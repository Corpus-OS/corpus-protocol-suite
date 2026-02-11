# LLM Protocol Conformance Test Coverage

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

This document tracks conformance test coverage for the **LLM Protocol V1.0** specification as defined in `SPECIFICATION.md §8`. Each test validates normative requirements (MUST/SHOULD) from the specification and shared behavior from the common foundation (errors, deadlines, observability, privacy).

This suite constitutes the **official LLM Protocol V1.0 Reference Conformance Test Suite**. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** LLM Protocol V1.0  
**Status:** Stable / Production-Ready  
**Last Updated:** 2026-01-19  
**Test Location:** `tests/llm/`  
**Performance:** 3.96s total (30ms/test average)

## Conformance Summary

**Overall Coverage: 132/132 tests (100%) ✅**

📊 **Total Tests:** 132/132 passing (100%)  
⚡ **Execution Time:** 3.96s (30ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Category | Tests | Coverage | Status |
|----------|-------|-----------|---------|
| **Core Operations** | 8/8 | 100% ✅ | Production Ready |
| **Message Validation** | 20/20 | 100% ✅ | Production Ready |
| **Sampling Parameters** | 41/41 | 100% ✅ | Production Ready |
| **Streaming Semantics** | 6/6 | 100% ✅ | Production Ready |
| **Error Handling** | 5/5 | 100% ✅ | Production Ready |
| **Capabilities Discovery** | 14/14 | 100% ✅ | Production Ready |
| **Observability & Privacy** | 8/8 | 100% ✅ | Production Ready |
| **Deadline Semantics** | 6/6 | 100% ✅ | Production Ready |
| **Token Counting** | 8/8 | 100% ✅ | Production Ready |
| **Health Endpoint** | 7/7 | 100% ✅ | Production Ready |
| **Wire Envelopes & Routing** | 11/11 | 100% ✅ | Production Ready |
| **Total** | **132/132** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 3.96 seconds total runtime
- **Average Per Test:** 30 milliseconds
- **Cache Efficiency:** 0 cache hits, 132 misses (cache size: 132)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Mock Adapter:** `tests.mock.mock_llm_adapter:MockLLMAdapter` - Deterministic mock for LLM operations
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Strict Mode:** Off (permissive testing)

### Certification Levels
- 🏆 **Platinum:** 132/132 tests (100%) with comprehensive coverage
- 🥇 **Gold:** 106+ tests (80%+ coverage)
- 🥈 **Silver:** 79+ tests (60%+ coverage)
- 🔬 **Development:** 66+ tests (50%+ coverage)

---

## Test Files

### `test_capabilities_shape.py`

**Specification:** §8.4 Model Discovery, §6.2 Capability Discovery  
**Status:** ✅ Complete (14 tests)

Tests all aspects of capability discovery:

* `test_capabilities_capabilities_shape_and_required_fields` - Quick smoke test of essential fields (§8.4)
* `test_capabilities_returns_correct_type` - Returns `LLMCapabilities` instance (§8.4)
* `test_capabilities_identity_fields` - `server`/`version`/`model_family` are non-empty strings (§6.2)
* `test_capabilities_resource_limits` - `max_context_length` positive and reasonable (≤ 10M) (§8.4)
* `test_capabilities_feature_flags_are_boolean` - All feature flags are booleans (§8.4)
* `test_capabilities_supported_models_structure` - Non-empty tuple/sequence of non-empty strings (§8.4)
* `test_capabilities_consistency_with_count_tokens` - Declared count-tokens support matches behavior (§8.4)
* `test_capabilities_consistency_with_streaming` - Declared streaming support matches behavior (§8.4)
* `test_capabilities_all_fields_present` - All required fields populated (§8.4)
* `test_capabilities_idempotency` - Multiple calls return consistent results (§6.2)
* `test_capabilities_reasonable_model_names` - Model names follow reasonable patterns (§8.4)
* `test_capabilities_no_duplicate_models` - Supported models list contains no duplicates (§8.4)
* `test_capabilities_model_gate_enforced_when_supported_models_listed` - Model gating validation (§8.4)
* `test_capabilities_tools_consistency_with_complete` - Tools capability consistency (§8.4)
* `test_capabilities_tools_flags_and_limits_valid` - Tools flags and limits validation (§8.4)

### `test_complete_basic.py`

**Specification:** §8.3 Operations  
**Status:** ✅ Complete (8 tests)

Validates basic completion contract:

* `test_core_ops_complete_basic_text_and_usage` - Non-empty text, token accounting present, model echoed, valid `finish_reason` (§8.3)
* `test_core_ops_complete_different_message_structures` - Handles various valid message formats (§8.3)
* `test_core_ops_complete_empty_messages_rejected` - Rejects empty message lists (§8.3)
* `test_core_ops_complete_response_contains_expected_fields` - Response includes all required fields (§8.3)
* `test_core_ops_complete_usage_accounting_consistent` - Token usage totals are mathematically consistent (§8.3)
* `test_core_ops_complete_different_models_produce_results` - Works across all supported models (§8.3)
* `test_complete_system_message_gated_by_capability` - System message capability gating (§8.3)
* `test_complete_tools_happy_path_emits_tool_calls_when_supported` - Tool calls emission when supported (§8.3)
* `test_complete_tool_choice_none_does_not_emit_tool_calls` - Tool choice none validation (§8.3)

### `test_streaming_semantics.py`

**Specification:** §8.3 Operations, §4.2.3 Streaming Frames  
**Status:** ✅ Complete (6 tests)

Validates streaming contract:

* `test_streaming_stream_has_single_final_chunk_and_progress_usage` - Progressive chunks with single terminal (§4.2.3)
* `test_streaming_stream_model_consistent_when_present` - Model field consistent across chunks (§8.3)
* `test_streaming_stream_early_cancel_then_new_stream_ok` - Resource cleanup on cancellation (§8.3)
* `test_streaming_stream_deadline_preexpired_yields_no_chunks` - Deadline enforcement in streaming (§12.1)
* `test_streaming_stream_content_progress_and_terminal_rules` - Content progression and terminal semantics (§4.2.3)
* `test_streaming_stream_body_matches_complete_result` - Streamed content parity with complete operation (§8.3)

### `test_count_tokens_consistency.py`

**Specification:** §8.3 Operations  
**Status:** ✅ Complete (8 tests)

Validates token counting behavior:

* `test_token_counting_count_tokens_monotonic` - Longer input never reports fewer tokens than shorter input (§8.3)
* `test_token_counting_empty_string` - Empty string returns 0 (or minimal constant) (§8.3)
* `test_token_counting_unicode_handling` - Unicode handled without error or negative counts (§8.3)
* `test_token_counting_whitespace_variations` - Various whitespace patterns handled correctly (§8.3)
* `test_token_counting_consistent_for_identical_inputs` - Same input yields same token count (§8.3)
* `test_token_counting_not_supported_raises_notsupported` - Not supported error handling (§8.5)
* `test_token_counting_model_gate_enforced_when_listed` - Model gating validation (§8.3)
* `test_token_counting_respects_context_limits` - Handles context length boundaries appropriately (§8.3)

### `test_message_validation.py`

**Specification:** §8.3 Operations - Message Format  
**Status:** ✅ Complete (20 tests) ⭐ Exemplary

Comprehensive schema validation:

* `test_message_validation_empty_messages_list_rejected` - Rejects empty message lists (§8.3)
* `test_message_validation_each_message_must_be_mapping` - Each message must be a mapping (§8.3)
* `test_message_validation_missing_role_field_rejected` - Rejects messages missing role field (§8.3)
* `test_message_validation_missing_content_field_rejected` - Rejects messages missing content field (§8.3)
* `test_message_validation_role_and_content_type_enforced` - Role and content type enforcement (§8.3)
* `test_message_validation_valid_roles_accepted` - Accepts standard roles (user, assistant) (§8.3)
* `test_message_validation_invalid_role_rejected_or_descriptive` - Rejects unknown/invalid role values (§8.3)
* `test_message_validation_empty_role_string_rejected_or_descriptive` - Rejects empty role strings (§8.3)
* `test_message_validation_system_role_requires_capability_best_effort` - System role respects capabilities (§8.3)
* `test_message_validation_empty_content_rejected_for_user_role` - Rejects empty content for user role (§8.3)
* `test_message_validation_whitespace_only_content_rejected` - Rejects whitespace-only content (§8.3)
* `test_message_validation_content_too_large_rejected` - Rejects excessively large content (§8.3)
* `test_message_validation_valid_content_types_accepted` - Accepts various valid content formats (§8.3)
* `test_message_validation_conversation_structure_accepted` - Accepts valid conversation structures (§8.3)
* `test_message_validation_tool_role_requires_tool_call_id` - Tool role validation (§8.3)
* `test_message_validation_mixed_invalid_and_valid_rejected` - Rejects conversations with mixed validity (§8.3)
* `test_message_validation_error_messages_are_descriptive` - Error messages are informative (§12.4)
* `test_message_validation_extra_keys_are_ignored` - Extra message keys are ignored (§4.2.5)
* `test_message_validation_messages_must_be_json_serializable` - Messages must be JSON serializable (§4.2.1)
* `test_message_validation_max_reasonable_messages_accepted` - Accepts reasonable message counts (§8.3)

### `test_sampling_params_validation.py`

**Specification:** §8.3 Operations - Sampling Parameters  
**Status:** ✅ Complete (41 tests)

Validates parameter ranges with extensive parameterization:

* `test_sampling_params_invalid_temperature_rejected` - 4 parameterized cases: -0.1, 2.1, -1.0, 999.0 (§8.3)
* `test_sampling_params_valid_temperature_accepted` - 5 parameterized cases: 0.0, 0.5, 1.0, 1.5, 2.0 (§8.3)
* `test_sampling_params_invalid_top_p_rejected` - 5 parameterized cases: 0.0, -0.1, 1.1, 2.0, -1.0 (§8.3)
* `test_sampling_params_valid_top_p_accepted` - 4 parameterized cases: 0.1, 0.5, 0.9, 1.0 (§8.3)
* `test_sampling_params_invalid_frequency_penalty_rejected` - 4 parameterized cases: -2.1, 2.1, -3.0, 5.0 (§8.3)
* `test_sampling_params_valid_frequency_penalty_accepted` - 5 parameterized cases: -2.0, -1.0, 0.0, 1.0, 2.0 (§8.3)
* `test_sampling_params_invalid_presence_penalty_rejected` - 4 parameterized cases: -2.1, 2.1, -3.0, 5.0 (§8.3)
* `test_sampling_params_valid_presence_penalty_accepted` - 5 parameterized cases: -2.0, -1.0, 0.0, 1.0, 2.0 (§8.3)
* `test_sampling_params_multiple_invalid_params_error_message` - Multiple invalid parameters error handling (§12.4)

Ensures strict adherence to:
* `temperature ∈ [0.0, 2.0]`
* `top_p ∈ (0.0, 1.0]`
* `frequency_penalty, presence_penalty ∈ [-2.0, 2.0]`

### `test_error_mapping_retryable.py`

**Specification:** §8.5 LLM-Specific Errors, §12.1 Retry Semantics, §12.4 Error Mapping Table  
**Status:** ✅ Complete (5 tests)

Validates classification and normalization:

* `test_error_handling_retryable_errors_with_hints` - Retryable errors with hints (§12.1)
* `test_error_handling_bad_request_is_non_retryable_and_no_retry_after` - BadRequest non-retryable validation (§12.4)
* `test_error_handling_deadline_exceeded_is_conditionally_retryable_with_no_chunks` - DeadlineExceeded semantics (§12.1)
* `test_error_handling_retryable_error_attributes_minimum_shape` - Error attributes consistency (§12.4)
* `test_error_handling_deadline_capability_alignment` - Deadline capability alignment (§12.1)

### `test_deadline_enforcement.py`

**Specification:** §6.1 Operation Context, §12.1 Retry Semantics, §12.4 Error Mapping Table  
**Status:** ✅ Complete (6 tests)

Validates deadline behavior:

* `test_deadline_deadline_budget_nonnegative_and_usable` - Derived budget never negative (§6.1)
* `test_deadline_deadline_exceeded_on_expired_budget` - Immediate `DeadlineExceeded` when deadline elapsed (§12.4)
* `test_deadline_deadline_exceeded_during_stream` - Streaming respects deadlines mid-generation (§12.1)
* `test_deadline_operations_complete_with_adequate_budget` - Operations succeed with adequate budget (§6.1)
* `test_deadline_budget_calculation_accuracy` - Budget calculations are accurate (§6.1)
* `test_deadline_not_enforced_when_capability_false` - Deadline capability alignment (§12.1)

### `test_health_report.py`

**Specification:** §8.3 Operations, §6.4 Observability Interfaces  
**Status:** ✅ Complete (7 tests)

Validates health contract:

* `test_health_health_has_required_fields` - `{"ok", "server", "version"}` present (§8.3)
* `test_health_health_shape_consistent_when_degraded` - Shape stable when degraded (§6.4)
* `test_health_health_identity_fields_are_stable_across_calls` - Identity fields stable across calls (§6.4)
* `test_health_health_deadline_preexpired_raises_deadline_exceeded` - Deadline enforcement in health checks (§12.1)
* `test_health_health_includes_optional_uptime_if_provided` - Optional uptime field (§6.4)
* `test_health_health_includes_optional_details_if_provided` - Optional details field (§6.4)
* `test_health_deadline_capability_alignment` - Deadline capability alignment (§12.1)

### `test_context_siem.py`

**Specification:** §13.1-§13.3 Observability and Monitoring, §15 Privacy Considerations, §6.1 Operation Context  
**Status:** ✅ Complete (8 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_observability_context_propagates_to_metrics_siem_safe` - Context propagation, no raw tenant IDs (§13.1, §15)
* `test_observability_metrics_emitted_on_error_path` - Metrics emitted on error paths (§13.1)
* `test_observability_streaming_metrics_siem_safe` - Streaming metrics SIEM safety (§13.1)
* `test_observability_token_counter_metrics_present` - Token counter metrics presence (§13.1)
* `test_observability_metrics_structure_consistency` - Metrics structure consistency (§13.1)
* `test_observability_no_metric_leakage_between_tenants` - No metric leakage between tenants (§15)
* `test_observability_tenant_hash_is_emitted_not_raw_tenant` - Tenant hashed, never raw (§15)
* `test_observability_error_metrics_include_code_and_no_prompt_leak` - Error metrics include code, no prompt leak (§13.1, §15)

### `test_wire_handler.py`

**Specification:** §4.2 Wire-First Canonical Form, §4.2.6 Operation Registry, §6.1 Operation Context, §6.3 Error Taxonomy, §8.3 Operations, §11.2 Consistent Observability, §13 Observability and Monitoring  
**Status:** ✅ Complete (11 tests)

Validates wire-level handler behavior:

* `test_wire_contract_capabilities_success_envelope` - Capabilities envelope structure (§4.2.1)
* `test_wire_contract_complete_roundtrip_and_context_plumbing` - Complete operation with context (§4.2.1, §6.1)
* `test_wire_contract_count_tokens_and_health_envelopes` - Count tokens and health envelopes (§4.2.1)
* `test_wire_contract_stream_success_chunks_and_context` - Streaming envelope handling (§4.2.3)
* `test_wire_strictness_missing_required_keys_maps_to_bad_request` - Missing required keys mapping (§4.2.1)
* `test_wire_strictness_ctx_and_args_must_be_objects` - Context and args must be objects (§4.2.1)
* `test_wire_contract_unknown_op_maps_to_not_supported` - Unknown operation mapping (§4.2.6)
* `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` - Invalid operation handling (§4.2.6)
* `test_wire_contract_maps_llm_adapter_error_to_normalized_envelope` - Error normalization (§6.3)
* `test_wire_contract_maps_unexpected_exception_to_unavailable_stable_message` - Exception mapping (§6.3)
* `test_wire_stream_error_envelope_terminates_stream` - Error envelope termination in streaming (§4.2.3)

---

## Specification Mapping

### §8.3 Operations - Complete Coverage

#### `complete()`

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns `LLMCompletion` | `test_complete_basic.py` | ✅ |
| Non-empty text response | `test_complete_basic.py` | ✅ |
| Token usage accounting present | `test_complete_basic.py` | ✅ |
| Valid `finish_reason` enum | `test_complete_basic.py` | ✅ |
| Validates message schema | `test_message_validation.py` | ✅ |
| Accepts standard roles | `test_message_validation.py` | ✅ |
| Sampling params in allowed ranges | `test_sampling_params_validation.py` | ✅ |
| Rejects invalid sampling params | `test_sampling_params_validation.py` | ✅ |
| Honors deadline semantics | `test_deadline_enforcement.py` | ✅ |
| Works across supported models | `test_complete_basic.py` | ✅ |
| JSON serializable messages | `test_message_validation.py` | ✅ |
| Extra message keys ignored | `test_message_validation.py` | ✅ |
| System message capability gating | `test_complete_basic.py` | ✅ |
| Tool calls emission when supported | `test_complete_basic.py` | ✅ |
| Tool choice none validation | `test_complete_basic.py` | ✅ |

#### `stream()`

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Yields `LLMChunk` instances | `test_streaming_semantics.py` | ✅ |
| Emits multiple chunks for non-trivial outputs | `test_streaming_semantics.py` | ✅ |
| Exactly one final chunk | `test_streaming_semantics.py` | ✅ |
| Final chunk is last | `test_streaming_semantics.py` | ✅ |
| `usage_so_far` monotonic over stream | `test_streaming_semantics.py` | ✅ |
| Aggregate text non-empty | `test_streaming_semantics.py` | ✅ |
| Respects deadline during streaming | `test_deadline_enforcement.py` | ✅ |
| Model consistency across chunks | `test_streaming_semantics.py` | ✅ |
| Resource cleanup on cancellation | `test_streaming_semantics.py` | ✅ |
| Early cancel then new stream works | `test_streaming_semantics.py` | ✅ |
| Body matches complete result | `test_streaming_semantics.py` | ✅ |

#### `count_tokens()`

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns non-negative integer | `test_count_tokens_consistency.py` | ✅ |
| Monotonic w.r.t. input length | `test_count_tokens_consistency.py` | ✅ |
| Handles empty string | `test_count_tokens_consistency.py` | ✅ |
| Handles Unicode safely | `test_count_tokens_consistency.py` | ✅ |
| Consistent for identical inputs | `test_count_tokens_consistency.py` | ✅ |
| Respects context limits | `test_count_tokens_consistency.py` | ✅ |
| Not supported error handling | `test_count_tokens_consistency.py` | ✅ |
| Model gating validation | `test_count_tokens_consistency.py` | ✅ |

#### `health()`

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns object/dict | `test_health_report.py` | ✅ |
| Includes `ok` (bool) | `test_health_report.py` | ✅ |
| Includes `server` (str) | `test_health_report.py` | ✅ |
| Includes `version` (str) | `test_health_report.py` | ✅ |
| Stable shape across ok/degraded/err | `test_health_report.py` | ✅ |
| Stable identity fields | `test_health_report.py` | ✅ |
| Honors deadline semantics | `test_health_report.py` | ✅ |
| Includes optional uptime | `test_health_report.py` | ✅ |
| Includes optional details | `test_health_report.py` | ✅ |
| Deadline capability alignment | `test_health_report.py` | ✅ |

### §8.4 Capabilities - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Returns `LLMCapabilities` | `test_capabilities_shape.py` | ✅ |
| `server` / `version` / `model_family` set | `test_capabilities_shape.py` | ✅ |
| Resource limits positive | `test_capabilities_shape.py` | ✅ |
| Feature flags are booleans | `test_capabilities_shape.py` | ✅ |
| `supported_models` well-formed | `test_capabilities_shape.py` | ✅ |
| Matches `count_tokens` behavior | `test_capabilities_shape.py` | ✅ |
| Matches streaming support | `test_capabilities_shape.py` | ✅ |
| All required fields present | `test_capabilities_shape.py` | ✅ |
| Idempotent across calls | `test_capabilities_shape.py` | ✅ |
| Reasonable model names | `test_capabilities_shape.py` | ✅ |
| No duplicate models | `test_capabilities_shape.py` | ✅ |
| Model gating validation | `test_capabilities_shape.py` | ✅ |
| Tools capability consistency | `test_capabilities_shape.py` | ✅ |
| Tools flags and limits validation | `test_capabilities_shape.py` | ✅ |

### §8.5 Error Handling - Complete Coverage

| Error / Behavior | Test File | Status |
|------------------|-----------|--------|
| `BadRequest` for validation failures | `test_message_validation.py`, `test_sampling_params_validation.py` | ✅ |
| `ResourceExhausted` with `retry_after_ms` hints | `test_error_mapping_retryable.py` | ✅ |
| `Unavailable` classified retryable | `test_error_mapping_retryable.py` | ✅ |
| `DeadlineExceeded` on timeout/deadline | `test_deadline_enforcement.py` | ✅ |
| `NotSupported` for unsupported features/models | `test_error_mapping_retryable.py`, `test_count_tokens_consistency.py` | ✅ |
| Normalized `code` + attributes on all errors | `test_error_mapping_retryable.py` | ✅ |
| Deadline capability alignment | `test_error_mapping_retryable.py`, `test_deadline_enforcement.py` | ✅ |
| BadRequest non-retryable validation | `test_error_mapping_retryable.py` | ✅ |
| Retryable error attributes shape | `test_error_mapping_retryable.py` | ✅ |
| Deadline exceeded with no chunks | `test_error_mapping_retryable.py` | ✅ |

### §13 Observability - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Tenant never logged raw | `test_context_siem.py` | ✅ |
| Tenant hashed in metrics | `test_context_siem.py` | ✅ |
| No prompt content in metrics | `test_context_siem.py` | ✅ |
| Metrics also on error paths | `test_context_siem.py` | ✅ |
| Streaming metrics SIEM safe | `test_context_siem.py` | ✅ |
| Token counter metrics present | `test_context_siem.py` | ✅ |
| Metrics structure consistency | `test_context_siem.py` | ✅ |
| No metric leakage between tenants | `test_context_siem.py` | ✅ |
| Error metrics include code, no prompt leak | `test_context_siem.py` | ✅ |

### §15 Privacy - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| No PII in telemetry | `test_context_siem.py` | ✅ |
| Tenant identifiers hashed | `test_context_siem.py` | ✅ |
| No metric leakage between tenants | `test_context_siem.py` | ✅ |

### §6.1 Context & Deadlines - Complete Coverage

| Requirement | Test File | Status |
|-------------|-----------|--------|
| Budget computation | `test_deadline_enforcement.py` | ✅ |
| Deadline enforcement | `test_deadline_enforcement.py` | ✅ |
| Streaming deadline respect | `test_deadline_enforcement.py` | ✅ |
| Operations succeed with adequate budget | `test_deadline_enforcement.py` | ✅ |
| Budget calculation accuracy | `test_deadline_enforcement.py` | ✅ |
| Deadline capability alignment | `test_deadline_enforcement.py` | ✅ |
| Deadline not enforced when capability false | `test_deadline_enforcement.py` | ✅ |

### §4.2 Wire Protocol - Partial Coverage (LLM-specific)
*Note: Complete wire protocol coverage is in the separate wire conformance suite*

| Requirement | Test File | Status |
|-------------|-----------|--------|
| LLM operation routing | `test_wire_handler.py` | ✅ |
| Error envelope normalization | `test_wire_handler.py` | ✅ |
| Context propagation | `test_wire_handler.py` | ✅ |
| Unknown operation handling | `test_wire_handler.py` | ✅ |
| Streaming envelope handling | `test_wire_handler.py` | ✅ |
| Error envelope termination in streaming | `test_wire_handler.py` | ✅ |
| Missing required keys mapping | `test_wire_handler.py` | ✅ |
| Context and args object validation | `test_wire_handler.py` | ✅ |

---

## Running Tests

### All LLM conformance tests (3.96s typical)
```bash
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest tests/llm/ -v
```

### Performance Optimized Runs
```bash
# Parallel execution (recommended for CI/CD) - ~2.0s
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest tests/llm/ -n auto

# With detailed timing report
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest tests/llm/ --durations=10

# Fast mode (skip slow markers)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest tests/llm/ -k "not slow"
```

### By category with timing estimates
```bash
# Core operations & streaming (~1.2s)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest \
  tests/llm/test_complete_basic.py \
  tests/llm/test_streaming_semantics.py \
  tests/llm/test_count_tokens_consistency.py \
  tests/llm/test_health_report.py -v

# Validation & parameters (~1.5s)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest \
  tests/llm/test_message_validation.py \
  tests/llm/test_sampling_params_validation.py -v

# Infrastructure & capabilities (~0.8s)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest \
  tests/llm/test_capabilities_shape.py \
  tests/llm/test_deadline_enforcement.py \
  tests/llm/test_context_siem.py -v

# Error handling (~0.3s)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest \
  tests/llm/test_error_mapping_retryable.py -v

# Wire handler (~0.4s)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter pytest \
  tests/llm/test_wire_handler.py -v
```

### With Coverage Report
```bash
# Basic coverage (4.5s typical)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/llm/ --cov=corpus_sdk.llm --cov-report=html

# Minimal coverage (4.0s typical)
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/llm/ --cov=corpus_sdk.llm --cov-report=term-missing

# CI/CD optimized (parallel + coverage) - ~2.5s
CORPUS_ADAPTER=tests.mock.mock_llm_adapter:MockLLMAdapter \
  pytest tests/llm/ -n auto --cov=corpus_sdk.llm --cov-report=xml
```

### Adapter-Agnostic Usage
To validate a **third-party** or custom LLM Protocol implementation:

1. Implement the LLM Protocol V1.0 interface as defined in `SPECIFICATION.md §8`
2. Provide a small adapter/fixture that binds these tests to your implementation
3. Run the full `tests/llm/` suite
4. If all 132 tests pass unmodified, you can accurately claim:
   **"LLM Protocol V1.0 - 100% Conformant (Corpus Reference Suite)"**

### With Makefile Integration
```bash
# Run all LLM tests (3.96s typical)
make test-llm

# Run LLM tests with coverage (4.5s typical)
make test-llm-coverage

# Run LLM tests in parallel (2.0s typical)
make test-llm-parallel

# Run specific categories
make test-llm-core      # Core operations
make test-llm-validation # Validation tests
make test-llm-errors    # Error handling
make test-llm-wire      # Wire handler
```

---

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new LLM adapter:

### ✅ Phase 1: Core Operations (8/8)
* [x] `capabilities()` returns valid `LLMCapabilities` with all fields (§8.4)
* [x] `complete()` returns `LLMCompletion` with usage + finish_reason (§8.3)
* [x] `stream()` emits chunks with exactly one final marker (§8.3, §4.2.3)
* [x] `count_tokens()` returns non-negative int with proper behavior (§8.3)
* [x] `health()` returns `{ok, server, version}` with all fields (§8.3)
* [x] Works across all supported models (§8.4)
* [x] System message capability gating (§8.3)
* [x] Tool calls emission when supported (§8.3)

### ✅ Phase 2: Message Validation (20/20)
* [x] Rejects empty messages (§8.3)
* [x] Rejects unknown roles (§8.3)
* [x] Rejects missing required fields (§8.3)
* [x] Accepts `system` / `user` / `assistant` (§8.3)
* [x] Handles large (reasonable) content (§8.3)
* [x] Validates conversation structures (§8.3)
* [x] Provides descriptive error messages (§12.4)
* [x] Rejects empty role strings (§8.3)
* [x] System role capability checking (§8.3)
* [x] Rejects empty content for user role (§8.3)
* [x] Rejects whitespace-only content (§8.3)
* [x] Accepts valid content types (§8.3)
* [x] Tool role validation (§8.3)
* [x] Mixed validity rejection (§8.3)
* [x] Extra keys ignored (§4.2.5)
* [x] JSON serializable messages (§4.2.1)
* [x] Reasonable message count acceptance (§8.3)
* [x] Role and content type enforcement (§8.3)
* [x] Each message must be mapping (§8.3)

### ✅ Phase 3: Parameter Validation (41/41)
* [x] Enforces `temperature` in [0.0, 2.0] (§8.3)
* [x] Enforces `top_p` in (0.0, 1.0] (§8.3)
* [x] Enforces `frequency_penalty` in [-2.0, 2.0] (§8.3)
* [x] Enforces `presence_penalty` in [-2.0, 2.0] (§8.3)
* [x] Valid parameter acceptance tested (§8.3)
* [x] Invalid parameter rejection tested (§8.3)
* [x] Multiple invalid parameter error messages (§12.4)

### ✅ Phase 4: Streaming Semantics (6/6)
* [x] Yields `LLMChunk` objects (§8.3)
* [x] Multiple chunks where applicable (§4.2.3)
* [x] Exactly one final chunk (§4.2.3)
* [x] Final chunk is last (§4.2.3)
* [x] `usage_so_far` monotonic and consistent (§8.3)
* [x] Model consistency across chunks (§8.3)
* [x] Resource cleanup on cancellation (§8.3)
* [x] Early cancellation safety (§8.3)
* [x] Deadline enforcement in streaming (§12.1)
* [x] Content progression rules (§4.2.3)
* [x] Body matches complete result (§8.3)

### ✅ Phase 5: Token Counting (8/8)
* [x] Non-negative integers (§8.3)
* [x] Monotonic vs input length (§8.3)
* [x] Correct empty-string handling (§8.3)
* [x] Robust Unicode handling (§8.3)
* [x] Consistent for identical inputs (§8.3)
* [x] Respects context limits (§8.3)
* [x] Not supported error handling (§8.5)
* [x] Model gating validation (§8.3)

### ✅ Phase 6: Error Handling (9/9)
* [x] Maps validation issues → `BadRequest` (§12.4)
* [x] Maps quotas/limits → `ResourceExhausted` (+ `retry_after_ms`) (§12.1)
* [x] Maps transient issues → `Unavailable` / retryable (§12.1)
* [x] Maps timeouts → `DeadlineExceeded` (§12.4)
* [x] Maps unsupported → `NotSupported` (§12.4)
* [x] Emits normalized `code` and attributes (§12.4)
* [x] Deadline capability alignment (§12.1)
* [x] Retryable error attributes shape (§12.4)
* [x] Deadline exceeded with no chunks (§12.1)
* [x] BadRequest non-retryable validation (§12.4)

### ✅ Phase 7: Deadline Enforcement (6/6)
* [x] Correct budget computation (§6.1)
* [x] Preflight deadline checks where applicable (§6.1)
* [x] Honors deadlines in unary calls (§12.1)
* [x] Honors deadlines mid-stream (§12.1)
* [x] Accurate budget calculations (§6.1)
* [x] Deadline not enforced when capability false (§12.1)

### ✅ Phase 8: Observability & Privacy (8/8)
* [x] Never logs raw tenant IDs (§15)
* [x] Uses tenant hash in metrics (§13.1, §15)
* [x] Excludes prompt content from metrics (§13.1)
* [x] Emits metrics on both success and error paths (§13.1)
* [x] Streaming metrics SIEM safe (§13.1)
* [x] Token counter metrics present (§13.1)
* [x] Metrics structure consistency (§13.1)
* [x] No metric leakage between tenants (§15)
* [x] Error metrics include code, no prompt leak (§13.1, §15)

### ✅ Phase 9: Wire Contract & Envelopes (11/11)
* [x] `WireLLMHandler` implements all `llm.*` operations (§4.2.6)
* [x] Success envelopes have correct `{ok, code, ms, result}` shape (§4.2.1)
* [x] Error envelopes normalize to `{ok=false, code, error, message, ...}` (§4.2.1)
* [x] `OperationContext` properly constructed from wire `ctx` (§6.1)
* [x] Unknown fields ignored in requests (§4.2.5)
* [x] Unknown operations map to `NotSupported` (§4.2.6)
* [x] Unexpected exceptions map to `Unavailable` (§6.3)
* [x] Missing required keys mapping (§4.2.1)
* [x] Context and args must be objects (§4.2.1)
* [x] Streaming envelope handling (§4.2.3)
* [x] Error envelope termination in streaming (§4.2.3)

---

## Conformance Badge

```text
🏆 LLM PROTOCOL V1.0 - PLATINUM CERTIFIED
   132/132 conformance tests passing (100%)

   📊 Total Tests: 132/132 passing (100%)
   ⚡ Execution Time: 3.96s (30ms/test avg)
   🏆 Certification: Platinum (100%)

   ✅ Core Operations: 8/8 (100%) - §8.3
   ✅ Message Validation: 20/20 (100%) - §8.3
   ✅ Sampling Parameters: 41/41 (100%) - §8.3
   ✅ Streaming Semantics: 6/6 (100%) - §8.3, §4.2.3
   ✅ Error Handling: 5/5 (100%) - §8.5, §12.1, §12.4
   ✅ Capabilities Discovery: 14/14 (100%) - §8.4, §6.2
   ✅ Observability & Privacy: 8/8 (100%) - §13, §15
   ✅ Deadline Semantics: 6/6 (100%) - §6.1, §12.1, §12.4
   ✅ Token Counting: 8/8 (100%) - §8.3
   ✅ Health Endpoint: 7/7 (100%) - §8.3, §6.4
   ✅ Wire Envelopes & Routing: 11/11 (100%) - §4.2

   Status: Production Ready 🏆 Platinum Certified
```

**Badge Suggestion:**
[![Corpus LLM Protocol](https://img.shields.io/badge/CorpusLLM%20Protocol-Platinum%20Certified-brightgreen)](./llm_conformance_report.json)

**Performance Benchmark:**
```text
Execution Time: 3.96s total (30ms/test average)
Cache Efficiency: 0 hits, 132 misses (cache size: 132)
Parallel Ready: Yes (optimized for pytest-xdist)
Memory Footprint: Minimal (deterministic mocks)
Specification Coverage: 100% of §8 requirements
Test Files: 11 comprehensive modules
```

**Last Updated:** 2026-01-19  
**Maintained By:** Corpus SDK Team  
**Test Suite:** `tests/llm/` (11 test files)  
**Specification Version:** V1.0.0 §8  
**Status:** 100% V1.0 Conformant - Platinum Certified (132/132 tests, 3.96s runtime)

---
