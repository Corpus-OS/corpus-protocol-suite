# LLM Protocol Conformance Test Coverage

**Table of Contents**
- [Overview](#overview)
- [Conformance Summary](#conformance-summary)
- [Test Files](#test-files)
- [Specification Mapping](#specification-mapping)
- [Running Tests](#running-tests)
- [Adapter Compliance Checklist](#adapter-compliance-checklist)
- [Conformance Badge](#conformance-badge)
- [Maintenance](#maintenance)

---

## Overview

This document tracks conformance test coverage for the **LLM Protocol V1.0** specification as defined in `SPECIFICATION.md §8`. Each test validates normative requirements (MUST/SHOULD) from the specification.

This suite constitutes the official LLM Protocol V1.0 Reference Conformance Test Suite. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** LLM Protocol V1.0
**Status:** Pre-Release
**Last Updated:** 2025-01-XX
**Test Location:** `tests/llm/`

## Conformance Summary

**Overall Coverage: 111/111 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 6/6   | 100% ✅   |
| Message Validation       | 15/15 | 100% ✅   |
| Sampling Parameters      | 41/41 | 100% ✅   |
| Streaming Semantics      | 6/6   | 100% ✅   |
| Error Handling           | 4/4   | 100% ✅   |
| Capabilities Discovery   | 12/12 | 100% ✅   |
| Observability & Privacy  | 6/6   | 100% ✅   |
| Deadline Semantics       | 5/5   | 100% ✅   |
| Token Counting           | 6/6   | 100% ✅   |
| Health Endpoint          | 6/6   | 100% ✅   |
| Wire Envelopes & Routing | 8/8   | 100% ✅   |

> Note: Categories are logical groupings. Individual tests may satisfy multiple normative requirements.

## Test Files

### test_capabilities_shape.py

**Specification:** §8.4 - Capabilities Discovery
**Status:** ✅ Complete (12 tests)

Tests all aspects of capability discovery:

* `test_capabilities_shape_and_required_fields` - Quick smoke test of essential fields
* `test_capabilities_returns_correct_type` - Returns `LLMCapabilities` instance
* `test_capabilities_identity_fields` - `server`/`version`/`model_family` are non-empty strings
* `test_capabilities_resource_limits` - `max_context_length` positive and reasonable (≤ 10M)
* `test_capabilities_feature_flags_are_boolean` - All feature flags are booleans
* `test_capabilities_supported_models_structure` - Non-empty tuple/sequence of non-empty strings
* `test_capabilities_consistency_with_count_tokens` - Declared count-tokens support matches behavior
* `test_capabilities_consistency_with_streaming` - Declared streaming support matches behavior
* `test_capabilities_all_fields_present` - All required fields populated
* `test_capabilities_idempotency` - Multiple calls return consistent results
* `test_capabilities_reasonable_model_names` - Model names follow reasonable patterns
* `test_capabilities_no_duplicate_models` - Supported models list contains no duplicates

### test_complete_basic.py

**Specification:** §8.3 - `complete` Operation
**Status:** ✅ Complete (6 tests)

Validates basic completion contract:

* `test_core_ops_complete_basic_text_and_usage` - Non-empty text, token accounting present, model echoed, valid `finish_reason`
* `test_core_ops_complete_different_message_structures` - Handles various valid message formats
* `test_core_ops_complete_empty_messages_rejected` - Rejects empty message lists
* `test_core_ops_complete_response_contains_expected_fields` - Response includes all required fields
* `test_core_ops_complete_usage_accounting_consistent` - Token usage totals are mathematically consistent
* `test_core_ops_complete_different_models_produce_results` - Works across all supported models

### test_streaming_semantics.py

**Specification:** §8.3 - `stream` Operation
**Status:** ✅ Complete (6 tests)

Validates streaming contract:

* `test_streaming_stream_has_single_final_chunk_and_progress_usage` - Progressive chunks with single terminal
* `test_streaming_stream_model_consistent_when_present` - Model field consistent across chunks
* `test_streaming_stream_early_cancel_then_new_stream_ok` - Resource cleanup on cancellation
* `test_streaming_stream_deadline_preexpired_yields_no_chunks` - Deadline enforcement in streaming
* `test_streaming_stream_content_progress_and_terminal_rules` - Content progression and terminal semantics
* `test_streaming_stream_body_matches_complete_result` - Streamed content parity with complete operation

### test_count_tokens_consistency.py

**Specification:** §8.3 - Token Counting
**Status:** ✅ Complete (6 tests)

Validates token counting behavior:

* `test_token_counting_count_tokens_monotonic` - Longer input never reports fewer tokens than shorter input
* `test_token_counting_empty_string` - Empty string returns 0 (or minimal constant)
* `test_token_counting_unicode_handling` - Unicode handled without error or negative counts
* `test_token_counting_whitespace_variations` - Various whitespace patterns handled correctly
* `test_token_counting_consistent_for_identical_inputs` - Same input yields same token count
* `test_token_counting_respects_context_limits` - Handles context length boundaries appropriately

### test_message_validation.py

**Specification:** §8.3 - Message Format
**Status:** ✅ Complete (15 tests) ⭐ Exemplary

Comprehensive schema validation:

* `test_message_validation_empty_messages_list_rejected` - Rejects empty message lists
* `test_message_validation_missing_role_field_rejected` - Rejects messages missing role field
* `test_message_validation_missing_content_field_rejected` - Rejects messages missing content field
* `test_message_validation_invalid_role_rejected` - Rejects unknown/invalid role values
* `test_message_validation_valid_roles_accepted` - Accepts standard roles (user, assistant)
* `test_message_validation_system_role_requires_capability` - System role respects capabilities
* `test_message_validation_empty_content_rejected_for_user_role` - Rejects empty content for user role
* `test_message_validation_content_too_large_rejected` - Rejects excessively large content
* `test_message_validation_valid_content_types_accepted` - Accepts various valid content formats
* `test_message_validation_conversation_structure_accepted` - Accepts valid conversation structures
* `test_message_validation_tool_role_requires_tool_call_id` - Tool role validation
* `test_message_validation_mixed_invalid_and_valid_rejected` - Rejects conversations with mixed validity
* `test_message_validation_error_messages_are_descriptive` - Error messages are informative
* `test_message_validation_whitespace_only_content_rejected` - Rejects whitespace-only content
* `test_message_validation_max_reasonable_messages_accepted` - Accepts reasonable message counts

### test_sampling_params_validation.py

**Specification:** §8.3 - Sampling Parameters
**Status:** ✅ Complete (41 tests)

Validates parameter ranges with extensive parameterization:

* `test_sampling_params_invalid_temperature_rejected` - 4 parameterized cases
* `test_sampling_params_valid_temperature_accepted` - 5 parameterized cases
* `test_sampling_params_invalid_top_p_rejected` - 5 parameterized cases
* `test_sampling_params_valid_top_p_accepted` - 4 parameterized cases
* `test_sampling_params_invalid_frequency_penalty_rejected` - 4 parameterized cases
* `test_sampling_params_valid_frequency_penalty_accepted` - 5 parameterized cases
* `test_sampling_params_invalid_presence_penalty_rejected` - 4 parameterized cases
* `test_sampling_params_valid_presence_penalty_accepted` - 5 parameterized cases
* `test_sampling_params_multiple_invalid_params_error_message` - Multiple invalid parameters

Ensures strict adherence to:
* `temperature ∈ [0.0, 2.0]`
* `top_p ∈ (0.0, 1.0]`
* `frequency_penalty, presence_penalty ∈ [-2.0, 2.0]`

### test_error_mapping_retryable.py

**Specification:** §8.5, §12.1, §12.4 - Error Handling
**Status:** ✅ Complete (4 tests)

Validates classification and normalization:

* `test_error_handling_retryable_errors_with_hints` - Retryable errors include appropriate `retry_after_ms`
* `test_error_handling_bad_request_is_non_retryable_and_no_retry_after` - BadRequest is non-retryable
* `test_error_handling_deadline_exceeded_is_conditionally_retryable_with_no_chunks` - DeadlineExceeded semantics
* `test_error_handling_retryable_error_attributes_minimum_shape` - Error attributes consistency

### test_deadline_enforcement.py

**Specification:** §8.3, §12.1 - Deadline Semantics
**Status:** ✅ Complete (5 tests)

Validates deadline behavior:

* `test_deadline_deadline_budget_nonnegative_and_usable` - Derived budget never negative
* `test_deadline_deadline_exceeded_on_expired_budget` - Immediate `DeadlineExceeded` when deadline elapsed
* `test_deadline_deadline_exceeded_during_stream` - Streaming respects deadlines mid-generation
* `test_deadline_operations_complete_with_adequate_budget` - Operations succeed with adequate budget
* `test_deadline_budget_calculation_accuracy` - Budget calculations are accurate

### test_health_report.py

**Specification:** §8.3 - Health Endpoint
**Status:** ✅ Complete (6 tests)

Validates health contract:

* `test_health_health_has_required_fields` - `{"ok", "server", "version"}` present
* `test_health_health_shape_consistent_when_degraded` - Shape stable when degraded
* `test_health_health_identity_fields_are_stable_across_calls` - Identity fields stable across calls
* `test_health_health_deadline_preexpired_raises_deadline_exceeded` - Deadline enforcement in health checks
* `test_health_health_includes_optional_uptime_if_provided` - Optional uptime field
* `test_health_health_includes_optional_details_if_provided` - Optional details field

### test_context_siem.py

**Specification:** §13.2, §15 - Observability & Privacy
**Status:** ✅ Complete (6 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_observability_context_propagates_to_metrics_siem_safe` - Context propagation, no raw tenant IDs
* `test_observability_metrics_emitted_on_error_path` - Metrics emitted on error paths
* `test_observability_streaming_metrics_siem_safe` - Streaming metrics privacy
* `test_observability_token_counter_metrics_present` - Token usage counters
* `test_observability_metrics_structure_consistency` - Consistent metrics structure
* `test_observability_no_metric_leakage_between_tenants` - Tenant isolation in metrics

### test_wire_handler.py

**Specification:** §4.1, §8.3, §8.4, §8.5, §13 - Wire Contract & Envelopes
**Status:** ✅ Complete (8 tests)

Validates wire-level handler behavior:

* `test_wire_contract_capabilities_success_envelope` - Capabilities envelope structure
* `test_wire_contract_complete_roundtrip_and_context_plumbing` - Complete operation with context
* `test_wire_contract_count_tokens_and_health_envelopes` - Count tokens and health envelopes
* `test_wire_contract_stream_success_chunks_and_context` - Streaming envelope handling
* `test_wire_contract_unknown_op_maps_to_not_supported` - Unknown operation mapping
* `test_wire_contract_missing_or_invalid_op_maps_to_bad_request` - Invalid operation handling
* `test_wire_contract_maps_llm_adapter_error_to_normalized_envelope` - Error normalization
* `test_wire_contract_maps_unexpected_exception_to_unavailable` - Exception mapping

## Specification Mapping

### §8.3 Operations - Complete Coverage

#### `complete()`

| Requirement                       | Test File                          | Status |
| --------------------------------- | ---------------------------------- | ------ |
| Returns `LLMCompletion`           | test_complete_basic.py             | ✅      |
| Non-empty text response           | test_complete_basic.py             | ✅      |
| Token usage accounting present    | test_complete_basic.py             | ✅      |
| Valid `finish_reason` enum        | test_complete_basic.py             | ✅      |
| Validates message schema          | test_message_validation.py         | ✅      |
| Accepts standard roles            | test_message_validation.py         | ✅      |
| Sampling params in allowed ranges | test_sampling_params_validation.py | ✅      |
| Rejects invalid sampling params   | test_sampling_params_validation.py | ✅      |
| Honors deadline semantics         | test_deadline_enforcement.py       | ✅      |
| Works across supported models     | test_complete_basic.py             | ✅      |

#### `stream()`

| Requirement                                   | Test File                    | Status |
| --------------------------------------------- | ---------------------------- | ------ |
| Yields `LLMChunk` instances                   | test_streaming_semantics.py  | ✅      |
| Emits multiple chunks for non-trivial outputs | test_streaming_semantics.py  | ✅      |
| Exactly one final chunk                       | test_streaming_semantics.py  | ✅      |
| Final chunk is last                           | test_streaming_semantics.py  | ✅      |
| `usage_so_far` monotonic over stream          | test_streaming_semantics.py  | ✅      |
| Aggregate text non-empty                      | test_streaming_semantics.py  | ✅      |
| Respects deadline during streaming            | test_deadline_enforcement.py | ✅      |
| Model consistency across chunks              | test_streaming_semantics.py  | ✅      |
| Resource cleanup on cancellation             | test_streaming_semantics.py  | ✅      |

#### `count_tokens()`

| Requirement                   | Test File                        | Status |
| ----------------------------- | -------------------------------- | ------ |
| Returns non-negative integer  | test_count_tokens_consistency.py | ✅      |
| Monotonic w.r.t. input length | test_count_tokens_consistency.py | ✅      |
| Handles empty string          | test_count_tokens_consistency.py | ✅      |
| Handles Unicode safely        | test_count_tokens_consistency.py | ✅      |
| Consistent for identical inputs | test_count_tokens_consistency.py | ✅      |
| Respects context limits       | test_count_tokens_consistency.py | ✅      |

#### `health()`

| Requirement                         | Test File             | Status |
| ----------------------------------- | --------------------- | ------ |
| Returns object/dict                 | test_health_report.py | ✅      |
| Includes `ok` (bool)                | test_health_report.py | ✅      |
| Includes `server` (str)             | test_health_report.py | ✅      |
| Includes `version` (str)            | test_health_report.py | ✅      |
| Stable shape across ok/degraded/err | test_health_report.py | ✅      |
| Stable identity fields              | test_health_report.py | ✅      |
| Honors deadline semantics           | test_health_report.py | ✅      |

### §8.4 Capabilities - Complete Coverage

| Requirement                               | Test File                  | Status |
| ----------------------------------------- | -------------------------- | ------ |
| Returns `LLMCapabilities`                 | test_capabilities_shape.py | ✅      |
| `server` / `version` / `model_family` set | test_capabilities_shape.py | ✅      |
| Resource limits positive                  | test_capabilities_shape.py | ✅      |
| Feature flags are booleans                | test_capabilities_shape.py | ✅      |
| `supported_models` well-formed            | test_capabilities_shape.py | ✅      |
| Matches `count_tokens` behavior           | test_capabilities_shape.py | ✅      |
| Matches streaming support                 | test_capabilities_shape.py | ✅      |
| All required fields present               | test_capabilities_shape.py | ✅      |
| Idempotent across calls                   | test_capabilities_shape.py | ✅      |
| Reasonable model names                    | test_capabilities_shape.py | ✅      |
| No duplicate models                       | test_capabilities_shape.py | ✅      |

### §8.5 Error Handling - Complete Coverage

| Error / Behavior                                | Test File                                                      | Status |
| ----------------------------------------------- | -------------------------------------------------------------- | ------ |
| `BadRequest` for validation failures            | test_message_validation.py, test_sampling_params_validation.py | ✅      |
| `ResourceExhausted` with `retry_after_ms` hints | test_error_mapping_retryable.py                                | ✅      |
| `Unavailable` classified retryable              | test_error_mapping_retryable.py                                | ✅      |
| `DeadlineExceeded` on timeout/deadline          | test_deadline_enforcement.py                                   | ✅      |
| `NotSupported` for unsupported features/models  | test_error_mapping_retryable.py                                | ✅      |
| Normalized `code` + attributes on all errors    | test_error_mapping_retryable.py                                | ✅      |

### §13.2 Observability - Complete Coverage

| Requirement                       | Test File            | Status |
| --------------------------------- | -------------------- | ------ |
| Tenant never logged raw           | test_context_siem.py | ✅      |
| Tenant hashed in metrics          | test_context_siem.py | ✅      |
| No prompt content in metrics      | test_context_siem.py | ✅      |
| Metrics also on error paths       | test_context_siem.py | ✅      |
| Token counters for LLM operations | test_context_siem.py | ✅      |
| Consistent metrics structure      | test_context_siem.py | ✅      |
| Tenant isolation in metrics       | test_context_siem.py | ✅      |

### §15 Privacy - Complete Coverage

| Requirement               | Test File            | Status |
| ------------------------- | -------------------- | ------ |
| No PII in telemetry       | test_context_siem.py | ✅      |
| Tenant identifiers hashed | test_context_siem.py | ✅      |

## Running Tests

### All LLM conformance tests

```bash
pytest tests/llm/ -v
```

### By category

```bash
# Core operations
pytest tests/llm/test_complete_basic.py \
       tests/llm/test_streaming_semantics.py \
       tests/llm/test_count_tokens_consistency.py \
       tests/llm/test_health_report.py -v

# Validation
pytest tests/llm/test_message_validation.py \
       tests/llm/test_sampling_params_validation.py -v

# Infrastructure & deadlines
pytest tests/llm/test_capabilities_shape.py \
       tests/llm/test_deadline_enforcement.py \
       tests/llm/test_context_siem.py -v

# Error handling
pytest tests/llm/test_error_mapping_retryable.py -v

# Wire contracts
pytest tests/llm/test_wire_handler.py -v
```

### With coverage report

```bash
pytest tests/llm/ --cov=corpus_sdk.llm --cov-report=html
```

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new LLM adapter:

### ✅ Phase 1: Core Operations

* [ ] `capabilities()` returns valid `LLMCapabilities`
* [ ] `complete()` returns `LLMCompletion` with usage + finish_reason
* [ ] `stream()` emits chunks with exactly one final marker
* [ ] `count_tokens()` returns non-negative int
* [ ] `health()` returns `{ok, server, version}`

### ✅ Phase 2: Message Validation

* [ ] Rejects empty messages
* [ ] Rejects unknown roles
* [ ] Rejects missing required fields
* [ ] Accepts `system` / `user` / `assistant`
* [ ] Handles large (reasonable) content
* [ ] Validates conversation structures
* [ ] Provides descriptive error messages

### ✅ Phase 3: Parameter Validation

* [ ] Enforces `temperature` in [0.0, 2.0]
* [ ] Enforces `top_p` in (0.0, 1.0]
* [ ] Enforces `frequency_penalty` in [-2.0, 2.0]
* [ ] Enforces `presence_penalty` in [-2.0, 2.0]

### ✅ Phase 4: Streaming Semantics

* [ ] Yields `LLMChunk` objects
* [ ] Multiple chunks where applicable
* [ ] Exactly one final chunk
* [ ] Final chunk is last
* [ ] `usage_so_far` monotonic and consistent
* [ ] Model consistency across chunks

### ✅ Phase 5: Token Counting

* [ ] Non-negative integers
* [ ] Monotonic vs input length
* [ ] Correct empty-string handling
* [ ] Robust Unicode handling
* [ ] Consistent for identical inputs
* [ ] Respects context limits

### ✅ Phase 6: Error Handling

* [ ] Maps validation issues → `BadRequest`
* [ ] Maps quotas/limits → `ResourceExhausted` (+ `retry_after_ms`)
* [ ] Maps transient issues → `Unavailable` / retryable
* [ ] Maps timeouts → `DeadlineExceeded`
* [ ] Maps unsupported → `NotSupported`
* [ ] Emits normalized `code` and attributes

### ✅ Phase 7: Deadline Enforcement

* [ ] Correct budget computation
* [ ] Preflight deadline checks where applicable
* [ ] Honors deadlines in unary calls
* [ ] Honors deadlines mid-stream
* [ ] Accurate budget calculations

### ✅ Phase 8: Observability & Privacy

* [ ] Never logs raw tenant IDs
* [ ] Uses tenant hash in metrics
* [ ] Excludes prompt content from metrics
* [ ] Emits token usage metrics
* [ ] Emits metrics on both success and error paths
* [ ] Maintains tenant isolation

### ✅ Phase 9: Wire Contract & Envelopes

* [ ] `WireLLMHandler` implements all `llm.*` operations
* [ ] Success envelopes have correct `{ok, code, ms, result}` shape
* [ ] Error envelopes normalize to `{ok=false, code, error, message, ...}`
* [ ] `OperationContext` properly constructed from wire `ctx`
* [ ] Unknown fields ignored in requests
* [ ] Unknown operations map to `NotSupported`
* [ ] Unexpected exceptions map to `Unavailable`

## Conformance Badge

```
✅ LLM Protocol V1.0 - 100% Conformant
   111/111 tests passing

   ✅ Core Operations: 6/6 (100%)
   ✅ Message Validation: 15/15 (100%)
   ✅ Sampling Parameters: 41/41 (100%)
   ✅ Streaming Semantics: 6/6 (100%)
   ✅ Error Handling: 4/4 (100%)
   ✅ Capabilities Discovery: 12/12 (100%)
   ✅ Observability & Privacy: 6/6 (100%)
   ✅ Deadline Semantics: 5/5 (100%)
   ✅ Token Counting: 6/6 (100%)
   ✅ Health Endpoint: 6/6 (100%)
   ✅ Wire Envelopes & Routing: 8/8 (100%)

   Status: Production Ready
```

## Maintenance

### Adding New Tests

1. Create test file: `test_<feature>_<aspect>.py`
2. Add SPDX license header and spec references in a docstring
3. Use `pytestmark = pytest.mark.asyncio` for async tests
4. Update this `CONFORMANCE.md` with new coverage
5. Update the summary and badge

### Updating for Specification Changes

1. Review `SPECIFICATION.md` changelog (Appendix F)
2. Identify new/changed requirements in §8 / cross-protocol sections
3. Add/update tests accordingly
4. Update protocol version / date in this document
5. Update the conformance badge

## Related Documentation

* `../../SPECIFICATION.md` - Full protocol specification (§8 LLM Protocol)
* `../../ERRORS.md` - Error taxonomy reference
* `../../METRICS.md` - Observability guidelines
* `../README.md` - General testing guidelines

---

**Last Updated:** 2025-01-XX
**Maintained By:** Corpus SDK Team
**Status:** 100% V1.0 Conformant - Production Ready
```
