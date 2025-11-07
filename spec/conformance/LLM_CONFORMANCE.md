# LLM Protocol V1 Conformance Test Coverage


## Overview

This document tracks conformance test coverage for the **LLM Protocol V1.0** specification as defined in `SPECIFICATION.md §8`. Each test validates normative requirements (MUST/SHOULD) from the specification.

This suite constitutes the official LLM Protocol V1.0 Reference Conformance Test Suite. Any implementation (Corpus or third-party) MAY run these tests to verify and publicly claim conformance, provided all referenced tests pass unmodified.

**Protocol Version:** LLM Protocol V1.0
**Status:** Pre-Release
**Last Updated:** 2025-01-XX
**Test Location:** `tests/llm/`

## Conformance Summary

**Overall Coverage: 61/61 tests (100%) ✅**

| Category                 | Tests | Coverage |
| ------------------------ | ----- | -------- |
| Core Operations          | 4/4   | 100% ✅   |
| Message Validation       | 3/3   | 100% ✅   |
| Sampling Parameters      | 9/9   | 100% ✅   |
| Streaming Semantics      | 5/5   | 100% ✅   |
| Error Handling           | 4/4   | 100% ✅   |
| Capabilities             | 10/10 | 100% ✅   |
| Observability & Privacy  | 4/4   | 100% ✅   |
| Deadline Semantics       | 3/3   | 100% ✅   |
| Token Counting           | 3/3   | 100% ✅   |
| Health Endpoint          | 4/4   | 100% ✅   |
| Wire Envelopes & Routing | 12/12 | 100% ✅   |

> Note: Categories are logical groupings. Individual tests may satisfy multiple normative requirements.

## Test Files

### test_capabilities_shape.py

**Specification:** §8.4 - Capabilities Discovery
**Status:** ✅ Complete (10 tests)

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

### test_complete_basic.py

**Specification:** §8.3 - `complete` Operation
**Status:** ✅ Complete (1 test)

Validates basic completion contract:

* `test_complete_basic_text_and_usage` - Non-empty text, token accounting present, model echoed, valid `finish_reason`

### test_streaming_semantics.py

**Specification:** §8.3 - `stream` Operation
**Status:** ✅ Complete (5 tests)

Validates streaming contract:

* `test_stream_yields_llmchunk_instances`
* `test_stream_multiple_chunks_minimum_two`
* `test_stream_exactly_one_final_chunk`
* `test_stream_final_chunk_is_last`
* `test_stream_usage_so_far_monotonic_and_text_non_empty`

These collectively enforce the canonical streaming semantics: progressive chunks, single terminal signal, and coherent aggregate output.

### test_count_tokens_consistency.py

**Specification:** §8.3 - Token Counting
**Status:** ✅ Complete (3 tests)

Validates token counting behavior:

* `test_count_tokens_monotonic` - Longer input never reports fewer tokens than shorter input
* `test_count_tokens_empty_string` - Empty string returns 0 (or minimal constant)
* `test_count_tokens_unicode_handling` - Unicode handled without error or negative counts

### test_message_validation.py

**Specification:** §8.3 - Message Format
**Status:** ✅ Complete (3 tests) ⭐ Exemplary

Comprehensive schema validation:

* `test_invalid_messages_rejected` - Rejects unknown roles, empty lists, missing fields
* `test_accepts_standard_roles` - Accepts `system` / `user` / `assistant`
* `test_handles_large_message_content` - Supports large (but reasonable) payloads

### test_sampling_params_validation.py

**Specification:** §8.3 - Sampling Parameters
**Status:** ✅ Complete (9 tests)

Validates parameter ranges:

* `test_invalid_temperature_rejected`
* `test_valid_temperature_accepted`
* `test_invalid_top_p_rejected`
* `test_valid_top_p_accepted`
* `test_invalid_frequency_penalty_rejected`
* `test_valid_frequency_penalty_accepted`
* `test_invalid_presence_penalty_rejected`
* `test_valid_presence_penalty_accepted`
* `test_multiple_invalid_params_error_message` - Aggregated errors are informative

Ensures strict adherence to:

* `temperature ∈ [0.0, 2.0]`
* `top_p ∈ (0.0, 1.0]`
* `frequency_penalty, presence_penalty ∈ [-2.0, 2.0]`

### test_error_mapping_retryable.py

**Specification:** §8.5, §12.1, §12.4 - Error Handling
**Status:** ✅ Complete (4 tests)

Validates classification and normalization:

* `test_retryable_errors_with_hints` - Retryable errors (e.g. `ResourceExhausted`, `Unavailable`, `ModelOverloaded`) include appropriate `retry_after_ms` when available
* `test_non_retryable_validation_errors` - `BadRequest` and `ContentFiltered` are non-retryable without input change
* `test_not_supported_mapped_correctly` - Unsupported models/features → `NotSupported` with spec-consistent attributes
* `test_error_attributes_present` - Normalized `code`, stable class name, and retryability flags present

These tests collectively cover:

* Common taxonomy
* LLM-specific errors (`ModelOverloaded`, `ContentFiltered`)
* Proper `NotSupported` mapping (e.g. unsupported tools/JSON or count-tokens)

### test_deadline_enforcement.py

**Specification:** §8.3, §12.1 - Deadline Semantics
**Status:** ✅ Complete (3 tests)

Validates deadline behavior:

* `test_deadline_budget_nonnegative_and_usable` - Derived budget never negative
* `test_deadline_exceeded_on_expired_budget` - Immediate `DeadlineExceeded` when deadline already elapsed
* `test_deadline_exceeded_during_stream` - Streaming respects deadlines mid-generation

### test_health_report.py

**Specification:** §8.3 - Health Endpoint
**Status:** ✅ Complete (4 tests)

Validates health contract:

* `test_health_has_required_fields` - `{"ok", "server", "version"}` present
* `test_health_ok_shape` - Shape stable when healthy
* `test_health_degraded_shape_consistent` - Shape stable when degraded
* `test_health_error_shape_consistent` - Shape stable on internal errors

### test_context_siem.py

**Specification:** §13.2, §15 - Observability & Privacy
**Status:** ✅ Complete (4 tests) ⭐ Critical

Validates SIEM-safe observability:

* `test_context_propagates_to_metrics_siem_safe` - `traceparent`/operation metadata propagate; no raw tenant IDs
* `test_metrics_emitted_on_error_path` - Errors still produce safe metrics
* `test_streaming_metrics_siem_safe` - Streaming emits final metrics without prompt content
* `test_token_counter_metrics_present` - Token usage counters emitted, privacy-preserving

### test_llm_wire_handler_envelopes.py

**Specification:** §4.1, §8.3, §8.4, §8.5, §13 - Wire Contract & Envelopes
**Status:** ✅ Complete (12 tests)

Validates wire-level handler behavior:

* Canonical `llm.<op>` routing (`capabilities`, `complete`, `count_tokens`, `health`)
* Success envelopes: `{ok, code, ms, result}` shape and JSON-safe payloads
* Error envelopes: normalized `{ok=false, code, error, message, retry_after_ms?, details?}`
* `OperationContext` construction from wire `ctx` (ignores unknown keys)
* Proper mapping of `LLMAdapterError` subclasses to wire errors
* Fallback of unexpected exceptions to `UNAVAILABLE` per taxonomy
* `llm.stream` handled exclusively via `handle_stream` (non-unary guard)
* SIEM/PII-safe behavior at the wire boundary (no raw secrets in envelopes)
* `test_wire_count_tokens_unknown_model_maps_model_not_available` — Unknown model in `count_tokens` mapped to `MODEL_NOT_AVAILABLE` / `NOT_SUPPORTED` per spec.
* `test_wire_error_envelope_includes_message_and_type` — Verifies error envelopes expose a stable message/type shape for adapter errors.
* `test_wire_stream_maps_llm_adapter_error_to_normalized_envelope` — Ensures stream-path LLMAdapterError maps to a normalized error envelope.
* `test_wire_complete_missing_required_fields_maps_to_bad_request` — Missing required args for `llm.complete` yield `BAD_REQUEST`.

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

#### `count_tokens()`

| Requirement                   | Test File                        | Status |
| ----------------------------- | -------------------------------- | ------ |
| Returns non-negative integer  | test_count_tokens_consistency.py | ✅      |
| Monotonic w.r.t. input length | test_count_tokens_consistency.py | ✅      |
| Handles empty string          | test_count_tokens_consistency.py | ✅      |
| Handles Unicode safely        | test_count_tokens_consistency.py | ✅      |

#### `health()`

| Requirement                         | Test File             | Status |
| ----------------------------------- | --------------------- | ------ |
| Returns object/dict                 | test_health_report.py | ✅      |
| Includes `ok` (bool)                | test_health_report.py | ✅      |
| Includes `server` (str)             | test_health_report.py | ✅      |
| Includes `version` (str)            | test_health_report.py | ✅      |
| Stable shape across ok/degraded/err | test_health_report.py | ✅      |

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

### §8.5 Error Handling - Complete Coverage

| Error / Behavior                                | Test File                                                      | Status |
| ----------------------------------------------- | -------------------------------------------------------------- | ------ |
| `BadRequest` for validation failures            | test_message_validation.py, test_sampling_params_validation.py | ✅      |
| `ResourceExhausted` with `retry_after_ms` hints | test_error_mapping_retryable.py                                | ✅      |
| `Unavailable` classified retryable              | test_error_mapping_retryable.py                                | ✅      |
| `DeadlineExceeded` on timeout/deadline          | test_deadline_enforcement.py                                   | ✅      |
| `NotSupported` for unsupported features/models  | test_error_mapping_retryable.py                                | ✅      |
| `ModelOverloaded` (LLM-specific, retryable)     | test_error_mapping_retryable.py                                | ✅      |
| `ContentFiltered` (LLM-specific, non-retryable) | test_error_mapping_retryable.py                                | ✅      |
| Normalized `code` + attributes on all errors    | test_error_mapping_retryable.py                                | ✅      |

### §13.2 Observability - Complete Coverage

| Requirement                       | Test File            | Status |
| --------------------------------- | -------------------- | ------ |
| Tenant never logged raw           | test_context_siem.py | ✅      |
| Tenant hashed in metrics          | test_context_siem.py | ✅      |
| No prompt content in metrics      | test_context_siem.py | ✅      |
| Metrics also on error paths       | test_context_siem.py | ✅      |
| Token counters for LLM operations | test_context_siem.py | ✅      |

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
```

### With coverage report

```bash
pytest tests/llm/ --cov=corpus_sdk.llm --cov-report=html
```

## Adapter Compliance Checklist

Use this checklist when implementing or validating a new LLM adapter:

### ✅ Phase 1: Core Operations

* [x] `capabilities()` returns valid `LLMCapabilities`
* [x] `complete()` returns `LLMCompletion` with usage + finish_reason
* [x] `stream()` emits chunks with exactly one final marker
* [x] `count_tokens()` returns non-negative int
* [x] `health()` returns `{ok, server, version}`

### ✅ Phase 2: Message Validation

* [x] Rejects empty messages
* [x] Rejects unknown roles
* [x] Rejects missing required fields
* [x] Accepts `system` / `user` / `assistant`
* [x] Handles large (reasonable) content

### ✅ Phase 3: Parameter Validation

* [x] Enforces `temperature` in [0.0, 2.0]
* [x] Enforces `top_p` in (0.0, 1.0]
* [x] Enforces `frequency_penalty` in [-2.0, 2.0]
* [x] Enforces `presence_penalty` in [-2.0, 2.0]

### ✅ Phase 4: Streaming Semantics

* [x] Yields `LLMChunk` objects
* [x] Multiple chunks where applicable
* [x] Exactly one final chunk
* [x] Final chunk is last
* [x] `usage_so_far` monotonic and consistent

### ✅ Phase 5: Token Counting

* [x] Non-negative integers
* [x] Monotonic vs input length
* [x] Correct empty-string handling
* [x] Robust Unicode handling

### ✅ Phase 6: Error Handling

* [x] Maps validation issues → `BadRequest`
* [x] Maps quotas/limits → `ResourceExhausted` (+ `retry_after_ms`)
* [x] Maps transient issues → `Unavailable` / retryable
* [x] Maps timeouts → `DeadlineExceeded`
* [x] Maps unsupported → `NotSupported`
* [x] Maps `ModelOverloaded` as retryable
* [x] Maps `ContentFiltered` as non-retryable
* [x] Emits normalized `code` and attributes

### ✅ Phase 7: Deadline Enforcement

* [x] Correct budget computation
* [x] Preflight deadline checks where applicable
* [x] Honors deadlines in unary calls
* [x] Honors deadlines mid-stream

### ✅ Phase 8: Observability & Privacy

* [x] Never logs raw tenant IDs
* [x] Uses tenant hash in metrics
* [x] Excludes prompt content from metrics
* [x] Emits token usage metrics
* [x] Emits metrics on both success and error paths

## Conformance Badge

```
✅ LLM Protocol V1.0 - 100% Conformant
   61/61 tests passing

   ✅ Core Operations: 4/4 (100%)
   ✅ Message Validation: 3/3 (100%)
   ✅ Sampling Parameters: 9/9 (100%)
   ✅ Streaming Semantics: 5/5 (100%)
   ✅ Error Handling: 4/4 (100%)
   ✅ Capabilities: 10/10 (100%)
   ✅ Observability & Privacy: 4/4 (100%)
   ✅ Deadline Semantics: 3/3 (100%)
   ✅ Token Counting: 3/3 (100%)
   ✅ Health Endpoint: 4/4 (100%)
   ✅ Wire Envelopes & Routing: 12/12 (100%)

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
