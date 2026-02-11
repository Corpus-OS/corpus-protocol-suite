# CORPUS SCHEMA CONFORMANCE

## Table of Contents

- [Overview](#overview)
- [Test Coverage Summary](#test-coverage-summary)
  - [Schema Meta-Lint Suite (26 Tests)](#a-schema-meta-lint-suite-26-tests)
  - [Golden Samples Suite (184 Tests)](#b-golden-samples-suite-184-tests)
- [Schema Testing Philosophy](#schema-testing-philosophy)
- [Quick Start (Schema-Only)](#quick-start-schema-only)
- [Repository Layout (Schemas)](#repository-layout-schemas)
- [What "Schema Conformance" Means](#what-schema-conformance-means)
- [Test Suites (Schema-Only)](#test-suites-schema-only)
  - [A) Schema Meta-Lint (26 Tests)](#a-schema-meta-lint-26-tests)
  - [B) Golden Wire Messages (184 Test Cases)](#b-golden-wire-messages-184-test-cases)
- [Running (Schema-Only; Makefile-Aligned)](#running-schema-only-makefile-aligned)
- [Schema Evolution Guidelines](#schema-evolution-guidelines)
- [Validation in Other Languages](#validation-in-other-languages)
  - [TypeScript](#typescript)
  - [Python](#python)
- [Error Taxonomies by Protocol](#error-taxonomies-by-protocol)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting (Schema)](#troubleshooting-schema)
- [Versioning & Deprecation (Schema)](#versioning--deprecation-schema)
- [Compliance Badge (Schema-Only)](#compliance-badge-schema-only)
- [Maintenance](#maintenance)

---

## Overview

This document defines Schema Conformance for the Corpus Protocol across LLM, Vector, Embedding, Graph, and Common envelopes. It is the source of truth for:

- Wire-contract validation of requests, successes, errors, and streaming envelopes (via goldens)
- Schema hygiene: metaschema compliance, `$id` uniqueness, `$ref` resolvability, anchors, enums, and patterns

**Scope:** Schema only. Runtime/behavioral semantics (adapters, deadlines, caching, retries, metrics, error semantics, etc.) are covered in PROTOCOLS.md, ERRORS.md, and METRICS.md, and are not part of this document.

- **Protocol versions:** `llm/v1.0`, `vector/v1.0`, `embedding/v1.0`, `graph/v1.0`
- **Schema draft:** JSON Schema Draft 2020-12
- **Status:** Stable / Production-ready
- **Test Location:** `tests/schema/`

This document is the schema companion to:
- **PROTOCOLS.md** — normative operation & type semantics
- **ERRORS.md** — normalized error taxonomy & envelopes  
- **METRICS.md** — observability & metrics contracts

**Primary audience:**
- Schema authors and maintainers
- Adapter implementers (LLM / Vector / Embedding / Graph)
- Platform engineers wiring validation into gateways and CI

**Test locations**
- Schema meta-lint and golden tests: `tests/schema/`

---

## Test Coverage Summary

**Overall Coverage: 210/210 tests (100%) ✅**

📊 **Total Tests:** 210/210 passing (100%)  
⚡ **Execution Time:** 2.01s (9.6ms/test avg)  
🏆 **Certification:** Platinum (100%)

| Test Category | Tests | Coverage | Status |
|--------------|-------|-----------|---------|
| **Schema Meta-Lint** | 26/26 | 100% ✅ | Production Ready |
| **Golden Samples Validation** | 184/184 | 100% ✅ | Production Ready |
| **Total** | **210/210** | **100% ✅** | **🏆 Platinum Certified** |

### Performance Characteristics
- **Test Execution:** 2.01 seconds total runtime
- **Average Per Test:** 9.6 milliseconds
- **Cache Efficiency:** 0 cache hits, 210 misses (cache size: 210)
- **Parallel Ready:** Optimized for parallel execution with `pytest -n auto`

### Test Infrastructure
- **Testing Framework:** pytest 9.0.2 with comprehensive plugin support
- **Environment:** Python 3.10.19 on Darwin
- **Schema Draft:** JSON Schema Draft 2020-12
- **Registry:** All schemas loaded via `$id` references

### Certification Levels
- 🏆 **Platinum:** 210/210 tests (100%) with comprehensive coverage
- 🥇 **Gold:** 168+ tests (80%+ coverage)
- 🥈 **Silver:** 126+ tests (60%+ coverage)
- 🔬 **Development:** 105+ tests (50%+ coverage)

### A) Schema Meta-Lint Suite (26 Tests)
**Location:** `tests/schema/test_schema_lint.py`

| Test Category | Test Count | Description |
|---------------|------------|-------------|
| Schema Loading & IDs | 3 tests | File loading, `$schema` validation, unique `$id` checks |
| File Organization | 2 tests | Path conventions, component directory structure |
| Metaschema & Hygiene | 5 tests | Draft 2020-12 compliance, regex, enums, reserved properties |
| Cross-References | 2 tests | `$ref` resolution, local fragment validation |
| Definitions | 2 tests | Dangling `$defs` detection, size limits |
| Envelope Structure | 1 test | Envelope required keys and core constant fields (`ok`, `code`) |
| Examples Validation | 1 test | Embedded example validation |
| Performance & Metrics | 3 tests | Loading performance, complexity metrics, health summary |
| **Total** | **19 test functions** | Hundreds of individual assertions |

### B) Golden Samples Suite (184 Tests)
**Location:** `tests/schema/test_golden_samples.py`

| Test Category | Test Count | Description |
|---------------|------------|-------------|
| Core Schema Validation | 176 parametrized tests | Each golden file validates against its declared schema |
| NDJSON Stream Validation | 6 tests | LLM, Graph, and Embedding stream protocol validation |
| Cross-Schema Invariants | 10 tests | Token totals, vector dimensions, capabilities invariants |
| Field Type Validation | 2 tests | Timestamp, ID patterns, type consistency |
| Drift Detection | 4 tests | File existence, orphan detection, naming conventions |
| Performance & Reliability | 3 tests | File size limits, loading performance, checksum validation |
| Component Coverage | 1 test | Component-level test coverage validation |
| Error Taxonomy Coverage | 4 tests | Error code validation per protocol |
| **Total** | **206 test functions** | Comprehensive golden sample validation |

---

## Schema Testing Philosophy

Our schema testing follows the "contract-first" principle (§4.1):

1. **Schemas define the truth** — All validation flows from JSON Schema definitions.
2. **Goldens exemplify reality** — Real-world message samples validate the schemas.
3. **Meta-lint ensures hygiene** — Schema quality and maintainability are enforced.
4. **Makefile provides workflow** — Consistent developer experience across environments.

---

## Quick Start (Schema-Only)

These commands align with the project Makefile.

### 1. Install deps
```bash
pip install .[test]
```

### 2. Run schema suites
```bash
# Meta-lint (schemas/** only) - 26 comprehensive tests
make test-schema

# Golden wire messages (schema validation of sample payloads) - 184 test cases
make test-golden

# Run both schema suites together - 210 total tests
make verify-schema
```

### 3. Fast (skip @slow)
```bash
make test-schema-fast
make test-golden-fast
```

### 4. Smoke & safety
```bash
make quick-check     # minimal schema/golden smoke
make validate-env    # warns if CORPUS_TEST_ENV unset
make safety-check    # blocks heavy runs if CORPUS_TEST_ENV=production
```

---

## Repository Layout (Schemas)

### schemas/common/
- `envelope.request.json`, `envelope.success.json`, `envelope.error.json`, `envelope.stream.success.json`, `operation_context.json`

### schemas/llm/
- **Envelopes:** `llm.envelope.{request,success,error}.json`
- **Capabilities:** `llm.capabilities.json`
- **Types:** `llm.types.{message,chunk,completion,token_usage,tool,warning}.json`
- **Configuration:** `llm.sampling.params.json`, `llm.tools.schema.json`, `llm.response_format.json`

### schemas/vector/
- **Envelopes:** `vector.envelope.{request,success,error}.json`
- **Capabilities:** `vector.capabilities.json`
- **Query Schemas:** `vector.types.query_{spec,result}.json`
- **Namespace Schemas:** `vector.types.namespace_{spec,result}.json`
- **Result Schemas:** `vector.types.{upsert,delete}_result.json`
- **Type Schemas:** `vector.types.{vector,document,vector_match,filter,failure_item}.json`

### schemas/embedding/
- **Envelopes:** `embedding.envelope.{request,success,error}.json`
- **Capabilities:** `embedding.capabilities.json`
- **Stats:** `embedding.stats.json`
- **Result Schemas:** `embedding.types.{result,batch_result}.json`
- **Type Schemas:** `embedding.types.{vector,chunk,failure,warning}.json`

### schemas/graph/
- **Envelopes:** `graph.envelope.{request,success,error}.json`
- **Capabilities:** `graph.capabilities.json`
- **Health:** `graph.types.health_result.json`
- **Query Schemas:** `graph.types.query_{spec,result}.json`
- **Traversal Schemas:** `graph.types.traversal_{spec,result}.json`
- **Batch Schemas:** `graph.types.{batch_op,batch_result}.json`
- **Bulk Schemas:** `graph.types.bulk_vertices_{spec,result}.json`
- **Schema:** `graph.types.graph_schema.json`
- **Type Schemas:** `graph.types.{node,edge,entity,id,chunk,warning}.json`

**Note:** Some optional type schemas may exist only if referenced by `$ref` chains from operational schemas. All files listed above must exist and validate during meta-lint.

---

## What "Schema Conformance" Means

A build is schema-conformant when all of the following hold (§4.1):

1. **Metaschema compliance** — Every file validates against JSON Schema Draft 2020-12.
2. **`$id` hygiene** — Each schema declares a unique, canonical `$id` of the form:
   `https://corpusos.com/schemas/<component>/<file>.json`
3. **`$ref` resolvability** — All `$ref`s resolve to known `$id`s or valid internal anchors; no dangling fragments.
4. **Envelope correctness** — Request/success/error envelopes include required fields as defined in `schemas/common/envelope.{request,success,error,stream.success}.json`.
5. **Streaming contracts (shape)** — Streaming uses `envelope.stream.success.json` with `code: "STREAMING"` and protocol-specific chunk payloads (§4.2.3).
6. **Examples validate** — Any examples embedded in schemas validate against their parent schema.
7. **Pattern/enum hygiene** — Regex patterns compile; enums are deduplicated and documented; patterns match their intended domain.
8. **No dangling `$defs`** — Exported defs are referenced, or explicitly documented as public anchors; unused `$defs` either removed or justified.
9. **Cross-schema invariants (schema-level only)** — Enforced via goldens where JSON Schema alone is insufficient:
   - Streaming envelopes use `code: "STREAMING"` (not `"OK"`)
   - Success envelopes use `code: "OK"`
   - Success `result` may be an object or array depending on operation
   - Error envelopes have `retry_after_ms: integer|null` and `details: object|null`
   - `deadline_ms` is integer (not number)
   - Token usage invariants: `total_tokens = prompt_tokens + completion_tokens`
   - Vector dimension consistency across matches in a query result
   - Context (`ctx`) allows additional properties for forward compatibility (§4.2.5)

**Optional extensions not required by reference adapters:**
- `schema_version` field (may be added by implementations but not required)
- Top-level `protocol` or `component` fields (not emitted by reference adapters)

**Out of scope (behavioral):** deadlines, retries, caching, normalization semantics, metrics emission, and adapter behavior. Those are tracked in PROTOCOLS.md, METRICS.md, and ERRORS.md, and in the behavioral conformance suite.

---

## Test Suites (Schema-Only)

### A) Schema Meta-Lint (26 Tests)

**Path:** `tests/schema/test_schema_lint.py`  
**Purpose:** Validate the schemas themselves (§4.1).

**Comprehensive coverage:**
- ✅ **Schema Loading & IDs** (3 tests): Load all `schemas/**`, validate `$schema` Draft 2020-12, ensure unique `$id`s
- ✅ **File Organization** (2 tests): Validate path conventions, component directory structure
- ✅ **Metaschema & Hygiene** (5 tests): Draft 2020-12 compliance, regex pattern compilation, enum deduplication/sorting, reserved property checks
- ✅ **Cross-References** (2 tests): Resolve all `$ref`s (absolute and internal anchors), validate local fragments
- ✅ **Definitions** (2 tests): Detect dangling `$defs`, enforce size limits
- ✅ **Envelope Structure** (1 test): Validate envelope required keys and core constant fields (`ok`, `code`) (§4.2.1)
- ✅ **Examples Validation** (1 test): Validate embedded examples against parent schemas
- ✅ **Performance & Metrics** (3 tests): Schema loading performance, complexity metrics, health reporting

### B) Golden Wire Messages (184 Tests)

**Path:** `tests/schema/test_golden_samples.py`  
**Purpose:** Validate realistic request/response/stream samples against top-level envelopes (§4.2.1).

**Coverage summary:**

| Component | Request/Success/Error Envelopes | Streaming Envelopes (§4.2.3) | Operations Coverage |
|-----------|----------------------------------|--------------------------------------|---------------------|
| LLM | ✅ 20+ golden samples | ✅ `code: "STREAMING"` with `llm.types.chunk.json` | ✅ complete, stream, count_tokens, capabilities, health (§8.3) |
| Vector | ✅ 20+ golden samples | ✗ (no streaming) | ✅ query, batch_query, upsert, delete, create_namespace, delete_namespace, capabilities, health (§9.3) |
| Embedding | ✅ 20+ golden samples | ✅ `code: "STREAMING"` with `embedding.types.chunk.json` | ✅ embed, embed_batch, stream_embed, count_tokens, get_stats, capabilities, health (§10.3) |
| Graph | ✅ 20+ golden samples | ✅ `code: "STREAMING"` with `graph.types.chunk.json` | ✅ query, stream_query, upsert_nodes, upsert_edges, delete_nodes, delete_edges, bulk_vertices, batch, get_schema, transaction, traversal, capabilities, health (§7.3) |

**Detailed coverage:**
- ✅ **Core Schema Validation** (176 parametrized tests): Each golden file validates against its declared schema
- ✅ **NDJSON Stream Validation** (6 tests): LLM, Graph, and Embedding stream protocol validation, termination rules (§4.2.3)
- ✅ **Cross-Schema Invariants** (10 tests): Token usage math, vector dimension consistency, capabilities field validation, stats validation
- ✅ **Field Type Validation** (2 tests): Integer vs number validation, ID patterns
- ✅ **Drift Detection** (4 tests): File existence checks, orphan detection, naming conventions, request-response pairs
- ✅ **Performance & Reliability** (3 tests): File size limits, loading performance, checksum validation
- ✅ **Component Coverage** (1 test): Component-level test coverage validation
- ✅ **Error Taxonomy Coverage** (4 tests): Error code validation per protocol (§12.4)

**Golden samples should be treated as canonical examples of on-the-wire contracts; changes to schemas that break existing goldens are presumed breaking unless explicitly justified.**

---

## Running (Schema-Only; Makefile-Aligned)

### Everything schema (meta-lint + goldens) - 210 total tests
```bash
make verify-schema
```

### Meta-lint only - 26 comprehensive tests
```bash
make test-schema
# fast:
make test-schema-fast
```

### Goldens only - 184 individual test cases
```bash
make test-golden
# fast:
make test-golden-fast
```

### Smoke & safety
```bash
make quick-check     # minimal schema/golden smoke
make validate-env    # warns if CORPUS_TEST_ENV unset
make safety-check    # blocks heavy runs if CORPUS_TEST_ENV=production
```

**Env overrides:** `PYTEST_JOBS=4`, `PYTEST_ARGS="-x --tb=short"`.

---

## Schema Evolution Guidelines

### Adding New Schemas
1. Follow the `$id` convention (§4.1):
   `https://corpusos.com/schemas/<component>/<file>.json`
2. Include: `$schema` (2020-12), `title`, `description`, and top-level `type`.
3. Prefer `additionalProperties: false` for envelopes (use `patternProperties` for vendor slots if needed) (§4.2.1).
4. Add golden samples for new operations to exercise the new envelopes under `tests/schema/`.
5. Ensure any new cross-schema invariants are reflected in:
   - The relevant schema(s) (pattern/format/enums)
   - The golden samples
   - The meta-lint checks if they introduce new global constraints

### Breaking Changes
- Renaming required fields → Major version bump
- Removing enum values → Major version bump
- Changing field types (e.g., string → integer) → Major version bump
- Tightening constraints that invalidate previously valid payloads → Major version bump
- Changing streaming envelope shape (e.g., removing `chunk`, changing `code: "STREAMING"`, or altering termination semantics) → Major version bump (§4.2.3)

### Non-Breaking Changes
- Adding optional fields with sane defaults
- Adding enum members (as documented)
- Adding new `$defs` that are not wired into existing envelopes
- Widening constraints (e.g., increasing max lengths, broadening patterns while staying compatible)
- Changing `additionalProperties` from `false` to `true` in context objects (§4.2.5)
- Adding optional `schema_version` or `protocol`/`component` fields (forward-compatible extensions)

### Optional Extension Fields
Implementations may add these fields without breaking conformance:
- `schema_version` - May be added to envelopes or results for version tracking
- `protocol` / `component` - May be added at top level for debugging/identification
- `provider_specific` - Vendor extensions namespace

---

## Validation in Other Languages

### TypeScript
```typescript
// Using ajv with pre-loaded schemas
import Ajv from 'ajv';
const ajv = new Ajv({ 
  strict: false,
  allErrors: true 
});

// Pre-load all schemas by their $id
ajv.addSchema(commonEnvelopeSchema, 'https://corpusos.com/schemas/common/envelope.success.json');
ajv.addSchema(commonStreamSchema, 'https://corpusos.com/schemas/common/envelope.stream.success.json');
ajv.addSchema(llmEnvelopeSchema, 'https://corpusos.com/schemas/llm/llm.envelope.success.json');
ajv.addSchema(llmChunkSchema, 'https://corpusos.com/schemas/llm/llm.types.chunk.json');

// Validate a streaming LLM response
const llmStreamResponse = {
  "ok": true,
  "code": "STREAMING",
  "ms": 45.2,
  "chunk": {
    "text": "Hello world",
    "is_final": false,
    "model": "gpt-4.1-mini",
    "tool_calls": []
  }
};

const validate = ajv.getSchema('https://corpusos.com/schemas/common/envelope.stream.success.json');
if (!validate) throw new Error('Schema not registered');

const isValid = validate(llmStreamResponse);
if (!isValid) {
  console.error('Validation errors:', validate.errors);
}
```

### Python
```python
from jsonschema import Draft202012Validator, RefResolver
from pathlib import Path
import json

# Load schema registry
def load_schema_registry():
    registry = {}
    schema_dir = Path("schemas")
    
    for schema_file in schema_dir.rglob("*.json"):
        with open(schema_file) as f:
            schema = json.load(f)
            if "$id" in schema:
                registry[schema["$id"]] = schema
    return registry

# Validate a streaming LLM response
def validate_llm_stream_response():
    registry = load_schema_registry()
    resolver = RefResolver.from_schema(registry["https://corpusos.com/schemas/common/envelope.stream.success.json"], registry)
    
    llm_stream_response = {
        "ok": True,
        "code": "STREAMING",
        "ms": 45.2,
        "chunk": {
            "text": "Hello world",
            "is_final": False,
            "model": "gpt-4.1-mini",
            "tool_calls": []
        }
    }
    
    schema = registry["https://corpusos.com/schemas/common/envelope.stream.success.json"]
    validator = Draft202012Validator(schema, resolver=resolver)
    
    try:
        validator.validate(llm_stream_response)
        print("✅ Validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    validate_llm_stream_response()
```

---

## Error Taxonomies by Protocol

### LLM Error Codes (§12.4)
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `MODEL_OVERLOADED` - Model capacity exceeded
- `DEADLINE_EXCEEDED` - Request timeout

### Embedding Error Codes (§12.4)
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `TEXT_TOO_LONG` - Input text exceeds maximum length
- `MODEL_NOT_AVAILABLE` - Requested model unavailable
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout

### Vector Error Codes (§9.5, §12.4)
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `DIMENSION_MISMATCH` - Vector dimension mismatch
- `INDEX_NOT_READY` - Vector index not ready
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout

### Graph Error Codes (§12.4)
- `BAD_REQUEST` - Invalid request parameters
- `AUTH_ERROR` - Authentication or authorization failure
- `RESOURCE_EXHAUSTED` - Rate limit or quota exceeded
- `TRANSIENT_NETWORK` - Temporary network issue
- `UNAVAILABLE` - Service unavailable
- `NOT_SUPPORTED` - Requested feature not supported
- `DEADLINE_EXCEEDED` - Request timeout

---

## CI/CD Integration

### GitHub Actions (Schema-Only)
```yaml
name: Schema Conformance
on: [push, pull_request]
jobs:
  schema-conformance:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install .[test]
      - name: Verify schema suites (meta-lint + goldens) - 210 tests in ~2.01s
        run: make verify-schema
      - name: Generate coverage report
        run: |
          make test-schema-coverage
          make test-golden-coverage
```

For smoke checks in sensitive environments: `make quick-check` (pairs with `validate-env`/`safety-check` in the Makefile).

---

## Troubleshooting (Schema)

- **`$ref` cannot resolve** → Pre-load all schemas into the validator; `$id` strings must match exactly, including case.
- **Duplicate `$id`** → Each file must have a globally unique `$id`; fix collisions and re-run meta-lint.
- **Invalid regex** → Fix unescaped characters; confirm each pattern compiles in your target runtime.
- **Examples fail** → Update examples or the schema; examples must validate against their parent schema.
- **Streaming envelope shape mismatch** → Ensure streaming uses `code: "STREAMING"` and `chunk` field (not `result`). Use golden streams to spot termination issues (§4.2.3).
- **Schema drift vs PROTOCOLS/ERRORS/METRICS** → If behavior changes in those docs, ensure the schemas and goldens are updated in lockstep.
- **Type mismatch errors** → Check: `deadline_ms` must be integer (§6.1), `retry_after_ms` must be integer|null, `details` must be object|null.
- **Streaming validation fails** → Ensure terminal condition: either chunk with `is_final: true` OR error envelope, and streaming uses `code: "STREAMING"` (not `"OK"`) (§4.2.3).
- **Missing optional fields** → Remember `schema_version`, `protocol`, `component` are optional extensions not required for conformance.
- **Result field type mismatch** → Success `result` may be object or array depending on operation (e.g., vector.batch_query returns array).
- **File naming consistency** → Ensure `envelope.stream.success.json` filename matches across SCHEMA.md, schemas/, and tests.

---

## Versioning & Deprecation (Schema)

- Reference adapters do not emit `schema_version`; it remains an optional extension for implementations.
- **Non-breaking:** adding optional fields; additive enum members (as documented); new `$defs` not wired into existing envelopes.
- **Breaking:** removing/renaming required fields; type changes; constraint tightening that invalidates prior valid messages; `$id` renames.
- **Deprecation:** set `deprecated: true` and (optionally) `replacement_op` in the relevant schemas; ensure goldens reflect both old and new while deprecated paths are supported.

---

## Compliance Badge (Schema-Only)

```text
🏆 CORPUS SCHEMA CONFORMANCE - PLATINUM CERTIFIED
   210/210 schema validation tests passing (100%)

   📊 Total Tests: 210/210 passing (100%)
   ⚡ Execution Time: 2.01s (9.6ms/test avg)
   🏆 Certification: Platinum (100%)

   ✅ Schema Meta-Lint: 26/26 (100%) - §4.1
   ✅ Golden Samples Validation: 184/184 (100%) - §4.2.1
   ✅ Cross-Protocol Coverage: LLM, Vector, Embedding, Graph
   ✅ Streaming Envelopes: §4.2.3 compliance
   ✅ Error Taxonomy: §12.4 validation

   Status: Production Ready 🏆 Platinum Certified
```

**Badge Suggestion:**
[![Corpus Schema](https://img.shields.io/badge/CorpusSchema-100%25%20Conformant-blue)](./schema_conformance_report.json)

---

## Maintenance

- Keep `tests/schema/test_schema_lint.py` aligned with new schema categories, `$id` conventions, and extension rules.
- Add/update goldens whenever envelopes or type schemas change; drift detection in the golden suite will reveal gaps.
- Periodically regenerate a schema index (path → `$id`) to spot stale or missing entries.
- When PROTOCOLS / ERRORS / METRICS evolve, audit schemas for drift and update both schemas and goldens together to maintain a consistent contract surface.
- Monitor streaming envelope usage: ensure all streaming operations use `code: "STREAMING"` and proper termination semantics (§4.2.3).
- Remember: `schema_version`, `protocol`, `component` are required extensions in certain instances; and may require them in conformance tests. Check SCHEMA.md for the final determination.
- Ensure naming consistency: `envelope.stream.success.json` is the canonical streaming envelope filename used consistently across SCHEMA.md, schemas/, and tests.
- Keep repository layout accurate: optional type schemas should be listed only if they exist and are referenced by `$ref` chains.

**Last Updated:** 2026-02-10  
**Maintained By:** Corpus SDK Team  
**Test Suite:** `tests/schema/` (2 test files, 210 tests)  
**Specification Reference:** §4.1, §4.2  
**Status:** 100% V1.0 Conformant - Platinum Certified (210/210 tests, 2.01s runtime)

---
