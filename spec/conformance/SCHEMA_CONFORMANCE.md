# SCHEMA_CONFORMANCE.md

**Corpus Protocol (v1.0) — Schema Conformance (Schema-Only)**  
**Components:** LLM / Vector / Embedding / Graph  
**JSON Schema Draft 2020-12**  
**Test suites:** Schema Meta-Lint (tests/schema/) & Golden Wire Messages (tests/golden/)

---

## Overview

This document defines Schema Conformance for the Corpus Protocol across LLM, Vector, Embedding, Graph, and Common envelopes. It is the source of truth for:

- Wire-contract validation of requests, successes, errors, and streaming frames (via goldens)
- Schema hygiene: metaschema compliance, `$id` uniqueness, `$ref` resolvability, anchors, enums, and patterns

**Scope:** Schema only. Runtime/behavioral semantics (adapters, deadlines, caching, retries, metrics, error semantics, etc.) are covered in PROTOCOLS.md, ERRORS.md, and METRICS.md, and are not part of this document.

- **Protocol versions:** `llm/v1.0`, `vector/v1.0`, `embedding/v1.0`, `graph/v1.0`
- **Schema draft:** JSON Schema Draft 2020-12
- **Status:** Stable / Production-ready

This document is the schema companion to:
- **PROTOCOLS.md** — normative operation & type semantics
- **ERRORS.md** — normalized error taxonomy & envelopes  
- **METRICS.md** — observability & metrics contracts

**Primary audience:**
- Schema authors and maintainers
- Adapter implementers (LLM / Vector / Embedding / Graph)
- Platform engineers wiring validation into gateways and CI

**Test locations**
- Schema meta-lint: `tests/schema/`
- Golden message validation: `tests/golden/`
- Utilities: `tests/utils/`

---

## Schema Testing Philosophy

Our schema testing follows the "contract-first" principle:

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
# Meta-lint (schemas/** only)
make test-schema

# Golden wire messages (schema validation of sample payloads)
make test-golden

# Run both schema suites together
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
- `envelope.request.json`, `envelope.success.json`, `envelope.error.json`, `operation_context.json`

### schemas/llm/
- **Envelopes:** `llm.envelope.{request,success,error}.json`
- **Ops:** `llm.complete.{request,success}.json`, `llm.count_tokens.{request,success}.json`, `llm.capabilities.success.json`, `llm.health.success.json`
- **Streaming:** `llm.stream.frame.{data,end,error}.json`, `llm.stream.frames.ndjson.schema.json`
- **Types/params:** `llm.types.{message,chunk,completion,token_usage,tool,warning,logprobs}.json`, `llm.sampling.params.json`, `llm.tools.schema.json`, `llm.response_format.json`

### schemas/vector/
- **Envelopes:** `vector.envelope.{request,success,error}.json`
- **Ops:** `vector.query.{request,success}.json`, `vector.upsert.{request,success}.json`, `vector.delete.{request,success}.json`
- **Namespace:** `vector.namespace.{create,delete}.{request,success}.json`
- **Health/Caps:** `vector.capabilities.{request,success}.json`, `vector.health.success.json`
- **Types:** `vector.types.{vector,vector_match,query_result,filter,partial_success_result,failure_item}.json`

### schemas/embedding/
- **Envelopes:** `embedding.envelope.{request,success,error}.json`
- **Ops:** `embedding.embed.{request,success}.json`, `embedding.embed_batch.{request,success}.json`, `embedding.count_tokens.{request,success}.json`
- **Health/Caps:** `embedding.capabilities.{request,success}.json`, `embedding.health.{request,success}.json`
- **Partial/types:** `embedding.partial_success.result.json`, `embedding.types.{vector,result,warning,failure}.json`

### schemas/graph/
- **Envelopes:** `graph.envelope.{request,success,error}.json`
- **Ops:** `graph.query.{request,success}.json`, `graph.stream_query.request.json`, `graph.vertex.{create,delete}.request.json`, `graph.edge.create.request.json`, `graph.batch.request.json`, `graph.id.success.json`, `graph.ack.success.json`
- **Health/Caps:** `graph.capabilities.{request,success}.json`, `graph.health.{request,success}.json`
- **Streaming:** `graph.stream.frame.{data,end,error}.json`, `graph.stream.frames.ndjson.schema.json`
- **Types:** `graph.types.{entity,id,row,batch_op,warning,partial_success_result}.json`

`*.types.*.json` files are validated indirectly via `$ref` chains from envelopes and stream frames. The schema meta-lint also validates each schema file in isolation for metaschema compliance, `$id`, `$ref`, and pattern/enum hygiene.

---

## What "Schema Conformance" Means

A build is schema-conformant when all of the following hold:

1. **Metaschema compliance** — Every file validates against JSON Schema Draft 2020-12.
2. **`$id` hygiene** — Each schema declares a unique, canonical `$id` of the form:
   `https://adaptersdk.org/schemas/<component>/<file>.json`
3. **`$ref` resolvability** — All `$ref`s resolve to known `$id`s or valid internal anchors; no dangling fragments.
4. **Envelope correctness** — Request/success/error envelopes include required fields, enums, and protocol/component constants as defined in `schemas/common/envelope.{request,success,error}.json`.
5. **Streaming contracts (shape)** — Frame schemas validate NDJSON/SSE/WebSocket data/end/error shapes for LLM and Graph streaming frames.
6. **Examples validate** — Any examples embedded in schemas validate against their parent schema.
7. **Pattern/enum hygiene** — Regex patterns compile; enums are deduplicated and documented; patterns match their intended domain (e.g., lower-hex, ISO 8601).
8. **No dangling `$defs`** — Exported defs are referenced, or explicitly documented as public anchors; unused `$defs` either removed or justified.
9. **Cross-schema invariants (schema-level only)** — Enforced via goldens where JSON Schema alone is insufficient:
   - `schema_version` present on success envelopes and matches SemVer (`^[0-9]+\.[0-9]+\.[0-9]+$`) — typically declared in `schemas/common/envelope.success.json`.
   - `result_hash` (when present) is a lower-hex string — e.g. `^[0-9a-f]{64}$` for SHA-256.
   - Identifier patterns (e.g., `request_id`, `id`) conform to documented patterns (UUID, slug, or lower-hex) as defined in their respective schemas.
   - Timestamps use format: `"date-time"` or an equivalent documented pattern (e.g. ISO 8601) as defined in `operation_context.json` and envelope schemas.
   - Partial-success envelopes include the minimal accounting fields (`processed_count`, `failed_count`, and `failures[]`) in line with `*.partial_success_result.json`.

**Out of scope (behavioral):** deadlines, retries, caching, normalization semantics, metrics emission, and adapter behavior. Those are tracked in PROTOCOLS.md, METRICS.md, and ERRORS.md, and in the behavioral conformance suite.

---

## Test Suites (Schema-Only)

### A) Schema Meta-Lint

**Path:** `tests/schema/test_schema_lint.py`  
**Purpose:** Validate the schemas themselves.

**Checks include:**
- Load all `schemas/**` and validate against the Draft 2020-12 metaschema.
- Build a `$id` index; detect duplicates and missing `$id`s.
- Resolve all `$ref`s (absolute and internal anchors) and ensure no dangling fragments.
- Compile regex patterns and sanity-check enums (no empty or duplicate values).
- Validate embedded examples arrays (when present).
- Enforce envelope constants (protocol/component) where specified in common envelope schemas.

### B) Golden Wire Messages

**Path:** `tests/golden/test_golden_samples.py`  
**Purpose:** Validate realistic request/response/stream samples against top-level envelopes and frame schemas.

**Coverage summary:**

| Component | Request/Success/Error Envelopes | Streaming Frames (data/end/error) | NDJSON Union Schema |
|-----------|----------------------------------|-----------------------------------|---------------------|
| LLM | ✅ | ✅ | ✅ (`llm.stream.frames.ndjson.schema.json`) |
| Vector | ✅ | ✗ | ✗ |
| Embedding | ✅ | ✗ | ✗ |
| Graph | ✅ | ✅ | ✅ (`graph.stream.frames.ndjson.schema.json`) |

**Covers:**
- LLM / Vector / Embedding / Graph: request & success/error envelopes.
- LLM & Graph streaming: data/end/error frame schemas and NDJSON stream union schemas where defined.

**Also enforces (schema-level invariants only):**
- `schema_version` present and SemVer-conformant on success envelopes.
- Minimal partial-success accounting fields present where used (`processed_count`, `failed_count`, `failures[]`).
- Identifier/timestamp pattern sanity (IDs, `request_id`, and timestamps conform to their documented patterns).
- Single terminal frame constraint for streaming:
  - Exactly one terminal frame per stream (`type: "end"` or `type: "error"`, but not both).
  - Ordering and shape rules are asserted over golden NDJSON sequences via a dedicated stream utility.

**Golden samples should be treated as canonical examples of on-the-wire contracts; changes to schemas that break existing goldens are presumed breaking unless explicitly justified.**

---

## Running (Schema-Only; Makefile-Aligned)

### Everything schema (meta-lint + goldens)
```bash
make verify-schema
```

### Meta-lint only
```bash
make test-schema
# fast:
make test-schema-fast
```

### Goldens only
```bash
make test-golden
# fast:
make test-golden-fast
```

### Smoke & safety
```bash
make quick-check
make validate-env
make safety-check
```

**Env overrides:** `PYTEST_JOBS=4`, `PYTEST_ARGS="-x --tb=short"`.

---

## Schema Evolution Guidelines

### Adding New Schemas
1. Follow the `$id` convention:
   `https://adaptersdk.org/schemas/<component>/<file>.json`
2. Include: `$schema` (2020-12), `title`, `description`, and top-level `type`.
3. Prefer `additionalProperties: false` for envelopes (use `patternProperties` for vendor slots if needed).
4. Add golden samples for new operations to exercise the new envelopes/frames under `tests/golden/`.
5. Ensure any new cross-schema invariants (e.g. new hash fields, version fields) are reflected in:
   - The relevant schema(s) (pattern/format/enums).
   - The golden samples.
   - The meta-lint checks if they introduce new global constraints.

### Breaking Changes
- Renaming required fields → Major version bump.
- Removing enum values → Major version bump.
- Changing field types (e.g., string → integer) → Major version bump.
- Tightening constraints that invalidate previously valid payloads (e.g., stricter patterns, lower max lengths) → Major version bump.

### Non-Breaking Changes
- Adding optional fields with sane defaults.
- Adding enum members (as documented).
- Adding new `$defs` that are not wired into existing envelopes.
- Widening constraints (e.g., increasing max lengths, broadening patterns while staying compatible).

Keep `schema_version` in success envelopes aligned with SemVer and maintain backward compatibility guidance in commit notes.

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
ajv.addSchema(commonEnvelopeSchema, 'https://adaptersdk.org/schemas/common/envelope.success.json');
ajv.addSchema(llmEnvelopeSchema, 'https://adaptersdk.org/schemas/llm/llm.envelope.success.json');
ajv.addSchema(llmCompleteSchema, 'https://adaptersdk.org/schemas/llm/llm.complete.success.json');

// Validate a complete LLM response
const llmResponse = {
  "ok": true,
  "code": "OK", 
  "ms": 45.2,
  "result": {
    "id": "chatcmpl-123",
    "model": "gpt-4",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello world"
      },
      "finish_reason": "stop"
    }],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 2,
      "total_tokens": 12
    }
  }
};

const validate = ajv.getSchema('https://adaptersdk.org/schemas/llm/llm.envelope.success.json');
if (!validate) throw new Error('Schema not registered');

const isValid = validate(llmResponse);
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

# Validate a complete LLM response
def validate_llm_response():
    registry = load_schema_registry()
    resolver = RefResolver.from_schema(registry["https://adaptersdk.org/schemas/common/envelope.success.json"], registry)
    
    llm_response = {
        "ok": True,
        "code": "OK",
        "ms": 45.2,
        "result": {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12
            }
        }
    }
    
    schema = registry["https://adaptersdk.org/schemas/llm/llm.envelope.success.json"]
    validator = Draft202012Validator(schema, resolver=resolver)
    
    try:
        validator.validate(llm_response)
        print("✅ Validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    validate_llm_response()
```

---

## CI/CD Integration

### GitHub Actions (Schema-Only)
```yaml
name: Schema Conformance
on: [push, pull_request]
jobs:
  schema-conformance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install .[test]
      - name: Verify schema suites (meta-lint + goldens)
        run: make verify-schema
```

For smoke checks in sensitive environments: `make quick-check` (pairs with `validate-env`/`safety-check` in the Makefile).

---

## Troubleshooting (Schema)

- **`$ref` cannot resolve** → Pre-load all schemas into the validator; `$id` strings must match exactly, including case and trailing slashes.
- **Duplicate `$id`** → Each file must have a globally unique `$id`; fix collisions and re-run meta-lint.
- **Invalid regex** → Fix unescaped characters; confirm each pattern compiles in your target runtime.
- **Examples fail** → Update examples or the schema; examples must validate against their parent schema.
- **NDJSON frame shape mismatch** → Re-check `*.stream.frame.*.json` and NDJSON union schema fields (`type`, `data`). Use golden streams to spot ordering or terminal-frame issues.
- **Schema drift vs PROTOCOLS/ERRORS/METRICS** → If behavior changes in those docs, ensure the schemas and goldens are updated in lockstep (and meta-lint extended if necessary).

---

## Versioning & Deprecation (Schema)

- SemVer is required for `schema_version` on success envelopes.
- **Non-breaking:** adding optional fields; additive enum members (as documented); new `$defs` not wired into existing envelopes.
- **Breaking:** removing/renaming required fields; type changes; constraint tightening that invalidates prior valid messages; `$id` renames.
- **Deprecation:** set `deprecated: true` and (optionally) `replacement_op` in the relevant schemas; ensure goldens reflect both old and new while deprecated paths are supported.

---

## Compliance Badge (Schema-Only)

After meta-lint + golden schema suites pass unmodified:

```
✅ Corpus Protocol (v1.0) — Schema Conformant
   • JSON Schema Draft 2020-12
   • LLM / Vector / Embedding / Graph
```

**Badge suggestion** (link to your generated artifact or CI run):

[![Corpus Protocol Schema Conformance](https://img.shields.io/badge/Corpus_Protocol-Schema_Conformant-green)](./conformance_report.json)

---

## Maintenance

- Keep `tests/schema/test_schema_lint.py` aligned with new schema categories, `$id` conventions, and extension rules.
- Add/update goldens whenever envelopes or frame schemas change; drift detection in the golden suite will reveal gaps.
- Periodically regenerate a schema index (path → `$id`) to spot stale or missing entries.
- When PROTOCOLS / ERRORS / METRICS evolve, audit schemas for drift and update both schemas and goldens together to maintain a consistent contract surface.

**Maintainers:** Corpus SDK Team  
**Last Updated:** 2025-11-12  
**Scope:** Schema contracts & wire shapes only (behavioral semantics are documented and tested elsewhere)

---

*End of SCHEMA_CONFORMANCE.md*