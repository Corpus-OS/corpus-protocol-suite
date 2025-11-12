---

# SCHEMA_CONFORMANCE.md

> ✅ **Corpus Protocol (v1.0) — Schema Conformance (Schema-Only)**
> Components: **LLM** / **Vector** / **Embedding** / **Graph**
> JSON Schema **Draft 2020-12**
> Test suites: **Schema Meta-Lint** (`tests/schema/`) & **Golden Wire Messages** (`tests/golden/`)

---

## Overview

This document defines **Schema Conformance** for the Corpus Protocol across **LLM**, **Vector**, **Embedding**, **Graph**, and **Common** envelopes. It is the source of truth for:

* **Wire-contract validation** of requests, successes, errors, and streaming frames (via goldens)
* **Schema hygiene**: metaschema compliance, `$id` uniqueness, `$ref` resolvability, anchors, enums, and patterns

> **Scope:** *Schema only.* Runtime/behavioral semantics (adapters, deadlines, caching, retries, etc.) are covered in a separate suite and are **not** part of this document.

**Protocol versions:** `llm/v1.0`, `vector/v1.0`, `embedding/v1.0`, `graph/v1.0`
**Schema draft:** JSON Schema **Draft 2020-12**
**Status:** Stable / Production-ready

**Test locations**

* Schema meta-lint: `tests/schema/`
* Golden message validation: `tests/golden/`
* Utilities: `tests/utils/`

---
## Schema Testing Philosophy

Our schema testing follows the "contract-first" principle:

1. **Schemas define the truth** - All validation flows from JSON Schema definitions
2. **Goldens exemplify reality** - Real-world message samples validate the schemas
3. **Meta-lint ensures hygiene** - Schema quality and maintainability are enforced
4. **Makefile provides workflow** - Consistent developer experience across environments
---
## Quick Start (Schema-Only)

> These commands align with the project **Makefile**.

1. **Install deps**

```bash
pip install .[test]
```

2. **Run schema suites**

```bash
# Meta-lint (schemas/** only)
make test-schema

# Golden wire messages (schema validation of sample payloads)
make test-golden

# Run both schema suites together
make verify-schema
```

3. **Fast (skip @slow)**

```bash
make test-schema-fast
make test-golden-fast
```

4. **Smoke & safety**

```bash
make quick-check     # minimal schema/golden smoke
make validate-env    # warns if CORPUS_TEST_ENV unset
make safety-check    # blocks heavy runs if CORPUS_TEST_ENV=production
```

---

## Repository Layout (Schemas)

* `schemas/common/`
  `envelope.request.json`, `envelope.success.json`, `envelope.error.json`, `operation_context.json`

* `schemas/llm/`
  Envelopes: `llm.envelope.{request,success,error}.json`
  Ops: `llm.complete.{request,success}.json`, `llm.count_tokens.{request,success}.json`, `llm.capabilities.success.json`, `llm.health.success.json`
  Streaming: `llm.stream.frame.{data,end,error}.json`, `llm.stream.frames.ndjson.schema.json`
  Types/params: `llm.types.{message,chunk,completion,token_usage,tool,warning,logprobs}.json`, `llm.sampling.params.json`, `llm.tools.schema.json`, `llm.response_format.json`

* `schemas/vector/`
  Envelopes: `vector.envelope.{request,success,error}.json`
  Ops: `vector.query.{request,success}.json`, `vector.upsert.{request,success}.json`, `vector.delete.{request,success}.json`
  Namespace: `vector.namespace.{create,delete}.{request,success}.json`
  Health/Caps: `vector.capabilities.{request,success}.json`, `vector.health.success.json`
  Types: `vector.types.{vector,vector_match,query_result,filter,partial_success_result,failure_item}.json`

* `schemas/embedding/`
  Envelopes: `embedding.envelope.{request,success,error}.json`
  Ops: `embedding.embed.{request,success}.json`, `embedding.embed_batch.{request,success}.json`, `embedding.count_tokens.{request,success}.json`
  Health/Caps: `embedding.capabilities.{request,success}.json`, `embedding.health.{request,success}.json`
  Partial/types: `embedding.partial_success.result.json`, `embedding.types.{vector,result,warning,failure}.json`

* `schemas/graph/`
  Envelopes: `graph.envelope.{request,success,error}.json`
  Ops: `graph.query.{request,success}.json`, `graph.stream_query.request.json`, `graph.vertex.{create,delete}.request.json`, `graph.edge.create.request.json`, `graph.batch.request.json`, `graph.id.success.json`, `graph.ack.success.json`
  Health/Caps: `graph.capabilities.{request,success}.json`, `graph.health.{request,success}.json`
  Streaming: `graph.stream.frame.{data,end,error}.json`, `graph.stream.frames.ndjson.schema.json`
  Types: `graph.types.{entity,id,row,batch_op,warning,partial_success_result}.json`

> `*.types.*.json` files are validated **indirectly** via `$ref` chains from envelopes and stream frames. The **schema meta-lint** also validates each schema file in isolation.

---

## What “Schema Conformance” Means

A build is **schema-conformant** when all of the following hold:

1. **Metaschema compliance** — Every file validates against JSON Schema Draft 2020-12.
2. **$id hygiene** — Each schema declares a **unique**, canonical `$id` (`https://adaptersdk.org/schemas/<component>/<file>.json`).
3. **$ref resolvability** — All `$ref`s resolve to known `$id`s or valid internal anchors; no dangling fragments.
4. **Envelope correctness** — Request/success/error envelopes include required fields, enums, protocol/component constants.
5. **Streaming contracts (shape)** — Frame schemas validate NDJSON/SSE/WebSocket **data/end/error** shapes.
6. **Examples validate** — Any `examples` embedded in schemas validate against their parent schema.
7. **Pattern/enum hygiene** — Regex patterns compile; enums are deduplicated and documented.
8. **No dangling `$defs`** — Exported defs are referenced, or explicitly documented as public anchors.
9. **Cross-schema invariants (schema-level only)** — Enforced via goldens where JSON Schema alone is insufficient:

   * Partial success accounting fields present (`successes`, `failures`, `items`)
   * `schema_version` present on success envelopes; semver format
   * `result_hash` (when present) is a lower-hex string
   * IDs/timestamps match documented patterns (format/pattern checks)

> **Out of scope (behavioral):** deadlines, retries, caching, normalization semantics, etc. Tracked in behavioral conformance.

---

## Test Suites (Schema-Only)

### A) Schema Meta-Lint

**Path:** `tests/schema/test_schema_lint.py`
**Purpose:** Validate the schemas themselves.

**Checks include:**

* Load all `schemas/**` and validate against the **Draft 2020-12 metaschema**
* Build a `$id` index; **detect duplicates** and **missing** `$id`s
* Resolve **all** `$ref`s (absolute and internal anchors)
* Compile **regex patterns** and sanity-check **enums**
* Validate embedded **`examples`** arrays (when present)
* Enforce **envelope constants** (protocol/component) where specified

### B) Golden Wire Messages

**Path:** `tests/golden/test_golden_samples.py`
**Purpose:** Validate **realistic** request/response/stream samples against top-level envelopes and frame schemas.

**Covers:**

* LLM / Vector / Embedding / Graph: request & success/error envelopes
* LLM & Graph streaming: `data/end/error` frame schemas and NDJSON union where defined

**Also enforces (schema-level invariants only):**

* `schema_version` present and semver
* Minimal partial-success accounting fields present where used
* Identifier/timestamp pattern sanity
* Single terminal frame constraint validated by stream utility (ordering/shape rules)

---

## Running (Schema-Only; Makefile-Aligned)

**Everything schema (meta-lint + goldens)**

```bash
make verify-schema
```

**Meta-lint only**

```bash
make test-schema
# fast:
make test-schema-fast
```

**Goldens only**

```bash
make test-golden
# fast:
make test-golden-fast
```

**Smoke & safety**

```bash
make quick-check
make validate-env
make safety-check
```

> Env overrides: `PYTEST_JOBS=4`, `PYTEST_ARGS="-x --tb=short"`.

---

## Schema Evolution Guidelines

### Adding New Schemas

1. Follow the `$id` convention:
   `https://adaptersdk.org/schemas/<component>/<file>.json`
2. Include: `$schema` (2020-12), `title`, `description`, and top-level `type`.
3. Prefer `additionalProperties: false` for **envelopes** (use `patternProperties` for vendor slots if needed).
4. Add **golden samples** for new operations to exercise the new envelopes/frames.

### Breaking Changes

* **Renaming required fields** → **Major** version bump
* **Removing enum values** → **Major** version bump
* **Loosening constraints** (e.g., widening types, adding optional fields) → **Minor** version bump

> Keep `schema_version` in success envelopes aligned with SemVer and maintain backward compatibility guidance in commit notes.

---

## Validation in Other Languages

### TypeScript

```typescript
// Using ajv with pre-loaded schemas
import Ajv from 'ajv';
const ajv = new Ajv({ strict: false });

// Pre-load all schemas by their $id into ajv.addSchema(schemaObject)
// ...
const validate = ajv.getSchema('https://adaptersdk.org/schemas/llm/llm.envelope.success.json');
if (!validate) throw new Error('Schema not registered');

const ok = validate(message);
if (!ok) console.error(validate.errors);
```

### Python (external validator)

```python
from jsonschema import Draft202012Validator

# Assume 'loaded_schema' is the dict for the target $id
# and 'resolver' is a RefResolver that knows all $id -> schema mappings.
def validate_message(message: dict, loaded_schema: dict, resolver):
    Draft202012Validator(loaded_schema, resolver=resolver).validate(message)
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

> For smoke checks in sensitive environments: `make quick-check` (pairs with `validate-env`/`safety-check` in the Makefile).

---

## Troubleshooting (Schema)

* **$ref cannot resolve** → Pre-load all schemas into the validator; `$id` strings must match exactly.
* **Duplicate `$id`** → Each file must have a globally unique `$id`; fix collisions.
* **Invalid regex** → Fix unescaped characters; confirm `pattern` compiles.
* **Examples fail** → Update examples or the schema; examples must validate against their parent.
* **NDJSON frame shape mismatch** → Re-check `*.stream.frame.*.json` and NDJSON union schema fields (`event`, payload).

---

## Versioning & Deprecation (Schema)

* **SemVer** is required for `schema_version` on success envelopes.
* **Non-breaking:** adding optional fields; additive enum members (as documented); new `$defs` not wired into existing envelopes.
* **Breaking:** removing/renaming required fields; type changes; constraint tightening that invalidates prior valid messages; `$id` renames.
* **Deprecation:** set `deprecated: true` and (optionally) `replacement_op` in the relevant schemas.

---

## Compliance Badge (Schema-Only)

After **meta-lint + golden** schema suites pass unmodified:

```
✅ Corpus Protocol (v1.0) — Schema Conformant
   • JSON Schema Draft 2020-12
   • LLM / Vector / Embedding / Graph
```

Badge suggestion (link to your generated artifact or CI run):

```
[![Corpus Protocol Schema Conformance](https://img.shields.io/badge/Corpus_Protocol-Schema_Conformant-green)](./conformance_report.json)
```

---

## Maintenance

* Keep `tests/schema/test_schema_lint.py` aligned with new schema categories, `$id` conventions, and extension rules.
* Add/update **goldens** whenever envelopes or frame schemas change; drift detection in the golden suite will reveal gaps.
* Periodically regenerate a schema index (path → `$id`) to spot stale or missing entries.

**Maintainers:** Corpus SDK Team
**Last Updated:** 2025-11-12
**Scope:** **Schema contracts & wire shapes only** (behavioral semantics are documented and tested elsewhere)
