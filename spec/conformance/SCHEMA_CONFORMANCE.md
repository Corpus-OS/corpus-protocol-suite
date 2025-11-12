# SCHEMA_CONFORMANCE.md

> ✅ **Corpus Protocol (v1.0) — Schema Conformance**
> Components: **LLM** / **Vector** / **Embedding** / **Graph** · JSON Schema **Draft 2020-12**
> Test suites: **Schema Meta-Lint** (`tests/schema/`) & **Golden Wire Messages** (`tests/golden/`)

---

## Overview

This document defines the **Schema Conformance** requirements and test coverage for the Corpus Protocol schemas (Draft **2020-12**) across **LLM**, **Vector**, **Embedding**, **Graph**, and **Common** envelopes. It is the single source of truth for **wire-contract** validation (requests, successes, errors, and streaming frames) and **schema hygiene** (metaschema compliance, `$id` uniqueness, and `$ref` resolvability).

This suite is intended for Corpus adapters and third-party implementations that claim compatibility with the **Corpus Protocol v1.0** family of components. **Passing this suite** means your JSON messages conform to the protocol’s schemas and cross-schema invariants.

* **Protocol versions:** `llm/v1.0`, `vector/v1.0`, `embedding/v1.0`, `graph/v1.0`
* **Schema draft:** JSON Schema **Draft 2020-12**
* **Status:** Stable / Production-ready
* **Test locations:**

  * Schema meta-lint: `tests/schema/`
  * Golden message validation: `tests/golden/`
  * Utilities: `tests/utils/`

---

## Quick Start for Implementers

1. **Clone** the repository and install test deps:

   ```bash
   git clone https://github.com/corpus/protocol-schemas
   cd protocol-schemas
   pip install -r requirements-dev.txt  # or: pip install pytest jsonschema rfc3339-validator
   ```
2. **Run** the full conformance suite:

   ```bash
   pytest -v
   ```
3. **Fix** any schema validation failures (metaschema / $ref / examples) and wire-shape mismatches in your messages.
4. **Re-run** until clean; then **claim** the conformance badge in your docs.

---

## Repository Layout (Schema Files)

* `schemas/common/`
  `envelope.request.json`, `envelope.success.json`, `envelope.error.json`, `operation_context.json`

* `schemas/llm/`
  Envelopes: `llm.envelope.{request,success,error}.json`
  Ops: `llm.complete.{request,success}.json`, `llm.count_tokens.{request,success}.json`, `llm.capabilities.success.json`, `llm.health.success.json`
  Streaming: `llm.stream.frame.{data,end,error}.json`, `llm.stream.frames.ndjson.schema.json`
  Types: `llm.types.{message,chunk,completion,token_usage,tool,warning,logprobs}.json`
  Params & helpers: `llm.sampling.params.json`, `llm.tools.schema.json`, `llm.response_format.json`

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
  Partial: `embedding.partial_success.result.json`
  Types: `embedding.types.{vector,result,warning,failure}.json`

* `schemas/graph/`
  Envelopes: `graph.envelope.{request,success,error}.json`
  Ops: `graph.query.{request,success}.json`, `graph.stream_query.request.json`, `graph.vertex.{create,delete}.request.json`, `graph.edge.create.request.json`, `graph.batch.request.json`, `graph.id.success.json`, `graph.ack.success.json`
  Health/Caps: `graph.capabilities.{request,success}.json`, `graph.health.{request,success}.json`
  Streaming: `graph.stream.frame.{data,end,error}.json`, `graph.stream.frames.ndjson.schema.json`
  Types: `graph.types.{entity,id,row,batch_op,warning,partial_success_result}.json`

> **Note:** Many `*.types.*.json` are validated **indirectly** via `$ref` chains from envelopes and stream frames. The **schema meta-lint** also validates each schema file in isolation.

---

## What “Schema Conformance” Means

A build is **schema-conformant** when all of the following hold:

1. **Metaschema compliance** — Every file validates against JSON Schema Draft 2020-12.
2. **$id hygiene** — Each schema declares a **unique**, canonical `$id` (`https://adaptersdk.org/schemas/<component>/<file>.json`).
3. **$ref resolvability** — All `$ref`s resolve to known `$id`s or valid internal anchors.
4. **Envelope correctness** — Request/success/error envelopes have correct required fields, enums, and protocol/component constants.
5. **Streaming contracts** — NDJSON/SSE/WebSocket frames obey: exactly **one terminal** frame; **no data after terminal**.
6. **Examples validate** — Any `examples` embedded in schemas validate against their parent.
7. **Pattern/enum hygiene** — Regex patterns compile; enums deduped.
8. **No dangling `$defs`** — Exported defs are referenced or explicitly documented as public anchors.
9. **Cross-schema invariants** (enforced at golden level):

   * Partial-success: `successes + failures == len(items)` and both ≥ 1
   * Token totals: `total_tokens == prompt_tokens + completion_tokens`
   * Vector dimensions: consistent within results
   * `schema_version`: present & semver
   * `result_hash`: SHA-256 of canonical `result` (if present)
   * IDs/timestamps: match published patterns; SIEM-safe

---

## Test Suites

### A) Schema Meta-Lint (schema-only)

* **Path:** `tests/schema/test_schema_lint.py`
* **Purpose:** Validate the schemas themselves (metaschema, `$id`, `$ref`, anchors, examples, regexes).
* **Checks:**
  Load all schemas → metaschema validity → build `$id` index → resolve **all** `$ref`s → compile regex → validate `examples` → detect duplicate `$id`s & missing `$id`s → envelope constants sanity.

### B) Golden Message Validation (wire-level)

* **Path:** `tests/golden/test_golden_samples.py`
* **Purpose:** Validate **realistic** request/response/stream samples against top-level envelopes and frame schemas.
* **Coverage:** LLM, Vector, Embedding, Graph requests/success/error; NDJSON stream frames & terminal rules.
* **Invariants:** partial-success math, token totals, vector dims, single terminal, `schema_version`, `result_hash`, timestamp/ID patterns.
* **Drift detection:** ensures CASES mapping ↔ on-disk goldens stay in sync; flags missing or orphaned goldens.

### C) Utilities

* `tests/utils/schema_registry.py` — `$id` → validator cache; `assert_valid(schema_id, obj, context=...)`.
* `tests/utils/stream_validator.py` — NDJSON line-by-line validation; enforces frame shape & terminal rules.

---

## Conformance Summary

* **Metaschema validity:** 100% (all discovered schema files)
* **$id uniqueness & convention:** 100%
* **$ref graph resolution:** 100%
* **Golden validation:** 100% of present fixtures
* **Invariants:** 100% (partial success, token totals, dimensions, terminal rules, schema_version, result_hash)

> Golden coverage grows with new ops/examples. Drift checks will prompt updates as you add fixtures.

---

## Per-Component Coverage Notes

**Common** — `envelope.*`, `operation_context`: envelope structure, SIEM-safe guidance, deadlines, tenant/cache semantics, extension slots.

**LLM** — complete & count_tokens requests/success; stream frames + NDJSON union; types (messages/chunks/usage/tools/warnings/logprobs) exercised via envelopes/frames.

**Vector** — query/upsert/delete/namespace ops; health/caps; invariants for vector dimension consistency.

**Embedding** — single & batch ops; count_tokens/caps/health; partial-success invariants and result shapes.

**Graph** — query & stream_query; batch, id/ack; caps/health; NDJSON frames validated; types (entity/id/row/batch_op/warning) covered via frames.

---

## What’s Missing vs. Other Conformance (Behavioral Semantics)

Schema conformance **does not** assert runtime behavior. The following are covered by the **Behavioral Conformance** suite (separate):

1. Truncation & normalization rules (unit norm, “normalizes at source”, etc.)
2. Token counting invariants (monotonicity, unicode safety, model-gating)
3. Partial-success runtime semantics (fallbacks, stable ordering, no silent drops)
4. Error taxonomy mapping & retryability hints (provider→canonical, validation_errors)
5. Deadlines & budgets (pre-expired fast-fail, non-negative budgets, observability tags)
6. Observability & SIEM hygiene (no secrets/text/vectors; tenant hash only; one observe/op)
7. Caching & idempotency (tenant-aware keys, hashed text, replay semantics)
8. Streaming behavior beyond shape (pacing/keep-alives; mid-stream error is terminal; incremental usage vs final totals)
9. Model/limits enforcement at runtime (unsupported model handling; context/batch/dim limits)
10. Versioning/compatibility (back/forward checks; deprecation and replacement_op honored)
11. Cross-component runtime invariants (vector dims from live adapters; graph partial side-effects; LLM tool validation lifecycle)
12. Performance/resilience guardrails (payload ceilings, graceful degradation, backoff adherence)

---

## Running the Suite

* **Everything (schema meta-lint + goldens)**

  ```bash
  pytest -v
  ```
* **Schema meta-lint only**

  ```bash
  pytest tests/schema -v
  ```
* **Golden fixtures only**

  ```bash
  pytest tests/golden -v
  ```

> CI should run **both** suites. Fail-fast on metaschema/$ref breakage; allow golden drift checks to **skip** temporarily while new fixtures land—then switch to **fail** once stabilized.

---

## Adding or Changing Schemas — Checklist

1. Choose a canonical `$id`: `https://adaptersdk.org/schemas/<component>/<file>.json`
2. Include `$schema` (2020-12), `title`, `description`, top-level `type`.
3. Prefer `additionalProperties: false` for envelopes/types; expose vendor slots via `patternProperties` `^(vendor:)[a-z0-9_.:-]+$`.
4. Reference shared types using **absolute** `$ref`s to stable `$id`s.
5. Add `examples` for tricky shapes (optional but recommended).
6. Update or add **goldens** that exercise new envelopes/frames.
7. Run locally: `pytest tests/schema -v && pytest tests/golden -v`.
8. Open PR; CI must pass meta-lint + goldens.

---

## Cross-Schema Invariants (Test-Enforced)

* Partial success: `successes + failures == len(items)` and both ≥ 1
* Token totals: `total_tokens == prompt_tokens + completion_tokens`
* Vector dimensions: consistent within result set; under documented caps
* Streaming terminal: **exactly one** terminal frame; **no data after terminal**
* Schema version: present on success envelopes; `^\d+\.\d+\.\d+$`
* Result hash: if present, equals `sha256(canonical_json(result))` (lower-hex)
* IDs & timestamps: match documented patterns; SIEM-safe

---

## Versioning & Deprecation

* **SemVer** for `schema_version` inside success envelopes.
* **Non-breaking:** adding optional fields; additive enums (as documented); new `$defs` not wired to envelopes.
* **Breaking:** removing/retrofitting required fields; type changes; constraint tightening that invalidates prior valid messages; `$id` renames.
* **Deprecation:** set `deprecated: true` and optionally `replacement_op`; update docs and goldens accordingly.

---

## Implementation Examples

### Python

```python
from tests.utils.schema_registry import assert_valid

def validate_llm_complete_success(message: dict) -> bool:
    assert_valid("https://adaptersdk.org/schemas/llm/llm.envelope.success.json", message)
    return True
```

### TypeScript

```ts
// Using AJV (example): pre-load all schemas by $id into the AJV instance,
// then validate messages against the top-level envelope $id you need.
```

---

## Common Issues & Solutions (Troubleshooting)

* **$ref resolution fails to adaptersdk.org** → Pre-load schemas into your validator registry; do not rely on network.
* **Partial success math fails** → Ensure each item is counted exactly once; no “uncategorized” items.
* **Multiple `end` frames in NDJSON** → Implement a strict stream state machine; one terminal frame only.
* **`schema_version` missing** → Add SemVer string to success envelopes; keep consistent within a component.
* **Regex/pattern errors** → Compile patterns during CI; fix invalid or unescaped sequences.

---

## CI/CD Integration (Example)

**GitHub Actions**

```yaml
name: Corpus Protocol Conformance
on: [push, pull_request]
jobs:
  schema-conformance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - name: Install deps
        run: pip install -r requirements-dev.txt
      - name: Schema meta-lint
        run: pytest tests/schema/ -v
      - name: Golden validation
        run: pytest tests/golden/ -v
```

---

## Related Documents

* **Behavioral Conformance Suite** — Runtime semantics & error handling (`../BEHAVIORAL_CONFORMANCE.md`)
* **Adapter Implementation Guide** — Building production adapters (`../ADAPTER_GUIDE.md`)
* **Protocol Specification** — Full protocol details (`../PROTOCOL.md`)
* **Error Taxonomy** — Canonical error classes & retryability (`../ERRORS.md`)
* **Metrics & SIEM** — Observability guidance (`../METRICS.md`)

---

## Frequently Asked Questions

**Q:** Do I need both schema and behavioral conformance?
**A:** **Yes** for production. Schema ensures wire compatibility; behavioral ensures correct runtime semantics.

**Q:** Can I claim conformance for a single component only?
**A:** Yes: e.g., “Corpus Protocol (v1.0) — **LLM Schema Conformant**”.

**Q:** Are vendor extensions allowed?
**A:** Yes, under the `vendor:` prefix pattern. They must not break required fields or cross-schema invariants.

---

## Version Compatibility

| Schema Version | Protocol Version | Status   | Notes                |
| -------------- | ---------------- | -------- | -------------------- |
| 1.0.0          | v1.0             | ✅ Stable | Initial production   |

---

## Compliance Badge

Embed in your README once both suites pass unmodified:

```
✅ Corpus Protocol (v1.0) — Schema Conformant
   • JSON Schema Draft 2020-12
   • LLM / Vector / Embedding / Graph
```

Shields suggestion:

```
[![Corpus Protocol Conformance](https://img.shields.io/badge/Corpus_Protocol-Conformant-green)](https://github.com/corpus/protocol-schemas/actions)
```

---

## Maintenance

* Keep `tests/schema/test_schema_lint.py` aligned with new schema categories, `$id` conventions, and extension rules.
* Add golden fixtures alongside new envelopes or frame shapes. Drift checks will remind you if you forget.
* Periodically regenerate a schema index (path → `$id`) and update this document accordingly.

**Maintainers:** Corpus SDK Team
**Last Updated:** 2025-11-12
**Scope:** **Schema** contracts & wire shapes (behavioral semantics are covered separately)
