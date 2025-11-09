# VERSIONING.md

**Corpus Protocol Suite — Versioning & Compatibility Policy**

> This document describes how we version the **specification**, **wire contracts**, **SDK libraries**, and **adapters** across the Corpus Protocol Suite (Graph, LLM, Vector, Embedding). It builds on the specification’s versioning rules (see *SPECIFICATION.md §18*) and adds concrete release and compatibility practices for maintainers and contributors.

---

## 1) Goals

* **Predictability.** Consumers can upgrade with clear expectations about breakage risk and compatibility.
* **Interoperability.** Adapters and clients that share a protocol **major** version must remain wire-compatible.
* **Velocity with safety.** Additive changes flow quickly under **MINOR/PATCH** without forcing lockstep upgrades.
* **Auditability.** Every release is traceable to a changelog entry and, when relevant, a documented migration path.

---

## 2) What we version (at a glance)

| Thing                             | Example                       | Scheme                                                                | Compatibility Contract                                                                                                           |
| --------------------------------- | ----------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Protocol spec** (per component) | `llm/v1.1`, `vector/v1.1`     | **SemVer** on spec: `MAJOR.MINOR` (optionally `.PATCH`)               | Same-**MAJOR** wire compatibility; **MINOR** is strictly additive; **MAJOR** may break                                           |
| **Wire envelopes / JSON schemas** | `embedding/v1` envelopes      | **SemVer** `$id` in schema; embedded `protocol` field in capabilities | Adding fields = **MINOR**; changing/removing required fields = **MAJOR**                                                         |
| **SDK packages** (language libs)  | `corpus-sdk-py 1.7.2`         | **SemVer**                                                            | Public API stability aligned with spec **MAJOR**; SDK **MINOR/PATCH** must not break user code                                   |
| **Adapters** (provider impls)     | `corpus-adapter-qdrant 2.3.0` | **SemVer**                                                            | Adapter `MAJOR` may drop support for old protocol **MAJOR**; must advertise `protocol: <component>/v<major>` in `capabilities()` |
| **Spec repository**               | Git tag `spec/v1.1.0`         | **SemVer** tags per component                                         | Tags track immutable snapshots for citations and conformance suites                                                              |

> **Key invariant:** If a client and adapter both support `X-Adapter-Protocol: <component>/v1`, they must interoperate on the wire, regardless of their library/package versions.

---

## 3) Semantic Versioning (how to decide MAJOR/MINOR/PATCH)

We follow **SemVer 2.0.0** for all artifacts.

### 3.1 Protocol & wire contracts (normative)

* **MAJOR** (`v2.0`): Any **breaking** change in required fields, field types, error semantics, or operation signatures.
  *Examples:* renaming or removing a required field; changing meaning of `score` from “higher is better” to the opposite; tightening validation that rejects previously valid requests.
* **MINOR** (`v1.2`): Any **additive** change.
  *Examples:* new optional fields; new error subtype; new capability flag; new operation that is optional to implement.
* **PATCH** (`v1.1.3`): Editorial clarifications, typos, non-behavioral fixes, test suite updates that don’t change the normative contract.

### 3.2 SDK libraries & adapters (public API)

* **MAJOR:** Removing/renaming public classes/functions; changing parameter types; altering default behavior in a way that breaks existing apps.
* **MINOR:** Adding public functions/classes/flags; introducing optional parameters with safe defaults; performance improvements; new adapters.
* **PATCH:** Bug fixes, docs, build tooling, non-breaking typing annotations.

---

## 4) Compatibility Matrix

**Within the same protocol MAJOR, wire compatibility is required.** Additive fields must be **ignored by older clients/servers** per *SPECIFICATION.md §6.2*.

| Client Protocol | Adapter Protocol | Expectation                                                                                                             |
| --------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `v1.x`          | `v1.y`           | **Must** interoperate. Unknown fields are ignored.                                                                      |
| `v1.x`          | `v2.y`           | **May fail**. Adapter **should** reject with `NotSupported` + advertise supported protocols in `capabilities.protocol`. |
| `v2.x`          | `v1.y`           | **May fail**. Client should negotiate down using `capabilities()`.                                                      |

> **Negotiation rule:** Clients **MAY** send `X-Adapter-Protocol: {component}/v{major}`. If unsupported, adapter **MUST** respond with `NotSupported` and include supported versions in `capabilities.protocol` (see *SPECIFICATION.md §18.2*).

---

## 5) Backward/Forward Compatibility Rules (Normative)

* **Unknown JSON keys**: **MUST** be ignored by both clients and servers (see *§4 Conventions* and *§6.2 Capability Discovery*).
* **Error taxonomy**: Adding new error **subtypes** = **MINOR**. Renaming/removing existing classes or changing retryability = **MAJOR** (see *§12.4*).
* **Observability fields**: Adding optional, low-cardinality metric attributes (e.g., `deadline_bucket`) = **MINOR** (see *§13.1/§13.2*). Removing a previously-emitted attribute expected by clients = **MAJOR**.
* **Security/Privacy**: Tightening redaction (e.g., banning new data classes from logs) is **MINOR**; loosening is **MAJOR** (see *§14/§15*).
* **Vector scoring conventions**: The “higher is better” rule (see *§4*) is **MAJOR** if changed.

---

## 6) Deprecation Policy

We implement the deprecation process described in *SPECIFICATION.md §18.4* with the following operational details:

1. **Announce** in release notes and `CHANGELOG.md` with a clear rationale and migration hints.
2. **Warn** at runtime where feasible (e.g., log a deprecation code in `observe.extra` without leaking PII).
3. **Minimum support window**:

   * Keep the deprecated feature for **at least one full MAJOR** after announcement, or **6 months**, whichever is longer, unless there is a security issue.
4. **Removal** occurs at the **next MAJOR** release.

**On the wire:** mark deprecated fields with `"deprecated": true` in published JSON Schema (when applicable) and keep parsing them until removal in the next **MAJOR**.

---

## 7) Release Process & Tagging

* **Git tags**

  * Protocol/spec changes: `spec/<component>/vX.Y.Z` (e.g., `spec/vector/v1.1.0`)
  * SDK libraries: `<lang>/vX.Y.Z` (e.g., `python/v1.7.2`)
  * Adapters: `adapter-<name>/vX.Y.Z` (e.g., `adapter-qdrant/v2.3.0`)
* **Branches**

  * `main`: active development (next **MINOR**).
  * `release/vX.Y`: stabilization branch for imminent release.
  * `maint/vX.Y`: maintenance branch for backports to an older **MINOR**.
* **Artifacts**

  * Every release updates `CHANGELOG.md` (keepers: human-readable, “Added/Changed/Fixed/Deprecated/Removed/Security”).
  * If wire/spec touched: bump spec tag and update `SPECIFICATION.md` appendix F (Change Log).
  * If schemas touched: bump schema `$id` with `X.Y.Z` and publish under `/schemas/<component>/<major>/`.

---

## 8) Pre-Releases, RCs, and Experimental Features

* **Pre-releases**: `v1.2.0-rc.1`, `v1.2.0-beta.2`. No compatibility guarantees; APIs may change before GA.
* **Experimental flags** (code): must be opt-in via explicit environment variable or constructor parameter; default **off**.
* **Experimental capabilities** (wire): expose under `extensions.vendor:<org>.*` per *§6.2* and **do not** rely on them for cross-vendor compatibility.

---

## 9) Long-Term Support (LTS)

* We designate certain **MINOR** lines as **LTS** (e.g., `1.4.x`) supporting **12 months** of security and critical fixes.
* LTS backports must not change public APIs; no new features.
* LTS releases follow semantic **PATCH** increments only.

---

## 10) Multi-Language Package Versioning

To avoid cross-language drift:

* **Source of truth**: protocol spec tags (`spec/<component>/vX.Y.Z`).
* **Language SDKs** must include a machine-readable mapping (e.g., `sdk_support.json`) indicating the set of supported protocol majors/minors:

  ```json
  {
    "llm": "v1.1",
    "vector": "v1.1",
    "graph": "v1.1",
    "embedding": "v1.1"
  }
  ```
* **Go**: Module path remains stable across **MINOR/PATCH**; **MAJOR** increments add `/v2` per Go module rules.
* **Python**: `__version__` follows SemVer; upload wheels with exact tag parity; pin protocol compatibility in package metadata.
* **TypeScript/Java**: publish with SemVer; maintain API surface docs and generated types in lockstep with protocol spec.

---

## 11) Capability Gating & Negotiation

* Adapters **MUST** return `capabilities()` with:

  * `protocol: "<component>/v<major>"`,
  * Feature flags (e.g., `supports_deadline`),
  * Limits (e.g., `max_top_k`, `max_dimensions`),
  * `extensions.vendor:*` for non-standard features.
    *(See SPECIFICATION.md §6.2, §8.4, §9.2, §10.4.)*
* Clients **SHOULD**:

  * Probe `capabilities()` at startup,
  * Gate optional features based on flags,
  * Avoid assuming presence of non-advertised fields.

**Adding a new capability flag** is **MINOR**. **Removing** or **inverting** its semantics is **MAJOR**.

---

## 12) Migration Checklist (for maintainers)

Before merging a change that affects public API/spec:

1. **Classify**: breaking vs additive vs editorial.
2. **Bump**: version(s) in code/spec/schemas accordingly.
3. **Negotiate**: ensure fallback behavior or capability flag for optional features.
4. **Docs**: update `SPECIFICATION.md`, `CHANGELOG.md`, and migration notes.
5. **Tests**: extend conformance tests (unit + interop).

   * Breaking → add negative tests for old behavior.
   * Additive → add positive tests that older peers ignore gracefully.
6. **Deprecation**: if applicable, mark deprecated and document timeline.
7. **Security/Privacy**: reconfirm that telemetry remains SIEM-safe per *§13/§15*.

---

## 13) Examples

### 13.1 Add a new optional match attribute (Vector)

* Change: add `normalized_score` to `VectorMatch`.
* Version: `vector/v1.2.0` (**MINOR**, additive).
* Behavior: Adapters may populate; clients not expecting it ignore the field.

### 13.2 Tighten error retryability (LLM)

* Change: classify `ContentFiltered` from retryable → non-retryable.
* Version: `llm/v2.0.0` (**MAJOR**, changes client guidance per *§12.4*).
* Migration: document new handling; add capability flag if transitional behavior is offered.

### 13.3 Add `supports_deadline` to capabilities (Embedding)

* Change: capability flag addition only.
* Version: `embedding/v1.1.0` (**MINOR**).
* Adapters not supporting deadlines keep flag `false`; clients must check before using.

---

## 14) Versioning for Documentation & Examples

* **SPECIFICATION.md** carries a normative version header for each protocol component.
* **Examples** are **Non-Normative** unless explicitly marked (see callouts in the spec); example updates don’t change the protocol version unless they demonstrate new normative behavior.
* **CONFORMANCE.md** files track test coverage per protocol version; when the spec **MINOR** bumps, conformance files must reflect test deltas.

---

## 15) Security, CVEs, and Backports

* **Security fixes** may be backported to currently supported LTS and the latest stable MINOR.
* If a security fix changes a normative behavior:

  * Prefer an **opt-in** flag defaulting to safe behavior.
  * Otherwise, release as **PATCH** with clear release notes and rationale.
* We publish advisories with fixed versions and upgrade guidance.

---

## 16) Notices & Legal

* **NOTICE** and license headers are maintained per Apache 2.0. See `CONTRIBUTING.md` Appendix A for when/why to update the **NOTICE** file.
* Version tags and changelogs must not include vendor confidential material.

---

## 17) FAQ

**Q: Can an SDK `1.9.0` support protocol `vector/v2`?**
A: Only if it explicitly advertises support (e.g., via `sdk_support.json`) and negotiates `X-Adapter-Protocol: vector/v2`. Otherwise, expect `NotSupported`.

**Q: Are additive validation checks breaking?**
A: If they reject previously valid inputs, yes → **MAJOR**. If they only reject previously undefined/invalid inputs, it’s **PATCH/MINOR** (document rationale).

**Q: Can we add a new error hint (e.g., `suggested_batch_reduction`)?**
A: **MINOR** additive; clients must ignore unknown hints per *§12.1/§12.4*.

---

## 18) Quick Decision Table

| Change Type                                            | Bump            | Notes                            |
| ------------------------------------------------------ | --------------- | -------------------------------- |
| Add optional field / capability flag                   | **MINOR**       | Must be ignorable by older peers |
| Add new operation (optional)                           | **MINOR**       | Gate by capability               |
| Tighten validation that rejects previously valid input | **MAJOR**       | Document migration               |
| Rename/remove required field                           | **MAJOR**       | Breaking wire change             |
| Change error retryability guidance                     | **MAJOR**       | Alters client behavior           |
| Add new error subtype                                  | **MINOR**       | Additive taxonomy                |
| Observability: add low-cardinality attribute           | **MINOR/PATCH** | Ensure privacy invariants        |
| Doc/typo/edit only                                     | **PATCH**       | No behavior change               |

---

## 19) Checklist for Release Managers

* [ ] Version(s) bumped across spec, SDK, adapters, schemas
* [ ] Changelog entries complete (Added/Changed/Deprecated/Removed/Fixed/Security)
* [ ] Conformance tests updated/passing
* [ ] Deprecations noted with timeline
* [ ] Capabilities and negotiation tested end-to-end
* [ ] Tags pushed; wheels/jars/modules published
* [ ] `SPECIFICATION.md` Appendix F updated if normative
* [ ] **NOTICE**/LICENSE updated if new third-party code included

---

**Change History**

* **v1.0 (Initial):** Establishes SemVer policy across spec/wire/SDK/adapters; defines negotiation rules, deprecation, LTS, and migration procedures.

