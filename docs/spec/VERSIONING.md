# VERSIONING

**Corpus Protocol Suite — Versioning & Compatibility Policy**

> This document describes how we version the **spec**, **wire contracts (schemas/envelopes)**, **SDK libraries**, and **adapters** across the Corpus Protocol Suite (Graph, LLM, Vector, Embedding).  
> It complements `SCHEMA.md` (wire shapes) and `PROTOCOLS.md` (operation semantics).

---

## Table of Contents

1. [Goals](#1-goals)
2. [What we version (at a glance)](#2-what-we-version-at-a-glance)
3. [Semantic Versioning (how to decide MAJOR/MINOR/PATCH)](#3-semantic-versioning-how-to-decide-majorminorpatch)
   - [3.1 Protocol & wire contracts (normative)](#31-protocol--wire-contracts-normative)
   - [3.2 SDK libraries & adapters (public API)](#32-sdk-libraries--adapters-public-api)
4. [Compatibility Matrix](#4-compatibility-matrix)
5. [Backward/Forward Compatibility Rules (Normative)](#5-backwardforward-compatibility-rules-normative)
   - [5.1 Unknown JSON keys (schema-governed)](#51-unknown-json-keys-schema-governed)
   - [5.2 Error taxonomy changes](#52-error-taxonomy-changes)
   - [5.3 Observability](#53-observability)
   - [5.4 Streaming lifecycle](#54-streaming-lifecycle)
   - [5.5 Vector scoring conventions](#55-vector-scoring-conventions)
6. [Deprecation Policy](#6-deprecation-policy)
7. [Release Process & Tagging](#7-release-process--tagging)
   - [7.1 Git tags](#71-git-tags)
   - [7.2 Branches](#72-branches)
   - [7.3 Artifacts](#73-artifacts)
8. [Pre-Releases, RCs, and Experimental Features](#8-pre-releases-rcs-and-experimental-features)
9. [Long-Term Support (LTS)](#9-long-term-support-lts)
10. [Multi-Language Package Versioning](#10-multi-language-package-versioning)
11. [Capability Gating & Negotiation](#11-capability-gating--negotiation)
12. [Migration Checklist (for maintainers)](#12-migration-checklist-for-maintainers)
13. [Examples](#13-examples)
   - [13.1 Add a new optional match attribute (Vector)](#131-add-a-new-optional-match-attribute-vector)
   - [13.2 Tighten error retryability (LLM)](#132-tighten-error-retryability-llm)
   - [13.3 Add `supports_deadline` to capabilities (Embedding)](#133-add-supports_deadline-to-capabilities-embedding)
14. [Versioning for Documentation & Examples](#14-versioning-for-documentation--examples)
15. [Security, CVEs, and Backports](#15-security-cves-and-backports)
16. [Notices & Legal](#16-notices--legal)
17. [FAQ](#17-faq)
18. [Quick Decision Table](#18-quick-decision-table)
19. [Checklist for Release Managers](#19-checklist-for-release-managers)
20. [Change History](#20-change-history)

---

## 1) Goals

- **Predictability.** Consumers can upgrade with clear expectations about breakage risk and compatibility.
- **Interoperability.** Adapters and clients that share a protocol **MAJOR** must remain wire-compatible.
- **Velocity with safety.** Additive changes flow under **MINOR/PATCH** without lockstep upgrades.
- **Auditability.** Every release is traceable to a changelog entry and migration path when needed.

---

## 2) What we version (at a glance)

| Thing | Example | Scheme | Compatibility Contract |
|---|---|---|---|
| **Protocol ID** (wire-level major identifier, per component) | `llm/v1.0`, `vector/v1.0` | **Major-stable identifier** | **Stable for the entire MAJOR**; does not change on MINOR/PATCH spec updates |
| **Protocol spec** (per component) | `protocols_version: 1.2` | **SemVer** (`MAJOR.MINOR[.PATCH]`) | Same-**MAJOR** wire compatibility; **MINOR** strictly additive; **MAJOR** may break |
| **Wire envelopes / JSON schemas** | `schema_version: 1.0.3` | **SemVer** (`MAJOR.MINOR.PATCH`) | Adding optional fields = **MINOR**; changing/removing required fields or types = **MAJOR** |
| **SDK packages** (language libs) | `corpus-sdk-py 1.7.2` | **SemVer** | Public API stability aligned with spec **MAJOR**; SDK **MINOR/PATCH** must not break user code |
| **Adapters** (provider impls) | `corpus-adapter-qdrant 2.3.0` | **SemVer** | Adapter `MAJOR` may drop support for old protocol **MAJOR**; must advertise `protocol: <component>/v<major>.0` in `capabilities()` |
| **Spec repository snapshot** | Git tag `spec/v1.2.0` | **SemVer** tags | Tags track immutable snapshots for citations + conformance suites |

> **Key invariant:** If a client and adapter both speak `<component>/v1.0`, they must interoperate on the wire, regardless of SDK/package versions.

---

## 3) Semantic Versioning (how to decide MAJOR/MINOR/PATCH)

We follow **SemVer 2.0.0** for all artifacts.

### 3.1 Protocol & wire contracts (normative)

- **MAJOR** (`2.0.0`): Any **breaking** change in required fields, field types, error semantics, or operation signatures.  
  *Examples:* renaming or removing a required field; changing meaning of `score` from "higher is better" to the opposite; tightening validation that rejects previously valid requests.
- **MINOR** (`1.2.0`): Any **additive** change.  
  *Examples:* new optional fields; new error subtype; new capability flag; new operation that is optional to implement.
- **PATCH** (`1.1.3`): Editorial clarifications, typos, non-behavioral fixes, and test suite updates that don't change the normative contract.

### 3.2 SDK libraries & adapters (public API)

- **MAJOR:** Removing/renaming public classes/functions; changing parameter types; altering default behavior in a way that breaks existing apps.
- **MINOR:** Adding public functions/classes/flags; introducing optional parameters with safe defaults; performance improvements; new adapters.
- **PATCH:** Bug fixes, docs, build tooling, non-breaking typing annotations.

---

## 4) Compatibility Matrix

**Within the same protocol MAJOR, wire compatibility is required.** Additive fields must be ignored **only where the schema explicitly permits extension**.

| Client Protocol | Adapter Protocol | Expectation |
|---|---|---|
| `v1.x` | `v1.y` | **Must** interoperate. Extension points are ignored as defined by schema. |
| `v1.x` | `v2.y` | **May fail**. Adapter SHOULD reject with `NotSupported` and include supported protocols in `error.details`. |
| `v2.x` | `v1.y` | **May fail**. Client should probe `capabilities()` and downgrade. |

### Negotiation rule (wire-level, transport-independent)

Clients MAY indicate desired protocol MAJOR via a transport mapping (e.g., HTTP header) or via the request envelope:

- Recommended: `ctx.attrs.accept_protocol = "<component>/v<major>.0"` (or equivalent)
- Request envelopes are open, so adding a hint field is allowed by the base request envelope schema.

If unsupported, adapter MUST return `NotSupported` and include:

```json
{
  "ok": false,
  "code": "NOT_SUPPORTED",
  "error": "NotSupported",
  "message": "Unsupported protocol requested",
  "retry_after_ms": null,
  "details": { "supported_protocols": ["llm/v1.0"] },
  "ms": 0.0
}
```

> **Note:** `capabilities.protocol` is a single value (const per schema). Supported protocol sets must be communicated via `error.details` (or via an explicitly versioned capabilities schema in a future MAJOR).

---

## 5) Backward/Forward Compatibility Rules (Normative)

### 5.1 Unknown JSON keys (schema-governed)

- **Request envelopes:** Top-level unknown keys MAY be present and MUST be ignored by adapters (request envelope is open).
- **ctx:** Unknown keys MUST be ignored by adapters (OperationContext is open).
- **args:** Unknown keys:
  - MUST be ignored only when the operation's args schema allows them (`additionalProperties: true`).
  - MUST be rejected (typically `BAD_REQUEST`) when the operation's args schema is closed (`additionalProperties: false`).
- **Success / Error / Streaming envelopes:** Closed by schema; unknown top-level keys MUST be rejected by schema validation.

### 5.2 Error taxonomy changes

- Adding new error subtypes (and associated wire codes) is **MINOR** (additive).
- Renaming/removing existing error classes or changing retryability guidance is **MAJOR**.
- Error payload extensions must go in `details` (which is `object|null` and open by schema).

### 5.3 Observability

- Adding optional, low-cardinality metric attributes is **MINOR/PATCH** depending on impact.
- Removing or reinterpreting a previously-emitted attribute that downstream systems rely on is **MAJOR**.
- Privacy tightening (more redaction) is **MINOR**; loosening is **MAJOR**.

### 5.4 Streaming lifecycle

- Streaming success frames MUST remain `{ok, code:"STREAMING", ms, chunk}` with no extra top-level keys (schema-closed).
- Chunk schema changes follow SemVer rules:
  - Adding optional fields where allowed = **MINOR**
  - Removing/changing required fields = **MAJOR**

### 5.5 Vector scoring conventions

- The "higher is better" score convention is normative; changing it is **MAJOR**.
- If a backend provides only one of `score` or `distance`, adapters MUST synthesize the other (where schema requires both).

---

## 6) Deprecation Policy

We implement a standard deprecation lifecycle:

1. **Announce** in release notes and `CHANGELOG.md` with rationale + migration hints.
2. **Warn** where feasible (e.g., log a deprecation code in telemetry without leaking PII).
3. **Minimum support window:**
   - Keep deprecated features for at least one full **MAJOR** after announcement, or **6 months**, whichever is longer, unless there is a security issue.
4. **Removal** occurs at the next **MAJOR** release.

On the wire: mark deprecated fields with `"deprecated": true` in published JSON Schema (when applicable) and continue accepting them until removal in the next MAJOR.

---

## 7) Release Process & Tagging

### 7.1 Git tags

- **Spec changes:** `spec/vX.Y.Z` (repo snapshot)
- **Optional component tags:** `spec/<component>/vX.Y.Z` (when component-specific release notes are maintained)
- **SDKs:** `<lang>/vX.Y.Z` (e.g., `python/v1.7.2`)
- **Adapters:** `adapter-<name>/vX.Y.Z` (e.g., `adapter-qdrant/v2.3.0`)

### 7.2 Branches

- **main:** active development (next MINOR)
- **release/vX.Y:** stabilization branch for imminent release
- **maint/vX.Y:** maintenance branch for backports to an older MINOR

### 7.3 Artifacts

- Every release updates `CHANGELOG.md` (format: Added/Changed/Fixed/Deprecated/Removed/Security).
- If wire/spec touched: update conformance suites and publish updated schema bundle.
- If schemas touched: bump `schema_version` and publish under a stable path.

---

## 8) Pre-Releases, RCs, and Experimental Features

- **Pre-releases:** `v1.2.0-rc.1`, `v1.2.0-beta.2`. No long-term compatibility guarantees; APIs may change before General Availability..
- **Experimental flags (code):** opt-in via explicit environment variable or constructor parameter; default off.
- **Experimental capabilities (wire):** expose under explicit vendor/extension namespaces (e.g., `x-` prefixed keys) and do not rely on them for cross-vendor compatibility.

---

## 9) Long-Term Support (LTS)

- Certain **MINOR** lines may be designated as LTS (e.g., `1.4.x`) supporting **12 months** of security and critical fixes.
- LTS backports must **not** change public APIs; no new features.
- LTS releases are **PATCH** increments only.

---

## 10) Multi-Language Package Versioning

To reduce cross-language drift:

- **Source of truth:** spec tags (`spec/vX.Y.Z`) and schema bundle versions (`schema_version`).
- SDKs must include a machine-readable mapping (e.g., `sdk_support.json`) indicating supported protocol majors:

```json
{
  "llm": "v1.0",
  "vector": "v1.0",
  "graph": "v1.0",
  "embedding": "v1.0"
}
```

**Language-specific notes:**

- **Go:** Module path remains stable across MINOR/PATCH; MAJOR increments add `/v2` per Go module rules.
- **Python:** `__version__` follows SemVer; publish wheels with parity; pin protocol compatibility in package metadata.
- **TypeScript/Java:** publish with SemVer; keep generated types in lockstep with schema bundles.

---

## 11) Capability Gating & Negotiation

Adapters **MUST** return `capabilities()` with:

- `protocol: "<component>/v<major>.0"` (schema const),
- feature flags (e.g., `supports_deadline`),
- limits (e.g., `max_top_k`, `max_dimensions`),
- vendor extensions only where schema allows (e.g., open capabilities schemas).

Clients **SHOULD**:

- probe `capabilities()` at startup,
- gate optional features based on flags,
- avoid assuming presence of non-advertised fields.

Adding a new capability flag is **MINOR**. Removing or inverting its semantics is **MAJOR**.

---

## 12) Migration Checklist (for maintainers)

Before merging a change that affects public API/spec:

1. **Classify:** breaking vs additive vs editorial.
2. **Bump:** version(s) in spec/schemas/SDKs/adapters accordingly.
3. **Negotiate:** ensure fallback behavior or capability flag for optional features.
4. **Docs:** update `CHANGELOG.md` and any migration notes.
5. **Tests:** extend conformance tests.
   - Breaking → add negative tests for old behavior.
   - Additive → add positive tests showing older peers ignore fields where allowed.
6. **Deprecation:** mark deprecated and document timeline if applicable.
7. **Security/Privacy:** reconfirm SIEM-safe invariants.

---

## 13) Examples

### 13.1 Add a new optional match attribute (Vector)

- **Change:** add `normalized_score` to `VectorMatch` only if the VectorMatch schema permits additional properties (or via a schema MINOR that adds an optional field).
- **Version:** **MINOR**.
- **Behavior:** Adapters may populate; clients not expecting it ignore the field (only if schema allows).

### 13.2 Tighten error retryability (LLM)

- **Change:** classify an error from retryable → non-retryable.
- **Version:** **MAJOR** (changes client behavior).
- **Migration:** document new handling; optionally add a transitional capability flag if both behaviors exist temporarily.

### 13.3 Add `supports_deadline` to capabilities (Embedding)

- **Change:** capability flag addition.
- **Version:** **MINOR**.
- **Behavior:** Adapters not supporting deadlines keep flag false; clients must check before using.

---

## 14) Versioning for Documentation & Examples

- Examples are non-normative unless explicitly marked.
- Example updates are typically **PATCH** unless they demonstrate new normative behavior (which must be versioned accordingly).
- Conformance suites track coverage deltas per protocol/spec version.

---

## 15) Security, CVEs, and Backports

- Security fixes may be backported to supported LTS lines and latest stable MINOR.
- If a security fix changes normative behavior:
  - Prefer an opt-in flag defaulting to safe behavior, or
  - ship as **PATCH** with explicit release notes if the change is clearly in-scope for safety.
- Publish advisories with fixed versions and upgrade guidance.

---

## 16) Notices & Legal

- `NOTICE` and license headers are maintained per Apache 2.0.
- Version tags and changelogs must not include vendor confidential material.

---

## 17) FAQ

**Q: Can an SDK 1.9.0 support protocol vector/v2?**  
A: Only if it explicitly advertises support (e.g., via `sdk_support.json`) and negotiates the requested protocol major. Otherwise, expect `NotSupported`.

**Q: Are additive validation checks breaking?**  
A: If they reject previously valid inputs, yes → **MAJOR**. If they only reject previously undefined/invalid inputs, it's **PATCH/MINOR** (document rationale).

**Q: Can we add a new error hint (e.g., `suggested_batch_reduction`)?**  
A: **MINOR** additive; put it in `error.details` and require clients to ignore unknown hints.

---

## 18) Quick Decision Table

| Change Type | Bump | Notes |
|---|---|---|
| Add optional field / capability flag | MINOR | Must be ignorable by older peers where schema allows |
| Add new operation (optional) | MINOR | Gate by capability |
| Tighten validation that rejects previously valid input | MAJOR | Document migration |
| Rename/remove required field | MAJOR | Breaking wire change |
| Change error retryability guidance | MAJOR | Alters client behavior |
| Add new error subtype | MINOR | Additive taxonomy |
| Observability: add low-cardinality attribute | MINOR/PATCH | Ensure privacy invariants |
| Doc/typo/edit only | PATCH | No behavior change |

---

## 19) Checklist for Release Managers

- [ ] Version(s) bumped across spec, schemas, SDKs, adapters
- [ ] Changelog entries complete (Added/Changed/Deprecated/Removed/Fixed/Security)
- [ ] Conformance tests updated/passing
- [ ] Deprecations noted with timeline
- [ ] Capabilities and negotiation tested end-to-end
- [ ] Tags pushed; artifacts published
- [ ] NOTICE/LICENSE updated if new third-party code included

---

## 20) Change History

- **v1.0** (Initial): Establishes SemVer policy across spec/wire/SDK/adapters; defines negotiation rules, deprecation, LTS, and migration procedures.