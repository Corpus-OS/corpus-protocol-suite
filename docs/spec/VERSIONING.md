# VERSIONING

**Corpus Protocol Suite — Versioning & Compatibility Policy**

> This document describes how we version the **spec**, **wire contracts (schemas/envelopes)**,
> **SDK libraries**, and **adapters** across the Corpus Protocol Suite (Graph, LLM, Vector, Embedding).
> It complements `SCHEMA.md` (wire shapes) and `PROTOCOLS.md` (operation semantics).

---

## 1) Goals

- **Predictability.** Consumers can upgrade with clear expectations about breakage risk and compatibility.
- **Interoperability.** Adapters and clients that share a protocol **MAJOR** must remain wire-compatible.
- **Velocity with safety.** Additive changes flow under **MINOR/PATCH** without lockstep upgrades.
- **Auditability.** Every release is traceable to a changelog entry and migration path when needed.

---

## 2) What we version (at a glance)

| Thing | Example | Scheme | Compatibility Contract |
| --- | --- | --- | --- |
| **Protocol ID** (wire-level major identifier, per component) | `llm/v1.0`, `vector/v1.0` | **Major-stable identifier** | **Stable for the entire MAJOR**; does not change on MINOR/PATCH spec updates |
| **Protocol spec** (per component) | `protocols_version: 1.2` | **SemVer** (`MAJOR.MINOR[.PATCH]`) | Same-**MAJOR** wire compatibility; **MINOR** strictly additive; **MAJOR** may break |
| **Wire envelopes / JSON schemas** | `schema_version: 1.0.3` | **SemVer** (`MAJOR.MINOR.PATCH`) | Adding optional fields = **MINOR**; changing/removing required fields or types = **MAJOR** |
| **SDK packages** (language libs) | `corpus-sdk-py 1.7.2` | **SemVer** | Public API stability aligned with protocol **MAJOR**; SDK **MINOR/PATCH** must not break user code |
| **Adapters** (provider impls) | `corpus-adapter-qdrant 2.3.0` | **SemVer** | Adapter **MAJOR** may drop old protocol **MAJOR**; MUST truthfully advertise its `protocol` in `capabilities()` |
| **Spec repository snapshot** | Git tag `spec/v1.2.0` | **SemVer** tags | Tags track immutable snapshots for citations + conformance suites |

> **Key invariant:** If a client and adapter both speak `<component>/v1.0`, they must interoperate on the wire, regardless of SDK/package versions.

---

## 3) Semantic Versioning (how to decide MAJOR/MINOR/PATCH)

We follow **SemVer 2.0.0** for all artifacts.

### 3.1 Protocol & wire contracts (normative)

- **MAJOR** (`2.0.0`): breaking change to required fields, field types, error semantics, or operation signatures.
  - Examples: rename/remove required field; change meaning of `score`; tighten validation so previously-valid inputs are rejected.
- **MINOR** (`1.2.0`): additive change.
  - Examples: new optional fields; new optional operation; new capability flag; new error subtype.
- **PATCH** (`1.1.3`): editorial clarifications, typos, non-behavioral fixes, test updates that do not change the normative contract.

### 3.2 SDK libraries & adapters (public API)

- **MAJOR:** breaking public API changes.
- **MINOR:** additive APIs and safe defaults; performance improvements.
- **PATCH:** bug fixes, docs, tooling, non-breaking typing.

---

## 4) Compatibility Matrix

**Within the same protocol MAJOR, wire compatibility is required.** Additive fields must be ignored
*only where the schema explicitly permits extension*.

| Client Protocol | Adapter Protocol | Expectation |
| --- | --- | --- |
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

Note: capabilities.protocol is a single value (const per schema). Supported protocol sets must be communicated via error.details (or via an explicitly versioned capabilities schema in a future MAJOR).

⸻

5) Backward/Forward Compatibility Rules (Normative)

5.1 Unknown JSON keys
	•	Request envelopes: MAY include extra top-level keys beyond op, ctx, args. Adapters MUST ignore them.
	•	ctx: unknown fields MUST be ignored (additionalProperties: true).
	•	args: behavior is schema-governed per operation:
	•	If the op’s args schema is open (additionalProperties: true), unknown keys MUST be ignored.
	•	If the op’s args schema is closed (additionalProperties: false), unknown keys MUST cause validation failure (BAD_REQUEST).
	•	Response envelopes:
	•	Unary success, error, and streaming success envelopes are closed objects where schema says so.
	•	Unknown top-level keys in closed envelopes MUST NOT be emitted.

5.2 Error taxonomy
	•	Adding new error subtypes = MINOR.
	•	Renaming/removing existing classes, changing retryability semantics, or changing canonical codes = MAJOR.

5.3 Observability fields
	•	Adding optional low-cardinality attributes = MINOR/PATCH (must remain SIEM-safe).
	•	Removing a previously-emitted attribute expected by consumers = MAJOR.

5.4 Security/Privacy
	•	Tightening redaction / expanding “do-not-log” rules = MINOR.
	•	Loosening redaction guarantees = MAJOR.

5.5 Vector scoring conventions
	•	“Higher is better” (or any scoring direction invariant) is MAJOR if changed.

⸻

6) Deprecation Policy

We implement the deprecation process with the following operational details:
	1.	Announce in release notes + CHANGELOG.md with rationale and migration hints.
	2.	Warn at runtime where feasible (e.g., a low-cardinality deprecation indicator in metrics).
	3.	Minimum support window: keep deprecated features for ≥ 6 months or one full MAJOR, whichever is longer (except urgent security issues).
	4.	Removal occurs only at the next MAJOR.

On the wire: mark deprecated fields with "deprecated": true in JSON Schema (when applicable) and keep accepting/parsing until the next MAJOR.

⸻

7) Release Process & Tagging

Git tags (recommended)
	•	Suite/spec snapshots: spec/vX.Y.Z (e.g., spec/v1.2.0)
	•	SDKs: <lang>/vX.Y.Z (e.g., python/v1.7.2)
	•	Adapters: adapter-<name>/vX.Y.Z (e.g., adapter-qdrant/v2.3.0)

Branches (recommended)
	•	main: active development (next MINOR).
	•	release/vX.Y: stabilization branch.
	•	maint/vX.Y: maintenance branch for backports.

Artifacts
	•	Every release updates CHANGELOG.md with: Added / Changed / Fixed / Deprecated / Removed / Security.
	•	If schemas touched: bump schema_version and update golden/conformance coverage accordingly.
	•	If protocol semantics touched: bump protocols_version and update conformance tests.

⸻

8) Pre-Releases, RCs, and Experimental Features
	•	Pre-releases: v1.2.0-rc.1, v1.2.0-beta.2 (no compatibility guarantees).
	•	Experimental flags (code): explicit opt-in; default off.
	•	Experimental wire features: only via explicit extension points; do not rely on for cross-vendor compatibility.

⸻

9) Long-Term Support (LTS)
	•	Designate certain MINOR lines as LTS (e.g., 1.4.x) supported 12 months for security/critical fixes.
	•	LTS releases are PATCH-only; no new features; no breaking behavior.

⸻

10) Multi-Language Package Versioning

To reduce cross-language drift:
	•	Source of truth: spec tag spec/vX.Y.Z + protocols_version.
	•	Language SDKs SHOULD ship a machine-readable mapping of supported protocol MAJORs (and optionally spec MINORs):

{
  "llm": "v1.0",
  "vector": "v1.0",
  "graph": "v1.0",
  "embedding": "v1.0"
}

	•	Go: /v2 module path only on SDK MAJOR.
	•	Python: __version__ SemVer; publish wheels; include protocol support mapping in package metadata.
	•	TypeScript/Java: publish SemVer; keep generated types in lockstep with spec tag.

⸻

11) Capability Gating & Negotiation

Adapters MUST return capabilities() with:
	•	protocol: "<component>/v<major>.0" (schema-const per component),
	•	feature flags and limits as defined by that component’s schema,
	•	vendor extensions only where the schema permits openness.

Clients SHOULD:
	•	probe capabilities() at startup,
	•	gate optional features on flags,
	•	avoid assuming presence of non-advertised fields.

Adding a new capability flag is MINOR only if it is compatible with the schema strategy for that component (open vs closed). If the capability result schema is closed, introducing new fields requires a schema/versioning plan consistent with SemVer and validation expectations.

⸻

12) Migration Checklist (for maintainers)
	1.	Classify: breaking vs additive vs editorial.
	2.	Bump versions: protocols_version, schema_version, SDK/adapters as applicable.
	3.	Ensure negotiation/fallback: use capability flags or explicit error.details for supported protocols.
	4.	Docs: update CHANGELOG.md, and any migration notes.
	5.	Tests:
	•	Breaking → add negative tests for old behavior.
	•	Additive → add positive tests + ensure older peers ignore via extension points.
	6.	Deprecation: mark + timeline.
	7.	Security: re-confirm SIEM-safe invariants.

⸻

13) Examples

13.1 Add a new optional match attribute (Vector)
	•	Change: add normalized_score to VectorMatch.
	•	Version: spec MINOR + schema MINOR.
	•	Behavior: adapters may emit; clients ignore only if the type schema permits it (or schema is bumped + validators updated).

13.2 Tighten error retryability (LLM)
	•	Change: classify ContentFiltered from retryable → non-retryable.
	•	Version: MAJOR (changes client behavior guidance).
	•	Migration: document new handling; add tests.

13.3 Add supports_deadline capability flag (Embedding)
	•	Change: capability flag addition.
	•	Version: spec/schema MINOR (if schema already includes the field; otherwise requires schema plan).
	•	Behavior: adapters not supporting deadlines keep it false; clients check before use.

⸻

14) Documentation & Examples
	•	Examples are Non-Normative unless explicitly marked “Normative Example”.
	•	Example updates don’t bump protocol MAJOR unless they introduce new normative requirements.
	•	Conformance tracking should reflect test deltas per spec/schema version.

⸻

15) Security, CVEs, and Backports
	•	Security fixes may be backported to supported LTS + latest stable MINOR.
	•	If a security fix changes normative behavior:
	•	prefer opt-in flags defaulting to safe behavior; otherwise document clearly even for PATCH.

⸻

16) Notices & Legal
	•	Maintain NOTICE and license headers per Apache 2.0.
	•	Tags/changelogs must not include confidential vendor details.

⸻

17) FAQ

Q: Can an SDK 1.9.0 support protocol vector/v2?
A: Only if it explicitly advertises support and negotiates v2. Otherwise expect NotSupported.

Q: Are additive validation checks breaking?
A: If they reject previously-valid inputs, yes → MAJOR. If they only reject previously-undefined/invalid inputs, PATCH/MINOR (document rationale).

Q: Can we add a new error hint (e.g., suggested_batch_reduction)?
A: Yes as MINOR, placed under error.details so unknown hints are ignored.

⸻

18) Quick Decision Table

Change Type	Bump	Notes
Add optional field / capability flag	MINOR	Must be ignorable via schema-defined extension points (or schema/version plan)
Add new operation (optional)	MINOR	Gate by capability
Tighten validation that rejects previously valid input	MAJOR	Document migration
Rename/remove required field	MAJOR	Breaking wire change
Change error retryability guidance	MAJOR	Alters client behavior
Add new error subtype	MINOR	Additive taxonomy
Observability: add low-cardinality attribute	MINOR/PATCH	Preserve privacy invariants
Doc/typo/edit only	PATCH	No behavior change


⸻

19) Checklist for Release Managers
	•	Versions bumped across spec/schema/SDK/adapter as applicable
	•	Changelog complete (Added/Changed/Deprecated/Removed/Fixed/Security)
	•	Conformance tests updated/passing
	•	Deprecations noted with timeline
	•	Capabilities + negotiation tested end-to-end
	•	Tags pushed; packages published
	•	NOTICE/LICENSE updated if needed

⸻

Change History
	•	v1.0 (Initial): Establishes SemVer policy across spec/wire/SDK/adapters; defines negotiation rules, deprecation, LTS, and migration procedures.

