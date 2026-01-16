# Relationship and Normative Priority

This repo defines the Corpus Protocol Suite across multiple documents. Some are normative
(binding) and some are descriptive. This file defines which document is authoritative for
which kind of requirement and how to resolve conflicts.

## 1) The Core Rule

When two documents disagree, implementations MUST follow the highest-precedence source
for the subject being defined (shape vs behavior vs guidance).

## 2) Precedence by Topic

### A) Wire Shapes and Validation (JSON field names/types/requiredness)
**Authoritative:** `SCHEMA.md` and the JSON Schema files under `schemas/`

- JSON field names, types, constraints, required/optional status, enums, and
  `additionalProperties` behavior are defined by JSON Schema and are normative.
- If `PROTOCOLS.md` or `SPECIFICATION.md` includes an example that differs from schema,
  the schema is authoritative and the prose MUST be updated to match.

**Rule:** If it must validate on the wire, the schemas decide.

### B) Operation Semantics (behavior and lifecycle)
**Authoritative:** `PROTOCOLS.md`

Examples:
- Deadline propagation / budget enforcement semantics
- Idempotency semantics
- Retryability and error classification semantics
- Streaming lifecycle semantics (terminal conditions, ordering, “no frames after terminal”)
- Batch semantics (partial failure rules, ordering guarantees, atomic vs non-atomic)

**Rule:** If it’s about what an operation means or how it behaves, PROTOCOLS decides.

### C) Architecture, Motivation, and Rationale
**Informative:** `SPECIFICATION.md`

- `SPECIFICATION.md` provides architecture, design intent, and context.
- It is descriptive unless a section is explicitly labeled “Normative”.
- If it conflicts with `SCHEMA.md` (shape) or `PROTOCOLS.md` (semantics), it defers.

## 3) Optional Companion Documents 

These files are authoritative for their subject matter, while `SCHEMA.md`
remains authoritative for the *shape* of any related envelopes/types.

- `ERRORS.md`: canonical error taxonomy / normalization meanings
- `METRICS.md`: observability requirements, naming, labels, redaction rules
- `IMPLEMENTATION.md`: patterns and non-normative guidance
