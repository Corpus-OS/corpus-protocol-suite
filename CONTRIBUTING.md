# CONTRIBUTING.md

> Version: **1.0** (Apache-2.0 compliant)
> Applies to: **Code**, **Specifications (RFC-style docs)**, and **Conformance Tests**
> Scope: **Corpus Protocol Suite** — Graph, LLM, Vector, Embedding adapters and shared infrastructure

---

## 1) Scope & Principles

* **Open by default, interoperable by design.** Contributions SHOULD preserve cross-protocol portability and capability discovery.
* **Privacy-first telemetry.** As per the spec, never log raw prompts, vectors, or tenant identifiers; prefer SIEM-safe hashes and low-cardinality attributes only.
* **Minimal, additive surfaces.** Favor optional capability flags over breaking method signatures; follow SemVer for protocol changes.
* **Testable contracts.** Add or update conformance tests when you change normative behavior.

---

## 2) Developer Certificate of Origin (DCO 1.1)

We use the **DCO** instead of a CLA. By contributing, you agree to the DCO terms (see [https://developercertificate.org/](https://developercertificate.org/)). Sign your commits with:

```
Signed-off-by: Your Name <you@example.com>
```

Most Git UIs can add this automatically with `--signoff`.

**Why DCO?** It gives us a clear, auditable provenance chain for contributions without the overhead of a separate legal agreement (CLA).

---

## 3) IP & Patent Policy (Apache-2.0)

* All contributions are accepted under **Apache License 2.0** with its **patent grant** (Apache §3).
* Do **not** contribute code/spec you don’t have the right to submit.
* If your contribution may include a patentable idea, **disclose** it in the PR using the provided template block (optional but recommended).
* If your employer owns your IP, ensure you have **corporate consent** (see “Corporate Contributions” below).

**Corporate contributions (optional but recommended):**
If you contribute in the scope of employment or using employer IP, we recommend the PR includes a short note that your employer authorizes the contribution under Apache-2.0. For larger changes, a brief email from a corporate signatory (to project maintainers) avoids later ownership disputes.

---

## 4) Contribution Types & Dual-Track Expectations

We accept three contribution types. Use this table to align your changes:

| Type                               | Normative?                        | Must/Should Language                                | Privacy Rules                                                | What to include                                                |
| ---------------------------------- | --------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| **Code** (adapters, libs)          | **Yes** (behavior)                | Use comments for rationale; follow spec MUST/SHOULD | No PII in logs; SIEM-safe metrics                            | Tests, benchmarks (if perf-related), docs                      |
| **Specification text** (RFC-style) | **Yes** (MUST/SHOULD/RECOMMENDED) | BCP 14 language only                                | No inclusion of proprietary or third-party confidential text | Change log note, rationale, examples labeled **Non-Normative** |
| **Conformance tests**              | **Yes** (define pass/fail)        | Tests must assert normative requirements            | No raw prompts/vectors/tenant IDs in logs                    | Test docstrings include section citations (e.g., “§8.3”)       |

> **Spec provenance (required):** By submitting or editing specification text, you confirm the text is **original** to you or **appropriately attributed** under a license compatible with Apache-2.0, and does **not** copy standards-essential text in a way that violates third-party rights.

---

## 5) Code Style & Quality Gates

* **Language:** Python ≥ 3.10 (type-annotated).
* **Lint/Format:** `ruff`, `black`, `isort`.
* **Typing:** `mypy --strict` (or project config).
* **Tests:** `pytest -q` with async markers where applicable.
* **Docs:** Update README/spec excerpts if behavior or flags change.

### 5.1) Performance Contributions (benchmarks & portability)

If your change claims performance improvements, include:

* **Minimal, reproducible benchmark**: inputs, environment, metrics, and how to run.
* **Document trade-offs**: latency vs. memory/cost/accuracy; when to prefer/avoid the change.
* **Portability notes**: CPU/GPU/OS/backends; specify fallbacks and capability gates.
* **Raw data or link** if numbers influence README/spec claims.

---

## 6) Security & Privacy

* **Secrets:** Never commit keys/tokens. Use env vars or secret stores.
* **PII:** Don’t log prompts, vectors, or raw tenant IDs. Hash tenant identifiers and limit metric cardinality.
* **Threat model notes:** If you alter auth/tenancy/error paths, add a brief threat-model comment and tests.

---

## 7) Observability & Metrics

* Emit **low-cardinality** metrics only: component/op/code + `tenant_hash`, `deadline_bucket`, etc.
* Streaming ops MUST emit a **single final outcome metric** after completion/error.
* Use `traceparent` for distributed tracing; don’t stuff content into span attributes.

---

## 8) Tests & Conformance

* Tests MUST cite the spec section(s) they validate (e.g., `“§9.3 — Query Operations”`).
* **Golden rule:** Normative change ⇒ test added/updated.
* Prefer **parametrized** tests for validation ranges and error classes.
* Keep **mock adapters** deterministic (seed RNG) for retry/error semantics.

---

## 9) Commit & PR Process

1. **Small, reviewable diffs**; separate refactors from behavior changes.
2. Link the relevant **spec section(s)** and update the change log when normative.
3. Include **before/after** notes for user-visible behavior.
4. Ensure CI: lint, type check, tests, and coverage gates pass.

### Pull Request Template (copy/paste)

```
## Summary
<what changed and why; user impact>

## Specification Mapping
- Affects: §<section> (<short title>)
- Normative? (Yes/No): 
- Capability flags/toggles: <list or N/A>

## Testing
- [ ] Unit tests updated/added
- [ ] Conformance tests updated/added
- [ ] Benchmarks (if perf claim) with reproducible steps

## Observability & Privacy
- [ ] No raw prompts/vectors/tenant IDs in logs/metrics
- [ ] Final stream outcome metric emitted (if streaming)
- [ ] Deadline bucket labeling preserved (if applicable)

## Docs
- [ ] README/spec snippets updated (if user-visible)
- [ ] CHANGELOG entry added (normative)

## Optional Patent Disclosure (non-binding)
If this contribution may include patentable subject matter, you may optionally describe it here:
- Summary of idea:
- Prior art considered:
- Why non-obvious:
(Disclosure is optional; contributions remain under Apache-2.0 with patent grant.)
```

---

## 10) Governance & Decision Making

* **Maintainers** review for correctness, portability, privacy, and spec alignment.
* **Normative** spec changes require explicit maintainer ACK and change-log updates.
* Disagreements are resolved via an issue thread referencing data, prior art, and user impact.

---

## Appendix A — Legal Notices & Attribution

### License

All contributions are under **Apache License, Version 2.0**. See `LICENSE`.

### NOTICE File (Apache §4) — When & How to Update

Update `NOTICE` when you:

* **Add** third-party Apache-compatible notices that require attribution.
* **Change** the project name, copyright holder, or significant attribution.
* **Bundle/redistribute** artifacts that mandate carrying upstream NOTICE text.

**Template bullets to add under `NOTICE`:**

* Project name and primary copyright holder
* Short attribution for included third-party components (name + license)
* Year ranges updated as needed (e.g., `© 2024–2025`)

---

## Appendix B — File/Folder Conventions

* `corpus_sdk/…` — source; `examples/…` — runnable examples; `tests/…` — conformance.
* SPDX headers at the top of every file:

  ```
  # SPDX-License-Identifier: Apache-2.0
  ```
* Keep examples small, deterministic, and privacy-safe.

---

## Appendix C — Example Checklists

**New Adapter (Vector/LLM/Graph/Embedding)**

* [ ] `capabilities()` returns complete, accurate limits/flags
* [ ] Errors mapped to taxonomy with hints (`retry_after_ms`, `throttle_scope`, etc.)
* [ ] Deadline honored with preflight fail-fast; final stream outcome metric (if streaming)
* [ ] SIEM-safe metrics only; `tenant_hash`, `deadline_bucket` used appropriately
* [ ] Conformance tests pass locally and in CI

**Spec Changes**

* [ ] BCP 14 language; examples flagged *Non-Normative*
* [ ] Provenance confirmed as original/attributed (see §4 callout)
* [ ] Change log updated; version bump if normative

---

## Appendix D — Contribution Workflow Quickstart

1. **Fork** → create a feature branch.
2. Write code/spec/tests with SPDX headers and DCO sign-off.
3. Run: `ruff`, `black`, `isort`, `mypy`, `pytest`.
4. Open PR with the template body; link spec sections and issues.
5. Address review; squash or rebase with signed-off commits.

---

**Thank you for contributing to the Corpus Protocol Suite!**

