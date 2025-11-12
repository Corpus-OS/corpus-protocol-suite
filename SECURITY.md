# SECURITY.md

**Security Policy — Corpus Protocol Suite & SDKs**

> This document defines how to report vulnerabilities, how we triage and remediate them, and how we coordinate disclosure across the **Corpus Protocol Suite** (Graph, LLM, Vector, Embedding) and the **Corpus SDK** implementations. It complements our privacy/observability rules in the spec (see §13 and §15) and versioning policy in `VERSIONING.md`.

> **Important:** This policy **does not** establish service-level agreements (SLAs). The project is community-driven and security handling is **best-effort** unless you have a commercial support contract.

---

## 1) Report a Vulnerability

* **Email (preferred):** [security@adaptersdk.org](mailto:security@adaptersdk.org)
* **security.txt:** `https://adaptersdk.org/.well-known/security.txt`
* **Public key (PGP):**

  ```
  -----BEGIN PGP PUBLIC KEY BLOCK-----
  [PGP PUBLIC KEY HERE]
  -----END PGP PUBLIC KEY BLOCK-----
  ```

  **PGP fingerprint:** `XXXX XXXX XXXX XXXX XXXX  XXXX XXXX XXXX XXXX XXXX`

We aim to acknowledge every report within **7 business days**.

> **Clarification:** “Business days” are counted globally; primary triage timezone is **US Central**. We attempt to reply sooner for high-severity issues.

### Include in your report (helps us triage fast)

* Affected component(s) and version(s) (commit/tag if possible)
* Reproduction steps, PoC, and minimal test case
* Impact assessment (what an attacker gains/denies)
* Environment details (OS, runtime, configuration)
* Any temporary mitigations you discovered

**Please do not** file security reports in public issues or PRs.

**Large PoCs & sensitive data.** If your PoC is large or contains sensitive material, please share via an **encrypted link** (PGP-encrypted if possible) rather than inline email content. Do **not** email secrets in cleartext.

---

## 2) Scope

**In scope**

* Protocol specs & reference/adapters (Graph, LLM, Vector, Embedding)
* SDK packages, example adapters, conformance tests, build scripts
* CI/CD pipelines and release tooling for the above

**Out of scope**

* Third-party providers/backends not maintained by us
* Social engineering, physical attacks, or hosting provider issues
* Non-security bugs (use normal issue tracker)

If unsure, send the report—we’ll redirect as needed.

---

## 3) Response Process & Expectations (Community-Driven, No SLAs)

We are a community-driven open-source project. Security reports are handled on a **best-effort** basis by volunteer maintainers and contributors.

**Our Commitment (non-binding goals):**

* We **aim** to acknowledge your report within **7 business days**.
* We **prioritize** issues by severity and user impact.
* We **communicate transparently** about triage and next steps.
* We **coordinate disclosure** with reporters and **credit** responsible disclosures.

**Typical (non-binding) timelines, for context only:**

* **Critical:** initial assessment within **1–2 weeks** (subject to availability)
* **High:** triage and remediation planning within **2–4 weeks**
* **Medium/Low:** addressed as capacity permits

**What we cannot guarantee (unless under a commercial contract):**

* Specific fix timelines or delivery dates
* 24/7 emergency response
* Immediate patches for all issues

**Need guaranteed SLAs?** We offer **commercial support** with defined response times and coordination commitments via our enterprise offerings.

> **Legal clarity:** This section sets **expectations**, not contractual obligations. There are **no implied warranties** or SLAs in the open-source project.

---

## 4) Coordinated Disclosure & **Embargo Policy**

* We follow **coordinated disclosure**: we work with the reporter on a fix and publish an advisory when patches are available.
* **Default embargo window:** up to **90 days** from initial acknowledgment, or until a fix is released on supported branches—whichever is sooner.
* **Expedite/Extend:**

  * We **shorten** the embargo if active exploitation is confirmed.
  * We may **extend** briefly (≤ 30 additional days) if patches are ready but require synchronized releases across multiple packages.
* Reporters may share details with affected vendors under the same embargo. Please notify us of cross-vendor coordination so we can align releases.

> **Note:** If we confirm **active exploitation**, we may publish an expedited advisory with mitigations prior to releasing full patches, then follow with patched releases and backports.

---

## 5) Safe Harbor for Researchers

We will not pursue legal action for **good-faith** research that:

* Avoids privacy violations, service disruption, and data destruction,
* Respects rate limits and **does not** access data you do not own,
* Does not exfiltrate raw prompts, vectors, or tenant identifiers,
* Follows the **embargo** and coordinated disclosure terms above.

If in doubt, ask us first at [security@adaptersdk.org](mailto:security@adaptersdk.org).

**Load/DoS testing restriction:** Please **do not** perform stress/load testing against production infrastructure without prior written consent.

---

## 6) Fix & Release Process

1. **Assign & reproduce** internally; confirm affected versions.
2. **Develop fix** on a private branch. Add tests and conformance checks.
3. **Backport** to **supported branches** (see table below).
4. **Sign releases** and publish patched versions (see `VERSIONING.md` for SemVer).
5. **Advisory** (GHSA/OSV): coordinated with reporter, includes CVSS, affected versions, mitigations, and acknowledgements.
6. **NOTICE updates** (Apache §4): update `NOTICE` when:

   * Copyright or attribution text changes,
   * We include third-party notices for newly bundled assets,
   * The name of the work or significant components change.

**NOTICE update template (3 bullets)**

* Project: **Corpus Protocol Suite** — updated attribution for `[component/version]`
* Third-party notice added/updated: `[library]` under `[license]`
* Effective as of release: `[tag/date]`

---

## 7) Supported Branches (Security Fix Eligibility)

> Maintainers: keep this in sync with `VERSIONING.md`.

|  Major | Status | Accepting Sec Fixes | Planned EOL |
| -----: | ------ | :-----------------: | :---------: |
| **v1** | Stable |          ✅          |  2026-12-31 |
| **v0** | Legacy |          ❌          |     EOL     |

We typically support the **current major** and backport **security-only** patches when feasible.

> **Best-effort backports:** When a new major is released, we generally provide security backports for the previous major for ~12 months.

---

## 8) Supply-Chain & Build Integrity

* **SBOMs**: generated per release (SPDX or CycloneDX) and attached to artifacts.
* **Signing**: tags and release artifacts are signed (GPG/Sigstore). Verify before deploying.
* **Dependencies**: pinned with hash/lockfiles; automated alerts for known CVEs.
* **CI hygiene**: principle of least privilege, ephemeral runners where possible, no plaintext secrets in logs.
* **Secrets management**: never commit secrets; rotate credentials on suspicion or leak.

If you find an exposed secret, **email us immediately** and do not test it.

**Example: Sigstore / cosign verification (adjust identity/issuer as appropriate)**

```bash
cosign verify-blob \
  --certificate-identity-regexp 'https://github.com/yourorg/yourrepo' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  --signature artifact.sig artifact.tar.gz
```

---

## 9) Handling Secrets Exposure

If a credential appears in code, logs, or artifacts:

1. Notify `security@adaptersdk.org` (PGP-encrypted if possible).
2. We will **revoke/rotate** affected credentials promptly.
3. We will assess blast radius and publish remediation steps if user-facing.
4. Public commit history may be rewritten to remove secrets (with notice).

---

## 10) Temporary Mitigations (When Patching Isn’t Immediate)

* Disable vulnerable feature flags or endpoints where documented.
* Tighten network policies, auth scopes, or rate limits.
* Increase observability on suspicious metrics (e.g., anomaly spikes).
* For protocol adapters, prefer **“thin” mode** behind a hardened router.

Mitigations will be listed in advisories when applicable.

---

## 11) Severity Guidance (Quick Reference)

We use **CVSS v3.1** to guide severity. Examples:

| Category     | Examples                                                                       |
| ------------ | ------------------------------------------------------------------------------ |
| **Critical** | RCE, auth bypass, arbitrary secret read, tenant escape                         |
| **High**     | Privilege escalation, SSRF with metadata access, code injection requiring auth |
| **Medium**   | DoS, significant info disclosure with constraints                              |
| **Low**      | Best-practice deviations, limited info leaks                                   |

Final severity may differ after full analysis.

---

## 12) Hardening Recommendations (For Deployers)

* Isolate tenants; never log raw **tenant IDs**, **prompts**, **vectors** (see spec §13, §15).
* Enforce absolute **deadlines** and timeouts; prefer fail-fast on expired budgets.
* Use minimal scopes for provider API keys; rotate regularly.
* Enable WAF/IDS and alert on unusual token throughput and concurrency.
* Keep SDKs and adapters up-to-date; subscribe to advisories.

---

### 12a) Advisory Publication & Ecosystem Coordination (Added)

* We publish advisories to **OSV** and **GHSA** (GitHub Security Advisories) when applicable.
* Where relevant, we may publish **VEX** (Vulnerability Exploitability eXchange) to document non-affected components or non-exploitable paths.
* For issues originating in **third-party dependencies**, we:

  * Track upstream advisories and fixes,
  * Assess impact on Corpus SDKs and protocol adapters,
  * Provide mitigations where feasible,
  * Reference upstream advisories in our notes.

---

## 13) Hall of Fame

We credit reporters who request acknowledgement.
Include the name or handle, and optional link, in your email if you wish to be recognized.

---

## 14) Changes to This Policy

We may revise this policy; material changes will be noted in the repository changelog and release notes. Prior versions remain available in git history.

---

## 15) Hotfix Advisory Template (for Maintainers)

```
Title: [CVE/GHSA] <Short vulnerability title>

Summary
- Affected components: <packages/modules>
- Impact: <what an attacker can do>
- Severity: <CVSS score + vector string>

Affected Versions
- <>= v1.2.0, < v1.4.3
- <other lines as needed>

Patched Versions
- v1.4.3, v1.3.9 (backport)

Mitigations
- <config flag> = false, or disable <endpoint>, or restrict <scope>

Indicators of Compromise
- <relevant logs/metrics to watch>

Credits
- Reported by <Name/Handle> (thanks!)

Timeline
- YYYY-MM-DD report received
- YYYY-MM-DD fix completed
- YYYY-MM-DD coordinated release

References
- OSV/GHSA link(s), commit hashes, release notes
```

---

## 16) Contact

* Security team: **[security@adaptersdk.org](mailto:security@adaptersdk.org)**
* Emergency (best effort): prefix subject with **[URGENT]**
* PGP fingerprint: `XXXX XXXX XXXX XXXX XXXX  XXXX XXXX XXXX XXXX XXXX`

**Thank you** for helping keep the Corpus ecosystem secure.

---

## Appendix A — `security.txt` Template (Added)

Place at: `https://adaptersdk.org/.well-known/security.txt` (and optionally `/security.txt` in the repo root).

```
Contact: mailto:security@adaptersdk.org
Encryption: https://adaptersdk.org/pgp.txt
Preferred-Languages: en
Policy: https://github.com/yourorg/yourrepo/blob/main/SECURITY.md
Canonical: https://adaptersdk.org/.well-known/security.txt
Expires: 2026-12-31T23:59:59Z
```

**Bug bounty stance:** We do **not** run a public bounty at this time. Responsible disclosures are acknowledged in our Hall of Fame. If this changes, we will update this policy and the `security.txt`.
