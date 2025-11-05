# SECURITY.md

**Security Policy — Corpus Protocol Suite & SDKs**

> This document defines how to report vulnerabilities, how we triage and remediate them, and how we coordinate disclosure across the **Corpus Protocol Suite** (Graph, LLM, Vector, Embedding) and the **Corpus SDK** implementations. It complements our privacy/observability rules in the spec (see §13 and §15) and versioning policy in `VERSIONING.md`.

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

We aim to acknowledge every report within **48 hours**.

### Include in your report (helps us triage fast)

* Affected component(s) and version(s) (commit/tag if possible)
* Reproduction steps, PoC, and minimal test case
* Impact assessment (what an attacker gains/denies)
* Environment details (OS, runtime, configuration)
* Any temporary mitigations you discovered

**Please do not** file security reports in public issues or PRs.

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

## 3) Triage & Response SLAs

We classify severity using **CVSS v3.1** and target these timelines:

| Severity                | Example Impact                          | First Response | Triage Complete | Fix or Mitigation |
| ----------------------- | --------------------------------------- | -------------: | --------------: | ----------------: |
| **Critical (9.0–10.0)** | RCE, auth bypass, secret exfiltration   |          ≤ 24h |           ≤ 72h |          ≤ 7 days |
| **High (7.0–8.9)**      | Privilege escalation, SSRF with privesc |          ≤ 48h |        ≤ 5 days |         ≤ 14 days |
| **Medium (4.0–6.9)**    | DoS requiring unusual preconditions     |       ≤ 5 days |       ≤ 10 days |         ≤ 30 days |
| **Low (0.1–3.9)**       | Info leaks with limited risk            |      ≤ 10 days |       ≤ 20 days |       Best effort |

We may update timelines case-by-case if exploitation in the wild is observed.

---

## 4) Coordinated Disclosure & **Embargo Policy**

* We follow **coordinated disclosure**: we work with the reporter on a fix and publish an advisory when patches are available.
* **Default embargo window:** up to **90 days** from initial acknowledgment, or until a fix is released on supported branches—whichever is sooner.
* **Expedite/Extend:**

  * We **shorten** the embargo if active exploitation is confirmed.
  * We may **extend** briefly (≤ 30 additional days) if patches are ready but require synchronized releases across multiple packages.
* Reporters may share details with affected vendors under the same embargo. Please notify us of cross-vendor coordination so we can align releases.

---

## 5) Safe Harbor for Researchers

We will not pursue legal action for **good-faith** research that:

* Avoids privacy violations, service disruption, and data destruction,
* Respects rate limits and **does not** access data you do not own,
* Does not exfiltrate raw prompts, vectors, or tenant identifiers,
* Follows the **embargo** and coordinated disclosure terms above.

If in doubt, ask us first at [security@adaptersdk.org](mailto:security@adaptersdk.org).

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

---

## 8) Supply-Chain & Build Integrity

* **SBOMs**: generated per release (SPDX or CycloneDX) and attached to artifacts.
* **Signing**: tags and release artifacts are signed (GPG/Sigstore). Verify before deploying.
* **Dependencies**: pinned with hash/lockfiles; automated alerts for known CVEs.
* **CI hygiene**: principle of least privilege, ephemeral runners where possible, no plaintext secrets in logs.
* **Secrets management**: never commit secrets; rotate credentials on suspicion or leak.

If you find an exposed secret, **email us immediately** and do not test it.

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
* Emergency (24/7 best effort): prefix subject with **[URGENT]**
* PGP fingerprint: `XXXX XXXX XXXX XXXX XXXX  XXXX XXXX XXXX XXXX XXXX`

Thank you for helping keep the Corpus ecosystem secure.

