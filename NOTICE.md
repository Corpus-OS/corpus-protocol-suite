# NOTICE

**Corpus Protocol Suite & SDKs**
Copyright © 2025 The Corpus Protocol Suite Contributors

This product is licensed under the **Apache License, Version 2.0** (the “License”).
You may not use this file except in compliance with the License.
You may obtain a copy of the License at: **[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)**

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an **“AS IS” BASIS**, **WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See the License for the specific language governing permissions and limitations under the License.

This product includes software developed by the **Corpus Protocol Suite** contributors:
**[https://github.com/adapter-sdk](https://github.com/adapter-sdk)** • **[https://adaptersdk.org](https://adaptersdk.org)**

---

## Attributions

The following third-party components may be included in source or binary distributions. Their licenses and notices are provided below or in the component’s source directory.

> Replace the placeholders with actual dependencies when they are bundled or redistributed.

* **[NAME OF LIBRARY]** — © [YEAR] [OWNER]. Licensed under **[LICENSE NAME & VERSION]**.
  Source: [URL] • License: [URL or see `THIRD_PARTY_LICENSES/NAME-LICENSE`]

* **[NAME OF SPEC TEXT / DIAGRAMS]** — Portions © [YEAR] [OWNER].
  Included per **[LICENSE]** with required attribution.

> If you **dynamically** depend on a library (and do not redistribute it), you typically do **not** list it here; follow the library’s license.

---

## Trademarks

“Corpus”, “Corpus Protocol Suite”, and related marks are trademarks or registered trademarks of their respective owners. This notice does **not** grant permission to use trademarks, logos, or brand features except as permitted by applicable law.

---

## NOTICE Update Guidance (Apache License §4)

Update this **NOTICE** file whenever your distribution requires additional attribution or notice content. Typical triggers:

1. **New or changed attributions** — You add/replace a third-party component that requires **NOTICE** text in redistributed source/binaries.
2. **Copyright updates** — The project copyright line or ownership statement needs revision.
3. **Work name/components change** — The official name of the work or significant subcomponents change and must be reflected here.

**Definition — “Material Changes.”** For clarity in this project, **material changes** are changes that affect **functionality, security, or interoperability** of the software or specifications.

**Three-bullet template (scannable):**

```
• Modifications by: <Your Org> © <Year>
• Changes: <brief_description>
• Third-party: <component, version, license>
```

---

## Files Referenced by NOTICE

* `LICENSE` — The Apache License, Version 2.0
* `THIRD_PARTY_LICENSES/` — Collated third-party license texts (if bundled)
* Component-specific `NOTICE` files — Some submodules may include their own `NOTICE` content; include or merge as required when redistributing.

---

## Attribution Guidance for Forks (Apache-2.0 compliant)

Forks and redistributions **MUST**:

* **Retain** this project’s **LICENSE** (Apache-2.0) and this **NOTICE** file (including our copyright line and any third-party notices) per **Apache License §4**.
* **Preserve** upstream **copyright** and **license notices** in all copied files.
* **Add/modify** NOTICE entries to reflect **material changes** you introduce (see definition above).
* **Avoid** any use of our **trademarks** (names, logos, brand elements) that suggests sponsorship or endorsement. Trademark use requires separate permission.

**Project Policy (recognition request; not a license condition):**
To maintain clear provenance and machine-readable compliance, **please keep** the SPDX header in inherited files:

```
SPDX-License-Identifier: Apache-2.0
```

If you modify files, **add** your own copyright line(s) **without removing** ours, e.g.:

```
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 The Corpus Protocol Suite Contributors
Copyright (c) 2026 Your Org
```

> This SPDX retention request is a **project policy** to aid recognition and compliance; the Apache-2.0 license itself does **not** mandate SPDX tags, but it **does** require preservation of license and attribution notices.

**Optional courtesy attribution (not a license condition):**
If you fork or redistribute, we kindly request you include the following in your README or about page:

> “This project is based on the Corpus Protocol Suite & SDKs (Apache-2.0): [https://github.com/adapter-sdk”](https://github.com/adapter-sdk”)

---

## Dynamic vs. Static Dependencies (Attribution Clarification)

To reduce confusion about when to add items to **NOTICE**:

* **Static linking / vendored code / bundled assets** → **Include attribution** in NOTICE (and add license text under `THIRD_PARTY_LICENSES/`).
* **Dynamic linking / system package / service dependency** → Typically **no NOTICE entry** (you’re not redistributing it). Follow the dependency’s license terms.
* **Generated files / embedded specs/diagrams** → If redistributed and the source license requires attribution, **add** to NOTICE.

---

If you have questions about whether **NOTICE** requires an update for your change, open a PR with your proposed additions under the **Attributions** section or ask in `#legal-oss` (internal) / open a discussion on the repository.

