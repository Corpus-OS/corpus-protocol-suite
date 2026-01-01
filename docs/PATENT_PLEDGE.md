## Defensive Patent Pledge 

> **Document Version:** 1.3  
> **Effective Date:** December 23, 2025  
> **Last Updated:** January 1, 2026  
> **Company:** Interoperable Intelligence Inc.

To encourage broad adoption and innovation around the Corpus Protocol Suite, **Interoperable Intelligence Inc.** makes the following **protocol-focused defensive patent pledge**.

This pledge is **intentionally limited to protocol-level essentials** and is **not a waiver** of rights in our routing, orchestration, conformance, cost-normalization, or optimization patents.

---

### 1. Patent Non-Assertion Pledge (Protocol Essentials Only)

**We pledge not to assert our Essential Protocol Claims against you for implementing the Corpus Protocol specifications**, except in defensive counterclaims and subject to the conditions below.

#### 1.1 Covered Activities

This pledge applies only to your activities that are **both**:

1. **Protocol-conformant**, and  
2. Limited to **Core Corpus Protocol interfaces** as defined in `docs/spec/core/` (or their officially designated successors).

Covered activities include:

- Implementing the **Corpus Core Wire Protocols** as described in `docs/spec/core/`
- Building **adapters, clients, servers, or tools** that:
  - Accept and emit Corpus Protocol messages as defined in the Core specifications; and
  - Do not incorporate non-covered patented features (see §1.3 and §1.4)
- Using, modifying, or distributing software implementing the **core wire protocols** under an **Apache-2.0–compatible license**

> ⚠️ **Important:** This pledge covers **protocol-level interoperability**, not all technologies that may coexist with or build on Corpus-compatible systems.

#### 1.2 Essential Protocol Claims

This pledge applies only to **Essential Protocol Claims**, meaning patent claims that are:

- **Necessarily infringed** by any **conformant implementation of the Core Corpus Protocol wire and schema specifications**, and  
- For which **no technically feasible alternative (as defined in §4)** exists to implement the **mandatory** Core Corpus Protocol requirements without infringing.

**Examples of Essential Protocol Claims (Covered):**

- Message formats and envelope structures required for conformance (e.g., required fields, normalization rules)
- Core request/response schemas and error object formats mandated by the Core specifications
- Required negotiation / capability discovery fields where conformance requires specific structures
- On-the-wire behaviors that the Core specification defines as **mandatory** (MUST-level requirements) and that cannot be implemented differently while remaining conformant

#### 1.3 Non-Essential, Non-Covered Claims

The following are **not** covered by this pledge, even if your system uses Corpus:

- **Routing & orchestration technologies**, including but not limited to:
  - Execution strategy selection and meta-routing
  - Multi-strategy fan-out or concurrent orchestration mechanisms
  - Strategy resolution engines and ensemble-based routing decisions
- **Conformance and validation engines**, including:
  - Systems that gate production traffic using golden protocol-layer messages
  - Semantic validation and capability manifests derived from conformance verdicts
  - Health probes and circuit-breakers driven by conformance replays
- **Cost & quota systems**, including:
  - Unified tokenizer-based cost normalization across providers
  - Tenant-level budget enforcement and resource controls tied to heterogeneous domains
- **Advanced error handling & resilience**, including:
  - Tenant-scoped circuit breakers and state machines
  - Backpressure strategies, queue management, and health monitors
- **Optional features, optimizations, and extensions**, including:
  - Extensions beyond the Core specifications in `docs/spec/core/`
  - Proprietary orchestration, ensemble optimization, ranking, or scheduling algorithms
  - Transport-level optimizations, caching strategies, and performance tuning that are not strictly required for Core protocol conformance
- **Any technology where a technically feasible, non-infringing alternative implementation exists** while still complying with the Core protocol specifications

> **Clarifying example:**  
> If you build a Corpus-compatible API gateway that:
>
> - Correctly parses and emits Core Corpus wire messages (covered under this pledge), and  
> - Uses a proprietary orchestration engine, routing strategy chooser, or conformance engine similar to ours (not covered),  
>
> then this pledge protects your **core protocol interoperability layer only**. It does **not** waive our rights with respect to **orchestration, routing, conformance, cost normalization, or similar systems**.

#### 1.4 Reference Implementation Safe Harbor (Wire-Level Only)

To provide additional certainty for protocol implementers:

- Implementation logic **copied from or reasonably derived from** an officially designated **Core wire-level reference implementation** (for example, a `corpus-core-ref` package or repository) that we explicitly mark as such in `docs/spec/core/` or associated documentation,  
- **and** used solely to implement the **Core Corpus Protocol wire and schema specifications**,  

will be treated as **Covered Activity** under this pledge, to the extent that such logic would otherwise infringe our Essential Protocol Claims.

This **safe harbor does _not_ extend to**:

- Any routing, orchestration, conformance, cost-normalization, or optimization components, even if they appear in the same repository or codebase; or  
- Any implementation logic that goes beyond what is necessary to implement the Core wire and schema requirements as defined in `docs/spec/core/`.

---

### 2. Eligibility, Reciprocity & Defensive Suspension

#### 2.1 Who Benefits from this Pledge

Subject to the conditions below, this pledge applies to:

- Individuals and organizations that implement the **Core Corpus Protocol** as specified in `docs/spec/core/` (or successors), and  
- Distribute or use those implementations in compliance with an **Apache-2.0–compatible license**, and  
- Refrain from hostile patent behavior relating to the protocol or its conformant implementers.

#### 2.2 Reciprocity Requirement (Protocol-Level Non-Assert)

This pledge is conditioned on **reciprocity** for protocol essentials:

You must **not** assert any patent claims that you own or control which would be considered **essential protocol claims** (under a definition materially similar to §1.2) **against**:

- Interoperable Intelligence Inc. or its affiliates, or  
- Any other implementer of the Corpus Protocol, **for their conformant Core protocol implementation**.

If you want the benefit of this pledge, you effectively agree to a **mutual patent peace around the Core protocol itself**.

**Reciprocity verification (larger patent holders).**  
For entities with substantial patent portfolios (for example, more than 100 patents or applications in relevant fields), Interoperable Intelligence Inc. may request:

- (a) A written attestation that the entity complies with this reciprocity requirement and has conducted a good-faith review to identify any patents or applications that might be considered essential protocol claims; or  
- (b) A confirmation that the entity has conducted a good-faith internal review and believes it meets the reciprocity conditions in this §2.2.

Failure to provide requested verification within sixty (60) days of a written request may result in that entity being deemed ineligible for the benefit of this pledge.  

Small entities, individual developers, and entities with minimal patent portfolios (for example, fewer than 10 patents) are **presumed compliant** and need not provide attestation unless they have publicly asserted relevant patent claims.

#### 2.3 Defensive Suspension

This pledge is **suspended** with respect to any entity that:

- Asserts any patent claim against Interoperable Intelligence Inc. (or its affiliates), including but not limited to:
  - Claims related to routing, orchestration, conformance, or AI infrastructure; or
- Asserts any patent claim **related to protocol implementation, interoperability, or AI infrastructure** against:
  - Any user, distributor, or implementer of the Corpus Protocol for their conformant implementation; or
- Challenges the validity of any Interoperable Intelligence Inc. patent in a judicial, administrative, or arbitral proceeding.

During suspension, Interoperable Intelligence Inc. may fully enforce its patents, including Essential Protocol Claims, against that entity.

---

### 3. Scope, Limitations & Geographic Coverage

#### 3.1 Relationship to Apache-2.0

- All **software** released by Interoperable Intelligence Inc. under **Apache License 2.0** remains governed by that license and its patent grant (Apache §3).  
- This pledge:
  - **Supplements** the Apache-2.0 patent grant for protocol-level essentials when you implement from specifications, and  
  - Does **not** narrow or expand the Apache-2.0 grant that already applies to code you receive under that license.

#### 3.2 Patent Coverage

- This pledge covers all **Essential Protocol Claims** in patents and patent applications:
  - Owned or controlled by Interoperable Intelligence Inc. as of the **Effective Date**, and  
  - That remain in force in relevant jurisdictions.
- For patents acquired **after** the Effective Date:
  - Coverage of Essential Protocol Claims applies to **conformant protocol implementations occurring on or after the date of acquisition**, and  
  - We will use reasonable efforts to update relevant public materials (including this pledge) within thirty (30) days after acquisition;  
  - Any such administrative delay does **not** create any window during which we may assert Essential Protocol Claims against conformant implementations that would otherwise be covered by this pledge.

“Controlled” means we have the right to grant non-assertion covenants without third-party consent.

#### 3.3 Geographic Scope

- This pledge applies to Essential Protocol Claims in **all jurisdictions** where Interoperable Intelligence Inc. holds such patents.  
- Where local law requires specific formalities, parties may request jurisdiction-specific covenants for additional certainty.

#### 3.4 Termination & Grandfathering

- Interoperable Intelligence Inc. may terminate this pledge with **90 days’ written notice** by:
  - Public announcement at `https://corpus.io/patents`, and  
  - Email notification to any parties who have requested signed covenants.
- Termination does **not** affect implementations:
  - Created, distributed, or deployed **before** the termination effective date, provided they were in good standing under this pledge at that time.
- Entities in good standing at termination retain pledge benefits for their existing **conformant Core protocol implementations** indefinitely, subject to §2 (reciprocity and defensive suspension).

#### 3.5 Successors & Assigns

This pledge is **binding on Interoperable Intelligence Inc. and its successors and assigns** with respect to Essential Protocol Claims, except to the limited extent that a successor or assign:

- Lawfully terminates this pledge under §3.4 with respect to **future** implementations; and  
- Such termination does **not** retroactively affect rights previously accrued by entities that were in good standing prior to the termination effective date.

---

### 4. Definitions

For purposes of this pledge:

- **“Implementing”**  
  Creating software that adheres to the **mandatory** requirements of the **Core Corpus Protocol wire and schema specifications** as published in `docs/spec/core/` and their authorized successors.

- **“Essential Protocol Claims”**  
  Patent claims that are necessarily infringed by **any** conformant implementation of the **Core Corpus Protocol wire and schema specifications**, where **no technically feasible alternative** (as defined below) exists to implement the mandatory Core protocol requirements without infringing.

- **“Technically feasible alternative”**  
  An implementation approach that:
  - (a) Achieves the same mandatory Core Corpus Protocol conformance; and  
  - (b) Does not require a **prohibitive** degradation in latency, throughput, or resource consumption relative to a straightforward implementation of the specification by a skilled practitioner; and  
  - (c) Does not require abandoning the fundamental architectural patterns the Core specification itself mandates (for example, replacing a required request/response pattern with an incompatible publish/subscribe pattern); and  
  - (d) Can be implemented by a person of ordinary skill in the relevant art without **undue experimentation** or **prohibitive cost**; and  
  - (e) Either:
    - (i) Passes the official Corpus Protocol conformance test suite for the relevant tier (where applicable), **or**  
    - (ii) Can be shown through technical analysis to satisfy all mandatory Core protocol requirements for the relevant version.

- **“Defensive counterclaim”**  
  Asserting patent claims only as a **counterclaim** in response to being sued first, limited to:
  - Claims covering the Corpus Protocol specifications, or  
  - Claims related to protocol implementation and interoperability.

- **“Corpus Protocol specifications”**  
  The officially designated and versioned documents published by Interoperable Intelligence Inc. in `docs/spec/`, including a clearly identified **Core** subset in `docs/spec/core/`.

- **“Core Corpus Protocol” / “Core specifications”**  
  The subset of the Corpus Protocol specifications in `docs/spec/core/` that we designate as mandatory for Core wire and schema interoperability, as opposed to optional extensions, experimental drafts, or proprietary/commercial specifications that may appear elsewhere in `docs/spec/`.

- **“Related to protocol implementation”**  
  Patent claims covering protocol interfaces, message formats, wire-level behaviors, and interoperability patterns that are **directly required** for conformant Core Corpus Protocol implementation.

- **“Conformant implementation”**  
  An implementation that satisfies all **mandatory** Core protocol requirements in `docs/spec/core/`, without requiring optional extensions or proprietary behaviors.

- **“Non-Essential Claims”**  
  Patent claims that:
  - Cover optional features, proprietary routing/orchestration/conformance/cost systems, or other differentiated technologies; or  
  - Can be avoided by a technically feasible alternative while remaining conformant with the Core protocol.

---

### 5. Trademark & Certification

Our **trademarks remain protected**. This pledge does not grant any trademark rights.

Any implementation that passes our conformance tests (e.g., Platinum/Gold/Silver tiers) may state:

- ✅ “Implements Corpus Protocol v1 specification”  
- ✅ “Corpus Protocol compatible”

But may **not** claim, without explicit authorization:

- ⚠️ “Official Corpus implementation”  
- ⚠️ “Certified by Corpus” (requires a separate certification agreement)  
- ⚠️ Use of “Corpus” logos or wordmarks in product names or primary branding

Use of our trademarks is governed by our separate trademark and certification policies.

---

### 6. Why This Pledge (Narrowed)

We believe:

- **Protocols should be open and interoperable** at the wire/schema level;  
- **Innovation thrives** when developers can implement standards without constant patent fear; and  
- **The ecosystem grows fastest** when protocol interoperability is safe, and differentiation focuses on higher-level capabilities.

Our patent portfolio is primarily aimed at:

- Protecting the ecosystem from **patent trolls** and hostile actors; and  
- Defending our investments in **advanced routing, orchestration, conformance, and optimization technologies**.

This pledge ensures our patents serve their intended purpose—**enabling safe protocol interoperability**—while preserving our rights in **non-essential, differentiated technologies**, including router-level functionality and advanced orchestration systems.

---

### 7. Legal Effect

This pledge is a **unilateral non-assertion commitment** by **Interoperable Intelligence Inc.**, effective as of the **Effective Date** above.

- It **supplements** but does not replace:
  - The Apache License 2.0 for software; or  
  - Any commercial license agreements.
- It applies only to **Essential Protocol Claims** as defined in §1.2 and §4.  
- All other patent rights (including non-essential and implementation-level claims) are **expressly reserved**.

For heightened certainty, parties may request a **signed patent non-assertion covenant** from Interoperable Intelligence Inc. This written pledge and any such covenants should be read together with the applicable open-source and/or commercial licenses.

---

### 8. Example Scenarios

> ✅ **Protected — Core Protocol Only**  
> You build an open-source adapter for a new vector database that implements the `vector/v1` **Core wire specification**, without copying or reimplementing our proprietary router/conformance systems.

> ✅ **Protected — Internal Use**  
> Your company creates an internal edge service that speaks the Core Corpus Protocol (as defined in `docs/spec/core/`) to talk to upstream systems, but uses your own internal business logic beyond that. The **protocol interoperability layer** is protected.

> ✅ **Protected — Commercial Gateway**  
> A startup builds a commercial API gateway using the **Core Corpus Protocol** wire messages, while adding its own proprietary authentication, billing, or routing logic. The Core protocol implementation is protected; its proprietary logic is outside the pledge but also outside our Essential Protocol Claims.

> ✅ **Protected — Unrelated Patent Dispute**  
> Company X and Company Y both use Corpus. X sues Y over **unrelated payment processing patents**. This does not suspend the pledge for either party.

> ⚠️ **Not Protected — Non-Essential Technology**  
> You implement the Core Corpus Protocol **plus** an ensemble routing algorithm that infringes our **orchestration/router patents**. The Core protocol portion is covered; the orchestration technology is **not**.

> ⚠️ **Not Protected — Optional Extension**  
> You implement an **optional extension** not in `docs/spec/core/`, and it infringes our patent on a particular conformance engine or cost-normalization logic. This pledge does **not** cover that extension.

> ⚠️ **Not Protected — Pledge Suspended**  
> You sue Interoperable Intelligence Inc. for patent infringement, or  
> You sue another Corpus implementer claiming that their **protocol adapter** infringes your essential protocol patents. Our pledge is suspended as to you under §2.3.

> ⚠️ **Not Protected — Trademarks**  
> You use “Corpus” or related marks in your product names or marketing without permission. Trademark law applies separately and is not granted by this pledge.

> ⚠️ **Separate License Required**  
> You want to sell a “Certified Corpus Router” or similar product using our trademarks and certification marks. This requires a separate certification and/or commercial agreement.

---

### 9. Questions & Contact

For clarification or to request a signed non-assertion covenant, contact:

- Email: `legal@interoperable.ai` or `legal@corpus.io`  
- Web: `https://corpus.io/patents`

For questions about whether specific patent claims or implementations are covered by this pledge, please contact our legal team with details of your use case.

---

© 2026 Interoperable Intelligence Inc.  
This pledge may be referenced and linked freely.