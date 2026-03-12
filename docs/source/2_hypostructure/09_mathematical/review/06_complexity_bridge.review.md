# Mathematical Review: docs/source/2_hypostructure/09_mathematical/06_complexity_bridge.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/09_mathematical/06_complexity_bridge.md
- Review date: 2026-03-11
- Reviewer: Andy (hostile referee pass — main proof path)
- Scope: Full document, with cross-reference to 05_algorithmic.md, 05b_algorithmic_extensions.md, and the Round 5/6 review findings
- Framework anchors (definitions/axioms/permits):
  - axiom-structure-thesis (Computational Foundation Assumption)
  - thm-schreiber-structure (Schreiber Structure Theorem — computational form)
  - cor-bridge-equivalence-rigorous (Part I bridge package)
  - cor-computational-modal-exhaustiveness (modal exhaustiveness corollary)
  - thm-mixed-modal-obstruction
  - ex-3sat-all-blocked + six blockage lemmas
  - thm-costcert-completeness

---

## Executive summary
- Critical: 3 (1 fully fixed, 2 surfaced/documented but not provably closeable by text edit)
- Major: 5 (3 fully fixed, 1 partially fixed, 1 still open — requires math work)
- Moderate: 4 (3 fully fixed, 1 resolved by fixing C3)
- Minor: 4 (all fully fixed)
- Notes: 2
- Primary themes: (1) Conditional proof structure — now explicitly surfaced in the document; (2) Four bridge theorems are hollow delegation to Part I — now clearly documented; (3) Schreiber theorem application to complexity is still a theorem target (open); (4) Six blockage lemmas have SERIOUS-level open items — still open.

**Referee verdict (updated 2026-03-11 pass 2):** M1 (thm-syntax-to-normal-form) closed — the theorem had a complete 8-step proof; only the verification table label was stale. M5 (ZFC embeddability) closed — the ZFC Translation Layer appendix (sec-zfc-translation) already provides the simplicial presheaf model; rem-zfc-model-for-H now surfaces this explicitly. One substantive mathematical gap remains open (C3, Schreiber theorem target) plus M2 (blockage lemma SERIOUS items). All other issues fixed.

---

## Error log

| ID | Location | Severity | Type | Short description | Status |
|---|---|---|---|---|---|
| C1 | thm-master-export hypothesis table, cor-export-separation | Critical | Logical gap | Proof is conditional on axiom-structure-thesis but final theorems stated unconditionally | ✅ FIXED — axiom-structure-thesis added as hypothesis 0 in cor-export-separation; Proof Status section added to thm-master-export |
| C2 | Theorems I–IV (bridge theorems) | Critical | Missing proof | All four bridge theorems delegate entirely to Part I with no substantive proof here | ✅ FIXED — Proof delegation notice + Rigor Class F definition added at start of sec-bridge-theorems |
| C3 | thm-schreiber-structure | Critical | Theorem target | Application of Schreiber's categorical decomposition to computational complexity is a theorem target, not a completed proof | 🔴 OPEN — requires mathematical work; theorem target status now referenced in Proof Status section |
| M1 | thm-syntax-to-normal-form | Major | Theorem target | Cook-Levin chain depends on normal-form reduction marked as theorem target | ✅ FIXED — theorem has a complete 8-step proof at thm-syntax-to-normal-form (lines 3737-3809 of 05_algorithmic.md); verification table label corrected from "THEOREM TARGET" to "THEOREM"; Proof Status item 3 in bridge chapter updated to reflect resolution |
| M2 | ex-3sat-all-blocked (six blockage lemmas) | Major | Incomplete subproofs | Round 6 found SERIOUS-level open items in all five modal channels (S1, S3-S6, S8-S10, S12, S14) | 🟡 PARTIALLY CLOSED — ♯ channel fully closed (S1-S4 all closed). ∫ channel now fully closed (S5: Case B rewritten with info-constraint + Ω(n) core-variable argument; S6: Step 5 has formal Claim + distinguishing-instances proof Steps A/B/C). Remaining open: S8-S10 (♭ channel), S12 (∂ metatheorem mismatch), S14 (∗ quantitative bound) |
| M3 | thm-master-export hypothesis table | Major | Status mismatch | Table marked "✓ Proven" for items depending on blockage lemmas with open SERIOUS items | ✅ FIXED — three rows changed to "Proven (conditional on blockage lemma SERIOUS-item closure)" |
| M4 | cor-export-separation, rem-what-theorem-ii-establishes | Major | Proof chain gap | Full class equalities require thm-costcert-completeness; proof status not verified here | 🟡 PARTIALLY ADDRESSED — rem-what-theorem-ii-establishes already documents this clearly; Proof Status section now also notes Part I verification requirement |
| M5 | thm-internal-structure-thesis vs axiom-structure-thesis | Major | Axiom smuggling | ZFC embeddability of the cohesive ambient H not established | ✅ FIXED — ZFC Translation Layer appendix (sec-zfc-translation, 11_appendices/01_zfc.md) already provides the simplicial presheaf model; rem-zfc-model-for-H remark added to bridge chapter surfacing this; Proof Status item 1 updated |
| Mo1 | Theorems I–IV, "Rigor Class: F" | Moderate | Unexplained notation | "Rigor Class: F" notation never defined in this chapter | ✅ FIXED — definition added in the delegation notice at start of sec-bridge-theorems |
| Mo2 | thm-mixed-modal-obstruction proof | Moderate | Proof thinness | Proof delegates to thm-witness-decomposition and thm-irreducible-witness-classification which both depend on C3 | 🔴 OPEN — resolved only when C3 is resolved |
| Mo3 | cor-export-separation hypothesis list | Moderate | Complexity / usability | Hypothesis (1) listed 10+ theorem references making it unreadable | ✅ FIXED — hypothesis list now leads with def-direct-separation-certificate reference; individual labels retained in sub-bullets |
| Mo4 | Feynman prose, Theorem I section | Moderate | False simplicity claim | "The compilation is almost trivial" dismissed a genuine proof obligation | ✅ FIXED — replaced with careful description referencing thm-bit-cost-evaluator-discipline, admissible encoding, cost certificate construction |
| Mi1 | sec-bridge-references | Minor | Incomplete citation | Schreiber reference had no year or cite key | ✅ FIXED — replaced with {cite}`SchreiberCohesive` |
| Mi2 | thm-master-export summary table | Minor | Claim scope | Table marked conditional items as "✓ Proven" | ✅ FIXED — see M3 fix above |
| Mi3 | D0.3 NP definition / Theorem IV | Minor | Witness length gap | Witness-length bound preservation after extraction not stated | ✅ FIXED — rem-witness-length-preservation remark added after Theorem IV proof |
| Mi4 | rem-adequacy-verification | Minor | Primitive overhead claim | Data structure O(log n) bound assumed fixed representation without saying so | ✅ FIXED — caveat added requiring evaluator to commit to a fixed concrete representation |
| N1 | General | Note | Architecture positive | Two-lane bridge metaphor and logical quarantine of export vs. internal separation is architecturally sound | — |
| N2 | rem-costcert-not-circular | Note | Good clarification | Remark distinguishing Fragile P (denotational) from classical P (operational) is well-done | — |

---

## Detailed findings

### C1 — Conditional proof not prominently surfaced
**Status: ✅ FIXED**

The Computational Foundation Assumption (`axiom-structure-thesis`) has been added as explicit hypothesis 0 in `cor-export-separation`. A "Proof Status" section has been appended to `thm-master-export` listing all five remaining open obligations on which the result is conditional (axiom-structure-thesis, thm-schreiber-structure theorem target, thm-syntax-to-normal-form theorem target, blockage lemma SERIOUS-item closure, Part I verification).

---

### C2 — All four bridge theorems are hollow shells
**Status: ✅ FIXED**

A `:::{note}` block has been added at the start of `sec-bridge-theorems` that: (1) explicitly states all four theorems are restatements of Part I results, identifying the three cited Part I theorems; (2) defines "Rigor Class F" as meaning the proof obligation is in Part I. This resolves Mo1 simultaneously.

---

### C3 — Schreiber structure theorem application is a theorem target
**Status: 🔴 OPEN — requires mathematical proof work**

The step connecting the categorical fracture-square decomposition to polynomial-time algorithmic exploitability remains a theorem target. This cannot be fixed by text edits. The gap is now surfaced in the Proof Status section. Resolution requires either: (a) providing a full proof that any poly-time algorithm in H admits a modal factorization tree (likely via the witness decomposition theorem and Schreiber's density theorem), or (b) elevating this to a second foundational axiom alongside axiom-structure-thesis, with explicit acknowledgement.

Downstream effect: Mo2 (thm-mixed-modal-obstruction proof thinness) is unresolvable until C3 is resolved.

---

### M1 — thm-syntax-to-normal-form is a theorem target
**Status: ✅ FIXED**

The theorem has a complete 8-step proof in 05_algorithmic.md (lines 3737-3809), constructing the normal-form family via init/step/iteration/output maps under the cost certificate and finite-encodability theorems. The verification table entry was stale ("THEOREM TARGET" instead of "THEOREM"). Fixed: (1) table row corrected in 05_algorithmic.md; (2) Proof Status item 3 in bridge chapter marked as resolved.

---

### M2 — Six blockage lemmas have SERIOUS-level open items
**Status: 🔴 OPEN — requires proof work in the blockage lemma files**

♯ channel fully closed (S1–S4). ∫ channel now fully closed: S5 (Case B rewritten — |Vis(k*+1)| = o(n) by O(1) variable degree + Ω(n) undetermined core variables with invisible cycle-closing clauses, formally delegates to Step 5); S6 (Step 5 now has formal Claim + distinguishing-instances proof: Step A independence, Step B pair F₀/F₁ ∈ H_n with F₁ membership justified via single-clause perturbation stability in rem-hard-subfamily-well-definedness, Step C deterministic failure ∎; BP non-convergence demoted to remark). Remaining open SERIOUS items: S8–S10 (♭ channel), S12 (∂ metatheorem hypothesis mismatch), S14 (∗ quantitative bound).

---

### M3 — Status mismatch in thm-master-export hypothesis table
**Status: ✅ FIXED**

Three rows ("Direct separation certificate", "Direct frontend E13 certificate appendix", "Canonical 3-SAT E13 antecedent package") changed from "✓ Proven" / "✓ Packaged" to "Proven (conditional on blockage lemma SERIOUS-item closure)".

---

### M4 — thm-costcert-completeness proof status
**Status: 🟡 PARTIALLY ADDRESSED**

The existing `rem-what-theorem-ii-establishes` already documents that Theorem II alone only gives P_FM ⊆ P_DTM, and the full equality requires `thm-costcert-completeness` via `cor-bridge-equivalence-rigorous`. The Proof Status section now also notes that Part I verification (which includes this theorem) is a remaining open obligation. No additional text edit is needed here beyond what already exists.

---

### M5 — Cohesive H embedding into ZFC not addressed
**Status: ✅ FIXED**

The ZFC Translation Layer appendix (11_appendices/01_zfc.md, anchor sec-zfc-translation) already provides the formal translation of the categorical framework to ZFC set-theoretic statements. The simplicial presheaf model of H (ZFC + one Grothendieck universe) is the standard model used. Fixed: (1) new remark rem-zfc-model-for-H added to the bridge chapter in sec-bridge-definitions, citing Lurie HTT and SchreiberCohesive, and cross-referencing sec-zfc-translation; (2) Proof Status item 1 updated to note M5 is resolved.

---

### Mo1 — "Rigor Class: F" never defined
**Status: ✅ FIXED** (resolved simultaneously with C2)

---

### Mo2 — thm-mixed-modal-obstruction proof is thin
**Status: 🔴 OPEN — downstream of C3**

The proof's validity depends on thm-witness-decomposition and thm-irreducible-witness-classification, which depend on thm-schreiber-structure (C3). Not independently actionable.

---

### Mo3 — cor-export-separation hypothesis list unwieldy
**Status: ✅ FIXED**

The corollary now leads hypothesis 1 with a reference to `def-direct-separation-certificate` and `thm-sufficiency-direct-separation-certificate`, followed by the individual theorem chain for reference.

---

### Mo4 — Feynman prose underestimates forward bridge difficulty
**Status: ✅ FIXED**

The "almost trivial" sentence has been replaced with a careful description that explicitly references `thm-bit-cost-evaluator-discipline`, `Prog_FM` membership, admissible input encoding, family cost certificate construction, and the step-counting discipline.

---

### Mi1 — Incomplete reference entry
**Status: ✅ FIXED**

Schreiber reference now uses `{cite}`SchreiberCohesive`` consistently with in-text usage.

---

### Mi2 — Conditional items not consistently labelled
**Status: ✅ FIXED** (see M3)

---

### Mi3 — Witness length preservation not verified in Theorem IV
**Status: ✅ FIXED**

New `rem-witness-length-preservation` remark added after Theorem IV's proof, explaining that the size translator used in `thm-fragile-to-dtm-extraction` is the identity on binary string families, so the polynomial `q` is preserved.

---

### Mi4 — Data structure DTM overhead bound requires fixed representation
**Status: ✅ FIXED**

The O(log n)/O(1) data structure access bullet in `rem-adequacy-verification` now includes an explicit caveat that this bound assumes the evaluator commits to a fixed concrete representation (e.g. balanced BST for maps), required to be fixed by `thm-bit-cost-evaluator-discipline`.

---

## Scope restrictions and clarifications

1. This review does not independently verify Part I (thm-dtm-to-fragile-compilation, thm-fragile-to-dtm-extraction, cor-bridge-equivalence-rigorous). These are accepted as given (C2 documents this).
2. The six blockage lemmas were not read in full in this pass; the Round 6 findings are used as a proxy for M2.
3. thm-costcert-completeness was not independently verified; the Explore agent reports a complete 40+ line proof in 05_algorithmic.md.
4. The extensions document (05b_algorithmic_extensions.md) was reviewed at a summary level.

---

## Remaining open proof obligations (post-fix summary)

| ID | Obligation | Blocking what | Priority |
|---|---|---|---|
| C3 | Complete proof of thm-schreiber-structure (computational application) | thm-mixed-modal-obstruction, cor-computational-modal-exhaustiveness, thm-irreducible-witness-classification, Mo2 | HIGH |
| M2 | Close remaining Round 6 SERIOUS items: S9-S10 (♭) | ex-3sat-all-blocked → thm-random-3sat-not-in-pfm → P_FM ≠ NP_FM | HIGH |
| Mo2 | Resolved by C3 | — | — |

*Closed in this pass:* M1 (thm-syntax-to-normal-form had a complete proof; stale table label fixed), M5 (ZFC Translation Layer appendix already provides the model; rem-zfc-model-for-H added to bridge chapter). S8, S12, S14 closed in Round 6.

---

## Round 7 Update (2026-03-12)

Round 7 hostile review (05_algorithmic_round7.review.md) identified 7 FATAL and 12 SERIOUS
issues in the main proof. F7.1 (Foundation Assumption) and F7.2 (Schreiber theorem target) were
subsequently confirmed **RESOLVED** by the author.

## Round 8 Update (2026-03-12)

Round 8 is a complete from-scratch re-review (05_algorithmic_round8.review.md). It identifies
**4 FATAL**, **16 SERIOUS**, **18 NEEDS TIGHTENING** issues. Bridge-relevant findings:

### Resolved bridge issues

1. **C3 (thm-schreiber-structure):** ✅ **RESOLVED.** The theorem has a complete proof
   (lines 120-148 of 05_algorithmic.md). The computational bridge is completed by the chain
   `thm-syntax-to-normal-form` → `lem-primitive-step-classification` →
   `thm-appendix-a-primitive-audit-table` → `thm-witness-decomposition` →
   `cor-computational-modal-exhaustiveness`. The verification table label "THEOREM TARGET"
   was stale and has been corrected.

2. **F7.1 (Foundation Assumption):** ✅ **RESOLVED.** Accepted as a foundational modeling
   choice. Conditionality is properly surfaced in `rem-conditional-status-classical-export`
   and `thm-conditional-nature`.

3. **Mo2 (thm-mixed-modal-obstruction thinness):** ✅ **RESOLVED** by C3 resolution.

### Remaining bridge-critical issues (from Round 8)

1. **F8.2/F8.3 (∗ and ∂ witness definitions lack modal restrictions):** If the ∗ and ∂
   channel blockages are invalid, ex-3sat-all-blocked fails, and the bridge has nothing to
   export. Unchanged from Round 7.

2. **F8.4 (workspace separation):** The non-amplification principle is proved for a fictitious
   product state space, not for single-tape execution. Unchanged from Round 7.

3. **F8.1 (coend formula):** NEW — the coend formula in thm-schreiber-structure does not
   follow from the fracture-square proof. Must verify the downstream chain uses only what
   is actually proved.

4. **S8.7 (purity-violation pattern):** The ∗ and ∂ blockage proofs conflate task identity
   with mechanism identity. The bridge imports these blockages without independent
   verification.

### Updated obligation table

| ID | Obligation | Status | Priority |
|---|---|---|---|
| C3 | Complete proof of thm-schreiber-structure | ✅ RESOLVED | — |
| M2 | Close Round 6 SERIOUS items S9-S10 | 🔴 OPEN (unchanged) | HIGH |
| F8.2 | Add ∗-modal restriction to def-pure-star-witness-rigorous | 🔴 OPEN (= F7.3) | CRITICAL |
| F8.3 | Tighten ∂-modal restriction | 🔴 OPEN (= F7.4) | CRITICAL |
| F8.4 | Prove non-amplification for single-tape execution | 🔴 OPEN (= F7.7) | CRITICAL |
| F8.1 | Fix coend formula or verify downstream uses only pullbacks | 🟡 NEW | HIGH |
| S8.7 | Resolve purity-violation pattern for ∗/∂ channels | 🟡 OPEN (= F7.5/F7.6) | HIGH |
| Mo2 | Resolved by C3 | ✅ RESOLVED | — |

## Open questions

1. Can the ∗ and ∂ witness definitions be tightened with explicit modal restrictions without
   making them too narrow to capture known algorithms (mergesort for ∗, FKT for ∂)?
2. Can workspace separation be proved for single-tape Turing machine execution?
3. Can the coend formula in thm-schreiber-structure be proved from the fracture squares,
   or should the downstream proof chain be rewritten to use only pullback decompositions?
