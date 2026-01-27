# Mathematical Review: docs/source/2_hypostructure/03_sieve/02_kernel.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/03_sieve/02_kernel.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-progress-measures (Progress measures) in docs/source/2_hypostructure/05_interfaces/03_contracts.md
  - mt-fact-surgery (Surgery Schema Factory) in docs/source/2_hypostructure/07_factories/01_metatheorems.md
  - def-typed-no-certificates (Binary Certificate Logic) in docs/source/2_hypostructure/03_sieve/01_structural.md
  - def-cert-finite and def-closure (Promotion closure) in this file
  - def-node-evaluation and def-edge-validity in this file

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 4
- Minor: 0
- Notes: 0
- Primary themes: Global termination proof needs explicit linkage to surgery progress measure; closure termination assumptions vs def-cert-finite; context multiplicity vs set semantics; NO-certificate typing/notation alignment.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | thm-finite-runs proof (Kernel Theorems) | Moderate | Proof gap / Citation error | Finite-runs proof does not cite the global progress measure defined later in mt-fact-surgery. |
| E-002 | thm-closure-termination statement + proof Steps 5–7 | Moderate | Scope restriction / Definition mismatch | Theorem claims full closure under def-cert-finite, but def-cert-finite allows depth budgets that only give partial closure. |
| E-003 | def-context, def-node-evaluation, thm-closure-termination Step 1 | Moderate | Definition mismatch | Context is defined as a multiset, but closure uses set/powerset lattice semantics. |
| E-004 | def-determinism, def-gate-permits, rem-dichotomy | Moderate | Notation conflict / Definition mismatch | NO certificates use K^?, K^- without explicit coproduct link to K^{inc}, and dichotomy semantics don’t restrict benign NO to witness-based cases. |

## Detailed findings

### [E-001] Finite-runs proof needs explicit global progress linkage
- Location: thm-finite-runs proof (Kernel Theorems)
- Severity: Moderate
- Type: Proof gap / Citation error
- Claim (paraphrase): Finite surgery types + per-surgery progress measures imply finitely many surgeries and epochs.
- Why this is an error in the framework: The kernel proof does not exhibit a single well-founded measure that decreases across all surgeries. The global lexicographic progress measure is defined later in mt-fact-surgery (Step 4), but the kernel theorem does not reference it or state its assumptions.
- Impact on downstream results: The kernel’s termination guarantee is under-justified unless the factory theorem (or a global progress lemma) is explicitly imported.
- Fix guidance (step-by-step):
  1. Add a citation to mt-fact-surgery Step 4 (global progress measure) in the proof of thm-finite-runs.
  2. Alternatively, add a short lemma in the kernel stating the existence of a global progress measure under the surgery schema contract.
  3. Make the dependency explicit in the hypotheses of thm-finite-runs.
- Required new assumptions/permits (if any): None if mt-fact-surgery is invoked; otherwise a “global progress composition” lemma.
- Framework-first proof sketch for the fix: Use the lexicographic measure (N_max - N_S, Phi_residual) from mt-fact-surgery; each surgery strictly decreases it, so no infinite surgery sequence exists.
- Validation plan: Verify that the surgery schema contract is referenced in the kernel and that each surgery node provides the required progress certificate.

### [E-002] Closure termination statement does not match certificate finiteness definition
- Location: thm-closure-termination statement and proof Steps 5–7
- Severity: Moderate
- Type: Scope restriction / Definition mismatch (secondary: Miswording)
- Claim (paraphrase): Under def-cert-finite, promotion closure Cl(Gamma) is computable in finite time and order-independent.
- Why this is an error in the framework: def-cert-finite allows either (i) finite certificate language or (ii) depth budget. The proof in Step 5 assumes |K(T)| is finite; Step 7 introduces a depth-bounded partial closure but the theorem statement does not reflect this conditionality.
- Impact on downstream results: Later results may assume full closure even when only a depth-bounded computation is available.
- Fix guidance (step-by-step):
  1. Split the theorem into a finite-language case (full closure + order independence) and a depth-budget case (partial closure + K_Promo^{inc}).
  2. Make the dependence on bounded description length explicit in the statement.
- Required new assumptions/permits (if any): Explicit finite-alphabet/bounded-length assumption when claiming |K(T)| < infinity.
- Framework-first proof sketch for the fix: Use Knaster-Tarski/Kleene for the finite-language case; define Gamma_{D_max} and a partial-closure certificate for the budgeted case.
- Validation plan: Cross-check references in foundations/notation to ensure they mirror the two-case formulation.

### [E-003] Context is a multiset but closure uses set semantics
- Location: def-context, def-node-evaluation, thm-closure-termination Step 1
- Severity: Moderate
- Type: Definition mismatch
- Claim (paraphrase): Gamma is a multiset, but closure treats Gamma as a set and uses a powerset lattice.
- Why this is an error in the framework: Multisets track multiplicity, while the closure proof collapses duplicates via set union and subset order. If multiplicity is intended to matter, the current lattice model is invalid; if it does not matter, Gamma should be defined as a set.
- Impact on downstream results: Closure and monotonicity arguments are formally unsupported under multiset semantics.
- Fix guidance (step-by-step):
  1. Decide whether certificate multiplicity is semantically relevant.
  2. If not, redefine Gamma as a finite set and keep set-based closure.
  3. If yes, redefine the lattice as N-valued multisets and re-prove monotonicity for promotion rules.
- Required new assumptions/permits (if any): A multiset order definition if multiplicity is retained.
- Framework-first proof sketch for the fix: If sets, show duplicate certificates are idempotent; if multisets, define Gamma: K -> N and use pointwise order to show F is monotone.
- Validation plan: Check obligation ledger and goal dependency cone for type consistency under the chosen interpretation.

### [E-004] NO-certificate typing and dichotomy semantics are underspecified
- Location: def-determinism, def-gate-permits, rem-dichotomy
- Severity: Moderate
- Type: Notation conflict / Definition mismatch
- Claim (paraphrase): UNKNOWN is treated as NO with certificate K^?, while gate permits define a single NO certificate K^- that may mean failure or lack of proof; dichotomy classifiers treat NO as a benign outcome.
- Why this is an error in the framework: Binary Certificate Logic defines K^- as the coproduct K^{wit} + K^{inc}. The current section uses K^? without identifying it with K^{inc} and does not restrict benign “NO” outcomes to the witness branch.
- Impact on downstream results: Inc-upgrades and routing semantics may be misapplied if inconclusive NOs are treated as benign classifications.
- Fix guidance (step-by-step):
  1. Define K_i^- := K_i^{wit} + K_i^{inc} in def-gate-permits and reference def-typed-no-certificates.
  2. Replace K^? with K^{inc} (or explicitly state K^? = K^{inc}) in def-determinism.
  3. Clarify that dichotomy classifier benign outcomes apply only to K^{wit}; K^{inc} follows the inconclusive routing convention.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use coproduct case analysis to separate witness vs inconclusive routing and promotion behavior.
- Validation plan: Cross-check gate-node definitions (e.g., def-node-compact, def-node-oscillate) for consistent NO typing.

## Scope restrictions and clarifications
- None beyond the issues above.

## Proposed edits (optional)
- Cite mt-fact-surgery in thm-finite-runs to supply the global progress measure.
- Split thm-closure-termination into finite-language and depth-budget cases.
- Align NO-certificate notation with def-typed-no-certificates.

## Open questions
- Should context multiplicity carry any semantic meaning, or is Gamma intended to be a set?
