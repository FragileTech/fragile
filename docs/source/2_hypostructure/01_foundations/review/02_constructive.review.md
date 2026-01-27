# Mathematical Review: docs/source/2_hypostructure/01_foundations/02_constructive.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/01_foundations/02_constructive.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-sieve-functor and rem-sieve-dual-role in this file
  - thm-expansion-adjunction in this file
  - def-rigor-classification and def-bridge-verification in this file
  - def-thin-objects in this file

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: Adjunction notation mismatch; literature-based analytic theorems lack Rigor Class L/bridge framing.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | rem-sieve-adjoint | Moderate | Definition mismatch | Uses F_sieve (classification) as left adjoint to U, contradicting rem-sieve-dual-role which restricts adjunction to the expansion functor. |
| E-002 | thm-rcd-dissipation-link, thm-log-sobolev-concentration, thm-cheeger-dissipation | Moderate | External dependency | Analytic results are presented as internal theorems without Rigor Class L / Bridge Verification. |

## Detailed findings

### [E-001] Adjunction uses the wrong functor symbol
- Location: rem-sieve-adjoint
- Severity: Moderate
- Type: Definition mismatch
- Claim (paraphrase): The Structural Sieve computes the left adjoint F_sieve ⊣ U.
- Why this is an error in the framework: def-sieve-functor defines F_sieve as a classification map to Result (not a functor into Hypo_T). rem-sieve-dual-role explicitly separates classification (F_sieve^{class}) from the categorical expansion (F). The adjunction applies to the expansion functor, not the classification output.
- Impact on downstream results: This conflates the diagnostic label map with the categorical construction, blurring which object has the universal property and potentially invalidating proofs that rely on the adjunction.
- Fix guidance (step-by-step):
  1. Replace F_sieve with the expansion functor symbol (\mathcal{F}) in rem-sieve-adjoint.
  2. Add a sentence clarifying that F_sieve^{class} is not part of the adjunction.
  3. Cross-check later references to “F_sieve is left adjoint” and align them to rem-sieve-dual-role.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use the adjunction proof in thm-expansion-adjunction and rem-sieve-dual-role to identify the correct functor.
- Validation plan: Ensure all adjunction references use \mathcal{F} and not F_sieve^{class}.

### [E-002] Analytic metric-measure theorems need Rigor Class L/bridge framing
- Location: thm-rcd-dissipation-link, thm-log-sobolev-concentration, thm-cheeger-dissipation
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): RCD, LSI, and Cheeger-energy results are stated as internal theorems with literature citations.
- Why this is an error in the framework: def-rigor-classification requires literature-anchored results to be labeled Rigor Class L with explicit Bridge Verification. These theorems import external analysis without stating the bridge hypotheses in certificate form.
- Impact on downstream results: It is unclear which Sieve permits discharge the required hypotheses (e.g., RCD(K,N) or LSI constants), and whether the conclusions are conditional or absolute within the framework.
- Fix guidance (step-by-step):
  1. Label these theorems as Rigor Class L and cite the Bridge Verification protocol.
  2. State the certificate prerequisites (e.g., RCD(K,N) permit, infinitesimal Hilbertianity permit) explicitly.
  3. If the results are only used as optional upgrades, mark them as conditional permits rather than unconditional theorems.
- Required new assumptions/permits (if any): Bridge Verification clauses tying RCD/LSI hypotheses to gate certificates.
- Framework-first proof sketch for the fix: Translate RCD/LSI hypotheses into interface permits on thin objects, then import the external conclusions as upgrade certificates.
- Validation plan: Ensure any later use of these results references the corresponding permits or upgrades rather than assuming them unconditionally.

## Scope restrictions and clarifications
- None beyond the issues above.

## Proposed edits (optional)
- Replace F_sieve with \mathcal{F} in rem-sieve-adjoint and add a short clarification sentence.
- Add Rigor Class L labels and Bridge Verification references for the metric-measure theorems.

## Open questions
- Should the metric-measure theorems be treated as optional upgrades (permits) rather than unconditional theorems?
