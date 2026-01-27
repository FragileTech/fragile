# Mathematical Review: docs/source/2_hypostructure/01_foundations/02_constructive.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/01_foundations/02_constructive.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-sieve-functor, rem-sieve-dual-role, and rem-sieve-adjoint in this file
  - thm-expansion-adjunction in this file
  - def-rigor-classification and def-bridge-verification in this file
  - def-thin-objects in this file

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: Adjunction/type mismatch between classification vs. expansion; literature-based analytic theorems lack Rigor Class L/bridge verification framing.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | def-sieve-functor, rem-sieve-dual-role, rem-sieve-adjoint | Moderate | Definition mismatch | Uses F_{\text{Sieve}} (classification) as left adjoint to U and as a hypostructure constructor, contradicting the explicit separation between classification and expansion. |
| E-002 | thm-rcd-dissipation-link, thm-log-sobolev-concentration, thm-cheeger-dissipation | Moderate | External dependency | Analytic results are presented as internal theorems without Rigor Class L labeling or Bridge Verification prerequisites despite the new protocol. |

## Detailed findings

### [E-001] Adjunction conflates classification with expansion (and flips the adjunction typing)
- Location: def-sieve-functor, rem-sieve-dual-role, rem-sieve-adjoint
- Severity: Moderate
- Type: Definition mismatch
- Claim (paraphrase): The Structural Sieve computes the left adjoint $F_{\text{Sieve}} \dashv U : \mathbf{Hypo}_T \rightleftarrows \mathbf{Thin}_T$, and its unit/counit live on $F_{\text{Sieve}}$.
- Why this is an error in the framework: def-sieve-functor defines $F_{\text{Sieve}}: \mathbf{Thin} \to \mathbf{Result}$ as a classification map, while rem-sieve-dual-role reserves the adjunction for the categorical expansion $\mathcal{F}$. Using $F_{\text{Sieve}}$ for the adjunction gives it the wrong codomain (Result instead of Hypo$_T$) and the displayed adjunction flips the domain/codomain order. This violates the explicit separation between classification and expansion.
- Impact on downstream results: It blurs which functor satisfies the universal property, risks using unit/counit on the wrong codomain, and introduces type errors in any formalization or certificate payload that expects $\mathcal{F}: \mathbf{Thin}_T \to \mathbf{Hypo}_T$.
- Fix guidance (step-by-step):
  1. In rem-sieve-adjoint, replace $F_{\text{Sieve}}$ with $\mathcal{F}$ and update the adjunction display to $\mathcal{F} \dashv U : \mathbf{Thin}_T \rightleftarrows \mathbf{Hypo}_T$.
  2. Ensure the unit/counit bullets reference $\mathcal{F}$ (or introduce an explicit $F_{\text{Sieve}}^{\text{exp}}$ name if you want to keep $F_{\text{Sieve}}$ reserved for classification).
  3. If $F_{\text{Sieve}}$ is meant to be the classification map, keep def-sieve-functor as-is and add a one-line reminder that only $\mathcal{F}$ participates in adjunctions.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use rem-sieve-dual-role to distinguish the two operations and thm-expansion-adjunction to identify the left adjoint as $\mathcal{F}$.
- Validation plan: Check that all adjunction references use $\mathcal{F}$ and that $F_{\text{Sieve}}$ only maps into Result labels.

### [E-002] Metric-measure analytic theorems lack Rigor Class L / Bridge Verification framing
- Location: thm-rcd-dissipation-link, thm-log-sobolev-concentration, thm-cheeger-dissipation
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): RCD, LSI, and Cheeger-energy results are stated as internal theorems with literature citations.
- Why this is an error in the framework: def-rigor-classification and def-bridge-verification now make Bridge Verification mandatory for literature-anchored results. These theorems import external analysis (RCD/EVI, LSI, $\Gamma_2$) without stating the bridge hypotheses, certificates, or the embedding $\iota$ needed to justify the import.
- Impact on downstream results: It is unclear which Sieve permits discharge the analytic hypotheses (e.g., RCD(K,N), infinitesimal Hilbertianity, LSI constants), and whether the conclusions are conditional upgrades or unconditional guarantees inside the framework.
- Fix guidance (step-by-step):
  1. Add explicit “Rigor Class L” headers to these theorems and cite {prf:ref}`def-bridge-verification`.
  2. State the Bridge Verification components: hypothesis translation (which Thin/Hypo certificates imply RCD/LSI/Γ2 assumptions), the domain embedding $\iota$, and the conclusion import to a named certificate.
  3. If these are optional upgrades, reframe them as conditional permits (e.g., “If $K_{\mathrm{RCD}}^+$ then…”), not unconditional theorems.
- Required new assumptions/permits (if any): Bridge Verification clauses tying RCD/LSI hypotheses to explicit certificates.
- Framework-first proof sketch for the fix: Express RCD/LSI hypotheses as certified predicates on Thin Objects, then import the literature conclusion as a bridge permit that yields a named certificate.
- Validation plan: Ensure all later uses of these theorems reference the corresponding bridge permits or conditional hypotheses.

## Scope restrictions and clarifications
- Until Bridge Verification is specified, treat the metric-measure theorems as conditional upgrades rather than unconditional guarantees.

## Proposed edits (optional)
- Replace $F_{\text{Sieve}}$ with $\mathcal{F}$ in rem-sieve-adjoint, fix the adjunction arrow direction, and align unit/counit bullets to $\mathcal{F}$.
- Add Rigor Class L labels and Bridge Verification components for thm-rcd-dissipation-link, thm-log-sobolev-concentration, and thm-cheeger-dissipation.

## Open questions
- Should the metric-measure results live in a dedicated “Bridge Permits” subsection rather than as unconditional theorems?
- Do you want $F_{\text{Sieve}}$ reserved strictly for classification, or should it be renamed and used for expansion as well?
