# Mathematical Review: docs/source/2_hypostructure/01_foundations/01_categorical.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/01_foundations/01_categorical.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-structural-flow-datum (structural flow datum) in this file
  - def-sieve-functor (Sieve outputs) in docs/source/2_hypostructure/01_foundations/02_constructive.md
  - def-rigor-classification and def-bridge-verification in docs/source/2_hypostructure/01_foundations/02_constructive.md
  - def-typed-no-certificates (Binary Certificate Logic) in docs/source/2_hypostructure/03_sieve/01_structural.md
  - def-progress-measures in docs/source/2_hypostructure/05_interfaces/03_contracts.md
  - def-germ-smallness (Germ Smallness Permit) in docs/source/2_hypostructure/06_modules/01_singularity.md
  - def-node-geom (ambient dimension d, codimension bounds) in docs/source/2_hypostructure/04_nodes/01_gate_nodes.md
  - Permit WP_{s_c} and ProfDec_{s_c,G} (critical exponent s_c) in docs/source/2_hypostructure/05_interfaces/01_gate_evaluator.md
  - def-kolmogorov-complexity, def-chaitin-omega, def-computational-depth, def-algorithmic-phases in docs/source/2_hypostructure/09_mathematical/04_taxonomy.md
  - def-heyting-boolean-distinction in docs/source/2_hypostructure/11_appendices/01_zfc.md

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 0
- Minor: 0
- Notes: 0
- Primary themes: All identified issues resolved (dual provenance, certificate-conditional logic, critical-index anchoring).

## Upgrade plan (pre-implementation)
- Add explicit **Rigor Class L** labels and **Bridge Verification** triples for the literature-anchored theorems in `01_categorical.md` (KRNL-Consistency, KRNL-Trichotomy, KRNL-Equivariance, Halting/AIT Sieve Thermodynamics), written entirely in terms of thin-interface certificates and existing definitions.
- For KRNL-Exclusion Step 6, rewrite the contrapositive step to be certificate-conditional: only a Lock block certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ yields Rep_K; otherwise emit $K^{\mathrm{inc}}$ and route to reconstruction/Horizon.
- Anchor the germ-set construction’s parameters by explicit cross-references to the ambient dimension $d$ (Node 6 / Cap_H) and critical exponent $s_c$ (WP_{s_c} / ProfDec_{s_c,G}), and scope the bound to types with certified scaling data.
- Update this review file after edits to mark E-001–E-003 as resolved and cite the new bridge blocks and references used.

## Reference audit (for bridge proofs)
- **Bridge protocol + Rigor Class definitions:** `docs/source/2_hypostructure/01_foundations/02_constructive.md` (labels `def-rigor-classification`, `def-bridge-verification`)
- **Lock certificates / typed NO semantics:** `docs/source/2_hypostructure/03_sieve/01_structural.md` (`def-typed-no-certificates`) and `docs/source/2_hypostructure/03_sieve/02_kernel.md` (Lock reconstruction routing)
- **Germ Smallness Permit:** `docs/source/2_hypostructure/06_modules/01_singularity.md` (`def-germ-smallness`)
- **Ambient dimension (codimension/Cap_H):** `docs/source/2_hypostructure/04_nodes/01_gate_nodes.md` (`def-node-geom`)
- **Critical exponent s_c:** `docs/source/2_hypostructure/05_interfaces/01_gate_evaluator.md` (`WP_{s_c}`, `ProfDec_{s_c,G}`)
- **AIT/Kolmogorov formalism:** `docs/source/2_hypostructure/09_mathematical/04_taxonomy.md` (`def-kolmogorov-complexity`, `def-chaitin-omega`, `def-computational-depth`, `def-algorithmic-phases`)
- **Decidability in internal logic:** `docs/source/2_hypostructure/11_appendices/01_zfc.md` (`def-heyting-boolean-distinction`)

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | mt-krnl-consistency, mt-krnl-trichotomy, mt-krnl-equivariance, thm-halting-ait-sieve-thermo | Moderate | External dependency / Definition mismatch | Multiple theorems cite external results without Rigor Class L and Bridge Verification. **Resolved** (dual-provenance added). |
| E-002 | mt-krnl-exclusion proof, Step 6 | Moderate | Proof gap / Scope restriction | Uses decidability of Hom-emptiness from “discrete fragment” without tying to Lock certificates or termination permits. **Resolved** (certificate-conditional contrapositive). |
| E-003 | Initiality Lemma germ set construction | Minor | Notation conflict / Scope restriction | Uses parameters (Hausdorff dimension d, critical exponent s_c) not defined or referenced in this chapter. **Resolved** (critical index definition + cross-refs). |

## Detailed findings

### [E-001] Literature-based theorems lack Rigor Class L/Bridge Verification
- Location: mt-krnl-consistency, mt-krnl-trichotomy, mt-krnl-equivariance, thm-halting-ait-sieve-thermo
- Severity: Moderate
- Type: External dependency (secondary: Definition mismatch)
- Claim (paraphrase): These theorems invoke LaSalle, Lions/Bahouri-Gerard/Kenig-Merle, Noether/Cohen-Welling, Levin-Schnorr, etc., as part of the framework’s guarantees.
- Why this is an error in the framework: def-rigor-classification requires literature-anchored results to be labeled Rigor Class L with an explicit Bridge Verification. These statements are presented as internal theorems without a bridge construction that verifies the hypotheses in the hypostructure setting.
- Impact on downstream results: Readers may treat these as fully framework-internal proofs, which undermines the formal provenance model and blurs which results require external hypothesis checks.
**Resolution status:** Implemented.
- Dual-provenance added: each theorem now has a Framework-Original (Class F) statement plus a Literature-Anchored (Class L) alternative proof with Bridge Verification triples.
- Bridge blocks are expressed in terms of thin-interface certificates and cross-reference {prf:ref}`def-bridge-verification`.
- Validation: verify that the four theorems each include both provenance paths and that the certificate payloads match.

### [E-002] Decidability in KRNL-Exclusion needs an operational certification link
- Location: mt-krnl-exclusion proof, Step 6 (contrapositive in internal logic)
- Severity: Moderate
- Type: Proof gap / Scope restriction
- Claim (paraphrase): Because Hom-emptiness is certified in the discrete fragment, the contrapositive is valid and yields Rep_K.
- Why this is an error in the framework: Discrete/Boolean internal logic justifies classical reasoning, but the Sieve’s operational decision is certificate-based and may be inconclusive. The step should be conditional on producing a Lock certificate (K_lock^{blk}) or otherwise route to K^{inc}. The proof currently blurs logical entailment with algorithmic decidability.
- Impact on downstream results: It suggests the Lock always decides Hom-emptiness, contradicting the explicit breached-inconclusive path used elsewhere (e.g., Lock reconstruction).
- Fix guidance (step-by-step):
  1. Restate Step 6 as: if K_lock^{blk} certifies Hom-emptiness, then Rep_K follows; otherwise produce K^{inc} and route to reconstruction/Horizon.
  2. Add an explicit citation to the Lock obstruction tactics and their possible inconclusive outcome.
  3. Tie the “decidable in discrete fragment” statement to def-cert-finite or a specific termination/decidability permit.
- Required new assumptions/permits (if any): None if Lock certificate logic is referenced explicitly; otherwise add a “Lock decidability” permit.
- Framework-first proof sketch for the fix: Use case analysis on K_{Cat_Hom}^- = K^{wit} + K^{inc}. Only the K^{wit} or K^{blk} branch yields Rep_K; the K^{inc} branch defers to reconstruction.
**Resolution status:** Implemented.
- Step 6 now conditions the contrapositive on $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ and explicitly routes $K^{\mathrm{br\text{-}inc}}$ to reconstruction/Horizon.
- Validation: confirm references to {prf:ref}`def-typed-no-certificates` and the inconclusive branch wording align with Lock routing.

### [E-003] Germ set parameters are not anchored in this chapter
- Location: Initiality Lemma germ set construction (definition of G_T)
- Severity: Minor
- Type: Notation conflict / Scope restriction
- Claim (paraphrase): Germs satisfy dim_H(P) <= d - 2 s_c for critical exponent s_c.
- Why this is an error in the framework: The parameters d and s_c are not defined or referenced in this chapter, so the bound is not well-typed for arbitrary T without explicit links to where these quantities are defined.
- Impact on downstream results: The classifiable-singularity definition may be ambiguous outside PDE contexts; readers cannot verify the hypothesis without a cross-reference.
**Resolution status:** Implemented.
- Added explicit references to ambient dimension $d$ (capacity interface) and critical index $s_c$ (new general definition), with a conditional clause when scaling data are unavailable.
- Replaced PDE-only norm references with $X_c$ (critical phase space) and noted PDE specialization.
- Validation: confirm cross-references to {prf:ref}`def-critical-index` and {prf:ref}`def-node-geom` in the Initiality Lemma.

## Scope restrictions and clarifications
- None.

## Proposed edits (optional)
- None (all items addressed).

## Open questions
- None (dual-provenance adopted and general critical index definition added).
