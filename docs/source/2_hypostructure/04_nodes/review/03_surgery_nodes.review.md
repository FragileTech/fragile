# Mathematical Review: docs/source/2_hypostructure/04_nodes/03_surgery_nodes.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/04_nodes/03_surgery_nodes.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-surgery-schema in this file
  - def-progress-measures in docs/source/2_hypostructure/05_interfaces/03_contracts.md
  - thm-finite-runs in docs/source/2_hypostructure/03_sieve/02_kernel.md

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: Semantic non-circularity relies on an undefined dependency-rank constraint.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | thm-non-circularity (Scope of Non-Circularity) | Moderate | Proof gap / Reference error | Uses a “derivation-dependency constraint” and proof-DAG ranking without defining or referencing it elsewhere. |

## Detailed findings

### [E-001] Semantic non-circularity relies on an undefined dependency constraint
- Location: thm-non-circularity, “Scope of Non-Circularity” paragraph
- Severity: Moderate
- Type: Proof gap / Reference error
- Claim (paraphrase): Semantic circularity is prevented by a derivation-dependency constraint requiring lower-rank lemmas in a proof DAG.
- Why this is an error in the framework: The dependency-rank constraint is asserted but not defined or referenced in the volume; only the syntactic check (K_i^- not in Gamma) is formalized. Without an explicit definition or permit, the semantic non-circularity claim is unsupported.
- Impact on downstream results: The termination argument that relies on semantic non-circularity is only partially justified; proofs may still admit hidden circularity.
- Fix guidance (step-by-step):
  1. Define the derivation-dependency constraint explicitly (e.g., a certificate rank function and allowed inference edges).
  2. Reference where the proof DAG order is constructed and how it is enforced in certificate generation.
  3. Clarify how this constraint is checked or certified at runtime.
- Required new assumptions/permits (if any): A “derivation-rank” permit or an explicit definition of the proof DAG order.
- Framework-first proof sketch for the fix: Define a well-founded rank on certificate proofs (e.g., by node topological order) and require that each certificate cites only lower-rank lemmas; then show this forbids semantic circularity.
- Validation plan: Ensure any modules claiming semantic non-circularity point to the new definition or permit.

## Scope restrictions and clarifications
- None beyond the definition gap noted above.

## Proposed edits (optional)
- Add a formal definition of the derivation-dependency constraint and cite it here.

## Open questions
- Where should the dependency-rank constraint live: kernel (proof system) or interface permits?
