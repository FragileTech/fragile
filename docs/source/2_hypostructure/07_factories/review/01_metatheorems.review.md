# Mathematical Review: docs/source/2_hypostructure/07_factories/01_metatheorems.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/07_factories/01_metatheorems.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-progress-measures in docs/source/2_hypostructure/05_interfaces/03_contracts.md
  - thm-finite-runs in docs/source/2_hypostructure/03_sieve/02_kernel.md
  - def-surgery-schema in docs/source/2_hypostructure/04_nodes/03_surgery_nodes.md

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: Progress measure well-foundedness relies on discrete energy drop but is not stated as a requirement in the measure definition.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | mt-fact-surgery proof, Step 4 (Progress Measure) | Moderate | Scope restriction / Proof gap | The lexicographic measure uses a real-valued component without explicitly requiring discrete progress; well-foundedness depends on a stated minimum drop. |

## Detailed findings

### [E-001] Progress measure needs explicit discrete progress condition
- Location: mt-fact-surgery proof, Step 4 (Progress Measure)
- Severity: Moderate
- Type: Scope restriction / Proof gap
- Claim (paraphrase): The lexicographic measure (N_max - N_S, Phi_residual) is well-founded, so surgeries terminate.
- Why this is an error in the framework: The second component lives in R_{>=0}, which is not well-founded under strict decrease unless a uniform minimum drop (epsilon) is guaranteed. The proof mentions a delta_surgery > 0 but does not tie it to the discrete progress constraint or explicitly require it in the measure definition.
- Impact on downstream results: Without an explicit discrete progress condition, infinite descending chains in Phi_residual are possible, undermining the termination argument.
- Fix guidance (step-by-step):
  1. Add an explicit requirement that Phi_residual decreases by at least epsilon_T per surgery, referencing def-progress-measures.
  2. Alternatively, discretize Phi into epsilon_T levels and define the measure on those discrete levels.
  3. Make the progress certificate record epsilon_T so termination is verifiable.
- Required new assumptions/permits (if any): Discrete Progress Constraint (already in def-progress-measures) should be cited here.
- Framework-first proof sketch for the fix: With epsilon_T > 0, each surgery reduces Phi_residual by at least epsilon_T, so only finitely many decreases are possible before reaching zero (or the budgeted floor). Combined with N_max, the lexicographic order is well-founded.
- Validation plan: Ensure all surgery specifications emit a progress certificate with the epsilon_T bound or an explicit bound on the number of surgeries.

## Scope restrictions and clarifications
- None beyond the discrete-progress requirement.

## Proposed edits (optional)
- Cite def-progress-measures in Step 4 and record epsilon_T in the progress certificate payload.

## Open questions
- Is N_max intended to be finite for all types, or is termination primarily via discrete energy drops?
