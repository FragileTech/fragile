# Mathematical Review: docs/source/2_hypostructure/05_interfaces/01_gate_evaluator.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/05_interfaces/01_gate_evaluator.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-typed-no-certificates in docs/source/2_hypostructure/03_sieve/01_structural.md
  - def-determinism in docs/source/2_hypostructure/03_sieve/02_kernel.md
  - def-interface-permit in this file

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: Interface evaluator outputs are specified as three-valued, conflicting with the binary certificate logic/determinism policy.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | def-interface-permit and early Feynman prose | Moderate | Definition mismatch | Evaluators return {YES, NO, INC} explicitly, but the kernel’s determinism policy requires binary YES/NO with INC encoded as a NO certificate subtype. |

## Detailed findings

### [E-001] Three-valued evaluator output conflicts with binary certificate logic
- Location: def-interface-permit and the “three possible answers” prose near the top of the file
- Severity: Moderate
- Type: Definition mismatch
- Claim (paraphrase): Evaluators output YES/NO/INC, treating INC as a third outcome.
- Why this is an error in the framework: Binary Certificate Logic defines NO as a coproduct of witness and inconclusive certificates, and def-determinism specifies UNKNOWN/INC is treated as NO for routing. The interface permit definition should therefore encode INC as a NO outcome with K^{inc}, not a third branch.
- Impact on downstream results: It creates ambiguity about routing and determinism and could be interpreted as a three-valued logic at the node level, contradicting the kernel specification.
- Fix guidance (step-by-step):
  1. Rewrite the evaluator codomain as {YES, NO} with NO carrying either K^{wit} or K^{inc}.
  2. Keep the prose distinction between witness and inconclusive, but explicitly state that INC is a NO subtype.
  3. Cross-reference def-typed-no-certificates and def-determinism in this section.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use the coproduct K^- = K^{wit} + K^{inc} and case analysis to model all evaluator outcomes while preserving binary routing.
- Validation plan: Check all interface specifications and node evaluators for consistent YES/NO-only branching.

## Scope restrictions and clarifications
- None beyond the evaluator codomain alignment.

## Proposed edits (optional)
- Update def-interface-permit to encode INC as a NO subtype and add explicit cross-references to Binary Certificate Logic.

## Open questions
- Should any evaluator ever emit a third routing edge, or is all inconclusiveness always encoded as NO with K^{inc}?
