# Mathematical Review: docs/source/2_hypostructure/06_modules/03_lock.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/06_modules/03_lock.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-typed-no-certificates in docs/source/2_hypostructure/03_sieve/01_structural.md
  - def-node-lock (Lock barrier) in docs/source/2_hypostructure/04_nodes/01_gate_nodes.md
  - mt-lock-reconstruction in docs/source/2_hypostructure/09_mathematical/02_algebraic.md

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: Inconsistent tactic-count references across the volume (E1–E13 here vs E1–E12 elsewhere).

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | E1–E13 tactic definitions and breached-inconclusive payload | Moderate | Notation conflict / Citation error | This module uses E1–E13, but other chapters reference E1–E12 for exhaustion, creating an inconsistent tactic set. |

## Detailed findings

### [E-001] Inconsistent exclusion-tactic count across the volume
- Location: “E1–E13 Exclusion Tactics” section and breached-inconclusive payload definition
- Severity: Moderate
- Type: Notation conflict / Citation error
- Claim (paraphrase): The Lock uses E1–E13 tactics, and breached-inconclusive lists {E1,…,E13} as exhausted.
- Why this is an error in the framework: Other documents (e.g., kernel and barrier descriptions) refer to E1–E12 exhaustion. This inconsistency creates ambiguity about the completeness condition and the payload schema for K_{Cat_Hom}^{br-inc}.
- Impact on downstream results: Implementations may disagree on the tactic set required for exhaustion, leading to mismatched certificates and reconstruction triggers.
- Fix guidance (step-by-step):
  1. Decide whether the exclusion tactic set is E1–E12 or E1–E13.
  2. Update all references to the chosen count in kernel, node definitions, and lock payload schemas.
  3. If E13 is a meta-tactic (exhaustive check), clarify its role and whether it should be included in the exhausted set.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Treat E13 as a meta-check over E1–E12, or incorporate it explicitly into the tactic list and adjust payloads accordingly.
- Validation plan: Search the volume for “E1–E12”/“E1–E13” and standardize the terminology.

## Scope restrictions and clarifications
- None beyond the tactic-set alignment.

## Proposed edits (optional)
- Normalize the tactic count across the volume and update the breached-inconclusive payload accordingly.

## Open questions
- Is E13 intended as a true tactic or a meta-condition summarizing exhaustion of E1–E12?
