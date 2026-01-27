# Mathematical Review: docs/source/2_hypostructure/11_appendices/03_faq.md

## Metadata
- Reviewed file: docs/source/2_hypostructure/11_appendices/03_faq.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Entire document
- Framework anchors (definitions/axioms/permits):
  - def-typed-no-certificates in docs/source/2_hypostructure/03_sieve/01_structural.md
  - def-determinism in docs/source/2_hypostructure/03_sieve/02_kernel.md
  - E1–E13 tactics in docs/source/2_hypostructure/06_modules/03_lock.md

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: FAQ wording conflicts with binary routing semantics; tactic-set mismatch repeats (E1–E12 vs E1–E13).

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | H.4.5 (Halting Problem) – INC handling | Moderate | Miswording / Definition mismatch | Says INC is “not treated as NO,” conflicting with binary routing where INC is a NO subtype. |
| E-002 | H.5.1 (Factory soundness) – tactic list | Moderate | Notation conflict | References E1–E12 tactics, inconsistent with the Lock module’s E1–E13 set. |

## Detailed findings

### [E-001] INC handling phrasing conflicts with binary routing semantics
- Location: H.4.5 “How Do You Avoid the Halting Problem in Verification?”
- Severity: Moderate
- Type: Miswording / Definition mismatch
- Claim (paraphrase): Timeouts return K^{inc} and this is “not treated as NO (failure).”
- Why this is an error in the framework: Binary Certificate Logic and def-determinism specify that INC is a NO subtype for routing purposes; it is not a witness-based failure, but it still follows the NO edge. The FAQ wording can be read as implying a third routing branch, which contradicts the kernel specification.
- Impact on downstream results: Readers may misunderstand routing behavior and believe INC bypasses NO edges, altering the operational semantics.
- Fix guidance (step-by-step):
  1. Clarify that INC is encoded as NO with K^{inc} and follows the NO edge, while remaining semantically non-refutational.
  2. Add an explicit cross-reference to def-typed-no-certificates or def-determinism.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use the coproduct K^- = K^{wit} + K^{inc}; routing is by case analysis, not by a third edge.
- Validation plan: Check other FAQ entries for consistent phrasing of INC handling.

### [E-002] Tactic enumeration mismatch in the FAQ
- Location: H.5.1 “Factory soundness” (mentions E1–E12 tactics)
- Severity: Moderate
- Type: Notation conflict
- Claim (paraphrase): The Lock uses E1–E12 tactics with inc fallback.
- Why this is an error in the framework: The Lock module defines E1–E13 tactics and includes E13 in the exhaustion payload. Referencing E1–E12 here is inconsistent with the canonical tactic set.
- Impact on downstream results: Conflicting expectations about completeness and payload schema for breached-inconclusive certificates.
- Fix guidance (step-by-step):
  1. Align the tactic count with the Lock module (E1–E13) or explicitly define E13 as a meta-check.
  2. Update this FAQ entry to match the chosen canonical set.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Treat E13 consistently as either a tactic or a meta-condition and update all references accordingly.
- Validation plan: Search the volume for E1–E12 vs E1–E13 and standardize.

## Scope restrictions and clarifications
- None beyond the wording alignment.

## Proposed edits (optional)
- Clarify INC routing as NO-with-incertitude in H.4.5.
- Standardize the tactic count across the volume and update the FAQ accordingly.

## Open questions
- Is E13 intended as a proper tactic or a meta-condition summarizing exhaustion of E1–E12?
