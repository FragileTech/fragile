# Review taxonomy

Use this taxonomy to classify issues by severity and error type. Assign a primary type; add a secondary type only if it materially changes the fix guidance.

## Severity levels

- Critical: Invalidates a main theorem, core algorithm, or primary conclusion. Results are unreliable without substantial rework.
- Major: Breaks an important lemma/proposition or a key step used later; downstream results may fail.
- Moderate: Local correctness issue that can be fixed with targeted edits or an added hypothesis; does not collapse the main line.
- Minor: Ambiguity, notational inconsistency, or imprecise wording that can mislead but does not change correctness.
- Note: Suggestions or clarity improvements; not a mathematical error.

## Error types

- Conceptual: Misunderstanding of the underlying framework or misuse of a core concept.
- Definition mismatch: Uses a term inconsistently with its definition or shifts definitions mid-proof.
- Invalid inference: A logical step does not follow from the previous statements within the framework.
- Proof gap / omission: A required argument, lemma, or permit is missing.
- External dependency: Relies on classical analysis or external results not derived in the framework.
- Scope restriction: Claim holds only under specific parameters (dimension, domain, boundary, regularity) but is stated as general.
- Parameter inconsistency: Conflicting parameter ranges or hidden constraints between steps.
- Dimensional mismatch: Shape, units, or dimension assumptions are violated.
- Algorithm mismatch: The described algorithm or step contradicts the formal definition elsewhere.
- Notation conflict: Symbols reused for different objects or overloaded without clarification.
- Miswording: Wording changes the logical meaning of a statement (not just style).
- Typo: Spelling or symbol typo that is obvious and local.
- Citation / reference error: Points to the wrong lemma, equation, or section.
- Circular reasoning: A statement is used to justify itself, directly or indirectly.
- Computational error: Algebraic, combinatorial, or arithmetic mistake within the framework.

## Framework-first rules

- Treat the document's stated framework as the only allowed foundation.
- Do not import external theorems unless the user explicitly requests it.
- If a step would be valid only with an external result, classify it as External dependency or Proof gap, and propose an internal permit/lemma.
- If an argument appears standard but is not derived in the framework, mark it as missing.

## Fix guidance checklist

For each issue, provide:
- Why it fails in the framework (cite the missing definition/permit or violated hypothesis).
- A minimal change that fixes the issue (rewrite, add hypothesis, or add a lemma/permit).
- Any new assumptions that would be required (explicitly listed).
- A short framework-first proof sketch for the fix, or a plan to derive it.
- How to validate the fix (local check or downstream impact check).
