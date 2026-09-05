# Codex Review: 01_fragile_gas_framework.md

## 1. Broken Cross-References (CRITICAL - Many lines affected)

### Wrong reference type (`{prf:ref}` instead of `{doc}`)
Lines 328, 645, 1652, 2051, 2163, 2948, 3031, 3067, 3087, 5516, 5540 use `{prf:ref}`02_euclidean_gas`` but this is a **document**, not a label. Should be `{doc}`02_euclidean_gas``.

### Malformed nested references
- **Line 495**: Nested `{prf:ref}` inside another reference
- **Lines 1207, 1209, 1234**: Lemma titles with embedded `{prf:ref}` fragments corrupting references
- **Lines 2342, 2567**: Broken references mixing labels and `{prf:ref}`
- **Lines 3020, 3035, 3400**: Corrupted anchor syntax like `[](#def-swarm ({prf:ref}...)...)`
- **Line 4325**: Invalid `:label` syntax; lines 4323, 4329 have inline `{prf:ref}` insertions breaking directives

### Undefined labels referenced
- Lines 1587, 1625, 5374 reference `proof-thm-total-error-status-bound` and `proof-lem-sub-unify-holder-terms` - **labels don't exist**

## 2. LaTeX Errors (CRITICAL)

### Math mode issues
- **Lines 1121, 1124**: Equations outside math mode with stray closing `$$`
- **Lines 2067, 2083**: Malformed inline math (invalid accented commands, missing delimiters)
- **Line 3388**: Invalid command `\cLipschitz` inside math
- **Line 3036**: Uses `\sigma\'_{\text{reg}}` (text-mode accent in math) - MathJax/LaTeX rejects this

### 48 occurrences of `\sigma\'` pattern
Lines 2067, 2071, 2074, 2077, 2079, 2083, 2117, 2120, 2124, 2131, 2142, 2148, 2898, 2919, 3024, 3029, 3035, 3036, 3041, 3062 (and more) - all use invalid accent syntax in math mode

## 3. Mathematical Inconsistencies

### Independence vs Coupling conflict
- **Line 689**: Axiom of Boundary Regularity allows "state-dependent coupling between walkers"
- **Line 579**: Assumption A requires within-step independence
- **Conflict**: These statements are potentially contradictory and need clarification

### Section numbering mismatch
- **Line 649**: States k=1 case handled in §16
- **Line 128**: Document structure places revival in Section 17
- Internal navigation inconsistent

## 4. Typos / Corrupted Text (SEVERE)

### Merged/corrupted words
- **Lines 1207, 1234**: "deatboundary", "boLipschitz" - corrupted lemma headings
- **Lines 3008, 3145**: "LetLipschitz…", "operatorLipschitz…" mid-sentence
- **Lines 3293, 3385**: "raw value" fragments injected mid-word
- **Line 1127**: Entire table row corrupted with broken markdown

### Structural issues
- **Line 149**: Subsection numbering restarts at 1.1 under Section 2

## Summary

This file has **severe text corruption** throughout, likely from a bad find-replace or merge operation. The pattern suggests `{prf:ref}` tags were inserted incorrectly into existing text, breaking:
- Cross-references
- LaTeX math
- Readable prose

**Recommendation**: This file needs systematic cleanup before further review.
