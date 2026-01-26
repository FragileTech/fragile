# Mathematical Review: 01_fragile_gas_framework.md

**Reviewed by:** Miau (Claude), Gemini CLI, Codex
**Date:** 2026-01-26
**File:** `docs/source/3_fractal_gas/appendices/01_fragile_gas_framework.md`
**Lines:** 5706

---

## Executive Summary

This is the foundational document establishing the Fragile Gas framework with 16 parametric axioms. The document is mathematically dense and generally well-structured. Key issues identified relate to notation consistency, implicit assumptions, and some proof gaps.

**Status:** Review in progress - agents still running

---

## Preliminary Findings (Miau Analysis)

### CRITICAL Issues

*None identified yet in reviewed sections - document appears mathematically rigorous*

---

### MAJOR Issues

#### 1. **Section Numbering Inconsistency**
**Severity:** MAJOR

The document uses inconsistent section numbering:
- Sections labeled "1.1, 1.2..." under "Section 2" (Global Conventions)
- This creates confusion between document structure and internal numbering

**Location:** Lines ~170-300

---

#### 2. **Duplicate Sub-Lemma Definitions**
**Severity:** MAJOR

Sub-lemmas appear twice with identical content:
- `lem-sub-stable-positional-error-bound` (appears at ~line 2500 and again)
- `lem-sub-stable-structural-error-bound` (similar duplication)

**Resolution needed:** Remove duplicate definitions

---

#### 3. **Kolmogorov Quotient Definition Missing Context**
**Severity:** MAJOR

**Location:** Around line 530

The metric quotient definition (`def-metric-quotient`) is introduced but the equivalence relation $\mathcal{S}_1 \sim \mathcal{S}_2$ is defined circularly referencing the pseudometric being defined.

**Resolution needed:** Clarify the order of definitions

---

### MINOR Issues

#### 4. **Walker Definition Redundant References**
**Severity:** MINOR

Throughout the document, `({prf:ref}\`def-walker\`)` appears repeatedly inline, making the text harder to read. Consider removing redundant back-references.

---

#### 5. **Missing Units/Dimensions**
**Severity:** MINOR

Some axiomatic parameters lack explicit dimensionality:
- $\kappa_{\text{revival}}$ - dimensionless ratio (clarified)
- $L_{\text{death}}$ - probability per unit distance? (unclear)
- $\alpha_B$ - Hölder exponent (should be in $(0,1]$ per definition)

---

#### 6. **Notation: $\sigma$ vs $\sigma'$**
**Severity:** MINOR

**Location:** Line ~2280 (Section 12)

The document explicitly addresses this with a "Notation freeze" admonition, which is good practice. However, earlier sections use $\sigma$ before this clarification appears.

**Recommendation:** Move notation freeze earlier or add forward reference.

---

#### 7. **Reference to Non-Existent Sections**
**Severity:** MINOR

Several references point to sections that may not exist in this document:
- `{doc}\`/source/1_agent/05_geometry/02_wfr_geometry\`` - external reference (OK if exists)
- Various `{prf:ref}` cross-references need verification

---

### Structural Observations

1. **Axiomatic Framework:** The 16 parametric axioms + 1 structural assumption are well-organized into:
   - Viability Axioms (survival)
   - Environmental Axioms (problem structure)
   - Regularity Axioms (algorithmic stability)

2. **Proof Structure:** Proofs generally follow a clear pattern:
   - State the bound/inequality
   - Decompose into components
   - Apply sub-lemmas
   - Combine with Q.E.D.

3. **Continuity Analysis:** The mean-square continuity framework (Sections 10-12) is mathematically rigorous with explicit error bounds.

---

## Agent Reviews (Pending)

### Codex Review
*Status: Running (session `tide-orbit`)*

### Gemini Review
*Status: Restarted (session `fresh-kelp`)*

---

## Summary Table (Preliminary)

| Issue | Section | Severity | Status |
|:------|:--------|:---------|:-------|
| Section numbering inconsistency | §2 | MAJOR | Verify |
| Duplicate sub-lemma definitions | §10.2 | MAJOR | Needs fix |
| Circular quotient definition | §1.6.1 | MAJOR | Needs clarification |
| Redundant walker references | Throughout | MINOR | Stylistic |
| Missing dimensionality | §3 axioms | MINOR | Add clarification |
| Notation freeze placement | §12 | MINOR | Relocate |
| Reference verification | Various | MINOR | Check links |

---

*Generated: 2026-01-26 (preliminary)*
*Will be updated when agent reviews complete*
