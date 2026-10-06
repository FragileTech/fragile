# Claude Review: 01_fragile_gas_framework.md

## ✅ CLEANUP COMPLETED

**Fixed ~100+ corrupted patterns including:**
- 48 instances of `\sigma\'` → `\sigma'`
- ~20 mid-word "Lipschitz ({prf:ref}...)" insertions
- ~15 mid-word "raw value ({prf:ref}...)" insertions
- ~10 mid-word "boundary ({prf:ref}...)" insertions
- Several completely corrupted lemma titles and theorem statements

## Overview (Pre-cleanup)
This file had **severe systematic corruption**. Codex identified the pattern: `{prf:ref}` tags appear to have been incorrectly inserted mid-word throughout the document, likely from a bad search-replace operation.

## 1. Root Cause Analysis

The corruption pattern appears to be:
- Text like "Lipschitz" became "Lipschitz ({prf:ref}`axiom-reward-regularity`)"
- Text like "swarm" became "swarm ({prf:ref}`def-swarm-and-state-space`)"
- These insertions happened **mid-word** in many cases, creating garbage like "operatorLipschitz" and "deatboundary"

**Hypothesis**: An automated tool attempted to add cross-references to technical terms but matched substrings incorrectly.

## 2. Mathematical Inconsistencies (Beyond Corruption)

### Assumption A vs Boundary Regularity Conflict
- **Line 567-579**: Assumption A requires within-step independence of walker random inputs
- **Line 689**: Axiom of Boundary Regularity explicitly allows "state-dependent coupling between walkers"

**Issue**: If walkers can have state-dependent coupling, how can their random inputs be independent? This needs clarification:
- Either the coupling is deterministic (conditional on current state) and randomness is still independent
- Or the axiom scope needs to be clarified

### Section Numbering Inconsistencies
The document outline (lines 122-130) describes Sections 2-21, but internal references cite different section numbers (e.g., "§16" for revival when outline says Section 17).

## 3. LaTeX Issues (Pattern)

48+ occurrences of `\sigma\'` with text-mode accent in math. This suggests systematic search-replace of `σ'` with `\sigma\'` instead of `\sigma'`.

## 4. Structural Issues

### Line 1127 - Corrupted Table
An entire Markdown table row is broken, with unbalanced `|` characters and `$...$` math that spans cells incorrectly.

### Line 1121-1124 - Math Outside Delimiters
```
which would otherwise allow a tautological "margin" by tuning $\lambda_{\mathrm{status}}$.
n_c\;\le\; ...
```
The second line should be inside `$$...$$` delimiters.

## 5. Recommendations

1. **Before fixing individual issues**: Run a systematic cleanup script to:
   - Remove all mid-word `{prf:ref}` insertions
   - Fix `\sigma\'` → `\sigma'`

2. **Verify after cleanup** with `sphinx-build` or similar

3. **Then review** for actual mathematical errors vs. corruption artifacts

## Summary

**Priority**: This file needs **automated cleanup first**, then manual review. The corruption is too systematic to fix line-by-line efficiently.
