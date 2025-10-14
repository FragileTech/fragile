# Deprecated Analysis Documents

**Date Created**: 2025-10-14

This folder contains analysis documents that have been superseded by corrected understanding.

## Why These Documents Are Obsolete

The documents in this folder were created during October 14, 2025 investigation of whether the Fragile Gas QSD satisfies the requirements for Haag-Kastler axioms (needed for Yang-Mills Millennium Prize proof).

**Initial concern**: The QSD appeared NOT to be a Gibbs state because:
1. Fitness formula uses power laws of Z-scores (not exponential in energy)
2. Detailed balance seemed to fail
3. Lindbladian is non-unitary

**Resolution** (same day): User pointed out two critical insights that were already proven in the framework:

1. **Riemannian Gibbs state** ([13_fractal_set_new/04_rigorous_additions.md](../13_fractal_set_new/04_rigorous_additions.md)):
   - QSD IS a Gibbs state on Riemannian manifold
   - Has form: ρ ∝ √(det g) · exp(-β H_eff)
   - √(det g) factor from Stratonovich calculus (fundamental, not correction)

2. **Quantum amplitude structure** ([13_fractal_set_new/01_fractal_set.md](../13_fractal_set_new/01_fractal_set.md)):
   - Amplitudes ψ_ik = √P · exp(iθ) with unitarity
   - Two-level structure: unitary at amplitude level, Lindbladian at measurement level
   - Resolves "non-unitary" objection

**Corrected analysis**: See [15_millennium_problem_completion.md](../15_millennium_problem_completion.md) §20.6.6-20.6.8 for the correct proof that QSD satisfies all five Haag-Kastler axioms.

## Documents in This Folder

### DETAILED_BALANCE_FAILURE.md
- **Status**: ❌ INCORRECT
- **Why obsolete**: Analyzed detailed balance at algorithmic level, missed that mean-field limit produces Gibbs state
- **Correct version**: See [22_geometrothermodynamics.md](../22_geometrothermodynamics.md) Theorem thm-qsd-canonical-ensemble

### QSD_GIBBS_CRITICAL_ISSUE.md
- **Status**: ❌ INCORRECT
- **Why obsolete**: Same issue as above

### QSD_THERMAL_EQUILIBRIUM_RESOLUTION.md
- **Status**: ⚠️ PARTIALLY CORRECT
- **Why superseded**: Correctly identified many-body nature of fitness, but didn't incorporate Riemannian geometry

### WIGHTMAN_AXIOMS_CRITICAL_ISSUE.md
- **Status**: ✅ PARTIALLY CORRECT
- **Why obsolete**: Correctly identified Wightman/Lindbladian incompatibility, but solution (Haag-Kastler) is now proven viable
- **Note**: The quantum amplitude structure resolves the unitarity requirement

### WARNINGS_ADDED_SUMMARY.md
- **Status**: ⚠️ OUTDATED
- **Why obsolete**: Lists warnings that have been resolved

## Lessons Learned

1. **Always check framework documents first**: The Riemannian structure and quantum amplitudes were already proven
2. **Distinguish algorithmic vs effective theory**: Detailed balance not required at implementation level
3. **Mean-field emergence is powerful**: Complex algorithmic rules → simple thermal equilibrium in N→∞ limit
4. **Trust proven theorems**: When something seems inconsistent, re-read the proofs carefully

## References to Corrected Analysis

- **Main document**: [15_millennium_problem_completion.md](../15_millennium_problem_completion.md) §20
- **Riemannian Gibbs**: [13_fractal_set_new/04_rigorous_additions.md](../13_fractal_set_new/04_rigorous_additions.md) lines 122-142
- **Quantum amplitudes**: [13_fractal_set_new/01_fractal_set.md](../13_fractal_set_new/01_fractal_set.md) lines 704-727
- **Thermodynamics**: [22_geometrothermodynamics.md](../22_geometrothermodynamics.md) §1

---

**Note**: These documents are preserved for historical record and to document the investigation process. Do NOT cite them in papers or use them for understanding the framework.
