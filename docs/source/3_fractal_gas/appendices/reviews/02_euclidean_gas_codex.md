# Codex Review: 02_euclidean_gas.md

## Summary

Analysis identified **broken cross-references** and **minor notation inconsistencies**. No severe corruption like 01_fragile_gas_framework.



## 🔴 Critical: Broken Cross-References

10+ instances of `def-axiom-*` that should be `axiom-*`:

| Line | Current Reference | Should Be |
|------|-------------------|-----------|
| 403 | `def-axiom-bounded-algorithmic-diameter` | `axiom-bounded-algorithmic-diameter` |
| 517 | `def-axiom-sufficient-amplification` | `axiom-sufficient-amplification` |
| 537 | `def-axiom-environmental-richness` | `axiom-environmental-richness` |
| 555 | `def-axiom-guaranteed-revival` | `axiom-guaranteed-revival` |
| 557 | `def-axiom-boundary-regularity` | `axiom-boundary-regularity` |
| 557 | `def-axiom-boundary-smoothness` | `axiom-boundary-smoothness` |
| 744 | `def-axiom-bounded-algorithmic-diameter` | `axiom-bounded-algorithmic-diameter` |
| 753 | `def-axiom-reward-regularity` | `axiom-reward-regularity` |
| 781 | `def-axiom-environmental-richness` | `axiom-environmental-richness` |
| 1247 | `def-axiom-non-degenerate-noise` | `axiom-non-degenerate-noise` |
| 1249 | `def-axiom-sufficient-amplification` | `axiom-sufficient-amplification` |
| 2215 | `def-axiom-bounded-algorithmic-diameter` | `axiom-bounded-algorithmic-diameter` |
| 2228-2230 | Table references | Same pattern |



## 🟡 Minor: Notation Inconsistency

**Lines 164, 175, 1321, 1331:**
- Uses both `σ'_{min,patch}` (with prime) and `σ_{min,patch}` (without prime)
- Definition at line 164: `σ'_{min,patch}`
- Definition at line 1321, 1331: `σ_{min,patch}`
- These should be consistent (probably `σ'_{min,patch}` is correct)



## 🟢 LaTeX: Aligned Environments OK

All `\begin{aligned}...\end{aligned}` blocks checked and properly use `\\` for line breaks:
- Lines 173-199 ✓
- Lines 475-479 ✓
- Lines 525-529 ✓
- Lines 591-594 ✓
- Lines 714-717 ✓
- Lines 905-909 ✓
- Lines 1447-1453 ✓
- Lines 1926-1930 ✓
- Lines 2140-2143 ✓



## Recommendations

1. **Find-replace all `def-axiom-` → `axiom-`** in this file
2. **Standardize notation** for regularized std dev floor variable
