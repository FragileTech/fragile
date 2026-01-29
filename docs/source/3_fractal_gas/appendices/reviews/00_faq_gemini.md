# Gemini Review: 00_faq.md

## 1. Broken Cross-References

- **Line 661 (Section N.11.4):** The "Hypocoercive rate" in the Rigor Classification table references only `{doc}`../convergence_program/10_kl_hypocoercive``. However, section N.4.4 references BOTH `10_kl_hypocoercive` AND `15_kl_convergence`. The table may be missing the second reference.

No other broken cross-references were found.

## 2. LaTeX Errors

**None found.** All mathematical formulas are well-formed.

## 3. Mathematical Inconsistencies

### Line 237 (Section N.2.3) - CRITICAL
The text claims the velocity update rule `v_new = α_rest * v_j + (1 - α_rest) * v_i` "conserves total momentum Σ m_i v_i."

**This is incorrect.** The change in total momentum is:
```
ΔP = m_i(v_new - v_i) = m_i α_rest(v_j - v_i) ≠ 0
```
This is a damping process, not momentum-conserving collision.

### Line 344 (Section N.4.1) - Dimensional Inconsistency
The Acoustic Stability constraint `γ > E[p_i] M² / (2dh)` has inconsistent units:
- If M is mass: `[M/T] > [M²/T]` implies `1 > M`
- If M is Lipschitz constant: `[M/T] > [M²/T⁵]`
Neither works without further context. Possible typo or missing terms.

### Lines 581 & 603 (Sections N.7.1 & N.7.2) - Gauge Group Confusion
- Line 581: Claims redundancy lifts to U(N) and yields SU(N)
- Line 603: Claims O(d) redundancy lifts to U(d) and yields SU(d)

N (population size) and d (latent dimension) are used interchangeably. The intended group should be **SU(d)**.

## 4. Typos and Grammatical Errors

### Line 242 (Section N.2.3)
"OS reconstruction" - acronym "OS" is not defined. Likely refers to **Osterwalder-Schrader**.

### Line 604 (Section N.7.2)
"$SU(d)$ (modulo the overall phase)" - Imprecise phrasing. The relationship is that U(d) ≅ SU(d) × U(1), where U(1) is the overall phase. "Modulo" is non-standard terminology here.
