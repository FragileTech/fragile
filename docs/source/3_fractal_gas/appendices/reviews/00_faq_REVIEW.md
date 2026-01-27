# Mathematical Review: 00_faq.md (Appendix N)

**Reviewed by:** Miau (Claude), Gemini CLI, Codex (pending)
**Date:** 2026-01-26
**File:** `docs/source/3_fractal_gas/appendices/00_faq.md`

---

## Executive Summary

This FAQ document addresses common objections to the Fractal Gas framework. The review identified several mathematical issues ranging from critical inconsistencies to minor notation concerns.

---

## Critical Issues

### 1. **Momentum Conservation Claim is Mathematically False** (N.2.3)
**Severity:** CRITICAL

**Issue:** Section N.2.3 claims the inelastic collision $v_{\text{new}} = \alpha_{\text{rest}} v_j + (1 - \alpha_{\text{rest}}) v_i$ conserves total momentum $\sum m_i v_i$.

**Analysis:** In a cloning operation where walker $i$ updates ($v_i \to v_{\text{new}}$) and walker $j$ acts as template (unchanged), the total momentum change is:
$$\Delta P = m(v_{\text{new}} - v_i) = m \alpha_{\text{rest}} (v_j - v_i)$$

This is **non-zero** unless $v_i = v_j$ or $\alpha_{\text{rest}} = 0$.

**Resolution needed:** Either (a) the source walker $j$ must explicitly recoil, or (b) the claim of momentum conservation should be removed/corrected.

---

### 2. **Falsifiability Prediction is Self-Contradictory** (N.11.3 vs N.2.1)
**Severity:** CRITICAL

**Issue:** The falsifiability test in N.11.3 contradicts the definition in N.2.1.

**Analysis:**
- N.2.1 defines: $S_i(j) = -S_j(i) \cdot \frac{V_j + \varepsilon}{V_i + \varepsilon}$
- N.11.3 predicts measuring $Q = S_i(j) + S_j(i) \cdot \frac{V_i + \varepsilon}{V_j + \varepsilon}$ yields zero

Substituting:
$$Q = \left[ -S_j(i) \frac{V_j + \varepsilon}{V_i + \varepsilon} \right] + S_j(i) \frac{V_i + \varepsilon}{V_j + \varepsilon} = S_j(i) \left[ \frac{V_i + \varepsilon}{V_j + \varepsilon} - \frac{V_j + \varepsilon}{V_i + \varepsilon} \right]$$

This equals zero **only if** $V_i = V_j$, which defeats the purpose of the test.

**Resolution needed:** Fix the algebraic identity or the falsifiability prediction.

---

## Major Issues

### 3. **Ambiguous Notation "D"** (N.1.3)
**Severity:** MAJOR

**Issue:** The symbol $D$ is overloaded:
- In fitness: $V_{\text{fit}} = (d')^\beta (r')^\alpha$ — here $d'$ is diversity
- In QSD: $\rho_{\text{QSD}}(z) \propto R(z)^{\alpha D/\beta}$ — here $D$ appears as a constant
- In phase: $\Gamma = \beta/(\alpha D \cdot h\gamma)$

**Problem:** If $D$ refers to variable diversity ($d'$), the QSD density formula has a spatially-varying exponent (invalid). If $D$ is dimension, it conflicts with $d$ used elsewhere.

**Resolution needed:** Explicitly define $D$ and distinguish from diversity $d'$ and dimension $d$.

---

### 4. **BAOAB Symplectic Claim is Imprecise** (N.3.1)
**Severity:** MAJOR

**Issue:** N.3.1 claims "BAOAB... is a symmetric splitting that preserves the symplectic form to $O(h^2)$."

**Analysis:** BAOAB for Langevin dynamics is **not** symplectic due to the stochastic O (Ornstein-Uhlenbeck) step. It's a splitting scheme that samples the correct Boltzmann distribution, but "symplectic preservation" is technically incorrect.

**Resolution needed:** Clarify that BAOAB preserves the correct invariant measure, not the symplectic form per se.

---

### 5. **Wilson Action on Triangles is Ill-Defined** (N.7.3)
**Severity:** MAJOR

**Issue:** The Wilson action $S_W = \beta \sum_{\text{plaquettes}} \text{Re Tr}(1 - U_{\square})$ defines plaquettes as pairs of adjacent triangles.

**Problem:** On a 2D simplicial complex, each triangle has 3 neighbors. There's no canonical pairing into quadrilaterals without double-counting or gaps, unless the mesh has specific structured geometry (contradicting the "fractal" nature).

**Resolution needed:** Define the pairing scheme explicitly or use triangle-based action directly.

---

### 6. **Inconsistent Doeblin Constraints** (N.4.1 vs N.10.2)
**Severity:** MAJOR

**Issue:**
- N.4.1: $\varepsilon \geq D_{\text{alg}} / \sqrt{2\ln((N-1)/p_{\min})}$
- N.10.2: $\varepsilon \geq D_{\text{alg}} / \sqrt{2\ln N}$

**Problem:** N.10.2 drops the $p_{\min}$ term. If $p_{\min}$ is small (e.g., $10^{-6}$), then $\ln((N-1)/p_{\min}) \gg \ln N$, making N.10.2 too loose.

**Resolution needed:** Use consistent constraint formulation.

---

## Minor Issues

### 7. **Notation Overload on "V"**
**Severity:** MINOR

Multiple uses:
- $v$ = velocity (N.1.1)
- $V_{\text{fit}}$ = fitness (N.1.3)
- $V_j$ = fitness in cloning score (N.2.1)
- $V_{\text{eff}}$ = effective potential (N.8.2)
- $\Delta V$ = fitness gap (N.8.4)

**Recommendation:** Use distinct symbols (e.g., $\mathbf{v}$ for velocity, $\mathcal{F}$ for fitness, $U$ for potential).

---

### 8. **Grassmann Encoding Gap** (N.2.2)
**Severity:** MINOR

**Issue:** The text asserts that scalar antisymmetry $S_i(j) \approx -S_j(i)$ "is encoded" by Grassmann variables $\theta_i$ with $\theta_i \theta_j = -\theta_j \theta_i$.

**Gap:** No bridging definition explains how a scalar algorithm variable transforms into an anticommuting algebraic object (e.g., via path integral formulation).

**Resolution needed:** Add clarifying bridge or reference to where this is formalized.

---

### 9. **$O(d)$ to $U(d)$ Jump** (N.7.2)
**Severity:** MINOR

**Issue:** Claims "isotropic coupling... lifts to $U(d)$" but real isotropic symmetry is $O(d)$. The jump to $U(d)$ requires a complex structure $J$ with $J^2 = -1$.

**Resolution needed:** Explain where the imaginary component arises in "momentum-phase complexification."

---

### 10. **"Forty Objections" Count** (Line 12)
**Severity:** MINOR

**Issue:** Document claims "forty rigorous objections" but actual FAQ count appears different.

**Resolution needed:** Verify count or revise claim.

---

### 11. **Mean-Field Limit Clarification** (N.1.4)
**Severity:** MINOR

**Issue:** Claims "Mean-field limit takes $N \to \infty$ with fixed density" — ambiguous whether this means fixed normalized density (probability measure) or thermodynamic limit (growing domain).

**Resolution needed:** Clarify which interpretation applies.

---

## Summary Table

| Issue | Section | Severity | Status |
|:------|:--------|:---------|:-------|
| Momentum conservation false | N.2.3 | CRITICAL | Needs fix |
| Falsifiability self-contradiction | N.11.3 | CRITICAL | Needs fix |
| Ambiguous D notation | N.1.3 | MAJOR | Needs clarification |
| BAOAB symplectic claim | N.3.1 | MAJOR | Needs precision |
| Wilson action ill-defined | N.7.3 | MAJOR | Needs definition |
| Inconsistent Doeblin bounds | N.4.1/N.10.2 | MAJOR | Needs consistency |
| V notation overload | Multiple | MINOR | Recommend fix |
| Grassmann encoding gap | N.2.2 | MINOR | Needs bridge |
| O(d)→U(d) jump | N.7.2 | MINOR | Needs explanation |
| Forty objections count | Line 12 | MINOR | Verify |
| Mean-field limit ambiguity | N.1.4 | MINOR | Clarify |

---

## Reviewers

**Miau (Claude Opus 4.5):** Primary analysis and compilation
**Gemini CLI (Gemini 3 Pro):** Detailed mathematical review
**Codex (GPT-5.2):** Review in progress

---

*Generated: 2026-01-26*
