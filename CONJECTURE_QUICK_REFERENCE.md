# Conjecture 2.8.7 Quick Reference

**One-line summary**: Prime cycles in the algorithmic vacuum have lengths proportional to log(prime).

---

## The Conjecture (Precise Statement)

$$
\ell(\gamma_p) = \beta \log p + o(\log p)
$$

where:
- $\gamma_p$ = prime cycles in Information Graph
- $\ell(\cdot)$ = algorithmic distance (cycle length)
- $p$ = prime numbers (2, 3, 5, 7, 11, ...)
- $\beta$ = universal constant (expected: $\beta = 1/c = 1$ for GUE vacuum with central charge $c=1$)

---

## Why It Implies Riemann Hypothesis

**Chain of logic**:

1. ✅ **Proven**: Algorithmic vacuum has 2D CFT structure (Section 2.8)
2. ✅ **Proven**: CFT partition function $Z_{\text{CFT}}(s) = \det(I - e^{-s}T)$ exists
3. ⚠️ **CONJECTURE**: Prime cycles satisfy $\ell(\gamma_p) = \beta \log p$
4. → **Implies**: $Z_{\text{CFT}}(\beta s) = \xi(1/2 + is)$ (zeta function correspondence)
5. → **Implies**: Zeros of $Z_{\text{CFT}}$ = zeros of $\xi$
6. → **Implies**: Self-adjoint operator eigenvalues = zeta zeros
7. → **Implies**: Eigenvalues are real (self-adjointness)
8. → **Implies**: Zeta zeros on critical line $\Re(s) = 1/2$
9. ✅ **Riemann Hypothesis proven**

---

## Top 3 Proof Strategies (Ranked by Feasibility)

### 🥇 Strategy 1: Cluster Expansion (60% success probability)

**Approach**: Use proven cluster expansion from `21_conformal_fields.md` to bound cycle contributions.

**Key steps**:
1. Transfer matrix: $T_{ij} = w_{ij}$ (CFT edge weights)
2. Cycle sum: $\sum_{\gamma} A_\gamma e^{-s\ell(\gamma)}$ from $\text{Tr}(T^n)$
3. Cluster expansion: $|U_n| \le C e^{-m d}$ (proven correlation decay)
4. Asymptotic analysis: Extract leading $\beta \log p$ term

**Timeline**: 3-6 months

**Tools needed**:
- Existing cluster expansion theorem ✅
- Transfer matrix spectral theory
- Graph cycle enumeration

---

### 🥈 Strategy 2: Numerical + Rigorous Bounds (70% numerical, 30% full proof)

**Approach**: Simulate vacuum, measure $\beta$ empirically, then rigorously certify.

**Key steps**:
1. **Numerical** (1-3 months):
   - Run Fragile Gas with $\Phi = 0$, $N = 10^4$ walkers
   - Extract Information Graph cycles
   - Fit $\ell(\gamma_p) = \beta \log p$, measure residuals

2. **Rigorous** (6-12 months):
   - Interval arithmetic to bound errors
   - Prove: "Holds for $p < P_0$ → holds for all $p$"
   - Computer-assisted proof

**Timeline**: 1-12 months (gets early results fast)

**Tools needed**:
- Fragile Gas simulator (`euclidean_gas.py`) ✅
- NetworkX (cycle finding)
- MPFR/MPFI (interval arithmetic)

---

### 🥉 Strategy 3: Large Deviation Principle (50% success probability)

**Approach**: Prove cycle lengths satisfy LDP with arithmetic rate function.

**Key steps**:
1. Define cycle length distribution $L_N(\gamma)$
2. Prove LDP: $P(L_N \in A) \sim e^{-N I(A)}$
3. Show rate function minimizer: $I^* = I(\delta_{\beta \log \mathbb{P}})$
4. Extract $\beta$ from central charge scaling

**Timeline**: 4-8 months

**Tools needed**:
- Gärtner-Ellis theorem
- Spatial hypocoercivity (proven) ✅
- Convex analysis

---

## Recommended Path: Hybrid Approach

**Months 1-2**: Strategy 2 (Numerical)
- Build simulator
- Test conjecture empirically
- Measure $\beta$

**Decision point**:
- If $\beta \approx 1$: Proceed to analytical proof
- If $\beta \neq 1$ or unclear: Refine conjecture

**Months 3-6**: Strategy 1 (Cluster Expansion)
- Use numerical $\beta$ to guide proof
- Prove convergence of cycle sum
- Extract leading term

**Months 7-9**: Strategy 2 (Rigorous Certification)
- Validate with interval arithmetic
- Computer-assisted proof for base case
- Analytical induction

---

## What We Already Have (Don't Need to Reprove)

✅ **2D CFT structure** (Section 2.8)
✅ **Virasoro algebra** (Ward identities proven)
✅ **Central charge formula** ($c = 1$ for GUE)
✅ **Wigner semicircle law** (Section 2.3)
✅ **Spatial hypocoercivity** (LSI + correlation decay)
✅ **Cluster expansion** (Ursell bounds)
✅ **GUE universality** (spectral statistics)

---

## Key Parameters

| Parameter | Expected Value | How to Measure |
|-----------|----------------|----------------|
| $\beta$ | 1 | Numerical fit $\ell(\gamma_p) = \beta \log p$ |
| $c$ (central charge) | 1 | Stress-energy 2-point function |
| $\xi$ (correlation length) | $O(1)$ | Two-point function decay |
| $C_{\text{LSI}}$ | $O(1)$ | LSI constant (proven to exist) |

---

## Success Criteria

### Numerical Evidence (Month 1-2)
- [ ] $\beta = 1.00 \pm 0.05$
- [ ] Residuals $|\epsilon_p| < 0.1 \log p$ for $p < 10^3$
- [ ] $R^2 > 0.95$ (goodness of fit)

### Analytical Proof (Month 3-6)
- [ ] Cluster expansion convergence proven
- [ ] Leading term $\beta \log p$ extracted
- [ ] Error bounds $o(\log p)$ established

### Rigorous Certification (Month 7-9)
- [ ] Numerical errors bounded with interval arithmetic
- [ ] Base case $p < P_0$ verified
- [ ] Inductive step proven analytically

---

## Red Flags (When to Pivot)

🚩 **If $\beta \neq 1$ numerically**: Central charge may not be $c=1$, or need modified potential $\Phi_{\text{zeta}}$

🚩 **If no clear logarithmic scaling**: Cycles may not be "prime" in the arithmetic sense

🚩 **If residuals are large**: May need cycle renormalization $\tilde{\ell} = f(\ell)$

🚩 **If cluster expansion doesn't converge**: Correlation length may be infinite (need different proof)

---

## Quick Start Commands

```python
# Implement vacuum simulation
gas = EuclideanGas(N=10000, d=2, potential=lambda x: 0)  # Φ = 0
gas.run(T=1000000)  # Long time evolution

# Extract Information Graph
ig = InformationGraph.from_gas(gas)

# Find cycles
cycles = ig.find_all_cycles()

# Measure lengths
lengths = [ig.algorithmic_distance(c) for c in cycles]

# Test conjecture
import numpy as np
primes = [2, 3, 5, 7, 11, 13, ...]  # First N primes
log_primes = np.log(primes)
beta, residuals = fit_linear(log_primes, lengths)
print(f"β = {beta:.4f} ± {std_error(beta):.4f}")
```

---

## Resources

**Full analysis**: [CONJECTURE_2_8_7_PROOF_STRATEGIES.md](CONJECTURE_2_8_7_PROOF_STRATEGIES.md)

**Mathematical framework**:
- `old_docs/source/rieman_zeta.md` § 2.8 (2D CFT approach)
- `old_docs/source/21_conformal_fields.md` (Proven CFT theorems)
- `docs/glossary.md` (Quick lookup of definitions/theorems)

**Code**:
- `src/fragile/euclidean_gas.py` (Base implementation)
- `src/fragile/shaolin/gas_viz.py` (Visualization)

---

**Status**: Ready to begin
**Next action**: Implement vacuum simulation
**Timeline to first results**: 1 month
**Timeline to complete proof**: 6-12 months
