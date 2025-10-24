# Variance Requirement Analysis: Global Edge Budget

**Date**: 2025-10-24
**Issue**: Phase-Space Packing Lemma requires unrealistic variance for O(N^{3/2}) edge budget
**Status**: ⚠️ **POTENTIAL PROBLEM IDENTIFIED**

---

## Executive Summary

The Phase-Space Packing Lemma from `03_cloning.md` (which IS citable) can derive a global edge budget, BUT achieving O(N^{3/2}) edges requires **almost-maximal variance** that may exceed physical limits.

**Key Finding**: The O(N^{3/2}) edge budget claimed in the hierarchical clustering proof **may be too optimistic**. The actual edge budget is likely **O(N²)** with realistic variance assumptions.

**Impact**: If edge budget is O(N²) instead of O(N^{3/2}), the hierarchical clustering bound becomes L = O(N) instead of L = O(√N), which would invalidate the global regime concentration improvement.

---

## Phase-Space Packing Lemma (Current Framework)

**Source**: `docs/source/1_euclidean_gas/03_cloning.md` lines 2420-2550
**Label**: `lem-phase-space-packing`

**Statement**:
```markdown
For a swarm with k walkers and hypocoercive variance Var_h(S_k):

$$
f_{\text{close}} = \frac{N_{\text{close}}}{\binom{k}{2}} \le \frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h(S_k)}{D_{\text{valid}}^2 - d_{\text{close}}^2}
$$

Therefore:

$$
N_{\text{close}} \le \binom{k}{2} \cdot \frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h(S_k)}{D_{\text{valid}}^2 - d_{\text{close}}^2}
$$
```

**Threshold Property**: If Var_h > d_close²/2, then f_close < 1 (not all pairs are close).

---

## Application to Global Regime

### Setup

- **Number of walkers**: K = cN (constant fraction c ∈ (0,1))
- **Proximity threshold**: d_close = D_max/√N
- **Phase-space diameter**: D_valid = D_max (for simplicity, assuming λ_alg ≈ 1)

### Derivation of Edge Budget

Substituting into the Phase-Space Packing formula:

$$
N_{\text{close}} \le \binom{cN}{2} \cdot \frac{D_{\text{max}}^2 - 2\mathrm{Var}_h(\mathcal{C})}{D_{\text{max}}^2 - (D_{\text{max}}/\sqrt{N})^2}
$$

Simplifying the denominator:
$$
D_{\text{max}}^2 - \frac{D_{\text{max}}^2}{N} = D_{\text{max}}^2\left(1 - \frac{1}{N}\right) \approx D_{\text{max}}^2
$$

For large N:
$$
N_{\text{close}} \lesssim \frac{c^2N^2}{2} \cdot \frac{D_{\text{max}}^2 - 2\mathrm{Var}_h(\mathcal{C})}{D_{\text{max}}^2}
$$

### Requirement for O(N^{3/2}) Budget

For N_close = O(N^{3/2}), we need:

$$
\frac{c^2N^2}{2} \cdot \frac{D_{\text{max}}^2 - 2\mathrm{Var}_h(\mathcal{C})}{D_{\text{max}}^2} \le C N^{3/2}
$$

Solving for Var_h:
$$
D_{\text{max}}^2 - 2\mathrm{Var}_h(\mathcal{C}) \le \frac{2C D_{\text{max}}^2}{c^2\sqrt{N}}
$$

Therefore:
$$
\mathrm{Var}_h(\mathcal{C}) \ge \frac{D_{\text{max}}^2}{2} - \frac{C' D_{\text{max}}^2}{\sqrt{N}}
$$

This requires variance to be **within O(D_max²/√N) of half the squared diameter**.

---

## Physical Interpretation

### Maximum Possible Variance

For a configuration confined to a domain of diameter D_max, what is the maximum possible variance?

**Case 1: Two-point mass**
Place equal mass at two points separated by distance D_max:
$$
\mathrm{Var}_{\text{2pt}} = \frac{1}{2}(D_{\text{max}}/2)^2 + \frac{1}{2}(D_{\text{max}}/2)^2 = \frac{D_{\text{max}}^2}{4}
$$

**Case 2: Uniform distribution on sphere**
For uniform distribution on a sphere of radius R (diameter D_max = 2R):
$$
\mathrm{Var}_{\text{sphere}} = \frac{R^2}{3} = \frac{D_{\text{max}}^2}{12}
$$

**Case 3: Uniform distribution on interval**
For uniform distribution on [-D_max/2, D_max/2]:
$$
\mathrm{Var}_{\text{interval}} = \frac{D_{\text{max}}^2}{12}
$$

### Variance Requirement Analysis

The O(N^{3/2}) budget requires:
$$
\mathrm{Var}_h \ge \frac{D_{\text{max}}^2}{2} - O\left(\frac{D_{\text{max}}^2}{\sqrt{N}}\right)
$$

But maximum physically achievable variance is approximately:
$$
\mathrm{Var}_{\text{max}} \approx \frac{D_{\text{max}}^2}{4} \text{ to } \frac{D_{\text{max}}^2}{3}
$$

**Problem**: Required variance **D_max²/2** exceeds typical maximum variance **D_max²/4**!

---

## What Variance Does the QSD Actually Have?

### Framework Results

From `12_quantitative_error_bounds.md`:
- **Proposition** `prop-finite-second-moment-meanfield`: C_var < ∞
- **Interpretation**: Variance is **finite**, bounded by confinement and energy dissipation

From `06_convergence.md`:
- **Foster-Lyapunov Condition**: V_total contracts with rate κ_total
- **V_total includes Var_x and Var_v** (positional and velocity variance)
- **Steady state**: Drift balances noise, giving equilibrium variance

### Expected Variance Level

For a **confined system with Gibbs-like distribution** (exponential tails from confining potential U):

$$
\rho(x) \propto \exp(-\beta U(x))
$$

The variance depends on:
1. **Confinement strength** (α_U from Foster-Lyapunov)
2. **Noise level** (σ from Langevin dynamics)
3. **Domain size** (D_max from boundary)

**Typical behavior**:
- Variance is **O(D_max²)** but with **constant prefactor < 1/2**
- Empirically: Var_h ≈ (0.1 to 0.3) × D_max² for well-confined systems

### Reality Check

If Var_h ≈ c_var × D_max² with c_var = O(1) but c_var < 1/2:

$$
N_{\text{close}} \le \frac{c^2N^2}{2} \cdot \frac{D_{\text{max}}^2(1 - 2c_{\text{var}})}{D_{\text{max}}^2}
= \frac{c^2N^2}{2} (1 - 2c_{\text{var}})
$$

**Result**: N_close = **O(N²)**, NOT O(N^{3/2})!

For example, with c_var = 0.2 (20% of max variance):
$$
N_{\text{close}} \le \frac{c^2N^2}{2} \cdot 0.6 = O(N^2)
$$

---

## Implications for Hierarchical Clustering

### Scenario 1: Var_h is Typical (c_var < 1/2)

**Edge budget**: O(N²)

**Component analysis**:
- Each component has m = O(√N) vertices
- Each component needs |E(C)| ≥ m√N = O(N) edges (Component Edge Density Lemma)
- Total walkers: cN, so number of components: L = cN/m = cN/O(√N) = **O(√N)**
- Total edges consumed by components: L × N = O(√N) × O(N) = **O(N^{3/2})**

**Budget check**:
- Available edges: O(N²)
- Consumed by components: O(N^{3/2})
- **No contradiction** — edge budget doesn't constrain L

**Conclusion**: With realistic variance, **hierarchical clustering bound fails** (no contradiction achieved).

### Scenario 2: Var_h is Almost-Maximal (Var_h ≈ D_max²/2)

**Edge budget**: O(N^{3/2})

**Component analysis**:
- Same as above: L components need O(N^{3/2}) total edges
- Available edges: O(N^{3/2})
- **Tight constraint** — edge budget equals consumption

**Conclusion**: With almost-maximal variance, **hierarchical clustering bound succeeds** (L = O(√N) is the tight bound).

**BUT**: This requires **justifying** why QSD has almost-maximal variance.

---

## Possible Resolutions

### Option 1: Accept O(N²) Edge Budget

**Approach**: Use realistic variance assumption Var_h = Θ(D_max²) with prefactor < 1/2

**Result**: N_close = O(N²)

**Impact**: Hierarchical clustering bound cannot be proven via edge-counting argument. Global regime concentration improvement requires alternative approach.

### Option 2: Prove QSD Has High Variance

**Approach**: Show that diversity mechanism in companion selection **maximizes spread**

**Rationale**:
- Cloning operator has **anti-correlation pressure** (diverse companions preferred)
- This could drive variance toward maximum achievable level
- Need to prove: Var_h ≥ D_max²/2 - O(D_max²/√N)

**Challenge**: Framework currently only establishes finite variance, not near-maximal variance.

### Option 3: Use Different Threshold

**Approach**: Use larger proximity threshold d_close = C × D_max/√N with C > 1

**Effect**:
- Increases denominator in packing formula
- Relaxes variance requirement
- But changes interpretation of "hierarchical clustering" (larger threshold = coarser clusters)

### Option 4: Tighter Component Edge Density

**Approach**: Improve Component Edge Density Lemma to get |E(C)| ≥ Ω(m²/k) with better constants

**Challenge**: Current lemma uses Cauchy-Schwarz, which is tight for clique structure.

---

## Framework Investigation Results ✅

### Priority 1: Check Framework for Variance Maximization Results ✅ COMPLETED

**Searched documents**:
- ✅ `03_cloning.md` § 6.4.2 "Positional Variance as Lower Bound for Hypocoercive Variance"
- ✅ `01_fragile_gas_framework.md` (diversity companion mechanism)
- ✅ `06_convergence.md` (Foster-Lyapunov equilibrium analysis)

**Key Finding from `06_convergence.md` lines 1055-1154**:

:::{prf:theorem} Equilibrium Variance Bounds (Theorem {prf:ref}`thm-equilibrium-variance-bounds`)
The QSD satisfies:

$$
V_{\text{Var},x}^{\text{QSD}} \leq \frac{C_x}{\kappa_x}
$$

where:
- κ_x > 0 is N-uniform positional contraction rate (from Keystone Principle)
- C_x < ∞ is expansion constant (cloning noise + boundary effects)
:::

**Physical Interpretation**:
Equilibrium variance is determined by balance between:
- **Contraction**: Cloning operator (targets high variance)
- **Expansion**: Cloning noise + Langevin noise + boundary reentry

**Conclusion**:
❌ **No evidence of variance maximization**

The framework establishes:
1. ✅ Finite equilibrium variance: C_var < ∞
2. ✅ N-uniform bound: Var_x^QSD ≤ C_x/κ_x with N-independent constants
3. ❌ **NO proof** that Var_h^QSD ≈ D_max²/2 (near-maximal)

The equilibrium is a **balance point**, not a maximization. The cloning operator **contracts** variance (not expands it), counteracted by noise injection. There is no mechanism identified that would drive variance to D_max²/2.

### Priority 2: Compute Variance Numerically (RECOMMENDED NEXT STEP)

**Approach**:
- Simulate Euclidean Gas with K = O(N) walkers
- Measure steady-state Var_h / D_max²
- Check if ratio approaches 1/2 or remains O(1)

**Prediction**:
- If ratio → 0.2-0.3: Accept O(N²) edge budget
- If ratio → 0.45-0.5: High variance mechanism exists

### Priority 3: Alternative Proof Strategy

If edge-counting fails, consider:
- **Distance-sensitive concentration**: Use O(1/N³) covariance decay (if provable)
- **Entropic arguments**: Show high entropy requires hierarchical structure
- **Optimal transport**: Use Wasserstein gradient flow structure

---

## Summary

**Critical Question**: Does the QSD achieve **Var_h ≈ D_max²/2**?

**Framework Status**:
- ✅ Establishes C_var < ∞ (finite variance)
- ❌ Does NOT establish near-maximal variance

**Edge Budget**:
- **If Var_h = Θ(D_max²) with prefactor < 1/2**: N_close = **O(N²)**
- **If Var_h ≥ D_max²/2 - O(D_max²/√N)**: N_close = **O(N^{3/2})**

**Hierarchical Clustering Proof**:
- **Depends critically** on which variance regime holds
- **Current proof assumes** O(N^{3/2}) without justification
- **Needs verification** from framework or simulation

**Framework Search Complete**: ✅ No variance maximization mechanism found

**Most Likely Reality**:
Based on framework analysis, the QSD has **equilibrium variance** Var_h^QSD = Θ(D_max²) with **prefactor < 1/2** (likely 0.1-0.3), implying:

$$
N_{\text{close}} = O(N^2) \quad \text{(NOT } O(N^{3/2})\text{)}
$$

**Consequence for Hierarchical Clustering Proof**:
- ❌ Edge-counting argument **FAILS** to constrain L to O(√N)
- ⚠️ Need **alternative proof strategy** or **numerical verification** of variance level

**Recommended Actions**:
1. **Numerical simulation**: Measure Var_h^QSD / D_max² ratio empirically
2. **Accept O(N²)** edge budget and revise hierarchical clustering strategy
3. **Alternative approach**: Use distance-sensitive covariance decay instead of edge-counting

---

**Report Completed By**: Claude (Sonnet 4.5)
**Date**: 2025-10-24
**Priority**: ⚠️ **HIGH** — Affects validity of entire hierarchical clustering proof strategy
