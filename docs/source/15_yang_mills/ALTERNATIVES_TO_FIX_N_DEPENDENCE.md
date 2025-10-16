# Alternatives to Fix N-Dependence in Viscous Coupling

**Date:** October 2025

**Problem:** The viscous coupling $\nu \sum_{j \neq i} K(x_i - x_j)(v_j - v_i)$ forms a graph Laplacian with eigenvalues that can scale with N, breaking the N-uniform Poincaré proof.

**Source:** Codex independent review and technical assessment

---

## Executive Summary

**Recommended Solution:** Use **normalized viscous coupling** (Approach #6)

**Key Insight:** Replace unnormalized coupling with row-normalized weights:

$$
\mathbf{F}_{\text{viscous},i} = \nu \sum_{j \neq i} \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i)
$$

This produces a normalized graph Laplacian with **eigenvalues bounded in [0,2]**, independent of N.

---

## Detailed Analysis of Alternatives

### Option 1: Mean-Field Scaling (ν → ν/N) ✅ VIABLE

**Modification:**
$$
\mathbf{F}_{\text{viscous},i} = \frac{\nu}{N} \sum_{j \neq i} K(x_i - x_j) (v_j - v_i)
$$

**Mathematical Justification:**
- Walkers have mass 1/N, drift becomes Riemann sum for continuum integral $\int K(x-y)(v(y)-v(x))\rho(y)dy$
- Dirichlet form: $\frac{\nu}{2N} \sum_{i,j} K(x_i - x_j) |v_i - v_j|^2$
- Eigenvalues: $\leq \text{const} \cdot \nu \sup K$ (N-independent)

**Physical Interpretation:**
- If ν encodes **material viscosity** (bulk property), then ν/N is correct for particle approximation
- Matches mean-field theory in `05_mean_field.md`
- Preserves dissipative structure
- Enables propagation of chaos estimates

**Required Assumptions:**
- Bounded kernel: $\|K\|_{L^\infty} < \infty$
- Quasi-uniform positions (already assumed in framework)

**Pros:**
- ✅ Clean N-uniform bounds
- ✅ Standard in particle approximations
- ✅ Physically justified (if ν = material viscosity)
- ✅ Matches mean-field literature

**Cons:**
- ⚠️ Changes physical interpretation (ν now bulk viscosity, not pairwise strength)
- ⚠️ Requires updating framework definition

**Codex Assessment:**
> "Mathematically standard for particle approximations of non-local viscous stresses. Physically justified if ν encodes material viscosity, not pairwise strength."

**Verdict:** ✅ **RECOMMENDED** if we interpret ν as material viscosity

---

### Option 2: Spectral Bounds on Unnormalized Laplacian ❌ NOT VIABLE

**Approach:** Prove eigenvalues of $L = D - W$ (unnormalized Laplacian) are O(1) for specific kernels.

**Mathematical Reality:**
- For fixed-support kernels: $\lambda_{\max} \leq 2 \cdot \deg_{\max}$
- If kernel has range $r$ independent of N, then $\deg_{\max} \sim N$ (all walkers in range)
- Eigenvalues scale with N unless:
  - Kernel radius shrinks like $N^{-1/3}$ (making $\deg_{\max} = O(1)$), OR
  - Kernel has alternating sign (unphysical)

**Required Assumptions:**
- Kernel with special geometric properties (non-standard, not in current framework)

**Codex Assessment:**
> "Without rescaling, λ_max ≤ 2·deg_max, so for fixed-support kernels deg_max ~ N and eigenvalues blow up. One can recover O(1) bounds only under strong geometric assumptions... These conditions are non-standard and absent from the current framework."

**Verdict:** ❌ **NOT VIABLE** without unrealistic assumptions

---

### Option 3: Weak Coupling (Small ν) ❌ INSUFFICIENT

**Approach:** Take ν small enough that even O(N) perturbation is manageable.

**Mathematical Reality:**
- Need $\nu \cdot \deg_{\max} = O(1)$ to control Hessian norm
- Since $\deg_{\max} \sim N$, requires $\nu = O(1/N)$
- This collapses the viscous effect to negligible

**Codex Assessment:**
> "You need ν·deg_max = O(1)... Since deg_max ∼ N, ν must scale like 1/N, collapsing the viscous effect. Unless the backbone already provides a uniform spectral gap that dominates an O(N) perturbation (which it does not), this does not repair the proof."

**Verdict:** ❌ **INSUFFICIENT** - forces ν → 0 as N → ∞

---

### Option 4: Conditional Poincaré (Fix Positions) 🟡 PARTIAL

**Approach:** Fix positions $x_1, \ldots, x_N$, prove velocity Poincaré conditionally, then integrate over position distribution.

**Mathematical Reality:**
- Reduces to bounding $\|\gamma I + \nu L(x)\|$ for each configuration
- Still need $\|L(x)\|$ uniformly bounded
- With quasi-uniform positions + shrinking kernel support: $\deg_{\max} = O(\log N)$ (high probability)
- Gives **polylog growth**, not strict N-independence

**Codex Assessment:**
> "Conditioning reduces the problem to bounding the spectrum of γI + νL(x) for each configuration. You still need ‖L(x)‖ uniformly bounded... This helps if you accept logarithmic constants, but it does not yield strict N-independence."

**Verdict:** 🟡 **PARTIAL** - gives $C_P = O(\log N)$, not O(1)

---

### Option 5: Dense Graph Cheeger Bounds ❌ WRONG TARGET

**Approach:** Use Cheeger inequality to bound eigenvalues.

**Mathematical Reality:**
- Cheeger estimates control **smallest positive eigenvalue** (spectral gap)
- Does NOT cap **largest eigenvalue**
- Doesn't solve the current issue unless you first normalize

**Codex Assessment:**
> "Cheeger estimates control the smallest positive eigenvalue of the normalized Laplacian. They do not cap the largest eigenvalue of the unnormalized operator."

**Verdict:** ❌ **WRONG TARGET** - addresses different eigenvalue

---

### Option 6: Row-Normalized Coupling ✅✅ BEST OPTION

**Modification:**
$$
\mathbf{F}_{\text{viscous},i} = \nu \sum_{j \neq i} \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i)
$$

Define normalized weights:
$$
a_{ij} = \frac{K(x_i - x_j)}{\deg(i)}, \quad \deg(i) = \sum_k K(x_i - x_k)
$$

**Mathematical Properties:**
- Produces **symmetric normalized Laplacian**: $L_{\text{norm}} = I - D^{-1/2} W D^{-1/2}$
- **Eigenvalues in [0, 2]**, independent of N
- Operator norm: $\|\nu L_{\text{norm}}\| \leq 2\nu$ (N-independent!)

**Poincaré Inequality:**
- Conditional Gibbs density for velocities: Gaussian with precision $\gamma I + \nu L_{\text{norm}}$
- Matrix Poincaré (Bakry-Émery): $\text{Var}(f) \leq (\gamma + 2\nu)^{-1} \mathbb{E}[|\nabla_v f|^2]$
- **Uniform in N** ✅

**Energy Dissipation:**
$$
\mathcal{D}_{\text{viscous}} = \nu \sum_i \frac{1}{\deg(i)} \sum_j K(x_i - x_j) |v_i - v_j|^2
$$

- Matches continuum stress-tensor form (up to SPH mass factors)
- Preserves dissipative structure

**Required Assumptions:**
- Kernel K non-negative, symmetric, bounded below on a ball (every walker has neighbors)
- Quasi-uniform positions (already in framework)
- $\deg(i) \geq \kappa > 0$ for all i (follows from kernel positivity + spatial confinement)

**Codex Assessment:**
> "Produces the symmetric normalized Laplacian L_norm = I - D^{-1/2}WD^{-1/2} with spectrum in [0,2], independent of N... This keeps the quadratic form comparable to the original energy dissipation while making the drift coefficients uniformly bounded."

**Pros:**
- ✅✅ **Eigenvalues bounded in [0,2]** independent of N
- ✅ Preserves dissipative structure
- ✅ Matches hydrodynamics (stress tensor form)
- ✅ Clean N-uniform Poincaré constant: $C_P = (\gamma + 2\nu)^{-1}$
- ✅ No physical reinterpretation needed
- ✅ Well-defined under standard assumptions

**Cons:**
- ⚠️ Requires updating framework definition (`def-velocity-modulated-viscous-force`)
- ⚠️ Normalization makes coupling **configuration-dependent** (but this is already true for $\Sigma_{\text{reg}}$)

**Verdict:** ✅✅ **BEST OPTION** - clean, rigorous, N-uniform

---

## Comparison Table

| Approach | N-Uniform Bounds | Physics Preserved | Framework Changes | Difficulty | Recommendation |
|:---------|:-----------------|:------------------|:------------------|:-----------|:---------------|
| 1. Mean-field (ν/N) | ✅ Yes | 🟡 Reinterpret ν | Moderate | Easy | ✅ Good option |
| 2. Spectral bounds | ❌ No | ✅ Yes | None | Impossible | ❌ Not viable |
| 3. Weak coupling | ❌ No (forces ν→0) | ❌ Kills effect | None | N/A | ❌ Insufficient |
| 4. Conditional | 🟡 O(log N) | ✅ Yes | None | Moderate | 🟡 Partial |
| 5. Cheeger | ❌ Wrong target | ✅ Yes | None | N/A | ❌ Not applicable |
| 6. Normalized | ✅✅ Yes | ✅ Yes | Moderate | Easy | ✅✅ **BEST** |

---

## Recommended Implementation Plan

### Choice: **Option 6 (Normalized Coupling)** with optional **Option 1 (Mean-field scaling)** for physical clarity

**Step 1: Update Viscous Force Definition**

**Current (framework):**
$$
\mathbf{F}_{\text{viscous},i} = \nu \sum_{j \neq i} K(x_i - x_j) (v_j - v_i)
$$

**Proposed (normalized):**
$$
\mathbf{F}_{\text{viscous},i} = \nu \sum_{j \neq i} \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i)
$$

**Optional (normalized + mean-field):**
$$
\mathbf{F}_{\text{viscous},i} = \frac{\nu}{N} \sum_{j \neq i} \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i)
$$

**Step 2: Update LSI Proof (Section 7.3)**

**New Poincaré Proof:**

1. **Uncoupled system**: Same as before (Gaussian velocities, $C_P \leq c_{\max}^2(\rho)/\gamma$)

2. **Normalized viscous coupling**:
   - Drift matrix: $\gamma I + \nu L_{\text{norm}}$
   - Eigenvalues: $[\gamma, \gamma + 2\nu]$ (N-independent)
   - Matrix Poincaré: $C_P^{\text{coupled}} = (\gamma + 2\nu)^{-1} c_{\max}^2(\rho)$

3. **N-uniformity**:
   - All quantities (γ, ν, c_max(ρ)) N-independent
   - **Poincaré constant N-uniform** ✅

**Step 3: Verify Framework Consistency**

- Check `def-velocity-modulated-viscous-force` in `07_adaptative_gas.md`
- Update energy dissipation lemma (`lem-viscous-dissipative`)
- Verify Foster-Lyapunov analysis still works with normalized coupling
- Update physical interpretation section

**Step 4: Documentation**

- Add remark comparing normalized vs. unnormalized coupling
- Explain physical interpretation (normalized = local fluid coupling)
- Note that normalization preserves dissipative structure
- Reference continuum hydrodynamics analogy

---

## Physical Interpretation

### Why Normalization Makes Sense:

**Unnormalized coupling:** $\nu \sum_j K(x_i - x_j)(v_j - v_i)$
- Interpretation: Pairwise viscous forces
- Issue: Total force on walker i grows with number of neighbors (N-dependent)

**Normalized coupling:** $\nu \sum_j \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i)$
- Interpretation: **Weighted average** of velocity differences
- Each neighbor contributes proportionally to its "visibility" weight
- Total coupling strength normalized by local "density" of neighbors
- Analogous to **SPH (Smoothed Particle Hydrodynamics)** kernels

**Physical analogy:**
- In a dense crowd, each person experiences viscous drag from neighbors
- Normalized: drag force is an **average** over visible neighbors (bounded)
- Unnormalized: drag force **sums** over all neighbors (grows with crowd size)

**Continuum limit:**
$$
\nu \sum_j \frac{K(x_i - x_j)}{\sum_k K(x_i - x_k)} (v_j - v_i) \approx \nu \int K_{\rho}(x - y) (v(y) - v(x)) dy
$$

where $K_{\rho}(x-y) = K(x-y)/\int K(x-z)dz$ is a normalized kernel.

---

## Next Steps

1. **Consult user**: Which option to implement?
   - Pure normalized (Option 6)?
   - Normalized + mean-field scaling (Option 6 + 1)?

2. **Update framework documents**:
   - `07_adaptative_gas.md` (definition)
   - `adaptive_gas_lsi_proof.md` (Section 7.3)
   - Energy dissipation lemmas

3. **Re-run Codex review** on updated proof

4. **If successful**: Update conjecture status, remove Yang-Mills disclaimers

---

## Conclusion

**The N-dependence issue CAN be fixed** using normalized viscous coupling.

**Key insight:** The problem wasn't with the Poincaré proof technique, but with the **definition of the viscous force**. Normalization makes the coupling **intrinsically N-uniform** while preserving physics.

**Recommendation:** Implement Option 6 (normalized coupling), optionally combined with Option 1 (mean-field scaling) for consistency with continuum hydrodynamics.

This is a **solvable problem** with a clean mathematical and physical solution.
