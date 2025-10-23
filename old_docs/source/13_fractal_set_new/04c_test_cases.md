# Test Cases: Determining the Correct Interpretation

**Purpose:** Concrete computational experiments to determine whether the proposed symmetry structure is:
1. Local gauge theory
2. Mean-field collective theory
3. Somewhere in between (crossover)

**Date:** 2025-10-23

---

## Overview

The correct interpretation depends critically on the **locality parameters** (ρ, ε_d, ε_c). We design three test cases spanning the parameter space:

| Test Case | ρ | ε_d, ε_c | Expected Interpretation |
|-----------|---|----------|------------------------|
| **1. Ultra-Local** | 0.01 | 0.01 | Local field theory |
| **2. Mean-Field** | ∞ | ∞ | Auxiliary collective fields |
| **3. Crossover** | 0.01 → ∞ | 0.01 → ∞ | Study transition |

---

## Test Case 1: Ultra-Local Regime

**Goal:** Test if collective fields behave as LOCAL fields and whether gauge covariance exists.

### 1.1. Setup

**System:**
- N = 1000 walkers
- d = 2 dimensions
- Domain: [0, 1] × [0, 1] (unit square)

**Parameters:**
- ρ = 0.01 (localization scale)
- ε_d = 0.01 (diversity companion range)
- ε_c = 0.01 (cloning companion range)
- Average neighbor distance: ⟨d_alg⟩ ≈ 0.03

**Expected:** Only ~5-10 nearest neighbors contribute to statistics.

### 1.2. Test A: Spatial Correlation Function

**Purpose:** Measure how quickly d'_i correlation decays with distance.

**Procedure:**
1. Run algorithm for 100 timesteps to reach quasi-equilibrium
2. For each timestep t, compute:
   ```
   C(r) = ⟨d'_i(t) · d'_j(t)⟩ for all pairs with |x_i - x_j| ≈ r
   ```
3. Bin by distance r ∈ [0, 0.5] with spacing Δr = 0.01
4. Plot C(r) vs r

**Expected result (local field theory):**
- Exponential decay: C(r) ~ C_0 · exp(-r²/ξ²)
- Correlation length: ξ ≈ ρ = 0.01
- Fit and extract ξ

**Expected result (mean-field):**
- Flat correlation: C(r) ≈ const (would contradict ρ = 0.01 setup)

**Interpretation:**
- If ξ ≈ ρ → Confirms local field structure ✓
- If ξ >> ρ → Unexpected, needs investigation

### 1.3. Test B: Field Gradient Magnitude

**Purpose:** Measure how strongly d'(x) varies spatially.

**Procedure:**
1. At quasi-equilibrium, compute spatial gradient:
   ```
   ∇d'(x_i) ≈ (d'_j - d'_i) / |x_j - x_i| for nearest neighbor j
   ```
2. Histogram |∇d'| across all walkers
3. Compute mean |∇d'|_avg and std

**Expected result (local field theory):**
- Strong gradients: |∇d'|_avg ~ O(1/ρ) ~ 100
- Significant variance (field is non-uniform)

**Expected result (mean-field):**
- Weak gradients: |∇d'|_avg ≈ 0 (field is uniform)

**Interpretation:**
- If |∇d'| ~ 1/ρ → Local field with structure ✓
- If |∇d'| ≈ 0 → Mean-field (shouldn't happen with ρ = 0.01)

### 1.4. Test C: Perturbation Response (Locality Test)

**Purpose:** Measure how far the influence of a single walker extends.

**Procedure:**
1. Record baseline configuration {d'_i}
2. Artificially perturb single walker k:
   - Change x_k → x_k + δx (small displacement)
   - Recompute only local statistics μ_ρ(i) for i near k
   - Recompute d'_i for affected walkers
3. Measure change: Δd'_i = |d'_i(perturbed) - d'_i(baseline)|
4. Plot Δd'_i vs distance r_ik = |x_i - x_k|

**Expected result (local field theory):**
- Exponential decay: Δd'_i(r) ~ A · exp(-r²/ρ²)
- Response localized within ~3ρ ≈ 0.03

**Expected result (mean-field):**
- All walkers affected equally: Δd'_i ≈ const (uniform response)

**Interpretation:**
- If Δd'(r) ~ exp(-r²/ρ²) → Local field ✓
- If Δd' uniform → Mean-field (contradicts ρ = 0.01)

### 1.5. Test D: Gauge Covariance (Critical Test)

**Purpose:** Determine if d'_i transforms non-trivially under local gauge transformation.

**Procedure:**

**Step 1:** Define local gauge transformation on subset of walkers.
- Select region R: walkers with x ∈ [0.4, 0.6] × [0.4, 0.6] (center square)
- Apply phase shift: α_i = α_0 for i ∈ R, α_i = 0 otherwise
- Modify companion selection probabilities:
  ```
  P'_comp(k|i) ∝ P_comp(k|i) · exp(i(α_i - α_k) / ℏ_eff)
  ```
  (phase-dependent weighting)

**Step 2:** Recompute collective fields with transformed probabilities.
- Sample new companions c'_div(i) from P'_comp
- Compute distances d'_i using transformed companions
- Compute statistics μ'_ρ(i), σ'_ρ(i) with transformed neighborhood
- Compute d'_i with transformed statistics

**Step 3:** Measure change in d'_i for walkers inside and outside R.
- Inside R: Δd'_in = ⟨|d'_i - d'_i,baseline|⟩ for i ∈ R
- Boundary: Δd'_bd = ⟨|d'_i - d'_i,baseline|⟩ for i within distance ρ of ∂R
- Outside R: Δd'_out = ⟨|d'_i - d'_i,baseline|⟩ for i far from R

**Expected result (gauge covariant):**
- Inside: Δd'_in ~ O(α_0) (compensates for phase shift)
- Boundary: Δd'_bd ~ O(α_0) (gradient at boundary)
- Outside: Δd'_out ≈ 0 (no response beyond ~ρ)

**Expected result (gauge invariant):**
- Everywhere: Δd' ≈ 0 (no response to gauge transformation)

**Interpretation:**
- If Δd' ~ O(α) locally → **GAUGE COVARIANT** ✓✓✓
- If Δd' ≈ 0 everywhere → **GAUGE INVARIANT** (local gauge theory fails)

**This is the critical test!**

### 1.6. Test E: Wave Excitations in d'(x) Field

**Purpose:** Check if d'(x) supports wave-like collective excitations (expected in local field theory).

**Procedure:**
1. Introduce localized perturbation in fitness:
   - Add temporary Gaussian potential: U_pert(x) = A · exp(-|x - x_0|²/σ²)
   - x_0 = (0.5, 0.5), σ = 0.05, A = 1
2. Apply perturbation for 5 timesteps, then remove
3. Record d'(x,t) field evolution for 50 timesteps after removal
4. Compute Fourier transform: d'(k,ω)

**Expected result (local field theory):**
- Propagating waves in d'(x,t) emanating from x_0
- Dispersion relation: ω(k) (frequency vs wave-vector)
- Characteristic velocity: v ≈ √(∂²V_eff/∂d'²) · ρ

**Expected result (mean-field):**
- Uniform relaxation (no propagation)
- Exponential decay: d'(x,t) ~ e^(-t/τ_relax)

**Interpretation:**
- If waves propagate → Local field theory with dynamics ✓
- If uniform decay → Mean-field

### 1.7. Success Criteria (Test Case 1)

**For LOCAL FIELD THEORY interpretation to be valid:**
- [ ] Correlation length ξ ≈ ρ (Test A)
- [ ] Field gradient |∇d'| ~ 1/ρ (Test B)
- [ ] Perturbation response localized within ~3ρ (Test C)
- [ ] **Gauge covariance: Δd' ~ O(α) under local transformation (Test D)**
- [ ] Wave excitations propagate (Test E)

**If all pass:** Interpret as **local gauge field theory** ✅

**If Test D fails but others pass:** Interpret as **local non-gauge field theory** (still interesting!)

---

## Test Case 2: Mean-Field Regime

**Goal:** Verify mean-field interpretation applies when ρ, ε → ∞.

### 2.1. Setup

**System:** Same as Test Case 1

**Parameters:**
- ρ = ∞ (global statistics)
- ε_d = ∞ (all companions equally likely)
- ε_c = ∞ (global cloning)

**Implementation:**
- μ_ρ(i) = μ_global = (1/N) Σ_j d_j (same for all i)
- σ_ρ(i) = σ_global (same for all i)
- P_comp(k|i) = 1/(N-1) (uniform)

**Expected:** All walkers contribute equally to statistics.

### 2.2. Tests (Same as Test Case 1)

Run Tests A-E with ρ = ∞.

**Expected results:**
- **Test A:** C(r) ≈ const (no spatial correlation)
- **Test B:** |∇d'| ≈ 0 (spatially uniform field)
- **Test C:** Δd'(r) ≈ const (global response to perturbation)
- **Test D:** Δd' ≈ 0 (gauge invariant, transformation has no effect)
- **Test E:** No waves, uniform relaxation

### 2.3. Additional Test F: Mean-Field Self-Consistency

**Purpose:** Verify d'_i satisfies mean-field equation.

**Procedure:**
1. Record swarm state {x_i, v_i}
2. Compute global statistics: μ_d, σ_d
3. Predict d'_i from mean-field formula:
   ```
   d'_i = g_A((d_i - μ_d) / σ_d) + η
   ```
4. Run algorithm forward one step
5. Measure actual d'_i(t+1) from algorithm
6. Compare predicted vs actual

**Expected result:**
- High correlation: corr(predicted, actual) > 0.95
- Mean-field approximation is accurate

### 2.4. Success Criteria (Test Case 2)

**For MEAN-FIELD interpretation to be valid:**
- [ ] No spatial correlations (Test A)
- [ ] Spatially uniform field (Test B)
- [ ] Global response to perturbations (Test C)
- [ ] Gauge invariant (Test D fails)
- [ ] No wave propagation (Test E)
- [ ] Mean-field self-consistency (Test F)

**If all pass:** Interpret as **auxiliary mean-field theory** ✅

---

## Test Case 3: Crossover Regime

**Goal:** Study transition from local to mean-field as ρ increases.

### 3.1. Setup

**System:** Same as Test Cases 1-2

**Parameter scan:**
- ρ ∈ [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, ∞]
- ε_d = ε_c = ρ (scale together)
- Run Tests A-E for each value of ρ

### 3.2. Observables to Plot vs ρ

1. **Correlation length ξ(ρ)** (from Test A fit)
   - Expected: ξ ≈ ρ for small ρ, then ξ → ∞ as ρ → ∞

2. **Field gradient |∇d'|(ρ)** (from Test B)
   - Expected: |∇d'| ~ 1/ρ for small ρ, then |∇d'| → 0 as ρ → ∞

3. **Perturbation response range R_resp(ρ)** (from Test C)
   - Define: Distance where Δd'(r) drops to 1/e of peak
   - Expected: R_resp ≈ 3ρ for small ρ, then R_resp → ∞ as ρ → ∞

4. **Gauge response amplitude Δd'_gauge(ρ)** (from Test D)
   - Expected: Δd' ~ O(α) for small ρ (if gauge covariant), then Δd' → 0 as ρ → ∞

5. **Wave velocity v_wave(ρ)** (from Test E)
   - Expected: v_wave ~ ρ · c_0 for small ρ, then v_wave → ∞ (instantaneous) as ρ → ∞

### 3.3. Identify Critical Scale

**Definition:** ρ_c where transition from local to mean-field occurs.

**Methods:**
1. **From correlation length:** ρ_c = value where ξ(ρ) starts growing faster than ρ
2. **From field gradient:** ρ_c = value where |∇d'| · ρ starts decreasing
3. **From gauge response:** ρ_c = value where Δd'_gauge starts dropping

**Hypothesis:** ρ_c ~ average neighbor distance ⟨d_alg⟩

**Interpretation:**
- ρ < ρ_c: Local field regime (few neighbors contribute)
- ρ > ρ_c: Mean-field regime (many neighbors contribute)
- ρ ≈ ρ_c: Crossover (most interesting physics!)

### 3.4. Phase Diagram

Create 2D phase diagram in (ρ, ε) space:

```
         ε (companion range)
         ^
    ∞    |     MEAN-FIELD
         |      (global)
         |
    ε_c  |------------------
         |       |
         |  CROSSOVER
         |       |
         |------------------
    0    | LOCAL FIELD
         |  (few neighbors)
         +------|----------|---> ρ (statistics range)
                0        ρ_c    ∞
```

**Regions:**
1. **Local field** (ρ, ε < ρ_c): Gauge theory viable
2. **Mean-field** (ρ, ε > ρ_c): Collective theory applies
3. **Crossover** (ρ ~ ρ_c or ε ~ ρ_c): Mixed physics

### 3.5. Success Criteria (Test Case 3)

**For CROSSOVER interpretation to be interesting:**
- [ ] Smooth transition observed (not sharp phase transition)
- [ ] Critical scale ρ_c identified
- [ ] Observables scale with ρ in predicted ways
- [ ] Physical mechanisms understood (how locality emerges/disappears)

---

## Test Case 4: Algorithmic Fitness Comparison

**Goal:** Compare proposed structure (collective field phases) with current structure (raw distance phases).

### 4.1. Setup

**Two implementations:**
1. **Current:** Phase θ_ik = -d_alg²(i,k) / (2ε²ℏ)
2. **Proposed:** Phase θ_i = (d'_i)^β / ℏ

**Benchmark problems:**
- Rastrigin function (many local minima)
- Rosenbrock function (narrow valley)
- Sphere function (simple convex)
- Atari game (complex, high-dimensional)

**Metrics:**
- Convergence rate (timesteps to reach 90% of optimum)
- Final fitness achieved
- Sample efficiency (evaluations needed)
- Computational cost (time per iteration)

### 4.2. Hypothesis

**Proposed structure should be better when:**
- Problem has hierarchical structure (local + global features)
- Algorithm operates in local regime (small ρ)
- Processed information more informative than raw distances

**Current structure should be better when:**
- Problem is simple (convex)
- Algorithm operates in mean-field regime (large ρ)
- Direct geometric distance is most relevant

### 4.3. Success Criteria

**For proposed structure to be "better":**
- [ ] Converges faster on at least 2/4 benchmarks
- [ ] Achieves equal or better final fitness
- [ ] No significant computational overhead

**If comparable:** Both valid, choice is interpretational preference.

---

## Implementation Roadmap

### Week 1: Infrastructure

- [ ] Implement ρ-localized statistics (μ_ρ(i), σ_ρ(i))
- [ ] Implement collective field computation (d'_i, r'_i with local stats)
- [ ] Verify against test case (small example: N=10)

### Week 2: Test Case 1 (Ultra-Local)

- [ ] Tests A-C (correlation, gradient, perturbation)
- [ ] Analyze results, verify local field structure
- [ ] **Test D (gauge covariance)** - critical!

### Week 3: Test Cases 2-3 (Mean-Field & Crossover)

- [ ] Test Case 2 (ρ = ∞)
- [ ] Test Case 3 (scan ρ)
- [ ] Identify ρ_c, create phase diagram

### Week 4: Analysis & Documentation

- [ ] Determine correct interpretation from results
- [ ] Test Case 4 (benchmark comparison)
- [ ] Write results document

---

## Expected Outcomes

### Scenario A: Gauge Covariance Proven (Test 1D passes)

**Result:** d'_i transforms non-trivially under local gauge transformation

**Interpretation:** Local gauge field theory in small ρ regime ✅

**Next steps:**
- Derive gauge connection A_μ from d'_i
- Construct Yang-Mills action
- Compute gauge boson spectrum
- Publish in mathematical physics journal

**Impact:** Strong connection to Standard Model, novel gauge theory

### Scenario B: Gauge Invariance Confirmed (Test 1D fails)

**Result:** d'_i remains invariant under all tested transformations

**Interpretation:** NOT a local gauge theory; mean-field or non-gauge field theory

**Next steps:**
- Develop mean-field formalism
- Find condensed matter analogs
- Study collective modes
- Publish in interdisciplinary journal

**Impact:** Still interesting (emergent collective fields), weaker SM connection

### Scenario C: Regime-Dependent (Test 3 reveals transition)

**Result:** Gauge covariance appears/disappears at critical ρ_c

**Interpretation:** Emergent gauge structure

**Next steps:**
- Study emergence mechanism
- Understand how locality generates gauge structure
- Most novel physics direction
- Publish in high-impact venue

**Impact:** Highest novelty, could explain how gauge theories emerge

---

## Deliverables

1. **Data:** Correlation functions, field configurations, phase diagrams
2. **Analysis:** Fits, scaling exponents, critical points
3. **Visualization:** Heatmaps of d'(x,t), movies of wave propagation
4. **Report:** Which interpretation is correct and why
5. **Code:** Reusable testing framework for future studies

---

## Summary

**Three test cases probe different regimes:**
- **Test 1:** Local regime - test gauge covariance
- **Test 2:** Mean-field regime - verify auxiliary field interpretation
- **Test 3:** Crossover - study emergence of locality

**Critical test:** Test 1D (gauge covariance)
- **If passes:** Local gauge theory ✅✅✅
- **If fails:** Mean-field or non-gauge theory ✅

**Timeline:** 4 weeks to definitive answer

**Risk:** Low (all interpretations are viable, just different frameworks)

**Reward:** High (determine correct physics interpretation, potentially discover emergent gauge structure)

---

**End of Test Cases Document**
