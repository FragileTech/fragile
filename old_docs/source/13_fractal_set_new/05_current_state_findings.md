# Current State Findings: Locality Parameters in Fragile Gas Implementation

**Date:** 2025-10-23
**Purpose:** Document current implementation status and identify gaps

---

## Key Finding: Current Code Operates in Mean-Field Regime

### Critical Discovery

The current Fragile Gas implementation in `src/fragile/core/fitness.py` **does NOT implement ρ-localized statistics**.

**Evidence:**

```python
# From src/fragile/core/fitness.py, lines 76-78:
if rho is not None:
    msg = "Localized standardization (finite rho) not yet implemented"
    raise NotImplementedError(msg)
```

**Implication:** The current code operates with **ρ → ∞** (global statistics), meaning it's in the **MEAN-FIELD REGIME** by default.

---

## Current Implementation Details

### 1. Statistics Computation (fitness.py)

**Function:** `patched_standardization(values, alive, rho=None, ...)`

**Current behavior:**
```python
# Compute masked mean over ALL alive walkers
mu = (values * alive_mask).sum() / n_alive_safe

# Compute masked variance over ALL alive walkers
sigma_sq = ((centered**2) * alive_mask).sum() / n_alive_safe
```

**What's implemented:**
- ✅ Global mean: μ_global = (1/k) Σ_{i ∈ alive} v_i
- ✅ Global std: σ_global = sqrt((1/k) Σ_i (v_i - μ_global)²)
- ✅ Patching for dead walkers (excluded from statistics)
- ✅ Regularization: σ' = sqrt(σ² + σ_min²)

**What's missing:**
- ❌ Localization kernel: K_ρ(i,j) = exp(-d_alg²(i,j)/(2ρ²))
- ❌ Local mean: μ_ρ(i) = Σ_j K_ρ(i,j) v_j / Σ_j K_ρ(i,j)
- ❌ Local std: σ_ρ(i) based on ρ-neighborhood

### 2. Companion Selection (companion_selection.py)

**Methods available:**
- `"uniform"`: All alive walkers equally likely (ε → ∞)
- `"distance"`: Weighted by algorithmic distance with parameter ε
- `"fitness"`: Weighted by fitness difference

**Default in tests:** `method="uniform"` (mean-field)

### 3. Typical Parameter Values

**From test fixtures (conftest.py):**
- N = 10 (small swarm for tests)
- d = 2 (2D problems)
- sigma_x = 0.1 (cloning noise)
- lambda_alg = 0.0 (no velocity weighting)
- epsilon_clone = 0.01 (regularization)
- Companion selection: uniform (ε → ∞)

**Regime analysis:**
- Companion selection: GLOBAL (uniform → ε = ∞)
- Statistics: GLOBAL (rho → ∞)
- **Verdict: MEAN-FIELD REGIME**

---

## Implications for Symmetry Redefinition Analysis

### 1. Current Collective Fields Are Global

Since ρ → ∞ in current code:

```python
# Current behavior:
d'_i = g_A((d_i - μ_global) / σ_global) + η

# Where μ_global, σ_global same for ALL walkers
```

**Properties:**
- d'_i varies mainly through d_i (individual distance)
- Small variations from global statistics
- Essentially mean-field variables

### 2. Gemini's Original Analysis Was Correct (For Current Code)

**Gemini's argument:**
> "Collective fields are gauge-invariant because built from gauge-invariant primitives (global statistics)."

**Status: CORRECT for ρ → ∞**

In the mean-field regime:
- Global statistics μ, σ are gauge-invariant
- Therefore d'_i is gauge-invariant
- Local gauge theory interpretation unlikely

### 3. But Our Corrected Analysis Is Also Valid (For ρ < ∞)

**Our correction:**
> "With ρ-localized statistics, d'_i depends only on ρ-neighborhood, making local gauge theory plausible."

**Status: POTENTIALLY CORRECT for ρ << L**

But this requires:
- Implementing ρ-localized statistics
- Testing with small ρ values
- Verifying gauge covariance experimentally

---

## What Needs To Be Implemented

### Priority 1: ρ-Localized Statistics

**File:** `src/fragile/core/fitness.py`

**Function to modify:** `patched_standardization()`

**Required additions:**

```python
def compute_localization_weights(
    positions: Tensor,  # [N, d]
    velocities: Tensor,  # [N, d]
    alive: Tensor,      # [N] bool
    rho: float,
    lambda_alg: float = 0.0,
) -> Tensor:
    """Compute K_ρ(i,j) localization weights for all pairs.

    Returns:
        weights: [N, N] tensor where weights[i,j] = K_ρ(i,j) * alive[j]
    """
    # Algorithmic distances [N, N]
    dx = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, d]
    dv = velocities.unsqueeze(1) - velocities.unsqueeze(0)  # [N, N, d]

    d_alg_sq = (dx**2).sum(dim=-1) + lambda_alg * (dv**2).sum(dim=-1)  # [N, N]

    # Localization kernel
    K_rho = torch.exp(-d_alg_sq / (2 * rho**2))  # [N, N]

    # Mask dead walkers
    alive_mask_2d = alive.unsqueeze(1) * alive.unsqueeze(0)  # [N, N]
    K_rho = K_rho * alive_mask_2d.float()

    return K_rho


def localized_statistics(
    values: Tensor,     # [N]
    weights: Tensor,    # [N, N] localization weights
    sigma_min: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Compute ρ-localized mean and std for each walker.

    Returns:
        mu_rho: [N] local means
        sigma_rho: [N] local stds
    """
    # Normalize weights for each walker
    weight_sum = weights.sum(dim=1, keepdim=True)  # [N, 1]
    weight_sum_safe = torch.clamp(weight_sum, min=1e-10)
    weights_norm = weights / weight_sum_safe  # [N, N]

    # Local mean: μ_ρ(i) = Σ_j w_ij v_j
    mu_rho = torch.matmul(weights_norm, values)  # [N]

    # Local variance: σ²_ρ(i) = Σ_j w_ij (v_j - μ_ρ(i))²
    centered = values.unsqueeze(0) - mu_rho.unsqueeze(1)  # [N, N]
    sigma_sq_rho = torch.sum(weights_norm * centered**2, dim=1)  # [N]

    # Regularized std
    sigma_rho = torch.sqrt(sigma_sq_rho + sigma_min**2)  # [N]

    return mu_rho, sigma_rho
```

**Modified `patched_standardization`:**

```python
def patched_standardization(
    values: Tensor,
    alive: Tensor,
    positions: Tensor | None = None,  # NEW
    velocities: Tensor | None = None, # NEW
    rho: float | None = None,
    lambda_alg: float = 0.0,  # NEW
    sigma_min: float = 1e-8,
    return_statistics: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Compute Z-scores with optional ρ-localization."""

    if rho is None:
        # Global case (current implementation)
        # ... existing code ...
    else:
        # Localized case (NEW)
        if positions is None or velocities is None:
            raise ValueError("positions and velocities required for localized stats")

        # Compute localization weights
        weights = compute_localization_weights(
            positions, velocities, alive, rho, lambda_alg
        )

        # Compute local statistics
        mu_rho, sigma_rho = localized_statistics(values, weights, sigma_min)

        # Compute Z-scores
        z_scores = (values - mu_rho) / sigma_rho
        z_scores_masked = z_scores * alive.float()

        if return_statistics:
            return z_scores_masked, mu_rho, sigma_rho
        return z_scores_masked
```

### Priority 2: Companion Selection with ε

**File:** `src/fragile/core/companion_selection.py`

**Check if distance-weighted selection already exists:**
- Appears to have `method="distance"` with `epsilon` parameter
- Need to verify it implements: P(j|i) ∝ exp(-d_alg²(i,j)/(2ε²))

### Priority 3: Test Infrastructure

**New test file:** `tests/core/test_locality_regime.py`

**Test functions:**
- `test_global_statistics_when_rho_infinite()`
- `test_local_statistics_when_rho_small()`
- `test_correlation_length_scales_with_rho()`
- `test_perturbation_response_locality()`
- `test_gauge_covariance()` **← CRITICAL**

---

## Experimental Plan

### Phase 1: Validate Mean-Field Regime (Current Code)

**Goal:** Confirm current code behaves as mean-field theory predicts

**Tests:**
1. Run Test Case 2 (mean-field) from 04c_test_cases.md
2. Verify: Correlation length ξ → ∞
3. Verify: Field gradient |∇d'| ≈ 0
4. Verify: Gauge invariance (no response to local transformations)

**Expected result:** Confirms mean-field interpretation ✓

### Phase 2: Implement and Validate Local Regime

**Goal:** Implement ρ-localized statistics and test local field behavior

**Steps:**
1. Implement `compute_localization_weights()` ✓
2. Implement `localized_statistics()` ✓
3. Modify `patched_standardization()` to support `rho` parameter ✓
4. Write unit tests for localized functions ✓

**Tests:**
1. Run Test Case 1 (ultra-local) with ρ = 0.01
2. Verify: Correlation length ξ ≈ ρ
3. Verify: Field gradient |∇d'| ~ 1/ρ
4. Verify: Perturbation response localized

**Expected result:** Confirms local field structure ✓

### Phase 3: Critical Gauge Covariance Test

**Goal:** Determine if d'_i is gauge-covariant or gauge-invariant in local regime

**Test:** Test Case 1D (04c_test_cases.md § 1.5)

**Implementation:**
1. Define local gauge transformation on subset of walkers
2. Modify companion selection probabilities with phase factors
3. Recompute d'_i with transformed probabilities
4. Measure change Δd'_i inside/outside transformed region

**Possible outcomes:**
- **Outcome A:** Δd' ~ O(α) locally → Gauge covariant ✓✓✓
- **Outcome B:** Δd' ≈ 0 everywhere → Gauge invariant

**Interpretation:**
- A → Local gauge theory viable, strong SM correspondence
- B → Mean-field or non-gauge theory, weaker SM correspondence

### Phase 4: Crossover Study

**Goal:** Map out transition from local to mean-field as ρ increases

**Test:** Test Case 3 (crossover scan)

**Scan:** ρ ∈ [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, ∞]

**Observables:**
- ξ(ρ): Correlation length
- |∇d'|(ρ): Field gradient magnitude
- R_resp(ρ): Perturbation response range
- Δd'_gauge(ρ): Gauge response amplitude (if any)

**Deliverable:** Phase diagram in (ρ, ε) space identifying local vs mean-field regions

---

## Timeline Estimate

| Phase | Tasks | Time Estimate |
|-------|-------|---------------|
| **Phase 1** | Validate mean-field (current code) | 2-3 days |
| **Phase 2** | Implement + validate ρ-localized stats | 1 week |
| **Phase 3** | Gauge covariance test (CRITICAL) | 3-4 days |
| **Phase 4** | Crossover study | 1 week |
| **Analysis** | Write final report with verdict | 3-4 days |
| **Total** | | **3-4 weeks** |

---

## Success Metrics

### Minimum Viable Product (2 weeks)

- [ ] ρ-localized statistics implemented and tested
- [ ] Test Case 1 (ultra-local) passes
- [ ] Test Case 1D (gauge covariance) attempted
- [ ] **Verdict delivered:** Gauge covariant or invariant?

### Full Analysis (4 weeks)

- [ ] All test cases (1, 2, 3) completed
- [ ] Phase diagram created
- [ ] Critical scale ρ_c identified
- [ ] Final report with interpretation
- [ ] Recommendation: Which regime to use

---

## Risk Assessment

**Low risk:**
- ✅ Mean-field analysis (already done, Gemini confirmed)
- ✅ Implementation of ρ-localized stats (straightforward PyTorch)
- ✅ Basic tests (correlation, gradient, perturbation)

**Medium risk:**
- ⚠️ Gauge covariance test (novel, might be inconclusive)
- ⚠️ Defining "local gauge transformation" precisely
- ⚠️ Interpreting results (might be regime-dependent)

**High risk:**
- 🔴 Finding gauge covariance might require non-standard transformation
- 🔴 Current transformation ansatz might be wrong
- 🔴 Physics interpretation might remain ambiguous

**Mitigation:**
- Start with simple tests (correlation, locality)
- Build confidence before gauge test
- Have fallback: even if gauge-invariant, still interesting physics!

---

## Recommendations

### Immediate Next Steps (This Week)

1. ✅ **Document findings** (this document)
2. 🎯 **Implement ρ-localized statistics** (Priority 1)
   - Start with `compute_localization_weights()`
   - Unit test: verify K_ρ(i,j) decays exponentially
3. 🎯 **Run Phase 1** (validate mean-field with current code)
   - Confirms baseline behavior
   - Quick win (should match predictions)

### Short-Term (Weeks 2-3)

4. **Run Phase 2** (local regime tests)
   - Verify local field structure emerges
5. **Run Phase 3** (gauge covariance test)
   - THE critical experiment
   - Determines interpretation

### Medium-Term (Week 4)

6. **Run Phase 4** (crossover study)
   - Map out full parameter space
7. **Write final report**
   - Clear verdict on interpretation
   - Recommendations for using the framework

---

## Expected Outcomes

### Scenario A: Gauge Covariance Found (Best Case)

**Result:** d'_i transforms non-trivially in local regime

**Impact:**
- ✅ Local gauge theory interpretation validated
- ✅ Strong SM correspondence
- ✅ Novel physics (emergent gauge structure)
- ✅ High-impact publication potential

**Next steps:**
- Derive gauge connection A_μ from d'_i
- Construct Yang-Mills action
- Compute gauge boson spectrum

### Scenario B: Gauge Invariance Confirmed (Still Good)

**Result:** d'_i remains invariant in all regimes

**Impact:**
- ✅ Mean-field interpretation confirmed
- ✅ Interesting collective field theory
- ✅ Condensed matter analogs
- ✅ Still publishable (different venue)

**Next steps:**
- Develop mean-field formalism
- Study collective modes (phonon-like)
- Find physical applications

### Scenario C: Regime-Dependent (Most Interesting)

**Result:** Gauge covariance appears/disappears at critical ρ_c

**Impact:**
- ✅ Emergent gauge structure discovery
- ✅ Highest novelty
- ✅ Explains how locality generates gauge theory
- ✅ Potential breakthrough result

**Next steps:**
- Study emergence mechanism in detail
- Theoretical explanation of transition
- High-impact venue (Nature Physics, PRX)

---

## Conclusion

**Current state:** Code operates in mean-field regime (ρ → ∞) by default.

**Gap:** ρ-localized statistics not implemented.

**Path forward:** Implement localization → Run tests → Determine regime → Interpret results.

**Timeline:** 3-4 weeks to definitive answer.

**Risk:** Low (all interpretations are viable and interesting).

**Recommendation:** Proceed with implementation plan. Start with Phase 1-2 for quick validation, then Phase 3 for critical gauge test.

---

**End of Findings Document**
