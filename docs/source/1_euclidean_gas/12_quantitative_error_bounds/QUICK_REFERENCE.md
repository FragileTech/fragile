# Chapter 12: Quantitative Error Bounds - Quick Reference

## Main Results (2)

### 1. thm-quantitative-propagation-chaos (Object 7, Part I)
**Mean-field convergence with optimal rate**
```
|E[phi(empirical_N)] - E_meanfield[phi]| <= C_obs * L_phi / sqrt(N)
```
- **Rate**: O(1/√N) - OPTIMAL (matches CLT)
- **Constant**: C_obs = √(C_var + C' C_int)
- **Dependencies**: Ch 09 thm-kl-convergence-euclidean (N-uniform LSI)

### 2. thm-total-error-bound (Object 18, Part IV)
**Complete discrete system error bound**
```
|E[phi(discrete_N)] - E_meanfield[phi]| <= (C_MF/sqrt(N) + C_discrete*dt/N) * ||phi||_C4
```
- **Rates**: O(1/√N) + O(Δt/N)
- **Key**: Discretization term NEGLIGIBLE for large N
- **Practical**: For N=10^4, dt=0.01: MF ~0.01, discrete ~10^-6

---

## Structure Overview

| Part | Objects | Main Result | Key Rate | Focus |
|------|---------|-------------|----------|-------|
| I | 1-7 | thm-quantitative-propagation-chaos | O(1/√N) | Mean-field convergence |
| II | 8-11 | thm-langevin-baoab-discretization-error | O((Δt)²) | BAOAB alone (no cloning) |
| III | 12-17 | thm-full-system-discretization-error | O(Δt) | Full system (Langevin + cloning) |
| IV | 18-19 | thm-total-error-bound | O(1/√N + Δt/N) | Total bound + constants |

---

## Critical Dependencies

### From Chapter 09 (KL Convergence)
- **thm-kl-convergence-euclidean** - N-uniform LSI
  - Constant: λ_LSI = γ κ_conf κ_W δ² / C₀
  - **MOST CRITICAL** external dependency

### From Chapter 06 (Convergence)
- thm-foster-lyapunov-main - Geometric ergodicity
- thm-main-convergence - QSD convergence
- thm-energy-bounds - Energy control

### From Chapter 08 (Propagation of Chaos)
- Framework for exchangeability
- Mean-field scaling principles

---

## Proof Chain for thm-total-error-bound

```
Ch 09: N-uniform LSI
    ↓
Part I: KL bound O(1/N) [lem 2 + prop 3]
    ↓
Part I: W₂² bound O(1/N) [prop 5]
    ↓
Part I: W₁ bound O(1/√N) [lem 4]
    ↓
Part I: Observable error O(1/√N) [thm 7] ← PART I RESULT
    ↓
    ↓ (combine with Part III)
    ↓
Part II: Fourth moments [prop 8]
    ↓
Part II: BAOAB weak error O((Δt)²) [lem 9]
    ↓
Part II: BAOAB invariant error O((Δt)²) [lem 10]
    ↓
Part III: Lie splitting local error O((Δt)²) [lem 13]
    ↓
Part III: Geometric ergodicity (uniform in Δt) [lem 14]
    ↓
Part III: Error propagation O((Δt)²) local → O(Δt) global [thm 17]
    ↓
    ↓ (Triangle inequality)
    ↓
Part IV: Total bound O(1/√N + Δt/N) [thm 18] ← FINAL RESULT
    ↓
Part IV: Explicit constants [prop 19]
```

---

## Key Objects by Importance

### ⭐⭐⭐ Critical (Main Results)
- **07**: thm-quantitative-propagation-chaos
- **18**: thm-total-error-bound

### ⭐⭐ High Importance (Key Lemmas)
- **02**: lem-quantitative-kl-bound - KL divergence O(1/N)
- **04**: lem-lipschitz-observable-error - Observable error conversion
- **13**: lem-lie-splitting-weak-error - Commutator calculation
- **14**: lem-uniform-geometric-ergodicity - Mixing uniformity
- **17**: thm-quantitative-error-propagation - Local → global

### ⭐ Important (Supporting)
- **03**: prop-interaction-complexity-bound - Explicit C_int
- **08**: prop-fourth-moment-baoab - Moment control
- **09**: lem-baoab-weak-error - Second-order weak convergence
- **10**: lem-baoab-invariant-measure-error - Talay's method
- **19**: prop-quantitative-explicit-constants - Implementation formulas

### Support (Prerequisites)
- **05**: prop-empirical-wasserstein-concentration
- **06**: prop-finite-second-moment-meanfield
- **11**: thm-langevin-baoab-discretization-error
- **16**: prop-mixing-rate-relationship

### Historical/External
- **01**: lem-wasserstein-entropy (not used)
- **15**: thm-meyn-tweedie-drift-minor (external)

---

## Explicit Constants (Object 19)

### Mean-field constant:
```
C_MF = sqrt(C_var + C' * C_int)
C_int = lambda * L_log(rho_0) * diam(Omega)
```

### Discretization constant:
```
C_discrete = (C_split * C_poisson) / kappa_mix
C_split = (1/2) * lambda * beta * C_chaos * max(C_F, sigma^2, ||nabla^2 U||)
kappa_mix = min(kappa_hypo, lambda)
```

### Typical values:
- γ (friction): 0.1 to 1.0
- σ (noise): 0.1 to 1.0
- λ (cloning): 0.01 to 0.1
- δ (cloning noise): 0.1 to 0.3
- β (fitness weight): 1 to 10

### Order of magnitude:
- C_MF ~ O(10)
- C_discrete ~ O(1) to O(10)

---

## Key Mathematical Techniques

| Part | Technique | Purpose |
|------|-----------|---------|
| I | Relative entropy method | Control D_KL evolution |
| I | Fournier-Guillin bound | Empirical measure concentration (exchangeable) |
| I | Kantorovich-Rubinstein | Convert W₁ to observable error |
| II | Lyapunov on E² | Fourth moment control |
| II | Backward error analysis | Generator expansion |
| II | Poisson equation | Finite-time → invariant error |
| III | Commutator calculation | [L_clone, L_Langevin] bound |
| III | Mean-field cancellation | N-uniform bound via fluctuations |
| III | Drift-minorization | Meyn-Tweedie ergodicity |
| IV | Triangle inequality | Combine error sources |

---

## Why Different Rates?

| Component | Rate | Reason |
|-----------|------|--------|
| Mean-field | O(1/√N) | Central Limit Theorem (OPTIMAL) |
| BAOAB alone | O((Δt)²) | Symmetric integrator (Talay cancellation) |
| Full system | O(Δt) | Operator splitting: [L_clone, L_Langevin] ≠ 0 |
| Empirical observable | Extra 1/N | Averaging over N particles |
| Total discrete | O(Δt/N) | NEGLIGIBLE vs mean-field |

---

## Practical Implications

### For N = 10,000 walkers, Δt = 0.01:
- Mean-field error: ~0.01 (dominant)
- Discretization error: ~10^-6 (negligible)
- **Conclusion**: Focus on increasing N, not refining Δt

### Balanced regime (equal contributions):
- Requires: Δt ~ 1/√N
- For N = 10^4: Δt ~ 0.01
- Rarely needed in practice

### No balancing needed:
- Increasing N reduces BOTH errors
- Simple scaling: N ↑ → both errors ↓
- Unlike typical particle methods

---

## File Locations

```
/home/guillem/fragile/docs/source/1_euclidean_gas/12_quantitative_error_bounds/
├── theorems/
│   ├── 01_lem_wasserstein_entropy.json
│   ├── 02_lem_quantitative_kl_bound.json
│   ├── ...
│   ├── 18_thm_total_error_bound.json
│   ├── 19_prop_quantitative_explicit_constants.json
│   └── INDEX.json
├── EXTRACTION_SUMMARY.md (this file)
└── QUICK_REFERENCE.md (comprehensive guide)
```

---

## Validation Status

- ✅ All 19 objects extracted
- ✅ Complete input/output specifications
- ✅ Parameter dependencies documented
- ✅ Proof strategies recorded
- ✅ Cross-chapter references identified
- ✅ Constants made explicit
- ✅ Main results clearly marked
- ✅ Ready for downstream processing

---

## Next Steps

1. **Integration**: Add to mathematical framework graph
2. **Validation**: Numerical experiments to verify bounds
3. **Constants**: Compute explicit C_int for test problems
4. **Automation**: Build proof-verification pipeline
5. **Extensions**: Adaptive Gas with mean-field interactions

---

## Quick Lookup

**Need mean-field convergence?** → Object 7 (thm-quantitative-propagation-chaos)
**Need total error bound?** → Object 18 (thm-total-error-bound)
**Need explicit constants?** → Object 19 (prop-quantitative-explicit-constants)
**Need BAOAB analysis?** → Objects 8-11 (Part II)
**Need splitting analysis?** → Objects 13-17 (Part III)
**Need KL bound?** → Object 2 (lem-quantitative-kl-bound)
**Need interaction complexity?** → Object 3 (prop-interaction-complexity-bound)
