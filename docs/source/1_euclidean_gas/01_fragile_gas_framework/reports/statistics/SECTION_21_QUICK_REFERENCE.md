# Section 21 Quick Reference Guide

## Most Important Objects for Parameter Tuning

### 1. Total Value Error Coefficient
**File**: `obj-standardization-total-value-error-coefficient.json`

**Formula**: 
```
C_V_total(S) = 3 * (C_V_direct + C_V_mu(S) + C_V_sigma(S))
           = 3 * (2/sigma'^2_min_bound + 64*V_max^4*L_sigma'_reg^2 / sigma'^4_min_bound)
```

**Key Parameter**: epsilon_std (scaling: O(epsilon_std^{-6}))

**Tuning**: Increase epsilon_std to reduce sensitivity

---

### 2. Aggregator Lipschitz Constants
**File**: `obj-aggregator-lipschitz-constants.json`

**Formulas**:
```
L_mu_M = 1/sqrt(k)
L_m2_M = 2*V_max/sqrt(k)
L_var = 4*V_max/sqrt(k)
```

**Key Parameter**: k (number of alive walkers)

**Tuning**: Maintain larger k for better stability (constants scale as 1/sqrt(k))

---

### 3. Cloning Probability Lipschitz Constants
**File**: `obj-cloning-probability-lipschitz-constants.json`

**Formulas**:
```
L_pi_c = 1/(p_max * epsilon_clone)
L_pi_i = (V_pot_max + epsilon_clone)/(p_max * epsilon_clone^2)
```

**Key Parameters**: p_max, epsilon_clone

**Tuning**: Conservative (larger) p_max and larger epsilon_clone shrink these constants

---

### 4. One-Step Continuity Bound
**File**: `obj-one-step-ms-continuity-coefficients.json`

**Formula**:
```
E[d_out^2] <= C_Psi_L * V_in + C_Psi_H * V_in^{alpha_H} + K_Psi
```

**Components**:
- C_Psi_L: Lipschitz coefficient (linear sensitivity)
- C_Psi_H: Hölder coefficient (sublinear sensitivity from boundary)
- K_Psi: Constant offset (perturbation noise and fluctuations)

**Tuning**: This is the final assembled bound - tune component parameters to control each part

---

### 5. Perturbation Constants
**File**: `obj-perturbation-constants.json`

**Formulas**:
```
M_pert^2 = d * sigma^2
B_M(N) = N * d * sigma^2
B_S(N, delta) = D_Y^2 * sqrt(N/2 * ln(2/delta))
```

**Key Parameter**: sigma (perturbation noise scale)

**Tuning**: Smaller sigma reduces perturbation bounds but increases L_death

---

## Parameter Trade-offs Summary

| Parameter | Increase → Effect | Related Constants |
|-----------|-------------------|-------------------|
| epsilon_std | ↓ C_V_total (good) | Standardization sensitivity |
| kappa_var_min | ↓ C_V_total (good) | Variance floor |
| k (alive walkers) | ↓ Aggregator Lipschitz (good) | All empirical moments |
| sigma | ↓ Perturbation bounds (good) <br> ↑ L_death (bad) | Noise-boundary trade-off |
| p_max | ↓ L_pi constants (good) | Cloning probability |
| eta | ↑ V_pot_min (good) | Fitness potential stability |
| epsilon_clone | ↓ L_pi constants (mixed) | Cloning regularization |

---

## Quick Diagnostic Checklist

When tuning algorithm stability, check these constants in order:

1. **Standardization Stage** (`obj-standardization-total-value-error-coefficient.json`)
   - Is C_V_total too large? → Increase epsilon_std
   - Check: C_V_total ~ O(epsilon_std^{-6})

2. **Aggregation Stage** (`obj-aggregator-lipschitz-constants.json`)
   - Are aggregator constants large? → Check k (need more alive walkers)
   - Check: L_var = 4*V_max/sqrt(k)

3. **Cloning Stage** (`obj-cloning-probability-lipschitz-constants.json`)
   - Is cloning too sensitive? → Increase p_max or epsilon_clone
   - Check: L_pi_i = (V_pot_max + epsilon_clone)/(p_max * epsilon_clone^2)

4. **Perturbation Stage** (`obj-perturbation-constants.json`)
   - Too much perturbation noise? → Decrease sigma
   - Check: B_M = N*d*sigma^2

5. **Boundary Stage** (`obj-boundary-regularity-constants.json`)
   - Death probability too sensitive? → Increase sigma (smooths boundary)
   - Check: L_death = C_d * Per(E) / sigma

6. **Final Bound** (`obj-one-step-ms-continuity-coefficients.json`)
   - Check assembled C_Psi_L, C_Psi_H, K_Psi
   - Verify all component improvements propagate

---

## File Index

| Entity | File | Size | Key Content |
|--------|------|------|-------------|
| Full Table | `obj-continuity-constants-table.json` | 3.1K | All 44+ constants |
| Aggregator | `obj-aggregator-lipschitz-constants.json` | 2.0K | 5 empirical moment constants |
| Standardization | `obj-standardization-total-value-error-coefficient.json` | 2.8K | C_V_total formula |
| Distance | `obj-distance-measurement-ms-constants.json` | 2.0K | 3 distance constants |
| Cloning Prob | `obj-cloning-probability-lipschitz-constants.json` | 2.2K | 4 cloning constants |
| Potential | `obj-potential-bounds.json` | 2.0K | V_pot bounds |
| Perturbation | `obj-perturbation-constants.json` | 2.2K | 3 perturbation bounds |
| Boundary | `obj-boundary-regularity-constants.json` | 2.2K | L_death, alpha_B |
| Rescale | `obj-rescale-lipschitz-constants.json` | 2.4K | 3 rescale constants |
| One-Step | `obj-one-step-ms-continuity-coefficients.json` | 4.2K | Final assembled bound |
| Std Bound | `obj-standardization-lower-bound.json` | 2.1K | sigma'_min_bound |

**Total**: 11 objects covering 44+ individual constants
