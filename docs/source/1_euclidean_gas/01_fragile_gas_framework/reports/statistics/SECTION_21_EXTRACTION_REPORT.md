# Section 21 Extraction Complete

## Summary Statistics

**Source Document**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md`

**Section**: §21 - Continuity Constants: Canonical Values and Glossary (lines 5140-5262)

**Total Lines Processed**: 122 lines

---

## Entities Extracted

### Total Count: **11 Mathematical Objects**

| Entity Type | Count |
|-------------|-------|
| Reference Tables | 1 |
| Constant Collections | 6 |
| Coefficients | 1 |
| Coefficient Collections | 1 |
| Bounds | 2 |
| **TOTAL** | **11** |

---

## Files Created

**Output Directory**: `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/`

### 11 Mathematical Object Files + 1 Summary File

1. `obj-continuity-constants-table.json` (3.1K)
   - Comprehensive table of 44+ constants organized by type
   
2. `obj-aggregator-lipschitz-constants.json` (2.0K)
   - 5 Lipschitz constants for empirical moments
   
3. `obj-standardization-total-value-error-coefficient.json` (2.8K)
   - Three-component coefficient for standardization
   
4. `obj-distance-measurement-ms-constants.json` (2.0K)
   - 3 constants for distance-to-companion measurement
   
5. `obj-cloning-probability-lipschitz-constants.json` (2.2K)
   - 4 constants for cloning probability continuity
   
6. `obj-potential-bounds.json` (2.0K)
   - Lower and upper bounds on fitness potential
   
7. `obj-perturbation-constants.json` (2.2K)
   - 3 constants for Gaussian perturbation bounds
   
8. `obj-boundary-regularity-constants.json` (2.2K)
   - 2 constants for boundary death probability
   
9. `obj-rescale-lipschitz-constants.json` (2.4K)
   - 3 constants for rescale function and composition
   
10. `obj-one-step-ms-continuity-coefficients.json` (4.2K)
    - Complete collection of final coefficients (Lipschitz, Hölder, offset)
    
11. `obj-standardization-lower-bound.json` (2.1K)
    - Uniform lower bound on smoothed standard deviation

12. `section_21_extraction_summary.json` (6.5K)
    - Metadata and extraction statistics

**Total Data Created**: ~33.9 KB

---

## Key Constants Coverage (44+ Total)

### By Category

- **Environment** (2): R_max, L_R
- **Geometry** (2): D_Y, L_phi
- **Noise** (2): sigma, delta
- **Dynamics** (2): alpha, beta
- **Regularization** (4): epsilon_std, kappa_var_min, eta, z_max
- **Cloning** (2): p_max, epsilon_clone
- **Structural** (2): k, n_c
- **Measurement** (1): V_max
- **Aggregation** (5): L_mu_M, L_m2_M, L_var, L_sigma_reg, L_sigma_M
- **Standardization** (5): sigma'_min_bound, C_V_direct, C_V_mu, C_V_sigma, C_V_total
- **Rescale** (3): L_P, L_g_A, L_g_A_circ_z
- **Potential** (2): V_pot_min, V_pot_max
- **Cloning Probability** (4): L_pi_c, L_pi_i, C_val_pi, C_struct_pi
- **Perturbation** (3): M_pert^2, B_M, B_S
- **Boundary** (2): L_death, alpha_B
- **Distance** (3): C_pos_d, C_status_d^{(1)}, C_status_d^{(2)}
- **One-Step** (6): C_Psi_L, C_Psi_H, K_Psi, C_clone_L, C_clone_H, K_clone

---

## Section Structure Mapping

```
§21 Continuity Constants
│
├── §21.1 Quick Table
│   ├── obj-continuity-constants-table.json
│   ├── obj-aggregator-lipschitz-constants.json
│   ├── obj-rescale-lipschitz-constants.json
│   └── obj-standardization-lower-bound.json
│
├── §21.2 Mean-Square Standardization
│   └── obj-standardization-total-value-error-coefficient.json
│
├── §21.3 Distance Measurement
│   └── obj-distance-measurement-ms-constants.json
│
├── §21.4 Cloning Probability & Potential
│   ├── obj-cloning-probability-lipschitz-constants.json
│   └── obj-potential-bounds.json
│
├── §21.5 Perturbation & Boundary
│   ├── obj-perturbation-constants.json
│   └── obj-boundary-regularity-constants.json
│
└── §21.6 One-Step Continuity
    └── obj-one-step-ms-continuity-coefficients.json
```

---

## Key Formulas Captured

### Total Value Error Coefficient
```
C_V_total(S) = 3 * (C_V_direct + C_V_mu(S) + C_V_sigma(S))
```
Scaling: **O(epsilon_std^{-6})** when kappa_var_min << epsilon_std^2

### One-Step Continuity Bound
```
E[d_out^2] <= C_Psi_L * V_in + C_Psi_H * V_in^{alpha_H} + K_Psi
```

### Aggregator Lipschitz Constants
```
L_mu_M = 1/sqrt(k)
L_m2_M = 2*V_max/sqrt(k)
L_var = 4*V_max/sqrt(k)
```

---

## Dependencies Identified

Referenced results (defined in other sections):
- `lem-sigma-reg-derivative-bounds` (provides L_sigma'_reg)
- `thm-distance-operator-mean-square-continuity` (uses distance constants)

---

## Parameter Tuning Guidance

### Sensitivity Reduction Strategies

1. **Increase** epsilon_std or kappa_var_min → decreases C_V_total
2. **Larger** k (alive walkers) → tightens empirical-moment Lipschitz via 1/sqrt(k)
3. **Smaller** sigma → reduces post-perturbation constants
4. **Conservative** p_max, larger eta → shrinks L_pi and cloning amplification

### Key Scaling Laws

- C_V_total ~ **O(epsilon_std^{-6})**
- Aggregator constants ~ **1/sqrt(k)**
- L_death ~ **Per(E)/sigma**
- Perturbation bounds ~ **O(N*d*sigma^2)**

---

## Quality Assurance

✓ All JSON files validated (valid JSON format)  
✓ All 11 objects + 1 summary created successfully  
✓ Complete coverage of section content  
✓ Dependencies tracked and documented  
✓ Source sections and line numbers recorded  
✓ Mathematical formulas preserved  
✓ Parameter descriptions included  
✓ Tuning guidance captured  

---

## Files Location

All extracted entities are located in:

```
/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/
```

To view a specific entity:
```bash
cat /home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/objects/obj-continuity-constants-table.json | python3 -m json.tool
```
