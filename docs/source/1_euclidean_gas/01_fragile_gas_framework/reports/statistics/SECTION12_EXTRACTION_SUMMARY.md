# Section 12 Extraction Summary

**Date:** 2025-10-27
**Section:** § 12. Standardization pipeline
**Source:** `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework.md` (lines 2751-3682)
**Output Directory:** `/home/guillem/fragile/docs/source/1_euclidean_gas/01_fragile_gas_framework/raw_data/`

---

## Extraction Results

### Total Entities: 15

- **Definitions:** 4
- **Lemmas:** 3
- **Theorems:** 8

---

## Extracted Entities

### Definitions (4)

1. **def-standardization-operator-n-dimensional**
   *N-Dimensional Standardization Operator*
   File: `definitions/def-standardization-operator-n-dimensional.json`

   The core operator that maps a swarm state to an N-dimensional standardized vector (Z-score vector). Parameterized by a raw value operator and an aggregation operator.

2. **def-statistical-properties-measurement**
   *Statistical Properties Measurement*
   File: `definitions/def-statistical-properties-measurement.json`

   Defines how to extract mean and regularized standard deviation from the swarm aggregation measure. Introduces the critical regularized standard deviation function: $\sigma'_{\text{reg}}(V) := \sqrt{V + \sigma'^2_{\min}}$

3. **def-value-error-coefficients**
   *Value Error Coefficients*
   File: `definitions/def-value-error-coefficients.json`

   Defines four coefficients for value error bounds: $C_{V,\text{direct}}$, $C_{V,\mu}(\mathcal{S})$, $C_{V,\sigma}(\mathcal{S})$, and $C_{V,\text{total}}(\mathcal{S})$.

4. **def-lipschitz-structural-error-coefficients**
   *Structural Error Coefficients*
   File: `definitions/def-lipschitz-structural-error-coefficients.json`

   Defines two coefficients for structural error bounds: $C_{S,\text{direct}}$ (linear term) and $C_{S,\text{indirect}}(\mathcal{S}_1, \mathcal{S}_2)$ (quadratic term).

---

### Lemmas (3)

1. **lem-sigma-reg-derivative-bounds**
   *Derivative Bounds for Regularized Standard Deviation*
   File: `lemmas/lem-sigma-reg-derivative-bounds.json`

   Establishes explicit bounds for all derivatives of $\sigma'_{\text{reg}}(V)$. Global Lipschitz constant: $L_{\sigma'_{\text{reg}}} = \frac{1}{2\sigma'_{\min}}$

2. **lem-stats-value-continuity**
   *Value Continuity of Statistical Properties*
   File: `lemmas/lem-stats-value-continuity.json`

   Proves that mean and regularized standard deviation are Lipschitz continuous with respect to changes in the raw value vector.

3. **lem-stats-structural-continuity**
   *Structural Continuity of Statistical Properties*
   File: `lemmas/lem-stats-structural-continuity.json`

   Proves that mean and regularized standard deviation are continuous with respect to changes in swarm structure.

---

### Theorems (8)

1. **thm-z-score-norm-bound**
   *General Bound on the Norm of the Standardized Vector*
   File: `theorems/thm-z-score-norm-bound.json`

   Universal bound: $\|\mathbf{z}\|_2^2 \le k \left( \frac{2V_{\max}}{\varepsilon_{\mathrm{std}}} \right)^2$

2. **thm-asymptotic-std-dev-structural-continuity**
   *Asymptotic Behavior of Structural Continuity*
   File: `theorems/thm-asymptotic-std-dev-structural-continuity.json`

   Shows $L_{\sigma',S}(k) \propto k^{p_{\text{worst-case}}}$ where $p_{\text{worst-case}} = \max(p_{\mu,S}, p_{m_2,S})$.

3. **thm-standardization-value-error-mean-square**
   *Bounding the Expected Squared Value Error*
   File: `theorems/thm-standardization-value-error-mean-square.json`

   Mean-square continuity bound for value error: $E_{V,ms}^2 \le C_{V,\text{total}} \cdot F_{V,ms}$

4. **thm-standardization-structural-error-mean-square**
   *Bounding the Expected Squared Structural Error*
   File: `theorems/thm-standardization-structural-error-mean-square.json`

   Mean-square continuity bound for structural error: $E_{S,ms}^2 \le C_{S,\text{direct}} \cdot n_c + C_{S,\text{indirect}} \cdot n_c^2$

5. **thm-deterministic-error-decomposition**
   *Decomposition of the Total Standardization Error*
   File: `theorems/thm-deterministic-error-decomposition.json`

   Fundamental decomposition: Total error = $2 \cdot E_{V}^2 + 2 \cdot E_{S}^2$

6. **thm-lipschitz-value-error-bound**
   *Bounding the Squared Value Error (Deterministic)*
   File: `theorems/thm-lipschitz-value-error-bound.json`

   Lipschitz continuity: $E_{V}^2 \le C_{V,\text{total}} \cdot \|\mathbf{v}_1 - \mathbf{v}_2\|_2^2$

7. **thm-lipschitz-structural-error-bound**
   *Bounding the Squared Structural Error (Deterministic)*
   File: `theorems/thm-lipschitz-structural-error-bound.json`

   Hölder continuity: $E_{S}^2 \le C_{S,\text{direct}} \cdot n_c + C_{S,\text{indirect}} \cdot n_c^2$

8. **thm-global-continuity-patched-standardization**
   *Global Continuity of the Patched Standardization Operator*
   File: `theorems/thm-global-continuity-patched-standardization.json`

   **Main Result:** Combines value and structural error bounds into a unified deterministic global continuity guarantee.

---

## Key Mathematical Insights

### 1. Regularization Innovation

The regularized standard deviation function $\sigma'_{\text{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}$ is the critical innovation that:
- Prevents division by zero
- Maintains $C^\infty$ smoothness
- Enables global Lipschitz continuity

### 2. Dual Continuity Framework

Section 12 provides **two complementary continuity analyses**:

- **§11.2: Mean-Square Continuity** (probabilistic)
  - Bounds expected squared error
  - Useful for typical case analysis

- **§11.3: Deterministic Lipschitz-Hölder Continuity**
  - Worst-case guarantees
  - Required for Feynman-Kac theory

### 3. Error Decomposition

Total error splits into:
- **Value error:** From changes in raw value distribution
- **Structural error:** From changes in alive set composition

### 4. Operational Regimes

The framework identifies two distinct behaviors:

**Normal Operation** ($n_c$ small):
- Error is constant w.r.t. swarm size
- Scales as $O(\varepsilon_{\text{std}}^{-6})$

**Catastrophic Collapse** ($n_c \propto k$):
- Error grows linearly with swarm size
- Scales as $O(k \cdot \varepsilon_{\text{std}}^{-6})$

### 5. Asymptotic Scaling

For large swarms:
$$L_{\sigma',S}(k) \propto k^{p_{\text{worst-case}}}$$

For the empirical aggregator: $p_{\text{worst-case}} = -1$ (structurally stable!)

---

## Dependencies

Entities in Section 12 depend on:

- `def-raw-value-operator`
- `def-swarm-aggregation-operator-axiomatic`
- `axiom-raw-value-mean-square-continuity`
- `lem-empirical-aggregator-properties`
- `thm-distance-operator-mean-square-continuity`

---

## Files Created

```
raw_data/
├── definitions/
│   ├── def-standardization-operator-n-dimensional.json
│   ├── def-statistical-properties-measurement.json
│   ├── def-value-error-coefficients.json
│   └── def-lipschitz-structural-error-coefficients.json
├── lemmas/
│   ├── lem-sigma-reg-derivative-bounds.json
│   ├── lem-stats-value-continuity.json
│   └── lem-stats-structural-continuity.json
├── theorems/
│   ├── thm-z-score-norm-bound.json
│   ├── thm-asymptotic-std-dev-structural-continuity.json
│   ├── thm-standardization-value-error-mean-square.json
│   ├── thm-standardization-structural-error-mean-square.json
│   ├── thm-deterministic-error-decomposition.json
│   ├── thm-lipschitz-value-error-bound.json
│   ├── thm-lipschitz-structural-error-bound.json
│   └── thm-global-continuity-patched-standardization.json
└── statistics/
    └── section12_extraction_statistics.json
```

---

## Notes

- **Extraction method:** Manual extraction by Claude Code
- **Sub-lemmas not extracted:** Algebraic decomposition sub-lemmas (e.g., `sub-lem-value-error-decomposition`) were not extracted as separate JSON files, but are referenced in proof sketches
- **Proofs:** Full proofs remain in source document; JSON files contain proof sketches for reference
- **Quality:** All entities include proper labels, dependencies, tags, and content formatting

---

## Next Steps

To use these extracted entities:

1. **Cross-referencing:** Use the `cross-referencer` agent to build the dependency graph
2. **Validation:** Verify all dependency references resolve correctly
3. **Integration:** Incorporate into the larger framework knowledge base
4. **Documentation:** Update glossary with Section 12 entries

---

**Status:** ✅ Extraction Complete
**Quality Check:** All entities include proper structure, dependencies, and tags
