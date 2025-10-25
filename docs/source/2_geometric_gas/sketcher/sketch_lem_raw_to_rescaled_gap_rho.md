# Proof Sketch for lem-raw-to-rescaled-gap-rho

**Document**: docs/source/2_geometric_gas/11_geometric_gas.md
**Theorem**: lem-raw-to-rescaled-gap-rho
**Generated**: 2025-10-25 09:05
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:lemma} Raw-Gap to Rescaled-Gap for ρ-Localized Pipeline
:label: lem-raw-to-rescaled-gap-rho

If the raw measurements satisfy:

$$
\max_{i \in \{1, \ldots, N\}} |d_i - \mu_\rho[f_k, d, x_{\text{ref}}]| \ge \kappa_{\text{raw}}
$$

for some reference point $x_{\text{ref}}$ and raw gap $\kappa_{\text{raw}} > 0$, then the rescaled measurements satisfy:

$$
\max_{i \in \{1, \ldots, N\}} |d'_i - \mu[d']| \ge \kappa_{\text{rescaled}}(\kappa_{\text{raw}}, \rho)
$$

where:

$$
\kappa_{\text{rescaled}}(\kappa_{\text{raw}}, \rho) := g'_{\min} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}}
$$
:::

**Informal Restatement**: A guaranteed gap in raw diversity measurements (at least one walker differs from the localized mean by κ_raw) propagates through the Z-score normalization and sigmoid rescaling pipeline to produce a guaranteed gap in the rescaled measurements. The rescaled gap depends on ρ through σ'_{ρ,max} (worst-case localized standard deviation) and is amplified by g'_min (minimum derivative of the rescale function).

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Method**: Constructive pairwise-gap propagation

**Key Steps**:
1. Formalize local environment assumption (μ_ρ and σ'_ρ approximately constant)
2. Convert gap-from-reference-mean to pairwise gap: max_{i,j} |d_i - d_j| ≥ κ_raw
3. Propagate to Z-scores: max_{i,j} |Z_i - Z_j| ≥ κ_raw/σ'_{ρ,max}
4. Apply MVT with g'_A ≥ g'_min: max_{i,j} |d'_i - d'_j| ≥ g'_min · κ_raw/σ'_{ρ,max}
5. Convert pairwise gap to gap-from-mean using max_i |y_i - ȳ| ≥ ½ max_{i,j} |y_i - y_j|
6. Identify factor-of-2 discrepancy with theorem statement

**Strengths**:
- Systematic tracking through each transformation stage
- Identifies critical technical issue: local parameter stability assumption
- Recognizes factor-of-2 discrepancy; suggests theorem may need correction

**Weaknesses**:
- Requires additional lemma for Step 1 (local parameter stability)
- Concludes with ½ κ_rescaled instead of κ_rescaled (potential theorem error)

**Framework Dependencies**:
- lem-rho-pipeline-bounds (σ'_ρ bounds, g'_A bounds)
- Mean Value Theorem
- Standard properties of means

---

### Strategy B: GPT-5's Approach

**Method**: Pairwise-gap propagation with anchored Z-score correction

**Key Steps**:
1. Extract pairwise gap via convex combination: ∃(i*, j*) with |d_{i*} - d_{j*}| ≥ κ_raw
2. Lower-bound Z-score difference with anchored reference:
   - Define Z^ref(u) := (u - μ_ρ[x_ref])/σ'_{ρ,max}
   - Bound |Z_ρ(x) - Z^ref(d(x))| using C¹/C² regularity
   - Get |Z_ρ(x_{i*}) - Z_ρ(x_{j*})| ≥ κ_raw/σ'_{ρ,max} - E_ρ
3. Apply MVT: |d'_{i*} - d'_{j*}| ≥ g'_min · (κ_raw/σ'_{ρ,max} - E_ρ)
4. Convert pairwise to mean gap with factor ½
5. Final bound: max_i |d'_i - μ[d']| ≥ (g'_min/2) · κ_raw/σ'_{ρ,max} - (g'_min/2) · E_ρ

**Strengths**:
- Rigorous treatment of Z-score cross-terms (different μ_ρ, σ'_ρ at each point)
- Introduces anchored Z-score to make naive inequality valid
- Explicit error term E_ρ from localization drift
- Provides pathway to rigor via existing C¹/C² bounds

**Weaknesses**:
- More complex due to anchored Z-score construction
- Requires additional Lemma L3 for anchored-to-actual Z-score control
- Also identifies factor-of-2 issue

**Framework Dependencies**:
- lem-rho-pipeline-bounds
- C¹/C² regularity bounds on μ_ρ, σ'_ρ (from 19_geometric_gas_cinf_regularity_simplified.md)
- MVT, properties of means

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Pairwise-gap propagation with rigorous Z-score analysis (GPT-5's approach) + clarification of theorem statement

**Rationale**:
Both strategists independently identify the same core approach (pairwise gap → Z-score gap → rescaled gap → mean gap) but differ in technical rigor:

1. **Gemini** correctly identifies the conceptual flow but flags the need for a local stability lemma without providing details
2. **GPT-5** provides the rigorous solution via anchored Z-scores and error term E_ρ

The key insight is that the naive inequality |Z_ρ(x_i) - Z_ρ(x_j)| ≥ |d_i - d_j|/σ'_{ρ,max} is NOT valid when μ_ρ and σ'_ρ vary with basepoint. GPT-5's anchored approach resolves this rigorously.

**Critical Finding**: Both strategists identify a factor-of-2 discrepancy. The rigorous proof yields:

$$
\max_i |d'_i - \mu[d']| \ge \frac{1}{2} \cdot g'_{\min} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}} - \frac{1}{2} g'_{\min} \cdot E_\rho
$$

The factor ½ arises from the standard inequality max_i |y_i - ȳ| ≥ ½ max_{i,j} |y_i - y_j| (tight for {0,1}).

**Integration**:
- Step 1: Use GPT-5's convex-combination argument (avoids Gemini's stability assumption)
- Step 2: Use GPT-5's anchored Z-score construction with error term E_ρ
- Steps 3-4: Both agree on MVT application and mean reduction
- Resolution: Either (a) theorem should include factor ½, or (b) additional assumptions eliminate the factor

**Verification Status**:
- ✅ All framework dependencies verified
- ⚠️ Requires Lemma L3: Anchored-to-actual Z-score control via C¹/C² bounds
- ⚠️ Theorem statement may need factor-of-2 correction OR additional structural assumption
- ✅ No circular reasoning

---

## III. Framework Dependencies

### Verified Dependencies

**Lemmas**:
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| lem-rho-pipeline-bounds | 11_geometric_gas.md | σ'_ρ ≤ σ'_{ρ,max}, g'_A ≥ g'_min | Steps 2-3 | ✅ |
| lem-variance-to-gap-adaptive | 11_geometric_gas.md | max\|x-μ\| ≥ σ | Alternative approach | ✅ |

**Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| Localization weights | 19 (line 262-272) | ∑_j w_ij(ρ) = 1, w_ij ≥ 0 | Convex combination in Step 1 |
| Z-score | 19 (line 290-348) | Z_ρ = (d - μ_ρ)/σ'_ρ | Step 2 |
| Rescale function | 11 (line 3187-3211) | g_A smooth, monotone, g'_A ≥ g'_min | Step 3 |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| σ'_{ρ,max} | Worst-case localized std dev | σ'_{ρ,max} := d_max | ρ-dependent bound |
| g'_min | Min derivative of g_A | g'_A(z) ≥ g'_min > 0 | Monotonicity bound |
| E_ρ | Localization drift error | Bounds via C¹/C² regularity | ρ-dependent, can be made small |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- **Lemma L1**: Weighted-average to pairwise gap - easy - Standard convex combination argument
- **Lemma L2**: Pair-to-mean reduction: max_i |y_i - ȳ| ≥ ½ max_{i,j} |y_i - y_j| - easy - Standard inequality
- **Lemma L3**: Anchored Z-score control - medium - Bound |Z_ρ(x) - Z^ref(d(x))| using C¹/C² bounds on μ_ρ, σ'_ρ from framework documents

**Uncertain Assumptions**:
- **Factor-of-2 issue**: Theorem statement vs. rigorous proof - Needs resolution - Either correct theorem or add structural assumption (e.g., μ[d'] at boundary)

---

## IV. Detailed Proof Sketch

### Overview

The proof tracks a guaranteed raw gap through the rescaling pipeline (raw → Z-score → sigmoid → rescaled). The challenge is that the Z-score transformation uses localized means and standard deviations that vary with walker position. The solution is to introduce an "anchored" Z-score using a fixed reference point, bound the deviation between anchored and actual Z-scores using smoothness of μ_ρ and σ'_ρ, then propagate the gap through the monotone rescale function g_A via the Mean Value Theorem. Finally, a standard statistical inequality converts pairwise gaps to gap-from-mean (introducing factor ½).

The key insight: The rescaling pipeline is Lipschitz-like (via MVT), so gaps shrink by at most a multiplicative factor. The rescaled gap constant κ_rescaled = g'_min · κ_raw/σ'_{ρ,max} captures this contraction, with ρ-dependence entering through σ'_{ρ,max}.

### Proof Outline (Top-Level)

1. **Extract pairwise raw gap**: Use convex combination property of μ_ρ to find pair (i*, j*) with |d_{i*} - d_{j*}| ≥ κ_raw
2. **Bound Z-score difference**: Use anchored Z-score Z^ref and bound drift |Z_ρ - Z^ref| via regularity
3. **Propagate through g_A**: Apply MVT with g'_A ≥ g'_min to get rescaled pairwise gap
4. **Convert to mean gap**: Use max_i |y_i - ȳ| ≥ ½ max_{i,j} |y_i - y_j|
5. **Assemble conclusion**: Combine constants to get final bound

---

### Detailed Step-by-Step Sketch

#### Step 1: Extract Pairwise Raw Gap

**Goal**: From max_i |d_i - μ_ρ[x_ref]| ≥ κ_raw, derive ∃(i*, j*): |d_{i*} - d_{j*}| ≥ κ_raw

**Substep 1.1**: Identify extreme walker
- **Justification**: Let i* attain max_i |d_i - μ_ρ[f_k, d, x_ref]| ≥ κ_raw
- **Why valid**: Maximum exists over finite set {1,...,N}
- **Expected result**: |d_{i*} - μ_ρ[x_ref]| ≥ κ_raw

**Substep 1.2**: Express localized mean as convex combination
- **Justification**: μ_ρ[f_k, d, x_ref] = ∑_j w_j(x_ref) d_j where w_j ≥ 0, ∑_j w_j = 1
- **Why valid**: Normalized localization weights (19, line 262-272); Gaussian kernel positive
- **Expected result**: μ_ρ is a weighted average with positive weights

**Substep 1.3**: Apply triangle inequality in reverse
- **Justification**: |d_{i*} - μ_ρ| = |∑_j w_j(d_{i*} - d_j)| ≤ ∑_j w_j · |d_{i*} - d_j| ≤ max_j |d_{i*} - d_j|
- **Why valid**: Triangle inequality + normalization ∑w_j = 1
- **Expected result**: ∃j* s.t. |d_{i*} - d_{j*}| ≥ |d_{i*} - μ_ρ| ≥ κ_raw

**Conclusion**: Pairwise raw gap |d_{i*} - d_{j*}| ≥ κ_raw established

**Dependencies**:
- Uses: Positive, normalized weights w_j(x_ref)
- Requires: k ≥ 1 (at least one alive walker)

**Potential Issues**:
- None - elementary convex combination argument

---

#### Step 2: Bound Z-Score Difference (Anchored Approach)

**Goal**: Lower-bound |Z_ρ(x_{i*}) - Z_ρ(x_{j*})| in terms of κ_raw and σ'_{ρ,max}

**Substep 2.1**: Define anchored Z-score
- **Justification**: Introduce Z^ref(u) := (u - μ_ρ[x_ref])/σ'_{ρ,max}
- **Why valid**: Uses fixed reference mean and worst-case denominator (eliminating cross-terms)
- **Expected result**: Z^ref is well-defined with |Z^ref(d_{i*}) - Z^ref(d_{j*})| = |d_{i*} - d_{j*}|/σ'_{ρ,max} ≥ κ_raw/σ'_{ρ,max}

**Substep 2.2**: Bound deviation from anchored to actual Z-score
- **Justification**: For any x, decompose:
  $$
  Z_\rho(x) - Z^{\text{ref}}(d(x)) = \frac{d(x) - \mu_\rho[x]}{\sigma'_\rho[x]} - \frac{d(x) - \mu_\rho[x_{\text{ref}}]}{\sigma'_{\rho,\max}}
  $$
- **Why valid**: Algebra + triangle inequality
- **Expected result**: |Z_ρ(x) - Z^ref(d(x))| ≤ E_ρ(x) where E_ρ depends on |μ_ρ[x] - μ_ρ[x_ref]| and |σ'_ρ[x] - σ'_{ρ,max}|

**Substep 2.3**: Bound E_ρ using regularity
- **Justification**: From C¹ regularity (19, line 514-522):
  - ‖∇μ_ρ‖ ≤ C_μ(ρ) → |μ_ρ[x] - μ_ρ[x_ref]| ≤ C_μ · ‖x - x_ref‖
  - ‖∇σ²_ρ‖ ≤ C_σ(ρ) → similar bound on σ'_ρ
- **Why valid**: Mean Value Theorem applied to smooth functions μ_ρ, σ'_ρ
- **Expected result**: E_ρ ≤ C_μ/σ'_min + C_σ · d_max/σ'^2_min (explicit bound via framework constants)

**Substep 2.4**: Apply triangle inequality to Z-score differences
- **Justification**:
  $$
  |Z_\rho(x_{i*}) - Z_\rho(x_{j*})| \ge |Z^{\text{ref}}(d_{i*}) - Z^{\text{ref}}(d_{j*})| - 2\sup_x E_\rho(x)
  $$
- **Why valid**: |a - b| ≥ |a' - b'| - |a-a'| - |b-b'| (reverse triangle inequality)
- **Expected result**: |Z_ρ(x_{i*}) - Z_ρ(x_{j*})| ≥ κ_raw/σ'_{ρ,max} - 2E_ρ

**Conclusion**: Z-score gap bounded from below with explicit error term

**Dependencies**:
- Uses: lem-rho-pipeline-bounds (σ'_ρ ≤ σ'_{ρ,max})
- Uses: C¹ regularity bounds on μ_ρ, σ'_ρ (framework documents)
- Requires: σ'_reg ≥ ε_σ > 0 (denominator bounded away from zero)

**Potential Issues**:
- ⚠ Error term E_ρ must be controlled; negligible if ρ large or positions close
- **Resolution**: Framework provides explicit C¹/C² bounds; E_ρ can be made arbitrarily small under appropriate localization assumptions

---

#### Step 3: Propagate Through Rescale Function

**Goal**: Bound |d'_{i*} - d'_{j*}| using |Z_ρ(x_{i*}) - Z_ρ(x_{j*})|

**Substep 3.1**: Apply Mean Value Theorem to g_A
- **Justification**: d'_i = g_A(Z_ρ(x_i)), so by MVT:
  $$
  |d'_{i*} - d'_{j*}| = |g'_A(c)| \cdot |Z_\rho(x_{i*}) - Z_\rho(x_{j*})|
  $$
  for some c between the two Z-scores
- **Why valid**: g_A is smooth (lem-rho-pipeline-bounds, Part 2)
- **Expected result**: |d'_{i*} - d'_{j*}| = g'_A(c) · |Z_ρ(x_{i*}) - Z_ρ(x_{j*})|

**Substep 3.2**: Apply lower bound on g'_A
- **Justification**: g'_A(z) ≥ g'_min > 0 for all z in attained range (bounded by d_max and σ'_reg ≥ ε_σ)
- **Why valid**: lem-rho-pipeline-bounds Part 2 (11_geometric_gas.md, line 3187-3211)
- **Expected result**: |d'_{i*} - d'_{j*}| ≥ g'_min · |Z_ρ(x_{i*}) - Z_ρ(x_{j*})|

**Substep 3.3**: Combine with Step 2
- **Justification**: Substitute lower bound from Step 2.4
- **Why valid**: Transitivity of inequalities
- **Expected result**: |d'_{i*} - d'_{j*}| ≥ g'_min · (κ_raw/σ'_{ρ,max} - 2E_ρ)

**Conclusion**: Rescaled pairwise gap bounded below

**Dependencies**:
- Uses: lem-rho-pipeline-bounds Part 2 (g'_A ≥ g'_min)
- Uses: MVT (standard calculus)
- Requires: g_A differentiable with bounded derivative

**Potential Issues**:
- None - straightforward MVT application

---

#### Step 4: Convert Pairwise Gap to Gap-from-Mean

**Goal**: Relate max_{i,j} |d'_i - d'_j| to max_i |d'_i - μ[d']|

**Substep 4.1**: State standard inequality
- **Justification**: For any finite set {y_1,...,y_N} with mean ȳ:
  $$
  \max_i |y_i - \bar{y}| \ge \frac{1}{2} \max_{i,j} |y_i - y_j|
  $$
- **Why valid**: Mean ȳ ∈ [min_i y_i, max_i y_i], so max deviation is at least half the range
- **Expected result**: This is a tight inequality (equality for {0,1})

**Substep 4.2**: Apply to rescaled measurements
- **Justification**: Set y_i = d'_i, use Step 3 lower bound on pairwise gap
- **Why valid**: Standard application
- **Expected result**:
  $$
  \max_i |d'_i - \mu[d']| \ge \frac{1}{2} |d'_{i*} - d'_{j*}| \ge \frac{g'_{\min}}{2} \left(\frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}} - 2E_\rho\right)
  $$

**Conclusion**: Gap-from-mean established with factor ½

**Dependencies**:
- Uses: Standard properties of means
- Requires: None beyond arithmetic

**Potential Issues**:
- ⚠ **Critical**: Factor ½ appears, conflicting with theorem statement
- **Resolution**: See discrepancy analysis in Section V

---

#### Step 5: Final Assembly

**Goal**: Express conclusion in terms of κ_rescaled

**Substep 5.1**: Collect terms
- **Justification**: From Step 4, we have:
  $$
  \max_i |d'_i - \mu[d']| \ge \frac{g'_{\min}}{2} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}} - g'_{\min} \cdot E_\rho
  $$
- **Why valid**: Algebra
- **Expected result**: Lower bound with explicit error term

**Substep 5.2**: Handle E_ρ term
- **Justification**: Under one of:
  - (a) Common-baseline assumption: E_ρ = 0 (all walkers use same μ_ρ, σ'_ρ)
  - (b) Strong localization: E_ρ → 0 as ρ → ∞ or positions cluster
  - (c) Explicit bound: E_ρ ≤ ε for small ε via C¹/C² regularity
- **Why valid**: Framework provides multiple pathways
- **Expected result**: E_ρ negligible or dominated by κ_raw/σ'_{ρ,max}

**Substep 5.3**: Compare with theorem statement
- **Justification**: Theorem claims κ_rescaled = g'_min · κ_raw/σ'_{ρ,max}
- **Why valid**: Our proof gives (g'_min/2) · κ_raw/σ'_{ρ,max} - g'_min · E_ρ
- **Expected result**: **Discrepancy identified** (see Section V)

**Conclusion**: Modulo factor-of-2 issue and E_ρ term, rescaled gap propagation proven

**Q.E.D.** (with caveats) ∎

---

## V. Technical Deep Dives

### Challenge 1: Z-Score Cross-Terms and Anchored Reference

**Why Difficult**: The naive bound |Z_ρ(x_i) - Z_ρ(x_j)| ≥ |d_i - d_j|/σ'_{ρ,max} is INVALID because:
$$
Z_\rho(x_i) = \frac{d_i - \mu_\rho[x_i]}{\sigma'_\rho[x_i]}, \quad Z_\rho(x_j) = \frac{d_j - \mu_\rho[x_j]}{\sigma'_\rho[x_j]}
$$

Subtracting these introduces cross-terms from different μ_ρ and σ'_ρ values.

**Proposed Solution** (GPT-5's approach):
1. Introduce anchored Z^ref(u) := (u - μ_ρ[x_ref])/σ'_{ρ,max} using fixed reference
2. For this, |Z^ref(d_i) - Z^ref(d_j)| = |d_i - d_j|/σ'_{ρ,max} holds exactly
3. Bound |Z_ρ(x) - Z^ref(d(x))| using:
   $$
   |Z_\rho(x) - Z^{\text{ref}}(d(x))| \le \frac{|\mu_\rho[x] - \mu_\rho[x_{\text{ref}}]|}{\sigma'_\rho[x]} + \left|\frac{1}{\sigma'_\rho[x]} - \frac{1}{\sigma'_{\rho,\max}}\right| \cdot |d(x) - \mu_\rho[x_{\text{ref}}]|
   $$
4. Use C¹ bounds on μ_ρ, σ'_ρ to control each term:
   - |μ_ρ[x] - μ_ρ[x_ref]| ≤ C_μ(ρ) · ‖x - x_ref‖ (from ∇μ_ρ bounds, line 514-522)
   - Similar for σ'_ρ variations
5. Get E_ρ ≤ C · (‖x_{i*} - x_ref‖ + ‖x_{j*} - x_ref‖) for explicit C

**Alternative Approach** (if main approach fails):
- Assume all walkers use common baseline (global model limit ρ → ∞)
- Or assume |∇μ_ρ|, |∇σ'_ρ| are small enough that E_ρ << κ_raw/σ'_{ρ,max}
- State lemma with E_ρ term explicitly in conclusion

**References**:
- C¹ regularity of μ_ρ: 19_geometric_gas_cinf_regularity_simplified.md, lem-mean-cinf-inductive (line 451-488)
- C¹ regularity of σ²_ρ: lem-variance-cinf-inductive (line 491-508)

---

### Challenge 2: Factor-of-2 Discrepancy

**Why Difficult**: The rigorous proof yields:
$$
\max_i |d'_i - \mu[d']| \ge \frac{1}{2} \cdot g'_{\min} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}}
$$

But theorem states κ_rescaled = g'_min · κ_raw/σ'_{ρ,max} (no factor ½).

**Why the Factor Appears**:
The inequality max_i |y_i - ȳ| ≥ ½ max_{i,j} |y_i - y_j| is TIGHT. Example:
- Set {0, 1}: ȳ = ½, max_{i,j} |y_i - y_j| = 1, max_i |y_i - ȳ| = ½
- Equality holds; cannot improve

**Proposed Resolution**:
1. **Option A**: Theorem statement should include factor ½:
   $$
   \kappa_{\text{rescaled}} := \frac{1}{2} g'_{\min} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}}
   $$
   This makes the proof rigorous as-is.

2. **Option B**: Add structural assumption that μ[d'] coincides with an extremal value:
   - If μ[d'] = min_i d'_i or max_i d'_i, then max_i |d'_i - μ[d']| = max_{i,j} |d'_i - d'_j|
   - This requires special measurement distribution (unlikely in general)

3. **Option C**: Redefine theorem to bound pairwise gap:
   $$
   \max_{i,j} |d'_i - d'_j| \ge g'_{\min} \cdot \frac{\kappa_{\text{raw}}}{\sigma'_{\rho,\max}}
   $$
   Then separately apply mean reduction with factor ½ when needed

**Recommendation**: **Option A** (correct theorem statement) is most mathematically honest. Both Gemini and GPT-5 independently identify this issue, suggesting it's a genuine discrepancy, not a proof error.

**References**:
- Both strategists flag this (Gemini Step 6, GPT-5 Step 4)
- Standard statistical inequality literature (e.g., Hoeffding's lemma variations)

---

### Challenge 3: Uniformity of g'_min Bound

**Why Difficult**: The bound g'_A(z) ≥ g'_min must hold on the **actual range** of Z-scores attained by the walkers, not just asymptotically.

**Proposed Solution**:
1. Bound Z-score range: Since d ∈ [0, d_max] (compact X) and σ'_reg ≥ ε_σ > 0:
   $$
   |Z_\rho(x)| \le \frac{|d(x) - \mu_\rho| + |\mu_\rho|}{\sigma'_{\rho,\min}} \le \frac{2d_{\max}}{\varepsilon_\sigma}
   $$
2. g_A is smooth on compact domain → g'_A is continuous and bounded
3. On interval [-Z_max, Z_max], define g'_min := min_{|z| ≤ Z_max} g'_A(z)
4. Monotonicity of g_A (assumed) → g'_A > 0 everywhere → g'_min > 0

**Alternative Approach** (if g_A has inflection points):
- Explicitly compute g'_A for chosen sigmoid (e.g., tanh, logistic)
- Verify g'_A ≥ g'_min on attained range numerically or analytically

**References**:
- Regularization bound σ'_reg ≥ ε_σ: framework axiom (01_fragile_gas_framework.md, line 2848)
- Bounded measurements: X compact, d continuous

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps justified; anchored Z-score resolves cross-term issue
- [x] **Hypothesis Usage**: Raw gap assumption drives entire derivation
- [⚠] **Conclusion Derivation**: Derived with factor ½ (see Challenge 2)
- [x] **Framework Consistency**: All dependencies verified (lem-rho-pipeline-bounds, C¹ regularity)
- [x] **No Circular Reasoning**: Proof flows from primitive bounds to conclusion
- [x] **Constant Tracking**: κ_rescaled explicitly defined; ρ-dependence through σ'_{ρ,max}
- [⚠] **Edge Cases**: E_ρ term requires control; negligible under localization assumptions
- [x] **Regularity Verified**: g_A smooth, monotone; μ_ρ, σ'_ρ ∈ C¹ from framework
- [x] **Measure Theory**: Not applicable (finite-sample inequality)

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Variance Route

**Approach**: Convert raw gap to raw variance, propagate variance through pipeline, apply variance-to-gap at end

**Pros**:
- Avoids pairwise comparisons entirely
- Variance propagation may have cleaner constants
- Connects to lem-variance-to-gap-adaptive directly

**Cons**:
- Gap-to-variance conversion is loose: Var(d) ≥ κ²_raw/N (depends on N)
- Variance propagation through nonlinear g_A requires Lipschitz analysis (complex)
- Likely still has factor-of-type constants in final gap-from-variance conversion

**When to Consider**: If pairwise approach fails or if variance bounds are already available from other results

---

### Alternative 2: Direct Gap-from-Mean Bound (No Pairwise)

**Approach**: Bound |d'_{i*} - μ[d']| directly without creating pairwise gap intermediate

**Pros**:
- Could potentially avoid factor ½
- More direct connection from hypothesis to conclusion

**Cons**:
- Requires bounding |d'_{i*} - (1/N)∑_j d'_j| = |(1/N)∑_j (d'_{i*} - d'_j)|
- Sum of signed terms with potential cancellation
- Need control on entire distribution of d'_j, not just one pair
- Significantly more complex than pairwise approach

**When to Consider**: If factor-of-2 issue is critical and cannot be resolved by correcting theorem

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Factor-of-2 resolution**: Clarify whether theorem statement needs correction or proof has tighter approach
2. **E_ρ term control**: Formalize Lemma L3 with explicit constants from C¹/C² bounds
3. **Optimality**: Is factor ½ tight, or can it be improved under additional distributional assumptions?

### Conjectures

1. **Improved constant under symmetry**: If distribution of d' is symmetric about μ[d'], the factor ½ might be improvable to 2/3 or even 1
2. **ρ-dependence of E_ρ**: E_ρ → 0 as ρ → ∞ (global limit) with explicit rate O(1/ρ) from Gaussian kernel decay

### Extensions

1. **Multivariate measurements**: Extend to d: X → ℝ^m with component-wise gaps
2. **Non-monotone g_A**: Handle rescale functions with critical points (requires different MVT application)
3. **Time-dependent gaps**: Track κ_rescaled(t) under adaptive dynamics

---

## IX. Expansion Roadmap

**Phase 1: Prove Supporting Lemmas** (Estimated: 3 hours)

1. **Lemma L1**: Weighted-average to pairwise gap - Formalize convex combination argument
2. **Lemma L2**: Pair-to-mean reduction with tight factor ½ - Provide two-point example showing tightness
3. **Lemma L3**: Anchored Z-score control - Extract C_μ, C_σ from framework docs; bound E_ρ explicitly

**Phase 2: Resolve Factor-of-2 Discrepancy** (Estimated: 2 hours)

1. Consult framework authors or trace through dependencies in 03_cloning.md
2. Check if global model (ρ → ∞) has factor 1 and localization introduces ½
3. Either correct theorem statement or identify structural assumption that eliminates factor

**Phase 3: Fill Technical Details** (Estimated: 3 hours)

1. Step 2.3: Explicit computation of E_ρ using C¹ bounds from lines 514-522, 491-508
2. Step 3.2: Verify g'_min on bounded Z-range using σ'_reg ≥ ε_σ
3. Add numerical examples for specific sigmoid functions (tanh, logistic)

**Phase 4: Review and Validation** (Estimated: 1 hour)

1. Framework cross-validation: Verify all line number citations
2. Edge case verification: k = 1 (single walker), κ_raw → 0 limit
3. Constant tracking: Trace ρ-dependence through all steps

**Total Estimated Expansion Time**: 9 hours to full publication-ready proof (with factor-of-2 resolution)

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`lem-rho-pipeline-bounds` - Bounds on σ'_ρ and g'_A
- {prf:ref}`lem-variance-to-gap-adaptive` - Alternative approach (not chosen)

**Definitions Used**:
- Localization weights (19_geometric_gas_cinf_regularity_simplified.md, line 262-272)
- Z-score (19_geometric_gas_cinf_regularity_simplified.md, line 290-348)
- Rescale function g_A (11_geometric_gas.md, line 3187-3211)

**Related Proofs** (for comparison):
- Global model version (03_cloning.md) - Non-ρ-dependent analog
- C¹ regularity of μ_ρ ({prf:ref}`lem-mean-cinf-inductive`) - Provides bounds for E_ρ term
- C¹ regularity of σ²_ρ ({prf:ref}`lem-variance-cinf-inductive`) - Provides bounds for E_ρ term

**Downstream Dependencies**:
- {prf:ref}`lem-log-gap-bounds-adaptive` - Uses rescaled gap for intelligent targeting
- {prf:ref}`thm-keystone-adaptive` - Relies on signal propagation established here

---

**Proof Sketch Completed**: 2025-10-25 09:05
**Ready for Expansion**: Needs resolution of factor-of-2 issue and Lemma L3
**Confidence Level**: Medium-High - Core approach is sound (both strategists agree); factor-of-2 discrepancy needs clarification; E_ρ term requires explicit computation but pathway is clear via existing C¹/C² bounds.
