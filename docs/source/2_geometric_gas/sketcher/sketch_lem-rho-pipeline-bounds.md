# Proof Sketch for lem-rho-pipeline-bounds

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/11_geometric_gas.md
**Lemma**: lem-rho-pipeline-bounds
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Lemma Statement

:::{prf:lemma} Uniform Bounds on the ρ-Localized Pipeline
:label: lem-rho-pipeline-bounds

For the ρ-localized rescaling pipeline with bounded measurements d ∈ [0, d_max]:

**1. Upper Bound on Localized Standard Deviation:**

$$
\sigma'_\rho[f, d, x] \le \sigma'_{\rho,\max} := d_{\max}
$$

for all f, x, ρ. This bound is **N-uniform** and **ρ-dependent** (it could be tighter for specific ρ, but this worst-case bound suffices).

**2. Lower Bound on Rescale Derivative:**

$$
g'_A(z) \ge g'_{\min} > 0
$$

for all z ∈ ℝ, where g_A is the smooth, monotone rescale function. This bound is **ρ-independent**.

:::

**Informal Restatement**: For the ρ-localized measurement rescaling pipeline, two key constants can be bounded uniformly: (1) the localized standard deviation never exceeds the measurement range, and (2) the rescale function derivative is bounded below by a positive constant on the relevant Z-score domain.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: Gemini returned empty response. No strategy available.

---

### Strategy B: GPT-5's Approach

**Method**: Direct bounding for Part 1 (measurement range) + Compactness argument for Part 2 (derivative lower bound)

**Key Steps**:
1. Bound μ_ρ within [0, d_max] via convex combination
2. Bound σ_ρ and σ'_ρ using measurement range
3. Uniformly bound Z-score using bounded numerator and positive denominator
4. Derive ρ-independent lower bound on g'_A via compactness
5. Verify N-uniformity and ρ-independence of constants

**Strengths**:
- Elementary and self-contained
- Identifies critical assumption: κ_var,min ≤ d_max
- Notes quantifier mismatch in g'_A bound (relevant range vs all of ℝ)
- Systematic verification of uniformity claims
- Clear framework assumption checking

**Weaknesses**:
- Requires clarification of g'_A domain (Z-range vs all ℝ)
- Depends on variance regularization parameter constraint

**Framework Dependencies**:
- Measurement boundedness d ∈ [0, d_max]
- Normalized weights for localized mean
- Variance regularization κ_var,min > 0
- Smooth monotone rescale function g_A

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Direct bounding + compactness (GPT-5's approach)

**Rationale**:
This is the natural and correct strategy for this lemma. The proof is already provided in the document (lines 3196-3207), and GPT-5's analysis correctly formalizes it while identifying implicit assumptions.

**Integration**:
- Use GPT-5's systematic 5-step breakdown
- Address the quantifier clarification (g'_A on Z-range, not all ℝ)
- Document the κ_var,min ≤ d_max assumption explicitly

**Verification Status**:
- ✅ Proof complete in document
- ✅ Framework assumptions identified
- ⚠ Requires κ_var,min ≤ d_max (or alternative formulation)
- ⚠ g'_A bound applies to Z-range (should clarify in theorem statement)

---

## III. Framework Dependencies

### Verified Dependencies

**Framework Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| Localized mean μ_ρ | 11_geometric_gas.md:235 | Convex combination with normalized weights | Part 1, Step 1 |
| Localized variance σ²_ρ | 11_geometric_gas.md:2543 | Variance of measurements under localized weights | Part 1, Step 2 |
| Regularized std σ'_ρ | 11_geometric_gas.md:2860 | max{σ_ρ, κ_var,min} or √(V + σ'²_min) | Part 1, Step 2 |
| Z-score Z_ρ | 11_geometric_gas.md:692 | (d(x) - μ_ρ)/σ'_ρ | Part 2, Step 3 |
| Rescale g_A | 11_geometric_gas.md:481 | Smooth, monotone increasing | Part 2, Step 4 |

**Framework Axioms**:
| Axiom | Statement | Used in | Verified |
|-------|-----------|---------|----------|
| Bounded measurements | d: X → [0, d_max] | Steps 1-3 | ✅ |
| Normalized weights | ∑_j w_ij(ρ) = 1, w_ij ≥ 0 | Step 1 | ✅ |
| Positive regularization | κ_var,min > 0 or σ'_min > 0 | Steps 2-3 | ✅ |
| Smooth monotone g_A | g_A ∈ C¹, g'_A > 0 | Step 4 | ✅ |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| d_max | Max measurement | Range of d | N-uniform, ρ-independent |
| κ_var,min | Variance regularizer | Positive constant | Assumed ≤ d_max |
| g'_min | Min rescale derivative | min\_{z∈[-Z_max, Z_max]} g'_A(z) | ρ-independent, Z_max-dependent |
| Z_max | Max Z-score | d_max/κ_var,min | ρ-independent |

### Missing/Uncertain Dependencies

**Requires Explicit Assumption**:
- κ_var,min ≤ d_max (or relax to σ'_ρ ≤ max{d_max, κ_var,min})
- g'_A strictly positive on compact interval [-Z_max, Z_max] (not all ℝ)

**Potential Sharpening**:
- Use Popoviciu's inequality: σ_ρ ≤ d_max/2 for tighter bound

---

## IV. Detailed Proof Sketch

### Overview

The lemma establishes two fundamental bounds for the ρ-localized rescaling pipeline. Part 1 uses direct bounding via measurement range: localized statistics are convex combinations of bounded values, so deviations remain bounded. Part 2 uses a compactness argument: the Z-score has bounded range (from Part 1 and regularization), and g'_A is continuous on this compact interval, hence has a positive minimum.

Both bounds are uniform in N (no swarm-size dependence) and apply for all ρ (worst-case over localization scales).

### Proof Outline (Top-Level)

1. **Bound Localized Mean**: μ_ρ ∈ [0, d_max] via convex combination
2. **Bound Localized Variance**: σ_ρ, σ'_ρ ≤ d_max via measurement range
3. **Bound Z-Score**: |Z_ρ| ≤ d_max/κ_var,min via bounded numerator and positive denominator
4. **Bound Derivative**: g'_A ≥ g'_min > 0 via compactness on Z-range
5. **Verify Uniformity**: Check N-uniformity and ρ-independence

---

### Detailed Step-by-Step Sketch

#### Step 1: Bound μ_ρ within [0, d_max]

**Goal**: Show localized mean stays within measurement range

**Substep 1.1**: Invoke normalized weight property
- **Justification**: Line 235 states weights sum to 1: ∑_j w_ij(ρ) = 1
- **Why valid**: Kernel-weighted normalization by definition
- **Expected result**: Weights form a probability distribution

**Substep 1.2**: Apply convex combination bound
- **Justification**: μ_ρ[f, d, x] = ∑_j w_ij(ρ) d(x_j) with d(x_j) ∈ [0, d_max]
- **Why valid**: Convex combination of values in [a, b] lies in [a, b]
- **Expected result**: μ_ρ ∈ [0, d_max]

**Conclusion**: Localized mean bounded by measurement range
- **Form**: 0 ≤ μ_ρ[f, d, x] ≤ d_max

**Dependencies**:
- Uses: Normalized weights (line 235), measurement boundedness
- Requires: w_ij(ρ) ≥ 0, ∑_j w_ij = 1

**Potential Issues**:
- None (elementary convexity)

---

#### Step 2: Bound σ_ρ and σ'_ρ

**Goal**: Establish upper bound on regularized standard deviation

**Substep 2.1**: Bound σ_ρ via measurement range
- **Justification**: For any random variable on [0, d_max], standard deviation ≤ range
- **Why valid**: σ_ρ² = E[(d - μ_ρ)²] where |d - μ_ρ| ≤ d_max (from Step 1)
- **Expected result**: σ_ρ ≤ d_max

**Substep 2.2**: Apply regularization
- **Justification**: σ'_ρ = max{σ_ρ, κ_var,min} (line 2860)
- **Why valid**: Definition of regularized standard deviation
- **Expected result**: σ'_ρ = max{σ_ρ, κ_var,min} ≤ max{d_max, κ_var,min}

**Substep 2.3**: Invoke parameter assumption
- **Justification**: Assume κ_var,min ≤ d_max (natural design choice)
- **Why valid**: Regularization should not exceed measurement range
- **Expected result**: σ'_ρ ≤ d_max

**Conclusion**: σ'_ρ ≤ σ'_ρ,max := d_max
- **Form**: Worst-case bound uniform in f, x, ρ

**Dependencies**:
- Uses: Step 1, variance regularization definition
- Requires: κ_var,min ≤ d_max (should be stated explicitly)

**Potential Issues**:
- ⚠ If κ_var,min > d_max, bound becomes σ'_ρ ≤ κ_var,min
- **Resolution**: Assume parameter constraint or relax bound

---

#### Step 3: Uniformly Bound Z-Score

**Goal**: Establish compact range for Z_ρ

**Substep 3.1**: Bound Z-score numerator
- **Justification**: |d(x) - μ_ρ| ≤ max(d_max - 0, d_max - 0) = d_max
- **Why valid**: Both d(x) and μ_ρ in [0, d_max] (from Steps 1-2)
- **Expected result**: |d(x) - μ_ρ| ≤ d_max

**Substep 3.2**: Bound Z-score denominator
- **Justification**: σ'_ρ ≥ κ_var,min > 0 by regularization (line 692)
- **Why valid**: Regularization ensures positive denominator
- **Expected result**: σ'_ρ bounded below

**Substep 3.3**: Combine to bound Z-score
- **Justification**: |Z_ρ| = |d(x) - μ_ρ|/σ'_ρ ≤ d_max/κ_var,min
- **Why valid**: Ratio of bounded numerator and positive denominator
- **Expected result**: Z_ρ ∈ [-Z_max, Z_max] where Z_max := d_max/κ_var,min

**Conclusion**: Z-score has compact ρ-independent range
- **Form**: |Z_ρ[f, d, x]| ≤ Z_max for all f, d, x, ρ

**Dependencies**:
- Uses: Steps 1-2, variance regularization positivity
- Requires: κ_var,min > 0

**Potential Issues**:
- None (direct from definitions)

---

#### Step 4: Derive ρ-Independent Lower Bound on g'_A

**Goal**: Establish g'_min > 0 on relevant Z-range

**Substep 4.1**: Identify compact domain
- **Justification**: From Step 3, Z_ρ ∈ [-Z_max, Z_max]
- **Why valid**: Z-score range is compact and ρ-independent
- **Expected result**: Relevant domain K = [-Z_max, Z_max]

**Substep 4.2**: Invoke smoothness of g_A
- **Justification**: g_A smooth, monotone increasing (line 481)
- **Why valid**: Framework assumption on rescale function
- **Expected result**: g'_A continuous on ℝ

**Substep 4.3**: Apply strict positivity assumption
- **Justification**: Monotone increasing implies g'_A ≥ 0; assume g'_A > 0 on K
- **Why valid**: Standard choice (e.g., logistic, tanh) has g'_A > 0 everywhere or on relevant range
- **Expected result**: g'_A strictly positive on compact K

**Substep 4.4**: Extract minimum via compactness
- **Justification**: Continuous positive function on compact set has positive minimum
- **Why valid**: Extreme value theorem
- **Expected result**: g'_min := min_{z ∈ K} g'_A(z) > 0

**Conclusion**: g'_A(z) ≥ g'_min > 0 for all z in Z-range
- **Form**: ρ-independent bound (K is ρ-independent)

**Dependencies**:
- Uses: Step 3 (compact Z-range), smoothness of g_A
- Requires: g'_A > 0 on [-Z_max, Z_max]

**Potential Issues**:
- ⚠ Theorem statement says "for all z ∈ ℝ" but proof uses compact range
- **Resolution**: Clarify statement to "for all z in range of Z_ρ" or assume global g'_A > 0

---

#### Step 5: Verify N-Uniformity and ρ-Independence

**Goal**: Confirm claimed uniformity properties

**Substep 5.1**: Check N-uniformity
- **Justification**: All constants (d_max, κ_var,min, g'_min) independent of N, k
- **Why valid**: No swarm-size factors appear in bounds
- **Expected result**: Bounds uniform in N

**Substep 5.2**: Check ρ-dependence
- **Justification**: Bounds hold for all ρ; constants worst-case over ρ
- **Why valid**: Localized statistics enter only through μ_ρ, σ_ρ which are bounded uniformly
- **Expected result**: Bounds ρ-independent (could be tightened for specific ρ)

**Conclusion**: Both parts satisfy claimed uniformity
- **Form**: N-uniform, ρ-worst-case bounds

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Quantifier Mismatch on g'_A Domain

**Why Difficult**: Theorem statement claims g'_A(z) ≥ g'_min "for all z ∈ ℝ", but the proof mechanism only establishes this on the compact Z-range [-Z_max, Z_max]. Many standard rescale functions (e.g., logistic) have inf_{z∈ℝ} g'_A(z) = 0.

**Proposed Solution**:
1. **Clarify theorem statement**: Replace "for all z ∈ ℝ" with "for all z in the range of Z_ρ[f, d, ·]"
2. **Justify scope**: The bound is only used when applying g'_A to Z-scores, so this is sufficient
3. **ρ-independence**: The Z-range [-Z_max, Z_max] is ρ-independent (depends only on d_max, κ_var,min)
4. **Document**: State explicitly in proof that g'_min = min_{|z| ≤ Z_max} g'_A(z)

**Alternative Approach**: Assume g_A has globally bounded derivative away from zero (stronger assumption than needed).

---

### Challenge 2: κ_var,min ≤ d_max Assumption

**Why Difficult**: The bound σ'_ρ ≤ d_max requires κ_var,min ≤ d_max, but this isn't stated explicitly in the theorem hypotheses.

**Proposed Solution**:
1. **Add to hypotheses**: State κ_var,min ≤ d_max as a parameter constraint
2. **Natural constraint**: Regularization floor should not exceed measurement range
3. **Alternative formulation**: Relax bound to σ'_ρ ≤ max{d_max, κ_var,min}
4. **Additive regularization**: If using σ'_reg(V) = √(V + σ'²_min), requires σ'_min ≤ (√3/2) d_max

**Mathematical Detail**: With σ'_ρ = max{σ_ρ, κ_var,min}:
- If κ_var,min ≤ d_max: σ'_ρ ≤ max{d_max, κ_var,min} = d_max
- If κ_var,min > d_max: σ'_ρ ≤ max{d_max, κ_var,min} = κ_var,min

---

## VI. Proof Validation Checklist

- [x] **Part 1 proven**: σ'_ρ ≤ d_max (under κ_var,min ≤ d_max)
- [x] **Part 2 proven**: g'_A ≥ g'_min > 0 on Z-range
- [x] **N-uniformity verified**: All constants independent of N, k
- [x] **ρ-dependence characterized**: Worst-case bounds valid for all ρ
- [⚠] **Edge cases**: Requires κ_var,min ≤ d_max assumption
- [⚠] **Domain clarification**: g'_A bound on Z-range, not all ℝ

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Sharpen with Popoviciu's Inequality

**Approach**: Use Popoviciu bound σ_ρ ≤ d_max/2 for support in [0, d_max] to tighten Part 1

**Pros**:
- Tighter bound (factor 2 improvement)
- Smaller Z_max → potentially larger g'_min

**Cons**:
- More complex proof
- Worst-case d_max bound already sufficient

**When to Consider**: If downstream results benefit from tighter constants

---

## VIII. Open Questions

### Remaining Gaps
1. **Explicit parameter constraint**: Add κ_var,min ≤ d_max to hypotheses
2. **g'_A domain**: Clarify "for all z in Z-range" vs "for all z ∈ ℝ"

### Extensions
1. **ρ-dependent tightening**: Derive ρ-specific bounds for σ'_ρ,max(ρ)
2. **Optimal regularization**: Characterize optimal κ_var,min choice

---

## IX. Expansion Roadmap

**Phase 1: Clarify Assumptions** (Estimated: 1 hour)
1. Add κ_var,min ≤ d_max to theorem hypotheses
2. Adjust g'_A bound statement to Z-range

**Phase 2: Optional Sharpening** (Estimated: 2 hours)
1. Apply Popoviciu's inequality for σ_ρ ≤ d_max/2
2. Derive explicit g'_min for standard g_A choices

**Total Estimated Expansion Time**: 3 hours

---

## X. Cross-References

**Definitions Used**:
- Localized mean μ_ρ (line 235)
- Localized variance σ²_ρ (line 2543)
- Regularized std σ'_ρ (line 2860)
- Z-score Z_ρ (line 692)
- Rescale g_A (line 481)

**Related Lemmas**:
- lem-variance-to-gap-adaptive (line 3160)
- lem-raw-to-rescaled-gap-rho (line 3212)

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Yes (requires minor clarifications)
**Confidence Level**: High - Proof is complete and elementary. Main tasks are clarifying domain and parameter assumptions.
