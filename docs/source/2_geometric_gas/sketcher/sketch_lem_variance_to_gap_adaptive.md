# Proof Sketch for lem-variance-to-gap-adaptive

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/11_geometric_gas.md
**Lemma**: lem-variance-to-gap-adaptive
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:lemma} Variance-to-Gap (from 03_cloning.md, Lemma 7.3.1)
:label: lem-variance-to-gap-adaptive

For any random variable $X$ with mean $\mu$ and variance $\sigma^2 > 0$:

$$
\max_{x \in \text{supp}(X)} |x - \mu| \ge \sigma
$$

**Proof:** This is a general statistical inequality that holds for any probability distribution. See `03_cloning.md`, Lemma 7.3.1 for the proof.
:::

**Informal Restatement**: If a random variable has positive variance σ², then at least one value in its support (the set of values it can take) must be at least distance σ away from the mean μ. This establishes a fundamental connection between spread (variance) and extreme deviation (maximum gap).

**Context in Framework**: This lemma appears in Appendix B.3.1 of the Geometric Gas document, titled "The Variance-to-Gap Lemma (Universal)". It is explicitly noted as "ρ-independent and applies universally" - meaning it is a pure statistical fact that does not depend on the localization scale or any algorithm-specific parameters. The lemma bridges variance guarantees (which come from geometric analysis) to gap guarantees (which are needed for signal propagation in the rescaling pipeline).

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: Gemini did not provide output (empty response received)

**Impact**: This means we only have Codex's strategy to work with. The dual verification protocol cannot provide cross-validation in this case.

---

### Strategy B: Codex (GPT-5)'s Approach

**Method**: Direct proof from variance definition

**Key Steps**:
1. **Formalize the left side as a radius**: Define $R := \sup_{x \in \text{supp}(X)} |x - \mu| \in [0, \infty]$
2. **Bound the variance by the radius**: Show $|X - \mu| \le R$ almost surely, hence $(X - \mu)^2 \le R^2$ a.s., thus $\sigma^2 = \mathbb{E}[(X - \mu)^2] \le R^2$
3. **Conclude $\sigma \le R$**: Take square roots to get $\sigma \le R$, which is equivalent to the statement
4. **Universality verification**: Confirm the argument works for discrete and continuous distributions
5. **Tightness demonstration**: Exhibit equality case with symmetric two-point distribution

**Strengths**:
- **Elementary and direct**: Uses only the variance definition and monotonicity of expectation
- **Minimal assumptions**: No boundedness, smoothness, or regularity required
- **Universal applicability**: Works for discrete, continuous, and mixed distributions
- **Tight bound**: Provides explicit distribution achieving equality
- **Handles unbounded case**: Correctly interprets max as sup for unbounded support

**Weaknesses**:
- **Max vs Sup ambiguity**: Requires careful interpretation when support is unbounded
- **Topological subtlety**: Distinction between topological support and essential range needs clarification

**Framework Dependencies**:
- Variance definition: $\sigma^2 = \mathbb{E}[(X - \mu)^2]$
- Monotonicity of expectation
- Definition of support (topological closure for continuous, atoms for discrete)
- Extreme value theorem (for bounded support compactness)

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Direct proof from variance definition (Codex's approach)

**Rationale**:
With only one strategist providing output, I adopt Codex's approach after independent verification:

1. **Mathematical validity**: The proof is completely sound - it uses only the variance definition and basic properties of expectation
2. **Simplicity**: This is the most direct route with minimal machinery
3. **Generality**: No additional assumptions beyond the lemma's hypotheses (finite variance σ² > 0)
4. **Tightness**: The explicit equality case confirms the bound cannot be improved

**Integration**:
- Steps 1-3: From Codex's strategy (verified independently)
- Step 4: Added clarification on max vs sup interpretation
- Step 5: Added explicit connection to framework usage

**Critical Enhancements** (Claude's additions):
1. **Precise max/sup handling**: Clarify that for bounded support, max exists by compactness; for unbounded support, the inequality is trivially satisfied (R = ∞)
2. **Essential supremum vs topological support**: Document that using topological support only strengthens the bound (it's larger than essential supremum)
3. **Connection to Chebyshev**: Note this is fundamentally different - Chebyshev bounds tail probabilities, while this bounds support extent

**Verification Status**:
- ✅ All framework dependencies verified (standard probability theory)
- ✅ No circular reasoning detected (uses only variance definition)
- ✅ Universality confirmed (discrete, continuous, mixed distributions)
- ✅ Tightness verified (explicit equality case provided)
- ⚠️ Interpretation note: "max" should be understood as "sup" for unbounded support

---

## III. Framework Dependencies

### Verified Dependencies

**Definitions** (standard probability theory):

| Concept | Definition | Used in Step | Verified |
|---------|------------|--------------|----------|
| Variance | $\sigma^2 = \mathbb{E}[(X - \mu)^2]$ | Step 2 | ✅ |
| Support (discrete) | $\text{supp}(X) = \{x : P(X = x) > 0\}$ | Step 1 | ✅ |
| Support (continuous) | Closure of $\{x : f_X(x) > 0\}$ | Step 1 | ✅ |
| Essential supremum | $\text{ess sup } Y = \inf\{M : P(|Y| \le M) = 1\}$ | Step 2 | ✅ |

**Standard Results**:

| Result | Statement | Used in Step | Verified |
|--------|-----------|--------------|----------|
| Monotonicity of expectation | If $Y_1 \le Y_2$ a.s., then $\mathbb{E}[Y_1] \le \mathbb{E}[Y_2]$ | Step 2 | ✅ |
| Extreme value theorem | Continuous function on compact set attains max | Step 1 (bounded case) | ✅ |
| Square root monotonicity | If $0 \le a \le b$, then $\sqrt{a} \le \sqrt{b}$ | Step 3 | ✅ |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $R$ | $\sup_{x \in \text{supp}(X)} \|x - \mu\|$ | $[0, \infty]$ | Supremum over support |
| $\sigma$ | Standard deviation $\sqrt{\text{Var}(X)}$ | $> 0$ (by hypothesis) | Given finite and positive |
| $\mu$ | Mean $\mathbb{E}[X]$ | Finite (implicit) | Must exist for variance to be finite |

### Missing/Uncertain Dependencies

**None** - This is a self-contained statistical result requiring no framework-specific axioms.

---

## IV. Detailed Proof Sketch

### Overview

The proof is remarkably simple and relies on a fundamental observation: variance measures the expected squared deviation from the mean, while the maximum gap measures the worst-case deviation. Since variance is an average of squared deviations, it cannot exceed the square of the maximum deviation - otherwise, the maximum wouldn't be maximal. The formal proof makes this intuition precise using the variance definition and monotonicity of expectation.

The key insight is that if all values in the support were within distance $\sigma - \epsilon$ of the mean (for some $\epsilon > 0$), then the variance would be bounded by $(\sigma - \epsilon)^2 < \sigma^2$, contradicting the hypothesis. Therefore, at least one value must be at least distance $\sigma$ away.

### Proof Outline (Top-Level)

The proof proceeds in 3 main stages:

1. **Define radius**: Formalize the left-hand side as $R := \sup_{x \in \text{supp}(X)} |x - \mu|$
2. **Variance bound**: Show $\sigma^2 \le R^2$ using the variance definition
3. **Conclude**: Take square roots to obtain $\sigma \le R$, equivalent to the statement

---

### Detailed Step-by-Step Sketch

#### Step 1: Define the Support Radius

**Goal**: Formalize "max over support" as a well-defined supremum

**Substep 1.1**: Define the radius
- **Action**: Let $R := \sup_{x \in \text{supp}(X)} |x - \mu| \in [0, \infty]$
- **Justification**: Supremum always exists in extended reals $[0, \infty]$ for any non-empty set
- **Why valid**: Support is non-empty (variance σ² > 0 implies support contains at least two distinct points)
- **Expected result**: $R$ is well-defined, possibly infinite

**Substep 1.2**: Interpret max vs sup
- **Action**: Distinguish two cases:
  - **Bounded support**: If $R < \infty$, support is contained in compact interval $[\mu - R, \mu + R]$
  - **Unbounded support**: If $R = \infty$, inequality $R \ge \sigma$ is trivially true
- **Justification**:
  - Bounded case: Closed bounded subset of ℝ is compact; continuous function $x \mapsto |x - \mu|$ attains maximum by extreme value theorem, so $\max = \sup$
  - Unbounded case: Infinity is greater than any finite σ
- **Why valid**: Standard topology of ℝ; extreme value theorem
- **Expected result**: For bounded support, "max" is well-defined and equals $R$; for unbounded support, inequality holds trivially

**Dependencies**:
- Uses: Definition of support (closed set)
- Requires: Extreme value theorem (standard analysis)

**Potential Issues**:
- ⚠️ **Potential problem**: Original statement uses "max" which may not exist for unbounded support
- **Resolution**: Interpret as supremum; for unbounded support the inequality is trivially satisfied

---

#### Step 2: Bound Variance by Squared Radius

**Goal**: Show $\sigma^2 = \mathbb{E}[(X - \mu)^2] \le R^2$

**Substep 2.1**: Establish pointwise bound
- **Action**: Note that by definition of $R$ as supremum over support:

$$
|x - \mu| \le R \quad \text{for all } x \in \text{supp}(X)
$$

- **Justification**: $R = \sup_{x \in \text{supp}(X)} |x - \mu|$ is the least upper bound
- **Why valid**: Definition of supremum
- **Expected result**: Almost surely, $|X - \mu| \le R$

**Substep 2.2**: Square the inequality
- **Action**: Since $|X - \mu| \le R$ almost surely, we have:

$$
(X - \mu)^2 \le R^2 \quad \text{almost surely}
$$

- **Justification**: Squaring preserves inequality for non-negative values
- **Why valid**: Standard algebra; $|X - \mu|$ and $R$ are both non-negative
- **Expected result**: Pointwise bound on squared deviation

**Substep 2.3**: Take expectation
- **Action**: Apply expectation to both sides:

$$
\mathbb{E}[(X - \mu)^2] \le \mathbb{E}[R^2] = R^2
$$

- **Justification**: Monotonicity of expectation; $R^2$ is constant
- **Why valid**: If $Y_1 \le Y_2$ almost surely and both have finite expectation, then $\mathbb{E}[Y_1] \le \mathbb{E}[Y_2]$
- **Expected result**: $\sigma^2 \le R^2$

**Substep 2.4**: Handle unbounded case
- **Action**: If $R = \infty$, then $R^2 = \infty$ and inequality $\sigma^2 \le \infty$ holds trivially
- **Justification**: Finite variance cannot exceed infinity
- **Why valid**: $\sigma^2$ is finite by hypothesis
- **Expected result**: Inequality holds in both bounded and unbounded cases

**Dependencies**:
- Uses: Variance definition, monotonicity of expectation
- Requires: $\sigma^2 < \infty$ (hypothesis), supremum definition

**Potential Issues**:
- ⚠️ **Potential problem**: Topological support vs essential support distinction
- **Resolution**: Essential supremum satisfies $\text{ess sup} |X - \mu| \le \sup_{x \in \text{supp}(X)} |x - \mu| = R$. Using the (possibly larger) topological support only strengthens the inequality, preserving correctness.

---

#### Step 3: Conclude $\sigma \le R$

**Goal**: Derive the final inequality from the variance bound

**Substep 3.1**: Take square root
- **Action**: From $\sigma^2 \le R^2$ with $\sigma, R \ge 0$, conclude:

$$
\sigma \le R
$$

- **Justification**: Square root is monotone increasing on $[0, \infty]$
- **Why valid**: Standard property of square root function
- **Expected result**: $\sigma \le \sup_{x \in \text{supp}(X)} |x - \mu|$

**Substep 3.2**: Rewrite as max (bounded case)
- **Action**: If support is bounded, $R = \max_{x \in \text{supp}(X)} |x - \mu|$ by Step 1
- **Justification**: Extreme value theorem on compact support
- **Why valid**: Continuous function on compact set attains supremum
- **Expected result**: $\sigma \le \max_{x \in \text{supp}(X)} |x - \mu|$ (statement proven)

**Substep 3.3**: Verify unbounded case
- **Action**: If support is unbounded, $R = \infty > \sigma$, so inequality holds
- **Justification**: Any finite number is less than infinity
- **Why valid**: Standard order on extended reals
- **Expected result**: Inequality holds for all cases

**Dependencies**:
- Uses: Monotonicity of square root, extreme value theorem
- Requires: Case analysis from Step 1

**Potential Issues**:
- None - straightforward algebra

---

#### Step 4: Verify Universality (Discrete and Continuous)

**Goal**: Confirm the proof works for all probability distributions

**Substep 4.1**: Discrete case
- **Action**: For discrete $X$ with atoms at $\{x_1, x_2, \ldots\}$ where $P(X = x_i) = p_i > 0$:
  - Support is $\{x_i : p_i > 0\}$
  - Variance is $\sigma^2 = \sum_i p_i (x_i - \mu)^2$
  - Maximum gap is $R = \max_i |x_i - \mu|$
- **Justification**: All steps used only expectation and support, which are well-defined for discrete distributions
- **Why valid**: Discrete case is easier - all sums are finite
- **Expected result**: Proof applies without modification

**Substep 4.2**: Continuous case
- **Action**: For continuous $X$ with density $f_X$:
  - Support is closure of $\{x : f_X(x) > 0\}$
  - Variance is $\sigma^2 = \int (x - \mu)^2 f_X(x) dx$
  - Maximum gap is $R = \sup_{x \in \text{supp}(X)} |x - \mu|$
- **Justification**: All steps used only expectation and support, which are well-defined for continuous distributions
- **Why valid**: Expectation is integration; monotonicity still applies
- **Expected result**: Proof applies without modification

**Substep 4.3**: Mixed distributions
- **Action**: For mixed distributions (both discrete and continuous components):
  - Support is union of atoms and continuous density support
  - Variance is weighted sum of discrete and continuous parts
  - All steps remain valid
- **Justification**: General probability theory applies
- **Why valid**: Expectation is well-defined; monotonicity holds
- **Expected result**: Universality confirmed

**Dependencies**:
- Uses: General probability theory (Lebesgue integration)
- Requires: No additional assumptions beyond finite variance

---

#### Step 5: Demonstrate Tightness

**Goal**: Show the bound is sharp (cannot be improved)

**Substep 5.1**: Construct equality case
- **Action**: Consider the symmetric two-point distribution:

$$
P(X = \mu + \sigma) = \frac{1}{2}, \quad P(X = \mu - \sigma) = \frac{1}{2}
$$

- **Justification**: This is a valid probability distribution
- **Why valid**: Probabilities sum to 1; values are real
- **Expected result**: Well-defined random variable

**Substep 5.2**: Verify variance
- **Action**: Calculate variance:

$$
\mathbb{E}[X] = \frac{1}{2}(\mu + \sigma) + \frac{1}{2}(\mu - \sigma) = \mu
$$

$$
\text{Var}(X) = \frac{1}{2}(\sigma)^2 + \frac{1}{2}(-\sigma)^2 = \sigma^2
$$

- **Justification**: Direct calculation from definition
- **Why valid**: Arithmetic
- **Expected result**: Mean is μ, variance is σ²

**Substep 5.3**: Verify gap equals σ
- **Action**: Calculate maximum gap:

$$
\max_{x \in \{\mu - \sigma, \mu + \sigma\}} |x - \mu| = \max(\sigma, \sigma) = \sigma
$$

- **Justification**: Support has two points equidistant from mean
- **Why valid**: Elementary calculation
- **Expected result**: Equality holds: $\max_{x \in \text{supp}(X)} |x - \mu| = \sigma$

**Substep 5.4**: Conclude tightness
- **Action**: Since equality can be achieved, the bound $\max |x - \mu| \ge \sigma$ is tight (cannot be replaced by $\max |x - \mu| \ge c\sigma$ for any $c > 1$)
- **Justification**: Explicit counterexample to any stronger bound
- **Why valid**: Existence proof
- **Expected result**: Bound is optimal

**Dependencies**:
- Uses: Direct calculation
- Requires: None

**Potential Issues**:
- None - this is a straightforward verification

---

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Max vs Sup Interpretation

**Why Difficult**: The statement uses "max" but the maximum may not exist as a finite real number if the support is unbounded. For example, if $X$ is exponentially distributed on $[0, \infty)$, the support is unbounded and there is no maximum value.

**Proposed Solution**:
Interpret the statement using the supremum (least upper bound) rather than maximum:
- **Bounded support**: If $\sup_{x \in \text{supp}(X)} |x - \mu| < \infty$, the support is contained in a compact interval. By the extreme value theorem, the continuous function $x \mapsto |x - \mu|$ attains its supremum on this compact set, so $\max = \sup$.
- **Unbounded support**: If $\sup_{x \in \text{supp}(X)} |x - \mu| = \infty$, the inequality $\infty \ge \sigma$ is trivially satisfied for any finite σ.

**Mathematical Rigor**: The use of extended reals $[0, \infty]$ makes this completely rigorous. We define $R := \sup_{x \in \text{supp}(X)} |x - \mu| \in [0, \infty]$ and prove $\sigma \le R$ in all cases.

**Alternative Approach** (if interpretation fails):
Explicitly rewrite the lemma statement as:

$$
\sup_{x \in \text{supp}(X)} |x - \mu| \ge \sigma
$$

This is mathematically equivalent for bounded support and makes the unbounded case explicit.

**References**:
- Standard real analysis: extended real numbers, supremum definition
- Topology: extreme value theorem on compact sets

---

### Challenge 2: Topological Support vs Essential Support

**Why Difficult**: In measure theory, there are two notions of "support":
1. **Topological support**: Closed set containing all values with positive probability density (or atoms)
2. **Essential support**: Smallest set containing the random variable with probability 1

These can differ. For example, a modified Gaussian with a single atom could have topological support including the atom, but essential support might exclude it if it has probability zero.

**Impact on Proof**: The proof bound $|X - \mu| \le R$ almost surely uses the essential supremum, while the statement uses the topological support.

**Proposed Solution**:
Observe that:

$$
\text{ess sup}_{X \sim P} |X - \mu| \le \sup_{x \in \text{supp}_{\text{top}}(X)} |x - \mu|
$$

The topological support contains all points in the essential support (up to null sets) plus possibly some additional points. Therefore, using the supremum over the (larger) topological support only strengthens the upper bound $R$, which preserves the inequality $\sigma \le R$.

**Why This Works**: We're proving a lower bound on the right-hand side. Replacing the right-hand side with something potentially larger (supremum over topological support instead of essential support) only makes our claim stronger.

**Mathematical Rigor**: For rigorous measure theory, use:
- $R_{\text{ess}} := \text{ess sup} |X - \mu| = \inf\{M : P(|X - \mu| \le M) = 1\}$
- $R_{\text{top}} := \sup_{x \in \text{supp}_{\text{top}}(X)} |x - \mu|$
- Then $R_{\text{ess}} \le R_{\text{top}}$ and both satisfy $\sigma \le R$

**Alternative Approach** (if measurement issues):
Work entirely with essential supremum, which is the measure-theoretic version used in the proof. This avoids topological subtleties.

**References**:
- Probability theory: essential supremum definition
- Measure theory: almost-sure bounds

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps (variance definition → monotonicity → square root)
- [x] **Hypothesis Usage**: Variance σ² > 0 ensures non-degeneracy; finiteness used in all bounds
- [x] **Conclusion Derivation**: Inequality $\max |x - \mu| \ge \sigma$ is fully derived from $\sigma^2 \le R^2$
- [x] **Framework Consistency**: Uses only standard probability theory, no framework-specific axioms
- [x] **No Circular Reasoning**: Variance definition is external; no assumption of conclusion
- [x] **Constant Tracking**: $R$ defined as supremum; $\sigma$ given as $\sqrt{\text{Var}(X)}$
- [x] **Edge Cases**: Unbounded support handled via extended reals; bounded support via compactness
- [x] **Regularity Verified**: No smoothness or continuity required beyond support being closed
- [x] **Measure Theory**: Expectation well-defined for finite variance; monotonicity applies

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Proof by Contradiction

**Approach**: Assume $\max_{x \in \text{supp}(X)} |x - \mu| < \sigma$ and derive a contradiction.

**Detailed Steps**:
1. Suppose $R := \max_{x \in \text{supp}(X)} |x - \mu| < \sigma$
2. Then $|X - \mu| \le R < \sigma$ almost surely
3. Therefore $(X - \mu)^2 \le R^2 < \sigma^2$ almost surely
4. Taking expectations: $\sigma^2 = \mathbb{E}[(X - \mu)^2] \le R^2 < \sigma^2$
5. This is a contradiction: $\sigma^2 < \sigma^2$
6. Therefore $R \ge \sigma$

**Pros**:
- Equally rigorous and concise
- Some find contradiction proofs more intuitive (assuming the opposite)
- Makes the impossibility explicit

**Cons**:
- Requires strict inequality handling: need to show $R < \sigma$ implies $\mathbb{E}[(X - \mu)^2] \le R^2$, which already uses the direct proof's key step
- Doesn't add mathematical insight beyond the direct proof
- Contradiction is somewhat artificial - the direct proof is more natural

**When to Consider**: If a reviewer prefers proof by contradiction style, this is an equivalent formulation.

---

### Alternative 2: Popoviciu's Inequality Route (Bounded Support Only)

**Approach**: Use Popoviciu's inequality to bound variance, then derive the gap.

**Detailed Steps**:
1. Assume support is bounded: $\text{supp}(X) \subseteq [a, b]$ for finite $a < b$
2. By Popoviciu's inequality: $\text{Var}(X) \le \frac{(b - a)^2}{4}$
3. Therefore $\sigma^2 \le \frac{(b - a)^2}{4}$, so $b - a \ge 2\sigma$
4. The mean $\mu$ lies in $[a, b]$, so:
   - Either $b - \mu \ge \sigma$ (if $\mu$ is in lower half)
   - Or $\mu - a \ge \sigma$ (if $\mu$ is in upper half)
   - Or both (if $\mu$ is centered)
5. In all cases, $\max(|b - \mu|, |\mu - a|) \ge \sigma$
6. Since $\max_{x \in [a,b]} |x - \mu| = \max(|b - \mu|, |\mu - a|)$, we conclude the result

**Pros**:
- Connects to well-known Popoviciu's inequality
- Provides additional insight into bounded distributions
- Shows how the bound saturates for specific distributions (two-point at endpoints)

**Cons**:
- **Only works for bounded support** - fails for unbounded distributions like Gaussian, exponential, etc.
- More machinery required (Popoviciu's inequality is non-trivial)
- Less direct than variance definition approach
- Doesn't generalize to the universal case without additional work

**When to Consider**:
- If the lemma were restricted to bounded random variables
- To provide additional geometric intuition in the bounded case
- For comparison with Popoviciu's inequality in educational contexts

**Why Not Chosen**: The direct approach is simpler, more general, and requires less machinery.

---

### Alternative 3: Chebyshev-Style Probability Bound (Indirect)

**Approach**: Use Chebyshev's inequality to bound tail probabilities, then argue about support.

**Detailed Steps**:
1. By Chebyshev's inequality: $P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}$ for any $k > 0$
2. Choose $k = 1$: $P(|X - \mu| \ge \sigma) \le 1$
3. This is trivial (probability ≤ 1 always), but shows $P(|X - \mu| \ge \sigma) > 0$ is possible
4. If $P(|X - \mu| \ge \sigma) > 0$, then support contains values at distance ≥ σ
5. Therefore $\max_{x \in \text{supp}(X)} |x - \mu| \ge \sigma$

**Pros**:
- Connects to Chebyshev's inequality (classical result)
- Probabilistic interpretation

**Cons**:
- **Indirect and unnecessarily complicated** - Chebyshev gives probability bounds, not support bounds
- Requires showing $P(|X - \mu| \ge \sigma) > 0$, which is not always true (consider the two-point distribution at μ ± σ/2)
- Actually **incorrect** as stated - there exist distributions with $\text{Var}(X) = \sigma^2$ but $P(|X - \mu| \ge \sigma) = 0$ (e.g., uniform on $[\mu - \sigma/\sqrt{3}, \mu + \sigma/\sqrt{3}]$)
- Much weaker than the direct proof

**Why Not Chosen**: This approach doesn't work correctly. Chebyshev gives tail probabilities, not support extent. The lemma is about the support maximum, not about probabilities.

**Conclusion**: This is a dead end and illustrates why the direct variance-based proof is the correct approach.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

**None** - This proof is complete and self-contained.

### Conjectures

**None** - This is a proven result with tight bounds.

### Extensions

1. **Multivariate Extension**: For $X \in \mathbb{R}^d$ with covariance matrix $\Sigma$:

$$
\max_{x \in \text{supp}(X)} \|x - \mu\|_{\Sigma^{-1}} \ge \sqrt{\text{trace}(\Sigma)}
$$

   where $\|\cdot\|_{\Sigma^{-1}}$ is the Mahalanobis distance. This generalizes the one-dimensional result.

2. **Higher Moments**: Analogous results for higher moments:

$$
\max_{x \in \text{supp}(X)} |x - \mu| \ge \left(\mathbb{E}[|X - \mu|^p]\right)^{1/p}
$$

   for any $p \ge 1$ (follows from Hölder's inequality).

3. **Fractional Moments**: The bound holds for fractional moments as well:

$$
\max_{x \in \text{supp}(X)} |x - \mu| \ge \left(\mathbb{E}[|X - \mu|^\alpha]\right)^{1/\alpha}
$$

   for any $\alpha > 0$.

4. **Connection to Concentration Inequalities**: Understanding how this fundamental gap bound relates to modern concentration of measure results (sub-Gaussian, sub-exponential tails).

---

## IX. Expansion Roadmap

**Phase 1: Write Full Proof** (Estimated: 1-2 hours)
1. Expand each substep into complete sentences with all justifications
2. Add explicit references to standard theorems (extreme value theorem, monotonicity of expectation)
3. Clarify max vs sup notation in the statement

**Phase 2: Add Pedagogical Elements** (Estimated: 1 hour)
1. Intuitive explanation: "variance is average squared distance, so max distance must be at least as large as the 'typical' distance"
2. Visual diagrams for discrete two-point case (tightness)
3. Comparison with related inequalities (Chebyshev, Popoviciu)

**Phase 3: Verify Consistency with 03_cloning.md** (Estimated: 30 minutes)
1. Read the original proof in 03_cloning.md, Lemma 7.3.1
2. Confirm the proofs are consistent (this is marked as a reference to that lemma)
3. Ensure notation matches between documents

**Phase 4: Review and Validation** (Estimated: 30 minutes)
1. Check all mathematical steps rigorously
2. Verify examples (two-point distribution calculation)
3. Ensure statement clarity (max vs sup)

**Total Estimated Expansion Time**: 3-4 hours

**Note**: This is a simple lemma, so expansion is minimal. Most effort will be in pedagogical clarity and notation consistency with the framework.

---

## X. Cross-References

**Theorems Used**:
- Extreme value theorem (standard analysis)
- Monotonicity of expectation (probability theory)
- Popoviciu's inequality (mentioned for context, not used in main proof)

**Definitions Used**:
- Variance: $\sigma^2 = \mathbb{E}[(X - \mu)^2]$
- Support: topological closure of probability mass/density
- Supremum: least upper bound

**Related Results**:
- Original proof: See `03_cloning.md`, Lemma 7.3.1 (this is a reference/restatement)
- {prf:ref}`lem-variance-to-mean-separation` - Uses this lemma to convert variance to mean separation for partitioned sets
- {prf:ref}`lem-raw-to-rescaled-gap-rho` - Signal propagation lemma that depends on variance-to-gap conversion
- Chebyshev's inequality - Related but different (tail probability bound vs support extent bound)
- Popoviciu's inequality - Provides maximum variance for bounded random variables

**Usage in Framework**:
- **Appendix B.3**: Signal Integrity verification for ρ-localized pipeline
- **Signal propagation chain**: Geometry → Raw variance → Gap (this lemma) → Rescaled gap → Mean separation

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Yes
**Confidence Level**: High - This is a fundamental statistical result with a simple, rigorous proof. The bound is tight (achievable by symmetric two-point distribution) and universal (works for all distributions with finite variance). The only subtlety is max vs sup interpretation for unbounded support, which is resolved using extended reals.
