# Proof of Variance-to-Gap Lemma (Adaptive Version)

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/11_geometric_gas.md
**Lemma**: lem-variance-to-gap-adaptive
**Status**: Complete proof (Annals rigor)
**Date**: 2025-10-25

---

## Lemma Statement

:::{prf:lemma} Variance-to-Gap (Universal Statistical Inequality)
:label: lem-variance-to-gap-adaptive

For any random variable $X$ with mean $\mu$ and variance $\sigma^2 > 0$:

$$
\sup_{x \in \text{supp}(X)} |x - \mu| \ge \sigma
$$

where $\text{supp}(X)$ denotes the topological support of the law of $X$. When the support is bounded, the supremum is attained and equals the maximum.

:::

**Informal Restatement**: If a random variable has positive variance $\sigma^2$, then the supremum of deviations from the mean over the support is at least $\sigma$. For bounded support, this supremum is attained, meaning at least one value in the support is at distance at least $\sigma$ from the mean $\mu$. This establishes a fundamental connection between variance (spread) and extreme deviation (maximum gap).

**Context in Framework**: This lemma appears in Appendix B.3.1 of the Geometric Gas document, titled "The Variance-to-Gap Lemma (Universal)". It is explicitly noted as "ρ-independent and applies universally"—meaning it is a pure statistical fact that does not depend on the localization scale or any algorithm-specific parameters. The lemma bridges variance guarantees (which come from geometric analysis) to gap guarantees (which are needed for signal propagation in the rescaling pipeline).

---

## Proof

**Strategy**: We prove this by defining the support radius $R := \sup_{x \in \text{supp}(X)} |x - \mu|$ and showing that the variance definition implies $\sigma^2 \le R^2$, from which the result follows by taking square roots.

**Step 1: Define the support radius**

Let

$$
R := \sup_{x \in \text{supp}(X)} |x - \mu| \in [0, \infty]
$$

This supremum always exists in the extended real numbers $[0, \infty]$ for any non-empty set. Since $\sigma^2 > 0$, the support must contain at least two distinct points (otherwise the variance would be zero), so the support is non-empty and $R$ is well-defined.

**Interpretation of max vs sup**: We distinguish two cases:
- **Bounded support** ($R < \infty$): The support is contained in the compact interval $[\mu - R, \mu + R]$. The continuous function $x \mapsto |x - \mu|$ attains its supremum on this compact set by the extreme value theorem, so $\max = \sup$ and the maximum exists as a finite real number.
- **Unbounded support** ($R = \infty$): The inequality $R \ge \sigma$ is trivially satisfied since any finite number is less than infinity.

**Step 2: Bound variance by squared radius**

By definition of $R$ as the supremum over the support:

$$
|x - \mu| \le R \quad \text{for all } x \in \text{supp}(X)
$$

This holds by the definition of supremum as the least upper bound. Since $X$ takes values only in its support (with probability 1), we have almost surely:

$$
|X - \mu| \le R
$$

Squaring both sides (which preserves the inequality for non-negative values):

$$
(X - \mu)^2 \le R^2 \quad \text{almost surely}
$$

Taking expectations of both sides and using monotonicity of expectation (if $Y_1 \le Y_2$ almost surely, then $\mathbb{E}[Y_1] \le \mathbb{E}[Y_2]$):

$$
\mathbb{E}[(X - \mu)^2] \le \mathbb{E}[R^2] = R^2
$$

where the last equality holds because $R^2$ is a constant. By definition of variance, $\sigma^2 = \mathbb{E}[(X - \mu)^2]$, so:

$$
\sigma^2 \le R^2
$$

**Handling the unbounded case**: If $R = \infty$, then $R^2 = \infty$ and the inequality $\sigma^2 \le \infty$ holds trivially since $\sigma^2$ is finite by hypothesis.

**Step 3: Conclude $\sigma \le R$**

From $\sigma^2 \le R^2$ with $\sigma, R \ge 0$, we apply the monotonicity of the square root function on $[0, \infty]$ to conclude:

$$
\sigma \le R = \sup_{x \in \text{supp}(X)} |x - \mu|
$$

For bounded support, by Step 1, this supremum is attained, so $R = \max_{x \in \text{supp}(X)} |x - \mu|$, yielding the statement of the lemma.

For unbounded support, $R = \infty > \sigma$, so the inequality holds.

**Q.E.D.**

---

## Universality Verification

The proof uses only:
1. **Variance definition**: $\sigma^2 = \mathbb{E}[(X - \mu)^2]$
2. **Monotonicity of expectation**: Standard property of expectation
3. **Extreme value theorem**: Standard result from analysis

These apply to all probability distributions (discrete, continuous, or mixed) with finite variance. We verify each case explicitly:

**Discrete case**: For discrete $X$ with atoms at $\{x_1, x_2, \ldots\}$ where $P(X = x_i) = p_i > 0$:
- Support is $\{x_i : p_i > 0\}$
- Variance is $\sigma^2 = \sum_i p_i (x_i - \mu)^2$
- Maximum gap is $R = \max_i |x_i - \mu|$
- All proof steps apply: the summation is a finite or countable expectation, and monotonicity holds.

**Continuous case**: For continuous $X$ (absolutely continuous with density $f_X$):
- Support is the topological closure of $\{x : f_X(x) > 0\}$
- Variance is $\sigma^2 = \int (x - \mu)^2 f_X(x) \, dx$
- Maximum gap is $R = \sup_{x \in \text{supp}(X)} |x - \mu|$
- All proof steps apply: expectation is integration, and monotonicity still holds.

**General measure-theoretic case**: The proof applies to **any** Borel probability measure on $\mathbb{R}$ with finite variance, including singular continuous distributions (e.g., Cantor-type). No assumption of a density is required; the argument is purely measure-theoretic.

**Mixed distributions**: For distributions with both discrete and continuous components:
- Support is the union of atoms and continuous density support
- Variance is the weighted sum of discrete and continuous parts
- All steps remain valid by general probability theory.

---

## Tightness of the Bound

The bound is **sharp** (cannot be improved) as demonstrated by the following equality case.

**Construction**: Consider the symmetric two-point distribution:

$$
P(X = \mu + \sigma) = \frac{1}{2}, \quad P(X = \mu - \sigma) = \frac{1}{2}
$$

**Verification of mean**:

$$
\mathbb{E}[X] = \frac{1}{2}(\mu + \sigma) + \frac{1}{2}(\mu - \sigma) = \mu
$$

**Verification of variance**:

$$
\text{Var}(X) = \frac{1}{2}(\sigma - 0)^2 + \frac{1}{2}(-\sigma - 0)^2 = \frac{1}{2}\sigma^2 + \frac{1}{2}\sigma^2 = \sigma^2
$$

**Verification of gap**:

$$
\max_{x \in \{\mu - \sigma, \mu + \sigma\}} |x - \mu| = \max(\sigma, \sigma) = \sigma
$$

Therefore, equality holds: $\max_{x \in \text{supp}(X)} |x - \mu| = \sigma$. This shows the bound cannot be replaced by $\max |x - \mu| \ge c\sigma$ for any $c > 1$.

---

## Technical Notes

### Max vs Sup Interpretation

The original statement uses "max" but the maximum may not exist as a finite real number if the support is unbounded. We resolve this by interpreting the statement using the supremum:

**Rigorous formulation**: Define $R := \sup_{x \in \text{supp}(X)} |x - \mu| \in [0, \infty]$ and prove $\sigma \le R$ in all cases.

**Bounded support**: If $R < \infty$, the support is contained in a compact interval. By the extreme value theorem, the continuous function $x \mapsto |x - \mu|$ attains its supremum on this compact set, so max exists and equals sup.

**Unbounded support**: If $R = \infty$, the inequality $\infty \ge \sigma$ is trivially satisfied for any finite $\sigma$.

### Topological Support vs Essential Support

In measure theory, there are two notions of support:
1. **Topological support**: Closed set containing all values with positive probability density (or atoms)
2. **Essential support**: Smallest set containing the random variable with probability 1

The proof uses the essential supremum $\text{ess sup} |X - \mu| = \inf\{M : P(|X - \mu| \le M) = 1\}$ in the almost-sure bound $|X - \mu| \le R$, while the statement uses the topological support.

**Relationship**: The essential supremum satisfies:

$$
\text{ess sup} |X - \mu| \le \sup_{x \in \text{supp}_{\text{top}}(X)} |x - \mu| = R
$$

The topological support contains all points in the essential support (up to null sets) plus possibly additional points. Therefore, using the supremum over the (possibly larger) topological support only strengthens the upper bound $R$, which preserves the inequality $\sigma \le R$. Since we are proving a **lower bound** on the right-hand side, replacing it with something potentially larger (topological support instead of essential support) makes our claim **weaker** (easier to prove).

**Conclusion**: The proof is valid for both interpretations of support. The essential support version would be a **stronger claim** (tighter bound), but the lemma as stated uses topological support, which is sufficient for framework applications and avoids measure-theoretic subtleties. The proof technique would apply equally to either definition.

---

## Relationship to 03_cloning.md, Lemma 7.3.1

This lemma is a **probability-theoretic reformulation** of Lemma 7.3.1 from 03_cloning.md, which states:

**Original (03_cloning.md, Lemma 7.3.1)**: For a set $\{v_i\}_{i=1}^k$ of $k \ge 2$ real numbers with empirical variance $\text{Var}(\{v_i\}) \ge \kappa > 0$:

$$
\max_{i,j} |v_i - v_j| \ge \sqrt{2\kappa}
$$

**This lemma (lem-variance-to-gap-adaptive)**: For a random variable $X$ with variance $\sigma^2 > 0$:

$$
\max_{x \in \text{supp}(X)} |x - \mu| \ge \sigma
$$

**Connection**: These are equivalent via the following observations:

1. **Empirical interpretation**: If $X$ is a discrete uniform random variable over the set $\{v_1, \ldots, v_k\}$ (i.e., $P(X = v_i) = 1/k$), then:
   - Mean: $\mu = \frac{1}{k}\sum_i v_i = \bar{v}$ (empirical mean)
   - Variance: $\sigma^2 = \frac{1}{k}\sum_i (v_i - \bar{v})^2 = \text{Var}(\{v_i\})$ (empirical variance)
   - Support maximum: $\max_{x \in \text{supp}(X)} |x - \mu| = \max_i |v_i - \bar{v}|$

2. **Conversion between formulations**: The maximum pairwise gap and maximum deviation from mean are related by the **inequality**:

$$
\max_i |v_i - \bar{v}| \le \max_{i,j} |v_i - v_j| \le 2 \max_i |v_i - \bar{v}|
$$

   The first inequality is trivial. The second holds because the maximum pairwise gap occurs between the largest and smallest values, and both are within distance $\max_i |v_i - \bar{v}|$ from the mean. Equality in the second inequality holds for symmetric two-point distributions where the mean is at the midpoint, but **not in general**. For example, for $\{-1, -1, 2\}$ with mean $\mu = 0$, we have $\max_i |v_i - \mu| = 2$ but $\max_{i,j} |v_i - v_j| = 3 \neq 2 \times 2$.

3. **Factor of $\sqrt{2}$**: The original lemma has a factor $\sqrt{2}$ because it bounds $\max_{i,j} |v_i - v_j|$ (maximum pairwise difference) while this lemma bounds $\max_i |v_i - \mu|$ (maximum deviation from mean). These are **related but distinct** quantities, connected by the inequalities above.

**Why both versions exist**: The original version (Lemma 7.3.1) is more natural for the cloning analysis, which works with empirical sets of walkers and needs bounds on pairwise gaps. The adaptive version (this lemma) is more natural for the ρ-localized pipeline, which works with probability distributions and localized statistics and needs bounds on deviations from the mean. Both are **consistent consequences** of the variance definition but are not simply related by a constant factor—they bound different (though related) quantities.

---

## Dependencies

**Standard Results Used**:
- Monotonicity of expectation
- Extreme value theorem (continuous function on compact set attains maximum)
- Square root monotonicity ($0 \le a \le b \implies \sqrt{a} \le \sqrt{b}$)

**Framework Dependencies**: None. This is a self-contained statistical result requiring no framework-specific axioms.

**Cross-References in Framework**:
- {prf:ref}`lem-variance-to-mean-separation` (03_cloning.md) - Uses this lemma to convert variance to mean separation for partitioned sets
- {prf:ref}`lem-raw-to-rescaled-gap-rho` (11_geometric_gas.md) - Signal propagation lemma that depends on variance-to-gap conversion
- {prf:ref}`thm-signal-generation-adaptive` (11_geometric_gas.md) - Generates raw variance in ρ-localized measurements

**Usage in Framework**: This lemma appears in the Signal Integrity verification for the ρ-localized pipeline (Appendix B.3 of 11_geometric_gas.md). The causal chain is:
1. Geometry → Raw variance (from {prf:ref}`thm-signal-generation-adaptive`)
2. Raw variance → Gap (from this lemma)
3. Gap → Rescaled gap (from {prf:ref}`lem-raw-to-rescaled-gap-rho`)
4. Rescaled gap → Mean separation (from {prf:ref}`lem-variance-to-mean-separation`)

---

## Alternative Proof (Proof by Contradiction)

For completeness, we provide an alternative proof by contradiction, which some readers may find more intuitive.

**Assumption**: Suppose, for the sake of contradiction, that $R := \sup_{x \in \text{supp}(X)} |x - \mu| < \sigma$.

**Consequence**: Then $|X - \mu| \le R < \sigma$ almost surely.

**Squaring**: Therefore $(X - \mu)^2 \le R^2 < \sigma^2$ almost surely.

**Taking expectations**: $\sigma^2 = \mathbb{E}[(X - \mu)^2] \le R^2 < \sigma^2$

**Contradiction**: This gives $\sigma^2 < \sigma^2$, which is impossible.

**Conclusion**: Therefore $R \ge \sigma$, completing the proof.

**Note**: This proof is logically equivalent to the direct proof but presents the argument in contrapositive form. Both are equally rigorous.

---

## Summary

This lemma establishes a fundamental relationship between variance (a second-moment property) and support extent (a zero-probability property). The proof is elementary, using only the variance definition and monotonicity of expectation. The bound is tight (achievable by symmetric two-point distributions) and universal (applies to all probability distributions with finite variance, regardless of discrete, continuous, or mixed structure).

The lemma plays a critical role in the ρ-localized signal propagation analysis by converting abstract variance guarantees into concrete gap guarantees that can be tracked through the rescaling pipeline.

---

## Important Note: Downstream Application in 11_geometric_gas.md

**WARNING**: When applying this lemma in the signal propagation chain (11_geometric_gas.md, lines ~3246-3251), care must be taken with the conversion factor. If you have a bound on the pairwise gap $\max_{i,j} |d'_i - d'_j| \ge C$, the bound on the maximum deviation from the mean is:

$$
\max_i |d'_i - \mu[d']| \ge \frac{C}{2}
$$

**NOT** $\ge C$. This factor of $1/2$ comes from the inequality $\max_{i,j} |v_i - v_j| \le 2 \max_i |v_i - \mu|$ (see the "Relationship to 03_cloning.md" section above). Omitting this factor will inflate the propagation constants and may invalidate downstream bounds.

**Action Required**: Verify that all applications of this lemma in the pipeline analysis correctly include the factor of $1/2$ when converting from pairwise gaps to mean deviations.
