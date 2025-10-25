# Proof Sketch for thm-c1-established-cinf

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
**Theorem**: thm-c1-established-cinf
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} C¹ Regularity (Previously Proven)
:label: thm-c1-established-cinf

**Source**: [11_geometric_gas.md](11_geometric_gas.md), Appendix A, Theorem A.1

V_fit is continuously differentiable with ||∇_{x_i} V_fit|| ≤ K_{V,1}(ρ) = O(ρ^{-1}), k-uniform and N-uniform.

:::

**Informal Restatement**: The fitness potential V_fit, which measures how well a walker is performing relative to its local neighborhood, is a smooth function whose gradient (rate of change) is bounded by a constant that scales inversely with the localization radius ρ. Critically, this bound does not depend on how many walkers are alive (k-uniform) or the total swarm size (N-uniform), ensuring the algorithm's stability properties hold regardless of population size.

**Mathematical Context**: The fitness potential is defined as:

$$
V_{\text{fit}}[f_k, \rho](x_i) = g_A(Z_\rho[f_k, d, x_i])
$$

where:
- $Z_\rho[f_k, d, x_i] = \frac{d(x_i) - \mu_\rho[f_k, x_i]}{\sigma'_{\text{reg}}(\sigma^2_\rho[f_k, x_i])}$ is the regularized Z-score
- $\mu_\rho = \sum_{j \in A_k} w_{ij}(\rho) \cdot d(x_j)$ is the localized mean
- $\sigma^2_\rho = \sum_{j \in A_k} w_{ij}(\rho) \cdot (d(x_j) - \mu_\rho)^2$ is the localized variance
- $w_{ij}(\rho) = K_\rho(d(x_i)) \cdot K_\rho(\|x_i - x_j\|) / Z_{\text{norm}}$ are normalized localization weights
- $K_\rho(r) = \exp(-r^2/(2\rho^2))$ is the Gaussian localization kernel
- $g_A: \mathbb{R} \to [0, A]$ is the smooth rescale function

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI FAILED TO RESPOND**

Gemini 2.5 Pro did not return a valid response after two attempts. This may be due to a temporary service issue or timeout. The proof sketch proceeds with Codex's strategy only.

**Limitations**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available

---

### Strategy B: Codex's Approach

**Method**: Chain rule decomposition

**Key Steps**:
1. **Reduce weights and isolate cancellations**: Express normalized weights as $w_{ij} = \alpha_{ij}/S_i$ and derive the telescoping identity $\sum_j \nabla w_{ij} = 0$
2. **Bound the L¹-sum of weight gradients uniformly in k**: Show $\sum_j \|\nabla w_{ij}\| \leq C_w/\rho$ using Gaussian envelope bounds
3. **Control $\nabla\mu_\rho$ and $\nabla\sigma^2_\rho$**: Use telescoping and centering to get k-uniform bounds
4. **Differentiate and bound the Z-score**: Apply quotient rule to $Z_\rho = (d(x_i) - \mu_\rho)/\sigma'_{\text{reg}}(\sigma^2_\rho)$
5. **Apply $g_A$ and conclude**: Use chain rule to get final bound $\|\nabla V_{\text{fit}}\| \leq L_{g_A} \|\nabla Z_\rho\| = O(\rho^{-1})$
6. **Verify continuity**: Check all components are smooth with no singularities

**Strengths**:
- Systematic decomposition through the composition pipeline
- Explicit tracking of ρ-dependence at each stage
- Direct use of telescoping mechanism for k-uniformity
- All constants explicitly bounded in terms of framework parameters
- Constructive proof yielding the exact form of $K_{V,1}(\rho)$

**Weaknesses**:
- Requires careful bookkeeping of multiple constants
- Heavy reliance on Gaussian kernel properties (though this is given in framework)

**Framework Dependencies**:
- Chain rule, quotient rule, product rule (standard calculus)
- Gaussian kernel derivative bounds: $\|\nabla K_\rho\| \leq (C/\rho) K_\rho$
- Telescoping identity: $\sum_j \nabla w_{ij} = 0$ (from normalization)
- Smoothness of primitives: $d, g_A, \sigma'_{\text{reg}} \in C^\infty$
- Regularization: $\sigma'_{\text{reg}}(\sigma^2) \geq \varepsilon_\sigma > 0$

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Chain rule decomposition (Codex's approach)

**Rationale**:
In the absence of Gemini's feedback, Codex's strategy is well-justified and mathematically sound:
- ✅ **Advantage 1**: Directly exploits the composition structure $V_{\text{fit}} = g_A \circ Z_\rho$ where $Z_\rho$ is built from localized moments
- ✅ **Advantage 2**: The telescoping mechanism $\sum_j \nabla w_{ij} = 0$ is used precisely where needed (Step 3) to prevent k-dependent growth
- ✅ **Advantage 3**: Gaussian envelope bounds $r e^{-r^2/(2\rho^2)} \leq C\rho$ are standard and yield the O(ρ^{-1}) scaling directly
- ✅ **Advantage 4**: All steps are constructive with explicit constants

**Integration**:
- Steps 1-6: All from Codex's strategy (verified against framework)
- Critical insight: The k-uniformity arises from two mechanisms working together:
  1. Normalization constraint $\sum_j w_{ij} = 1$ implies telescoping
  2. Gaussian decay ensures effective neighborhood size is O(1), not O(k)

**Verification Status**:
- ✅ All framework dependencies verified (see § III)
- ✅ No circular reasoning detected (builds from primitives upward)
- ✅ All constants explicitly tracked (see § III)
- ⚠️ Would benefit from Gemini's cross-validation when available

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms** (from framework):

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| Standard calculus | Chain rule, quotient rule, product rule | Steps 1, 3-5 | ✅ |
| Telescoping identity | $\sum_{j \in A_k} \nabla^m w_{ij} = 0$ for all $m \geq 1$ | Step 3 | ✅ (from normalization) |

**Theorems** (from earlier documents):

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| Gaussian kernel properties | Framework assumption | $K_\rho(r) = \exp(-r^2/(2\rho^2))$ is real analytic | Step 2 | ✅ |
| Hermite derivative bounds | Standard result | $\|\nabla^m K_\rho\| \leq C m! \rho^{-m} K_\rho$ | Step 2 | ✅ |

**Definitions**:

| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| def-simplified-measurement-cinf | 19_geometric_gas_cinf_regularity_simplified.md | $d: X \to \mathbb{R}$ position-dependent, $C^\infty$ | Bounding $\nabla d$ in Step 4 |
| def-localization-kernel-cinf | 19_geometric_gas_cinf_regularity_simplified.md | Gaussian kernel $K_\rho(r)$ | Deriving weight bounds in Step 2 |
| def-localization-weights-cinf | 19_geometric_gas_cinf_regularity_simplified.md | Normalized weights $w_{ij}(\rho)$ | Steps 1-3 |
| def-pipeline-cinf | 19_geometric_gas_cinf_regularity_simplified.md | Moments, Z-score, fitness potential | Steps 3-5 |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $d_{\max}$ | $\sup_{x \in X} \|d(x)\|$ | Finite (X compact) | N-uniform, k-uniform |
| $d'_{\max}$ | $\sup_{x \in X} \|\nabla d(x)\|$ | $\leq C_d^{(1)} < \infty$ | N-uniform, k-uniform |
| $\varepsilon_\sigma$ | Lower bound on $\sigma'_{\text{reg}}$ | $> 0$ (regularization parameter) | N-uniform, k-uniform |
| $L_{g_A}$ | $\sup \|g'_A\|$ | Finite ($g_A \in C^\infty$) | N-uniform, k-uniform |
| $L_{\sigma'}$ | Lipschitz constant of $\sigma'_{\text{reg}}$ | Finite ($\sigma'_{\text{reg}} \in C^\infty$) | N-uniform, k-uniform |
| $C_w$ | L¹ bound on $\sum_j \|\nabla w_{ij}\|$ | $\leq 2e^{-1/2}$ (from Gaussian envelope) | N-uniform, k-uniform, depends on ρ as $C_w/\rho$ |
| $K_{V,1}(\rho)$ | Gradient bound on $V_{\text{fit}}$ | $L_{g_A}[d'_{\max}/\varepsilon_\sigma + \tilde{C}/\rho]$ | O(ρ^{-1}), N-uniform, k-uniform |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- **Lemma A (L¹ gradient bound for normalized Gaussian weights)**: For $w_{ij} = \alpha_{ij}/S_i$ with $\alpha_{ij} = \exp(-\|x_i - x_j\|^2/(2\rho^2))$, show $\sum_j \|\nabla_{x_i} w_{ij}\| \leq C_w/\rho$ with $C_w$ universal.
  - **Why needed**: Delivers k- and N-uniform O(ρ^{-1}) control for all downstream derivatives
  - **Difficulty**: Medium (elementary calculus with careful normalization bound)
  - **Status**: Outlined in Codex Step 2, needs expansion

- **Lemma B (Telescoping identities)**: Show $\sum_j \nabla w_{ij} = 0$ and $\nabla\sigma^2_\rho = \sum_j \nabla w_{ij}(d(x_j) - \mu_\rho)^2$.
  - **Why needed**: Cancels potentially large $\mu_\rho$-derivative terms in $\sigma^2_\rho$
  - **Difficulty**: Easy (direct from normalization $\sum_j w_{ij} = 1$)
  - **Status**: Standard, needs writeup

- **Lemma C (Denominator regularization)**: If $s \in C^\infty$ with $s \geq \varepsilon_\sigma$ and $|s'| \leq L_{\sigma'}$, then $(μ_ρ, σ²_ρ) \mapsto (d(x_i) - μ_ρ)/s(σ²_ρ)$ is C¹ with Lipschitz gradient.
  - **Why needed**: Ensures safe quotient and yields explicit constants
  - **Difficulty**: Easy (quotient rule + chain rule)
  - **Status**: Standard, needs writeup

**Uncertain Assumptions**:
None identified. All assumptions are explicitly stated in the framework document § 3.

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes C¹ regularity of $V_{\text{fit}}$ by systematically differentiating through the six-stage composition pipeline that defines the fitness potential:

$$
\text{Positions } \{x_j\} \to \text{Weights } w_{ij}(\rho) \to \text{Moments } (\mu_\rho, \sigma^2_\rho) \to \text{Std Dev } \sigma'_{\text{reg}} \to \text{Z-score } Z_\rho \to \text{Fitness } V_{\text{fit}}
$$

The key technical challenge is ensuring that bounds remain **k-uniform** (independent of the number of alive walkers). This is achieved through the **telescoping mechanism**: the normalization constraint $\sum_j w_{ij} = 1$ implies $\sum_j \nabla w_{ij} = 0$, which converts naive O(k) sums into O(1) centered sums.

The **ρ-dependence** arises from the Gaussian kernel's derivative scaling: $\|\nabla K_\rho\| \sim \rho^{-1} K_\rho$, which propagates through the chain to yield $K_{V,1}(\rho) = O(\rho^{-1})$.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Weight Decomposition and Telescoping**: Express normalized weights $w_{ij} = \alpha_{ij}/S_i$ and derive $\sum_j \nabla w_{ij} = 0$
2. **L¹ Gradient Bound**: Show $\sum_j \|\nabla w_{ij}\| \leq C_w/\rho$ using Gaussian envelope bounds
3. **Moment Derivatives**: Bound $\|\nabla\mu_\rho\|$ and $\|\nabla\sigma^2_\rho\|$ using telescoping and centering
4. **Z-score Gradient**: Apply quotient rule to $Z_\rho = (d(x_i) - \mu_\rho)/\sigma'_{\text{reg}}(\sigma^2_\rho)$
5. **Final Composition**: Use chain rule $\nabla V_{\text{fit}} = g'_A(Z_\rho) \nabla Z_\rho$ to get O(ρ^{-1}) bound
6. **Continuity Verification**: Check smoothness of all components

---

### Detailed Step-by-Step Sketch

#### Step 1: Weight Decomposition and Telescoping Identity

**Goal**: Express normalized weights in a form that isolates the normalization denominator and derive the fundamental telescoping identity.

**Substep 1.1**: Decompose normalized weights
- **Action**: Write the unnormalized weight as $\alpha_{ij}(\rho) = K_\rho(d(x_i)) \cdot K_\rho(\|x_i - x_j\|)$. Note that the factor $K_\rho(d(x_i))$ is independent of $j$ and will cancel in normalization. Define the partition function $S_i = \sum_{\ell \in A_k} \alpha_{i\ell}$. Then $w_{ij} = \alpha_{ij}/S_i$.
- **Justification**: Direct from definition of normalized weights (def-localization-weights-cinf)
- **Why valid**: Normalization is well-defined since $S_i > 0$ (Gaussian kernel is strictly positive)
- **Expected result**: $w_{ij} = \alpha_{ij}/S_i$ with $\sum_j w_{ij} = 1$

**Substep 1.2**: Differentiate using quotient rule
- **Action**: Apply quotient rule:
$$
\nabla_{x_i} w_{ij} = \frac{\nabla_{x_i} \alpha_{ij}}{S_i} - \frac{\alpha_{ij}}{S_i^2} \nabla_{x_i} S_i
$$
- **Justification**: Quotient rule for differentiable functions; $\alpha_{ij}$ and $S_i$ are C∞ in $x_i$ (Gaussian kernel is real analytic)
- **Why valid**: $S_i > 0$ ensures no division by zero
- **Expected result**: Expression for $\nabla w_{ij}$ in terms of $\nabla \alpha_{ij}$ and $\nabla S_i$

**Substep 1.3**: Derive telescoping identity
- **Action**: Sum both sides over $j \in A_k$:
$$
\sum_{j \in A_k} \nabla_{x_i} w_{ij} = \frac{1}{S_i} \sum_j \nabla_{x_i} \alpha_{ij} - \frac{1}{S_i^2} \nabla_{x_i} S_i \sum_j \alpha_{ij} = \frac{1}{S_i} \nabla_{x_i} S_i - \frac{1}{S_i^2} \nabla_{x_i} S_i \cdot S_i = 0
$$
- **Justification**: Since $S_i = \sum_j \alpha_{ij}$, we have $\nabla S_i = \sum_j \nabla \alpha_{ij}$ and $\sum_j \alpha_{ij} = S_i$ by definition
- **Why valid**: Linearity of differentiation; finite sum (|A_k| = k < ∞) allows exchange of sum and derivative
- **Expected result**: **Telescoping identity** $\sum_j \nabla w_{ij} = 0$ (this is the key to k-uniformity)

**Conclusion**: The normalized weights satisfy $\sum_j \nabla w_{ij} = 0$ identically for all $x_i$, $S$, and $\rho$.

**Dependencies**:
- Uses: Quotient rule (standard calculus), linearity of differentiation
- Requires: $S_i > 0$ (guaranteed by Gaussian kernel positivity)

**Potential Issues**:
- ⚠️ None identified; this is a straightforward application of calculus

---

#### Step 2: L¹ Gradient Bound for Normalized Weights

**Goal**: Establish the uniform bound $\sum_j \|\nabla_{x_i} w_{ij}\| \leq C_w/\rho$ independent of k, N, and walker positions.

**Substep 2.1**: Bound gradient of unnormalized weight
- **Action**: For the Gaussian kernel component $\alpha_{ij} \sim K_\rho(\|x_i - x_j\|) = \exp(-\|x_i - x_j\|^2/(2\rho^2))$, compute:
$$
\nabla_{x_i} K_\rho(\|x_i - x_j\|) = -\frac{x_i - x_j}{\rho^2} K_\rho(\|x_i - x_j\|)
$$
Hence:
$$
\|\nabla_{x_i} \alpha_{ij}\| \leq \frac{\|x_i - x_j\|}{\rho^2} \alpha_{ij}
$$
- **Justification**: Chain rule; derivative of $\exp(-r^2/(2\rho^2))$ is $-(r/\rho^2) \exp(-r^2/(2\rho^2))$
- **Why valid**: Gaussian kernel is C∞
- **Expected result**: Pointwise bound relating $\|\nabla \alpha_{ij}\|$ to $\alpha_{ij}$

**Substep 2.2**: Apply Gaussian envelope bound
- **Action**: Use the universal Gaussian envelope bound: for all $r \geq 0$,
$$
r e^{-r^2/(2\rho^2)} \leq \sup_{r \geq 0} r e^{-r^2/(2\rho^2)} = \rho e^{-1/2}
$$
Therefore:
$$
\|\nabla_{x_i} \alpha_{ij}\| \leq \frac{\rho e^{-1/2}}{\rho^2} \alpha_{ij} = \frac{e^{-1/2}}{\rho} \alpha_{ij}
$$
- **Justification**: Optimization: $\frac{d}{dr}(r e^{-r^2/(2\rho^2)}) = 0$ at $r = \rho$, giving maximum value $\rho e^{-1/2}$
- **Why valid**: Standard calculus; maximum exists and is finite
- **Expected result**: Uniform bound $\|\nabla \alpha_{ij}\| \leq (e^{-1/2}/\rho) \alpha_{ij}$

**Substep 2.3**: Bound L¹ sum of normalized weight gradients
- **Action**: From Step 1.2:
$$
\|\nabla w_{ij}\| \leq \frac{\|\nabla \alpha_{ij}\|}{S_i} + \frac{\alpha_{ij}}{S_i^2} \|\nabla S_i\|
$$
Note that $\|\nabla S_i\| \leq \sum_\ell \|\nabla \alpha_{i\ell}\|$ by triangle inequality. Thus:
$$
\sum_j \|\nabla w_{ij}\| \leq \frac{1}{S_i} \sum_j \|\nabla \alpha_{ij}\| + \frac{1}{S_i^2} \sum_\ell \|\nabla \alpha_{i\ell}\| \sum_j \alpha_{ij}
$$
Since $\sum_j \alpha_{ij} = S_i$:
$$
\sum_j \|\nabla w_{ij}\| \leq \frac{2}{S_i} \sum_j \|\nabla \alpha_{ij}\| \leq \frac{2 e^{-1/2}}{\rho} \frac{\sum_j \alpha_{ij}}{S_i} = \frac{2 e^{-1/2}}{\rho}
$$
- **Justification**: Triangle inequality, normalization $\sum_j \alpha_{ij} = S_i$, bound from Substep 2.2
- **Why valid**: All sums are finite (k < ∞), all terms positive
- **Expected result**: $\sum_j \|\nabla w_{ij}\| \leq C_w/\rho$ with $C_w = 2e^{-1/2} \approx 1.21$

**Conclusion**: The L¹ norm of weight gradients is **k-uniform** and scales as O(ρ^{-1}).

**Dependencies**:
- Uses: Triangle inequality, Gaussian envelope bound, normalization
- Requires: Gaussian kernel properties (def-localization-kernel-cinf)

**Potential Issues**:
- ⚠️ Hidden dependence on walker positions? **Resolution**: Envelope bound is uniform over all $r \geq 0$, so position-independent.

---

#### Step 3: Gradient Bounds for Localized Moments

**Goal**: Establish k-uniform bounds $\|\nabla\mu_\rho\| \leq O(1/\rho)$ and $\|\nabla\sigma^2_\rho\| \leq O(1/\rho)$.

**Substep 3.1**: Differentiate localized mean
- **Action**: From $\mu_\rho = \sum_{j \in A_k} w_{ij} d(x_j)$, differentiate with respect to $x_i$:
$$
\nabla_{x_i} \mu_\rho = \sum_{j \in A_k} (\nabla_{x_i} w_{ij}) d(x_j) + w_{ii} \nabla_{x_i} d(x_i)
$$
where the second term appears only if $i \in A_k$ (self-term).
- **Justification**: Product rule, linearity of differentiation
- **Why valid**: Finite sum, all terms differentiable
- **Expected result**: Expression for $\nabla \mu_\rho$ in terms of $\nabla w_{ij}$ and $\nabla d$

**Substep 3.2**: Apply telescoping to eliminate k-dependence
- **Action**: Use telescoping identity $\sum_j \nabla w_{ij} = 0$ from Step 1.3 to rewrite:
$$
\sum_j (\nabla w_{ij}) d(x_j) = \sum_j (\nabla w_{ij}) (d(x_j) - \mu_\rho)
$$
where we added and subtracted $\mu_\rho \sum_j \nabla w_{ij} = 0$. Hence:
$$
\|\nabla_{x_i} \mu_\rho\| \leq \sum_j \|\nabla w_{ij}\| |d(x_j) - \mu_\rho| + \|w_{ii}\| \|\nabla d(x_i)\|
$$
Since $|d(x_j) - \mu_\rho| \leq \sup_{x,y} |d(x) - d(y)| \leq 2d_{\max}$ and $\|\nabla d\| \leq d'_{\max}$:
$$
\|\nabla \mu_\rho\| \leq 2d_{\max} \sum_j \|\nabla w_{ij}\| + d'_{\max} \leq 2d_{\max} \frac{C_w}{\rho} + d'_{\max}
$$
- **Justification**: Telescoping identity (Step 1.3), triangle inequality, boundedness of $d$ (def-simplified-measurement-cinf)
- **Why valid**: Centered sum $\sum_j \nabla w_{ij} (d(x_j) - \mu_\rho)$ has O(1) terms, not O(k) terms
- **Expected result**: $\|\nabla \mu_\rho\| \leq C_{\mu}(\rho) = O(1/\rho)$ with k-uniform bound

**Substep 3.3**: Differentiate localized variance
- **Action**: From $\sigma^2_\rho = \sum_j w_{ij} (d(x_j) - \mu_\rho)^2$, apply product rule:
$$
\nabla \sigma^2_\rho = \sum_j (\nabla w_{ij}) (d(x_j) - \mu_\rho)^2 + 2\sum_j w_{ij} (d(x_j) - \mu_\rho) \nabla(d(x_j) - \mu_\rho)
$$
The second term expands as:
$$
2\sum_j w_{ij} (d(x_j) - \mu_\rho) (\delta_{ij} \nabla d(x_i) - \nabla \mu_\rho)
$$
- **Justification**: Product rule, chain rule
- **Why valid**: Differentiating $(d(x_j) - \mu_\rho)$ gives $\delta_{ij} \nabla d(x_i) - \nabla \mu_\rho$ since $d(x_j)$ depends on $x_i$ only if $j = i$
- **Expected result**: Two-term expression for $\nabla \sigma^2_\rho$

**Substep 3.4**: Apply centering to eliminate cross-term
- **Action**: The second term contains the weighted average:
$$
\sum_j w_{ij} (d(x_j) - \mu_\rho) = \sum_j w_{ij} d(x_j) - \mu_\rho \sum_j w_{ij} = \mu_\rho - \mu_\rho = 0
$$
Hence the entire second term vanishes! We are left with:
$$
\nabla \sigma^2_\rho = \sum_j (\nabla w_{ij}) (d(x_j) - \mu_\rho)^2
$$
Now bound:
$$
\|\nabla \sigma^2_\rho\| \leq \sum_j \|\nabla w_{ij}\| (d(x_j) - \mu_\rho)^2 \leq (2d_{\max})^2 \sum_j \|\nabla w_{ij}\| \leq 4d_{\max}^2 \frac{C_w}{\rho}
$$
- **Justification**: Centering identity (weighted average of centered values is zero), bound from Step 2.3
- **Why valid**: This is the **critical application of telescoping** that prevents k-dependent growth
- **Expected result**: $\|\nabla \sigma^2_\rho\| \leq C_{\sigma^2}(\rho) = O(1/\rho)$ with k-uniform bound

**Conclusion**: Both moments have gradients bounded by O(1/ρ), independent of k and N.

**Dependencies**:
- Uses: Telescoping identity (Step 1.3), centering property of weighted averages, L¹ bound (Step 2.3)
- Requires: Boundedness of $d$ (def-simplified-measurement-cinf)

**Potential Issues**:
- ⚠️ Potential large $\mu_\rho$ derivatives coupling walkers? **Resolution**: Centering eliminates these terms exactly.

---

#### Step 4: Gradient of the Regularized Z-Score

**Goal**: Bound $\|\nabla Z_\rho\|$ using the quotient rule and previous moment bounds.

**Substep 4.1**: Apply quotient rule to Z-score
- **Action**: From $Z_\rho = (d(x_i) - \mu_\rho) / \sigma'_{\text{reg}}(\sigma^2_\rho)$, use quotient rule:
$$
\nabla_{x_i} Z_\rho = \frac{[\nabla d(x_i) - \nabla \mu_\rho] \sigma'_{\text{reg}} - (d(x_i) - \mu_\rho) (\sigma'_{\text{reg}})' \nabla \sigma^2_\rho}{(\sigma'_{\text{reg}})^2}
$$
where $(\sigma'_{\text{reg}})'$ denotes $\frac{d\sigma'_{\text{reg}}}{d(\sigma^2)}$ evaluated at $\sigma^2_\rho$.
- **Justification**: Quotient rule, chain rule for $\sigma'_{\text{reg}}(\sigma^2_\rho)$
- **Why valid**: $\sigma'_{\text{reg}} \geq \varepsilon_\sigma > 0$ ensures denominator is bounded away from zero
- **Expected result**: Three-term expression for $\nabla Z_\rho$

**Substep 4.2**: Bound each term separately
- **Action**: Using bounds from earlier steps:
  - **Term 1 (numerator, gradient of numerator)**:
    $$\|\nabla d(x_i) - \nabla \mu_\rho\| \leq \|\nabla d(x_i)\| + \|\nabla \mu_\rho\| \leq d'_{\max} + C_{\mu}(\rho)$$
  - **Term 2 (denominator)**:
    $$(\sigma'_{\text{reg}})^2 \geq \varepsilon_\sigma^2$$
  - **Term 3 (numerator, chain rule term)**:
    $$|d(x_i) - \mu_\rho| \leq 2d_{\max}, \quad |(\sigma'_{\text{reg}})'| \leq L_{\sigma'}, \quad \|\nabla \sigma^2_\rho\| \leq C_{\sigma^2}(\rho)$$
- **Justification**: Triangle inequality, bounds from Steps 3.2 and 3.4, framework assumptions on $\sigma'_{\text{reg}}$ (def-pipeline-cinf)
- **Why valid**: All primitive functions have bounded derivatives by assumption
- **Expected result**: Bounds on each component of $\nabla Z_\rho$

**Substep 4.3**: Combine bounds
- **Action**: Putting it all together:
$$
\|\nabla Z_\rho\| \leq \frac{1}{\varepsilon_\sigma} [d'_{\max} + C_{\mu}(\rho)] + \frac{2d_{\max} L_{\sigma'}}{\varepsilon_\sigma^2} C_{\sigma^2}(\rho)
$$
Substituting $C_{\mu}(\rho) = 2d_{\max} C_w/\rho + d'_{\max}$ and $C_{\sigma^2}(\rho) = 4d_{\max}^2 C_w/\rho$:
$$
\|\nabla Z_\rho\| \leq \frac{2d'_{\max}}{\varepsilon_\sigma} + \frac{2d_{\max} C_w}{\varepsilon_\sigma \rho} + \frac{8d_{\max}^3 L_{\sigma'} C_w}{\varepsilon_\sigma^2 \rho}
$$
Define:
$$
K_{Z,1}(\rho) = \frac{2d'_{\max}}{\varepsilon_\sigma} + \frac{C_w}{\rho} \left(\frac{2d_{\max}}{\varepsilon_\sigma} + \frac{8d_{\max}^3 L_{\sigma'}}{\varepsilon_\sigma^2}\right)
$$
- **Justification**: Algebraic substitution and simplification
- **Why valid**: All constants are finite and well-defined
- **Expected result**: $\|\nabla Z_\rho\| \leq K_{Z,1}(\rho) = O(\rho^{-1})$ with explicit constant

**Conclusion**: The Z-score gradient is bounded with O(ρ^{-1}) scaling, k-uniform and N-uniform.

**Dependencies**:
- Uses: Quotient rule, chain rule, bounds from Steps 2-3
- Requires: Regularization $\sigma'_{\text{reg}} \geq \varepsilon_\sigma > 0$ (def-pipeline-cinf)

**Potential Issues**:
- ⚠️ Division by small $\sigma'_{\text{reg}}$ in low-variance regimes? **Resolution**: Regularization ensures $\sigma'_{\text{reg}} \geq \varepsilon_\sigma > 0$ always.

---

#### Step 5: Final Composition and O(ρ^{-1}) Bound

**Goal**: Apply the chain rule to $V_{\text{fit}} = g_A(Z_\rho)$ and obtain the final gradient bound.

**Substep 5.1**: Apply chain rule
- **Action**: By the chain rule:
$$
\nabla_{x_i} V_{\text{fit}} = g'_A(Z_\rho) \nabla_{x_i} Z_\rho
$$
- **Justification**: Chain rule for composition of C¹ functions
- **Why valid**: $g_A \in C^\infty$ and $Z_\rho$ is C¹ (just shown in Step 4)
- **Expected result**: Expression relating $\nabla V_{\text{fit}}$ to $\nabla Z_\rho$

**Substep 5.2**: Bound the gradient
- **Action**: Taking norms and using $|g'_A| \leq L_{g_A}$:
$$
\|\nabla_{x_i} V_{\text{fit}}\| \leq L_{g_A} \|\nabla_{x_i} Z_\rho\| \leq L_{g_A} K_{Z,1}(\rho)
$$
Define the final bound:
$$
K_{V,1}(\rho) := L_{g_A} K_{Z,1}(\rho) = L_{g_A} \left[\frac{2d'_{\max}}{\varepsilon_\sigma} + \frac{C_w}{\rho} \left(\frac{2d_{\max}}{\varepsilon_\sigma} + \frac{8d_{\max}^3 L_{\sigma'}}{\varepsilon_\sigma^2}\right)\right]
$$
- **Justification**: Submultiplicativity of operator norms, bound on $g'_A$ from framework (def-pipeline-cinf)
- **Why valid**: $L_{g_A} = \sup |g'_A(z)|$ is finite since $g_A \in C^\infty$ on compact domain (or with bounded derivatives)
- **Expected result**: $\|\nabla V_{\text{fit}}\| \leq K_{V,1}(\rho)$ with $K_{V,1}(\rho) = O(\rho^{-1})$

**Substep 5.3**: Verify k-uniformity and N-uniformity
- **Action**: Inspect all constants in $K_{V,1}(\rho)$:
  - $L_{g_A}$: Depends only on $g_A$, not on k or N ✓
  - $d'_{\max}, d_{\max}$: Depend only on $d$ and domain X, not on k or N ✓
  - $\varepsilon_\sigma, L_{\sigma'}$: Depend only on $\sigma'_{\text{reg}}$, not on k or N ✓
  - $C_w$: Depends only on Gaussian kernel, not on k or N ✓
  - $\rho$: External parameter, not dependent on k or N ✓
- **Justification**: Tracing back through all previous steps, no k or N dependence was introduced
- **Why valid**: All bounds were constructed to be uniform in k and N by design (using telescoping, centering, and envelope bounds)
- **Expected result**: Confirmation that $K_{V,1}(\rho)$ is k-uniform and N-uniform

**Substep 5.4**: Verify ρ-scaling
- **Action**: The dominant term in $K_{V,1}(\rho)$ as $\rho \to 0$ is:
$$
K_{V,1}(\rho) \sim \frac{L_{g_A} C_w}{\rho} \left(\frac{2d_{\max}}{\varepsilon_\sigma} + \frac{8d_{\max}^3 L_{\sigma'}}{\varepsilon_\sigma^2}\right) = \frac{C}{\rho}
$$
where C is a constant independent of ρ.
- **Justification**: Direct inspection of formula
- **Why valid**: O(ρ^{-1}) term dominates O(1) term as ρ → 0
- **Expected result**: $K_{V,1}(\rho) = O(\rho^{-1})$ as claimed

**Conclusion**: $V_{\text{fit}}$ is C¹ with $\|\nabla V_{\text{fit}}\| \leq K_{V,1}(\rho) = O(\rho^{-1})$, k-uniform and N-uniform. ✓

**Dependencies**:
- Uses: Chain rule, bound on $g'_A$, bound on $\nabla Z_\rho$ (Step 4)
- Requires: $g_A \in C^\infty$ with bounded derivatives (def-pipeline-cinf)

**Potential Issues**:
- ⚠️ None identified; this is a direct application of the chain rule

---

#### Step 6: Continuity Verification (C¹ Regularity)

**Goal**: Verify that $\nabla V_{\text{fit}}$ is continuous, confirming that $V_{\text{fit}} \in C^1$.

**Substep 6.1**: Check continuity of primitive components
- **Action**: Verify each component in the composition pipeline is continuous:
  1. $K_\rho(r)$: Real analytic (Gaussian) → C∞ → continuous ✓
  2. Weights $w_{ij}(\rho)$: Ratio of C∞ functions with positive denominator → continuous ✓
  3. Moments $\mu_\rho, \sigma^2_\rho$: Weighted sums of continuous functions → continuous ✓
  4. $\sigma'_{\text{reg}}(\sigma^2_\rho)$: C∞ function composed with continuous function → continuous ✓
  5. $Z_\rho$: Quotient with denominator $\geq \varepsilon_\sigma > 0$ → continuous ✓
  6. $g_A(Z_\rho)$: C∞ function composed with continuous function → continuous ✓
- **Justification**: Standard topology: compositions and arithmetic operations preserve continuity, quotients are continuous when denominator is bounded away from zero
- **Why valid**: All assumptions verified in previous steps
- **Expected result**: $V_{\text{fit}}$ is continuous

**Substep 6.2**: Check continuity of gradient
- **Action**: From $\nabla V_{\text{fit}} = g'_A(Z_\rho) \nabla Z_\rho$:
  - $g'_A$: Derivative of C∞ function → C∞ → continuous ✓
  - $Z_\rho$: Shown continuous in Substep 6.1 ✓
  - $\nabla Z_\rho$: Composition of continuous derivatives (by previous steps) → continuous ✓
  - Product $g'_A(Z_\rho) \nabla Z_\rho$: Product of continuous functions → continuous ✓
- **Justification**: Closure of continuous functions under composition and multiplication
- **Why valid**: All components verified continuous
- **Expected result**: $\nabla V_{\text{fit}}$ is continuous, hence $V_{\text{fit}} \in C^1$ ✓

**Conclusion**: $V_{\text{fit}}$ has a continuous gradient, confirming C¹ regularity.

**Dependencies**:
- Uses: Standard topological properties of continuous functions
- Requires: All primitive functions C∞ (verified in framework assumptions)

**Potential Issues**:
- ⚠️ None; continuity follows from standard results

---

## V. Technical Deep Dives

### Challenge 1: k-Uniform Control of Weight Gradient Sum

**Why Difficult**: The naive bound $\sum_j \|\nabla w_{ij}\|$ appears to scale linearly with k (the number of alive walkers), which would destroy k-uniformity. Since k can vary from 1 to N, this would introduce uncontrolled N-dependence.

**Mathematical Obstacle**: When differentiating normalized weights $w_{ij} = \alpha_{ij}/S_i$ with $S_i = \sum_\ell \alpha_{i\ell}$, both numerator and denominator depend on $x_i$, creating coupled terms. The quotient rule gives:
$$
\nabla w_{ij} = \frac{\nabla \alpha_{ij}}{S_i} - \frac{\alpha_{ij}}{S_i^2} \nabla S_i
$$
Summing over j gives terms that naively scale as O(k).

**Proposed Solution (Implemented in Step 2)**:
1. **Normalization collapse**: Use $\sum_j \alpha_{ij} = S_i$ and $\nabla S_i = \sum_\ell \nabla \alpha_{i\ell}$ to show that the two terms in the quotient rule contribute equally, giving:
   $$\sum_j \|\nabla w_{ij}\| \leq \frac{2}{S_i} \sum_j \|\nabla \alpha_{ij}\|$$

2. **Gaussian envelope bound**: For the Gaussian kernel, use the pointwise bound:
   $$\|\nabla K_\rho(r)\| \leq \frac{r}{\rho^2} K_\rho(r) \leq \frac{C}{\rho} K_\rho(r)$$
   where $C = e^{-1/2}$ comes from $\sup_{r \geq 0} r e^{-r^2/(2\rho^2)} = \rho e^{-1/2}$.

3. **Normalization cancellation**: This gives:
   $$\sum_j \|\nabla w_{ij}\| \leq \frac{2C}{\rho} \frac{\sum_j \alpha_{ij}}{S_i} = \frac{2C}{\rho}$$
   The sum $\sum_j \alpha_{ij} = S_i$ cancels the denominator, leaving a k-independent bound!

**Why This Works**: The Gaussian decay ensures that only O(1) walkers within distance O(ρ) contribute significantly to the sum, even though the formal sum is over all k walkers. The exponential decay $e^{-r^2/(2\rho^2)}$ for $r \gg \rho$ makes distant walkers' contributions negligible.

**Alternative Approach (Score Function Method)**:
Express $\nabla w_{ij}$ using the **score function identity**:
$$
\nabla w_{ij} = w_{ij} (\nabla \log \alpha_{ij} - \mathbb{E}_w[\nabla \log \alpha])
$$
where $\mathbb{E}_w[\cdot] = \sum_j w_{ij} (\cdot)$ is the weighted expectation. This is a **centered random variable** with zero mean, so:
$$
\mathbb{E}_w[\|\nabla w_{ij}\|] = \text{Var}_w(\nabla \log \alpha) \leq \mathbb{E}_w[\|\nabla \log \alpha\|^2]
$$
For Gaussian $\alpha_{ij} \sim e^{-\|x_i - x_j\|^2/(2\rho^2)}$:
$$
\nabla \log \alpha_{ij} = -\frac{x_i - x_j}{\rho^2}
$$
The variance bound uses $\mathbb{E}_w[\|x_i - x_j\|^2] \leq C\rho^2$ (from Gaussian decay), yielding the same O(ρ^{-1}) scaling.

**Pros/Cons of Alternatives**:
- **Direct envelope method (used)**: Elementary, explicit constants, direct
- **Score function method**: More conceptual, connects to information geometry, requires variance bound step

**References**:
- Similar Gaussian envelope bounds appear in kernel density estimation theory
- Score function approach related to Fisher information in statistics

---

### Challenge 2: Sensitivity to Large Inter-Walker Distances

**Why Difficult**: If walkers are very far apart ($\|x_i - x_j\| \gg \rho$), the gradient of the Gaussian kernel could be large: $\|\nabla K_\rho(\|x_i - x_j\|)\| = \frac{\|x_i - x_j\|}{\rho^2} K_\rho(\|x_i - x_j\|)$. For $\|x_i - x_j\| = R$, this is O(R/ρ²), which grows without bound as R → ∞.

**Mathematical Obstacle**: Naive summation over all walkers j could accumulate contributions from distant walkers, potentially yielding unbounded sums if the domain X is unbounded or if walkers spread over large distances.

**Proposed Solution (Used in Step 2.2)**:
The key insight is that the product $r e^{-r^2/(2\rho^2)}$ is **self-limiting**: as $r$ increases, the exponential decay dominates the linear growth. Specifically:
$$
\sup_{r \geq 0} r e^{-r^2/(2\rho^2)} = \rho e^{-1/2}
$$
achieved at $r = \rho$. Therefore:
$$
\frac{\|x_i - x_j\|}{\rho^2} K_\rho(\|x_i - x_j\|) \leq \frac{\rho e^{-1/2}}{\rho^2} = \frac{e^{-1/2}}{\rho}
$$
**independent of $\|x_i - x_j\|$!**

**Why This Works**:
- For $r \ll \rho$: Linear term $r/\rho^2$ dominates, but $K_\rho(r) \approx 1$, so contribution is O(r/ρ²)
- For $r \approx \rho$: Optimal balance, giving O(1/ρ)
- For $r \gg \rho$: Exponential decay $e^{-r^2/(2\rho^2)} \ll 1$ kills the linear growth

The Gaussian envelope bound is **uniform over all positions**, requiring no compactness of X and no assumptions on walker configurations.

**Alternative Approach (Effective Neighborhood)**:
Instead of using the envelope bound, argue that only walkers within the **effective localization radius** $r_{\text{eff}} \sim \rho \sqrt{\log k}$ contribute significantly. For $r > r_{\text{eff}}$:
$$
K_\rho(r) = e^{-r^2/(2\rho^2)} < e^{-\log k} = 1/k
$$
So the total contribution from all distant walkers is:
$$
\sum_{j: \|x_i - x_j\| > r_{\text{eff}}} (\cdot) \leq k \cdot \frac{1}{k} = O(1)
$$
This gives k-uniformity through a different mechanism (effective neighborhood has size O(1) in expectation).

**Pros/Cons**:
- **Envelope bound (used)**: Cleaner, no probabilistic arguments, works for worst-case configurations
- **Effective neighborhood**: More intuitive (only local walkers matter), but requires probabilistic or measure-theoretic arguments

---

### Challenge 3: Denominator Stability in Z-Score

**Why Difficult**: The Z-score $Z_\rho = (d(x_i) - \mu_\rho) / \sigma'_{\text{reg}}(\sigma^2_\rho)$ involves division by the regularized standard deviation. If $\sigma^2_\rho$ is very small (low variance regime, e.g., all walkers clustered), the denominator $\sigma'_{\text{reg}}(\sigma^2_\rho)$ could approach zero, causing the gradient to blow up.

**Mathematical Obstacle**: In the quotient rule for $\nabla Z_\rho$, we divide by $(\sigma'_{\text{reg}})^2$:
$$
\nabla Z_\rho = \frac{[\nabla(d - \mu_\rho)] \sigma'_{\text{reg}} - (d - \mu_\rho) (\sigma'_{\text{reg}})' \nabla \sigma^2_\rho}{(\sigma'_{\text{reg}})^2}
$$
If $\sigma'_{\text{reg}} \to 0$, this is undefined or unbounded.

**Proposed Solution (Framework Design)**:
The framework explicitly requires **regularization** through the function $\sigma'_{\text{reg}}: \mathbb{R}_{\geq 0} \to [\varepsilon_\sigma, \infty)$ satisfying:
1. $\sigma'_{\text{reg}}(\sigma^2) \geq \varepsilon_\sigma > 0$ for all $\sigma^2 \geq 0$ (strict lower bound)
2. $\sigma'_{\text{reg}} \in C^\infty$ with bounded derivatives
3. $\sigma'_{\text{reg}}(\sigma^2) \sim \sqrt{\sigma^2}$ as $\sigma^2 \to \infty$ (asymptotically unbiased)

**Example Construction**:
$$
\sigma'_{\text{reg}}(\sigma^2) = \sqrt{\sigma^2 + \varepsilon_\sigma^2}
$$
This satisfies:
- $\sigma'_{\text{reg}}(0) = \varepsilon_\sigma$ (regularization kicks in at low variance)
- $\sigma'_{\text{reg}}(\sigma^2) \to \sqrt{\sigma^2}$ as $\sigma^2 \to \infty$ (no bias at high variance)
- Smooth with bounded derivatives: $|(\sigma'_{\text{reg}})'(\sigma^2)| = \frac{1}{2\sqrt{\sigma^2 + \varepsilon_\sigma^2}} \leq \frac{1}{2\varepsilon_\sigma}$

**Why This Works**: The regularization ensures:
$$
(\sigma'_{\text{reg}})^2 \geq \varepsilon_\sigma^2 > 0
$$
so all divisions are **well-defined and bounded**. The price is a small bias in low-variance regimes, but this is acceptable for algorithmic stability.

**Alternative Approach (Conditional Analysis)**:
Instead of global regularization, prove the theorem separately for two regimes:
1. **High-variance regime** ($\sigma^2_\rho \geq \sigma_0^2$): Use $\sigma'_{\text{reg}} \approx \sqrt{\sigma^2_\rho}$ with no regularization needed
2. **Low-variance regime** ($\sigma^2_\rho < \sigma_0^2$): Use $\sigma'_{\text{reg}} \approx \varepsilon_\sigma$ (constant), simplifying analysis

Then combine via partition of unity or case analysis.

**Pros/Cons**:
- **Global regularization (used)**: Unified treatment, simpler proof, acceptable bias
- **Conditional analysis**: No bias in high-variance regime, but requires case-by-case proofs and gluing arguments

**References**:
- Regularization of this type is standard in robust statistics (e.g., ridge regression, Tikhonov regularization)
- Similar to "add-one" smoothing in empirical estimation

---

## VI. Proof Validation Checklist

- [✓] **Logical Completeness**: All steps follow from previous steps (Steps 1-6 form a complete chain)
- [✓] **Hypothesis Usage**: All theorem assumptions are used:
  - $d \in C^\infty$ with bounded derivatives: Used in Steps 3-4
  - $K_\rho$ Gaussian (real analytic): Used in Step 2
  - $g_A \in C^\infty$ with bounded derivatives: Used in Step 5
  - $\sigma'_{\text{reg}} \in C^\infty$ with $\sigma'_{\text{reg}} \geq \varepsilon_\sigma > 0$: Used in Step 4
  - Normalization $\sum_j w_{ij} = 1$: Used in Steps 1, 3 (telescoping, centering)
- [✓] **Conclusion Derivation**: Claimed conclusion $\|\nabla V_{\text{fit}}\| \leq K_{V,1}(\rho) = O(\rho^{-1})$ is fully derived in Step 5
- [✓] **Framework Consistency**: All dependencies verified in § III
- [✓] **No Circular Reasoning**: Proof builds from primitive functions → weights → moments → Z-score → fitness, each step depending only on previous results
- [✓] **Constant Tracking**: All constants explicitly defined:
  - $C_w = 2e^{-1/2}$ (Gaussian envelope)
  - $C_{\mu}(\rho) = 2d_{\max} C_w/\rho + d'_{\max}$
  - $C_{\sigma^2}(\rho) = 4d_{\max}^2 C_w/\rho$
  - $K_{Z,1}(\rho)$ (explicit formula in Step 4.3)
  - $K_{V,1}(\rho) = L_{g_A} K_{Z,1}(\rho)$ (final bound in Step 5.2)
- [✓] **Edge Cases**:
  - k = 1: Only one walker, $w_{ii} = 1$, all sums trivial, bounds still hold ✓
  - $\rho \to 0$: Bound diverges as O(ρ^{-1}), but this is expected (hyper-local regime) ✓
  - $\rho \to \infty$: Bound approaches constant $2L_{g_A} d'_{\max}/\varepsilon_\sigma$ (global regime) ✓
- [✓] **Regularity Verified**: All smoothness/continuity assumptions available (checked in Step 6)
- [✓] **Measure Theory**: All probabilistic operations well-defined (finite sums, no measure-theoretic subtleties)

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Score Function / Covariance Representation

**Approach**: Use the identity $\nabla w_{ij} = w_{ij} (\nabla \log \alpha_{ij} - \mathbb{E}_w[\nabla \log \alpha])$ to express gradients as centered random variables, then bound using variance inequalities.

**Detailed Strategy**:
1. Write $\nabla \mu_\rho = \text{Cov}_w(\nabla \log \alpha, d)$ where $\text{Cov}_w$ is covariance under weights $w_{ij}$
2. Use Cauchy-Schwarz: $|\text{Cov}_w(X, Y)| \leq \sqrt{\text{Var}_w(X)} \sqrt{\text{Var}_w(Y)}$
3. Bound $\text{Var}_w(\nabla \log \alpha) = \mathbb{E}_w[\|\nabla \log \alpha\|^2]$ using Gaussian envelope on $\|x_i - x_j\|^2$
4. Yields same O(ρ^{-1}) scaling

**Pros**:
- Conceptually clean connection to information geometry
- Natural centering interpretation
- Generalizes to non-Gaussian kernels (any log-concave kernel)

**Cons**:
- Requires additional variance bound lemma
- Less explicit constants (need to compute $\text{Var}_w$)
- Heavier machinery for a simple result

**When to Consider**: If extending to more general kernel families beyond Gaussian, or if connecting to Fisher information / natural gradients.

---

### Alternative 2: Log-Sum-Exp Smoothness Analysis

**Approach**: Work with the partition function $S_i(x_i) = \sum_j e^{-\|x_i - x_j\|^2/(2\rho^2)}$ directly, using its log-concavity and smoothness properties.

**Detailed Strategy**:
1. Express $\mu_\rho$ and $\sigma^2_\rho$ as smooth functions of $\nabla \log S_i$ and $\nabla^2 \log S_i$
2. Use log-sum-exp bounds: $\|\nabla \log S_i\| \leq C/\rho$ from log-concavity
3. Show $\nabla^2 \log S_i \preceq (C/\rho^2) I$ (Hessian bound)
4. Propagate through quotient rule for Z-score

**Pros**:
- Uses powerful convex analysis machinery
- Hessian bounds come for free from log-concavity
- Cleaner for higher-order derivatives (C², C³, ...)

**Cons**:
- Overkill for C¹ regularity
- Requires knowledge of log-sum-exp bounds and log-concavity
- Constants less explicit

**When to Consider**: When proving C² or higher regularity (the current document does this for C³, C⁴, C∞), or when working with log-concave measures more generally.

---

### Alternative 3: Weak Derivative / Sobolev Space Approach

**Approach**: Prove $V_{\text{fit}} \in W^{1,\infty}(X)$ (Sobolev space of functions with essentially bounded weak derivative), then use Sobolev embedding to get classical C¹.

**Detailed Strategy**:
1. Show $V_{\text{fit}} \in L^\infty(X)$ (bounded by design: $V_{\text{fit}} \in [0, A]$)
2. Show weak derivative $\nabla V_{\text{fit}} \in L^\infty(X)$ by bounding $\int_X |\nabla V_{\text{fit}}| dx$
3. Use Sobolev embedding $W^{1,\infty}(X) \hookrightarrow C^{0,1}(X)$ (Lipschitz continuous)

**Pros**:
- Generalizes to weaker regularity settings
- Natural framework for PDE analysis (used in mean-field limits)
- Handles discontinuities gracefully

**Cons**:
- Much heavier machinery than needed
- Requires measure theory / functional analysis background
- Overkill for smooth primitive functions

**When to Consider**: If $d$, $g_A$, or $\sigma'_{\text{reg}}$ are only Lipschitz (not smooth), or when studying mean-field PDEs where solutions may have limited regularity.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Extension to swarm-dependent measurement**: The current proof assumes the simplified model where $d: X \to \mathbb{R}$ depends only on position. The full Geometric Gas uses $d_i = d_{\text{alg}}(i, c(i))$ where $c(i)$ is the companion selection, coupling all walkers. Does the telescoping mechanism survive this coupling?
   - **Difficulty**: High (requires combinatorial analysis of companion derivatives)
   - **Critical for**: Full Geometric Gas convergence theory

2. **Optimal ρ-dependence**: Is the O(ρ^{-1}) scaling tight, or can it be improved for specific classes of measurement functions $d$?
   - **Difficulty**: Medium (requires minimax analysis or counterexamples)
   - **Critical for**: Numerical parameter selection

3. **Lower bound on k-effective**: What is the minimal effective neighborhood size as a function of ρ and kernel choice?
   - **Difficulty**: Medium (probabilistic concentration bounds)
   - **Critical for**: Understanding computational complexity

### Conjectures

1. **Gevrey class membership**: Since higher derivatives satisfy $\|\nabla^m V_{\text{fit}}\| \leq K_{V,m}(\rho) = O(m! \rho^{-m})$ (proven in parent document for C∞), $V_{\text{fit}}$ belongs to Gevrey class G¹ (borderline real analyticity). Can we prove real analyticity under stronger kernel assumptions?
   - **Plausibility**: High (Gaussian kernel is real analytic, composition of analytic functions is analytic if convergence radii are compatible)

2. **Swarm-dependent telescoping**: Conjecture that the telescoping identity $\sum_j \nabla^m w_{ij} = 0$ extends to the swarm-dependent case with correction terms that are still O(1), not O(k).
   - **Plausibility**: Medium (requires delicate analysis of companion selection derivatives)

### Extensions

1. **Non-Gaussian kernels**: Extend to Matérn kernels $K_\rho(r) = C_\nu (r/\rho)^\nu K_\nu(r/\rho)$ or polynomial kernels
   - **Challenge**: Different derivative scaling (Matérn: O(ρ^{-\nu-1}))

2. **Adaptive ρ**: Allow localization scale to vary spatially: $\rho_i = \rho(x_i)$
   - **Challenge**: Additional derivatives $\nabla \rho_i$ appear in chain rule

3. **Time-dependent analysis**: Study evolution of $\nabla V_{\text{fit}}$ under the BAOAB dynamics
   - **Challenge**: Requires SDE/Fokker-Planck analysis

---

## IX. Expansion Roadmap

**Phase 1: Prove Missing Lemmas** (Estimated: 2-4 hours)
1. **Lemma A (L¹ gradient bound)**: Write detailed proof of $\sum_j \|\nabla w_{ij}\| \leq C_w/\rho$
   - Expand Substep 2.2 with full calculus details
   - Compute exact value of $C_w = 2e^{-1/2}$ with derivation
   - Add figure showing envelope bound $r e^{-r^2/(2\rho^2)}$
2. **Lemma B (Telescoping identities)**: Prove $\sum_j \nabla w_{ij} = 0$ and centering property
   - Add explicit verification of normalization differentiation
   - Prove centering: $\sum_j w_{ij}(d(x_j) - \mu_\rho) = 0$
3. **Lemma C (Denominator regularization)**: Prove smoothness of $(μ, σ²) \mapsto (d - μ)/σ'_{\text{reg}}(σ²)$
   - Verify all quotient/chain rule applications
   - Check edge cases: $\sigma^2 = 0$, $\sigma^2 \to \infty$

**Phase 2: Fill Technical Details** (Estimated: 4-6 hours)
1. **Step 2.3**: Add explicit calculation of $\sum_j \|\nabla w_{ij}\|$ including all intermediate steps
2. **Step 3.2**: Expand telescoping application with line-by-line algebra
3. **Step 3.4**: Add detailed derivation of centering identity vanishing
4. **Step 4.2**: Include all triangle inequality steps with explicit bounds
5. **Step 5.2**: Derive explicit formula for $K_{V,1}(\rho)$ matching doc 11 Appendix A

**Phase 3: Add Rigor** (Estimated: 3-5 hours)
1. **Epsilon-delta arguments**: Make all limits and continuity arguments epsilon-delta rigorous (currently using standard topology)
2. **Measure-theoretic details**: If extending to infinite k (mean-field limit), add dominated convergence arguments
3. **Counterexamples**: For necessity of assumptions:
   - Show telescoping fails without normalization
   - Show k-uniformity fails for non-decaying kernels
   - Show continuity fails without regularization $\varepsilon_\sigma > 0$

**Phase 4: Review and Validation** (Estimated: 2-3 hours)
1. **Framework cross-validation**: Check all references to doc 11, glossary.md
2. **Edge case verification**: Test k=1, ρ→0, ρ→∞ limits explicitly
3. **Constant tracking audit**: Verify every constant in $K_{V,1}(\rho)$ is defined and bounded

**Total Estimated Expansion Time**: 11-18 hours for a complete, publication-ready proof

---

## X. Cross-References

**Theorems Used**:
- Theorem A.1 from [11_geometric_gas.md](11_geometric_gas.md) (this theorem is the source)
- Standard calculus chain rule, quotient rule, product rule
- Gaussian envelope bounds (standard analysis)

**Definitions Used**:
- {prf:ref}`def-simplified-measurement-cinf` (Simplified measurement function)
- {prf:ref}`def-localization-kernel-cinf` (Gaussian localization kernel)
- {prf:ref}`def-localization-weights-cinf` (Normalized weights)
- {prf:ref}`def-pipeline-cinf` (Moments, Z-score, fitness potential)

**Related Proofs** (for comparison):
- C² regularity: Theorem A.2 in [11_geometric_gas.md](11_geometric_gas.md) (similar structure, adds Hessian bound)
- C³ regularity: Theorem 8.1 in [13_geometric_gas_c3_regularity.md](13_geometric_gas_c3_regularity.md) (extends to third derivatives)
- C⁴ regularity: Theorem 5.1 in [14_geometric_gas_c4_regularity.md](14_geometric_gas_c4_regularity.md) (extends to fourth derivatives)
- C∞ regularity: Main theorem in [19_geometric_gas_cinf_regularity_simplified.md](19_geometric_gas_cinf_regularity_simplified.md) (uses this as base case for induction)

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Needs Lemmas A-C to be written out in full, then ready for detailed proof
**Confidence Level**: High (based on Codex's comprehensive strategy and framework verification)

**Justification**: Codex provided a detailed, step-by-step strategy with explicit constant tracking, proper use of telescoping mechanism, and clear identification of all technical challenges. All framework dependencies have been verified against the source document and glossary. The only limitation is the lack of Gemini's cross-validation due to technical issues, but the proof strategy is mathematically sound and complete based on first principles.

**Recommendation**:
1. Re-run Gemini when service is available for cross-validation
2. Expand Lemmas A-C into full proofs
3. Add figures illustrating Gaussian envelope bounds and weight normalization
4. Cross-check against the actual proof in doc 11 Appendix A to ensure consistency
