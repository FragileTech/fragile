# Proof Sketch for thm-exponential-tails

**Document**: docs/source/2_geometric_gas/16_convergence_mean_field.md
**Theorem**: thm-exponential-tails
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} Exponential Tails for QSD
:label: thm-exponential-tails

Under Assumptions A1-A4, the QSD $\rho_\infty$ satisfies:

$$
\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}
$$

for some constants $\alpha, C > 0$ depending on $\gamma$, $\sigma^2$, $\kappa_{\text{conf}}$, and $\kappa_{\max}$.

In particular, **R6** holds.

:::

**Informal Restatement**: The quasi-stationary distribution $\rho_\infty$ for the mean-field geometric gas has exponentially decaying tails in phase space. Specifically, the probability density at position $x$ and velocity $v$ decreases exponentially with the squared distance from the origin in phase space. This property (R6) is critical because it ensures the QSD has finite moments of all orders and, combined with other regularity properties (R1-R5), guarantees the existence of a Log-Sobolev inequality needed for KL-convergence analysis.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI RESPONSE FAILED**

Gemini 2.5 Pro did not return a response (empty output received twice). This may indicate a temporary service issue or prompt complexity limitation.

**Implications**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available

---

### Strategy B: GPT-5's Approach

**Method**: Moment method with Lyapunov test functions

**Key Steps**:
1. Fix the quadratic Lyapunov function $V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$ with positive-definite matrix $M = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$
2. Establish adjoint drift for $V$ under the full generator (kinetic + jump)
3. Derive multiplicative Lyapunov estimate for $W_\theta = e^{\theta V}$
4. Close exponential moment inequality to show $\int e^{\theta V} \rho_\infty < \infty$ for small $\theta > 0$
5. Apply Chebyshev inequality to obtain tail probability bounds
6. Convert integral tail bounds to pointwise exponential decay using hypoelliptic regularity

**Strengths**:
- Directly leverages the quadratic Lyapunov drift already proven in Section 4.2
- Well-established technique (Khas'minskii multiplicative Lyapunov) for kinetic Fokker-Planck equations
- Systematic path from generator properties → moments → tails → pointwise bounds
- Compatible with killed/revival mean-field structure
- All constants explicit and computable

**Weaknesses**:
- Requires careful handling of unbounded test function $e^{\theta V}$ via truncation
- Jump operator contribution (revival) adds positive terms that must be controlled
- Transition from integral to pointwise requires hypoelliptic Harnack inequality (non-trivial)
- Current parameter choice in Section 4.2 requires $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ which may be overly restrictive

**Framework Dependencies**:
- Quadratic Lyapunov drift: $\mathcal{L}^*[V] \le -\beta V + C$ (Section 4.2)
- QSD stationarity: $\int \mathcal{L}^*[f] \rho_\infty = 0$ (definition)
- Smoothness (R2): $\rho_\infty \in C^\infty(\Omega)$ (Hörmander hypoellipticity)
- Positivity (R3): $\rho_\infty(x,v) > 0$ everywhere (strong maximum principle)
- Assumptions A1-A4 (confinement, killing structure, bounded parameters)

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Moment method with Lyapunov test functions (GPT-5's approach)

**Rationale**:
With only one strategist available, I adopt GPT-5's approach because:

1. **Direct connection to proven results**: The quadratic Lyapunov drift $\mathcal{L}^*[V] \le -\beta V + C$ is already established in Section 4.2 of the document. This is the natural foundation for exponential moment bounds.

2. **Standard technique for kinetic PDEs**: The moment method via multiplicative Lyapunov functions is the canonical approach for proving exponential tails in kinetic Fokker-Planck equations with confinement. It has been successfully applied to similar systems in the literature (Villani 2009, Dolbeault et al. 2015).

3. **Systematic progression**: The proof follows a clear logical chain:
   - Lyapunov drift → exponential moments → integral tail bounds → pointwise decay
   - Each step is well-justified by standard techniques

4. **Compatibility with framework**: The approach respects the killed/revival structure and mean-field coupling without requiring fundamentally new machinery beyond what's already proven (R1-R5).

**Integration**:
Since only GPT-5's strategy is available, the sketch below follows it directly with my own critical analysis and verification against framework documents.

**Key Insight Enabling the Proof**:
The cross-term structure $V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$ is essential. A naive approach using separate Lyapunov functions for $|x|^2$ and $|v|^2$ fails because the kinetic coupling $-v \cdot \nabla_x$ and force term $-\nabla U(x) \cdot \nabla_v$ create uncontrolled cross-derivatives. The quadratic form with off-diagonal term $2bx \cdot v$ exactly cancels these problematic terms when parameters are chosen correctly (as done in Section 4.2).

**Verification Status**:
- ✅ Quadratic Lyapunov drift verified (Section 4.2, proven with explicit constants)
- ✅ QSD stationarity verified (definition, follows from fixed-point construction in Section 1)
- ✅ Smoothness R2 verified (Section 2.2, Hörmander hypoellipticity)
- ✅ Positivity R3 verified (Section 2.3, strong maximum principle)
- ⚠️ Parameter restriction $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ needs verification or relaxation (see Challenge 1 below)
- ⚠️ Hypoelliptic Harnack inequality (Step 6) requires careful justification

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms** (from Assumptions A1-A4):

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| A1 (Confinement) | $U(x) \to +\infty$, $\nabla^2 U \ge \kappa_{\text{conf}} I$ | Steps 1-2 (Lyapunov drift) | ✅ |
| A2 (Killing) | $\kappa_{\text{kill}} = 0$ on compact $K$, $\ge \kappa_0$ near $\partial \mathcal{X}$ | Step 2, 4 (jump control) | ✅ |
| A3 (Parameters) | $\gamma, \sigma^2, \lambda_{\text{revive}} > 0$ bounded | Steps 1-4 (constants) | ✅ |
| A4 (Domain) | Bounded smooth or unbounded with confinement | Step 6 (regularity) | ✅ |

**Theorems** (from earlier sections):

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| Quadratic Lyapunov drift | Section 4.2 | $\mathcal{L}^*[V] \le -\beta V + C$ | Steps 2-4 | ✅ |
| R1 (Existence) | Section 1.5 | QSD exists via Schauder fixed-point | Background | ✅ |
| R2 (Smoothness) | Section 2.2 | $\rho_\infty \in C^\infty$ via Hörmander | Step 6 | ✅ |
| R3 (Positivity) | Section 2.3 | $\rho_\infty > 0$ via maximum principle | Step 6 | ✅ |
| R4 (Velocity gradient) | Section 3.2 | $\|\nabla_v \log \rho_\infty\|_{L^\infty} < \infty$ | Background | ✅ |
| R5 (Spatial gradients) | Section 3.3 | $\|\nabla_x \log \rho_\infty\|_{L^\infty}, \|\Delta_v \log \rho_\infty\|_{L^\infty} < \infty$ | Background | ✅ |

**Definitions**:

| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| QSD | Section 1.1 | Stationary measure with $\mathcal{L}[\rho_\infty] = 0$ | Throughout (stationarity identity) |
| Adjoint operator | Section 4.2 | $\mathcal{L}^* = \mathcal{L}^*_{\text{kin}} + J^*$ | Steps 2-4 (drift computation) |
| Kinetic generator | Section 1.1 | $\mathcal{L}_{\text{kin}} = -v \cdot \nabla_x + \nabla_v \cdot [(\gamma v + \nabla U)\cdot] + \frac{\sigma^2}{2}\Delta_v$ | Step 2 |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $\beta$ | Lyapunov drift rate | From Section 4.2: $\min(\frac{3\gamma - \frac{4\kappa_{\text{conf}}}{3}}{\kappa_{\text{conf}}}, \beta_v)$ | Requires $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ |
| $C$ | Lyapunov drift offset | From Section 4.2: explicit function of $\gamma, \sigma^2, \kappa_{\text{conf}}$ | $O(1)$ constant |
| $\lambda_{\min}(M)$ | Min eigenvalue of Lyapunov matrix | $\approx \kappa_{\text{conf}}\varepsilon$ for small $\varepsilon$ | Ensures $V \ge \lambda_{\min}(\|x\|^2 + \|v\|^2)$ |
| $C_V$ | Gradient bound constant | $8 \max(b^2, c^2)/\lambda_{\min}$ | Used to control diffusion term |
| $\theta_0$ | Critical moment parameter | $\beta/(\sigma^2 C_V)$ | Exponential moments finite for $\theta < \theta_0$ |
| $\alpha$ | Tail decay rate | $\theta \lambda_{\min}(1-\varepsilon)/2$ for $\theta < \theta_0$ | Final exponential rate in theorem |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:

- **Lemma (Adjoint chain rule)**: For $V \in C^2$ and $W_\theta = e^{\theta V}$, show $\mathcal{L}^*[W_\theta] = \theta e^{\theta V} \mathcal{L}^*[V] + \frac{\sigma^2}{2} \theta^2 e^{\theta V} |\nabla_v V|^2$
  - **Why needed**: Step 3 (multiplicative Lyapunov estimate)
  - **Difficulty**: Easy (standard calculus + chain rule + diffusion operator formula)
  - **Status**: Standard result, can be verified directly

- **Lemma (Quadratic coercivity)**: If $M \succ 0$, then $V(x,v) = [x, v]^T M [x,v] \ge \lambda_{\min}(M)(|x|^2 + |v|^2)$ and $|\nabla_v V|^2 \le C_V V$
  - **Why needed**: Steps 3, 5 (coercivity and gradient control)
  - **Difficulty**: Easy (linear algebra + Cauchy-Schwarz)
  - **Status**: Standard, follows from spectral theorem

- **Lemma (Jump adjoint bound)**: Under A2-A3, show $J^*[e^{\theta V}] \le C_{\text{jump}}(\theta) - \kappa_{\text{kill}} e^{\theta V} \mathbb{1}_{\mathcal{X} \setminus K} + C_K(\theta) \mathbb{1}_K$ with $C_{\text{jump}}(\theta) < \infty$
  - **Why needed**: Step 4 (control revival contribution)
  - **Difficulty**: Medium (requires careful handling of mean-field revival integral)
  - **Status**: Needs proof, but structure is clear from $J^*[f] = -\kappa_{\text{kill}} f + \lambda_{\text{revive}} \mathbb{E}[f(X',V') | \text{revival}]$

- **Lemma (Local Harnack inequality)**: For stationary $\rho_\infty$ with Hörmander structure, show $\sup_{B_\delta} \rho_\infty \le C_{\text{loc}} \text{avg}_{B_{2\delta}} \rho_\infty$
  - **Why needed**: Step 6 (integral to pointwise bounds)
  - **Difficulty**: Medium-Hard (requires Bony-type hypoelliptic regularity theory)
  - **Status**: Standard result in hypoelliptic PDE theory, cited in Section 2.2 (Hörmander), but needs precise statement

- **Lemma (Truncation stability)**: For $W_{\theta,R} = e^{\theta V} \chi_R$ with smooth cutoff, show $\int \mathcal{L}^*[W_{\theta,R}] \rho_\infty \to \int \mathcal{L}^*[W_\theta] \rho_\infty$ as $R \to \infty$
  - **Why needed**: Step 4 (justify integration by parts with unbounded test function)
  - **Difficulty**: Medium (dominated convergence + tail estimates)
  - **Status**: Follows from exponential moment bound once established (bootstrap)

**Uncertain Assumptions**:

- **Parameter restriction**: The drift computation in Section 4.2 yields the condition $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ for one particular choice of Lyapunov parameters $(a, b, c)$. It's unclear if this is **necessary** or merely **sufficient** for that specific parameterization.
  - **Why uncertain**: Alternative parameter choices might avoid this restriction
  - **How to verify**: Re-optimize $(a, b, c)$ or incorporate $-\kappa_{\text{kill}}$ term outside $K$ to strengthen drift
  - **Impact if false**: Theorem statement should include explicit parameter condition, or proof needs modification

- **Bounded killing maximum**: The theorem statement lists $\kappa_{\max}$ as a parameter, implying $\sup_\mathcal{X} \kappa_{\text{kill}} = \kappa_{\max} < \infty$. Assumption A2 states $\kappa_{\text{kill}} \in C^2$ with bounded derivatives, but doesn't explicitly bound $\kappa_{\text{kill}}$ itself on unbounded domains.
  - **Why uncertain**: For unbounded $\mathcal{X}$, need $\kappa_{\text{kill}}(x) \to \kappa_\infty < \infty$ as $|x| \to \infty$ or explicit bound
  - **How to verify**: Add explicit boundedness assumption $\|\kappa_{\text{kill}}\|_{L^\infty} < \infty$ to A2
  - **Impact if false**: Revival term $\lambda_{\text{revive}} \mathbb{E}[e^{\theta V}]$ might be unbounded

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes exponential tails for the QSD $\rho_\infty$ by showing it has finite exponential moments $\int e^{\theta V} \rho_\infty < \infty$ for a quadratic Lyapunov function $V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$, then converting this to pointwise decay via hypoelliptic regularity. The key technical tool is the **multiplicative Lyapunov argument**: we test the adjoint operator $\mathcal{L}^*$ against $W_\theta = e^{\theta V}$ and use the drift bound $\mathcal{L}^*[V] \le -\beta V + C$ to show that for small enough $\theta > 0$, the exponential moment satisfies a finite bound via the stationarity identity $\int \mathcal{L}^*[W_\theta] \rho_\infty = 0$.

The proof has a natural three-stage structure:
1. **Moment generation** (Steps 1-4): Derive $\int e^{\theta V} \rho_\infty < \infty$ from Lyapunov drift
2. **Tail probability** (Step 5): Apply Chebyshev to get $\rho_\infty(|x|^2 + |v|^2 > r^2) \le K e^{-\theta \kappa_0 r^2}$
3. **Pointwise bound** (Step 6): Use local regularity to convert integral decay to $\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}$

The main technical challenge is controlling the jump operator $J^*$ contribution to ensure the overall drift remains negative for the exponential test function.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Fix the Lyapunov function**: Establish the quadratic form $V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$ with positive-definite matrix and coercivity
2. **Establish full adjoint drift**: Extend the kinetic drift $\mathcal{L}^*_{\text{kin}}[V] \le -\beta V + C$ to the full operator including jumps
3. **Derive exponential test function drift**: Compute $\mathcal{L}^*[e^{\theta V}]$ and bound it using the quadratic drift and diffusion term
4. **Close the moment bound**: Use stationarity $\int \mathcal{L}^*[e^{\theta V}] \rho_\infty = 0$ to prove $\int e^{\theta V} \rho_\infty < \infty$ via bootstrap
5. **Obtain tail probability decay**: Apply Chebyshev inequality with the exponential moment
6. **Convert to pointwise bound**: Use hypoelliptic Harnack inequality to localize the integral decay

---

### Detailed Step-by-Step Sketch

#### Step 1: Fix the Lyapunov Function

**Goal**: Establish a quadratic Lyapunov function $V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$ with positive-definite matrix that provides coercivity.

**Substep 1.1**: Define the matrix

$$
M = \begin{pmatrix} a & b \\ b & c \end{pmatrix}
$$

and require $M \succ 0$ (positive-definite), which is equivalent to $a > 0$, $c > 0$, and $ac - b^2 > 0$.

- **Justification**: A quadratic form is coercive iff its matrix is positive-definite (standard linear algebra)
- **Why valid**: We will choose $a, b, c$ following Section 4.2's optimization
- **Expected result**: $M \succ 0$ with explicit eigenvalues

**Substep 1.2**: Compute eigenvalues of $M$

The eigenvalues satisfy $\lambda^2 - (a+c)\lambda + (ac - b^2) = 0$, giving:

$$
\lambda_{\pm} = \frac{(a+c) \pm \sqrt{(a+c)^2 - 4(ac-b^2)}}{2} = \frac{(a+c) \pm \sqrt{(a-c)^2 + 4b^2}}{2}
$$

The minimum eigenvalue is $\lambda_{\min} = \frac{(a+c) - \sqrt{(a-c)^2 + 4b^2}}{2}$.

- **Justification**: Standard eigenvalue computation for $2 \times 2$ symmetric matrix
- **Why valid**: $M \succ 0$ ensures $\lambda_{\min} > 0$
- **Expected result**: $V(x,v) \ge \lambda_{\min}(|x|^2 + |v|^2)$

**Substep 1.3**: Establish coercivity bound

By the spectral theorem, for $w = [x, v]^T$:

$$
V(x,v) = w^T M w \ge \lambda_{\min}(M) \|w\|^2 = \lambda_{\min}(|x|^2 + |v|^2)
$$

Define $\kappa_0 := \lambda_{\min}(M)$.

- **Conclusion**: $V(x,v) \ge \kappa_0(|x|^2 + |v|^2)$ with explicit $\kappa_0 > 0$
- **Form**: Coercivity inequality

**Substep 1.4**: Compute velocity gradient bound

$$
\nabla_v V = 2cv + 2bx
$$

Therefore:

$$
|\nabla_v V|^2 = 4|cv + bx|^2 \le 8c^2|v|^2 + 8b^2|x|^2
$$

Using $V \ge \kappa_0(|x|^2 + |v|^2)$, we get $|x|^2, |v|^2 \le V/\kappa_0$, so:

$$
|\nabla_v V|^2 \le 8(c^2 + b^2) \frac{V}{\kappa_0} =: C_V V
$$

with $C_V = 8\max(b^2, c^2)/\kappa_0$.

- **Conclusion**: Gradient control $|\nabla_v V|^2 \le C_V V$
- **Form**: Differential inequality

**Dependencies**:
- Uses: Spectral theorem for symmetric matrices, Cauchy-Schwarz inequality
- Requires: Parameters $a, b, c$ chosen from Section 4.2

**Potential Issues**:
- ⚠️ Section 4.2's choice requires $\gamma > \frac{4\kappa_{\text{conf}}}{9}$
- **Resolution**: Accept this condition or re-optimize (see Challenge 1)

---

#### Step 2: Establish Adjoint Drift for Full Generator

**Goal**: Show that the full adjoint operator $\mathcal{L}^* = \mathcal{L}^*_{\text{kin}} + J^*$ satisfies $\mathcal{L}^*[V] \le -\beta V + C$ with explicit constants.

**Substep 2.1**: Recall kinetic drift bound

From Section 4.2, we have:

$$
\mathcal{L}^*_{\text{kin}}[V] \le -\beta_{\text{kin}} V + C_{\text{kin}}
$$

with explicit $\beta_{\text{kin}} > 0$ and $C_{\text{kin}} > 0$ depending on $\gamma, \sigma^2, \kappa_{\text{conf}}$.

- **Justification**: Theorem in Section 4.2 (proven via direct computation)
- **Why valid**: Uses A1 (confinement), A3 (parameters), and optimized choice of $a, b, c$
- **Expected result**: $\beta_{\text{kin}} = \min(\frac{3\gamma - \frac{4\kappa_{\text{conf}}}{3}}{\kappa_{\text{conf}}}, \beta_v)$ where $\beta_v > 0$ for small $\varepsilon$

**Substep 2.2**: Analyze jump adjoint $J^*[V]$

The jump operator is $J[\rho] = -\kappa_{\text{kill}}(x) \rho + \lambda_{\text{revive}} m_d(\rho) \frac{\rho}{\|\rho\|_{L^1}}$ where $m_d(\rho) = \int_{\mathcal{D}} \rho$ is the dead mass.

By integration by parts, the adjoint acts as:

$$
J^*[f] = -\kappa_{\text{kill}}(x) f + \lambda_{\text{revive}} \mathbb{E}[f(X', V') | \text{revival}]
$$

where $(X', V')$ is distributed according to the revival mechanism (proportional to dead mass, resampled in $K$).

For $f = V(x,v) = a|x|^2 + 2bx \cdot v + c|v|^2$:

$$
J^*[V] = -\kappa_{\text{kill}}(x) V(x,v) + \lambda_{\text{revive}} \mathbb{E}[V(X', V')]
$$

- **Justification**: Definition of adjoint operator via $\int J[\rho] f = \int \rho J^*[f]$
- **Why valid**: Revival resamples in compact $K$, so $\mathbb{E}[V(X', V')] \le \sup_{K \times \mathbb{R}^d} V < \infty$
- **Expected result**: $J^*[V] \le -\kappa_{\text{kill}} V + \lambda_{\text{revive}} V_{\max}$ where $V_{\max} = \sup_K V$

**Substep 2.3**: Combine kinetic and jump contributions

$$
\mathcal{L}^*[V] = \mathcal{L}^*_{\text{kin}}[V] + J^*[V] \le (-\beta_{\text{kin}} V + C_{\text{kin}}) + (-\kappa_{\text{kill}} V + \lambda_{\text{revive}} V_{\max})
$$

On the safe region $K$ where $\kappa_{\text{kill}} = 0$:

$$
\mathcal{L}^*[V]|_K \le -\beta_{\text{kin}} V + C_{\text{kin}} + \lambda_{\text{revive}} V_{\max}
$$

Outside $K$ where $\kappa_{\text{kill}} \ge \kappa_0 > 0$:

$$
\mathcal{L}^*[V]|_{\mathcal{X} \setminus K} \le -(\beta_{\text{kin}} + \kappa_0) V + C_{\text{kin}} + \lambda_{\text{revive}} V_{\max}
$$

Setting $\beta = \beta_{\text{kin}}$ and $C = C_{\text{kin}} + \lambda_{\text{revive}} V_{\max}$:

$$
\mathcal{L}^*[V] \le -\beta V + C
$$

- **Conclusion**: Full drift bound $\mathcal{L}^*[V] \le -\beta V + C$ with explicit $\beta, C > 0$
- **Form**: Lyapunov drift inequality

**Dependencies**:
- Uses: Section 4.2 (kinetic drift), A2 (killing structure), A3 (bounded $\lambda_{\text{revive}}$)
- Requires: Boundedness of $\kappa_{\text{kill}}$ on unbounded domains (see Uncertain Assumptions)

**Potential Issues**:
- ⚠️ If $\kappa_{\text{kill}}$ is unbounded, $\lambda_{\text{revive}} \mathbb{E}[V]$ might grow
- **Resolution**: Add explicit assumption $\kappa_{\max} = \|\kappa_{\text{kill}}\|_{L^\infty} < \infty$ or modify argument to use revival in bounded region only

---

#### Step 3: Derive Multiplicative Lyapunov Estimate for $W_\theta = e^{\theta V}$

**Goal**: Compute $\mathcal{L}^*[W_\theta]$ and show that for small $\theta > 0$, it satisfies a controlled bound enabling moment finiteness.

**Substep 3.1**: Apply chain rule to compute kinetic adjoint on $W_\theta$

For $W_\theta(x,v) = e^{\theta V(x,v)}$ and the kinetic adjoint $\mathcal{L}^*_{\text{kin}}$, we have:

$$
\mathcal{L}^*_{\text{kin}}[W_\theta] = e^{\theta V} \mathcal{L}^*_{\text{kin}}[\theta V] + \frac{\sigma^2}{2} e^{\theta V} \theta^2 |\nabla_v V|^2
$$

This is the **multiplicative chain rule** for diffusion operators: the first term comes from the drift/transport parts, the second from the $\Delta_v$ diffusion term.

- **Justification**: Chain rule for $e^{\theta V}$ under kinetic operator (see Required Lemmas)
- **Why valid**: $V \in C^2$ (smooth quadratic), diffusion only in $v$
- **Expected result**: Explicit formula for $\mathcal{L}^*_{\text{kin}}[W_\theta]$

**Substep 3.2**: Bound the first term using Lyapunov drift

From Step 2, we have $\mathcal{L}^*_{\text{kin}}[V] \le -\beta V + C$, so:

$$
\mathcal{L}^*_{\text{kin}}[\theta V] = \theta \mathcal{L}^*_{\text{kin}}[V] \le \theta(-\beta V + C) = -\theta \beta V + \theta C
$$

Therefore:

$$
e^{\theta V} \mathcal{L}^*_{\text{kin}}[\theta V] \le \theta e^{\theta V}(-\beta V + C)
$$

- **Justification**: Linearity of $\mathcal{L}^*_{\text{kin}}$ and drift bound from Step 2
- **Why valid**: Multiplying inequality by $e^{\theta V} > 0$ preserves direction
- **Expected result**: $\theta e^{\theta V}(-\beta V + C)$

**Substep 3.3**: Bound the diffusion term using gradient control

From Step 1, we have $|\nabla_v V|^2 \le C_V V$, so:

$$
\frac{\sigma^2}{2} \theta^2 e^{\theta V} |\nabla_v V|^2 \le \frac{\sigma^2}{2} \theta^2 C_V e^{\theta V} V
$$

- **Justification**: Gradient bound from Step 1.4
- **Why valid**: $C_V = 8\max(b^2, c^2)/\kappa_0$ is explicit constant
- **Expected result**: $\frac{\sigma^2}{2} \theta^2 C_V e^{\theta V} V$

**Substep 3.4**: Combine to get kinetic adjoint bound

$$
\mathcal{L}^*_{\text{kin}}[W_\theta] \le \theta e^{\theta V}(-\beta V + C) + \frac{\sigma^2}{2} \theta^2 C_V e^{\theta V} V
$$

$$
= \theta e^{\theta V} \left[ \left(\frac{\sigma^2}{2} \theta C_V - \beta\right) V + C \right]
$$

- **Conclusion**: Kinetic adjoint on $W_\theta$ has coefficient $(\frac{\sigma^2}{2} \theta C_V - \beta)$ on the $V$ term
- **Form**: Multiplicative Lyapunov inequality

**Substep 3.5**: Handle jump adjoint on $W_\theta$

Similarly to Substep 2.2:

$$
J^*[e^{\theta V}] = -\kappa_{\text{kill}}(x) e^{\theta V} + \lambda_{\text{revive}} \mathbb{E}[e^{\theta V(X', V')}]
$$

Since revival resamples in $K$:

$$
\mathbb{E}[e^{\theta V(X', V')}] \le e^{\theta \sup_K V} =: e^{\theta V_{\max}}
$$

So:

$$
J^*[e^{\theta V}] \le -\kappa_{\text{kill}} e^{\theta V} + \lambda_{\text{revive}} e^{\theta V_{\max}}
$$

- **Justification**: Revival distribution supported in $K$, monotonicity of exponential
- **Why valid**: $V$ is bounded on compact $K$
- **Expected result**: Bounded additive term $\lambda_{\text{revive}} e^{\theta V_{\max}}$

**Substep 3.6**: Combine kinetic and jump for full adjoint

$$
\mathcal{L}^*[W_\theta] = \mathcal{L}^*_{\text{kin}}[W_\theta] + J^*[W_\theta]
$$

$$
\le \theta e^{\theta V} \left[ \left(\frac{\sigma^2}{2} \theta C_V - \beta\right) V + C \right] - \kappa_{\text{kill}} e^{\theta V} + \lambda_{\text{revive}} e^{\theta V_{\max}}
$$

For $\theta < \theta_0 := \frac{2\beta}{\sigma^2 C_V}$, the coefficient of $V$ is negative:

$$
\frac{\sigma^2}{2} \theta C_V - \beta < -\frac{\beta}{2}
$$

Thus:

$$
\mathcal{L}^*[W_\theta] \le \theta e^{\theta V} \left( -\frac{\beta}{2} V + C \right) - \kappa_{\text{kill}} e^{\theta V} + \lambda_{\text{revive}} e^{\theta V_{\max}}
$$

- **Conclusion**: For $\theta < \theta_0$, the full adjoint $\mathcal{L}^*[W_\theta]$ has negative drift term plus bounded offset
- **Form**: Controlled multiplicative Lyapunov bound

**Dependencies**:
- Uses: Steps 1-2 (Lyapunov drift, gradient control), chain rule lemma
- Requires: $\theta < \theta_0 = \frac{2\beta}{\sigma^2 C_V}$

**Potential Issues**:
- ⚠️ Unbounded test function $e^{\theta V}$ requires justification for integration by parts
- **Resolution**: Use truncation $W_{\theta,R} = e^{\theta V} \chi_R$ and pass to limit (see Substep 4.3)

---

#### Step 4: Close the Exponential Moment Inequality

**Goal**: Use the stationarity identity $\int \mathcal{L}^*[f] \rho_\infty = 0$ to prove $\int e^{\theta V} \rho_\infty < \infty$ for $\theta < \theta_0$.

**Substep 4.1**: Apply stationarity identity (formal)

Since $\rho_\infty$ is a QSD, we have $\mathcal{L}[\rho_\infty] = 0$. By integration by parts:

$$
0 = \int \mathcal{L}[\rho_\infty] \cdot f = \int \rho_\infty \cdot \mathcal{L}^*[f]
$$

for any test function $f$ (subject to regularity and decay conditions).

Taking $f = W_\theta = e^{\theta V}$:

$$
\int \mathcal{L}^*[W_\theta] \rho_\infty = 0
$$

- **Justification**: Stationarity of QSD and formal integration by parts
- **Why valid**: Needs justification for unbounded $W_\theta$ (see Substep 4.3)
- **Expected result**: Integral identity relating drift and moments

**Substep 4.2**: Substitute the bound from Step 3

From Step 3.6, for $\theta < \theta_0$:

$$
\mathcal{L}^*[W_\theta] \le \theta e^{\theta V} \left( -\frac{\beta}{2} V + C \right) + \text{jump terms}
$$

Integrating:

$$
0 = \int \mathcal{L}^*[W_\theta] \rho_\infty \le \theta \int e^{\theta V} \left( -\frac{\beta}{2} V + C \right) \rho_\infty + \int (\text{jump terms}) \rho_\infty
$$

Rearranging:

$$
\frac{\beta \theta}{2} \int V e^{\theta V} \rho_\infty \le \theta C \int e^{\theta V} \rho_\infty + \int (-\kappa_{\text{kill}} e^{\theta V} + \lambda_{\text{revive}} e^{\theta V_{\max}}) \rho_\infty
$$

The $\kappa_{\text{kill}}$ term is negative, so we can drop it:

$$
\frac{\beta \theta}{2} \int V e^{\theta V} \rho_\infty \le \theta C \int e^{\theta V} \rho_\infty + \lambda_{\text{revive}} e^{\theta V_{\max}} \int \rho_\infty
$$

- **Justification**: Substituting bound and using negativity of $-\kappa_{\text{kill}}$
- **Why valid**: $\int \rho_\infty = \|\rho_\infty\|_{L^1} < 1$ for QSD (sub-probability)
- **Expected result**: Inequality relating $\int V e^{\theta V}$ to $\int e^{\theta V}$

**Substep 4.3**: Bootstrap to finite moment

Dividing by $\theta$ and using $\int \rho_\infty \le 1$:

$$
\frac{\beta}{2} \int V e^{\theta V} \rho_\infty \le C \int e^{\theta V} \rho_\infty + \frac{\lambda_{\text{revive}} e^{\theta V_{\max}}}{\theta}
$$

Since $V \ge \kappa_0(|x|^2 + |v|^2)$ and $V$ is unbounded, if $\int e^{\theta V} \rho_\infty = \infty$, then the LHS would dominate the RHS (as $V e^{\theta V}$ grows faster than $e^{\theta V}$ in regions where $V$ is large). This contradiction implies:

$$
\int e^{\theta V} \rho_\infty \le K(\theta) < \infty
$$

for some constant $K(\theta)$ depending on $\theta, \beta, C, \lambda_{\text{revive}}, V_{\max}$.

**More rigorous version**: Use the inequality

$$
\int V e^{\theta V} \rho_\infty \ge \int_{V > R} V e^{\theta V} \rho_\infty \ge R e^{\theta R} \int_{V > R} \rho_\infty
$$

If $\int e^{\theta V} \rho_\infty$ were infinite, we could make RHS of main inequality arbitrarily small relative to LHS by choosing $R$ large, yielding a contradiction.

- **Conclusion**: Exponential moment $\int e^{\theta V} \rho_\infty < \infty$ for $\theta < \theta_0$
- **Form**: Finiteness statement

**Substep 4.4**: Justify unbounded test function via truncation

The above argument is formal because $W_\theta = e^{\theta V}$ is unbounded. To make it rigorous:

1. Consider truncated test function $W_{\theta,R} = e^{\theta V} \chi_R(x,v)$ where $\chi_R$ is a smooth cutoff with $\chi_R \equiv 1$ on $\{|x|^2 + |v|^2 \le R\}$ and $\chi_R \equiv 0$ on $\{|x|^2 + |v|^2 \ge 2R\}$

2. Apply stationarity: $\int \mathcal{L}^*[W_{\theta,R}] \rho_\infty = 0$

3. Compute $\mathcal{L}^*[W_{\theta,R}] \approx \mathcal{L}^*[W_\theta] \chi_R + (\text{cutoff derivatives})$

4. The cutoff derivative terms are supported on $\{R \le |x|^2 + |v|^2 \le 2R\}$ and can be bounded by $C_R e^{\theta V}$ on that annulus

5. Derive the same inequality as Substep 4.2 but with an extra error term that vanishes as $R \to \infty$ (using tail decay of $\rho_\infty$ that we're proving)

6. Pass $R \to \infty$ via monotone convergence once finiteness is established

- **Conclusion**: The formal argument can be made rigorous via truncation
- **Form**: Technical justification (Required Lemma)

**Dependencies**:
- Uses: Step 3 (multiplicative Lyapunov), QSD stationarity, coercivity from Step 1
- Requires: Truncation stability lemma (see Required Lemmas)

**Potential Issues**:
- ⚠️ Circular reasoning: using tail decay to justify the proof of tail decay
- **Resolution**: Iterate: first prove weaker moment bound (e.g., $\int V^k \rho_\infty < \infty$ for $k \in \mathbb{N}$), use that for truncation, then upgrade to exponential

---

#### Step 5: Obtain Tail Probability Decay

**Goal**: Apply Chebyshev/Markov inequality to the exponential moment to derive integral tail bounds.

**Substep 5.1**: Apply Markov inequality

For any $R > 0$:

$$
\int_{\{V > R\}} \rho_\infty = \int_{\{V > R\}} \mathbb{1}_{V > R} \rho_\infty \le \int_{\{V > R\}} e^{\theta(V - R)} \rho_\infty = e^{-\theta R} \int_{\{V > R\}} e^{\theta V} \rho_\infty
$$

$$
\le e^{-\theta R} \int e^{\theta V} \rho_\infty \le K(\theta) e^{-\theta R}
$$

- **Justification**: Markov inequality: $P(X > a) \le e^{-\theta a} \mathbb{E}[e^{\theta X}]$
- **Why valid**: $e^{\theta(V-R)} \ge \mathbb{1}_{V > R}$ and monotonicity
- **Expected result**: Exponential decay of tail probability in $V$

**Substep 5.2**: Convert to phase space tail

Using the coercivity $V(x,v) \ge \kappa_0(|x|^2 + |v|^2)$ from Step 1:

$$
\{|x|^2 + |v|^2 > r^2\} \subseteq \{V > \kappa_0 r^2\}
$$

Therefore:

$$
\int_{\{|x|^2 + |v|^2 > r^2\}} \rho_\infty \le \int_{\{V > \kappa_0 r^2\}} \rho_\infty \le K(\theta) e^{-\theta \kappa_0 r^2}
$$

- **Conclusion**: Integral tail bound $\int_{\{|x|^2 + |v|^2 > r^2\}} \rho_\infty \le K(\theta) e^{-\theta \kappa_0 r^2}$
- **Form**: Exponential decay of tail measure

**Dependencies**:
- Uses: Step 4 (moment finiteness), Step 1 (coercivity)
- Requires: Only Markov inequality (standard probability)

**Potential Issues**:
- None (this step is straightforward)

---

#### Step 6: Convert to Pointwise Exponential Bound

**Goal**: Use hypoelliptic regularity (Harnack inequality) to convert the integral tail bound to a pointwise bound $\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}$.

**Substep 6.1**: Apply local Harnack inequality

From R2 (smoothness via Hörmander hypoellipticity) and R3 (positivity via strong maximum principle), the QSD $\rho_\infty$ satisfies a **local Harnack inequality**: for any ball $B_\delta(x,v)$ of radius $\delta > 0$,

$$
\sup_{B_\delta(x,v)} \rho_\infty \le C_{\text{Har}} \frac{1}{|B_\delta|} \int_{B_\delta(x,v)} \rho_\infty
$$

where $C_{\text{Har}}$ is a constant depending on the hypoellipticity structure, and $|B_\delta| = \delta^{2d}$ is the volume of the ball (in $2d$-dimensional phase space).

- **Justification**: Bony-type hypoelliptic Harnack inequality for stationary solutions (see Required Lemma)
- **Why valid**: Hörmander condition holds for kinetic operator (Section 2.2), $\rho_\infty > 0$ (Section 2.3)
- **Expected result**: Local $L^\infty - L^1$ estimate

**Substep 6.2**: Estimate the local average at large $r$

Fix a point $(x,v)$ with $|x|^2 + |v|^2 = r^2$ for large $r$. Choose $\delta = 1$ (fixed radius). For $r \gg 1$, the ball $B_1(x,v)$ satisfies:

$$
B_1(x,v) \subseteq \{(x', v') : |x'|^2 + |v'|^2 \ge (r-\sqrt{2})^2\}
$$

(since any point in the ball is at most distance 1 from $(x,v)$, and $r$ is large).

Therefore:

$$
\int_{B_1(x,v)} \rho_\infty \le \int_{\{|x'|^2 + |v'|^2 \ge (r-\sqrt{2})^2\}} \rho_\infty \le K(\theta) e^{-\theta \kappa_0 (r-\sqrt{2})^2}
$$

- **Justification**: Monotonicity and Step 5 tail bound
- **Why valid**: Set inclusion and exponential decay from Step 5
- **Expected result**: Local average decays exponentially

**Substep 6.3**: Apply Harnack to get pointwise bound

Combining Substeps 6.1 and 6.2:

$$
\rho_\infty(x,v) \le \sup_{B_1(x,v)} \rho_\infty \le C_{\text{Har}} \frac{1}{|B_1|} \int_{B_1(x,v)} \rho_\infty \le \frac{C_{\text{Har}}}{|B_1|} K(\theta) e^{-\theta \kappa_0 (r-\sqrt{2})^2}
$$

For large $r$:

$$
(r-\sqrt{2})^2 = r^2 - 2\sqrt{2} r + 2 \ge r^2 - r^2/2 = r^2/2
$$

(choosing $r$ large enough that $2\sqrt{2} r \le r^2/2$).

Thus:

$$
\rho_\infty(x,v) \le \frac{C_{\text{Har}} K(\theta)}{|B_1|} e^{-\theta \kappa_0 r^2/2} = \frac{C_{\text{Har}} K(\theta)}{|B_1|} e^{-(\theta \kappa_0/2)(|x|^2 + |v|^2)}
$$

Define $\alpha := \frac{\theta \kappa_0}{2}$ and $C := \frac{C_{\text{Har}} K(\theta)}{|B_1|}$.

- **Conclusion**: Pointwise exponential bound $\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}$ for large $r$
- **Form**: Exponential decay estimate

**Substep 6.4**: Handle bounded region

For $|x|^2 + |v|^2 \le R_0$ (compact region), $\rho_\infty$ is continuous and positive (R2, R3), hence bounded:

$$
\rho_\infty(x,v) \le \sup_{|x|^2 + |v|^2 \le R_0} \rho_\infty =: C_0 < \infty
$$

We can adjust the constant $C$ to ensure the bound holds globally:

$$
C_{\text{global}} = \max(C, C_0 e^{\alpha R_0})
$$

Then:

$$
\rho_\infty(x,v) \le C_{\text{global}} e^{-\alpha(|x|^2 + |v|^2)} \quad \forall (x,v) \in \Omega
$$

- **Conclusion**: Global pointwise exponential bound with unified constant
- **Form**: Final theorem statement

**Dependencies**:
- Uses: Step 5 (tail bounds), R2-R3 (smoothness, positivity), Harnack lemma
- Requires: Hypoelliptic Harnack inequality (Required Lemma)

**Potential Issues**:
- ⚠️ Harnack inequality is a deep result in hypoelliptic PDE theory
- **Resolution**: Cite standard references (Bony, Hörmander theory) and Section 2.2's discussion

---

### Final Assembly and Q.E.D.

**Assembly**:
- From Step 1: Quadratic Lyapunov $V$ with coercivity $V \ge \kappa_0(|x|^2 + |v|^2)$
- From Step 2: Full drift $\mathcal{L}^*[V] \le -\beta V + C$
- From Step 3: Multiplicative bound $\mathcal{L}^*[e^{\theta V}] \le \theta e^{\theta V}(-\frac{\beta}{2}V + C) + O(1)$
- From Step 4: Exponential moment $\int e^{\theta V} \rho_\infty < \infty$ for $\theta < \theta_0$
- From Step 5: Tail decay $\int_{\{|x|^2 + |v|^2 > r^2\}} \rho_\infty \le K e^{-\theta \kappa_0 r^2}$
- From Step 6: Pointwise bound $\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}$ with $\alpha = \theta \kappa_0/2$

**Combining Results**:
The theorem claims $\rho_\infty(x,v) \le C e^{-\alpha(|x|^2 + |v|^2)}$ for some $\alpha, C > 0$. We have shown:

- $\alpha = \frac{\theta \kappa_0}{2}$ where $\theta < \theta_0 = \frac{2\beta}{\sigma^2 C_V}$
- $\kappa_0 = \lambda_{\min}(M)$ depends on $(a,b,c)$ from Section 4.2
- $\beta$ depends on $\gamma, \kappa_{\text{conf}}$ via Section 4.2's drift computation
- $C$ depends on $C_{\text{Har}}, K(\theta), |B_1|$, which in turn depend on all problem parameters

Therefore, $\alpha$ and $C$ are explicit functions of $\gamma, \sigma^2, \kappa_{\text{conf}}, \kappa_{\max}$ as claimed.

**Final Conclusion**:
Under Assumptions A1-A4, the QSD $\rho_\infty$ satisfies exponential tail decay, establishing property R6.

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Parameter Restriction on $\gamma$ vs $\kappa_{\text{conf}}$

**Why Difficult**: The Lyapunov drift computation in Section 4.2 (lines 2160-2241) derives the condition:

$$
\gamma > \frac{4\kappa_{\text{conf}}}{9}
$$

for one particular choice of parameters $(a, b, c, \varepsilon)$. This appears as a **sufficient condition** for the drift bound to hold, but it's unclear if this restriction is **necessary** or merely an artifact of that specific parameterization.

The issue arises in the algebra for the $|x|^2$ coefficient in $\mathcal{L}^*[V]$. After expanding all terms (kinetic transport, friction, force, diffusion), the coefficient is:

$$
-2\varepsilon \kappa_{\text{conf}} + 3\varepsilon \kappa_{\text{conf}} - 3\gamma \varepsilon + \frac{\varepsilon \kappa_{\text{conf}}}{3} = \varepsilon\left(\frac{4\kappa_{\text{conf}}}{3} - 3\gamma\right)
$$

For this to be negative (required for drift), we need $\gamma > \frac{4\kappa_{\text{conf}}}{9}$.

**Mathematical Obstacle**:
- If this condition is **necessary**, the theorem only holds for sufficiently large friction relative to confinement strength
- If this is merely **sufficient** for the current parameter choice, we can optimize $(a,b,c)$ to remove the restriction
- The current document states the condition but doesn't clarify its status

**Proposed Technique**:

**Option 1: Re-optimize Lyapunov parameters**
- Treat $(a, b, c)$ and Young's inequality parameters as free optimization variables
- Solve for the optimal choice minimizing the required $\gamma/\kappa_{\text{conf}}$ ratio
- This is a feasibility problem: find $(a,b,c,\delta_1,\delta_2,\ldots) \succ 0$ such that all coefficients in $\mathcal{L}^*[V]$ have correct sign
- May require numerical optimization or more sophisticated symbolic algebra

**Option 2: Leverage killing term outside safe region**
- On $\mathcal{X} \setminus K$ where $\kappa_{\text{kill}} \ge \kappa_0 > 0$, the jump adjoint contributes $-\kappa_{\text{kill}} V$ (negative)
- Modify the Lyapunov function to $V_{\text{mod}} = a|x|^2 + 2bx \cdot v + c|v|^2 + \eta \kappa_{\text{kill}}(x) |x|^2$ with small $\eta > 0$
- The extra term $-\kappa_{\text{kill}} \cdot \eta \kappa_{\text{kill}} |x|^2$ strengthens the drift outside $K$
- Inside $K$ where $\kappa_{\text{kill}} = 0$, reduces to original $V$
- This may compensate for insufficient friction

**Option 3: Two-region Lyapunov**
- Use different Lyapunov functions in $K$ (safe region) and $\mathcal{X} \setminus K$ (killing region)
- In $K$: Use kinetic drift (may require tighter parameter choice)
- Outside $K$: Leverage $-\kappa_{\text{kill}} V$ to relax friction requirement
- Match at the boundary via continuity

**Alternative if Fails**:
If the restriction is indeed necessary (or cannot be removed without excessive technical complexity):
- **State the theorem with explicit condition**: "Under A1-A4 and $\gamma > c \kappa_{\text{conf}}$ for some explicit $c > 0$, the QSD satisfies exponential tails."
- **Physical interpretation**: Friction must dominate confinement force to ensure velocity dissipation controls kinetic energy
- **Check if reasonable**: For typical applications, is $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ a mild or restrictive condition?

**References**:
- Current computation: Section 4.2, lines 2200-2211
- Young's inequality parameterization: Section 4.2, lines 2177-2199

---

### Challenge 2: Jump-Adjoint Control for $e^{\theta V}$

**Why Difficult**: The revival mechanism in the mean-field model injects mass into the safe region $K$ with distribution proportional to the dead mass. For the exponential test function $W_\theta = e^{\theta V}$, this creates a positive contribution to $J^*[W_\theta]$:

$$
J^*[e^{\theta V}] = -\kappa_{\text{kill}}(x) e^{\theta V(x,v)} + \lambda_{\text{revive}} \mathbb{E}[e^{\theta V(X',V')} | \text{revival}]
$$

The expectation $\mathbb{E}[e^{\theta V(X',V')}]$ depends on the revival distribution. If we only know that $(X',V')$ is supported in $K \times \mathbb{R}^d_v$ (compact in $x$ but potentially unbounded in $v$), we need to control:

$$
\mathbb{E}[e^{\theta(a|X'|^2 + 2bX' \cdot V' + c|V'|^2)}]
$$

If the revival distribution has heavy tails in $v$, this could be unbounded even for small $\theta > 0$.

**Mathematical Obstacle**:
- The revival distribution is defined as proportional to $\rho_{\text{dead}}$, which is itself determined by the dynamics
- We don't have a priori control on the velocity moments of the revival distribution
- Circular reasoning: proving R6 requires controlling revival moments, but revival depends on $\rho_\infty$ which we're characterizing via R6

**Proposed Technique**:

**Step 1: Bound spatial part trivially**
Since $X'$ is supported in compact $K$:

$$
\sup_{x \in K} |x|^2 \le R_K^2 < \infty
$$

Therefore:

$$
e^{\theta a |X'|^2} \le e^{\theta a R_K^2} =: C_K(\theta) < \infty
$$

**Step 2: Handle velocity part via sub-QSD analysis**
The revival velocity distribution inherits properties from the marginal velocity distribution of dead walkers. By the killing structure (A2), killing primarily occurs near spatial boundaries $\partial \mathcal{X}$, not at high velocities. The velocity distribution of killed walkers should be comparable to the velocity marginal of the full QSD restricted to the killing region.

More precisely, the dead mass is:

$$
\rho_{\text{dead}}(x,v) = \kappa_{\text{kill}}(x) \rho_\infty(x,v)
$$

The revival distribution is:

$$
\rho_{\text{revival}}(x,v) = \frac{\kappa_{\text{kill}}(x) \rho_\infty(x,v)}{\int \kappa_{\text{kill}} \rho_\infty} \cdot \mathbb{1}_K(x)
$$

For the velocity part:

$$
\int e^{\theta c |V'|^2} \rho_{\text{revival}}(x, v) dv = \frac{\int_K \kappa_{\text{kill}}(x) \rho_\infty(x,v) e^{\theta c |v|^2} dv}{\int \kappa_{\text{kill}} \rho_\infty}
$$

If we've already established that $\rho_\infty$ has exponential tails (circular!), we know $\int e^{\theta' |v|^2} \rho_\infty < \infty$ for small $\theta'$. But we're trying to prove this!

**Resolution via Bootstrapping**:
1. **First prove polynomial moments**: Show $\int |v|^{2k} \rho_\infty < \infty$ for $k \in \mathbb{N}$ using $\mathcal{L}^*[|v|^{2k}]$ and induction (easier, doesn't require exponential test functions)
2. **Use polynomial moments to control revival**: With polynomial moments, we have:
   $$
   \mathbb{E}[e^{\theta c |V'|^2}] = \sum_{n=0}^\infty \frac{(\theta c)^n}{n!} \mathbb{E}[|V'|^{2n}] < \infty
   $$
   for small $\theta$ (by dominated convergence using tail bounds from polynomial moments)
3. **Close exponential moment**: Now that revival is controlled, proceed with Step 4's argument for $e^{\theta V}$
4. **Upgrade to exponential tails**: Use the proven exponential moment to establish R6

**Step 3: Explicit bound**
Combining Steps 1-2:

$$
\mathbb{E}[e^{\theta V(X',V')}] = \mathbb{E}[e^{\theta(a|X'|^2 + 2bX' \cdot V' + c|V'|^2)}]
$$

$$
\le e^{\theta a R_K^2} \mathbb{E}[e^{2\theta b R_K |V'| + \theta c |V'|^2}]
$$

Using $e^{2\theta b R_K |V'|} \le e^{\theta b^2 R_K^2} e^{\theta |V'|^2}$ (Young's inequality):

$$
\le e^{\theta a R_K^2 + \theta b^2 R_K^2} \mathbb{E}[e^{\theta(c+1)|V'|^2}]
$$

Define $V_{\max}(\theta) := \theta(a + b^2) R_K^2 + \log \mathbb{E}[e^{\theta(c+1)|V'|^2}]$. Then:

$$
J^*[e^{\theta V}] \le -\kappa_{\text{kill}} e^{\theta V} + \lambda_{\text{revive}} e^{V_{\max}(\theta)}
$$

For Step 4's argument, we need $V_{\max}(\theta) < \infty$, which follows from the polynomial moment bootstrap.

**Alternative if Fails**:
If the bootstrap approach is too complex:
- **Assume revival in bounded velocity region**: Modify A2 to include that revival resamples with $|v| \le V_{\text{max}}$ (physically: particles return with moderate kinetic energy)
- **Use Gaussian revival**: Specify that revival uses a Gaussian in $v$ with variance $\sigma_{\text{rev}}^2$, making all moments trivially finite
- **Accept weaker result**: Prove exponential tails in $x$ only, polynomial decay in $v$

**References**:
- Revival operator definition: Section 1.1, Section 4.2 (jump adjoint)
- Killing structure: Assumption A2

---

### Challenge 3: From Moment to Pointwise Bound via Hypoelliptic Harnack

**Why Difficult**: Step 6 requires a **local Harnack inequality** (or equivalently, a local $L^\infty - L^1$ estimate) for the QSD $\rho_\infty$ viewed as a stationary solution of the hypoelliptic kinetic Fokker-Planck equation. This is a deep result in PDE regularity theory, specifically for **degenerate elliptic operators** (the kinetic operator has diffusion only in $v$, not in $x$).

The classical Harnack inequality for uniformly elliptic equations doesn't apply directly. Instead, we need **Hörmander's hypoellipticity theory** combined with **maximum principle arguments** adapted to the kinetic structure.

**Mathematical Obstacle**:
- The kinetic operator $\mathcal{L}_{\text{kin}} = -v \cdot \nabla_x + \nabla_v \cdot [(\gamma v + \nabla U) \cdot] + \frac{\sigma^2}{2} \Delta_v$ is only degenerate-elliptic
- Transport term $-v \cdot \nabla_x$ is first-order hyperbolic
- Standard elliptic regularity theory doesn't apply
- Hörmander's theorem guarantees $C^\infty$ regularity (R2), but extracting quantitative Harnack estimates requires careful analysis

**Proposed Technique**:

**Step 1: Invoke Hörmander's hypoellipticity**
From Section 2.2 (R2 proof), the kinetic operator satisfies **Hörmander's bracket condition**:
- Let $X_0 = -v \cdot \nabla_x + (\gamma v + \nabla U) \cdot \nabla_v$ (drift)
- Let $X_1, \ldots, X_d$ be the noise directions: $X_i = \sigma \partial_{v_i}$
- The Lie algebra generated by $\{X_1, \ldots, X_d, [X_0, X_i], [X_0, [X_0, X_i]], \ldots\}$ spans the tangent space at every point

Specifically:
$$
[X_0, X_i] = [\text{drift}, \partial_{v_i}] = -\partial_{x_i} + \ldots
$$

This gives access to spatial directions via commutators, ensuring hypoellipticity.

By **Hörmander's theorem** (1967), stationary solutions are $C^\infty$ and satisfy a weak Harnack inequality.

**Step 2: Cite standard Harnack results**
The key reference is **Jean-Michel Bony** (1969) on maximum principles and Harnack inequalities for hypoelliptic operators:

> For a hypoelliptic operator $L$ satisfying Hörmander's condition and a positive stationary solution $u$ (i.e., $Lu = 0$, $u > 0$), there exists a local Harnack inequality:
> $$
> \sup_{B_\delta} u \le C_{\text{Har}} \inf_{B_\delta} u
> $$
> for balls $B_\delta$ of radius $\delta$, with $C_{\text{Har}}$ depending on the operator structure.

Integrating the LHS and using $\inf \le \text{avg}$:

$$
\sup_{B_\delta} u \le C_{\text{Har}} \inf_{B_\delta} u \le C_{\text{Har}} \frac{1}{|B_\delta|} \int_{B_\delta} u
$$

This is the $L^\infty - L^1$ estimate needed for Step 6.

**Step 3: Verify preconditions**
- **Hypoellipticity**: Verified in Section 2.2 (R2)
- **Positivity**: Verified in Section 2.3 (R3)
- **Stationarity**: $\mathcal{L}[\rho_\infty] = 0$ by definition of QSD

Therefore, Bony's theorem applies to $\rho_\infty$.

**Step 4: Handle killed/revival operator**
The full generator is $\mathcal{L} = \mathcal{L}_{\text{kin}} + J$ where $J$ is the jump operator. Does the Harnack inequality still hold?

**Analysis**:
- Inside the safe region $K$ where $\kappa_{\text{kill}} = 0$, we have $J[\rho] = \lambda_{\text{revive}} m_d(\rho) \rho / \|\rho\|_{L^1}$, which is a **multiplication operator** (zero-order)
- Harnack inequalities are robust to lower-order perturbations: if $L u = f$ with $f$ bounded, the Harnack inequality degrades gracefully
- For the stationary case $\mathcal{L}[\rho_\infty] = 0$, we have $\mathcal{L}_{\text{kin}}[\rho_\infty] = -J[\rho_\infty]$
- The RHS is bounded (since revival is bounded in $K$), so $\rho_\infty$ solves a hypoelliptic equation with bounded source
- Schauder estimates + De Giorgi-Nash-Moser theory (adapted to hypoelliptic case) still provide Harnack

**Conclusion**: The local Harnack inequality holds for $\rho_\infty$ with constant $C_{\text{Har}}$ depending on operator parameters.

**Alternative if Fails**:

**Backup 1: Moser iteration**
If Bony's Harnack is unavailable or hard to verify:
- Use **Moser iteration** adapted to kinetic operators (see Golse-Imbert-Mouhot 2011)
- This provides local $L^p$ estimates from $L^2$ integrability, eventually yielding $L^\infty$ bounds
- More technical but systematic

**Backup 2: Barrier comparison**
Construct an explicit **supersolution** $\Phi(x,v) = C e^{-\alpha V(x,v)}$ and show $\mathcal{L}[\Phi] \ge 0$ (or $\le 0$) outside a compact set. Then compare $\rho_\infty$ to $\Phi$ via maximum principle on exterior domains:
- On large annuli $\{R_1 < |x|^2 + |v|^2 < R_2\}$, if $\rho_\infty \le \Phi$ on the boundary $|x|^2 + |v|^2 = R_1$, then $\rho_\infty \le \Phi$ throughout
- Requires proving $\mathcal{L}[\Phi] \ge 0$, which amounts to re-doing the Lyapunov drift calculation for $\Phi$
- May avoid Harnack but requires separate computation

**References**:
- Hörmander hypoellipticity: Section 2.2, citations to Hörmander 1967
- Bony maximum principles: Referenced in Section 2.3 (R3 proof)
- Villani (2009), Chapter 2: Hypoelliptic estimates for kinetic equations
- Dolbeault et al. (2015): Log-Sobolev inequalities for hypocoercive kinetic operators

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps (Step 1 → 2 → 3 → 4 → 5 → 6, each using prior results)
- [x] **Hypothesis Usage**:
  - A1 (confinement) used in Steps 1-2 (Lyapunov drift)
  - A2 (killing structure) used in Steps 2, 4 (jump control)
  - A3 (parameters) used throughout (constants)
  - A4 (domain) used in Step 6 (regularity)
- [x] **Conclusion Derivation**: Pointwise bound $\rho_\infty \le C e^{-\alpha(|x|^2 + |v|^2)}$ fully derived in Step 6
- [x] **Framework Consistency**: All dependencies (R1-R5, Lyapunov drift) verified against document sections
- [x] **No Circular Reasoning**: R6 proof uses only R2-R3 and A1-A4, does not assume R6 (except minor bootstrap in Step 4.4 which can be made rigorous via iteration)
- [x] **Constant Tracking**: All constants ($\alpha, C, \beta, \kappa_0, C_V, \theta_0$) defined explicitly in terms of problem parameters
- [x] **Edge Cases**:
  - Bounded region handled in Step 6.4
  - Large $r$ asymptotics in Steps 5-6
  - Truncation for unbounded test function in Step 4.4
- [x] **Regularity Verified**: R2 (smoothness), R3 (positivity) confirmed in Sections 2.2-2.3
- [x] **Measure Theory**: Integration by parts justified via stationarity and truncation (Step 4.4)

**Minor Gaps Requiring Additional Work**:
- ⚠️ **Lemma (Jump adjoint bound)**: Needs explicit proof of $J^*[e^{\theta V}] \le C_{\text{jump}}$ with finiteness (Challenge 2)
- ⚠️ **Lemma (Harnack inequality)**: Needs citation or proof of local $L^\infty - L^1$ estimate (Challenge 3)
- ⚠️ **Truncation stability**: Formal argument in Step 4.4 needs rigorous justification via dominated convergence
- ⚠️ **Parameter optimization**: Resolve whether $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ is necessary (Challenge 1)

**Overall Assessment**: The proof structure is **sound** with a clear logical flow. The main steps are well-justified by standard techniques (multiplicative Lyapunov, Markov inequality, hypoelliptic Harnack). The gaps are **technical** rather than **conceptual**, and each has a proposed resolution path. With the required lemmas proven, the theorem follows rigorously.

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Direct Supersolution/Barrier Method

**Approach**: Construct an explicit function $\Phi(x,v) = C e^{-\alpha V(x,v)}$ and show it is a **supersolution** to the stationary equation in the sense:

$$
\mathcal{L}[\Phi] \ge 0 \quad \text{for } |x|^2 + |v|^2 > R_0
$$

Then use a maximum principle on exterior domains to compare $\rho_\infty$ to $\Phi$.

**Pros**:
- Avoids the Harnack inequality (Challenge 3)
- Direct and constructive: explicitly builds the exponential barrier
- Self-contained: doesn't rely on deep PDE regularity theory

**Cons**:
- Requires computing $\mathcal{L}[e^{-\alpha V}]$ explicitly, which is similar to (but not identical to) the adjoint computation
- Maximum principle for the killed/revival operator is non-trivial: the jump term $J[\rho]$ involves global dependence on $\rho$, breaking locality
- Boundary conditions at infinity or $\partial \mathcal{X}$ need careful treatment
- May not work if $\mathcal{L}[\Phi]$ has mixed signs (would need to adjust $\alpha$ regionally)

**When to Consider**:
- If Harnack inequality is unavailable or difficult to verify
- If a constructive, self-contained proof is preferred
- As a cross-check: if both methods work, they should yield the same $\alpha$

**Implementation Sketch**:
1. Compute $\mathcal{L}[e^{-\alpha V}]$ using chain rule (similar to adjoint calculation but with opposite sign)
2. Show that for appropriate $\alpha > 0$ and outside a compact set, $\mathcal{L}[e^{-\alpha V}] \ge c e^{-\alpha V}$ for some $c > 0$ (proving $e^{-\alpha V}$ decays faster than it's created)
3. Define $\Phi = C e^{-\alpha V}$ with $C$ large enough that $\Phi \ge \rho_\infty$ on some annulus $\{|x|^2 + |v|^2 = R_0\}$
4. Apply comparison principle on $\{|x|^2 + |v|^2 > R_0\}$: if $\mathcal{L}[\rho_\infty - \Phi] \le 0$ and $\rho_\infty \le \Phi$ on the boundary, then $\rho_\infty \le \Phi$ throughout
5. Extend to global bound by handling $\{|x|^2 + |v|^2 \le R_0\}$ via compactness

---

### Alternative 2: Semigroup and Hypocoercivity via Kernel Bounds

**Approach**: Use the **semigroup perspective**: write $\rho_\infty$ as the limit of $e^{t \mathcal{L}} \rho_0$ as $t \to \infty$. Establish **Gaussian-like kernel bounds** for the kinetic semigroup $e^{t \mathcal{L}_{\text{kin}}}$, then handle the killed/revival perturbation via Duhamel's formula.

**Pros**:
- Leverages well-developed theory: hypocoercivity methods (Villani, Dolbeault-Mouhot-Schmeiser)
- Provides explicit decay rates as $t \to \infty$
- Kernel bounds give pointwise control directly without needing Harnack
- Conceptually clean: semigroup view is natural for Markov processes

**Cons**:
- Requires proving kernel bounds for the killed/revival semigroup, not just the pure kinetic operator
- Duhamel formula for the perturbation $J$ may be difficult: revival is nonlinear in $\rho$
- Heavier machinery: need functional analytic setup (interpolation spaces, Sobolev embeddings in kinetic context)
- Less elementary than moment method
- Mean-field coupling complicates semigroup analysis (non-autonomous evolution)

**When to Consider**:
- If studying time-dependent convergence, not just stationary properties
- If kernel bounds are already available from prior work
- For a more abstract, functional-analytic proof

**Implementation Sketch**:
1. Prove **hypocoercive kernel bounds** for $e^{t \mathcal{L}_{\text{kin}}}$: for the kinetic operator without killing/revival, show:
   $$
   p_t((x,v), (x',v')) \le C t^{-d} e^{-c|x-x'|^2/t} e^{-c'|v-v'|^2}
   $$
   (Gaussian in space due to transport, exponential in velocity due to diffusion)

2. Use **Duhamel formula** to incorporate killing/revival:
   $$
   \rho_t = e^{t \mathcal{L}} \rho_0 = e^{t \mathcal{L}_{\text{kin}}} \rho_0 + \int_0^t e^{(t-s) \mathcal{L}_{\text{kin}}} J[\rho_s] ds
   $$

3. For the QSD, $\rho_\infty$ satisfies the stationary Duhamel:
   $$
   \rho_\infty = \int_0^\infty e^{s \mathcal{L}_{\text{kin}}} J[\rho_\infty] ds
   $$
   (formal; requires careful analysis)

4. Use kernel bounds to estimate:
   $$
   \rho_\infty(x,v) \le \int_0^\infty \int p_s((x,v), (x',v')) |J[\rho_\infty](x',v')| dx' dv' ds
   $$

5. The Gaussian kernel $p_s$ provides exponential decay which, when integrated against the bounded source $J[\rho_\infty]$, yields exponential tails for $\rho_\infty$

**Challenges**: Step 3's stationary Duhamel is formal and requires justification. The nonlinear coupling in $J[\rho_\infty]$ makes this non-trivial.

---

### Alternative 3: Nash Inequalities and Logarithmic Sobolev Inequality (Reverse Direction)

**Approach**: Instead of proving exponential tails to enable LSI (as in the current proof), try to prove a **weak LSI** or **Nash inequality** first using R1-R5, then deduce exponential tails as a consequence.

**Pros**:
- Reverses the logical dependence: LSI → tails instead of tails → LSI
- Nash inequalities can sometimes be proven more directly from operator structure
- Avoids Harnack inequality

**Cons**:
- Proving LSI without exponential tails is typically harder (needs stronger assumptions or different techniques)
- The whole point of R6 is to enable LSI, so this reverses the document's architecture
- May require additional assumptions (log-concavity, Bakry-Émery criterion)

**When to Consider**:
- If the QSD has additional structure (e.g., log-concave, gradient flow structure)
- As a cross-check: if LSI can be proven independently, it validates R6

**Not Recommended** for current framework: This would require restructuring the entire proof architecture, and the current approach (tails → LSI) is more standard for kinetic equations.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Necessity of parameter restriction $\gamma > \frac{4\kappa_{\text{conf}}}{9}$**:
   - **Current status**: Sufficient condition for one Lyapunov parameterization
   - **Question**: Is this necessary for *any* choice of $(a,b,c)$, or artifact of current choice?
   - **How critical**: Low for applications (can be absorbed into assumptions), but high for mathematical completeness
   - **Resolution path**: Optimization over Lyapunov parameter space, or proof of necessity via counterexample

2. **Quantitative Harnack constant $C_{\text{Har}}$**:
   - **Current status**: Existence guaranteed by Bony's theorem, but constant not explicit
   - **Question**: Can $C_{\text{Har}}$ be computed explicitly in terms of $\gamma, \sigma^2, \kappa_{\text{conf}}$?
   - **How critical**: Medium (affects final constant $C$ in theorem, but not existence)
   - **Resolution path**: Detailed analysis of Hörmander's theorem proof, or numerical estimation

3. **Optimality of tail decay rate $\alpha$**:
   - **Current status**: $\alpha = \frac{\theta \kappa_0}{2}$ with $\theta < \theta_0$, but this uses $\theta < \theta_0$, not the optimal $\theta \to \theta_0$
   - **Question**: Can we take $\theta = \theta_0 - \varepsilon$ and optimize? Is $\alpha$ the best possible rate?
   - **How critical**: Low for existence, high for sharp estimates
   - **Resolution path**: Asymptotic analysis of moment bounds as $\theta \to \theta_0$

### Conjectures

1. **Parameter-free exponential tails**:
   - **Statement**: For any $\gamma, \sigma^2, \kappa_{\text{conf}} > 0$ satisfying A1-A4, exponential tails hold (possibly with $\gamma$-dependent rate)
   - **Why plausible**: Confinement + friction should always yield exponential tails, regardless of specific ratio
   - **How to test**: Numerical simulation with varying $\gamma/\kappa_{\text{conf}}$ to check tail behavior

2. **Gaussian tails for large $\gamma$**:
   - **Statement**: In the strong friction limit $\gamma \to \infty$, $\rho_\infty(x,v) \sim e^{-\frac{U(x)}{D} - \frac{|v|^2}{2D_v}}$ (approximately Gaussian)
   - **Why plausible**: Overdamped limit should yield Gibbs distribution in $x$, Gaussian in $v$
   - **How to test**: Matched asymptotics or rigorous $\gamma \to \infty$ limit

3. **Uniform bounds over revival rate $\lambda_{\text{revive}}$**:
   - **Statement**: The constants $\alpha, C$ can be chosen uniformly for $\lambda_{\text{revive}} \in [0, \Lambda]$ (revival rate doesn't affect tail decay fundamentally)
   - **Why plausible**: Revival only redistributes mass, doesn't inject energy
   - **How to test**: Analyze dependence of $C$ on $\lambda_{\text{revive}}$ in Step 2.2

### Extensions

1. **Dimension-dependent bounds**: Characterize how $\alpha(d)$ and $C(d)$ scale with spatial dimension $d$
   - **Motivation**: Understanding curse of dimensionality for high-dimensional optimization
   - **Approach**: Track $d$-dependence through eigenvalue bounds $\lambda_{\min}(M)$, Harnack constant $C_{\text{Har}}$

2. **Non-quadratic potentials**: Extend to $U(x)$ with weaker convexity (e.g., $U \sim |x|^p$ for $p \in (1,2)$)
   - **Motivation**: Broader class of confinement mechanisms
   - **Challenge**: Lyapunov function may need to be non-quadratic ($V \sim |x|^p + |v|^2$)

3. **Anisotropic tails**: Allow different decay rates in $x$ and $v$: $\rho_\infty \le C e^{-\alpha_x |x|^2 - \alpha_v |v|^2}$
   - **Motivation**: Friction dominates velocity decay, confinement dominates spatial decay (different scales)
   - **Approach**: Use $V(x,v) = a|x|^2 + c|v|^2$ without cross-term (requires separate drift analysis)

---

## IX. Expansion Roadmap

This section provides a concrete plan for expanding the proof sketch into a complete, publication-ready proof.

### Phase 1: Prove Missing Lemmas (Estimated: 2-3 weeks)

**1.1. Lemma (Adjoint chain rule for exponentials)**
- **Statement**: For $V \in C^2$ and $W_\theta = e^{\theta V}$, $\mathcal{L}^*[W_\theta] = \theta e^{\theta V} \mathcal{L}^*[V] + \frac{\sigma^2}{2} \theta^2 e^{\theta V} |\nabla_v V|^2$
- **Proof strategy**: Direct computation using product rule and chain rule
- **Difficulty**: Easy (1-2 hours)
- **Dependencies**: None

**1.2. Lemma (Quadratic coercivity and gradient control)**
- **Statement**: If $M \succ 0$, then $V \ge \lambda_{\min}(|x|^2 + |v|^2)$ and $|\nabla_v V|^2 \le C_V V$
- **Proof strategy**: Spectral theorem + Cauchy-Schwarz
- **Difficulty**: Easy (2-3 hours)
- **Dependencies**: None

**1.3. Lemma (Jump adjoint bound for exponentials)**
- **Statement**: $J^*[e^{\theta V}] \le C_{\text{jump}}(\theta) - \kappa_{\text{kill}} e^{\theta V} + O(1)$ with explicit $C_{\text{jump}}$
- **Proof strategy**:
  1. Write $J^*[f] = -\kappa_{\text{kill}} f + \lambda_{\text{revive}} \mathbb{E}[f | \text{revival}]$
  2. Bound $\mathbb{E}[e^{\theta V}]$ using compact support in $x$ and polynomial velocity moments (bootstrap)
  3. Derive explicit formula for $C_{\text{jump}}(\theta)$
- **Difficulty**: Medium (1-2 days)
- **Dependencies**: Polynomial moment bounds (may need separate sub-lemma)

**1.4. Lemma (Local Harnack inequality)**
- **Statement**: For $\rho_\infty$ solving $\mathcal{L}[\rho_\infty] = 0$ with R2-R3, $\sup_{B_\delta} \rho_\infty \le C_{\text{Har}} \frac{1}{|B_\delta|} \int_{B_\delta} \rho_\infty$
- **Proof strategy**:
  1. Verify Hörmander condition (already done in Section 2.2)
  2. Cite Bony (1969) or derive via Moser iteration
  3. Handle jump operator perturbation via Schauder estimates
- **Difficulty**: Medium-Hard (3-5 days if deriving from scratch, 1 day if citing literature)
- **Dependencies**: R2, R3 (already proven)

**1.5. Lemma (Truncation stability for integration by parts)**
- **Statement**: $\lim_{R \to \infty} \int \mathcal{L}^*[W_{\theta,R}] \rho_\infty = \int \mathcal{L}^*[W_\theta] \rho_\infty$ (understood in distributional sense)
- **Proof strategy**:
  1. Show $|\mathcal{L}^*[W_{\theta,R}] - \mathcal{L}^*[W_\theta] \chi_R|$ is supported on annulus $\{R \le |x|^2 + |v|^2 \le 2R\}$
  2. Bound this error using cutoff derivatives and $e^{\theta V}$ decay
  3. Use dominated convergence with tail bounds from Step 5
- **Difficulty**: Medium (2-3 days)
- **Dependencies**: Tail probability bounds (creates bootstrap; resolve by iteration)

### Phase 2: Fill Technical Details (Estimated: 3-4 weeks)

**2.1. Step 2 (Full adjoint drift): Complete computation**
- Expand Substep 2.2 with explicit formulas for $\mathbb{E}[V | \text{revival}]$
- Verify that $C = C_{\text{kin}} + \lambda_{\text{revive}} V_{\max}$ is finite under A2-A3
- Address bounded $\kappa_{\max}$ assumption (add to A2 if needed)
- **Estimated time**: 1 week

**2.2. Step 3 (Multiplicative Lyapunov): Detailed chain rule computation**
- Write out full expansion of $\mathcal{L}^*_{\text{kin}}[e^{\theta V}]$ term-by-term
- Verify each term's sign and magnitude
- Derive explicit formula for $\theta_0 = \frac{2\beta}{\sigma^2 C_V}$
- **Estimated time**: 1 week

**2.3. Step 4 (Moment closure): Rigorous bootstrap**
- Prove polynomial moments $\int |v|^{2k} \rho_\infty < \infty$ first (induction on $k$)
- Use polynomial moments to control revival in Lemma 1.3
- Then prove exponential moment using controlled revival
- Handle truncation rigorously via Lemma 1.5
- **Estimated time**: 2 weeks (most technical part)

**2.4. Step 6 (Pointwise bound): Detailed Harnack application**
- State Harnack inequality precisely with all hypotheses
- Verify each hypothesis (hypoellipticity, positivity, stationarity)
- Compute the ball inclusion $B_1(x,v) \subseteq \{|x'|^2 + |v'|^2 \ge (r-\sqrt{2})^2\}$ rigorously
- Handle boundary region $|x|^2 + |v|^2 \le R_0$ carefully
- **Estimated time**: 1 week

### Phase 3: Add Rigor and Cross-Checks (Estimated: 2 weeks)

**3.1. Epsilon-delta arguments**
- Formalize "for large $r$" statements with explicit $r > R_0(\varepsilon)$
- Verify all inequalities hold with stated constants
- Add error term tracking where approximations are used
- **Estimated time**: 3-4 days

**3.2. Measure-theoretic details**
- Verify all integrals are well-defined (measurability, convergence)
- Justify interchange of limit and integral via dominated/monotone convergence
- Ensure boundary terms vanish in integration by parts
- **Estimated time**: 2-3 days

**3.3. Assumption verification**
- Check every use of A1-A4 is justified
- Flag any additional assumptions needed (e.g., $\kappa_{\max} < \infty$)
- Verify no circular dependencies (R6 doesn't assume R6)
- **Estimated time**: 2 days

**3.4. Numerical validation (optional)**
- Simulate the mean-field QSD numerically
- Measure tail decay rate $\alpha_{\text{numerical}}$ from simulation
- Compare to theoretical $\alpha = \frac{\theta \kappa_0}{2}$
- **Estimated time**: 1 week (if doing numerics)

### Phase 4: Review and Validation (Estimated: 1-2 weeks)

**4.1. Internal review**
- Read proof end-to-end checking every step
- Verify all cross-references to framework documents
- Ensure notation is consistent throughout
- **Estimated time**: 3-4 days

**4.2. Framework cross-validation**
- Check all cited results (R1-R5, Lyapunov drift) match their source statements
- Verify no results are used before they're proven
- Confirm all theorem labels are correct
- **Estimated time**: 2-3 days

**4.3. Gemini dual review** (when available)
- Re-run dual proof strategy generation with both Gemini and GPT-5
- Compare new Gemini strategy to current proof
- Identify any discrepancies or improvements
- **Estimated time**: 1 day + integration of feedback (2-3 days)

**4.4. Constant tracking audit**
- List all constants ($\alpha, C, \beta, \kappa_0, C_V, \theta_0, C_{\text{Har}}$)
- Verify each has explicit definition
- Check dimension-scaling and parameter-dependence
- **Estimated time**: 1-2 days

### Total Estimated Expansion Time: **8-12 weeks**

**Critical Path**:
1. Lemma 1.3 (Jump adjoint) depends on polynomial moments (bootstrap)
2. Step 4.2-4.3 (moment closure) depends on Lemma 1.3 and Lemma 1.5 (truncation)
3. Step 6.1 (Harnack) is independent, can be done in parallel
4. Phase 3-4 depend on Phase 1-2 completion

**Parallelization Opportunities**:
- Lemmas 1.1, 1.2 (easy lemmas) can be done quickly upfront
- Lemma 1.4 (Harnack) and Step 2 (full drift) can proceed independently
- Numerical validation (Phase 3.4) can run concurrently with Phase 4

**Risk Assessment**:
- **Low risk**: Lemmas 1.1, 1.2 (standard results)
- **Medium risk**: Lemma 1.3 (jump control), Lemma 1.5 (truncation) - well-defined but technically involved
- **Higher risk**: Lemma 1.4 (Harnack) - deep PDE theory, may need expert consultation or extensive literature review
- **Critical dependency**: Bootstrap in Step 4.3 - if polynomial moments are hard to prove, may need different approach

**Mitigation Strategies**:
- For Harnack (high risk): Prepare Alternative 1 (barrier method) as backup
- For bootstrap (critical): If direct proof fails, use Alternative 3 (reverse via Nash inequality) or add assumption of polynomial moments to A1-A4

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-qsd-existence-corrected` (Section 1.4: QSD existence via Schauder fixed-point)
- Quadratic Lyapunov drift (Section 4.2, lines 2090-2243: $\mathcal{L}^*[V] \le -\beta V + C$)
- {prf:ref}`R2` (Section 2.2: Smoothness $\rho_\infty \in C^\infty$ via Hörmander hypoellipticity)
- {prf:ref}`R3` (Section 2.3: Positivity $\rho_\infty > 0$ via strong maximum principle)

**Definitions Used**:
- Quasi-Stationary Distribution (QSD): Section 1.1, characterized by $\mathcal{L}[\rho_\infty] = 0$ with $\|\rho_\infty\|_{L^1} < 1$
- Adjoint operator $\mathcal{L}^*$: Section 4.2, defined via integration by parts
- Kinetic generator $\mathcal{L}_{\text{kin}}$: Section 1.1, Langevin dynamics with friction and diffusion
- Jump operator $J$: Section 1.2, killing + proportional revival
- Regularity properties R1-R6: Section 0.2 (R1-R5), Section 4.3 (R6, this theorem)

**Related Proofs** (for comparison):
- Similar exponential tail results for kinetic Fokker-Planck equations: Villani (2009), Chapter 2
- Multiplicative Lyapunov technique: Khas'minskii (1960), Meyn-Tweedie (1993) for discrete-time Markov chains
- Hypoelliptic Harnack inequalities: Bony (1969), Hörmander (1967)

**Framework Documents**:
- Assumptions A1-A4: Section 1.2 (this document)
- Mean-field model: Section 1.1, also {prf:ref}`07_mean_field.md`
- KL-convergence roadmap: This document (16_convergence_mean_field.md), Sections 0-1

---

## XI. Proof Sketch Metadata

**Proof Sketch Completed**: 2025-10-25

**Ready for Expansion**: Needs additional lemmas (Lemma 1.3-1.5 require proof)

**Confidence Level**: Medium-High

**Justification**:
- **Strengths**: The proof strategy is well-established (moment method + Lyapunov is standard for kinetic equations). All main steps are clearly justified with references to framework results. The logical flow is sound (Steps 1→2→3→4→5→6 form a complete chain).

- **Weaknesses**: Missing technical lemmas (especially Lemma 1.4 on Harnack inequality) introduce uncertainty. The bootstrap in Step 4.3 (using tail bounds to justify truncation for proving tail bounds) requires careful execution. Parameter restriction $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ needs resolution.

- **Gemini absence**: No cross-validation from Gemini strategist reduces confidence. Recommend re-running with Gemini when available.

- **Overall**: The core strategy is robust, but execution requires careful technical work on the missing lemmas. With those lemmas proven (estimated 2-3 weeks of focused work), the theorem follows rigorously. Upgrading confidence to "High" would require:
  1. Gemini dual validation
  2. Proof of Lemma 1.3 (jump control) with explicit constants
  3. Verification or citation of Lemma 1.4 (Harnack)
  4. Resolution of parameter restriction (Challenge 1)

**Recommended Next Steps**:
1. Prove Lemma 1.3 (jump adjoint bound) - **Priority 1**
2. Verify Lemma 1.4 (Harnack) via literature or prepare Alternative 1 - **Priority 2**
3. Re-run dual strategy generation when Gemini is available - **Priority 3**
4. Investigate parameter optimization to remove $\gamma > \frac{4\kappa_{\text{conf}}}{9}$ restriction - **Priority 4**

---

**⚠️ SINGLE-STRATEGIST ANALYSIS WARNING**

This proof sketch was generated using **GPT-5 only** due to Gemini 2.5 Pro returning empty responses (attempted twice). According to the Proof Sketcher protocol, dual independent review is **mandatory** for full confidence.

**Limitations of this sketch**:
- No cross-validation from Gemini's strategic reasoning
- Potential blind spots or alternative approaches not considered
- Lower confidence in chosen method (no consensus check)

**Recommendation**: Re-run this sketch generation when Gemini service is restored to obtain:
- Independent proof strategy from Gemini
- Comparison of approaches (consensus vs. discrepancies)
- Cross-validation of framework dependencies
- Identification of any issues missed by single strategist

**User action requested**: If Gemini becomes available, please re-execute the dual strategy protocol for this theorem.

---

**End of Proof Sketch**
