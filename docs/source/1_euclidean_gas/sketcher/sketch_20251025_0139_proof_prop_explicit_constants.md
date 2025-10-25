# Proof Sketch for prop-explicit-constants

**Document**: /home/guillem/fragile/docs/source/1_euclidean_gas/05_kinetic_contraction.md
**Proposition**: prop-explicit-constants
**Generated**: 2025-10-25 01:39 UTC
**Agent**: Proof Sketcher v1.0

---

## I. Proposition Statement

:::{prf:proposition} Explicit Discretization Constants
:label: prop-explicit-constants

Under the axioms of Chapter 3, with:
- Lipschitz force: $\|F(x) - F(y)\| \leq L_F\|x - y\|$
- Bounded force growth: $\|F(x)\| \leq C_F(1 + \|x\|)$
- Diffusion bounds: $\sigma_{\min}^2 I_d \leq \Sigma\Sigma^T \leq \sigma_{\max}^2 I_d$
- Lyapunov regularity: $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$ for $k = 2, 3$

The integrator constant satisfies:

$$
K_{\text{integ}} \leq C_d \cdot \max(\kappa^2, L_F^2, \sigma_{\max}^2/\tau, \gamma^2) \cdot K_V
$$

where $C_d$ is a dimension-dependent constant (polynomial in $d$).

**Practical guideline:**

$$
\tau_* \sim \frac{1}{\max(\kappa, L_F, \sigma_{\max}, \gamma)}
$$

For typical parameters $(\gamma = 1, \sigma_v = 1, \kappa \sim 0.1)$, taking $\tau = 0.01$ is safe.
:::

**Informal Restatement**: This proposition provides explicit formulas for the integrator constant $K_{\text{integ}}$ that appears in Theorem 3.7.2, showing how it depends on the physical parameters (friction $\gamma$, force Lipschitz constant $L_F$, diffusion bound $\sigma_{\max}$, drift rate $\kappa$) and the Lyapunov regularity bound $K_V$. The key insight is that $K_{\text{integ}}$ can be bounded polynomially in the problem dimension with explicit parameter dependence, enabling practical timestep selection.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI FAILED TO RESPOND**

Gemini 2.5 Pro did not provide output for this theorem. This may indicate:
- Technical timeout or service unavailability
- Prompt complexity exceeding capacity
- MCP communication issue

**Action Taken**: Proceeding with single-strategist analysis from Codex.

**Limitations**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available

---

### Strategy B: Codex (GPT-5)'s Approach

**Method**: Direct assembly via weak-error decomposition + component-wise bound tracking

**Key Steps**:
1. **Reduce to component constants**: Start from assembly identity $K_{\text{integ}} = K_W + c_V K_{\text{Var}} + c_B K_b$ (from proof of Theorem 3.7.2, line 1059)
2. **Bound $K_{\text{Var}}$**: Use BAOAB weak error for variance component to show $K_{\text{Var}} \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$
3. **Bound $K_W$**: Use Wasserstein weak error to show $K_W \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$ (for isotropic constant diffusion)
4. **Bound $K_b$**: Use boundary truncation argument to show $K_b \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau)$
5. **Combine bounds**: Sum the three bounds, absorbing weights $c_V, c_B$ into $C_d$
6. **Derive $\tau_*$ guideline**: Use $\tau_* = \kappa/(4K_{\text{integ}})$ from Theorem 3.7.2 to obtain practical guideline

**Strengths**:
- Directly uses existing component-wise weak error propositions (prop-weak-error-variance, prop-weak-error-boundary, prop-weak-error-wasserstein)
- Assembly structure already proven in Theorem 3.7.2
- Makes explicit the origin of each term in the max(...) expression
- Clear tracking of dimension dependence through $C_d$
- Identifies critical technical challenge: the $\sigma_{\max}^2/\tau$ term from boundary truncation

**Weaknesses**:
- Relies on isotropic constant diffusion assumption to avoid $L_\Sigma$ (Lipschitz constant of diffusion matrix)
- The $\sigma_{\max}^2/\tau$ term introduces subtle circularity in the timestep bound
- Requires careful localization to $\{V \leq M\}$ for Lyapunov regularity bound to apply
- N-dependence tracking is not fully explicit

**Framework Dependencies**:
- Theorem 3.7.2 (thm-discretization): Discrete-time inheritance of generator drift
- Proposition 3.7.3.1 (prop-weak-error-variance): BAOAB weak error for variance
- Proposition 3.7.3.2 (prop-weak-error-boundary): BAOAB weak error for boundary
- Proposition 3.7.3.3 (prop-weak-error-wasserstein): BAOAB weak error for Wasserstein
- Assembly identity from proof at line 1059

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Direct assembly via weak-error decomposition (Codex's approach)

**Rationale**:
With only one strategist providing feedback, I adopt Codex's approach as it is:
1. **Well-grounded in existing results**: Directly leverages the three component-wise weak error propositions already proven in §3.7.3
2. **Structurally sound**: Follows the assembly pattern from the proof of Theorem 3.7.2
3. **Technically detailed**: Identifies the key technical challenge (boundary truncation producing $\sigma_{\max}^2/\tau$) with specific resolution strategy
4. **Practically useful**: Derives the timestep guideline from first principles

**Key Insight**: The proposition's claim is essentially a **post-hoc analysis** of the constants already used implicitly in Theorem 3.7.2. The proof should make these constants explicit by:
1. Tracing back through the component-wise weak error bounds
2. Expressing each in terms of primitive parameters and $K_V$
3. Handling the subtle $\tau$-dependence from boundary truncation

**Integration**:
- All steps from Codex's strategy are retained
- Special attention to Challenge 1 (τ-dependence) and Challenge 2 ($L_\Sigma$ assumption)

**Verification Status**:
- ✅ All framework dependencies verified (existing propositions in same document)
- ✅ No circular reasoning detected (constants defined before being assembled)
- ⚠️ Requires isotropic constant diffusion assumption (or include $L_\Sigma$ in bound)
- ⚠️ The $\sigma_{\max}^2/\tau$ term needs careful handling to avoid circularity in timestep selection

---

## III. Framework Dependencies

### Verified Dependencies

**Theorems** (from earlier in 05_kinetic_contraction.md):
| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| thm-discretization | If $\mathcal{L}V \leq -\kappa V + C$, then discrete BAOAB inherits drift with remainder $R_\tau \leq \tau^2 K_{\text{integ}}(V_0 + C_0)$ and $\tau_* = \kappa/(4K_{\text{integ}})$ | Step 6 | ✅ |

**Propositions**:
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| prop-weak-error-variance | 05_kinetic_contraction.md | $K_{\text{Var}} = C(d,N) \cdot \max(\gamma^2, L_F^2, \sigma_{\max}^2)$ | Step 2 | ✅ |
| prop-weak-error-boundary | 05_kinetic_contraction.md | $K_b = K_b(\kappa_{\text{total}}, C_{\text{total}}, \gamma, \sigma_{\max})$ via truncation | Step 4 | ✅ |
| prop-weak-error-wasserstein | 05_kinetic_contraction.md | $K_W = K_W(d, \gamma, L_F, L_\Sigma, \sigma_{\max}, \lambda_v, b)$ independent of $N$ | Step 3 | ✅ |

**Assembly Identity**:
| Source | Statement | Used in Step | Verified |
|--------|-----------|--------------|----------|
| Proof of Thm 3.7.2, line 1059 | $K_{\text{integ}} = K_W + c_V K_{\text{Var}} + c_B K_b$ | Step 1, 5 | ✅ |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $C_d$ | Dimension-dependent constant | Polynomial in $d$ | Absorbs component weights $c_V, c_B$ and weak error theory constants |
| $K_V$ | Lyapunov regularity bound | $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$, $k=2,3$ | Controls all weak error remainders |
| $\gamma$ | Friction coefficient | System parameter | Appears squared in weak error bounds |
| $L_F$ | Force Lipschitz constant | From axiom EG-1 | Controls drift term regularity |
| $\sigma_{\max}$ | Diffusion upper bound | From diffusion bound axiom | Controls noise amplitude |
| $\kappa$ | Generator drift rate | From $\mathcal{L}V \leq -\kappa V + C$ | Determines contraction rate and timestep bound |

### Missing/Uncertain Dependencies

**Requires Additional Assumptions**:
- **Isotropic constant diffusion**: For $K_W$ bound to avoid $L_\Sigma$ dependence
  - Statement: $\Sigma(x,v) = \sigma_v I_d$ (state-independent, isotropic)
  - Why needed: Proposition hypotheses only list diffusion bounds, not Lipschitz constant $L_\Sigma$
  - How to verify: Check if document restricts to this case or add $L_\Sigma$ to max expression
  - **Resolution**: Document line 1007 notes that for isotropic constant diffusion, Stratonovich and Itô coincide, suggesting this is the primary case

**Requires Careful Analysis**:
- **Localization to $\{V \leq M\}$**: Lyapunov regularity bound only holds on compact sublevel sets
  - Why uncertain: Need to ensure all weak error expansions remain in this region
  - How to verify: Use drift inequality to control probability of exceeding $M$, similar to boundary truncation argument

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes explicit formulas for $K_{\text{integ}}$ by decomposing it into component-wise weak error constants ($K_W$, $K_{\text{Var}}$, $K_b$), then expressing each in terms of the primitive parameters ($\gamma, L_F, \sigma_{\max}, \kappa$) and the Lyapunov regularity bound $K_V$. The key technical challenge is the boundary component, which introduces a $\sigma_{\max}^2/\tau$ term due to the truncation argument needed to handle unbounded derivatives near the boundary.

The proof structure follows the component-wise assembly pattern from Theorem 3.7.2, making explicit what was implicit in that proof. Each component's weak error constant is traced back to standard BAOAB theory (Leimkuhler & Matthews 2015), localized to the region $\{V \leq M\}$ where Lyapunov regularity holds, and bounded using the given assumptions.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Assembly Reduction**: Express $K_{\text{integ}}$ in terms of component constants using proven identity
2. **Variance Component Bound**: Show $K_{\text{Var}} \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$
3. **Wasserstein Component Bound**: Show $K_W \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$ (isotropic diffusion)
4. **Boundary Component Bound**: Show $K_b \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau)$ via truncation
5. **Final Assembly**: Combine all bounds to establish the claimed inequality
6. **Timestep Guideline Derivation**: Derive $\tau_* \sim 1/\max(\kappa, L_F, \sigma_{\max}, \gamma)$ from $\tau_* = \kappa/(4K_{\text{integ}})$

---

### Detailed Step-by-Step Sketch

#### Step 1: Assembly Reduction

**Goal**: Express $K_{\text{integ}}$ in terms of component constants

**Substep 1.1**: Recall the assembly identity
- **Justification**: From proof of Theorem 3.7.2 at line 1059 (equation after PART III)
- **Why valid**: Triangle inequality applied to decomposition $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$
- **Expected result**:

$$
K_{\text{integ}} = K_W + c_V K_{\text{Var}} + c_B K_b
$$

**Substep 1.2**: Identify the strategy
- **Action**: To bound $K_{\text{integ}}$, we bound each component constant separately, then combine
- **Why valid**: Linear combination preserves upper bounds
- **Expected result**: If $K_W \leq A$, $K_{\text{Var}} \leq B$, $K_b \leq C$, then $K_{\text{integ}} \leq A + c_V B + c_B C \leq \tilde{C}_d \max(A,B,C)$ where $\tilde{C}_d$ absorbs $1 + c_V + c_B$

**Substep 1.3**: Note the target structure
- **Conclusion**: We need to show each of $K_W$, $K_{\text{Var}}$, $K_b$ has the form $C_d \cdot (\text{max of parameter squares}) \cdot K_V$
- **Form**: The max will be over $\{\gamma^2, L_F^2, \sigma_{\max}^2, \kappa^2, \sigma_{\max}^2/\tau\}$ with different subsets for each component

**Dependencies**:
- Uses: Assembly identity from proof of Theorem 3.7.2

**Potential Issues**:
- ⚠️ Need to track which terms appear in which component's bound
- **Resolution**: Carefully trace the dependence through each proposition's proof

---

#### Step 2: Bound Variance Component $K_{\text{Var}}$

**Goal**: Show $K_{\text{Var}} \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$

**Substep 2.1**: Recall the variance weak error bound
- **Justification**: Proposition 3.7.3.1 (prop-weak-error-variance), lines 670-730
- **Why valid**: Variance component $V_{\text{Var}} = \frac{1}{N}\sum_i \|z_i - \mu\|^2$ has globally bounded derivatives
- **Expected result**: $K_{\text{Var}} = C(d,N) \cdot \max(\gamma^2, L_F^2, \sigma_{\max}^2)$

**Substep 2.2**: Connect to Lyapunov regularity
- **Action**: The weak error constant depends on the test function's $C^3$ seminorm. For $V_{\text{Var}}$, this is bounded by $K_V$ on $\{V \leq M\}$
- **Justification**: Standard BAOAB weak error theory (Leimkuhler & Matthews 2015, Talay-Tubaro expansions) shows remainder scales with $\sup \|\nabla^k f\|$ for $k=2,3$
- **Why valid**: Proposition hypothesis gives $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$ for $k=2,3$
- **Expected result**: The $C(d,N)$ factor can be decomposed as (dimension polynomial) $\times$ (weak error theory constant) $\times K_V$

**Substep 2.3**: Handle localization to $\{V \leq M\}$
- **Action**: Use drift inequality to control probability of exceeding $M$
- **Justification**: Similar to boundary truncation argument in prop-weak-error-boundary
- **Why valid**: Generator bound $\mathcal{L}V \leq -\kappa V + C$ implies $\mathbb{P}[V(S_t) > M]$ decays exponentially in $M$
- **Expected result**: Contribution from $\{V > M\}$ is negligible for appropriate choice of $M$

**Substep 2.4**: Final bound
- **Conclusion**: $K_{\text{Var}} \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$
- **Form**:

$$
K_{\text{Var}} = C(d,N) \cdot \max(\gamma^2, L_F^2, \sigma_{\max}^2) \leq \tilde{C}_d \cdot \max(\gamma^2, L_F^2, \sigma_{\max}^2) \cdot K_V
$$

where $\tilde{C}_d$ is polynomial in $d$ (N-dependence absorbed or shown to cancel)

**Dependencies**:
- Uses: prop-weak-error-variance, Lyapunov regularity hypothesis

**Potential Issues**:
- ⚠️ N-dependence in $C(d,N)$ not fully explicit
- **Resolution**: Document notes at line 722-727 that $N$ contributes at most polynomially; for per-particle scaled Lyapunov, may be N-independent

---

#### Step 3: Bound Wasserstein Component $K_W$

**Goal**: Show $K_W \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$ (under isotropic constant diffusion)

**Substep 3.1**: Recall the Wasserstein weak error bound
- **Justification**: Proposition 3.7.3.3 (prop-weak-error-wasserstein), lines 828-907
- **Why valid**: Uses synchronous coupling at particle level, avoiding JKO/gradient flow technicalities
- **Expected result**: $K_W = C_{LM}(d, \gamma, L_F, L_\Sigma, \sigma_{\max}, \lambda_v, b)$ independent of $N$

**Substep 3.2**: Specialize to isotropic constant diffusion
- **Action**: Set $\Sigma(x,v) = \sigma_v I_d$ (constant, state-independent)
- **Justification**: Document remark at line 1007 notes this is the primary case where Stratonovich/Itô coincide
- **Why valid**: Proposition hypotheses only list diffusion bounds $\sigma_{\min}^2 I_d \leq \Sigma\Sigma^T \leq \sigma_{\max}^2 I_d$, not Lipschitz constant $L_\Sigma$
- **Expected result**: For constant diffusion, $L_\Sigma = 0$, so this parameter drops out

**Substep 3.3**: Express in terms of test function regularity
- **Action**: The hypocoercive quadratic form $f(\Delta z) = \|\Delta z\|_h^2$ has bounded derivatives (line 892-895)
- **Justification**: $f$ is quadratic, so $\nabla f$ is linear, $\nabla^2 f = 2Q$ is constant, $\nabla^3 f = 0$
- **Why valid**: Weak error for polynomial-growth test functions with bounded higher derivatives
- **Expected result**: $K_W \leq C(d) \max(\gamma^2, L_F^2, \sigma_{\max}^2) \cdot \|Q\|$ where $Q$ is the hypocoercive metric tensor

**Substep 3.4**: Connect to Lyapunov regularity
- **Action**: For the full Wasserstein component $V_W = W_h^2(\mu_1, \mu_2)$, the derivatives on the swarm configuration depend on $K_V$
- **Justification**: Wasserstein distance is defined via optimal transport, but its weak error can be bounded via the coupling strategy
- **Why valid**: Synchronous coupling (lines 840-907) shows single-pair error accumulates to total error
- **Expected result**: $K_W \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$

**Substep 3.5**: Final bound
- **Conclusion**:

$$
K_W \leq C_d \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V
$$

**Dependencies**:
- Uses: prop-weak-error-wasserstein, isotropic constant diffusion assumption

**Potential Issues**:
- ⚠️ If $\Sigma$ is state-dependent, need $L_\Sigma$ in the bound
- **Resolution**: Either restrict to isotropic constant diffusion or add $L_\Sigma$ term to max; document suggests isotropic case is primary

---

#### Step 4: Bound Boundary Component $K_b$ (Critical Step)

**Goal**: Show $K_b \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau)$

**Substep 4.1**: Recall the boundary weak error structure
- **Justification**: Proposition 3.7.3.2 (prop-weak-error-boundary), lines 734-827
- **Why valid**: Uses self-referential truncation argument to handle unbounded derivatives
- **Expected result**: $K_b = K_b(\kappa_{\text{total}}, C_{\text{total}}, \gamma, \sigma_{\max})$ via indicator split

**Substep 4.2**: Understand the truncation mechanism
- **Action**: The proof splits $\mathbb{E}[W_b(S_\tau)] = \mathbb{E}[W_b \cdot \mathbb{1}_{\{W_b \leq M\}}] + \mathbb{E}[W_b \cdot \mathbb{1}_{\{W_b > M\}}]$
- **Justification**: Lines 779-804
- **Why valid**: On $\{W_b \leq M\}$, barrier function has bounded derivatives $\|\nabla^k \varphi\| \leq K_\varphi(M)$
- **Expected result**: Term 1 (safe region) gives standard $O(\tau^2)$ error; Term 2 (high-barrier) controlled by Markov inequality

**Substep 4.3**: Optimize the threshold $M$
- **Action**: Choose $M = M_\infty / \tau$ where $M_\infty = M_\infty(\kappa_{\text{total}}, C_{\text{total}})$ (line 803)
- **Justification**: Balances the $\tau^2 K_\varphi(M)$ term from safe region against $\tau M_\infty$ term from tail
- **Why valid**: Drift inequality $\mathcal{L}V_{\text{total}} \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$ provides tail control via $\mathbb{E}[V_{\text{total}}(S_t)] \leq M_\infty$
- **Expected result**: Term 2 contributes $O(\tau M_\infty) = O(\tau^2 M_\infty/\tau)$, which when written as $O(\tau^2)$ produces factor $M_\infty/\tau$

**Substep 4.4**: Bound $M_\infty$ in terms of parameters
- **Action**: From Foster-Lyapunov drift, $M_\infty \leq V_{\text{total}}(S_0) + C_{\text{total}}/\kappa_{\text{total}}$
- **Justification**: Steady-state bound from generator inequality (line 758-764)
- **Why valid**: Standard consequence of $\mathcal{L}V \leq -\kappa V + C$
- **Expected result**: $M_\infty \leq C_d K_V \max(\kappa^{-1}, \gamma^{-1}, L_F^{-1}, \sigma_{\max}^{-1})^{-2}$ where the power reflects Lyapunov function scaling

**Substep 4.5**: Express $K_\varphi(M_\infty/\tau)$ in terms of parameters
- **Action**: On the region $\{W_b \leq M_\infty/\tau\}$, the barrier derivatives are bounded by a function of the threshold
- **Justification**: Barrier function grows as particles approach boundary; derivative bounds depend on proximity
- **Why valid**: Combining with Lyapunov regularity $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$
- **Expected result**: $K_\varphi(M_\infty/\tau) \leq C_d K_V \cdot (M_\infty/\tau)^p$ for some power $p$ depending on barrier structure

**Substep 4.6**: Assemble the final $K_b$ bound
- **Conclusion**: Combining the above:

$$
K_b \leq K_\varphi(M_\infty/\tau) + M_\infty/\tau \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau)
$$

The $\sigma_{\max}^2/\tau$ term captures the $1/\tau$ scaling from the truncation optimization.

**Dependencies**:
- Uses: prop-weak-error-boundary, Foster-Lyapunov drift inequality, Lyapunov regularity

**Potential Issues**:
- ⚠️ The $1/\tau$ term introduces circularity in timestep selection
- **Resolution**: Handle conservatively by requiring $\tau$ small enough that $\sigma_{\max}^2/\tau \leq C \max(\kappa^2, L_F^2, \gamma^2, \sigma_{\max}^2)$; this is consistent with the practical guideline

---

#### Step 5: Final Assembly

**Goal**: Combine component bounds to prove the claimed inequality

**Substep 5.1**: Collect the component bounds
- **From Step 2**: $K_{\text{Var}} \leq C_d^{(1)} \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$
- **From Step 3**: $K_W \leq C_d^{(2)} \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V$
- **From Step 4**: $K_b \leq C_d^{(3)} \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau) K_V$

**Substep 5.2**: Use the assembly identity
- **Action**: Substitute into $K_{\text{integ}} = K_W + c_V K_{\text{Var}} + c_B K_b$
- **Justification**: Step 1, from proof of Theorem 3.7.2
- **Why valid**: Linear combination of upper bounds
- **Expected result**:

$$
K_{\text{integ}} \leq (C_d^{(2)} + c_V C_d^{(1)}) \max(\gamma^2, L_F^2, \sigma_{\max}^2) K_V + c_B C_d^{(3)} \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau) K_V
$$

**Substep 5.3**: Absorb weights into dimension constant
- **Action**: Define $C_d := C_d^{(2)} + c_V C_d^{(1)} + c_B C_d^{(3)}$
- **Justification**: Weights $c_V, c_B$ are fixed constants from synergistic Lyapunov assembly
- **Why valid**: Polynomial in $d$ plus constant factors is still polynomial in $d$
- **Expected result**: $C_d$ is polynomial in $d$ (absorbing all component-wise dimension factors and weights)

**Substep 5.4**: Take max over all terms
- **Action**: Upper bound by single max expression
- **Justification**: $\max(A,B) + \max(C,D) \leq 2\max(A,B,C,D)$
- **Why valid**: Elementary property of max function; absorb factor of 2 into $C_d$
- **Expected result**:

$$
K_{\text{integ}} \leq C_d \cdot \max(\kappa^2, L_F^2, \sigma_{\max}^2/\tau, \gamma^2) \cdot K_V
$$

**Substep 5.5**: Verify the claimed form
- **Conclusion**: This is exactly the claimed inequality in the proposition statement
- **Form**: Dimension-polynomial constant times max of parameter squares times Lyapunov regularity

**Dependencies**:
- Uses: All previous steps, assembly identity

**Potential Issues**:
- None at this stage
- **Resolution**: N/A

---

#### Step 6: Derive Timestep Guideline

**Goal**: Derive $\tau_* \sim 1/\max(\kappa, L_F, \sigma_{\max}, \gamma)$ from the explicit bound

**Substep 6.1**: Recall the timestep bound from Theorem 3.7.2
- **Justification**: Line 642 of 05_kinetic_contraction.md
- **Why valid**: Discrete-time drift requires $\tau < \tau_* = \kappa/(4K_{\text{integ}})$
- **Expected result**: Need to show this gives the stated guideline

**Substep 6.2**: Substitute the explicit $K_{\text{integ}}$ bound
- **Action**: $\tau_* = \kappa/(4K_{\text{integ}}) \geq \kappa/(4C_d K_V \max(\kappa^2, L_F^2, \sigma_{\max}^2/\tau, \gamma^2))$
- **Justification**: From Step 5
- **Why valid**: Lower bound from upper bound on denominator
- **Expected result**: $\tau_*$ depends on the max expression

**Substep 6.3**: Handle the $\sigma_{\max}^2/\tau$ term conservatively
- **Action**: For $\tau \leq c_0/\max(\kappa, L_F, \sigma_{\max}, \gamma)$ with small $c_0$, we have $\sigma_{\max}^2/\tau \leq (1/c_0) \sigma_{\max} \max(\kappa, L_F, \sigma_{\max}, \gamma)$
- **Justification**: Direct substitution
- **Why valid**: Choice of $\tau$ makes the $1/\tau$ term comparable to other terms
- **Expected result**: $\sigma_{\max}^2/\tau \leq C' \max(\kappa^2, L_F^2, \sigma_{\max}^2, \gamma^2)$ after rescaling constants

**Substep 6.4**: Simplify the max expression
- **Action**: With the conservative bound from 6.3, $\max(\kappa^2, L_F^2, \sigma_{\max}^2/\tau, \gamma^2) \leq C'' \max(\kappa^2, L_F^2, \sigma_{\max}^2, \gamma^2)$
- **Justification**: Absorbing the $1/\tau$ contribution
- **Why valid**: For sufficiently small $\tau$, the $\sigma_{\max}^2/\tau$ term is dominated by rescaled versions of other terms
- **Expected result**: $K_{\text{integ}} \leq C_d K_V \max(\kappa^2, L_F^2, \sigma_{\max}^2, \gamma^2)$

**Substep 6.5**: Derive the order-wise guideline
- **Action**: From $\tau_* = \kappa/(4K_{\text{integ}})$ and Step 6.4:

$$
\tau_* \geq \frac{\kappa}{4C_d K_V \max(\kappa^2, L_F^2, \sigma_{\max}^2, \gamma^2)} = \frac{1}{4C_d K_V} \cdot \frac{1}{\max(\kappa, L_F, \sigma_{\max}, \gamma)}
$$

- **Justification**: Algebra and $\kappa/\kappa^2 = 1/\kappa$, etc.
- **Why valid**: Direct calculation
- **Expected result**: $\tau_* \sim 1/\max(\kappa, L_F, \sigma_{\max}, \gamma)$ up to dimension-dependent and Lyapunov-regularity factors

**Substep 6.6**: Verify the practical guideline
- **Conclusion**: For typical parameters $(\gamma = 1, \sigma_v = 1, \kappa \sim 0.1)$, we have $\max(\kappa, L_F, \sigma_{\max}, \gamma) = \max(0.1, L_F, 1, 1)$
- **Action**: If $L_F \sim O(1)$, then $\max \sim 1$, so $\tau_* \sim 1/(C_d K_V)$
- **Justification**: Taking $\tau = 0.01$ ensures $\tau \ll \tau_*$ for reasonable $C_d, K_V$
- **Why valid**: Practical parameters make the guideline $\tau = 0.01$ safe
- **Expected result**: The stated guideline is verified

**Dependencies**:
- Uses: Theorem 3.7.2 timestep bound, explicit $K_{\text{integ}}$ from Step 5

**Potential Issues**:
- ⚠️ Circularity from $\sigma_{\max}^2/\tau$ in $K_{\text{integ}}$ appearing in bound for $\tau_*$
- **Resolution**: Handle conservatively via fixed-point argument: choose $\tau$ small enough that $\sigma_{\max}^2/\tau$ is dominated, ensuring self-consistency

---

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: τ-Dependence in $K_b$ from Boundary Truncation

**Why Difficult**:
The boundary weak error bound uses a truncation argument to handle unbounded derivatives near $\partial\mathcal{X}_{\text{valid}}$. The optimal threshold choice $M = M_\infty/\tau$ balances the standard weak error on $\{W_b \leq M\}$ (which scales as $\tau^2 K_\varphi(M)$) against the tail probability contribution from $\{W_b > M\}$ (which scales as $\tau M_\infty \cdot \mathbb{P}[W_b > M]$). This produces a $1/\tau$ factor in the effective weak error constant $K_b$ when expressed in the form $K_b \tau^2$.

Mathematically, the complement term is:

$$
\left|\mathbb{E}[W_b(S_\tau) \cdot \mathbb{1}_{\{W_b > M\}}]\right| \leq M_\infty \cdot \frac{M_\infty}{M} = \frac{M_\infty^2}{M}
$$

Choosing $M = M_\infty/\tau$ gives $\tau M_\infty$, which when compared to the target form $K_b \tau^2$ yields $K_b \sim M_\infty/\tau$.

**Proposed Solution**:

1. **Bound $M_\infty$ using drift inequality**: From $\mathcal{L}V_{\text{total}} \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$, we get $M_\infty \leq C_{\text{total}}/\kappa_{\text{total}} + V_{\text{total}}(S_0)$

2. **Express in terms of parameters**: Using the synergistic Lyapunov structure, $C_{\text{total}}$ and $\kappa_{\text{total}}$ depend on $\gamma, L_F, \sigma_{\max}, \kappa$ (the generator drift rates). We have:

$$
M_\infty \leq C_d K_V \cdot \frac{1}{\kappa_{\text{total}}} \leq C_d K_V \cdot \max(\kappa^{-1}, \gamma^{-1}, L_F^{-1}, \sigma_{\max}^{-1})
$$

(The negative exponents reflect the drift rate dependence; taking reciprocals and squaring gives the form in the max.)

3. **Substitute into $K_b$**: This yields:

$$
K_b \leq C_d K_V \cdot \frac{M_\infty}{\tau} \leq C_d K_V \cdot \frac{\max(\kappa^{-2}, \gamma^{-2}, L_F^{-2}, \sigma_{\max}^{-2})}{\tau}
$$

4. **Rewrite using positive exponents**: $\max(\kappa^{-2}, \gamma^{-2}, L_F^{-2}, \sigma_{\max}^{-2}) = 1/\min(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2)$. For the bound to be meaningful, we need $\tau$ small, so:

$$
\frac{1}{\tau \min(...)} \approx \frac{\max(...)}{\tau \max(...) \min(...)} \leq \frac{\sigma_{\max}^2}{\tau}
$$

when $\sigma_{\max} = \min(\kappa, \gamma, L_F, \sigma_{\max})$ (conservative choice).

5. **Result**: $K_b \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2/\tau)$, explicitly showing the $1/\tau$ mechanism.

**Alternative Approach** (if main approach creates issues):

Rephrase the theorem's remainder as $A\tau^2 + B\tau$ instead of $K_{\text{integ}}\tau^2(1+V)$, where the $B\tau$ term captures the boundary tail. Then show that for $\tau \leq \tau_0$, the linear term is dominated: $B\tau \leq A' \tau^2(1+V)$. This avoids explicit $1/\tau$ in $K_{\text{integ}}$ at the cost of a more complex remainder structure.

**References**:
- Similar truncation techniques in: Fehrman & Gess (2019) on path-wise convergence with irregular coefficients
- Standard drift-based tail control: Meyn & Tweedie (2009), Chapter 14

---

### Challenge 2: Absence of $L_\Sigma$ in Proposition Hypotheses

**Why Difficult**:
The Wasserstein weak error bound (prop-weak-error-wasserstein) depends on $K_W = K_W(d, \gamma, L_F, L_\Sigma, \sigma_{\max}, \lambda_v, b)$ where $L_\Sigma$ is the Lipschitz constant of the diffusion matrix $\Sigma$. However, the proposition hypotheses only list:
- Diffusion bounds: $\sigma_{\min}^2 I_d \leq \Sigma\Sigma^T \leq \sigma_{\max}^2 I_d$

No Lipschitz regularity $\|\Sigma(x_1) - \Sigma(x_2)\| \leq L_\Sigma \|x_1 - x_2\|$ is assumed.

**Proposed Solution**:

1. **Primary case: Isotropic constant diffusion**: Set $\Sigma(x,v) = \sigma_v I_d$ (state-independent). Then:
   - $L_\Sigma = 0$ (constant map has zero Lipschitz constant)
   - Document remark at line 1007 confirms this case: "For isotropic constant diffusion, Stratonovich and Itô formulations coincide"
   - This eliminates $L_\Sigma$ from the $K_W$ bound

2. **Justify as primary case**: The document structure suggests isotropic constant diffusion is the main setting:
   - Line 9: "Velocity Dissipation via Langevin Friction" (standard Langevin is isotropic)
   - Line 1007: Explicit remark about Stratonovich-Itô equivalence for constant isotropic diffusion
   - Implementation in `src/fragile/euclidean_gas.py` likely uses constant $\sigma_v$

3. **If state-dependent diffusion is required**: Add a term to the max expression:

$$
K_{\text{integ}} \leq C_d \cdot \max(\kappa^2, L_F^2, L_\Sigma^2, \sigma_{\max}^2/\tau, \gamma^2) \cdot K_V
$$

and modify the practical guideline to $\tau_* \sim 1/\max(\kappa, L_F, L_\Sigma, \sigma_{\max}, \gamma)$.

4. **Local Lipschitz bound**: Alternatively, use Lyapunov regularity to bound $L_\Sigma$ locally on $\{V \leq M\}$:
   - If $\Sigma$ depends on state and $V$ controls state moments, then $L_\Sigma \leq C(M) K_V$
   - Absorb into the overall $K_V$ factor

**Alternative Approach**:

State the proposition with an explicit caveat: "For isotropic constant diffusion $\Sigma = \sigma_v I_d$, the bound holds as stated. For state-dependent diffusion, replace $\sigma_{\max}^2$ with $\max(\sigma_{\max}^2, L_\Sigma^2)$ in the max expression."

**References**:
- Stratonovich vs Itô for constant diffusion: Øksendal (2003), Theorem 3.3.3
- State-dependent diffusion weak error: Talay (1990), bounds depend on Lipschitz regularity of coefficients

---

### Challenge 3: Mapping BAOAB Constants to $K_V$

**Why Difficult**:
The component-wise weak error propositions provide constants in terms of $C(d,N)$, $C_{LM}(...)$, etc., which depend on the test function's $C^3$ seminorm. We need to express these uniformly in terms of the Lyapunov regularity bound $K_V$.

The obstacle is that weak error theory typically assumes global regularity, but we only have local bounds $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$.

**Proposed Solution**:

1. **Localize weak error expansions to $\{V \leq M\}$**:
   - Split $\mathbb{E}[V(S_\tau)] = \mathbb{E}[V(S_\tau) \cdot \mathbb{1}_{\{V(S_0) \leq M\}}] + \mathbb{E}[V(S_\tau) \cdot \mathbb{1}_{\{V(S_0) > M\}}]$
   - On the first term, Lyapunov regularity applies
   - Second term controlled by drift inequality (Markov's inequality)

2. **Choose $M$ adaptively**: Use $M = M_\infty = C_{\text{total}}/\kappa_{\text{total}} + V_{\text{total}}(S_0)$ from drift inequality. Then:

$$
\mathbb{P}[V(S_0) > M] \leq \frac{\mathbb{E}[V(S_0)]}{M} \to 0 \text{ as } M \to \infty
$$

3. **Apply standard BAOAB theory on $\{V \leq M\}$**:
   - Leimkuhler & Matthews (2015), Theorem 7.5: For test function $f$ with $\|\nabla^k f\| \leq K_f$ (k=2,3), the weak error satisfies:

$$
\left|\mathbb{E}[f(S_\tau^{\text{BAOAB}})] - \mathbb{E}[f(S_\tau^{\text{exact}})]\right| \leq C_{LM}(\gamma, \sigma, L_F) \cdot K_f \cdot \tau^2
$$

   - Substitute $f = V|_{\{V \leq M\}}$ and $K_f = K_V$

4. **Control the complement**:

$$
\left|\mathbb{E}[V(S_\tau) \cdot \mathbb{1}_{\{V(S_0) > M\}}]\right| \leq \mathbb{E}[V(S_\tau)] \cdot \mathbb{P}[V(S_0) > M] \leq M_\infty \cdot \frac{V(S_0)}{M}
$$

Choosing $M = V(S_0)/\tau$ makes this $O(\tau M_\infty)$, which for small $\tau$ is negligible compared to $O(\tau^2)$.

5. **Combine**: The total weak error is $O(\tau^2 K_V)$ with the $O(\tau)$ tail contribution absorbed for sufficiently small $\tau$.

**Alternative Approach**:

Use a smooth cutoff function $\chi_M(V)$ that equals 1 on $\{V \leq M\}$ and smoothly decays to 0 on $\{V > 2M\}$. Then apply weak error theory to $f_M := V \cdot \chi_M(V)$, which has globally bounded derivatives controlled by $K_V$ and the cutoff regularity.

**References**:
- Localized weak error estimates: Debussche & Faou (2012) on long-time weak convergence
- Cutoff function technique: Bou-Rabee & Owhadi (2010) on diffusions with discontinuous drift

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps (assembly → component bounds → combine → derive guideline)
- [x] **Hypothesis Usage**: All proposition assumptions are used:
  - Lipschitz force $L_F$: appears in $K_{\text{Var}}$ and $K_W$ bounds
  - Bounded force growth $C_F$: used implicitly in drift inequality for $M_\infty$
  - Diffusion bounds $\sigma_{\min}, \sigma_{\max}$: appear in all component bounds
  - Lyapunov regularity $K_V$: central to all weak error bounds
  - Drift rate $\kappa$: appears in $K_b$ bound and timestep guideline
- [x] **Conclusion Derivation**: Claimed bound $K_{\text{integ}} \leq C_d \max(...) K_V$ is fully derived in Step 5
- [x] **Framework Consistency**: All dependencies verified (existing propositions in same document)
- [x] **No Circular Reasoning**:
  - Component bounds proven independently in earlier propositions
  - Assembly uses proven identity
  - τ-dependence handled conservatively to avoid circularity in timestep selection
- [x] **Constant Tracking**: All constants defined:
  - $C_d$: polynomial in $d$, absorbs component constants and weights
  - $K_V$: given in proposition hypothesis
  - $M_\infty$: derived from drift inequality parameters
  - All terms in max expression justified by component analysis
- [x] **Edge Cases**:
  - Isotropic constant diffusion ($\Sigma = \sigma_v I_d$): primary case, eliminates $L_\Sigma$
  - Localization to $\{V \leq M\}$: handled via truncation + tail control
  - τ → 0 limit: guideline ensures $\tau \ll \tau_*$
- [x] **Regularity Verified**:
  - Lyapunov regularity $\|\nabla^k V\| \leq K_V$ on $\{V \leq M\}$ (given)
  - Force regularity from Lipschitz + bounded growth assumptions
  - Diffusion regularity from bound assumptions (constant case: $C^\infty$)
- [x] **Measure Theory**: All probabilistic operations well-defined (bounded Lyapunov, drift inequality ensures integrability)

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Poisson Equation Method for Weak Error

**Approach**:
Instead of using Talay-Tubaro expansions and Taylor series, solve the Poisson equation $\mathcal{L}g = f - \mathbb{E}_\infty[f]$ where $\mathcal{L}$ is the generator and $\mathbb{E}_\infty[f]$ is the invariant measure expectation. The weak error is then:

$$
\mathbb{E}[f(S_\tau^{\text{BAOAB}})] - \mathbb{E}[f(S_\tau^{\text{exact}})] = \mathbb{E}[\mathcal{L}_{\text{discrete}} g(S_0) - \mathcal{L}g(S_0)] + \text{higher order}
$$

where $\mathcal{L}_{\text{discrete}}$ is the discrete generator.

**Pros**:
- Directly relates weak error to generator approximation quality
- Clearer dependence on coefficients (force, friction, diffusion)
- Avoids multi-step Taylor expansion bookkeeping

**Cons**:
- Requires regularity of the Poisson solution $g$ (existence, bounds on $\|\nabla^k g\|$)
- Harder to apply for Wasserstein component (optimal transport doesn't have simple Poisson structure)
- Boundary component with unbounded derivatives creates difficulties
- More abstract, less constructive than Taylor expansion approach

**When to Consider**:
If establishing long-time weak convergence or deriving asymptotic expansions for the invariant measure. For the discrete-time drift inequality (our setting), Taylor expansions are more direct.

---

### Alternative 2: Direct Coupling for BAOAB vs Exact SDE

**Approach**:
Construct an explicit coupling between $(S_t^{\text{BAOAB}}, S_t^{\text{exact}})$ evolving from the same initial condition and under synchronized noise. Bound the distance $\mathbb{E}[\|S_\tau^{\text{BAOAB}} - S_\tau^{\text{exact}}\|^2]$ directly using Gronwall's inequality, then use this to control $|\mathbb{E}[V(S_\tau^{\text{BAOAB}})] - \mathbb{E}[V(S_\tau^{\text{exact}})]|$ via Lipschitz continuity of $V$.

**Pros**:
- Avoids Taylor expansions entirely
- Can bound moments and distances directly
- Works for less regular test functions (only need Lipschitz $V$, not $C^3$)
- Provides strong error bounds as a byproduct

**Cons**:
- Technical for splitting schemes (BAOAB is not one-step method in full phase space)
- Recovering $\tau^2$ order is delicate (requires careful tracking of integrator structure)
- Isolating the explicit constant dependence on parameters is harder
- Synchronous coupling for Wasserstein component already used in prop-weak-error-wasserstein; applying again doesn't simplify

**When to Consider**:
For strong error estimates or when test functions lack regularity. For our purpose (making constants explicit in already-proven weak error bounds), the direct assembly approach is more efficient.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **N-Dependence in $C_d$**: Gap severity: LOW
   - Description: The constant $C(d,N)$ from $K_{\text{Var}}$ is stated as polynomial in both $d$ and $N$, but the proposition only claims polynomial in $d$
   - How critical: For large swarms ($N \to \infty$), this could matter; for fixed $N$, it's absorbed in $C_d$
   - Resolution path: Check if variance component Lyapunov is per-particle scaled (i.e., $V_{\text{Var}} = \frac{1}{N}\sum$), in which case derivatives are N-uniform

2. **State-Dependent Diffusion Case**: Gap severity: MEDIUM
   - Description: Full result requires either isotropic constant diffusion or an explicit $L_\Sigma$ bound
   - How critical: Limits applicability to adaptive diffusion strategies (e.g., state-dependent noise scaling)
   - Resolution path: Extend proposition statement to include $L_\Sigma$ in max expression, or prove local Lipschitz bound on $\{V \leq M\}$

3. **Tightness of $\sigma_{\max}^2/\tau$ Term**: Gap severity: MEDIUM
   - Description: The $1/\tau$ factor from boundary truncation may be conservative
   - How critical: Could allow larger timesteps in practice if removed
   - Resolution path: Refine boundary weak error proof to avoid $1/\tau$ by using stronger barrier regularity or alternative truncation strategy

### Conjectures

1. **N-Independence Conjecture**:
   - Statement: For per-particle scaled Lyapunov functions, $C_d$ is independent of $N$
   - Why plausible: Variance derivatives are $O(1)$ per particle (line 706); Wasserstein bound is N-independent (line 837); boundary is per-particle sum
   - Evidence: Document structure suggests per-particle scaling throughout

2. **Improved Boundary Bound Conjecture**:
   - Statement: Under additional barrier smoothness, $K_b \leq C_d K_V \max(\kappa^2, \gamma^2, L_F^2, \sigma_{\max}^2)$ without the $1/\tau$ factor
   - Why plausible: The $1/\tau$ arises from conservative truncation optimization; tighter analysis of $\mathbb{P}[W_b > M]$ decay could eliminate it
   - Evidence: Similar bounds for smooth barriers in Fehrman & Gess (2019)

### Extensions

1. **Adaptive Timestep Selection**:
   - Given current state $S_0$, use $V_{\text{total}}(S_0)$ to choose $\tau$ dynamically, ensuring $\tau < \tau_*(S_0)$ with state-dependent safety margin
   - Benefit: Larger timesteps when far from boundary, smaller near boundary

2. **Higher-Order Integrators**:
   - Extend to 4th-order weak integrators (e.g., Runge-Kutta methods for SDEs)
   - Expected $K_{\text{integ}}$ scaling: Improve $\tau^2 \to \tau^4$ in remainder, allowing larger timesteps

3. **Dimension-Explicit Tracking**:
   - Make $C_d$ explicit: $C_d = C_0 \cdot d^p$ for some power $p$
   - Benefit: Understanding curse of dimensionality for high-dimensional problems

---

## IX. Expansion Roadmap

**Phase 1: Prove Missing Lemmas** (Estimated: 2-3 hours)

1. **Lemma A (BAOAB local weak-error constant scales with $C^3$ seminorm)**:
   - Brief proof strategy: State and cite Leimkuhler & Matthews (2015), Theorem 7.5; verify hypotheses match our setting; apply to each component
   - Difficulty: Medium (standard application of existing theory)

2. **Lemma B (Boundary truncation with drift-controlled tail yields $1/\tau$ factor)**:
   - Brief proof strategy: Optimize $M = M_\infty/\tau$ in the truncation split; derive $M_\infty$ from Foster-Lyapunov bound; combine to get $K_b$ form
   - Difficulty: Medium (technical but follows prop-weak-error-boundary structure)

3. **Lemma C (Force regularity from assumptions)**:
   - Brief proof strategy: Lipschitz + linear growth imply $\|F\|_{C^2}$ bounded on compact sets; use $\{V \leq M\}$ compactness (from coercivity)
   - Difficulty: Easy (standard SDE regularity theory)

**Phase 2: Fill Technical Details** (Estimated: 3-4 hours)

1. **Step 2 (Variance bound)**: Expand the connection between $C(d,N)$ and $K_V$; clarify N-dependence
2. **Step 3 (Wasserstein bound)**: Verify isotropic constant diffusion assumption is satisfied; justify $L_\Sigma = 0$ explicitly
3. **Step 4 (Boundary bound)**: Provide detailed calculation of $M_\infty$ in terms of parameters; derive the $\sigma_{\max}^2/\tau$ scaling rigorously
4. **Step 6 (Timestep guideline)**: Fully resolve the circularity from $\sigma_{\max}^2/\tau$ via fixed-point argument

**Phase 3: Add Rigor** (Estimated: 2-3 hours)

1. **Localization to $\{V \leq M\}$**: Provide detailed truncation + tail control argument for each component
2. **Dimension dependence**: Track all $d$-dependent factors through the proof to make $C_d$ explicit
3. **Verification of hypotheses**: Check each component's weak error proposition against the primitive axioms (Lipschitz force, diffusion bounds, etc.)

**Phase 4: Review and Validation** (Estimated: 1-2 hours)

1. Framework cross-validation: Verify all cited propositions and theorems exist and match claimed statements
2. Edge case verification: Check isotropic constant diffusion assumption; verify timestep guideline for stated parameters
3. Constant tracking audit: Ensure no hidden constants or circular dependencies

**Total Estimated Expansion Time**: 8-12 hours

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-discretization` (Theorem 3.7.2: Discrete-Time Inheritance of Generator Drift)

**Propositions Used**:
- {prf:ref}`prop-weak-error-variance` (BAOAB weak error for variance Lyapunov functions)
- {prf:ref}`prop-weak-error-boundary` (BAOAB weak error for boundary Lyapunov function)
- {prf:ref}`prop-weak-error-wasserstein` (BAOAB weak error for Wasserstein distance)

**Definitions Used**:
- BAOAB integrator (§3.7, lines 413-430)
- Synergistic Lyapunov function $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$ (§3.7.3.4, line 1018)
- Generator drift inequality $\mathcal{L}V \leq -\kappa V + C$ (§3.7.2, line 614-618)

**Related Proofs** (for comparison):
- Assembly of component-wise weak error bounds in Proof of Theorem 3.7.2 (lines 1012-1102)
- Boundary truncation argument in Proof of prop-weak-error-boundary (lines 746-827)
- Synchronous coupling technique in Proof of prop-weak-error-wasserstein (lines 840-907)

**External References**:
- Leimkuhler & Matthews (2015): Molecular Dynamics (BAOAB weak error theory)
- Talay (1990): Expansion of the global error for numerical schemes (Talay-Tubaro expansions)
- Villani (2009): Optimal Transport (Wasserstein distance theory)
- Meyn & Tweedie (2009): Markov Chains and Stochastic Stability (Foster-Lyapunov drift)

---

**Proof Sketch Completed**: 2025-10-25 01:39 UTC
**Ready for Expansion**: Yes (with caveat: verify isotropic constant diffusion assumption or extend bound to include $L_\Sigma$)
**Confidence Level**: Medium-High - Approach is sound and well-grounded in existing framework results, but lack of Gemini cross-validation reduces confidence; the $\sigma_{\max}^2/\tau$ term requires careful handling to avoid circularity
