# Proof Sketch for thm-optimal-parameter-scaling

**Document**: docs/source/2_geometric_gas/16_convergence_mean_field.md
**Theorem**: thm-optimal-parameter-scaling
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} Optimal Parameter Scaling
:label: thm-optimal-parameter-scaling

For a landscape with Lipschitz constant $L_U$ and minimum Hessian eigenvalue $\lambda_{\min}$, the optimal parameter scaling is:

$$
\begin{aligned}
\gamma^* &\sim L_U^{3/7} \\
\sigma^* &\sim L_U^{9/14} \\
\tau^* &\sim L_U^{-12/7} \\
\lambda_{\text{revive}}^* &\sim \kappa_{\max}
\end{aligned}
$$

yielding convergence rate:

$$
\alpha_{\text{net}}^* \sim \gamma^* \sim L_U^{3/7}
$$

:::

**Informal Restatement**:

For the mean-field Geometric Gas converging to a quasi-stationary distribution (QSD), there exists an optimal choice of algorithmic parameters that maximizes the exponential convergence rate. The theorem provides the precise scaling laws: as the landscape becomes more complex (larger Lipschitz constant $L_U$), the optimal friction $\gamma$ and diffusion $\sigma$ must increase with specific power laws ($L_U^{3/7}$ and $L_U^{9/14}$ respectively), while the time step $\tau$ must decrease as $L_U^{-12/7}$. These scalings yield an optimal convergence rate that grows as $L_U^{3/7}$, meaning harder landscapes (paradoxically) allow faster convergence when parameters are tuned correctly.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: ⚠️ **GEMINI FAILED TO RESPOND**

Gemini 2.5 Pro returned an empty response despite multiple attempts. This is likely due to a temporary API issue or timeout. The proof sketch proceeds with GPT-5's strategy alone.

**Limitations**:
- No cross-validation from second strategist
- Lower confidence in chosen approach
- Recommend re-running sketch when Gemini is available

---

### Strategy B: GPT-5's Approach

**Method**: Optimization-based proof with scaling balance and KKT conditions

**Key Steps**:
1. Reduce $\alpha_{\text{net}}$ to dominant parametric form using explicit formula
2. State and justify admissible regime constraints (critical diffusion, time step bounds)
3. Build constrained optimization problem with all dependencies
4. Nondimensionalize parameters and extract $L_U$ exponents via scaling ansatz
5. Solve the KKT exponent system to recover exact scalings (3/7, 9/14, -12/7)
6. Optimize $\lambda_{\text{revive}}$ and verify jump term contribution

**Strengths**:
- **Systematic and rigorous**: Uses explicit formulas from framework rather than heuristics
- **Tracks all dependencies**: Accounts for LSI constant, coupling terms, jump expansion
- **Exact exponents**: Derives 3/7, 9/14, -12/7 from KKT conditions, not dimensional analysis
- **Constraint-aware**: Recognizes that optimizers saturate boundary constraints
- **Well-structured**: Clear progression from setup to solution

**Weaknesses**:
- **Algebraically intensive**: Requires solving coupled nonlinear system with multiple terms
- **Dominance arguments needed**: Must prove certain terms are subleading in the optimal regime
- **No numerical verification**: Strategy proposes but doesn't execute symbolic solution
- **Lemmas not fully proven**: Relies on several medium-difficulty lemmas (B, C, D)

**Framework Dependencies**:
- Theorem thm-alpha-net-explicit (16_convergence_mean_field.md § 2.1)
- LSI constant approximation (16_convergence_mean_field.md § 1.2)
- Coupling constants formulas (16_convergence_mean_field.md § 1.3)
- $C_{\nabla x}$ scaling estimate (16_convergence_mean_field.md § 1.1)
- Critical diffusion threshold (16_convergence_mean_field.md § 2.2)
- Jump expansion formula (16_convergence_mean_field.md § 1.4)

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Optimization-based proof with KKT conditions and scaling balance (GPT-5's approach)

**Rationale**:

Given the absence of Gemini's response, I adopt GPT-5's strategy with the following enhancements:

1. **Why optimization-based is correct**: The theorem asks for *optimal* parameter scaling, which naturally frames as a constrained optimization problem. The explicit convergence rate formula provides the objective function.

2. **Why KKT conditions are necessary**: The constraints (critical diffusion $\sigma^4 \gtrsim L_U^3/\gamma$, admissible time step $\tau \lesssim \sigma/\gamma^2$) are inequality constraints. At the optimum, we expect these to be active (saturated), which KKT theory formalizes.

3. **Why scaling ansatz works**: The problem has a natural scaling structure—all parameters have dimensions that can be expressed as powers of $L_U$. Substituting $\gamma = L_U^p \Gamma$, $\sigma = L_U^q \Sigma$, $\tau = L_U^r T$ reduces the problem to finding $(p, q, r)$ that balance the dominant terms.

4. **Key insight enabling the proof**: The convergence rate formula has three competing terms:
   - **LSI benefit**: $\sim \gamma$ (promotes large friction)
   - **Coupling penalty 1**: $\sim \gamma^2 \tau / \sigma$ (penalizes large $\gamma$, $\tau$, rewards large $\sigma$)
   - **Landscape roughness penalty**: $\sim \gamma L_U^3 / \sigma^4$ (penalizes landscape complexity, rewards diffusion)

   At the optimum, these three forces balance via the constraints, yielding the specific exponents.

**Integration**:
- **Steps 1-2**: From GPT-5 (formula reduction, constraint identification)
- **Step 3**: From GPT-5 (optimization problem setup)
- **Step 4**: From GPT-5 (nondimensionalization and exponent extraction)
- **Step 5**: From GPT-5 with **Claude's enhancement**: I will explicitly solve the linear system to verify the exponents
- **Step 6**: From GPT-5 (revival rate optimization)

**Verification Status**:
- ✅ All framework dependencies verified (theorem thm-alpha-net-explicit, LSI formula, coupling formulas)
- ✅ No circular reasoning detected (uses explicit formulas from earlier in same document)
- ⚠ **Requires Lemma B**: Dominance of $C_{\nabla x} \approx \sqrt{L_U/\gamma}$ over additive $\gamma$ term
- ⚠ **Requires Lemma C**: Active constraint for $\sigma$ (critical diffusion bound is saturated)
- ⚠ **Requires Lemma D**: Solution of KKT exponent system yields exact values 3/7, 9/14, -12/7

---

## III. Framework Dependencies

### Verified Dependencies

**Theorems** (from same document):

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| thm-alpha-net-explicit | $\alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - C_{\text{jump}}\right]$ | Step 1, 3 | ✅ (16_convergence_mean_field.md:4449-4472) |

**Formulas** (from same document):

| Symbol/Label | Definition | Used in Step | Verified |
|--------------|------------|--------------|----------|
| $\lambda_{\text{LSI}}$ | $\frac{\gamma}{\sigma^2(1 + \gamma/\lambda_{\min} + \lambda_{\text{revive}}/(M_\infty \gamma))}$ | Step 1, 3 | ✅ (16_convergence_mean_field.md:4344) |
| $C_{\text{Fisher}}^{\text{coup}}$ | $(C_{\nabla x} + \gamma) \sigma\tau\sqrt{2d} + \frac{L_U^3}{2\sigma^2}$ | Step 1, 3, 4 | ✅ (16_convergence_mean_field.md:4376) |
| $C_{\nabla x}$ | $\sqrt{\frac{\kappa_{\max}}{\sigma^2}} + \sqrt{\frac{L_U}{\gamma}}$ | Step 4 | ✅ (16_convergence_mean_field.md:4269) |
| $C_{\text{KL}}^{\text{coup}}$ | $(C_{\nabla x} + \gamma) \sqrt{2d\sigma^2/\gamma}$ | Step 1, 3 | ✅ (16_convergence_mean_field.md:4360) |
| $A_{\text{jump}}$ | $2\kappa_{\max} + \frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^2}$ | Step 6 | ✅ (16_convergence_mean_field.md:4427) |
| $\sigma_{\text{crit}}$ | $\left(\frac{2L_U^3}{\gamma}\right)^{1/4}$ | Step 2, 5 | ✅ (16_convergence_mean_field.md:4487) |
| $\tau_{\text{max}}$ | $\frac{\sigma}{2\gamma^2\sqrt{2d}}$ | Step 2, 5 | ✅ (16_convergence_mean_field.md:4522) |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| $L_U$ | Lipschitz constant of $\nabla U$ | Landscape-dependent | Fixed (input to optimization) |
| $\lambda_{\min}$ | Minimum Hessian eigenvalue of $U$ | Landscape-dependent | Fixed (defines weak-damping regime $\gamma \ll \lambda_{\min}$) |
| $\kappa_{\max}$ | Maximum killing rate | Landscape-dependent | Fixed (appears in $A_{\text{jump}}$) |
| $d$ | Dimension | Problem-dependent | Fixed (appears in $\sqrt{2d}$ factors) |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- **Lemma A (Monotonicity in $\tau$)**: For fixed $(\gamma, \sigma, \lambda_{\text{revive}})$, $\alpha_{\text{net}}$ decreases strictly with $\tau$ in the small-$\tau$ regime - **Difficulty: easy**
  - Why needed: Justifies that $\tau$ optimizer saturates its admissible upper bound
  - Proof sketch: $\frac{\partial \alpha_{\text{net}}}{\partial \tau} \approx -\frac{\gamma^2\sqrt{2d}}{\sigma} < 0$ from explicit formula

- **Lemma B (Dominance of $C_{\nabla x}$)**: In the weak-damping scaling region, $C_{\nabla x} \approx \sqrt{L_U/\gamma}$ dominates the additive $\gamma$ term inside coupling constants - **Difficulty: medium**
  - Why needed: Simplifies the $L_U$ exponent extraction in Step 4
  - Proof sketch: For $\gamma \sim L_U^{3/7}$ (the claimed scaling), $\sqrt{L_U/\gamma} \sim L_U^{1/2 - 3/14} = L_U^{4/14} = L_U^{2/7} \gg \gamma = L_U^{3/7}$ is FALSE! Need to reconsider.
  - ⚠️ **ISSUE FLAGGED**: This dominance claim needs verification. If $\gamma \sim L_U^{3/7}$, then $\sqrt{L_U/\gamma} \sim L_U^{2/7}$ and $\gamma \sim L_U^{3/7}$, so $\gamma$ actually dominates! This affects the derivation.

- **Lemma C (Active $\sigma$-bound)**: At the maximizer, the critical diffusion threshold is saturated - **Difficulty: medium**
  - Why needed: Reduces optimization DOF and provides one KKT equation
  - Proof sketch: Show that $\alpha_{\text{net}}$ has no interior $\sigma$-optimum by analyzing first-order condition; conclude optimizer is on boundary

- **Lemma D (KKT exponent system)**: With Lemmas A–C, the KKT conditions reduce to a linear system yielding $p = 3/7$, $q = 9/14$, $r = -12/7$ - **Difficulty: medium**
  - Why needed: This is the core calculation proving the theorem
  - Proof sketch: Substitute scaling ansatz, balance dominant $L_U$ exponents, solve linear system

**Uncertain Assumptions**:
- **Dominance of specific terms**: The simplified formula drops $C_{\text{KL}}^{\text{coup}}$ from the full expression. Need to verify it's subdominant at the optimizer.
  - Why uncertain: $C_{\text{KL}}^{\text{coup}} = (C_{\nabla x} + \gamma) \sqrt{2d\sigma^2/\gamma}$ could potentially compete with other terms
  - How to verify: Track this term through the scaling analysis and show its $L_U$ exponent is less dominant than the kept terms

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes optimal parameter scaling laws by formulating and solving a constrained optimization problem. The key ingredients are:

1. **Explicit convergence rate formula** (thm-alpha-net-explicit): Provides the objective function $\alpha_{\text{net}}(\tau, \gamma, \sigma, \lambda_{\text{revive}})$ with explicit dependence on all parameters and landscape characteristics.

2. **Physical constraints**: Critical diffusion threshold $\sigma^4 \gtrsim L_U^3/\gamma$ ensures convergence; time step bound $\tau \lesssim \sigma/\gamma^2$ maintains numerical accuracy.

3. **Scaling structure**: All parameters have natural dimensions relative to $L_U$. By assuming power-law scalings $\gamma \sim L_U^p$, $\sigma \sim L_U^q$, $\tau \sim L_U^r$, we reduce the problem to finding the exponents $(p, q, r)$.

4. **KKT conditions**: At the optimal parameter choice, constraints are active (saturated) and first-order optimality holds for free variables. This yields a system of algebraic equations in the exponents.

5. **Linear exponent system**: The KKT conditions, when expressed in logarithmic derivatives, form a linear system whose unique solution is $(p, q, r) = (3/7, 9/14, -12/7)$.

The proof is constructive in the sense that it explicitly determines the optimizer rather than just proving existence.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Formula Reduction**: Identify dominant terms in $\alpha_{\text{net}}$ for the weak-damping, small-$\tau$ regime
2. **Constraint Formulation**: State admissible parameter regime and justify inequality constraints
3. **Optimization Problem Setup**: Formulate constrained maximization with all dependencies explicit
4. **Nondimensionalization**: Introduce scaling ansatz and extract $L_U$ exponents for all terms
5. **KKT Solution**: Solve the exponent system from active constraints and first-order optimality
6. **Revival Optimization**: Determine optimal $\lambda_{\text{revive}}$ and verify overall convergence rate

---

### Detailed Step-by-Step Sketch

#### Step 1: Reduce $\alpha_{\text{net}}$ to Dominant Parametric Form

**Goal**: Identify the dominant terms in the convergence rate formula for the optimization regime

**Substep 1.1**: Start from explicit formula
- **Justification**: Use thm-alpha-net-explicit (16_convergence_mean_field.md:4449)
- **Formula**:
  $$
  \alpha_{\text{net}} \approx \frac{1}{2}\left[\gamma - \frac{2\gamma^2\tau\sqrt{2d}}{\sigma} - \frac{2\gamma L_U^3}{\sigma^4} - 2\kappa_{\max} - C_{\text{jump}}\right]
  $$
- **Why valid**: This is the simplified form for $\tau \ll 1$, $\gamma \ll \lambda_{\min}$, which defines our optimization regime
- **Expected result**: Explicit objective function for optimization

**Substep 1.2**: Identify parameter dependencies
- **Action**: Recognize three types of terms:
  - **T0 (LSI benefit)**: $\gamma$ — linear benefit from friction
  - **Tτ (coupling penalty)**: $-\frac{2\gamma^2\tau\sqrt{2d}}{\sigma}$ — discretization error
  - **TLU (landscape penalty)**: $-\frac{2\gamma L_U^3}{\sigma^4}$ — roughness penalty
  - **Tjump (killing penalty)**: $-2\kappa_{\max} - C_{\text{jump}}$ — boundary effects
- **Justification**: Each term has distinct parametric structure
- **Expected result**: Classification of terms by their role in optimization

**Substep 1.3**: Determine which terms dominate the optimization
- **Action**: Note that:
  - $\kappa_{\max}$ and $C_{\text{jump}}$ are $L_U$-independent constants (fixed by landscape killing)
  - $\gamma$, $\tau$, $\sigma$ are the optimization variables
  - $L_U$ is the scaling parameter we analyze
- **Conclusion**: For $L_U$-scaling analysis, focus on T0, Tτ, TLU terms; Tjump contributes $O(1)$ offset

**Dependencies**:
- Uses: thm-alpha-net-explicit
- Requires: Weak-damping regime $\gamma \ll \lambda_{\min}$, small time step $\tau \ll 1$

**Potential Issues**:
- ⚠ The full formula includes $C_{\text{KL}}^{\text{coup}} = (C_{\nabla x} + \gamma) \sqrt{2d\sigma^2/\gamma}$ which is dropped
- **Resolution**: Must verify this term is subdominant in the optimal regime (part of Lemma B analysis)

---

#### Step 2: State and Justify Admissible Regime Constraints

**Goal**: Formalize the inequality constraints that bound the feasible parameter region

**Substep 2.1**: Critical diffusion threshold
- **Justification**: From Section 2.2 (16_convergence_mean_field.md:4487)
- **Constraint**:
  $$
  \sigma^4 \gtrsim \frac{2L_U^3}{\gamma}
  $$
- **Why valid**: For $\alpha_{\text{net}} > 0$, diffusion must overcome landscape roughness. Below this threshold, the TLU penalty dominates and $\alpha_{\text{net}} < 0$.
- **Expected result**: Lower bound on $\sigma$ relative to $\gamma$ and $L_U$

**Substep 2.2**: Time step admissibility
- **Justification**: From Section 2.3 (16_convergence_mean_field.md:4522)
- **Constraint**:
  $$
  \tau \lesssim \frac{\sigma}{2\gamma^2\sqrt{2d}}
  $$
- **Why valid**: For larger $\tau$, the Tτ coupling penalty grows and time-discretization errors dominate
- **Expected result**: Upper bound on $\tau$ relative to $\gamma$ and $\sigma$

**Substep 2.3**: Weak-damping regime
- **Action**: Assume $\gamma \ll \lambda_{\min}$ (stated in theorem context)
- **Why valid**: This is the regime where the simplified LSI formula $\lambda_{\text{LSI}} \approx \gamma/\sigma^2$ holds
- **Expected result**: Simplification of LSI constant dependence

**Dependencies**:
- Uses: Critical regime analysis (16_convergence_mean_field.md § 2.2)
- Uses: Optimal time step heuristic (16_convergence_mean_field.md § 2.3)

**Potential Issues**:
- ⚠ Constraints are stated as "approximately" ($\gtrsim$, $\lesssim$) rather than sharp equalities
- **Resolution**: In optimization, interpret these as defining the admissible region; at the optimum, expect them to be active (saturated)

---

#### Step 3: Build the Constrained Optimization Problem

**Goal**: Formulate the complete optimization problem with objective, variables, and constraints

**Substep 3.1**: State the optimization problem
- **Objective**: Maximize $\alpha_{\text{net}}(\gamma, \sigma, \tau, \lambda_{\text{revive}})$
- **Variables**: $(\gamma, \sigma, \tau, \lambda_{\text{revive}}) \in \mathbb{R}_+^4$
- **Fixed parameters**: $(L_U, \lambda_{\min}, \kappa_{\max}, d)$
- **Constraints**:
  - $g_1(\gamma, \sigma)$: $\sigma^4 - \frac{2L_U^3}{\gamma} \geq 0$ (critical diffusion)
  - $g_2(\gamma, \sigma, \tau)$: $\frac{\sigma}{2\gamma^2\sqrt{2d}} - \tau \geq 0$ (time step bound)
  - $h_1(\gamma)$: $\gamma \leq \epsilon \lambda_{\min}$ for small $\epsilon$ (weak damping)

**Substep 3.2**: Formulate KKT conditions
- **Action**: Write Lagrangian with multipliers $\mu_1, \mu_2 \geq 0$:
  $$
  \mathcal{L} = \alpha_{\text{net}}(\gamma, \sigma, \tau, \lambda_{\text{revive}}) + \mu_1 g_1(\gamma, \sigma) + \mu_2 g_2(\gamma, \sigma, \tau)
  $$
- **Stationarity**:
  $$
  \begin{aligned}
  \frac{\partial \mathcal{L}}{\partial \gamma} &= 0 \\
  \frac{\partial \mathcal{L}}{\partial \sigma} &= 0 \\
  \frac{\partial \mathcal{L}}{\partial \tau} &= 0 \\
  \frac{\partial \mathcal{L}}{\partial \lambda_{\text{revive}}} &= 0
  \end{aligned}
  $$
- **Complementarity**: $\mu_1 g_1 = 0$, $\mu_2 g_2 = 0$
- **Feasibility**: $g_1, g_2 \geq 0$, $\mu_1, \mu_2 \geq 0$

**Substep 3.3**: Predict active constraints
- **Lemma A**: $\frac{\partial \alpha_{\text{net}}}{\partial \tau} < 0$ (monotone decreasing in $\tau$)
- **Conclusion**: At optimum, $\tau$ saturates its upper bound, so $g_2 = 0$ is active ($\mu_2 > 0$)
- **Lemma C**: No interior $\sigma$-optimum exists
- **Conclusion**: At optimum, $\sigma$ saturates critical diffusion, so $g_1 = 0$ is active ($\mu_1 > 0$)
- **Expected result**: Two active constraints reduce 4-variable problem to 2-variable problem

**Dependencies**:
- Uses: thm-alpha-net-explicit for $\alpha_{\text{net}}$
- Requires: Lemma A (monotonicity in $\tau$)
- Requires: Lemma C (active $\sigma$ constraint)

**Potential Issues**:
- ⚠ KKT theory requires constraint qualification (e.g., LICQ); must verify
- **Resolution**: Constraints $g_1, g_2$ have linearly independent gradients in the interior, so LICQ holds

---

#### Step 4: Nondimensionalization and Exponent Extraction

**Goal**: Introduce scaling ansatz and extract $L_U$ exponents for all terms in $\alpha_{\text{net}}$

**Substep 4.1**: Define scaling ansatz
- **Action**: Assume power-law scalings:
  $$
  \begin{aligned}
  \gamma &= L_U^p \cdot \Gamma \\
  \sigma &= L_U^q \cdot \Sigma \\
  \tau &= L_U^r \cdot T \\
  \lambda_{\text{revive}} &= L_U^s \cdot \Lambda
  \end{aligned}
  $$
  where $(\Gamma, \Sigma, T, \Lambda)$ are $O(1)$ dimensionless constants and $(p, q, r, s)$ are the exponents to determine.
- **Why valid**: All parameters have dimensions that can be expressed relative to $L_U$ (which has dimensions of inverse length × force)
- **Expected result**: Parametrization of solution class

**Substep 4.2**: Substitute into active constraints
- **Constraint 1** ($g_1 = 0$, critical diffusion):
  $$
  \sigma^4 = \frac{2L_U^3}{\gamma} \implies (L_U^q \Sigma)^4 = \frac{2L_U^3}{L_U^p \Gamma}
  $$
  $$
  L_U^{4q} = L_U^{3-p} \implies 4q = 3 - p
  $$
- **Constraint 2** ($g_2 = 0$, time step saturation):
  $$
  \tau = \frac{\sigma}{2\gamma^2\sqrt{2d}} \implies L_U^r T = \frac{L_U^q \Sigma}{2(L_U^p \Gamma)^2\sqrt{2d}}
  $$
  $$
  L_U^r = L_U^{q - 2p} \implies r = q - 2p
  $$
- **Expected result**: Two algebraic relations among exponents
  - **Relation 1**: $4q = 3 - p$
  - **Relation 2**: $r = q - 2p$

**Substep 4.3**: Substitute into $\alpha_{\text{net}}$ and extract exponents
- **T0 term**: $\gamma = L_U^p \Gamma$ contributes $L_U^p$
- **Tτ term**:
  $$
  \frac{\gamma^2\tau}{\sigma} = \frac{(L_U^p)^2 \cdot L_U^r}{L_U^q} = L_U^{2p + r - q}
  $$
  Using $r = q - 2p$: $2p + (q-2p) - q = 0$, so this term is $O(1)$ in $L_U$!

- **TLU term**:
  $$
  \frac{\gamma L_U^3}{\sigma^4} = \frac{L_U^p \cdot L_U^3}{(L_U^q)^4} = L_U^{p + 3 - 4q}
  $$
  Using $4q = 3 - p$: $p + 3 - (3-p) = 2p$, so this term is $L_U^{2p}$

- **Why this matters**: For $\alpha_{\text{net}} \sim L_U^p$ (claimed in theorem), we need the T0 and TLU terms to balance at leading order.

**Substep 4.4**: Balance the dominant terms
- **Observation**: We have
  $$
  \alpha_{\text{net}} \sim \gamma - \frac{2\gamma L_U^3}{\sigma^4} \sim L_U^p - L_U^{2p}
  $$
- **For maximum**: Set $\frac{\partial}{\partial p}[\text{dominant terms}] = 0$
- **But wait**: This doesn't balance! If $p < 2p$, then $L_U^{2p}$ dominates for large $L_U$ and $\alpha_{\text{net}} < 0$. We need another approach.

**Substep 4.5**: Reconsider the balance
- **Key insight**: At the optimum, the terms should balance such that increasing $p$ doesn't help:
  $$
  L_U^p \sim L_U^{2p} \implies p = 2p \text{ (impossible unless } p=0 \text{)}
  $$
- **Alternative**: The optimum occurs when the marginal benefit of increasing $\gamma$ equals the marginal cost. Using the first-order condition:
  $$
  \frac{\partial \alpha_{\text{net}}}{\partial \gamma} = 1 - \frac{2L_U^3}{\sigma^4} = 0
  $$
  This gives $\sigma^4 = 2L_U^3$, but we also have $\sigma^4 = 2L_U^3/\gamma$ from the active constraint.
  So $2L_U^3 = 2L_U^3/\gamma \implies \gamma = 1$, which doesn't give scaling!

**Substep 4.6**: Include the Tτ term properly
- **Correction**: The Tτ term is NOT $O(1)$—I made an error. Let me recalculate:
  $$
  \frac{\gamma^2\tau}{\sigma} = \frac{(L_U^p \Gamma)^2 \cdot L_U^r T}{L_U^q \Sigma} = L_U^{2p+r-q} \frac{\Gamma^2 T}{\Sigma}
  $$
  With $r = q - 2p$:
  $$
  2p + r - q = 2p + (q-2p) - q = 0
  $$
  So it IS $O(1)$, but its coefficient depends on $(\Gamma, \Sigma, T)$.

- **Key realization**: The active constraint $g_2 = 0$ means $T = \Sigma/(2\Gamma^2\sqrt{2d})$, so:
  $$
  \text{Tτ term} = \frac{\Gamma^2}{\Sigma} \cdot \frac{\Sigma}{2\Gamma^2\sqrt{2d}} = \frac{1}{2\sqrt{2d}} = O(1)
  $$
  This is a pure constant, independent of $L_U$ AND of $(\Gamma, \Sigma)$!

**Substep 4.7**: Correct balance equation
- **Action**: Now $\alpha_{\text{net}}$ simplifies to:
  $$
  \alpha_{\text{net}} \sim L_U^p \Gamma - C_1 - L_U^{2p} \frac{\Gamma}{\Sigma^4} \cdot 2L_U^{3-4q}
  $$
  Using $4q = 3-p$:
  $$
  \alpha_{\text{net}} \sim L_U^p \Gamma - C_1 - L_U^{2p} \cdot \text{const}
  $$
  where const = $2\Gamma / \Sigma^4 \cdot L_U^{3-4q} = 2\Gamma/\Sigma^4 \cdot L_U^p$... wait, that's $L_U^{3p}$!

- **Re-re-calculation**: Using $\sigma^4 = 2L_U^3/\gamma$:
  $$
  \frac{\gamma L_U^3}{\sigma^4} = \frac{\gamma L_U^3}{2L_U^3/\gamma} = \frac{\gamma^2}{2}
  $$
  So this term is actually $(\gamma)^2 / 2 = L_U^{2p} \Gamma^2/2$!

**Substep 4.8**: Final balance with all constraints applied
- **After applying both active constraints**:
  $$
  \alpha_{\text{net}} \approx \frac{1}{2}\left[\Gamma L_U^p - C_{\tau} - \frac{\Gamma^2}{1} L_U^{2p} - 2\kappa_{\max}\right]
  $$
  where $C_{\tau}$ is the $O(1)$ Tτ contribution.

- **Maximize over $p$ (and implicitly $\Gamma$)**: Taking derivative with respect to $p$:
  $$
  \frac{\partial}{\partial p}[\Gamma L_U^p - \Gamma^2 L_U^{2p}] = \Gamma L_U^p \ln L_U - 2\Gamma^2 L_U^{2p} \ln L_U = 0
  $$
  $$
  \Gamma L_U^p = 2\Gamma^2 L_U^{2p} \implies 1 = 2\Gamma L_U^p \implies \Gamma = \frac{1}{2 L_U^p}
  $$

- **This is still wrong**: The issue is that $\Gamma$ should be $O(1)$, not scaling with $L_U$.

**Substep 4.9**: Correct approach—include the missing Tτ dependence
- **Realization**: I need to include the **full** coupling term structure. The simplified formula drops $C_{\text{KL}}^{\text{coup}}$, but the full formula has both Fisher and KL coupling. Let me reconsider using the FULL formula before simplification.

**Dependencies**:
- Uses: Active constraints from Step 3
- Requires: Careful algebra and dimensional analysis

**Potential Issues**:
- ⚠️ **CALCULATION ERROR IDENTIFIED**: The scaling analysis above has issues. Need to use the FULL formula, not the simplified one, OR need to derive the simplified formula's regime of validity more carefully.
- **Resolution**: This is the critical technical challenge that Lemma D must resolve. The full calculation requires tracking MORE terms than the simplified formula shows.

---

#### Step 5: Solve the KKT Exponent System

**Goal**: Obtain the exact exponents $(p, q, r) = (3/7, 9/14, -12/7)$ from the KKT conditions

**Substep 5.1**: Use the correct optimization approach
- **Key insight from GPT-5**: The optimal scalings emerge from balancing THREE contributions:
  1. LSI benefit $\sim \lambda_{\text{LSI}} \sigma^2$
  2. Fisher coupling penalty $\sim \lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}}$
  3. Landscape roughness in coupling $\sim L_U^3/\sigma^2$ term

- **From the FULL formula** (before simplification):
  $$
  \alpha_{\text{net}} = \frac{1}{2}(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}})
  $$

**Substep 5.2**: Substitute scalings into each component
- **LSI constant**:
  $$
  \lambda_{\text{LSI}} \approx \frac{\gamma}{\sigma^2} = \frac{L_U^p \Gamma}{L_U^{2q} \Sigma^2} = L_U^{p-2q} \frac{\Gamma}{\Sigma^2}
  $$

- **First term** ($\lambda_{\text{LSI}} \sigma^2$):
  $$
  L_U^{p-2q} \cdot L_U^{2q} = L_U^p
  $$

- **Fisher coupling** (with $C_{\nabla x} \sim \sqrt{L_U/\gamma}$ dominant):
  $$
  C_{\text{Fisher}}^{\text{coup}} \sim \sqrt{L_U/\gamma} \cdot \sigma\tau + \frac{L_U^3}{\sigma^2}
  $$
  $$
  \sim L_U^{1/2 - p/2} \cdot L_U^{q} \cdot L_U^{r} + L_U^{3-2q}
  $$
  $$
  = L_U^{1/2 - p/2 + q + r} + L_U^{3-2q}
  $$

- **Second term** ($2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}}$):
  $$
  L_U^{p-2q} \cdot [L_U^{1/2 - p/2 + q + r} + L_U^{3-2q}]
  $$
  $$
  = L_U^{p-2q + 1/2 - p/2 + q + r} + L_U^{p-2q+3-2q}
  $$
  $$
  = L_U^{p/2 - q + 1/2 + r} + L_U^{p+3-4q}
  $$

**Substep 5.3**: Apply active constraints to reduce exponents
- **From $4q = 3-p$**: $L_U^{p+3-4q} = L_U^{p+p} = L_U^{2p}$
- **From $r = q-2p$**: $L_U^{p/2 - q + 1/2 + r} = L_U^{p/2 - q + 1/2 + q - 2p} = L_U^{1/2 - 3p/2}$

**Substep 5.4**: Require all terms to have commensurate scaling
- **For optimal balance**: The benefit (first term $\sim L_U^p$) and penalties (second term) should scale similarly:
  $$
  L_U^p \sim L_U^{2p} \quad \text{OR} \quad L_U^p \sim L_U^{1/2 - 3p/2}
  $$

- **First balance** $p = 2p$ gives $p = 0$ (trivial, not useful)

- **Second balance** $p = 1/2 - 3p/2$ gives:
  $$
  p + 3p/2 = 1/2 \implies 5p/2 = 1/2 \implies p = 1/5
  $$
  But this doesn't match the claimed $p = 3/7$!

**Substep 5.5**: Three-way balance
- **Alternative**: Require all three terms (benefit, Fisher coupling, landscape roughness) to balance:
  $$
  L_U^p \sim L_U^{1/2 - 3p/2} \sim L_U^{2p}
  $$

- **From first two**: $p = 1/2 - 3p/2 \implies p = 1/5$
- **From first and third**: $p = 2p$ (impossible)
- **From second and third**: $1/2 - 3p/2 = 2p \implies 1/2 = 7p/2 \implies p = 1/7$

- **None of these give $p = 3/7$!**

**Substep 5.6**: ISSUE IDENTIFIED—Need different approach
- **Problem**: The scaling balance approach is not yielding the correct exponents
- **Likely cause**:
  1. Missing important terms in the balance
  2. Incorrect dominance assumptions (e.g., maybe $\gamma$ term in $C_{\nabla x} + \gamma$ is NOT negligible)
  3. Need to use first-order condition in $\Gamma$ (dimensionless coefficient optimization)

- **Resolution needed**: This is exactly what **Lemma D** must provide—the correct derivation of the exponent system.

**Dependencies**:
- Uses: Active constraints $4q = 3-p$, $r = q-2p$
- Requires: **Lemma D** (the actual correct derivation)

**Potential Issues**:
- ✗ **MAJOR ISSUE**: My scaling analysis does not reproduce the claimed exponents $p = 3/7$
- **Resolution**: The actual proof must use a more sophisticated approach. Possibilities:
  1. Optimize over the dimensionless coefficients $(\Gamma, \Sigma, T)$ FIRST, then extract $L_U$ scaling
  2. Include the $C_{\text{KL}}^{\text{coup}}$ term (which I've been ignoring)
  3. Use variational approach rather than dimensional analysis

---

#### Step 6: Optimize $\lambda_{\text{revive}}$ and Verify Overall Rate

**Goal**: Determine optimal revival rate and confirm the final convergence rate scaling

**Substep 6.1**: Analyze jump expansion dependence
- **Justification**: From 16_convergence_mean_field.md:4427
- **Formula**:
  $$
  A_{\text{jump}} = 2\kappa_{\max} + \frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^2}
  $$
- **Why valid**: This captures the KL expansion from killing and revival operators

**Substep 6.2**: Minimize with respect to $\lambda_{\text{revive}}$
- **Action**: Take derivative:
  $$
  \frac{\partial A_{\text{jump}}}{\partial \lambda_{\text{revive}}} = \frac{\partial}{\partial \lambda_{\text{revive}}}\left[\frac{\kappa_0(\lambda_{\text{revive}} + \kappa_0)^2}{\lambda_{\text{revive}}^2}\right]
  $$
  $$
  = \kappa_0 \frac{2(\lambda_{\text{revive}} + \kappa_0) \lambda_{\text{revive}}^2 - (\lambda_{\text{revive}} + \kappa_0)^2 \cdot 2\lambda_{\text{revive}}}{\lambda_{\text{revive}}^4}
  $$
  $$
  = \kappa_0 (\lambda_{\text{revive}} + \kappa_0) \frac{2\lambda_{\text{revive}} - 2(\lambda_{\text{revive}} + \kappa_0)}{\lambda_{\text{revive}}^3}
  $$
  $$
  = -\frac{2\kappa_0^2(\lambda_{\text{revive}} + \kappa_0)}{\lambda_{\text{revive}}^3}
  $$
- **Sign**: Always negative, so $A_{\text{jump}}$ is monotone decreasing in $\lambda_{\text{revive}}$!

**Substep 6.3**: Interpret the result
- **Observation**: Since $\frac{\partial A_{\text{jump}}}{\partial \lambda_{\text{revive}}} < 0$, minimizing $A_{\text{jump}}$ (which maximizes $\alpha_{\text{net}}$) requires $\lambda_{\text{revive}} \to \infty$
- **But**: There may be practical constraints or other dependencies on $\lambda_{\text{revive}}$ through $M_{\infty}$ in the LSI constant
- **From LSI formula**:
  $$
  \lambda_{\text{LSI}} \approx \frac{\gamma}{\sigma^2(1 + \gamma/\lambda_{\min} + \lambda_{\text{revive}}/(M_\infty \gamma))}
  $$
  Increasing $\lambda_{\text{revive}}$ DECREASES $\lambda_{\text{LSI}}$, which DECREASES $\alpha_{\text{net}}$ through the first term!

**Substep 6.4**: Optimal balance for $\lambda_{\text{revive}}$
- **Trade-off**:
  - Increasing $\lambda_{\text{revive}}$ decreases $A_{\text{jump}}$ (good)
  - Increasing $\lambda_{\text{revive}}$ decreases $\lambda_{\text{LSI}}$ (bad)
- **Optimal point**: Balance these effects
- **Heuristic from document**: $\lambda_{\text{revive}}^* \approx \kappa_0$ balances killing and revival
- **Theorem claim**: $\lambda_{\text{revive}}^* \sim \kappa_{\max}$

**Substep 6.5**: L_U independence of jump term
- **Key observation**: Both $\kappa_0$ and $\kappa_{\max}$ are landscape-dependent but NOT $L_U$-dependent
- **Conclusion**: The jump term contributes $O(1)$ to $\alpha_{\text{net}}$, not affecting the $L_U$ scaling
- **Therefore**: Can set $\lambda_{\text{revive}} \sim \kappa_{\max}$ (or $\sim \kappa_0$) without changing the exponents $(p, q, r)$

**Dependencies**:
- Uses: Jump expansion formula (16_convergence_mean_field.md:4427)
- Uses: LSI constant formula (16_convergence_mean_field.md:4344)

**Potential Issues**:
- ⚠ The optimal $\lambda_{\text{revive}}$ requires balancing LSI and jump effects, which is not fully worked out
- **Resolution**: Since it contributes only $O(1)$ to $\alpha_{\text{net}}$, this doesn't affect the main theorem (the exponents)

---

## V. Technical Deep Dives

### Challenge 1: Deriving the Exact Exponents 3/7, 9/14, -12/7

**Why Difficult**:
- Multiple competing terms with different $L_U$ scalings
- Constraints couple the exponents in a nonlinear way
- Dimensional analysis alone doesn't uniquely determine the balance point
- Need to account for optimization over dimensionless coefficients $(\Gamma, \Sigma, T)$

**Proposed Solution** (beyond my current sketch):

The key is to recognize that after substituting the active constraints and the scaling ansatz, $\alpha_{\text{net}}$ becomes a function of $p$ and the dimensionless coefficients:

$$
\alpha_{\text{net}}(p, \Gamma, \Sigma) \sim f(p, \Gamma, \Sigma) \cdot L_U^p
$$

where the function $f$ encodes the balance of terms. The correct approach is:

1. **First**: Optimize over $(\Gamma, \Sigma)$ for fixed $p$ to get $f(p) = \max_{\Gamma,\Sigma} f(p, \Gamma, \Sigma)$
2. **Second**: Optimize over $p$ to find $p^* = \arg\max_p f(p) \cdot L_U^p$

For large $L_U$, this reduces to finding the $p$ that makes $f(p)$ positive and as large as possible, subject to the constraint that all kept terms remain commensurate.

**Detailed approach** (sketch):
- Use the FULL coupling formula including both Fisher and KL terms
- After active constraints: $\sigma^4 = 2L_U^3/\gamma$, $\tau = \sigma/(2\gamma^2)$
- Substitute into $\alpha_{\text{net}}$ to get (after considerable algebra):
  $$
  \alpha_{\text{net}} \sim L_U^p \left[a_1 \Gamma - a_2 \Gamma^{3/2} - a_3 \Gamma^{1/2} L_U^{p/2} - a_4 \Gamma^2 \right]
  $$
  for some positive constants $a_i$.
- Optimize over $\Gamma$ (set derivative to zero)
- The optimal $\Gamma^*$ will depend on $p$ and $L_U$
- Require $\Gamma^*$ to be $O(1)$ (independent of $L_U$) for consistency
- This condition determines $p$

**Expected result**: The requirement that $\Gamma^* = O(1)$ forces specific cancellations that yield $p = 3/7$.

**Alternative Approach**:

Use the fact that at the optimum, the first-order condition must hold:
$$
\frac{\partial \alpha_{\text{net}}}{\partial \gamma} = 0
$$
along with the active constraints. This gives a system of three equations in three unknowns, whose solution (in terms of $L_U$ scalings) determines the exponents.

**Reference**:
The document mentions this scaling (3/7, 9/14, -12/7) in the theorem statement but the DERIVATION is not shown explicitly in the provided context. The full proof likely appears elsewhere in the document or requires a detailed calculation that I cannot complete in this sketch.

---

### Challenge 2: Verifying Dominance of $\sqrt{L_U/\gamma}$ term

**Why Difficult**:
The coupling constant $C_{\nabla x} = \sqrt{\kappa_{\max}/\sigma^2} + \sqrt{L_U/\gamma}$ has two terms, and we need to determine which dominates at the optimal scaling.

**Analysis**:

With the claimed scalings $\gamma \sim L_U^{3/7}$, $\sigma \sim L_U^{9/14}$:

- **First term**:
  $$
  \sqrt{\frac{\kappa_{\max}}{\sigma^2}} \sim \sqrt{\frac{1}{L_U^{9/7}}} = L_U^{-9/14}
  $$
  (assuming $\kappa_{\max}$ is $O(1)$)

- **Second term**:
  $$
  \sqrt{\frac{L_U}{\gamma}} = \sqrt{\frac{L_U}{L_U^{3/7}}} = \sqrt{L_U^{4/7}} = L_U^{2/7}
  $$

**Comparison**:
$L_U^{2/7} \gg L_U^{-9/14}$ for large $L_U$ (since $2/7 = 4/14 > 0 > -9/14$).

**Conclusion**:
✅ The $\sqrt{L_U/\gamma}$ term DOES dominate in $C_{\nabla x}$ at the optimal scaling.

**What about $\gamma$ vs $C_{\nabla x}$?**

- $C_{\nabla x} \sim L_U^{2/7}$
- $\gamma \sim L_U^{3/7}$

Since $3/7 > 2/7$, we have $\gamma \gg C_{\nabla x}$ at the optimal scaling!

**Implication**:
⚠️ **LEMMA B NEEDS REVISION**: The claim that "$C_{\nabla x}$ dominates $\gamma$" appears to be BACKWARDS. At the optimal scaling, $\gamma$ actually dominates inside the coupling terms.

**Resolution**:
Need to redo the scaling analysis keeping BOTH terms $(C_{\nabla x} + \gamma)$ and verify which one matters for the exponent derivation. It may be that BOTH contribute, or that different terms dominate in different coupling constants.

---

### Challenge 3: Handling the $C_{\text{KL}}^{\text{coup}}$ Term

**Why Difficult**:
The simplified formula used in the theorem statement omits the KL coupling term:
$$
C_{\text{KL}}^{\text{coup}} = (C_{\nabla x} + \gamma) \sqrt{2d\sigma^2/\gamma}
$$

We need to verify this is indeed subdominant.

**Scaling Analysis**:

With $\gamma \sim L_U^{3/7}$, $\sigma \sim L_U^{9/14}$, $C_{\nabla x} + \gamma \sim \gamma \sim L_U^{3/7}$:

$$
C_{\text{KL}}^{\text{coup}} \sim L_U^{3/7} \cdot \sqrt{\frac{L_U^{9/7}}{L_U^{3/7}}} = L_U^{3/7} \cdot L_U^{3/7} = L_U^{6/7}
$$

**Comparison with other terms**:
- LSI benefit: $\sim L_U^{3/7}$ (leading)
- Landscape penalty: $\sim L_U^{6/7}$ (same as KL coupling!)
- Tτ penalty: $\sim O(1)$

**Issue**:
✗ The $C_{\text{KL}}^{\text{coup}}$ term scales as $L_U^{6/7}$, which is LARGER than the $L_U^{3/7}$ benefit for large $L_U$!

**Possible resolutions**:
1. The full formula has additional prefactors that make KL coupling smaller than landscape penalty
2. The KL coupling is included in a different way (perhaps in the full derivation of $\alpha_{\text{net}}$)
3. My scaling analysis has an error
4. The theorem statement's formula is an approximation that requires $L_U$ not too large

**What this means for the proof**:
⚠️ **CRITICAL ISSUE**: Need to check the FULL formula for $\alpha_{\text{net}}$ (before simplification) and verify which terms actually dominate at the optimal scaling. The simplified formula may not be valid for extracting the exponents.

---

## VI. Proof Validation Checklist

- [❓] **Logical Completeness**: All steps follow from previous steps — **Partially**, but Step 5 has issues
- [✅] **Hypothesis Usage**: All theorem assumptions used (landscape $L_U$, $\lambda_{\min}$, weak damping, small $\tau$)
- [❌] **Conclusion Derivation**: Claimed exponents 3/7, 9/14, -12/7 NOT fully derived in this sketch
- [✅] **Framework Consistency**: All dependencies verified against source document
- [✅] **No Circular Reasoning**: Uses explicit formulas from same section, no circular dependencies
- [⚠️] **Constant Tracking**: Most constants tracked, but $C_{\text{KL}}^{\text{coup}}$ status unclear
- [❌] **Edge Cases**: Boundary cases not fully addressed
- [⚠️] **Regularity Verified**: Weak-damping regime assumed but not verified to be self-consistent
- [✅] **Measure Theory**: Not applicable (deterministic optimization)

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Direct Numerical Optimization

**Approach**: For specific values of $(L_U, \lambda_{\min}, \kappa_{\max}, d)$, numerically maximize $\alpha_{\text{net}}$ over $(\gamma, \sigma, \tau, \lambda_{\text{revive}})$, then fit power laws to extract exponents.

**Pros**:
- Computationally straightforward
- Verifies the exponents empirically
- Can handle the full formula without approximations

**Cons**:
- Not a proof, only numerical evidence
- Requires many $(L_U, \lambda_{\min}, \kappa_{\max})$ combinations to confirm scaling
- Doesn't provide insight into WHY these exponents emerge

**When to Consider**:
As a validation step after completing the analytical proof, or if the analytical approach proves intractable.

---

### Alternative 2: Perturbation Theory from Simple Limit

**Approach**: Start from a simplified limit (e.g., $L_U \to 0$ or $\tau \to 0$) where the optimal scaling is trivial, then perturb around this limit to find how exponents change.

**Pros**:
- May provide clearer intuition
- Systematic expansion procedure

**Cons**:
- Unclear which limit is the right starting point
- Perturbation series may not converge or may be singular
- Still requires significant calculation

**When to Consider**:
If the direct approach (Chosen Strategy) gets stuck, this provides an alternative angle of attack.

---

### Alternative 3: Variational Principle with Test Functions

**Approach**: Assume a one-parameter family of scalings (e.g., $\gamma \sim L_U^p$), substitute into $\alpha_{\text{net}}$, and find the $p$ that optimizes the leading-order growth.

**Pros**:
- More general than assuming specific functional forms for all parameters
- May reveal whether the optimum is unique

**Cons**:
- Still requires choosing appropriate test function family
- May lead to the same algebra as the direct approach

**When to Consider**:
As a consistency check or if multiple optima are suspected.

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Gap 1: Exact exponent derivation** — **CRITICAL**
   - The scaling analysis in Step 5 does NOT yield the claimed exponents 3/7, 9/14, -12/7
   - Need either: (a) a more careful derivation including ALL terms, (b) numerical verification, or (c) reference to a complete calculation elsewhere in the framework
   - **How critical**: Essential—this is the main claim of the theorem

2. **Gap 2: Role of $C_{\text{KL}}^{\text{coup}}$ term** — **HIGH PRIORITY**
   - The simplified formula omits this term, but scaling analysis suggests it's $O(L_U^{6/7})$, comparable to other terms
   - Need to verify whether: (a) the simplified formula is valid in the optimal regime, or (b) the full formula must be used
   - **How critical**: Affects correctness of the optimization setup

3. **Gap 3: Uniqueness of optimal scaling** — **MEDIUM PRIORITY**
   - Have not verified that the claimed scaling is the UNIQUE optimum
   - Could there be other local optima or boundary solutions?
   - **How critical**: Relevant for completeness but probably not a fundamental issue

4. **Gap 4: Rigorous justification of active constraints** — **MEDIUM PRIORITY**
   - Lemma A (monotonicity in $\tau$) is easy to verify
   - Lemma C (active $\sigma$ constraint) needs a full proof using KKT theory
   - **How critical**: Important for rigor but likely tractable

### Conjectures

1. **Conjecture 1**: The exponents $(3/7, 9/14, -12/7)$ are UNIQUE given the weak-damping regime
   - **Why plausible**: Optimization over a smooth objective with convex constraints typically has a unique maximizer (modulo degeneracies)

2. **Conjecture 2**: The optimal convergence rate $\alpha_{\text{net}}^* \sim L_U^{3/7}$ is the BEST possible for any parameter choice
   - **Why plausible**: The theorem claims these are optimal, implying no other scaling does better

3. **Conjecture 3**: Including the $C_{\text{KL}}^{\text{coup}}$ term yields the SAME exponents (perhaps with different prefactors)
   - **Why plausible**: If the theorem statement is correct, the full and simplified formulas should agree on scaling exponents

### Extensions

1. **Extension 1**: Optimal scaling for the FULL regime (not just weak damping $\gamma \ll \lambda_{\min}$)
   - What are the optimal scalings when $\gamma \sim \lambda_{\min}$?
   - Does the LSI constant formula change significantly?

2. **Extension 2**: Dimension dependence of the scaling
   - The factors $\sqrt{2d}$ appear in several terms
   - How do the exponents depend on $d$ (if at all)?

3. **Extension 3**: Finite-$N$ corrections to the optimal scaling
   - The current theorem is for the mean-field limit
   - Do the exponents change for finite $N$?

---

## IX. Expansion Roadmap

**Phase 1: Prove Missing Lemmas** (Estimated: 6-8 hours)
1. **Lemma A (Monotonicity in $\tau$)**:
   - Compute $\frac{\partial \alpha_{\text{net}}}{\partial \tau}$ from explicit formula
   - Verify sign is negative in the regime of interest
   - Estimated: 1 hour (straightforward differentiation)

2. **Lemma B (Dominance of $\sqrt{L_U/\gamma}$)**:
   - With optimal scalings, compute both terms in $C_{\nabla x}$
   - **REVISION NEEDED**: Current analysis shows $\gamma$ dominates $C_{\nabla x}$, opposite of claimed
   - Reconcile with GPT-5's strategy or revise the approach
   - Estimated: 2-3 hours (requires careful re-examination)

3. **Lemma C (Active $\sigma$ constraint)**:
   - Assume interior optimum exists: $\frac{\partial \alpha_{\text{net}}}{\partial \sigma} = 0$
   - Show this leads to contradiction or non-positive convergence rate
   - Conclude constraint must be active
   - Estimated: 2-3 hours (KKT analysis)

4. **Lemma D (KKT exponent system)**:
   - Formulate the system of equations from active constraints and first-order conditions
   - Solve for $(p, q, r)$ algebraically or symbolically (e.g., using Mathematica/SymPy)
   - Verify solution is $(3/7, 9/14, -12/7)$
   - Estimated: 3-4 hours (algebraically intensive, possibly requires symbolic software)

**Phase 2: Fill Technical Details** (Estimated: 8-10 hours)
1. **Step 4: Complete nondimensionalization**
   - Systematically substitute scaling ansatz into FULL formula (not simplified)
   - Track all $L_U$ exponents including $C_{\text{KL}}^{\text{coup}}$ term
   - Verify which terms are dominant, subdominant, negligible
   - Estimated: 3-4 hours

2. **Step 5: Complete exponent extraction**
   - Using results from Lemma D, explicitly solve the linear system
   - Show the solution is unique and matches theorem claim
   - Estimated: 2-3 hours

3. **Step 6: Verify $\lambda_{\text{revive}}$ optimization**
   - Include LSI dependence on $\lambda_{\text{revive}}$ through $M_{\infty}$ term
   - Find the true optimum balancing jump and LSI effects
   - Confirm $\lambda_{\text{revive}}^* \sim \kappa_{\max}$ or $\sim \kappa_0$
   - Estimated: 2-3 hours

**Phase 3: Add Rigor** (Estimated: 4-6 hours)
1. **Verify KKT conditions**:
   - Check constraint qualification (LICQ or similar)
   - Verify second-order sufficient conditions for local maximum
   - Rule out boundary optima (e.g., $\gamma \to 0$ or $\gamma \to \lambda_{\min}$)
   - Estimated: 2-3 hours

2. **Uniqueness proof**:
   - Show the KKT system has a unique solution
   - Verify no other critical points exist
   - Estimated: 1-2 hours

3. **Regime of validity**:
   - Verify the weak-damping assumption $\gamma^* \ll \lambda_{\min}$ is self-consistent with the optimal scaling
   - Verify the small time step assumption $\tau^* \ll 1$ is self-consistent
   - Estimated: 1-2 hours

**Phase 4: Review and Validation** (Estimated: 3-4 hours)
1. **Framework cross-validation**:
   - Re-check all cited formulas against source document
   - Verify no dependencies on unproven results
   - Estimated: 1 hour

2. **Numerical verification** (optional but recommended):
   - Implement numerical optimization of $\alpha_{\text{net}}$ for various $L_U$ values
   - Fit power laws to extracted optimal parameters
   - Confirm exponents match 3/7, 9/14, -12/7
   - Estimated: 2-3 hours (coding + analysis)

3. **Constant tracking audit**:
   - Verify all $O(1)$ claims are justified
   - Track prefactors to ensure they don't introduce hidden $L_U$ dependence
   - Estimated: 1 hour

**Total Estimated Expansion Time**: 21-28 hours

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-alpha-net-explicit` (Mean-Field Convergence Rate - Explicit)

**Formulas Used**:
- LSI constant approximation (16_convergence_mean_field.md § 1.2, equation 4344)
- Coupling constants (16_convergence_mean_field.md § 1.3, equations 4360, 4376)
- $C_{\nabla x}$ scaling (16_convergence_mean_field.md § 1.1, equation 4269)
- Jump expansion (16_convergence_mean_field.md § 1.4, equation 4427)
- Critical diffusion threshold (16_convergence_mean_field.md § 2.2, equation 4487)
- Time step bound (16_convergence_mean_field.md § 2.3, equation 4522)

**Definitions Used**:
- Mean-field convergence rate $\alpha_{\text{net}}$ (16_convergence_mean_field.md § 0)
- Quasi-stationary distribution (QSD) $\rho_{\infty}$ (16_convergence_mean_field.md § 0)
- Parameters: $\gamma$ (friction), $\sigma$ (diffusion), $\tau$ (time step), $\lambda_{\text{revive}}$ (revival rate)
- Landscape characteristics: $L_U$ (Lipschitz constant), $\lambda_{\min}$ (minimum Hessian eigenvalue), $\kappa_{\max}$ (maximum killing rate)

**Related Proofs** (for comparison):
- Similar optimization in Section 4 (Parameter Tuning Strategies) provides heuristic recipes
- Section 3 (Parameter Sensitivity Analysis) provides first-order sensitivity bounds

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: ⚠️ **Needs Lemma D (critical)** — The core calculation yielding the exponents must be completed
**Confidence Level**: **Medium-Low** — GPT-5's strategy is sound, but my attempt to execute Step 5 (exponent extraction) did not reproduce the claimed values. This indicates either:
1. Missing technical details in the scaling analysis
2. Need to use the FULL formula (not simplified)
3. Requirement for symbolic calculation beyond manual algebra
4. Possible error in my interpretation of the scaling approach

**Recommendation**: Proceed with numerical verification (Phase 4.2) to empirically confirm the exponents, then work backwards to identify the correct algebraic pathway in Lemma D.
