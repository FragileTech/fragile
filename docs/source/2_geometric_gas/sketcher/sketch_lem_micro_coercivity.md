# Proof Sketch: Microscopic Coercivity (Step A)

**Label:** `lem-micro-coercivity`

**Source:** `/home/guillem/fragile/docs/source/2_geometric_gas/11_geometric_gas.md` (line 2304)

**Status:** DRAFT - Awaiting review from GPT-5 (Codex)

**Note on Review Protocol:** This sketch uses single-strategist review (GPT-5 only) due to Gemini MCP issues. This provides lower confidence than the standard dual-review protocol. Human verification is recommended.

---

## I. Theorem Statement

:::{prf:lemma} Microscopic Coercivity (Step A)
:label: lem-micro-coercivity

There exists $\lambda_{\text{mic}} > 0$ such that:

$$
D_{\text{kin}}(h \cdot \rho_{\text{QSD}}) \ge \lambda_{\text{mic}} \|(I - \Pi) h\|^2_{L^2(\rho_{\text{QSD}})}
$$

where:
- $D_{\text{kin}}(f) = \int f \|\nabla_v \log(f / \rho_{\text{QSD}})\|^2_{G_{\text{reg}}} \, dx \, dv$ is the kinetic dissipation
- $\Pi h(x) := \int_{\mathbb{R}^d} h(x, v) \rho_{\text{QSD}}(v | x) \, dv$ is the hydrodynamic projection
- $(I - \Pi) h := h - \Pi h$ is the microscopic fluctuation
- $\rho_{\text{QSD}}(v | x)$ is the conditional QSD measure on velocities given position $x$
- $G_{\text{reg}}(x, S) = (H(x, S) + \epsilon_\Sigma I)^{-1}$ is the regularized Riemannian metric
:::

**Informal Restatement:** This lemma establishes that the kinetic dissipation (arising from velocity diffusion with the regularized metric $G_{\text{reg}}$) provides coercivity on the microscopic (velocity-dependent) fluctuations of any function $h$. The key claim is that velocity diffusion alone controls the $L^2$ norm of the velocity-dependent part of $h$, provided we measure with respect to the QSD $\rho_{\text{QSD}}$.

**Physical Interpretation:** In the context of the Geometric Gas dynamics, the kinetic operator represents the Langevin velocity diffusion with anisotropic noise determined by $G_{\text{reg}}$. This lemma says that the velocity randomization is strong enough to dissipate velocity fluctuations at an exponential rate, independent of the position variable. This is the "microscopic" part of hypocoercivity: velocity dissipation acts only in the $v$-directions, but it acts uniformly over all positions $x$.

**Role in Hypocoercivity:** This is Step A in the three-step hypocoercivity framework:
- **Step A (THIS LEMMA):** Velocity dissipation controls microscopic fluctuations
- **Step B (lem-macro-transport):** Transport couples macroscopic and microscopic scales
- **Step C (lem-micro-reg):** Cross-term is controlled by velocity dissipation

Together, these three lemmas yield the full LSI for the mean-field generator.

---

## II. Dependencies and Context

### Framework Dependencies

1. **QSD Existence and Regularity:**
   - Theorem `thm-qsd-existence` (11_geometric_gas.md, line 2106): The quasi-stationary distribution $\rho_{\text{QSD}}$ exists, is $C^2$ smooth, and has exponential concentration.
   - This guarantees that the conditional measure $\rho_{\text{QSD}}(v | x)$ is well-defined for each $x$.

2. **Uniform Ellipticity of Regularized Metric:**
   - Axiom/Definition from Chapter 3 (11_geometric_gas.md): The regularized metric satisfies uniform bounds:
     $$
     c_{\min}(\rho) I \preceq G_{\text{reg}}(x, S) \preceq c_{\max}(\rho) I
     $$
   - This ensures the velocity diffusion is uniformly elliptic with explicit N-uniform, $\rho$-dependent constants.

3. **Microlocal Decomposition:**
   - Definition `def-microlocal` (11_geometric_gas.md, line 2290): The projection $\Pi$ and fluctuation operator $(I - \Pi)$ decompose functions into macroscopic and microscopic parts.

4. **Kinetic Dissipation Definition:**
   - Lemma `lem-dissipation-decomp` (11_geometric_gas.md, line 2271): The kinetic dissipation is defined as:
     $$
     D_{\text{kin}}(f) = \int f \|\nabla_v \log(f / \rho_{\text{QSD}})\|^2_{G_{\text{reg}}} \, dx \, dv
     $$

### Related Results from Euclidean Gas

The Euclidean Gas framework (Chapter 1) establishes analogous results for the isotropic case where $G_{\text{reg}} = \sigma^2 I$:
- **Poincaré inequality for Gaussian velocity:** In 09_kl_convergence.md, the standard Poincaré inequality for Gaussian measures gives velocity coercivity with constant $\gamma / \sigma_v^2$.
- The Geometric Gas extends this to the anisotropic setting where the effective "temperature" varies with position and swarm state via $G_{\text{reg}}$.

---

## III. Proof Strategy

The proof proceeds in four steps:

### Step 1: Conditional Disintegration of Kinetic Dissipation

**Goal:** Decompose $D_{\text{kin}}(h \cdot \rho_{\text{QSD}})$ into an integral over positions of conditional velocity dissipations.

**Approach:**
- Write $f = h \cdot \rho_{\text{QSD}}$ and use the fact that $\rho_{\text{QSD}}(x, v) = \rho_{\text{QSD}}^x(x) \cdot \rho_{\text{QSD}}(v | x)$ where $\rho_{\text{QSD}}^x$ is the marginal on positions.
- Compute:
  $$
  \nabla_v \log(f / \rho_{\text{QSD}}) = \nabla_v \log h
  $$
  since $\rho_{\text{QSD}}$ appears in both numerator and denominator.
- Then:
  $$
  D_{\text{kin}}(h \cdot \rho_{\text{QSD}}) = \int_{\mathcal{X}} \rho_{\text{QSD}}^x(x) \left[ \int_{\mathbb{R}^d} h(x, v) \rho_{\text{QSD}}(v | x) \|\nabla_v \log h\|^2_{G_{\text{reg}}(x)} \, dv \right] dx
  $$
- Define the **conditional kinetic dissipation** at position $x$:
  $$
  D_{\text{kin}}^x(h) := \int_{\mathbb{R}^d} h(x, v) \rho_{\text{QSD}}(v | x) \|\nabla_v \log h\|^2_{G_{\text{reg}}(x)} \, dv
  $$
- Then $D_{\text{kin}}(h \cdot \rho_{\text{QSD}}) = \int_{\mathcal{X}} \rho_{\text{QSD}}^x(x) D_{\text{kin}}^x(h) \, dx$.

**Technical Subtlety:**
- We need to verify that $G_{\text{reg}}(x)$ can be treated as position-dependent only (not velocity-dependent). In the mean-field limit, $G_{\text{reg}}(x, S)$ depends on the swarm state $S$, but for a fixed QSD $\rho_{\text{QSD}}$, we can treat it as a deterministic function of $x$ by replacing $S$ with the mean-field swarm configuration. This is justified by the QSD regularity results.

### Step 2: Poincaré Inequality for Conditional Velocity Measures

**Goal:** Establish a Poincaré inequality for the conditional measure $\rho_{\text{QSD}}(v | x)$ with respect to the metric $G_{\text{reg}}(x)$.

**Key Claim:**
For each fixed $x \in \mathcal{X}$, there exists $\lambda_{\text{Poin}}(x) > 0$ such that for all functions $g(v)$ with $\int_{\mathbb{R}^d} g(v) \rho_{\text{QSD}}(v | x) \, dv = 0$:

$$
\int_{\mathbb{R}^d} g(v)^2 \rho_{\text{QSD}}(v | x) \, dv \le \frac{1}{\lambda_{\text{Poin}}(x)} \int_{\mathbb{R}^d} \|\nabla_v g\|^2_{G_{\text{reg}}(x)} \rho_{\text{QSD}}(v | x) \, dv
$$

**Approach:**
- The conditional measure $\rho_{\text{QSD}}(v | x)$ arises from the velocity dynamics at position $x$, which is driven by the kinetic operator with anisotropic diffusion $G_{\text{reg}}(x)$.
- We need to establish that $\rho_{\text{QSD}}(v | x)$ satisfies exponential concentration in the metric $G_{\text{reg}}(x)$.

**Two Possible Routes:**

**Route A (Spectral Gap for Ornstein-Uhlenbeck):**
- If $\rho_{\text{QSD}}(v | x)$ is close to a Gaussian measure (which is expected from the Langevin dynamics), we can invoke the spectral gap for the Ornstein-Uhlenbeck operator with metric $G_{\text{reg}}(x)$.
- The spectral gap for an OU process with friction $\gamma$ and covariance $G_{\text{reg}}(x)$ is:
  $$
  \lambda_{\text{Poin}}(x) = \gamma \cdot \lambda_{\min}(G_{\text{reg}}(x)^{-1})
  $$
- By uniform ellipticity ($c_{\min}(\rho) I \preceq G_{\text{reg}} \preceq c_{\max}(\rho) I$), we have:
  $$
  \lambda_{\min}(G_{\text{reg}}(x)^{-1}) \ge \frac{1}{c_{\max}(\rho)}
  $$
- Thus $\lambda_{\text{Poin}}(x) \ge \gamma / c_{\max}(\rho) =: \lambda_{\text{mic}}$ uniformly in $x$.

**Route B (Bakry-Émery Theory):**
- Use the Bakry-Émery criterion: if the velocity dynamics satisfy a curvature-dimension condition $\text{CD}(\kappa, \infty)$ with $\kappa > 0$, then the Poincaré constant is bounded by $1/\kappa$.
- For the Geometric Gas, the velocity dynamics with metric $G_{\text{reg}}$ should satisfy such a condition due to the ellipticity of $G_{\text{reg}}$ and the friction term.
- This requires verifying the Bakry-Émery $\Gamma_2$ inequality for the conditional velocity operator at each $x$.

**Recommended Route:** Route A is more direct and relies only on the known spectral gap for OU processes. Route B would provide a deeper geometric understanding but requires additional regularity assumptions on $G_{\text{reg}}(x)$ as a function of $x$.

**Required Lemma:**

:::{prf:lemma} Uniform Velocity Poincaré Constant
:label: lem-uniform-velocity-poincare

For each $x \in \mathcal{X}$, the conditional velocity measure $\rho_{\text{QSD}}(v | x)$ satisfies a Poincaré inequality with respect to the metric $G_{\text{reg}}(x)$, with constant:

$$
\lambda_{\text{Poin}}(x) \ge \lambda_{\text{mic}} := \frac{\gamma}{c_{\max}(\rho)}
$$

where $\gamma$ is the friction coefficient and $c_{\max}(\rho)$ is the uniform upper bound on $G_{\text{reg}}$.
:::

**Proof Sketch of Required Lemma:**
1. The velocity dynamics at position $x$ are governed by the SDE:
   $$
   dv_t = -\gamma v_t \, dt + \sqrt{2} G_{\text{reg}}(x)^{1/2} dW_t
   $$
2. This is an Ornstein-Uhlenbeck process with stationary distribution proportional to:
   $$
   \exp\left( -\frac{1}{2} v^T G_{\text{reg}}(x)^{-1} v \right)
   $$
3. The spectral gap of the OU generator is:
   $$
   \lambda_{\text{OU}} = \gamma \cdot \lambda_{\min}(G_{\text{reg}}(x)^{-1}) \ge \gamma \cdot \frac{1}{c_{\max}(\rho)}
   $$
4. By the variational characterization of the spectral gap, this gives the Poincaré constant.

### Step 3: Apply Conditional Poincaré to $(I - \Pi) h$

**Goal:** Use the Poincaré inequality at each $x$ to control the $L^2$ norm of the microscopic fluctuation.

**Approach:**
- For any function $h$, the microscopic fluctuation $(I - \Pi) h$ satisfies:
  $$
  \int_{\mathbb{R}^d} (I - \Pi) h(x, v) \rho_{\text{QSD}}(v | x) \, dv = 0
  $$
  by definition of $\Pi$.
- Apply the Poincaré inequality (Step 2) to $g(v) = (I - \Pi) h(x, v)$ at each fixed $x$:
  $$
  \int_{\mathbb{R}^d} |(I - \Pi) h(x, v)|^2 \rho_{\text{QSD}}(v | x) \, dv \le \frac{1}{\lambda_{\text{mic}}} \int_{\mathbb{R}^d} \|\nabla_v [(I - \Pi) h]\|^2_{G_{\text{reg}}(x)} \rho_{\text{QSD}}(v | x) \, dv
  $$
- Since $\Pi h$ does not depend on $v$, we have $\nabla_v [(I - \Pi) h] = \nabla_v h$.
- Thus:
  $$
  \int_{\mathbb{R}^d} |(I - \Pi) h(x, v)|^2 \rho_{\text{QSD}}(v | x) \, dv \le \frac{1}{\lambda_{\text{mic}}} \int_{\mathbb{R}^d} \|\nabla_v h\|^2_{G_{\text{reg}}(x)} \rho_{\text{QSD}}(v | x) \, dv
  $$

**Technical Note:**
- We need to handle the fact that the Poincaré inequality is stated for mean-zero functions, but $(I - \Pi) h$ is not globally mean-zero—it is mean-zero only when integrating over $v$ at fixed $x$. This is exactly what we need.

### Step 4: Integrate Over Positions and Compare with Kinetic Dissipation

**Goal:** Integrate the conditional Poincaré inequality over $x$ and relate to $D_{\text{kin}}$.

**Approach:**
- Integrate the inequality from Step 3 over $x$ with respect to $\rho_{\text{QSD}}^x(x)$:
  $$
  \int_{\mathcal{X}} \rho_{\text{QSD}}^x(x) \left[ \int_{\mathbb{R}^d} |(I - \Pi) h(x, v)|^2 \rho_{\text{QSD}}(v | x) \, dv \right] dx \le \frac{1}{\lambda_{\text{mic}}} \int_{\mathcal{X}} \rho_{\text{QSD}}^x(x) \left[ \int_{\mathbb{R}^d} \|\nabla_v h\|^2_{G_{\text{reg}}(x)} \rho_{\text{QSD}}(v | x) \, dv \right] dx
  $$
- The left-hand side is:
  $$
  \int_{\mathcal{X} \times \mathbb{R}^d} |(I - \Pi) h|^2 \rho_{\text{QSD}}(x, v) \, dx \, dv = \|(I - \Pi) h\|^2_{L^2(\rho_{\text{QSD}})}
  $$
- For the right-hand side, we need to relate $\|\nabla_v h\|^2$ to the kinetic dissipation.
- Recall from Step 1:
  $$
  D_{\text{kin}}(h \cdot \rho_{\text{QSD}}) = \int_{\mathcal{X}} \rho_{\text{QSD}}^x(x) \left[ \int_{\mathbb{R}^d} h(x, v) \rho_{\text{QSD}}(v | x) \|\nabla_v \log h\|^2_{G_{\text{reg}}(x)} \, dv \right] dx
  $$
- We need to bound $\|\nabla_v \log h\|^2$ in terms of $\|\nabla_v h\|^2 / h^2$ (standard relationship).

**Key Inequality:**
- By Cauchy-Schwarz or integration by parts (depending on regularity assumptions):
  $$
  \int h \|\nabla_v \log h\|^2 \rho_{\text{QSD}}(v | x) \, dv = \int \frac{|\nabla_v h|^2}{h} \rho_{\text{QSD}}(v | x) \, dv \ge C \int |\nabla_v h|^2 \rho_{\text{QSD}}(v | x) \, dv
  $$
  under appropriate bounds on $h$ (e.g., $h \ge c > 0$ or $h$ bounded).
- Alternatively, use the fact that for functions $h$ close to 1 (perturbative regime), $\log h \approx h - 1$, so $\nabla_v \log h \approx \nabla_v h$.

**Potential Issue:**
- The factor of $h$ in the integrand of $D_{\text{kin}}$ complicates the comparison. This is where the lemma might require additional assumptions on $h$ (e.g., boundedness or regularity).
- A standard approach is to use the **Holley-Stroock perturbation lemma** or work in the perturbative regime where $h = 1 + \varepsilon g$ for small $\varepsilon$.

**Resolution Strategy:**
- For the LSI proof, it suffices to work with $h = f / \rho_{\text{QSD}}$ where $f$ is a probability density. In this regime, $h$ is positive and integrates to 1, which provides the necessary control.
- Alternatively, use weighted Sobolev space techniques to handle the $h$-weighting rigorously.

**Final Comparison:**
- Combining the above, we obtain:
  $$
  \|(I - \Pi) h\|^2_{L^2(\rho_{\text{QSD}})} \le \frac{1}{\lambda_{\text{mic}}} \cdot C \cdot D_{\text{kin}}(h \cdot \rho_{\text{QSD}})
  $$
- This yields the desired coercivity inequality with $\lambda_{\text{mic}}$ replaced by $\lambda_{\text{mic}} / C$ (absorbing the constant into the definition).

---

## IV. Technical Challenges and Resolutions

### Challenge 1: Regularity of Conditional Measure $\rho_{\text{QSD}}(v | x)$

**Issue:** We need to ensure that $\rho_{\text{QSD}}(v | x)$ is well-defined and sufficiently regular (e.g., smooth, exponentially decaying) for the Poincaré inequality to hold uniformly in $x$.

**Resolution:**
- The QSD existence theorem (`thm-qsd-existence`) guarantees that $\rho_{\text{QSD}}$ is $C^2$ smooth in the joint $(x, v)$ space.
- By standard conditional measure theory, the conditional $\rho_{\text{QSD}}(v | x)$ inherits smoothness from the joint measure.
- The exponential concentration of $\rho_{\text{QSD}}$ ensures that $\rho_{\text{QSD}}(v | x)$ has exponential tails, which is sufficient for the Poincaré inequality.

**Required Regularity Lemma:**

:::{prf:lemma} Regularity of Conditional QSD
:label: lem-conditional-qsd-regularity

For each $x \in \mathcal{X}$ in the interior of the domain, the conditional measure $\rho_{\text{QSD}}(v | x)$ is a smooth probability measure on $\mathbb{R}^d$ with exponential tails:

$$
\rho_{\text{QSD}}(v | x) \le C(x) \exp\left( -c(x) |v|^2 \right)
$$

for some position-dependent constants $C(x), c(x) > 0$ that are uniformly bounded away from zero on compact subsets of $\mathcal{X}$.
:::

### Challenge 2: Position-Dependence of $G_{\text{reg}}(x)$

**Issue:** The metric $G_{\text{reg}}(x)$ varies with position, so the Poincaré constant $\lambda_{\text{Poin}}(x)$ is position-dependent. We need a uniform lower bound.

**Resolution:**
- The uniform ellipticity axiom provides:
  $$
  c_{\min}(\rho) I \preceq G_{\text{reg}}(x, S) \preceq c_{\max}(\rho) I
  $$
  uniformly in $x$ and $S$.
- This immediately gives:
  $$
  \lambda_{\text{Poin}}(x) \ge \frac{\gamma}{c_{\max}(\rho)} =: \lambda_{\text{mic}}
  $$
  uniformly in $x$.
- The $\rho$-dependence of the constants is tracked explicitly throughout the framework.

### Challenge 3: Handling the $h$-Weighting in $D_{\text{kin}}$

**Issue:** The kinetic dissipation involves $\int h \|\nabla_v \log h\|^2$ rather than $\int \|\nabla_v h\|^2$, which complicates the comparison with the Poincaré inequality.

**Resolution:**
- **Option A (Perturbative Regime):** Work with $h = 1 + \varepsilon g$ for small $\varepsilon$, so that $\nabla_v \log h = \nabla_v g / (1 + \varepsilon g) \approx \nabla_v g$ to leading order.
- **Option B (Fisher Information Identity):** Use the identity:
  $$
  \int h \|\nabla_v \log h\|^2 \rho \, dv = \int \frac{|\nabla_v h|^2}{h} \rho \, dv = 4 \int \|\nabla_v \sqrt{h}\|^2 \rho \, dv
  $$
  and work with $\sqrt{h}$ instead of $h$.
- **Option C (Weighted Sobolev Spaces):** Define the appropriate weighted Sobolev space where the Poincaré inequality holds with the $h$-weighting naturally incorporated.

**Recommended Approach:** Option B (Fisher Information Identity) is standard in LSI theory and avoids perturbative assumptions.

### Challenge 4: Justifying the Mean-Field Approximation of $G_{\text{reg}}$

**Issue:** In the mean-field limit, $G_{\text{reg}}(x, S)$ depends on the swarm state $S$. For a fixed QSD, we need to replace $S$ with a deterministic mean-field configuration.

**Resolution:**
- The QSD $\rho_{\text{QSD}}$ is the unique stationary measure of the mean-field dynamics, so we can define $G_{\text{reg}}(x) := G_{\text{reg}}(x, S_{\text{QSD}})$ where $S_{\text{QSD}}$ is the swarm state corresponding to sampling from $\rho_{\text{QSD}}$.
- By the law of large numbers (in the mean-field limit $N \to \infty$), the empirical swarm state concentrates around this deterministic configuration.
- This justification is part of the propagation of chaos framework (Chapter 1, docs 07_mean_field.md and 08_propagation_chaos.md).

---

## V. Required Technical Lemmas

To complete the proof rigorously, the following auxiliary lemmas must be established:

1. **Lemma `lem-uniform-velocity-poincare` (Uniform Velocity Poincaré Constant):**
   - Stated in Step 2 above.
   - Difficulty: MEDIUM
   - Expansion estimate: 4-6 hours
   - Approach: Standard spectral gap theory for OU processes with anisotropic diffusion.

2. **Lemma `lem-conditional-qsd-regularity` (Regularity of Conditional QSD):**
   - Stated in Challenge 1 above.
   - Difficulty: MEDIUM
   - Expansion estimate: 4-6 hours
   - Approach: Conditional measure theory + QSD regularity results from `thm-qsd-existence`.

3. **Lemma `lem-fisher-sqrt-comparison` (Fisher Information for $\sqrt{h}$):**
   - Statement: For probability densities $f = h \cdot \rho_{\text{QSD}}$:
     $$
     D_{\text{kin}}(f) = 4 \int \|\nabla_v \sqrt{h}\|^2 \rho_{\text{QSD}} \, dx \, dv
     $$
   - Difficulty: LOW
   - Expansion estimate: 1-2 hours
   - Approach: Standard Fisher information identity (integration by parts).

4. **Lemma `lem-mean-field-metric-deterministic` (Mean-Field Metric Approximation):**
   - Statement: In the QSD regime, $G_{\text{reg}}(x, S)$ can be replaced by a deterministic function $G_{\text{reg}}(x)$ with error $O(1/N)$.
   - Difficulty: MEDIUM
   - Expansion estimate: 3-5 hours
   - Approach: Propagation of chaos + concentration of empirical swarm state.

---

## VI. Difficulty Assessment

**Overall Difficulty:** MEDIUM

**Justification:**
- The core idea is standard in hypocoercivity theory (conditional Poincaré inequality for velocity).
- The main novelty is adapting to the anisotropic metric $G_{\text{reg}}(x)$ and handling the mean-field limit.
- The required technical lemmas are all tractable using existing tools in the framework.

**Breakdown by Component:**
- Step 1 (Conditional Disintegration): LOW difficulty - straightforward measure theory
- Step 2 (Poincaré Inequality): MEDIUM difficulty - requires spectral gap theory
- Step 3 (Apply to Fluctuation): LOW difficulty - direct application
- Step 4 (Integration and Comparison): MEDIUM difficulty - requires careful handling of $h$-weighting

**Risk Factors:**
- **Medium Risk:** Verifying uniform Poincaré constant across all positions $x$ (depends on regularity of $G_{\text{reg}}(x)$).
- **Low Risk:** Handling the $h$-weighting (standard techniques available).

---

## VII. Expansion Time Estimate

**Estimated Time to Full Proof:** 12-16 hours

**Breakdown:**
1. **Preliminary Setup (2 hours):**
   - Formalize notation and state all assumptions clearly.
   - Verify dependencies on QSD regularity and uniform ellipticity.

2. **Step 1 - Conditional Disintegration (2 hours):**
   - Formal measure-theoretic setup of conditional measures.
   - Verify that disintegration formula holds with proper regularity.

3. **Step 2 - Poincaré Inequality (4-6 hours):**
   - Prove `lem-uniform-velocity-poincare` using OU spectral gap.
   - Verify uniform ellipticity bounds propagate correctly.
   - Address potential regularity issues in conditional measure.

4. **Step 3 - Apply to Fluctuation (2 hours):**
   - Formal verification that $(I - \Pi) h$ satisfies mean-zero condition at each $x$.
   - Apply Poincaré inequality conditionally.

5. **Step 4 - Integration and Comparison (4-6 hours):**
   - Establish Fisher information identity (`lem-fisher-sqrt-comparison`).
   - Integrate conditional bounds over position space.
   - Handle measure-theoretic details (Fubini's theorem, regularity requirements).
   - Verify final constant $\lambda_{\text{mic}}$ is explicit and N-uniform.

6. **Additional Technical Lemmas (4-6 hours total):**
   - `lem-conditional-qsd-regularity`: 4-6 hours
   - `lem-fisher-sqrt-comparison`: 1-2 hours
   - `lem-mean-field-metric-deterministic`: 3-5 hours
   - (Some of these may be parallelized with main proof development)

**Note:** This estimate assumes familiarity with hypocoercivity theory and spectral gap techniques. Additional time may be needed for literature review if specific techniques (e.g., Bakry-Émery theory for anisotropic metrics) are unfamiliar.

---

## VIII. Connection to Broader Framework

**Upstream Dependencies:**
- QSD Existence and Regularity (`thm-qsd-existence`)
- Uniform Ellipticity of Regularized Metric (Chapter 3, 11_geometric_gas.md)
- Microlocal Decomposition (`def-microlocal`)

**Downstream Usage:**
- This lemma is Step A in the three-step hypocoercivity argument for proving the mean-field LSI (`thm-lsi-mean-field`).
- Combined with Steps B (`lem-macro-transport`) and C (`lem-micro-reg`), it yields exponential convergence to QSD.
- The constant $\lambda_{\text{mic}}$ contributes to the final LSI constant $\lambda_{\text{LSI}}$ via the hypocoercive assembly formula.

**Physical Significance:**
- This lemma quantifies the rate at which velocity randomization homogenizes the velocity distribution at each position.
- The uniformity of $\lambda_{\text{mic}}$ across positions ensures that the velocity dissipation is not "trapped" in any region of state space.
- The $\rho$-dependence of $\lambda_{\text{mic}}$ (via $c_{\max}(\rho)$) tracks how the diversity parameter $\rho$ affects the microscopic thermalization rate.

---

## IX. Review Notes

**Review Protocol Used:** Single-strategist (GPT-5 / Codex only)

**Reason:** Gemini MCP currently has technical issues preventing dual review.

**Confidence Level:** MEDIUM - Lower than dual-review protocol

**Recommended Next Steps:**
1. Submit this sketch to GPT-5 (Codex) for independent review.
2. When Gemini MCP is restored, re-run dual review for cross-validation.
3. Address any discrepancies or gaps identified by reviewers.
4. Proceed to full proof expansion only after reviewer consensus.

**Human Verification Recommended:** Due to single-strategist limitation, human expert review is advised before proceeding to full proof.

---

## X. References

1. **Villani's Hypocoercivity Theory:**
   - Villani, C. (2009). "Hypocoercivity." *Memoirs of the AMS*, Vol. 202, No. 950.
   - Standard reference for conditional Poincaré inequalities in kinetic theory.

2. **Spectral Gap for Ornstein-Uhlenbeck:**
   - Bakry, D., Gentil, I., Ledoux, M. (2014). "Analysis and Geometry of Markov Diffusion Operators." Springer.
   - Chapter on OU processes and spectral gaps.

3. **Framework Documents:**
   - `09_kl_convergence.md` (Euclidean Gas hypocoercivity)
   - `11_geometric_gas.md` § 9.3 (Mean-field LSI proof strategy)
   - `16_convergence_mean_field.md` (Complete mean-field LSI proof for Euclidean Gas)

---

**End of Proof Sketch**

**File:** `/home/guillem/fragile/docs/source/2_geometric_gas/sketcher/sketch_lem_micro_coercivity.md`

**Status:** DRAFT - Awaiting GPT-5 (Codex) review

**Next Action:** Submit to Codex for rigor check and gap analysis.
