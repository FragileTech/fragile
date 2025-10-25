# Proof Sketch: Total Convergence Rate (Parameter-Explicit)

**Theorem Label:** `thm-total-rate-explicit`
**Source Document:** `docs/source/1_euclidean_gas/06_convergence.md` (lines 1715-1792)
**Theorem Type:** Theorem
**Date:** 2025-10-25

---

## Theorem Statement

:::{prf:theorem} Total Convergence Rate (Parameter-Explicit)
:label: thm-total-rate-explicit

The total geometric convergence rate is:

$$
\kappa_{\text{total}} = \min(\kappa_x, \kappa_v, \kappa_W, \kappa_b) \cdot (1 - \epsilon_{\text{coupling}})
$$

where $\epsilon_{\text{coupling}} \ll 1$ is the expansion-to-contraction ratio:

$$
\epsilon_{\text{coupling}} = \max\left(
\frac{\alpha_v C_{xv}}{\kappa_v V_{\text{Var},v}},
\frac{\alpha_W C_{xW}}{\kappa_W V_W},
\frac{C_{vx}}{\kappa_x V_{\text{Var},x}},
\ldots
\right)
$$

The equilibrium constant is:

$$
C_{\text{total}} = \frac{C_x + \alpha_v C_v' + \alpha_W C_W' + \alpha_b C_b}{\kappa_{\text{total}}}
$$

**Explicit formulas:**

Substituting from previous sections:

$$
\kappa_{\text{total}} \sim \min\left(
\lambda, \quad 2\gamma, \quad \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}, \quad \lambda \frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}
\right) \cdot (1 - O(\tau))
$$

$$
C_{\text{total}} \sim \frac{1}{\kappa_{\text{total}}} \left(
\frac{\sigma_v^2 \tau^2}{\gamma \lambda} + \frac{d\sigma_v^2}{\gamma} + \frac{\sigma_v^2 \tau}{N^{1/d}} + \frac{\sigma_v^2 \tau}{d_{\text{safe}}^2}
\right)
$$
:::

---

## Dependencies

### Required Results

1. **Foster-Lyapunov Condition** (`thm-foster-lyapunov-main` from Section 3.4):
   - Establishes the synergistic composition framework
   - Provides the general form: $\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$
   - Shows that coupling constants $(c_V, c_B)$ can be chosen to achieve net contraction

2. **Component Rate Formulas** (from Section 5):
   - `prop-velocity-rate-explicit` (Section 5.1): $\kappa_v = 2\gamma - O(\tau)$
   - `prop-position-rate-explicit` (Section 5.2): $\kappa_x = \lambda \cdot \mathbb{E}[\text{Cov}(f, \|x - \bar{x}\|^2)/V_{\text{Var},x}] + O(\tau)$
   - Wasserstein contraction rate: $\kappa_W = \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}$ (from `05_kinetic_contraction.md`)
   - Boundary safety rate: $\kappa_b = \lambda \frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}$ (from `03_cloning.md`)

3. **Component Equilibrium Constants** (from Section 5):
   - $C_v' = \frac{d\sigma_v^2}{\gamma} + O(\tau\sigma_v^2)$
   - $C_x = O(\frac{\sigma_v^2 \tau^2}{\gamma\lambda})$
   - $C_W' = O(\frac{\sigma_v^2 \tau}{N^{1/d}})$ (Wasserstein diffusion)
   - $C_b = O(\frac{\sigma_v^2 \tau}{d_{\text{safe}}^2})$ (boundary perturbation)

4. **Coupling Analysis** (from Section 3.5-3.6):
   - Cross-component expansion terms: $C_{xv}, C_{xW}, C_{vx}$, etc.
   - Coupling penalty formula relating these to $\epsilon_{\text{coupling}}$

### Framework Context

This theorem synthesizes the **synergistic dissipation** paradigm established throughout Chapter 6:
- Cloning operator contracts position ($\kappa_x$) and boundary risk ($\kappa_b$) but expands velocity
- Kinetic operator contracts velocity ($\kappa_v$) and Wasserstein distance ($\kappa_W$) but expands position
- Proper coupling allows net contraction of the total Lyapunov function

---

## Proof Strategy

### High-Level Approach

The proof follows a **bottleneck principle with coupling penalty** structure:

1. **Establish the minimum-rate formula** by showing that the weakest contraction component dominates the overall convergence
2. **Quantify the coupling penalty** $\epsilon_{\text{coupling}}$ arising from cross-component expansions
3. **Substitute explicit parameter formulas** for each component rate to obtain the concrete expression
4. **Derive the equilibrium constant** by balancing all source terms at stationarity

The key insight is that in the synergistic composition framework, the total rate is limited by:
- The **slowest contracting component** (bottleneck)
- The **cost of coupling** (energy wasted compensating other operators' expansions)

### Proof Outline

#### Step 1: Derive the Bottleneck Principle

**Goal:** Show $\kappa_{\text{total}} = \min_i(\kappa_i) \cdot (1 - \epsilon_{\text{coupling}})$

**Approach:**
1. Start from the Foster-Lyapunov condition (Theorem `thm-foster-lyapunov-main`):
   $$
   \mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}
   $$
   where $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$.

2. Decompose the drift into component contributions:
   $$
   \mathbb{E}[\Delta V_{\text{total}}] = \sum_i c_i \mathbb{E}[\Delta V_i]
   $$

3. Use the component drift inequalities from prerequisite documents:
   - Cloning: $\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x$
   - Kinetic: $\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v}\tau + d\sigma_{\max}^2\tau$
   - Etc. (see component drift table in Section 3.3)

4. Identify **contraction terms** (negative drift) vs. **expansion terms** (positive drift):
   - Contraction: $-\sum_i c_i \kappa_i V_i$
   - Expansion: Cross-component coupling terms like $\alpha_v C_{xv}$

5. The effective contraction rate is the weakest component rate minus the coupling penalty:
   $$
   \kappa_{\text{total}} = \min_i(\kappa_i) - \text{(coupling losses)}
   $$

6. Express coupling losses as a fraction $\epsilon_{\text{coupling}}$ of the minimum rate.

**Critical Estimate:**
The coupling penalty arises from terms like:
$$
\frac{\alpha_v C_{xv}}{\kappa_v V_{\text{Var},v}^{\text{eq}}}
$$
which represents the expansion of $V_{\text{Var},x}$ due to kinetic operator divided by the contraction of $V_{\text{Var},v}$.

#### Step 2: Quantify the Coupling Ratio

**Goal:** Show $\epsilon_{\text{coupling}} = \max(\ldots) \ll 1$ under proper parameter choices

**Approach:**
1. For each cross-component expansion term, compute the ratio:
   $$
   \text{expansion rate} / \text{contraction rate}
   $$

2. Examples:
   - Velocity expansion by cloning: $\frac{C_v}{\kappa_v V_{\text{Var},v}^{\text{eq}}}$
   - Position expansion by kinetics: $\frac{C_{\text{kin},x}}{\kappa_x V_{\text{Var},x}^{\text{eq}}}$

3. Use the equilibrium values:
   - $V_{\text{Var},v}^{\text{eq}} = \frac{C_v'}{\kappa_v} = \frac{d\sigma_v^2}{\gamma}$ (from velocity thermalization)
   - $V_{\text{Var},x}^{\text{eq}} = \frac{C_x}{\kappa_x}$ (from position contraction)

4. Substitute to obtain explicit parameter dependence:
   $$
   \epsilon_{\text{coupling}} = \max\left(
   \frac{C_v \gamma}{C_v' \kappa_v}, \quad
   \frac{C_{\text{kin},x} \kappa_x}{C_x \kappa_v}, \quad
   \ldots
   \right)
   $$

5. Show that for small timestep ($\tau \ll 1$) and proper parameter scaling:
   $$
   \epsilon_{\text{coupling}} = O(\tau)
   $$
   hence the $(1 - O(\tau))$ factor in the explicit formula.

**Critical Estimate:**
For typical parameters:
- $C_v \sim O(1)$ (bounded cloning perturbation)
- $\kappa_v \sim 2\gamma \sim O(1)$
- $C_v' \sim d\sigma_v^2/\gamma$
- Thus: $\frac{C_v \gamma}{C_v' \kappa_v} \sim \frac{\gamma}{d\sigma_v^2/\gamma \cdot 2\gamma} = O(\gamma^2/(d\sigma_v^2)) \ll 1$ if noise is sufficiently strong.

#### Step 3: Substitute Explicit Component Rates

**Goal:** Obtain the concrete formula with primitive parameters

**Approach:**
1. Substitute the explicit formulas from Sections 5.1-5.4:
   - $\kappa_v = 2\gamma - O(\tau)$ (velocity dissipation)
   - $\kappa_x \sim \lambda$ (cloning rate, assuming strong fitness-variance correlation)
   - $\kappa_W \sim \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}$ (hypocoercive Wasserstein contraction)
   - $\kappa_b \sim \lambda \frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}$ (boundary safety)

2. Take the minimum:
   $$
   \kappa_{\text{total}} \sim \min\left(
   \lambda, \quad 2\gamma, \quad \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}, \quad \lambda \frac{\Delta f_{\text{boundary}}}{f_{\text{typical}}}
   \right)
   $$

3. Apply the coupling penalty $(1 - O(\tau))$.

**Note:** The minimum identifies the **bottleneck component**:
- If $\lambda < 2\gamma$: position is bottleneck (cloning too slow)
- If $2\gamma < \lambda$: velocity is bottleneck (friction too weak)
- If Wasserstein term is smallest: inter-swarm mixing is bottleneck
- If boundary term is smallest: boundary risk dominates

#### Step 4: Derive the Equilibrium Constant

**Goal:** Show $C_{\text{total}} = \frac{C_x + \alpha_v C_v' + \alpha_W C_W' + \alpha_b C_b}{\kappa_{\text{total}}}$

**Approach:**
1. At equilibrium, the drift vanishes:
   $$
   \mathbb{E}[\Delta V_{\text{total}}] = 0
   $$

2. From the Foster-Lyapunov condition:
   $$
   -\kappa_{\text{total}} V_{\text{total}}^{\text{eq}} + C_{\text{total}} = 0
   $$

3. Solve for equilibrium:
   $$
   V_{\text{total}}^{\text{eq}} = \frac{C_{\text{total}}}{\kappa_{\text{total}}}
   $$

4. The constant $C_{\text{total}}$ aggregates all source terms:
   - $C_x$ from positional diffusion by kinetics
   - $\alpha_v C_v'$ from velocity noise (weighted by Lyapunov coefficient)
   - $\alpha_W C_W'$ from Wasserstein diffusion
   - $\alpha_b C_b$ from boundary perturbations

5. Substitute the explicit formulas:
   - $C_x \sim \frac{\sigma_v^2 \tau^2}{\gamma\lambda}$
   - $C_v' \sim \frac{d\sigma_v^2}{\gamma}$
   - $C_W' \sim \frac{\sigma_v^2 \tau}{N^{1/d}}$
   - $C_b \sim \frac{\sigma_v^2 \tau}{d_{\text{safe}}^2}$

6. Combine to get the total equilibrium constant formula.

**Critical Estimate:**
The equilibrium variance scales as:
$$
V_{\text{total}}^{\text{eq}} \sim \frac{1}{\kappa_{\text{total}}} \left(
\frac{\sigma_v^2 \tau^2}{\gamma\lambda} + \frac{d\sigma_v^2}{\gamma} + \frac{\sigma_v^2 \tau}{N^{1/d}} + \frac{\sigma_v^2 \tau}{d_{\text{safe}}^2}
\right)
$$

This shows:
- Faster convergence ($\kappa_{\text{total}} \uparrow$) leads to tighter equilibrium
- Stronger noise ($\sigma_v \uparrow$) leads to wider equilibrium
- Larger swarm ($N \uparrow$) reduces Wasserstein contribution

---

## Key Technical Points

### Point 1: Why the Minimum?

The bottleneck principle follows from the fact that $V_{\text{total}}$ is a **weighted sum** of component variances. If one component contracts slowly, it limits the overall contraction even if other components contract quickly.

**Formal argument:**
Suppose $\kappa_i$ are the component rates. The total drift is:
$$
\mathbb{E}[\Delta V_{\text{total}}] = -\sum_i c_i \kappa_i V_i + \text{(coupling terms)}
$$

If we write $V_{\text{total}} = \sum_i c_i V_i$, the effective rate for uniform contraction is:
$$
\kappa_{\text{total}} = \min_i \kappa_i
$$
because the slowest component dominates the long-time behavior.

### Point 2: Coupling Penalty Interpretation

The coupling penalty $\epsilon_{\text{coupling}}$ represents the **fraction of contraction power wasted** on compensating other operators' expansions.

**Example:**
- Cloning contracts $V_{\text{Var},x}$ at rate $\kappa_x$
- But kinetic expands $V_{\text{Var},x}$ at rate $C_{\text{kin},x}\tau$
- Net contraction: $\kappa_x - \frac{C_{\text{kin},x}\tau}{V_{\text{Var},x}^{\text{eq}}}$
- Penalty: $\frac{C_{\text{kin},x}\tau/V_{\text{Var},x}^{\text{eq}}}{\kappa_x} = \epsilon_{\text{coupling},x}$

As long as $\epsilon_{\text{coupling}} < 1$, net contraction is preserved.

### Point 3: Parameter Scaling

For **balanced convergence** (no single bottleneck), choose:
$$
\lambda \sim \gamma \sim c_{\text{hypo}}^2 \gamma / (1 + \gamma/\lambda_{\min}) \sim \lambda \Delta f_{\text{boundary}}/f_{\text{typical}}
$$

This is generally satisfied by:
$$
\gamma \sim \lambda \sim \lambda_{\min}
$$
where $\lambda_{\min}$ is the smallest eigenvalue of the Hessian of the potential (landscape curvature).

**Interpretation:** Match the thermalization rate to the landscape's natural timescale.

---

## Potential Difficulties and Resolutions

### Difficulty 1: Verifying $\epsilon_{\text{coupling}} \ll 1$

**Challenge:**
The coupling terms involve ratios of equilibrium variances, which are themselves functions of the parameters. Circular dependencies may arise.

**Resolution:**
Use **self-consistent analysis**:
1. Assume $V_i^{\text{eq}} = C_i/\kappa_i$ (from equilibrium condition)
2. Substitute into coupling ratios
3. Verify that resulting $\epsilon_{\text{coupling}}$ is consistent with $(1 - \epsilon_{\text{coupling}})$ factor

For small $\tau$, this iteration converges and yields $\epsilon_{\text{coupling}} = O(\tau)$.

### Difficulty 2: Handling the "..." in the Coupling Formula

**Challenge:**
The theorem statement shows "$\ldots$" indicating additional coupling terms not explicitly listed.

**Resolution:**
The complete coupling penalty includes all pairwise cross-component terms:
$$
\epsilon_{\text{coupling}} = \max\left(
\frac{\alpha_v C_{xv}}{\kappa_v V_{\text{Var},v}},
\frac{\alpha_W C_{xW}}{\kappa_W V_W},
\frac{C_{vx}}{\kappa_x V_{\text{Var},x}},
\frac{C_{Wv}}{\kappa_v V_{\text{Var},v}},
\frac{C_{bx}}{\kappa_x V_{\text{Var},x}},
\ldots
\right)
$$

In the full proof, we enumerate all such terms systematically using the component drift table (Section 3.3) and verify each is $O(\tau)$ or smaller.

### Difficulty 3: Justifying the Approximation $\kappa_x \sim \lambda$

**Challenge:**
The exact formula is:
$$
\kappa_x = \lambda \cdot \frac{\text{Cov}(f_i, \|x_i - \bar{x}\|^2)}{V_{\text{Var},x}}
$$
which depends on the fitness-variance correlation.

**Resolution:**
Under the **Keystone Principle** (Theorem 5.1 from `03_cloning.md`), walkers with high fitness tend to have low variance from the swarm center. This anti-correlation is quantified by:
$$
\frac{\text{Cov}(f_i, \|x_i - \bar{x}\|^2)}{V_{\text{Var},x}} \sim -\Theta(1)
$$
with the negative sign absorbed into the contraction interpretation.

For typical fitness landscapes (unimodal with well-separated optima), this correlation is strong, so:
$$
\kappa_x \sim \lambda \cdot O(1) = \Theta(\lambda)
$$

In pathological cases (flat fitness), $\kappa_x \to 0$ and cloning provides no contraction.

### Difficulty 4: N-Dependence in Wasserstein Term

**Challenge:**
The Wasserstein equilibrium constant depends on $N$ as:
$$
C_W' \sim \frac{\sigma_v^2 \tau}{N^{1/d}}
$$
raising questions about the mean-field limit ($N \to \infty$).

**Resolution:**
As $N \to \infty$:
- $C_W' \to 0$ (Wasserstein diffusion vanishes)
- $V_W^{\text{eq}} \to 0$ (swarms become identical)
- The Wasserstein component no longer contributes to the bottleneck
- The limit is well-defined, with $\kappa_{\text{total}}$ approaching the minimum of the remaining components

This validates the mean-field limit and shows the finite-$N$ analysis degenerates correctly.

---

## Summary

### Main Steps

1. **Bottleneck Principle:** Derive $\kappa_{\text{total}} = \min_i(\kappa_i) \cdot (1 - \epsilon_{\text{coupling}})$ from the Foster-Lyapunov framework
2. **Coupling Quantification:** Show $\epsilon_{\text{coupling}} = O(\tau)$ by estimating all cross-component expansion ratios
3. **Parameter Substitution:** Substitute explicit component formulas to obtain the concrete rate expression
4. **Equilibrium Constant:** Aggregate all source terms to derive $C_{\text{total}}$ formula

### Key Insights

- **Synergistic dissipation:** Neither operator alone achieves convergence, but their composition does
- **Bottleneck dominance:** The slowest component limits overall convergence
- **Coupling penalty:** Small timestep ensures coupling losses are negligible
- **N-uniformity:** All constants are $N$-independent, validating mean-field analysis

### Mathematical Tools Required

- Foster-Lyapunov drift condition (prerequisite theorem)
- Component-wise drift decomposition
- Equilibrium variance formulas ($V_i^{\text{eq}} = C_i/\kappa_i$)
- Asymptotic expansions in $\tau$ (small timestep regime)
- Bottleneck analysis (min over components)

---

## Next Steps for Full Proof

1. **Enumerate all coupling terms** from the component drift table (Section 3.3)
2. **Compute each coupling ratio** explicitly using equilibrium formulas
3. **Verify $\epsilon_{\text{coupling}} = O(\tau)$** for all terms
4. **Formalize the bottleneck argument** with rigorous min-max analysis
5. **Validate the explicit formulas** through numerical examples (as shown in Section 5.6)
6. **Extend to sensitivity analysis** (Chapter 6) for robustness

---

**End of Sketch**
