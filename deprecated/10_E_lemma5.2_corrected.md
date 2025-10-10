# Lemma 5.2: Corrected Mean-Field Version

## Critical Fix: Operator Decomposition

**Issue identified by Gemini**: The previous version incorrectly decomposed the operator as $T = T_{\text{noise}} \circ T_{\text{clone}}$, where both operators applied Gaussian noise with variance $\delta^2$. This **double-counts** the noise.

**Correct model** (from [05_mean_field.md](05_mean_field.md)):

The discrete algorithm has composition:

$$
\Psi = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}
$$

where:
- $\Psi_{\text{clone}}$: Cloning operator **with post-cloning noise** $Q_\delta$ built-in
- $\Psi_{\text{kin}}$: Kinetic Langevin operator (separate from cloning noise)

The noise $Q_\delta(z \mid z_c)$ is part of the **source term** $S_{\text{src}}$ in the mean-field cloning generator.

---

## Lemma 5.2 (Corrected Mean-Field Version)

:::{prf:lemma} Entropy Dissipation Under Cloning (Mean-Field)
:label: lem-mean-field-cloning-dissipation

**Context**: This lemma analyzes **only the cloning step** of the algorithm. The full LSI proof requires combining this with the kinetic operator analysis.

**Hypotheses:**

1. $\mu, \pi$ are probability measures on $\Omega$ with **smooth densities** $\rho_\mu, \rho_\pi$:
   - $\rho_\mu, \rho_\pi \in C^2(\Omega)$
   - $\rho_\mu, \rho_\pi > 0$ on $\Omega$ (strictly positive)
   - $\int_\Omega \rho_\mu = \int_\Omega \rho_\pi = 1$

2. $\pi = \pi_{\text{QSD}}$ is log-concave (Axiom 3.5): $\rho_\pi(z) = \exp(-V_{\text{QSD}}(z))$ for convex $V_{\text{QSD}}$

3. $T_{\text{clone}}: \mathcal{P}(\Omega) \to \mathcal{P}(\Omega)$ is the mean-field cloning operator with post-cloning noise $\delta^2$ built-in (Definition {prf:ref}`def-cloning-generator` in 05_mean_field.md)

4. **W₂ Contraction Assumption** (to be proven in 03_cloning.md):

   $$
   W_2^2(T_{\text{clone}} \# \mu, \pi) \leq (1 - \kappa_W) W_2^2(\mu, \pi)
   $$

   for some $\kappa_W > 0$

**Conclusion:**

For $\mu' = T_{\text{clone}} \# \mu$, the cloning operator provides **bounded KL divergence change**:

$$
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi) \leq C_{\text{clone}}
$$

where $C_{\text{clone}}$ is a constant depending on $\delta^2$ and the cloning probability.

**Furthermore**, if the potential energy reduction dominates entropy increase:

$$
D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi) \leq -\beta D_{\text{KL}}(\mu \| \pi)
$$

for some $\beta > 0$ (strong cloning regime).

:::

---

## Proof Structure

### Step 0: Cloning Operator Definition

From [05_mean_field.md](05_mean_field.md), the cloning operator is defined by its infinitesimal generator $S[\rho]$:

$$
\rho_{\mu'} = \rho_\mu + \tau S[\rho_\mu] + O(\tau^2)
$$

where:

$$
\begin{aligned}
S[\rho](z) &= S_{\text{src}}[\rho](z) - S_{\text{sink}}[\rho](z) \\
S_{\text{src}}[\rho](z) &= \frac{1}{m_a} \int_{\Omega \times \Omega} \rho(z_d) \rho(z_c) \, P_{\text{clone}}(V[z_d], V[z_c]) \, \underbrace{Q_\delta(z \mid z_c)}_{\text{post-cloning noise}} \, \mathrm{d}z_d \, \mathrm{d}z_c \\
S_{\text{sink}}[\rho](z) &= \rho(z) \int_{\Omega} P_{\text{clone}}(V[z], V[z']) \, \frac{\rho(z')}{m_a} \, \mathrm{d}z'
\end{aligned}
$$

**KEY**: The noise kernel $Q_\delta(z \mid z_c) = \mathcal{N}(z; z_c, \delta^2 I)$ is **already included** in $S_{\text{src}}$. There is **no additional noise step**.

---

### Step 1: KL Divergence Change Under Cloning

Define:

$$
\Delta_{\text{clone}} := D_{\text{KL}}(\mu' \| \pi) - D_{\text{KL}}(\mu \| \pi)
$$

where $\mu' = T_{\text{clone}} \# \mu$ with density $\rho_{\mu'}$.

### Step 2: Entropy-Potential Decomposition

Recall:

$$
D_{\text{KL}}(\mu \| \pi) = \underbrace{-\int_\Omega \rho_\mu(z) \log \rho_\mu(z) \, \mathrm{d}z}_{-H(\mu)} + \underbrace{\int_\Omega \rho_\mu(z) V_{\text{QSD}}(z) \, \mathrm{d}z}_{E_\mu[\pi]}
$$

where:
- $H(\mu) = -\int \rho_\mu \log \rho_\mu$: **Differential entropy**
- $E_\mu[\pi] = \int \rho_\mu V_{\text{QSD}}$: **Potential energy**

Therefore:

$$
\Delta_{\text{clone}} = [H(\mu) - H(\mu')] + [E_{\mu'}[\pi] - E_\mu[\pi]]
$$

We bound each term separately.

---

### Step 3: Bounding Potential Energy Change

**From [10_D_step3_cloning_bounds.md](10_D_step3_cloning_bounds.md), Part A**:

The cloning operator:
- **Removes** mass from low-fitness (high potential) regions via $S_{\text{sink}}$
- **Adds** mass to high-fitness (low potential) regions via $S_{\text{src}}$

For infinitesimal time step $\tau$:

$$
E_{\mu'}[\pi] - E_\mu[\pi] = \tau \int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z + O(\tau^2)
$$

**Key result** (derived in Part A of 10_D):

$$
\int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z = \frac{1}{m_a} \int_{\Omega \times \Omega} \rho_\mu(z_d) \rho_\mu(z_c) P_{\text{clone}}(V_d, V_c) [V_{\text{QSD}}(z_c) - V_{\text{QSD}}(z_d)] \, \mathrm{d}z_d \mathrm{d}z_c
$$

Since $P_{\text{clone}} \propto V_d / V_c$ and high fitness correlates with low potential (for QSD):

$$
\int_\Omega S[\rho_\mu](z) \, V_{\text{QSD}}(z) \, \mathrm{d}z \leq -C_{\text{pot}} D_{\text{KL}}(\mu \| \pi)
$$

for some $C_{\text{pot}} > 0$ (assuming fitness-potential anti-correlation).

**Therefore**:

$$
E_{\mu'}[\pi] - E_\mu[\pi] \leq -\tau \lambda_{\text{clone}} C_{\text{pot}} D_{\text{KL}}(\mu \| \pi)
$$

---

### Step 4: Bounding Entropy Change

**From [10_D_step3_cloning_bounds.md](10_D_step3_cloning_bounds.md), Part B**:

The entropy change has two competing effects:

1. **Selection** (via $S_{\text{sink}}$): Reduces entropy by concentrating distribution
2. **Noise injection** (via $Q_\delta$ in $S_{\text{src}}$): Increases entropy by diffusion

For infinitesimal time step:

$$
H(\mu') - H(\mu) = -\tau \int_\Omega S[\rho_\mu](z) [\log \rho_\mu(z) + 1] \, \mathrm{d}z + O(\tau^2)
$$

**Heuristic bound** (from Part B of 10_D):

$$
H(\mu) - H(\mu') \approx \tau \lambda_{\text{clone}} \left[\text{Var}_\mu[\log \rho_\mu] - \frac{d}{2} \log(2\pi e \delta^2)\right]
$$

**Key insight**: For sufficiently large cloning noise $\delta^2$, the noise-induced entropy gain dominates the selection-induced entropy loss.

**Bounded entropy change**:

$$
H(\mu) - H(\mu') \leq \tau \lambda_{\text{clone}} C_{\text{ent}}
$$

where $C_{\text{ent}}$ depends on $\delta^2$, variance bounds, and regularity of $\rho_\mu$.

---

### Step 5: Combined Bound

Combining Steps 3 and 4:

$$
\begin{aligned}
\Delta_{\text{clone}} &= [H(\mu) - H(\mu')] + [E_{\mu'}[\pi] - E_\mu[\pi]] \\
&\leq \tau \lambda_{\text{clone}} C_{\text{ent}} - \tau \lambda_{\text{clone}} C_{\text{pot}} D_{\text{KL}}(\mu \| \pi)
\end{aligned}
$$

**Case 1: Weak cloning** ($C_{\text{pot}}$ small):

$$
\Delta_{\text{clone}} \leq C_{\text{clone}} := \tau \lambda_{\text{clone}} C_{\text{ent}}
$$

(Bounded expansion, not contraction)

**Case 2: Strong cloning** ($C_{\text{pot}}$ large):

If $C_{\text{pot}} > C_{\text{ent}} / D_{\text{KL}}(\mu \| \pi)$, then:

$$
\Delta_{\text{clone}} \leq -\beta D_{\text{KL}}(\mu \| \pi)
$$

where $\beta := \tau \lambda_{\text{clone}} C_{\text{pot}} - \tau \lambda_{\text{clone}} C_{\text{ent}} / D_{\text{KL}}(\mu \| \pi) > 0$.

**Conclusion**:

In the weak cloning regime, $\Delta_{\text{clone}} \leq C_{\text{clone}}$ (bounded).

In the strong cloning regime, $\Delta_{\text{clone}} \leq -\beta D_{\text{KL}}(\mu \| \pi)$ (contractive). $\square$

---

## Remaining Gaps

To make this proof fully rigorous:

1. **Prove fitness-potential correlation** (Assumption in Step 3)
   - Show that high fitness $V[z]$ correlates with low potential $V_{\text{QSD}}(z)$
   - This should follow from the QSD definition and algorithm design

2. **Rigorously bound variance terms** (Step 4)
   - Use Poincaré/log-Sobolev inequalities for $\text{Var}_\mu[\log \rho_\mu]$
   - Apply entropy power inequality for noise contribution

3. **Complete the full LSI proof**
   - Combine cloning analysis with **kinetic operator** analysis
   - The kinetic operator provides additional dissipation via Langevin dynamics
   - Final result: exponential convergence of full operator $\Psi = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$

---

## Connection to Full LSI

This lemma analyzes **only the cloning operator**. For the complete LSI:

$$
D_{\text{KL}}(\rho_{t+\tau} \| \pi) - D_{\text{KL}}(\rho_t \| \pi) \leq -\kappa_{\text{total}} D_{\text{KL}}(\rho_t \| \pi)
$$

we need to combine:

1. **Cloning contribution**: $\Delta_{\text{clone}}$ (this lemma)
2. **Kinetic contribution**: De Bruijn identity + LSI for Langevin dynamics

The kinetic operator $\Psi_{\text{kin}}$ provides **additional dissipation** through:
- Friction term $-\gamma v$ (velocity damping)
- Diffusion term $\Sigma \,dW$ (regularization)
- Potential force $\nabla R(x)$ (confinement)

Together, these ensure:

$$
\kappa_{\text{total}} = \underbrace{\kappa_{\text{kin}}}_{\text{Langevin LSI}} + \underbrace{\beta_{\text{clone}}}_{\text{Cloning (if strong)}} - \underbrace{C_{\text{clone}}/D_{\text{KL}}}_{\text{Cloning expansion (if weak)}} > 0
$$

---

## Summary of Corrections

**What changed from previous version**:

1. ✅ **Removed double-counting**: No separate $T_{\text{noise}}$ operator
2. ✅ **Correct operator**: Single $T_{\text{clone}}$ with noise built-in
3. ✅ **Simplified structure**: Direct analysis of $\Delta_{\text{clone}}$ using entropy-potential decomposition
4. ✅ **Clear scope**: This lemma covers cloning only; full LSI requires kinetic operator

**What remains the same**:

1. ✅ Entropy-potential decomposition (valid fix for convexity subtraction error)
2. ✅ Smooth density formalism (valid fix for discrete-vs-continuous error)
3. ✅ Mean-field setting (correct approach for Part A)

**Status**: Proof structure is now mathematically consistent. Remaining gaps are in completing the technical bounds (Steps 3-4) and combining with kinetic operator analysis.
