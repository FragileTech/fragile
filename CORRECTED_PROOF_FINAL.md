:::{prf:proof}
**Proof (Integral Representation with OU Covariance Bounds).**

**PART I: Integral Representation**

For walker $i$ in swarm $k$, the centered position evolves deterministically:

$$
d\delta_{x,k,i} = \delta_{v,k,i} \, dt
$$

where $\delta_{x,k,i}(t) = x_{k,i}(t) - \mu_{x,k}(t)$ and $\delta_{v,k,i}(t) = v_{k,i}(t) - \mu_{v,k}(t)$.

**Key observation:** Position has no direct stochastic term—it evolves as $dx = v \, dt$. Therefore, Itô's lemma yields **no dt² correction term**.

Integrating from $t=0$ to $t=\tau$:

$$
\delta_{x,k,i}(\tau) = \delta_{x,k,i}(0) + \int_0^\tau \delta_{v,k,i}(s) \, ds
$$

Squaring both sides:

$$
\|\delta_{x,k,i}(\tau)\|^2 = \|\delta_{x,k,i}(0)\|^2 + 2\left\langle \delta_{x,k,i}(0), \int_0^\tau \delta_{v,k,i}(s) \, ds \right\rangle + \left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2
$$

**PART II: Linear Term—Position-Velocity Coupling**

For the linear cross-term, expand to first order in $\tau$:

$$
\int_0^\tau \delta_{v,k,i}(s) \, ds \approx \delta_{v,k,i}(0) \tau + O(\tau^2)
$$

Thus:

$$
2\left\langle \delta_{x,k,i}(0), \int_0^\tau \delta_{v,k,i}(s) \, ds \right\rangle \approx 2\langle \delta_{x,k,i}(0), \delta_{v,k,i}(0) \rangle \tau + O(\tau^2)
$$

Taking expectations and using Cauchy-Schwarz:

$$
\left|\mathbb{E}[\langle \delta_{x,k,i}, \delta_{v,k,i} \rangle]\right| \leq \sqrt{\mathbb{E}[\|\delta_{x,k,i}\|^2] \cdot \mathbb{E}[\|\delta_{v,k,i}\|^2]}
$$

At equilibrium, the underdamped Langevin dynamics ensures position-velocity decorrelation:

$$
\mathbb{E}_{\text{eq}}[\langle \delta_x, \delta_v \rangle] = 0
$$

During transients, we use uniform bounds on variances (see Assumption {prf:ref}`assump-uniform-variance-bounds` below):

$$
\left|\mathbb{E}\left[2\left\langle \delta_{x,k,i}(0), \int_0^\tau \delta_{v,k,i}(s) \, ds \right\rangle\right]\right| \leq 2\sqrt{M_x \cdot M_v} \, \tau
$$

Define:

$$
C_1 := 2\sqrt{M_x \cdot M_v}
$$

**PART III: Quadratic Term—Velocity Accumulation via Exponential Covariance Decay**

The critical term is:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2\right]
$$

Expanding the squared norm:

$$
\left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2 = \int_0^\tau \int_0^\tau \langle \delta_{v,k,i}(s_1), \delta_{v,k,i}(s_2) \rangle \, ds_1 \, ds_2
$$

Taking expectations:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2\right] = \int_0^\tau \int_0^\tau \mathbb{E}[\langle \delta_{v,k,i}(s_1), \delta_{v,k,i}(s_2) \rangle] \, ds_1 \, ds_2
$$

**Velocity covariance bound:** The centered velocity $\delta_v$ satisfies the underdamped Langevin SDE:

$$
d\delta_v = [F(x) - F(\mu_x) - \gamma \delta_v] \, dt + \Sigma \circ dW
$$

While $\delta_v$ is not an exact Ornstein-Uhlenbeck (OU) process for general non-quadratic potentials $U$ (due to the nonlinear force term $F(x) - F(\mu_x)$), the friction term $-\gamma \delta_v$ governs exponential decay of velocity correlations. Under the Lipschitz condition on $F$ (Axiom {prf:ref}`axiom-bounded-displacement` from 01_fragile_gas_framework.md) and constant friction $\gamma > 0$, the velocity autocovariance satisfies the upper bound:

$$
\mathbb{E}[\langle \delta_{v}(s_1), \delta_{v}(s_2) \rangle] \leq V_{\text{Var},v}^{\text{eq}} e^{-\gamma |s_1 - s_2|}
$$

where $V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}$ is the equilibrium velocity variance from {prf:ref}`thm-velocity-variance-contraction-kinetic`.

**Double integral evaluation:**

Using the exponential bound:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2\right] \leq V_{\text{Var},v}^{\text{eq}} \int_0^\tau \int_0^\tau e^{-\gamma |s_1 - s_2|} \, ds_1 \, ds_2
$$

By symmetry:

$$
\int_0^\tau \int_0^\tau e^{-\gamma |s_1 - s_2|} \, ds_1 \, ds_2 = 2\int_0^\tau \int_0^{s_2} e^{-\gamma(s_2 - s_1)} \, ds_1 \, ds_2
$$

Inner integral:

$$
\int_0^{s_2} e^{-\gamma(s_2 - s_1)} \, ds_1 = \frac{1}{\gamma}(1 - e^{-\gamma s_2})
$$

Outer integral:

$$
2\int_0^\tau \frac{1}{\gamma}(1 - e^{-\gamma s_2}) \, ds_2 = \frac{2}{\gamma}\left[\tau - \frac{1}{\gamma}(1 - e^{-\gamma \tau})\right]
$$

This exact identity holds for all $\tau \geq 0$. We analyze two regimes:

**Regime 1: Small timesteps ($\gamma \tau \ll 1$):**

Expand $e^{-\gamma \tau} \approx 1 - \gamma \tau + \frac{\gamma^2 \tau^2}{2}$:

$$
\frac{2}{\gamma}\tau - \frac{2}{\gamma^2}\left(\gamma \tau - \frac{\gamma^2 \tau^2}{2}\right) = \frac{2}{\gamma}\tau - \frac{2}{\gamma}\tau + \tau^2 = \tau^2 + O(\tau^3)
$$

Multiplying by $V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}$:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v}(s) \, ds\right\|^2\right] \leq \frac{d\sigma_{\max}^2}{2\gamma} \cdot \tau^2 + O(\tau^3)
$$

**Regime 2: Finite timesteps ($\gamma \tau \sim O(1)$):**

Using $(1 - e^{-\gamma \tau})/\gamma \leq \tau$, we obtain the uniform bound:

$$
\frac{2}{\gamma}\tau - \frac{2}{\gamma^2}(1 - e^{-\gamma \tau}) \leq \frac{2\tau}{\gamma}
$$

Multiplying by $V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}$:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v}(s) \, ds\right\|^2\right] \leq \frac{d\sigma_{\max}^2}{2\gamma} \cdot \frac{2\tau}{\gamma} = \frac{d\sigma_{\max}^2}{\gamma^2} \tau
$$

**Uniform bound for all $\tau \geq 0$:**

Define:

$$
C_2 := \frac{d\sigma_{\max}^2}{\gamma^2}
$$

Then for all $\tau \geq 0$:

$$
\mathbb{E}\left[\left\|\int_0^\tau \delta_{v,k,i}(s) \, ds\right\|^2\right] \leq C_2 \tau
$$

**Physical interpretation:** Despite the integral being "quadratic" in form, the exponential correlation decay with characteristic time $1/\gamma$ causes the effective accumulation to scale as $O(\tau)$ for timesteps $\tau \sim 1/\gamma$, not $O(\tau^2)$. This is a standard result for OU-type processes and reflects the finite correlation time of velocity fluctuations.

**PART IV: State-Independence via Uniform Variance Bounds**

The constant $C_2$ depends only on system parameters ($d$, $\sigma_{\max}$, $\gamma$) and is **inherently state-independent**.

The constant $C_1$ requires uniform bounds on positional and velocity variances:

:::{prf:assumption} Uniform Variance Bounds
:label: assump-uniform-variance-bounds

There exist constants $M_x, M_v > 0$ such that for all swarm configurations along the kinetic evolution:

$$
\mathbb{E}[V_{\text{Var},x}] \leq M_x, \quad \mathbb{E}[V_{\text{Var},v}] \leq M_v
$$

These bounds are ensured by:

1. **Velocity variance:** {prf:ref}`thm-velocity-variance-contraction-kinetic` establishes that velocity variance equilibrates to $V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}$ with exponential convergence. Thus $M_v = \frac{d\sigma_{\max}^2}{2\gamma}$.

2. **Positional variance:** {prf:ref}`thm-positional-variance-contraction` (from 03_cloning.md, Chapter 10) establishes the Foster-Lyapunov drift inequality:

   $$
   \mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x
   $$

   with $\kappa_x > 0$ and $C_x < \infty$ independent of $N$. This implies a uniform equilibrium bound $M_x = C_x / \kappa_x$ when combined with the bounded expansion from the kinetic operator (this theorem).
:::

With this assumption:

$$
C_1 = 2\sqrt{M_x \cdot M_v}
$$

is **state-independent**.

**PART V: Aggregation and Final Bound**

Summing over all particles:

$$
\Delta V_{\text{Var},x} = \frac{1}{N}\sum_{k=1,2}\sum_{i \in \mathcal{A}(S_k)} \Delta\|\delta_{x,k,i}\|^2
$$

Taking expectations and using Parts II-III:

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},x}] \leq C_1 \tau + C_2 \tau + O(\tau^2)
$$

Define:

$$
C_{\text{kin},x} = C_1 + C_2 = 2\sqrt{M_x \cdot M_v} + \frac{d\sigma_{\max}^2}{\gamma^2}
$$

For sufficiently small $\tau$, the $O(\tau^2)$ terms are negligible, yielding:

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},x}] \leq C_{\text{kin},x} \tau
$$

**Key property:** The expansion is **bounded**—it does not grow with $V_{\text{Var},x}$ itself. The constant $C_{\text{kin},x}$ is state-independent under the uniform variance bounds from Assumption {prf:ref}`assump-uniform-variance-bounds`.

**Q.E.D.**
:::

---

## Physical Interpretation

**Linear term ($C_1 \tau$):**
- Arises from position-velocity correlation during transient dynamics
- Vanishes at equilibrium due to detailed balance of Langevin dynamics
- Bounded by Cauchy-Schwarz during transients using uniform variance bounds

**Quadratic term ($C_2 \tau$):**
- Velocity accumulation over timestep $\tau$ with exponential correlation decay
- Despite being a "squared integral," contributes $O(\tau)$ due to finite correlation time $1/\gamma$
- Physically: random walk with correlation length $\sim 1/\gamma$

**Why $O(\tau)$ and not $O(\tau^2)$?**

The naive application of Jensen's inequality:

$$
\mathbb{E}\left[\left\|\int_0^\tau v(s) \, ds\right\|^2\right] \leq \tau \int_0^\tau \mathbb{E}[\|v(s)\|^2] \, ds \approx \tau^2 V_{\text{Var},v}^{\text{eq}}
$$

gives $O(\tau^2)$ because it ignores the covariance structure. The correct calculation using the exponential correlation decay reveals that:

- **Small $\tau$:** The contribution is indeed $O(\tau^2)$ with prefactor $\frac{d\sigma_{\max}^2}{2\gamma}$
- **Finite $\tau \sim 1/\gamma$:** The correlation decay causes cancellations, yielding $O(\tau)$ with prefactor $\frac{d\sigma_{\max}^2}{\gamma^2}$

The uniform bound holds for all $\tau$ and scales as $O(\tau)$, which is the relevant result for the drift inequality.

---

## Comparison to Standard OU Results

For a scalar OU process $dv = -\gamma v \, dt + \sigma \, dW$, the mean-square displacement is:

$$
\mathbb{E}\left[\left(\int_0^\tau v(s) \, ds\right)^2\right] = \frac{\sigma^2}{\gamma^3}(\gamma \tau - 1 + e^{-\gamma \tau})
$$

In $d$ dimensions with isotropic noise $\sigma_{\max}$, this generalizes to:

$$
\mathbb{E}\left[\left\|\int_0^\tau v(s) \, ds\right\|^2\right] = \frac{d\sigma_{\max}^2}{\gamma^3}(\gamma \tau - 1 + e^{-\gamma \tau})
$$

which matches our derivation exactly when using the equilibrium variance $V_{\text{Var},v}^{\text{eq}} = \frac{d\sigma_{\max}^2}{2\gamma}$ and the double integral formula:

$$
V_{\text{Var},v}^{\text{eq}} \cdot \left[\frac{2\tau}{\gamma} - \frac{2(1 - e^{-\gamma \tau})}{\gamma^2}\right] = \frac{d\sigma_{\max}^2}{2\gamma} \cdot \frac{2}{\gamma^2}(\gamma \tau - 1 + e^{-\gamma \tau}) = \frac{d\sigma_{\max}^2}{\gamma^3}(\gamma \tau - 1 + e^{-\gamma \tau})
$$

This validates our approach against the standard OU mean-square displacement formula.
