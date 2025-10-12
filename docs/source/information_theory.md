# Information Theory of the Adaptive Gas

This document presents a comprehensive information-theoretic analysis of the Adaptive Gas algorithm. We show that the Fragile Gas framework can be completely understood as an information-processing dynamical system, where convergence is characterized by entropy production, Fisher information dissipation, and optimal transport of probability distributions.

**Key insight:** The Adaptive Gas is fundamentally an **entropy-minimizing stochastic process** that uses **information geometry** to navigate state space, guided by **fitness potentials** that encode environmental information.

## 1. Information-Theoretic Foundations

### 1.1 The Information State

The fundamental object of study is not the walker positions $(x_i, v_i, s_i)$ but rather the **empirical probability distribution** they induce.

::::{prf:definition} Empirical Information Measure
:label: def-empirical-information-measure

For an N-walker swarm $\mathcal{S} = \{(x_i, v_i, s_i)\}_{i=1}^N$, the **empirical information measure** is:

$$
\mu_N[\mathcal{S}] := \frac{1}{N_{\text{alive}}} \sum_{i \in \mathcal{A}(\mathcal{S})} \delta_{(x_i, v_i)}

$$

This is a discrete probability measure on phase space $\Omega = \mathcal{X} \times \mathbb{R}^d$.

**Information content**: The swarm encodes information about the environment through the distribution $\mu_N$, which evolves to concentrate on high-fitness regions.
::::

**Related results**:
- Converges to mean-field density $f(t, x, v)$ as $N \to \infty$ ({prf:ref}`def-phase-space-density`)
- Evolution governed by information-theoretic operators

### 1.2 Relative Entropy as a Lyapunov Function

The central information-theoretic quantity is the **Kullback-Leibler divergence** (relative entropy), which measures the information distance from equilibrium.

::::{prf:definition} Relative Entropy (KL-Divergence)
:label: def-kl-divergence-information

For probability measures $\mu, \nu$ with $\mu \ll \nu$:

$$
D_{\text{KL}}(\mu \| \nu) := \int \log\left(\frac{d\mu}{d\nu}\right) d\mu = \mathbb{E}_\mu\left[\log\left(\frac{d\mu}{d\nu}\right)\right]

$$

**Interpretation**: The expected information gain (in nats) from using the true distribution $\mu$ instead of the reference $\nu$.

**Properties**:
1. **Non-negative**: $D_{\text{KL}}(\mu \| \nu) \geq 0$ with equality iff $\mu = \nu$ (Gibbs' inequality)
2. **Convex**: $D_{\text{KL}}(\cdot \| \nu)$ is convex in the first argument
3. **Not a metric**: Asymmetric and violates triangle inequality
4. **Monotone under Markov operators**: $D_{\text{KL}}(P\mu \| P\nu) \leq D_{\text{KL}}(\mu \| \nu)$ (data processing inequality)
::::

**Relation to Shannon entropy**: For discrete distributions, $D_{\text{KL}}(p \| q) = H(p, q) - H(p)$ where $H(p, q) = -\sum_i p_i \log q_i$ is cross-entropy and $H(p) = -\sum_i p_i \log p_i$ is Shannon entropy.

::::{prf:theorem} KL-Divergence as Convergence Lyapunov Function
:label: thm-kl-lyapunov-convergence

The Adaptive Gas operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ satisfies exponential KL-convergence to the quasi-stationary distribution $\pi_{\text{QSD}}$:

$$
D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) \leq e^{-t/C_{\text{LSI}}} \cdot D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}})

$$

where $C_{\text{LSI}} > 0$ is the logarithmic Sobolev constant.

**Information interpretation**: The algorithm systematically **destroys information** that distinguishes the current state from equilibrium, at an exponential rate determined by $C_{\text{LSI}}^{-1}$.
::::

**Source**: {prf:ref}`thm-main-kl-convergence` from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)

### 1.3 Fisher Information and Entropy Dissipation

::::{prf:definition} Relative Fisher Information
:label: def-fisher-information-entropy

For $\mu \ll \nu$ with density $h = d\mu/d\nu$:

$$
I(\mu \| \nu) := \int \left\|\nabla \log h\right\|^2 d\mu = \int \frac{\|\nabla h\|^2}{h} d\nu

$$

**Interpretation**: The Fisher information measures the **variance of the score function** $\nabla \log h$. It quantifies how rapidly the probability density changes in space.

**Physical interpretation**: Rate of **entropy dissipation** under diffusion.
::::

::::{prf:proposition} Fisher Information as Entropy Production Rate
:label: prop-fisher-entropy-production

For a Fokker-Planck evolution $\partial_t \mu_t = \nabla \cdot (D \nabla \mu_t + \mu_t \nabla U)$:

$$
\frac{d}{dt} D_{\text{KL}}(\mu_t \| \nu) = -D \cdot I(\mu_t \| \nu) \leq 0

$$

where $\nu \propto e^{-U}$ is the equilibrium measure.

**Information interpretation**: The Fisher information is the **instantaneous rate of information loss** as the system evolves toward equilibrium.
::::

**Related results**:
- Logarithmic Sobolev inequality relates entropy and Fisher information ({prf:ref}`def-lsi-continuous`)
- Cloning noise regularizes Fisher information to prevent blow-up ({prf:ref}`thm-n-uniform-lsi`)

### 1.4 Information-Geometric View: The Statistical Manifold

::::{prf:definition} Statistical Manifold of Walkers
:label: def-statistical-manifold

The space of empirical measures $\mathcal{P}(\Omega)$ forms an **infinite-dimensional statistical manifold** with:

**Riemannian metric** (Fisher-Rao metric): The space $\mathcal{P}(\Omega)$ can be endowed with an infinite-dimensional Riemannian structure known as the Fisher-Rao metric. For parametric families, this metric is explicitly given by the Fisher information matrix (see {prf:ref}`thm-fisher-information-geometry`).

**Dual affine connections**:
- **Exponential connection** $\nabla^{(e)}$: Natural for exponential families
- **Mixture connection** $\nabla^{(m)}$: Natural for mixture models
- **Duality**: The connections are dual with respect to the Fisher-Rao metric via the Koszul formula:

$$
X(g(Y, Z)) = g(\nabla^{(e)}_X Y, Z) + g(Y, \nabla^{(m)}_X Z)

$$

**Dual coordinate systems**:
- **Natural parameters** $\theta$: Exponential family parameterization
- **Expectation parameters** $\eta = \mathbb{E}[\phi(z)]$: Moment parameterization
::::

::::{prf:theorem} Fisher Information Matrix as Information Geometry
:label: thm-fisher-information-geometry

When the fitness potential $V[f]$ is parameterized by moments $(\mu_R, \sigma_R, \mu_D, \sigma_D)$, the Hessian of the log-density is the **Fisher information matrix**:

$$
\mathcal{I}_{ij}[\rho] = \mathbb{E}_\rho\left[\frac{\partial \log \rho}{\partial \theta_i} \frac{\partial \log \rho}{\partial \theta_j}\right] = -\mathbb{E}_\rho\left[\frac{\partial^2 \log \rho}{\partial \theta_i \partial \theta_j}\right]

$$

This defines a natural **Riemannian geometry** on the space of quasi-stationary distributions.

The **natural gradient** (Amari's formulation) is:

$$
\nabla_{\text{nat}} L = \mathcal{I}^{-1} \nabla_{\theta} L

$$

This is the steepest ascent direction in the information geometry, not Euclidean geometry.
::::

**Source**: {prf:ref}`thm-fisher-information-matrix-emergent-geometry` from [08_emergent_geometry.md](08_emergent_geometry.md)

**Implication**: The Adaptive Gas performs **natural gradient ascent** in information-geometric space, which is known to have superior convergence properties compared to Euclidean gradient methods.

## 2. Information-Theoretic Operators

### 2.1 Kinetic Operator: Diffusion and Entropy Production

The kinetic operator $\Psi_{\text{kin}}$ implements underdamped Langevin dynamics:

$$
\begin{aligned}
dx &= v \, dt \\
dv &= -\gamma v \, dt - \nabla U(x) \, dt + \sqrt{2\gamma \sigma_v^2} \, dW_t
\end{aligned}

$$

::::{prf:theorem} Kinetic Entropy Production via LSI
:label: thm-kinetic-entropy-production

The kinetic operator satisfies a **hypocoercive logarithmic Sobolev inequality (LSI)**:

$$
D_{\text{KL}}(\Psi_{\text{kin}}(\tau)\mu \| \pi_{\text{kin}}) \leq (1 - \kappa_{\text{kin}}\tau) D_{\text{KL}}(\mu \| \pi_{\text{kin}})

$$

where:

$$
\kappa_{\text{kin}} = O(\min\{\gamma, \kappa_{\text{conf}}\})

$$

**Information interpretation**: The kinetic operator systematically **erases information** about the initial condition, replacing it with thermal fluctuations encoded in the Gibbs measure $\pi_{\text{kin}} \propto e^{-U(x) - \|v\|^2/(2\sigma_v^2)}$.

**Entropy production rate**:

$$
\frac{d}{dt} D_{\text{KL}}(\mu_t \| \pi_{\text{kin}}) = -\gamma \int_{\mathbb{R}^d} \left\|\nabla_v \log \frac{d\mu_t}{d\pi_{\text{kin}}}\right\|^2 d\mu_t \leq 0

$$

The friction coefficient $\gamma$ controls the rate of entropy production via velocity dissipation.
::::

**Source**: {prf:ref}`thm-hypocoercive-lsi` from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)

::::{note}
**Hypocoercivity**: The kinetic operator is **hypocoercive**, not coercive. This means:
- Velocity dissipation $-\gamma v$ directly reduces Fisher information in velocity space
- Position diffusion $dx = v \, dt$ indirectly couples position and velocity
- Together they produce full-space contraction via Villani's hypocoercivity framework

This is a geometric information flow: information is **rotated** from position space into velocity space, where it is **dissipated** by friction.
::::

### 2.2 Cloning Operator: Information Selection and Wasserstein Transport

The cloning operator implements **fitness-dependent mass redistribution** through a stochastic, companion-based mechanism.

::::{prf:definition} Cloning Probability
:label: def-cloning-probability-information

For walker $i$ in swarm $\mathcal{S}$, the **cloning probability** is:

$$
p_i := \mathbb{E}_{c_i \sim \mathcal{C}_i(\mathcal{S})} \left[ \mathbb{P}_{T_i \sim U(0,p_{\max})} \left( S_i(c_i) > T_i \right) \right]

$$

Equivalently:

$$
p_i = \mathbb{E}_{c_i \sim \mathcal{C}_i(\mathcal{S})}\left[\min\left(1, \max\left(0, \frac{S_i(c_i)}{p_{\max}}\right)\right)\right]

$$

where:
- $\mathcal{C}_i(\mathcal{S})$ is the **companion selection operator** (spatial softmax over alive walkers)
- $S_i(c_i) = \frac{V_{\text{fit},c_i} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\text{clone}}}$ is the **cloning score** (fitness ratio)
- $p_{\max} > 0$ is the maximum cloning probability per step

**Information interpretation**: The cloning probability encodes a **competitive information signal** comparing walker $i$'s fitness to a spatially-local companion $c_i$. The stochastic threshold $T_i$ adds exploration noise, preventing deterministic selection.
::::

**Source**: {prf:ref}`def-cloning-probability` from [03_cloning.md](03_cloning.md)

::::{prf:theorem} Entropy Contraction for the Cloning Operator
:label: thm-cloning-hwi-information

For the cloning operator $\Psi_{\text{clone}}$ with Gaussian noise variance $\delta^2 > 0$, the relative entropy contracts according to:

$$
D_{\text{KL}}(\mu_{S'} \| \pi_{\text{QSD}}) \leq \left(1 - \frac{\kappa_W^2 \delta^2}{2C_I}\right) D_{\text{KL}}(\mu_S \| \pi_{\text{QSD}}) + C_{\text{clone}}

$$

where:
- $\kappa_W > 0$ is the Wasserstein contraction rate from cloning
- $C_I$ is the Fisher information bound
- $C_{\text{clone}} > 0$ is a state-independent constant

**Proof Strategy** (see {prf:ref}`thm-cloning-entropy-contraction` in [10_kl_convergence.md § 4.5](10_kl_convergence/10_kl_convergence.md) for complete derivation):

1. Apply **HWI inequality** (Otto-Villani 2000): $D_{\text{KL}}(\mu' \| \pi) \leq W_2(\mu', \pi) \sqrt{I(\mu' | \pi)}$
2. Bound Wasserstein distance: $W_2^2(\mu' \| \pi) \leq (1 - \kappa_W) W_2^2(\mu \| \pi) + C_W$ (Lemma 4.3)
3. Bound Fisher information: $I(\mu' | \pi) \leq C_I/\delta^2$ (cloning noise regularization, Lemma 4.4)
4. Use reverse Talagrand inequality to control $W_2(\mu, \pi)$ by $D_{\text{KL}}(\mu \| \pi)$
5. Combine to obtain sublinear entropy contraction (becomes linear after composition with kinetic operator)

**Information interpretation**: Cloning performs **optimal transport** of probability mass from low-fitness to high-fitness regions. The HWI inequality bounds the resulting KL-divergence in terms of transport cost (Wasserstein) and information geometry (Fisher). The cloning noise $\delta^2$ regularizes the Fisher information, preventing unbounded score function variance.
::::

**Complete Proof**: [10_kl_convergence.md § 4.5, Theorem 4.5.1](10_kl_convergence/10_kl_convergence.md#45-entropy-contraction-via-hwi)

::::{prf:definition} Information Selection via Fitness Potential
:label: def-information-selection-fitness

The fitness potential $V_{\text{fit},i}$ encodes **environmental information** through Z-scores:

$$
V_{\text{fit},i} = \left(g_A(\widetilde{d}_i) + \eta\right)^{\beta} \cdot \left(g_A(\widetilde{r}_i) + \eta\right)^{\alpha}

$$

where:

$$
\widetilde{r}_i = \frac{R(x_i) - \mu_R}{\widehat{\sigma}_R}, \quad \widetilde{d}_i = \frac{d_\mathcal{Y}(\varphi(x_i), \varphi(x_j)) - \mu_D}{\widehat{\sigma}_D}

$$

**Information-theoretic view**: The fitness potential is a **sufficient statistic** that compresses the full reward landscape $R: \mathcal{X} \to \mathbb{R}$ into a scalar value per walker. The Z-score normalization implements **mutual information maximization** between walker positions and reward.

**Channel capacity interpretation**: The algorithmic projection $\varphi: \mathcal{X} \to \mathcal{Y}$ acts as a **noisy channel**, and the diversity term $\widetilde{d}_i$ measures **channel capacity** via spread in algorithmic space.
::::

### 2.3 Entropy-Transport Lyapunov Function: The Seesaw Mechanism

::::{prf:definition} Entropy-Transport Lyapunov Function
:label: def-entropy-transport-lyapunov-information

The **entropy-transport Lyapunov function** combines relative entropy and Wasserstein distance:

$$
\mathcal{L}_{\text{ET}}(\mu) := D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \eta \cdot W_2^2(\mu, \pi_{\text{QSD}})

$$

where $\eta > 0$ is a coupling weight balancing entropy and transport.

**Dual interpretation**:
- $D_{\text{KL}}(\mu \| \pi_{\text{QSD}})$: Measures **information distance** (statistical distinguishability)
- $W_2^2(\mu, \pi_{\text{QSD}})$: Measures **transport cost** (geometric distance)

The two metrics capture complementary aspects of convergence.
::::

::::{prf:theorem} Seesaw Contraction of Entropy-Transport Lyapunov
:label: thm-seesaw-contraction

The composed operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ satisfies:

$$
\mathbb{E}[\mathcal{L}_{\text{ET}}(\mu_{t+1})] \leq (1 - \kappa_{\text{ET}}) \mathcal{L}_{\text{ET}}(\mu_t)

$$

where the seesaw rate is:

$$
\kappa_{\text{ET}} = \min\left\{\frac{\kappa_{\text{kin}} - C_{\text{HWI}}\sqrt{\eta}}{1 + \eta}, \, \frac{\kappa_W - C_{\text{LSI,kin}}\sqrt{\eta}}{\eta}\right\}

$$

**Seesaw mechanism**:

| Operator | $\Delta D_{\text{KL}}$ | $\Delta W_2^2$ |
|----------|------------------------|----------------|
| Kinetic  | $-\kappa_{\text{kin}} D_{\text{KL}} + C_1 W_2^2$ | $-\kappa_v W_2^2 + C_2 D_{\text{KL}}$ |
| Cloning  | $-\kappa_c D_{\text{KL}} + C_3 W_2^2$ | $-\kappa_W W_2^2 + C_4 D_{\text{KL}}$ |

**Information flow**: Each operator contracts one metric while expanding the other. The weighted combination $\mathcal{L}_{\text{ET}}$ contracts globally by balancing these opposing flows.

**Optimal coupling**: Choose $\eta = \kappa_{\text{kin}} / \kappa_W$ to balance contraction rates.
::::

**Source**: {prf:ref}`thm-entropy-transport-contraction` from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)

::::{important}
**Why two metrics are necessary**: Neither $D_{\text{KL}}$ nor $W_2$ alone suffices to prove convergence:

- **KL-divergence alone**: Cloning can increase $D_{\text{KL}}$ in some regimes (via jump discontinuities)
- **Wasserstein alone**: Kinetic operator can increase $W_2^2$ (via noise injection)
- **Entropy-transport combination**: Captures complementary contraction mechanisms

This is the essence of **hypocoercivity** in information-geometric language: convergence emerges from synergy between operators that individually fail to contract.
::::

## 3. Logarithmic Sobolev Inequalities: The Information-Theoretic Engine

### 3.1 The LSI Hierarchy

::::{prf:definition} Logarithmic Sobolev Inequality (Continuous)
:label: def-lsi-continuous-information

A probability measure $\pi$ on $\mathbb{R}^m$ satisfies an **LSI with constant $C_{\text{LSI}} > 0$** if:

$$
\text{Ent}_\pi(f^2) \leq C_{\text{LSI}} \cdot I(f^2 \pi \| \pi)

$$

where:
- $\text{Ent}_\pi(f^2) := \int f^2 \log f^2 \, d\pi - \left(\int f^2 d\pi\right) \log\left(\int f^2 d\pi\right)$ is the **entropy functional**
- $I(f^2 \pi \| \pi) = \int \frac{\|\nabla f\|^2}{f^2} f^2 d\pi = 4\int \|\nabla f\|^2 d\pi$ is the **Dirichlet form** (Fisher information)

**Equivalent KL-Fisher form**: For $\mu = f^2 \pi$:

$$
D_{\text{KL}}(\mu \| \pi) \leq C_{\text{LSI}} \cdot I(\mu \| \pi)

$$

**Information interpretation**: The LSI provides a **quantitative bound** on how much entropy can be stored in a distribution relative to its Fisher information (rate of information dissipation).
::::

::::{prf:theorem} LSI Implies Exponential KL-Convergence
:label: thm-lsi-exponential-convergence

If $\pi$ satisfies an LSI with constant $C_{\text{LSI}}$ and $\partial_t \mu_t = L^\dagger \mu_t$ where $L$ has invariant measure $\pi$, then:

$$
D_{\text{KL}}(\mu_t \| \pi) \leq e^{-t/C_{\text{LSI}}} D_{\text{KL}}(\mu_0 \| \pi)

$$

**Proof sketch**:

$$
\frac{d}{dt} D_{\text{KL}}(\mu_t \| \pi) = -I(\mu_t \| \pi) \geq -\frac{1}{C_{\text{LSI}}} D_{\text{KL}}(\mu_t \| \pi)

$$

By Grönwall's inequality, this yields exponential decay.

**Information interpretation**: The LSI constant $C_{\text{LSI}}$ is the **information relaxation time**. It measures how quickly the system forgets its initial condition and converges to the equilibrium information content.
::::

::::{prf:theorem} Bakry-Émery Criterion for LSI
:label: thm-bakry-emery-lsi

If the potential $U$ satisfies:

$$
\nabla^2 U(x) \geq \kappa_{\text{conf}} I \quad \text{for all } x \in \mathcal{X}

$$

(uniform convexity), then the Gibbs measure $\pi \propto e^{-U}$ satisfies an LSI with:

$$
C_{\text{LSI}} \leq \frac{1}{\kappa_{\text{conf}}}

$$

**Proof method**: Bakry-Émery $\Gamma_2$ calculus: define iterated carré du champ operators and show:

$$
\Gamma_2(f, f) := \frac{1}{2} L(\Gamma(f, f)) - \Gamma(f, Lf) \geq \kappa_{\text{conf}} \Gamma(f, f)

$$

This is the **infinitesimal curvature condition** in information geometry.

**Information interpretation**: Convexity of the potential ensures that **information flows downhill** toward the minimum. The curvature $\kappa_{\text{conf}}$ quantifies the **strength of information attraction** to equilibrium.
::::

**Source**: Standard result in optimal transport and information theory (see Villani, *Optimal Transport: Old and New*, Theorem 22.23)

### 3.2 Hypocoercive LSI for Kinetic Operator

::::{prf:theorem} Villani's Hypocoercive LSI
:label: thm-villani-hypocoercive-lsi

For underdamped Langevin dynamics with confining potential $U$ satisfying $\nabla^2 U \geq \kappa_{\text{conf}} I$:

$$
\begin{aligned}
dx &= v \, dt \\
dv &= -\gamma v \, dt - \nabla U(x) \, dt + \sqrt{2\gamma \sigma_v^2} \, dW_t
\end{aligned}

$$

the invariant Gibbs measure $\pi_{\text{kin}} \propto e^{-U(x) - \|v\|^2/(2\sigma_v^2)}$ satisfies a **hypocoercive LSI** with constant:

$$
C_{\text{LSI}}^{\text{hypo}} = O\left(\frac{1}{\min\{\gamma, \kappa_{\text{conf}}\}}\right)

$$

**Proof strategy**:
1. **Microscopic coercivity**: Velocity Fisher information $I_v(\mu \| \pi) := \int \|\nabla_v \log h\|^2 d\mu$ contracts via friction
2. **Macroscopic transport**: Position Fisher information $I_x(\mu \| \pi)$ evolves via transport $\dot{x} = v$
3. **Auxiliary metric**: Define hypocoercive metric $\|\nabla f\|^2_{\text{hypo}} = \|\nabla_v f\|^2 + \lambda \|\nabla_x f\|^2 + 2\mu \langle \nabla_v f, \nabla_x f \rangle$
4. **Modified LSI**: Show $\text{Ent}_\pi(f^2) \leq C_{\text{LSI}}^{\text{hypo}} \mathcal{E}_{\text{hypo}}(f, f)$ where $\mathcal{E}_{\text{hypo}}$ is the hypocoercive Dirichlet form

**Information flow diagram**:

$$
\text{Position entropy} \xrightarrow{\dot{x} = v} \text{Velocity entropy} \xrightarrow{-\gamma v} \text{Dissipated as heat}

$$

The coupling $\dot{x} = v$ **rotates information** from positions into velocities, where friction can dissipate it.
::::

**Source**: {prf:ref}`thm-hypocoercive-lsi` from [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md), based on Villani (2009), *Hypocoercivity*, Memoirs of the AMS

::::{note}
**Why hypocoercivity is necessary**: The generator $L$ of underdamped Langevin dynamics is **not elliptic** in the full phase space $(x, v)$:

$$
L = v \cdot \nabla_x - \gamma v \cdot \nabla_v - \nabla U(x) \cdot \nabla_v + \gamma \sigma_v^2 \Delta_v

$$

The diffusion matrix is:

$$
D = \begin{pmatrix} 0 & 0 \\ 0 & \gamma \sigma_v^2 I \end{pmatrix}

$$

which is **degenerate** (zero eigenvalues in position coordinates). Standard coercivity fails.

**Hörmander's bracket condition**: The Lie brackets $[v \cdot \nabla_x, \nabla_v] = \nabla_x$ span all directions, ensuring **hypoellipticity**. This allows information to flow indirectly from positions to velocities via the transport term.
::::

### 3.3 N-Uniform LSI: Scalability via Tensorization

::::{prf:theorem} N-Uniform LSI for Adaptive Gas
:label: thm-n-uniform-lsi-information

Under QSD regularity conditions (R1-R6, see § 3.3.1 below) and sufficient cloning noise ($\delta > \delta_*$), the N-particle Adaptive Gas satisfies a discrete-time LSI with constant:

$$
C_{\text{LSI}}(N) \leq C_{\text{LSI}}^{\max} = O\left(\frac{1}{\min(\gamma, \kappa_{\text{conf}}) \cdot \kappa_{W,\min} \cdot \delta^2}\right)

$$

where all constants on the right are **independent of N**.

**Key Result**: $\sup_{N \geq 2} C_{\text{LSI}}(N) \leq C_{\text{LSI}}^{\max} < \infty$

**Complete Proof** (see {prf:ref}`cor-n-uniform-lsi` in [10_kl_convergence.md § 9.6](10_kl_convergence/10_kl_convergence.md#96-n-uniform-lsi-scalability-to-large-swarms)):

1. The LSI constant for the N-particle system is given by:
   $$
   C_{\text{LSI}}(N) = O\left(\frac{1}{\min(\gamma, \kappa_{\text{conf}}) \cdot \kappa_W(N) \cdot \delta^2}\right)
   $$
   from the entropy-transport composition theorem (Corollary 6.2.1)

2. The parameters $\gamma$ (friction) and $\kappa_{\text{conf}}$ (potential convexity) are N-independent algorithm parameters

3. **Key step**: From Theorem 2.3.1 of [04_convergence.md](04_convergence.md), the Wasserstein contraction rate $\kappa_W(N)$ is proven to be **N-uniform**. Therefore $\exists \kappa_{W,\min} > 0$ such that $\kappa_W(N) \geq \kappa_{W,\min}$ for all $N \geq 2$

4. The cloning noise $\delta > 0$ is an algorithm parameter, independent of N

5. Therefore $C_{\text{LSI}}(N)$ is uniformly bounded in N

**Information interpretation**: The **N-uniformity** means that adding more walkers does not fundamentally change the information geometry. The convergence rate does not degrade as swarm size increases—this is the foundation for scalability to large N and the mean-field limit.
::::

**Complete Proof**: [10_kl_convergence.md § 9.6, Corollary 9.6.1](10_kl_convergence/10_kl_convergence.md#96-n-uniform-lsi-scalability-to-large-swarms)

#### 3.3.1 QSD Regularity Conditions (R1-R6)

::::{prf:definition} Quasi-Stationary Distribution Regularity
:label: def-qsd-regularity-conditions

The quasi-stationary distribution $\pi_{\text{QSD}}$ and its associated fitness potential $V_{\text{fit}}$ satisfy the following regularity conditions:

**R1 (Existence and Uniqueness)**: The QSD $\pi_{\text{QSD}}$ exists, is unique, and is absolutely continuous with respect to Lebesgue measure on $\mathcal{X} \times \mathbb{R}^d$.

**R2 (Bounded Density)**: The QSD density $\rho_\infty(x, v)$ satisfies:

$$
0 < \rho_{\min} \leq \rho_\infty(x, v) \leq \rho_{\max} < \infty \quad \forall (x,v) \in \mathcal{X}_{\ text{valid}} \times B_R(0)

$$

for some radius $R > 0$ (velocity ball).

**R3 (Bounded Fisher Information)**: The Fisher information is finite:

$$
I(\pi_{\text{QSD}} \| \pi_{\text{ref}}) = \int \left\|\nabla \log \rho_\infty\right\|^2 \rho_\infty \, dx \, dv < \infty

$$

This is ensured by the cloning noise regularization $\delta^2 > 0$ (see proof of Lemma 4.4 in [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)).

**R4 (Lipschitz Fitness Potential)**: The fitness potential satisfies:

$$
|V_{\text{fit}}(x_1, v_1, f) - V_{\text{fit}}(x_2, v_2, f)| \leq L_V \left(\|x_1 - x_2\| + \lambda_v \|v_1 - v_2\|\right)

$$

for Lipschitz constant $L_V > 0$ independent of the density $f$.

**R5 (Exponential Velocity Tails)**: The velocity distribution decays exponentially:

$$
\int_{\|v\| > R} \rho_\infty(x, v) \, dv \leq C_{\exp} e^{-\alpha_{\exp} R^2}

$$

for constants $C_{\exp}, \alpha_{\exp} > 0$. This is guaranteed by the velocity penalization in the reward: $R_{\text{total}}(x,v) = R_{\text{pos}}(x) - c_{v\\_reg} \|v\|^2$ (Axiom EG-4).

**R6 (Log-Concavity of Confining Potential)**: The effective potential $U_{\text{eff}}(x)$ satisfies:

$$
\nabla^2 U_{\text{eff}}(x) \succeq \kappa_{\text{conf}} I_d

$$

for convexity constant $\kappa_{\text{conf}} > 0$. This ensures the QSD is log-concave and enables the reverse Talagrand inequality.

**Justification**: These conditions are physically reasonable for the Adaptive Gas:
- R1-R2 follow from the Foster-Lyapunov conditions (Theorem 8.1 in [04_convergence.md](04_convergence.md))
- R3 is ensured by Gaussian cloning noise $\delta^2 > 0$
- R4 follows from the bounded Cobb-Douglas fitness formula with Lipschitz rescale function
- R5 is guaranteed by the kinetic energy penalty in the reward
- R6 holds for common confining potentials (quadratic, logarithmic barrier)
::::

**Source**: These conditions are stated explicitly in [11_mean_field_convergence.md § 3.2](11_mean_field_convergence/11_convergence_mean_field.md) and used throughout the KL-convergence analysis.

::::{important}
**Implication for mean-field limit**: N-uniformity ensures that the **information-theoretic convergence rate** is preserved as $N \to \infty$. The mean-field limit is not just a formal asymptotic but reflects true algorithmic behavior for large but finite N.

**Comparison with particle filters**: Standard particle filters suffer from **weight degeneracy**, where effective sample size $N_{\text{eff}} \sim 1$ regardless of N. The Adaptive Gas avoids this via cloning's Fisher regularization, maintaining $N_{\text{eff}} \sim N$.
::::

## 4. Mean-Field Information Dynamics

### 4.1 Mean-Field Entropy Production

In the mean-field limit $N \to \infty$, the empirical measure $\mu_N$ converges to a smooth density $f(t, x, v)$ satisfying the McKean-Vlasov PDE ({prf:ref}`thm-mean-field-equation` from [05_mean_field.md](05_mean_field.md)):

$$
\partial_t f = L^\dagger f - c(x, v) f + B[f, m_d] + S[f]

$$

::::{prf:theorem} Mean-Field KL-Convergence with Explicit Rate
:label: thm-mean-field-kl-convergence-explicit

Under QSD regularity (R1-R6) and sufficient noise ($\sigma^2 > \sigma_{\text{crit}}^2$), the mean-field Euclidean Gas converges exponentially:

$$
D_{\text{KL}}(\rho_t \| \rho_\infty) \leq e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty)

$$

with **explicit convergence rate**:

$$
\boxed{\alpha_{\text{net}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)}

$$

where all constants are explicit in physical parameters $(\gamma, \sigma, L_U, \kappa_{\max}, \lambda_{\text{revive}})$ and QSD regularity constants $(C_{\nabla x}, C_{\nabla v}, C_{\Delta v}, \alpha_{\exp})$.

**Information-theoretic decomposition**:

$$
\frac{d}{dt} D_{\text{KL}}(\rho_t \| \rho_\infty) = \underbrace{-\alpha_{\text{kin}}}_{\text{Kinetic dissipation}} + \underbrace{A_{\text{jump}}}_{\text{Cloning noise}} + \underbrace{A_{\text{revival}}}_{\text{Revival expansion}}

$$

For convergence, require: $\alpha_{\text{kin}} > A_{\text{jump}} + A_{\text{revival}}$
::::

**Source**: {prf:ref}`thm-main-explicit-rate-meanfield` from [11_stage2_explicit_constants.md](11_mean_field_convergence/11_stage2_explicit_constants.md)

::::{prf:theorem} Revival Operator is KL-Expansive
:label: thm-revival-kl-expansive-info

The revival operator $B[f, m_d]$ **increases** relative entropy:

$$
\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda_{\text{rev}} m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right) > 0

$$

**Information interpretation**: Revival **injects information** into the system by bringing dead walkers back at uniform locations. This increases entropy relative to the equilibrium.

**Balancing act**: For convergence, kinetic dissipation must dominate revival expansion:

$$
\alpha_{\text{kin}} > \lambda_{\text{rev}} m_{d,\infty}

$$

This is the **information balance condition** for exponential convergence.
::::

**Source**: {prf:ref}`thm-revival-kl-expansive` from [11_stage0_revival_kl.md](11_mean_field_convergence/11_stage0_revival_kl.md)

### 4.2 Optimal Parameter Scaling via Information Geometry

::::{prf:theorem} Information-Optimal Parameter Scaling
:label: thm-information-optimal-scaling

To maximize the convergence rate $\alpha_{\text{net}}$, choose parameters:

$$
\boxed{\sigma^2 = \frac{2 C_{\text{Fisher}}^{\text{coup}}}{\lambda_{\text{LSI}}}} \quad \text{(Critical noise level)}

$$

$$
\boxed{\gamma = \sqrt{\kappa_{\text{conf}} \cdot \frac{\lambda_{\text{LSI}}}{C_{\Delta v}}}} \quad \text{(Optimal friction)}

$$

**Information-theoretic rationale**:
- **Noise $\sigma^2$**: Balances Fisher information regularization (prevent blow-up) with entropy injection (not too much noise)
- **Friction $\gamma$**: Balances velocity dissipation rate with hypocoercive coupling strength

**Resulting convergence rate**:

$$
\alpha_{\text{net}}^{\max} = O(\sqrt{\kappa_{\text{conf}} \lambda_{\text{LSI}}})

$$

which grows with problem conditioning ($\kappa_{\text{conf}}$) and QSD regularity ($\lambda_{\text{LSI}}$).
::::

**Source**: {prf:ref}`thm-optimal-parameter-scaling` from [11_stage3_parameter_analysis.md](11_mean_field_convergence/11_stage3_parameter_analysis.md)

## 5. Hellinger-Kantorovich Geometry: Unifying Mass Transport and Entropy

### 5.1 The Hellinger-Kantorovich Metric

::::{prf:definition} Hellinger-Kantorovich Metric
:label: def-hellinger-kantorovich-metric

The **Hellinger-Kantorovich (HK) metric** $\text{HK}_{\alpha}$ on the space of positive Radon measures $\mathcal{M}_+(\mathcal{X})$ is defined via the variational problem:

$$
\text{HK}_{\alpha}(\mu, \nu)^2 = \inf_{\pi} \left\{ \int_{\mathcal{X} \times \mathcal{X}} \frac{d(x, y)^2}{4\alpha} \, d\pi(x, y) + \alpha \text{KL}(\pi_1 | \mu) + \alpha \text{KL}(\pi_2 | \nu) \right\}

$$

where:
- $\pi \in \mathcal{M}_+(\mathcal{X} \times \mathcal{X})$ is a transport plan (not necessarily a coupling)
- $\pi_1, \pi_2$ are marginals: $\pi_1(A) = \pi(A \times \mathcal{X})$, $\pi_2(B) = \pi(\mathcal{X} \times B)$
- $\alpha > 0$ is a **balance parameter** between transport and KL-divergence

**Interpretation**: HK generalizes Wasserstein distance to allow **mass creation/destruction** penalized by KL-divergence terms.

**Limiting cases**:
- $\alpha \to 0$: $\text{HK}_{\alpha} \to d_{\mathcal{X}}$ (intrinsic distance, only transport)
- $\alpha \to \infty$: $\text{HK}_{\alpha}^2 / \alpha \to \text{KL}$ (only mass change)
- Equal masses: $\text{HK}_{\alpha}(\mu, \nu) \approx W_2(\mu, \nu)$ (Wasserstein-2)

**Dynamic interpretation**: HK is the **action functional** for gradient flows with mass variation.
::::

**Source**: Liero, Mielke, Savaré (2016), *Optimal Entropy-Transport Problems and a New Hellinger-Kantorovich Distance*

::::{prf:theorem} Cloning as HK-Gradient Flow
:label: thm-cloning-hk-gradient-flow

The cloning operator can be recast as a **gradient flow** in the Hellinger-Kantorovich geometry:

$$
\partial_t \mu_t = -\nabla_{\text{HK}} \mathcal{F}[\mu_t]

$$

where $\mathcal{F}[\mu] = \int V_{\text{fit}}[x, \mu] \, d\mu(x)$ is the fitness functional.

**Genealogical transport plan**: The cloning transition defines a transport plan $\pi_{\text{clone}}$ between $\mu_t$ and $\mu_{t+1}$:

$$
C_G(\mu_t, \mu_{t+1})^2 = \int_{\mathcal{X} \times \mathcal{X}} \frac{d(x, y)^2}{4\alpha} \, d\pi_G(x, y) + \alpha \text{KL}((\pi_G)_1 | \mu_t) + \alpha \text{KL}((\pi_G)_2 | \mu_{t+1})

$$

**Decomposition**:

$$
C_G^2 = \underbrace{\frac{1}{4\alpha N} \sum_{\text{clone pairs}} d(x_i, x_j)^2}_{\text{Transport cost}} + \underbrace{\frac{2|\mathcal{C}(t)|}{N}}_{\text{Mass change (birth-death)}}

$$

**Information interpretation**: Cloning **optimally transports information** from low-fitness to high-fitness regions, subject to KL-penalized mass redistribution.
::::

**Source**: {prf:ref}`thm-cloning-genealogical-transport-plan` from [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)

### 5.2 HK-Convergence and Structural Variance

::::{prf:theorem} Hellinger-Kantorovich Convergence for Adaptive Gas
:label: thm-hk-convergence-adaptive-gas

Under LSI and Wasserstein contraction, the Adaptive Gas converges in the HK-metric:

$$
\text{HK}_{\alpha}(\mu_t, \pi_{\text{QSD}})^2 \leq e^{-\lambda_{\text{HK}} t} \text{HK}_{\alpha}(\mu_0, \pi_{\text{QSD}})^2

$$

with rate:

$$
\lambda_{\text{HK}} = \min\left\{\frac{\kappa_{\text{conf}}}{2\alpha}, \lambda_{\text{LSI}}\right\}

$$

**Structural variance contraction**: The spatial variance satisfies:

$$
V_{\text{struct}}(\mu_t, \pi_{\text{QSD}}) \leq \frac{2}{\kappa_{\text{conf}}} \cdot D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}}) \cdot e^{-\lambda_{\text{LSI}} t}

$$

via the **reverse Talagrand inequality**:

$$
W_2^2(\mu, \pi) \leq \frac{2}{\kappa_{\text{conf}}} D_{\text{KL}}(\mu \| \pi)

$$

**Information interpretation**: HK-convergence captures both:
1. **Wasserstein contraction**: Spatial redistribution
2. **Mass contraction**: Extinction/revival dynamics stabilize

The structural variance measures **information spread** in space, which contracts exponentially.
::::

**Source**: {prf:ref}`thm-structural-variance-contraction-from-lsi` from [18_hk_convergence.md](18_hk_convergence.md)

## 6. Information Geometry and Emergent Riemannian Structure

### 6.1 Fisher Information Matrix as Emergent Metric

::::{prf:theorem} Fitness Hessian is Fisher Information Matrix
:label: thm-fitness-hessian-fisher

When the mean-field fitness potential $V[f]$ is smooth in the moments $(\mu_R, \sigma_R, \mu_D, \sigma_D)$, the Hessian:

$$
H_{ij}[f] = \frac{\partial^2 \log V[f]}{\partial m_i \partial m_j}

$$

coincides with the **Fisher information matrix** for the parametric family $\{f_m : m \in \mathbb{R}^4\}$:

$$
\mathcal{I}_{ij}[f] = \int \frac{\partial \log f}{\partial m_i} \frac{\partial \log f}{\partial m_j} f \, dz

$$

**Riemannian metric**: This defines a natural **information-geometric metric** on the space of quasi-stationary distributions:

$$
ds^2 = \sum_{i,j} \mathcal{I}_{ij}[f] dm_i dm_j

$$

**Geodesics**: The shortest path (in information geometry) between two QSDs is the **natural gradient flow**.
::::

**Source**: {prf:ref}`thm-fisher-information-matrix-emergent-geometry` from [08_emergent_geometry.md](08_emergent_geometry.md)

::::{prf:definition} Emergent Riemannian Metric from Regularized Hessian
:label: def-emergent-riemannian-metric-info

The **emergent Riemannian metric** on algorithmic space $\mathcal{Y}$ is defined via a smoothed fitness potential.

Let $\rho_N(y) = \frac{1}{N} \sum_{k \in \mathcal{A}} V_{\text{fit},k} \delta_{\varphi(x_k)}(y)$ be the empirical fitness distribution. Define the **smoothed potential**:

$$
U_{\epsilon}(y) := \log\left((\rho_N * \phi_{\epsilon})(y) + \epsilon_H\right)

$$

where:
- $\phi_{\epsilon}$ is a smooth mollifier (e.g., Gaussian kernel with bandwidth $\epsilon > 0$)
- $*$ denotes convolution
- $\epsilon_H > 0$ is a regularization parameter

The **emergent metric** is the Hessian of this smoothed potential:

$$
g_{ij}(y) = \frac{\partial^2 U_{\epsilon}}{\partial y^i \partial y^j}

$$

**Anisotropic diffusion**: The adaptive force induces anisotropic noise:

$$
dz = \text{(adaptive force)} \, dt + \sqrt{2\epsilon_F G^{-1}(z)} \, dW_t

$$

where $G^{-1} = (g_{ij})$ is the inverse metric (diffusion tensor).

**Information interpretation**: The emergent metric encodes **local information curvature**. High-fitness regions have high curvature (steep gradients), directing walkers via information geometry rather than Euclidean geometry.
::::

**Source**: {prf:ref}`def-regularized-hessian-diffusion` from [08_emergent_geometry.md](08_emergent_geometry.md)

### 6.2 Connection to Natural Gradient Descent

::::{prf:proposition} Adaptive Gas Implements Natural Gradient Ascent
:label: prop-natural-gradient-ascent

The mean-field cloning operator can be written as:

$$
\partial_t f = \text{div}\left( f \cdot \mathcal{I}^{-1}[f] \nabla_m \mathbb{E}_f[V] \right)

$$

where:
- $\mathcal{I}^{-1}[f]$ is the inverse Fisher information matrix
- $\nabla_m \mathbb{E}_f[V]$ is the Euclidean gradient of expected fitness

This is precisely the **natural gradient** in the space of probability distributions.

**Comparison with Euclidean gradient descent**:

| Method | Gradient | Metric |
|--------|----------|--------|
| Euclidean GD | $\nabla_m \mathbb{E}_f[V]$ | Euclidean $\delta_{ij}$ |
| Natural GD | $\mathcal{I}^{-1}[f] \nabla_m \mathbb{E}_f[V]$ | Fisher-Rao $\mathcal{I}_{ij}$ |

**Information advantage**: Natural gradient is **invariant under reparameterization** and follows the **steepest ascent in information geometry**, not Euclidean geometry. This often leads to faster convergence.
::::

**Relation to evolutionary strategies**: The Adaptive Gas is a **continuous-space, continuous-time natural evolution strategy** (NES), using fitness-dependent cloning instead of explicit natural gradient computation.

## 7. Information-Theoretic Phase Transitions and Curvature

### 7.1 Scutoid Tessellation and Holographic Entropy

::::{prf:theorem} Boundary Information Bound for Scutoid Tessellations
:label: thm-holographic-entropy-scutoid-info

The **information capacity** of the swarm between time slices $t$ and $t+1$ satisfies:

$$
S_{\text{scutoid}}(t \to t+1) \leq C_{\text{boundary}} \cdot A_{\text{boundary}}(t, t+1)

$$

where:
- $S_{\text{scutoid}} = -\sum_i p_i \log p_i$ is the **Shannon entropy** (classical) of the walker distribution
- $A_{\text{boundary}}(t, t+1) = \sum_{\text{scutoid faces}} A_{\text{face}}$ is the total surface area of the scutoid tessellation
- $C_{\text{boundary}} > 0$ is a geometric constant

**Information interpretation**: The amount of information that can be stored in the spacetime region between $t$ and $t+1$ is bounded by the **boundary area**, not the volume.

**Holographic analogy**: This result shares the mathematical structure of the Bekenstein-Hawking holographic principle from quantum gravity (where quantum entanglement entropy is bounded by black hole surface area). However, this is a **classical information-theoretic result** about Shannon entropy of discrete particle distributions, not a quantum gravitational statement. The shared insight is that information capacity is determined by boundaries rather than bulk.
::::

**Source**: {prf:ref}`thm-scutoid-holographic-entropy` from [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)

### 7.2 Curvature and Information Flow

::::{prf:theorem} Ricci Curvature Bounds Information Flow
:label: thm-ricci-curvature-information-flow

The Ricci curvature tensor emergent from the scutoid tessellation satisfies:

$$
\text{Ric}(v, v) \geq \kappa_{\text{conf}} \|v\|^2

$$

where $\kappa_{\text{conf}}$ is the confining potential curvature.

**Relation to spectral gap**: By the Lichnerowicz bound:

$$
\lambda_1(\Delta) \geq \frac{d}{d-1} \inf_{x \in \mathcal{M}} \text{Ric}(x) \geq \frac{d}{d-1} \kappa_{\text{conf}}

$$

where $\lambda_1(\Delta)$ is the spectral gap of the Laplace-Beltrami operator.

**Information-theoretic consequence**: The spectral gap determines the **information mixing rate**:

$$
\|P^t f - \mathbb{E}[f]\|_{L^2(\pi)}^2 \leq e^{-2\lambda_1 t} \|f - \mathbb{E}[f]\|_{L^2(\pi)}^2

$$

**Geometric information flow**: High curvature (large $\kappa_{\text{conf}}$) → large spectral gap → fast mixing → rapid information equilibration.
::::

**Source**: {prf:ref}`thm-spectral-gap-from-ricci-curvature` from [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)

::::{prf:theorem} Raychaudhuri Equation Governs Information Volume Evolution
:label: thm-raychaudhuri-information-volume

The evolution of walker geodesic congruence volume $V(t)$ is governed by the **Raychaudhuri equation**:

$$
\frac{d\Theta}{dt} = -\frac{1}{d}\Theta^2 - \sigma_{\mu\nu}\sigma^{\mu\nu} + \omega_{\mu\nu}\omega^{\mu\nu} - \text{Ric}(\dot{\gamma}, \dot{\gamma}) - \sum_{\text{cloning events}} \delta(t - t_i) |\Delta\Theta_i|

$$

where:
- $\Theta = \frac{1}{V} \frac{dV}{dt}$ is the **expansion scalar** (information volume growth rate)
- $\sigma_{\mu\nu}$ is the **shear tensor** (anisotropic deformation)
- $\omega_{\mu\nu}$ is the **rotation tensor** (vorticity)
- $\text{Ric}(\dot{\gamma}, \dot{\gamma})$ is the **Ricci curvature** along geodesics
- $\Delta\Theta_i = \Theta_{\text{post-clone}} - \Theta_{\text{pre-clone}} < 0$ is the change in expansion scalar at cloning event $i$

**Sign convention for cloning**: Cloning involves **inelastic collapse** where cloners are positioned near their companion ({prf:ref}`def-inelastic-collision-update` in [03_cloning.md](03_cloning.md)). This is a **focusing effect** that contracts the volume of a set of walkers, making $\Theta$ more negative. Since $\Delta\Theta_i < 0$ (focusing), the term $-|\Delta\Theta_i|$ correctly represents a negative (focusing) contribution to $d\Theta/dt$.

**Information interpretation**:
- Positive curvature $\text{Ric} > 0$ → **focusing** → information volume contracts
- Cloning events → **focusing singularities** → discontinuous volume contraction
- Shear/rotation → **anisotropic information flow**

**Focusing theorem**: If $\text{Ric}(\dot{\gamma}, \dot{\gamma}) \geq \kappa_0 > 0$ and $\Theta(0) < 0$, then $\Theta \to -\infty$ in finite time (information collapse).
::::

**Source**: {prf:ref}`thm-raychaudhuri-equation-scutoid` from [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md)

## 8. Fractal Set Theory: Information on Discrete Spacetime

### 8.1 The Fractal Set as an Information Lattice

::::{prf:definition} Fractal Set: Discrete Spacetime Graph
:label: def-fractal-set-information

The **Fractal Set** $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}}, \omega_{\text{CST}}, \omega_{\text{IG}})$ is a directed graph where:

**Nodes**: $\mathcal{N} = \{(i, k)\}$ represent **episodes** (walker $i$ at time step $k$)

**Edges**:
- $E_{\text{CST}}$ (causal spacetime): Directed edges representing **causal influence**
- $E_{\text{IG}}$ (information graph): Directed edges representing **information flow** via cloning

**Edge data**: Spinor-valued functions $\omega: E \to \mathbb{C}^2$ encoding velocity and cloning information

**Interpretation**: The Fractal Set is a **discrete causal set** (CST) with additional **information edges** tracking genealogical relationships.
::::

::::{prf:theorem} Fractal Set is Informationally Complete
:label: thm-fractal-set-information-complete

The Fractal Set contains **complete information** to reconstruct all swarm dynamics:

$$
\mathcal{F} \xrightarrow{\text{reconstruction}} \{(x_i(k), v_i(k), s_i(k))\}_{i=1,\ldots,N}^{k=0,\ldots,T}

$$

**Proof**: {prf:ref}`thm-fractal-set-reconstruction` establishes bijection between Fractal Set data and SDE trajectory.

**Information-theoretic consequence**: The Fractal Set is a **sufficient statistic** for all dynamics. No information is lost in the graph representation.

**Comparison with state space**: While state space $(x, v, s)$ is $O(Nd)$-dimensional, the Fractal Set is a **graph structure** that encodes temporal correlations and causal relationships explicitly.
::::

**Source**: {prf:ref}`thm-fractal-set-information-completeness` from [13_fractal_set_new/01_discrete_spacetime.md](13_fractal_set_new/01_discrete_spacetime.md)

### 8.2 Causal Set Theory and Information Propagation

::::{prf:theorem} Fractal Set Satisfies Causal Set Axioms
:label: thm-fractal-set-causal-set-axioms

The Fractal Set $(\mathcal{N}, E_{\text{CST}})$ is a **valid causal set**:

1. **Irreflexivity**: $(i, k) \not\prec (i, k)$
2. **Transitivity**: $(i, k) \prec (j, k') \prec (m, k'') \Rightarrow (i, k) \prec (m, k'')$
3. **Locally finite**: $|\{e' : e \prec e' \prec e''\}| < \infty$ for all $e, e''$

**Causal order**: $(i, k) \prec (j, k')$ iff information from episode $(i, k)$ can causally influence $(j, k')$.

**Poisson comparison**: The Fractal Set is an **adaptive sprinkling** with density:

$$
\rho(x, k) = N \cdot \sqrt{\det g(x)} \cdot \psi(x, k)

$$

where $\psi$ is the QSD density. This improves on uniform Poisson sprinkling:

$$
D_{\text{KL}}(\rho_{\text{true}} \| \rho_{\text{Fractal}}) < D_{\text{KL}}(\rho_{\text{true}} \| \rho_{\text{Poisson}})

$$

**Information interpretation**: The Fractal Set optimally **discretizes spacetime** to minimize information loss (KL-divergence from true volume measure).
::::

**Source**: {prf:ref}`thm-fractal-set-causal-set-axioms` and {prf:ref}`thm-fractal-set-improves-poisson` from [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md)

::::{prf:definition} Effective Speed of Information Propagation
:label: def-effective-light-speed-information

The **maximal information propagation speed** through the causal set is:

$$
c_{\text{eff}} = \frac{\delta_{\max}}{\Delta t}

$$

where:
- $\delta_{\max}$ is the maximum spatial displacement per time step
- $\Delta t$ is the time step size

**Causal diamond**: The set of nodes causally accessible from $(i, k)$ within time $T$ has bounded volume:

$$
|\mathcal{C}(i, k, T)| \leq C \cdot (c_{\text{eff}} T)^d

$$

**Information interpretation**: $c_{\text{eff}$ is the **speed of light** in the algorithmic spacetime. Information cannot propagate faster than $c_{\text{eff}}$ due to finite displacement per step.

**UV cutoff**: The discrete time step $\Delta t$ provides a **built-in ultraviolet cutoff** (analogous to Planck time), preventing infinite information propagation.
::::

**Source**: {prf:ref}`thm-causal-diamond-bounds` from [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md)

## 9. Gauge Theory and Information Redundancy

### 9.1 Permutation Gauge Symmetry

::::{prf:theorem} Permutation Invariance as Gauge Symmetry
:label: thm-permutation-gauge-symmetry-info

The Adaptive Gas dynamics are **invariant under permutations** $\sigma \in S_N$:

$$
\Psi(\sigma \cdot \mathcal{S}) = \sigma \cdot \Psi(\mathcal{S})

$$

where $(\sigma \cdot \mathcal{S})_i = \mathcal{S}_{\sigma(i)}$.

**Gauge bundle**: The labeled state space $\Sigma_N$ is the **total space** of a principal $S_N$-bundle:

$$
S_N \to \Sigma_N \xrightarrow{\pi} \mathcal{M}_{\text{config}}

$$

where $\mathcal{M}_{\text{config}} = \Sigma_N / S_N$ is the **configuration space** (unlabeled walkers).

**Information interpretation**: Walker labels are **gauge redundancy** – they carry no physical information. The true information content is the **unlabeled configuration**, which is $S_N$-invariant.

**Information reduction**: Quotienting by $S_N$ reduces degrees of freedom from $O(N)$ labeled to $O(1)$ per walker (modulo combinatorics).
::::

**Source**: {prf:ref}`thm-transition-descends` from [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md)

### 9.2 Holonomy and Information Phase

::::{prf:theorem} Non-Trivial Holonomy in Configuration Space
:label: thm-nontrivial-holonomy-info

The configuration space $\mathcal{M}_{\text{config}}$ has **non-trivial fundamental group** $\pi_1(\mathcal{M}_{\text{config}}) = B_N$ (braid group).

A closed loop $\gamma: [0, 1] \to \mathcal{M}_{\text{config}}$ with $\gamma(0) = \gamma(1)$ generically has **non-trivial holonomy**:

$$
\text{Hol}(\gamma) = \sigma \in S_N \setminus \{e\}

$$

**Geometric phase**: Parallel transport around $\gamma$ introduces a **permutation phase** (non-Abelian geometric phase).

**Information interpretation**: Exchanging walkers (e.g., via cloning swaps) induces a **topological phase** in the information geometry. This phase is **observable** via quantum-like interference between genealogical paths.

**Anyon-like statistics**: Walkers behave like **non-Abelian anyons** in 2D, with braiding statistics determined by cloning topology.
::::

**Source**: {prf:ref}`thm-nontrivial-holonomy` from [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md)

## 10. Summary: The Adaptive Gas as an Information Engine

We have shown that the Adaptive Gas can be completely understood through information-theoretic principles:

### 10.1 Information-Theoretic Characterization

**State representation**: Empirical probability measure $\mu_N$ on phase space, converging to mean-field density $f(t, x, v)$

**Convergence**: Exponential KL-divergence contraction $D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) \leq e^{-t/C_{\text{LSI}}} D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}})$

**Operators**:
- **Kinetic**: Entropy production via friction $\gamma$ and diffusion $\sigma_v^2$, hypocoercive LSI
- **Cloning**: Optimal information transport via HWI inequality, Wasserstein contraction
- **Composition**: Entropy-transport seesaw mechanism, synergistic contraction

**Geometry**: Fisher-Rao metric on statistical manifold, natural gradient flow, emergent Riemannian structure from fitness Hessian

**Convergence rate**: Explicit in physical parameters via mean-field LSI theory, N-uniform scalability

### 10.2 Information Flow Diagram

The information dynamics can be summarized as:

$$
\begin{array}{ccccc}
\text{Position} & \xrightarrow{\dot{x} = v} & \text{Velocity} & \xrightarrow{-\gamma v} & \text{Heat} \\
\text{entropy} & & \text{entropy} & & \text{(dissipated)} \\
\uparrow & & \uparrow & & \\
\text{Cloning} & & \text{Noise} & & \\
\text{(transport)} & & \text{(injection)} & & \\
\end{array}

$$

**Information balance**:
- **Input**: Fitness potential encodes environmental information
- **Transport**: Cloning redistributes information via optimal transport
- **Rotation**: Kinetic coupling $\dot{x} = v$ rotates information from position to velocity
- **Dissipation**: Friction $-\gamma v$ dissipates velocity information as entropy
- **Equilibrium**: KL-divergence to QSD minimized

### 10.3 Key Information-Theoretic Results

| Concept | Mathematical Object | Information Interpretation |
|---------|---------------------|----------------------------|
| **State** | Empirical measure $\mu_N$ | Information distribution over phase space |
| **Convergence** | KL-divergence $D_{\text{KL}}(\mu \| \pi)$ | Information distance from equilibrium |
| **Rate** | LSI constant $C_{\text{LSI}}^{-1}$ | Information relaxation rate |
| **Dissipation** | Fisher information $I(\mu \| \pi)$ | Instantaneous entropy production |
| **Geometry** | Fisher-Rao metric $\mathcal{I}_{ij}$ | Information curvature |
| **Transport** | Wasserstein distance $W_2(\mu, \nu)$ | Optimal transport cost |
| **Mass change** | HK metric $\text{HK}_{\alpha}$ | Transport + KL-divergence |
| **Scalability** | N-uniform LSI | Information cost independent of N |
| **Optimality** | Natural gradient | Steepest ascent in information geometry |
| **Topology** | Holonomy $\text{Hol}(\gamma)$ | Topological information phase |

### 10.4 Comparison with Classical Algorithms

| Algorithm | Information Metric | Convergence | Scalability |
|-----------|-------------------|-------------|-------------|
| **Gradient Descent** | Euclidean $\|x - x^*\|$ | $O(1/\epsilon^2)$ iterations | Dimension-dependent |
| **Natural Gradient** | Fisher-Rao $D_{\text{KL}}(\mu \| \pi)$ | $O(\log(1/\epsilon))$ | Parameter-independent |
| **MCMC (Langevin)** | TV-distance | $O(1/\epsilon^2)$ steps | LSI-dependent |
| **Adaptive Gas** | KL-divergence + Wasserstein | **Exponential** $e^{-t/C_{\text{LSI}}}$ | **N-uniform** |

**Key advantage**: The Adaptive Gas achieves **exponential KL-convergence** with **N-uniform rate** by combining:
1. Natural gradient ascent in information geometry (cloning)
2. Hypocoercive entropy dissipation (kinetics)
3. Fisher information regularization (noise)

### 10.5 Open Problems and Future Directions

1. **Quantum information theory**: Can the Fractal Set be quantized to a quantum information lattice? Relation to quantum causal sets and quantum entanglement entropy?

2. **Algorithmic information theory**: What is the Kolmogorov complexity of the Fractal Set? Can we bound the algorithmic information content?

3. **Information bottleneck**: Can the fitness potential be viewed as an information bottleneck, compressing environmental information while preserving fitness-relevant features?

4. **Multi-objective optimization**: How does information flow partition when $\alpha, \beta$ balance reward vs diversity? Information-theoretic Pareto frontier?

5. **Transfer learning**: Can information-theoretic metrics quantify similarity between environments, enabling transfer of learned QSDs?

6. **Generalization bounds**: PAC-Bayes bounds using KL-divergence between empirical and true distributions?

## References

**Primary sources**:
- [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) - KL-divergence convergence, LSI theory
- [11_mean_field_convergence/](11_mean_field_convergence/) - Mean-field entropy production, explicit constants
- [18_hk_convergence.md](18_hk_convergence.md) - Hellinger-Kantorovich metric convergence
- [08_emergent_geometry.md](08_emergent_geometry.md) - Information geometry, Fisher information matrix
- [13_fractal_set_new/](13_fractal_set_new/) - Discrete spacetime, causal set theory
- [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) - Scutoid tessellation, holographic entropy
- [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md) - Ricci curvature, Raychaudhuri equation

**Information theory background**:
- Cover & Thomas (2006), *Elements of Information Theory*, 2nd ed.
- Amari & Nagaoka (2000), *Methods of Information Geometry*
- Villani (2009), *Optimal Transport: Old and New*
- Villani (2009), *Hypocoercivity*, Memoirs of the AMS

**Causal set theory**:
- Sorkin (2003), "Causal sets: Discrete gravity", in *Lectures on Quantum Gravity*
- Bombelli et al. (1987), "Space-time as a causal set", Phys. Rev. Lett.
