# The Fractal Gas Algorithm Family

:::{div} feynman-prose
Imagine you're trying to find the highest point in a vast, fog-covered mountain range. You can't see the peaks, but you have a team of hikers scattered across the landscape. Each hiker can sense the altitude at their location and can communicate with nearby teammates. The clever strategy isn't to have everyone climb independently—it's to have hikers in promising locations "clone" themselves while hikers in dead ends "die off," with the dead hikers respawning near successful ones.

This is the essence of the Fractal Gas: a swarm of *walkers* exploring a space, where *fitness* determines who thrives and who must relocate, and *companion selection* determines who learns from whom. The result is a self-organizing search process that automatically concentrates computational effort in promising regions while maintaining diversity to avoid local traps.
:::

---

## Introduction

A **Fractal Gas** is a population-based algorithm for optimization and sampling in high-dimensional spaces. It operates through three fundamental mechanisms:

1. **Companion Selection**: Each walker probabilistically selects nearby walkers as "companions" for information exchange
2. **Fitness Evaluation**: A dual-channel fitness function balances *exploitation* (reward) and *exploration* (diversity)
3. **Cloning/Revival**: Low-fitness walkers are replaced by perturbed copies of high-fitness companions

The algorithm exhibits behavior across three timescales:

| Timescale | Description | Mathematical Object |
|-----------|-------------|---------------------|
| **Discrete** | Individual algorithmic steps | Markov transition kernel $P_\tau$ |
| **Scaling** | Many walkers, small steps | Mean-field limit, propagation of chaos |
| **Continuum** | Infinite walkers, continuous time | WFR (Wasserstein-Fisher-Rao) PDE |

:::{div} feynman-prose
The Fractal Gas sits at the intersection of several algorithmic traditions. Like *particle swarm optimization*, it uses a population of interacting agents. Like *genetic algorithms*, it employs selection and reproduction. Like *MCMC methods*, it samples from a target distribution. But unlike any of these, it achieves a precise mathematical characterization: in the continuum limit, the swarm density evolves according to a reaction-diffusion equation where mass flows toward high-fitness regions while maintaining diversity through diffusion.
:::

### Relationship to Other Algorithms

| Algorithm Family | Shared Feature | Key Difference |
|------------------|----------------|----------------|
| Particle Swarm | Population-based, local communication | Fractal Gas: explicit selection pressure, no global best |
| Genetic Algorithms | Selection and reproduction | Fractal Gas: continuous state space, soft selection |
| MCMC / Langevin | Sampling from distributions | Fractal Gas: selection creates non-equilibrium dynamics |
| Interacting Particle Systems | Mean-field limits | Fractal Gas: dual-channel fitness, momentum conservation |

---

(sec-state-space)=
## 1. State Space and Walkers

The fundamental unit of a Fractal Gas is the **walker**: an entity with position, velocity, and alive/dead status.

:::{prf:definition} Walker
:label: def-fg-walker

A **walker** is a tuple $w = (z, v, s)$ where:
- $z \in \mathcal{Z}$ is the **position** in latent space
- $v \in T_z\mathcal{Z}$ is the **velocity** (tangent vector at $z$)
- $s \in \{0, 1\}$ is the **status** (0 = dead, 1 = alive)

The latent space $\mathcal{Z}$ is equipped with a Riemannian metric $G$, inducing the inner product $\langle u, w \rangle_G = u^\top G(z) w$ for tangent vectors $u, w \in T_z\mathcal{Z}$.
:::

:::{prf:definition} Swarm State
:label: def-fg-swarm-state

A **swarm** of $N$ walkers is the tuple $\mathcal{S} = (w_1, \ldots, w_N)$ with state space

$$
\Sigma_N = (\mathcal{Z} \times T\mathcal{Z} \times \{0,1\})^N.
$$

For a swarm $\mathcal{S}$, we define:
- **Alive set**: $\mathcal{A}(\mathcal{S}) = \{i \in [N] : s_i = 1\}$
- **Dead set**: $\mathcal{D}(\mathcal{S}) = \{i \in [N] : s_i = 0\}$
- **Alive count**: $n_{\text{alive}} = |\mathcal{A}(\mathcal{S})|$
:::

:::{div} feynman-prose
Why do walkers have both position *and* velocity? In many optimization problems, the direction you're moving matters as much as where you are. A walker heading toward a promising region is more valuable than one heading away from it. The velocity also enables *momentum-conserving* dynamics: when walkers clone, we can ensure that the total momentum of the swarm is preserved, preventing artificial acceleration or deceleration of the search.
:::

### Algorithmic Distance

Walkers interact based on their proximity in an **algorithmic metric** that combines position and velocity.

:::{prf:definition} Algorithmic Distance
:label: def-fg-algorithmic-distance

The **algorithmic distance** between walkers $i$ and $j$ is

$$
d_{\text{alg}}(i, j)^2 = \|z_i - z_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|_G^2
$$

where:
- $\|z_i - z_j\|^2$ is the squared Euclidean distance in latent coordinates
- $\|v_i - v_j\|_G^2 = (v_i - v_j)^\top G(z_i)(v_i - v_j)$ is the squared metric norm of velocity difference
- $\lambda_{\text{alg}} \geq 0$ is a **velocity weight** parameter

When $\lambda_{\text{alg}} = 0$, only positions matter; when $\lambda_{\text{alg}} > 0$, walkers with similar velocities are considered "closer."
:::

---

(sec-companion-selection)=
## 2. Companion Selection

The core of the Fractal Gas is **soft companion selection**: each walker probabilistically chooses companions from the swarm, with nearby walkers more likely to be selected.

:::{prf:definition} Soft Companion Kernel
:label: def-fg-soft-companion-kernel

For alive walkers $i \in \mathcal{A}(\mathcal{S})$, define the **Gaussian weights**

$$
w_{ij} = \exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon^2}\right), \quad j \neq i
$$

where $\epsilon > 0$ is the **kernel bandwidth**. The **companion distribution** for walker $i$ is the softmax:

$$
P_i(j) = \frac{w_{ij}}{\sum_{l \in \mathcal{A}, l \neq i} w_{il}}, \quad j \in \mathcal{A}(\mathcal{S}) \setminus \{i\}
$$

For **dead walkers** $i \in \mathcal{D}(\mathcal{S})$, the companion distribution is uniform over the alive set:

$$
P_i(j) = \frac{1}{|\mathcal{A}(\mathcal{S})|}, \quad j \in \mathcal{A}(\mathcal{S})
$$

The kernel is **degenerate** when $|\mathcal{A}(\mathcal{S})| < 2$; this case triggers transition to a **cemetery state** $\dagger$.
:::

:::{div} feynman-prose
Why *soft* selection instead of just picking the nearest neighbor? Three reasons:

1. **Smoothness**: The soft kernel makes the companion distribution a differentiable function of walker positions. This is crucial for theoretical analysis—it lets us apply calculus tools to understand the algorithm's behavior.

2. **Minorization**: Even distant walkers have a small but positive probability of being selected. This ensures the Markov chain is *irreducible*: any configuration can eventually reach any other. Without this, the swarm could fragment into disconnected clusters.

3. **Robustness**: Hard selection (always pick the nearest) is sensitive to small perturbations—a tiny position change can completely switch the selected companion. Soft selection degrades gracefully.
:::

### The Minorization Floor

The soft kernel guarantees a minimum selection probability for any pair of alive walkers.

:::{prf:proposition} Companion Minorization
:label: prop-companion-minorization

Let $D_{\text{alg}} = \max_{i,j \in \mathcal{A}} d_{\text{alg}}(i,j)$ be the algorithmic diameter of the alive set. Then for any $i, j \in \mathcal{A}(\mathcal{S})$ with $i \neq j$:

$$
P_i(j) \geq \frac{m_\epsilon}{n_{\text{alive}} - 1}
$$

where $m_\epsilon = \exp(-D_{\text{alg}}^2 / (2\epsilon^2))$ is the **kernel floor**.
:::

*Proof.* The numerator $w_{ij} \geq m_\epsilon$ by definition. The denominator $\sum_{l \neq i} w_{il} \leq n_{\text{alive}} - 1$ since each weight is at most 1. $\square$

:::{note}
:class: feynman-added
The minorization floor $m_\epsilon$ depends on both the kernel bandwidth $\epsilon$ and the swarm diameter $D_{\text{alg}}$. A larger $\epsilon$ (wider kernel) increases $m_\epsilon$, making selection more uniform. A smaller $\epsilon$ (tighter kernel) decreases $m_\epsilon$, making selection more localized.
:::

### Types of Companions

Each walker samples **two independent companions** from the same distribution $P_i$:

1. **Distance companion** $c_i^{\text{dist}} \sim P_i$: Used for the *diversity* channel of fitness
2. **Clone companion** $c_i^{\text{clone}} \sim P_i$: Used for cloning decisions

The independence of these samples is important: it decorrelates the fitness evaluation from the cloning decision, reducing bias.

---

(sec-fitness-computation)=
## 3. Fitness Computation

Fitness determines which walkers thrive and which must relocate. The Fractal Gas uses a **dual-channel** fitness function that balances reward (exploitation) with diversity (exploration).

### 3.1 The Dual-Channel Design

:::{div} feynman-prose
Here's the central tension in any search algorithm: you want to *exploit* what you've learned (go where rewards are high) but also *explore* new regions (maintain diversity). Pure exploitation leads to premature convergence on local optima. Pure exploration wastes computation on unpromising areas.

The Fractal Gas resolves this with two fitness channels:
- **Reward channel**: How good is this walker's current reward signal?
- **Diversity channel**: How far is this walker from its companion?

Walkers that are both high-reward *and* well-separated from companions get the highest fitness. This automatically balances exploitation and exploration.
:::

:::{prf:definition} Reward and Diversity Channels
:label: def-fg-reward-diversity

For walker $i$ with distance companion $c_i^{\text{dist}}$:

**Diversity (distance to companion)**:
$$
d_i = \sqrt{d_{\text{alg}}(i, c_i^{\text{dist}})^2 + \epsilon_{\text{dist}}^2}
$$

where $\epsilon_{\text{dist}} > 0$ is a regularization constant preventing division by zero.

**Reward (application-specific)**:
$$
r_i = \langle \mathcal{R}(z_i), v_i \rangle_G
$$

where $\mathcal{R}: \mathcal{Z} \to T^*\mathcal{Z}$ is a **reward 1-form** (e.g., gradient of a potential, or environment feedback).
:::

### 3.2 Standardization Pipeline

Raw reward and distance values are standardized to ensure comparable scales and bounded outputs.

:::{prf:definition} Fitness Standardization
:label: def-fg-standardization

**Step 1: Z-score normalization** (computed over alive walkers only):

$$
z_r(i) = \frac{r_i - \mu_r}{\sigma_r + \sigma_{\min}}, \quad z_d(i) = \frac{d_i - \mu_d}{\sigma_d + \sigma_{\min}}
$$

where $\mu_r, \sigma_r$ (resp. $\mu_d, \sigma_d$) are the mean and standard deviation of rewards (resp. distances) over $\mathcal{A}(\mathcal{S})$, and $\sigma_{\min} > 0$ is a regularizer.

**Step 2: Logistic rescaling**:

$$
r_i' = g_A(z_r(i)) + \eta, \quad d_i' = g_A(z_d(i)) + \eta
$$

where $g_A(z) = \frac{A}{1 + e^{-z}}$ is the logistic function with amplitude $A > 0$, and $\eta > 0$ is a **positivity floor**.
:::

:::{note}
:class: feynman-added
The standardization serves several purposes:
- **Z-score**: Makes the algorithm invariant to scale and shift of raw values
- **Logistic rescaling**: Bounds the output to $[\eta, A + \eta]$, preventing extreme fitness values
- **Positivity floor $\eta$**: Ensures rescaled values never reach zero, which is critical for the revival guarantee (see Section 5)
:::

### 3.3 Fitness Formula

:::{prf:definition} Dual-Channel Fitness
:label: def-fg-fitness

The **fitness** of walker $i$ is

$$
V_{\text{fit}, i} = (d_i')^{\beta_{\text{fit}}} \cdot (r_i')^{\alpha_{\text{fit}}}
$$

where:
- $\alpha_{\text{fit}} \geq 0$ is the **reward exponent**
- $\beta_{\text{fit}} \geq 0$ is the **diversity exponent**

**Dead walkers** have fitness set to $V_{\text{fit}, i} = 0$ by convention.

**Fitness bounds**: For alive walkers,

$$
V_{\min} = \eta^{\alpha_{\text{fit}} + \beta_{\text{fit}}} \leq V_{\text{fit}, i} \leq (A + \eta)^{\alpha_{\text{fit}} + \beta_{\text{fit}}} = V_{\max}
$$
:::

:::{div} feynman-prose
The exponents $\alpha_{\text{fit}}$ and $\beta_{\text{fit}}$ control the balance between exploitation and exploration:

- **$\alpha_{\text{fit}} > \beta_{\text{fit}}$**: Reward-dominated. The swarm aggressively pursues high-reward regions, risking premature convergence.
- **$\alpha_{\text{fit}} < \beta_{\text{fit}}$**: Diversity-dominated. The swarm maintains spread, risking slow convergence.
- **$\alpha_{\text{fit}} = \beta_{\text{fit}} = 1$** (default): Balanced. Fitness is proportional to the *product* of normalized reward and distance.

The multiplicative form $(d')^\beta (r')^\alpha$ rather than additive $(d')^\beta + (r')^\alpha$ is deliberate: it means a walker needs *both* good reward *and* good diversity to achieve high fitness. A walker with excellent reward but zero diversity (on top of its companion) gets low fitness.
:::

---

### 3.4 Worked Example: Fitness Computation

:::{admonition} Example: Computing Fitness for a Small Swarm
:class: feynman-added example

Consider a swarm of $N = 4$ walkers with the following raw values (all alive):

| Walker $i$ | Distance $d_i$ | Reward $r_i$ |
|------------|----------------|--------------|
| 1 | 0.5 | 2.0 |
| 2 | 1.2 | 1.5 |
| 3 | 0.8 | 3.0 |
| 4 | 2.0 | 0.5 |

**Step 1: Compute statistics**
- $\mu_d = (0.5 + 1.2 + 0.8 + 2.0)/4 = 1.125$
- $\sigma_d = 0.574$ (standard deviation)
- $\mu_r = (2.0 + 1.5 + 3.0 + 0.5)/4 = 1.75$
- $\sigma_r = 0.901$

**Step 2: Z-scores** (with $\sigma_{\min} = 10^{-8}$)

| Walker | $z_d(i)$ | $z_r(i)$ |
|--------|----------|----------|
| 1 | -1.09 | 0.28 |
| 2 | 0.13 | -0.28 |
| 3 | -0.57 | 1.39 |
| 4 | 1.52 | -1.39 |

**Step 3: Logistic rescaling** (with $A = 2$, $\eta = 0.1$)

| Walker | $d_i' = g_2(z_d) + 0.1$ | $r_i' = g_2(z_r) + 0.1$ |
|--------|-------------------------|-------------------------|
| 1 | 0.60 | 1.24 |
| 2 | 1.16 | 0.96 |
| 3 | 0.82 | 1.80 |
| 4 | 1.92 | 0.40 |

**Step 4: Fitness** (with $\alpha_{\text{fit}} = \beta_{\text{fit}} = 1$)

| Walker | $V_{\text{fit}} = d' \cdot r'$ | Interpretation |
|--------|-------------------------------|----------------|
| 1 | 0.74 | Good reward, low diversity |
| 2 | 1.11 | Moderate both |
| 3 | 1.48 | **Highest**: good reward AND diversity |
| 4 | 0.77 | High diversity, low reward |

Walker 3 has the highest fitness because it balances both channels well.
:::

---

(sec-cloning-mechanics)=
## 4. Cloning Mechanics

Cloning is the selection mechanism that drives the swarm toward high-fitness regions. Low-fitness walkers are replaced by perturbed copies of their companions.

### 4.1 The Cloning Decision

:::{prf:definition} Cloning Score and Decision
:label: def-fg-cloning-decision

For walker $i$ with clone companion $c_i^{\text{clone}}$, define:

**Cloning score** (relative fitness advantage of companion):
$$
S_i = \frac{V_{\text{fit}, c_i^{\text{clone}}} - V_{\text{fit}, i}}{V_{\text{fit}, i} + \varepsilon_{\text{clone}}}
$$

where $\varepsilon_{\text{clone}} > 0$ is a regularizer preventing division by zero.

**Cloning threshold**: Sample $T_i \sim \text{Uniform}(0, p_{\max})$ independently for each walker.

**Cloning decision**: Walker $i$ **clones from** $c_i^{\text{clone}}$ if and only if $S_i > T_i$.
:::

:::{div} feynman-prose
The cloning score $S_i$ measures how much better the companion is than the walker itself:
- $S_i > 0$: Companion is fitter → walker may clone
- $S_i \leq 0$: Walker is at least as fit → no cloning

The randomized threshold $T_i \sim \text{Uniform}(0, p_{\max})$ introduces stochasticity: even a small fitness advantage can trigger cloning (if $T_i$ is small), and even a large advantage might not (if $T_i$ is large). This prevents deterministic collapse and maintains exploration.
:::

### 4.2 Position Update (Jitter)

When walker $i$ clones from companion $c_i^{\text{clone}}$, it doesn't copy the companion's position exactly—it adds Gaussian noise.

:::{prf:definition} Cloning Jitter
:label: def-fg-cloning-jitter

When walker $i$ clones from companion $c$, its new position is:

$$
z_i' = z_c + \sigma_x \zeta_i, \quad \zeta_i \sim \mathcal{N}(0, I_{d_z})
$$

where $\sigma_x > 0$ is the **jitter scale** and $d_z = \dim(\mathcal{Z})$.
:::

:::{note}
:class: feynman-added
Why add jitter? Without it, all walkers cloning from the same companion would end up at the same position, destroying diversity. The jitter maintains a cloud of walkers around successful locations, enabling continued local exploration. The scale $\sigma_x$ controls the trade-off: larger jitter maintains diversity but may overshoot optima; smaller jitter enables precision but risks collapse.
:::

### 4.3 Velocity Update (Inelastic Collision)

Position updates are straightforward, but what about velocities? The Fractal Gas uses a **momentum-conserving** velocity update inspired by inelastic collisions.

:::{prf:definition} Inelastic Collision Velocity Update
:label: def-fg-inelastic-collision

When a set of walkers $G$ (a "collision group") all clone from the same companion, their velocities are updated as follows:

**Step 1**: Compute the center-of-mass velocity:
$$
V_{\text{COM}} = \frac{1}{|G|} \sum_{k \in G} v_k
$$

**Step 2**: Apply restitution toward the center of mass:
$$
v_k' = V_{\text{COM}} + \alpha_{\text{rest}} (v_k - V_{\text{COM}}), \quad k \in G
$$

where $\alpha_{\text{rest}} \in [0, 1]$ is the **restitution coefficient**.

**Properties**:
- **Momentum conservation**: $\sum_{k \in G} v_k' = \sum_{k \in G} v_k$ (total momentum preserved)
- $\alpha_{\text{rest}} = 1$: Elastic (velocities unchanged)
- $\alpha_{\text{rest}} = 0$: Perfectly inelastic (all walkers get $V_{\text{COM}}$)
:::

:::{div} feynman-prose
Why conserve momentum? In physical systems, momentum conservation is fundamental. For the Fractal Gas, it serves a practical purpose: it prevents the cloning operation from artificially injecting or removing kinetic energy from the swarm. Without momentum conservation, cloning could create runaway acceleration or deceleration, destabilizing the dynamics.

The restitution coefficient $\alpha_{\text{rest}}$ controls how much individual velocity information is retained. At $\alpha_{\text{rest}} = 0$, all walkers in a collision group end up with the same velocity—maximum coordination. At $\alpha_{\text{rest}} = 1$, velocities are unchanged—maximum independence. The default $\alpha_{\text{rest}} = 0.5$ balances these extremes.
:::

---

(sec-revival-guarantee)=
## 5. The Revival Guarantee

A critical property of the Fractal Gas is that **dead walkers are guaranteed to be revived** (as long as at least one walker remains alive). This prevents gradual extinction and ensures the swarm maintains its population.

:::{prf:definition} Revival Constraint
:label: def-fg-revival-constraint

The algorithm parameters must satisfy the **revival inequality**:

$$
\varepsilon_{\text{clone}} \cdot p_{\max} < \eta^{\alpha_{\text{fit}} + \beta_{\text{fit}}} = V_{\min}
$$

where:
- $\varepsilon_{\text{clone}}$ is the cloning score regularizer
- $p_{\max}$ is the maximum cloning threshold
- $\eta$ is the positivity floor
- $V_{\min}$ is the minimum possible fitness for alive walkers
:::

:::{prf:proposition} Guaranteed Revival
:label: prop-fg-guaranteed-revival

Under the revival constraint, if $|\mathcal{A}(\mathcal{S})| \geq 1$, then every dead walker clones with probability 1.
:::

*Proof.* For a dead walker $i$, we have $V_{\text{fit}, i} = 0$. Its companion $c$ is alive (since dead walkers sample from $\mathcal{A}$), so $V_{\text{fit}, c} \geq V_{\min}$. The cloning score is:

$$
S_i = \frac{V_{\text{fit}, c} - 0}{0 + \varepsilon_{\text{clone}}} = \frac{V_{\text{fit}, c}}{\varepsilon_{\text{clone}}} \geq \frac{V_{\min}}{\varepsilon_{\text{clone}}} > p_{\max}
$$

The last inequality uses the revival constraint. Since $T_i \leq p_{\max}$ always, we have $S_i > T_i$ with probability 1. $\square$

:::{div} feynman-prose
The revival guarantee is the safety net of the Fractal Gas. No matter how poorly the search is going, dead walkers will always be reborn near alive ones. The only failure mode is **catastrophic**: all walkers dying simultaneously. This is typically a measure-zero event (assuming continuous dynamics with noise), so in practice the swarm persists indefinitely.

This is a key difference from genetic algorithms, where population can dwindle through selection pressure. The Fractal Gas maintains constant population $N$—it's the *distribution* of walkers that evolves, not the count.
:::

---

(sec-kinetic-dynamics)=
## 6. Kinetic Dynamics

Between cloning events, walkers evolve according to a **kinetic operator** that moves them through the latent space. The Fractal Gas framework is agnostic to the specific kinetic scheme; different choices yield different algorithm variants.

### 6.1 Abstract Kinetic Interface

:::{prf:definition} Kinetic Operator (Abstract)
:label: def-fg-kinetic-abstract

A **kinetic operator** is a Markov transition kernel $K_\tau: (\mathcal{Z} \times T\mathcal{Z}) \to (\mathcal{Z} \times T\mathcal{Z})$ that:

1. **Preserves the alive set**: $K_\tau$ acts only on $(z, v)$, not on status $s$
2. **Admits a stationary measure**: There exists a reference measure $\mu$ such that $K_\tau^* \mu = \mu$ (or $K_\tau$ contracts toward $\mu$)
3. **Injects noise**: $K_\tau$ is not deterministic; it has a positive diffusion component

The kinetic operator advances walkers by time step $\tau$:

$$
(z_i', v_i') \sim K_\tau(\cdot \mid z_i, v_i)
$$
:::

### 6.2 Concrete Example: Boris-BAOAB

The **Boris-BAOAB** scheme is a splitting integrator for Langevin dynamics on Riemannian manifolds, adapted for the Fractal Gas with anisotropic diffusion.

:::{prf:definition} Boris-BAOAB Splitting
:label: def-fg-boris-baoab

For a walker at $(z, p)$ with $p = G(z) v$ (metric momentum), the Boris-BAOAB step with time step $h$ is:

**B (half-kick)**:
$$
p \leftarrow p - \frac{h}{2} \nabla \Phi_{\text{eff}}(z)
$$

**A (half-drift)**:
$$
z \leftarrow \mathrm{Exp}_z\left(\frac{h}{2} G^{-1}(z) p\right)
$$

**O (thermostat)**:
$$
p \leftarrow c_1 p + c_2 G^{1/2}(z) \Sigma_{\text{reg}}(z) \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

where $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1 - c_1^2) T_c}$, $\gamma$ is friction, and $T_c$ is cognitive temperature.

**A (half-drift)**: Repeat the geodesic step.

**B (half-kick)**: Repeat the momentum kick.
:::

:::{note}
:class: feynman-added
The five steps spell **BAOAB**:
- **B** = kick (momentum update from potential gradient)
- **A** = drift (position update along geodesic)
- **O** = thermostat (noise injection with friction)

The **Boris** prefix refers to an optional rotation step for curl fields (used in electromagnetic simulations). For most optimization applications, the curl term is zero and Boris-BAOAB reduces to standard BAOAB.
:::

### 6.3 Anisotropic Diffusion

The thermostat step uses an **anisotropic diffusion tensor** $\Sigma_{\text{reg}}(z)$ that adapts to the local fitness landscape.

:::{prf:definition} Regularized Anisotropic Diffusion
:label: def-fg-anisotropic-diffusion

The **regularized diffusion tensor** is:

$$
\Sigma_{\text{reg}}(z) = \left(\nabla_z^2 V_{\text{fit}}(z) + \epsilon_\Sigma I\right)^{-1/2}
$$

where $\nabla_z^2 V_{\text{fit}}$ is the Hessian of fitness with respect to position, and $\epsilon_\Sigma > 0$ is a regularizer ensuring positive definiteness.
:::

:::{div} feynman-prose
Why anisotropic diffusion? Consider a fitness landscape with a narrow valley: stiff in one direction (walls of the valley) and flat in another (along the valley floor). Isotropic diffusion would waste energy bouncing off the walls. Anisotropic diffusion, scaled by the inverse square root of the Hessian, injects more noise along flat directions and less along stiff directions. The walker slides along the valley floor instead of bouncing off walls.

This is the same principle behind preconditioned gradient descent and natural gradient methods—adapting the step size to the local geometry of the loss landscape.
:::

---

(sec-step-operator)=
## 7. The Complete Step Operator

The full Fractal Gas step combines fitness evaluation, cloning, and kinetics into a single Markov transition.

:::{prf:definition} Fractal Gas Step Operator
:label: def-fg-step-operator

The **one-step operator** $P_\tau: \Sigma_N \to \Sigma_N$ acts as follows:

**Input**: Swarm state $\mathcal{S} = ((z_i, v_i, s_i))_{i=1}^N$

**Step 1 (Fitness)**:
1. Sample distance companions: $c_i^{\text{dist}} \sim P_i$ for each $i$
2. Compute diversity $d_i$ and reward $r_i$
3. Standardize and rescale to get $d_i', r_i'$
4. Compute fitness $V_{\text{fit}, i} = (d_i')^{\beta_{\text{fit}}} (r_i')^{\alpha_{\text{fit}}}$

**Step 2 (Cloning)**:
1. Sample clone companions: $c_i^{\text{clone}} \sim P_i$ for each $i$
2. Compute cloning scores $S_i$ and sample thresholds $T_i$
3. For each $i$ with $S_i > T_i$:
   - Update position: $z_i' = z_{c_i^{\text{clone}}} + \sigma_x \zeta_i$
   - Group walkers by recipient companion and apply inelastic collision
4. Set status: $s_i' = 1$ for all cloned walkers

**Step 3 (Kinetics)**:
1. Apply kinetic operator: $(z_i', v_i') \sim K_\tau(\cdot \mid z_i, v_i)$ for each $i$

**Output**: Updated swarm $\mathcal{S}' = ((z_i', v_i', s_i'))_{i=1}^N$

**Cemetery**: If $|\mathcal{A}(\mathcal{S})| < 2$ at any point, transition to absorbing state $\dagger$.
:::

:::{div} feynman-prose
The order of operations matters. Fitness is computed first because it depends on the current swarm configuration. Cloning comes next because it uses the fitness values. Kinetics comes last because it advances positions *after* cloning has redistributed walkers.

The cemetery state $\dagger$ is a theoretical necessity: if only one or zero walkers remain alive, companion selection is undefined. In practice, the revival guarantee ensures this almost never happens—it requires all walkers to die in a single step, which has probability zero under continuous dynamics with noise.
:::

---

(sec-algorithm-variants)=
## 8. Algorithm Variants

The Fractal Gas framework admits several instantiations depending on the choice of state space, reward function, and kinetic operator.

### Latent Fractal Gas

The **Latent Fractal Gas** operates in a learned latent space $\mathcal{Z}$ with Riemannian metric $G$. Key features:
- Position $z$ is a latent representation (e.g., from a VAE encoder)
- Metric $G(z)$ may be learned or derived from the latent structure
- Reward $r_i = \langle \mathcal{R}(z_i), v_i \rangle_G$ is the inner product of a reward 1-form with velocity
- Kinetics: Boris-BAOAB with anisotropic diffusion

This is the variant formally treated in {doc}`02_fractal_gas_latent`.

### Fragile Gas

The **Fragile Gas** is the RL instantiation where reward comes from environment interaction:
- Position $z$ encodes environment state
- Reward $r_i$ is the environment reward signal
- The swarm explores the state-action space guided by environment feedback

### Abstract Fractal Gas

The **Abstract Fractal Gas** is the minimal specification:
- Arbitrary state space $\mathcal{X}$ with distance $d$
- Arbitrary reward function $r: \mathcal{X} \to \mathbb{R}$
- Minimal kinetics (e.g., Brownian motion)

This is the version formalized in the Hypostructure framework (Volume 2, Part X).

---

(sec-continuum-preview)=
## 9. Connection to Continuum Theory

In the limit of many walkers ($N \to \infty$) and small time steps ($\tau \to 0$), the Fractal Gas converges to a continuum description governed by a reaction-diffusion PDE.

### The Darwinian Ratchet

The one-step operator decomposes into **transport** (kinetics) and **reaction** (selection/cloning):

$$
P_\tau = R_\tau \circ T_\tau
$$

where:
- $T_\tau$ = transport (Langevin diffusion)
- $R_\tau$ = reaction (fitness-weighted resampling)

In the continuum limit, this becomes the **WFR (Wasserstein-Fisher-Rao)** equation:

$$
\partial_t \rho = \underbrace{\nabla \cdot (\rho \nabla \Phi) + \Delta \rho}_{\text{transport (WFR)}} + \underbrace{\rho (V_{\text{fit}} - \bar{V}_{\text{fit}})}_{\text{reaction (replicator)}}
$$

:::{div} feynman-prose
The Darwinian Ratchet is a beautiful synthesis: the diffusion term spreads mass outward (exploration), the drift term pulls mass toward low-potential regions (gradient descent), and the reaction term amplifies mass in high-fitness regions while depleting low-fitness regions (natural selection). Together, they create a "ratchet" that steadily concentrates mass at optima while maintaining diversity.

This is why the Fractal Gas works: it's not just a heuristic algorithm—it's a discretization of a well-understood PDE. The convergence guarantees of the PDE transfer (with appropriate error bounds) to the discrete algorithm.
:::

### Mean-Field Limit

As $N \to \infty$, the empirical measure $\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{(z_i, v_i)}$ converges to a deterministic measure $\mu_t$ solving the mean-field equation. The error bound (propagation of chaos) is:

$$
\mathbb{E}[W_2(\mu_N, \mu)] \lesssim \frac{e^{-\kappa_W t}}{\sqrt{N}}
$$

where $\kappa_W > 0$ is the Wasserstein contraction rate certified by the sieve.

For formal treatment of convergence, see {doc}`02_fractal_gas_latent` (Part III-B).

---

(sec-parameter-glossary)=
## Parameter Glossary

| Category | Symbol | Default | Description |
|----------|--------|---------|-------------|
| **Swarm** | $N$ | 50 | Number of walkers |
| **Companion** | $\epsilon$ | 0.1 | Kernel bandwidth |
| **Companion** | $\lambda_{\text{alg}}$ | 0.0 | Velocity weight in $d_{\text{alg}}$ |
| **Fitness** | $\alpha_{\text{fit}}$ | 1.0 | Reward exponent |
| **Fitness** | $\beta_{\text{fit}}$ | 1.0 | Diversity exponent |
| **Fitness** | $\eta$ | 0.1 | Positivity floor |
| **Fitness** | $A$ | 2.0 | Logistic amplitude |
| **Fitness** | $\sigma_{\min}$ | 1e-8 | Standardization regularizer |
| **Fitness** | $\epsilon_{\text{dist}}$ | 1e-8 | Distance regularizer |
| **Cloning** | $\varepsilon_{\text{clone}}$ | 0.01 | Score regularizer |
| **Cloning** | $p_{\max}$ | 1.0 | Max threshold |
| **Cloning** | $\sigma_x$ | 0.1 | Jitter scale |
| **Cloning** | $\alpha_{\text{rest}}$ | 0.5 | Restitution coefficient |
| **Kinetic** | $h$ | 0.01 | Time step |
| **Kinetic** | $\gamma$ | 1.0 | Friction coefficient |
| **Kinetic** | $T_c$ | 1.0 | Cognitive temperature |
| **Kinetic** | $\epsilon_\Sigma$ | 1e-4 | Diffusion regularizer |

:::{admonition} Revival Constraint Check
:class: warning

Parameters must satisfy:
$$
\varepsilon_{\text{clone}} \cdot p_{\max} < \eta^{\alpha_{\text{fit}} + \beta_{\text{fit}}}
$$

With defaults: $0.01 \times 1.0 = 0.01 < 0.1^2 = 0.01$. **Warning**: Default parameters are at the boundary! Use $\eta = 0.15$ or $\varepsilon_{\text{clone}} = 0.005$ for safety margin.
:::

---

## Summary

The Fractal Gas is a principled algorithm for optimization and sampling that combines:

1. **Soft companion selection**: Smooth, differentiable pairing with minorization guarantees
2. **Dual-channel fitness**: Balanced exploitation (reward) and exploration (diversity)
3. **Momentum-conserving cloning**: Population maintenance with physical consistency
4. **Guaranteed revival**: Dead walkers always return, preventing extinction
5. **Flexible kinetics**: Pluggable dynamics (Langevin, Boris-BAOAB, etc.)

The formal treatment in {doc}`02_fractal_gas_latent` proves that this algorithm:
- Defines a valid Markov transition kernel
- Converges to a quasi-stationary distribution
- Admits a mean-field limit with explicit error bounds
- Satisfies the sieve conditions for the Hypostructure framework

:::{seealso}
- {doc}`02_fractal_gas_latent`: Proof object with full sieve verification
- {doc}`../2_hypostructure/10_information_processing/02_fractal_gas`: Hypostructure metatheorems
- {doc}`appendices/01_fragile_gas_framework`: Axiom system and revival guarantee proof
:::
