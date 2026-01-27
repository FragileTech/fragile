# The Fractal Gas Algorithm Family

## TLDR

**Population-Based Search with Selection Pressure**: The Fractal Gas is a swarm algorithm where walkers explore a state
space through soft companion selection, dual-channel fitness evaluation (balancing reward and diversity), and
momentum-conserving cloning. Low-fitness walkers are replaced by perturbed copies of high-fitness companions, creating
an adaptive search that automatically concentrates effort in promising regions.

**Three-Timescale Analysis**: The algorithm admits rigorous mathematical characterization across three timescales:
discrete (individual Markov transitions), scaling (mean-field limit with $N \to \infty$ walkers), and continuum (WFR
reaction-diffusion PDE). This hierarchy connects practical implementation to theoretical guarantees.

**Guaranteed Revival and Population Maintenance**: Dead walkers are forced to clone from the alive set, so if at least
one walker remains alive, every dead walker revives with probability 1. This prevents gradual extinction and ensures
the swarm maintains constant population $N$—only the distribution of walkers evolves, not their count (the only
catastrophic failure is the all-dead event).

**Flexible Instantiation Framework**: The core mechanisms (companion selection, fitness computation, cloning, kinetics)
define an abstract framework that admits multiple instantiations: the Latent Fractal Gas (learned latent space with
Riemannian geometry), the Fragile Gas (RL with environment feedback), and the Abstract Fractal Gas (minimal
specification for theoretical analysis).

## Introduction

This chapter provides an intuitive, implementation-oriented introduction to the Fractal Gas algorithm family. Where the formal treatment in {doc}`02_fractal_gas_latent` develops the mathematical machinery for convergence proofs and sieve verification, this chapter focuses on *what the algorithm does* and *why each component exists*. The goal is to build understanding before formalism—to make the subsequent mathematical analysis feel inevitable rather than arbitrary.

**Prerequisites**: This chapter is designed to be self-contained for readers with basic familiarity with optimization and probability. No prior knowledge of the Fragile framework or Hypostructure is required—those concepts are developed in subsequent chapters. Readers seeking the rigorous mathematical treatment should proceed to {doc}`02_fractal_gas_latent` after building intuition here; those interested in the complete axiomatic foundations will find them in {doc}`../appendices/01_fragile_gas_framework`.

The chapter is organized around the algorithm's core components:

- State space and walker definitions
- Soft companion selection with minorization guarantees
- Dual-channel fitness (exploitation vs. exploration)
- Momentum-conserving cloning and the revival guarantee
- Kinetic operator interfaces and the complete step operator
- Algorithm variants and the continuum-limit interpretation

A parameter glossary at the end provides quick reference for tunable quantities.

:::{div} feynman-prose
Imagine you're trying to find the highest point in a vast, fog-covered mountain range. You can't see the peaks, but you
have a team of hikers scattered across the landscape. Each hiker can sense the altitude at their location and can
communicate with nearby teammates. The clever strategy isn't to have everyone climb independently—it's to have hikers in
promising locations "clone" themselves while hikers in dead ends "die off," with the dead hikers respawning near
successful ones.

This is the essence of the Fractal Gas: a swarm of *walkers* exploring a space, where *fitness* determines who thrives and who must relocate, and *companion selection* determines who learns from whom. The result is a self-organizing search process that automatically concentrates computational effort in promising regions while maintaining diversity to avoid local traps.
:::

---

(sec-overview)=
## Overview

:::{div} feynman-prose
Before we dive into the machinery, let me tell you what problem we are really solving here. You have a function you want
to optimize, or a distribution you want to sample from, and the space is enormous. Maybe it is the space of all possible
neural network weights, or the space of all configurations of a protein, or the space of all strategies an agent could
follow. The space is so big that exhaustive search is hopeless, and gradient descent gets stuck in local minima.

What do you do? You deploy a *swarm*. Not one searcher, but many. And here is the key insight: these searchers should
not operate independently. They should *talk to each other*. When one searcher finds a promising region, others should
know about it and move toward it. When a searcher is stuck in a dead end, it should be able to "die" and respawn near a
more successful colleague.

This is not a new idea. Particle swarm optimization does something like this. Genetic algorithms do something like this.
But the Fractal Gas does it in a way that admits precise mathematical analysis. We can prove things about convergence
rates, about the conditions under which the algorithm succeeds, about how it behaves in the limit of infinitely many
walkers. That is rare in the world of optimization heuristics, and it is why the Fractal Gas is worth understanding in
detail.
:::

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
The Fractal Gas sits at the intersection of several algorithmic traditions.

Like *particle swarm optimization* {cite}`kennedy1995particle,shi1998modified`, it uses a population of interacting
agents. Like *genetic algorithms* {cite}`holland1992genetic,goldberg1989genetic`, it employs selection and
reproduction. Like *MCMC methods* {cite}`metropolis1953equation,hastings1970monte`, it samples from a target
distribution.

But unlike any of these, it admits a precise mathematical characterization: in the continuum limit, the swarm density
evolves according to a reaction-diffusion equation where mass flows toward high-fitness regions while maintaining
diversity through diffusion.
:::

### Relationship to Other Algorithms

| Algorithm Family | Shared Feature | Key Difference |
|------------------|----------------|----------------|
| Particle Swarm {cite}`kennedy1995particle` | Population-based, local communication | Fractal Gas: explicit selection pressure, no global best |
| Genetic Algorithms {cite}`holland1992genetic` | Selection and reproduction | Fractal Gas: continuous state space, soft selection |
| MCMC / Langevin {cite}`metropolis1953equation,roberts1996exponential` | Sampling from distributions | Fractal Gas: selection creates non-equilibrium dynamics |
| Interacting Particle Systems {cite}`del2004feynman` | Mean-field limits | Fractal Gas: dual-channel fitness, momentum conservation |

## Minimal Pseudocode

```python
# Pseudocode: one Fractal Gas step (high level)
# State: walkers i=1..N with (z_i, v_i, s_i) where s_i ∈ {alive, dead}
#
# 1) Rewards + alive mask: compute reward signal and mark in-bounds walkers as alive
# 2) Companion selection (distance): sample c_i^dist using a soft kernel on d_alg(i,j)
# 3) Fitness: compute standardized V_fit(i) from reward and diversity terms
# 4) Companion selection (cloning): sample c_i^clone from the same kernel
# 5) Cloning + revival (reaction): low-fitness walkers may clone; dead walkers always clone
# 6) Kinetics (transport + killing): advance (z_i, v_i) and update alive/dead by a boundary check
# 7) Repeat; the population size N stays constant by construction
```

:::{figure} figures/fg-overview-flow.svg
:alt: Flow diagram of a Fractal Gas step from companion selection to kinetics and repeat.
:width: 95%
:align: center

Fractal Gas step overview: selection, fitness, cloning, kinetics, and repeat with constant population.
:::

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

The latent space $\mathcal{Z}$ is equipped with a Riemannian metric $G$ {cite}`lee2018introduction`, inducing the inner product $\langle u, w \rangle_G = u^\top G(z) w$ for tangent vectors $u, w \in T_z\mathcal{Z}$.
:::

:::{prf:definition} Swarm State
:label: def-fg-swarm-state

A **swarm** of $N$ walkers is the tuple $\mathcal{S} = (w_1, \ldots, w_N)$ with state space

$$
\Sigma_N = (\mathcal{Z} \times T\mathcal{Z} \times \{0,1\})^N.
$$

For a swarm $\mathcal{S}$, we define (where $[N] := \{1, \ldots, N\}$):
- **Alive set**: $\mathcal{A}(\mathcal{S}) = \{i \in [N] : s_i = 1\}$
- **Dead set**: $\mathcal{D}(\mathcal{S}) = \{i \in [N] : s_i = 0\}$
- **Alive count**: $n_{\text{alive}} = |\mathcal{A}(\mathcal{S})|$
:::

:::{note}
:class: feynman-added
In most variants, a walker is marked **dead** ($s=0$) when it violates constraints (for example, leaving an admissible
region $B$) during the kinetic update stage (via a boundary/constraints check). Dead walkers remain in the swarm and are
revived via cloning (Section 5).
:::

:::{div} feynman-prose
Why do walkers have both position *and* velocity? In many optimization problems, the direction you're moving matters as
much as where you are. A walker heading toward a promising region is more valuable than one heading away from it.

Velocity also enables *momentum-conserving* dynamics: when walkers clone, we can preserve the swarm's total momentum,
preventing artificial acceleration or deceleration of the search.
:::

### Algorithmic Distance

Walkers interact based on their proximity in an **algorithmic metric** that combines position and velocity.

:::{div} feynman-prose
Now here is something subtle that deserves your attention. When we say two walkers are "close," we do not just mean they
are at nearby positions. We also care about whether they are moving in similar directions.

Think about it this way. Suppose you have two hikers on a mountain, both at the same elevation. One is climbing toward
the peak. The other is heading toward a cliff. Should they be considered "close" in any meaningful sense? Positionally,
yes. But operationally, no. They are on very different trajectories, and what works for one will not work for the other.

The algorithmic distance captures this by including a velocity term. Two walkers are considered close if they are at
similar positions *and* moving in similar directions. The parameter $\lambda_{\text{alg}}$ controls how much the
velocity matters. When $\lambda_{\text{alg}} = 0$, only position counts. When $\lambda_{\text{alg}} > 0$, walkers moving
in different directions are considered "farther apart" even if their positions coincide.

This is the same principle behind phase space in classical mechanics. The state of a particle is not just where it is,
but where it is *and* how fast it is moving. The Fractal Gas operates in a similar extended state space.
:::

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

:::{note}
:class: feynman-added
On a curved manifold, velocities live in different tangent spaces. In practice, $d_{\text{alg}}$ is computed in a common
coordinate chart (or after transporting velocities) so the difference $v_i - v_j$ is well-defined.
:::

:::{figure} figures/fg-alg-distance.svg
:alt: Diagram showing algorithmic distance as position distance plus velocity distance.
:width: 95%
:align: center

Algorithmic distance combines position similarity with velocity similarity via $\lambda_{\text{alg}}$.
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
P_i(j) = \frac{w_{ij}}{\sum_{l \in \mathcal{A}(\mathcal{S}), l \neq i} w_{il}}, \quad j \in \mathcal{A}(\mathcal{S}) \setminus \{i\}
$$

For **dead walkers** $i \in \mathcal{D}(\mathcal{S})$, the companion distribution is uniform over the alive set:

$$
P_i(j) = \frac{1}{|\mathcal{A}(\mathcal{S})|}, \quad j \in \mathcal{A}(\mathcal{S})
$$

The kernel is **degenerate** when $|\mathcal{A}(\mathcal{S})| < 2$ (an alive walker has no valid companion). Analyses
often handle this with an explicit **cemetery state** $\dagger$, while implementations may special-case the
single-survivor regime by freezing the lone alive walker and reviving everyone else from it.
:::

:::{figure} figures/fg-companion-kernel.svg
:alt: Soft companion selection with Gaussian weights and a minorization probability floor.
:width: 95%
:align: center

Soft companion selection: nearby walkers are weighted more, but every alive walker keeps a nonzero chance.
:::

:::{div} feynman-prose
Why *soft* selection instead of just picking the nearest neighbor? Three reasons:

1. **Smoothness**: The soft kernel makes the companion distribution a differentiable function of walker positions. This is crucial for theoretical analysis—it lets us apply calculus tools to understand the algorithm's behavior.

2. **Minorization**: Even distant walkers have a small but positive probability of being selected. This ensures the Markov chain is *irreducible* {cite}`meyn2012markov`: any configuration can eventually reach any other. Without this, the swarm could fragment into disconnected clusters.

3. **Robustness**: Hard selection (always pick the nearest) is sensitive to small perturbations—a tiny position change can completely switch the selected companion. Soft selection degrades gracefully.
:::

### The Minorization Floor

The soft kernel guarantees a minimum selection probability for any pair of alive walkers.

:::{prf:proposition} Companion Minorization
:label: prop-companion-minorization

Let $D_{\text{alg}} = \max_{i,j \in \mathcal{A}(\mathcal{S})} d_{\text{alg}}(i,j)$ be the algorithmic diameter of the alive set. Then for any $i, j \in \mathcal{A}(\mathcal{S})$ with $i \neq j$:

$$
P_i(j) \geq \frac{m_\epsilon}{n_{\text{alive}} - 1}
$$

where $m_\epsilon = \exp(-D_{\text{alg}}^2 / (2\epsilon^2))$ is the **kernel floor**.
:::

*Proof.* The numerator $w_{ij} \geq m_\epsilon$ by definition. The denominator $\sum_{l \neq i} w_{il} \leq n_{\text{alive}} - 1$ since each weight is at most 1. $\square$

:::{div} feynman-prose
Here is something worth sitting with for a moment. The minorization bound says that no matter how far apart two walkers
are, there is always a positive probability that one will select the other as a companion. This is not just a
mathematical convenience. It is the reason the swarm cannot fragment into disconnected islands that never communicate.
Even walkers on opposite ends of the state space have a small but nonzero chance of interacting, and that small chance
is enough to guarantee that information eventually flows everywhere.

The exponential in $m_\epsilon = \exp(-D_{\text{alg}}^2/(2\epsilon^2))$ falls off rapidly with distance, so distant pairs
are rarely selected. But "rarely" is infinitely different from "never." In Markov chain theory
{cite}`meyn2012markov,robert2004monte`, this distinction is the difference between an irreducible chain and a reducible
one, between a system that eventually mixes and one that gets stuck forever.
:::

:::{note}
:class: feynman-added
The minorization floor $m_\epsilon$ depends on both the kernel bandwidth $\epsilon$ and the swarm diameter $D_{\text{alg}}$. A larger $\epsilon$ (wider kernel) increases $m_\epsilon$, making selection more uniform. A smaller $\epsilon$ (tighter kernel) decreases $m_\epsilon$, making selection more localized.
:::

### Types of Companions

Each walker samples **two independent companions** from the same distribution $P_i$:

1. **Distance companion** $c_i^{\text{dist}} \sim P_i$: Used for the *diversity* channel of fitness
2. **Clone companion** $c_i^{\text{clone}} \sim P_i$: Used for cloning decisions

The independence of these samples is important: it decorrelates the fitness evaluation from the cloning decision, reducing bias.

:::{div} feynman-prose
You might wonder: why two companions instead of one? Why not use the same companion for both fitness computation and
cloning?

The answer is subtle but important. If we used the same companion for both, the fitness evaluation would be *correlated*
with the cloning decision in a way that introduces bias. A walker would be most likely to clone from whatever companion
happened to make its diversity score look bad (since low diversity relative to that particular companion triggers
cloning). This creates a systematic bias toward certain cloning patterns.

By sampling two independent companions, we break this correlation. The distance companion determines whether you look
diverse. The clone companion determines who you clone from if you do clone. These are independent questions that deserve
independent answers.

This is the same principle behind using separate training and validation sets in machine learning: you do not want the
data that determines your model's parameters to also determine your estimate of its performance.
:::

:::{tip}
:class: feynman-added
In practice, the two companion samples come from the same distribution but are drawn independently. This means the same
walker *could* be selected for both roles, but usually will not be. The key point is that the selection for one role
does not influence the selection for the other.
:::

---

(sec-fitness-computation)=
## 3. Fitness Computation

Fitness determines which walkers thrive and which must relocate. The Fractal Gas uses a **dual-channel** fitness function that balances reward (exploitation) with diversity (exploration).

### 3.1 The Dual-Channel Design

:::{div} feynman-prose
Here's the central tension in any search algorithm: you want to *exploit* what you've learned (go where rewards are high) but also *explore* new regions (maintain diversity) {cite}`sutton2018reinforcement`. Pure exploitation leads to premature convergence on local optima. Pure exploration wastes computation on unpromising areas.

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

**Reward (application-specific scalar)**:

$$
r_i = R(z_i)
$$

where $R:\mathcal{Z}\to\mathbb{R}$ is a scalar reward signal (for example, $R=-U$ for potential minimization).
In agent/1-form settings, the reward is directional:
$
r_i = \langle \mathcal{R}(z_i), v_i \rangle_G
$
with $\mathcal{R}: \mathcal{Z} \to T^*\mathcal{Z}$ a reward 1-form.
:::

:::{div} feynman-prose
The reward 1-form notation $\langle \mathcal{R}(z), v \rangle_G$ might look intimidating, but here is what it really means: the reward depends not just on where you are, but on which direction you are moving.

Think of skiing downhill. Your position on the mountain matters, but what really determines whether you are having a
good time is whether you are moving toward the bottom or toward the cliff. The inner product $\langle \mathcal{R}, v
\rangle$ asks: "Is this walker's velocity aligned with the reward gradient?" A walker heading toward high reward gets
positive contribution. A walker heading away gets negative contribution.

The simplest case is when $\mathcal{R} = \nabla \Phi$ for some potential $\Phi$. Then the reward reduces to
$r = \langle \nabla \Phi, v \rangle$, which is the directional derivative. It measures how fast the potential is
increasing in the direction of motion. But the framework allows more general reward signals that do not come from a
potential, such as environmental feedback in reinforcement learning.
:::

:::{figure} figures/fg-fitness-pipeline.svg
:alt: Pipeline diagram for dual-channel fitness: reward and diversity are standardized and multiplied.
:width: 95%
:align: center

Dual-channel fitness pipeline: reward and diversity are standardized, bounded, and multiplied into $V_{\text{fit}}$.
:::

### 3.2 Standardization Pipeline

Raw reward and distance values are standardized to ensure comparable scales and bounded outputs.

:::{div} feynman-prose
Now here is a piece of engineering that looks pedestrian but is actually rather clever. Suppose your reward signal ranges
from 0 to 1000, and your distance measurements range from 0.001 to 0.1. If you multiply them directly, the reward
completely dominates. Even a tiny reward difference swamps a huge distance difference. The standardization pipeline fixes
this by asking: "How unusual is this value compared to what we are seeing right now?"

The z-score transformation centers each quantity at zero and scales it by its current variability. A reward two standard deviations above average gets the same standardized value as a distance two standard deviations above average. Now they speak the same language.

But z-scores can be arbitrarily large, which causes numerical problems and makes the fitness function too sensitive to
outliers. The logistic squashing fixes this: it takes any real number and maps it smoothly into a bounded interval
{cite}`bishop2006pattern`. Extreme values saturate rather than explode. And the positivity floor ensures we never divide
by zero or take logarithms of zero later in the pipeline.
:::

:::{prf:definition} Fitness Standardization
:label: def-fg-standardization

**Step 1: Z-score normalization** (computed over alive walkers only, with regularized std):

$$
z_r(i) = \frac{r_i - \mu_r}{\sigma_r'}, \quad z_d(i) = \frac{d_i - \mu_d}{\sigma_d'}
$$

where $\mu_r, \sigma_r$ (resp. $\mu_d, \sigma_d$) are the mean and standard deviation of rewards (resp. distances) over $\mathcal{A}(\mathcal{S})$, and
$
\sigma_r' = \sqrt{\sigma_r^2 + \sigma_{\min}^2}, \quad \sigma_d' = \sqrt{\sigma_d^2 + \sigma_{\min}^2}
$
with $\sigma_{\min} > 0$ a regularizer.

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
- **Positivity floor $\eta$**: Ensures rescaled values never reach zero, stabilizing scores and (in variants without forced revival) preventing zero-probability cloning
:::

### 3.3 Fitness Formula

:::{div} feynman-prose
Now we put the pieces together. We have a normalized reward value and a normalized diversity value, both living in the
same bounded interval. How do we combine them into a single fitness score?

The simplest thing would be to add them: fitness = reward + diversity. But that has a problem. A walker with fantastic
reward but zero diversity would still get a decent fitness score from the reward term alone. We want walkers to need
*both* qualities to thrive.

The solution is to *multiply* them. If either term is small, the product is small. You cannot compensate for being on
top of your neighbor (low diversity) by having great reward. You cannot compensate for poor reward by being far from
everyone else. You need both.

The exponents $\alpha_{\text{fit}}$ and $\beta_{\text{fit}}$ let you tune how much each channel matters. Equal exponents
mean balanced importance. Unequal exponents let you bias toward exploitation (favor reward) or exploration (favor
diversity) depending on your problem.
:::

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

**Step 1: Compute statistics** (population mean/std, rounded)
- $\mu_d = (0.5 + 1.2 + 0.8 + 2.0)/4 = 1.125$
- $\sigma_d \approx 0.563$
- $\mu_r = (2.0 + 1.5 + 3.0 + 0.5)/4 = 1.75$
- $\sigma_r \approx 0.901$

**Step 2: Z-scores** (with $\sigma_{\min} = 10^{-8}$)

| Walker | $z_d(i)$ | $z_r(i)$ |
|--------|----------|----------|
| 1 | -1.11 | 0.28 |
| 2 | 0.13 | -0.28 |
| 3 | -0.58 | 1.39 |
| 4 | 1.55 | -1.39 |

**Step 3: Logistic rescaling** (with $A = 2$, $\eta = 0.1$)

| Walker | $d_i' = g_2(z_d) + 0.1$ | $r_i' = g_2(z_r) + 0.1$ |
|--------|-------------------------|-------------------------|
| 1 | 0.60 | 1.24 |
| 2 | 1.17 | 0.96 |
| 3 | 0.82 | 1.70 |
| 4 | 1.75 | 0.50 |

**Step 4: Fitness** (with $\alpha_{\text{fit}} = \beta_{\text{fit}} = 1$)

| Walker | $V_{\text{fit}} = d' \cdot r'$ | Interpretation |
|--------|-------------------------------|----------------|
| 1 | 0.74 | Good reward, low diversity |
| 2 | 1.12 | Moderate both |
| 3 | 1.39 | **Highest**: good reward AND diversity |
| 4 | 0.88 | High diversity, low reward |

Walker 3 has the highest fitness because it balances both channels well.
:::

---

(sec-cloning-mechanics)=
## 4. Cloning Mechanics

:::{div} feynman-prose
Now we come to the heart of the algorithm: the mechanism by which the swarm *learns*. Everything up to this point---the
walker states, the companion selection, the fitness computation---has been preparation. This is where the action happens.

The idea is natural selection, but in a computational setting. Walkers with low fitness "die" and are replaced by
perturbed copies of high-fitness companions. Over time, the swarm concentrates in high-fitness regions while maintaining
enough diversity (through the cloning jitter) to continue exploring.

But here is the thing that makes this work mathematically: the cloning is *soft*. It is not a deterministic rule that
says "clone if your companion is fitter." It is a probabilistic rule that says "clone with probability that increases
with your companion's fitness advantage." This stochasticity is not a bug---it is what allows the algorithm to escape
local optima and what enables the nice mathematical properties we will prove later.
:::

Cloning is the selection mechanism that drives the swarm toward high-fitness regions. Low-fitness walkers are replaced by perturbed copies of their companions.

:::{figure} figures/fg-cloning-revival.svg
:alt: Diagram of cloning decision, jittered position updates, velocity blending, and revival of dead walkers.
:width: 95%
:align: center

Cloning mechanics: fitter companions attract low-fitness walkers, jitter maintains diversity, and revival preserves population size.
:::

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

The randomized threshold $T_i \sim \text{Uniform}(0, p_{\max})$ introduces stochasticity: even a small fitness advantage
can trigger cloning (if $T_i$ is small), and even a large advantage might not (if $T_i$ is large)—unless $S_i > p_{\max}$
(then cloning is certain). This prevents deterministic collapse and maintains exploration. The stochastic acceptance
mechanism is reminiscent of simulated annealing {cite}`kirkpatrick1983optimization` and Metropolis-Hastings sampling
{cite}`hastings1970monte`. In the gauge-theoretic analogy, the threshold draw is the algorithmic "measurement" that
selects which cloning outcome is realized.
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

:::{div} feynman-prose
Here is where the physics intuition becomes useful. When walkers clone, they are essentially "colliding" in an abstract
sense---multiple walkers end up at nearby positions because they all cloned from the same parent. What should happen to
their velocities?

In a perfectly elastic collision, kinetic energy is conserved and particles bounce off each other. In a perfectly
inelastic collision, particles stick together and move as one unit. The Fractal Gas uses something in between: a
*partially inelastic* collision where walkers' velocities are blended toward their common center of mass.

Why does this matter? Two reasons. First, momentum conservation prevents the cloning operation from artificially pumping
energy into or out of the swarm. If cloning could change the total momentum, the algorithm would have a systematic bias
toward acceleration or deceleration that has nothing to do with the fitness landscape. Second, the partial blending
creates *coordination* among related walkers without complete synchronization. They move in similar directions but retain
some individual variation.

The restitution coefficient $\alpha_{\text{rest}}$ is the dial that controls this tradeoff. At $\alpha_{\text{rest}} = 0$,
all cloned walkers get exactly the center-of-mass velocity---maximum coordination, minimum diversity. At
$\alpha_{\text{rest}} = 1$, velocities are unchanged---minimum coordination, maximum diversity. The default of 0.5
balances these extremes.
:::

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

:::{div} feynman-prose
Here is something that should make you sit up and pay attention. In most evolutionary algorithms, there is a real risk of
*extinction*---the population can dwindle as selection pressure eliminates less fit individuals faster than they can be
replaced. You start with 100 agents, and after a few generations you might have 50, then 20, then 5, then extinction.

The Fractal Gas is different. The population size $N$ is *exactly conserved*. Not approximately, not on average---exactly.
Every step has the same number of walkers as the step before. The cloning operation does not create new walkers or
destroy old ones. It *replaces* low-fitness walkers with copies of high-fitness ones.

This conservation is enforced by the revival rule: any walker that "dies" (violates constraints, falls off the
boundary, etc.) is forced to clone from the alive set in the very next step. The dead walker samples a companion
uniformly at random from the alive population and respawns as a perturbed copy of that companion.

The beauty is that this guarantee is enforced directly by the cloning rule. No extra parameter constraint is required
for revival; the only failure mode is catastrophic (all walkers die simultaneously).
:::

A critical property of the Fractal Gas is that **dead walkers are guaranteed to be revived** (as long as at least one walker remains alive). This prevents gradual extinction and ensures the swarm maintains its population.

:::{prf:definition} Revival Rule (Implementation)
:label: def-fg-revival-rule

Dead walkers are forced to clone: the cloning decision for any walker with $s_i = 0$ is set to **true** (equivalently,
its cloning probability is set to 1). Companions for dead walkers are drawn uniformly from the alive set.
:::

:::{prf:proposition} Guaranteed Revival
:label: prop-fg-guaranteed-revival

Under the revival rule, if $|\mathcal{A}(\mathcal{S})| \geq 1$, then every dead walker clones with probability 1.
:::

*Proof.* The cloning operator explicitly sets the cloning decision for dead walkers to true. Since each dead walker
draws its companion from $\mathcal{A}(\mathcal{S})$, it respawns as a perturbed copy of an alive walker with
probability 1. $\square$

:::{note}
:class: feynman-added
If you disable the explicit dead-always-clone override and apply the same stochastic rule to all walkers, a sufficient
condition for guaranteed revival is $\varepsilon_{\text{clone}} \cdot p_{\max} < V_{\min}$. This is **not** required in
the default implementation.
:::

:::{div} feynman-prose
The revival guarantee is the safety net of the Fractal Gas. No matter how poorly the search is going, dead walkers will
always be reborn near alive ones. The only failure mode is **catastrophic**: all walkers dying simultaneously. In many
settings this probability decays rapidly with $N$ (for example, like $p^N$ if deaths are weakly dependent), so for
moderate swarm sizes it is typically negligible—but it is not literally zero.

This is a key difference from genetic algorithms {cite}`goldberg1989genetic`, where population can dwindle through selection pressure. The Fractal Gas maintains constant population $N$—it's the *distribution* of walkers that evolves, not the count.
:::

---

(sec-kinetic-dynamics)=
## 6. Kinetic Dynamics

Between cloning events, walkers evolve according to a **kinetic operator** that moves them through the latent space. The Fractal Gas framework is agnostic to the specific kinetic scheme; different choices yield different algorithm variants.

:::{div} feynman-prose
So far we have talked about the swarm-level intelligence: how walkers select companions, how fitness is computed, how
cloning redistributes the population. But we have not said anything about how individual walkers *move*.

This is by design. The Fractal Gas is a framework, not a single algorithm. The companion selection and cloning mechanics
are fixed---they define what it means to be a Fractal Gas. But the kinetic operator is a plug-in module. You can choose
Brownian motion if you want pure diffusion. You can choose Langevin dynamics if you want to balance diffusion with drift
toward low-potential regions. You can choose Hamiltonian Monte Carlo if you want to exploit gradient information
efficiently.

The choice of kinetic operator affects the algorithm's exploration characteristics. Fast diffusion explores quickly but
might overshoot optima. Slow diffusion is more precise but might get stuck. Gradient-based kinetics can be very efficient
if the gradient is informative, but can mislead if the local gradient points away from global optima.

The key constraint is that the kinetic operator must inject *some* noise. A completely deterministic kinetic operator
would make the swarm's exploration entirely dependent on the cloning jitter, which is usually not enough.
:::

### 6.1 Abstract Kinetic Interface

:::{div} feynman-prose
The kinetic operator is deliberately left abstract here because the Fractal Gas framework does not care how walkers move between cloning events. It only cares that they move in a way that injects some noise and respects some basic regularity conditions.

This is a good design principle: separate the concerns. The companion selection and cloning mechanics handle the
swarm-level intelligence, the selection pressure that drives walkers toward good regions. The kinetic operator handles
the individual-level dynamics, how each walker explores its local neighborhood.

You can swap in different kinetic operators without changing the rest of the algorithm. Brownian motion works
{cite}`einstein1905bewegung`. Langevin dynamics works {cite}`leimkuhler2015molecular`. Hamiltonian Monte Carlo works
{cite}`neal2011mcmc`. Each choice gives different exploration characteristics, but the overall framework remains the
same.
:::

:::{prf:definition} Kinetic Operator (Abstract)
:label: def-fg-kinetic-abstract

A **kinetic operator** is a Markov transition kernel $K_\tau: (\mathcal{Z} \times T\mathcal{Z}) \to (\mathcal{Z} \times T\mathcal{Z})$ that:

1. **Preserves the alive set**: $K_\tau$ acts only on $(z, v)$, not on status $s$
2. **Admits a stationary measure**: There exists a reference measure $\mu$ such that $K_\tau^* \mu = \mu$ (or $K_\tau$ contracts toward $\mu$) {cite}`leimkuhler2015molecular`
3. **Injects noise**: $K_\tau$ is not deterministic; it has a positive diffusion component

The kinetic operator advances walkers by time step $\tau$:

$$
(z_i', v_i') \sim K_\tau(\cdot \mid z_i, v_i)
$$
:::

### 6.2 Concrete Example: Boris-BAOAB

The **Boris-BAOAB** scheme is a splitting integrator for Langevin dynamics {cite}`leimkuhler2016efficient` on Riemannian manifolds, adapted for the Fractal Gas with anisotropic diffusion.

:::{div} feynman-prose
Let me explain what BAOAB actually does, because the notation can be intimidating but the idea is simple.

Imagine you have a particle moving in a potential landscape with friction. At each moment, three things are happening:
(1) the potential is pushing the particle downhill, (2) friction is slowing the particle down, and (3) random thermal
noise is jittering the particle around.

The challenge is that these three effects interact in complicated ways. If you try to simulate them all at once, you get
numerical errors that accumulate over time. The BAOAB scheme splits the dynamics into separate steps that can each be
handled exactly or nearly exactly.

**B** (kick) handles the potential force: update momentum based on the gradient of the potential.

**A** (drift) handles the motion: update position based on the current momentum.

**O** (thermostat) handles friction and noise together: damp the momentum and add random thermal noise.

By interleaving these steps in the order B-A-O-A-B, you get a symmetric integrator that has much better long-term
accuracy than simpler schemes. The symmetry is important---it prevents systematic drift in the energy of the system.

The "Boris" prefix refers to an optional rotation step for particles in magnetic fields. For most optimization
applications, you can ignore it.
:::

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
\Sigma_{\text{reg}}(z_i) = \left(\nabla_{z_i}^2 V_{\text{fit}}^{(i)} + \epsilon_\Sigma I\right)^{-1/2}
$$

where $\nabla_{z_i}^2 V_{\text{fit}}^{(i)}$ is the per-walker Hessian of fitness with respect to $z_i$ (companions and other walkers treated as frozen), and $\epsilon_\Sigma > 0$ is a regularizer ensuring positive definiteness. The OU step scales this shape by the thermostat amplitude $c_2$.
:::

:::{prf:remark} Mean-Field Fitness Field
:label: rem-mean-field-fitness-field-intuition

In the mean-field limit $N \to \infty$, the per-walker fitness induces a deterministic field
$V_{\mathrm{fit}}(z; \mu)$ obtained by averaging over companion selection and using statistics
computed from the limiting measure $\mu$ (global if $\rho=\varnothing$, localized if $\rho$ is finite).
For finite $N$, the algorithm samples this field only at walker locations. See Definition
{prf:ref}`def-mean-field-fitness-field`.
:::

:::{div} feynman-prose
Why anisotropic diffusion? Consider a fitness landscape with a narrow valley: stiff in one direction (walls of the
valley) and flat in another (along the valley floor). Isotropic diffusion would waste energy bouncing off the walls.
Anisotropic diffusion, scaled by the inverse square root of the Hessian, injects more noise along flat directions and
less along stiff directions. The walker slides along the valley floor instead of bouncing off walls.

This is the same principle behind preconditioned gradient descent and natural gradient methods {cite}`amari1998natural`—adapting the step size to the local geometry of the loss landscape.
:::

---

(sec-step-operator)=
## 7. The Complete Step Operator

:::{div} feynman-prose
We have now seen all the pieces: companion selection, fitness computation, cloning, and kinetics. Let me show you how
they fit together into a single coherent step.

The order matters. First we evaluate fitness based on the current configuration. Then we make cloning decisions based on
that fitness. Then we move the walkers according to the kinetic operator. This ordering ensures that cloning decisions
are based on where walkers *are*, not where they are about to move to.

You might wonder: could we reorder these steps? Could we move first and then clone? The answer is yes, you could, and
you would get a slightly different algorithm. The choice of ordering is a design decision. The fitness-then-clone-then-move
ordering has the property that newly cloned walkers immediately begin exploring from their new positions, which spreads
information through the swarm more quickly.
:::

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
   - Update position: $\tilde{z}_i = z_{c_i^{\text{clone}}} + \sigma_x \zeta_i$
4. For each $i$ with $S_i \leq T_i$: keep $\tilde{z}_i = z_i$, $\tilde{v}_i = v_i$, and $\tilde{s}_i = s_i$
5. Group cloned walkers by parent and apply inelastic collision to produce intermediate velocities $\tilde{v}_i$ (non-cloned walkers keep $\tilde{v}_i = v_i$)
6. Set intermediate status: $\tilde{s}_i = 1$ for cloned walkers

**Step 3 (Kinetics + Killing)**:
1. Apply kinetic operator: $(z_i', v_i') \sim K_\tau(\cdot \mid \tilde{z}_i, \tilde{v}_i)$ for each $i$
2. Apply a status check (boundary/constraints) to set $s_i' \in \{0,1\}$ from $(z_i', v_i')$

**Output**: Updated swarm $\mathcal{S}' = ((z_i', v_i', s_i'))_{i=1}^N$

**Cemetery**: If companion selection for alive walkers is undefined (for example, $|\mathcal{A}(\mathcal{S})| < 2$) and
no special case is used, transition to absorbing state $\dagger$.
:::

:::{div} feynman-prose
The order of operations matters. Fitness is computed first because it depends on the current swarm configuration. Cloning comes next because it uses the fitness values. Kinetics comes last because it advances positions *after* cloning has redistributed walkers.

The cemetery state $\dagger$ is a theoretical convenience: if fewer than two walkers remain alive, the companion kernel
for alive walkers needs a special case. In practice, catastrophic extinction (all walkers dying in one step) can be made
extremely unlikely, and the single-survivor regime can be handled explicitly (see Section 2).
:::

---

(sec-algorithm-variants)=
## 8. Algorithm Variants

:::{div} feynman-prose
The Fractal Gas is not one algorithm but a *family* of algorithms. The core mechanisms---soft companion selection,
dual-channel fitness, momentum-conserving cloning---are fixed. But the state space, the reward function, and the kinetic
operator are all customizable.

This is a good design pattern. The fixed parts are the parts that give the algorithm its mathematical guarantees: the
minorization property that ensures mixing, the revival guarantee that prevents extinction, the momentum conservation that
prevents energy injection. The customizable parts are the parts that connect the abstract algorithm to your specific
problem.

Let me describe three important variants. They are not fundamentally different algorithms---they are the same algorithm
applied to different settings.
:::

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

:::{div} feynman-prose
Now we come to what I consider the most beautiful aspect of the Fractal Gas. We have described a discrete algorithm: $N$
walkers, discrete time steps, stochastic cloning decisions. But there is a hidden continuous structure underneath.

As you take more and more walkers and smaller and smaller time steps, the noisy discrete swarm smooths out into a
continuous density. Instead of tracking $N$ individual walkers, you track a density function $\rho(z, t)$ that tells you
how much "walker mass" is at each location at each time. And this density evolves according to a partial differential
equation---a reaction-diffusion equation with a very specific structure.

Why does this matter? Because PDEs are things we understand extremely well. Two hundred years of mathematical analysis
have given us powerful tools for understanding how solutions behave, when they converge, how fast they converge. By
showing that the Fractal Gas converges to a well-behaved PDE, we import all that mathematical machinery. The convergence
guarantees of the PDE become (with appropriate error bounds) convergence guarantees for the discrete algorithm.

This is the payoff for all the careful design: the soft companion selection, the dual-channel fitness, the momentum
conservation. Each of these choices was made to ensure a clean continuum limit. The algorithm was not designed first and
then analyzed---it was designed *for* analyzability.
:::

In the limit of many walkers ($N \to \infty$) and small time steps ($\tau \to 0$), the Fractal Gas converges to a continuum description governed by a reaction-diffusion PDE.

### The Darwinian Ratchet

The one-step operator decomposes into **transport** (kinetics) and **reaction** (selection/cloning):

$$
P_\tau = T_\tau \circ R_\tau
$$

where:
- $T_\tau$ = transport (Langevin diffusion)
- $R_\tau$ = reaction (fitness-weighted resampling)

In a simplified Euclidean setting (no boundary killing, scalar potential reward), this yields a **reaction-diffusion / replicator** equation; WFR provides the natural metric geometry for transport+reaction {cite}`liero2018optimal,chizat2018interpolating`:

$$
\partial_t \rho = \underbrace{\nabla \cdot (\rho \nabla \Phi) + \Delta \rho}_{\text{transport (Fokker--Planck)}} + \underbrace{\rho (V_{\text{fit}} - \bar{V}_{\text{fit}})}_{\text{reaction (replicator)}}
$$

:::{div} feynman-prose
Let me explain what each piece of this equation is doing, because once you see it, the whole algorithm starts to feel inevitable.

The term $\Delta \rho$ is diffusion. It spreads the walkers out, like cream dispersing in coffee. If you did nothing but diffuse, the swarm would eventually become a uniform smear across the entire space. This is pure exploration with no exploitation.

The term $\nabla \cdot (\rho \nabla \Phi)$ is drift. It pushes walkers downhill on the potential $\Phi$. If the potential represents negative reward, this pulls walkers toward high-reward regions. If you did nothing but drift, the swarm would collapse to a single point at the global minimum. This is pure exploitation with no exploration.

The term $\rho(V_{\text{fit}} - \bar{V}_{\text{fit}})$ is the replicator dynamics {cite}`hofbauer1998evolutionary`. It says that density grows where fitness exceeds the mean and shrinks where fitness is below the mean. The $-\bar{V}_{\text{fit}}$ term is crucial: it conserves total mass. Without it, the equation would create or destroy walkers out of nothing.

The Darwinian Ratchet is a beautiful synthesis: the diffusion term spreads mass outward (exploration), the drift term pulls mass toward low-potential regions (gradient descent), and the reaction term amplifies mass in high-fitness regions while depleting low-fitness regions (natural selection). Together, they create a "ratchet" that steadily concentrates mass at optima while maintaining diversity.

This is why the Fractal Gas works: it's not just a heuristic algorithm—it's a discretization of a well-understood PDE. The convergence guarantees of the PDE transfer (with appropriate error bounds) to the discrete algorithm.
:::

### Mean-Field Limit

As $N \to \infty$, the empirical measure $\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{(z_i, v_i)}$ converges to a deterministic measure $\mu_t$ solving the mean-field equation {cite}`sznitman1991topics,del2004feynman`. The error bound (propagation of chaos) is:

$$
\mathbb{E}[W_2(\mu_N, \mu)] \lesssim \frac{e^{-\kappa_W t}}{\sqrt{N}}
$$

where $\kappa_W > 0$ is the Wasserstein contraction rate certified by the sieve.

For formal treatment of convergence, see {doc}`02_fractal_gas_latent` (Part III-B).

---

(sec-parameter-glossary)=
## Parameter Glossary

:::{div} feynman-prose
Here are all the knobs you can turn. It looks like a lot, but let me organize them for you by function:

**Swarm size** ($N$): How many walkers you have. More walkers means better coverage of the space but more computation.
Start with 50 and increase if you see signs of poor coverage.

**Locality** ($\epsilon$, $\lambda_{\text{alg}}$): How local is the companion selection? Smaller $\epsilon$ means walkers
only talk to very nearby companions. $\lambda_{\text{alg}}$ controls whether velocity similarity matters.

**Exploitation vs. exploration** ($\alpha_{\text{fit}}$, $\beta_{\text{fit}}$): These exponents control the tradeoff.
Start with both equal to 1. If your swarm converges too fast to suboptimal solutions, increase $\beta_{\text{fit}}$
(favor diversity). If it explores forever without converging, increase $\alpha_{\text{fit}}$ (favor reward).

**Cloning aggressiveness** ($\varepsilon_{\text{clone}}$, $p_{\max}$, $\sigma_x$, $\alpha_{\text{rest}}$): How
aggressively does the swarm clone, and how similar are the clones? Higher $p_{\max}$ means more cloning. Smaller
$\sigma_x$ means clones are closer to their parents. Smaller $\alpha_{\text{rest}}$ means cloned walkers move more
similarly.

**Kinetic temperature** ($h$, $\gamma$, $T_c$): How fast and how noisily do walkers move? Higher $T_c$ means more thermal
noise (more exploration). Higher $\gamma$ means more friction (faster equilibration but slower exploration). Smaller $h$
means more accurate integration but more computation.
:::

| Category | Symbol | Default | Unit | Description |
|----------|--------|---------|------|-------------|
| **Swarm** | $N$ | 50 | [count] | Number of walkers |
| **Companion** | $\epsilon$ | 0.1 | [distance] | Kernel bandwidth |
| **Companion** | $\lambda_{\text{alg}}$ | 0.0 | [dimensionless] | Velocity weight in $d_{\text{alg}}$ |
| **Fitness** | $\alpha_{\text{fit}}$ | 1.0 | [dimensionless] | Reward exponent |
| **Fitness** | $\beta_{\text{fit}}$ | 1.0 | [dimensionless] | Diversity exponent |
| **Fitness** | $\eta$ | 0.1 | [dimensionless] | Positivity floor |
| **Fitness** | $A$ | 2.0 | [dimensionless] | Logistic amplitude |
| **Fitness** | $\sigma_{\min}$ | 1e-8 | [dimensionless] | Standardization regularizer |
| **Fitness** | $\epsilon_{\text{dist}}$ | 1e-8 | [dimensionless] | Distance regularizer |
| **Cloning** | $\varepsilon_{\text{clone}}$ | 0.01 | [dimensionless] | Score regularizer |
| **Cloning** | $p_{\max}$ | 1.0 | [probability] | Max threshold |
| **Cloning** | $\sigma_x$ | 0.1 | [distance] | Jitter scale |
| **Cloning** | $\alpha_{\text{rest}}$ | 0.5 | [dimensionless] | Restitution coefficient |
| **Kinetic** | $h$ | 0.01 | [time] | Time step |
| **Kinetic** | $\gamma$ | 1.0 | [1/time] | Friction coefficient |
| **Kinetic** | $T_c$ | 1.0 | [dimensionless] | Cognitive temperature |
| **Kinetic** | $\epsilon_\Sigma$ | 1e-4 | [dimensionless] | Diffusion regularizer |

:::{admonition} Revival Constraint Check
:class: feynman-added warning

In the default implementation, dead walkers are forced to clone, so **no parameter constraint is required for revival**.
If you remove that override and use the stochastic rule for all walkers, a sufficient condition is:

$$
\varepsilon_{\text{clone}} \cdot p_{\max} < \eta^{\alpha_{\text{fit}} + \beta_{\text{fit}}}.
$$
:::

:::{admonition} Troubleshooting Guide
:class: feynman-added tip

**Swarm collapses to a single point:** Increase $\beta_{\text{fit}}$ (favor diversity), increase $\sigma_x$ (larger
cloning jitter), or increase $\epsilon$ (wider companion selection).

**Swarm never converges:** Decrease $\beta_{\text{fit}}$ (reduce diversity pressure), increase $\alpha_{\text{fit}}$
(favor reward), or decrease $T_c$ (cooler kinetics).

**Dead walkers stay dead:** Check the revival constraint. Decrease $\varepsilon_{\text{clone}}$ or $p_{\max}$, or
increase $\eta$.

**Swarm oscillates wildly:** Increase $\gamma$ (more friction), decrease $T_c$ (cooler temperature), or decrease $h$
(smaller time steps).

**Slow convergence despite good fitness landscape:** Decrease $\epsilon$ (tighter companion selection), increase
$p_{\max}$ (more aggressive cloning), or increase the kinetic step size $h$.
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

:::{div} feynman-prose
And there it is. The Fractal Gas is not a collection of heuristics cobbled together. It is a principled system where
every piece has a reason: soft selection guarantees mixing, dual-channel fitness balances exploration and exploitation,
momentum conservation prevents artificial energy injection, and the revival guarantee prevents extinction. Each piece
enables a mathematical property, and together they yield a discretization of a well-understood reaction-diffusion PDE.

The beauty is that you do not need to understand the PDE to use the algorithm. You can treat it as a black box:
initialize some walkers, run the loop, and watch them converge to the good regions. But if you want to know why it
works, why it has to work given its construction, the mathematics is there waiting for you.
:::

:::{seealso}
:class: feynman-added
- {doc}`02_fractal_gas_latent`: Proof object with full sieve verification
- {doc}`../../2_hypostructure/10_information_processing/02_fractal_gas`: Hypostructure metatheorems
- {doc}`../appendices/01_fragile_gas_framework`: Axiom system and revival guarantee proof
:::
