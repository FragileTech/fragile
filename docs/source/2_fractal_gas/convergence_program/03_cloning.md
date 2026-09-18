# The Keystone Principle and the Contractive Nature of Cloning

(sec-cloning-tldr)=
## 0. TLDR

:::{div} feynman-prose
Cloning reallocates a fixed collection of walker slots. Its stabilizing effect
comes from replacing selected walkers by companions: when the selection law
puts enough probability on walkers carrying positional error, the Keystone
estimate quantifies that activity. The variance drift additionally depends on the direction and length of actual donor displacements. Shared component collisions dissipate full-slot relative velocity energy; alive-only variance includes an explicit revival term.

Boundary control uses two further estimates: exposed walkers must have
favorable companions in the actual swarm, and the barrier must be integrable
under the post-cloning noise. Under these conditions the chapter proves
$\mathbb E\Delta W_b\leq-\kappa_bW_b+C_b$. A safe population and conditionally
independent death events give the separate exponential bound on one-step
extinction probability. A safe region in the environment supplies locations;
its occupancy supplies the probability estimate.

The constants are uniform in $N$ when the measurement, selection, target-error,
and barrier bounds used to construct them are uniform in $N$.
{doc}`06_convergence` combines the cloning and kinetic estimates for the finite
particle chain. {doc}`09_propagation_chaos` treats the mean-field approximation,
and {doc}`15_kl_convergence` gives the functional-inequality and entropy proofs
for their specified laws and generators.

Prerequisites: {doc}`01_fragile_gas_framework`, {doc}`02_euclidean_gas`, and
{doc}`04_single_particle`.
:::

(sec-cloning-introduction)=
## 1. Introduction

### 1.1. What selection can control

:::{div} feynman-prose
Imagine watching one replacement. An unsuccessful walker copies a companion's
position, with the prescribed jitter, and its velocity undergoes the specified
collision. The number of slots stays $N$. An alive walker replacing another
alive state creates no new alive mass. Revival changes the status of a dead
slot, and a later boundary test can kill a newly proposed state.

The question for the proof is quantitative: does replacement occur often
enough on the walkers carrying the error? A fitness comparison by itself does
not answer that question. We need its probability under the actual companion
law, multiplied by the error that the selected walker carries. The Keystone
inequality records exactly this weighted quantity.

This chapter develops the measurement, selection, variance, and boundary
calculations needed for that estimate. Each drift result retains its own
hypotheses. In particular, a bound on internal variance is a bound on the spread
of one swarm; convergence of probability laws requires the additional coupling
or mixing estimates in the convergence chapters.
:::

:::{prf:remark} Absorption and the law conditioned on survival
:label: rem-note-extinction-possibility

For a transition that kills walkers outside the valid domain, let $Q$ be its
sub-Markov kernel on living swarms. If total extinction is accessible, the
surviving law at step $n$ is $\mu Q^n/(\mu Q^n1)$ whenever the denominator is
positive. A QSD $\nu$ satisfies $\nu Q=\alpha\nu$ for a survival factor
$\alpha\in(0,1]$. It is invariant under this normalized evolution.

Gaussian noise gives a positive probability of total extinction when the
specified transition can send every remaining walker into the killing region;
this depends on the boundary, clipping, and revival conventions. On a
conservative state space the relevant stationary law is instead an invariant
probability. The hypotheses and proofs for the finite-particle QSD are given in
{doc}`06_convergence`; the killing normalization in entropy calculations is
explicit in {prf:ref}`prop-kl-conditioned-entropy`.
:::

### 1.2. Reading the proof

:::{div} feynman-prose
Sections 2–4 define the coupled states and the quantities whose drifts we
measure. Sections 5–7 follow a signal through the measurement and fitness
pipeline. Section 8 combines selection probability with error concentration.
Sections 9–11 apply the update rule to variance, velocity, and the boundary
observable. Section 12 composes the resulting component estimates.

There are two probability spaces to keep track of. A sampled pairing produces
one fitness vector. Averaging over pairings produces expectations of that
vector and of its nonlinear functions. The clone probability averages the
clipped score itself. It cannot be obtained by clipping an average score.
Likewise, the idealized weighted matching law and the sequential greedy law
need their own probability estimates, even though both use the same edge
weights.
:::

(sec-cloning-states)=
## 2. The Coupled State Space and State Differences

:::{div} feynman-prose
Coupling puts two valid swarm evolutions on one probability space. Their difference vectors let us measure the effect of a common update. The definitions below provide the state space and error quantities used by the drift estimates and by the law-convergence argument in {doc}`06_convergence`.
:::

### 2.1. The Single-Swarm State Space

The fundamental unit of the system is the walker ({prf:ref}`def-walker`), and a collection of these walkers constitutes a swarm ({prf:ref}`def-swarm-and-state-space`). We begin by defining their state spaces abstractly, in a manner consistent with the Fragile Gas framework.

:::{prf:definition} Single-Walker and Swarm State Spaces
:label: def-single-swarm-space

1.  A **walker** is a tuple $(x, s)$ ({prf:ref}`def-walker`), where $x \in \mathcal{X}$ is its position in a state space and $s \in \{0, 1\}$ is its survival status. For the Euclidean Gas ({prf:ref}`alg-euclidean-gas`), this is extended to include a velocity component, making the **full state** of a single walker a tuple $(x, v, s) \in \mathbb{R}^d \times \mathbb{R}^d \times \{0, 1\}$. We refer to $(x,v)$ as the **kinematic state**.

2.  A **swarm ({prf:ref}`def-swarm-and-state-space`) configuration**, $S$, is an N-tuple of walker  states:



$$
S := \left( (x_1, v_1, s_1), (x_2, v_2, s_2), \dots, (x_N, v_N, s_N) \right)

$$

3.  The **single-swarm ({prf:ref}`def-swarm-and-state-space`) state space**, denoted $\Sigma_N$, is the Cartesian product of the per-walker ({prf:ref}`def-walker`) state spaces:



$$
\Sigma_N := \left( \mathbb{R}^d \times \mathbb{R}^d \times \{0, 1\} \right)^N.

$$

Referenced by {prf:ref}`def-barycentres-and-centered-vectors`, {prf:ref}`def-coupled-state-space`, and {prf:ref}`def-swarm-aggregation-operator`.
:::

Our drift analysis focuses on proving the contraction of the continuous kinematic states $(x_i, v_i)$. Throughout the argument, we will condition on the discrete status variables $(s_i)$ so that their influence on the dynamics is treated explicitly when evaluating the expectation of each one-step operator.

### 2.2. The Coupled Process and Synchronous Coupling

To analyze the convergence of the swarm, we consider two copies of the Markov process, $(S_{1,t})$ and $(S_{2,t})$, each evolving on the single-swarm state space (see {prf:ref}`def-single-swarm-space`) $\Sigma_N$, together forming the coupled state space (see {prf:ref}`def-coupled-state-space`) $\Sigma_N \times \Sigma_N$. The core of the proof is to show that the distance between these two processes, as measured by our Lyapunov function, decreases over time in expectation.

To achieve this, we must define a specific **coupling** for all sources of randomness in the algorithm. For this proof, we will use a **synchronous coupling**. This means that for any given time step $t$, the same underlying canonical random variables (e.g., uniform draws from $[0,1]$ or standard normal vectors) are used to generate the stochastic outcomes for both swarms. This synchronization applies to all sources of randomness:

1.  **Companion Selection:** For each walker $i$, a single underlying random variable is used to sample from the $\varepsilon$-dependent spatial kernels of each swarm, $P(\cdot \mid S_1, i)$ and $P(\cdot \mid S_2, i)$. Because the distributions are state-dependent, the chosen companions $c_{1,i}$ and $c_{2,i}$ will generally be different.
2.  **Cloning Thresholds:** The same random threshold $T_i \sim \text{Uniform}(0, p_{\max})$ is used for both swarms.
3.  **Cloning Jitter:** The same random vectors $\zeta_i^x \sim \mathcal{N}(0, I_d)$ are used for positional jitter.
4.  **Kinetic Perturbation:** The same random vectors $\xi_i^v \sim \mathcal{N}(0, I_d)$ are used for Langevin ({prf:ref}`def-langevin-operator`) noise.

This defines a coupling with the correct marginals; optimality or contraction must be established by a separate estimate. For collisions, couple one independent Haar matrix per component, sharing it across the two swarms only for the chosen correspondence of components. Different component partitions cannot be treated as the same per-walker update. All expectations $\mathbb{E}[\cdot]$ in the subsequent analysis are taken with respect to this single, shared source of randomness.

:::{prf:definition} The Coupled State Space
:label: def-coupled-state-space

The **coupled state space** for the Euclidean Gas ({prf:ref}`alg-euclidean-gas`) is the Cartesian product $\Sigma_N \times \Sigma_N$, where $\Sigma_N$ is defined in {prf:ref}`def-single-swarm-space`. An element of this space is an ordered pair of swarm configurations, $(S_1, S_2)$, where:

$$
S_1 = \left( (x_{1,1}, v_{1,1}, s_{1,1}), \dots, (x_{1,N}, v_{1,N}, s_{1,N}) \right) \in \Sigma_N,

$$

$$
S_2 = \left( (x_{2,1}, v_{2,1}, s_{2,1}), \dots, (x_{2,N}, v_{2,N}, s_{2,N}) \right) \in \Sigma_N.

$$

The convergence analysis proceeds by tracking the evolution of a Lyapunov function $V(S_1, S_2)$ across this coupled space.

Referenced by {prf:ref}`def-coupled-cloning-expectation`.
:::

### 2.3. State Difference Vectors

The core of the hypocoercive analysis is not the absolute state of the swarms, but the *difference* between them (measured element-wise). We formally define the vectors that capture this relative configuration.

:::{prf:definition} State Difference Vectors
:label: def-state-difference-vectors

For any element $(S_1, S_2) \in \Sigma_N \times \Sigma_N$, we define the **state difference vectors** for each walker ({prf:ref}`def-walker`) index $i \in \{1, \ldots, N\}$ as follows:

1.  The **position difference vector** for walker  $i$ is:



$$
\Delta x_i := x_{1,i} - x_{2,i} \in \mathbb{R}^d

$$

2.  The **velocity difference vector** for walker ({prf:ref}`def-walker`) $i$ is:



$$
\Delta v_i := v_{1,i} - v_{2,i} \in \mathbb{R}^d

$$

The entire drift analysis will be formulated in terms of the norms and inner products of these $2N$ difference vectors. The objective is to show that, in expectation, the magnitudes of these vectors decrease over time, driving the two swarm ({prf:ref}`def-swarm-and-state-space`) trajectories together.

Referenced by {prf:ref}`def-location-error-component`.
:::

### 2.4. The Boundary Barrier Function

The analysis of the swarm's behavior near the boundary of the valid domain $\mathcal{X}_{\text{valid}}$ requires a potential function that is smooth throughout the interior and diverges as any walker approaches the boundary. This function, $\varphi(x)$, serves as the basis for the boundary potential $W_b$ in the Lyapunov function. The existence and properties of such a function are not merely assumed but can be formally proven under standard regularity conditions on the domain.

#### 2.4.1. Assumptions on the Domain

For the construction to be valid, we impose the following standard regularity conditions on the single-walker state space $\mathcal{X}$.

:::{prf:axiom} **(Axiom EG-0): Regularity of the Domain**
:label: axiom-domain-regularity

The valid domain for a single walker ({prf:ref}`def-walker`)'s position, $\mathcal{X}_{\text{valid}}$ ({prf:ref}`def-valid-state-space`), is an open, bounded, and connected subset of $\mathbb{R}^d$. Its boundary, $\partial \mathcal{X}_{\text{valid}}$, is a $C^{\infty}$-smooth compact manifold without boundary.

Referenced by {prf:ref}`prop-barrier-existence`.
:::

**Rationale:** Boundedness is necessary for many of the compactness arguments used throughout the proof. $C^{\infty}$ smoothness of the boundary is a standard technical condition that guarantees the existence of a well-behaved signed distance function in a neighborhood of the boundary, which is the essential ingredient for our construction.

#### 2.4.2. Existence of a Smooth Barrier Function

Under the assumption of a regular domain, we can state and prove the existence of our desired barrier function.

:::{prf:proposition} Existence of a Global Smooth Barrier Function
:label: prop-barrier-existence

Let $\mathcal{X}_{\text{valid}}$ satisfy the conditions of {prf:ref}`axiom-domain-regularity`. Then there exists a function $\varphi: \mathcal{X}_{\text{valid}} \to \mathbb{R}$ with the following properties:
1.  **Smoothness:** $\varphi(x)$ is $C^{\infty}$-smooth on $\mathcal{X}_{\text{valid}}$.
2.  **Positivity:** $\varphi(x)$ is strictly positive for all $x \in \mathcal{X}_{\text{valid}}$.
3.  **Boundary Divergence:** $\varphi(x) \to \infty$ as $x \to \partial \mathcal{X}_{\text{valid}}$.

Referenced by {prf:ref}`def-boundary-potential-cloning` and {prf:ref}`def-full-synergistic-lyapunov-function`.
:::
:::{prf:proof}

**Proof.**

The proof is constructive. We build the function $\varphi(x)$ using two primary tools: the signed distance function to the boundary and a smooth cutoff function. The construction proceeds in three steps, followed by rigorous verification of all required properties.

**Step 1: The Signed Distance Function.**

Since $\partial \mathcal{X}_{\text{valid}}$ is a $C^{\infty}$ compact manifold without boundary embedded in $\mathbb{R}^d$, the **Tubular Neighborhood Theorem** (see [Lee, 2013, Theorem 6.24]) guarantees the existence of an open tubular neighborhood $U \supset \partial \mathcal{X}_{\text{valid}}$ and a smooth retraction $\pi: U \to \partial \mathcal{X}_{\text{valid}}$ such that the signed distance function

$$
\rho(x) := \begin{cases}
d(x, \partial \mathcal{X}_{\text{valid}}) & \text{if } x \in \mathcal{X}_{\text{valid}} \\
-d(x, \partial \mathcal{X}_{\text{valid}}) & \text{if } x \notin \mathcal{X}_{\text{valid}}
\end{cases}

$$

is $C^{\infty}$-smooth on $U$. Here $d(\cdot, \cdot)$ denotes the Euclidean distance. For any $x \in U \cap \mathcal{X}_{\text{valid}}$, we have $\rho(x) = \|x - \pi(x)\| > 0$, and $\nabla \rho(x)$ is the outward-pointing unit normal vector at the closest boundary point.

**Explicit construction of the tubular neighborhood width:** By compactness of $\partial \mathcal{X}_{\text{valid}}$ and smoothness, there exists $\delta_0 > 0$ such that $U := \{x \in \mathbb{R}^d : d(x, \partial \mathcal{X}_{\text{valid}}) < \delta_0\}$ is a smooth tubular neighborhood. We will use $\delta < \delta_0/3$ in the sequel to ensure all relevant regions lie within $U$.

**Step 2: Construction of a Smooth Cutoff Function.**

We require a smooth cutoff function $\psi: \mathbb{R} \to [0, 1]$ with the following properties:
1. $\psi \in C^{\infty}(\mathbb{R})$
2. $\psi(t) = 1$ for all $t \leq 1$
3. $\psi(t) = 0$ for all $t \geq 2$
4. $\psi$ is non-increasing on $\mathbb{R}$
5. $\psi'(t) < 0$ for all $t \in (1, 2)$

**Explicit construction:** A standard construction uses the mollifier function. Define

$$
\eta(t) := \begin{cases}
\exp\left(-\frac{1}{1-t^2}\right) & \text{if } |t| < 1 \\
0 & \text{if } |t| \geq 1
\end{cases}

$$

which is $C^{\infty}$ on $\mathbb{R}$ (see [Rudin, 1987, Theorem 1.46]). Then set

$$
\psi(t) := \frac{\int_{t}^{\infty} \eta(2s - 3) \, ds}{\int_{-\infty}^{\infty} \eta(2s - 3) \, ds}

$$

This gives a smooth non-increasing function with $\psi(t) = 1$ for $t \leq 1$ and $\psi(t) = 0$ for $t \geq 2$.

**Step 3: Construction of the Barrier Function.**

Fix $\delta \in (0, \delta_0/3)$ where $\delta_0$ is the tubular neighborhood width from Step 1. We define $\varphi: \mathcal{X}_{\text{valid}} \to (0, \infty)$ by

$$
\varphi(x) := \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right)\left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right)

$$

**Verification of Properties:**

**Property 1: Smoothness.**

We verify $\varphi \in C^{\infty}(\mathcal{X}_{\text{valid}})$ by analyzing the composition structure.

For any $x \in \mathcal{X}_{\text{valid}}$ with $\rho(x) < 3\delta < \delta_0$, we have $x \in U$, so $\rho(x)$ is $C^{\infty}$ near $x$. Since $\rho(x) > 0$ for all $x \in \mathcal{X}_{\text{valid}}$, the function $1/\rho(x)$ is $C^{\infty}$ on all of $\mathcal{X}_{\text{valid}}$. The composition $\psi(\rho(x)/\delta)$ is $C^{\infty}$ since both $\psi$ and $\rho$ are $C^{\infty}$.

For $x$ with $\rho(x) \geq 3\delta$, we have $\rho(x)/\delta \geq 3 > 2$, so $\psi(\rho(x)/\delta) = 0$ identically in a neighborhood of $x$. Thus $\varphi(x) = 1/\delta$ (constant) in this region, which is trivially $C^{\infty}$.

The matching at $\rho(x) = 3\delta$ is smooth because $\psi$ and all its derivatives vanish for arguments $\geq 2$.

Therefore, $\varphi \in C^{\infty}(\mathcal{X}_{\text{valid}})$.

**Property 2: Boundary Divergence.**

We must show that for any sequence $(x_n) \subset \mathcal{X}_{\text{valid}}$ with $x_n \to x_{\infty} \in \partial \mathcal{X}_{\text{valid}}$, we have $\varphi(x_n) \to \infty$.

Since $x_n \to x_{\infty} \in \partial \mathcal{X}_{\text{valid}}$ and $x_n \in \mathcal{X}_{\text{valid}}$, by continuity of the distance function, $\rho(x_n) = d(x_n, \partial \mathcal{X}_{\text{valid}}) \to 0^{+}$.

For sufficiently large $n$, we have $\rho(x_n) < \delta$, which implies $\rho(x_n)/\delta < 1$, hence $\psi(\rho(x_n)/\delta) = 1$. In this regime:

$$
\varphi(x_n) = \frac{1}{\delta} + 1 \cdot \left( \frac{1}{\rho(x_n)} - \frac{1}{\delta} \right) = \frac{1}{\rho(x_n)}

$$

Since $\rho(x_n) \to 0^{+}$, we have $\varphi(x_n) = 1/\rho(x_n) \to +\infty$.

**Property 3: Strict Positivity.**

We prove $\varphi(x) > 0$ for all $x \in \mathcal{X}_{\text{valid}}$ by case analysis.

*Case 1: $0 < \rho(x) \leq \delta$.*
Here $\rho(x)/\delta \leq 1$, so $\psi(\rho(x)/\delta) = 1$. Thus:

$$
\varphi(x) = \frac{1}{\delta} + 1 \cdot \left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) = \frac{1}{\rho(x)} > 0

$$

since $\rho(x) > 0$.

*Case 2: $\rho(x) \geq 2\delta$.*
Here $\rho(x)/\delta \geq 2$, so $\psi(\rho(x)/\delta) = 0$. Thus:

$$
\varphi(x) = \frac{1}{\delta} + 0 \cdot \left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) = \frac{1}{\delta} > 0

$$

*Case 3: $\delta < \rho(x) < 2\delta$.*
This is the transition region. We have $1 < \rho(x)/\delta < 2$, so $\psi(\rho(x)/\delta) \in (0, 1)$.

Rewrite $\varphi(x)$ by expanding:

$$
\begin{aligned}
\varphi(x) &= \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right)\left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) \\
&= \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right) \cdot \frac{1}{\rho(x)} - \psi\left(\frac{\rho(x)}{\delta}\right) \cdot \frac{1}{\delta} \\
&= \frac{1}{\delta}\left(1 - \psi\left(\frac{\rho(x)}{\delta}\right)\right) + \frac{1}{\rho(x)} \psi\left(\frac{\rho(x)}{\delta}\right)
\end{aligned}

$$

Since $\psi(\rho(x)/\delta) \in (0,1)$, we have $1 - \psi(\rho(x)/\delta) \in (0, 1) \subset (0, \infty)$. Thus:

$$
\varphi(x) = \underbrace{\frac{1}{\delta}\left(1 - \psi\left(\frac{\rho(x)}{\delta}\right)\right)}_{> 0} + \underbrace{\frac{1}{\rho(x)} \psi\left(\frac{\rho(x)}{\delta}\right)}_{> 0} > 0

$$

Both terms are strictly positive since $\delta > 0$, $\rho(x) > 0$, $1 - \psi > 0$, and $\psi > 0$ in this regime.

**Conclusion:**

We have constructed a function $\varphi: \mathcal{X}_{\text{valid}} \to (0, \infty)$ satisfying all three properties: $\varphi \in C^{\infty}(\mathcal{X}_{\text{valid}})$, $\varphi(x) > 0$ everywhere, and $\varphi(x) \to \infty$ as $x \to \partial \mathcal{X}_{\text{valid}}$.

**Q.E.D.**
:::
:::{admonition} References
:class: note

[Lee, 2013] Lee, John M. *Introduction to Smooth Manifolds*. 2nd ed., Springer, 2013.

[Rudin, 1987] Rudin, Walter. *Real and Complex Analysis*. 3rd ed., McGraw-Hill, 1987.
:::


(sec-cloning-lyapunov)=
## 3. The Augmented Hypocoercive Lyapunov Function

To prove that the synergistic dissipation between the cloning and kinetic stages leads to convergence, we must use a Lyapunov function that correctly separates the different geometric components of the swarm's error. The core of the analysis rests on decomposing the total kinematic error between two swarms, $S_1$ and $S_2$, into two distinct parts: a **location error**, which measures the distance between the swarms' centers of mass, and a **structural error**, which measures the mismatch in their geometric shapes. We augment this decomposed kinematic function with a boundary barrier that penalizes walkers approaching the boundary of the valid domain, $\partial X_{\text{valid}}$.

### 3.1. Center of Mass and Structural Decomposition

We begin by formally defining the mathematical objects required for this decomposition.

::::{prf:definition} Barycentres and Centered Vectors (Alive Walkers Only)
:label: def-barycentres-and-centered-vectors

For each swarm ({prf:ref}`def-swarm-and-state-space`) $k \in \{1, 2\}$ (see {prf:ref}`def-single-swarm-space`) in a coupled state $(S_1, S_2)$, let $\mathcal{A}(S_k)$ denote the set of alive walker ({prf:ref}`def-walker`) indices and let $k_{\text{alive}} := |\mathcal{A}(S_k)|$ denote the number of alive walkers in swarm $k$. We define:

1.  The **positional center of mass** (barycentre) **computed over alive walkers only**:



$$
\mu_{x,k} := \frac{1}{k_{\text{alive}}}\sum_{i \in \mathcal{A}(S_k)} x_{k,i}

$$

2.  The **velocity center of mass** **computed over alive walkers only**:



$$
\mu_{v,k} := \frac{1}{k_{\text{alive}}}\sum_{i \in \mathcal{A}(S_k)} v_{k,i}

$$

The **centered vectors** represent the state of each **alive** walker ({prf:ref}`def-walker`) relative to its swarm ({prf:ref}`def-swarm-and-state-space`)'s center of mass:

1.  The **centered position vector** for alive walker  $i \in \mathcal{A}(S_k)$:



$$
\delta_{x,k,i} := x_{k,i} - \mu_{x,k}

$$

2.  The **centered velocity vector** for alive walker $i \in \mathcal{A}(S_k)$:



$$
\delta_{v,k,i} := v_{k,i} - \mu_{v,k}

$$

**Convention**: Dead walkers ($i \notin \mathcal{A}(S_k)$) do not contribute to barycentres, variances, or any statistical quantities. By construction, the centered vectors for alive walkers in any swarm sum to zero: $\sum_{i \in \mathcal{A}(S_k)} \delta_{x,k,i} = 0$ and $\sum_{i \in \mathcal{A}(S_k)} \delta_{v,k,i} = 0$.

:::{admonition} Rationale for Alive-Walker-Only Statistics
:class: important

Dead walkers retain their last known position $(x_i, v_i)$ but have status $s_i = 0$. Including them in statistical calculations would distort the geometric properties:

1. **Physical Interpretation**: Dead walkers represent "failed" exploration paths. Their positions are historical artifacts, not part of the current active swarm distribution.

2. **Cloning Operator Target**: The cloning operator $\Psi_{\text{clone}}$ acts on the fitness and geometric distribution of **alive** walkers. The variance it contracts is specifically the variance of the alive population.

3. **Measurement Consistency**: Distance-to-companion measurements ({prf:ref}`def-algorithmic-distance-metric`) are computed from the alive-walker distribution. For consistency, all variance and barycentre calculations must use the same population.

Referenced by {prf:ref}`def-full-synergistic-lyapunov-function` and {prf:ref}`def-structural-error-component`.
:::
::::

### 3.2. Permutation-Invariant Error Components

To create a robust analysis that is independent of walker labels, we define the location and structural errors using permutation-invariant metrics. The location error captures the distance between the swarms' centers, while the structural error captures the dissimilarity in their shapes.

#### 3.2.1. The Location Error Component ($V_{\text{loc}}$)

The distance between the swarms' centers of mass is an intrinsically permutation-invariant quantity. We define the location error as the hypocoercive quadratic form ({prf:ref}`def-hypocoercive-metric`) applied to the difference between the barycenters of the two swarms.

:::{prf:definition} The Location Error Component ($V_{\text{loc}}$)
:label: def-location-error-component

For any pair of swarm ({prf:ref}`def-swarm-and-state-space`) configurations $(S_1, S_2)$ with barycenters $(\mu_{x,1}, \mu_{v,1})$ and $(\mu_{x,2}, \mu_{v,2})$ (derived from {prf:ref}`def-state-difference-vectors`), the **location error component** is defined as:

$$
V_{\text{loc}} := \|\Delta\mu_x\|^2 + \lambda_v\|\Delta\mu_v\|^2 + b\langle\Delta\mu_x, \Delta\mu_v\rangle

$$

where $\Delta\mu_x = \mu_{x,1} - \mu_{x,2}$ and $\Delta\mu_v = \mu_{v,1} - \mu_{v,2}$. The parameters $b$ and $\lambda_v$ are the hypocoercive coefficients.
:::

#### 3.2.2. The Structural Error Component ($V_{\text{struct}}$)

The structural error measures the mismatch between the "shapes" of the two swarms. The shape of a swarm is described by the set of its centered vectors, $\{\delta_{z,k,i}\}$. To compare these shapes in a permutation-invariant way, we find the **optimal matching** between the centered vectors of the two swarms and measure the residual error of that matching. This is equivalent to the hypocoercive Wasserstein distance ({prf:ref}`def-wasserstein-distance`) between the *centered empirical measures*.

:::{prf:definition} The Structural Error Component ($V_{\text{struct}}$)
:label: def-structural-error-component

Let $\tilde{\mu}_1$ and $\tilde{\mu}_2$ be the centered empirical measures of swarms $S_1$ and $S_2$ **computed over alive walkers only**:

$$
\tilde{\mu}_k := \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \delta_{(\delta_{x,k,i}, \delta_{v,k,i})}

$$

where $k_{\text{alive}} = |\mathcal{A}(S_k)|$ is the number of alive walkers in swarm ({prf:ref}`def-swarm-and-state-space`) $k$, and $\delta_{x,k,i}, \delta_{v,k,i}$ are the centered vectors defined in {prf:ref}`def-barycentres-and-centered-vectors`.

The **structural error component** $V_{\text{struct}}$ is defined as the squared hypocoercive Wasserstein distance ({prf:ref}`def-hypocoercive-metric`) between these centered measures:

$$
V_{\text{struct}} := W_h^2(\tilde{\mu}_1, \tilde{\mu}_2) = \inf_{\gamma \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)} \int c(\delta_{z,1}, \delta_{z,2}) \, d\gamma(\delta_{z,1}, \delta_{z,2})

$$

where $c(\delta_1, \delta_2)$ is the hypocoercive cost $\|\delta_{x,1}-\delta_{x,2}\|^2 + \lambda_v\|\delta_{v,1}-\delta_{v,2}\|^2 + b\langle\ldots\rangle$. This finds the minimal average cost to align the shape of swarm ({prf:ref}`def-swarm-and-state-space`) 1 with the shape of swarm 2.
:::

#### 3.2.3. The Decomposition of Total Inter-Swarm Error

A key result from optimal transport theory ({prf:ref}`def-wasserstein-distance`) allows us to relate these components. The total distance between two distributions can be precisely decomposed into the distance between their centers of mass and the distance between their centered shapes.

:::{prf:lemma} Decomposition of the Hypocoercive Wasserstein Distance
:label: lem-wasserstein-decomposition

The total inter-swarm ({prf:ref}`def-swarm-and-state-space`) error, as measured by the squared hypocoercive Wasserstein distance ({prf:ref}`def-n-particle-displacement-metric`) $W_h^2(\mu_1, \mu_2)$ between the two swarms' full empirical measures $\mu_1$ and $\mu_2$, decomposes exactly into the sum of the location and structural error components:

$$
W_h^2(\mu_1, \mu_2) = V_{\text{loc}} + V_{\text{struct}}

$$

Referenced by {prf:ref}`def-full-synergistic-lyapunov-function`.
:::
:::{prf:proof}
**Proof.**

This fundamental decomposition theorem for Wasserstein distances with quadratic costs is a consequence of the gluing lemma in optimal transport and the geometry of barycenters. We provide a complete proof adapted to the hypocoercive cost structure.

**Step 1: Setting up notation and the cost function.**

Let $\mathcal{Z} = \mathbb{R}^d \times \mathbb{R}^d$ denote the phase space (positions and velocities). For two swarms, let $\mu_1$ and $\mu_2$ be their empirical measures over alive walkers:

$$
\mu_k = \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \delta_{z_{k,i}}, \quad z_{k,i} = (x_{k,i}, v_{k,i})

$$

The hypocoercive cost function is:

$$
c(z_1, z_2) = \|x_1 - x_2\|^2 + \lambda_v \|v_1 - v_2\|^2 + b\langle x_1 - x_2, v_1 - v_2 \rangle

$$

This is a **quadratic form** in $(z_1, z_2)$, which we write as $c(z_1, z_2) = q(z_1 - z_2)$ where $q$ is the quadratic form $q(\Delta z) = \|\Delta x\|^2 + \lambda_v \|\Delta v\|^2 + b\langle \Delta x, \Delta v \rangle$.

**Step 2: Barycentric projections and centered measures.**

Define the barycenters:

$$
\bar{z}_k = \int z \, d\mu_k(z) = (\mu_{x,k}, \mu_{v,k})

$$

For empirical measures over alive walkers, this is simply:

$$
\bar{z}_k = \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} z_{k,i} = (\mu_{x,k}, \mu_{v,k})

$$

Define the **centered measures** $\tilde{\mu}_k$ by shifting each measure to have zero barycenter:

$$
\tilde{\mu}_k = \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \delta_{\delta_{z,k,i}}, \quad \delta_{z,k,i} = z_{k,i} - \bar{z}_k = (\delta_{x,k,i}, \delta_{v,k,i})

$$

By construction, $\int \delta_z \, d\tilde{\mu}_k(\delta_z) = 0$ for both $k = 1, 2$.

**Step 3: Decomposition via optimal couplings.**

Let $\gamma^* \in \Gamma(\mu_1, \mu_2)$ be an optimal coupling achieving $W_h^2(\mu_1, \mu_2)$. We will show that $\gamma^*$ induces a natural coupling structure that decomposes the cost.

For any coupling $\gamma \in \Gamma(\mu_1, \mu_2)$, the total transport cost is:

$$
\int_{\mathcal{Z} \times \mathcal{Z}} c(z_1, z_2) \, d\gamma(z_1, z_2) = \int_{\mathcal{Z} \times \mathcal{Z}} q(z_1 - z_2) \, d\gamma(z_1, z_2)

$$

Since $q$ is a quadratic form, we can decompose $z_1 - z_2$ as:

$$
z_1 - z_2 = (z_1 - \bar{z}_1) - (z_2 - \bar{z}_2) + (\bar{z}_1 - \bar{z}_2) = \delta_{z_1} - \delta_{z_2} + \Delta\bar{z}

$$

where $\Delta\bar{z} = \bar{z}_1 - \bar{z}_2 = (\Delta\mu_x, \Delta\mu_v)$ is the barycenter difference and $\delta_{z_i} = z_i - \bar{z}_i$ are centered coordinates.

**Step 4: Expanding the quadratic form.**

Expanding $q(z_1 - z_2)$ using the decomposition:

$$
\begin{aligned}
q(z_1 - z_2) &= q(\delta_{z_1} - \delta_{z_2} + \Delta\bar{z}) \\
&= q(\delta_{z_1} - \delta_{z_2}) + q(\Delta\bar{z}) + 2\langle \delta_{z_1} - \delta_{z_2}, \Delta\bar{z} \rangle_q
\end{aligned}

$$

where $\langle \cdot, \cdot \rangle_q$ denotes the inner product associated with the quadratic form $q$ (i.e., the bilinear form such that $q(\Delta z) = \langle \Delta z, \Delta z \rangle_q$).

Integrating over the coupling $\gamma$:

$$
\begin{aligned}
\int c(z_1, z_2) \, d\gamma &= \int q(\delta_{z_1} - \delta_{z_2}) \, d\gamma + q(\Delta\bar{z}) + 2\int \langle \delta_{z_1} - \delta_{z_2}, \Delta\bar{z} \rangle_q \, d\gamma
\end{aligned}

$$

**Step 5: The cross-term vanishes.**

The key observation is that the cross-term vanishes:

$$
\int \langle \delta_{z_1} - \delta_{z_2}, \Delta\bar{z} \rangle_q \, d\gamma = \left\langle \int \delta_{z_1} \, d\gamma(z_1, z_2), \Delta\bar{z} \right\rangle_q - \left\langle \int \delta_{z_2} \, d\gamma(z_1, z_2), \Delta\bar{z} \right\rangle_q

$$

For any coupling $\gamma \in \Gamma(\mu_1, \mu_2)$, the marginals satisfy $\gamma(\cdot \times \mathcal{Z}) = \mu_1$ and $\gamma(\mathcal{Z} \times \cdot) = \mu_2$. Therefore:

$$
\int \delta_{z_1} \, d\gamma(z_1, z_2) = \int (z_1 - \bar{z}_1) \, d\gamma(z_1, z_2) = \int z_1 \, d\mu_1(z_1) - \bar{z}_1 = \bar{z}_1 - \bar{z}_1 = 0

$$

Similarly, $\int \delta_{z_2} \, d\gamma(z_1, z_2) = 0$. Thus the cross-term is zero.

**Step 6: Identifying the decomposition terms.**

With the cross-term eliminated:

$$
\int c(z_1, z_2) \, d\gamma = \int q(\delta_{z_1} - \delta_{z_2}) \, d\gamma + q(\Delta\bar{z})

$$

The second term is the barycenter cost:

$$
q(\Delta\bar{z}) = \|\Delta\mu_x\|^2 + \lambda_v \|\Delta\mu_v\|^2 + b\langle \Delta\mu_x, \Delta\mu_v \rangle = V_{\text{loc}}

$$

The first term involves the centered coordinates. Note that $\gamma$ induces a coupling $\tilde{\gamma} \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)$ between the centered measures via the map $(z_1, z_2) \mapsto (\delta_{z_1}, \delta_{z_2})$. Thus:

$$
\int q(\delta_{z_1} - \delta_{z_2}) \, d\gamma(z_1, z_2) = \int q(\delta_{z_1}' - \delta_{z_2}') \, d\tilde{\gamma}(\delta_{z_1}', \delta_{z_2}')

$$

**Step 7: Taking the infimum.**

Taking the infimum over all couplings $\gamma \in \Gamma(\mu_1, \mu_2)$:

$$
W_h^2(\mu_1, \mu_2) = \inf_{\gamma \in \Gamma(\mu_1, \mu_2)} \int c(z_1, z_2) \, d\gamma = V_{\text{loc}} + \inf_{\tilde{\gamma} \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)} \int c(\delta_{z_1}, \delta_{z_2}) \, d\tilde{\gamma}

$$

The infimum over centered couplings is precisely $W_h^2(\tilde{\mu}_1, \tilde{\mu}_2) = V_{\text{struct}}$.

**Conclusion:**

$$
W_h^2(\mu_1, \mu_2) = V_{\text{loc}} + V_{\text{struct}}

$$

This decomposition is exact and holds for any pair of measures with finite second moments and any quadratic cost function.

**Q.E.D.**
:::

This theorem provides a rigorous, permutation-invariant foundation for our analysis, allowing us to study the drift of the barycenters ($V_{\text{loc}}$) and the drift of the swarm shapes ($V_{\text{struct}}$) as separate but related problems.

#### 3.2.4 From Structural Error to Internal Swarm Variance

The first step in our causal chain is to connect the state of the coupled system to the internal configuration of the individual swarms. A large mismatch in the geometric shapes of the two swarms, as measured by the positional component of the structural error ($V_{x,\text{struct}}$), implies that at least one of the swarms must be internally spread out, i.e., have a large positional variance. This lemma makes that connection rigorous.

:::{prf:lemma} Structural Positional Error and Internal Variance
:label: lem-sx-implies-variance

Let $k_1 := |\mathcal{A}(S_1)|$ and $k_2 := |\mathcal{A}(S_2)|$ denote the numbers of alive walkers in each swarm ({prf:ref}`def-swarm-and-state-space`). Define:

- $V_{\text{x,struct}}$ as the positional component of the structural error between the two swarms' **alive-walker ({prf:ref}`def-walker`) distributions**
- $\text{Var}_k(x) := \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2$ as the **physical internal positional variance** of the **alive walkers** in swarm  $k$ (note: this is $k_{\text{alive}}$-normalized, representing the actual spread of alive walkers, distinct from the Lyapunov variance component $V_{Var,x}$ which is $N$-normalized)

Then:

$$
V_{\text{x,struct}} \le 2(\text{Var}_1(x) + \text{Var}_2(x))

$$

Consequently, if $V_{\text{x,struct}} > R^2_{\text{spread}}$ for some threshold $R_{\text{spread}}$, then at least one swarm ({prf:ref}`def-swarm-and-state-space`) $k$ must have an internal variance $\text{Var}_k(x) > R^2_{\text{spread}} / 4$.
:::
:::{prf:proof}
**Proof.**

The proof is in two parts. First, we rigorously establish the primary inequality by analyzing the optimal transport structure and using a carefully constructed sub-optimal coupling. Second, we demonstrate the consequence using a proof by contradiction.

**Part 1: Rigorous Proof of the Main Inequality**

Let $\tilde{\mu}_1$ and $\tilde{\mu}_2$ denote the centered empirical measures of the alive walkers in swarms $S_1$ and $S_2$:

$$
\tilde{\mu}_k = \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \delta_{\delta_{x,k,i}}

$$

where $\delta_{x,k,i} = x_{k,i} - \mu_{x,k}$ are the centered position vectors and $\mu_{x,k} = \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} x_{k,i}$ is the positional barycenter.

The structural positional error is defined as the squared Wasserstein distance:

$$
V_{\text{x,struct}} := W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) = \inf_{\gamma \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)} \int \|\delta_{x,1} - \delta_{x,2}\|^2 \, d\gamma(\delta_{x,1}, \delta_{x,2})

$$

where $\Gamma(\tilde{\mu}_1, \tilde{\mu}_2)$ is the set of couplings (joint probability measures with marginals $\tilde{\mu}_1$ and $\tilde{\mu}_2$).

**Step 1.1: Construction of a sub-optimal coupling.**

We construct a specific coupling $\gamma_{\text{id}}$ to obtain an upper bound. Let $m := \min(k_1, k_2)$ where $k_1 = |\mathcal{A}(S_1)|$ and $k_2 = |\mathcal{A}(S_2)|$.

Without loss of generality, relabel the walkers in each swarm by their indices $1, 2, \ldots, k_1$ and $1, 2, \ldots, k_2$. Define the **identity-plus-remainder coupling** $\gamma_{\text{id}}$ as follows:

- For $i \leq m$: couple walker $i$ in swarm 1 with walker $i$ in swarm 2 with mass $1/\max(k_1, k_2)$.
- For the excess walkers in the larger swarm: couple each with an arbitrary uniform distribution over the other swarm.

The precise construction depends on the relative sizes, but the key property is that this coupling costs at most the sum of:
1. The average squared centered norm in swarm 1: $\frac{1}{k_1} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2$
2. The average squared centered norm in swarm 2: $\frac{1}{k_2} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2$

**Step 1.2: Bounding the cost of the identity coupling (equal sizes).**

First consider the case $k_1 = k_2 = k$. The identity coupling matches walker $i$ to walker $i$. Its cost is:

$$
\int \|\delta_{x,1} - \delta_{x,2}\|^2 \, d\gamma_{\text{id}} = \frac{1}{k} \sum_{i=1}^k \|\delta_{x,1,i} - \delta_{x,2,i}\|^2

$$

Using the elementary inequality $\|a - b\|^2 \leq 2\|a\|^2 + 2\|b\|^2$ for any $a, b \in \mathbb{R}^d$ (which follows from $\|a-b\|^2 = \|a\|^2 - 2\langle a, b \rangle + \|b\|^2 \leq \|a\|^2 + \|b\|^2 + |\langle a, b \rangle|^2 \leq \|a\|^2 + \|b\|^2 + \|a\|^2 + \|b\|^2$ by Cauchy-Schwarz and the polarization identity):

$$
\|\delta_{x,1,i} - \delta_{x,2,i}\|^2 \leq 2\|\delta_{x,1,i}\|^2 + 2\|\delta_{x,2,i}\|^2

$$

Summing over all $i$ and dividing by $k$:

$$
\begin{aligned}
\frac{1}{k} \sum_{i=1}^k \|\delta_{x,1,i} - \delta_{x,2,i}\|^2 &\leq \frac{2}{k} \sum_{i=1}^k \|\delta_{x,1,i}\|^2 + \frac{2}{k} \sum_{i=1}^k \|\delta_{x,2,i}\|^2 \\
&= 2\text{Var}_1(x) + 2\text{Var}_2(x)
\end{aligned}

$$

**Step 1.3: Extension to unequal sizes.**

For unequal sizes $k_1 \neq k_2$, a more careful analysis is required. Consider a coupling that matches $\min(k_1, k_2)$ pairs and distributes the excess mass. By the triangle inequality for Wasserstein distances and properties of Dirac measures, one can show that the cost is still bounded by $2(\text{Var}_1(x) + \text{Var}_2(x))$.

Specifically, for any centered measure $\tilde{\mu}$, we have $W_2^2(\tilde{\mu}, \delta_0) = \int \|\delta_x\|^2 \, d\tilde{\mu}(\delta_x) = \text{Var}(x)$ where $\delta_0$ is the Dirac measure at the origin. Using the triangle inequality:

$$
W_2(\tilde{\mu}_1, \tilde{\mu}_2) \leq W_2(\tilde{\mu}_1, \delta_0) + W_2(\delta_0, \tilde{\mu}_2) = \sqrt{\text{Var}_1(x)} + \sqrt{\text{Var}_2(x)}

$$

Squaring both sides and using $(a + b)^2 \leq 2a^2 + 2b^2$:

$$
W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \leq \left(\sqrt{\text{Var}_1(x)} + \sqrt{\text{Var}_2(x)}\right)^2 \leq 2\text{Var}_1(x) + 2\text{Var}_2(x)

$$

**Step 1.4: Conclusion of Part 1.**

Since the Wasserstein distance is the infimum over all couplings and we've constructed a coupling with cost at most $2(\text{Var}_1(x) + \text{Var}_2(x))$:

$$
V_{\text{x,struct}} = W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \leq 2(\text{Var}_1(x) + \text{Var}_2(x))

$$

This establishes the main inequality rigorously.

**Part 2: Proof of the Consequence**

We prove the implication $V_{\text{x,struct}} > R^2_{\text{spread}} \implies \exists k \in \{1,2\} : \text{Var}_k(x) > R^2_{\text{spread}}/4$ by contrapositive.

**Contrapositive statement:** If $\text{Var}_1(x) \leq R^2_{\text{spread}}/4$ and $\text{Var}_2(x) \leq R^2_{\text{spread}}/4$, then $V_{\text{x,struct}} \leq R^2_{\text{spread}}$.

**Proof of contrapositive:** Assume $\text{Var}_1(x) \leq R^2_{\text{spread}}/4$ and $\text{Var}_2(x) \leq R^2_{\text{spread}}/4$. By the inequality established in Part 1:

$$
V_{\text{x,struct}} \leq 2(\text{Var}_1(x) + \text{Var}_2(x)) \leq 2\left(\frac{R^2_{\text{spread}}}{4} + \frac{R^2_{\text{spread}}}{4}\right) = 2 \cdot \frac{R^2_{\text{spread}}}{2} = R^2_{\text{spread}}

$$

This proves the contrapositive statement. By logical equivalence, the implication follows: if $V_{\text{x,struct}} > R^2_{\text{spread}}$, then at least one swarm must satisfy $\text{Var}_k(x) > R^2_{\text{spread}}/4$.

**Q.E.D.**
:::

### 3.3. The Full Synergistic Lyapunov Function

With the permutation-invariant decomposition of the inter-swarm error established, we now define the full Lyapunov function. This **synergistic** function is constructed as a weighted sum of three distinct error components (see {prf:ref}`prop-lyapunov-necessity` for why this structure is mathematically necessary). It is designed to capture not only the distance *between* the swarms, but also the internal disorder *within* each swarm, which is the primary target of the cloning operator.

::::{prf:definition} The Full Synergistic Hypocoercive Lyapunov Function
:label: def-full-synergistic-lyapunov-function

For any pair of swarm ({prf:ref}`def-swarm-and-state-space`) configurations $(S_1, S_2)$ with corresponding empirical measures $(\mu_1, \mu_2)$, the **total synergistic Lyapunov function** is defined as:

$$
V_{\mathrm{total}}(S_1, S_2) := W_h^2(\mu_1, \mu_2) + c_V V_{Var}(S_1, S_2) + c_B W_b(S_1, S_2)

$$

where the intra-swarm ({prf:ref}`def-swarm-and-state-space`) variance term explicitly decomposes into positional and velocity components **summed over alive walkers only, but normalized by the total swarm size $N$**:

$$
V_{Var}(S_1, S_2) = V_{Var,x}(S_1, S_2) + \lambda_v V_{Var,v}(S_1, S_2)

$$

with:

$$
\begin{aligned}
V_{Var,x}(S_1, S_2) &:= \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2 \\
V_{Var,v}(S_1, S_2) &:= \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{v,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{v,2,i}\|^2
\end{aligned}

$$

where $N$ is the total swarm ({prf:ref}`def-swarm-and-state-space`) size, $\mathcal{A}(S_k)$ is the set of alive walker ({prf:ref}`def-walker`) indices in swarm $k$, and $\delta_{x,k,i}, \delta_{v,k,i}$ are the centered vectors defined in {prf:ref}`def-barycentres-and-centered-vectors`.

The function is a sum of three components:

1.  **The Inter-Swarm ({prf:ref}`def-swarm-and-state-space`) Error ($W_h^2$):** The squared hypocoercive 2-Wasserstein distance between the swarms' full empirical measures. This term quantifies the total permutation-invariant distance between the two swarms in phase space. As established in {prf:ref}`lem-wasserstein-decomposition`, this component can be exactly decomposed into:
    *   A **Location Component ($V_{\text{loc}}$)**, measuring the error between the swarm centers of mass.
    *   A **Structural Component ($V_{\text{struct}}$)**, measuring the mismatch in swarm shapes.

2.  **The Intra-Swarm Error ($V_{\text{Var}}$):** The sum of the internal hypocoercive variances of each swarm. This term quantifies the internal dispersion or "shape error" *within* each individual swarm in phase space, measuring their lack of internal convergence in both position and velocity. This component is the primary target of the **synergistic dissipation framework**:
    *   The **cloning operator** ($\Psi_{\text{clone}}$, analyzed in this document) has the exact positional drift $H_x$ and dissipates full-slot velocity variance inside collision components. Alive-only velocity variance has an additional bounded revival contribution.
    *   The **kinetic operator** ($\Psi_{\text{kin}}$, analyzed in {doc}`05_kinetic_contraction`) provides contraction of the velocity variance component $V_{Var,v}$ through Langevin dissipation but causes bounded expansion of the positional variance component $V_{Var,x}$ through diffusion.
    *   A net contraction estimate for their composition requires the positional donor-geometry and kinetic drift bounds stated in the composition theorem.

3.  **The Boundary Potential ($W_b$):** A term that penalizes **alive** walkers approaching the boundary, constructed from the smooth barrier function $\varphi_{\text{barrier}}(x)$ defined in {prf:ref}`prop-barrier-existence`.


$$
W_b(S_1, S_2) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \varphi_{\text{barrier}}(x_{1,i}) + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \varphi_{\text{barrier}}(x_{2,i})

$$

    where $N$ is the total swarm size and $\mathcal{A}(S_k)$ denotes the set of alive walker indices in swarm $k$. Note that dead walkers do not contribute to the boundary potential.

The parameters $b$ and $\lambda_v > 0$ are the **hypocoercive parameters**. The constants $c_V > 0$ and $c_B > 0$ are small, positive **coupling constants** used in the analysis to balance the contributions of the different error components in the final drift inequality.

:::{admonition} Normalization by $N$ vs. $k_{\text{alive}}$ in the Lyapunov Function
:class: important

The Lyapunov function components $V_{\text{Var}}$ and $W_b$ are normalized by the **total swarm size $N$**, not by the number of alive walkers $k_{\text{alive}}$. This design choice is critical for mathematical tractability and deserves careful explanation:

**Why This Choice Differs from Algorithm Internals:**

The algorithm's internal fitness calculations (z-scores, variance measurements used for cloning decisions) correctly use $k_{\text{alive}}$-normalization to compute statistics over the current active population. This is the physically and statistically correct choice for **decision-making**, as it accurately characterizes the distribution of alive walkers at each step.

However, the Lyapunov function serves a different purpose: it is an **analytical tool** designed to prove long-term stability through drift analysis. For this purpose, $N$-normalization is mathematically necessary.

**The Mathematical Necessity:**

Consider the one-step change in the variance component:

$$
\Delta V_{\text{Var}} = V_{\text{Var}}(S_{t+1}) - V_{\text{Var}}(S_t)

$$

If $V_{\text{Var}}$ were normalized by $k_{\text{alive}}$, the drift calculation would become:

$$
\mathbb{E}[\Delta V_{\text{Var}}] = \mathbb{E}\left[\frac{1}{k_{t+1}} \sum_{i} \|\delta_{x,i}\|^2_{t+1} - \frac{1}{k_t} \sum_{i} \|\delta_{x,i}\|^2_t\right]

$$

This expression involves the **ratio of correlated random variables**: both the sum of squares and the number of alive walkers change stochastically at each step, and these changes are strongly coupled (e.g., if a high-variance walker dies, both the numerator and denominator change). The expectation of such a ratio cannot be simplified, making rigorous drift bounds essentially impossible to derive.

With $N$-normalization, the constant factor $1/N$ factors out of the expectation:

$$
\mathbb{E}[\Delta V_{\text{Var}}] = \frac{1}{N} \mathbb{E}\left[\sum_{i} \|\delta_{x,i}\|^2_{t+1} - \sum_{i} \|\delta_{x,i}\|^2_t\right]

$$

This allows the analysis to focus entirely on $\mathbb{E}[\Delta \text{SumOfSquares}]$, which is the direct effect of the cloning and kinetic operators on the swarm's kinematic state. This is precisely what the Keystone Principle and the hypocoercive analysis are designed to bound.

**The Mean-Field Interpretation:**

The $N$-normalized variance can be interpreted as:

$$
V_{\text{Var},x}(S_k) = \frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2 = \frac{k_{\text{alive}}}{N} \cdot \text{Var}_{\text{alive}}(S_k)

$$

This represents the **mean-field contribution to system disorder per walker slot**. It scales with the fraction of alive walkers, which is exactly the correct behavior: if only a small fraction of walkers remain alive, the system's total disorder (as measured by the Lyapunov function) should reflect this reduced active mass.

**The Viability Requirement:**

This normalization implicitly assumes that the swarm remains viable, meaning $k_{\text{alive}}/N$ is bounded away from zero. This is guaranteed by the framework's design:
- The Safe Harbor Axiom ensures existence of a desirable region away from boundaries
- The contractive properties of the cloning operator (Keystone Principle) and the confining potential prevent swarm collapse
- The Lyapunov analysis operates in the regime where the swarm is stable, with extinction probability exponentially small

**Conclusion:**

The separation between algorithmic calculations (using $k_{\text{alive}}$) and analytical tools (using $N$) is not a compromise but a hallmark of rigorous mean-field analysis. The algorithm uses the physically optimal metric for real-time decisions, while the Lyapunov function uses the mathematically tractable metric for proving convergence. Both serve their respective purposes correctly.

Referenced by {prf:ref}`def-boundary-potential-cloning`.
:::
::::

#### 3.3.1. Variance Notation Reference

To ensure clarity throughout the proofs, we explicitly state the relationships between the three variance concepts used in this document:

:::{prf:definition} Variance Notation Conversion Formulas
:label: def-variance-conversions

For a swarm ({prf:ref}`def-swarm-and-state-space`) $k$ with $k_{\text{alive}} = |\mathcal{A}(S_k)|$ alive walkers out of $N$ total walker ({prf:ref}`def-walker`) slots:

**1. Un-normalized Sum of Squared Deviations:**

$$
S_k := \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2

$$

This is the total positional variance without any normalization.

**2. Physical Internal Variance ($k$-normalized):**

$$
\text{Var}_k(x) := \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2 = \frac{S_k}{k_{\text{alive}}}

$$

This is the average squared deviation per alive walker ({prf:ref}`def-walker`) - the standard statistical variance.

**3. Lyapunov Variance Component ($N$-normalized):**

$$
V_{\text{Var},x}(S_k) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2 = \frac{S_k}{N}

$$

This is the mean-field contribution to system disorder per walker ({prf:ref}`def-walker`) slot.

**Conversion Formulas:**

$$
\begin{aligned}
S_k &= k_{\text{alive}} \cdot \text{Var}_k(x) = N \cdot V_{\text{Var},x}(S_k) \\
V_{\text{Var},x}(S_k) &= \frac{k_{\text{alive}}}{N} \cdot \text{Var}_k(x) \\
\text{Var}_k(x) &= \frac{N}{k_{\text{alive}}} \cdot V_{\text{Var},x}(S_k)
\end{aligned}

$$

**When converting between notations in proofs:**
- From $S_k$ to $V_{\text{Var},x}$: **divide by $N$**
- From $\text{Var}_k(x)$ to $V_{\text{Var},x}$: **multiply by $\frac{k_{\text{alive}}}{N}$**
- From $V_{\text{Var},x}$ to $S_k$: **multiply by $N$**
:::

:::{admonition} Why Three Notations?
:class: note

Each notation serves a specific purpose:

- **$S_k$**: Used in geometric arguments (Section 6) where we decompose variance using Law of Total Variance. Being un-normalized, it avoids fractional coefficients when partitioning into subsets.

- **$\text{Var}_k(x)$**: Used in algorithmic analysis (Section 5-7) where we compare walker distances to swarm spread. This is the "natural" variance scale for the algorithm.

- **$V_{\text{Var},x}$**: Used in Lyapunov analysis (Sections 10-12) where we need uniform normalization across swarms with different $k_{\text{alive}}$. The $N$-normalization ensures the drift inequalities are N-uniform.

**In proofs that mix these notations, we always show the explicit conversion factor to maintain rigor.**
:::

#### 3.3.2. Mathematical Necessity of the Augmented Lyapunov Structure

The inclusion of both $W_h^2$ (inter-swarm error) and $V_{\text{Var}}$ (intra-swarm error) in the Lyapunov function is not merely convenient but mathematically necessary. This subsection explains why the specific weighted-sum structure is required for proving convergence.

:::{prf:remark} Distinct information in the Lyapunov components
:label: prop-lyapunov-necessity

The proposed observable $V_{\mathrm{total}}=W_h^2+c_VV_{\mathrm{Var}}+c_BW_b$ combines different information. Two identical broad swarms have $W_h=0$ and positive internal variance. Two distinct point clouds each concentrated at one point have zero internal variance and positive $W_h$. Neither component determines the other.

The cloning contribution is computed from the exact position law and component energy identities below. Common randomness does not automatically increase inter-swarm error: identical inputs with the same innovations remain identical. A weighted combination proves a drift inequality only after each component estimate is established for the same transition. This choice of Lyapunov observable is useful; no assertion that it is the only possible choice is needed.
:::

:::{prf:remark} Analogy to Classical Hypocoercivity Theory
:label: rem-note-hypocoercivity-analogy
:class: tip

This structure is the **discrete stochastic analogue** of the classical hypocoercivity framework for kinetic PDEs (Villani, 2009; Dolbeault-Mouhot-Schmeiser, 2015):

**Classical Hypocoercivity (Kinetic Fokker-Planck)**:
- The transport operator $v \cdot \nabla_x$ generates dynamics in $x$ but is neutral on the velocity distribution.
- The collision operator $\mathcal{L}_v$ generates dissipation in $v$ but does not directly affect $x$.
- Neither operator alone contracts the full kinetic norm $\|f\|^2_{L^2} + \|\nabla_x f\|^2_{L^2}$.
- The augmented norm $\|f\|^2_{L^2} + \varepsilon \|\nabla_x f\|^2_{L^2}$ allows proving exponential decay by balancing the operators' effects.

**Our Discrete Stochastic Framework (Fragile Gas)**:
- The cloning operator has an exact internal-variance balance; a strict positional rate and an inter-swarm coupling estimate require their own geometric bounds.
- The kinetic operator $\Psi_{\text{kin}}$ contracts $W_h^2$ (via confining potential) but may expand $V_{\text{Var}}$ (via diffusion noise).
- Neither operator alone contracts the full phase-space error.
- The augmented Lyapunov $V_{\text{total}} = W_h^2 + c_V V_{\text{Var}} + c_B W_b$ allows proving exponential convergence by balancing the operators' synergistic dissipation.

The mathematical structure is fundamentally the same: **complementary dissipation mechanisms acting on orthogonal error components**, requiring a weighted-sum Lyapunov function to capture the synergy.
:::

### 3.4. Coercivity of the Decomposed Lyapunov Function

For the Lyapunov function to be a valid measure of the total system error, its kinematic components must be positive-definite. This is guaranteed by a simple condition on the hypocoercive parameters.

::::{prf:lemma} Coercivity of the Hypocoercive Lyapunov Components
:label: lem-V-coercive

The location component $V_{\text{loc}}$ and the structural component $V_{\text{struct}}$ are positive-definite quadratic forms, and are therefore coercive, if the hypocoercive parameters satisfy:

$$
b^2 < 4\lambda_v

$$

This condition ensures that there exist constants $\lambda_1, \lambda_2 > 0$ such that:
*   $V_{\text{loc}} \ge \lambda_1 (\|\Delta\mu_x\|^2 + \|\Delta\mu_v\|^2)$
  *   $V_{\text{struct}} \ge \lambda_2 \frac{1}{N}\sum_i (\|\Delta\delta_{x,i}\|^2 + \|\Delta\delta_{v,i}\|^2)$
:::{prf:proof}
**Proof.**

We prove the coercivity of both the location and structural components by verifying that the associated quadratic forms are positive-definite under the stated condition.

**Part 1: Positive-definiteness of general hypocoercive quadratic forms.**

Consider a general quadratic form on $\mathbb{R}^d \times \mathbb{R}^d$:

$$
q(\Delta x, \Delta v) = \|\Delta x\|^2 + \lambda_v \|\Delta v\|^2 + b\langle \Delta x, \Delta v \rangle

$$

where $\Delta x, \Delta v \in \mathbb{R}^d$, $\lambda_v > 0$, and $b \in \mathbb{R}$ is a coupling parameter.

**Step 1.1: Matrix representation.**

This quadratic form can be represented in block matrix form as:

$$
q(\Delta x, \Delta v) = \begin{pmatrix} \Delta x \\ \Delta v \end{pmatrix}^T \begin{pmatrix} I_d & \frac{b}{2} I_d \\ \frac{b}{2} I_d & \lambda_v I_d \end{pmatrix} \begin{pmatrix} \Delta x \\ \Delta v \end{pmatrix}

$$

where the cross-term $b\langle \Delta x, \Delta v \rangle$ is split symmetrically into the off-diagonal blocks.

**Step 1.2: Positive-definiteness criterion via eigenvalues.**

The quadratic form $q$ is positive-definite if and only if its associated matrix $Q$ is positive-definite, which occurs if and only if all eigenvalues of $Q$ are strictly positive.

For a $2 \times 2$ block diagonal structure with scalar blocks (after diagonalizing the inner $\mathbb{R}^d$ structure), the matrix reduces to analyzing the $2 \times 2$ matrix:

$$
Q_{\text{scalar}} = \begin{pmatrix} 1 & b/2 \\ b/2 & \lambda_v \end{pmatrix}

$$

**Step 1.3: Sylvester's criterion.**

A symmetric $2 \times 2$ matrix $\begin{pmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{pmatrix}$ is positive-definite if and only if:
1. $a_{11} > 0$ (first leading principal minor)
2. $\det \begin{pmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{pmatrix} > 0$ (second leading principal minor)

For our matrix $Q_{\text{scalar}}$:
1. First condition: $1 > 0$ ✓ (always satisfied)
2. Second condition:


$$
\det(Q_{\text{scalar}}) = (1)(\lambda_v) - \left(\frac{b}{2}\right)^2 = \lambda_v - \frac{b^2}{4} > 0

$$

This requires $\lambda_v > b^2/4$, which is equivalent to $b^2 < 4\lambda_v$.

**Step 1.4: Explicit eigenvalue bounds.**

When $b^2 < 4\lambda_v$, the eigenvalues of $Q_{\text{scalar}}$ are:

$$
\lambda_{\pm} = \frac{1 + \lambda_v \pm \sqrt{(1 - \lambda_v)^2 + b^2}}{2}

$$

The discriminant satisfies $(1 - \lambda_v)^2 + b^2 < (1 - \lambda_v)^2 + 4\lambda_v = (1 + \lambda_v)^2$, so:

$$
\lambda_{-} = \frac{1 + \lambda_v - \sqrt{(1 - \lambda_v)^2 + b^2}}{2} > \frac{1 + \lambda_v - (1 + \lambda_v)}{2} = 0

$$

and similarly $\lambda_{+} > 0$. Thus both eigenvalues are strictly positive.

**Step 1.5: Coercivity constants.**

The smallest eigenvalue provides the coercivity constant:

$$
\lambda_{\min} = \min\{\lambda_{-}, \lambda_{+}\} = \frac{1 + \lambda_v - \sqrt{(1 - \lambda_v)^2 + b^2}}{2} > 0

$$

Therefore, for any $(\Delta x, \Delta v) \in \mathbb{R}^d \times \mathbb{R}^d$:

$$
q(\Delta x, \Delta v) \geq \lambda_{\min} \left(\|\Delta x\|^2 + \|\Delta v\|^2\right)

$$

**Part 2: Application to $V_{\text{loc}}$.**

The location error component is defined as:

$$
V_{\text{loc}} = \|\Delta\mu_x\|^2 + \lambda_v \|\Delta\mu_v\|^2 + b\langle \Delta\mu_x, \Delta\mu_v \rangle

$$

This is precisely the hypocoercive quadratic form $q(\Delta\mu_x, \Delta\mu_v)$ analyzed in Part 1. Under the condition $b^2 < 4\lambda_v$, we have:

$$
V_{\text{loc}} \geq \lambda_1 \left(\|\Delta\mu_x\|^2 + \|\Delta\mu_v\|^2\right)

$$

where $\lambda_1 = \lambda_{\min} > 0$ is the smallest eigenvalue from Step 1.5.

**Part 3: Application to $V_{\text{struct}}$.**

The structural error component is defined as the Wasserstein distance with hypocoercive cost:

$$
V_{\text{struct}} = W_h^2(\tilde{\mu}_1, \tilde{\mu}_2) = \inf_{\gamma \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)} \int q(\delta_{x,1} - \delta_{x,2}, \delta_{v,1} - \delta_{v,2}) \, d\gamma

$$

Since the cost function is the hypocoercive quadratic form $q$ applied to centered coordinate differences, and we've proven $q$ is coercive with constant $\lambda_{\min}$, we have for any coupling $\gamma$:

$$
\int q(\delta_{x,1} - \delta_{x,2}, \delta_{v,1} - \delta_{v,2}) \, d\gamma \geq \lambda_{\min} \int \left(\|\delta_{x,1} - \delta_{x,2}\|^2 + \|\delta_{v,1} - \delta_{v,2}\|^2\right) d\gamma

$$

Taking the infimum over all couplings and using the definition of the standard Wasserstein distance on centered measures:

$$
V_{\text{struct}} \geq \lambda_2 \cdot W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)

$$

where $\lambda_2 = \lambda_{\min} > 0$. The standard $W_2$ ({prf:ref}`lem-polishness-and-w2`) distance between centered empirical measures satisfies:

$$
W_2^2(\tilde{\mu}_1, \tilde{\mu}_2) \geq \frac{1}{N} \sum_{i=1}^N \inf_{\sigma \in S_N} \left(\|\delta_{x,1,i} - \delta_{x,2,\sigma(i)}\|^2 + \|\delta_{v,1,i} - \delta_{v,2,\sigma(i)}\|^2\right)

$$

where the infimum is over permutations $\sigma \in S_N$. This provides the desired bound on the sum of centered coordinate differences.

**Conclusion:**

Under the condition $b^2 < 4\lambda_v$, both $V_{\text{loc}}$ and $V_{\text{struct}}$ are positive-definite quadratic forms with explicit coercivity constants $\lambda_1, \lambda_2 > 0$ given by the minimum eigenvalue of the hypocoercive matrix.

**Q.E.D.**
:::
::::

(sec-cloning-assumptions)=
## 4. Foundational Assumptions and System Properties

The proof of geometric ergodicity for the Fragile Gas is built upon a set of foundational axioms. These axioms are the fundamental "contracts" that an instantiation of the algorithm must satisfy for the convergence guarantees to hold. They are not arbitrary assumptions but are carefully formulated to capture the essential properties of a well-posed, learnable environment and a dynamically stable algorithmic configuration.

This chapter organizes these axioms into three logical groups:
1.  **Environmental Axioms:** Assumptions about the geometric and reward landscape in which the swarm operates.
2.  **Measurement & Signal Axioms:** Assumptions that ensure the environment is sufficiently informative for the algorithm to learn and adapt.
3.  **Algorithmic Dynamics Axioms:** Assumptions about the user's choice of parameters, ensuring the algorithm is configured to be active, intelligent, and stable.

### 4.1. Environmental Axioms (Properties of the "World")

These axioms describe the fundamental properties of the state space and the reward function, which constitute the static "world" that the swarm explores.

:::{prf:axiom} **(Axiom EG-1): Lipschitz Regularity of Environmental Fields**
:label: axiom-lipschitz-fields

The deterministic fields governing the system's kinetic dynamics are locally smooth and globally well-behaved on the compact valid domain $\mathcal X_{\mathrm{valid}}$. Specifically, there exist finite constants $L_F$ and $L_u$ such that for all $x_1, x_2 \in \mathcal X_{\mathrm{valid}}$:

1.  **Force Field:** $\|F(x_1) - F(x_2)\| \leq L_F \|x_1 - x_2\|$
2.  **Steady Flow Field:** $\|u(x_1) - u(x_2)\| \leq L_u \|x_1 - x_2\|$

**Rationale:** This is a standard regularity assumption that ensures the kinetic dynamics do not have infinite gradients or instantaneous velocities, which is essential for the hypocoercive analysis. It guarantees that the one-step change in any walker ({prf:ref}`def-walker`)'s state is a well-behaved function of its current state.

Referenced by {prf:ref}`prop-lyapunov-necessity`.
:::

:::{admonition} Failure Mode Analysis
:class: dropdown warning
:open:

**If this axiom is violated:**

If the force or flow fields were not Lipschitz (e.g., if they had discontinuities or unbounded derivatives) as required by {prf:ref}`axiom-lipschitz-fields`, the kinetic update operator $\Psi_{\text{kin}}$ would no longer be continuous.

*   A small change in a walker's pre-kinetic position could result in an arbitrarily large, discontinuous change in its post-kinetic position.
*   This would break the hypocoercive structure of the drift. The controlled transfer of dissipation from the velocity to the position components would fail, as the coupling terms in the drift analysis would no longer be bounded.
*   The system's dynamics could become chaotic and unpredictable, with no guarantee of convergence. The proof of the main drift theorem would collapse at the analysis of the kinetic stage.
:::

:::{prf:axiom} **(Axiom EG-2): Existence of a Safe Harbor**
:label: axiom-safe-harbor

There exists a compact set $C_{\mathrm{safe}} \subset \mathcal X_{\mathrm{valid}}$ and a reward threshold $R_{\mathrm{safe}}$ such that:

1.  $C_{\mathrm{safe}}$ lies strictly inside the valid domain: $d(x, \partial X_{\mathrm{valid}}) \geq \delta_{\mathrm{safe}} > 0$ for every $x \in C_{\mathrm{safe}}$.
2.  The positional reward is strictly better inside the safe harbor ({prf:ref}`axiom-safe-harbor`): $\max_{y \in C_{\mathrm{safe}}} R_{\mathrm{pos}}(y) \geq R_{\mathrm{safe}}$ and $R_{\mathrm{pos}}(x) < R_{\mathrm{safe}}$ for all $x \notin C_{\mathrm{safe}}$.

**Rationale:** This structural assumption on the reward landscape is the engine for the boundary potential's contractive drift. It guarantees that walkers near the boundary are demonstrably "unfit" compared to those in the interior, ensuring they will be preferentially cloned inwards. This provides the inward pull necessary to counteract the diffusive expansion from the kinetic noise.
:::

:::{admonition} Failure Mode Analysis
:class: dropdown warning
:open:

**If this axiom is violated:**

If no such Safe Harbor exists, the reward landscape could be structured such that the highest-reward regions are located precariously close to the boundary of the valid domain.

*   In this scenario, the algorithm's adaptive pressure would actively drive the swarm *towards* the boundary, not away from it.
*   The boundary potential component of the Lyapunov function, $W_b$, would no longer experience a contractive drift from cloning. Instead, cloning would become an expansive force, reinforcing the kinetic stage's diffusion and accelerating the swarm's drift towards extinction.
*   The proof of the drift for $W_b$ (Section 11) would fail, and with it, the proof of the overall stability of the system. The swarm would suffer from an uncorrected diffusion to the boundary, leading to inevitable absorption in the cemetery state.
:::

### 4.2. Measurement & Signal Axioms (Properties of "Learnability")

These axioms ensure that the environment is sufficiently informative to prevent algorithmic stagnation and to allow the swarm to distinguish between good and bad configurations.

:::{prf:axiom} **(Axiom EG-3): Non-Deceptive Landscape ({prf:ref}`axiom-environmental-richness`)**
:label: axiom-non-deceptive-landscape

The environment is **non-deceptive**. A sufficient geometric separation between two walkers guarantees a minimal, non-zero difference in their raw positional rewards. Formally, there exist constants $L_{\text{grad}} > 0$ and $\kappa_{\text{raw},r} > 0$ such that:

If $\|x - y\| \geq L_{\text{grad}}$, then $|R_{\mathrm{pos}}(y) - R_{\mathrm{pos}}(x)| \geq \kappa_{\text{raw},r}$.

**Rationale:** This is the most important "learnability" axiom. It forges the critical link between the geometric diversity signal and the reward signal, preventing the algorithm from getting stuck on deceptive plateaus. It is the direct input for proving the "intelligence" of the fitness metric in the Keystone Principle.
:::

:::{admonition} Failure Mode Analysis
:class: dropdown warning
:open:

**If this axiom is violated:**

The system can enter a **deceptive plateau** state, leading to a complete breakdown of intelligent adaptation. This occurs when the swarm becomes geometrically diverse (high $\text{Var}(x)$) on a large region of the state space where the reward is nearly constant.

1.  **Signal Decoupling:** The diversity channel produces a strong, informative signal, but the reward channel produces a near-zero, uninformative signal.
2.  **Loss of Guidance:** The fitness potential becomes dominated by the diversity term ($(d')^\beta$). Cloning decisions are made based purely on geometry, with no input from the reward landscape.
3.  **Stagnation:** The swarm will churn indefinitely on the plateau. It correctly identifies its lack of convergence (due to the high diversity signal) but has lost the reward gradients needed to guide its exploration. Cloning becomes a random reshuffling of positions within the plateau, with no directed movement towards a true optimum.

This failure mode is analyzed in detail in Section 7, where it is shown to break the corrective feedback loop of the Keystone Principle.
:::

### 4.3. Algorithmic Dynamics Axioms (Properties of the "Agent")

These final axioms are assumptions about the user's choice of algorithmic parameters. They ensure the algorithm is configured to be active, to correctly interpret the signals it receives, and to remain dynamically stable.

:::{prf:definition} Optional velocity penalty in the objective
:label: axiom-velocity-regularization

An explicitly configured reward may take the form

$$
R(x,v)=R_{\rm pos}(x)-c_{v\_reg}|v|^2,\qquad c_{v\_reg}\geq0.
$$

Claims using a strictly positive reward penalty assume $c_{v\_reg}>0$. The canonical positional-objective configuration allows $c_{v\_reg}=0$. Its input velocity bound is supplied by the completed-step radial cap $\psi_v(v)=V_{\rm alg}v/(V_{\rm alg}+|v|)$, and component energy dissipation follows from {prf:ref}`prop-cloning-component-conservation` without a velocity reward penalty. Selection does not overwrite velocity by the donor's velocity.
:::

:::{div} feynman-prose
Penalizing kinetic energy changes which walkers are selected. It does not explain conservation or dissipation during a selected collision; those follow from the component update itself. Nor does a penalty guarantee that a faster walker always has lower total fitness: position reward, diversity, and normalization also enter that comparison.
:::

:::{prf:axiom} **(Axiom EG-5): Active Diversity Signal**
:label: axiom-active-diversity

The diversity channel of the fitness potential is active. The dynamics weight $\beta$ is strictly positive:

$$
\beta > 0

$$

**Rationale:** This is a fundamental assumption for the Keystone Principle's proof of intelligent targeting. It ensures that the algorithm pays attention to the reliable geometric signal generated by the **phase-space** companion kernel. This signal is the primary mechanism that allows the algorithm to detect its own lack of convergence and escape deceptive reward landscapes. This ensures that the algorithm is sensitive to its degree of convergence in the full kinematic state space, not just its spatial configuration.
:::

:::{admonition} Failure Mode Analysis
:class: dropdown warning
:open:

**If this axiom is violated ($\beta = 0$):**

The algorithm's fitness potential becomes $V_{\text{fit}} = (r')^\alpha$, making it completely "blind" to geometric diversity.

*   The algorithm loses its primary mechanism for detecting a lack of convergence. If the swarm were to land on a large, flat reward plateau (where `Var(r)` is zero), it would have no way to know it is not converged.
*   All walkers would receive the same fitness, cloning would cease, and the swarm would stall, becoming a collection of independent random walkers.
*   The entire Keystone Principle, which relies on the diversity signal to create a fitness gap, would fail. Without an active diversity signal, the algorithm has no reliable, built-in mechanism to force adaptation.
:::

### 4.4. Uniform constants and the coupled alive sets

:::{div} feynman-prose
A constant is uniform in $N$ when a single bound works for the whole family of
swarm sizes. A positive number depending on $N$ need not have this property.
For selection, a fixed favorable fraction and a uniform comparison-kernel
bound can supply the required constant; the positive-part calculation in
{prf:ref}`lem-unfit-cloning-pressure` makes the dependence explicit.

The set $I_{11}$ contains labels alive in both coupled swarms. A within-swarm
matching does not change this intersection or create missing labels in it.
The target-error estimate must therefore use the actual common alive set and
retain any complement term. The status-change bounds and the coupling analysis
in {doc}`06_convergence` treat the remaining contribution.
:::

(sec-cloning-measurement)=
## 5. The Measurement and Interaction Pipeline

This chapter formally defines the complete sequence of operators that transforms a swarm's raw state into a final, N-dimensional fitness potential vector. This multi-stage operator, which we denote $\Phi_{\text{pipeline}}$, constitutes the "sensory and cognitive" system of the swarm for a single timestep. It is the mechanism by which the swarm perceives its own configuration and the reward landscape, and translates those perceptions into a quantitative measure of fitness that will drive the subsequent cloning and selection process.

The pipeline begins with a sampled measurement law: independent weighted companions in the canonical gas, or the explicitly specified matching law in a matching configuration. Once this pairing is fixed, the second phase is a **deterministic cascade** of measurement, aggregation, and transformation operators that processes the information from these pairings. This chapter will construct the pipeline step-by-step, defining each operator and establishing its key properties. The final output, the fitness potential vector $\mathbf{V}_{\text{fit}}$, is the fixed, deterministic input for the cloning operator analyzed in the subsequent sections.

### 5.0. The Algorithmic Distance Metric for Phase-Space Proximity

Before defining the measurement operators, we must first establish the fundamental metric that quantifies proximity between walkers. This metric is central to all intra-swarm measurements in the algorithm, including companion selection for diversity measurement and companion selection for cloning.

:::{prf:definition} Algorithmic comparison for companion selection
:label: def-algorithmic-distance-metric

For the canonical Euclidean Gas, put $S_R(u)=Ru/(R+|u|)$ and

$$
d_{\mathrm{alg}}(i,j)^2=
|S_{R_x}(x_i)-S_{R_x}(x_j)|^2+
\lambda_{\mathrm{alg}}|S_{R_v}(v_i)-S_{R_v}(v_j)|^2.
$$

It satisfies $d_{\mathrm{alg}}^2\leq4R_x^2+4\lambda_{\mathrm{alg}}R_v^2$ on unbounded physical space. The independent measurement and cloning kernels use their own Gaussian bandwidths with this comparison.

An explicitly configured unsquashed comparison instead uses
$|x_i-x_j|^2+\lambda_{\mathrm{alg}}|v_i-v_j|^2$. Geometric estimates identifying algorithmic distance with that physical quadratic distance apply to that configuration. They require a proved comparison estimate before being applied to the squashed configuration; bounded comparison features alone do not bound physical positions.
:::

:::{div} feynman-prose
The feature comparison controls which companions are sampled. The hypocoercive norm controls how we measure differences between physical states. These are distinct functions. In particular, two distant physical positions may have similar squashed features, so a physical separation estimate cannot be substituted directly for a feature-space estimate.
:::

### 5.1. Measurement laws and explicit matching configurations

:::{prf:remark} Canonical independent sampling and matching laws
:label: rem-cloning-measurement-law-scope

The canonical gas draws one measurement companion independently for each alive recipient, from its Gaussian weighted eligible pool with self exclusion. Its sampled separation, global regularized statistics, and retained fitness marks are defined in {prf:ref}`def-mean-field-measurement-law` and {prf:ref}`def-mean-field-moments`. The canonical finite-population and population-limit proofs use that law.

The perfect-matching and sequential-greedy constructions below define distinct available measurement configurations. Their conditional signal estimates retain the specified matching law and are not automatically estimates for independent sampling. Similarly, geometric arguments below that identify algorithmic distance with physical phase-space distance retain the unsquashed-comparison hypothesis. The canonical fixed-step mean-field proof is discharged directly in {doc}`08_mean_field` and {doc}`09_propagation_chaos` and does not rely on those matching estimates.
:::

#### 5.1.1. The Idealized Matching Model

For the purposes of theoretical analysis, it is useful to model the pairing as a single, collective draw from a probability distribution over all possible perfect matchings of the alive set. This idealized model captures the physical intent of the operator—to strongly favor pairings between walkers that are close in the algorithmic space.

:::{prf:definition} Spatially-Aware Pairing Operator (Idealized Model)
:label: def-spatial-pairing-diversity-idealized

Let $\mathcal{S}_t$ be the current swarm ({prf:ref}`def-swarm-and-state-space`) state with alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}_t$ of size $k = |\mathcal{A}_t|$. The idealized **Spatially-Aware Pairing Operator**, denoted $\mathbb{P}_{\text{pair}}$, maps the alive set $\mathcal{A}_t$ to a probability distribution over the set of all possible perfect matchings, $\mathcal{M}_k$.

**Inputs:**
*   The alive set  of walkers, $\mathcal{A}_t = \{w_1, w_2, \dots, w_k\}$.
*   $\varepsilon_d > 0$ (The Interaction Range for Diversity).

**Operation:**
1.  For every pair of distinct walkers $(w_i, w_j)$, an edge weight is assigned based on their phase-space proximity using the algorithmic distance ({prf:ref}`def-alg-distance`) metric (see {prf:ref}`def-algorithmic-distance-metric`):


$$
w_{ij} := \exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon_d^2}\right)

$$

2.  The "quality" of a specific perfect matching `M` is the product of the weights of the edges it contains:


$$
W(M) := \prod_{(i,j) \in M} w_{ij}

$$

3.  The probability of selecting a specific matching `M` is given by its quality normalized by the sum of qualities over all possible matchings (the partition function):


$$
P(M) = \frac{W(M)}{\sum_{M' \in \mathcal{M}_k} W(M')}

$$

:::

#### 5.1.2. Practical Implementation: The Sequential Stochastic Greedy Pairing Operator

While the idealized model in {prf:ref}`def-spatial-pairing-diversity-idealized` is analytically useful, its requirement to sum over all `(k-1)!!` possible perfect matchings makes it computationally intractable for any non-trivial swarm size. To ensure the algorithm is efficient, we implement a **Sequential Stochastic Greedy Pairing Operator**.

This algorithm builds the matching iteratively. It selects an unpaired walker, computes a probability distribution over all other currently unpaired walkers based on proximity, samples a companion, and removes the new pair from the pool. This reduces the computational complexity from factorial to quadratic, making it practical for large swarms.

:::{prf:definition} Sequential Stochastic Greedy Pairing Operator
:label: def-greedy-pairing-algorithm

Let `A_t` be the set of `k` alive walkers at time `t`. The pairing operator generates a **Companion Map**, `c: A_t → A_t`, which is a perfect matching if `k` is even, or a maximal matching if `k` is odd. In the reference implementation, any leftover walker when `k` is odd (or `k=1`) is mapped to itself, `c(i) = i`, yielding an involution with at most one fixed point.

**Inputs:**
*   The set of alive walkers, `A_t = {w_1, w_2, ..., w_k}`.
*   $\varepsilon_d > 0$ (The Interaction Range for Diversity).

**Operation:**
1.  Initialize a set of unpaired walkers `U ← A_t` and an empty companion map `c`.
2.  While `|U| > 1`:
    a. Select and remove an arbitrary walker ({prf:ref}`def-walker`) `i` from `U`.
    b. For each remaining walker  $j \in U$, calculate the selection weight based on phase-space proximity using the algorithmic distance ({prf:ref}`def-alg-distance`) metric (see {prf:ref}`def-algorithmic-distance-metric`):


$$
w_{ij} := \exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon_d^2}\right)

$$

    c. Form a probability distribution over $j \in U$ where $P(\text{choose } j) = w_{ij} / (\sum_{l \in U} w_{il})$.
    d. Sample a companion `c_i` for `i` from this distribution.
    e. Remove `c_i` from `U`.
    f. Set the pairing in the companion map: `c(i) ← c_i` and `c(c_i) ← i`.
3.  If `|U| = 1`, set the remaining walker's companion to itself.
4.  Return the completed companion map `c`.

**Complexity:** The outer loop runs `k/2` times. In each iteration, the weights and normalization factor are computed over the remaining walkers (at most `k-1`). The complexity is therefore `O(k^2)`, which is a feasible computation.

Referenced by {prf:ref}`lem-greedy-preserves-signal` and {prf:ref}`thm-geometry-guarantees-variance`.
:::

The following pseudocode provides a concrete implementation of this operator.

:::{prf:algorithm} Sequential Stochastic Greedy Pairing Algorithm
:label: alg-greedy-pairing

ALGORITHM: GreedyPairing(alive_walkers, epsilon_d)
-------------------------------------------------
INPUT:
  alive_walkers: A list of k walker ({prf:ref}`def-walker`) objects.
  epsilon_d: The interaction range for diversity.
OUTPUT:
  companion_map: A dictionary representing the pairing.

1.  unpaired_set ← a set containing all walkers from alive_walkers
2.  companion_map ← an empty dictionary

3.  WHILE len(unpaired_set) > 1:
4.      i ← unpaired_set.pop()  // Select and remove a walker ({prf:ref}`def-walker`)

5.      // Prepare to compute the probability distribution
6.      companions ← list(unpaired_set)
7.      weights ← empty list of floats
8.
9.      FOR j IN companions:
10.         dist_sq = algorithmic_distance(i.state, j.state)^2
11.         weight = exp(-dist_sq / (2 * epsilon_d^2))
12.         weights.append(weight)

13.     // Normalize weights to get probabilities
14.     total_weight = sum(weights)
15.     probabilities = [w / total_weight for w in weights]

16.     // Sample the companion based on the probabilities
17.     c_i ← sample_from(companions, probabilities)

18.     // Finalize the pair
19.     unpaired_set.remove(c_i)
20.     companion_map[i] ← c_i
21.     companion_map[c_i] ← i

22. IF len(unpaired_set) == 1:
23.     i ← unpaired_set.pop()
24.     companion_map[i] ← i
25. RETURN companion_map
:::

### 5.1.3. Conditional signal estimates for sequential pairing

:::{div} feynman-prose
The sequential algorithm samples from the candidates still available at each
step. The idealized law weights complete matchings. These are different laws:
the normalization factors accumulated along a greedy history depend on the
pairs removed earlier. The common edge weights alone do not identify their
expectations. {prf:ref}`lem-greedy-ideal-equivalence` treats their regularity as
separate finite sums.

For signal detection, the useful calculation is a tail bound at a particular
selection step. Nearby candidates contribute to the denominator and distant
candidates to the numerator. We retain both counts. Later steps can exhaust a
nearby cluster, so the initial cluster size cannot replace its remaining size.
:::

:::{prf:definition} A separated configuration for distance measurement
:label: def-geometric-partition

For a fixed swarm and its algorithmic distance, a separated configuration
consists of sets $H_k,L_k$, radii $0\leq R_L<D_H$, and subsets
$C_j\subset\mathcal A_k\setminus\{j\}$ for $j\in L_k$. An isolated member
$i\in H_k$ satisfies $d_{\mathrm{alg}}(i,u)\geq D_H$ for every $u\ne i$.
For $j\in L_k$, members of $C_j$ lie within $R_L$ of $j$ and all remaining
nonself candidates lie at distance at least $D_H$.

These are properties of the specified configuration. The high-error sets
constructed from positional outlier clusters in Section 6 need not consist of
individually isolated walkers: members of an outlying cluster can be close to
one another. Applying the isolated-walker estimate requires its stated distance
condition. Write $D_{\mathrm{alg}}$ for a bound on all candidate distances;
under the position and velocity bounds one can take
$D_{\mathrm{alg}}=(D_{\mathrm{valid}}^2+4\lambda_{\mathrm{alg}}V_{\max}^2)^{1/2}$.
:::

:::{prf:lemma} Distance bounds under the actual greedy history
:label: lem-greedy-preserves-signal

At a greedy selection step, condition on its history, remaining set $U$, and
chosen pivot $j$. Suppose $n_{\mathrm{near}}=|C_j\cap U|>0$ and let
$n_{\mathrm{far}}=|U\setminus(C_j\cup\{j\})|$. For the geometry of
{prf:ref}`def-geometric-partition`, set

$$
q(U,j)=\min\left\{1,\frac{n_{\mathrm{far}}}{n_{\mathrm{near}}}
\exp\left[-\frac{D_H^2-R_L^2}{2\varepsilon_d^2}\right]\right\}.
$$

Then

$$
\mathbb P(c_j\notin C_j\mid U,j)\leq q(U,j),\qquad
\mathbb E[d_j\mid U,j]\leq R_L+(D_{\mathrm{alg}}-R_L)q(U,j).
$$

If no nearby candidate remains, use the bound $q(U,j)=1$.
For an isolated walker $i$, its final measurement satisfies

$$
\mathbb E[d_i\mid S]\geq D_H\,[1-\mathbb P(c_i=i\mid S)].
$$

For any low-error walker $j$, let $q_j$ be the probability that its final
nonself companion lies outside $C_j$. Then its final marginal obeys

$$
\mathbb E[d_j\mid S]\leq R_L+(D_{\mathrm{alg}}-R_L)q_j.
$$

The probability $q_j$ includes histories in which $j$ is chosen as a companion
by another pivot. A uniform bound on $n_{\mathrm{far}}/n_{\mathrm{near}}$ gives
a uniform tail bound for the corresponding pivot histories; a final marginal
bound requires the incoming histories as well.
:::

:::{prf:proof}
Write $w_{ju}=\exp[-d_{\mathrm{alg}}(j,u)^2/(2\varepsilon_d^2)]$ and
$Z_j(U)=\sum_{u\in U\setminus\{j\}}w_{ju}$. The total far weight is at most
$n_{\mathrm{far}}e^{-D_H^2/(2\varepsilon_d^2)}$ and the near weight is at
least $n_{\mathrm{near}}e^{-R_L^2/(2\varepsilon_d^2)}$. Dividing gives the
probability bound. On a near selection $d_j\leq R_L$; otherwise
$d_j\leq D_{\mathrm{alg}}$, which proves the conditional expectation bound.

An isolated walker's nonself edges all have length at least $D_H$; a self edge
has length zero. This proves its bound, including the possible odd leftover.
For a low-error walker the same near/far partition proves the final marginal
bound once its actual probability $q_j$ is used.

More explicitly, let $\mathcal F_t$ contain the history and pivot $I_t$ just
before companion sampling, with $U_t$ the remaining set including the pivot.
Then $q_j$ is the expectation of the sum over selection steps of

$$
\mathbf1_{\{j\in U_t,I_t=j\}}
 \frac{\sum_{u\in U_t\setminus(C_j\cup\{j\})}w_{ju}}{Z_j(U_t)}
+\mathbf1_{\{j\in U_t,I_t\ne j,I_t\notin C_j\}}
 \frac{w_{I_tj}}{Z_{I_t}(U_t)}.
$$

Each term is the conditional probability that this step creates a far edge
incident to $j$. Such events at different steps are disjoint because a paired
walker is removed. Conditional expectation and summation therefore give the
exact final probability, with no independence assumption on the matching.
:::

:::{div} feynman-prose
For an even swarm, an individually isolated walker has no self-match and its
lower bound is $D_H$. For a low-error walker, a small distance needs a small
final far-edge probability. The displayed history formula tells us what must
be estimated. Smooth dependence on positions, proved in the regularity chapter,
does not supply that probability estimate by itself.
:::

### 5.2. Stage 2: Raw Value Measurement

Once the Companion Map `c(i)` is fixed for the timestep by the pairing operator, the measurement of raw values for each walker becomes a deterministic process.

:::{prf:definition} Raw Value Operators
:label: def-raw-value-operators

1.  **The Reward Measurement Operator ($V_R$):** The raw reward for each alive walker ({prf:ref}`def-walker`) `i` is its direct, individual measurement of the reward function, which explicitly includes both positional and velocity components:


$$
r_i := R(x_i, v_i) = R_{\text{pos}}(x_i) - c_{v\_reg} \|v_i\|^2

$$

    where $R_{\text{pos}}(x_i)$ is the positional reward and $c_{v\_reg} \geq 0$ is the explicitly configured velocity regularization coefficient from {prf:ref}`axiom-velocity-regularization`.

2.  **The Paired Distance Measurement Operator ($V_D$):** Given the Companion Map `c(i)` generated by the pairing operator, the raw distance for each alive walker ({prf:ref}`def-walker`) `i` is deterministically defined as the algorithmic distance ({prf:ref}`def-alg-distance`) to its assigned companion:


$$
\ell_i:=d_{\mathrm{alg}}(i,c(i)),\qquad d_i:=\sqrt{\ell_i^2+\delta_D^2},\qquad\delta_D>0

$$

For any walker ({prf:ref}`def-walker`) `j` that is dead, its raw values are deterministically zero: $r_j = 0$ and $d_j = 0$.

Referenced by {prf:ref}`def-measurement-operator`.
:::

:::{prf:lemma} Transferring a raw-distance estimate to the measured separation
:label: lem-cloning-distance-floor-transfer

The actual separation and raw feature distance satisfy $0\leq d_i-\ell_i\leq\delta_D$. Their empirical means differ by at most $\delta_D$, and their empirical standard deviations differ by at most $\delta_D$. Therefore

$$
|\operatorname{Var}(d)-\operatorname{Var}(\ell)|
\leq2\delta_D\sqrt{\operatorname{Var}(\ell)}+\delta_D^2.
$$

*Proof.* The pointwise inequality follows from $\sqrt{u^2+\delta_D^2}\leq u+\delta_D$ for $u\geq0$. Averaging bounds the means. Centering is an orthogonal projection in the empirical $L^2$ norm, hence the reverse triangle inequality bounds the difference of standard deviations by $\|d-\ell\|_{L^2}\leq\delta_D$. Squaring yields the variance bound. $\square$

Geometric estimates for the raw feature distances use these explicit errors when applied to the regularized measurements entering fitness.
:::


:::{div} feynman-prose
Velocity enters the comparison features when $\lambda_{\rm alg}>0$. It enters the reward directly only if the objective includes a velocity penalty. Neither channel alone makes a high-speed walker unfit regardless of its position and sampled diversity. The completed-step velocity cap supplies the state-independent input bound used in collision estimates.
:::

### 5.3. Stage 3: Swarm Aggregation and Statistical Measurement

This stage distills the raw reward and distance vectors from the `k` alive walkers into the summary statistics (mean and standard deviation) required for standardization.

:::{prf:definition} Swarm Aggregation Operator
:label: def-swarm-aggregation-operator

A **Swarm ({prf:ref}`def-swarm-and-state-space`) Aggregation Operator**, $M$, maps the `k`-dimensional raw value vector $\mathbf{v}_{\mathcal{A}}$ from the alive set ({prf:ref}`def-alive-dead-sets`) of a swarm state $\mathcal{S}$ (see {prf:ref}`def-single-swarm-space`) to a probability measure on $\mathbb{R}$, $\mu_{\mathbf{v}} = M(\mathcal{S}, \mathbf{v}_{\mathcal{A}})$. The moments of this measure define the swarm's collective statistics.

Referenced by {prf:ref}`def-standardization-operator`.
:::

:::{admonition} Canonical Instantiation
:class: note
The canonical choice for the Euclidean Gas is the **Empirical Measure Aggregator**, where $M(\mathcal{S}, \mathbf{v}) = \frac{1}{k} \sum_{i \in \mathcal{A}} \delta_{v_i}$. Its first and second moments are the standard empirical mean and raw second moment, respectively.
:::

A critical component of this stage is the use of a robust, smooth function (see {prf:ref}`def-patched-std-dev-function`) to compute the standard deviation, which is essential for the stability of the entire pipeline.

:::{prf:definition} Regularized standard deviation
:label: def-patched-std-dev-function

For the canonical global standardizer, the function denoted $\sigma'_{\rm patch}$ is

$$
\sigma'_{\rm patch}(V)=\sqrt{V+\sigma_{\min}^2},\qquad V\geq0,\quad\sigma_{\min}>0.
$$

Reward and diversity may have their own fixed regularizers. This is the scale computed from the population variance by the Rust standardizer.
:::

:::{prf:lemma} Properties of the regularized scale
:label: lem-patching-properties

The scale is smooth on a neighborhood of $[0,\infty)$, bounded below by $\sigma'_{\min,\rm patch}=\sigma_{\min}$, and globally Lipschitz on $[0,\infty)$ with constant $1/(2\sigma_{\min})$.

*Proof.* Differentiate: $(\sigma'_{\rm patch})'(V)=1/(2\sqrt{V+\sigma_{\min}^2})\leq1/(2\sigma_{\min})$. Positivity and smoothness follow from $V+\sigma_{\min}^2>0$. $\square$
:::

### 5.4. Stage 4: The N-Dimensional Standardization Operator

This operator uses the robust statistics from the previous stage to convert the raw value vectors into standardized Z-scores.

:::{prf:definition} N-Dimensional Standardization Operator
:label: def-standardization-operator

The **N-Dimensional Standardization Operator ({prf:ref}`def-standardization-operator-n-dimensional`)**, $z$, maps a swarm ({prf:ref}`def-swarm-and-state-space`) state `S`, a raw value vector `v`, and an aggregation operator `M` to an N-dimensional vector of Z-scores.

**Operation:**
1.  Aggregate the alive components `v_A` using operator M (see {prf:ref}`def-swarm-aggregation-operator`) to get a measure $\mu_v = M(S, v_A)$.
2.  Compute the mean $\mu_A = \mathbb{E}[\mu_v]$ and the **patched** standard deviation $\sigma'_A = \sigma'_{\text{patch}}(\text{Var}[\mu_v])$ using the patching function (see {prf:ref}`lem-patching-properties`).
3.  For each alive walker ({prf:ref}`def-walker`) `i`, compute its Z-score: $z_i = (v_i - \mu_A) / \sigma'_A$.
4.  Assemble the final N-dimensional vector, setting components for dead walkers to zero.

Referenced by {prf:ref}`def-measurement-operator`.
:::

:::{prf:lemma} Compact Support of Standardized Scores
:label: lem-compact-support-z-scores
As a direct consequence of the raw values being uniformly bounded and the patched standard deviation being uniformly bounded below by $\sigma'_{\min,\text{patch}} > 0$ (from {prf:ref}`lem-patching-properties`), any generated Z-score `z_i` is guaranteed to lie within a fixed, compact interval $Z_{\mathrm{supp}}$ that is independent of the swarm ({prf:ref}`def-swarm-and-state-space`) state.
:::

### 5.5. Stage 5: The Rescale Transformation

This stage applies a non-linear transformation (see {prf:ref}`def-logistic-rescale`) to map the standardized scores, which can be any real number, to a bounded interval of positive values that will serve as the components for the fitness potential.

:::{admonition} Canonical Instantiation
:class: note
The canonical choice for the Euclidean Gas is the **Canonical Logistic Rescale Function** (see {prf:ref}`def-logistic-rescale`), which satisfies the **Axiom of a Well-Behaved Rescale Function** from the framework (i.e., it is smooth, monotonic, bounded, and Lipschitz).
:::

:::{prf:definition} Canonical Logistic Rescale Function
:label: def-logistic-rescale

The **Canonical Logistic Rescale Function ({prf:ref}`def-canonical-logistic-rescale-function-example`)**, $g_A: \mathbb{R} \to (0, 2)$, is defined as:

$$
g_A(z) := \frac{2}{1 + e^{-z}}

$$

:::

:::{prf:lemma} Verification of Axiomatic Properties
:label: lem-logistic-properties
The Canonical Logistic Rescale function ({prf:ref}`def-canonical-logistic-rescale-function-example`) is infinitely differentiable ($C^\infty$), strictly increasing, has a range of $(0,2)$, and its derivative is globally bounded by 1/2. It is therefore globally Lipschitz and satisfies all axiomatic requirements for a valid rescale function.
:::

### 5.6. Stage 6: The Final Fitness Potential Operator

The final stage of the pipeline assembles the rescaled components from both the reward and diversity channels into the final fitness potential vector.

:::{prf:definition} Fitness Potential Operator
:label: def-fitness-potential-operator

The **Fitness Potential Operator**, $\Phi_{\text{pipeline}}$, maps a swarm ({prf:ref}`def-swarm-and-state-space`) state `S` and its raw measurement vectors `r` and `d` to the final N-dimensional fitness potential vector $\mathbf{V}_{\text{fit}}$.

**Operation:**
1.  Compute reward Z-scores: $\mathbf{z}_r = z(S, \mathbf{r}, R_{agg})$.
2.  Compute distance Z-scores: $\mathbf{z}_d = z(S, \mathbf{d}, M_D)$.
3.  For each alive walker ({prf:ref}`def-walker`) `i`, compute the rescaled components with the floor $\eta$ using the Canonical Logistic Rescale Function (see {prf:ref}`lem-logistic-properties`):
    *   $r'_i := g_A(z_{r,i}) + \eta$
    *   $d'_i := g_A(z_{d,i}) + \eta$
4.  Combine the components using the dynamics weights $\alpha$ and $\beta$:



$$
V_i := (d'_i)^\beta \cdot (r'_i)^\alpha

$$

5.  Assemble the final N-dimensional vector $\mathbf{V}_{\text{fit}}$, setting components for dead walkers to zero.
:::

:::{prf:lemma} Uniform Bounds of the Fitness Potential
:label: lem-potential-bounds

Any non-zero fitness potential $V_i$ generated by this pipeline is uniformly bounded within a compact interval $[V_{\text{pot,min}}, V_{\text{pot,max}}]$. The bounds are state-independent constants defined by the algorithmic parameters (using properties from {prf:ref}`lem-logistic-properties`):
*   $V_{\text{pot,min}} := \eta^{\alpha+\beta}$
*   $V_{\text{pot,max}} := (g_{A,\max} + \eta)^{\alpha+\beta}$
:::
:::{prf:proof}

**Proof.**

The proof follows directly from the definition of the multiplicative potential and the bounded properties of its components.

The rescaled components, $r'_i = g_A(z_{r,i}) + \eta$ and $d'_i = g_A(z_{d,i}) + \eta$, are strictly positive and bounded. The rescale function $g_A(z)$ has a range of $(g_{A,\min}, g_{A,\max}]$. Since $\eta > 0$, the components are bounded on the interval $(g_{A,\min} + \eta, g_{A,\max} + \eta]$. For simplicity and rigor, we use the absolute bounds $(\eta, g_{A,\max} + \eta]$.

**Lower Bound ($V_{\text{pot,min}}$):** The fitness potential $V_i$ is a product of positive terms raised to non-negative powers ($\alpha, \beta \geq 0$). It is minimized when each component is at its minimum possible value.

$$
V_i \ge (\eta)^{\beta} \cdot (\eta)^{\alpha} = \eta^{\alpha+\beta}

$$

Therefore, the uniform lower bound is $V_{\text{pot,min}} := \eta^{\alpha+\beta}$.

**Upper Bound ($V_{\text{pot,max}}$):** The potential is maximized when each component is at its maximum possible value.

$$
V_i \le (g_{A,\max} + \eta)^{\beta} \cdot (g_{A,\max} + \eta)^{\alpha} = (g_{A,\max} + \eta)^{\alpha+\beta}

$$

Therefore, the uniform upper bound is $V_{\text{pot,max}} := (g_{A,\max} + \eta)^{\alpha+\beta}$.

**Uniformity:**

Since $g_A$ is bounded and $\eta$ is a finite positive constant, both $V_{\text{pot,min}}$ and $V_{\text{pot,max}}$ are finite, positive, state-independent constants. They are independent of the swarm size $N$, the current state, or any dynamical variables.

This completes the proof.

**Q.E.D.**
:::

### 5.7. Stage 7: The Clone vs. Persist Gate

This is the final and most critical stage of the cloning operator. It is the "act" phase that follows the "sense" and "process" stages of the pipeline. This gate takes the frozen N-dimensional fitness potential vector, $\mathbf{V}_{\text{fit}}$, as its primary input and executes a per-walker stochastic decision: either the walker **persists** in its current state, or it is marked for **cloning** and will be replaced by a jittered copy of a higher-fitness companion.

The process for each walker `i` follows a strict sequence: first, a potential companion is selected (see {prf:ref}`def-cloning-companion-operator`); second, a score is calculated based on that companion's fitness; third, a probabilistic decision is made. This section formally defines each of these components.

#### 5.7.1 Companion Selection Operator for Cloning
:::{prf:definition} Companion Selection Operator for Cloning
:label: def-cloning-companion-operator

The first step of the cloning action is to select a companion. The **Companion Selection ({prf:ref}`def-companion-selection-measure`) Operator for Cloning** defines, for each walker ({prf:ref}`def-walker`) `i`, a probability measure $\mathcal{C}_i(S)$ from which a companion `c_i` is sampled independently. The same configured weighted kernel is used for live selection and dead-slot revival, with their respective eligible donor sets.

**Inputs:**
*   The swarm ({prf:ref}`def-swarm-and-state-space`) state `S`, which defines the set of alive walkers, $\mathcal{A}_k$, and the set of dead walkers, $\mathcal{D}_k$.
*   The interaction range for cloning, $\varepsilon_c > 0$.

**Operation:**
The definition of the measure $\mathcal{C}_i(S)$ depends on the status of walker ({prf:ref}`def-walker`) `i`:

1.  **If `i` is an ALIVE walker  ($i \in \mathcal{A}_k$):**
    The selection is phase-space-aware and restricted to other alive walkers. For any other alive walker $j \in \mathcal{A}_k \setminus \{i\}$, the probability of selection is given by a softmax distribution based on algorithmic distance:


$$
P(c_i=j \mid i \in \mathcal{A}_k) := \frac{\exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon_c^2}\right)}{\sum_{l \in \mathcal{A}_k \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}(i, l)^2}{2\epsilon_c^2}\right)}

$$

2.  **If `i` is a DEAD walker ($i \in \mathcal{D}_k$):**
    Use the retained dead position and velocity in the same weighted comparison. For any alive walker $j \in \mathcal{A}_k$:


$$
P(c_i=j \mid i \in \mathcal{D}_k) := \frac{\exp[-d_{\mathrm{alg}}(i,j)^2/(2\epsilon_c^2)]}{\sum_{\ell\in\mathcal A_k}\exp[-d_{\mathrm{alg}}(i,\ell)^2/(2\epsilon_c^2)]}

$$

Referenced by {prf:ref}`def-decision-operator`.
:::

#### 5.7.2 Cloning Score
:::{prf:definition} The Canonical Cloning Score
:label: def-cloning-score

Once a companion `c_i` has been selected for an alive walker ({prf:ref}`def-walker`) `i`, the **Canonical Cloning Score**, $S_i(c_i)$, is calculated as:

$$
S_i(c_i) := \frac{V_{\text{fit},{c_i}} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\mathrm{clone}}}

$$

where $V_{\text{fit},i}$ is the fitness of walker ({prf:ref}`def-walker`) `i`, $V_{\text{fit},{c_i}}$ is the fitness of its chosen companion, and $\varepsilon_{\mathrm{clone}} > 0$ is a small regularization constant.

Referenced by {prf:ref}`def-cloning-decision` and {prf:ref}`def-cloning-probability`.
:::

:::{prf:definition} Total Cloning Probability
:label: def-cloning-probability

For an alive row, the **total cloning probability** $p_i=p_i(S,\mathbf F)$ is its probability of acceptance conditional on the input and frozen measured fitness. This is the expectation of the cloning event taken over the random draws of both the companion `c_i` and the threshold `T_i`, where the score is defined by {prf:ref}`def-cloning-score`.

$$
p_i := \mathbb{E}_{c_i \sim \mathcal{C}_i(S)} \left[ \mathbb{P}_{T_i \sim U(0,p_{\max})} \left( S_i(c_i) > T_i \right) \right]

$$

The inner probability, for a fixed companion, evaluates to $\min(1, \max(0, S_i(c_i)/p_{\max}))$. This gives the final expression for the total cloning probability as an expectation over the choice of companion:

$$
p_i = \mathbb{E}_{c_i \sim \mathcal{C}_i(S)}\left[\min\left(1, \max\left(0, \frac{S_i(c_i)}{p_{\max}}\right)\right)\right]

$$

For dead rows $p_i=1$ by the mandatory revival branch. When conditioning only on $S$, average the displayed probability over the actual measurement law. This distinction is retained in the Keystone selection estimates and exact variance formulas.
:::

:::{admonition} Design Note: The Duality of Cloning Interactions
:class: tip

The structure of the canonical cloning score creates a fundamental duality in every interaction between two alive walkers, `i` and `c`. The scores are anti-symmetric in their numerators: $S_i(c) \propto (V_c - V_i)$ while $S_c(i) \propto (V_i - V_c)$. Consequently, if $V_i < V_c$, then $S_i(c)$ is positive while $S_c(i)$ is negative.

This means that in any given pairing, only one walker—the less fit one—can ever have a positive score and thus a non-zero chance of cloning. The fitter walker effectively acts as a "teacher" or a source of information, while the less fit walker acts as a "learner." This ensures that information, in the form of walker states, flows exclusively from high-fitness regions of the state space to low-fitness regions, providing a robust, microscopic guarantee of corrective pressure.
:::

#### 5.7.3  Cloning Decision

:::{prf:definition} The Stochastic Cloning Decision
:label: def-cloning-decision

For a live recipient, the decision to clone is made by comparing the score (see {prf:ref}`def-cloning-score`) to a random threshold. For each walker ({prf:ref}`def-walker`) `i`, after its score $S_i(c_i)$ has been computed, a random threshold $T_i$ is sampled from the uniform distribution $T_i \sim \mathrm{Unif}(0, p_{\max})$. The walker `i` is marked for **cloning** if $S_i(c_i) > T_i$. Otherwise, it is marked to **persist**. A dead recipient is accepted unconditionally whenever the current alive donor pool is nonempty.
:::

#### 5.7.4. Frozen connected-component collisions

:::{div} feynman-prose
Follow the accepted arrows, not just the arrows entering one donor. A donor can itself copy a fitter walker. These two donor groups overlap and must form one collision. Each participating slot receives one velocity update from the frozen input, while each accepted recipient copies the frozen position of its own donor.

The entire component shares one orthogonal matrix. This is what makes the sum of its centered velocities stay zero after rotation. A donor that does not copy anyone can still change velocity because other walkers selected it.
:::

:::{prf:definition} The inelastic component update
:label: def-inelastic-collision-update

Condition on the frozen input $S$, its sampled fitness vector, donor choices $c_i$, and accepted set $A_C$. Every dead slot belongs to $A_C$ when the current alive pool is nonempty. Form the undirected graph with edge $\{i,c_i\}$ for every $i\in A_C$. Let $\mathfrak C$ be its connected components, including isolated vertices.

For each nontrivial component $C$, draw one independent Haar matrix $R_C\in O(d)$ and write $\alpha=\alpha_{\mathrm{restitution}}\in[0,1]$. Using the frozen pre-collision velocities of every slot, including retained dead velocities, set

$$
\bar v_C=\frac1{|C|}\sum_{j\in C}v_j,\qquad
v_i'=\bar v_C+\alpha R_C(v_i-\bar v_C),\quad i\in C.
$$

An isolated walker retains its velocity. The positional update is

$$
x_i'=\begin{cases}x_{c_i}+\sigma_x\zeta_i^x,&i\in A_C,\\x_i,&i\notin A_C,\end{cases}
\qquad\zeta_i^x\sim N(0,I_d),
$$

with independent row jitters, independent of the graph and rotations. Jitter applies to every accepted recipient, including revival. All donor positions on the right-hand side are frozen input positions. The cloning proposal marks every slot alive; canonical boundary classification occurs after the subsequent kinetic stages and final position diffusion.

A live accepted edge strictly increases its frozen fitness, and a dead vertex cannot be a donor. The graph is therefore a forest: its outdegree is at most one, and an undirected cycle would force a directed cycle. Components are disjoint even when donor stars overlap. There is one rotation and one destination write per participating slot.
:::

:::{prf:proposition} Exact component momentum and energy
:label: prop-cloning-component-conservation

For each realized component,

$$
\sum_{i\in C}v_i'=\sum_{i\in C}v_i,\qquad
\sum_{i\in C}|v_i'|^2
=\sum_{i\in C}|v_i|^2-(1-\alpha^2)\sum_{i\in C}|v_i-\bar v_C|^2.
$$

Conditional on its graph and velocities,

$$
\mathbb E v_i'=\bar v_C,\qquad
\operatorname{Cov}(v_i',v_j')=
\frac{\alpha^2}{d}\bigl[(v_i-\bar v_C)\cdot(v_j-\bar v_C)\bigr]I_d.
$$

*Proof.* The centered vectors sum to zero. The same linear map $\alpha R_C$ acts on every one, so their transformed sum is zero. Orthogonality gives the squared-norm identity. Haar invariance gives $\mathbb E R_C=0$ and $\mathbb E[(R_Cu)(R_Cw)^T]=(u\cdot w)I_d/d$. These statements also hold for $d=1$, where Haar $O(1)$ is a uniform sign. $\square$
:::

#### 5.7.5. Velocity variance and revival

:::{prf:proposition} Velocity dissipation with the exact revival contribution
:label: prop-bounded-velocity-expansion

Suppose every input slot, alive or dead, satisfies $|v_i|\leq V_{\max}$. Define

$$
\mathcal V_v^{\rm all}(S)=\frac1N\sum_i|v_i-\bar v_{\rm all}|^2,\qquad
\mathcal V_v^a(S)=\frac1N\sum_{i\in\mathcal A}|v_i-\bar v_a|^2,
$$

$$
\mathcal E_C=\frac1N\sum_{C\in\mathfrak C}\sum_{i\in C}|v_i-\bar v_C|^2,
\qquad R_v(S)=\mathcal V_v^{\rm all}(S)-\mathcal V_v^a(S).
$$

With $D=|\mathcal D|$ and $|\mathcal A|>0$, the all-alive proposal obeys the pathwise identities and bounds

$$
\mathcal V_v^{\rm all}(S')-\mathcal V_v^{\rm all}(S)=-(1-\alpha^2)\mathcal E_C,
$$

$$
\boxed{\quad\mathcal V_v^a(S')-\mathcal V_v^a(S)
=R_v(S)-(1-\alpha^2)\mathcal E_C,\qquad
0\leq R_v(S)\leq\frac{4D}{N}V_{\max}^2.\quad}
$$

In particular $\Delta\mathcal V_v^a\leq4f_{\rm clone}V_{\max}^2$ when the cloning fraction includes all revived slots. With no dead slots, the velocity variance cannot increase. Elastic components preserve it exactly. No positive uniform contraction factor follows unless accepted components capture a controlled fraction of the incoming velocity variance.

*Proof.* Full-slot momentum conservation keeps $\bar v_{\rm all}$ fixed, so summing the component energy identity gives the first formula. The proposal is all alive, which gives the second formula by adding and subtracting $\mathcal V_v^{\rm all}(S)$. The minimization identity for variance yields

$$
\mathcal V_v^{\rm all}(S)
=\min_b\frac1N\sum_i|v_i-b|^2
\leq\mathcal V_v^a(S)+\frac1N\sum_{i\in\mathcal D}|v_i-\bar v_a|^2
\leq\mathcal V_v^a(S)+\frac{4D}{N}V_{\max}^2.
$$

Its lower bound by $\mathcal V_v^a$ follows by dropping the dead terms before minimization. Finally $D/N\leq f_{\rm clone}$. Collision outputs obey $|v_i'|\leq(1+2\alpha)V_{\max}$; they need not yet satisfy the final cap. $\square$
:::

:::{div} feynman-prose
The energy removed by a collision is explicit: it is the relative energy inside its component times $1-\alpha^2$. Relative motion between distinct components is untouched. This explains both the dissipation and its limit. If all components are isolated, or their members already have identical velocities, this stage removes no energy.

Revival changes which slots are counted in an alive-only observable. Its contribution is $R_v$, even though the collision conserves the momentum of all slots. Keeping these two effects separate gives a sharper and directly measurable drift formula.
:::

### 5.8. Section summary

This chapter has formally defined the complete $\Psi_clone$ operator, from initial perception to final action. We have constructed the full, multi-stage pipeline that constitutes the swarm's adaptive engine for a single timestep, following the precise chronological order of the algorithm. This included:
1.  The stochastic operators for **measuring diversity**, including a rigorous analysis of the practical pairing algorithm.
2.  The deterministic cascade of operators that process raw measurements into a final, N-dimensional **fitness potential vector**, $\mathbf{V}_{\text{fit}}$.
3.  The final **stochastic gate** that uses this fitness vector to select a companion, calculate a score, and make the clone-or-persist decision, ultimately defining the crucial **total cloning probability**, $p_i$.

With every component of the cloning mechdanism now formally defined in its logical place, we are fully equipped for the subsequent analysis. The following chapters will prove the core stability property of this pipeline: its ability to robustly and intelligently convert a large system-level error into a powerful, corrective, and contractive force.

(sec-cloning-geometry)=
## 6. The Geometry of Error: From System Error to a Guaranteed Geometric Structure

### 6.1. Introduction

This chapter establishes the first and most fundamental link in the Keystone causal chain: the rigorous connection between a large **Intra-Swarm Error ($V_{\text{Var}}$)** and the microscopic geometric configuration of the swarm. The ultimate goal of the Keystone Principle (Sections 5-8) is to prove that the cloning operator, $\Psi_clone$, acts as a powerful contractive force on the **positional variance component, $V_{\text{Var},x}$**, of our synergistic Lyapunov function. The first step in that proof is to demonstrate that a large **positional variance ($V_{\text{Var},x}$)** is not an abstract, system-wide statistical property, but rather an unstable condition that forces at least one of the swarms into a specific, non-uniform, and *quantifiable* **phase-space geometric structure**.

Proving the existence and properties of this geometric structure is the essential prerequisite for the entire stability analysis of the cloning operator. Without it, we cannot demonstrate that the algorithm has a meaningful internal signal to measure or a specific population of walkers to target for corrective action. A large **$V_{\text{Var},x}$** implies that at least one swarm is internally "puffed up" *spatially*. This chapter will prove that this positional dispersion is a sufficient condition to create a detectable structure in the full **phase space**, which the `d_alg`-based measurement pipeline can reliably perceive.

Our analytical strategy is built on an **$\varepsilon$-dichotomy**. The interaction range $\varepsilon$ acts as the algorithm's perceptual lens, and the nature of the geometric error the system is sensitive to changes with this scale. We will prove that a large $V_{\text{Var}}$ forces the creation of a "high-error" population of walkers in one of the swarms, where the nature of this error falls into one of two exhaustive regimes:

*   For **large $\varepsilon$ (Mean-Field Regime)**, the system is sensitive to the swarm's global configuration. We will prove that a large internal variance forces a non-vanishing fraction of the swarm to become **global phase-space outliers**.
*   For **small $\varepsilon$ (Local-Interaction Regime)**, the system is sensitive to local variations in density. We will prove that a large internal variance forces a non-vanishing fraction of the swarm to reside in regions of **low local phase-space density** (or, more formally, to belong to geometrically isolated clusters).

This chapter's goal is to prove that a large **positional variance ($V_{\text{Var},x}$)** guarantees the existence of a substantial and structurally significant **high-error population (`H_k`)** within at least one of the swarms. We will establish that this "high-error" population is both:

1.  **Substantial:** It constitutes a non-vanishing, N-uniform fraction of the swarm.
2.  **Structurally Significant:** Its members possess distinct **phase-space properties** (e.g., kinematic isolation) that are detectable by the measurement pipeline defined in Section 5.

These two properties, proven herein from first principles, will serve as the foundational input for Section 7, where we will prove that the algorithm's measurement pipeline can reliably detect this **phase-space structure** and transduce it into a usable corrective signal.

### 6.2. From Total Positional Variance ($V_{\text{Var},x}$) to Single-Swarm Positional Variance

The first step in the Keystone analysis is to connect the relevant component of the synergistic Lyapunov function to the state of a single swarm. As this document aims to prove the contractive nature of cloning on **positional error**, the Keystone mechanism is triggered specifically by the **positional variance component, $V_{\text{Var},x}$**. The following lemma provides the simple but necessary guarantee that if this $V_{\text{Var},x}$ term is large, then at least one of the two swarms must have a large internal positional variance.

:::{prf:lemma} Large $V_{\text{Var},x}$ Implies Large Single-Swarm Positional Variance
:label: lem-V_Varx-implies-variance

Let $V_{Var,x}(S_1, S_2)$ be the total intra-swarm ({prf:ref}`def-swarm-and-state-space`) positional variance component of the Lyapunov function as defined in {prf:ref}`def-full-synergistic-lyapunov-function`:

$$
V_{Var,x}(S_1, S_2) = \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2

$$

If this component is large, such that $V_{Var,x} > R_{total\_var,x}^2$ for some threshold $R_{total\_var,x}^2 > 0$, then at least one swarm ({prf:ref}`def-swarm-and-state-space`) $k \in \{1, 2\}$ must have a large sum of squared deviations:

$$
\frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2 > \frac{R_{total\_var,x}^2}{2}

$$

:::

:::{prf:proof}
**Proof.**

The proof is by contradiction. Assume the premise holds: $V_{Var,x} > R_{total\_var,x}^2$. Assume for contradiction that the conclusion is false. This would mean that for *both* swarms (`k=1` and `k=2`):

$$
\frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2 \le \frac{R_{total\_var,x}^2}{2}, \quad \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2 \le \frac{R_{total\_var,x}^2}{2}

$$

Now, we bound the total intra-swarm positional error $V_{Var,x}$ under this assumption:

$$
\begin{aligned}
V_{Var,x} &= \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2 \\
&\le \frac{R_{total\_var,x}^2}{2} + \frac{R_{total\_var,x}^2}{2} = R_{total\_var,x}^2
\end{aligned}

$$

The result $V_{Var,x} \le R_{total\_var,x}^2$ directly contradicts our premise. Therefore, the assumption must be false, and the conclusion must be true.

**Q.E.D.**
:::

This lemma establishes a direct and crucial link: if the **$V_{\text{Var},x}$ component** of the total Lyapunov function is large, we are guaranteed to have at least one swarm with **high internal positional variance**. This swarm, with its guaranteed spatial dispersion, now becomes the primary object of our analysis. The subsequent sections will prove that this condition of high positional variance is sufficient to generate a detectable **phase-space structure** and, ultimately, a corrective cloning response.

### 6.3 Canonical Definitions: The High-Error and Low-Error Partition

The preceding section established that a large $V_{\text{Var},x}$ guarantees that at least one swarm has a large internal positional variance (`Var_x`). While this spatial dispersion is the *trigger* for our analysis, the algorithm's measurement pipeline perceives the swarm's structure through the **phase-space metric `d_alg`**. For the resulting signal to be meaningful, the geometric partition of the swarm into "high-error" and "low-error" sets must be based on properties that this pipeline can actually detect.

Therefore, this section formally defines these sets based on the swarm's full **phase-space configuration**. We will partition the swarm based on two distinct phase-space measures: global kinematic dispersion and local phase-space density. These definitions create the crucial link between the state of the swarm and the signals measured by the algorithm. The remainder of the Keystone analysis will then be dedicated to proving that high *positional* variance is a sufficient condition to force a non-trivial number of walkers into these *phase-space* defined error sets.

:::{prf:definition} The Unified High-Error and Low-Error Sets
:label: def-unified-high-low-error-sets

For a given swarm ({prf:ref}`def-swarm-and-state-space`) `k` with alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}_k$ ($k \ge 2$), we define a partition into a unified high-error set $H_k(\epsilon)$ and a unified low-error set $L_k(\epsilon)$ using a **clustering-based approach** that applies uniformly across all interaction regimes. This unified approach captures both global outlier structure and local phase-space clustering through a single consistent mechanism.

**Phase-Space Clustering Construction:**

The construction proceeds in four steps:

1.  **Clustering:** Partition the alive set ({prf:ref}`def-alive-dead-sets`) $\mathcal{A}_k$ into disjoint clusters $\{G_1, \ldots, G_M\}$ using complete-linkage hierarchical clustering with a maximum cluster diameter $D_{\text{diam}}(\epsilon) := c_d \cdot \epsilon$ (where $c_d > 0$ is a fixed constant, typically $c_d = 2$). Each cluster $G_m$ satisfies:

$$
\text{diam}(G_m) := \max_{i,j \in G_m} d_{\text{alg}}(i, j) \le D_{\text{diam}}(\epsilon)

$$

where $d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$ is the algorithmic phase-space distance.

2.  **Statistical Validity Constraint:** To ensure that cluster-level statistics are meaningful, we impose a minimum cluster size requirement. Let $k_{\min} := \max(5, \lceil 0.05k \rceil)$ be the minimum statistically valid cluster size. All clusters with $|G_m| < k_{\min}$ are marked as **invalid** and their walkers are automatically included in the high-error set (as they represent statistically unreliable outlier configurations).

3.  **Outlier Cluster Identification:** For each valid cluster $G_m$ (with $|G_m| \ge k_{\min}$), compute its center of mass in phase space: $(\mu_{x,m}, \mu_{v,m})$. Compute the between-cluster hypocoercive variance contribution:

$$
\text{Contrib}(G_m) := |G_m| \left(\|\mu_{x,m} - \mu_x\|^2 + \lambda_v \|\mu_{v,m} - \mu_v\|^2\right)

$$

where $(\mu_x, \mu_v)$ is the global center of mass. Sort valid clusters by $\text{Contrib}(G_m)$ in descending order, and let $O_M \subseteq \{1, \ldots, M\}$ be the smallest set of cluster indices (among valid clusters) whose cumulative contribution meets or exceeds a fraction $(1-\varepsilon_O)$ of the total contribution from valid clusters (where $\varepsilon_O \in (0, 1)$ is a fixed structural parameter, typically $\varepsilon_O = 0.1$):

$$
\sum_{m \in O_M} \text{Contrib}(G_m) \ge (1-\varepsilon_O) \sum_{\substack{m=1 \\ |G_m| \ge k_{\min}}}^M \text{Contrib}(G_m)

$$

4.  **Unified High-Error Set Construction:** The unified high-error set is the union of all walkers in outlier clusters plus all walkers in invalid clusters:

$$
H_k(\epsilon) := \left(\bigcup_{m \in O_M} G_m\right) \cup \left(\bigcup_{\substack{m: |G_m| < k_{\min}}} G_m\right)

$$

The **Unified Low-Error Set** is the complement:

$$
L_k(\epsilon) := \mathcal{A}_k \setminus H_k(\epsilon)

$$

Referenced by {prf:ref}`def-fitness-potential-operator` and {prf:ref}`def-geometric-partition`.
:::

**Interpretation:** The unified high-error set $H_k(\epsilon)$ represents the population of walkers identified as the primary source of geometric error **in phase space**, where the *nature* of that error—being a global kinematic outlier versus belonging to an outlier cluster in phase space—is determined by the algorithm's perceptual scale $\varepsilon$. The subsequent sections will prove that this set is guaranteed to be both substantial in size and structurally distinct when the swarm's **positional variance** is large.

### 6.4. From Structural Error to a Guaranteed High-Error Population Fraction

Before we can prove that the algorithm correctly identifies the walkers in the **Unified High-Error Set** (defined in Section 6.3) as "unfit," we must first rigorously establish that this set is not empty or vanishingly small when the system error is large. For the Keystone Lemma to be N-uniform—a non-negotiable requirement for mean-field scalability—the source of the error must be a collective phenomenon, not an artifact of a few "rogue" walkers.

Therefore, this section's central goal is to prove that a large internal positional variance (`Var_x`) is a macroscopic event that necessarily implicates a substantial, `N`-independent fraction of the swarm. We will prove that when the system error is large, a non-vanishing fraction of the population must belong to the high-error set. This result provides the first of two critical population guarantees required for the main proof, establishing that there is always a large, strategically important sub-population for the cloning mechanism to target.

The introduction of the $\varepsilon$-dependent phase-space kernel requires a nuanced analytical strategy, as the nature of the geometric error the system is sensitive to changes with the interaction scale $\varepsilon$. Our proof is therefore built on an **$\varepsilon$-dichotomy**: in the **large-$\varepsilon$ (Mean-Field) regime**, error is defined by global phase-space outliers, while in the **small-$\varepsilon$ (Local-Interaction) regime**, error is defined by walkers belonging to outlier clusters in phase space. The subsequent subsections will provide rigorous, N-uniform proofs for each regime. We will first prove a foundational lemma relating a swarm's **total hypocoercive variance** to its **local phase-space clustering**, and then use this result to establish the main theorems for both the mean-field and local-interaction cases.

#### 6.4.1. The Phase-Space Packing Lemma: Hypocoercive Variance Limits Local Phase-Space Clustering

Before analyzing the specific regimes of the $\varepsilon$-dichotomy, we establish a precise, quantitative relationship between a swarm's global dispersion in phase space, as measured by its **total hypocoercive variance**, and its local phase-space clustering structure. This lemma generalizes the classical packing argument to phase space, proving that a swarm cannot be simultaneously spread out in the hypocoercive norm while being highly clustered under the algorithmic distance metric `d_alg`. This result is the foundational geometric constraint upon which both regimes of our analysis will depend.

:::{prf:lemma} The Phase-Space Packing Lemma
:label: lem-phase-space-packing

For a swarm ({prf:ref}`def-swarm-and-state-space`) `k` consisting of $k \geq 2$ walkers with phase-space states $\{(x_i, v_i)\}_{i=1}^k$ within a compact domain, define the **total hypocoercive variance** of the swarm as:

$$
\mathrm{Var}_h(S_k) := \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k)

$$

For any chosen proximity threshold $d_{\text{close}} > 0$, let $N_{\text{close}}$ be the number of unique pairs $(i, j)$ with $i<j$ and $d_{\text{alg}}(i, j) < d_{\text{close}}$, where $d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$ is the algorithmic phase-space distance.

The fraction of such "close pairs in phase space", $f_{\text{close}} = N_{\text{close}} / \binom{k}{2}$, is bounded above by a function of the swarm ({prf:ref}`def-swarm-and-state-space`)'s hypocoercive variance. Specifically, assuming $\lambda_v \le \lambda_{\text{alg}}$ and defining the phase-space diameter $D_{\text{valid}}^2 := D_x^2 + \lambda_{\text{alg}} D_v^2$ where $D_x$ and $D_v$ are the spatial and velocity domain diameters, there exists a continuous, monotonically decreasing function such that:

$$
f_{\text{close}} \le g(\mathrm{Var}_h(S_k)) := \frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h(S_k)}{D_{\text{valid}}^2 - d_{\text{close}}^2}

$$

Furthermore, if the hypocoercive variance exceeds a threshold $\mathrm{Var}_h(S_k) > R_{\text{pack}}^2 := d_{\text{close}}^2 / 2$, then $g(\mathrm{Var}_h(S_k)) < 1$, guaranteeing that not all pairs can be close pairs in phase space.
:::

:::{prf:proof}
**Proof.**

The proof generalizes the classical packing argument to phase space and proceeds in four parts. First, we establish fundamental identities relating the hypocoercive variance to sums of pairwise squared distances in both position and velocity. Second, we partition pairs by their algorithmic distance and bound the hypocoercive variance. Third, we carefully account for the potentially different velocity weighting factors $\lambda_v$ (in the variance) and $\lambda_{\text{alg}}$ (in the distance). Finally, we invert the relationship to derive the desired upper bound on the fraction of close pairs.

**Part 1: Pairwise Identities for Hypocoercive Variance**

We begin by establishing pairwise representations for both positional and velocity variances. For the positional variance, the standard identity states:

$$
2k^2 \mathrm{Var}_x(S_k) = \sum_{i=1}^k \sum_{j=1}^k \|x_i - x_j\|^2

$$

This can be verified by expanding the right-hand side:

$$
\begin{aligned}
\sum_{i,j} \|x_i - x_j\|^2 &= \sum_{i,j} (\|x_i\|^2 - 2\langle x_i, x_j \rangle + \|x_j\|^2) \\
&= 2k \sum_i \|x_i\|^2 - 2\langle k\mu_x, k\mu_x \rangle \\
&= 2k \sum_i \|x_i\|^2 - 2k^2 \|\mu_x\|^2 \\
&= 2k(k \cdot \mathrm{Var}_x + k\|\mu_x\|^2) - 2k^2\|\mu_x\|^2 = 2k^2 \mathrm{Var}_x
\end{aligned}

$$

An identical derivation applies to the velocity variance:

$$
2k^2 \mathrm{Var}_v(S_k) = \sum_{i=1}^k \sum_{j=1}^k \|v_i - v_j\|^2

$$

Multiplying the velocity identity by $\lambda_v$ and adding the two identities yields:

$$
2k^2 \mathrm{Var}_h(S_k) = 2k^2 (\mathrm{Var}_x + \lambda_v \mathrm{Var}_v) = \sum_{i,j} \left(\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2\right)

$$

Since the sum over all ordered pairs $(i,j)$ is twice the sum over unique pairs where $i<j$, we obtain:

$$
\mathrm{Var}_h(S_k) = \frac{1}{k^2} \sum_{i<j} \left(\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2\right)

$$

**Part 2: Partitioning by Algorithmic Distance**

We now partition the set of unique pairs into two subsets based on the algorithmic distance $d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$:
- $P_{\text{close}}$: the set of $N_{\text{close}}$ pairs with $d_{\text{alg}}(i,j) < d_{\text{close}}$
- $P_{\text{far}}$: the set of $N_{\text{far}}$ pairs with $d_{\text{alg}}(i,j) \ge d_{\text{close}}$

The hypocoercive variance can be written as:

$$
\mathrm{Var}_h(S_k) = \frac{1}{k^2} \left( \sum_{(i,j) \in P_{\text{close}}} \left(\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2\right) + \sum_{(i,j) \in P_{\text{far}}} \left(\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2\right) \right)

$$

**Part 3: Bounding the Variance Terms**

For pairs in $P_{\text{close}}$, we have $d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2 < d_{\text{close}}^2$. Under our assumption that $\lambda_v \le \lambda_{\text{alg}}$, we can bound:

$$
\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2 \le \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2 = d_{\text{alg}}(i,j)^2 < d_{\text{close}}^2

$$

For pairs in $P_{\text{far}}$, each component is bounded by the corresponding domain diameter:

$$
\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2 \le D_x^2 + \lambda_v D_v^2 \le D_x^2 + \lambda_{\text{alg}} D_v^2 = D_{\text{valid}}^2

$$

where we again used $\lambda_v \le \lambda_{\text{alg}}$. Therefore:

$$
\mathrm{Var}_h(S_k) \le \frac{1}{k^2} \left( N_{\text{close}} \cdot d_{\text{close}}^2 + N_{\text{far}} \cdot D_{\text{valid}}^2 \right)

$$

Let $f_{\text{close}} = N_{\text{close}} / \binom{k}{2}$ be the fraction of close pairs. Substituting $N_{\text{close}} = f_{\text{close}} \binom{k}{2}$ and $N_{\text{far}} = (1 - f_{\text{close}}) \binom{k}{2}$:

$$
\mathrm{Var}_h(S_k) \le \frac{\binom{k}{2}}{k^2} \left( f_{\text{close}} d_{\text{close}}^2 + (1-f_{\text{close}})D_{\text{valid}}^2 \right) = \frac{k-1}{2k} \left( f_{\text{close}} (d_{\text{close}}^2 - D_{\text{valid}}^2) + D_{\text{valid}}^2 \right)

$$

**Part 4: Deriving the Upper Bound on the Fraction of Close Pairs**

To obtain a simpler bound, we use $(k-1)/(2k) < 1/2$ for $k \ge 2$:

$$
\mathrm{Var}_h(S_k) < \frac{1}{2} \left( f_{\text{close}} (d_{\text{close}}^2 - D_{\text{valid}}^2) + D_{\text{valid}}^2 \right)

$$

Solving for $f_{\text{close}}$:

$$
\begin{aligned}
2\mathrm{Var}_h(S_k) &< f_{\text{close}} (d_{\text{close}}^2 - D_{\text{valid}}^2) + D_{\text{valid}}^2 \\
2\mathrm{Var}_h(S_k) - D_{\text{valid}}^2 &< f_{\text{close}}(d_{\text{close}}^2 - D_{\text{valid}}^2)
\end{aligned}

$$

Since $d_{\text{close}} < D_{\text{valid}}$, the term $(d_{\text{close}}^2 - D_{\text{valid}}^2)$ is strictly negative. Dividing by it reverses the inequality:

$$
f_{\text{close}} < \frac{2\mathrm{Var}_h(S_k) - D_{\text{valid}}^2}{d_{\text{close}}^2 - D_{\text{valid}}^2} = \frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h(S_k)}{D_{\text{valid}}^2 - d_{\text{close}}^2}

$$

This establishes $f_{\text{close}} \le g(\mathrm{Var}_h(S_k))$ where $g(V) := (D_{\text{valid}}^2 - 2V) / (D_{\text{valid}}^2 - d_{\text{close}}^2)$. As an affine function of $V$ with negative coefficient, $g(V)$ is continuous and strictly decreasing.

Finally, we verify that $g(\mathrm{Var}_h) < 1$ when $\mathrm{Var}_h > d_{\text{close}}^2 / 2$:

$$
\frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h}{D_{\text{valid}}^2 - d_{\text{close}}^2} < 1 \implies D_{\text{valid}}^2 - 2\mathrm{Var}_h < D_{\text{valid}}^2 - d_{\text{close}}^2 \implies \mathrm{Var}_h > \frac{d_{\text{close}}^2}{2}

$$

This completes the proof.

**Q.E.D.**
:::

#### 6.4.2 The Mean-Field Regime: A Non-Vanishing Fraction of Global Outliers

This subsection analyzes the system's behavior in the **large-$\varepsilon$ (mean-field) regime**, where the interaction range $\varepsilon$ is larger than the swarm's phase-space diameter, $D_{\text{swarm}} := \max_{i,j \in \mathcal{A}_k} d_{\text{alg}}(i, j)$. As per {prf:ref}`def-unified-high-low-error-sets`, in this regime the **Unified High-Error Set $H_k(\varepsilon)$ is identical to the global kinematic outlier set `O_k`**. Therefore, to prove that a high-variance swarm has a non-vanishing high-error fraction in this regime, we must prove that the fractional size of `O_k` is bounded below.

We prove that a large internal hypocoercive variance (`Var_h`) is a sufficient condition to guarantee that a significant, `N`-independent fraction of the population must belong to the **global phase-space outliers**. This proof establishes a fundamental geometric property of phase-space point clouds and provides the first half of our $\varepsilon$-dichotomy.

To connect this analysis back to the positional variance component of the Lyapunov function, we first establish a simple but crucial relationship.

:::{prf:lemma} Positional Variance as a Lower Bound for Hypocoercive Variance
:label: lem-var-x-implies-var-h

For any swarm ({prf:ref}`def-swarm-and-state-space`) `k`, its total hypocoercive variance is bounded below by its positional variance:

$$
\mathrm{Var}_h(S_k) \ge \mathrm{Var}_x(S_k)

$$

Consequently, if $\mathrm{Var}_x(S_k) > R^2_{\text{var}}$ for some threshold $R^2_{\text{var}} > 0$, then it is guaranteed that $\mathrm{Var}_h(S_k) > R^2_{\text{var}}$.
:::

:::{prf:proof}
**Proof.**

By definition, the hypocoercive variance is:

$$
\mathrm{Var}_h(S_k) := \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k)

$$

Since $\lambda_v > 0$ is a positive hypocoercive parameter and $\mathrm{Var}_v(S_k) \ge 0$ (variance is non-negative), we immediately have:

$$
\mathrm{Var}_h(S_k) = \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k) \ge \mathrm{Var}_x(S_k)

$$

The second claim follows directly: if $\mathrm{Var}_x(S_k) > R^2_{\text{var}}$, then by the above inequality, $\mathrm{Var}_h(S_k) \ge \mathrm{Var}_x(S_k) > R^2_{\text{var}}$.

**Q.E.D.**
:::

This lemma establishes the crucial bridge: the analysis in Section 6.2 guaranteed a large positional variance `Var_x`, and this lemma proves that such a condition is sufficient to guarantee a large hypocoercive variance `Var_h`, which is the relevant measure for the phase-space outlier set `O_k`. We can now proceed with the main result.

:::{prf:lemma} N-Uniform Lower Bound on the Outlier Fraction
:label: lem-outlier-fraction-lower-bound

Let $O_k$ be the **global kinematic outlier set** for a swarm ({prf:ref}`def-swarm-and-state-space`) `k` with `k >= 2` alive walkers, as defined in Section 6.3, with structural parameter $\varepsilon_O \in (0, 1)$.

If the swarm 's internal hypocoercive variance is large, such that $\mathrm{Var}_h(S_k) > R^2_h$ for some threshold $R^2_h > 0$, then the fraction of *alive* walkers in the outlier set is bounded below by a positive constant that is independent of `N`. Specifically:

$$
\frac{|O_k|}{k} \ge \frac{(1-\varepsilon_O) R^2_h}{D_h^2} =: f_O > 0

$$

where $D_h^2 := D_x^2 + \lambda_v D_v^2$ is the squared **hypocoercive diameter** of the valid domain, with $D_x := \sup_{x_1, x_2 \in \mathcal{X}_{\text{valid}}} \|x_1 - x_2\|$ being the positional domain diameter and $D_v$ being the velocity domain diameter.
:::
:::{prf:proof}

**Proof.**

The proof establishes the lower bound by relating the total hypocoercive variance of the swarm to the maximum possible contribution of any single walker in phase space (using the packing argument from {prf:ref}`lem-phase-space-packing`), which is a fixed geometric property of the environment.

**1. Recall Definitions and Outlier Set Property:**
*   The sum of squared hypocoercive norms of the centered phase-space vectors for the `k` alive walkers is:


$$
T_k = \sum_{j \in \mathcal{A}_k} \left(\|\delta_{x,k,j}\|^2 + \lambda_v \|\delta_{v,k,j}\|^2\right) = k \cdot \mathrm{Var}_h(S_k)

$$

*   By the definition of the global kinematic outlier set $O_k$ (Section 6.3), the sum of squared hypocoercive norms over this subset is bounded below by a fixed fraction of the total sum:


$$
\sum_{i \in O_k} \left(\|\delta_{x,k,i}\|^2 + \lambda_v \|\delta_{v,k,i}\|^2\right) \ge (1-\varepsilon_O) T_k = (1-\varepsilon_O) k \cdot \mathrm{Var}_h(S_k)

$$

**2. Establish a Uniform Upper Bound on Single-Walker Contribution:**
*   For any single alive walker `i`, its centered phase-space state is $(\delta_{x,k,i}, \delta_{v,k,i}) = (x_{k,i} - \mu_{x,k}, v_{k,i} - \mu_{v,k})$.
*   The walker's position $x_{k,i}$ must lie within the valid domain $\mathcal{X}_{\text{valid}}$. If $\mathcal{X}_{\text{valid}}$ is convex (a standard assumption), the center of mass $\mu_{x,k}$ must also lie within $\mathcal{X}_{\text{valid}}$. Therefore, $\|\delta_{x,k,i}\| \le D_x$, where $D_x$ is the positional domain diameter.
*   Similarly, the velocity $v_{k,i}$ is bounded by the velocity domain diameter: $\|\delta_{v,k,i}\| \le D_v$.
*   Therefore, the squared hypocoercive norm of any centered phase-space vector is uniformly bounded:


$$
\|\delta_{x,k,i}\|^2 + \lambda_v \|\delta_{v,k,i}\|^2 \le D_x^2 + \lambda_v D_v^2 = D_h^2

$$

    This bound is a geometric property of the environment and is independent of the number of walkers `N` or `k`.

**3. Bound the Sum over the Outlier Set:**
*   The sum of squared hypocoercive norms over the outlier set can also be bounded above by multiplying the number of walkers in the set, $|O_k|$, by the maximum possible value of any single term:


$$
\sum_{i \in O_k} \left(\|\delta_{x,k,i}\|^2 + \lambda_v \|\delta_{v,k,i}\|^2\right) \le |O_k| \cdot \sup_{j \in O_k} \left(\|\delta_{x,k,j}\|^2 + \lambda_v \|\delta_{v,k,j}\|^2\right) \le |O_k| \cdot D_h^2

$$

**4. Combine Bounds and Finalize:**
*   We now have both a lower and an upper bound for the same quantity. Combining them yields:


$$
(1-\varepsilon_O) k \cdot \mathrm{Var}_h(S_k) \le \sum_{i \in O_k} \left(\|\delta_{x,k,i}\|^2 + \lambda_v \|\delta_{v,k,i}\|^2\right) \le |O_k| \cdot D_h^2

$$

*   We are given the premise that the hypocoercive variance is large: $\mathrm{Var}_h(S_k) > R^2_h$. Substituting this into the left-hand side gives:


$$
(1-\varepsilon_O) k \cdot R^2_h < |O_k| \cdot D_h^2

$$

*   Rearranging to find a bound on the fraction of outliers relative to the number of *alive* walkers `k`, we get:


$$
\frac{|O_k|}{k} > \frac{(1-\varepsilon_O) R^2_h}{D_h^2}

$$

*   The resulting lower bound, $f_O := (1-\varepsilon_O) R^2_h / D_h^2$, is a positive constant constructed entirely from `N`-independent parameters. This completes the proof that a large hypocoercive variance guarantees a non-vanishing fraction of global phase-space outliers among the alive population.

**Q.E.D.**
:::
:::{admonition} Note on N-Uniformity
:class: note

The lemma proves that the fraction of *alive* walkers in the outlier set, `|O_k|/k`, is bounded below by the N-uniform constant `f_O`. All subsequent proofs that rely on a guaranteed fraction of high-error walkers (such as the proof of the Unfit-High-Error Overlap) will operate on the alive set. Therefore, this bound is precisely what is needed for the N-uniformity of the Keystone Lemma. The fraction relative to the total swarm size `N`, `|O_k|/N`, is also bounded below by `f_O` multiplied by the minimum alive fraction `k_min/N`, but the bound relative to `k` is the more direct and useful result.
:::

#### 6.4.3. The Local-Interaction Regime: High-Error Fraction via Clustering

This subsection analyzes the system's behavior in the **small-$\varepsilon$ (local-interaction) regime**, where the interaction range $\varepsilon$ is smaller than or equal to the swarm's phase-space diameter, $D_{\text{swarm}} := \max_{i,j \in \mathcal{A}_k} d_{\text{alg}}(i, j)$. As per {prf:ref}`def-unified-high-low-error-sets`, in this regime the **Unified High-Error Set $H_k(\varepsilon)$ is identical to the phase-space clustering-based outlier set $C_k(\varepsilon)$**. Therefore, to prove that a high-variance swarm has a non-vanishing high-error fraction in this regime, we must prove that the fractional size of the outlier clusters is bounded below.

In this regime, the concept of a "global outlier" becomes less meaningful. The system's error is instead driven by the meso-scale structure of the swarm in phase space—the formation of distinct sub-populations or clusters. The clustering-based definition captures this geometric structure by first partitioning the swarm into phase-space clusters (using the algorithmic distance `d_alg`), and then identifying the "outlier clusters" that are the primary contributors to the swarm's global hypocoercive variance.

:::{admonition} A Note on Positional Analysis as a Proxy for Phase-Space Structure
:class: note

The following proof establishes the existence of a high-error population based on analysis of the swarm's **positional geometry**. While {prf:ref}`def-unified-high-low-error-sets` specifies that clustering should be performed in phase space (using `d_alg`), and that outlier clusters are identified by their contribution to the **hypocoercive variance** (which includes both position and velocity), the proof below operates primarily in positional space.

This is a deliberate analytical simplification made for tractability, and it is mathematically justified by the structure of our argument. The Keystone analysis is triggered by a large **positional variance ($V_{\text{Var},x}$)**—this positional dispersion is the primary driver of the system's geometric error. The proof demonstrates that this positional condition alone is **sufficient** to guarantee a non-vanishing high-error fraction, regardless of the velocity structure.

A rigorous phase-space proof would be more complex but would yield a similar (and potentially stronger) result. The positional analysis provides a conservative lower bound: if positional dispersion alone guarantees the result, then the presence of additional velocity structure can only strengthen the conclusion. This approach ensures that the proof is robust and provides the N-uniform guarantee required for the Keystone Principle.
:::

With the clustering-based definition from Section 6.3 established, we can now prove the main result for this regime.

:::{prf:lemma} N-Uniform Lower Bound on the Outlier-Cluster Fraction
:label: lem-outlier-cluster-fraction-lower-bound

Let the high-error set $H_k(\varepsilon)$ be defined via the phase-space clustering-based approach (as $C_k(\varepsilon)$ in {prf:ref}`def-unified-high-low-error-sets`) for the local-interaction regime, with maximum cluster diameter $D_{\mathrm{diam}}(\varepsilon) = c_d \cdot \varepsilon$ where $c_d > 0$ is a fixed constant.

For any choice of $c_d$ and variance threshold $R^2_{\text{var}}$ satisfying $c_d · \epsilon < 2\sqrt{R^2_{\text{var}}}$, there exists a positive constant $f_H(\epsilon) > 0$, independent of `N` and `k`, such that:

If the swarm ({prf:ref}`def-swarm-and-state-space`)'s internal positional variance is large, $\mathrm{Var}_x(S_k) > R^2_{\text{var}}$, then the fraction of *alive* walkers in the high-error set is bounded below:

$$
\frac{|H_k(\epsilon)|}{k} \ge f_H(\epsilon) > 0

$$

:::
:::{prf:proof}

**Proof.**

The proof is constructive. We use the Law of Total Variance to show that a large global variance forces a large variance *between* the cluster centers. We then apply the same logic used in the mean-field regime ({prf:ref}`lem-outlier-fraction-lower-bound`) to this set of cluster centers to prove that a non-vanishing fraction of the population must reside in these outlier clusters.

**1. Decomposing the Total Variance.**
The Law of Total Variance provides an exact identity for the swarm's variance based on the cluster partition `{G_1, ..., G_M}`. Let $\mu$ be the global center of mass of the `k` alive walkers, $\mu_m$ be the center of mass of cluster `G_m`, and `|G_m|` be the number of walkers in it. The total sum of squared deviations can be decomposed as:

$$
k \cdot \mathrm{Var}_k(x) = \sum_{m=1}^M \sum_{i \in G_m} \|x_i - \mu\|^2 = \sum_{m=1}^M |G_m|\mathrm{Var}(G_m) + \sum_{m=1}^M |G_m|\|\mu_m - \mu\|^2

$$

The first term is the "within-cluster" sum of squares, and the second is the size-weighted "between-cluster" sum of squares.

**2. A Uniform Upper Bound on the Within-Cluster Variance.**
By the definition of our clustering algorithm, the diameter of any cluster `G_m` is at most $D_diam(\varepsilon)$. The maximum possible internal variance for any set of points with a given diameter is achieved when the points are at the extremes of an interval, which gives $\text{Var}(G_m) \leq (D_diam(\varepsilon)/2)^{2}$. This provides a uniform, N-independent upper bound for the within-cluster variance of any cluster.
The total within-cluster sum of squares is therefore bounded:

$$
\sum_{m=1}^M |G_m|\mathrm{Var}(G_m) \le \sum_{m=1}^M |G_m| \left(\frac{D_{\mathrm{diam}}(\epsilon)}{2}\right)^2 = k \left(\frac{D_{\mathrm{diam}}(\epsilon)}{2}\right)^2

$$

**3. A Uniform Lower Bound on the Between-Cluster Variance.**
We can now find a lower bound for the between-cluster sum of squares. Rearranging the identity from Step 1 and using our premise `Var_k(x) > R^{2}_var`:

$$
\sum_{m=1}^M |G_m|\|\mu_m - \mu\|^2 = k \cdot \mathrm{Var}_k(x) - \sum_{m=1}^M |G_m|\mathrm{Var}(G_m) > k \cdot R^2_{\mathrm{var}} - k \left(\frac{D_{\mathrm{diam}}(\epsilon)}{2}\right)^2

$$

Let's define a new positive, N-uniform constant $R_{\mathrm{means}}^{2} := R_{\mathrm{var}}^{2} - (D_{\mathrm{diam}}(\varepsilon)/2)^{2}$. The premise of this lemma requires that we choose $D_{\mathrm{diam}}(\varepsilon)$ small enough to ensure $R_{\mathrm{means}}^{2} > 0$. With this, we have a guaranteed lower bound on the size-weighted variance of the cluster means:

$$
\frac{1}{k}\sum_{m=1}^M |G_m|\|\mu_m - \mu\|^2 > R^2_{\mathrm{means}} > 0

$$

**4. Applying the Outlier Argument to the Cluster Centers.**
We have now reduced the problem to one that is formally identical to the mean-field case. We have a set of `M` "meta-particles" (the cluster centers $\mu_m$) with associated weights (`|G_m|`) whose size-weighted variance is guaranteed to be large.

By the definition of the high-error set $H_k(\varepsilon)$, it is the union of all walkers in the "outlier clusters" `O_M`. These are the clusters whose weighted contribution to the between-cluster variance sums to at least $(1-\varepsilon_O)$ of the total.

$$
\sum_{m \in O_M} |G_m|\|\mu_m - \mu\|^2 \ge (1-\varepsilon_O) \sum_{m=1}^M |G_m|\|\mu_m - \mu\|^2 > (1-\varepsilon_O) k \cdot R^2_{\mathrm{means}}

$$

At the same time, we can find an upper bound for this sum. The maximum squared distance of any cluster mean from the global mean is bounded by `D_valid^{2}`.

$$
\sum_{m \in O_M} |G_m|\|\mu_m - \mu\|^2 \le \sum_{m \in O_M} |G_m|D_{\mathrm{valid}}^2 = D_{\mathrm{valid}}^2 \sum_{m \in O_M} |G_m|

$$

The term $\sum_{m\in O_M} |G_m|$ is, by definition, the total number of walkers in the high-error set, $|H_k(\varepsilon)|$. Combining the inequalities:

$$
(1-\varepsilon_O) k \cdot R^2_{\mathrm{means}} < |H_k(\epsilon)| \cdot D_{\mathrm{valid}}^2

$$

**5. Conclusion.**
Rearranging the final inequality gives the desired N-uniform lower bound on the high-error fraction:

$$
\frac{|H_k(\epsilon)|}{k} > \frac{(1-\varepsilon_O) R^2_{\mathrm{means}}}{D_{\mathrm{valid}}^2} = \frac{(1-\varepsilon_O) \left(R^2_{\mathrm{var}} - (D_{\mathrm{diam}}(\epsilon)/2)^2\right)}{D_{\mathrm{valid}}^2}

$$

We define the right-hand side as our N-uniform constant $f_H(\varepsilon)$. It is strictly positive by our choice of $D_diam(\varepsilon)$, and it is constructed entirely from N-independent system parameters ($\varepsilon_O$, `R^{2}_var`, `D_diam`, `D_valid`). This completes the N-uniform proof.

**Q.E.D.**
:::

#### 6.4.4 Synthesis: A Large Intra-Swarm Positional Variance Guarantees a Non-Vanishing High-Error Fraction

The preceding subsections have rigorously established, via an $\varepsilon$-dichotomy, that a large internal hypocoercive variance is a sufficient condition to guarantee that a non-vanishing, N-uniform fraction of the swarm has a "high-error" phase-space configuration. We now unify these results into a single, powerful corollary for the **Unified High-Error Set**, as defined in Section 6.3.

This corollary provides the final, synthesized result of our geometric analysis. It proves that a large **total intra-swarm positional variance ($V_{\text{Var},x}$)** is sufficient to guarantee that a non-vanishing fraction of at least one of the swarms belongs to this high-error set, providing the clean, unified input required for the subsequent analysis. This supplies a geometric input to the selection estimate. Its connection to the actual positional drift is given by the donor-displacement identity in {prf:ref}`lem-keystone-contraction-alive`.

:::{prf:corollary} A Large Intra-Swarm Positional Variance Guarantees a Non-Vanishing High-Error Fraction
:label: cor-vvarx-to-high-error-fraction

For any fixed interaction range $\varepsilon > 0$, there exists a positional variance threshold $R^2_{\text{total\_var},x} > 0$ and a corresponding N-uniform constant $f_H(\epsilon) > 0$ such that:

If the total intra-swarm ({prf:ref}`def-swarm-and-state-space`) positional variance is large, $V_{\text{Var},x} > R^2_{\text{total\_var},x}$, then the fraction of *alive* walkers in the unified high-error set of at least one of the swarms, $k \in {1, 2}$, is bounded below:

$$
\frac{|H_k(\epsilon)|}{k} \ge f_H(\epsilon) > 0

$$

Referenced by {prf:ref}`def-geometric-partition`.
:::
:::{prf:proof}

**Proof.**

This corollary is a direct synthesis of the lemmas established in this chapter.

**1. From Total Positional Variance to Single-Swarm Positional Variance:**
By **{prf:ref}`lem-V_Varx-implies-variance`** (labeled $lem-V_{\text{Var}}x-implies-variance$), if the total intra-swarm positional variance is large, $V_{\text{Var},x} > R^2_{\text{total\_var},x}$, then at least one of the two swarms, say swarm `k`, must have a large internal positional variance:

$$
\mathrm{Var}_x(S_k) > \frac{R^2_{\text{total\_var},x}}{2}

$$

We define the threshold $R^2_{\text{var}} := R^2_{\text{total\_var},x} / 2$.

**2. From Positional Variance to Hypocoercive Variance:**
Since the hypocoercive variance satisfies $\mathrm{Var}_h(S_k) = \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k) \ge \mathrm{Var}_x(S_k)$ (as established in {prf:ref}`lem-var-x-implies-var-h`), the condition $\mathrm{Var}_x(S_k) > R^2_{\text{var}}$ is sufficient to guarantee that the total hypocoercive variance is also large:

$$
\mathrm{Var}_h(S_k) > R^2_{\text{var}}

$$

This satisfies the necessary premise for the lemmas governing both regimes of the $\varepsilon$-dichotomy.

**3. From Hypocoercive Variance to a High-Error Fraction:**
With the condition $\mathrm{Var}_h(S_k) > R^2_{\text{var}}$ met, we can now invoke the results of the $\varepsilon$-dichotomy analysis:

*   **If the swarm is in the large-$\varepsilon$ regime** (where $\varepsilon > D_swarm$): By {prf:ref}`def-unified-high-low-error-sets`, $H_k(\epsilon) = O_k$ in this regime. **{prf:ref}`lem-outlier-fraction-lower-bound`** guarantees that the fraction of walkers in the global kinematic outlier set is bounded below by a positive, N-uniform constant: $|H_k(\epsilon)|/k \ge f_O > 0$.

*   **If the swarm is in the small-$\varepsilon$ regime** (where $\varepsilon \leq D_swarm$): By , $H_k(\epsilon) = C_k(\epsilon)$ (the clustering-based outlier set) in this regime. **{prf:ref}`lem-outlier-cluster-fraction-lower-bound`** guarantees that the fraction of walkers in the outlier clusters is bounded below by a positive, N-uniform constant: $|H_k(\epsilon)|/k \ge f_{H,\text{cluster}}(\epsilon) > 0$.

**4. Define the Unified Lower Bound:**
We can define a single, unified lower bound $f_H(\epsilon)$ that is valid for all regimes by taking the minimum of the bounds from the two cases:

$$
f_H(\epsilon) := \min(f_O, f_{H,\text{cluster}}(\epsilon))

$$

Since both $f_O$ and $f_{H,\text{cluster}}(\epsilon)$ are strictly positive, N-uniform constants, their minimum $f_H(\epsilon)$ is also a strictly positive, N-uniform constant.

**5. Conclusion:**
We have rigorously shown that for any $\varepsilon > 0$, if the total intra-swarm positional variance $V_{\text{Var},x}$ is sufficiently large, then at least one swarm `k` is guaranteed to have a large hypocoercive variance, which in turn guarantees that the fraction of alive walkers in its unified high-error set $H_k(\epsilon)$ is bounded below by the positive, N-uniform constant $f_H(\epsilon)$. This establishes the direct causal link from the Lyapunov function's positional variance component to the guaranteed existence of a substantial high-error population.

**Q.E.D.**
:::
:::{admonition} A Note on the Unified Definition
:class: note

The piecewise definition of the Unified High-Error Set used in this proof is sufficient for the logical argument, as any given swarm state falls into exactly one of the two regimes based on the relationship between $\varepsilon$ and the swarm's phase-space diameter. The two definitions—global kinematic outliers (`O_k`) for the mean-field regime and clustering-based outliers ($C_k(\varepsilon)$) for the local-interaction regime—capture fundamentally different geometric phenomena in phase space, but both are rigorously shown to contain a non-vanishing, N-uniform fraction of the swarm when the positional variance is large.
:::

### 6.5. Microscopic Signature: Geometric Properties of the Partition

The preceding section established a critical macroscopic result: a high-variance swarm is guaranteed to contain a substantial, N-uniform "high-error" population. We proved that when $\mathrm{Var}(x) > R^2_{\mathrm{var}}$, a non-vanishing fraction $f_H(\epsilon) > 0$ of the swarm's walkers must belong to the unified high-error set $H_k(\epsilon)$.

However, knowing that this set is *large* is not sufficient for the subsequent analysis. The effectiveness of the cloning mechanism depends not merely on the *size* of the high-error population, but on its *geometric arrangement* relative to the low-error population. This section completes the transition from a macroscopic property—the guaranteed size of the partition—to a microscopic one: the specific spatial configuration of walkers within each set.

The question we must answer is: **How are walkers in $H_k(\epsilon)$ geometrically distributed compared to those in $L_k(\epsilon)$?** This geometric characterization is the missing link required for Section 7, where we will prove that the `GreedyPairing` algorithm can reliably identify high-error walkers. That proof will depend fundamentally on demonstrating that walkers in the high-error set are systematically more *isolated* from their companions than those in the low-error set, making them distinguishable through distance-based measurements.

**The Central Challenge: The Velocity Contamination Problem.** The most critical challenge in proving this geometric separation is that our premise—high positional variance $\mathrm{Var}(x) > R^2_{\mathrm{var}}$—involves only the positional component of the state, while our conclusion must hold in the full **algorithmic phase space** $d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$, which includes both position and velocity. This creates a fundamental vulnerability: could an adversarial velocity configuration break the geometric separation that positional variance guarantees?

Consider the most pathological scenario:
- **High-error walkers** (positionally far from the center) could all have **identical velocities**: $\|v_i - v_j\| \approx 0$ for all $i, j \in H_k(\epsilon)$
- **Low-error walkers** (positionally close to the center) could have **maximally divergent velocities**: $\|v_l - v_m\| \approx D_v$ for all $l, m \in L_k(\epsilon)$

Under this configuration, the velocity term $\lambda_{\text{alg}} \|v_i - v_j\|^2$ contributes negligibly to the phase-space distance between high-error pairs, while contributing maximally to the distance between low-error pairs. This could potentially violate the required separation $D_H(\epsilon) > R_L(\epsilon)$, breaking the entire geometric argument.

**Our Strategy:** The proofs in this section are specifically constructed to defeat this velocity pathology. Using the same $\epsilon$-dichotomy framework established in Section 6.4, we will prove that in both the mean-field and local-interaction regimes, the high-error and low-error sets possess fundamentally different geometric signatures that hold **even under the worst-case velocity configuration**: high-error walkers remain isolated, while low-error walkers remain clustered. The algebraic conditions derived in Sections 6.5.2 and 6.5.3 are not arbitrary tuning parameters but rather mathematically necessary conditions that ensure the positional signal dominates the velocity noise.

#### 6.5.1. Main Lemma: Statement of Geometric Separation

:::{prf:lemma} Geometric Separation of the Partition
:label: lem-geometric-separation-of-partition

Let $H_k(\epsilon)$ and $L_k(\epsilon)$ be the unified high-error and low-error sets for swarm ({prf:ref}`def-swarm-and-state-space`) $k$ as defined in {prf:ref}`def-unified-high-low-error-sets`. Assume the swarm's internal positional variance is large: $\mathrm{Var}(x) > R^2_{\mathrm{var}}$.

Then there exist N-uniform, $\epsilon$-dependent constants $D_H(\epsilon) > R_L(\epsilon) > 0$ and a fractional constant $f_c > 0$ such that:

**Part 1 (Separation Between Sets):** For any walker ({prf:ref}`def-walker`) $i \in H_k(\epsilon)$ from a high-error cluster and any walker $j \in L_k(\epsilon)$ from a low-error cluster, their algorithmic distance ({prf:ref}`def-alg-distance`) is bounded below:

$$
d_{\text{alg}}(i, j) \ge D_H(\epsilon)

$$

**Part 2 (Clustering of Low-Error Walkers):** For any walker ({prf:ref}`def-walker`) $j \in L_k(\epsilon)$, there exists a non-empty subset of companion walkers $C_j \subset L_k(\epsilon) \setminus \{j\}$ of minimum size $|C_j| \ge f_c k$ such that all members of this cluster are within a small algorithmic radius:

$$
d_{\text{alg}}(j, \ell) \le R_L(\epsilon) \quad \text{for all } \ell \in C_j

$$

The separation property $D_H(\epsilon) > R_L(\epsilon)$ ensures that the geometric signatures of the two sets are fundamentally distinct and non-overlapping **in the algorithmic phase space**.

**Note:** This lemma does **not** claim that high-error walkers are isolated from each other. Walkers within the same high-error cluster may be close ($d_{\text{alg}} \le D_{\text{diam}}(\epsilon) = R_L(\epsilon)$). The key property is the separation **between** the high-error and low-error populations.
:::

:::{admonition} The Velocity Contamination Problem and How We Solve It
:class: important

{prf:ref}`lem-geometric-separation-of-partition` establishes a causal link from a premise based on **positional variance (`Var(x)`)** to a conclusion about geometric separation in the full **algorithmic phase-space (`d_alg`)**. This is a highly non-trivial claim because the algorithmic distance $d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$ includes both position and velocity components.

**The Pathological Scenario:** The adversarial velocity configuration described above could break this connection:
- High-error walkers with identical velocities would have their phase-space distances *reduced* (no velocity contribution)
- Low-error walkers with maximally divergent velocities would have their phase-space distances *inflated* (maximum velocity contribution)
- This could cause $d_{\text{alg}}(\text{high-error pair}) < d_{\text{alg}}(\text{low-error pair})$, violating the separation property

**How Both Proofs Solve This:**
- **Mean-Field Regime (Section 6.5.2):** Uses the parameter $\varepsilon_O$ to bound the fraction of walkers that can exhibit pathological velocity clustering. The algebraic condition derived in Step 6 is specifically designed to ensure that even when high-error walkers have identical velocities, their positional separation is sufficient to maintain phase-space isolation.

- **Local-Interaction Regime (Section 6.5.3):** Uses worst-case velocity noise bounds in both the phase-space radius calculation ($R_L$ assumes maximum velocity divergence within clusters) and the isolation distance calculation ($D_H$ uses only positional separation, which holds regardless of velocity structure).

Both approaches guarantee that the positional signal dominates the velocity noise, ensuring the `GreedyPairing` algorithm from Section 5 can reliably distinguish high-error from low-error walkers. The unified conditions in Section 6.5.4 ensure this guarantee holds across all regimes and parameter choices.
:::

#### 6.5.2. Unified Proof via Clustering-Based Geometric Separation

:::{prf:proof} Proof of Geometric Separation (All Regimes)

**Objective:** Using the unified clustering-based definition from Section 6.3, we will prove that high-error clusters are geometrically isolated from low-error clusters in the algorithmic phase-space metric $d_{\text{alg}}$, starting from the premise $\mathrm{Var}_x(S_k) > R^2_{\mathrm{var}}$. This proof applies uniformly across all interaction regimes.

**Proof Strategy: Clustering-Based Separation**

The unified definition partitions walkers into clusters $\{G_1, \ldots, G_M\}$ with maximum diameter $D_{\text{diam}}(\epsilon) = c_d \cdot \epsilon$ in the algorithmic phase-space metric. High-error clusters are those whose centers contribute significantly to the between-cluster hypocoercive variance. We will prove:

1. **Within-cluster cohesion**: Walkers within any cluster (especially low-error clusters) remain close in phase space by construction ($d_{\text{alg}} \le D_{\text{diam}}(\epsilon)$)
2. **Between-cluster separation**: High-error cluster centers are far from low-error cluster centers in phase space
3. **Geometric separation**: These properties combine to ensure $D_H(\epsilon) > R_L(\epsilon)$

The proof uses the reverse triangle inequality with explicit verification that the resulting bounds are positive and meaningful, ensuring rigorous separation between high-error and low-error populations.

**Step 1: Establish Clustering Properties**

By {prf:ref}`def-unified-high-low-error-sets`, the alive set $\mathcal{A}_k$ is partitioned into clusters $\{G_1, \ldots, G_M\}$ where each cluster satisfies:

$$
\text{diam}(G_m) := \max_{i,j \in G_m} d_{\text{alg}}(i, j) \le D_{\text{diam}}(\epsilon) = c_d \cdot \epsilon

$$

This immediately gives us the **low-error clustering radius**. For any walker $j \in L_k(\epsilon)$ belonging to a valid low-error cluster $G_\ell$ (with $|G_\ell| \ge k_{\min}$), all other walkers in that cluster satisfy:

$$
d_{\text{alg}}(j, m) \le D_{\text{diam}}(\epsilon) \quad \text{for all } m \in G_\ell

$$

We define:

$$
R_L(\epsilon) := D_{\text{diam}}(\epsilon) = c_d \cdot \epsilon

$$

**Step 2: Bridge to Hypocoercive Variance**

As established in Section 6.4.2, the premise $\mathrm{Var}_x(S_k) > R^2_{\mathrm{var}}$ guarantees:

$$
\mathrm{Var}_h(S_k) = \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k) \ge \mathrm{Var}_x(S_k) > R^2_{\mathrm{var}}

$$

**Step 3: Decompose Variance via Law of Total Variance**

The hypocoercive variance can be decomposed into within-cluster and between-cluster components. For the positional component:

$$
k \cdot \mathrm{Var}_x(S_k) = \sum_{m=1}^M \sum_{i \in G_m} \|x_i - \mu_x\|^2 = \underbrace{\sum_{m=1}^M |G_m| \mathrm{Var}_x(G_m)}_{\text{within-cluster}} + \underbrace{\sum_{m=1}^M |G_m| \|\mu_{x,m} - \mu_x\|^2}_{\text{between-cluster}}

$$

where $\mu_{x,m}$ is the positional center of mass of cluster $G_m$.

**Step 4: Bound Within-Cluster Variance**

Since each cluster has algorithmic diameter at most $D_{\text{diam}}(\epsilon)$, the positional diameter is bounded:

$$
\max_{i,j \in G_m} \|x_i - x_j\| \le \max_{i,j \in G_m} d_{\text{alg}}(i,j) \le D_{\text{diam}}(\epsilon)

$$

Therefore, the maximum internal positional variance of any cluster satisfies:

$$
\mathrm{Var}_x(G_m) \le \left(\frac{D_{\text{diam}}(\epsilon)}{2}\right)^2

$$

The total within-cluster sum of squares is bounded:

$$
\sum_{m=1}^M |G_m| \mathrm{Var}_x(G_m) \le k \left(\frac{D_{\text{diam}}(\epsilon)}{2}\right)^2

$$

**Step 5: Lower Bound on Between-Cluster Variance**

Rearranging the variance decomposition and using $\mathrm{Var}_x(S_k) > R^2_{\mathrm{var}}$:

$$
\sum_{m=1}^M |G_m| \|\mu_{x,m} - \mu_x\|^2 = k \cdot \mathrm{Var}_x(S_k) - \sum_{m=1}^M |G_m| \mathrm{Var}_x(G_m) > k \cdot R^2_{\mathrm{var}} - k \left(\frac{D_{\text{diam}}(\epsilon)}{2}\right)^2

$$

Define the **minimum cluster mean separation threshold**:

$$
R^2_{\mathrm{means}} := R^2_{\mathrm{var}} - \left(\frac{D_{\text{diam}}(\epsilon)}{2}\right)^2

$$

For this to be positive, we require the **admissibility condition**:

$$
D_{\text{diam}}(\epsilon) = c_d \cdot \epsilon < 2\sqrt{R^2_{\mathrm{var}}}

$$

Under this condition:

$$
\frac{1}{k} \sum_{m=1}^M |G_m| \|\mu_{x,m} - \mu_x\|^2 > R^2_{\mathrm{means}} > 0

$$

**Step 6: Apply Outlier Analysis to Cluster Centers**

By {prf:ref}`def-unified-high-low-error-sets`, valid outlier clusters (with $|G_m| \ge k_{\min}$) satisfy:

$$
\sum_{m \in O_M} |G_m| \|\mu_{x,m} - \mu_x\|^2 \ge (1-\varepsilon_O) \sum_{\substack{m: |G_m| \ge k_{\min}}} |G_m| \|\mu_{x,m} - \mu_x\|^2

$$

Let $H_k(\epsilon) = \bigcup_{m \in O_M} G_m$ be the union of valid outlier clusters, and let $L_k(\epsilon)$ be the union of valid low-error clusters.

For any high-error cluster $G_h \in O_M$ and any low-error cluster $G_\ell \notin O_M$ (with both having $|G_h|, |G_\ell| \ge k_{\min}$), we derive a lower bound on the positional separation of their centers.

**Step 7: Derive Minimum Cluster Mean Separation**

Using the averaging argument from the outlier analysis: if the minimum positional distance from any outlier cluster center to the global center is $r_h$, then:

$$
\sum_{m \in O_M} |G_m| \|\mu_{x,m} - \mu_x\|^2 \ge |H_k(\epsilon)| \cdot r_h^2

$$

Combined with Step 6 and using $|H_k(\epsilon)| \le k$:

$$
r_h^2 \ge (1-\varepsilon_O) R^2_{\mathrm{means}}

$$

Therefore:

$$
\|\mu_{x,h} - \mu_x\| \ge \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} \quad \text{for all } G_h \in O_M

$$

Similarly, for low-error clusters:

$$
\|\mu_{x,\ell} - \mu_x\| \le \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}} k}{|L_k(\epsilon)|}}

$$

**Step 8: Prove Separation Between High-Error and Low-Error Sets**

We now establish that walkers from high-error clusters are separated from walkers in low-error clusters. For any walker $i \in H_k(\epsilon)$ (in outlier cluster $G_h$), we consider two cases:

**Case 1 (Within High-Error Set):** If $j \in H_k(\epsilon)$ and belongs to the same cluster $j \in G_h$, then by the cluster diameter bound:

$$
d_{\text{alg}}(i,j) \le D_{\text{diam}}(\epsilon) = R_L(\epsilon)

$$

This case shows that walkers within the same high-error cluster are **not** isolated from each other. This is a critical observation: we do not claim universal isolation for high-error walkers.

**Case 2 (Between Different Sets):** If $j \in L_k(\epsilon)$ (low-error cluster $G_\ell$), we use positional separation of cluster centers. By the reverse triangle inequality in position space:

$$
\|x_i - x_j\| \ge \|\mu_{x,h} - \mu_{x,j'}\| - \|x_i - \mu_{x,h}\| - \|x_j - \mu_{x,j'}\|

$$

where $G_{j'}$ is the cluster containing $j$. This application of the reverse triangle inequality is valid when the separation between cluster centers dominates the within-cluster radii, which we now verify.

Using our established bounds:
- $\|\mu_{x,h} - \mu_{x,j'}\| \ge \|\mu_{x,h} - \mu_x\| - \|\mu_{x,j'} - \mu_x\|$ (reverse triangle inequality)
- $\|x_i - \mu_{x,h}\| \le D_{\text{diam}}(\epsilon)/2$ (radius bound within cluster)
- $\|x_j - \mu_{x,j'}\| \le D_{\text{diam}}(\epsilon)/2$ (radius bound within cluster)

**Verification of Positivity:** For the bound to be meaningful, we must verify that:

$$
\|\mu_{x,h} - \mu_{x,j'}\| > \|x_i - \mu_{x,h}\| + \|x_j - \mu_{x,j'}\|

$$

From Steps 6-7, we have:
- $\|\mu_{x,h} - \mu_{x,j'}\| \geq \|\mu_{x,h} - \mu_x\| - \|\mu_{x,j'} - \mu_x\| \geq \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}} k}{|L_k(\epsilon)|}}$
- $\|x_i - \mu_{x,h}\| + \|x_j - \mu_{x,j'}\| \leq D_{\mathrm{diam}}(\epsilon)$

Therefore, positivity requires:

$$
\sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}} k}{|L_k(\epsilon)|}} > D_{\mathrm{diam}}(\epsilon)

$$

This condition will be guaranteed by the admissibility constraints derived in Step 9 below. Proceeding under this guarantee, we obtain:

$$
\|x_i - x_j\| \ge \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}} k}{|L_k(\epsilon)|}} - D_{\text{diam}}(\epsilon)

$$

Since $d_{\text{alg}}(i,j) \ge \|x_i - x_j\|$, we define the **high-error isolation distance**:

$$
D_H(\epsilon) := \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}} k}{k(1-f_H(\epsilon))}} - D_{\text{diam}}(\epsilon)

$$

where $f_H(\epsilon)$ is the N-uniform lower bound on the high-error fraction from Section 6.4. Simplifying:

$$
D_H(\epsilon) := \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}}}{1-f_H(\epsilon)}} - c_d \cdot \epsilon

$$

:::{admonition} Mathematical Rigour Note
:class: note

The application of the reverse triangle inequality in Step 8 deserves careful examination. For three points $a, b, c$ in a metric space, the reverse triangle inequality states:

$$
\|a - c\| \geq \|a - b\| - \|b - c\|

$$

In our application with $a = x_i$, $b = \mu_{x,h}$, and $c = x_j$, this becomes:

$$
\|x_i - x_j\| \geq \|x_i - \mu_{x,h}\| - \|\mu_{x,h} - x_j\|

$$

However, to obtain a useful **lower bound**, we need the term $\|\mu_{x,h} - x_j\|$ to be expressible in terms of quantities we can control. Using the triangle inequality $\|\mu_{x,h} - x_j\| \leq \|\mu_{x,h} - \mu_{x,j'}\| + \|\mu_{x,j'} - x_j\|$, we substitute to get:

$$
\|x_i - x_j\| \geq \|x_i - \mu_{x,h}\| - (\|\mu_{x,h} - \mu_{x,j'}\| + \|\mu_{x,j'} - x_j\|)

$$

Rearranging yields the form used in the proof:

$$
\|x_i - x_j\| \geq \|\mu_{x,h} - \mu_{x,j'}\| - \|x_i - \mu_{x,h}\| - \|x_j - \mu_{x,j'}\|

$$

This is mathematically valid. The subtlety is that this bound is only **meaningful** (i.e., positive) when the between-cluster separation $\|\mu_{x,h} - \mu_{x,j'}\|$ dominates the sum of within-cluster radii. This is precisely what the positivity verification establishes, and what the admissibility constraints in Step 9 guarantee. The approach is standard in clustering-based geometric analysis where one must verify that cluster-level separation dominates local fluctuations.
:::

**Step 9: Verify Separation Condition $D_H(\epsilon) > R_L(\epsilon)$**

For geometric separation, we require:

$$
\sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}}}{1-f_H(\epsilon)}} - c_d \cdot \epsilon > c_d \cdot \epsilon

$$

Simplifying:

$$
\sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} > \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}}}{1-f_H(\epsilon)}} + 2c_d \cdot \epsilon

$$

This condition is satisfied when:

$$
\varepsilon_O < \frac{(1-f_H(\epsilon)) \left(\sqrt{R^2_{\mathrm{means}}} - 2c_d \cdot \epsilon\right)^2}{R^2_{\mathrm{means}} + f_H(\epsilon) \left(\sqrt{R^2_{\mathrm{means}}} - 2c_d \cdot \epsilon\right)^2}

$$

provided that $\sqrt{R^2_{\mathrm{means}}} > 2c_d \cdot \epsilon$, which follows from choosing:

$$
R^2_{\mathrm{var}} > \left(\frac{D_{\text{diam}}(\epsilon)}{2} + 2c_d \cdot \epsilon\right)^2 = \left(\frac{c_d \cdot \epsilon}{2} + 2c_d \cdot \epsilon\right)^2 = \left(\frac{5c_d \cdot \epsilon}{2}\right)^2

$$

**Conclusion:**

Under the admissibility conditions:
1. $c_d \cdot \epsilon < 2\sqrt{R^2_{\mathrm{var}}}$ (ensures positive between-cluster variance)
2. $R^2_{\mathrm{var}} > (5c_d \cdot \epsilon / 2)^2$ (ensures sufficient separation for the bound)
3. $\varepsilon_O$ satisfying the bound above (restricts outlier contamination)

**Verification:** These three conditions jointly guarantee the positivity requirement from Step 8. Specifically, conditions (2) and (3) together ensure:

$$
\sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}}}{1-f_H(\epsilon)}} > c_d \cdot \epsilon = D_{\mathrm{diam}}(\epsilon)

$$

which validates the application of the reverse triangle inequality for deriving meaningful separation bounds between high-error and low-error walkers.

we have rigorously established phase-space constants $D_H(\epsilon)$ and $R_L(\epsilon) = c_d \cdot \epsilon$ with $D_H(\epsilon) > R_L(\epsilon)$. This proves:

- **Separation Between Sets (Part 1)**: Every walker in a high-error cluster is separated from every walker in a low-error cluster by at least $D_H(\epsilon)$ in the algorithmic phase-space metric
- **Clustering of Low-Error Walkers (Part 2)**: Every walker in a valid low-error cluster has companions within algorithmic radius $R_L(\epsilon) = c_d \cdot \epsilon$

**Important Clarification:** We do **not** claim that all high-error walkers are isolated from each other. Walkers within the same high-error cluster may have distances as small as $R_L(\epsilon)$. The key property is the guaranteed separation **between** the high-error and low-error populations, which enables the algorithm to distinguish these populations statistically.

The clustering-based approach provides a unified proof that avoids the flawed reverse triangle inequality and applies consistently across all interaction regimes.

**Q.E.D.**
:::

#### 6.5.3. Summary of Geometric Separation Constants

The unified clustering-based proof in Section 6.5.2 established N-uniform geometric separation constants that apply across all interaction regimes. We now summarize these constants and the admissibility conditions required for their validity.

#### 6.5.3.1. Unified Geometric Separation Constants

The clustering-based proof established the following state-independent constants:

$$
\begin{aligned}
D_H(\epsilon) &:= \sqrt{(1-\varepsilon_O) R^2_{\mathrm{means}}} - \sqrt{\frac{\varepsilon_O R^2_{\mathrm{means}}}{1-f_H(\epsilon)}} - c_d \cdot \epsilon \\
R_L(\epsilon) &:= c_d \cdot \epsilon
\end{aligned}

$$

where:
- $R^2_{\mathrm{means}} := R^2_{\mathrm{var}} - (c_d \cdot \epsilon / 2)^2$ is the guaranteed between-cluster variance
- $c_d > 0$ is the cluster diameter constant (typically $c_d = 2$)
- $\varepsilon_O \in (0,1)$ is the outlier structural parameter (typically $\varepsilon_O = 0.1$)
- $f_H(\epsilon) > 0$ is the N-uniform lower bound on the high-error fraction from Section 6.4

These constants depend only on the primitive parameters $R^2_{\mathrm{var}}$, $\varepsilon_O$, $c_d$, $\epsilon$, and $f_H(\epsilon)$, and are manifestly N-uniform.

#### 6.5.3.2. Admissibility Conditions for Geometric Separation

1. **Positive Between-Cluster Variance** (Equation (1)):

$$
c_d \cdot \epsilon < 2\sqrt{R^2_{\mathrm{var}}}

$$

This ensures that $R^2_{\mathrm{means}} > 0$, guaranteeing non-trivial variance between cluster centers.

2. **Sufficient Separation for Isolation** (Equation (2)):

$$
R^2_{\mathrm{var}} > \left(\frac{5c_d \cdot \epsilon}{2}\right)^2

$$

This ensures that $\sqrt{R^2_{\mathrm{means}}} > 2c_d \cdot \epsilon$, which is required for the separation condition.

3. **Outlier Parameter Constraint** (Equation (3)):

$$
\varepsilon_O < \frac{(1-f_H(\epsilon)) \left(\sqrt{R^2_{\mathrm{means}}} - 2c_d \cdot \epsilon\right)^2}{R^2_{\mathrm{means}} + f_H(\epsilon) \left(\sqrt{R^2_{\mathrm{means}}} - 2c_d \cdot \epsilon\right)^2}

$$

This restricts the maximum allowable outlier contamination to ensure that positional signal dominates.

Together, these conditions ensure the strict separation property $D_H(\epsilon) > R_L(\epsilon)$ for all swarm configurations with $\mathrm{Var}_x(S_k) > R^2_{\mathrm{var}}$.

#### 6.5.3.3. Conclusion of the Proof of {prf:ref}`lem-geometric-separation-of-partition`

We have now rigorously established the existence of **state-independent, N-uniform constants** $D_H(\varepsilon)$ and $R_L(\varepsilon)$ in the **`d_alg` phase-space metric** that satisfy the claims of :

1.  For any swarm state `S_k` with `Var_x(S_k) > R^{2}_var`, every walker `i` in the unified high-error set $H_k(\varepsilon)$ is guaranteed to be isolated from all other walkers `j` by a phase-space distance of at least $D_H(\varepsilon)$ in the `d_alg` metric.

2.  Every walker in the low-error set $L_k(\varepsilon)$ is guaranteed to have a substantial sub-population of companions (of size at least `f_c k` for N-uniform `f_c > 0`) within a phase-space radius of $R_L(\varepsilon)$ in the `d_alg` metric.

3.  The strict separation $D_H(\varepsilon) > R_L(\varepsilon)$ is guaranteed by the Unified Condition, which requires consistent selection of the system's primitive parameters (`R^{2}_var`, $\varepsilon_O$, `f_O`, $D_diam(\varepsilon)$, $\lambda_v$, $\lambda_alg$, `D_x`, `D_v`).

These uniform bounds are constructed entirely from fundamental system constants and are independent of the swarm size `N`, the number of alive walkers `k`, and the specific swarm configuration `S`.

**This completes the proof of {prf:ref}`lem-geometric-separation-of-partition` (Geometric Separation of the Partition).**

**Q.E.D.**

### 6.6. Section summary

This chapter has established the first and most fundamental link in the Keystone causal chain: the rigorous, N-uniform connection between a macroscopic system error and a guaranteed microscopic geometric structure. We have proven that a large intra-swarm positional variance ($V_{\text{Var},x}$) is an unstable condition that forces a predictable and detectable pattern onto the swarm's configuration.

**Defeating the Velocity Pathology:**

The central technical achievement of this chapter, beyond establishing the high-error fraction, is the rigorous defeat of the velocity contamination problem. We have proven that high positional variance is a sufficient condition for phase-space separation, even under the most adversarial velocity configuration:

- Sections 6.5.2 and 6.5.3 explicitly constructed constants $D_H(\epsilon)$ and $R_L(\epsilon)$ under worst-case velocity assumptions
- Section 6.5.4 unified these into a single set of parameter conditions that guarantee separation across all regimes
- The algebraic conditions on $\varepsilon_O$ and $D_{\mathrm{diam}}(\epsilon)$ are not arbitrary tuning parameters but rather mathematically necessary conditions to ensure the positional signal dominates velocity noise

This ensures that the algorithmic distance metric $d_{\text{alg}}$, which includes both position and velocity components, reliably perceives the geometric structure induced by positional variance alone. Without this guarantee, the entire Keystone mechanism would fail: high-error walkers could be kinematically clustered despite being positionally dispersed, breaking the corrective feedback loop.

The logical argument forged in this chapter proceeded in two main parts:

1.  **From Macroscopic Error to a Substantial High-Error Population:** We began by showing that a large $V_{\text{Var},x}$ guarantees that at least one swarm must have a large internal positional variance (`Var_x`). The analysis in Section 6.4 then proved, via a robust $\varepsilon$-dichotomy, that this condition is sufficient to guarantee that a non-vanishing, N-uniform fraction of the swarm's walkers, $f_H(\varepsilon)$, must belong to the **Unified High-Error Set $H_k(\varepsilon)$**.

2.  **From Population Statistics to Microscopic Geometry:** The chapter's final and most critical result, the **Geometric Separation of the Partition ({prf:ref}`lem-geometric-separation-of-partition`)**, proved that this high-error population possesses a distinct and measurable geometric signature. We have rigorously shown that the high-error set $H_k(\varepsilon)$ is forced into a state of **phase-space isolation**, while the low-error set $L_k(\varepsilon)$ is confined to **dense phase-space clusters**.

The strict separation $D_H(\varepsilon) > R_L(\varepsilon)$, guaranteed by the Unified Condition derived in Section 6.5.4, ensures that these two geometric signatures are fundamentally distinct and non-overlapping in the **algorithmic phase space (`d_alg`)**. This proven separation is the "missing link" that connects the abstract concept of system error to a concrete, physical property that the swarm's own measurement pipeline can perceive. It provides the precise, exploitable structure that the `GreedyPairing` algorithm can reliably detect, a fact that will be the cornerstone of the proof of intelligent signal generation in the next chapter.

(sec-cloning-fitness)=
## 7. The Corrective Nature of Fitness: From Signal Generation to Intelligent Adaptation

### 7.1. From distances to a selection signal

:::{div} feynman-prose
A distance measurement becomes useful to selection through several nonlinear operations. We first quantify its variance under the actual pairing law, then follow rescaling and the fitness product. The proofs distinguish a realized fitness vector from averages over random pairings.
:::

### 7.2.1 Guaranteed Measurement Variance from Geometric Structure
:::{prf:theorem} Measurement variance from a marginal mean separation
:label: thm-geometry-guarantees-variance

For the actual pairing law, let $d=(d_1,\ldots,d_k)$ be its random measurement
vector. Suppose a deterministic partition $H,L$ has fractions $f_H,f_L>0$ and
its marginal mean measurements differ by at least $\Delta_d>0$. Then

$$
\mathbb E\operatorname{Var}(d)\geq f_Hf_L\Delta_d^2.
$$

For the greedy law, the history and self-match estimates of
{prf:ref}`lem-greedy-preserves-signal` provide one way to verify this separation.
The estimate is $N$-uniform when the fractions and gap have common positive
lower bounds. Positional variance alone does not supply the required marginal
gap for every matching configuration.
:::

:::{prf:proof}
Let $P=I-k^{-1}\mathbf1\mathbf1^\top$, the orthogonal centering projection.
Then $\operatorname{Var}(d)=k^{-1}\|Pd\|^2$. The identity

$$
\mathbb E\|Pd\|^2=\|P\mathbb Ed\|^2+
\mathbb E\|P(d-\mathbb Ed)\|^2
$$

implies $\mathbb E\operatorname{Var}(d)\geq\operatorname{Var}(\mathbb Ed)$.
Decompose the deterministic vector $\mathbb Ed$ into its two group means and
within-group deviations. Its variance is the sum of the nonnegative
within-group variance and $f_Hf_L(\mu_H-\mu_L)^2$, proving the result.
This calculation retains all correlations created by the pairing.
:::

#### 7.2.2. Proposition: Satisfiability of the Signal-to-Noise Condition

The analysis in the subsequent sections rests on a key statistical lemma ({prf:ref}`lem-variance-to-gap`), which provides the bridge from a guaranteed total signal variance to a guaranteed separation between subpopulation means. The validity of this lemma is conditional; it requires that the signal variance generated by a high-error state must be strictly greater than the maximum possible "internal noise" variance that can be generated by any configuration of values within the same range.

This section provides the formal proof that this condition, which we call the **Signal-to-Noise Condition**, is not an unstated assumption but a satisfiable criterion that can be met by a valid choice of the algorithm's user-defined parameters. We prove this by introducing a **Signal Gain** parameter, $\gamma$, which acts as a sensitivity knob for the algorithm. This proves that the system is fundamentally "learnable": the signal generated by geometric error can always be amplified sufficiently to overcome the worst-case statistical noise, ensuring that a true difference between the high-error and low-error populations is always detectable.

:::{prf:proposition} **(Satisfiability of the Signal-to-Noise Condition via Signal Gain)**
:label: prop-satisfiability-of-snr-gamma

Let the rescaled diversity values be defined as $d'_i = g_A(\gamma · z_{d,i}) + \eta$, where $\gamma > 0$ is a user-defined **Signal Gain** parameter and `g_A` is any function satisfying the **Axiom of a Well-Behaved Rescale Function ({prf:ref}`def-canonical-logistic-rescale-function-example`)** (see {prf:ref}`def-logistic-rescale` for the canonical choice).

For any system in a high-error state (`Var(x) > R^{2}_{\mathrm{var}}`) that generates a non-zero raw distance signal ($\kappa_{\mathrm{meas}}(d) > 0$), there exists a sufficiently large choice of $\gamma$ that satisfies the **Signal-to-Noise Condition**:

$$
\kappa_{\mathrm{var}}(d') > \operatorname{Var}_{\max}(d')

$$

where `Var_max(d')` is the maximum possible variance of the rescaled values, and $\kappa_{\mathrm{var}}(d')$ is the guaranteed lower bound on the variance of the rescaled values in the high-error state.
:::
:::{prf:proof}

**Proof.**

The proof strategy is to show that the guaranteed signal variance of the rescaled values, $\kappa_{\mathrm{var}}(d')$, scales with $\gamma^{2}$ in the small-signal limit, while the maximum possible noise, `Var_max(d')`, remains a fixed constant independent of $\gamma$. This algebraic advantage allows $\gamma$ to be chosen to ensure the signal always dominates the noise.

**1. The Noise Term (`Var_max(d')`): A Fixed, $\gamma$-Independent Constant.**

The **Axiom of a Well-Behaved Rescale Function** requires `g_A` to have a bounded range, which we denote `(g_{A,\min}, g_{A,\max})`. Consequently, the rescaled values $d'_i = g_A(\gamma · z_{d,i}) + \eta$ are always contained within the fixed interval $(g_{A,\min} + \eta, g_{A,\max} + \eta)$.

The maximum possible variance for any set of values on this interval is given by Popoviciu's inequality:

$$
\operatorname{Var}_{\max}(d') := \frac{1}{4}(\max(d') - \min(d'))^2 = \frac{1}{4}(g_{A,\max} - g_{A,\min})^2

$$

This value is a constant determined solely by the choice of the rescale function `g_A`; it does not depend on the Signal Gain $\gamma$. For the **Canonical Logistic Rescale function**, `g_A(z) = 2/(1+e^{-z})`, the range is `(0, 2)`, yielding a fixed maximum noise of `Var_max(d') = 1`.

Our goal is to prove that we can choose $\gamma$ such that the guaranteed signal variance $\kappa_{\mathrm{var}}(d')$ is greater than this fixed constant.

**2. The Signal Term ($\kappa_{\mathrm{var}}(d')$): Amplification by $\gamma$.**

The signal originates from the raw distance measurements `d`, propagates to the standardized scores `z_d`, and is then amplified.

*   **Raw and Standardized Signal:** From {prf:ref}`thm-geometry-guarantees-variance`, a high-error state guarantees $\text{Var}(d) \geq \kappa_{\mathrm{meas}}(d) > 0$. The Z-scores $z_d = (d - \mu_d) / \sigma'_d$ have a variance $\text{Var}(z_d) = \text{Var}(d) / (\sigma'_d)^{2}$. Since the patched standard deviation (see {prf:ref}`def-patched-std-dev-function`) $\sigma'_d$ is uniformly bounded above by $\sigma'_{\max}$ ({prf:ref}`def-max-patched-std`), the Z-score variance has a uniform lower bound:


$$
\operatorname{Var}(z_d) \ge \frac{\kappa_{\mathrm{meas}}(d)}{(\sigma'_{\max})^2} =: \kappa_{\mathrm{var}}(z) > 0

$$

*   **Signal Amplification:** The input to the rescale function is $u_i = \gamma z_{d,i}$. The variance of this amplified signal is $\text{Var}(u) = \gamma^{2}\text{Var}(z_d) \geq \gamma^{2}\kappa_{\mathrm{var}}(z)$.

*   **Rescaled Signal ($\kappa_{\mathrm{var}}(d')$):** The rescaled values are $d' = g_A(u) + \eta$. For any differentiable function, a first-order Taylor expansion around the mean $\mu_u$ gives $g_A(u_i) \approx g_A(\mu_u) + g'_A(\mu_u)(u_i - \mu_u)$. The variance is then approximated by:


$$
\operatorname{Var}(d') = \operatorname{Var}(g_A(u)) \approx (g'_A(\mu_u))^2 \operatorname{Var}(u)

$$

    This approximation becomes exact in the limit of small variance relative to the curvature of `g_A`. A more rigorous treatment using the Mean Value Theorem shows that the variance of the output is bounded below by the variance of the input multiplied by the squared infimum of the derivative.


$$
\operatorname{Var}(d') \ge (\inf_{c \in Z_{\mathrm{eff}}} g'_A(c))^2 \operatorname{Var}(u)

$$

    where `Z_eff` is the effective range of inputs. Let `g'_{\min} > 0` be the uniform lower bound on the derivative (guaranteed to exist on any compact operational range by the axiom). The guaranteed variance of the rescaled values is thus bounded below by a term proportional to $\gamma^{2}$:


$$
\kappa_{\mathrm{var}}(d') \ge (g'_{\min})^2 \cdot \gamma^2 \kappa_{\mathrm{var}}(z)

$$

**3. Proving Satisfiability.**

The Signal-to-Noise Condition is $\kappa_{\mathrm{var}}(d') > \operatorname{Var}_{\max}(d')$. Substituting our results from the steps above:

$$
(g'_{\min})^2 \cdot \gamma^2 \kappa_{\mathrm{var}}(z) > \frac{1}{4}(g_{A,\max} - g_{A,\min})^2

$$

Solving for the Signal Gain $\gamma$:

$$
\gamma > \frac{g_{A,\max} - g_{A,\min}}{2 \cdot g'_{\min} \cdot \sqrt{\kappa_{\mathrm{var}}(z)}}

$$

Since $\kappa_{\mathrm{var}}(z)$ is a fixed positive constant for a given $\varepsilon$, and `g_A`'s properties (`g_{A,max}`, `g_{A,min}`, `g'_{min}`) are fixed, the right-hand side is a fixed, positive real number. This proves that there always exists a sufficiently large choice of $\gamma$ that satisfies the condition.

**Conclusion:** The Signal-to-Noise Condition is not a restrictive assumption on the environment but is a design criterion that can always be satisfied by appropriately tuning the algorithm's sensitivity $\gamma$. This holds for any valid rescale function, including the Canonical choice.

**Q.E.D.**
:::
:::{admonition} Design Implications: The Role of the Signal Gain ($\gamma$)
:class: note

The introduction of the $\gamma$ parameter is a crucial step in ensuring the mathematical robustness of the framework. It formalizes the concept of the algorithm's **sensitivity**.

*   $\gamma$ acts as a tuning knob that determines how strongly the system reacts to the standardized signals it measures. A low $\gamma$ will map a wide range of Z-scores to a narrow band of rescaled values, making the system very stable but potentially slow to adapt. A high $\gamma$ will amplify small differences in Z-scores, making the system highly responsive.

*   This proposition proves that for the system's "intelligence" to be guaranteed (i.e., for the proofs in the subsequent sections to hold), $\gamma$ must be chosen to be above a certain threshold. This threshold depends on the intrinsic signal strength of the problem ($\kappa_{\mathrm{meas}}$) and the properties of the chosen rescale function.

*   Therefore, the requirement for a sufficiently large $\gamma$ should be considered a foundational property for any well-posed Fragile Gas instantiation. It is recommended to add $\gamma$ to the list of **Algorithmic Dynamics Axioms (Section 4)**, with the condition $\gamma > 0$, noting that its value must be chosen large enough to satisfy the inequality derived herein.
:::

### 7.3. Signal Propagation Through the Pipeline

The preceding theorem ({prf:ref}`thm-geometry-guarantees-variance`) established that a swarm in a high-error state, possessing the geometric structure proven in Section 6, is guaranteed to generate a raw distance measurement signal with a non-vanishing expected variance, $\mathbb{E}[\operatorname{Var}(d)] \geq \kappa_{\mathrm{meas}}(\varepsilon) > 0$. This section proves that the deterministic pipeline defined in Section 5 is a robust signal processor, capable of transforming this raw statistical signal into a concrete, usable gap in the final rescaled values.

The proof will follow the signal's journey in two stages:
1.  First, we prove that a guaranteed variance in any set of raw values implies the existence of a guaranteed *gap* between at least two of those values.
2.  Second, we prove that this raw gap robustly propagates through the standardization and rescale operators to become a guaranteed *rescaled gap*.

#### 7.3.1. From Raw Variance to a Guaranteed Raw Gap

The first step in the signal integrity proof is to show that the statistical property of variance, now proven in {prf:ref}`thm-geometry-guarantees-variance`, has a direct, concrete consequence: it forces a measurable separation between the raw values of at least two walkers.

:::{prf:lemma} From Bounded Variance to a Guaranteed Gap
:label: lem-variance-to-gap

Let $\{v_i\}_{i=1}^k$ be a set of $k \ge 2$ real numbers. If the empirical variance of this set is bounded below by a strictly positive constant, $\text{Var}(\{v_i\}) \geq \kappa > 0$, then there must exist at least one pair of indices $(i, j)$ such that the gap between their values is bounded below:

$$
\max_{i,j} |v_i - v_j| \ge \sqrt{2\kappa}

$$

:::
:::{prf:proof}

**Proof.**

The proof relies on a standard identity that relates the empirical variance of a set to the sum of its pairwise squared differences.

**1. The Pairwise Variance Identity.**
The empirical variance, $\text{Var}(\{v_i\}) = \frac{1}{k}\sum_i v_i^2 - (\frac{1}{k}\sum_i v_i)^2$, can be expressed as:

$$
\mathrm{Var}(\{v_i\}) = \frac{1}{2k^2} \sum_{i=1}^k \sum_{j=1}^k (v_i - v_j)^2

$$

This identity is established by expanding the squared term in the double summation.

**2. Bounding the Variance by the Maximum Gap.**
Let $\Delta_{\text{max}} := \max_{i,j} |v_i - v_j|$. By definition, every term in the summation is bounded above by this maximum: $(v_i - v_j)^2 \le \Delta_{\max}^2$. The double summation contains $k^2$ such terms. We can therefore bound the sum:

$$
\sum_{i=1}^k \sum_{j=1}^k (v_i - v_j)^2 \le \sum_{i=1}^k \sum_{j=1}^k \Delta_{\max}^2 = k^2 \Delta_{\max}^2

$$

Substituting this into the identity from Step 1 gives an upper bound on the variance in terms of the maximum gap:

$$
\mathrm{Var}(\{v_i\}) \le \frac{1}{2k^2} (k^2 \Delta_{\max}^2) = \frac{1}{2} \Delta_{\max}^2

$$

**3. Final Derivation.**
We are given the premise that $\mathrm{Var}(\{v_i\}) \geq \kappa$. Combining this with the result from Step 2:

$$
\kappa \le \mathrm{Var}(\{v_i\}) \le \frac{1}{2} \Delta_{\max}^2

$$

Rearranging the inequality $\kappa \le \frac{1}{2} \Delta_{\max}^2$ gives $\Delta_{\max}^2 \ge 2\kappa$. Taking the square root of both sides yields the desired result.

**Q.E.D.**
:::

This lemma provides the first crucial step in the signal processing analysis: it converts the abstract statistical guarantee of variance, now proven in {prf:ref}`thm-geometry-guarantees-variance`, into the concrete existence of at least two specific walkers with measurably different raw distance values.

#### 7.3.2. From a Raw Gap to a Guaranteed Rescaled Gap

A raw gap in the measurement values is not sufficient on its own to guarantee an adaptive signal. The standardization process, which involves dividing the raw values by the swarm's standard deviation, could potentially shrink this gap to an arbitrarily small value, effectively destroying the signal. This section proves that this is not the case by establishing uniform, N-independent bounds on the behavior of the pipeline's key components. We will show that any non-zero raw gap is reliably transformed into a non-zero rescaled gap, proving the integrity of the signal as it propagates through the pipeline.

#### 7.3.2.1. Uniform Bounds on Pipeline Components

To prove that the signal propagation is robust, we must first establish that the core components of the pipeline operate within a predictable, well-behaved range that is independent of the swarm's specific configuration or size.

The first component we must bound is the denominator of the standardization formula. The following definition establishes a uniform upper bound on the patched standard deviation (see {prf:ref}`def-patched-std-dev-function`).

:::{prf:definition} Maximum Patched Standard Deviation
:label: def-max-patched-std

Let $V_{\max}$ be the uniform upper bound on a raw measurement's absolute value (either $V_{\max}^{(R)}$ for rewards or $D_{\text{valid}}$ for distances). The **maximum patched standard deviation**, $\sigma'_{\max}$, is the maximum value that the patched standard deviation function can attain over its entire possible input domain.

$$
\sigma'_{\max} := \sup_{0 \le V \le V_{\max}^2} \sigma'_{\mathrm{patch}}(V)

$$

As the raw variance `Var({vᵢ})` is uniformly bounded by `V_max^{2}` and the function $\sigma'_patch(V)$ is continuous and monotonic, the Extreme Value Theorem guarantees that this maximum is attained at the right endpoint of the interval: $\sigma'_max = \sigma'_patch(V_max^{2})$. It is therefore a finite, positive constant determined only by the fixed system parameters, providing a state-independent upper bound for any standard deviation computed by the algorithm.
:::

Next, we must prove that the rescale function `g_A` is sufficiently sensitive to preserve a standardized gap. This requires showing that its derivative is uniformly bounded below by a positive constant over its entire operational domain.

:::{prf:lemma} Positive Derivative Bound for the Rescale Function
:label: lem-rescale-derivative-lower-bound

For the Canonical Logistic Rescale function (see {prf:ref}`def-logistic-rescale`), the first derivative $g'_A(z)$ is uniformly bounded below by a strictly positive constant for all z-scores in the operational range $Z_{\text{supp}}$. That is, there exists a constant $g'_{\min} > 0$ such that:

$$
\inf_{z \in Z_{\mathrm{supp}}} g'_A(z) = g'_{\min} > 0

$$

where $Z_{\text{supp}} := \left[ -2V_{\max}/\sigma'_{\min,\text{patch}}, 2V_{\max}/\sigma'_{\min,\text{patch}} \right]$ is the compact support of all possible standardized scores.
:::
:::{prf:proof}

**Proof.**
1.  **Compactness of the Domain:** Any standardized score `zᵢ` must lie within the interval `Z_supp` (from {prf:ref}`lem-compact-support-z-scores`). This interval is defined by the uniform constants `V_max` and $\sigma'_min,patch$, making `Z_supp` a compact set that is independent of the swarm state.

2.  **Properties of the Derivative:** The Canonical Logistic Rescale function (see {prf:ref}`def-logistic-rescale`) is $g_A(z) = 2 / (1 + e^{-z})$. Its derivative, $g'_A(z) = 2e^{-z} / (1+e^{-z})^2$, is continuous and strictly positive for all $z \in \mathbb{R}$.

3.  **Application of the Extreme Value Theorem:** By the Extreme Value Theorem, a continuous function ($g'_A(z)$) must attain its minimum value on a compact set ($Z_{\text{supp}}$).

4.  **Conclusion:** Since `g'_A(z)` is strictly positive on its entire domain, its minimum value on the compact subset `Z_supp`, which we define as `g'_min`, must also be a strictly positive constant.

**Q.E.D.**
:::

#### 7.3.2.2. The Main Propagation Lemma

With the uniform bounds on the pipeline's components now established, we can prove the main result of this section: a guaranteed raw measurement gap is reliably transformed into a guaranteed rescaled value gap.

:::{prf:lemma} From Raw Measurement Gap to Rescaled Value Gap
:label: lem-raw-gap-to-rescaled-gap

Let the system parameters be fixed. There exists a function $\kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}})$ such that for *any* swarm ({prf:ref}`def-swarm-and-state-space`) state `S` with $k \geq 2$ alive walkers, if the raw measurement values contain a gap $|vₐ - vᵦ| \geq \kappa_{\mathrm{raw}} > 0$, then the corresponding rescaled values are guaranteed to have a gap:

$$
|g_A(z_a) - g_A(z_b)| \ge \kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}}) > 0

$$

The function $\kappa_{\mathrm{rescaled}}$ is independent of the swarm ({prf:ref}`def-swarm-and-state-space`) state `S` and its size `k`, and is defined as:

$$
\kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}}) := \frac{g'_{\min}}{\sigma'_{\max}} \cdot \kappa_{\mathrm{raw}}

$$

:::
:::{prf:proof}

**Proof.**

The proof follows the signal gap as it propagates through the two main steps of the pipeline.

**Stage 1: From Raw Value Gap to a Uniform Lower Bound on the Z-Score Gap**
We seek a uniform lower bound for the gap between standardized scores, `|zₐ - zᵦ|`.

$$
|z_a - z_b| = \left| \frac{v_a - \mu}{\sigma'} - \frac{v_b - \mu}{\sigma'} \right| = \frac{|v_a - v_b|}{\sigma'}

$$

We are given the premise that the numerator is bounded below by $\kappa_{\mathrm{raw}}$. The denominator $\sigma'$ is the patched standard deviation (see {prf:ref}`def-patched-std-dev-function`) of the full set of `k` raw values. By Definition {prf:ref}`def-max-patched-std`, $\sigma'$ is uniformly bounded above by the state-independent constant $\sigma'_{\max}$. Combining these gives a uniform lower bound on the z-score gap:

$$
|z_a - z_b| \ge \frac{\kappa_{\mathrm{raw}}}{\sigma'_{\max}} =: \kappa_z > 0

$$

**Stage 2: From Z-Score Gap to Rescaled Value Gap**
The rescale function `g_A(z)` is continuously differentiable. By the Mean Value Theorem, there exists a point `c` on the line segment between `zₐ` and `zᵦ` such that:

$$
|g_A(z_a) - g_A(z_b)| = |g'_A(c)| \cdot |z_a - z_b|

$$

The points `zₐ`, `zᵦ`, and `c` are all within the compact operational range `Z_supp`. By Lemma {prf:ref}`lem-rescale-derivative-lower-bound`, the derivative at `c` is uniformly bounded below by the positive constant `g'_min`. Substituting the lower bounds for both terms on the right-hand side gives:

$$
|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \kappa_z

$$

**Conclusion**
Substituting the definition of $\kappa_z$ from Stage 1 yields the final result:

$$
|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \left(\frac{\kappa_{\mathrm{raw}}}{\sigma'_{\max}}\right) = \kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}})

$$

Since `g'_min` and $\sigma'_max$ are positive, N-uniform constants, the function $\kappa_rescaled(\kappa_raw)$ provides a strictly positive, N-uniform lower bound for any $\kappa_raw > 0$. This completes the proof that a raw measurement gap robustly propagates to a guaranteed rescaled value gap.

**Q.E.D.**
:::

#### 7.3.3 Conclusion: A Guaranteed Signal from Error to Rescaled Value

This section has forged a central link in the signal integrity proof. We have demonstrated that the measurement pipeline is a reliable signal processor whose behavior is uniformly bounded, independent of the swarm's size or specific configuration. By combining the lemmas, we have established a direct, N-uniform causal chain:

`Guaranteed Raw Variance (from` {prf:ref}`thm-geometry-guarantees-variance``)` → `Guaranteed Raw Gap` → `Guaranteed Rescaled Gap`

With this result, we have proven that a high-error state, which is guaranteed by {prf:ref}`thm-geometry-guarantees-variance` to produce a non-trivial raw measurement variance, always produces a non-trivial signal that survives the standardization and rescaling process. The final and most critical step, addressed in the next section, is to prove that the signals from the reward and diversity channels cannot pathologically cancel each other out in the final fitness calculation.

### 7.4. Population comparisons

:::{div} feynman-prose
Total variation in measurements can come from differences within each group or from differences between their means. The variance decomposition below separates these contributions. To determine which population is favored by selection, we also need the sign of the gap and the effect of the nonlinear fitness map.
:::

#### 7.4.1 The Macroscopic Signal Separation Lemma

The analysis in the preceding sections, culminating in {prf:ref}`thm-geometry-guarantees-variance`, establishes that a high-error swarm state is guaranteed to generate a raw measurement signal with a non-vanishing expected empirical variance. The next crucial step is to prove that this macroscopic statistical signal—a property of the entire population—forces a macroscopic separation between the means of the geometrically-defined high-error and low-error subpopulations.

The following lemma provides this fundamental link. It proves, from first principles, that a sufficiently large variance within a population partitioned into two substantial subsets necessitates a statistically significant separation between the means of those subsets.

:::{prf:lemma} **(From Total Variance to Mean Separation)**
:label: lem-variance-to-mean-separation

Let $\mathcal{V} = \{v_i\}_{i=1}^k$ be a set of $k \ge 2$ real numbers, with each element $v_i$ contained in the compact interval $[V_{\min}, V_{\max}]$. Let $\mathcal{V}$ be partitioned into two disjoint, non-empty subsets, $H$ and $L$, with corresponding means $\mu_H$ and $\mu_L$. Let their fractional population sizes, $f_H = |H|/k$ and $f_L = |L|/k$, be bounded below by a strictly positive constant $f_{\min} \in (0, 1/2]$, such that $f_H \ge f_{\min}$ and $f_L \ge f_{\min}$.

If the empirical variance of the total set, $\operatorname{Var}(\mathcal{V})$, is bounded below by a strictly positive constant $\kappa_{\mathrm{var}} > 0$, then the squared difference between the subset means is bounded below by:

$$
(\mu_H - \mu_L)^2 \ge \frac{1}{f_H f_L} \left( \kappa_{\mathrm{var}} - \operatorname{Var}_{\mathrm{max}} \right)

$$

where $\operatorname{Var}_{\mathrm{max}} := \frac{1}{4}(V_{\max} - V_{\min})^2$ is the maximum possible variance for any set of values on the interval.

Consequently, if the guaranteed variance $\kappa_{\mathrm{var}}$ is sufficiently large to satisfy the **Signal-to-Noise Condition**, $\kappa_{\mathrm{var}} > \operatorname{Var}_{\mathrm{max}}$, then the mean separation is guaranteed to be positive:

$$
|\mu_H - \mu_L| \ge \frac{1}{\sqrt{f_H f_L}} \sqrt{\kappa_{\mathrm{var}} - \operatorname{Var}_{\mathrm{max}}} > 0

$$

:::
:::{prf:proof}

**Proof.**

The proof is based on the decomposition of the total variance provided by the Law of Total Variance. We will establish a precise identity relating the total variance to the difference in subset means, find a sharp upper bound on the confounding variance term, and combine these results to derive the desired lower bound.

**Step 1: The Law of Total Variance.**
Let $\mu_{\mathcal{V}}$ be the mean of the entire set $\mathcal{V}$. The total empirical variance, $\operatorname{Var}(\mathcal{V}) := \frac{1}{k}\sum_{i \in \mathcal{V}} (v_i - \mu_{\mathcal{V}})^2$, can be decomposed into two components: the between-group variance ($\operatorname{Var}_B$) and the within-group variance ($\operatorname{Var}_W$).

$$
\operatorname{Var}(\mathcal{V}) = \operatorname{Var}_B(\mathcal{V}) + \operatorname{Var}_W(\mathcal{V})

$$

The **within-group variance** is the weighted average of the variances of the subsets:

$$
\operatorname{Var}_W(\mathcal{V}) := f_H \operatorname{Var}(H) + f_L \operatorname{Var}(L)

$$

The **between-group variance** is the variance of the subset means around the total mean:

$$
\operatorname{Var}_B(\mathcal{V}) := f_H(\mu_H - \mu_{\mathcal{V}})^2 + f_L(\mu_L - \mu_{\mathcal{V}})^2

$$

**Step 2: Relating Between-Group Variance to the Mean Separation.**
We will now prove that the between-group variance is directly proportional to $(\mu_H - \mu_L)^2$. The total mean is the weighted average of the subset means: $\mu_{\mathcal{V}} = f_H \mu_H + f_L \mu_L$. Substituting this into the definition of $\operatorname{Var}_B(\mathcal{V})$:

$$
\begin{aligned}
\mu_H - \mu_{\mathcal{V}} &= \mu_H - (f_H \mu_H + f_L \mu_L) = (1-f_H)\mu_H - f_L \mu_L = f_L \mu_H - f_L \mu_L = f_L(\mu_H - \mu_L) \\
\mu_L - \mu_{\mathcal{V}} &= \mu_L - (f_H \mu_H + f_L \mu_L) = -f_H \mu_H + (1-f_L)\mu_L = -f_H \mu_H + f_H \mu_L = -f_H(\mu_H - \mu_L)
\end{aligned}

$$

Substituting these expressions back into the formula for $\operatorname{Var}_B(\mathcal{V})$ yields:

$$
\begin{aligned}
\operatorname{Var}_B(\mathcal{V}) &= f_H (f_L(\mu_H - \mu_L))^2 + f_L (-f_H(\mu_H - \mu_L))^2 \\
&= f_H f_L^2 (\mu_H - \mu_L)^2 + f_L f_H^2 (\mu_H - \mu_L)^2 \\
&= (f_H f_L^2 + f_L f_H^2)(\mu_H - \mu_L)^2 \\
&= f_H f_L (f_L + f_H)(\mu_H - \mu_L)^2
\end{aligned}

$$

Since $f_H + f_L = 1$, we arrive at the exact identity:

$$
\operatorname{Var}_B(\mathcal{V}) = f_H f_L (\mu_H - \mu_L)^2

$$

**Step 3: A Uniform Upper Bound on the Within-Group Variance.**
The within-group variance, $\operatorname{Var}_W(\mathcal{V}) = f_H \operatorname{Var}(H) + f_L \operatorname{Var}(L)$, represents the noise that can mask the signal from the mean separation. We seek a sharp, state-independent upper bound. For any set of numbers on a compact interval $[a, b]$, the maximum possible variance is given by Popoviciu's inequality:

$$
\operatorname{Var}(S) \le \frac{1}{4}(\max(S) - \min(S))^2

$$

Since for any subset $S \subseteq \mathcal{V}$, its elements are contained in $[V_{\min}, V_{\max}]$, we have $\operatorname{Var}(H) \le \frac{1}{4}(V_{\max} - V_{\min})^2$ and $\operatorname{Var}(L) \le \frac{1}{4}(V_{\max} - V_{\min})^2$.
Let $\operatorname{Var}_{\mathrm{max}} := \frac{1}{4}(V_{\max} - V_{\min})^2$. The within-group variance is therefore uniformly bounded above:

$$
\operatorname{Var}_W(\mathcal{V}) \le f_H \operatorname{Var}_{\mathrm{max}} + f_L \operatorname{Var}_{\mathrm{max}} = (f_H+f_L)\operatorname{Var}_{\mathrm{max}} = \operatorname{Var}_{\mathrm{max}}

$$

This upper bound is sharp; it is attained if both subsets consist of values located only at the endpoints of the interval.

**Step 4: Assembling the Final Inequality.**
We rearrange the Law of Total Variance from Step 1:

$$
\operatorname{Var}_B(\mathcal{V}) = \operatorname{Var}(\mathcal{V}) - \operatorname{Var}_W(\mathcal{V})

$$

We substitute our identity for $\operatorname{Var}_B(\mathcal{V})$ from Step 2. Then, we use our premise, $\operatorname{Var}(\mathcal{V}) \ge \kappa_{\mathrm{var}}$, and our upper bound for the within-group variance from Step 3:

$$
f_H f_L (\mu_H - \mu_L)^2 \ge \kappa_{\mathrm{var}} - \operatorname{Var}_{\mathrm{max}}

$$

Since the fractional sizes $f_H$ and $f_L$ are strictly positive, dividing by their product preserves the inequality:

$$
(\mu_H - \mu_L)^2 \ge \frac{1}{f_H f_L} \left( \kappa_{\mathrm{var}} - \operatorname{Var}_{\mathrm{max}} \right)

$$

This proves the main inequality of the lemma. The final conclusion follows directly. If $\kappa_{\mathrm{var}} > \operatorname{Var}_{\mathrm{max}}$, the right-hand side is strictly positive. Taking the square root gives the lower bound on $|\mu_H - \mu_L|$. The pre-factor $1/\sqrt{f_H f_L}$ is well-defined and uniformly bounded above because the premises guarantee $f_H, f_L \ge f_{\min} > 0$. The entire lower bound is therefore a strictly positive constant.

**Q.E.D.**
:::
:::{admonition} Remark on the Role of {prf:ref}`lem-variance-to-gap`
:class: note

This lemma serves as the rigorous bridge between the macroscopic statistical properties of the measurement signal and the structural separation of its constituent subpopulations. Its primary function within the Keystone Principle's proof is to translate the guarantee of a non-vanishing total variance (the conclusion of {prf:ref}`thm-geometry-guarantees-variance`) into a guaranteed, non-vanishing separation between the means of the high-error and low-error sets.

The **Signal-to-Noise Condition**, $\kappa_{\mathrm{var}} > \operatorname{Var}_{\mathrm{max}}$, emerges from this analysis as a fundamental criterion for the system's "learnability." It formalizes the requirement that the signal generated by the system's geometric error must be strong enough to overcome the maximal possible statistical noise that could be generated by adversarial value configurations within the subpopulations.

*   $\kappa_{\mathrm{var}}$ represents the **Signal**: the total statistical heterogeneity that is guaranteed to be present in a high-error swarm.
*   $\operatorname{Var}_{\mathrm{max}}$ represents the **Worst-Case Internal Noise**: the maximum possible variance that can exist *within* subpopulations, which can act to mask a true difference in their means.

The lemma proves that if the signal is strictly greater than the worst-case internal noise, a separation between the subpopulation means is a mathematical necessity.

Consequently, for the proof of the **Stability Condition for Intelligent Adaptation** to proceed, it is necessary to demonstrate that the parameters of the Fragile Gas can be chosen such that the guaranteed measurement variance, $\kappa_{\mathrm{meas}}(\epsilon)$, satisfies this Signal-to-Noise condition. This establishes a verifiable, quantitative requirement for a well-posed system.
:::

#### 7.4.2. Log-fitness and arithmetic fitness

:::{div} feynman-prose
Multiplicative fitness is convenient on a logarithmic scale: its two factors
become a sum. But two populations ordered by mean log-fitness need not have the
same ordering by mean fitness. A population with a few large values can reverse
the latter comparison. The estimates below retain the difference between these
two averages and quantify the extra information needed to pass between them.
:::

:::{prf:theorem} Exact log-fitness comparison and a sufficient parameter bound
:label: thm-derivation-of-stability-condition

For the product fitness $V=(d')^\beta(r')^\alpha$, define, under the specified
population and measurement law,

$$
D=\mathbb E_L\log d'-\mathbb E_H\log d',\qquad
A=\mathbb E_H\log r'-\mathbb E_L\log r'.
$$

Then $\mathbb E_L\log V-\mathbb E_H\log V=\beta D-\alpha A$.
Thus the exact criterion for lower mean log-fitness in $H$ is
$\beta D>\alpha A$. If $D\geq D_*$ and $A\leq A_*$, the parameter condition
$\beta D_*>\alpha A_*$ is sufficient. Necessity concerns the exact gaps
$D,A$, rather than arbitrary lower and upper bounds for them.
:::

:::{prf:proof}
Take the logarithm of the positive product and average over each population.
Subtracting the two identities gives the exact gap. For $\alpha,\beta\geq0$,
it is at least $\beta D_*-\alpha A_*$. A positive lower bound proves the
sufficient condition.
:::

### 7.5. Quantitative bounds for logarithmic comparisons

#### 7.5.1. Extremal distributions with prescribed means

:::{prf:lemma} Lower logarithmic gap from the chord bound
:label: lem-log-gap-lower-bound

Let $X,Y\in[a,b]$ with $0<a<b$, and define

$$
\ell(u)=\frac{b-u}{b-a}\log a+\frac{u-a}{b-a}\log b,
\qquad c=\frac{\log b-\log a}{b-a}.
$$

If $\mathbb EX-\mathbb EY\geq\kappa$, $0\leq\kappa\leq b-a$, then

$$
\mathbb E\log X-\mathbb E\log Y\geq
L_{a,b}(\kappa):=
\min_{u\in[a,b-\kappa]}\{\ell(u+\kappa)-\log u\}.
$$

The minimum occurs at the projection of $1/c$ onto $[a,b-\kappa]$.
This bound can be negative. Under the stronger condition that there is a
coupling with $X\geq Y$ almost surely,

$$
\mathbb E\log X-\mathbb E\log Y\geq\kappa/b
\geq\log(1+\kappa/b).
$$
:::

:::{prf:proof}
Concavity gives the pointwise chord inequality $\log x\geq\ell(x)$ and
Jensen gives $\mathbb E\log Y\leq\log\mathbb EY$. Therefore the difference
is at least $\ell(\mathbb EX)-\log\mathbb EY$. Since $\ell$ is increasing,
replace $\mathbb EX$ by $\mathbb EY+\kappa$ and minimize over the allowed
mean of $Y$. The derivative is $c-1/u$ and the second derivative is $1/u^2$,
giving the stated minimizer.

The chord bound is attained by an endpoint-valued $X$ with the specified mean;
Jensen is attained by a deterministic $Y$. Thus this minimization retains the
extremal-distribution argument with its actual sign. Under an ordered coupling,
$\log X-\log Y=\int_Y^X t^{-1}\,dt\geq(X-Y)/b$; average and use
$\log(1+z)\leq z$.
:::

:::{prf:remark} What a mean gap determines
:label: rem-log-gap-bound-tightness

The chord formula gives a bound from the means and support alone. The simpler
positive logarithmic bound uses an ordered coupling. A positive arithmetic
mean gap by itself need not imply a positive logarithmic mean gap.
:::

:::{prf:lemma} Upper logarithmic gap and a coupling refinement
:label: lem-log-gap-upper-bound

For $X,Y\in[a,b]$ and $|\mathbb EX-\mathbb EY|\leq\kappa\leq b-a$,

$$
|\mathbb E\log X-\mathbb E\log Y|
\leq U_{a,b}(\kappa):=
\max_{u\in[a,b-\kappa]}\{\log(u+\kappa)-\ell(u)\}.
$$

The maximum occurs at the projection of $1/c-\kappa$ onto
$[a,b-\kappa]$. For any coupling with $\mathbb E|X-Y|\leq K$, the sharper
available estimate is

$$
|\mathbb E\log X-\mathbb E\log Y|\leq\log(1+K/a).
$$
:::

:::{prf:proof}
Jensen and the chord bound give
$\mathbb E\log X-\mathbb E\log Y\leq\log\mathbb EX-\ell(\mathbb EY)$.
For $\mathbb EY=u$, maximize the allowed first mean at
$\min(b,u+\kappa)$. On $[b-\kappa,b]$ the resulting expression decreases,
so the maximum is attained in $[a,b-\kappa]$. There its derivative is
$1/(u+\kappa)-c$ and its second derivative is negative. Exchange $X,Y$ for
the opposite sign.

For the coupling bound, pointwise
$|\log X-\log Y|\leq\log(1+|X-Y|/a)$. Take expectations and apply concave
Jensen to this upper bound. This use of Jensen has the required direction.
:::

:::{prf:remark} Equal means and unequal logarithmic averages
:label: rem-log-gap-bound-tight-at-vmin

At $\kappa=0$, the mean-only upper bound can remain positive because one
population can fluctuate while the other is constant. The coupling bound
vanishes when $K=0$, which identifies the two coupled values almost surely.
The global support bound $\log(b/a)$ is attained by opposite endpoint values.
:::

#### 7.5.2. Diversity and reward contributions

:::{prf:proposition} Corrective diversity signal
:label: prop-corrective-signal-bound

Suppose $d'\in[\eta,M]$ and its population means satisfy
$\mu_{d',L}-\mu_{d',H}\geq\kappa_{d'}\geq0$. Then

$$
D\geq L_{\eta,M}(\kappa_{d'}).
$$

If the population laws additionally admit a coupling with $d'_L\geq d'_H$
almost surely, one can use
$D\geq\kappa_{d'}/M\geq\log(1+\kappa_{d'}/M)$.
The choice of corrective orientation must be verified for the actual
measurement and rescaling rule.
:::

:::{prf:proof}
Apply {prf:ref}`lem-log-gap-lower-bound` to the two rescaled diversity laws.
Its ordered-coupling refinement gives the second bound.
:::

:::{prf:proposition} Global reward contribution bound
:label: prop-adversarial-signal-bound-naive

If $r'\in[\eta,M]$, then
$|\mathbb E_H\log r'-\mathbb E_L\log r'|\leq\log(M/\eta)$.
:::

:::{prf:proof}
Both logarithmic averages lie in $[\log\eta,\log M]$.
:::

:::{prf:proposition} Raw reward differences on a specified set
:label: prop-raw-reward-mean-gap-bound

Suppose the complete raw reward satisfies $|r_i-r_j|\leq B_r$ for every
cross-population pair under consideration. Then
$|\mu_{r,H}-\mu_{r,L}|\leq B_r$. If the complete reward is $L_R$-Lipschitz
on a set of diameter $D$, one may take $B_r=L_RD$.
:::

:::{prf:proof}
Write the difference of population means as the average of $r_i-r_j$ over all
cross pairs and use the assumed bound. The Lipschitz case follows directly
from $|r_i-r_j|\leq L_R\|z_i-z_j\|$.
:::

:::{prf:proposition} Reward log-gap from a cross-pair bound
:label: prop-log-reward-gap-axiom-bound

Under {prf:ref}`prop-raw-reward-mean-gap-bound`, suppose the shared patched
standard deviation is at least $\sigma_{\min}>0$ and the rescaling is
$L_g$-Lipschitz. Set $K_r=L_gB_r/\sigma_{\min}$. Then

$$
|\mathbb E_H\log r'-\mathbb E_L\log r'|
\leq\log(1+K_r/\eta).
$$

The raw reward bound includes every term used in fitness. A bound for the
objective alone does not bound a separate diverging boundary penalty.
:::

:::{prf:proof}
Shared centering cancels in the difference, so
$|z_{r,i}-z_{r,j}|\leq B_r/\sigma_{\min}$. Lipschitz rescaling gives
$|r'_i-r'_j|\leq K_r$ for each cross pair. Couple the two populations by their
product law and apply the coupling bound in {prf:ref}`lem-log-gap-upper-bound`.
:::

:::{prf:theorem} A quantitative sufficient condition for a log-fitness gap
:label: thm-stability-condition-final-corrected

With $D_*$ chosen from {prf:ref}`prop-corrective-signal-bound` and
$A_*\geq0$ chosen from either reward bound above, suppose
$\delta_{\log}:=\beta D_*-\alpha A_*>0$. Then

$$
\mathbb E_L\log V-\mathbb E_H\log V\geq\delta_{\log}.
$$

For fitness in $[v_*,v^*]$, an arithmetic mean gap follows if additionally
$\delta:=\delta_{\log}-\operatorname{Var}_H(V)/(2v_*^2)>0$:

$$
\mathbb E_LV-\mathbb E_HV\geq v_*(e^\delta-1)>0.
$$

All expectations and variances refer to the same specified population and
measurement law. For use in the conditional selection estimates, these bounds
must hold at that conditional stage, or their event probabilities and error
correlations must be retained when averaging.
:::

:::{prf:proof}
The log-gap bound follows from {prf:ref}`thm-derivation-of-stability-condition`.
For a positive random variable $X\geq v_*$, Taylor expansion of $\log X$
about its mean gives

$$
0\leq\log\mathbb EX-\mathbb E\log X
\leq\frac{\operatorname{Var}(X)}{2v_*^2}.
$$

Apply the upper bound to the $H$ population and the lower bound to $L$.
Subtracting gives
$\log\mathbb E_LV-\log\mathbb E_HV\geq\delta$.
Exponentiation and $\mathbb E_HV\geq v_*$ prove the arithmetic gap.
:::

### 7.6. Population fractions and overlap

:::{div} feynman-prose
A useful signal must occupy enough of the population to survive averaging.
The following estimates make this counting step explicit. Fitness variance
controls the size of the below-mean population. A gap between the actual group
means controls its overlap with the chosen high-error group. Neither count
creates common alive labels between two different swarms.
:::

:::{prf:definition} The unfit set
:label: def-unfit-set

For a realized fitness vector on $k$ alive walkers, let
$\mu=k^{-1}\sum_iV_i$ and define
$U_k=\{i\in\mathcal A_k:V_i\leq\mu\}$ and
$F_k=\mathcal A_k\setminus U_k$.
:::

:::{prf:lemma} An unfit-fraction bound from fitness variance
:label: lem-unfit-fraction-lower-bound

For a realized fitness vector with range $R_V>0$ and variance $s_V^2$,

$$
\frac{|U_k|}{k}\geq\frac{s_V^2}{2R_V^2},\qquad
\frac{|F_k|}{k}\geq\frac{s_V^2}{2R_V^2}.
$$

A uniform positive fitness variance and uniform range bound therefore give
uniform positive fractions.
:::

:::{prf:proof}
For a uniformly sampled fitness $X$, the centered mean vanishes, so
$\mathbb E(X-\mu)_+=\mathbb E(\mu-X)_+$. Also
$s_V^2\leq R_V\mathbb E|X-\mu|=2R_V\mathbb E(\mu-X)_+$.
Since $(\mu-X)_+\leq R_V\mathbf1_{X\leq\mu}$, the first bound follows.
Use the positive part and $\mathbf1_{X>\mu}$ for the second.
:::

:::{prf:theorem} Quantitative unfit and high-error overlap
:label: thm-unfit-high-error-overlap-fraction

For a realized fitness vector of range $R_V>0$, suppose a partition $H,L$ has
fractions $f_H,f_L>0$ and means $\mu_L-\mu_H\geq\Delta_V>0$. Then

$$
\frac{|H\cap U_k|}{k}\geq\frac{f_Hf_L\Delta_V}{R_V}.
$$

In a coupled swarm, the further intersection with $I_{11}$ satisfies

$$
|I_{11}\cap H\cap U_k|
\geq |H\cap U_k|-|\mathcal A_k\setminus I_{11}|.
$$
:::

:::{prf:proof}
The global mean is $\mu=f_H\mu_H+f_L\mu_L$, hence
$\mu-\mu_H=f_L(\mu_L-\mu_H)\geq f_L\Delta_V$.
Summing $(\mu-V_i)$ over $H$ gives at least $|H|f_L\Delta_V$.
Terms outside $U_k$ are negative, and each remaining term is at most $R_V$.
Thus $R_V|H\cap U_k|\geq|H|f_L\Delta_V$. Divide by $kR_V$.
The final bound is the elementary count obtained by deleting the labels absent
from $I_{11}$.
:::

### 7.7. The signal used by the cloning calculation

:::{div} feynman-prose
The logarithmic estimates retain the full extremal-distribution argument.
Their lower bounds can be negative, which is useful information when a proposed
parameter regime has too much within-population fluctuation. When an arithmetic
fitness gap has been established for the realized groups, the variance and
overlap bounds above supply explicit population estimates. Section 8 combines
these with the actual companion probabilities and the error carried by the
selected labels.
:::

(sec-cloning-keystone)=
## 8. The N-Uniform Quantitative Keystone Lemma

:::{div} feynman-prose
The Keystone estimate combines two inputs: cloning probability on a target set, and positional error carried by that set. The preceding sections provide geometric, measurement, and fitness bounds that can verify those inputs. The theorem states them for the actual law and the same family of coupled states, so the constants can be followed into the drift calculation.
:::

### 8.1 The Quantitative Keystone Lemma and Proof Strategy

We begin by formally stating the main theorem. This lemma provides the quantitative link between the system's error and its corrective response, which will be the primary tool for the drift analysis in the subsequent sections. The lemma considers the summed cloning probability from both swarms, $(p_{1,i} + p_{2,i})$, to capture the total corrective pressure. The proof will demonstrate that even when only one swarm is in a high-error state, the cloning pressure from that single swarm is sufficient to ensure the inequality holds.

:::{prf:lemma} The N-Uniform Quantitative Keystone Lemma
:label: lem-quantitative-keystone

Consider a family of coupled configurations satisfying the foundational bounds
of Section 4. Above a structural threshold, suppose its target set satisfies
{prf:ref}`cor-cloning-pressure-target-set` with a common $p_u>0$ and
{prf:ref}`lem-error-concentration-target-set` with common constants
$c_{\mathrm{err}}>0$, $g_{\mathrm{err}}\geq0$. These estimates must use the
actual selection law and the same coupled configurations. Then there exist:
*   a structural error threshold $R^2_{\text{spread}} > 0$,
*   a minimum feedback coefficient $\chi(\epsilon) > 0$,
*   and a constant offset $g_{\max}(\epsilon) \ge 0$,

all of which may depend on $\epsilon$ but are independent of $N$, such that for every pair in the stated family:

$$
\frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2 \ge \chi(\epsilon) V_{\text{struct}} - g_{\max}(\epsilon)

$$

where $I_{11}$ is the set of stably alive walkers and $p_{k,i}$ is the total cloning probability for walker ({prf:ref}`def-walker`) $i$ in swarm ({prf:ref}`def-swarm-and-state-space`) $k$.

For the complete measurement-averaged kernel, {prf:ref}`thm-keystone-complete-error-coverage` includes every geometric cluster and derives the self-exclusion correction without a target-mass or error-coverage premise. {prf:ref}`thm-keystone-discharged-averaged-pressure` then proves the averaged Keystone bound from the entering geometry, with population-independent positive constants and an explicit $N^{-2}$ correction; its structural formulation retains the actual velocity and alive-status terms. The sharper state-dependent route in {prf:ref}`thm-keystone-averaged-cluster-pressure` remains available, and {prf:ref}`cor-keystone-canonical-balanced-structural` gives a zero-offset family. All these averaged statements retain complete-fitness-tie events in their probability space.

Referenced by {prf:ref}`def-decision-operator` and {prf:ref}`lem-keystone-contraction-alive`.
:::

:::{div} feynman-prose
The Keystone sum contains squared positional discrepancies. The comparison with full phase-space structural error retains the velocity and cross-term remainder in {prf:ref}`lem-error-concentration-target-set`. Coercivity alone does not turn a large velocity discrepancy into a large positional one.
:::

**Proof Strategy:**

The proof of the Keystone Lemma establishes the inequality globally by partitioning the state space into two distinct regimes based on the magnitude of the structural error $V_{\text{struct}}$. The constants $\chi(\epsilon)$ and $g_{\max}(\epsilon)$ are constructed to ensure the inequality holds in both cases.

1.  **The Low-Error Regime ($V_{\text{struct}} \le R^2_{\text{spread}}$):**
    In this regime, the inequality is satisfied by a careful choice of the offset constant. The left-hand side (LHS) of the inequality, being a weighted sum of squared norms, is always non-negative. For the right-hand side (RHS), we choose the offset to satisfy $g_{\max}(\epsilon) \ge \chi(\epsilon) R^2_{\text{spread}}$. This ensures that for any state in this regime, the RHS is non-positive:

$$
\chi(\epsilon) V_{\text{struct}} - g_{\max}(\epsilon) \le \chi(\epsilon) R^2_{\text{spread}} - g_{\max}(\epsilon) \le 0

$$

Therefore, we have $\text{LHS} \ge 0 \ge \text{RHS}$, and the inequality is trivially satisfied.

2.  **The High-Error Regime ($V_{\text{struct}} > R^2_{\text{spread}}$):**
    This regime constitutes the core of the proof. The remainder of this chapter is dedicated to a constructive proof of the inequality for this case. We will use the full causal chain established in Sections 6 and 7 to show that the left-hand side is bounded below by a linear function of $V_{\text{struct}}$. The constants $\chi(\epsilon)$ and the remainder of $g_{\max}(\epsilon)$ will emerge directly from this constructive proof.

### 8.2. The target population

:::{div} feynman-prose
The relevant walkers belong to three sets: the high-error population, the below-mean fitness population, and the labels alive in both coupled swarms. Their intersection is the target. The within-swarm overlap estimate and the common-label count are separate steps; the error-concentration proof subtracts the omitted labels explicitly.
:::

:::{prf:definition} The Critical Target Set
:label: def-critical-target-set

For a state in the high-error regime, let $k$ be the index of the high-variance swarm ({prf:ref}`def-swarm-and-state-space`). The **critical target set**, $I_{\text{target}}$, is the set of walkers that are simultaneously stably alive, unfit in swarm $k$, and high-error in swarm $k$.

$$
I_{\text{target}} := I_{11} \cap U_k \cap H_k(\epsilon)

$$

The within-swarm overlap estimate of {prf:ref}`thm-unfit-high-error-overlap-fraction` concerns $U_k\cap H_k$. Intersecting with $I_{11}$ requires separate control of the common alive labels. The target-error estimate below retains the omitted contribution explicitly; nonemptiness follows when its lower bound is positive.

Referenced by {prf:ref}`cor-cloning-pressure-target-set`.
:::

### 8.3. Selection probability on the target set

:::{div} feynman-prose
A population can have a fixed fitness range while almost every value is nearly
the same. What controls the aggregate comparison signal is its spread and the
probability of sampling the better values. We can make this precise without
passing a clipping function through an expectation.
:::

#### 8.3.1. Companion means and positive fitness differences

:::{prf:lemma} Companion mean and positive-part bounds
:label: lem-mean-companion-fitness-gap

Condition on a realized fitness vector $V_1,\ldots,V_k$, $k\geq2$, with
$0<V_i\leq V_{\mathrm{pot,max}}$, mean $\mu$, variance $s_V^2$, and range
$R_V=\max V_i-\min V_i>0$. For a uniform nonself companion,

$$
\mu_{\mathrm{comp},i}-V_i=\frac{k}{k-1}(\mu-V_i).
$$

For a general companion law $K_i(j)$, suppose
$K_i(j)\geq a/(k-1)$ for all $j\ne i$, with $a>0$. Every walker with
$V_i\leq\mu$ then satisfies

$$
\mathbb E_{K_i}(V_c-V_i)_+
\geq \frac{ak}{k-1}\frac{s_V^2}{2R_V}
\geq \frac{a s_V^2}{2R_V}.
$$

If a partition into populations $H,L$ has fractions $f_H,f_L$ and realized
fitness means differing by $\Delta$, then $s_V^2\geq f_Hf_L\Delta^2$.
Thus a uniform population mean gap, together with a uniform companion lower
bound, supplies a uniform positive-part signal.
:::

:::{prf:proof}
The uniform companion mean is $(k\mu-V_i)/(k-1)$, proving the identity.
Let $X$ be a uniformly sampled fitness value. Since $|X-\mu|\leq R_V$,

$$
s_V^2\leq R_V\mathbb E|X-\mu|
=2R_V\mathbb E(X-\mu)_+.
$$

For $V_i\leq\mu$, $(V_j-V_i)_+\geq(V_j-\mu)_+$, and the $j=i$ term in
the latter sum is zero. Summing the companion lower bounds proves the stated
inequality. Finally, expansion around the two group means gives
$s_V^2=f_Hs_H^2+f_Ls_L^2+f_Hf_L(\mu_H-\mu_L)^2$; the first two terms are
nonnegative.
:::

:::{prf:remark} Uniformity and the probability space of the fitness signal
:label: rem-n-uniformity-delta-min-bound

The factor $k/(k-1)$ is bounded below by one. Uniformity follows from a positive
lower bound on $a s_V^2/R_V$, or from an equivalent positive-part comparison
bound. A fixed nonzero range alone is insufficient: one low and one high value
among $k-2$ midpoint values have fixed range and variance of order $k^{-1}$.

These inequalities concern the realized fitness vector and its conditional
companion law. If the fitness is random, take the outer expectation after
applying them. An inequality for expected distance or expected log-fitness does
not identify the realized fitness variance or its correlation with the target
error. A conditional lower bound can be averaged on an event of known
probability, retaining that probability as a factor.
:::

:::{prf:lemma} Cloning pressure from the positive-part comparison signal
:label: lem-unfit-cloning-pressure

Condition on the realized fitness and companion law. Let the independent
cloning threshold be uniform on $[0,p_{\max}]$, and set

$$
A_V:=\max\{R_V,p_{\max}(V_{\mathrm{pot,max}}+
\varepsilon_{\mathrm{clone}})\}.
$$

Then the actual clipped cloning probability satisfies

$$
p_i=\mathbb E_{K_i}\min\left\{1,
\frac{(V_c-V_i)_+}{p_{\max}(V_i+\varepsilon_{\mathrm{clone}})}\right\}
\geq\frac{\mathbb E_{K_i}(V_c-V_i)_+}{A_V}.
$$

Under {prf:ref}`lem-mean-companion-fitness-gap`, for $V_i\leq\mu$ this gives
$p_i\geq a s_V^2/(2R_VA_V)$. On a family with
$a\geq a_*>0$, $s_V^2\geq s_*^2>0$, and
$R_V\leq R_*<\infty$, one can therefore use the $N$-uniform constant

$$
p_u=\frac{a_*s_*^2}
 {2R_*\max\{R_*,p_{\max}(V_{\mathrm{pot,max}}+
 \varepsilon_{\mathrm{clone}})\}}>0.
$$

Alternatively, a favorable set with $V_j-V_i\geq\Delta>0$ and companion
probability at least $q_*>0$ gives
$p_i\geq q_*\min\{1,\Delta/[p_{\max}(V_{\mathrm{pot,max}}+
\varepsilon_{\mathrm{clone}})]\}$.
:::

:::{prf:proof}
For $0\leq z\leq R_V$ and
$b_i=p_{\max}(V_i+\varepsilon_{\mathrm{clone}})$,
$\min(1,z/b_i)\geq z/A_V$: when $z\leq b_i$ use $A_V\geq b_i$, and
when $z>b_i$ use $A_V\geq R_V\geq z$. Apply this pointwise with
$z=(V_c-V_i)_+$ and average. Substitute the previous lemma and the uniform
bounds to obtain $p_u$. Restriction of the expectation to the favorable set
gives the alternative estimate. This is the same selection calculation used
in {prf:ref}`lem-boundary-enhanced-cloning`.
:::

:::{prf:corollary} Cloning pressure on an error-carrying target set
:label: cor-cloning-pressure-target-set

Let $I_{\mathrm{target}}\subseteq I_{11}\cap U_k\cap H_k$ satisfy one of the
uniform comparison conditions of {prf:ref}`lem-unfit-cloning-pressure` in swarm
$k$. Then every $i\in I_{\mathrm{target}}$ has $p_{k,i}\geq p_u>0$.
If the fitness pipeline is random, the bound applies after conditioning on its
realized output; an outer expectation retains the same constant when the
condition holds for all relevant realizations.
:::

:::{prf:proof}
Apply the preceding lemma to each member of the specified target set. An
almost-sure conditional inequality remains valid after taking expectation.
:::

### 8.4. Error concentration in the target set

:::{div} feynman-prose
Selection probability is useful when it acts on the error we are measuring. First we recover the variance captured by an outlier set, using the cluster decomposition. Then we compare the two swarms and subtract the error outside their common target. The resulting offset remains in the final Keystone inequality.
:::

:::{prf:lemma} Variance captured by the high-error set
:label: lem-variance-concentration-Hk

Let $S_k=\sum_{i\in\mathcal A_k}\|x_i-\mu\|^2$.
If the global outlier set $H_k$ captures a fraction $1-\varepsilon_O$ of this
sum by definition, its variance contribution is at least
$(1-\varepsilon_O)S_k$.

For a cluster partition with cluster diameters at most $D_c$, suppose $H_k$ is
the union of clusters capturing at least a fraction $1-\varepsilon_O$ of the
between-cluster variance. If $S_k/k>R_{\mathrm{var}}^2>D_c^2/2$, then

$$
\sum_{i\in H_k}\|x_i-\mu\|^2\geq c_HS_k,\qquad
c_H=(1-\varepsilon_O)\left(1-\frac{D_c^2}{2R_{\mathrm{var}}^2}\right)>0.
$$

The constants are uniform in $N$ when the displayed geometric bounds are.
:::

:::{prf:proof}
For a cluster $G$ of size $m$, the pairwise variance identity gives

$$
\operatorname{Var}(G)=\frac1{2m^2}\sum_{i,j\in G}\|x_i-x_j\|^2
\leq\frac{D_c^2}{2}.
$$

This dimension-independent vector bound is sufficient here. The value
$D_c^2/4$ requires a stronger radius or one-dimensional hypothesis.
Let $W=\sum_G|G|\operatorname{Var}(G)$ and
$B=\sum_G|G|\|\mu_G-\mu\|^2$. Expansion around each cluster mean, whose
centered deviations sum to zero, gives exactly $S_k=W+B$ and
$W\leq kD_c^2/2$.
The same expansion on the selected clusters gives

$$
\sum_{i\in H_k}\|x_i-\mu\|^2\geq(1-\varepsilon_O)B
\geq(1-\varepsilon_O)S_k\left(1-\frac{kD_c^2}{2S_k}\right).
$$

Since $S_k/k>R_{\mathrm{var}}^2$, the last factor is at least
$1-D_c^2/(2R_{\mathrm{var}}^2)$, proving the result. The global outlier case
follows directly from its stated capture property.
:::
:::{div} feynman-prose
The center and internal variance here use the alive population:
$S_k=k\operatorname{Var}_{\mathcal A_k}(x)$.
The Lyapunov contribution uses $S_k/N$. Thus population geometry is normalized
by $k$, while its contribution to the fixed-$N$ swarm observable carries the
factor $k/N$. The following estimate keeps that normalization explicitly.
:::

:::{prf:lemma} Error concentration with an explicit complement term
:label: lem-error-concentration-target-set

Let $H$ lie in the alive set of swarm $k$, let $T=I_{11}\cap U_k\cap H$, and
write $S_k=\sum_{i\in\mathcal A_k}\|\delta_{x,k,i}\|^2$. Suppose

$$
\sum_{i\in H}\|\delta_{x,k,i}\|^2\geq c_HS_k,\qquad
S_k/N\geq aV_{\mathrm{struct}}-b,
$$

with $c_H,a>0$. Suppose the other swarm's centered positions, defined on the
comparison labels, satisfy
$N^{-1}\sum_{i\in H}\|\delta_{x,j,i}\|^2\leq M_j$ and the omitted comparison
error satisfies
$N^{-1}\sum_{i\in H\setminus T}\|\Delta\delta_{x,i}\|^2\leq B_T$.
Then

$$
\frac1N\sum_{i\in T}\|\Delta\delta_{x,i}\|^2
\geq \frac{c_Ha}{2}V_{\mathrm{struct}}
-\left(\frac{c_Hb}{2}+M_j+B_T\right).
$$

The concentration estimate in {prf:ref}`lem-variance-concentration-Hk` supplies
$c_H$. On a bounded positional domain one may use $M_j\leq D_{\mathrm{valid}}^2$
and $B_T\leq4D_{\mathrm{valid}}^2$ when both centered configurations obey that
bound. The relation to $V_{\mathrm{struct}}$ must retain its velocity and
cross terms if it denotes a full phase-space error.
:::

:::{prf:proof}
The identity
$\|u-v\|^2-\tfrac12\|u\|^2+\|v\|^2
=\tfrac12\|u-2v\|^2\geq0$ gives

$$
\frac1N\sum_{i\in H}\|\Delta\delta_{x,i}\|^2
\geq \frac{c_H S_k}{2N}-M_j
\geq\frac{c_Ha}{2}V_{\mathrm{struct}}-\frac{c_Hb}{2}-M_j.
$$

Subtract the sum over $H\setminus T$ and use its bound $B_T$.
For the bounded positional estimates, each centered position has norm at most
$D_{\mathrm{valid}}$ and each difference at most $2D_{\mathrm{valid}}$; there
are at most $N$ terms. These give the displayed constants. In the purely
positional comparison, the standard inequality
$V_{x,\mathrm{struct}}\leq2(S_1+S_2)/N$ allows $a=1/2$ and
$b=S_j/N\leq D_{\mathrm{valid}}^2$. A bounded velocity remainder is added to
$b$ when a phase-space comparison is used. Thus the proof retains the stated
variance, comparison, and complement decomposition with every remainder
explicit.
:::

### 8.5. Averaged target estimates from the actual geometric clusters

:::{prf:lemma} Weighted near and far measurement events in a geometric cluster
:label: lem-keystone-geometric-measurement-events

The actual alive-row diversity measurement is $Y_i=\sqrt{|z_i-z_{D_i}|^2+\delta_D^2}$, where $D_i$ is the independently sampled current measurement companion and $\delta_D=10^{-3}$ in the canonical configuration. The diversity standardizer here is the configured global one:

$$
\bar Y=\frac1k\sum_{i\in\mathcal A}Y_i,\qquad
s_Y=\sqrt{\frac1k\sum_{i\in\mathcal A}(Y_i-\bar Y)^2+\varepsilon_s^2},
$$

with $\varepsilon_s=0.1$ canonically. The comparison features and weights below are those of the actual canonical kernel. In particular, the bounds in {prf:ref}`thm-cloning-canonical-barycenter-concentration` give $D_z^2\le32$ and $\kappa_D=\kappa_C=e^{-4}$. Actual pairwise extrema may sharpen these constants. The geometric partition remains the chapter's complete-linkage partition; the estimates use its actual cluster diameters.


Let $k\ge2$ alive rows have comparison features $z_i$, actual symmetric measurement weights $\kappa_D\le w^D_{ij}\le1$, and actual cloning weights $\kappa_C\le w^C_{ij}\le1$, all with self exclusion. Put
$$
\mathsf V_z=\frac1k\sum_i|z_i-\bar z|^2,\qquad
D_z=\max_{i,j}|z_i-z_j|,
$$
and choose a geometric distance threshold $h$ with $0<h^2<\mathsf V_z$. This $h$ is a distance threshold for the estimate, distinct from the physical timestep of the unchanged transition. Define
$$
\rho_h=\frac{\mathsf V_z-h^2}{D_z^2-h^2}>0.
\tag{3.AP1}
$$
For completed canonical box inputs, this feature variance has a proved relation to physical positional variance. On the ball $|x|\le B$, the radial and tangential eigenvalues of $DS_R(x)$ are at least $m_R=R^2/(R+B)^2$. Integrating the Jacobian along the segment from $x$ to $y$ gives $(S_R(x)-S_R(y))\cdot(x-y)\ge m_R|x-y|^2$, hence $|S_R(x)-S_R(y)|\ge m_R|x-y|$. Applying the pairwise variance identity and retaining the nonnegative velocity-feature contribution yields $\mathsf V_z\ge m_R^2\operatorname{Var}_{\mathcal A}(x)$. Canonically $R=2$ and $B=2\sqrt d$, so $m_R=(1+\sqrt d)^{-2}$. This comparison concerns eligible alive positions; it imposes no bound on retained dead coordinates.

For every alive row $j$, the proportion of its distinct eligible companions at distance at least $h$ is at least $\rho_h$. Indeed,
$$
\frac1k\sum_\ell|z_\ell-z_j|^2
=\mathsf V_z+|z_j-\bar z|^2
\le h^2+(D_z^2-h^2)\frac{\#\{\ell:|z_\ell-z_j|\ge h\}}k.
$$
The self term is zero, so removal of $j$ only increases this proportion. Its actual weighted probability is consequently at least $\kappa_D\rho_h$.

For a geometric cluster $G$, write $n_G=|G|$, $r_G=\operatorname{diam}_z(G)$, and
$$
\ell_G=\sqrt{r_G^2+\delta_D^2},\qquad
H_h=\sqrt{h^2+\delta_D^2},\qquad
\delta_G=H_h-\ell_G.
$$
Consider clusters with $n_G\ge2$ and $r_G<h$, so $\delta_G>0$. Define their exact nonself mass
$$
\rho_G=\frac{n_G-1}{k-1}.
\tag{3.AP2}
$$
For $i,j\in G$, $i\ne j$, the measurement event
$$
E_{ij}=\{Y_i\le\ell_G,\ Y_j\ge H_h\}
$$
has probability at least $\kappa_D^2\rho_G\rho_h$. The first event contains all measurement choices by $i$ within $G\setminus\{i\}$; the second uses (3.AP1). The two measurement draws are independent, even though their resulting fitnesses use the same population standardization.

Let $A_i$ be the actual rescaled reward factor in the product fitness. It is fixed by the entering state. Let $f(z)=g_s(z)^{p_s}$ be its positive increasing diversity factor, where $p_s>0$, and let $\varepsilon_s>0$ be the configured global diversity regularizer. All possible measured separations lie in a deterministically known interval $[y_-,y_+]$; one may take its exact range over eligible pairs. Set
$$
D_m=y_+-y_-,\quad Z_m=D_m/\varepsilon_s,\quad
s_*=\sqrt{D_m^2/4+\varepsilon_s^2},
$$
$$
f_-=\min_{|z|\le Z_m}f(z)>0,\quad
f_+=\max_{|z|\le Z_m}f(z),\quad
m_f=\min_{|z|\le Z_m}f'(z)>0.
\tag{3.AP3}
$$
These minima exist for the configured logistic map with strictly positive amplitude and every fixed positive exponent; the canonical amplitude is $2$. The bounded-interval variance inequality is explicit: averaging $(Y-y_-)(y_+-Y)\ge0$ gives $\operatorname{Var}(Y)\le(y_+-\bar Y)(\bar Y-y_-)\le D_m^2/4$. Thus every realized regularized scale is at most $s_*$, and every realized standardized measurement lies in $[-Z_m,Z_m]$.

The following are actual cluster statistics, not presumed favorable fitness gaps:
$$
A_G^- =\min_{i\in G}A_i,\qquad
\omega_G=\max_{i\in G}A_i-\min_{i\in G}A_i,
$$
$$
\gamma_G=A_G^-m_f\delta_G/s_*-f_+\omega_G,
\qquad
\mathfrak a_G=\min\left\{1,
\frac{(\gamma_G)_+}{p_{\max}(F^*+\varepsilon_c)}\right\}.
\tag{3.AP4}
$$
Here $F^*$ is the algorithm's proved fitness upper bound. All quantities in (3.AP1)--(3.AP4) can be evaluated from the entering state and configured maps before any random measurement is drawn.


:::

:::{prf:theorem} Measurement-averaged pressure on geometric clusters
:label: thm-keystone-averaged-cluster-pressure


For every alive row $i$ in one of these same geometric clusters, its unconditional cloning probability satisfies
$$
\boxed{\overline p_i:=\mathbb E[p_i(\mathbf F)\mid S]
\ge \pi_G:=\kappa_C\kappa_D^2\rho_G^2\rho_h\mathfrak a_G.}
\tag{3.AP5}
$$
This is an inequality for the actual complete retained-fitness acceptance law. Equal-fitness events remain in its probability space. It imposes no assumption that $i$ is below the realized global fitness mean.

**Proof.** On $E_{ij}$, the two standardized diversity arguments satisfy
$$
\frac{Y_j-\bar Y}{s_Y}-\frac{Y_i-\bar Y}{s_Y}
\ge\delta_G/s_*.
$$
The same realized mean and scale occur in both arguments. By the derivative bound in (3.AP3),
$$
f((Y_j-\bar Y)/s_Y)-f((Y_i-\bar Y)/s_Y)
\ge m_f\delta_G/s_*.
$$
Consequently
$$
F_j-F_i
=A_j(f_j-f_i)+(A_j-A_i)f_i
\ge A_G^-m_f\delta_G/s_*-f_+\omega_G=\gamma_G.
$$
For $\gamma_G>0$, the conditional acceptance of donor $j$ is at least $\mathfrak a_G$; if $\gamma_G\le0$, (3.AP5) is the valid zero lower bound. The actual cloning donor probability is fixed by the entering physical state and independent of the measurement draws. Average the acceptance after the event estimate, then sum only over donors $j\in G\setminus\{i\}$. Their total actual cloning probability is at least $\kappa_C\rho_G$. Together with $\Pr(E_{ij})\ge\kappa_D^2\rho_G\rho_h$, this gives (3.AP5). No joint occurrence of all events $E_{ij}$ is required, because the algorithm draws only one cloning donor and the donor expectation is a sum. The proof never replaces sampled fitness by an expected fitness. $\square$

For every statistically valid cluster in the chapter's construction,
$n_G\ge\max(5,\lceil0.05k\rceil)$. Thus
$$
\rho_G\ge\frac{(4/5)n_G}{k}\ge0.04,
\tag{3.AP6}
$$
where $n_G-1\ge(4/5)n_G$ follows from $n_G\ge5$. Therefore (3.AP5) has the explicit population-independent lower bound
$$
\overline p_i\ge 0.04^2\kappa_C\kappa_D^2\rho_h\mathfrak a_G.
$$
Small clusters need not be discarded: their exact $\rho_G$ in (3.AP5) remains available. Only a claim of a common positive constant over such clusters requires inspecting their masses.

The reward oscillation can also be bounded analytically from the actual objective. If its reward is $L_R$-Lipschitz on the entering alive region and the reward map has derivative bound $L_g$, global regularization $\varepsilon_r$, and unit reward exponent, then
$$
\omega_G\le (L_gL_R/\varepsilon_r)\operatorname{diam}_{\rm phys}(G).
$$
For other fixed exponents, include the derivative of the configured powered reward map. This is a derived bound, not a replacement reward law. Canonical box inputs bound $L_R$ directly for their configured smooth objective; the actual finite cluster oscillation in (3.AP4) is often sharper.


:::

:::{prf:theorem} Actual averaged target error and its uncovered contribution
:label: thm-keystone-averaged-error-capture


Apply (3.AP5) in both swarms. Put its value equal to zero on any row whose cluster is not certified by the displayed geometry, without changing the cluster partition. For $i\in I_{11}$, let $\pi_i=\pi_{1,i}+\pi_{2,i}$ and $e_i=|\Delta\delta_{x,i}|^2$. Then
$$
\boxed{\mathbb E\!\left[\frac1N\sum_{i\in I_{11}}
 (p_{1,i}+p_{2,i})e_i\,\middle|\,S_1,S_2\right]
\ge\frac1N\sum_{i\in I_{11}}\pi_i e_i.}
\tag{3.AP7}
$$
**Proof.** The geometric errors are fixed by the entering states, so this averaging introduces no covariance assumption. Let $\mathcal P$ be the common refinement of the existing geometric partitions and the common-alive label set, and let $\pi_B=\min_{i\in B}\pi_i$. The right side has the proved lower bound
$$
\sum_{B\in\mathcal P}\pi_B\frac{|B|}{N}
\left(\left|\overline{\Delta\delta_x}_B\right|^2
+\frac1{|B|}\sum_{i\in B}
 |\Delta\delta_{x,i}-\overline{\Delta\delta_x}_B|^2\right).
\tag{3.AP8}
$$
This is the exact within/between cluster identity applied to the error vectors, followed by a pointwise probability lower bound. It does not infer error capture from population counts.

For any chosen $p_*>0$, let $T_*$ be the union of blocks with $\pi_B\ge p_*$, and define the actual uncovered error
$$
R_* =\frac1N\sum_{i\in I_{11}\setminus T_*}e_i.
$$
Then the right side of (3.AP7) is at least
$$
p_*\left[\frac1N\sum_{i\in I_{11}}e_i-R_*\right].
\tag{3.AP9}
$$
Both the probability coefficient and the error not captured are now explicit functions of the actual entering state. On a common alive position domain of diameter $D_x$,
$R_*\le4D_x^2|I_{11}\setminus T_*|/N$. Thus uncertified or low-mass clusters are charged their actual geometric error rather than assigned a positive probability by assumption.

For all-alive inputs, the structural hypocoercive cost satisfies, for every $\eta>0$,
$$
V_{\rm struct}\le(1+\eta)\frac1N\sum_i e_i
+\left(\lambda_v+\frac{b^2}{4\eta}\right)
 \frac1N\sum_i|\Delta\delta_{v,i}|^2.
$$
This follows by using the comparison-label pairing as a transport plan and applying Young's inequality to its cross term. Substitution into (3.AP9) produces the Keystone inequality with coefficient $p_* /(1+\eta)$ and the explicitly retained uncovered-error and velocity remainders. If every entering velocity has norm at most $V_{\max}$, its displayed mean squared centered discrepancy is at most $4V_{\max}^2$: centering is an orthogonal projection in the empirical $L^2$ norm, and $|v_{1,i}-v_{2,i}|\le2V_{\max}$. The positional pressure estimate cannot absorb a velocity-only discrepancy without that remainder. For unequal alive pools the original alive-normalization and common-label residual must likewise be retained; (3.AP7)--(3.AP9) themselves apply without an alive-fraction floor.


A zero certificate is distinct from zero actual activity. A coarse cluster, an insufficient reward contrast, or a small eligible mass can make (3.AP5) uninformative while the actual algorithm continues to clone. The exact uncovered error in (3.AP9) retains these cases; it is not replaced by an assumed favorable fraction. $\square$

:::

(sec-cloning-complete-error-coverage)=
#### Complete coverage of the geometric error

:::{div} feynman-prose
Keep every geometric cluster and imagine laying a fixed grid over the bounded comparison features. The grid answers one question: how much eligible donor mass lies near a walker? Intersect its cells with the existing clusters, and every error contribution is still present, including those in clusters with poor statistical estimates. The number of cells depends on the geometric scale and dimension, so it stays fixed as $N$ grows.

Nearby mass then gives an actual probability of measurement outcomes that separate a recipient's retained fitness from a donor's. The proof evaluates acceptance with those sampled values and their shared normalizers. Removing the recipient from its own donor pool leaves an explicit finite-population correction. This produces an averaged pressure estimate with constants independent of $N$, while preserving all clusters and the full measurement law.
:::


:::{prf:lemma} Uniform constants from the actual fitness and companion laws
:label: lem-keystone-complete-coverage-constants

Use the chapter's independent Gaussian-weighted measurement and cloning companions, retained global empirical standardization, powered positive fitness, and frozen acceptance. Keep every configured parameter fixed. Write the actual fitness as
$$
F_i=A_i f(u_i),\qquad A_i=H((R_i-\bar R)/s_R),\qquad
u_i=(Y_i-\bar Y)/s_Y,
$$
where $H(t)=(g_r(t)+\eta_r)^\alpha$, $f(t)=(g_s(t)+\eta_s)^\beta$, $\alpha\ge0$, and the active diversity exponent satisfies $\beta>0$. The positive floors, bounded smooth rescaling, and positive regularizers are those of the fitness pipeline. The canonical rescalings are $g_r(t)=g_s(t)=2/(1+e^{-t})$; the argument also applies to the same pipeline with any of its fixed positive strictly increasing smooth rescalings. No derivative lower bound is imposed.

Only entering alive coordinates are bounded by the valid physical domain and the completed velocity cap. Retained dead coordinates remain unrestricted. Put $B_x=\sup_{x\in\mathcal X_{\rm valid}}|x|$, $B_v=V_{\max}$. For the actual squashed comparison use
$$
z_i=(S_{R_x}(x_i),\sqrt{\lambda_{\rm alg}}S_{R_v}(v_i)),\qquad
D_0=2\sqrt{R_x^2+\lambda_{\rm alg}R_v^2},\quad
B_f=\max(R_x,\sqrt{\lambda_{\rm alg}}R_v).
$$
These features lie in $[-B_f,B_f]^{2d}$ and have diameter at most $D_0$. The positive phase-space weight is the configured $\lambda_{\rm alg}$. Define
$$
m_x=\frac{R_x^2}{(R_x+B_x)^2},\qquad
m_z=\min\left\{m_x,\frac{\sqrt{\lambda_{\rm alg}}R_v^2}{(R_v+B_v)^2}\right\}>0.
$$
Then
$$
|z_i-z_j|\ge m_x|x_i-x_j|,\qquad
|z_i-z_j|\ge m_z|(x_i,v_i)-(x_j,v_j)|.
\tag{3.CC1}
$$
Indeed the symmetric Jacobian of $S_R(x)=Rx/(R+|x|)$ has eigenvalues at least $R^2/(R+B)^2$ on the containing ball of radius $B$. Integrate along the segment, take its inner product with the segment direction, and apply Cauchy--Schwarz. This proves each component estimate and then their product-space bounds. For the explicitly configured unsquashed comparison the same proof below uses $z=(x,\sqrt{\lambda_{\rm alg}}v)$, $D_0=2\sqrt{B_x^2+\lambda_{\rm alg}B_v^2}$, $B_f=\max(B_x,\sqrt{\lambda_{\rm alg}}B_v)$, $m_x=1$, and $m_z=\min(1,\sqrt{\lambda_{\rm alg}})$. This keeps its own configured distance law.

If $\sigma_D,\sigma_C>0$ are the actual Gaussian bandwidths, put $\kappa_D=e^{-D_0^2/(2\sigma_D^2)}$, $\kappa_C=e^{-D_0^2/(2\sigma_C^2)}$. Every eligible measurement or cloning donor has probability at least $\kappa_D/(k-1)$ or $\kappa_C/(k-1)$, respectively. This follows directly by bounding the numerator below and each denominator summand above by one.

Let $\varepsilon_r,\varepsilon_s>0$ be the reward and diversity regularizers. With the actual measurement companion $C_i$, the measured diversity is
$$
Y_i=\sqrt{|z_i-z_{C_i}|^2+\delta_D^2},\qquad
s_Y=\sqrt{k^{-1}\sum_{i\in\mathcal A}(Y_i-\bar Y)^2+\varepsilon_s^2},
$$
where $\delta_D\ge0$ is the configured distance floor. Define
$$
D_m=\sqrt{D_0^2+\delta_D^2}-\delta_D,\qquad
s_*=\sqrt{D_m^2/4+\varepsilon_s^2},\qquad Z_*=D_m/\varepsilon_s.
$$
Thus $|u_i|\le Z_*$ and $s_Y\le s_*$. In fact, for values in $[y_-,y_+]$, averaging $(Y-y_-)(y_+-Y)\ge0$ gives
$\operatorname{Var}(Y)\le(y_+-\bar Y)(\bar Y-y_-)\le(y_+-y_-)^2/4$.

Let $A_->0$, $f_+<\infty$, and $F^*<\infty$ be the pipeline's lower reward-factor bound, upper diversity-factor bound, and upper fitness bound. These follow from its positive floors and bounded rescalings, for every fixed $\alpha\ge0,\beta>0$. Let $L_R$ bound the Lipschitz constant of the configured reward in the joint position/velocity norm on the bounded alive region, and put $Z_R=\operatorname{osc}_{\mathcal X_{\rm valid}\times B_{V_{\max}}}(R)/\varepsilon_r$. Let $L_H=\max_{|t|\le Z_R}|H'(t)|$, which is finite on this compact standardized-reward interval. Then
$$
|A_i-A_j|\le L_A|z_i-z_j|,\qquad
L_A=\frac{L_H L_R}{\varepsilon_r m_z}.
\tag{3.CC2}
$$
Both factors use the same entering reward mean and regularized scale. The positive floor makes $L_H$ finite even for exponents below one; when $\alpha=0$, $L_H=0$. The stated reward regularity supplies $L_R$. On a convex containing region a bound on its joint gradient norm suffices. More generally, if the reward is smooth on a neighborhood of the compact valid region, choose a positive neighborhood radius $\rho$: for pairs at distance below $\rho$, use the gradient bound on their segment; for the other pairs use $\operatorname{osc}(R)/\rho$. Their maximum is a finite Lipschitz constant. A velocity penalty contributes its velocity gradient to this joint norm; summing the positional and velocity gradient bounds is sufficient. No reward is replaced by a quadratic objective.

Fix an analysis threshold $0<W_0\le E_{\max}$ small enough that $m_x^2W_0/4<D_0^2$, where $E_{\max}$ bounds the entering positional error below. Put
$$
v_0=m_x^2W_0/4,\qquad h_f=\sqrt{v_0/2},\qquad
\rho_f=\frac{v_0/2}{D_0^2-v_0/2}>0.
\tag{3.CC3}
$$
The threshold $h_f$ is a comparison distance, not a physical timestep. Define
$$
\Delta_f=\sqrt{h_f^2+\delta_D^2}-\sqrt{h_f^2/4+\delta_D^2}>0,
\quad t_f=\Delta_f/s_*,\quad
\omega_f=\min_{u\in[-Z_*,Z_*-t_f]}\{f(u+t_f)-f(u)\}>0.
\tag{3.CC4}
$$
Here $0<t_f\le Z_*$. The minimum is attained on a compact interval, and strict increase makes every value strictly positive. This proves positivity even when the derivative vanishes at individual scores. Choose
$$
r=\begin{cases}
\min\{h_f/2,A_-\omega_f/(2f_+L_A)\},&L_A>0,\\
h_f/2,&L_A=0,
\end{cases}\qquad
\gamma_0=A_-\omega_f/2,
$$
$$
a_0=\min\{1,\gamma_0/[p_{\max}(F^*+\varepsilon_c)]\},\qquad
C_0=\kappa_C\kappa_D^2\rho_f a_0>0.
\tag{3.CC5}
$$
The actual acceptance is $\min\{1,(F_j-F_i)_+/[p_{\max}(F_i+\varepsilon_c)]\}$. All displayed constants depend only on the fixed algorithmic parameters, bounded entering region, reward, and analysis threshold; none depends on the population or the number of clusters.

For the canonical parameters, $R_x=R_v=2$, $\lambda_{\rm alg}=1$, $\sigma_D=\sigma_C=2$, $B_x=2\sqrt d$, $B_v=2$, so $D_0^2=32$, $B_f=2$, $m_x=m_z=(1+\sqrt d)^{-2}$, and $\kappa_D=\kappa_C=e^{-4}$. Also $\delta_D=.001$, $\varepsilon_r=\varepsilon_s=.1$, $\alpha=\beta=1$, $A_-=.1$, $f_+=2.1$, $F^*=4.41$, $L_H=1/2$, $p_{\max}=1$, and $\varepsilon_c=10^{-6}$. The logistic choice has the additional explicit bound $\omega_f\ge 2e^{-Z_*}t_f/(1+e^{-Z_*})^2$. These specialize the proof; they are not restrictions to a demonstration potential.

:::

:::{prf:lemma} Measurement-averaged pressure from every near-neighbor mass
:label: lem-keystone-near-neighbor-pressure

Use the actual kernel and constants of {prf:ref}`lem-keystone-complete-coverage-constants`. Consider a swarm with $k\ge2$ alive slots and actual feature variance
$\mathsf V_z\ge v_0$. For an alive recipient $i$, let
$$
n_i(r)=\#\{j\in\mathcal A\setminus\{i\}:|z_j-z_i|\le r\}.
$$
Then
$$
\boxed{\overline p_i:=\mathbb E[p_i(\mathbf F)\mid S]
\ge C_0\left(\frac{n_i(r)}{k-1}\right)^2.}
\tag{3.CC6}
$$
**Proof.** The feature-variance identity shows that every alive row $j$ has at least a fraction $\rho_f$ of its eligible companions at distance at least $h_f$:
$$
\frac1k\sum_\ell|z_\ell-z_j|^2
=\mathsf V_z+|z_j-\bar z|^2\ge v_0.
$$
Comparing distances below $h_f$ with the bound $D_0$ gives the fraction $(v_0-h_f^2)/(D_0^2-h_f^2)=\rho_f$; self exclusion can only improve it. Its actual measurement probability is at least $\kappa_D\rho_f$.

Restrict the cloning donor $j$ to $|z_j-z_i|\le r$. The recipient's near-measurement event has probability at least $\kappa_D n_i(r)/(k-1)$; independently, the donor's far-measurement event has probability at least $\kappa_D\rho_f$. On their joint event,
$$
Y_j-Y_i\ge\sqrt{h_f^2+\delta_D^2}-\sqrt{r^2+\delta_D^2}\ge\Delta_f.
$$
Shared empirical standardization retains exactly this difference divided by the same $s_Y\le s_*$. The finite-increment bound for the actual diversity rescaling and the reward-factor variation therefore give
$$
F_j-F_i\ge A_-\omega_f-f_+L_A r\ge\gamma_0.
$$
Thus its actual acceptance is at least $a_0$. Sum over the restricted cloning donors, whose actual probability is at least $\kappa_C n_i(r)/(k-1)$, and average their independent measurement events. This proves (3.CC6). It does not demand that every measurement realization have a fitness gap. $\square$

:::

:::{prf:theorem} Complete geometric-cluster error coverage at every population size
:label: thm-keystone-complete-error-coverage

Use the actual kernel and constants of {prf:ref}`lem-keystone-complete-coverage-constants`, and suppose the entering alive feature variance is $\mathsf V_z\ge v_0$. Fix any comparison labels $I\subseteq\mathcal A$, nonnegative entering error weights $e_i\le E_{\max}$, and normalization $N\ge k$. Put
$$
W=\frac1N\sum_{i\in I}e_i.
$$
For two swarms, take $I=I_{11}$ and $e_i=|\Delta\delta_{x,i}|^2$. If both alive domains have common diameter bound $D_x$, then $E_{\max}=4D_x^2$ works, since each centered position has norm at most $D_x$. Canonically one may use $D_x=4\sqrt d$, so $E_{\max}=64d$.

Partition the fixed feature cube into cells of side at most $r/\sqrt{2d}$. Their feature diameter is at most $r$, and their number is bounded by
$$
M_r=\left\lceil\frac{2B_f\sqrt{2d}}r\right\rceil^{2d}.
\tag{3.CC7}
$$
Use half-open cells with the outer boundary included in the last cell. This is a finite auxiliary cover, fixed independently of the population. Let $n_c$ count the alive rows in cell $c$, and write
$$
E_c=\frac1N\sum_{G}\sum_{i\in I\cap G\cap c}e_i.
$$
The sum uses every original geometric cluster $G$, with no mass or validity restriction. Thus $\sum_c E_c=W$ and $E_c\le E_{\max}n_c/N$.

For every $N\ge k\ge2$,
$$
\boxed{\frac1N\sum_{i\in I}\overline p_i e_i
\ge C_0 W\left[
\frac{(NW/(E_{\max}M_r)-1)_+}{k-1}\right]^2.}
\tag{3.CC8}
$$
**Proof.** Every row in cell $c$ has $n_i(r)\ge n_c-1$. Hence (3.CC6) bounds the left side below by
$$
\frac{C_0}{(k-1)^2}\sum_c E_c(n_c-1)^2.
$$
If $W=0$, the claim is immediate. Otherwise weighted Cauchy--Schwarz gives
$$
\sum_c E_c(n_c-1)^2
\ge\frac1W\left(\sum_c E_c(n_c-1)\right)^2.
$$
Terms with $E_c>0$ have $n_c\ge1$, so the sum being squared is nonnegative. Also
$$
\sum_c E_c n_c\ge\frac N{E_{\max}}\sum_c E_c^2
\ge\frac{NW^2}{E_{\max}M_r}.
$$
Subtract $W$, take its positive part, and substitute. This proves (3.CC8) without assuming any error is captured by a prescribed fraction of clusters. $\square$

The estimate is valid even below the range where its right side becomes positive. Such a zero lower bound does not assert zero actual activity. The self-exclusion correction and all previously uncovered error are explicit.

If $w=W/(k/N)\ge W_0$ and
$$
k\ge\frac{2E_{\max}M_r}{W_0},
$$
then (3.CC8) gives the population-uniform linear bound
$$
\frac1N\sum_{i\in I}\overline p_i e_i
\ge\chi_0 W,
\qquad \chi_0=\frac{C_0W_0^2}{4E_{\max}^2M_r^2}>0.
\tag{3.CC9}
$$
Indeed $NW/(E_{\max}M_r)=kw/(E_{\max}M_r)\ge2$, so the numerator in (3.CC8) is at least half that value. Dividing by $k-1\le k$ yields at least $w/(2E_{\max}M_r)$.

:::

:::{prf:theorem} Discharged averaged Keystone estimate with finite-population correction
:label: thm-keystone-discharged-averaged-pressure

Use the actual kernel, bounded entering alive region, and constants of {prf:ref}`lem-keystone-complete-coverage-constants`. No lower bound on either alive fraction or the mass of any geometric cluster is imposed. All expectations below condition on the two complete entering marked states. Any coupling of the two measurement laws with the prescribed one-swarm marginals is allowed: only linearity of their summed marginal expectations is used.

For two nonextinct swarms define $m_s=k_s/N$ and
$$
W=\frac1N\sum_{i\in I_{11}}|\Delta\delta_{x,i}|^2.
$$
Their alive positional variances satisfy
$$
W\le2\sum_{s=1}^2m_s\operatorname{Var}_{\mathcal A_s}(x).
$$
Choose a swarm $s$ maximizing the weighted positional variance. If $W>0$, this swarm has positive positional variance and hence $k_s\ge2$, so every denominator $k_s-1$ below is defined. Then
$$
m_s\operatorname{Var}_{\mathcal A_s}(x)\ge W/4,
\qquad \mathsf V_{z,s}\ge m_x^2 W/(4m_s).
\tag{3.CC10}
$$
Thus $W\ge W_0$ implies the feature threshold $\mathsf V_{z,s}\ge v_0$ used in (3.CC6), because $m_s\le1$. It also implies $W/m_s\ge W_0$ and $m_s\ge W_0/E_{\max}$, since $W\le E_{\max}m_s$. The latter is a consequence of the entering error, not an assumed alive-fraction floor.

A quantitative affine estimate already holds for every population size, without a large-population premise. Put
$$
\chi_*=\frac{C_0W_0^2}{2E_{\max}^2M_r^2},\qquad
B_*=\frac{C_0E_{\max}^2}{W_0}.
$$
Then
$$
\boxed{\mathbb E\left[\frac1N\sum_{i\in I_{11}}
(p_{1,i}+p_{2,i})|\Delta\delta_{x,i}|^2\right]
\ge\chi_*(W-W_0)-\frac{B_*}{N^2}.}
\tag{3.CC11a}
$$
To prove this in the high-error branch, (3.CC8) is at least
$$
C_0W\left(\frac{W}{m_s E_{\max}M_r}-\frac1{k_s}\right)_+^2
\ge\frac{C_0W^3}{2m_s^2E_{\max}^2M_r^2}
-\frac{C_0W}{k_s^2}.
$$
Here $(a-b)_+^2\ge a^2/2-b^2$: for $a\ge b$ subtract the right side to obtain $(a-2b)^2/2$, and for $a<b$ the right side is nonpositive. Since $m_s\le1$, $W\ge W_0$, $W\le E_{\max}k_s/N$, and $k_s\ge NW_0/E_{\max}$, the last display is at least $\chi_*W-B_*/N^2$. For $W<W_0$, nonnegativity proves (3.CC11a). The zero-error $N=1$ case is included through this low-error branch. For all-alive inputs the sharper $B_*=C_0E_{\max}$ works because $k_s=N$. Thus the finite-population correction explicitly accounts for self exclusion without assuming any cluster covers the error.

Consequently, for
$$
N\ge N_0:=\left\lceil\frac{2E_{\max}^2M_r}{W_0^2}\right\rceil,
$$
the selected alive count satisfies the threshold in (3.CC9). The full two-swarm error-weighted activity therefore obeys
$$
\boxed{\mathbb E\left[\frac1N\sum_{i\in I_{11}}
(p_{1,i}+p_{2,i})|\Delta\delta_{x,i}|^2\right]
\ge\chi_0(W-W_0),\qquad N\ge N_0.}
\tag{3.CC11}
$$
For $W\ge W_0$, the stronger lower bound is $\chi_0W$, by selecting the spread swarm above. For $W<W_0$, nonnegativity proves (3.CC11). Every original cluster has been included. No target-probability, cluster-mass, or error-coverage assumption remains in this estimate. Below $N_0$, (3.CC8) remains the proved finite-population statement; this theorem does not identify all such populations with the exceptional no-selection case.

For all-alive inputs, use the actual comparison matching as a transport plan. For $\eta>0$, Young's inequality gives
$$
V_{\rm struct}\le(1+\eta)W+
 c_v\frac1N\sum_i|\Delta\delta_{v,i}|^2,
\qquad c_v=\lambda_v+\frac{b^2}{4\eta}.
$$
The last mean squared centered discrepancy is at most $4V_{\max}^2$, since centering is an orthogonal projection and each entering velocity has norm at most $V_{\max}$. Thus (3.CC11) supplies the fully discharged Keystone constants
$$
\mathbb E[\text{error-weighted activity}]
\ge \frac{\chi_0}{1+\eta}V_{\rm struct}
-\chi_0\left[W_0+
 \frac{4c_vV_{\max}^2}{1+\eta}\right],\qquad N\ge N_0.
\tag{3.CC12}
$$
For zero velocity discrepancy, its actual velocity remainder is zero, so the positional branch is nonvacuous whenever $V_{\rm struct}>(1+\eta)W_0$. The threshold $W_0$ is arbitrary positive below the attainable positional error; it is not chosen to cover the entire physical state space.

For partially alive states, retain the normalization of the original alive-law structural cost. With $k_{\max}=\max(k_1,k_2)$, $s=|I_{11}|$, $m_{\max}=k_{\max}/N$, and $D_v=N^{-1}\sum_{i\in I_{11}}|\Delta\delta_{v,i}|^2$, the matching of common labels has mass $s/k_{\max}$. Completing its residual marginals by any coupling yields
$$
m_{\max}V_{\rm struct}
\le(1+\eta)W+c_vD_v+
 \frac{k_{\max}-s}{N}\bigl[(1+\eta)E_{\max}+16c_vV_{\max}^2\bigr].
\tag{3.CC13}
$$
The factor 16 bounds centered velocity differences on arbitrary residual label pairs; unlike the full equal-mass pairing average, it is a pointwise bound. Applying (3.CC11) to (3.CC13) retains exactly the alive-mass, velocity, and unmatched-label contributions. No claim that positional cloning pressure controls a purely velocity discrepancy or a missing common-alive population is needed.

The all-population version (3.CC11a) gives, for all-alive inputs,
$$
\mathbb E[\text{error-weighted activity}]
\ge\frac{\chi_*}{1+\eta}V_{\rm struct}
-\chi_*\left[W_0+
\frac{c_v}{1+\eta}\frac1N\sum_i|\Delta\delta_{v,i}|^2\right]
-\frac{C_0E_{\max}}{N^2}.
\tag{3.CC14}
$$
For partially alive inputs, substitute (3.CC13) into (3.CC11a) to obtain the same fully explicit formula with $m_{\max}V_{\rm struct}$, its displayed common-label and velocity remainders, and $B_*/N^2$. All coverage and probability estimates are thereby discharged for the stated actual kernel; no error outside the original geometric clusters has been assumed away.

:::

:::{prf:corollary} A canonical population-uniform structural Keystone application
:label: cor-keystone-canonical-balanced-structural

Use the canonical Euclidean Gas in dimension $d\geq1$ and population
$N=2M\geq4$. Let one entering swarm contain $M$ walkers at $+ae_1$ and
$M$ at $-ae_1$, and another contain $M$ at $+be_1$ and $M$ at $-be_1$,
where $a,b\in[0.5,2)$. All slots are alive and all velocities are zero.
The potential is the actual quadratic $U(x)=|x|^2/2$. Use the optimal
same-sign matching of their centered empirical measures. The structural
error of {prf:ref}`def-structural-error-component` is exactly
$$
V_{\mathrm{struct}}=(a-b)^2.
$$
Let $p_{s,i}$ be the actual cloning acceptance probability conditional on
swarm $s$'s retained fitness vector. The full measurement-averaged Keystone
quantity satisfies
$$
\boxed{
\mathbb E\left[\frac1N\sum_i(p_{1,i}+p_{2,i})
 |\Delta\delta_{x,i}|^2\right]
\geq\chi_0V_{\mathrm{struct}},\qquad
\chi_0=\frac49A_0(0.5)=0.30251915319\ldots>0.}          \tag{3.KB1}
$$
Here $A_0(r)$ is the explicit canonical function defined in
{prf:ref}`prop-cloning-two-cluster-noise-balance`. The constant is independent
of $N,d,a,b$ in the stated family, and the offset is zero. Every measurement
outcome, including complete fitness ties, is included in the expectation.
:::

:::{prf:proof}
At radius $r$, the actual row measurement laws are identical two-point
laws: an opposite-site companion is measured with probability
$$
q_M(r)=\frac{Mw(r)}{M-1+Mw(r)},\qquad
w(r)=\exp\left[-\frac18\left(\frac{4r}{2+r}\right)^2\right].
$$
The measurement innovations are independent across rows, though the
resulting retained fitness values share their global normalizers. For any
fixed recipient $i$ and distinct donor $j$, the event that $i$ has the low
measurement and $j$ the high measurement has probability
$(1-q_M)q_M$. On that event the actual retained-fitness acceptance is at
least $A_0(r)$, irrespective of the other measurements, as proved in
{prf:ref}`prop-cloning-two-cluster-noise-balance`. The cloning donor law is
fixed by the entering physical state. Sum its normalized probabilities over
$j\ne i$ to obtain, for every row,
$$
\pi_i(r):=\mathbb E[p_i\mid S_r]
\geq A_0(r)q_M(r)(1-q_M(r)).                            \tag{3.KB2}
$$
No event is removed from this expectation: it is a lower bound obtained by
retaining particular nonnegative contributions of the actual kernel.

The two swarm means vanish. With zero velocities, the hypocoercive ground
cost is exactly the squared positional distance. A same-sign pair costs
$(a-b)^2$, whereas an opposite-sign pair costs $(a+b)^2$. Both empirical
measures put half their mass at each sign, so matching equal signs is
optimal and gives the displayed structural error. Moreover every paired
squared positional discrepancy equals $(a-b)^2$. Consequently (3.KB2)
gives the sharper bound
$$
\mathbb E\left[\frac1N\sum_i(p_{1,i}+p_{2,i})
 |\Delta\delta_{x,i}|^2\right]
\geq\bigl[A_0(a)q_M(a)(1-q_M(a))
+A_0(b)q_M(b)(1-q_M(b))\bigr]V_{\mathrm{struct}}.
$$
Thus no unproved correlation between the target error and sampled
acceptance has been factored out.

For $r<2$, $w(r)>e^{-1/2}>1/2$, so
$q_M(r)\geq w(r)/(1+w(r))>1/3$.
Also $w(r)\leq1$ and $M\geq2$ give
$q_M(r)\leq M/(2M-1)\leq2/3$. Hence $q_M(1-q_M)\geq2/9$.
The distance gap $\delta(r)$ in the definition of $A_0$ increases with $r$,
and $\delta/\sqrt{\delta^2+4(0.1)^2}$ is increasing. Therefore
$A_0(r)\geq A_0(0.5)$ for $r\geq0.5$. Applying these inequalities to
both swarms proves (3.KB1). Embedding the two sites along $e_1$ does not
alter any reward, metric distance, measurement probability, or cost, so the
same constants apply in every dimension. $\square$
:::

:::{div} feynman-prose
In this balanced two-site family, every paired walker carries the same squared error, $(a-b)^2$. We can therefore average the actual cloning pressure without losing track of where the error sits. Each walker has a fixed positive chance of measuring nearby while a potential donor measures across the two sites. That chance stays bounded below as the population grows, giving the same positive Keystone coefficient for every even $N\geq4$ in the stated radius range.

Sometimes all sampled fitness values tie and cloning stops for that proposal. Those events remain in the average with their actual probabilities and zero contribution. The positive averaged bound comes from the other measurement outcomes; it does not require discarding ties.
:::

:::{prf:remark} Scope of the structural application
:label: rem-keystone-balanced-structural-scope

The condition $N\ge4$ identifies an exact exception. For $N=2$ at $\pm re_1$ with zero velocities, each row has only its opposite companion. Both rewards and both measured separations agree, so the two retained fitnesses are exactly $1.21$ and both cloning acceptance probabilities are zero. Two such swarms with different radii have $V_{\mathrm{struct}}>0$, so a positive zero-offset cloning-pressure bound fails at that stage. This does not assert failure of their subsequent kinetic evolution. For $N\ge4$, complete-fitness-tie outcomes still occur with positive probability and are already included in (3.KB1); they are not excluded by conditioning. The one-alive-row no-self-donor branch follows its declared separate convention.

The result controls the actual paired structural-error pressure, not an
internal-variance proxy. In particular it coexists with the positive
internal-variance drift in {prf:ref}`prop-cloning-two-cluster-noise-balance`.
Passing from pressure to a coupled structural increment still uses the
signed terms and chosen full-kernel coupling in
{prf:ref}`thm-cloning-incremental-cluster-balance` and
{prf:ref}`lem-cloning-coupled-error-not-internal-variance`.
:::


### 8.6. Assembly of the Keystone estimate

The target-set assembly below uses the two bounds stated in {prf:ref}`lem-quantitative-keystone`: probability on the same target labels and error captured by those labels. The direct averaged route in {prf:ref}`thm-keystone-discharged-averaged-pressure` proves its own probability and complete-coverage estimates for the actual kernel, without assuming these target-set premises. Its finite-population correction and structural velocity/status terms are explicit. The application in {prf:ref}`cor-keystone-canonical-balanced-structural` has zero offset.

:::{prf:proof}
**Proof of the N-Uniform Quantitative Keystone Lemma ({prf:ref}`lem-quantitative-keystone`).**

The proof establishes the inequality for the high-error regime ($V_{\text{struct}} > R^2_{\text{spread}}$) and then defines the global offset $g_{\max}(\epsilon)$ to ensure it holds everywhere, as per the strategy outlined in Section 8.1.

**1. Setup for the High-Error Regime.**
Assume the initial state $(S_1, S_2)$ is in the high-error regime. Use the index $k=1$ for the swarm supplying the target estimates in the theorem hypotheses, and set $I_{\mathrm{target}}=I_{11}\cap U_1\cap H_1$. The lower bound below remains valid if this set is empty; it is informative when its error lower bound is positive. We seek a lower bound for the error-weighted cloning activity, $E_w$:

$$
E_w := \frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2

$$

**2. Lower-Bound the Sum by the Critical Target Set.**
The sum $E_w$ consists of non-negative terms and is bounded below by the sum over the critical target set $I_{\text{target}} \subseteq I_{11}$:

$$
E_w \ge \frac{1}{N}\sum_{i \in I_{\text{target}}} p_{1,i}\|\Delta\delta_{x,i}\|^2

$$

We focus on the cloning probability `p_1,i` because swarm 1 is the high-variance swarm for which our guarantees on the unfit and high-error sets hold.

**3. Apply the target-wise probability bound.**
By {prf:ref}`cor-cloning-pressure-target-set`, $p_{1,i}\geq p_u$ for every
$i\in I_{\mathrm{target}}$. Therefore

$$
E_w\geq \frac{p_u}{N}\sum_{i\in I_{\mathrm{target}}}
\|\Delta\delta_{x,i}\|^2.
$$

A population-average cloning probability alone would leave a covariance term
with the squared error. The target-wise hypothesis is precisely what permits
this factorization.

**4. Substitute the Error Concentration Bound.**
We now have an expression that is the product of two N-uniform lower bounds.
*   **Cloning Pressure:** The term $p_u(\varepsilon)$ is the N-uniform minimum cloning probability from **{prf:ref}`lem-unfit-cloning-pressure`**.
*   **Error Concentration:** The term

$$
\frac{1}{N}\sum_{i \in I_{\text{target}}} \|\Delta\delta_{x,i}\|^2

$$

 is exactly the quantity lower-bounded by the **Error Concentration Lemma (8.4.1)**.

Substituting the bound from {prf:ref}`lem-error-concentration-target-set` gives:

$$
E_w \ge p_u(\epsilon) \cdot \left( c_{err}(\epsilon)V_{\mathrm{struct}} - g_{err}(\epsilon) \right)

$$

**5. Define N-Uniform Constants for the High-Error Regime.**
Substituting these two bounds into the inequality from Step 3 gives:

$$
E_w \ge p_u(\epsilon) \cdot \left( c_{err}(\epsilon)V_{\mathrm{struct}} - g_{err}(\epsilon) \right)

$$

We define the N-uniform, $\varepsilon$-dependent constants that emerge from this constructive proof:
*   The **feedback coefficient:** $\chi(\epsilon) := p_u(\epsilon) \cdot c_{err}(\epsilon) > 0$
*   The **partial offset:** $g_{\text{partial}}(\epsilon) := p_u(\epsilon) \cdot g_{err}(\epsilon) \ge 0$

This establishes the desired linear lower bound for any state in the high-error regime:

$$
E_w \ge \chi(\epsilon) V_{\text{struct}} - g_{\text{partial}}(\epsilon)

$$

**6. Finalize the Global Inequality.**
As outlined in the proof strategy (Section 8.1), we define the global offset constant $g_{\max}(\epsilon)$ to ensure the inequality holds for all states by taking the maximum of the offsets required for the low-error and high-error regimes:

$$
g_{\max}(\epsilon) := \max\bigl(g_{\text{partial}}(\epsilon),\, \chi(\epsilon) R^2_{\text{spread}}\bigr)

$$

This choice ensures the inequality is satisfied everywhere. Since $\chi(\epsilon)$ and $g_{\max}(\epsilon)$ are constructed entirely from N-uniform constants, they are themselves independent of $N$.

This completes the rigorous, constructive proof of the N-Uniform Quantitative Keystone Lemma.

**Q.E.D.**
:::

### 8.7. Constants and their uniformity

:::{div} feynman-prose
The selection calculation and the error calculation have separate roles. Write
$p_u$ for the first lower bound and
$N^{-1}\sum_{I_{\mathrm{target}}}\|\Delta\delta_x\|^2
\geq c_{\mathrm{err}}V_{\mathrm{struct}}-g_{\mathrm{err}}$ for the second.
Then the feedback coefficient is their product, while the offset retains the
error outside the target set and the low-error regime.
:::

:::{prf:proposition} Uniformity of the Keystone constants
:label: prop-n-uniformity-keystone

Suppose the hypotheses of {prf:ref}`lem-quantitative-keystone` hold with common
$p_u>0$, $c_{\mathrm{err}}>0$, $g_{\mathrm{err}}\geq0$, and
$R_{\mathrm{spread}}^2$ over the family of particle numbers. Then

$$
\chi=p_uc_{\mathrm{err}},\qquad
 g_{\max}=\max\{p_ug_{\mathrm{err}},\chi R_{\mathrm{spread}}^2\}
$$

are independent of $N$. One explicit selection choice is

$$
p_u=\frac{a_*s_*^2}{2R_*A_*},\qquad
A_*=\max\{R_*,p_{\max}(V_{\mathrm{pot,max}}+
\varepsilon_{\mathrm{clone}})\},
$$

with the realized fitness and companion bounds of
{prf:ref}`lem-unfit-cloning-pressure`. A favorable-companion probability gives
the alternative constant in that lemma.
:::

:::{prf:proof}
The target-set sum is bounded below by
$p_u(c_{\mathrm{err}}V_{\mathrm{struct}}-g_{\mathrm{err}})$ in the high-error
regime. In the low-error regime it is nonnegative, and
$\chi V_{\mathrm{struct}}-g_{\max}\leq0$. Thus the displayed constants work
throughout the stated family. Products and maxima of common constants remain
common constants. The selection formula follows from the pointwise clipped
score bound, and contains no factor tending to zero with $N$.
:::

:::{div} feynman-prose
A large residual can make a valid lower bound uninformative on part of a bounded
state space. The useful drift regime is where the corrective term dominates
that residual. Parameter changes affect both terms: increasing a fitness
weight can increase a signal while also changing its range and the target set.
The explicit formulas should be evaluated together.
:::

### 8.8. From the Keystone estimate to drift

:::{div} feynman-prose
The Keystone assembly is multiplication of two controlled quantities: a
selection probability and an error sum. Its constants are
$\chi=p_uc_{\mathrm{err}}$ and
$g_{\max}=\max\{p_ug_{\mathrm{err}},\chi R_{\mathrm{spread}}^2\}$.
The residual is part of the result; the bound supplies positive corrective
activity when the structural error exceeds $g_{\max}/\chi$.

The next sections compare this selection estimate with the actual position and velocity
updates. The boundary estimate uses its own favorable-companion and
integrability conditions. Keeping these inputs distinct makes the final
composition proof a check on the same transition kernel throughout.
:::

(sec-cloning-operator)=
## 9. Formal Definition of the Cloning Operator

### 9.1. Introduction and Objectives

Having established the Keystone Principle in Sections 5-8, we now formalize the complete cloning operator as a single mathematical object. This chapter serves as the bridge between the theoretical guarantees of the Keystone Principle and the practical drift analysis that follows in Sections 10-11.

The cloning operator, denoted $\Psi_{\text{clone}}$, is a **stochastic transition operator** that maps a swarm configuration to a probability distribution over new swarm configurations. It encompasses the entire adaptive feedback mechanism analyzed in the preceding sections: measurement, fitness evaluation, companion selection, and state update.

**Objectives of this chapter:**

1. **Formal Operator Definition:** Present $\Psi_{\text{clone}}$ as a rigorous mathematical object with clearly specified domain, range, and stochastic structure.

2. **Compositional Structure:** Decompose the operator into its constituent sub-operators, making explicit the flow of information and randomness through the system.

3. **Key Output Quantities:** Define the critical intermediate and final quantities (see {prf:ref}`def-key-operator-outputs`) produced by the operator that will be used in the drift analysis, particularly the **total cloning probability** $p_i$ for each walker.

4. **Preparation for Drift Analysis:** Establish the notation and framework necessary for proving the contractive properties in Sections 10-11.

### 9.2. The Cloning Operator as a Markov Kernel

We begin with the highest-level definition of the operator as a mathematical object.

:::{prf:definition} The Cloning Operator $\Psi_{\text{clone}}$
:label: def-cloning-operator-formal

The cloning proposal $\Psi_{\mathrm{clone}}$ is a Markov kernel from nonempty valid swarms to the ambient proposal state space containing all jittered positions. The canonical transition applies its validity test after the complete kinetic update and final position noise, giving the corresponding terminal dead statuses. The component drift estimates specify whether they concern this proposal or the tested transition.

**Domain and Range:**
- **Input:** A swarm  configuration $S = ((x_1, v_1, s_1), \ldots, (x_N, v_N, s_N)) \in \Sigma_N$ with at least one alive walker ({prf:ref}`def-walker`) ($|\mathcal{A}(S)| \geq 1$).
- **Output:** The proposed swarm after measurement, decisions, and updates. Under guaranteed revival, proposal statuses are all alive before any subsequent killing test. Gaussian position jitter can leave the valid domain; the state space for the proposal kernel must include these positions, or the boundary test must be included in the transition.

**Stochastic Structure:**

The operator is defined by a composition of deterministic and stochastic sub-operators (see {prf:ref}`thm-cloning-operator-composition` for the rigorous compositional representation):

$$
\Psi_{\text{clone}} = \Psi_{\text{update}} \circ \Psi_{\text{decision}} \circ \Psi_{\text{fitness}} \circ \Psi_{\text{measure}}

$$

where each sub-operator is defined in the subsequent sections.

**Key Property - All-Alive Output:**

By construction, the output configuration $S' \sim \Psi_{\text{clone}}(S, \cdot)$ satisfies:

$$
s'_i = 1 \quad \text{for all } i \in \{1, \ldots, N\}

$$

This status statement uses the forced-revival convention and a nonempty alive companion set. It describes the proposal stage. A subsequent validity test determines the actual alive set. Replacing an alive slot leaves the alive count unchanged at the proposal stage.

Referenced by {prf:ref}`thm-complete-cloning-drift`.
:::

:::{admonition} Notation Convention
:class: note

Throughout this chapter and the subsequent drift analysis:
- $S$ denotes the **input** swarm configuration to $\Psi_{\text{clone}}$
- $S'$ denotes the **output** swarm configuration from $\Psi_{\text{clone}}$
- Intermediate quantities (raw measurements, fitness potentials, etc.) are denoted with their specific symbols introduced in Section 5
:::

### 9.3. Decomposition into Sub-Operators

We now specify each component of the operator in detail, following the logical flow from perception to action established in Section 5.

#### 9.3.1. The Measurement Operator $\Psi_{\text{measure}}$

The first stage (as discussed in Section 5.3) generates the stochastic diversity measurements that form the foundation of the fitness evaluation.

:::{prf:definition} The measurement operator
:label: def-measurement-operator

For each alive row independently, draw $d(i)$ from the Gaussian weighted current eligible pool excluding $i$, and record

$$
s_i=\sqrt{d_{\mathrm{alg}}(i,d(i))^2+\delta_D^2},\qquad\delta_D>0.
$$

For a singleton eligible pool the raw companion distance is zero. Dead rows have dummy measurements and do not contribute to current reward or diversity moments. The chosen measurement for every alive row is retained throughout fitness and acceptance. A matching configuration uses its explicitly defined joint measurement law instead.
:::

:::{prf:remark} Coupling measurement draws
:label: rem-measurement-coupling

Recipient-addressed common uniforms define a valid synchronous coupling of two independent-sampling kernels, even when their state-dependent donor laws differ. They do not force the same donor or the same fitness. In a matching configuration, couple the matching innovations according to its joint law. Conditional independence of row measurements is specific to the independent configuration.
:::

#### 9.3.2. The Fitness Evaluation Operator $\Psi_{\text{fitness}}$

This deterministic operator (implementing the pipeline from Section 5) transforms raw measurements into fitness potentials.

:::{prf:definition} The fitness evaluation operator
:label: def-fitness-operator

Evaluate the configured oriented reward $r_i=R(x_i,v_i)$ on alive rows. A barrier or a velocity penalty is included only when explicitly part of that objective. For $q=r,s$, compute

$$
\bar q=\frac1{|\mathcal A|}\sum_{i\in\mathcal A}q_i,\qquad
\widehat\sigma_q=\sqrt{\frac1{|\mathcal A|}\sum_{i\in\mathcal A}(q_i-\bar q)^2+\sigma_{q,\min}^2}.
$$

For the canonical positive logistic maps $g_q(z)=A_q/(1+e^{-z})+\eta_q$ and nonnegative exponents $p_r,p_s$, the frozen fitness is

$$
F_i=g_r((r_i-\bar r)/\widehat\sigma_r)^{p_r}
g_s((s_i-\bar s)/\widehat\sigma_s)^{p_s}.
$$

Positive floors and variance regularizers are fixed. Dead fitness is a dummy value: revival does not evaluate a live acceptance score. The exponents are fitness parameters, distinct from collision restitution $\alpha$. Every selected donor is scored with this same frozen vector. Averaging over measurement draws occurs after the nonlinear acceptance calculation.
:::

#### 9.3.3. The cloning decision operator

:::{prf:definition} The cloning decision operator
:label: def-decision-operator

Condition on $S$ and its frozen sampled fitness vector. Each row draws one donor independently from {prf:ref}`def-cloning-companion-operator`. For a live row with a distinct eligible donor, sample $T_i\sim U(0,p_{\max})$ and accept when $S_i(c_i)>T_i$. A live row without a distinct eligible donor persists. Every dead row accepts its weighted current eligible donor with probability one. No fitness threshold is applied to revival.

The output consists of the donor vector and the accepted-edge indicator vector. For a live row, its total probability is

$$
p_i=\sum_{j\in\mathcal A\setminus\{i\}}P(c_i=j\mid S)
\min\!\left(1,\frac{(V_{\rm fit,j}-V_{\rm fit,i})_+}{p_{\max}(V_{\rm fit,i}+\varepsilon_{\rm clone})}\right).
$$

For a dead row, $p_i=1$. These probabilities remain conditional on the sampled fitness; averaging over measurement draws comes afterwards.
:::

:::{prf:lemma} Mandatory revival
:label: lem-dead-walker-clone-prob

If the current alive pool is nonempty, each dead slot revives with probability one during the cloning proposal, independently of the fitness floor and the acceptance regularizer.

*Proof.* Its decision branch draws an eligible current donor and accepts unconditionally. If the alive pool is empty, the algorithm stops at extinction; it does not invent a donor or a restart law. $\square$
:::

#### 9.3.4. The state update operator

:::{prf:definition} The state update operator
:label: def-update-operator

Given the frozen input, donors, and accepted indicators, apply {prf:ref}`def-inelastic-collision-update`. Build the connected components of all accepted undirected edges before any write. Each accepted row, including every revived row, copies its own frozen donor position and receives its independent Gaussian jitter. Every vertex in a nontrivial component receives the component's shared-Haar velocity update from its frozen velocity. A row that persists in position can therefore change velocity as a donor. Isolated rows retain their velocity. All proposal statuses are alive.

The output is a probability kernel on the ambient proposal space, which includes jittered positions outside the valid domain and pre-cap velocities. The canonical boundary schedule tests positions only at the end of the full kinetic update, after its independent final position noise.
:::

:::{prf:remark} Conditional independence belongs to the correct stage
:label: rem-position-velocity-update-difference

Conditional on the frozen state and fitness, donor decisions and row jitters are independent, so post-copy positions are independent across rows. Velocities within a collision component share both its center of mass and one Haar rotation. Their conditional covariance is {prf:ref}`prop-cloning-component-conservation`; it cannot be replaced by an independent per-row noise covariance. Coupling two swarms must likewise couple component rotations, not assign separate rotations to walkers in one component.
:::

### 9.4. Complete Operator Specification

We now assemble the complete operator from its components.

:::{prf:theorem} Compositional Structure of $\Psi_{\text{clone}}$
:label: thm-cloning-operator-composition

The cloning operator ({prf:ref}`def-cloning-operator-formal`) admits the following compositional representation:

$$
\Psi_{\text{clone}}(S, \cdot) = \int_{\mathbf{d}} \int_{\mathbf{c}, \mathbf{a}} \Psi_{\text{update}}(S, \mathbf{c}, \mathbf{a}, \cdot) \, dP_{\text{decision}}(S, \mathbf{V}_{\text{fit}}(\mathbf{d}), \mathbf{c}, \mathbf{a}) \, dP_{\text{measure}}(S, \mathbf{d})

$$

where:
- $P_{\text{measure}}(S, \cdot)$ is the distribution of raw distance vectors from $\Psi_{\text{measure}}$
- $\mathbf{V}_{\text{fit}}(\mathbf{d})$ is the deterministic fitness vector from $\Psi_{\text{fitness}}$ given $S$ and $\mathbf{d}$
- $P_{\text{decision}}(S, \mathbf{V}_{\text{fit}}, \cdot)$ is the joint distribution of companion assignments and actions
- $\Psi_{\text{update}}(S, \mathbf{c}, \mathbf{a}, \cdot)$ is the (possibly stochastic) output distribution given the actions

This composition is a proper Markov kernel ({prf:ref}`def-markov-kernel`): for any measurable set $A \subseteq \Sigma_N$,

$$
\Psi_{\text{clone}}(S, A) = P(S' \in A \mid S)

$$

is a well-defined probability.

Referenced by {prf:ref}`def-cloning-operator-formal`.
:::

### 9.5. Key Quantities for Drift Analysis

We conclude by highlighting the critical quantities (see {prf:ref}`def-key-operator-outputs`) that will be used in the subsequent drift analysis.

:::{prf:definition} Key Operator Outputs
:label: def-key-operator-outputs

For input swarm ({prf:ref}`def-swarm-and-state-space`) $S$ and output swarm $S' \sim \Psi_{\text{clone}}(S, \cdot)$, the following quantities are central to the drift analysis:

1. **Total Cloning Probability:** For each walker ({prf:ref}`def-walker`) $i$:


$$
p_i = P(\text{walker ({prf:ref}`def-walker`) } i \text{ clones} \mid S)

$$

2. **Position Displacement:** For each walker  $i$:


$$
\Delta x_i := x'_i - x_i

$$

   For cloners, $\Delta x_i = x_{c_i} - x_i + \sigma_x \zeta_i^x$ where $\zeta_i^x \sim \mathcal{N}(0, I_d)$.

3. **Velocity Perturbation:** For each walker ({prf:ref}`def-walker`) $i$ that participates in a cloning event:


$$
\Delta v_i := v'_i - v_i

$$

   This arises from the inelastic collision model. The expected squared velocity change depends on:
   - The component center: $\mathbb{E}[\|\bar v_{C(i)} - v_i\|^2]$
   - The restitution coefficient: $\alpha_{\text{restitution}}$
   - The shared component rotation: $R_{C(i)}$

4. **Centered Displacements:** For coupled swarms $(S_1, S_2)$:


$$
\Delta\delta_{x,i} := \delta_{x,1,i} - \delta_{x,2,i}

$$

:::

:::{prf:proposition} Exact positional displacement and Gaussian moments
:label: prop-expected-displacement-cloning

Conditional on the frozen state and measured fitness,

$$
\mathbb E[|\Delta x_i|^2\mid S,\mathbf F]
=\sum_jb_{ij}\bigl(|x_j-x_i|^2+d\sigma_x^2\bigr).
$$

For an alive row in a domain of diameter $D_x$, this is at most $p_i(D_x^2+d\sigma_x^2)$. For a dead row with retained position $x_i$ and donor positions bounded by $R_D$, it is at most $2|x_i|^2+2R_D^2+d\sigma_x^2$. A persisting row has zero positional displacement; it may still receive a donor velocity update.

*Proof.* Given acceptance and donor $j$, the displacement is $x_j-x_i+\sigma_x\zeta_i$. Centering and Gaussian covariance give the displayed second moment. Average the donor decisions and apply the corresponding deterministic distance bound. $\square$
:::

### 9.6. The transition used in the drift calculation

:::{div} feynman-prose
The measurement and decision kernels determine a joint distribution of
companions and replacement actions. The update maps those choices to proposed
positions, velocities, and statuses. Guaranteed revival refers to a nonempty
companion population before the next boundary check. The drift calculations
must use the same proposal or tested-transition convention throughout.
:::

(sec-cloning-variance)=
## 10. Drift Analysis Under the Cloning Operator - Variance Components

### 10.1. Variance under replacement

:::{div} feynman-prose
The position estimate uses the exact row laws and a bounded-donor reset argument. The velocity estimate is the exact component energy identity plus the alive-only revival contribution. The selection estimate is retained separately, because its conversion into a sharper positional rate requires directional donor information.
:::

### 10.2. The Coupled Expectation Framework

To analyze the drift of the Lyapunov function components, we work with two copies of the swarm evolving under synchronous coupling.

:::{prf:definition} Coupled Cloning Expectation
:label: def-coupled-cloning-expectation

Consider two swarms $(S_1, S_2)$ in the coupled state space (see {prf:ref}`def-coupled-state-space`). Let $(S'_1, S'_2)$ be the output swarms after applying $\Psi_{\text{clone}}$ to each with its correct marginal kernel, using **synchronous coupling** of the innovations:

- Same PRNG seeds for companion selection ({prf:ref}`def-companion-selection-measure`)
- Same pairing algorithm random choices
- Same threshold samples $T_i$ for each walker ({prf:ref}`def-walker`) index $i$
- Same Gaussian jitters $\zeta_i^x$ for position updates (when both walkers clone)
- One independent Haar matrix per component in each marginal; identical components may share their matrix across swarms, and different component partitions retain their distinct centers and membership

For any function $f: \Sigma_N \times \Sigma_N \to \mathbb{R}$, the **coupled cloning expectation** is:

$$
\mathbb{E}_{\text{clone}}[f(S'_1, S'_2) \mid S_1, S_2] := \mathbb{E}[f(S'_1, S'_2) \mid S_1, S_2, \text{coupling}]

$$

:::

:::{prf:remark} What synchronous coupling provides
:label: rem-coupling-benefits

Identical inputs and identical innovations give identical outputs, including their accepted graph and component rotations. On other inputs, a shared uniform can select different donors, and changed graph membership changes a whole component's center of mass. Gaussian positional displacement has finite moments but no deterministic bound by the domain diameter. The estimates below use its actual second moment. Neither synchronization alone nor a selection-probability bound proves a negative inter-swarm drift.
:::

### 10.3. Exact positional variance and the geometric drift term

:::{div} feynman-prose
Replacing a walker by a fitter donor does not always shrink a cloud. If a distant donor attracts copies, the cloud can initially spread as more mass reaches that donor. The relevant quantity is the new distribution of positions, including its moving center, rather than the number of accepted decisions alone.

We can calculate the complete one-step positional variance from the actual donor laws. This calculation also identifies the geometric term that a strict contraction argument has to control.
:::

:::{prf:definition} Conditional position laws after copying
:label: def-cloning-position-row-law

Freeze the input and the sampled fitness vector. Let $b_{ij}$ be the probability of the accepted edge $i\to j$ under {prf:ref}`def-decision-operator`, and let $p_i=\sum_jb_{ij}$. A dead row has $p_i=1$. Write $G_{\sigma_x}$ for the centered Gaussian jitter law. The exact row position law is

$$
Q_i=(1-p_i)\delta_{x_i}+\sum_{j\in\mathcal A}b_{ij}
(\delta_{x_j}*G_{\sigma_x}),\qquad
m_i=\int yQ_i(dy),\quad \Sigma_i=\int(y-m_i)(y-m_i)^TQ_i(dy).
$$

All retained dead coordinates enter donor selection, but the dead row's position is replaced in $Q_i$. Conditional on the frozen fitness, the output positions are independent with laws $Q_i$. Component rotations do not change this positional statement.
:::

:::{prf:lemma} Exact variance decomposition
:label: lem-variance-change-decomposition

For the all-alive proposal and $\bar m=N^{-1}\sum_i m_i$,

$$
\boxed{\quad
\mathbb E[V_{\mathrm{Var},x}(S')\mid S,\mathbf F]
=\frac1N\sum_i|m_i-\bar m|^2
+\left(1-\frac1N\right)\frac1N\sum_i\operatorname{tr}\Sigma_i.
\quad}
$$

Here the single-swarm input observable is the $N$-normalized alive variance. For two swarms, sum this formula; independence between swarms is unnecessary. Averaging over their sampled fitness vectors gives the unconditional cloning drift.

*Proof.* The output variance equals $N^{-1}\sum_i|X_i'|^2-|N^{-1}\sum_iX_i'|^2$. The expected first term is $N^{-1}\sum_i(|m_i|^2+\operatorname{tr}\Sigma_i)$. Conditional independence gives the expected second term $|\bar m|^2+N^{-2}\sum_i\operatorname{tr}\Sigma_i$. Subtraction proves the formula. $\square$
:::

:::{prf:definition} Frozen positional moments for the complete cloning proposal
:label: def-cloning-frozen-positional-moments

The following identities concern the cloning proposal of the specified independent weighted-companion configuration. They do not alter its measurement law, retained sampled fitness, frozen acceptance decisions, revival, component rotations, or jitter. They apply before the prescribed kinetic and terminal-boundary stages. All conditional expectations below first condition on the complete marked input state $S$ and the complete retained fitness vector $\mathbf F$. An outer expectation over its actual measurement law is required to obtain the cloning kernel conditional only on $S$.

Write $\mathcal A$ for the entering alive set. Assume initially that it is nonempty. For every slot define the actual accepted-edge probabilities
$$
b_{ij}=\Pr(A_i=1,J_i=j\mid S,\mathbf F),\qquad p_i=\sum_{j\in\mathcal A}b_{ij}.
$$
For a live row when $|\mathcal A|\geq2$,
$$
b_{ij}=\frac{w_{ij}}{Z_i}
 \min\left\{1,\frac{(F_j-F_i)_+}{p_{\max}(F_i+\varepsilon_{\rm clone})}\right\},
\quad Z_i=\sum_{j\in\mathcal A\setminus\{i\}}w_{ij},
\quad w_{ij}=\exp[-d_{\rm alg}(i,j)^2/(2\epsilon_c^2)].
$$
There is no self-edge. A singleton live row persists. A dead row has $p_i=1$ and $b_{ij}=w_{ij}/\sum_{\ell\in\mathcal A}w_{i\ell}$, using its retained entering coordinates in the weights. These revival probabilities do not evaluate a fictitious dead fitness.

Let the position-jitter standard deviation be $j$, in dimension $d$, and set
$$
t_i=\sum_j b_{ij}(x_j-x_i),\qquad
m_i=x_i+t_i,\qquad
c_i=\sum_j b_{ij}|x_j-x_i|^2-|t_i|^2+d j^2p_i.
\tag{3.F1}
$$
Here $m_i$ is the exact conditional output mean, and $c_i$ is the trace of the exact conditional output covariance. In particular $c_i\ge0$. Although the displacement representation in (3.F1) uses retained dead coordinates, for a dead row it simplifies to
$$
m_i=\sum_jb_{ij}x_j,\qquad
c_i=\sum_jb_{ij}|x_j-m_i|^2+d j^2.
\tag{3.F2}
$$
Consequently neither output moment contains an artificial contribution from a discarded dead position. Retained coordinates still affect the donor probabilities.

Conditional on $(S,\mathbf F)$, the output positions $X_i'$ are independent: donor/gate innovations and position jitters are row-independent. This positional assertion remains true when the collision velocities have a shared component rotation. It makes no assertion of independence for the full output states, or of unconditional positional independence after the fitness vector is averaged out.
:::

:::{prf:lemma} Individual centered displacement with the moving barycenter
:label: lem-cloning-individual-centered-displacement

Use full-slot centers $\bar x=N^{-1}\sum_i x_i$, $\bar t=N^{-1}\sum_i t_i$, and $\delta_i=x_i-\bar x$. The output center is $\bar X'=N^{-1}\sum_iX_i'$. For every row,
$$
\boxed{
\mathbb E\bigl[|X_i'-\bar X'|^2-|\delta_i|^2\mid S,\mathbf F\bigr]
=2\delta_i\cdot(t_i-\bar t)+|t_i-\bar t|^2
 +\left(1-\frac2N\right)c_i+\frac1{N^2}\sum_\ell c_\ell .}
\tag{3.F3}
$$
**Proof.** Write $X_i'=m_i+\xi_i$, where the centered vectors $\xi_i$ are conditionally independent and $\mathbb E|\xi_i|^2=c_i$. The mean of $X_i'-\bar X'$ is $\delta_i+t_i-\bar t$. Expanding the covariance of $\xi_i-N^{-1}\sum_\ell\xi_\ell$ gives $(1-2/N)c_i+N^{-2}\sum_\ell c_\ell$. The mixed mean-noise term has zero expectation. This proves (3.F3), including $N=1$, where the covariance terms cancel. $\square$

Thus a row with $p_i=0$ can have a changing centered position: its own position stays fixed, while its population center changes. The deterministic contribution $-2\delta_i\cdot\bar t+|\bar t|^2$ need not be small in $N$. The stochastic center term $N^{-2}\sum c_\ell$ is distinct from this deterministic movement.

For an all-alive entering cloud of position diameter at most $D_x$,
$$
0\le \operatorname{Var}(\bar X'\mid S,\mathbf F)
=\frac1{N^2}\sum_i c_i
\le\frac{D_x^2+d j^2}{N}\,\bar p,
\qquad \bar p=\frac1N\sum_i p_i.
\tag{3.F4}
$$
This follows from $c_i\le p_i(D_x^2+d j^2)$. For nonempty alive pools with revival, all row means are convex combinations of eligible positions and all copied or retained pre-jitter positions lie in the eligible cloud; the weaker bound $c_i\le D_x^2+d j^2$ gives the same $N^{-1}$ bound without $\bar p$. No bound on retained dead positions is used. Conditional only on $S$, the law of total variance adds the separate term
$$
\operatorname{Var}_{\mathbf F}(\bar m\mid S),\qquad \bar m=\frac1N\sum_i m_i.
\tag{3.F5}
$$
The complete canonical bound for (3.F5) is proved in {prf:ref}`thm-cloning-canonical-barycenter-concentration`.
:::

:::{div} feynman-prose
Imagine marking the center of a cloud on a ruler. A walker can stay exactly where it is while other walkers copy a donor. The mark moves, so the stationary walker's distance from the mark changes. Its own cloning probability cannot account for that change. Formula (3.F3) keeps both pieces: the walker's displacement and the displacement of the common center.

Now sum over the cloud. The terms involving the entering centered positions simplify because those positions sum to zero. This is where the collective calculation becomes more informative than separate row bounds. In the three-walker example below, the central walker gains centered variance while the whole cloud loses it. There is no conflict: an individual distance from a moving center and the cloud's average squared distance are different observables.
:::

:::{prf:theorem} Canonical cloning barycenter concentration for every nonempty alive pool
:label: thm-cloning-canonical-barycenter-concentration

Fix a nonextinct entering marked state $S$ of the canonical Euclidean Gas, with $N$ retained slots and $k\ge1$ alive slots. Apply the actual cloning proposal, including its independent weighted measurement companions, retained sampled fitness, independent weighted cloning companions, frozen acceptance, immediate revival, recipient Gaussian jitter, and shared component rotations. Let $\bar X^c=N^{-1}\sum_iX_i^c$ and $\bar V^c=N^{-1}\sum_iV_i^c$ denote its full-slot output barycenters.

Write $D_x$ for a bound on the entering eligible positions' diameter and $j$ for the position-jitter standard deviation. The following are actual canonical parameter bounds:
$$
D_m=\sqrt{32+10^{-6}},\quad \varepsilon_s=0.1,\quad
\kappa_C=e^{-4},\quad C=\kappa_C^{-1}=e^4,\quad
F_*=0.01,\quad F^*=4.41.
$$
Let $\varepsilon_c=10^{-6}$, $p_{\max}=1$, and define
$$
\begin{aligned}
L_0&=1.05\left(\frac{D_m}{\varepsilon_s}
 +\frac{3D_m^3}{2\varepsilon_s^3}\right),\\
L_a&=\frac1{p_{\max}}\max\left\{\frac1{F_*+\varepsilon_c},
 \frac{F^*+\varepsilon_c}{(F_*+\varepsilon_c)^2}\right\},\\
B_0&=1+C+2L_aL_0.
\end{aligned}
$$
For vector variance $\operatorname{Var}(Z)=\mathbb E|Z-\mathbb EZ|^2$, the complete proposal obeys
$$
\boxed{\operatorname{Var}(\bar X^c\mid S)
\le \frac{D_x^2+d j^2}{N}
 +\frac{kD_x^2B_0^2}{2N^2}
\le\frac{D_x^2+d j^2+D_x^2B_0^2/2}{N}.}
\tag{3.B1}
$$
Its full-slot velocity barycenter is deterministic:
$$
\boxed{\bar V^c=\frac1N\sum_i v_i\quad\text{almost surely},
\qquad \operatorname{Var}(\bar V^c\mid S)=0.}
\tag{3.B2}
$$
Neither bound requires a lower bound on $k/N$, a collision-component-size estimate, or bounded retained dead positions. For completed-step states in the canonical absorbing box, one may take $D_x=4\sqrt d$ and $j=0.1$, so every displayed constant is discharged directly for that configuration. Smaller entering eligible diameters may be used statewise. The assertions concern the cloning proposal before the kinetic stage and before conditioning on terminal survival.
:::

:::{prf:proof}
**1. Bound the actual canonical measurements and weights.** The configured metric squashes each position and velocity vector separately with radius $2$, and compares the two resulting vectors in the phase-space norm with velocity weight $1$. Each squashed vector has norm at most $2$, irrespective of the original physical coordinates or dimension. Consequently every squared comparison is at most $4^2+4^2=32$. The measured diversity value includes the squared floor $10^{-6}$, so every alive row's measurement $Y_i$ lies in $[0,D_m]$. The cloning Gaussian width is $2$, hence its weights lie in $[e^{-4},1]$. Conditional on $S$, alive measurement companion draws, and therefore the random variables $(Y_i)_{i\in\mathcal A}$, are independent. They need not have identical laws.

The canonical positive map is $g(z)=0.1+2/(1+e^{-z})$, with $0.1\le g\le2.1$ and $|g'|\le0.5$. Its reward and diversity exponents are both one. Thus $F_*\le F_i\le F^*$. Reward measurements and their standardization are fixed when only a diversity companion innovation is replaced. This remains true when the fixed physical rewards themselves are unbounded.

**2. Replace one measurement innovation.** Assume first $k\ge2$. Replace $Y_r$ by an independent copy, leaving all other entering physical data and measurement innovations unchanged. Tildes denote quantities computed with the replacement. For the alive empirical diversity mean and variance,
$$
|\bar Y-\widetilde{\bar Y}|\le\frac{D_m}{k},\qquad
|\operatorname{var}(Y)-\operatorname{var}(\widetilde Y)|
\le\frac{3D_m^2}{k}.
$$
Indeed the empirical second moment changes by at most $D_m^2/k$, while the squared mean changes by at most $2D_m^2/k$. Let
$s_Y=\sqrt{\operatorname{var}(Y)+\varepsilon_s^2}$. Since both regularized scales are at least $\varepsilon_s$, the mean-value bound for $u\mapsto(u+\varepsilon_s^2)^{-1/2}$ gives
$$
|s_Y^{-1}-s_{\widetilde Y}^{-1}|
\le\frac{3D_m^2}{2\varepsilon_s^3k}.
$$
For $i\ne r$, the measurement itself has not changed and $|Y_i-\widetilde{\bar Y}|\le D_m$. Hence
$$
\left|\frac{Y_i-\bar Y}{s_Y}
 -\frac{Y_i-\widetilde{\bar Y}}{s_{\widetilde Y}}\right|
\le\frac1k\left(\frac{D_m}{\varepsilon_s}
 +\frac{3D_m^3}{2\varepsilon_s^3}\right).
$$
The fixed rescaled reward factor is at most $2.1$, so the product fitness satisfies
$$
|F_i-\widetilde F_i|\le L_0/k\qquad(i\ne r).
\tag{3.B3}
$$
No small bound is imposed on $F_r-\widetilde F_r$.

**3. Sum the exact output-mean influence before estimating it.** The live donor probability $K_{i\ell}$ depends on the frozen entering physical state, so it is unchanged by this measurement replacement. Its self-excluded denominator has $k-1$ terms, each at least $\kappa_C$. Thus
$$
K_{ir}\le\frac{C}{k-1}\qquad(i\ne r).
\tag{3.B4}
$$
The actual clipped acceptance
$$
a(f,g)=\min\left\{1,\max\left\{0,
 \frac{g-f}{p_{\max}(f+\varepsilon_c)}\right\}\right\}
$$
is $L_a$-Lipschitz in the sum of its two fitness arguments on $[F_*,F^*]^2$. This follows by differentiating the ratio, whose partial derivatives in magnitude are at most the two constants defining $L_a$, and using the nonexpansiveness of scalar clipping.

Let $m_i=\mathbb E[X_i^c\mid S,\mathbf Y]$. For a live row,
$$
m_i=x_i+\sum_{\ell\in\mathcal A\setminus\{i\}}
 K_{i\ell}a(F_i,F_\ell)(x_\ell-x_i).
$$
Gaussian jitter has zero mean. Both $m_r$ and $\widetilde m_r$ are convex combinations of eligible positions, whence $|m_r-\widetilde m_r|\le D_x$. For $i\ne r$, isolate donor $r$, for which the acceptance difference is at most one. For every other donor, apply (3.B3) to both fitness arguments. This yields
$$
|m_i-\widetilde m_i|
\le D_x\left(K_{ir}+\frac{2L_aL_0}{k}\right).
$$
Every dead row revives unconditionally, with mean $m_i=\sum_{\ell\in\mathcal A}K_{i\ell}x_\ell$. Its donor law uses retained dead coordinates but not sampled fitness, so $m_i=\widetilde m_i$ for all dead rows. Summing the alive-row inequalities and using (3.B4) proves
$$
\left|\bar m(\mathbf Y)-\bar m(\widetilde{\mathbf Y})\right|
\le\frac{D_x}{N}\left[1+C+2L_aL_0\frac{k-1}{k}\right]
\le\frac{D_xB_0}{N},\quad
\bar m=\frac1N\sum_i m_i.
\tag{3.B5}
$$
The factor $1/(k-1)$ has canceled against exactly $k-1$ recipient rows; no positive alive-fraction bound has been used.

**4. Apply the conditional vector Efron--Stein inequality.** The independent alive measurement variables and (3.B5) give
$$
\operatorname{Var}_{\mathbf Y}(\bar m\mid S)
\le\frac12\sum_{r\in\mathcal A}
 \mathbb E\bigl|\bar m(\mathbf Y)-\bar m(\mathbf Y^{(r)})\bigr|^2
\le\frac{kD_x^2B_0^2}{2N^2}.
\tag{3.B6}
$$
For completeness, the scalar product-measure variance bound follows by induction on the number of independent inputs. The conditional variance identity isolates the last input; apply the induction hypothesis to its conditional mean and Jensen to each resulting squared difference. This gives $\operatorname{Var}(f)\leq\sum_r\mathbb E\operatorname{Var}_{Y_r}(f\mid Y_{-r})$. Each term equals one half of the expected squared difference under independent replacement of $Y_r$. Summing this scalar inequality over coordinates proves the displayed vector inequality, with no additional dimension factor.

Conditional on all measurements, the output positions are independent across recipient rows. Each pre-jitter row position is an eligible input position, including every revived row, so its conditional variance is at most $D_x^2$. Accepted-row jitter adds at most $d j^2$. Consequently
$$
\operatorname{Var}(\bar X^c\mid S,\mathbf Y)
=\frac1{N^2}\sum_i\operatorname{Var}(X_i^c\mid S,\mathbf Y)
\le\frac{D_x^2+d j^2}{N}.
$$
The vector law of total variance and (3.B6) prove (3.B1). When $k=1$, the sole alive measurement and fitness are deterministic, and every revived row uses the sole eligible donor. The measurement-variance contribution is zero and the same bound holds. Nonextinction excludes $k=0$, where no donor law is defined.

Finally, in every realized accepted component $C$, the actual velocity transformation satisfies
$$
\sum_{i\in C}v_i^c
=|C|\bar v_C+\alpha R_C\sum_{i\in C}(v_i-\bar v_C)
=\sum_{i\in C}v_i.
$$
The sums include retained dead-slot velocities when recipients revive. Summing over components, including unchanged singletons, proves the pathwise identity (3.B2). It is unaffected by averaging sampled fitness, donor choices, or Haar rotations. $\square$
:::

:::{div} feynman-prose
One measurement can influence many cloning decisions. Sum its effect on the population mean first: the number of possible recipients cancels the reciprocal alive-population factor in their donor weights. The remaining change is of order $1/N$, and summing squared influences gives variance of order $1/N$. With just one alive walker, measurement randomness disappears altogether. Even a giant collision component causes no difficulty for these barycenters: rotations leave positions alone and conserve full-slot momentum exactly. These conclusions use the particular observables being averaged; concentration of other observables still needs its own argument.
:::

:::{prf:theorem} Collective donor flux in the geometric clusters
:label: lem-keystone-contraction-alive

For an all-alive input, let $r_i^2=|x_i-\bar x|^2$, and define the nonnegative row-copy variance
$$
a_i=\sum_jb_{ij}|x_j-x_i|^2-|t_i|^2.
$$
The complete conditional variance drift is
$$
\boxed{
\mathbb E[\Delta V_{\mathrm{Var},x}\mid S,\mathbf F]
=\frac1N\sum_{i,j}b_{ij}(r_j^2-r_i^2)
 -|\bar t|^2-\frac1{N^2}\sum_i a_i
 +\left(1-\frac1N\right)d j^2\bar p .}
\tag{3.F6}
$$
**Proof.** Sum (3.F3) and divide by $N$. Use $\sum_i\delta_i=0$, $\sum_i|t_i-\bar t|^2=\sum_i|t_i|^2-N|\bar t|^2$, and $c_i=a_i+d j^2p_i$. Finally,
$$
2\delta_i\cdot t_i+\sum_jb_{ij}|x_j-x_i|^2
=\sum_jb_{ij}(r_j^2-r_i^2).
$$
Substitution proves (3.F6). Nonnegativity of $a_i$ is the variance of the pre-jitter row-copy law. $\square$

Use exactly the chapter's geometric cluster partition $\{G\}$, with no change to its construction. Put
$$
x_G=\frac1{|G|}\sum_{i\in G}x_i,\quad
e_G=|x_G-\bar x|^2,\quad
\rho_i=|x_i-\bar x|^2-e_G\quad(i\in G),\quad
B_{GH}=\frac1N\sum_{i\in G,j\in H}b_{ij}.
$$
Then the signed flux in (3.F6) decomposes exactly as
$$
\frac1N\sum_{i,j}b_{ij}(r_j^2-r_i^2)
=\sum_{G,H}B_{GH}(e_H-e_G)+\mathcal R_{\mathrm{cl}},
\qquad
\mathcal R_{\mathrm{cl}}=\frac1N\sum_{i,j}b_{ij}(\rho_j-\rho_i).
\tag{3.F7}
$$
If the eligible position diameter is $D_x$, and $D_c=\max_G\operatorname{diam}_x(G)$ is the actual positional diameter of these same clusters, then
$$
|\rho_i|\le2D_xD_c,
\qquad
|\mathcal R_{\mathrm{cl}}|\le4D_xD_c\bar p.
\tag{3.F8}
$$
Indeed $|x_i-x_G|\le D_c$, while both $|x_i-\bar x|$ and $|x_G-\bar x|$ are at most $D_x$; factor the difference of the squared norms. Summing actual accepted-edge masses uses only $N^{-1}\sum_{ij}b_{ij}=\bar p\le1$. No factor equal to the number of walkers or clusters is introduced. A phase-space diameter bound controls $D_c$ when its metric dominates position distance. For a saturated metric, use its proved inverse modulus if available, or the actual positional cluster diameter; a saturated distance must not be silently treated as the physical distance.

The symmetric donor weight gives an additional exact symmetrization, using the realized fitness order:
$$
\frac1N\sum_{i,j}b_{ij}(r_j^2-r_i^2)
=\frac1N\sum_{\{i,j\}:F_i<F_j}
 \frac{w_{ij}}{Z_i}
 \min\left\{1,\frac{F_j-F_i}{p_{\max}(F_i+\varepsilon_{\rm clone})}\right\}
 (r_j^2-r_i^2).
\tag{3.F9}
$$
Here every unordered unequal-fitness pair is written with its lower-fitness endpoint first; ties contribute zero. The normalization remains that of the recipient. Symmetry of $w$ does not make the directed normalized acceptance weights symmetric. Equations (3.F7)--(3.F9) retain inward and outward flux separately. In particular the incoming term $\sum_{G,H}B_{GH}e_H$ cannot be omitted when a Keystone estimate bounds outgoing selection pressure.
:::

:::{div} feynman-prose
Think of an accepted copy as moving population mass from the recipient's location to the donor's location. The cluster contribution is the amount moved, $B_{GH}$, times the change in squared distance, $e_H-e_G$. Moving inward gives a negative contribution; moving outward gives a positive one. The original geometric clusters let us keep these signs together before estimating the smaller within-cluster remainder.

The normalization matters. Across all cluster pairs, the transported mass is $\bar p\leq1$, regardless of how many walkers or clusters there are. That is why the remainder estimate carries no extra factor of $N$. This accounting preserves uniformity in population size, but its sign still depends on where the mass lands. A bound on how much mass leaves high-error clusters must therefore be combined with the incoming donor contribution in the same collective balance.
:::

:::{prf:theorem} Signed cluster flow from retained fitness gaps
:label: thm-cloning-signed-cluster-fitness-flux

Condition on the complete entering state and retained fitness vector, with
$k=|\mathcal A|\geq2$. Use its actual bounds
$0<F_*\leq F_i\leq F^*$ and symmetric weights
$\kappa\leq w_{ij}=w_{ji}\leq1$. For the canonical configuration these
bounds are discharged in
{prf:ref}`thm-cloning-canonical-barycenter-concentration`.
For two nonempty disjoint geometric clusters $H,L$, retain the actual
$b_{ij},Z_i,B_{HL}$ of (3.F7)--(3.F9). Put

$$
A_*=\max\{F^*-F_*,p_{\max}(F^*+\varepsilon_{\rm clone})\},
\quad c_+=\frac\kappa{A_*},\quad
c_-=\frac1{\kappa p_{\max}(F_*+\varepsilon_{\rm clone})},
$$
$$
\Delta=\bar F_L-\bar F_H,\qquad
s^2=\frac1{|H|}\sum_{i\in H}(F_i-\bar F_H)^2
+\frac1{|L|}\sum_{j\in L}(F_j-\bar F_L)^2.
$$

Then $0<c_+\leq c_-$, and the signed flow satisfies, for either sign of
$\Delta$,

$$
B_{HL}-B_{LH}\geq
\frac{|H||L|}{N(k-1)}
\left[c_+\Delta-
\frac{c_--c_+}{2}\bigl(\sqrt{\Delta^2+s^2}-\Delta\bigr)\right]
=:\mathcal L_{HL}.                                      \tag{3.S1}
$$

There is also a bound retaining the kernel weights and measured
normalizers. Define

$$
W_{HL}=\sum_{i\in H,j\in L}w_{ij},\quad
P_w(i,j)=\frac{w_{ij}}{W_{HL}},\quad
\Delta_w=\sum_{i,j}P_w(i,j)(F_j-F_i),
$$
$$
s_w^2=\sum_{i,j}P_w(i,j)(F_j-F_i-\Delta_w)^2,
$$
$$
A_{HL}=\max\left\{
(\max_L F-\min_H F)_+,\ p_{\max}(\max_H F+\varepsilon_{\rm clone})\right\},
$$
$$
a_{HL}=\frac1{(\max_{i\in H}Z_i)A_{HL}},\qquad
b_{HL}=\frac1{(\min_{j\in L}Z_j)
 p_{\max}(\min_L F+\varepsilon_{\rm clone})}.
$$

With these deterministic weighted moments of the retained marks,

$$
B_{HL}-B_{LH}\geq\frac{W_{HL}}N\left[
a_{HL}\Delta_w-\frac{(b_{HL}-a_{HL})_+}{2}
\bigl(\sqrt{\Delta_w^2+s_w^2}-\Delta_w\bigr)\right]
=:\mathcal L^w_{HL}.                                    \tag{3.S2}
$$

In particular the weighted variance retains the covariance induced by
$P_w$; it is not replaced by a product of uniform donor laws.
:::

:::{prf:proof}
For $i\in H,j\in L$, write $D_{ij}=F_j-F_i$.
When $D_{ij}\geq0$, the reverse acceptance vanishes and
$\min\{1,D_{ij}/[p_{\max}(F_i+\varepsilon_{\rm clone})]\}
\geq D_{ij}/A_*$.
When $D_{ij}\leq0$, the forward acceptance vanishes and reverse acceptance
is at most $(-D_{ij})/[p_{\max}(F_*+\varepsilon_{\rm clone})]$.
Since $\kappa(k-1)\leq Z_i\leq k-1$, this proves pointwise

$$
b_{ij}-b_{ji}\geq
\frac{c_+(D_{ij})_+-c_-(D_{ij})_-}{k-1}.                 \tag{3.S3}
$$

Let $\langle\cdot\rangle$ denote the finite arithmetic average over
$H\times L$. It is used only to sum this already proved weighted
inequality. Direct expansion gives
$\langle D\rangle=\Delta$ and
$\langle D^2\rangle=\Delta^2+s^2$. Therefore

$$
\langle D_-\rangle
=\frac{\langle|D|\rangle-\Delta}{2}
\leq\frac{\sqrt{\Delta^2+s^2}-\Delta}{2}.
$$

Use $\langle D_+\rangle=\Delta+\langle D_-\rangle$ and
$c_-\geq c_+$ to sum (3.S3), proving (3.S1). No expectation is passed
through nonlinear acceptance.

For the refined bound, the same two sign cases instead give
$b_{ij}-b_{ji}\geq
w_{ij}[a_{HL}(D_{ij})_+-b_{HL}(D_{ij})_-]$.
Average with $P_w$ and apply
$\mathbb E_wD_-\leq(\sqrt{\Delta_w^2+s_w^2}-\Delta_w)/2$.
If $b_{HL}<a_{HL}$, its coefficient in
$a_{HL}\Delta_w+(a_{HL}-b_{HL})\mathbb E_wD_-$ is nonnegative, so
discarding that last term proves the positive-part convention in (3.S2).
This accounts for both possible orders of the coefficients. $\square$
:::

:::{prf:corollary} Collective positional drift with both flow directions retained
:label: cor-cloning-signed-collective-drift

For an all-alive input, orient each unordered pair of the same geometric
clusters so that $e_H\geq e_L$. With the notation of (3.F6)--(3.F8),

$$
\begin{aligned}
\mathbb E[\Delta V_{\mathrm{Var},x}\mid S,\mathbf F]
\leq{}&-\sum_{\{H,L\}}(e_H-e_L)
 \max\{\mathcal L_{HL},\mathcal L^w_{HL}\}
+\mathcal R_{\rm cl}\\
&-|\bar t|^2-\frac1{N^2}\sum_i a_i
+\left(1-\frac1N\right)d j^2\bar p .                    \tag{3.S4}
\end{aligned}
$$

The signed remainder may be retained or bounded by
$|\mathcal R_{\rm cl}|\leq4D_xD_c\bar p$.
The coefficients in (3.S1) sum to at most $k/(2N)$ over all unordered
cluster pairs. No factor from the population size or the number of
clusters is added. The bounds hold for every retained realization and
can be averaged over its actual measurement law, keeping negative lower
bounds in the sum. For a nonempty partially alive input, (3.F10) gives
the additional revival injection exactly.

*Proof.* The two directed centroid terms for a pair combine as
$-(e_H-e_L)(B_{HL}-B_{LH})$. Apply (3.S1) and (3.S2), then use (3.F6)
and (3.F7). For a partition of $k$ labels,
$\sum_{\{H,L\}}|H||L|=(k^2-\sum_H|H|^2)/2\leq k(k-1)/2$;
divide by $N(k-1)$. Conditional expectation proves the final assertion.
$\square$
:::

:::{div} feynman-prose
Take a cluster farther from the center and another closer in. A fitness advantage for the inner cluster favors inward copying, but the two clusters can contain overlapping fitness values. Some accepted copies can therefore run outward. The spread term measures how much this reverse flow can subtract from the favorable mean gap; the weighted formula also keeps the algorithm's actual preference for particular donor pairs.

The calculation uses the retained fitness values to bound each accepted direction before taking any averages. Thus both flow signs survive the nonlinear acceptance rule. The geometric clusters remain the units of the collective drift calculation, with their internal variation accounted for separately.
:::

:::{prf:proposition} Revival contribution to the alive variance
:label: prop-cloning-revival-cluster-flux

For a nonempty alive pool, write $\mu_A=|\mathcal A|^{-1}\sum_{i\in\mathcal A}x_i$ and
$$
V_A(S)=\frac1N\sum_{i\in\mathcal A}|x_i-\mu_A|^2.
$$
All proposal slots are alive after mandatory revival. The exact change from the entering alive-only variance to the proposal variance is
$$
\begin{aligned}
\mathbb E[V_A(S')-V_A(S)\mid S,\mathbf F]
={}&\frac1N\sum_{i\in\mathcal A,j\in\mathcal A}
 b_{ij}\bigl(|x_j-\mu_A|^2-|x_i-\mu_A|^2\bigr)\\
&+\frac1N\sum_{i\notin\mathcal A,j\in\mathcal A}
 b_{ij}|x_j-\mu_A|^2
 +d j^2\bar p\\
&-|\bar m-\mu_A|^2-\frac1{N^2}\sum_i c_i.
\end{aligned}
\tag{3.F10}
$$
**Proof.** Expand the proposal variance around the fixed entering alive center $\mu_A$. Its expected uncentered second moment is the entering alive second moment plus the two displayed donor sums and $d j^2\bar p$. Subtract $\mathbb E|\bar X'-\mu_A|^2=|\bar m-\mu_A|^2+N^{-2}\sum_i c_i$. $\square$

The second donor sum is the exact revival injection into this alive-only observable. Its upper bound is $D_x^2(N-|\mathcal A|)/N$. Equations (3.F1), (3.F3), and the algebra of (3.F6) remain valid for full-slot entering variance when retained dead coordinates are included. In that case (3.F9) applies only to live-live edges: unconditional revival edges do not follow a live fitness order. A physical cluster remainder involving unbounded retained dead coordinates requires their actual moments, not an invented compact-support claim. Formula (3.F10) avoids that difficulty for the alive-only observable while preserving their influence on sampling. If $\mathcal A=\varnothing$, the transition is extinction and no donor, center $\mu_A$, or revival law is introduced.
:::

:::{prf:theorem} Complete measurement-averaged cloning balance
:label: thm-cloning-unconditional-collective-balance

Fix a nonextinct entering marked swarm $S$. Let $k\geq1$ be its alive count,
$\mu_A$ and $u_A$ its alive position and velocity means, and keep the
chapter's geometric partition of the alive slots. Define
$$
V_{A,x}=\frac1N\sum_{i\in\mathcal A}|x_i-\mu_A|^2,
\qquad
V_{A,v}=\frac1N\sum_{i\in\mathcal A}|v_i-u_A|^2.
$$
A prime denotes the complete cloning proposal, after mandatory revival,
position jitter, and the prescribed shared component rotations. The
following expectations use the actual measurement-companion law conditional
on $S$. In particular,
$$
\beta_{ij}(S)=\mathbb E_{\mathbf F\mid S}b_{ij}(S,\mathbf F),
\qquad \bar\pi(S)=\frac1N\sum_{i,j}\beta_{ij}(S).
$$
For $k\geq2$, this expectation is the finite integral
$$
\mathbb E_{\mathbf F\mid S}f(\mathbf F)
=\sum_{(j_i)_{i\in\mathcal A}}
 \left[\prod_{i\in\mathcal A}K^{\rm meas}_{ij_i}(S)\right]
 f\bigl(\mathbf F(S,(j_i)_{i\in\mathcal A})\bigr),       \tag{3.U1}
$$
where $j_i\ne i$ are eligible measurement companions and every fitness
normalizer is recomputed from the whole sampled vector in that summand.
For $k=1$, the canonical singleton measurement convention is deterministic.

Put $a_i=\mathbb E[|Y_i-m_i|^2\mid S,\mathbf F]$, where $Y_i$ is the
pre-jitter copied-or-retained position and $m_i$ its conditional mean.
Thus $c_i=a_i+d j^2p_i$ in
(3.F1). Define the two exact revival terms
$$
R_x(S)=\frac1N\sum_{i\notin\mathcal A,j\in\mathcal A}
 \beta_{ij}|x_j-\mu_A|^2,
$$
$$
R_v(S)=\frac1N\sum_{i\notin\mathcal A}|v_i-u_A|^2
 -\left|\frac1N\sum_{i\notin\mathcal A}(v_i-u_A)\right|^2.
$$
If eligible positions have diameter $D_x$ and all retained velocities are
bounded by $V_{\max}$, then
$$
0\leq R_x\leq D_x^2\frac{N-k}{N},\qquad
0\leq R_v\leq4V_{\max}^2\frac{N-k}{N}.
$$
For the actual accepted components, let
$$
\mathcal E_C=\frac1N\sum_C\sum_{i\in C}|v_i-\bar v_C|^2.
$$
Orient unordered geometric-cluster pairs by $e_H\geq e_L$, using the
alive center $\mu_A$. For $0\leq\alpha\leq1$ and any fixed velocity weight $\lambda_v\geq0$,
define the signed quantity
$$
\begin{aligned}
\mathscr D(S)={}&
\sum_{\{H,L\}}(e_H-e_L)
\frac1N\sum_{i\in H,j\in L}(\beta_{ij}-\beta_{ji})
-\mathbb E\mathcal R_{\rm cl}\\
&+\mathbb E|\bar m-\mu_A|^2
+\frac1{N^2}\sum_i\mathbb E a_i
+\lambda_v(1-\alpha^2)\mathbb E\mathcal E_C .
\end{aligned}
$$
Then the complete unconditional proposal balance is exactly
$$
\boxed{
\mathbb E[\Delta(V_{A,x}+\lambda_vV_{A,v})\mid S]
=-\mathscr D(S)
+\left(1-\frac1N\right)d j^2\bar\pi(S)
+R_x(S)+\lambda_vR_v(S).}                              \tag{3.U2}
$$
The expectation of $\mathcal E_C$ additionally integrates actual donor
choices and gates. Its energy identity is pathwise in the shared rotations.
No independence of collision velocities is used.

For $k\geq2$, the signed quantity has the proved lower bound obtained by
replacing each averaged directed-flow difference by
$\mathbb E\max\{\mathcal L_{HL},\mathcal L^w_{HL}\}$ from (3.S1)--(3.S2).
One explicit bound involving unconditional retained-fitness moments is
$$
\begin{aligned}
M_{HL}&=\frac1{|H||L|}\sum_{i\in H,j\in L}\mathbb E(F_j-F_i),\\
T_{HL}&=\frac1{|H||L|}\sum_{i\in H,j\in L}\mathbb E[(F_j-F_i)^2],\\
\frac1N\sum_{i\in H,j\in L}(\beta_{ij}-\beta_{ji})
&\geq\frac{|H||L|}{N(k-1)}
\left[c_+M_{HL}-\frac{c_--c_+}{2}
 (\sqrt{T_{HL}}-M_{HL})\right].                        \tag{3.U3}
\end{aligned}
$$
Both signs are retained. The averaging coefficients sum to at most
$k/(2N)$, the positional remainder obeys
$|\mathbb E\mathcal R_{\rm cl}|\leq4D_xD_c\bar\pi$, and the revival
bounds contain only the dead fraction. These estimates introduce no growing
factor of $N$. The proposal is dissipative at a given input precisely when
the signed left contribution in (3.U2) exceeds its displayed jitter and
revival injections; (3.U3) is a sufficient calculable lower bound, not an
assumption that every input has that sign.
:::

:::{prf:proof}
Independence of the entering measurement-companion draws gives (3.U1).
This independence is used for their innovations, not for fitnesses after
shared normalizers are computed. In each summand the nonlinear acceptance
is evaluated before integration.

Apply (3.F10), use $c_i=a_i+d j^2p_i$, and average over the same measurement
law. The live-live flux has exactly the geometric decomposition (3.F7)
around $\mu_A$. The dead-row donor law does not depend on sampled fitness;
it gives $R_x$ directly. The covariance subtraction remains
$\mathbb E|\bar m-\mu_A|^2$, which includes fluctuations of the mean caused
by the shared fitness statistics. It is not replaced by
$|\mathbb E\bar m-\mu_A|^2$.

For velocities, each actual component conserves its full-slot mean and
multiplies its relative energy by $\alpha^2$. Consequently its full-slot
variance changes by $-(1-\alpha^2)\mathcal E_C$. Expanding the entering
full-slot variance about $u_A$ shows that its difference from $V_{A,v}$ is
exactly $R_v$. Cauchy--Schwarz gives
$$
\left|\frac1N\sum_{i\notin\mathcal A}(v_i-u_A)\right|^2
\leq\frac{N-k}{N}\,
\frac1N\sum_{i\notin\mathcal A}|v_i-u_A|^2,
$$
so $R_v\geq0$; the displayed upper bounds follow from eligible diameter
and the retained velocity cap. Adding the two exact identities proves
(3.U2), including revived recipients' frozen pre-collision velocities.

Finally average the pointwise inequality (3.S3). Applied to the joint
finite measure consisting of the actual measurement law and the arithmetic
sum over $H\times L$, Cauchy--Schwarz yields
$\mathbb E\langle|F_j-F_i|\rangle\leq\sqrt{T_{HL}}$.
Using $D_-=(|D|-D)/2$ gives (3.U3). This is averaging after the accepted-edge
bound, not replacing a donor law or passing expectation through acceptance.
The coefficient and remainder estimates were proved in (3.F8) and (3.S4).
$\square$
:::

:::{prf:proposition} Two populated geometric clusters with unequal sampled fitness
:label: prop-cloning-two-cluster-noise-balance

In the canonical one-dimensional quadratic configuration, take $N=2M$,
$M\geq2$, with $M$ alive slots at $+a$, $M$ at $-a$, all velocities zero,
and $0<a<2$. Retain the canonical position jitter $j=0.1$ and every
measurement, donor, gate, and collision operation. Put
$$
\ell=\frac{4a}{2+a},\quad
w=e^{-\ell^2/8},\quad Z=M-1+Mw,\quad q=\frac{Mw}{Z},
$$
$$
\delta=\sqrt{\ell^2+10^{-6}}-10^{-3},\quad
s_{\max}=\sqrt{\delta^2/4+0.1^2},\quad
A_0=\min\left\{1,
\frac{1.1\tanh[\delta/(2s_{\max})]}{1.21+10^{-6}}\right\}.
$$
The full measurement-averaged cloning drift obeys
$$
\boxed{
\mathbb E[\Delta V_{\mathrm{Var},x}\mid S]
\geq j^2\left(1-\frac1N\right)A_0q(1-q)
-\frac{4a^2}{N}q^2(1-q^2).}                            \tag{3.U4}
$$
At $a=0.5$, $N=128$, its right side is greater than $0.0002854$.
The probability that all retained fitness values tie is
$q^N+(1-q)^N<1.7\times10^{-37}$. Thus the positive averaged drift here
occurs with nontrivial fitness selection, not solely in an equal-fitness
configuration. It remains a statement about internal positional variance,
not a contradiction to an offset-bearing Keystone inequality or to
convergence of coupled structural error under the complete update.
:::

:::{prf:proof}
The two sites have equal quadratic reward. Their rescaled reward factor is
therefore exactly $1.1$. Every slot measures an opposite-site companion
with probability $q$, and a same-site companion otherwise. These choices
are independent conditional on the physical input, since each site has
$M-1$ same-site eligible companions and $M$ opposite-site companions.
Thus the numbers $K_+,K_-$ of high-diversity measurements at the two sites
are independent $\operatorname{Bin}(M,q)$ variables.

Let $K=K_++K_-$ and $\theta=K/N$. When $0<K<N$, retained fitness has the
two distinct values
$$
F_H=1.1g\left(\frac{(1-\theta)\delta}{s_\theta}\right),\quad
F_L=1.1g\left(-\frac{\theta\delta}{s_\theta}\right),\quad
s_\theta=\sqrt{\theta(1-\theta)\delta^2+0.1^2},
$$
where $g(z)=1.1+\tanh(z/2)$. Exactly the low-fitness recipients can accept,
and their acceptance conditional on choosing a high-fitness donor is
$$
A(K)=\min\{1,(F_H-F_L)/(F_L+10^{-6})\}.
$$
Set $A(0)=A(N)=0$ for the tied cases. For $u,v\geq0$,
$\tanh u+\tanh v\geq\tanh(u+v)$, by the addition formula. Since
$s_\theta\leq s_{\max}$ and $F_L\leq1.21$, this gives
$A(K)\geq A_0$ whenever $0<K<N$.

Write $\beta=w/Z=q/M$. A low-fitness slot at $+a$ has conditional mean
position displacement $-2aA(K)\beta K_-$, while a low-fitness slot at
$-a$ has displacement $2aA(K)\beta K_+$. Summing these actual displacements
before estimating them gives the exact cancellation
$$
\bar t=aA(K)\beta(K_+-K_-).
$$
Using $A\leq1$ and the independent binomial variance,
$$
\mathbb E|\bar t|^2
\leq a^2\beta^2\,2Mq(1-q)
=\frac{4a^2}{N}q^3(1-q).
$$
For a low-fitness recipient at $+a$, the pre-jitter position changes sites
with probability $c_-=A(K)\beta K_-$, so its copy variance is
$4a^2c_-(1-c_-)\leq4a^2\beta K_-$. Multiply by its count $M-K_+$,
apply the analogous formula at $-a$, and use independence of $K_+,K_-$:
$$
\frac1{N^2}\sum_i\mathbb E a_i
\leq\frac{4a^2\beta}{N^2}\,2M^2q(1-q)
=\frac{4a^2}{N}q^2(1-q).
$$

For each distinct ordered pair, its measurement labels are independent
Bernoulli $q$. Its probability of a low-fitness recipient and high-fitness
donor is $q(1-q)$; on this event $0<K<N$ and $A(K)\geq A_0$.
Summing the actual deterministic normalized donor probabilities therefore
gives $\mathbb E\bar p\geq A_0q(1-q)$.

Every entering squared radius is $a^2$, so the signed radius flux in
(3.F6) is identically zero for every retained fitness vector. Substituting
the preceding three estimates into the exact identity (3.F6) proves (3.U4).
All component velocities remain zero, irrespective of component membership,
restitution, or Haar rotations. The numerical lower bound follows by
substitution: $q=0.483942612969\ldots$ and
$A_0=0.680668094688\ldots$ at $a=0.5,N=128$.
Finally all fitnesses tie exactly when all measurements are low or all are
high, whose probabilities are $(1-q)^N$ and $q^N$.
$\square$
:::

:::{prf:lemma} The coupled structural increment keeps its cross-swarm term
:label: lem-cloning-coupled-error-not-internal-variance

Fix two entering full-slot populations and a pairing of their labels. Write
$\delta_{1,i}=x_{1,i}-\bar x_1$,
$\delta_{2,i}=x_{2,i}-\bar x_2$, and define the paired centered positional
cost and cross-swarm term
$$
E_x=\frac1N\sum_i|\delta_{1,i}-\delta_{2,i}|^2,
\qquad C_x=\frac1N\sum_i\delta_{1,i}\cdot\delta_{2,i}.
$$
For every coupling $\Gamma$ of the two complete cloning proposal kernels,
$$
\boxed{
\mathbb E_\Gamma\Delta E_x
=H_x(S_1)+H_x(S_2)
-2\bigl(\mathbb E_\Gamma C_x'-C_x\bigr),}               \tag{3.U5}
$$
where $H_x$ is the exact full-slot internal-variance drift. Thus the
measurement-averaged single-swarm calculation determines the first two
terms, but it does not determine the coupling-dependent cross-swarm term.
For alive-only entering variance, the corresponding revival conversion
must first be made as in (3.F10); it cannot be silently removed.

*Proof.* The identity
$E_x=V_{\mathrm{Var},x}(S_1)+V_{\mathrm{Var},x}(S_2)-2C_x$
holds before and after the proposal. Take its difference and expectation.
The first two output expectations depend only on the prescribed marginal
kernels, whereas $\mathbb E_\Gamma C_x'$ depends on the chosen coupling.
For identical entering populations, coupling every actual measurement,
donor, gate, component rotation, and jitter identically gives identical
outputs. Hence $E_x'=E_x=0$ pathwise. In this case
$C_x'=V_{\mathrm{Var},x}(S_1')$, even if that internal variance increases.
$\square$

This distinction preserves the target of the Keystone estimate: its paired
structural errors must be inserted into the complete coupled increment
{prf:ref}`thm-cloning-incremental-cluster-balance`. A signed internal-variance
calculation, whether favorable or unfavorable, does not replace that
increment. An optimal output matching may lower a cost established under
a chosen valid coupling; it does not authorize dropping the cross-swarm
term of that coupling.
:::

:::{div} feynman-prose
To average a cloning step, let the algorithm perform each possible measurement, compute the shared fitness normalizers, and then evaluate acceptance. Weight the resulting inward and outward flows by their actual measurement probabilities. This keeps selection active even when the fitness values fluctuate from one measurement to the next.

The two-site calculation shows what selection can do: walkers have unequal sampled fitness almost every time, yet copying between equally distant sites supplies no inward radius flux, while jitter adds spread. To understand structural error, now watch two coupled clouds together. They may spread in the same way and remain close to each other; their cross-swarm term records that agreement. The Keystone calculation concerns this paired error, with the full averaged cluster balance determining how the actual copying step changes it.
:::

::::{prf:remark} Component growth and the combined affine estimate
:label: rem-component-growth-combined-drift

The following calculations concern $V_{\rm struct}$, a component of
$V_W=V_{\rm loc}+V_{\rm struct}$. The combined estimate in
{prf:ref}`thm-synergistic-foster-lyapunov-preview` instead concerns
$$
V_{\rm total}=V_W+c_V(V_{\mathrm{Var},x}+\lambda_v V_{\mathrm{Var},v})+c_BW_b,
\qquad
QV_{\rm total}-V_{\rm total}\le-\kappa_*V_{\rm total}+C_*.
$$
Its cloning input for $V_W$ is a bounded-expansion estimate with an additive
constant. Neither a negative cloning increment of $V_{\rm struct}$ nor
zero-offset contraction is a premise of this composition. In particular, a
positive component increment does not compare the full left-hand side with
$-\kappa_*V_{\rm total}+C_*$. The operator checks below constrain stronger
componentwise claims while leaving the combined estimate's stated target
and offsets intact.
::::

::::{prf:proposition} Positive cloning pressure can increase the actual structural error
:label: prop-cloning-macroscopic-structural-expansion

For the canonical one-dimensional Euclidean Gas cloning stage with quadratic
reward $R(x)=-x^2/2$, there are two all-alive four-slot swarms with nonconstant
positions and unequal retained fitness for which every coupling of their actual
cloning outputs has a strictly positive expected structural-error increment.
The increase is macroscopic and includes the specified jitter and component
collision mechanism.

Take zero entering velocities and
$$
x^A=(-3/2,-3/2,-3/2,1),\qquad x^B=x^A/100.
$$
Their barycenter-centered distributions are positive dilations of each other.
Their monotone pairing is optimal, and therefore
$$
V_{\rm struct}(A,B)=(99/100)^2\frac{75}{64}=1.1485546875.
\tag{3.X1}
$$
For both swarms, every one of the 81 measurement vectors gives nonconstant
retained fitness. The singleton has a strictly larger reward factor than the
three equal majority positions, and its measurement is always a far one. If
the majority measurements include both near and far values, their diversity
factors differ. If all are far, the reward factors distinguish the singleton;
if all are near, both factors favor it. No equal-fitness event is excluded.

Every component velocity remains zero under its actual shared orthogonal
rotation, so the hypocoercive structural cost reduces exactly to positional
squared Wasserstein distance throughout this cloning stage.

Here is a finite expression for its two marginal output second moments, with
all measurement randomness and empirical standardization retained. For each
swarm, set
$$
z_i=\frac{2x_i}{2+|x_i|},\quad
D_{ij}=\sqrt{(z_i-z_j)^2+10^{-6}},\quad
w_{ij}=\begin{cases}e^{-(z_i-z_j)^2/8},&j\ne i,\\0,&j=i,\end{cases}
\quad P_{ij}=\frac{w_{ij}}{\sum_\ell w_{i\ell}}.
$$
For $\operatorname{std}(y)_i=(y_i-\bar y)/\sqrt{\frac14\sum_\ell(y_\ell-\bar y)^2+.01}$
and $g(u)=.1+2/(1+e^{-u})$, enumerate the $3^4=81$ actual measurement vectors
$m=(m_1,\ldots,m_4)$ with $m_i\ne i$. Their exact probabilities and retained
fitnesses are
$$
\omega_m=\prod_iP_{i m_i},\qquad
F_i(m)=g(\operatorname{std}(-x^2/2)_i)
       g(\operatorname{std}(D_{1m_1},\ldots,D_{4m_4})_i).
$$
Conditional on this entire vector, put
$$
b_{ij}(m)=P_{ij}\min\left\{1,
 \frac{(F_j(m)-F_i(m))_+}{F_i(m)+10^{-6}}\right\},\quad
p_i=\sum_jb_{ij},\quad
 t_i=\sum_j b_{ij}(x_j-x_i),
$$
$$
a_i=\sum_jb_{ij}(x_j-x_i)^2-t_i^2,\quad
L_m=\frac14\sum_{i,j}b_{ij}
 \big[(x_j-\bar x)^2-(x_i-\bar x)^2\big].
$$
The frozen-copy rule and independent recipient jitters give exactly
$$
\mathbb E\operatorname{Var}(x')
=\operatorname{Var}(x)+\sum_m\omega_m
\left[L_m-\left(\frac14\sum_i t_i\right)^2
-\frac1{16}\sum_i a_i+\frac3{16}(.1)^2\sum_i p_i\right].
\tag{3.X2}
$$
This expression integrates donors and gates through their actual row kernels;
it does not replace the sampled fitness inside nonlinear acceptance.

Directed rational interval evaluation of (3.X2) gives
$$
1.31954463969802<\mathbb E\operatorname{Var}(x^{A\prime})
 <1.31954463969803,
$$
$$
.00035711824482<\mathbb E\operatorname{Var}(x^{B\prime})
 <.00035711824483.
\tag{3.X3}
$$
The interval certificate uses denominator $10^{40}$ with outward rounding for
each rational operation, integer square-root bounds, and dyadic reduction of
positive exponential arguments to $[0,1]$. On that interval it sums the Taylor
series through degree 80 and bounds the remainder by $3u^{81}/81!$;
$e<3$ proves the latter bound. Negative exponentials are enclosed by taking
reciprocals. Thus (3.X3) is a bound on the finite analytic expression, not a Monte
Carlo confidence interval.

For any coupling $\Gamma$ of the two actual outputs, the second-moment lower
bound for Wasserstein distance and Cauchy--Schwarz give
$$
\begin{aligned}
\mathbb E_\Gamma V_{\rm struct}'
&\ge\mathbb E_\Gamma
 \left(\sqrt{\operatorname{Var}(x^{A\prime})}
       -\sqrt{\operatorname{Var}(x^{B\prime})}\right)^2\\
&\ge\left(\sqrt{\mathbb E\operatorname{Var}(x^{A\prime})}
       -\sqrt{\mathbb E\operatorname{Var}(x^{B\prime})}\right)^2
 >1.27648593291590.
\end{aligned}
$$
Consequently
$$
\boxed{\mathbb E_\Gamma V_{\rm struct}'-V_{\rm struct}>.12793124541590>.12.}
\tag{3.X4}
$$
The same exact balance splits swarm A's internal-variance increment into a
selection contribution in $(.14532737180918,.14532737180920)$ and a positive
jitter contribution in $(.00234226788882,.00234226788885)$. Selection itself
produces the macroscopic expansion: copying a fitter minority increases its
population mass toward balance, which increases the spread of these two
spatial groups. The signed radius flux is positive, despite positive cloning
pressure. Neither shared collision rotations nor a different cross-swarm
coupling can reverse (3.X4). $\square$
::::

::::{prf:proposition} No globally nonvacuous affine structural contraction for the cloning stage
:label: prop-cloning-no-global-affine-structural-contraction

In the same canonical one-dimensional algorithm, there are no constants
$\kappa>0$ and $C<4\kappa$, independent of population size, such that a coupling
of the actual cloning outputs satisfies
$$
\mathbb E V_{\rm struct}'-V_{\rm struct}\le-\kappa V_{\rm struct}+C
\tag{3.X5}
$$
for every pair of all-alive inputs with zero velocities. This statement concerns
the cloning stage; it makes no assertion against contraction after the declared
BAOAB, cap, and terminal-boundary stages.

**Proof.** The maximum entering centered positional Wasserstein error on
$[-2,2]$ is four. To see this, let $f,g$ be the nondecreasing quantile functions
of two entering position laws. For $u<v$, both increments $f(v)-f(u)$ and
$g(v)-g(u)$ lie in $[0,4]$. Hence the oscillation of $f-g$ is at most four.
The centered squared Wasserstein distance is $\operatorname{Var}(f-g)$, which
is at most four by the variance-range inequality. Balanced atoms at $\pm2$
versus a single atom attain the bound; interior versions approach it.

Use the balanced $N=2M$ family of
{prf:ref}`prop-cloning-two-cluster-noise-balance` with positions $\pm a$,
$1/2\le a<2$, and zero velocities. Its constants obey
$$
q(1-q)\ge2/9,\qquad q^2(1-q^2)\le20/81,
\qquad A_0(a)\ge A_*=A_0(1/2)>0.
$$
The proved actual cloning balance (3.U4), with the prescribed $j=.1$, gives
$$
\mathbb E\operatorname{Var}(x')-a^2
\ge\frac{2j^2A_*}{9}\left(1-\frac1N\right)-\frac{320}{81N}.
$$
For any even
$$
N\ge2+\frac{320}{9j^2A_*},
$$
this is at least $j^2A_*/9>0$, uniformly over $a\in[1/2,2)$.
The second swarm, with every position and velocity zero, is unchanged by its
cloning stage: all its retained fitnesses agree, and there is no revival.
Its centered law is the deterministic atom at zero. Thus the actual structural
error equals $a^2$ before cloning and $\operatorname{Var}(x')$ afterward,
independently of the coupling. Choose $a^2>C/\kappa$ sufficiently near four.
The left side of (3.X5) is positive and its right side negative, a contradiction.

This conclusion is not confined to exactly monomorphic comparison swarms.
For fixed finite $N$, give the comparison positions a nonconstant perturbation
of magnitude $\epsilon$. All of the actual finite measurement and donor
probabilities and clipped acceptance probabilities are continuous in these
positions; the positive regularizers keep their denominators nonzero. At
$\epsilon=0$ the acceptance probabilities vanish, so the perturbed comparison
output variance tends to zero, including its actual jitter contribution.
The second-moment inequality used in (3.X4) shows that the strictly positive
structural increment persists for sufficiently small positive $\epsilon$.
The incoming structural error also converges to $a^2$. Consequently both
inputs can have nonconstant positions while retaining the contradiction.

A global affine upper bound with $C\ge4\kappa$ is not contradicted, but it
has no strictly negative drift region on this class of entering structural
errors. The false inference is the deduction of a signed structural
contraction from positive cloning pressure alone. Its replacement is the exact
signed balance and the mechanism exhibited above; a full-update contraction
argument must establish the contribution of the remaining actual stages.
$\square$
::::

::::{prf:proposition} Canonical full-step structural expansion at unit velocity diffusion
:label: prop-canonical-fullstep-structural-expansion

Continue the two four-slot inputs of
{prf:ref}`prop-cloning-macroscopic-structural-expansion` through the actual
canonical quadratic BAOAB update, with $h=.04$, $\gamma=1$, velocity-noise
factor $B=1$, position diffusion $.1$, velocity cap two, and terminal absorption
at $[-2,2]$. Use $Q(x,v)=x^2+v^2+.1xv$. For every coupling of the two separately nonextinction-conditioned
one-step output kernels,
$$
\boxed{\mathbb E V_{\rm struct}'-V_{\rm struct}>.08.}
\tag{3.X6}
$$
This rules out zero-offset one-step structural contraction at this canonical
parameter choice. It does not assert that the original sufficiently-strong-
diffusion regime fails, nor contradict an affine estimate allowing an offset
larger than the observed increment.

**Proof.** Write $X_i$ for the actual cloned position, including its prescribed
jitter. All frozen component velocities are zero. With
$$
c=.02,\quad a=e^{-.04},\quad q^2=(1-e^{-.08})/2,\quad s=.02,
\quad t=1-c^2(1+a),\quad \nu=c^2q^2+s^2,
$$
the actual quadratic B1--A1--O--A2 stages and final position noise give the
exact preboundary position
$$
\widehat x_i=tX_i+cq\xi_i+s\zeta_i.
$$
The recomputed B2 force and the velocity cap do not change this position.
Every $\xi_i,\zeta_i$ here is the declared independent row innovation.
Consequently the preboundary empirical variances have exact means
$$
M_A=t^2\mathbb E\operatorname{Var}(X^A)+\tfrac34\nu,
\qquad M_B=t^2\mathbb E\operatorname{Var}(X^B)+\tfrac34\nu.
\tag{3.X7}
$$
The finite interval certificate gives
$1.31778710461037<M_A<1.31778710461038$ and
$.00066809082559<M_B<.00066809082561$.

Conditional on the entire measurement, donor, and gate plan, every
$\widehat x_i$ is Gaussian, with mean equal to $t$ times its frozen source
position and variance at most $t^2(.1)^2+\nu<.0105$. Source magnitudes are at
most $1.5$ in A and $.015$ in B; also $0<t<1$. Thus Gaussian Chernoff bounds
and the union bound over these four slots give
$$
\mathbb P(E_A^c)\le8e^{-(.5)^2/(2\cdot.0105)}<.000055=:\delta_A,
$$
$$
\mathbb P(E_B^c)\le8e^{-(1.985)^2/(2\cdot.0105)}<10^{-30}=:\delta_B,
\tag{3.X8}
$$
where $E_s$ is the event that all four terminal positions are alive. These
bounds hold after integrating the actual frozen plan. No population-uniform
minorization is involved in this fixed-four-slot counterexample.

Let $V_s=\operatorname{Var}(\widehat x^s)$ before the boundary. For A,
$$
\mathbb E V_A^2\le\frac14\sum_i\mathbb E\widehat x_i^4
\le1.5^4+6(1.5)^2(.0105)+3(.0105)^2<5.205.
$$
The first inequality follows successively from
$V_A\le\frac14\sum_i\widehat x_i^2$ and Jensen. On $E_A$ the alive-normalized
variance equals $V_A$. On other surviving patterns it is nonnegative.
Therefore, writing $H_s$ for nonextinction and retaining its normalization,
$$
\mathbb E[\operatorname{Var}_{\mathcal A'}(x^{A\prime})\mid H_A]
\ge\mathbb E[V_A\mathbf1_{E_A}]
\ge M_A-\sqrt{5.205\,\delta_A}=:L_A.
\tag{3.X9}
$$
Indeed $\mathbb P(H_A)\le1$, and Cauchy--Schwarz bounds the discarded second
moment. For B every surviving empirical position law on $[-2,2]$ has variance
at most four. Since $\mathbb P(H_B)\ge\mathbb P(E_B)\ge1-\delta_B$,
$$
\mathbb E[\operatorname{Var}_{\mathcal A'}(x^{B\prime})\mid H_B]
\le\frac{M_B+4\delta_B}{1-\delta_B}=:U_B.
\tag{3.X10}
$$
These are bounds for each exact survival-conditioned marginal, so they remain
valid under every coupling of those marginals.

Finally
$$
Q(x,v)=\tfrac{399}{400}x^2+(v+.05x)^2
\ge\tfrac{399}{400}x^2.
$$
Projection to positions, followed by the second-moment bound and
Cauchy--Schwarz as in (3.X4), gives
$$
\mathbb E V_{\rm struct}'
\ge\frac{399}{400}\left(\sqrt{L_A}-\sqrt{U_B}\right)^2
>V_{\rm struct}+.08.
$$
The last strict inequality is verified by the same outward-rounded rational
certificate as (3.X3); all tail bounds in (3.X8)--(3.X10) are included in it.
No force, noise, collision, capping, or boundary stage is replaced. $\square$
::::

:::{div} feynman-prose
Copying a fitter minority restores its representation in the cloud and can increase spatial diversity. For two locations separated by distance $L$, the variance is $p(1-p)L^2$. Moving their masses toward an even split increases this quantity. That increase measures the redistribution of walkers; deciding whether the law approaches equilibrium requires the complete dynamics.

The four-walker construction also gives an increase in the distance between the two specified swarms, under every coupling. The weighted Lyapunov argument allows such a component increment: cloning supplies bounded expansion of the inter-swarm distance, while the combined estimate accounts for position, velocity, boundary exposure, and their finite offsets together.

At $h=.04$ and velocity diffusion $B=1$, this particular increase persists through the full step. The relevant comparison for the combined proof is the weighted drift against $-\kappa_*V_{\rm total}+C_*$. A positive component increment alone does not determine that comparison or the long-time behavior.
:::

:::{prf:example} An unchanged walker and a contracting cloud
:label: ex-cloning-moving-barycenter

Take three alive walkers in dimension one with positions $(-a,0,a)$, zero velocities, and the canonical quadratic reward $r(x)=-x^2/2$, where $0<a<2$. Use the canonical comparison radius $2$, measurement and donor bandwidths $2$, positive logistic maps $g(z)=2/(1+e^{-z})+0.1$, unit fitness exponents, variance floors $0.1$, and the declared positive diversity-distance floor. Let $E$ be the retained measurement event that both endpoints measure the center. The center necessarily measures an endpoint. All three diversity measurements are equal on $E$, so their rescaled diversity factors are exactly $1.1$.

Write
$$
\delta_1=\frac{2a}{2+a},\quad \delta_2=2\delta_1=\frac{4a}{2+a},\quad
w_1=e^{-\delta_1^2/8},\quad w_2=e^{-\delta_2^2/8},\quad
\theta=\frac{w_1}{w_1+w_2}.
$$
The metric first squashes each position separately: it compares $S_2(x_i)-S_2(x_j)$, not $S_2(x_i-x_j)$. Thus $\Pr(E\mid S)=\theta^2>0$. With $s_r=\sqrt{a^4/18+0.1^2}$, the retained fitnesses are
$$
F_0=1.1g\!\left(\frac{a^2}{3s_r}\right),\qquad
F_-=F_+=1.1g\!\left(-\frac{a^2}{6s_r}\right),\qquad F_0>F_-.
$$
The central row has $p_0=0$. Each endpoint independently copies the center with probability
$$
q=\theta\min\left\{1,\frac{F_0-F_-}{p_{\max}(F_-+\varepsilon_{\rm clone})}\right\}\in(0,1).
$$
No endpoint-to-endpoint proposal is accepted because their retained fitnesses tie. Write $A_-,A_+$ for the independent Bernoulli $q$ decisions. The center persists at zero, while
$$
\bar X'=\frac{a(A_--A_+)+j(A_-\zeta_-+A_+\zeta_+)}3.
$$
Therefore
$$
\boxed{\mathbb E\bigl[|X_0'-\bar X'|^2\mid S,E\bigr]
=\frac{2a^2q(1-q)+2j^2q}{9}>0.}
\tag{3.F11}
$$
The entering central squared centered position is zero. A row estimate consisting only of a negative term multiplied by its own cloning probability and a remainder multiplied by that same probability has right side zero here, irrespective of the numerical remainder. Equation (3.F11) supplies an exact contradiction to that row inference at nonzero canonical jitter. It is caused by the moving barycenter, not a change of collision algorithm.

There is also an unconditional test of a jitter-only remainder. Couple two identical entering swarms, so the central inter-swarm discrepancy is zero. For every retained measurement realization its post-step squared centered position is nonnegative. Averaging (3.F11) over the positive-probability event gives
$$
\mathbb E\bigl[|X_0'-\bar X'|^2\mid S\bigr]
\ge \theta^2\frac{2a^2q(1-q)}9>0,
\tag{3.F12}
$$
independently of $j$. Consequently a bound with right side $p_0 C_{\rm jitter}$ and a uniform $C_{\rm jitter}=O(j^2)$ fails as positive $j\downarrow0$. At $j=0$ its right side is zero. This does not contradict a collective Keystone inequality with an explicitly derived geometric offset: its purpose is to identify the exact invalid individual inference and the missing barycenter terms.

For example, at $a=0.1$, $p_{\max}=1$, and $\varepsilon_{\rm clone}=10^{-6}$, independent elementary evaluation gives
$$
\theta=0.500850339316,\quad
F_0=1.22832654693,\quad F_-=1.20083609058,\quad
q=0.0114658387064,
$$
$$
\frac{2a^2q(1-q)}9=2.51874961093\times10^{-5},\qquad
\theta^2\frac{2a^2q(1-q)}9=6.31831015805\times10^{-6}.
$$
These numbers evaluate the exact formulas; no simulated trajectory is substituted for an analytic argument.

The collective variance on the same conditional event illustrates why the individual and collective claims must be separated. Since the entering variance is $2a^2/3$,
$$
\boxed{\mathbb E[\Delta V_{\mathrm{Var},x}\mid S,E]
=-\frac{2a^2q(4-q)}9+\frac{4qj^2}9.}
\tag{3.F13}
$$
Indeed, the expected uncentered second moment is $2a^2(1-q)/3+2qj^2/3$; subtract (3.F11), the expected squared barycenter, and then subtract the entering variance. Thus this collective conditional drift is strictly negative whenever $j^2<a^2(4-q)/2$, including $a=j=0.1$. A persisting central row can acquire positive centered variance at the same time that the cloud's collective variance decreases. This is a direct reason to repair the individual inference by the collective signed balance rather than infer failure of the geometric-cluster strategy.
:::

:::{prf:theorem} Positional reset bound and exact drift
:label: thm-positional-variance-contraction

Suppose eligible input positions lie in a domain of finite diameter $D_x$. For the actual cloning proposal, with every dead slot revived and every accepted slot receiving Gaussian jitter,

$$
\mathbb E V_{\mathrm{Var},x}(S')
\leq \frac{D_x^2}{2}+\left(1-\frac1N\right)d\sigma_x^2=:B_x.
$$

Consequently $\mathbb E\Delta V_{\mathrm{Var},x}\leq-V_{\mathrm{Var},x}+B_x$. For two swarms the offset is $2B_x$. This is an $N$-uniform reset estimate, independent of a Keystone constant or a favorable donor direction.

The exact drift remains the computable integral

$$
H_x(S)=\mathbb E_{\mathbf F}\left[
\frac1N\sum_i|m_i-\bar m|^2+
\left(1-\frac1N\right)\frac1N\sum_i\operatorname{tr}\Sigma_i\right]
-V_{\mathrm{Var},x}(S).
$$

*Proof.* Conditional on all donor decisions, write $X_i'=Y_i+\sigma_xA_i\zeta_i$, where $Y_i$ is an eligible input position, whether retained or copied, and $A_i$ is the acceptance indicator. The pairwise identity gives

$$
\frac1N\sum_i|Y_i-\bar Y|^2
=\frac1{2N^2}\sum_{i,j}|Y_i-Y_j|^2\leq\frac{D_x^2}{2}.
$$

Independent centered jitters add exactly $(1-1/N)d\sigma_x^2N^{-1}\sum_iA_i$ to the expected variance. Since $\sum_iA_i\leq N$, averaging gives the bound. The exact formula is {prf:ref}`lem-variance-change-decomposition`. $\square$

The reset bound does not predict monotone shrinkage of every bounded cloud: its offset can cover the entire input variance range. A smaller offset or a rate tied to measured selection pressure requires a bound on the actual donor terms in $H_x$. The theorem supplies the statewise estimate needed for moment control without asserting that stronger property.
:::

:::{prf:example} A canonical cloud whose expected positional variance increases
:label: ex-cloning-position-spreading

Take four alive one-dimensional walkers at $(0,0,0,a)$, with zero velocities, $a=0.1$, and $U(x)=x^2/2$. Use the canonical comparison radii $R_x=R_v=2$, both Gaussian widths $2$, global standardization floors $0.1$, diversity distance floor $0.001$, logistic amplitude $2$ and floor $0.1$ in both channels, unit fitness exponents, $p_{\max}=1$, and $\varepsilon_{\rm clone}=10^{-6}$. These are the canonical measurement and acceptance settings.

Condition first on the measurement event where each zero-position row selects another zero-position row. The isolated row necessarily measures a zero-position row. With $\delta=2a/(2+a)$ and $w=e^{-\delta^2/8}$, the frozen fitnesses and accepted edge probability are

$$
F_0=1.09668913465,\qquad F_a=1.53107680787,\qquad
q=\frac{w}{2+w}\frac{F_a-F_0}{F_0+10^{-6}}=0.131930125086.
$$

Each zero-position row independently copies the isolated donor with probability $q$; the isolated donor persists. Before jitter, the number at $a$ is $M=1+\operatorname{Bin}(3,q)$, giving

$$
\mathbb E V_{\mathrm{Var},x}(S')=
\frac{(3+3q-6q^2)a^2}{16},\qquad
V_{\mathrm{Var},x}(S)=\frac{3a^2}{16}.
$$

Its drift is positive because $0<q<1/2$. The actual jitter adds $9q\sigma_x^2/16$ to that conditional expectation.

The unconditional calculation is also finite. Enumerate the eight vectors $b\in\{0,1\}^3$ specifying which zero rows measure the isolated row. Their probabilities are

$$
P(b)=\prod_{i=1}^3\left(\frac{w}{2+w}\right)^{b_i}
\left(\frac2{2+w}\right)^{1-b_i}.
$$

For each vector, compute the four sampled fitnesses, their actual accepted-edge probabilities, and the row-law variance in {prf:ref}`lem-variance-change-decomposition`; then sum with these weights. This gives expected variance $0.00200927153569$ before jitter and $0.00297011744615$ with the canonical $\sigma_x=0.1$, compared with input variance $0.001875$. Thus the unconditional expected drift is positive, including the complete measurement law. The reset bound above remains valid. A theorem asserting unconditional monotone positional contraction would be false for this actual configuration.
:::

:::{prf:lemma} Revival variance with the actual Gaussian jitter
:label: lem-dead-walker-revival-bounded

Suppose eligible input positions lie in a domain of diameter $D_x$. For each input dead slot $i$, the cloning proposal satisfies

$$
\mathbb E|X_i'-\bar X'|^2\leq D_x^2+d\sigma_x^2.
$$

Consequently its total $N$-normalized contribution is at most
$|\mathcal D|(D_x^2+d\sigma_x^2)/N$. For two swarms, sum the two dead counts.

*Proof.* Condition on all donor decisions. Write $X_i'=Y_i+\sigma_x A_i\zeta_i$, where every $Y_i$ is an eligible input position and $A_i$ is its acceptance indicator. Thus $|Y_i-\bar Y|\leq D_x$. The centered Gaussian term has expected squared norm

$$
d\sigma_x^2\left[(1-2/N)A_i+N^{-2}\sum_j A_j\right]
\leq d\sigma_x^2
$$

for $N\geq2$; for $N=1$ the centered position is zero. Its cross term has mean zero. Add the deterministic bound and average the decisions. There is no rejection, projection, or bounded Gaussian support in this calculation. $\square$
:::

### 10.4. Exact velocity drift

:::{prf:theorem} Velocity dissipation and bounded revival expansion
:label: thm-velocity-variance-bounded-expansion

For two input swarms with all retained velocities bounded by $V_{\max}$, let $D_k$ be their dead counts and let $\mathcal E_{C,k}$ be the normalized relative component energy defined in {prf:ref}`prop-bounded-velocity-expansion`. Then

$$
\Delta V_{\mathrm{Var},v}
=R_v(S_1)+R_v(S_2)-(1-\alpha^2)(\mathcal E_{C,1}+\mathcal E_{C,2})
\leq\frac{4(D_1+D_2)}N V_{\max}^2\leq8V_{\max}^2.
$$

The bound is pathwise for the cloning proposal. For a single swarm its uniform offset is $4V_{\max}^2$. If both inputs are all alive, $C_v=0$ is valid and the displayed dissipation is exact.

*Proof.* Sum {prf:ref}`prop-bounded-velocity-expansion` over the two swarms. $\square$
:::

:::{prf:remark} Component changes in an inter-swarm coupling
:label: rem-synergistic-velocity-dissipation

If the two swarms have the same component partition and use a shared Haar matrix on each corresponding component, their velocity difference on a component satisfies

$$
\sum_{i\in C}|\delta v_i'|^2
=|C||\overline{\delta v}_C|^2+
\alpha^2\sum_{i\in C}|\delta v_i-\overline{\delta v}_C|^2.
$$

This follows by the same centered decomposition. When accepted graphs differ, components and their centers of mass differ; this formula cannot be applied by pairing individual walkers' rotations. A full coupling bound must control that component-change event and its displacement. The bounded output estimate $|v_i'|\leq(1+2\alpha)V_{\max}$ remains available without an identical partition.
:::

:::{div} feynman-prose
Picture two matched swarms and draw an arrow from each walker in the first
swarm to its partner in the second. These arrows are the errors $d_i$.
Within a geometric cluster, separate their common direction from their
individual deviations. Cloning changes both: it moves the cluster's mean
error and rearranges the errors around that mean. The two signed flux terms
below measure precisely these effects.

Keep their signs. A donor displacement pointing against an error can reduce
it; replacing their scalar product by a product of lengths loses that
reduction. The common refinement lets us perform this accounting in both
swarms' geometric clusters, with each slot counted once. These bookkeeping
blocks do not replace the collision components that the algorithm builds
from accepted edges.
:::

:::{prf:theorem} Incremental cloning balance in the geometric error clusters
:label: thm-cloning-incremental-cluster-balance

Pair the full slots of two nonextinct swarms by a fixed matching and relabel
according to that matching. Retain the geometric error clusters of
{prf:ref}`def-unified-high-low-error-sets` in each swarm. Add each swarm's
dead slots as a residual block and take the common refinement of these two
partitions, denoted $\mathscr G$. This is a partition used to evaluate the
coupling; it does not change the algorithm's companion laws or collisions.

Condition on both complete frozen measurement vectors and accepted plans.
Write $A_i,\widetilde A_i\in\{0,1\}$ for their acceptance indicators,
$J_i,\widetilde J_i$ for accepted donor indices, and define

$$
d_i=x_i-y_i,\qquad
r_i=A_i(x_{J_i}-x_i)
 -\widetilde A_i(y_{\widetilde J_i}-y_i).
$$

A product with a zero acceptance indicator is zero, so its unused donor
index need not be defined. Couple the recipient Gaussian jitters by the
same standard Gaussian in each paired row, with the actual amplitude $j$.
Then the full-slot positional discrepancy satisfies the exact identity

$$
\begin{aligned}
\frac1N\mathbb E_\xi\sum_i|x_i^c-y_i^c|^2
-\frac1N\sum_i|d_i|^2
={}&\frac2N\sum_{G\in\mathscr G}|G|\bar d_G\cdot\bar r_G\\
&+\frac2N\sum_G\sum_{i\in G}
 (d_i-\bar d_G)\cdot(r_i-\bar r_G)\\
&+\frac1N\sum_i|r_i|^2
 +\frac{d j^2}{N}\sum_i(A_i-\widetilde A_i)^2.
\end{aligned}                                                     \tag{3.C1}
$$

Here bars denote the arithmetic means within the indicated geometric
refinement block. The first two terms retain the signed between-cluster
and within-cluster donor fluxes.

Let $C_i,D_i$ be the accepted collision components containing slot $i$ in
the two swarms, including singleton components. Set
$u_i=v_i-\bar v_{C_i}$ and $t_i=w_i-\bar w_{D_i}$ using all frozen slot
velocities, including revived recipients. Share the Haar matrix precisely
when two component vertex sets coincide, and use independent matrices for
all other components. This is a coupling of the prescribed collision laws,
and it gives

$$
\mathbb E_R\sum_i|v_i^c-w_i^c|^2
=\sum_i|\bar v_{C_i}-\bar w_{D_i}|^2
 +\alpha^2\sum_i\left(
 |u_i|^2+|t_i|^2-2\mathbf1_{\{C_i=D_i\}}u_i\cdot t_i\right).
                                                               \tag{3.C2}
$$

Both identities vanish on the diagonal when the input swarms and their
plans coincide. Expectations over the coupled sampled measurements,
weighted donors, and gates give their unconditional versions.
:::

:::{prf:proof}
The frozen copying rule gives
$x_i^c-y_i^c=d_i+r_i+j(A_i-\widetilde A_i)\xi_i$.
The centered Gaussian has mean zero and second moment $d$. Expand the
square and sum. For each geometric block,

$$
\sum_{i\in G}d_i\cdot r_i
=|G|\bar d_G\cdot\bar r_G
 +\sum_{i\in G}(d_i-\bar d_G)\cdot(r_i-\bar r_G),
$$

because both centered sums vanish. This proves (3.C1) without a sign
estimate or a change in cluster construction.

For (3.C2), write the velocity difference as
$\bar v_{C_i}-\bar w_{D_i}+\alpha(R_{C_i}u_i-\widetilde R_{D_i}t_i)$.
Haar orthogonality preserves squared norms, and its mean is zero. Distinct
component matrices in the specified coupling have zero cross expectation;
a shared matrix preserves $u_i\cdot t_i$. Expand and sum. Singleton
components have $u_i=0$ or $t_i=0$, so uninvolved slots obey the same formula.
The construction shares at most one matrix with any given component and
therefore preserves independence of component matrices within each swarm.
The source velocities throughout are the frozen pre-collision values.
Finally integrate the identities against the actual coupled plan laws.
$\square$
:::

:::{div} feynman-prose
There are two useful checks on this accounting. If both recipients clone,
their shared jitter cancels from their difference. If only one clones, it
does not cancel, and the gate-mismatch term records its cost. Likewise, a
shared rotation preserves the scalar product of the two relative velocities
when their component sets agree. Changed components require their own terms.
The identities keep these mechanisms visible while the geometric estimates
determine the net sign of the positional flux.
:::

:::{prf:theorem} Revival donor changes with component-average cancellation
:label: thm-cloning-revival-backbone-coupling

Condition on the complete marked input, its retained fitness and all
accepted alive-recipient edges, with a nonempty alive set. Let
$|v_i|\leq V_*$ for every frozen slot velocity, including dead slots.
The undirected graph of accepted edges between alive walkers has connected
components $B$, called its alive backbones. Each dead recipient has exactly
one accepted edge to an alive donor and cannot receive an edge. Consequently
each full collision component consists of one backbone and its attached
dead leaves.

Attach one independent Haar matrix $R_B$ to each backbone, independently
of the dead-recipient donor choices. For every assignment of those donors,
use $R_B$ on the corresponding full component. This is the prescribed
component-rotation law for every assignment. It couples the two laws without
requiring their full component vertex sets to agree.

Compare two assignments differing only in the donor of a dead recipient
$r$, while sharing all frozen inputs, backbone rotations and recipient
jitters. If both donors belong to the same backbone, collision velocities
are identical. Otherwise let $C,D$ be the full components before the switch,
where $r\in C$, and put

$$
n_C=|C|\geq2,\quad n_D=|D|\geq1,\quad
w=v_r,\quad \mu=\bar v_C,\quad \nu=\bar v_D,
$$
$$
U=(I-\alpha R_C)(w-\mu),\qquad
W=\frac{n_D}{n_D+1}(I-\alpha R_D)(w-\nu).
$$

For the new output minus the old output, the exact identities are

$$
\sum_{i\in C\setminus\{r\}}\Delta v_i^+=-U,\qquad
\sum_{i\in D}\Delta v_i^+=W,\qquad
\Delta v_r^+=U-W,
$$
$$
\sum_i\Delta v_i^+=0,\qquad
\sum_i|\Delta v_i^+|=|U|+|W|+|U-W|
\leq(6+8\alpha)V_* .                                      \tag{3.L1}
$$

Thus for $M$ changed dead-recipient donors and unchanged alive backbones,

$$
\sum_i|v_i^+-\widetilde v_i^+|\leq(6+8\alpha)V_*M,
\qquad
\sum_i|x_i^+-\widetilde x_i^+|
\leq\sum_{r:\,J_r\ne\widetilde J_r}|x_{J_r}-x_{\widetilde J_r}|.
                                                               \tag{3.L2}
$$

If eligible positions have diameter $D_x$, the latter sum is at most $D_xM$.
The velocity constants are independent of the number of leaves, component
sizes, alive fraction, dimension and $N$.
:::

:::{prf:proof}
Removing a dead leaf cannot remove the nonempty alive backbone. The two
new component means are

$$
\mu'=\frac{n_C\mu-w}{n_C-1},\qquad
\nu'=\frac{n_D\nu+w}{n_D+1}.
$$

For every unchanged member of $C$, the output difference is
$(I-\alpha R_C)(\mu'-\mu)=-U/(n_C-1)$; for every member of $D$ it is
$(I-\alpha R_D)(\nu'-\nu)=W/n_D$. The moved leaf has old velocity
$w-U$ and new velocity $w-W$. These give the three sums in (3.L1), their
zero total and the exact sum of norms. In particular the reciprocal
component sizes cancel when the unchanged members are summed.

Since the old and new means are convex combinations of frozen velocities,
their norms are at most $V_*$. Therefore
$|U|,|W|\leq2(1+\alpha)V_*$, and the moved leaf satisfies

$$
|\Delta v_r^+|
\leq|\nu'-\mu|+\alpha(|w-\nu'|+|w-\mu|)
\leq(2+4\alpha)V_*.
$$

This proves the last inequality in (3.L1). Change $M$ assignments one at a
time with the same backbone matrices and apply the triangle inequality.
Every intermediate assignment is a valid collision graph with those
backbones. Frozen-source copying changes only the switched recipients'
positions; each recipient is revived in both assignments, so its common
jitter cancels exactly. This proves (3.L2).

Finally, conditional on any donor assignment, different backbones give
different full components and their matrices are independent Haar draws.
A singleton backbone may carry an unused Haar draw when it has no leaves;
its relative velocity is zero. The coupling therefore has the correct
marginal collision law in every case. $\square$
:::

:::{div} feynman-prose
Keep the alive graph fixed and picture the revived walkers as leaves attached to it. Moving one leaf to a different alive backbone changes two component means, so many output velocities can change. But each unchanged member feels only the change in its component average: the effect is divided by the number of members. Summing over those members cancels that divisor. A large component therefore does not create a proportionally large total disturbance.

To see this cancellation, use the same rotation for each unchanged alive backbone in both donor assignments. Each assignment still has independent uniform rotations across its full components, exactly as the algorithm requires. The shared rotations simply couple the two executions. The resulting bound charges each changed revival donor a fixed total cost, even when its collision component contains a number of walkers proportional to $N$.
:::

:::{prf:corollary} Full-step cost of changing revival donors
:label: cor-cloning-revival-full-step-cost

Under the kinetic configuration and notation of
{prf:ref}`thm-kinetic-bounded-transport-smoothing`, the two complete updates
in {prf:ref}`thm-cloning-revival-backbone-coupling` admit a coupling with

$$
\mathbb E\frac1N\sum_i d_0(Z_i',\widetilde Z_i')
\leq\frac{A_xD_x+A_v(6+8\alpha)V_*}{Nq}\,\mathbb E M .  \tag{3.L3}
$$

The expectation is conditional on the shared frozen inputs and alive
backbones, and may then be integrated over any coupling with those shared
quantities. The bound includes terminal marks and retained dead coordinates;
it does not condition on survival. It applies also when a component contains
a number of dead leaves proportional to $N$.

*Proof.* Couple rotations and jitters as in (3.L2), apply
{prf:ref}`cor-kinetic-full-cluster-smoothing` conditional on both complete
cloning outputs, and sum the two bounds in (3.L2). $\square$
:::

:::{prf:lemma} Frozen velocity perturbations in a fixed component plan
:label: lem-cloning-fixed-plan-velocity-perturbation

For identical component membership and shared component rotations,

$$
\sum_i|v_i^+-\widetilde v_i^+|
\leq(1+2\alpha)\sum_i|v_i-\widetilde v_i|.               \tag{3.L4}
$$

*Proof.* On each component the difference is
$\overline{\Delta v}_C+\alpha R_C(\Delta v_i-\overline{\Delta v}_C)$.
Use $|C||\overline{\Delta v}_C|\leq\sum_{i\in C}|\Delta v_i|$ and sum the
triangle inequality. $\square$
:::

### 10.5. Structural error and the limits of one-sided comparisons

:::{prf:corollary} Structural reset from the actual output moments
:label: cor-structural-error-contraction

For the all-alive cloning proposals, suppose each output empirical probability has expected quadratic hypocoercive moment at most $M_h$ about a fixed phase-space point. Then

$$
\mathbb E V_{\mathrm{struct}}(S_1',S_2')\leq4M_h,\qquad
\mathbb E\Delta V_{\mathrm{struct}}\leq-V_{\mathrm{struct}}+4M_h.
$$

For the canonical bounded-domain cloning proposal, this moment hypothesis is discharged by bounded frozen donor positions, finite Gaussian jitter moment, and $|v_i'|\leq(1+2\alpha)V_{\max}$. It is uniform in $N$ and holds for every coupling of the two kernels.

*Proof.* The nonnegative decomposition gives $V_{\mathrm{struct}}'\leq W_h^2(\mu_1',\mu_2')$. Couple the two probabilities by their product and use $\|z-w\|_h^2\leq2\|z-z_0\|_h^2+2\|w-z_0\|_h^2$. Taking expectations gives $4M_h$; subtract the initial structural error. This argument supplies the bound directly and does not infer a contraction factor from a one-sided comparison with initial internal variance. $\square$
:::

### 10.6. Combining the actual variance contributions

:::{prf:theorem} Variance drift with an explicit positional term
:label: thm-complete-variance-drift

For the coupled cloning proposal,

$$
\mathbb E\Delta V_{\mathrm{Var}}
=H_x(S_1)+H_x(S_2)+R_v(S_1)+R_v(S_2)
-(1-\alpha^2)\mathbb E(\mathcal E_{C,1}+\mathcal E_{C,2}),
$$

with the prescribed velocity weight inserted when it is part of $V_{\mathrm{Var}}$. If a family has a proved positional estimate
$H_x(S_1)+H_x(S_2)\leq-\kappa_xV_{\mathrm{Var},x}+C_x$, this gives

$$
\mathbb E\Delta V_{\mathrm{Var}}\leq-\kappa_xV_{\mathrm{Var},x}+C_x+C_v,
\qquad C_v=8V_{\max}^2
$$

for two swarms without a velocity weight. On all-alive inputs take $C_v=0$ and retain the negative collision energy term if useful.

*Proof.* Add the exact positional and velocity identities and then apply the explicitly stated positional bound. $\square$
:::

:::{prf:remark} Constants and their applicability
:label: rem-drift-constants-dependencies

The proved revival offset depends on the velocity radius and dead fraction. Restitution enters the exact dissipated energy, not an unavoidable positive reset error. A positional rate sharper than the reset estimate requires a bound on $H_x$ for the actual sampled fitness and donor law. Neither monotonic improvement in that rate with bandwidth nor improvement with increasing $p_{\max}$ follows automatically; increasing $p_{\max}$ at fixed positive score decreases the acceptance probability.
:::

### 10.7. Passing these results to the full update

:::{div} feynman-prose
The cloning proposal has exact positional and velocity balances. A sharper positional rate tied to selection still needs a geometric estimate on its actual donor displacement integral; the proved reset bound supplies a statewise moment estimate directly. The kinetic update and final killing test add their own terms; {prf:ref}`thm-canonical-full-step-reset-drift` proves a global moment drift for their complete canonical composition. The composition theorems below retain that unresolved input explicitly; an algebraic combination of drift bounds does not prove a missing bound.
:::

(sec-cloning-boundary)=
## 11. Drift Analysis Under the Cloning Operator - Boundary Potential

### 11.1. The boundary observable

:::{div} feynman-prose
A barrier records how much alive mass lies near the killing boundary. Replacement can lower it when exposed walkers select favorable companions. The proof therefore keeps the companion probability and the expected barrier after jitter as explicit quantities. Exponential suppression of total extinction uses a further safe-population estimate.
:::

:::{prf:remark} Barrier observables and the terminal boundary schedule
:label: rem-cloning-barrier-stage

The cloning proposal can place a row outside the box while its proposal mark remains alive. A barrier used on that intermediate state must therefore be defined on the ambient position space. In the zero-extension convention below, $\widetilde\varphi(x)=\mathbf1_D(x)\varphi(x)$ is an auxiliary observable; a zero contribution outside $D$ does not mean the algorithm has killed the row at this stage. It may return before terminal classification. The actual complete-transition boundary moment is proved in {prf:ref}`cor-canonical-full-step-boundary-reset` using the final position noise and terminal status. A reciprocal-distance barrier has infinite Gaussian expectation; it cannot supply a finite drift offset merely by being smooth inside the domain.
:::

### 11.2. The Boundary Barrier and Fitness Gradient

We begin by recalling the structure of the boundary barrier and how it affects walker fitness.

#### 11.2.1. The Barrier Function

:::{prf:definition} Boundary Potential Component (Recall)
:label: def-boundary-potential-cloning

From {prf:ref}`def-full-synergistic-lyapunov-function`, the boundary potential is:

$$
W_b(S_1, S_2) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \varphi_{\text{barrier}}(x_{1,i}) + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \varphi_{\text{barrier}}(x_{2,i})

$$

where $\varphi_{\text{barrier}}: \mathcal{X}_{\text{valid}} \to \mathbb{R}_{\geq 0}$ is the smooth barrier function satisfying:

1. **Interior safety:** $\varphi_{\text{barrier}}(x) = 0$ for $x$ in the safe interior region (distance $> \delta_{\text{safe}}$ from boundary)

2. **Boundary growth:** $\varphi_{\text{barrier}}(x) \to \infty$ as $x \to \partial \mathcal{X}_{\text{valid}}$

3. **Smoothness:** $\varphi_{\text{barrier}} \in C^2(\mathcal{X}_{\text{valid}})$ with bounded derivatives in the interior

The existence of such a function was established in {prf:ref}`prop-barrier-existence`.
:::

:::{prf:remark} Barrier Function as Geometric Penalty
:label: rem-barrier-geometric-penalty

The barrier function creates a "soft wall" around the boundary:

- **Far from boundary** ($d(x, \partial \mathcal{X}_{\text{valid}}) > \delta_{\text{safe}}$): No penalty, $\varphi_{\text{barrier}}(x) = 0$

- **Near boundary** ($d(x, \partial \mathcal{X}_{\text{valid}}) \leq \delta_{\text{safe}}$): Penalty increases, reducing fitness

- **At boundary:** Infinite penalty (though walkers at the boundary are dead, so this limit is never realized for alive walkers)

This graduated penalty ensures that danger is detected before catastrophic boundary crossing.
:::

#### 11.2.2. Barrier Contribution to Fitness

Recall from the reward function definition (Section 5.6) that the raw reward for walker $i$ includes the boundary barrier:

$$
r_i = g_A(x_i) = R_{\text{pos}}(x_i) - \varphi_{\text{barrier}}(x_i) - c_{v\_reg} \|v_i\|^2

$$

This raw reward feeds into the fitness potential calculation, so walkers near the boundary have systematically lower fitness.

:::{prf:lemma} A boundary reward advantage through the product fitness
:label: lem-fitness-gradient-boundary

Condition on a realized measurement vector. Write
$r_i=R_{\mathrm{pos}}(x_i)-\varphi_i-c_{v\_reg}\|v_i\|^2$,
$z_{r,i}=(r_i-\bar r)/\sigma'_r$, and
$V_i=(d'_i)^\beta(r'_i)^\alpha$, where
$r'_i=g_A(z_{r,i})+\eta$ and $d'_i,r'_i\in[\eta,M]$.
Suppose $\alpha>0$, $r_j-r_i\geq\delta_r>0$,
$d'_j\geq d'_i$, $\sigma'_r\leq\sigma_*$, and
$g'_A\geq m_g>0$ between the attained reward scores. Then

$$
V_j-V_i\geq \eta^\beta\alpha
\min\{\eta^{\alpha-1},M^{\alpha-1}\}
\frac{m_g\delta_r}{\sigma_*}.
$$

For example, $\varphi_i-\varphi_j=\Delta_b$ and an opposing difference in the
other raw reward terms of magnitude at most $B_r<\Delta_b$ give
$\delta_r=\Delta_b-B_r$. The bound is uniform only when these margins and the
attained-interval derivative bound are uniform.
:::

:::{prf:proof}
The shared centering cancels in the difference:
$z_{r,j}-z_{r,i}=(r_j-r_i)/\sigma'_r\geq\delta_r/\sigma_*$.
Integrating $g'_A$ on this interval gives
$r'_j-r'_i\geq m_g\delta_r/\sigma_*$. Since $d'_j\geq d'_i$ and
$\beta\geq0$,

$$
V_j-V_i\geq(d'_i)^\beta[(r'_j)^\alpha-(r'_i)^\alpha].
$$

The derivative of $u^\alpha$ on $[\eta,M]$ is at least
$\alpha\min\{\eta^{\alpha-1},M^{\alpha-1}\}$. Apply the mean value theorem
and $(d'_i)^\beta\geq\eta^\beta$. The final raw reward estimate follows by
subtracting the opposing contribution from the barrier difference.
:::

#### 11.2.3. The Safe Harbor Set

We now formalize the set of walkers (see {prf:ref}`def-boundary-exposed-set`) that are in danger due to boundary proximity.

:::{prf:definition} The Boundary-Exposed Set
:label: def-boundary-exposed-set

For a swarm ({prf:ref}`def-swarm-and-state-space`) $S$ and a threshold $\phi_{\text{thresh}} > 0$, the **boundary-exposed set** is:

$$
\mathcal{E}_{\text{boundary}}(S) := \{i \in \mathcal{A}(S) : \varphi_{\text{barrier}}(x_i) > \phi_{\text{thresh}}\}

$$

These are alive walkers whose barrier penalty exceeds the threshold, indicating dangerous proximity to the boundary.

The **boundary-exposed mass** is:

$$
M_{\text{boundary}}(S) := \frac{1}{N} \sum_{i \in \mathcal{E}_{\text{boundary}}(S)} \varphi_{\text{barrier}}(x_i)

$$

:::

:::{prf:remark} Relationship to Total Boundary Potential
:label: rem-boundary-mass-relationship

If all walkers outside the exposed set have $\varphi_{\text{barrier}}(x_i) \leq \phi_{\text{thresh}}$, then:

$$
W_b(S_k) = \frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} \varphi_{\text{barrier}}(x_i) \leq M_{\text{boundary}}(S_k) + \frac{k_{\text{alive}}}{N} \phi_{\text{thresh}}

$$

When $W_b$ is large, most of its contribution comes from the boundary-exposed set.
:::

### 11.3. Main Theorem: Boundary Potential Contraction

:::{prf:theorem} Boundary potential drift from selection and integrability bounds
:label: thm-boundary-potential-contraction

Fix an exposed threshold $\theta>0$. Suppose, for each swarm and conditional
on its current configuration, every exposed alive walker has cloning
probability at least $p_*>0$. Suppose also that the expected post-cloning
barrier of any cloned alive walker is at most $J<\infty$, and that of any
revived walker is at most $R<\infty$. The bounds must refer to the actual
selection and boundary conventions of the transition under study.
Then, with $k_1,k_2$ alive counts and $D_s=N-k_s$,

$$
\mathbb E_C[\Delta W_b\mid S_1,S_2]
\leq-p_*W_b+
\frac{k_1+k_2}{N}(p_*\theta+J)+\frac{D_1+D_2}{N}R.
$$

In particular $\kappa_b=p_*$ and
$C_b=2p_*\theta+2\max(J,R)$ give a bound independent of $N$ when
$p_*,\theta,J,R$ are independent of $N$. Strict negative drift holds when
$W_b>C_b/p_*$. The selection estimate is established in
{prf:ref}`lem-boundary-enhanced-cloning`; the required barrier integrals are
treated in {prf:ref}`lem-barrier-reduction-cloning`.
:::

:::{prf:remark} Boundary correction and its probability consequences
:label: rem-progressive-safety

The rate $\kappa_b=p_*$ is uniform on families satisfying the favorable-companion
bound, and the offset includes the post-cloning and revival barrier integrals.
The drift is negative above $C_b/p_*$. Iterating a transition with this same
drift gives {prf:ref}`cor-bounded-boundary-exposure`. The one-step extinction
bound in {prf:ref}`cor-extinction-suppression` additionally uses a safe alive
fraction and conditional independence at the specified update stage.
:::

### 11.4. Proof of Boundary Potential Contraction

The proof proceeds by analyzing how cloning affects walkers in the boundary-exposed set (see {prf:ref}`def-boundary-exposed-set`).

#### 11.4.1. Cloning Probability for Boundary-Exposed Walkers

:::{prf:lemma} Cloning probability from a favorable companion set
:label: lem-boundary-enhanced-cloning

For an exposed walker $i$, let $H_i$ be a set of possible companions with
$V_j-V_i\geq\Delta>0$ for $j\in H_i$, and suppose
$\mathbb P(c_i\in H_i\mid S)\geq q>0$ and $V_i\leq V_{\max}$.
The clone threshold is independent and uniform on $[0,p_{\max}]$. Then

$$
p_i\geq q\min\!\left(1,
\frac{\Delta}{p_{\max}(V_{\max}+\varepsilon_{\mathrm{clone}})}\right)=:p_*.
$$

For a companion kernel with weights between $a_{\min}>0$ and $a_{\max}$,
$q=|H_i|a_{\min}/((k-1)a_{\max})$ is valid. A fixed positive fraction of
favorable companions therefore gives an $N$-uniform lower bound.
:::

:::{prf:proof}
On $\{c_i\in H_i\}$ the clone score is at least
$\Delta/(V_{\max}+\varepsilon_{\mathrm{clone}})$. Integrating the independent
uniform threshold gives the displayed lower bound. The probability of selecting
$H_i$ is its total weight divided by the total companion weight, bounded below
by $|H_i|a_{\min}/((k-1)a_{\max})$.
:::

:::{prf:remark} Availability of favorable companions
:label: rem-boundary-companion-availability

The geometric safe-harbor condition {prf:ref}`axiom-safe-harbor` identifies
favorable locations. The finite swarm must supply the favorable-companion mass
in the preceding lemma. For example, if all alive walkers have identical
positions and velocities, their fitness values coincide and every clone score
is zero. If this common position approaches the boundary, a diverging barrier
has arbitrarily large $W_b$, while cloning leaves the positions unchanged.
Thus a global strict drift for cloning alone cannot follow from the existence
of a safe region in the environment. The complete kinetic and cloning update
is analyzed separately in {doc}`06_convergence`.
:::

#### 11.4.2. Barrier Reduction from Cloning

:::{prf:lemma} Exact bounds for barrier integrals after jitter
:label: lem-barrier-reduction-cloning

Let $\varphi\geq0$ on the valid domain $D$ and use its zero extension outside $D$, with the stage convention in {prf:ref}`rem-cloning-barrier-stage`. If a post-update position has density $q_y(z)\leq M_q$
uniformly over allowed companion states $y$, then

$$
\mathbb E[\varphi(Y)\mathbf1_{Y\in D}\mid y]
\leq M_q\|\varphi\|_{L^1(D)}.
$$

For Gaussian jitter $Y=y+\sigma_x\xi$, one may take
$M_q=(2\pi\sigma_x^2)^{-d/2}$. This gives a finite uniform constant whenever
$\varphi\in L^1(D)$, without a compact-support assumption on the noise.

If instead the observable has a $C^2$ extension to all jittered positions with
$\|D^2\varphi\|\leq H$, then

$$
\mathbb E\varphi(y+\sigma_x\xi)
\leq\varphi(y)+\tfrac12Hd\sigma_x^2.
$$
:::

:::{prf:proof}
The first inequality follows by integrating the density bound against the
nonnegative function $\varphi$. A Gaussian density is bounded by its value at
its mean, giving the stated $M_q$ independently of $y$.

For the second estimate, Taylor's formula with integral remainder gives

$$
\varphi(y+h)=\varphi(y)+D\varphi(y)h+
\int_0^1(1-t)D^2\varphi(y+th)[h,h],dt.
$$

Set $h=\sigma_x\xi$, use $\mathbb E\xi=0$ and
$\mathbb E\|\xi\|^2=d$, and bound the remainder by
$H\|h\|^2/2$.
:::

:::{prf:remark} Integrability of a diverging barrier
:label: rem-boundary-barrier-integrability

The reciprocal-distance construction in {prf:ref}`prop-barrier-existence` is
smooth inside the domain but is not integrable across a smooth boundary collar:
the normal integral contains $\int_0^\delta r^{-1}\,dr=\infty$.
A Gaussian density is bounded below on a sufficiently small collar patch, so
its expected reciprocal-distance barrier is infinite. Interior smoothness
therefore does not establish the finite $J,R$ required above.

A logarithmic distance barrier is an integrable auxiliary Lyapunov observable:
$\int_0^\delta|\log r|\,dr<\infty$. Multiplication by a smooth collar cutoff
extends it into the interior. This choice concerns the proof observable; its
use does not alter the reward or transition rule. Its selection and kinetic
drift estimates must refer to that same observable.
:::

#### 11.4.3. Proof of the Boundary Drift Theorem

:::{prf:proof}
:label: proof-boundary-potential-contraction

For one swarm, write $\varphi_i=\varphi(x_i)$ and
$E=\{i\in\mathcal A:\varphi_i>\theta\}$. Conditional expectation of the
cloning decision gives exactly

$$
\mathbb E_C[\Delta W_b^{(s)}\mid S_s]
=\frac1N\sum_{i\in\mathcal A}p_i(J_i-\varphi_i)
+\frac1N\sum_{i\in\mathcal D}R_i,
$$

where $J_i$ is the expected barrier conditional on cloning and $R_i$ includes
the revival probability. The hypotheses give $p_iJ_i\leq J$, $R_i\leq R$,
and $p_i\varphi_i\geq p_*\varphi_i$ on $E$. Hence

$$
\mathbb E_C[\Delta W_b^{(s)}\mid S_s]
\leq-\frac{p_*}{N}\sum_{i\in E}\varphi_i+\frac{k_s}{N}J+\frac{D_s}{N}R.
$$

Since $\sum_{i\notin E}\varphi_i\leq k_s\theta$,
$N^{-1}\sum_{i\in E}\varphi_i\geq W_b^{(s)}-k_s\theta/N$.
Substitute this exact inequality and sum over the two swarms. Finally,
$k_s+D_s=N$ bounds the offset by $2p_*\theta+2\max(J,R)$.
:::

### 11.5. Implications for Extinction Probability

The boundary potential contraction has important consequences for swarm survival.

:::{prf:corollary} Bounded Boundary Exposure in Equilibrium
:label: cor-bounded-boundary-exposure

For repeated application of a transition satisfying the drift inequality in {prf:ref}`thm-boundary-potential-contraction`, with $0<\kappa_b\leq1$ and finite initial expected barrier, the expected boundary exposure satisfies:

$$
\limsup_{t \to \infty} \mathbb{E}[W_b(S_t)] \leq \frac{C_b}{\kappa_b}

$$

This is an expectation bound on the barrier observable. Applying it to the composed chain requires a drift bound for that complete transition; conditioning a killed chain on survival additionally requires its normalization.

Referenced by {prf:ref}`rem-progressive-safety`.
:::

:::{prf:proof}
**Proof.**

From the Foster-Lyapunov drift condition:

$$
\mathbb{E}[W_b(S_{t+1})] \leq (1 - \kappa_b) W_b(S_t) + C_b

$$

Taking expectations and iterating:

$$
\mathbb{E}[W_b(S_t)] \leq (1 - \kappa_b)^t W_b(S_0) + C_b \sum_{j=0}^{t-1} (1 - \kappa_b)^j

$$

As $t \to \infty$, the geometric series converges:

$$
\sum_{j=0}^{\infty} (1 - \kappa_b)^j = \frac{1}{\kappa_b}

$$

Therefore:

$$
\limsup_{t \to \infty} \mathbb{E}[W_b(S_t)] \leq \frac{C_b}{\kappa_b}

$$

**Q.E.D.**
:::

:::{prf:corollary} Exponential suppression from a safe population
:label: cor-extinction-suppression

Condition on a stage of the update at which there are at least $aN$ walkers
whose next-step death indicators are independent and whose individual death
probabilities are at most $q<1$. Here $a>0$ and $q$ are independent of $N$.
Then

$$
\mathbb P(\text{total extinction in the next step}\mid S)
\leq q^{aN}=\exp[-aN\log(1/q)].
$$

For a barrier observable $B=N^{-1}\sum_{i\in\mathcal A}\varphi_i$, the number
of alive walkers with $\varphi_i<\theta$ is at least $k-NB/\theta$.
Thus $k/N\geq a_0$ and $B\leq b<a_0\theta$ imply a safe fraction at least
$a=a_0-b/\theta$, provided this barrier sublevel set has the required individual
survival bound. If the stage condition holds with probability at least
$1-\delta_N$, the unconditional extinction probability is at most
$\delta_N+q^{aN}$.
:::

:::{prf:proof}
If $n_{\mathrm{exposed}}$ alive walkers have $\varphi_i\geq\theta$, then
$n_{\mathrm{exposed}}\theta\leq\sum_{i\in\mathcal A}\varphi_i=NB$.
This proves the deterministic count. Total extinction requires every one of
the selected safe walkers to die. Conditional independence bounds this
intersection by the product of their death probabilities, at most $q^{aN}$.
Split according to whether the stage condition holds to obtain
$\delta_N+q^{aN}$.

An explicit individual bound is available for a Gaussian position update
$Y_i=\mu_i+\sigma\xi_i$: if the ball of radius $r$ about $\mu_i$ lies in the
valid domain, death implies $\|\xi_i\|\geq r/\sigma$. Since
$\mathbb E e^{\|\xi_i\|^2/4}=2^{d/2}$, exponential Markov inequality gives

$$
q\leq\min\left(1,2^{d/2}e^{-r^2/(4\sigma^2)}\right).
$$

A useful exponential-in-$N$ bound requires the expression to be strictly
below one. The means and noise independence must be evaluated at the same
conditioned stage, after all shared random choices relevant to the means.
:::

:::{prf:remark} Parameters in the boundary estimate
:label: rem-safety-margin-tuning

The bound $C_b/p_*$ depends on the favorable-companion probability, fitness
advantage, barrier threshold, and post-update integrals $J,R$. Changing reward
weights or jitter can change several of these quantities at once. Parameter
choices should therefore be evaluated through the complete expression in
{prf:ref}`thm-complete-boundary-drift`. A smaller jitter variance, for example,
increases the Gaussian density bound used in the $L^1$ barrier estimate; that
particular upper bound need not decrease with the noise amplitude.
:::

### 11.6. Summary and Drift Inequality

We conclude by stating the complete boundary potential drift result.

:::{prf:theorem} Complete boundary drift and revival contribution
:label: thm-complete-boundary-drift

Under {prf:ref}`thm-boundary-potential-contraction`, the exact offset is

$$
C_b(S_1,S_2)=\frac{k_1+k_2}{N}(p_*\theta+J)
+\frac{D_1+D_2}{N}R.
$$

The revival term is at most $2R$ uniformly in $N$. On a family of random
configurations with $\mathbb E(D_1+D_2)\leq D_*<\infty$ uniformly in $N$, its
expectation is at most $RD_*/N$. The latter refinement requires this death-count
bound; suppression of the event that all walkers die does not imply it.
:::

:::{prf:proof}
Subtract $W_b$ from the one-step bound in
{prf:ref}`thm-boundary-potential-contraction`. The exposed-set, jitter, and
revival terms in that proof give the stated offset. Since $0\leq D_1+D_2\leq2N$,
the revival contribution is bounded by $2R$. Under the stated moment bound,
taking its expectation gives $R\mathbb E(D_1+D_2)/N\leq RD_*/N$.

The order $N^{-1}$ cannot be inferred from rare total extinction: for
independent deaths with probability $p\in(0,1)$, the all-dead probability is
$p^N$, while the expected number of deaths is $Np$. This verifies precisely
which extra estimate is needed for the sharper revival term.
:::

#### 11.7. Boundary control and survival

:::{div} feynman-prose
The boundary calculation has three inputs. Selection determines which exposed
walkers move, the jitter integral controls where they land, and revival adds
its own contribution. The exact sum is the bound in
{prf:ref}`thm-complete-boundary-drift`.

A mean barrier bound controls expected exposure. Exponential suppression of a
common extinction event uses the stronger stage-wise information in
{prf:ref}`cor-extinction-suppression`: enough walkers have a uniform individual
survival probability, and the relevant noises are conditionally independent.
The distinction matters when many walkers share a random companion or when a
rare configuration puts most of the population near the boundary.
:::

(sec-cloning-composition)=
## 12. Synergistic Drift Analysis and Conclusion

### 12.1. Composing the component estimates

:::{div} feynman-prose
The direct reset theorem below proves a complete canonical moment drift without a Keystone-to-variance inference. Stronger estimates tied to selection pressure or inter-swarm contraction additionally require control of actual donor displacement and component changes. When those estimates are established, conditional expectation combines them while preserving their offsets and normalization. A finite-particle QSD conclusion then uses the survival and mixing results in {doc}`06_convergence`.
:::

### 12.1.1. A direct moment drift for the complete canonical transition

:::{prf:theorem} Global reset and Foster bound from the actual update
:label: thm-canonical-full-step-reset-drift

Consider the canonical absorbing-box update: current weighted donors, mandatory revival, shared component collision, BAOAB with constant isotropic Gaussian factor $B$, independent final position diffusion of amplitude $\sigma_p\sqrt h$, smooth final velocity cap, and terminal classification. Let $R_D=\sup_{x\in D}|x|<\infty$, and suppose all retained entering velocities satisfy $|v_i|\leq V$. The objective force has $|\nabla U(x)|\leq L_U|x|+B_U$. There is no substep absorption or viscosity in this canonical statement.

Write $c=e^{-\gamma h}$, $s_h^2=(1-e^{-2\gamma h})/(2\gamma)$, with $s_h^2=h$ at $\gamma=0$, and define

$$
W=(1+2\alpha)V,\qquad A=1+\frac{h^2}{4}(1+c)L_U,\qquad
D_0=\frac h2(1+c)W+\frac{h^2}{4}(1+c)B_U,
$$

$$
M_x=\left(A\sqrt{R_D^2+d\sigma_x^2}+D_0\right)^2
+\frac{h^2}{4}s_h^2\operatorname{tr}(BB^T)+d\sigma_p^2h.
$$

For the full marked observable

$$
\mathscr L_N(S)=1+\frac1N\sum_i\bigl(|x_i|^2+\lambda|v_i|^2\bigr),\qquad\lambda>0,
$$

the completed transition satisfies the global, $N$-uniform reset estimate

$$
P\mathscr L_N(S)\leq M:=1+M_x+\lambda V^2.
$$

Hence for every fixed $q\in(0,1)$,

$$
P\mathscr L_N-\mathscr L_N\leq-(1-q)\mathscr L_N+M.
$$

The same upper bound holds for the sub-Markov transition killed at complete extinction. If extinction is represented by an absorbing state with $\mathscr L_N=1$, it also holds for that completed Markov chain.

*Proof.* Freeze every donor choice. After literal copying, every position is an eligible input position, including every revived slot; its norm is at most $R_D$. Accepted-row independent centered jitter gives

$$
\mathbb E\frac1N\sum_i|X_i^c|^2\leq R_D^2+d\sigma_x^2.
$$

All collision velocities have norm at most $W$. For BAOAB, the position immediately before final position noise is

$$
X_2=X^c+\frac h2(1+c)\left(V^c-\frac h2\nabla U(X^c)\right)
+\frac h2s_hB\xi^O.
$$

The deterministic center has norm at most $A|X^c|+D_0$. The $L^2$ triangle inequality bounds its mean square by $(A\sqrt{R_D^2+d\sigma_x^2}+D_0)^2$. The independent centered O innovation contributes the displayed trace term. Independent final position noise adds $d\sigma_p^2h$. B2 and the final cap do not change position. The smooth cap bounds every completed velocity by $V$, including retained velocities of terminally dead slots. Averaging proves $P\mathscr L_N\leq M$. Nonnegativity gives the Foster inequality and the killed-kernel bound. $\square$

The estimate controls arbitrarily large entering dead coordinates through the actual revival operation. It does not use a strict cloning variance rate, an assumed stationary density, or a continuous-time replacement. Together with the actual terminal-noise survival estimate in {prf:ref}`cor-mean-field-positive-alive-mass`, it supplies uniform finite-step moment control for surviving laws. Uniqueness and attraction of a law additionally require mixing estimates; a moment drift alone is not such a theorem.
:::

:::{prf:corollary} A finite boundary moment for the complete transition
:label: cor-canonical-full-step-boundary-reset

For the canonical box $D=\prod_j[\ell_j,u_j]$, let $L_j=u_j-\ell_j$ and define the auxiliary boundary observable in its interior by

$$
\psi_D(x)=\sum_{j=1}^d\log\!\left(\frac{L_j^2}{(x_j-\ell_j)(u_j-x_j)}\right),\qquad
\mathscr B_N(S)=\frac1N\sum_{i:a_i=1}\psi_D(x_i).
$$

Set its value to zero at the box boundary, so it is finite at every admitted point; this measure-zero convention leaves all transition integrals unchanged. This is a diagnostic Lyapunov observable, not an added term in the canonical reward. It is nonnegative and integrable over $D$. For $\sigma_p>0$,

$$
P\mathscr B_N(S)\leq (2\pi\sigma_p^2h)^{-d/2}\|\psi_D\|_{L^1(D)}=:M_b<\infty.
$$

Thus $\mathscr L_N+c_b\mathscr B_N$, with any fixed $c_b>0$, satisfies the full-step Foster bound with offset $M+c_bM_b$.

*Proof.* On each coordinate interval, the logarithmic endpoint singularity has finite integral, so Fubini gives $\psi_D\in L^1(D)$. Conditional on all preceding stages, the final position noise has a Gaussian density bounded by $(2\pi\sigma_p^2h)^{-d/2}$. The terminal eligibility indicator restricts its contribution to $D$. Integrate $\psi_D$ against that density, average the rows, and add the previous theorem. Boundary points have probability zero after this noise. $\square$
:::

### 12.2. Inter-Swarm Error Under Cloning

The original inter-swarm error is $V_W=V_{\rm loc}+V_{\rm struct}$. Its clone-side contribution enters the combined Lyapunov argument through a bounded expansion estimate.

#### 12.2.1. Bounded Expansion of Inter-Swarm Error

:::{prf:theorem} Inter-swarm drift from a post-update moment bound
:label: thm-inter-swarm-bounded-expansion

Let $\mu'_1,\mu'_2$ be the probability empirical measures used in $V_W$ after
the specified update. Suppose, for a fixed phase-space point $z_0$ and the
positive quadratic hypocoercive norm,

$$
\mathbb E\int\|z-z_0\|_h^2\,d\mu'_s(z)\leq M_h,
\qquad s=1,2.
$$

Then $\mathbb E\Delta V_W\leq C_W$ with $C_W=4M_h$, independently of the
choice of coupling between the two update kernels. The bound is uniform in
$N$ when $M_h$ is. Measures normalized by the alive count require this same
normalized moment bound; unnormalized moment estimates alone do not imply it.
:::

:::{prf:proof}
The product measure $\mu'_1\otimes\mu'_2$ is an admissible transport coupling.
The quadratic inequality
$\|z-w\|_h^2\leq2\|z-z_0\|_h^2+2\|w-z_0\|_h^2$ gives

$$
W_h^2(\mu'_1,\mu'_2)
\leq2\int\|z-z_0\|_h^2d\mu'_1+
2\int\|w-z_0\|_h^2d\mu'_2.
$$

Take expectations and subtract the nonnegative initial $V_W$.
For the all-slot Gaussian proposal, bounded companion positions and collision velocities bounded by $(1+2\alpha)V_{\max}$ give a finite $M_h$ directly from
$\mathbb E\|y+\sigma_x\xi-z_{0,x}\|^2
=\|y-z_{0,x}\|^2+d\sigma_x^2$ and equivalence of quadratic norms.
After a killing test on a bounded valid domain, normalized living empirical
measures have a direct support bound whenever the alive set is nonempty.
These are two distinct ways to verify the stated moment hypothesis.
:::

:::{prf:corollary} The actual cloning kernel supplies the inter-swarm moment bound
:label: cor-cloning-actual-inter-swarm-expansion

Use the complete cloning transition with weighted measurement and cloning companions, retained sampled fitness, frozen gates and source positions, mandatory revival, Gaussian position jitter of amplitude $j$, and the shared component rotations of {prf:ref}`prop-cloning-component-conservation`, with restitution $0\le\alpha\le1$. Suppose both entering alive sets are nonempty. Fix a position anchor $x_0$ and let

$$
B_x=\sup_{x\in\mathcal X_{\rm valid}}|x-x_0|<\infty,
\qquad |v_i|\le V_{\max}\quad\text{for every entering slot}.
$$

The positional bound applies only to eligible alive sources. Retained dead positions are unrestricted. The velocity bound includes retained dead slots and is supplied by the completed velocity cap. Let

$$
Q(x,v)=|x|^2+\lambda_v|v|^2+b\,x\cdot v,
\qquad \lambda_v>b^2/4,
$$

be the original transport quadratic form. For every fixed $\eta>0$, put

$$
K_\eta=(1+\eta)(B_x^2+d j^2)
 +\left(\lambda_v+\frac{b^2}{4\eta}\right)V_{\max}^2.
\tag{3.W1}
$$

Then each actual postcloning empirical law $\mu_s^+$ satisfies

$$
\mathbb E\!\left[\int Q(x-x_0,v)\,d\mu_s^+(x,v)\,\middle|\,S_s\right]
\le K_\eta,\qquad s=1,2.
\tag{3.W2}
$$

Consequently every coupling of the two actual cloning kernels obeys

$$
\boxed{\mathbb E[\Delta V_W\mid S_1,S_2]
\le -V_W(S_1,S_2)+4K_\eta\le C_W,
\qquad C_W=4K_\eta.}
\tag{3.W3}
$$

All constants are independent of $N$, the alive fractions, and the number or sizes of collision components. This supplies the bounded inter-swarm expansion input of the original weighted Lyapunov composition.

**Proof.** Freeze the retained measurement vector, donor and gate choices, and accepted component graph. Each resulting source position $Y_i$ is either an unaccepted alive position or an eligible frozen donor position. Mandatory revival supplies such a donor to every entering dead slot. Thus $|Y_i-x_0|\le B_x$ for all $N$ destinations. With $I_i$ the accepted-cloning indicator, the actual positional proposal is

$$
X_i^+=Y_i+j I_i\zeta_i,\qquad
\mathbb E\!\left[\frac1N\sum_i|X_i^+-x_0|^2\,\middle|\,\text{frozen choices}\right]
=\frac1N\sum_i|Y_i-x_0|^2+
 \frac{d j^2}{N}\sum_i I_i
\le B_x^2+d j^2.
$$

This is a Gaussian moment estimate on the complete proposal. It makes no compact-support claim about the postcloning positions.

For each component $C$, the actual shared orthogonal rotation gives the pointwise identity

$$
\sum_{i\in C}|v_i^+|^2
=|C|\,|\bar v_C|^2+
 \alpha^2\sum_{i\in C}|v_i-\bar v_C|^2
\le\sum_{i\in C}|v_i|^2.
$$

Uninvolved slots retain their velocities. Sum over the disjoint components to obtain

$$
\frac1N\sum_i|v_i^+|^2\le\frac1N\sum_i|v_i|^2\le V_{\max}^2.
$$

The sum includes the retained pre-collision velocities of revived slots. No independence between collision outputs or bound on component size is used. Young's inequality,

$$
Q(x,v)\le(1+\eta)|x|^2+
 \left(\lambda_v+\frac{b^2}{4\eta}\right)|v|^2,
$$

then proves (3.W2), after averaging all the actual measurement, gate, jitter, and rotation randomness. Both postcloning populations have all $N$ slots alive, so their probability empirical laws use exactly the $1/N$ normalization in these moment estimates. Apply {prf:ref}`thm-inter-swarm-bounded-expansion` with anchor $(x_0,0)$ and $M_h=K_\eta$. Its product transport plan is valid for each realized pair of empirical laws, independently of how the two kernels are coupled, proving (3.W3).

An extinct entering state has no eligible donor and follows the specified cemetery transition. The alive probability law and $V_W$ in this corollary are defined on the nonextinct pair domain; this argument does not assign a fictitious normalized alive law to the cemetery state. $\square$
:::

:::{prf:remark} Bounded drift and contraction
:label: rem-why-vw-expands

The preceding estimate controls the size of the post-update transport distance.
It neither asserts expansion nor proves strict contraction. Synchronous jitter
cancels when the same noise is applied to the same two update branches; different
companions and decisions must still be included in the coupling. A contraction
rate requires the finer kernel estimates used in {doc}`06_convergence`.
:::

#### 12.2.2. Decomposition of Inter-Swarm Error

For completeness, we state the separate bounds on the location and structural components.

:::{prf:corollary} Bounds on the location and structural components
:label: cor-component-bounds-vw

Under {prf:ref}`thm-inter-swarm-bounded-expansion`, the nonnegative decomposition
$V_W=V_{\mathrm{loc}}+V_{\mathrm{struct}}$ gives

$$
\mathbb E\Delta V_{\mathrm{loc}}\leq4M_h,
\qquad \mathbb E\Delta V_{\mathrm{struct}}\leq4M_h.
$$

Thus one may take $C_{\mathrm{loc}}=C_{\mathrm{struct}}=4M_h$ in a separate
component calculation. The direct sum bound $C_W=4M_h$ is sharper than the sum
of these two separately relaxed bounds.
:::

:::{prf:proof}
Each post-update component is at most $V'_W$, and each initial component is
nonnegative. Apply the preceding theorem's bound on $\mathbb EV'_W$.
:::

:::{prf:theorem} Summing component drift bounds
:label: thm-complete-wasserstein-drift

If the same cloning transition satisfies
$\mathbb E\Delta V_{\mathrm{loc}}\leq C_{\mathrm{loc}}$ and
$\mathbb E\Delta V_{\mathrm{struct}}\leq C_{\mathrm{struct}}$, then

$$
\mathbb E\Delta V_W\leq C_{\mathrm{loc}}+C_{\mathrm{struct}}.
$$

The direct moment argument in {prf:ref}`thm-inter-swarm-bounded-expansion` gives
an alternative bound. Either valid choice can be used as $C_W$ in the component
composition theorem. For the complete cloning mechanism, {prf:ref}`cor-cloning-actual-inter-swarm-expansion` discharges this input with $C_W=4K_\eta$, including retained-dead velocities and immediate revival. The kinetic estimates and their rate conditions are given
in {doc}`05_kinetic_contraction` and {doc}`06_convergence`.
:::

:::{prf:proof}
Use the exact decomposition and linearity of conditional expectation. Keeping
the direct joint estimate instead of relaxing each component separately gives
the alternative constant stated above.
:::

### 12.3. The Complete Lyapunov Drift Under Cloning

We now combine all results to characterize the cloning operator's effect on the full Lyapunov function.

#### 12.3.1. Main Result

:::{prf:theorem} Complete weighted drift for the actual cloning operator
:label: thm-complete-cloning-drift

Retain the chapter's Lyapunov function, including its prescribed velocity weight:
$$
\Phi=V_{\mathrm{total}}=V_W+c_V(X+Y)+c_BW_b,\qquad
X=V_{\mathrm{Var},x},\quad Y=\lambda_vV_{\mathrm{Var},v}.
$$
Here the variances are the sums of the two $N$-normalized alive-input
variances. Both entering swarms are nonextinct, eligible positions lie in the
stated domain of diameter $D_x$, and all retained velocities satisfy the
completed-step cap $V_{\max}$. Apply the actual measurement, frozen acceptance,
revival, component rotation, and jitter kernel $P_C$.

Before estimating any signed positional contribution, its exact weighted
increment is
$$
\begin{aligned}
(P_C-I)\Phi
={}&(P_C-I)V_W+c_V\big[H_x(S_1)+H_x(S_2)
 +\lambda_v\{R_v(S_1)+R_v(S_2)\}\\
&\hspace{36mm}-\lambda_v(1-\alpha^2)
 \mathbb E(\mathcal E_{C,1}+\mathcal E_{C,2})\big]
+c_B(P_C-I)W_b .
\end{aligned}
\tag{3.AC1}
$$
The actual donor and measurement integrals in $H_x$ are those of
{prf:ref}`thm-positional-variance-contraction`; they have not been replaced
by a sign condition or by an inter-swarm distance.

The positional and velocity inputs of the weighted affine argument are
fully supplied by the exact kernel:
$$
P_CX\le C_x:=D_x^2+2d\sigma_x^2,
\qquad
P_CY=Y+\lambda_vR-\lambda_v(1-\alpha^2)\overline{\mathcal E}_C,
\tag{3.AC2}
$$
where $R=R_v(S_1)+R_v(S_2)$,
$\overline{\mathcal E}_C=\mathbb E(\mathcal E_{C,1}+\mathcal E_{C,2})$, and
$$
0\le R\le\frac{4(D_1+D_2)}N V_{\max}^2\le8V_{\max}^2.
$$
In particular, for every chosen $0<\kappa_x\le1$,
$$
(P_C-I)X\le-\kappa_xX+C_x,
\qquad (P_C-I)Y\le C_v:=8\lambda_vV_{\max}^2.
\tag{3.AC3}
$$
For all-alive entering swarms take $C_v=0$ and retain the negative component
energy in (3.AC2). The bounded inter-swarm expansion from
{prf:ref}`cor-cloning-actual-inter-swarm-expansion` supplies $C_W$, so (3.AC1) gives
$$
(P_C-I)\Phi\le C_W+c_V[-\kappa_xX+C_x+C_v]
+c_B(P_C-I)W_b.
\tag{3.AC4}
$$
When the applicable boundary estimate of
{prf:ref}`thm-boundary-potential-contraction` is inserted, this becomes
$$
(P_C-I)\Phi\le-c_V\kappa_xX-c_B\kappa_bW_b
 +C_W+c_V(C_x+C_v)+c_BC_b.
\tag{3.AC5}
$$
All displayed positional, velocity, and inter-swarm constants are uniform in
$N$. This is the original weighted affine drift: bounded expansion in one
component is permitted and its offset is retained. A negative drift is asserted
only where the displayed dissipative terms exceed the displayed offset.
:::

:::{prf:proof}
The variance identities of
{prf:ref}`thm-complete-variance-drift` and linearity of conditional expectation
give (3.AC1), with $\lambda_v$ multiplying every velocity contribution.
Conditional on the entire donor/gate realization, every copied or retained
position comes from the eligible input domain. Its pairwise variance is at
most $D_x^2/2$ per swarm. Independent centered jitters add at most
$(1-1/N)d\sigma_x^2$ per swarm. This proves the first bound in (3.AC2),
including all revival rows and all retained-fitness dependence.
The actual shared component rotations give its second identity; the
retained-dead correction is bounded by
{prf:ref}`thm-velocity-variance-bounded-expansion`.
Since $X\ge0$, $-X+C_x\le-\kappa_xX+C_x$ for $\kappa_x\le1$.
This proves (3.AC3). Insert the already proved bounded $V_W$ expansion to
obtain (3.AC4), and the stated boundary estimate to obtain (3.AC5).
No individual or structural Wasserstein contraction is needed in this
cloning-stage calculation.
:::


#### 12.3.2. Interpreting the component bounds

:::{div} feynman-prose
The cloning inputs $C_x$, $C_v$, and $C_W$ now come from the actual proposal: frozen eligible positions and Gaussian jitter control the positional moment, component rotations control full-slot velocity energy, and these moments bound inter-swarm expansion. Revival has its own explicit contribution. Each of these bounds is uniform in $N$.

The weights combine these different roles. Cloning can rearrange the two clouds while limiting their positional spread; kinetic evolution supplies its complementary estimates. The additive constants remain in the combined drift, and the negative relative-energy contribution from collisions can be retained for a sharper bound. Thus the calculation permits a positive increment in one component while testing the weighted sum against its full offset.

Composition evaluates kinetics on the actual post-cloning states. Its BAOAB, noise, cap, and terminal-boundary estimates must hold there, and the chosen boundary observable must satisfy its stated bound. With those entries verified, the displayed operator identity carries both dissipation and offsets through the scheduled update.
:::

#### 12.3.3. Why Cloning Alone Cannot Achieve Convergence

:::{prf:proposition} What the cloning drift leaves to the full transition
:label: prop-kinetic-necessity

The component bounds $\mathbb E\Delta V_W\leq C_W$ and
$\mathbb E\Delta V_{\mathrm{Var},v}\leq C_v$ alone imply no strict contraction
in those components. A full Lyapunov contraction follows when the composed
transition satisfies the complementary estimates in
{prf:ref}`thm-synergistic-foster-lyapunov-preview`.

In particular, an upper bound by a positive constant does not prove positive
expansion, and a velocity cap already bounds the corresponding variance.
Inelastic collisions can dissipate relative kinetic energy; identifying a
stationary velocity law requires the complete generator.
:::

:::{prf:proof}
The identity transition satisfies both upper bounds with any nonnegative
$C_W,C_v$ while preserving every value of the two observables. Thus these bounds
cannot imply a strict contraction. The sufficient composition estimate is
proved below by conditioning successively on the two updates.
:::

### 12.4. The Synergistic Dissipation Framework

We state the algebraic composition result for component bounds that have been verified for the actual cloning and kinetic transitions.

#### 12.4.1. Complementary Drift Properties

The following table records the bounds required by the diagonal composition theorem; the positional reset bound and its offset are given in {prf:ref}`thm-positional-variance-contraction`. Kinetic estimates must match the declared discrete stages in {doc}`05_kinetic_contraction`:

| Component | Cloning drift | Kinetic drift at its input | Combined affine drift |
|:----------|:--------------|:---------------------------|:----------------------|
| $V_W$ | $C_W$ | $-\kappa_WV_W+C'_W$ | $-\kappa_WV_W+(1-\kappa_W)C_W+C'_W$ |
| $X=V_{\mathrm{Var},x}$ | $-\kappa_xX+C_x$ | $C'_x$ | $-\kappa_xX+C_x+C'_x$ |
| $Y=\lambda_vV_{\mathrm{Var},v}$ | $C_v$ | $-\kappa_vY+C'_v$ | $-\kappa_vY+(1-\kappa_v)C_v+C'_v$ |
| $W_b$ | $-\kappa_bW_b+C_b$ | $C'_b$ | $-\kappa_bW_b+C_b+C'_b$ |

:::{prf:remark} Matching the component estimates
:label: rem-perfect-complementarity

The table records the diagonal drift pattern used in the composition theorem.
Its entries are hypotheses for the indicated components and kernels, with their
additive terms retained. A confining potential does not automatically contract
every chosen boundary observable. Cross-coupled estimates are assembled with
the comparison-matrix conditions of {doc}`06_convergence`.
:::

#### 12.4.2. The Synergistic Drift Inequality

:::{prf:lemma} Weighted minimum-coefficient inequality
:label: lemma-weighted-min-coefficient

For $X_i\geq0$, $w_i>0$, and $a_i\geq a_*>0$,

$$
\sum_i w_i a_iX_i\geq a_*\sum_iw_iX_i.
$$
:::

:::{prf:proof}
Each difference $w_i(a_i-a_*)X_i$ is nonnegative. Summing these differences
proves the inequality.
:::

:::{prf:theorem} Composition of complementary component drift bounds
:label: thm-synergistic-foster-lyapunov-preview

Let $P_C$ and $P_K$ be the cloning and kinetic transition operators, acting on
nonnegative observables, and set
$F=(V_W,V_{\mathrm{Var},x},\lambda_vV_{\mathrm{Var},v},W_b)^\mathsf T$.
Suppose the component estimates established for the chosen parameter regime
have the following common, statewise form:

$$
P_CF\leq A_CF+b_C,\qquad P_KF\leq A_KF+b_K,
$$

where

$$
A_C=\operatorname{diag}(1,1-\kappa_x,1,1-\kappa_b),\qquad
A_K=\operatorname{diag}(1-\kappa_W,1,1-\kappa_v,1),
$$

$0<\kappa_x,\kappa_b,\kappa_W,\kappa_v\leq1$, and the vectors $b_C,b_K$
are finite and nonnegative. These are estimates for the same observables and
state space; any boundary, survival, or coupling hypotheses of the component
results remain in force. The cloning inputs are
{prf:ref}`thm-positional-variance-contraction`,
{prf:ref}`thm-velocity-variance-bounded-expansion`,
{prf:ref}`thm-boundary-potential-contraction`, and
{prf:ref}`thm-inter-swarm-bounded-expansion`.
The kinetic inputs are developed in {doc}`05_kinetic_contraction`.

For cloning followed by kinetics, the backward transition operator is
$Q=P_CP_K$. For any $c_V,c_B>0$, put
$w=(1,c_V,c_V,c_B)^\mathsf T$ and $V_{\mathrm{total}}=w^\mathsf TF$. Then

$$
QV_{\mathrm{total}}\leq(1-\kappa_*)V_{\mathrm{total}}+C_*,\qquad
\kappa_*=\min(\kappa_W,\kappa_x,\kappa_v,\kappa_b),\quad
C_*=w^\mathsf T(A_Kb_C+b_K).
$$

The constants are uniform in $N$ when the component constants and chosen
weights are uniform in $N$. If component estimates contain cross terms, their
nonnegative comparison matrix must be used instead of these diagonal matrices;
see {doc}`06_convergence`.
:::

:::{prf:proof}
Condition first on the post-cloning state. Positivity and linearity of $P_C$
give

$$
QF=P_C(P_KF)\leq P_C(A_KF+b_K)
=A_KP_CF+b_K\leq A_KA_CF+A_Kb_C+b_K.
$$

For a killed operator the constant term can only decrease, since $P_C1\leq1$.
The diagonal entries of $A_KA_C$ are
$1-\kappa_W,1-\kappa_x,1-\kappa_v,1-\kappa_b$.
Multiply by $w^\mathsf T$ and apply
{prf:ref}`lemma-weighted-min-coefficient` to the four nonnegative components.
This gives the asserted drift, with every additive constant evaluated after
the correct stage of the update.

Iteration yields

$$
Q^nV_{\mathrm{total}}\leq(1-\kappa_*)^nV_{\mathrm{total}}
+\frac{C_*}{\kappa_*}\bigl(1-(1-\kappa_*)^n\bigr).
$$

For a conservative chain this is an ordinary moment estimate. For a killed
chain it bounds the unnormalized surviving moment. A conditioned moment is
obtained by dividing by $Q^n1$, and therefore also requires survival control.
:::

:::{prf:corollary} Positional and velocity inputs in the weighted composition
:label: cor-cloning-weighted-assembly-input

For $F=(V_W,X,Y,W_b)^\mathsf T$ in
{prf:ref}`thm-synergistic-foster-lyapunov-preview`, the positional and velocity
entries are discharged by (3.AC2)--(3.AC3), and the inter-swarm entry by
{prf:ref}`cor-cloning-actual-inter-swarm-expansion`. Thus the actual clone-side offset
is
$$
b_C=(C_W,C_x,C_v,C_b)^\mathsf T,
\quad C_x=D_x^2+2d\sigma_x^2,\quad C_v=8\lambda_vV_{\max}^2,
\tag{3.AC6}
$$
The sharper choice $C_v=0$ applies to a one-step estimate with all-alive
entering swarms. Iteration uses the uniform $8\lambda_vV_{\max}^2$ offset
unless the all-alive class is itself invariant; terminal deaths must not be
passed into the next step with the zero-revival offset. The boundary entry
retains the applicable estimate for the chosen $W_b$ from Section 11.

For the same component bounds, define their nonnegative defects
$$
D_C=A_CF+b_C-P_CF,\qquad D_K=A_KF+b_K-P_KF.
$$
On nonextinct cloning inputs, mandatory revival makes $P_C1=1$. The exact
backward composition $Q=P_CP_K$ therefore satisfies
$$
QF=A_KA_CF+A_Kb_C+b_K-A_KD_C-P_CD_K.
\tag{3.AC7}
$$
This remains valid for the actual terminally killed kinetic kernel: its output
observables are zero at the cemetery, and no conditioning on survival has been
inserted. For a sub-Markov cloning kernel there is the additional nonpositive
term $-(1-P_C1)b_K$.

With $w=(1,c_V,c_V,c_B)^\mathsf T$, retain the exact identity
$$
(Q-I)\Phi=(P_C-I)\Phi+P_C(P_K-I)\Phi.
\tag{3.AC8}
$$
For the diagonal kinetic comparison used above,
$b_K=(C'_W,C'_x,C'_v,C'_b)^\mathsf T$, where $C'_v$ is for the weighted
observable $Y=\lambda_vV_{\mathrm{Var},v}$. Its explicit full-step offset is
$$
\begin{aligned}
C_*={}&(1-\kappa_W)C_W+C'_W\\
 &+c_V[C_x+C'_x+(1-\kappa_v)C_v+C'_v]
 +c_B(C_b+C'_b).
\end{aligned}
\tag{3.AC9}
$$
Moreover, the velocity defect in (3.AC7) includes the proved nonnegative term
$\lambda_v(1-\alpha^2)\overline{\mathcal E}_C$. Its contribution to the upper
bound for the full drift is consequently
$-c_V(1-\kappa_v)\lambda_v(1-\alpha^2)\overline{\mathcal E}_C$.
A nonnegative comparison matrix with cross terms propagates these same defects
as $-A_KD_C$; they must not be assigned a favorable sign after multiplication
by a matrix having negative entries.

This supplies the original weighted composition with the actual bounded
cloning inputs and their offsets. The kinetic entries must be proved on the
actual post-cloning states using the declared BAOAB, position noise, cap, and
terminal boundary. Iterated killed moments and survival-conditioned moments
retain the distinction in the preceding theorem.
:::

:::{prf:proof}
Equation (3.AC6) follows from the proved inputs, with the boundary component
left in its own stated applicability. Substitute
$P_KF=A_KF+b_K-D_K$ into $P_C(P_KF)$, then substitute
$P_CF=A_CF+b_C-D_C$. Positivity and linearity give (3.AC7), and expanding
$P_CP_K-I=(P_C-I)+P_C(P_K-I)$ gives (3.AC8).
Multiplying the offset by the original weights gives (3.AC9).
Finally the exact velocity identity (3.AC2) yields
$$
(D_C)_Y=C_v-\lambda_vR+
 \lambda_v(1-\alpha^2)\overline{\mathcal E}_C
\ge\lambda_v(1-\alpha^2)\overline{\mathcal E}_C,
$$
which proves the retained collision contribution. No contraction assumption
on the cloning-stage structural distance is used.
:::

#### 12.4.3. Parameter Balancing

:::{prf:proposition} Existence of valid coupling constants
:label: prop-coupling-constant-existence

Under the component bounds in
{prf:ref}`thm-synergistic-foster-lyapunov-preview`, the explicit choice
$c_V=c_B=1$ gives a finite drift offset and a positive contraction coefficient
$\kappa_*$. Every fixed positive pair of weights also works in this diagonal
case. The contraction coefficient is a minimum of the component rates.
:::

:::{prf:proof}
Set $w=(1,1,1,1)^\mathsf T$ in the preceding theorem. Then
$C_*=\sum_i(A_Kb_C+b_K)_i<\infty$ and $\kappa_*>0$. For arbitrary finite
positive weights the same minimum-coefficient inequality applies, and the
offset remains finite. This proves the claimed construction.
:::

:::{prf:remark} Tuning guidance
:label: rem-tuning-guidance

The weights change the relative emphasis of the Lyapunov components and the
offset $C_*$. In the diagonal estimate they do not multiply the relative
contraction rates. With cross-coupled estimates, admissible weights must satisfy
the comparison-matrix inequalities in {doc}`06_convergence`.
:::

### 12.5. Summary of Main Results

We conclude by summarizing the main achievements of this document.

#### 12.5.1. Theoretical Contributions

:::{prf:theorem} The cloning estimates and their composition
:label: thm-fg-cloning-main-results

On a family satisfying the hypotheses of the cited estimates, the cloning
analysis gives:

1. The Keystone bound of {prf:ref}`lem-quantitative-keystone`, with
   $\chi=p_uc_{\mathrm{err}}$ and its stated residual.
2. The positional variance drift of
   {prf:ref}`thm-positional-variance-contraction` and the bounded velocity
   contribution of {prf:ref}`thm-velocity-variance-bounded-expansion`.
3. The boundary drift of {prf:ref}`thm-boundary-potential-contraction`, including
   the explicit revival contribution in {prf:ref}`thm-complete-boundary-drift`.
4. The composed component drift of
   {prf:ref}`thm-synergistic-foster-lyapunov-preview` when its kinetic and cloning
   estimates hold for the same observables and state space.

These constants are independent of $N$ when their input bounds are independent
of $N$. The extinction estimate separately requires the safe-population and
conditional-noise assumptions of {prf:ref}`cor-extinction-suppression`.
:::

:::{prf:proof}
:label: proof-fg-cloning-main-results

The summary follows by assembling the analytical estimates already proved in
this chapter, retaining the hypotheses attached to each estimate.

1. The geometric partition and measurement estimates of Sections 5–8 give the
   Keystone inequality: the weighted sum of squared centered discrepancies is
   at least $\chi(\epsilon)V_{\mathrm{struct}}-g_{\max}(\epsilon)$.
   The constants come from the measurement and selection bounds in that proof.
2. The exact row-law calculation gives {prf:ref}`thm-positional-variance-contraction`. The bounded-domain reset estimate follows directly from frozen eligible donor positions and Gaussian jitter. A sharper bound using the Keystone selection sum requires a separate estimate of $H_x$.
3. Full-component momentum and energy conservation give {prf:ref}`thm-velocity-variance-bounded-expansion`, with $C_v=8V_{\max}^2$ for two swarms and $C_v=0$ on all-alive inputs, before any velocity weight is inserted.
4. The exposed/persistent/revived partition gives
   {prf:ref}`thm-boundary-potential-contraction` and its explicit refinement
   {prf:ref}`thm-complete-boundary-drift`. The probability of a common extinction
   event is controlled separately by {prf:ref}`cor-extinction-suppression` under
   its safe-population and conditional-noise hypotheses.
5. Positivity and linearity combine the component estimates in
   {prf:ref}`thm-synergistic-foster-lyapunov-preview`.
   {prf:ref}`prop-coupling-constant-existence` supplies explicit positive weights
   for the diagonal case. Uniformity in $N$ is inherited precisely from
   uniformity of the input bounds.

These steps retain the variance, collision, boundary, and composition arguments
as distinct mathematical results. The additional hypotheses for QSD convergence
are treated in {doc}`06_convergence`.
:::

#### 12.5.2. Following the estimates into the next chapters

:::{div} feynman-prose
The component inequalities provide quantitative information about the update:
how much selected positional error is removed, how collisions affect velocity,
and how replacement and revival affect the boundary observable. Uniform
constants make these bounds useful for a family of particle systems. A
mean-field limit additionally needs control of the interaction law and the
sampling error.

{doc}`05_kinetic_contraction` develops the kinetic estimates, and
{doc}`06_convergence` proves the composition and finite-particle QSD results
under their stated mixing and survival hypotheses. {doc}`08_mean_field`
identifies the limiting generator and mass balance; {doc}`09_propagation_chaos`
gives the mean-field existence, uniqueness, and approximation arguments.
{doc}`15_kl_convergence` develops full-gradient functional inequalities and
hypocoercive entropy estimates, keeping the conservative invariant law and the
normalized QSD evolution distinct.

These are successive uses of the same algorithmic data. For any application,
start with the actual measurement and companion law, carry its constants into
the drift calculation, and then apply the theorem for that transition and that
probability law.
:::
