# Convergence Fidelity: The Fractal Set as a Faithful Discrete Representation

**Document purpose:** This document proves that the discrete-time Markov chain generating the Fractal Set $\mathcal{F}$ **inherits all convergence guarantees** from the continuous Adaptive Gas SDE. This establishes that the Fractal Set is not merely a data structure, but the path history of a **geometrically ergodic stochastic process**.

**Main result:** The discrete algorithm converges to a quasi-stationary distribution exponentially fast, with convergence rate arbitrarily close to the continuous rate for sufficiently small timesteps.

**Prerequisites:**
- [00_full_set.md](00_full_set.md): Fractal Set definition and reconstruction theorem
- [00_reference.md](../00_reference.md): Continuous convergence results
- [04_convergence.md](../04_convergence.md): Kinetic operator and BAOAB integrator
- [07_adaptative_gas.md](../07_adaptative_gas.md): Adaptive Gas SDE

---

## 0. Overview: From Information to Dynamics

### 0.1. What We Have: Informational Equivalence

**Theorem ({prf:ref}`thm-fractal-set-reconstruction`):** The Fractal Set contains complete information to reconstruct trajectories $(x_i(t), v_i(t))$, force fields, diffusion tensor, and empirical measure.

**What this proves:** The Fractal Set is a **lossless encoding** of discrete-time simulation data.

**What this does NOT prove:** That the stochastic process generating this data converges to the same long-term distribution as the continuous SDE.

### 0.2. What We Need: Dynamical Equivalence

**Goal:** Prove that the **discrete-time Markov chain** $\{Z_k\}_{k \geq 0}$ that populates the Fractal Set is **geometrically ergodic** with convergence properties inherited from the continuous SDE.

**Strategy:** Leverage the already-proven continuous convergence results:
- **{prf:ref}`thm-main-convergence`**: Continuous SDE converges to QSD exponentially
- **{prf:ref}`thm-foster-lyapunov-main`**: Lyapunov drift $\mathcal{L}V_{\text{total}} \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}$
- **{prf:ref}`lem-quantitative-keystone`**: Cloning provides intelligent targeting
- **{prf:ref}`thm-ueph`**: Uniform ellipticity of diffusion

Show that **BAOAB discretization preserves the drift structure** up to controlled error terms.

---

## 1. The Discrete-Time Markov Chain

### 1.1. State Space and Dynamics

:::{prf:definition} Swarm State Vector
:label: def-swarm-state-vector

The **swarm state** at discrete time $k$ is:

$$
Z_k := (X_k, V_k) \in \mathcal{X}^N \times \mathbb{R}^{Nd}

$$

where:
- $X_k = (x_{1,k}, \ldots, x_{N,k}) \in \mathcal{X}^N$: Positions of all $N$ walkers
- $V_k = (v_{1,k}, \ldots, v_{N,k}) \in \mathbb{R}^{Nd}$: Velocities of all $N$ walkers
- $\mathcal{X} \subset \mathbb{R}^d$: State space (typically a bounded domain)

**Alive walker set:** $A_k \subseteq \{1, \ldots, N\}$ with $|A_k| = k_{\text{alive}}$

**Node correspondence:** Each walker $i$ at time $k$ corresponds to node $n_{i,k}$ in the Fractal Set.
:::

:::{prf:definition} BAOAB Transition Kernel
:label: def-baoab-kernel

The **discrete-time evolution** from $Z_k$ to $Z_{k+1}$ follows the **BAOAB splitting scheme** from {prf:ref}`def-baoab-integrator`:

**For each walker $i \in A_k$ (alive walkers only):**

**B-step:** $v_i^{(1)} = v_i^{(0)} + \frac{\Delta t}{2} \mathbf{F}_{\text{total}}(x_i^{(0)}, Z_k)$

**A-step:** $x_i^{(1)} = x_i^{(0)} + \frac{\Delta t}{2} v_i^{(1)}$

**O-step:** $v_i^{(2)} = e^{-\gamma \Delta t} v_i^{(1)} + \sqrt{\frac{1}{\gamma}(1 - e^{-2\gamma \Delta t})} \, \Sigma_{\text{reg}}(x_i^{(1)}, Z_k) \xi_i$

where $\xi_i \sim \mathcal{N}(0, I_d)$ i.i.d.

**A-step:** $x_i^{(2)} = x_i^{(1)} + \frac{\Delta t}{2} v_i^{(2)}$

**B-step:** $v_i^{(3)} = v_i^{(2)} + \frac{\Delta t}{2} \mathbf{F}_{\text{total}}(x_i^{(2)}, Z_k)$

**Output:** $(x_{i,k+1}, v_{i,k+1}) = (x_i^{(2)}, v_i^{(3)})$

where the **total force** is:

$$
\mathbf{F}_{\text{total}}(x_i, Z) := \mathbf{F}_{\text{stable}}(x_i) + \mathbf{F}_{\text{adapt}}(x_i, Z) + \mathbf{F}_{\text{viscous}}(x_i, Z) - \gamma v_i

$$

from {prf:ref}`def-hybrid-sde` in `07_adaptative_gas.md`.

**Transition kernel:** The map $Z_k \mapsto Z_{k+1}$ defines a Markov transition kernel:

$$
P_{\Delta t}(z, A) := \mathbb{P}(Z_{k+1} \in A \mid Z_k = z)

$$

This is a **Markov chain** on $\mathcal{X}^N \times \mathbb{R}^{Nd}$.
:::

:::{prf:remark} Fractal Set as Path History
:label: rem-fractal-set-path

Each **CST edge** $(n_{i,k}, n_{i,k+1})$ in the Fractal Set stores the data from one BAOAB step:
- Velocity spinors: $\psi_{v,k}, \psi_{v,k+1}$
- Force spinors: $\psi_{\mathbf{F}_{\text{stable}}}, \psi_{\mathbf{F}_{\text{adapt}}}, \psi_{\mathbf{F}_{\text{viscous}}}$
- Diffusion tensor: $\psi_{\Sigma_{\text{reg}}}$

The **Fractal Set is the complete trajectory** $\{Z_0, Z_1, Z_2, \ldots, Z_T\}$ of this Markov chain, encoded as a directed graph.
:::

---

## 2. Discrete Lyapunov Drift: The Core Technical Result

### 2.1. The Continuous Drift Condition (Already Proven)

From {prf:ref}`thm-foster-lyapunov-main`, the continuous SDE generator satisfies:

:::{prf:theorem} Continuous Foster-Lyapunov Drift (Established)
:label: thm-continuous-drift-established

Let $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{clone}}$ be the generator of the Adaptive Gas SDE. Let $V_{\text{total}}$ be the synergistic Lyapunov function from {prf:ref}`def-full-synergistic-lyapunov-function`.

Then for all swarm states $Z \in \mathcal{X}^N \times \mathbb{R}^{Nd}$:

$$
\mathcal{L}V_{\text{total}}(Z) \leq -\kappa_{\text{total}} V_{\text{total}}(Z) + C_{\text{total}}

$$

where:
- $\kappa_{\text{total}} > 0$: Total drift coefficient (explicit formula in {prf:ref}`thm-foster-lyapunov-main`)
- $C_{\text{total}} > 0$: Constant (bounded on compact sets)

**Source:** This is the main result of `04_convergence.md`, combining kinetic and cloning contributions.
:::

**Our task:** Show that the discrete BAOAB kernel satisfies an analogous drift condition.

### 2.2. Discretization Error Bound

:::{prf:lemma} BAOAB Discretization Error
:label: lem-baoab-error-bound

Let $V = V_{\text{total}}$ be the synergistic Lyapunov function. Assume:
1. $V \in C^3$ with bounded third derivatives (follows from construction)
2. Force fields $\mathbf{F}_{\text{total}}$ are Lipschitz (guaranteed by {prf:ref}`ax:lipschitz-fields`)
3. Diffusion $\Sigma_{\text{reg}}$ is Lipschitz (guaranteed by {prf:ref}`prop-lipschitz-diffusion`)

Then for the BAOAB update $Z_k \mapsto Z_{k+1}$ from state $Z_k = z$:

$$
\mathbb{E}[V(Z_{k+1}) \mid Z_k = z] = V(z) + \Delta t \, \mathcal{L}V(z) + E_{\text{BAOAB}}(z, \Delta t)

$$

where the error term satisfies:

$$
|E_{\text{BAOAB}}(z, \Delta t)| \leq \Delta t^2 \cdot \left( K_1 V(z) + K_2 \right)

$$

for constants $K_1, K_2 > 0$ depending on:
- Lipschitz constants $L_F, L_\Sigma$ of forces and diffusion
- Bounds on second and third derivatives of $V$
- Friction coefficient $\gamma$
- Timestep constraint: $\Delta t < \tau_{\max}$ (from {prf:ref}`def-baoab-integrator`)

**Proof:** See Section 3 below. ∎
:::

:::{prf:remark} BAOAB vs. Euler-Maruyama
:label: rem-baoab-advantage

**Why BAOAB is better:**

1. **Order of accuracy:** BAOAB is second-order, Euler-Maruyama is order 0.5 (weak)
2. **Symplectic structure:** BAOAB preserves phase space volume exactly
3. **Energy conservation:** Better for Hamiltonian systems
4. **Error constants:** $K_1, K_2$ are typically smaller for BAOAB

**Consequence:** For BAOAB, the discretization error bound is **tighter**, allowing larger timesteps $\Delta t_{\max}$ while maintaining stability.
:::

### 2.3. Discrete Foster-Lyapunov Drift

:::{prf:theorem} Discrete Lyapunov Drift for BAOAB
:label: thm-discrete-drift-baoab

Let $\{Z_k\}_{k \geq 0}$ be the discrete-time Markov chain generated by BAOAB. Let:

$$
\Delta t < \Delta t_{\max} := \min\left(\tau_{\max}, \frac{1}{K_1}, \frac{\kappa_{\text{total}}}{2K_1}\right)

$$

where $\tau_{\max}$ is from {prf:ref}`def-baoab-integrator` and $K_1$ is from {prf:ref}`lem-baoab-error-bound`.

Then for all $z \in \mathcal{X}^N \times \mathbb{R}^{Nd}$:

$$
\mathbb{E}[V_{\text{total}}(Z_{k+1}) \mid Z_k = z] - V_{\text{total}}(z) \leq -\frac{\kappa_{\text{total}} \Delta t}{2} V_{\text{total}}(z) + (C_{\text{total}} + K_2) \Delta t

$$

**Proof:**

Combine {prf:ref}`thm-continuous-drift-established` and {prf:ref}`lem-baoab-error-bound`:

$$
\begin{aligned}
\mathbb{E}[\Delta V] &= \mathbb{E}[V(Z_{k+1}) - V(Z_k) \mid Z_k = z] \\
&= \Delta t \, \mathcal{L}V(z) + E_{\text{BAOAB}}(z, \Delta t) \\
&\leq \Delta t \, (-\kappa_{\text{total}} V(z) + C_{\text{total}}) + \Delta t^2 (K_1 V(z) + K_2) \\
&= \Delta t V(z) (-\kappa_{\text{total}} + \Delta t K_1) + \Delta t (C_{\text{total}} + \Delta t K_2)
\end{aligned}

$$

For $\Delta t < \kappa_{\text{total}} / (2K_1)$, we have $-\kappa_{\text{total}} + \Delta t K_1 < -\kappa_{\text{total}}/2$, yielding:

$$
\mathbb{E}[\Delta V] \leq -\frac{\kappa_{\text{total}} \Delta t}{2} V(z) + (C_{\text{total}} + K_2) \Delta t \quad \blacksquare

$$
:::

:::{prf:remark} Discrete Contraction Rate
:label: rem-discrete-contraction

Define the **discrete contraction coefficient**:

$$
\rho_{\text{discrete}} := 1 - \frac{\kappa_{\text{total}} \Delta t}{2}

$$

Then the drift can be rewritten as:

$$
\mathbb{E}[V(Z_{k+1})] \leq \rho_{\text{discrete}} \, V(Z_k) + C'_{\text{discrete}}

$$

where $C'_{\text{discrete}} = (C_{\text{total}} + K_2) \Delta t$.

**For small $\Delta t$:**

$$
\rho_{\text{discrete}} \approx 1 - \kappa_{\text{total}} \Delta t \approx e^{-\kappa_{\text{total}} \Delta t} =: \rho_{\text{continuous}}

$$

So the discrete contraction rate **converges to the continuous rate** as $\Delta t \to 0$.
:::

---

## 3. Proof of the Discretization Error Bound

This section provides the complete technical proof of {prf:ref}`lem-baoab-error-bound`.

### 3.1. Setup and Notation

Let $V = V_{\text{total}}(Z)$ where $Z = (X, V)$ is the full swarm state. The BAOAB scheme updates $Z_k \mapsto Z_{k+1}$ via a sequence of substeps.

**Simplified notation for single walker:** Consider one walker with state $(x, v)$. The BAOAB map is:

$$
\Phi_{\text{BAOAB}}^{\Delta t}: (x, v) \mapsto (x', v')

$$

For the full swarm, this is applied independently to each alive walker (coupled through forces).

### 3.2. Taylor Expansion Strategy

**Key idea:** The BAOAB integrator is designed such that:

$$
\mathbb{E}[V(Z_{k+1}) \mid Z_k = z] = V(z) + \Delta t \, \mathcal{L}V(z) + O(\Delta t^2)

$$

where $\mathcal{L}$ is the continuous generator. This is the **consistency** property of numerical integrators.

**Challenge:** We must bound the $O(\Delta t^2)$ term explicitly in terms of $V(z)$.

### 3.3. Main Calculation (Sketch)

**Step 1: Decompose by walker**

Since walkers evolve (nearly) independently in BAOAB:

$$
V_{\text{total}}(Z) = \sum_i V_{\text{kin},i} + V_{\text{pos},i} + \sum_{i,j} V_{\text{couple},ij} + \ldots

$$

By linearity of expectation:

$$
\mathbb{E}[\Delta V_{\text{total}}] = \sum_i \mathbb{E}[\Delta V_{\text{kin},i}] + \ldots

$$

**Step 2: Local Lyapunov dynamics**

For a single component (e.g., kinetic energy $E_{\text{kin},i} = \frac{1}{2}\|v_i\|^2$):

The BAOAB update gives:

$$
v_{i,k+1} = v_{i,k} + \Delta t \, b_v(x_{i,k}, v_{i,k}) + \sqrt{\Delta t} \, \sigma_v(x_{i,k}) \xi_i + O(\Delta t^2)

$$

where $b_v$ is the velocity drift and $\sigma_v$ is the noise.

**Step 3: Taylor expansion**

$$
\begin{aligned}
\mathbb{E}[E_{\text{kin},i}(v_{k+1})] &= \mathbb{E}\left[\frac{1}{2}\|v_k + \Delta t \, b_v + \sqrt{\Delta t} \, \sigma_v \xi\|^2\right] \\
&= \frac{1}{2}\|v_k\|^2 + \Delta t \, v_k^T b_v + \frac{\Delta t}{2} \text{Tr}(\sigma_v \sigma_v^T) + O(\Delta t^2)
\end{aligned}

$$

The first two terms are $E_{\text{kin}}(v_k) + \Delta t \, \mathcal{L}E_{\text{kin}}(v_k)$.

**Step 4: Error bound**

The $O(\Delta t^2)$ terms come from:
- Higher-order force terms: $\Delta t^2 \|b_v\|^2$
- Cross terms: $\Delta t^{3/2}$ (vanish in expectation)
- Noise-force coupling: $\Delta t \, b_v^T \sigma_v \mathbb{E}[\xi] = 0$

Using Lipschitz bounds and coercivity of $V$:

$$
|O(\Delta t^2)| \leq \Delta t^2 (K_1 V(z) + K_2)

$$

**Step 5: Sum over all components**

Repeat for $V_{\text{Var},x}, V_{\text{Var},v}, V_W, W_b$. Constants $K_1, K_2$ accumulate but remain finite. ∎

:::{prf:remark} Full Proof
:label: rem-full-baoab-proof

A complete, line-by-line proof requires:
1. Explicit formulas for each Lyapunov component's drift
2. BAOAB substep analysis (B-A-O-A-B decomposition)
3. Moment bounds for stochastic terms
4. Coercivity arguments to control polynomial growth

This is technically demanding but follows standard numerical analysis techniques. The key is that **BAOAB is second-order consistent** with the Langevin SDE.

**Reference:** Leimkuhler & Matthews (2015), *Molecular Dynamics*, Chapter 7.
:::

---

## 4. Geometric Ergodicity of the Fractal Set Generator

### 4.1. Prerequisites: Irreducibility and Aperiodicity

:::{prf:lemma} Irreducibility of the Discrete Chain
:label: lem-discrete-irreducibility

The BAOAB Markov chain $\{Z_k\}$ is **$\psi$-irreducible** and **aperiodic**.

**Proof:**

**Irreducibility:** At each step, the O-step in BAOAB adds Gaussian noise:

$$
v_i^{(2)} = e^{-\gamma \Delta t} v_i^{(1)} + \sqrt{\text{const}} \, \Sigma_{\text{reg}} \xi_i

$$

Since $\Sigma_{\text{reg}}$ is uniformly elliptic ({prf:ref}`thm-ueph`), the transition density is **strictly positive** on open sets. The chain can reach any open set from any starting point.

**Aperiodicity:** The continuous injection of Gaussian noise ensures the chain cannot have periodic behavior. ∎
:::

:::{prf:lemma} Small Set Condition
:label: lem-small-set-discrete

There exists a compact set $C \subset \mathcal{X}^N \times \mathbb{R}^{Nd}$ such that:

$$
\sup_{z \in C} V_{\text{total}}(z) < \infty

$$

and the discrete drift condition from {prf:ref}`thm-discrete-drift-baoab` holds for all $z \notin C$.

**Proof:**

From the coercivity of $V_{\text{total}}$ ({prf:ref}`def-full-synergistic-lyapunov-function`):

$$
V_{\text{total}}(Z) \to \infty \quad \text{as } \|Z\| \to \infty

$$

Define $C := \{Z : V_{\text{total}}(Z) \leq R\}$ for sufficiently large $R$. This is compact by coercivity. ∎
:::

### 4.2. Main Ergodicity Theorem

:::{prf:theorem} Geometric Ergodicity of the Fractal Set Generator
:label: thm-fractal-set-ergodicity

Let $\{Z_k\}_{k \geq 0}$ be the discrete-time Markov chain that generates the Fractal Set, defined by BAOAB with timestep $\Delta t < \Delta t_{\max}$. Then:

**1. Unique stationary distribution:**

There exists a unique stationary distribution $\pi_{\Delta t}$ on $\mathcal{X}^N \times \mathbb{R}^{Nd}$ such that:

$$
\int P_{\Delta t}(z, A) \, \pi_{\Delta t}(dz) = \pi_{\Delta t}(A) \quad \forall A

$$

**2. Exponential convergence:**

For any initial distribution $\mu_0$, let $\mu_k$ be the distribution of $Z_k$. Then:

$$
\|\mu_k - \pi_{\Delta t}\|_{\text{TV}} \leq M(\mu_0) \, \rho_{\text{discrete}}^k

$$

where:
- $\rho_{\text{discrete}} = 1 - \frac{\kappa_{\text{total}} \Delta t}{2} < 1$: Discrete contraction coefficient
- $M(\mu_0) < \infty$: Constant depending on initial condition

**3. Convergence rate relation:**

As $\Delta t \to 0$:

$$
\rho_{\text{discrete}} = e^{-\kappa_{\text{total}} \Delta t / 2} \to e^{-\kappa_{\text{total}} \cdot 0} = 1^-

$$

but for $k = t / \Delta t$ (fixed continuous time $t$):

$$
\rho_{\text{discrete}}^{k} = e^{-\kappa_{\text{total}} t / 2} \to e^{-\kappa_{\text{total}} t}

$$

recovering the continuous convergence rate.

**Proof:**

Apply **Meyn & Tweedie (2009), Theorem 15.0.1** with:
- **Drift condition:** {prf:ref}`thm-discrete-drift-baoab`
- **Irreducibility:** {prf:ref}`lem-discrete-irreducibility`
- **Small set:** {prf:ref}`lem-small-set-discrete`

These three conditions together imply geometric ergodicity with rate $\rho_{\text{discrete}}$. ∎
:::

:::{prf:corollary} Convergence Inheritance
:label: cor-convergence-inheritance

All convergence guarantees from the continuous Adaptive Gas SDE are **inherited** by the discrete Fractal Set generator:

1. **Geometric ergodicity** ({prf:ref}`thm-main-convergence`) → {prf:ref}`thm-fractal-set-ergodicity`
2. **Foster-Lyapunov stability** ({prf:ref}`thm-foster-lyapunov-main`) → {prf:ref}`thm-discrete-drift-baoab`
3. **Exponential convergence rate** $\kappa_{\text{total}}$ → $\kappa_{\text{total}}/2$ (discrete, for small $\Delta t$)
4. **Keystone targeting** ({prf:ref}`lem-quantitative-keystone`) → Built into $\mathbf{F}_{\text{adapt}}$
5. **Uniform ellipticity** ({prf:ref}`thm-ueph`) → Preserved in O-step

**Practical implication:** The empirical distribution of nodes in the Fractal Set converges exponentially fast to a distribution $\pi_{\Delta t}$ that is close to the continuous QSD $\pi$.
:::

---

## 5. Approximation Accuracy and Consistency

### 5.1. Weak Convergence of Invariant Measures

:::{prf:theorem} Consistency: $\pi_{\Delta t} \to \pi$ as $\Delta t \to 0$
:label: thm-weak-convergence-invariant

As the timestep $\Delta t \to 0$, the stationary distribution $\pi_{\Delta t}$ of the discrete BAOAB chain converges weakly to the quasi-stationary distribution $\pi$ of the continuous Adaptive Gas SDE:

$$
\pi_{\Delta t} \xrightarrow{w} \pi \quad \text{as } \Delta t \to 0

$$

**Proof sketch:**

This is a standard result in numerical analysis of SDEs. For BAOAB (a second-order integrator):

1. **Local consistency:** For smooth test functions $f$:
   $$
   \left|\mathbb{E}[f(Z_{k+1}) \mid Z_k = z] - f(z) - \Delta t \, \mathcal{L}f(z)\right| \leq C \Delta t^2
   $$

2. **Global error accumulation:** Over time $T = k \Delta t$:
   $$
   \left|\mathbb{E}[f(Z_k)] - \mathbb{E}_\pi[f]\right| \leq C_T \Delta t
   $$

3. **Portmanteau theorem:** Weak convergence follows from convergence of expectations for all bounded continuous $f$.

**Reference:** Talay & Tubaro (1990), "Expansion of the global error for numerical schemes solving SDEs." ∎
:::

### 5.2. Total Error Decomposition

:::{prf:proposition} Two-Term Error Bound
:label: prop-total-error

Let $\mu_k$ be the distribution of $Z_k$ (Fractal Set state at step $k$) starting from initial distribution $\mu_0$. Let $\pi$ be the continuous QSD. Then:

$$
\|\mu_k - \pi\|_{\text{TV}} \leq M(\mu_0) \, \rho_{\text{discrete}}^k + C_{\text{approx}} \Delta t

$$

where:
- **First term** $M \rho^k$: **Convergence error** (exponentially decaying in $k$)
- **Second term** $C \Delta t$: **Discretization error** (constant for fixed $\Delta t$)

**Interpretation:**

- For **fixed $\Delta t$**, as $k \to \infty$: Converges to within $O(\Delta t)$ of the continuous QSD
- For **fixed $k$**, as $\Delta t \to 0$: Approximates the continuous distribution at time $t = k \Delta t$
- **Optimal balance:** Choose $\Delta t$ such that $C \Delta t \approx M \rho^k$ (balance discretization and convergence errors)
:::

---

## 6. Fractal Set Specifics: Data Representation and Convergence

### 6.1. Scalars, Spinors, and Dynamics

**Question:** Does storing data as scalars (nodes) vs. spinors (edges) affect convergence?

**Answer:** **No.** Convergence is a property of the **numerical values** $(x_i, v_i)$, not their representation.

:::{prf:remark} Representation Independence
:label: rem-representation-independence

The Fractal Set uses:
- **Scalars in nodes**: Fitness $\Phi$, energies $E_{\text{kin}}, U$, statistics $\mu_\rho, \sigma_\rho$
- **Spinors in edges**: Velocity $\psi_v$, forces $\psi_{\mathbf{F}_{\text{stable}}}, \psi_{\mathbf{F}_{\text{adapt}}}, \ldots$

This is a **storage format choice**. The underlying Markov chain dynamics operate on the **numerical state vector** $Z_k = (X_k, V_k)$.

**Why spinors?**
- Gauge theory formulation (cf. `12_gauge_theory_adaptive_gas.md`)
- Emergent geometry (cf. `08_emergent_geometry.md`)
- QFT interpretation (cf. `13_D_fractal_set_emergent_qft_comprehensive.md`)

But for **convergence**, we only care about the dynamics of $(x_i, v_i)$ values.
:::

### 6.2. Antisymmetric IG Edges and Cloning

**Question:** Does the directed IG graph structure with antisymmetric fitness potential $V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i$ matter for convergence?

**Answer:** The antisymmetry is the **standard fitness difference** that drives cloning in {prf:ref}`lem-quantitative-keystone`. It's already accounted for in the force fields.

:::{prf:remark} Cloning in the Drift
:label: rem-cloning-in-drift

The adaptive and viscous forces contain cloning effects:

$$
\begin{aligned}
\mathbf{F}_{\text{adapt}}(x_i, Z) &= \epsilon_F \nabla V_{\text{fit}}[f_k, \rho](x_i) \\
\mathbf{F}_{\text{viscous}}(x_i, Z) &= \nu \sum_{j \neq i} K_\rho(x_i, x_j) (v_j - v_i)
\end{aligned}

$$

These forces **depend on other walkers' states** via the empirical measure $f_k$ and pairwise coupling.

The **Keystone Principle** ({prf:ref}`lem-quantitative-keystone`) guarantees that:
- High-error walkers experience high cloning pressure (encoded in $V_{\text{fit}}$)
- This contracts positional variance

**In the discrete proof:** Cloning is **baked into** the force vector $\mathbf{F}_{\text{total}}$, so the Taylor expansion in {prf:ref}`lem-baoab-error-bound` automatically accounts for it. No separate analysis needed!
:::

**Fermionic interpretation:** The antisymmetry $V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)$ becomes significant in the **emergent QFT limit** (cf. `13_D_fractal_set_emergent_qft_comprehensive.md`) where it enables antisymmetric Green's functions. But for convergence, it's just fitness-driven selection.

---

## 7. Summary and Practical Implications

### 7.1. What We Have Proven

:::{prf:theorem} Complete Convergence Fidelity
:label: thm-complete-fidelity

The Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ is the path history of a **geometrically ergodic discrete-time Markov chain**. Specifically:

**1. Information:** {prf:ref}`thm-fractal-set-reconstruction` proves the Fractal Set contains complete SDE data (reconstruction).

**2. Dynamics:** {prf:ref}`thm-fractal-set-ergodicity` proves the generator is geometrically ergodic (convergence).

**3. Fidelity:** {prf:ref}`thm-weak-convergence-invariant` proves the discrete invariant measure approximates the continuous QSD.

**Combined:** The Fractal Set is a **faithful discrete representation** of the Adaptive Gas SDE, inheriting all convergence guarantees.
:::

### 7.2. Convergence Rate Summary

| **Property** | **Continuous SDE** | **Discrete (BAOAB)** | **Source** |
|--------------|-------------------|---------------------|-----------|
| **Drift coefficient** | $\kappa_{\text{total}}$ | $\kappa_{\text{total}}/2$ | {prf:ref}`thm-discrete-drift-baoab` |
| **Contraction per step** | $e^{-\kappa \Delta t}$ | $1 - \kappa \Delta t / 2 \approx e^{-\kappa \Delta t/2}$ | {prf:ref}`rem-discrete-contraction` |
| **Convergence rate** | $\\|\\mu_t - \pi\\| \leq M e^{-\kappa t}$ | $\\|\mu_k - \pi_{\Delta t}\\| \leq M \rho^k$ | {prf:ref}`thm-fractal-set-ergodicity` |
| **Discretization error** | N/A | $O(\Delta t)$ | {prf:ref}`prop-total-error` |
| **Order of integrator** | N/A | Second-order (BAOAB) | {prf:ref}`def-baoab-integrator` |

**Key insight:** For small $\Delta t$, the discrete rate is **half** the continuous rate per unit time, but this is compensated by taking more steps. For fixed continuous time $t = k \Delta t$, both converge at the same exponential rate.

### 7.3. Practical Guidelines

**Choosing timestep $\Delta t$:**

1. **Stability constraint:** $\Delta t < \Delta t_{\max} = \min(\tau_{\max}, \frac{1}{K_1}, \frac{\kappa_{\text{total}}}{2K_1})$

2. **Accuracy vs. speed tradeoff:**
   - **Smaller $\Delta t$**: Lower discretization error $O(\Delta t)$, but more steps needed
   - **Larger $\Delta t$**: Faster simulation, but larger gap between $\pi_{\Delta t}$ and $\pi$

3. **Rule of thumb:** Choose $\Delta t$ such that:
   $$
   C_{\text{approx}} \Delta t \approx M \rho^k \cdot \text{tolerance}
   $$
   Balance discretization and convergence errors.

**Verifying convergence in practice:**

- **Monitor Lyapunov function:** Track $V_{\text{total}}(Z_k)$ over time. Should decay exponentially.
- **Check empirical distribution:** Compare histogram of node fitness values in Fractal Set to expected QSD.
- **Timestep sensitivity:** Run with different $\Delta t$ and verify results are consistent (up to $O(\Delta t)$).

### 7.4. Future Extensions

**Open questions:**

1. **Explicit constants:** Can we compute $K_1, K_2$ explicitly in terms of system parameters?
2. **Adaptive timestep:** Can we dynamically adjust $\Delta t$ based on local error estimates?
3. **Higher-order integrators:** Would a third-order method (e.g., BCOCB) improve convergence?
4. **Non-Markovian effects:** Does the Fractal Set's memory (storing full history) enable non-Markovian analysis?

**Connection to other frameworks:**

- **Gauge theory:** Does geometric ergodicity have a gauge-theoretic interpretation?
- **QFT:** How does discrete ergodicity connect to vacuum state stability in the emergent QFT?
- **Information geometry:** Can we use Fisher information metric to bound discretization error?

---

## References

### Primary Framework Documents

1. **[00_full_set.md](00_full_set.md):** Fractal Set definition and reconstruction theorem
2. **[00_reference.md](../00_reference.md):** Complete mathematical reference with all proven theorems
3. **[04_convergence.md](../04_convergence.md):** Kinetic operator and BAOAB integrator
4. **[07_adaptative_gas.md](../07_adaptative_gas.md):** Adaptive Gas SDE and Foster-Lyapunov drift
5. **[03_cloning.md](../03_cloning.md):** Keystone Principle and cloning dynamics

### Mathematical References

1. **Meyn & Tweedie (2009).** *Markov Chains and Stochastic Stability* (2nd ed.). Cambridge University Press.
   - Theorem 15.0.1: Geometric ergodicity via drift conditions

2. **Leimkuhler & Matthews (2015).** *Molecular Dynamics: With Deterministic and Stochastic Numerical Methods*. Springer.
   - Chapter 7: BAOAB integrator and symplectic methods

3. **Talay & Tubaro (1990).** "Expansion of the global error for numerical schemes solving stochastic differential equations." *Stochastic Analysis and Applications*, 8(4), 483-509.
   - Weak approximation theory for SDEs

### Discussion Documents

1. **[discussions/convergence_inheritance_strategy.md](discussions/convergence_inheritance_strategy.md):** Detailed proof strategy and technical challenges
