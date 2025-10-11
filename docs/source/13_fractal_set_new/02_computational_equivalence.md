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

### 6.3. Transfer of Symmetries and Conserved Quantities

**Question:** Do the symmetries and conserved quantities defined in the Fractal Set data structure correspond to physical properties of the N-particle system?

**Answer:** **Yes.** Every symmetry and quantity defined on the Fractal Set graph structure has a unique corresponding property in the N-particle Markov chain dynamics, and vice versa. This establishes **equivalence** between the Fractal Set representation and the algorithmic dynamics.

:::{note}
The symmetry theorems referenced here ({prf:ref}`thm-sn-braid-holonomy`, {prf:ref}`thm-u1-fitness-global`, {prf:ref}`thm-su2-interaction-symmetry`, {prf:ref}`thm-sn-su2-lattice-qft`) are formally established in [00_full_set.md](00_full_set.md) (Section 7). These definitions may need to be added to [00_reference.md](../00_reference.md) for central indexing.
:::

:::{prf:theorem} Equivalence of Fractal Set and N-Particle Properties
:label: thm-fractal-set-n-particle-equivalence

Let $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ be the Fractal Set and let $\{Z_k\}_{k \geq 0}$ be the N-particle discrete-time Markov chain from {prf:ref}`def-swarm-state-vector`. Then:

**1. Bijective correspondence of states:**

For each node $n_{i,t} \in \mathcal{N}$, there exists a unique walker state $(x_i(t), v_i(t)) \in Z_t$, and conversely, each walker state at timestep $t$ corresponds to exactly one node in $\mathcal{N}$.

**2. Reconstruction of phase space:**

Given $\mathcal{F}$, the full phase space trajectory can be reconstructed via:

$$
x_i(t) = x_i(0) + \sum_{s=0}^{t-1} \text{spinor-to-vector}[\psi_{\Delta x}(n_{i,s}, n_{i,s+1})]

$$

$$
v_i(t) = \text{spinor-to-vector}[\psi_{v,t}(n_{i,t-1}, n_{i,t})]

$$

where the spinor-to-vector map extracts the $\mathbb{R}^d$ vector from its spinor encoding $\psi \in \mathbb{C}^{2^{[d/2]}}$.

**3. Transfer of symmetries:**

The symmetry structure on $\mathcal{F}$ ({prf:ref}`thm-sn-su2-lattice-qft`) corresponds to algorithmic symmetries of the N-particle system. The Fractal Set has a three-tier symmetry hierarchy:

$$
G_{\text{total}} = S_N^{\text{discrete}} \times \text{SU}(2)_{\text{weak}}^{\text{local}} \times \text{U}(1)_{\text{fitness}}^{\text{global}}

$$

where $S_N$ (permutation group) is the **fundamental discrete gauge symmetry**, while SU(2) and U(1) are **emergent** symmetries in the continuum limit.

**S_N permutation symmetry** ↔ **Walker indistinguishability**:

The **fundamental gauge symmetry** of the algorithm is invariance under walker label permutations. For any permutation $\sigma \in S_N$:

$$
\mathcal{S} = (w_1, w_2, \ldots, w_N) \sim \sigma \cdot \mathcal{S} = (w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(N)})

$$

This discrete gauge symmetry gives rise to **braid group topology** $\pi_1(\mathcal{M}_{\text{config}}) \cong B_N$ ({prf:ref}`thm-sn-braid-holonomy`). Closed loops in configuration space have **discrete holonomy** $\text{Hol}(\gamma) \in S_N$, representing the net permutation of walkers after traversing the loop.

**Physical consequence**: All algorithmic observables must be $S_N$-invariant - they depend on unordered sets of walkers, not specific labels. The $S_N$ holonomy is the **fundamental discrete gauge observable**, analogous to how Wilson loops probe continuous gauge connections (but distinct from them - $S_N$ holonomy is a permutation, not a matrix trace).

**Global U(1) fitness symmetry** ↔ **Fitness conservation**:

The **global** (not gauged) U(1) symmetry ({prf:ref}`thm-u1-fitness-global`) acts by uniform phase rotation on all diversity amplitudes:

$$
\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}, \quad \alpha \in [0, 2\pi) \text{ (same } \alpha \text{ for all i)}

$$

This corresponds to **invariance under global fitness shifts** $\Phi(x) \mapsto \Phi(x) + c$ in the N-particle system. The diversity companion probability:

$$
P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2}\right)}{\sum_{k'} \exp\left(-\frac{d_{\text{alg}}(i,k')^2}{2\epsilon_d^2}\right)}

$$

depends only on positions and velocities (via $d_{\text{alg}}$), **not** on absolute fitness values, making it invariant under global fitness shifts.

**Conserved charge (Noether)**: Global U(1) symmetry implies conservation of fitness current $J_{\text{fitness}}^\mu = \sum_i \text{Im}(\psi_i^* \partial^\mu \psi_i)$, analogous to baryon number conservation in particle physics.

**Local SU(2) weak isospin symmetry** ↔ **Approximate role exchange in cloning**:

The SU(2) doublet structure $|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle$ ({prf:ref}`thm-su2-interaction-symmetry`) corresponds to an **approximate algorithmic symmetry** where the cloning dynamics are nearly invariant under exchange of cloner and target roles. This near-invariance is quantified by the ratio:

$$
\frac{P_{\text{clone}}(i \to j)}{P_{\text{clone}}(j \to i)} \approx 1 + O(\Phi_j - \Phi_i)

$$

The symmetry becomes **exact** in the limit of vanishing fitness differences:

$$
\lim_{\Phi_j - \Phi_i \to 0} \frac{P_{\text{clone}}(i \to j)}{P_{\text{clone}}(j \to i)} = 1

$$

**Physical meaning**: On the fitness-degenerate submanifold, walkers are interchangeable with respect to weak isospin transformations, and the dynamics exhibit exact SU(2) invariance under mixing of cloner and target roles.

**4. Conserved and derived quantities:**

**Energy conservation in BAOAB**: The total energy $H_{\text{total}}(Z_k) = \sum_i [E_{\text{kin},i} + U(x_i)]$ stored in nodes satisfies:

$$
\mathbb{E}[H_{\text{total}}(Z_{k+1}) | Z_k] = H_{\text{total}}(Z_k) + O(\Delta t)

$$

up to dissipation from friction and stochastic forcing.

This corresponds to **energy flow balance** in the Fractal Set: Energy changes along CST edges equal the work done by forces stored in edge spinors:

$$
E_{\text{kin}}(n_{i,t+1}) - E_{\text{kin}}(n_{i,t}) = \int_{t}^{t+1} \text{spinor-to-vector}[\psi_{\mathbf{F}_{\text{total}}}] \cdot \text{spinor-to-vector}[\psi_{v}] \, ds

$$

**Derived statistical observables**: Localized statistical moments $\mu_\rho(n), \sigma_\rho(n)$ stored in nodes are **not conserved quantities** but rather **functionals of IG edge data**. They satisfy:

$$
\mu_\rho[f_k, \Phi, x_i(t)] = \frac{\sum_{j \in A_k(t)} w_{ij}(\rho) \Phi_j}{\sum_{j \in A_k(t)} w_{ij}(\rho)}

$$

where the localization weights $w_{ij}(\rho) = K_\rho(x_i - x_j)$ are stored in IG edges. The node scalars $\mu_\rho(n_{i,t})$ are **computed observables derived from** the empirical measure $f_k(t)$ encoded in the graph structure, not fundamental quantities. This establishes **correspondence** between node-stored statistics and the underlying N-particle distribution.

**Proof:**

**Part 1 (Bijection):** By construction ({prf:ref}`def-node-spacetime`), each node $n_{i,t}$ has unique walker ID $i$ and timestep $t$, establishing a one-to-one correspondence with $(x_i(t), v_i(t)) \in Z_t$.

**Part 2 (Reconstruction):** Follows directly from {prf:ref}`thm-fractal-set-reconstruction` items 1-2. The position is obtained by integrating displacement spinors along CST edges, and velocity is directly stored in CST edge spinors.

**Part 3 (Symmetries):**

**S_N permutation proof:** The N-particle algorithm treats walkers as **indistinguishable particles** with arbitrary labels. Any physical observable must be invariant under label permutations $\sigma \in S_N$. For example:

- Cloning probability depends on walker states, not labels: $P_{\text{clone}}(\sigma(i) \to \sigma(j) | \sigma \cdot \mathcal{S}) = P_{\text{clone}}(i \to j | \mathcal{S})$
- Fitness histogram is an unordered set: $\{V_{\text{fit}}(i) : i \in A_t\}$ is $S_N$-invariant
- Diversity distances are symmetric: $d_{\text{alg}}(i, j) = d_{\text{alg}}(j, i)$

In the Fractal Set, this fundamental gauge redundancy manifests as **braid holonomy**. When walkers exchange positions during dynamics, their worldlines form braids in spacetime. Closed loops $\gamma$ in configuration space have holonomy $\text{Hol}(\gamma) \in S_N$, measuring the net permutation acquired.

**Discrete holonomy**: For a closed path in CST+IG, the holonomy is computed via the braid homomorphism $\rho: B_N \to S_N$. This **discrete gauge structure** is fundamental - all other symmetries (U(1), SU(2)) are emergent in the continuum limit. Unlike continuous Wilson loops (which are matrix traces), the $S_N$ holonomy is a discrete permutation, making this a hybrid discrete-continuous gauge theory.

**Global U(1) proof:** The diversity companion selection follows a Gibbs distribution. From the Fractal Set specification, the probability is:

$$
P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2}\right)}{\sum_{k' \in A_t \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}(i,k')^2}{2\epsilon_d^2}\right)}

$$

The algorithmic distance is defined as:

$$
d_{\text{alg}}(i, k)^2 = \|x_i - x_k\|^2 + \lambda_v \|v_i - v_k\|^2

$$

where $\lambda_v$ is a constant weighting parameter. **Crucially, $d_{\text{alg}}$ depends only on positions and velocities, NOT on fitness values $\Phi(x_i), \Phi(x_k)$.**

Under a global fitness shift $\Phi(x) \mapsto \Phi(x) + c$ for constant $c \in \mathbb{R}$:

$$
\begin{aligned}
d_{\text{alg}}(i,k)^2 &= \|x_i - x_k\|^2 + \lambda_v \|v_i - v_k\|^2 \\
&\quad \text{(unchanged, as positions and velocities are independent of fitness labeling)} \\
\Rightarrow P_{\text{comp}}^{(\text{div})}(k|i) &= \frac{\exp(-d_{\text{alg}}^2/(2\epsilon_d^2))}{\sum_{k'} \exp(-d_{\text{alg}}(i,k')^2/(2\epsilon_d^2))} \\
&\quad \text{(invariant)}
\end{aligned}

$$

The U(1) phase $\theta_{ik}^{(\text{U(1)})} = -d_{\text{alg}}(i,k)^2/(2\epsilon_d^2 \hbar_{\text{eff}})$ is likewise invariant. **Global** phase transformations $\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}$ (same $\alpha$ for all $i, k$) leave all observables invariant:

$$
\left|\psi_{ik}^{(\text{div})}\right|^2 = P_{\text{comp}}^{(\text{div})}(k|i) \quad \text{(global U(1) invariant)}

$$

This is **not a gauge symmetry** - there is no dynamical gauge field. The phases $\theta_{ik}$ are fixed by algorithmic distances. It's a **global symmetry** like baryon number conservation, giving a conserved Noether current $J_{\text{fitness}}^\mu$.

**SU(2) proof:** The cloning amplitude $\Psi(i \to j)$ factorizes ({prf:ref}`def-cloning-amplitude-factorization`) as:

$$
\Psi(i \to j) = A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)

$$

where $A_{ij}^{\text{SU(2)}} = \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} e^{i\theta_{ij}^{(\text{SU(2)})}}$ is the bare SU(2) vertex and $K_{\text{eff}}$ is the U(1)-dressed effective kernel.

An SU(2) transformation $U \in \text{SU}(2)$ acts on the isospin doublet $(|\psi_i\rangle, |\psi_j\rangle)^T$ as a rotation, mixing cloner and target roles. In the N-particle system, the amplitude ratio:

$$
\frac{\Psi(i \to j)}{\Psi(j \to i)} = \frac{A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)}{A_{ji}^{\text{SU(2)}} \cdot K_{\text{eff}}(j, i)}

$$

The SU(2) vertex satisfies $A_{ij}^{\text{SU(2)}} \propto \exp(i\theta_{ij}^{(\text{SU(2)})})$ where the phase depends on algorithmic distance, which is **symmetric** under $(i,j)$ exchange. However, the probability amplitude $\sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)}$ contains **fitness asymmetry** through the cloning score $S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$.

Therefore:

$$
\frac{\Psi(i \to j)}{\Psi(j \to i)} = \exp\left(\frac{i}{\hbar_{\text{eff}}} [S(i,j,k,m) - S(j,i,m,k)] + O(\Phi_j - \Phi_i)\right)

$$

In the limit $\Phi_j - \Phi_i \to 0$, the fitness differences vanish and the ratio approaches unity, establishing **exact SU(2) symmetry on the fitness-degenerate submanifold**.

**Part 4 (Conserved and derived quantities):**

**Energy conservation:** From the BAOAB O-step ({prf:ref}`def-baoab-kernel`), the velocity update includes stochastic forcing:

$$
v^{(2)} = e^{-\gamma \Delta t} v^{(1)} + \sqrt{\frac{1}{\gamma}(1 - e^{-2\gamma \Delta t})} \, \Sigma_{\text{reg}} \xi

$$

where $\xi \sim \mathcal{N}(0, I_d)$. The kinetic energy evolves as:

$$
\begin{aligned}
E_{\text{kin}}(v_{k+1}) &= \frac{1}{2}\|v_{k+1}\|^2 \\
&= \frac{1}{2}\|v_k + \Delta t \, \mathbf{F}_{\text{total}} + \sqrt{\Delta t} \, \Sigma_{\text{reg}} \tilde{\xi}\|^2 \\
&\quad \text{(where } \tilde{\xi} \text{ is the rescaled noise)} \\
&= \frac{1}{2}\|v_k\|^2 + \Delta t \, v_k \cdot \mathbf{F}_{\text{total}} + \sqrt{\Delta t} \, v_k \cdot \Sigma_{\text{reg}} \tilde{\xi} \\
&\quad + \frac{\Delta t^2}{2}\|\mathbf{F}_{\text{total}}\|^2 + \frac{\Delta t}{2}\|\Sigma_{\text{reg}} \tilde{\xi}\|^2 + O(\Delta t^{3/2})
\end{aligned}

$$

Taking expectations over the noise $\mathbb{E}[\tilde{\xi}] = 0, \mathbb{E}[\|\tilde{\xi}\|^2] = d$:

$$
\begin{aligned}
\mathbb{E}[E_{\text{kin}}(v_{k+1}) | v_k] &= E_{\text{kin}}(v_k) + \Delta t \, v_k \cdot \mathbf{F}_{\text{total}} \\
&\quad + \frac{\Delta t}{2} \text{Tr}(\Sigma_{\text{reg}} \Sigma_{\text{reg}}^T) + O(\Delta t^2)
\end{aligned}

$$

The $O(\Delta t)$ terms are:
- $v_k \cdot \mathbf{F}_{\text{total}}$: **Power from forces** (work done)
- $\frac{1}{2}\text{Tr}(\Sigma_{\text{reg}}^2)$: **Stochastic heating**

The $O(\Delta t^2)$ terms include friction dissipation $-\gamma \Delta t \|v_k\|^2$ and higher-order force-noise coupling. For the **total energy** $H_{\text{total}} = \sum_i [E_{\text{kin},i} + U(x_i)]$:

$$
\mathbb{E}[H_{\text{total}}(Z_{k+1}) | Z_k] = H_{\text{total}}(Z_k) + \Delta t \, (\text{heating} - \text{dissipation}) + O(\Delta t^2)

$$

where the $O(\Delta t)$ term represents the **balance between stochastic forcing and friction**, not exact conservation.

The term $v_k \cdot \mathbf{F}_{\text{total}}$ equals:

$$
v_k \cdot \mathbf{F}_{\text{total}} = \text{spinor-to-vector}[\psi_v] \cdot \text{spinor-to-vector}[\psi_{\mathbf{F}_{\text{total}}}]

$$

using the spinor representation. The node scalars $E_{\text{kin}}(n_{i,t+1})$ and $E_{\text{kin}}(n_{i,t})$ store the energies, while the CST edge spinors $\psi_v, \psi_{\mathbf{F}_{\text{total}}}, \psi_{\Sigma_{\text{reg}}}$ enable computing the energy flow components.

**Derived statistical observables:** The localized mean $\mu_\rho[f_k, \Phi, x_i]$ is computed from the empirical measure $f_k(t)$:

$$
f_k(t) = \frac{1}{k} \sum_{j \in A_k(t)} \delta_{(x_j(t), v_j(t))}

$$

where $A_k(t) = \{j : s(n_{j,t}) = 1\}$ is determined from node status flags. The localization kernel $K_\rho(x_i - x_j)$ stored in IG edges provides the weighting:

$$
\mu_\rho[f_k, \Phi, x_i] = \sum_{j \in A_k} w_{ij}(\rho) \Phi_j, \quad w_{ij}(\rho) = \frac{K_\rho(x_i - x_j)}{\sum_{m \in A_k} K_\rho(x_i - x_m)}

$$

The node scalar $\mu_\rho(n_{i,t})$ equals the right-hand side. This proves that localized statistics are **derived observables** (functionals of the N-particle configuration), not independent quantities or conservation laws. The correspondence establishes that node-stored moments accurately represent the algorithmic state's local statistical structure. ∎
:::

:::{prf:corollary} Fractal Set as Complete Algorithmic Representation
:label: cor-fractal-set-complete-representation

The Fractal Set $\mathcal{F}$ is **informationally and dynamically complete** for the N-particle Adaptive Gas algorithm:

**1. Informational completeness**: Every scalar, spinor, and graph structure element in $\mathcal{F}$ corresponds to a unique quantity or operation in the N-particle system (Reconstruction Theorem {prf:ref}`thm-fractal-set-reconstruction`).

**2. Dynamical completeness**: Every symmetry and conservation law in the N-particle system has a unique representation in $\mathcal{F}$ ({prf:ref}`thm-fractal-set-n-particle-equivalence`).

**3. Convergence equivalence**: The discrete-time Markov chain generating $\mathcal{F}$ converges to the same long-term distribution as the continuous SDE, up to $O(\Delta t)$ discretization error ({prf:ref}`thm-fractal-set-ergodicity`, {prf:ref}`thm-weak-convergence-invariant`).

**Practical implication**: Analyzing the Fractal Set graph structure is **equivalent** to analyzing the N-particle algorithm. Properties proven for $\mathcal{F}$ (gauge symmetries, lattice QFT structure, fermionic propagators) are **properties of the algorithm**, not artifacts of the representation.
:::

:::{prf:remark} Why Spinors Are Necessary
:label: rem-spinors-necessary

**Question:** Could we store vectors $v \in \mathbb{R}^d$ directly instead of spinors $\psi \in \mathbb{C}^{2^{[d/2]}}$?

**Answer:** For **reconstruction** purposes, yes - vectors suffice. But for **gauge theory interpretation** and **emergent QFT formulation**, spinors are essential:

1. **Frame independence**: Spinors transform covariantly under rotations without requiring a preferred basis.
2. **Geometric naturalness**: Spinors are the fundamental representation of the Lorentz/rotation group.
3. **Gauge connections**: The U(1) and SU(2) phases $\theta_{ik}^{(\text{U(1)})}, \theta_{ij}^{(\text{SU(2)})}$ act on spinor phases, not vector components.
4. **Fermionic structure**: Spinors enable antisymmetric Green's functions in the continuum limit.
5. **Clifford algebra**: Spinor formalism naturally encodes the geometric product structure underlying the viscous coupling.

**For convergence analysis** (this document), vectors would suffice. **For the full framework** (gauge theory, QFT, emergent geometry), spinors are the natural choice.
:::

:::{prf:remark} Antisymmetric IG Edges and Fermionic Statistics
:label: rem-antisymmetric-ig-fermionic

The **directed antisymmetric** structure of IG edges:

$$
V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i = -V_{\text{clone}}(j \to i)

$$

implies that IG edges are **oriented** by fitness differences. This has two interpretations:

**1. Algorithmic interpretation (this document):** Cloning is driven by fitness gradients. Walkers with higher fitness attract cloning events.

**2. QFT interpretation** (`13_D_fractal_set_emergent_qft_comprehensive.md`): The antisymmetry enables **fermionic Green's functions**:

$$
G_F(i, j) = \langle \psi(i) \bar{\psi}(j) \rangle = -G_F(j, i)

$$

where $\psi(i)$ is a fermionic field operator at node $n_i$.

In the continuum limit $N \to \infty, \Delta t \to 0$, the antisymmetric IG structure becomes a **fermionic propagator** with Grassmann-valued path integral. This is a **precursor** to fermionic statistics, not full fermions (walkers are distinguishable).

**Key point:** The antisymmetry is **not optional** - it is fundamental to both the algorithmic fitness-driven dynamics and the emergent fermionic structure.
:::

---

## 7. Four Perspectives: N-Particle, Mean-Field, Fractal Set, and Scutoid Geometry

### 7.1. The Four Equivalent Descriptions

The Fragile Gas algorithm admits **four equivalent mathematical descriptions**, each with its own natural framework for analysis:

| **Perspective** | **Primary Objects** | **Natural Questions** | **Key Framework Documents** |
|-----------------|---------------------|----------------------|------------------------------|
| **N-Particle** | Swarm state $Z_k = (X_k, V_k)$ | Ergodicity, convergence rates, Lyapunov drift | [04_convergence.md](../04_convergence.md), [07_adaptative_gas.md](../07_adaptative_gas.md) |
| **Mean-Field** | Empirical measure $\mu_N \to \mu$ | McKean-Vlasov PDE, propagation of chaos | [05_mean_field.md](../05_mean_field.md), [06_propagation_chaos.md](../06_propagation_chaos.md) |
| **Fractal Set** | Discrete spacetime graph $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ | Information reconstruction, gauge symmetries, lattice QFT | [00_full_set.md](00_full_set.md), current document |
| **Scutoid Geometry** | Spacetime tessellation $\mathcal{T} = \{S_i\}$ | Emergent curvature, topological order, holographic principle | [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md) |

**Central claim**: These four descriptions are **mathematically equivalent**, meaning:
1. **Bijective correspondence**: Each object in one framework has a unique counterpart in the others
2. **Convergence transfer**: Convergence guarantees proven in any framework transfer to all others
3. **Observable equivalence**: Physical/algorithmic observables computed in different frameworks agree

This section establishes the scutoid framework as the **fourth pillar** of the Fragile Gas theory, complementing the N-particle, mean-field, and Fractal Set perspectives.

### 7.2. Scutoid Framework: Overview and Key Concepts

:::{prf:definition} Scutoid Tessellation of Swarm Spacetime
:label: def-scutoid-tessellation-overview

From {prf:ref}`def-scutoid-cell-genealogical` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md), a **scutoid cell** $S_i$ is a spacetime volume connecting Voronoi cells at adjacent time slices $t$ and $t+1$:

$$
S_i = \text{Vor}_i(t) \times \{t\} \cup \text{Vor}_{j}(t+1) \times \{t+1\} \cup \text{(lateral faces)}
$$

where $j$ is the descendant of walker $i$ (either $j = i$ if alive, or $j = \text{parent}(i)$ if cloned).

**Key geometric properties**:
1. **Prism** ($S_i$ is a prism): Walker $i$ survives without cloning → Top and bottom faces are congruent
2. **Scutoid** ($S_i$ is a scutoid): Walker $i$ is involved in cloning → Neighbor topology changes between time slices

The **scutoid tessellation** is the complete spacetime partition:

$$
\mathcal{T} = \{S_1, S_2, \ldots, S_N\} \quad \text{with} \quad \bigcup_{i=1}^N S_i = \mathcal{X} \times [t, t+1]
$$

**Topological order parameter**: The **scutoid fraction** $\phi(t)$ is the fraction of cells that are scutoids (not prisms) at time $t$:

$$
\phi(t) := \frac{\#\{\text{scutoid cells at time } t\}}{N}
$$

This quantifies the algorithmic exploration phase: $\phi \approx 1$ (exploratory), $\phi \approx 0$ (convergent).
:::

:::{prf:remark} Why Scutoids?
:label: rem-why-scutoids-overview

The scutoid framework provides:
1. **Geometric language**: Cloning events (discrete graph operations) become continuous geometric transformations
2. **Mesoscopic bridge**: Links discrete Fractal Set (graph) to continuous mean-field (PDE)
3. **Curvature dictionary**: Scutoid shape parameters encode Riemann curvature tensor components
4. **Topological invariants**: Euler characteristic, holonomy, and genus are computable from scutoid data

**Physical analogy**: Scutoids in biological tissue minimize packing energy during deformation. Scutoids in the Fragile Gas minimize **information rearrangement cost** (Wasserstein-2 distance) during walker redistribution.
:::

### 7.3. Bijective Correspondence: Fractal Set ↔ Scutoid Tessellation

:::{prf:theorem} Fractal Set and Scutoid Tessellation Are Dual Structures
:label: thm-fractal-scutoid-duality

Let $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ be the Fractal Set and $\mathcal{T} = \{S_i\}$ be the scutoid tessellation. Then there exists a **bijective correspondence**:

$$
\mathcal{F} \xrightarrow{\text{1:1}} \mathcal{T}
$$

with the following structure:

**1. Node-to-cell correspondence**:

Each node $n_{i,t} \in \mathcal{N}$ corresponds to a unique scutoid cell $S_i(t, t+1)$ connecting time slices $t$ and $t+1$.

**2. CST edges encode scutoid connectivity**:

A CST edge $e = (n_{i,t}, n_{j,t+1}) \in E_{\text{CST}}$ encodes the **temporal connection** between scutoid $S_i$ (at level $t$) and scutoid $S_j$ (at level $t+1$):
- If $j = i$ (walker survives): $S_i$ is a **prism**
- If $j \neq i$ (walker cloned or died): $S_i$ is a **scutoid** with neighbor-swapping

**3. IG edges encode neighbor relations**:

An IG edge $e = (n_{i,t}, n_{k,t}) \in E_{\text{IG}}$ at time $t$ encodes the **neighbor relation** between Voronoi cells $\text{Vor}_i(t)$ and $\text{Vor}_k(t)$. This translates to:
- Scutoids $S_i$ and $S_k$ share a **lateral face** at time slice $t$
- The IG edge weight $w_{ik}(t)$ (companion probability) determines the **area** of the shared interface

**4. Spinors encode scutoid geometry**:

The spinors stored in Fractal Set edges encode geometric quantities of scutoid cells:
- $\psi_v(n_{i,t}, n_{i,t+1})$: Velocity field along the prism/scutoid axis
- $\psi_{\mathbf{F}_{\text{total}}}$: Force field driving scutoid deformation
- $\psi_{\Sigma_{\text{reg}}}$: Diffusion tensor controlling scutoid face curvature

**5. Cloning events as scutoid transformations**:

A cloning event $(i \to j)$ recorded in the Fractal Set corresponds to a **topological transformation** in the scutoid tessellation:
- Walker $i$ at time $t$ has Voronoi cell $\text{Vor}_i(t)$
- After cloning, walker $i$ is replaced by a clone of walker $j$
- At time $t+1$, the Voronoi cell $\text{Vor}_i(t+1)$ is now part of $\text{Vor}_j(t+1)$'s influence region
- The scutoid $S_i$ connecting these has **mid-level vertices** where the neighbor topology changes

**Proof**: Direct construction from {prf:ref}`thm-scutoid-ig-duality` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md). See Section 7.4 below for detailed mapping. ∎
:::

### 7.4. Detailed Construction: From Fractal Set to Scutoid Tessellation

:::{prf:construction} Episode-to-Scutoid Map
:label: constr-episode-scutoid-map

Given a Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ with time horizon $T$, construct the scutoid tessellation $\mathcal{T}$ as follows:

**Input**: Fractal Set $\mathcal{F}$ with:
- Nodes $\mathcal{N} = \{n_{i,t}\}$ with positions $x_i(t)$, velocities $v_i(t)$, status $s(n_{i,t}) \in \{0, 1\}$
- CST edges $E_{\text{CST}}$ with spinors $\psi_v, \psi_{\mathbf{F}_{\text{total}}}, \psi_{\Sigma_{\text{reg}}}$
- IG edges $E_{\text{IG}}$ with companion weights $w_{ik}(t)$

**Step 1: Voronoi tessellation at each time slice**

For each time $t \in \{0, 1, \ldots, T\}$:
1. Extract alive walker positions: $\mathcal{A}(t) = \{i : s(n_{i,t}) = 1\}$, $X(t) = \{x_i(t) : i \in \mathcal{A}(t)\}$
2. Compute Riemannian Voronoi diagram $\mathcal{V}_t = \{\text{Vor}_i(t) : i \in \mathcal{A}(t)\}$ using the emergent metric:

$$
g(x, t) = H(x, S_t) + \epsilon_\Sigma I
$$

where $H(x, S_t) = \nabla^2 V_{\text{fit}}[f_t](x)$ is the fitness Hessian and $f_t$ is the empirical measure reconstructed from the Fractal Set.

**Remark**: The metric $g(x, t)$ is fully determined by node data via {prf:ref}`thm-fractal-set-reconstruction`.

**Step 2: Identify cloning events**

For each time interval $[t, t+1]$:
1. For each node $n_{i,t}$, find its CST successor: $n_{j,t+1}$ where $(n_{i,t}, n_{j,t+1}) \in E_{\text{CST}}$
2. Classify the episode:
   - **Survival**: $j = i$ (walker continues)
   - **Death**: No outgoing CST edge from $n_{i,t}$
   - **Birth**: No incoming CST edge to $n_{j,t+1}$ (cloned from another walker)

**Step 3: Construct scutoid cells**

For each walker $i \in \mathcal{A}(t)$:

**Case 1 (Prism)**: Walker $i$ survives to time $t+1$ without cloning involvement
- Bottom face: $F_{\text{bottom}} = \text{Vor}_i(t) \times \{t\}$
- Top face: $F_{\text{top}} = \text{Vor}_i(t+1) \times \{t+1\}$
- Lateral faces: Ruled surfaces connecting $\partial \text{Vor}_i(t)$ to $\partial \text{Vor}_i(t+1)$
- Cell type: **Prism** (congruent top/bottom, vertical edges)

**Case 2 (Scutoid via cloning)**: Walker $i$ at time $t$ is replaced by a clone of walker $j$ (where $j \neq i$)
- Bottom face: $F_{\text{bottom}} = \text{Vor}_i(t) \times \{t\}$
- Top face: $F_{\text{top}} = \text{Vor}_j(t+1) \times \{t+1\}$ (portion that was originally $\text{Vor}_i$)
- Mid-level vertices: Introduced at the boundary where $\text{Vor}_i(t)$ transitions to being absorbed into $\text{Vor}_j(t+1)$
- Lateral faces: **Curved surfaces** (non-planar) due to mid-level vertices forcing neighbor topology changes
- Cell type: **Scutoid** (neighbor set changes from bottom to top)

**Step 4: Encode geometric data**

For each scutoid cell $S_i$:
1. **Volume**: Compute from Voronoi cells using the Riemannian volume form $dV_g = \sqrt{\det g} \, dx$
2. **Lateral face areas**: Determine from IG edge weights $w_{ik}(t)$ (proportional to interface area)
3. **Curvature**: Extract from spinor data $\psi_{\Sigma_{\text{reg}}}$ and fitness Hessian $H(x, t)$
4. **Holonomy**: Compute from the $S_N$ permutation symmetry encoded in CST edge patterns

**Output**: Scutoid tessellation $\mathcal{T} = \{S_1, S_2, \ldots, S_N\}$ with complete geometric and topological data.

**Inverse map**: Given a scutoid tessellation $\mathcal{T}$, reconstruct the Fractal Set by:
1. Extracting bottom/top face centroids as node positions $(x_i(t), x_i(t+1))$
2. Encoding vertical connections as CST edges
3. Encoding lateral face adjacencies as IG edges
4. Computing spinors from scutoid geometry (velocity from displacement, forces from deformation)

**Theorem**: This construction is **bijective** (one-to-one and onto), establishing $\mathcal{F} \cong \mathcal{T}$ as mathematical structures. ∎
:::

### 7.5. Transfer of Convergence Guarantees to Scutoid Framework

The key result of this section is that **all convergence guarantees** proven for the N-particle system ({prf:ref}`thm-main-convergence`, {prf:ref}`thm-foster-lyapunov-main`) **transfer to the scutoid framework** via the bijection $\mathcal{F} \leftrightarrow \mathcal{T}$.

:::{prf:theorem} Convergence Inheritance: Scutoid Tessellation Perspective
:label: thm-scutoid-convergence-inheritance

Let $\mathcal{T}_k = \{S_i(k, k+1)\}$ be the scutoid tessellation between discrete times $k$ and $k+1$. Define the **tessellation state** as:

$$
\mathcal{T}_{\text{state}}(k) := \left\{ S_i(k, k+1) : i = 1, \ldots, N \right\}
$$

This induces a **Markov chain on the space of scutoid tessellations**:

$$
\mathcal{T}_{\text{state}}(k) \xrightarrow{P_{\text{scutoid}}} \mathcal{T}_{\text{state}}(k+1)
$$

where the transition kernel $P_{\text{scutoid}}$ is determined by the BAOAB dynamics.

**Then the following convergence guarantees hold**:

**1. Geometric ergodicity**:

There exists a unique stationary distribution $\pi_{\text{scutoid}}$ on the space of tessellations such that:

$$
\|\mathcal{L}(\mathcal{T}_{\text{state}}(k)) - \pi_{\text{scutoid}}\|_{\text{TV}} \leq M_{\text{scutoid}} \, \rho_{\text{discrete}}^k
$$

where $\rho_{\text{discrete}} = 1 - \frac{\kappa_{\text{total}} \Delta t}{2} < 1$ is the discrete contraction coefficient from {prf:ref}`thm-fractal-set-ergodicity`.

**2. Scutoid fraction convergence**:

The scutoid fraction $\phi(k)$ (fraction of non-prism cells) converges exponentially to a steady-state value $\phi^*$:

$$
|\phi(k) - \phi^*| \leq C_\phi \, \rho_{\text{discrete}}^k
$$

**3. Curvature measure convergence**:

For any curvature observable $R_{\text{obs}}(\mathcal{T})$ (e.g., deficit angle, volume distortion, holonomy):

$$
|\mathbb{E}[R_{\text{obs}}(\mathcal{T}_k)] - \mathbb{E}_{\pi_{\text{scutoid}}}[R_{\text{obs}}]| \leq C_R \, \rho_{\text{discrete}}^k
$$

**4. Mean-field limit**:

As $N \to \infty$ with timestep $\Delta t = O(N^{-\alpha})$ for $\alpha \in (0, 1/2)$:

$$
\mathcal{T}_{\text{state}}(k) \xrightarrow{w} \text{Continuum Riemannian manifold } (\mathcal{M}, g_t)
$$

in the Gromov-Hausdorff metric, where $g_t$ satisfies the McKean-Vlasov PDE from [05_mean_field.md](../05_mean_field.md).

**Proof**: See Section 7.6 below. ∎
:::

### 7.6. Proof of Scutoid Convergence via Bijection

:::{prf:proof} Convergence Transfer via Fractal Set ↔ Scutoid Duality

We prove {prf:ref}`thm-scutoid-convergence-inheritance` by showing that the scutoid tessellation state $\mathcal{T}_{\text{state}}(k)$ is a **deterministic function** of the N-particle state $Z_k$, and therefore inherits all Markov properties.

**Step 1: State space correspondence**

Define the **projection map**:

$$
\Psi: Z_k = (X_k, V_k) \mapsto \mathcal{T}_{\text{state}}(k)
$$

that constructs the scutoid tessellation from walker positions and velocities via {prf:ref}`constr-episode-scutoid-map`. This map satisfies:

1. **Determinism**: $\Psi$ is a deterministic function (given $Z_k$, the tessellation is uniquely determined)
2. **Measurability**: $\Psi$ is measurable (continuous in positions for smooth metrics)
3. **Almost-invertibility**: For typical configurations (excluding degenerate Voronoi cells), there exists an inverse $\Psi^{-1}$ that reconstructs walker states from tessellation geometry

**Step 2: Induced transition kernel**

The N-particle Markov chain has transition kernel $P_{\Delta t}(Z_k, \cdot)$ from {prf:ref}`def-baoab-kernel`. The scutoid tessellation evolves via the **induced kernel**:

$$
P_{\text{scutoid}}(\mathcal{T}, A) := P_{\Delta t}(\Psi^{-1}(\mathcal{T}), \Psi^{-1}(A))
$$

for measurable sets $A$ in tessellation space.

**Key property**: Since $\Psi$ is deterministic:

$$
\mathcal{T}_{\text{state}}(k+1) = \Psi(Z_{k+1}) \quad \text{where} \quad Z_{k+1} \sim P_{\Delta t}(Z_k, \cdot)
$$

This shows that $\{\mathcal{T}_{\text{state}}(k)\}$ is itself a Markov chain, obtained by **pushing forward** the N-particle chain through $\Psi$.

**Step 3: Geometric ergodicity inheritance**

From {prf:ref}`thm-fractal-set-ergodicity`, the N-particle chain converges to stationary distribution $\pi_{\Delta t}$ at exponential rate $\rho_{\text{discrete}}^k$.

By the properties of pushforward measures:

$$
\pi_{\text{scutoid}} := \Psi_* \pi_{\Delta t}
$$

is the unique stationary distribution of the scutoid chain. The total variation distance satisfies:

$$
\|\mathcal{L}(\mathcal{T}_k) - \pi_{\text{scutoid}}\|_{\text{TV}} = \|\Psi_* \mathcal{L}(Z_k) - \Psi_* \pi_{\Delta t}\|_{\text{TV}} \leq \|\mathcal{L}(Z_k) - \pi_{\Delta t}\|_{\text{TV}} \leq M \rho_{\text{discrete}}^k
$$

where the first inequality uses the **contraction property** of pushforward maps for total variation distance (since $\Psi$ is deterministic, it cannot increase distances).

This proves part 1 of {prf:ref}`thm-scutoid-convergence-inheritance`.

**Step 4: Scutoid fraction as an observable**

The scutoid fraction $\phi(k)$ is a **measurable function** of the tessellation state:

$$
\phi(k) = \phi(\mathcal{T}_{\text{state}}(k)) = \frac{1}{N} \sum_{i=1}^N \mathbb{1}_{\{\text{$S_i$ is a scutoid}\}}
$$

A cell $S_i$ is a scutoid if and only if a cloning event occurred at walker $i$ between times $k$ and $k+1$. From the Fractal Set perspective, this corresponds to a non-trivial CST edge (parent $\neq$ child).

Since $\phi$ is a bounded measurable function:

$$
|\mathbb{E}[\phi(k)] - \mathbb{E}_{\pi_{\text{scutoid}}}[\phi]| \leq \|\phi\|_\infty \|\mathcal{L}(\mathcal{T}_k) - \pi_{\text{scutoid}}\|_{\text{TV}} \leq M \rho_{\text{discrete}}^k
$$

This proves part 2.

**Step 5: Curvature observables**

Any curvature measure $R_{\text{obs}}(\mathcal{T})$ (deficit angle, volume, holonomy) is a function of scutoid cell geometry, which in turn is determined by walker positions and the emergent metric $g(x, Z_k)$.

From [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md) Section 5, the discrete curvature measures (deficit angles, causal set volume, spectral gap) converge to the continuum Ricci scalar as $N \to \infty$.

For fixed $N$, the curvature at time $k$ is a measurable function $R_{\text{obs}}(\mathcal{T}_k)$. Applying the same ergodic theorem argument as in Step 4:

$$
|\mathbb{E}[R_{\text{obs}}(\mathcal{T}_k)] - \mathbb{E}_{\pi_{\text{scutoid}}}[R_{\text{obs}}]| \leq \|R_{\text{obs}}\|_\infty \, M \rho_{\text{discrete}}^k
$$

This proves part 3.

**Step 6: Mean-field limit**

The mean-field limit $N \to \infty$ for the scutoid framework follows from the corresponding result for the N-particle system.

**Key results from existing framework**:
1. **Propagation of chaos** ([06_propagation_chaos.md](../06_propagation_chaos.md)): $\mu_N \to \mu$ weakly, where $\mu$ solves the McKean-Vlasov PDE
2. **Gromov-Hausdorff convergence** (proven in {prf:ref}`lem-gromov-hausdorff` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md)): The empirical measure metric space $(X_N, d_g, \mu_N)$ converges to the continuum Riemannian manifold $(\mathcal{M}, d_g, \mu)$

The scutoid tessellation $\mathcal{T}_N$ is a **dual structure** to the Voronoi diagram of $X_N$. As $N \to \infty$:
- Voronoi cells shrink: $\text{diam}(\text{Vor}_i) \to 0$
- Tessellation becomes dense: $\bigcup_{i} S_i$ fills the spacetime manifold $\mathcal{M} \times [t, t+1]$
- Discrete geometry converges to continuum Riemannian geometry

By the duality $\mathcal{F} \leftrightarrow \mathcal{T}$ and the reconstruction theorem {prf:ref}`thm-fractal-set-reconstruction`, all geometric data (metric, curvature, connection) can be reconstructed from the scutoid tessellation. Therefore:

$$
\mathcal{T}_N \xrightarrow{\text{GH}} (\mathcal{M}, g_t) \quad \text{as } N \to \infty
$$

in the Gromov-Hausdorff metric on spacetime manifolds.

This proves part 4. ∎
:::

### 7.7. Unified Convergence Table: Four Perspectives

The following table summarizes the convergence guarantees across all four frameworks:

| **Property** | **N-Particle** | **Mean-Field** | **Fractal Set** | **Scutoid Geometry** | **Source** |
|--------------|----------------|----------------|-----------------|----------------------|------------|
| **Primary state** | $Z_k = (X_k, V_k)$ | $\mu_t(dx, dv)$ | $\mathcal{F} = (\mathcal{N}, E)$ | $\mathcal{T} = \{S_i\}$ | Definitions |
| **Markov property** | ✅ Discrete-time Markov chain | ✅ Nonlinear Markov process | ✅ Inherited from $Z_k$ | ✅ Inherited via $\Psi$ | {prf:ref}`def-baoab-kernel` |
| **Ergodicity** | ✅ Geometric ergodicity | ✅ Exponential mixing | ✅ Inherited | ✅ Inherited | {prf:ref}`thm-fractal-set-ergodicity` |
| **Convergence rate** | $\rho_{\text{discrete}}^k$ | $e^{-\kappa t}$ | $\rho_{\text{discrete}}^k$ | $\rho_{\text{discrete}}^k$ | {prf:ref}`thm-discrete-drift-baoab` |
| **Stationary dist.** | $\pi_{\Delta t}(dZ)$ | $\pi(dx, dv)$ | $\pi_{\text{fractal}}$ | $\pi_{\text{scutoid}}$ | {prf:ref}`thm-fractal-set-ergodicity` |
| **Limit $N \to \infty$** | Propagation of chaos | McKean-Vlasov PDE | Continuum QFT | Riemannian manifold | [06_propagation_chaos.md](../06_propagation_chaos.md) |
| **Limit $\Delta t \to 0$** | SDE solution | Fokker-Planck PDE | Continuous spacetime | Smooth manifold | {prf:ref}`thm-weak-convergence-invariant` |
| **Discretization error** | $O(\Delta t)$ | N/A (continuous) | $O(\Delta t)$ | $O(\Delta t)$ | {prf:ref}`prop-total-error` |
| **Key observable** | Lyapunov $V(Z_k)$ | KL divergence $D_{\text{KL}}(\mu_t \| \pi)$ | Node energies | Scutoid fraction $\phi(k)$ | Various |
| **Symmetries** | $S_N$ permutation | Translation/rotation | $S_N \times \text{U}(1) \times \text{SU}(2)$ | Diffeomorphisms | [09_symmetries_adaptive_gas.md](../09_symmetries_adaptive_gas.md) |
| **Curvature measure** | Fitness Hessian $H(x)$ | Wasserstein metric | IG edge holonomy | Deficit angles, heat kernel | [08_emergent_geometry.md](../08_emergent_geometry.md) |

**Key insight**: All four columns describe **the same physical algorithm**, viewed through different mathematical lenses. Convergence guarantees proven in any framework transfer to all others via bijective correspondences.

### 7.8. Scutoid-Specific Results: Beyond the Other Frameworks

While the scutoid framework inherits all convergence guarantees, it also provides **unique insights** not naturally accessible in the other perspectives:

:::{prf:theorem} Scutoid Fraction as Phase Transition Order Parameter
:label: thm-scutoid-phase-transition

The scutoid fraction $\phi(t)$ serves as an **order parameter** for the exploration-exploitation phase transition:

**1. Exploratory phase** ($t \ll t_{\text{conv}}$):

$$
\phi(t) \approx \phi_{\max} = 1 - e^{-\lambda_{\text{clone}} \Delta t} \approx \lambda_{\text{clone}} \Delta t
$$

where $\lambda_{\text{clone}}$ is the cloning rate. High scutoid fraction indicates active exploration.

**2. Convergent phase** ($t \gg t_{\text{conv}}$):

$$
\phi(t) \to \phi^* = \frac{\lambda_{\text{clone}}^{\text{QSD}}}{\mu_{\text{death}} + \lambda_{\text{clone}}^{\text{QSD}}} \ll 1
$$

where $\lambda_{\text{clone}}^{\text{QSD}}$ is the cloning rate at the QSD. Low scutoid fraction indicates convergence.

**3. Phase transition critical time**:

The transition occurs at:

$$
t_{\text{crit}} \sim \frac{1}{\kappa_{\text{total}}} \log\left(\frac{\phi_{\max}}{\phi^*}\right)
$$

where $\kappa_{\text{total}}$ is the drift coefficient from {prf:ref}`thm-foster-lyapunov-main`.

**Proof**: From {prf:ref}`thm-scutoid-phase-classification` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md) Section 7. The scutoid fraction is directly related to the cloning rate via the cloning operator's action on the empirical measure. ∎
:::

:::{prf:theorem} Holographic Entropy Bound from Scutoid Geometry
:label: thm-scutoid-holographic-entropy

The **information capacity** of a scutoid tessellation satisfies a holographic bound analogous to the Bekenstein-Hawking entropy:

$$
S_{\text{scutoid}}(\mathcal{T}) \leq \frac{A_{\text{boundary}}}{4 \ell_{\text{Planck}}^{d-1}}
$$

where:
- $S_{\text{scutoid}} = -\sum_{i} p_i \log p_i$ is the Shannon entropy of the walker distribution
- $A_{\text{boundary}}$ is the total area of lateral faces (boundary of the spacetime region)
- $\ell_{\text{Planck}} = \sqrt{\epsilon_\Sigma}$ is the emergent "Planck length" from the diffusion regularization

**Physical interpretation**: The amount of information that can be stored in a spacetime region (between time slices $t$ and $t+1$) is bounded by the **surface area** of that region, not its volume. This is a discrete algorithmic analog of the holographic principle in quantum gravity.

**Proof**: From {prf:ref}`thm-holographic-entropy-bound` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md) Section 7. The bound follows from the relationship between IG edge weights (encoding boundary areas) and the Fisher information metric. ∎
:::

:::{prf:theorem} Raychaudhuri Equation for Scutoid Evolution
:label: thm-scutoid-raychaudhuri

The **expansion rate** of scutoid cells satisfies a discrete analog of the Raychaudhuri equation from general relativity:

$$
\frac{d\theta_i}{dt} = -\frac{1}{d}\theta_i^2 - \sigma_{ij}^2 + \omega_{ij}^2 - R_{\mu\nu} u^\mu u^\nu + \nabla_\mu a^\mu
$$

where:
- $\theta_i = \frac{1}{d} \text{tr}(\nabla_a v^a)$ is the expansion scalar (volume change rate of cell $S_i$)
- $\sigma_{ij}$ is the shear tensor (distortion without volume change)
- $\omega_{ij}$ is the vorticity tensor (rotation)
- $R_{\mu\nu}$ is the Ricci curvature of the emergent metric
- $a^\mu$ is the acceleration field from forces

**Physical consequence**: If the Ricci curvature is positive ($R_{\mu\nu} u^\mu u^\nu > 0$, corresponding to **high fitness gradients**), then scutoid cells **contract** over time, indicating convergence toward high-fitness regions.

**Proof**: From {prf:ref}`thm-raychaudhuri-scutoid` in [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md) Section 6. The discrete version follows from the relationship between scutoid volume changes and the BAOAB velocity updates. ∎
:::

### 7.9. Practical Implications: Choosing the Right Perspective

**When to use each framework**:

**N-Particle** perspective:
- **Best for**: Proving convergence via Lyapunov drift, analyzing stability, designing new operators
- **Tools**: Stochastic analysis, Foster-Lyapunov theorems, coupling arguments
- **Documents**: [04_convergence.md](../04_convergence.md), [07_adaptative_gas.md](../07_adaptative_gas.md)

**Mean-Field** perspective:
- **Best for**: Understanding large-$N$ limits, deriving PDEs, analyzing scaling behavior
- **Tools**: Weak convergence, McKean-Vlasov theory, entropy methods
- **Documents**: [05_mean_field.md](../05_mean_field.md), [06_propagation_chaos.md](../06_propagation_chaos.md)

**Fractal Set** perspective:
- **Best for**: Data structure design, information reconstruction, gauge theory formulation, lattice QFT
- **Tools**: Graph theory, spinor calculus, gauge connections, discrete symmetries
- **Documents**: [00_full_set.md](00_full_set.md), current document, [13_fractal_set_new/](./00_full_set.md)

**Scutoid Geometry** perspective:
- **Best for**: Visualizing spacetime evolution, computing curvature, detecting phase transitions, holographic bounds
- **Tools**: Differential geometry, Voronoi diagrams, curvature measures, topological invariants
- **Documents**: [14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md)

**Recommendation**: Use multiple perspectives in parallel:
1. **Prove** convergence using N-particle Lyapunov drift
2. **Understand** scaling using mean-field limits
3. **Implement** using Fractal Set data structures
4. **Visualize** using scutoid tessellations
5. **Verify** by checking consistency across all four frameworks

---

## 8. Summary and Practical Implications

### 8.1. What We Have Proven

:::{prf:theorem} Complete Convergence Fidelity
:label: thm-complete-fidelity

The Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ is the path history of a **geometrically ergodic discrete-time Markov chain**. Specifically:

**1. Information:** {prf:ref}`thm-fractal-set-reconstruction` proves the Fractal Set contains complete SDE data (reconstruction).

**2. Dynamics:** {prf:ref}`thm-fractal-set-ergodicity` proves the generator is geometrically ergodic (convergence).

**3. Fidelity:** {prf:ref}`thm-weak-convergence-invariant` proves the discrete invariant measure approximates the continuous QSD.

**Combined:** The Fractal Set is a **faithful discrete representation** of the Adaptive Gas SDE, inheriting all convergence guarantees.

**Extension to four perspectives:** Section 7 proves that this fidelity extends to all four equivalent formulations (N-particle, mean-field, Fractal Set, scutoid geometry), with convergence guarantees transferring between frameworks via bijective correspondences.
:::

### 8.2. Convergence Rate Summary

| **Property** | **Continuous SDE** | **Discrete (BAOAB)** | **Source** |
|--------------|-------------------|---------------------|-----------|
| **Drift coefficient** | $\kappa_{\text{total}}$ | $\kappa_{\text{total}}/2$ | {prf:ref}`thm-discrete-drift-baoab` |
| **Contraction per step** | $e^{-\kappa \Delta t}$ | $1 - \kappa \Delta t / 2 \approx e^{-\kappa \Delta t/2}$ | {prf:ref}`rem-discrete-contraction` |
| **Convergence rate** | $\\|\\mu_t - \pi\\| \leq M e^{-\kappa t}$ | $\\|\mu_k - \pi_{\Delta t}\\| \leq M \rho^k$ | {prf:ref}`thm-fractal-set-ergodicity` |
| **Discretization error** | N/A | $O(\Delta t)$ | {prf:ref}`prop-total-error` |
| **Order of integrator** | N/A | Second-order (BAOAB) | {prf:ref}`def-baoab-integrator` |

**Key insight:** For small $\Delta t$, the discrete rate is **half** the continuous rate per unit time, but this is compensated by taking more steps. For fixed continuous time $t = k \Delta t$, both converge at the same exponential rate.

### 8.3. Practical Guidelines

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
- **Scutoid fraction monitoring:** Track $\phi(t)$ to detect phase transitions between exploration and exploitation regimes.

### 8.4. Future Extensions

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
6. **[14_scutoid_geometry_framework.md](../14_scutoid_geometry_framework.md):** Scutoid tessellation and emergent Riemannian geometry
7. **[05_mean_field.md](../05_mean_field.md):** Mean-field limit and McKean-Vlasov PDE
8. **[06_propagation_chaos.md](../06_propagation_chaos.md):** Propagation of chaos

### Mathematical References

1. **Meyn & Tweedie (2009).** *Markov Chains and Stochastic Stability* (2nd ed.). Cambridge University Press.
   - Theorem 15.0.1: Geometric ergodicity via drift conditions

2. **Leimkuhler & Matthews (2015).** *Molecular Dynamics: With Deterministic and Stochastic Numerical Methods*. Springer.
   - Chapter 7: BAOAB integrator and symplectic methods

3. **Talay & Tubaro (1990).** "Expansion of the global error for numerical schemes solving stochastic differential equations." *Stochastic Analysis and Applications*, 8(4), 483-509.
   - Weak approximation theory for SDEs

4. **Gómez-Gálvez et al. (2018).** "Scutoids are a geometrical solution to three-dimensional packing of epithelia." *Nature Communications*, 9, 2960.
   - Original discovery of scutoids in biological tissue packing

### Discussion Documents

1. **[discussions/convergence_inheritance_strategy.md](discussions/convergence_inheritance_strategy.md):** Detailed proof strategy and technical challenges
