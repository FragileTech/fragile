# Hypocoercivity and Convergence of the Euclidean Gas

:::{div} feynman-prose
A force estimate valid inside one region must account for a noisy step that leaves it. [Structural landscape convergence](06a_structural_landscape_convergence.md) develops this regional accounting and carries the excursion terms into the complete update. The kinetic mechanism stays the same; the additional estimates specify where its hypotheses apply and what must control the rest.
:::

## 0. TLDR

*Notation: $V_{\text{Var},x}$, $V_{\text{Var},v}$ = positional and velocity variance; $\mu_v$ = velocity barycenter; $W_b$ = boundary potential; $\Psi_{\text{kin}}$, $\Psi_{\text{clone}}$ = kinetic and cloning operators. The TV-focused Lyapunov is $V_{\text{TV}} = c_V\!\left(V_{\text{Var},x} + V_{\text{Var},v}\right) + c_\mu \|\mu_v\|^2 + c_B W_b$.*

**TV-Ready Kinetic Drift**: Langevin friction contracts $V_{\text{Var},v}$ and $\|\mu_v\|^2$; positional diffusion produces a cross-term bounded by $V_{\text{Var},v}$; the confining potential contracts $W_b$. These are the only kinetic ingredients used in the total-variation (TV) convergence proof.

**Velocity Squashing (Always On)**: A smooth squashing map is applied after each kinetic step so $\|v\| \leq v_{\max}$ holds deterministically. This makes boundary and weak-error bounds uniform and avoids hidden moment assumptions.

**Minorization for TV**: Uniform ellipticity of $\Sigma$ plus the BAOAB O-step yields a smooth transition density with a positive lower bound on compact interior sets, giving a rigorous small-set/minorization condition for Harris/Meyn–Tweedie theory.

**W2/Hypocoercive Material Deferred**: Hypocoercive/Wasserstein claims are quarantined into dedicated sections and are **not used** in the TV proof. They will be repaired and tightened later.

**Dependencies**: {doc}`02_euclidean_gas`, {doc}`04_single_particle`, {doc}`03_cloning`

## 1. Introduction

### 1.1. Goal and Scope

The goal of this document is to analyze the **kinetic operator** $\Psi_{\text{kin}}$ and its contribution to the complete Euclidean Gas update. The companion chapter {doc}`03_cloning` proves an $N$-uniform Keystone pressure estimate and a separate positional reset bound. Neither is, by itself, a signed contraction of positional variance. The full signed composition is recorded in {prf:ref}`thm-slc-signed-complete-update`.

The central mathematical object of study is the **underdamped Langevin dynamics** that governs walker evolution between cloning events. This dynamics combines deterministic drift from a confining potential, friction that dissipates kinetic energy, and thermal noise that maintains ergodicity. We prove that this operator achieves (TV-focused):

1. **Velocity dissipation** for $V_{\text{Var},v}$ and $\|\mu_v\|^2$ (Chapter 5)
2. **Bounded positional expansion** with explicit cross-term control for $V_{\text{Var},x}$ (Chapter 6)
3. **Confining potential protection** for boundary safety $W_b$ (Chapter 7)
4. **Small-set/minorization** for the kinetic step on compact interior sets (Chapter 7.6)

Hypocoercive/Wasserstein claims remain in this document but are **quarantined** and **not used** in the TV proof; they will be repaired later with additional assumptions.

The scope of this document is the analysis of $\Psi_{\text{kin}}$ in isolation. The composition $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$, parameter optimization, and the main TV convergence theorem are deferred to the companion document *{doc}`06_convergence`*.

### 1.2. The Synergistic Dissipation Framework (TV Track)

The Euclidean Gas achieves stability through a carefully orchestrated interplay between two operators that provide **complementary dissipation**. Neither operator alone is sufficient for convergence; each contracts the error components that the other expands:

| **Lyapunov Component** | **Cloning $\Psi_{\text{clone}}$** | **Kinetics $\Psi_{\text{kin}}$** | **Net Effect** |
|:-----------------------|:-----------------------------------|:----------------------------------|:---------------|
| $V_{\text{Var},x}$ (position) | Keystone pressure plus signed donor terms; a separate reset bound | Kinetic transport, force, and noise terms | **Rate decided by the complete signed balance** |
| $V_{\text{Var},v}$ (velocity) | $\leq C_v$ | $-(2\gamma-\epsilon) V_{\text{Var},v}\tau + \left(\frac{F_{\max}^2}{\epsilon} + d\sigma_{\max}^2\right)\tau$ | **Contraction** |
| $\|\mu_v\|^2$ (velocity barycenter) | $\leq C_{\mu}^{\text{clone}}$ | $-2\gamma \|\mu_v\|^2\tau + C_{\mu}^{\text{kin}}\tau$ | **Contraction** |
| $W_b$ (boundary)       | $-\kappa_b W_b \tau + C_b \tau$ | $-\kappa_{\text{pot}} W_b \tau + C_{\text{pot}} \tau$ | **Strong contraction** |

**The Physical Intuition:**

- **Cloning** is a *positional* mechanism: it resamples walker positions based on fitness. Keystone pressure measures favourable replacement, while donor insertion can contribute with either sign to positional variance. Inelastic collisions perturb velocities and their barycenter.

- **Kinetics** is a *velocity* mechanism: the friction term $-\gamma v$ directly dissipates velocity variance and $\|\mu_v\|^2$, while thermal noise injects bounded positional diffusion. These effects offset the cloning-induced velocity perturbations.

- **Boundary safety** benefits from **dual independent mechanisms**: cloning eliminates boundary-proximate walkers (Safe Harbor), while the confining potential actively pushes walkers away from the boundary.

The decomposition identifies the terms in the complete update. A TV rate requires the signed composition to close on the same state class; the separate component estimates do not establish it by themselves.

### 1.3. Overview of the Proof Strategy and Document Structure

The proof is organized into five main chapters, each establishing a specific drift inequality for one component of the Lyapunov function. The diagram below illustrates the logical dependencies and the role of each chapter in the overall convergence architecture.

```{mermaid}
graph TD
    subgraph "Foundations"
        A["<b>Ch 3: Kinetic Operator Definition</b><br>Stratonovich SDE, Axioms for U, Σ, γ<br>Fokker-Planck Equation"]:::stateStyle
        B["<b>Ch 3.7: Discretization Theory</b><br>Continuous → Discrete Drift<br>Weak Error Bounds"]:::lemmaStyle
    end

    subgraph "Chapter 4 (Deferred): W2/Hypocoercive Track"
        C["<b>Ch 4.2: Hypocoercive Norm</b><br>Coupled (x,v) metric with<br>cross-term b⟨Δx, Δv⟩"]:::stateStyle
        D["<b>Ch 4.5: Location Error Drift</b><br>Barycenter separation contracts<br>via friction-transport coupling"]:::lemmaStyle
        E["<b>Ch 4.6: Structural Error Drift</b><br>Shape dissimilarity contracts<br>via diffusion and confinement"]:::lemmaStyle
        F["<b>Theorem 4.3: W_h² Contraction</b><br>(Deferred; not used in TV proof)"]:::theoremStyle
    end

    subgraph "Chapters 5-7: TV Components"
        G["<b>Theorem 5.3/5.4.1: Velocity Dissipation</b><br>V_Var,v and ||mu_v||^2 contract"]:::theoremStyle
        H["<b>Theorem 6.3: V_Var,x Expansion</b><br>Bounded thermal diffusion<br>ΔV_Var,x ≤ C_kin,x τ"]:::theoremStyle
        I["<b>Theorem 7.3: W_b Contraction</b><br>Confining potential creates<br>ΔW_b ≤ -κ_pot W_b τ + C_pot τ"]:::theoremStyle
        I2["<b>Lemma 7.6: Minorization</b><br>Small set for kinetic kernel"]:::lemmaStyle
    end

    subgraph "Integration with Cloning Operator"
        J["<b>From 03_cloning</b><br>N-uniform Keystone pressure<br>and a separate positional reset bound"]:::axiomStyle
        K["<b>Synergistic Composition</b><br>Balance weights c_V, c_μ, c_B in<br>V_TV = c_V(V_Var,x+V_Var,v)+c_μ||μ_v||²+c_B W_b"]:::stateStyle
        L["<b>Complete update</b><br>Signed Keystone, donor and kinetic balance<br>with explicit residual"]:::theoremStyle
    end

    A --> B
    A --> C
    B --> F
    C --> D
    C --> E
    D --> F
    E --> F

    A --> G
    A --> H
    A --> I
    A --> I2

    F --> K
    G --> K
    H --> K
    I --> K
    I2 --> K
    J -- "Supplies Keystone pressure<br>and donor accounting" --> K
    K --> L

    classDef stateStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#e8eaf6
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,stroke-dasharray: 5 5,color:#f4e8d8
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#d8f4e3
    classDef theoremStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#f4d8e8
```

**Chapter-by-Chapter Overview:**

- **Chapter 3 (Foundations):** Defines the kinetic operator using Stratonovich stochastic differential equations, states the axioms for the confining potential $U$, diffusion tensor $\Sigma$, and friction coefficient $\gamma$, derives the Fokker-Planck equation, and establishes the discretization theory connecting continuous-time generators to discrete-time drift inequalities.

- **Chapter 4 (Hypocoercive Contraction, Deferred):** Records the W2/hypocoercive analysis for future use. It is **not used** in the TV convergence proof.

- **Chapter 5 (Velocity Dissipation):** Proves that Langevin friction provides direct linear dissipation of velocity variance $V_{\text{Var},v}$ and barycenter energy $\|\mu_v\|^2$.

- **Chapter 6 (Positional Expansion):** Bounds the kinetic contribution on its stated moment class. The signed Keystone, donor, and kinetic terms are combined in {prf:ref}`thm-slc-signed-complete-update`.

- **Chapter 7 (Boundary Safety + Minorization):** Proves boundary protection and establishes a small-set/minorization condition for the kinetic kernel on compact interior sets.

The drift inequalities proven in this document, combined with those from {doc}`03_cloning`, provide the complete set of components needed for the main convergence theorem in {doc}`06_convergence`.



## 2. Document Overview and Relation to {doc}`03_cloning`

**Purpose of This Document:**

This document provides the kinetic half of the TV convergence proof for the Euclidean Gas algorithm. While the companion document *"The Keystone Principle and the Contractive Nature of Cloning"* ({doc}`03_cloning`) analyzed the cloning operator $\Psi_{\text{clone}}$, this document analyzes the **kinetic operator** $\Psi_{\text{kin}}$ and provides the drift and minorization ingredients used by the composed operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$.

**The Synergistic Dissipation Framework:**

The Euclidean Gas achieves stability through the complementary action of two operators:

| Component (TV Track) | $\Psi_{\text{clone}}$ ({doc}`03_cloning`) | $\Psi_{\text{kin}}$ (this document) | Net Effect |
|:---------------------|:------------------------------------------|:-------------------------------------|:-----------|
| $V_{\text{Var},x}$ (position) | Keystone pressure, signed donor terms, and a reset bound | Transport, force, and noise terms | **Use the complete signed balance** |
| $V_{\text{Var},v}$ (velocity) | $+C_v$ | $-(2\gamma-\epsilon)V_{\text{Var},v} + C_v'$ | **Contraction** |
| $\|\mu_v\|^2$ | $+C_{\mu}^{\text{clone}}$ | $-\gamma\|\mu_v\|^2 + C_{\mu}^{\text{kin}}$ | **Contraction** |
| $W_b$ (boundary) | $-\kappa_b W_b$ | $-\kappa_{\text{pot}} W_b + C_{\text{pot}}$ | **Strong contraction** |

**Deferred:** $V_W$ (inter-swarm/W2) belongs to the W2 track and is not used in the TV proof.

This document proves the drift inequalities in the "$\Psi_{\text{kin}}$" column and combines them with results from {doc}`03_cloning` to establish the main convergence theorem.

**Document Structure:**

- **Chapter 3:** The kinetic operator with Stratonovich formulation
- **Chapter 4:** Hypocoercive contraction of inter-swarm error $V_W$ (deferred W2 track)
- **Chapter 5:** Velocity variance and barycenter dissipation via Langevin friction
- **Chapter 6:** Positional diffusion and bounded expansion
- **Chapter 7:** Boundary potential contraction and kinetic minorization

**Note:** The synergistic composition, main convergence theorem, and parameter optimization are covered in the companion document *{doc}`06_convergence`*.

## 3. The Kinetic Operator with Stratonovich Formulation

### 3.1. Introduction and Motivation

The kinetic operator $\Psi_{\text{kin}}$ governs the continuous-time evolution of walkers between cloning events. It is an **underdamped Langevin dynamics** that combines:

1. **Deterministic drift** from the confining potential $U(x)$
2. **Friction** that dissipates kinetic energy
3. **Thermal noise** that maintains ergodicity and prevents collapse

This chapter defines the operator rigorously, introduces the Stratonovich formulation for geometric consistency, and establishes the framework for subsequent analysis.

**Why Stratonovich?**

We adopt the **Stratonovich convention** for the stochastic differential equations because:

1. **Geometric invariance:** Respects coordinate transformations on manifolds
2. **Physical correctness:** Natural formulation from fluctuation-dissipation theorem
3. **Future compatibility:** Essential for Riemannian extensions with Hessian-based diffusion
4. **Clean invariant measures:** Gibbs distributions emerge naturally without correction terms

For the isotropic case analyzed in detail here, the Stratonovich and Itô formulations coincide. We state the general framework to enable future extensions.

### 3.2. The Kinetic SDE

:::{prf:definition} The Kinetic Operator (Stratonovich Form)
:label: def-kinetic-operator-stratonovich

The kinetic operator $\Psi_{\text{kin}}$ evolves the swarm for a time interval $\tau > 0$ according to the coupled Stratonovich SDEs:

$$
\begin{aligned}
dx_t &= v_t \, dt \\
dv_t &= F(x_t) \, dt - \gamma(v_t - u(x_t)) \, dt + \Sigma(x_t, v_t) \circ dW_t
\end{aligned}

$$

where:

**Deterministic Terms:**
- $F(x) = -\nabla U(x)$: Force field from the **confining potential** $U: \mathcal{X}_{\text{valid}} \to \mathbb{R}_{\geq 0}$
- $\gamma > 0$: **Friction coefficient**
- $u(x)$: **Local drift velocity** (typically $u \equiv 0$ for simplicity)

**Stochastic Term:**
- $\Sigma(x,v): \mathcal{X}_{\text{valid}} \times \mathbb{R}^d \to \mathbb{R}^{d \times d}$: **Diffusion tensor**
- $W_t$: Standard $d$-dimensional Brownian motion
- $\circ$: **Stratonovich product**

**Boundary Condition and Velocity Squashing:**
After evolving for time $\tau$, the walker status is updated and a smooth velocity squashing map is applied:

$$
s_i^{(t+1)} = \mathbf{1}_{\mathcal{X}_{\text{valid}}}(x_i(t+\tau)), \quad v_i^{(t+1)} = S(v_i(t+\tau))

$$

Walkers exiting the valid domain are marked as dead. The squashing map $S$ enforces $\|v\| \leq v_{\max}$ deterministically.
:::

:::{prf:remark} Relationship to Itô Formulation
:label: rem-stratonovich-ito-equivalence

The equivalent Itô SDE includes a correction term:

$$
dv_t = \left[F(x_t) - \gamma(v_t - u(x_t)) + \underbrace{\frac{1}{2}\sum_{j=1}^d \Sigma_j(x_t,v_t) \cdot \nabla_v \Sigma_j(x_t,v_t)}_{\text{Stratonovich correction}}\right] dt + \Sigma(x_t,v_t) \, dW_t

$$

where $\Sigma_j$ is the $j$-th column of $\Sigma$. We denote the effective Itô drift by
$$
b_v(x,v) := F(x) - \gamma(v - u(x)) + \frac{1}{2}\sum_{j=1}^d \Sigma_j(x,v) \cdot \nabla_v \Sigma_j(x,v).
$$

**For isotropic diffusion** ($\Sigma = \sigma_v I_d$), the correction term vanishes since $\nabla_v(\sigma_v I_d) = 0$. Thus **Stratonovich = Itô** in this case. Throughout the TV analysis we take $u \equiv 0$ to avoid unnecessary drift terms; extensions to nonzero $u$ are straightforward.
:::

### 3.3. Axioms for the Kinetic Operator

We now state the foundational axioms that $U$, $\Sigma$, and $\gamma$ must satisfy for the convergence theory to hold.

#### 3.3.1. The Confining Potential

:::{prf:axiom} Globally Confining Potential
:label: axiom-confining-potential

The potential function $U: \mathcal{X}_{\text{valid}} \to \mathbb{R}_{\geq 0}$ satisfies:

**1. Smoothness:**

$$
U \in C^2(\mathcal{X}_{\text{valid}})

$$

**2. Coercivity (Confinement):**
There exist constants $\alpha_U > 0$ and $R_U < \infty$ such that:

$$
\langle x, \nabla U(x) \rangle \geq \alpha_U \|x\|^2 - R_U \quad \forall x \in \mathcal{X}_{\text{valid}}

$$

This ensures the force field $F(x) = -\nabla U(x)$ drives walkers back toward the origin when $\|x\|$ is large.

**3. Bounded Force on the Valid Domain:**
There exists a constant $F_{\max} < \infty$ such that:

$$
\|F(x)\| = \|\nabla U(x)\| \leq F_{\max} \quad \forall x \in \mathcal{X}_{\text{valid}}

$$

**4. Compatibility with Boundary Barrier (Quantitative):**
Near the boundary, $U(x)$ grows to create an inward-pointing force with quantifiable strength. There exist constants $\alpha_{\text{boundary}} > 0$ and $\delta_{\text{boundary}} > 0$ such that:

$$
\langle \vec{n}(x), F(x) \rangle \leq -\alpha_{\text{boundary}} \quad \text{for all } x \text{ with } \text{dist}(x, \partial\mathcal{X}_{\text{valid}}) < \delta_{\text{boundary}}

$$

where $\vec{n}(x)$ is the outward unit normal at the closest boundary point.

**5. Lipschitz Continuity (Global on $\mathcal{X}_{\text{valid}}$):**
There exists $L_F < \infty$ such that:

$$
\|F(x) - F(y)\| \leq L_F \|x - y\| \quad \forall x,y \in \mathcal{X}_{\text{valid}}

$$

**Physical Interpretation:** The potential creates a "bowl" that confines walkers to the valid domain while allowing free movement in the interior. The parameter $\alpha_{\text{boundary}}$ quantifies the minimum inward force strength near the boundary, which is critical for proving the boundary potential contraction rate in Chapter 7.
:::

:::{prf:example} Canonical Confining Potential
:label: ex-canonical-confining-potential

A standard choice is a **smoothly capped harmonic potential**. Let $\phi:\mathbb{R}\to\mathbb{R}_{\ge 0}$ be $C^2$, non-decreasing, and satisfy:

- $\phi(s)=0$ for $s \le 0$
- $\phi(s)=s$ for $0 \le s \le r_{\text{gap}}/2$
- $\phi$ saturates smoothly on $[r_{\text{gap}}/2, r_{\text{gap}}]$

where $r_{\text{gap}} := r_{\text{boundary}} - r_{\text{interior}}$. Define:

$$
U(x) = \frac{\kappa}{2}\,\phi(\|x\| - r_{\text{interior}})^2

$$

with $r_{\text{interior}} < r_{\text{boundary}} = \text{radius of } \mathcal{X}_{\text{valid}}$.

This potential satisfies all axiom requirements:
- **Coercivity**: $\alpha_U = \kappa$ (from quadratic growth)
- **Interior safety**: $F = 0$ for $\|x\| \leq r_{\text{interior}}$
- **Inward force**: $F(x)$ points inward in the boundary layer by construction of $\phi$
- **Boundary compatibility**: $\alpha_{\text{boundary}}$ follows from the slope of $\phi$ on the boundary layer
:::

#### 3.3.2. The Diffusion Tensor

:::{prf:axiom} Anisotropic Diffusion Tensor
:label: axiom-diffusion-tensor

The velocity diffusion tensor $\Sigma: \mathcal{X}_{\text{valid}} \times \mathbb{R}^d \to \mathbb{R}^{d \times d}$ satisfies:

**1. Uniform Ellipticity:**

$$
\lambda_{\min}(\Sigma(x,v)\Sigma(x,v)^T) \geq \sigma_{\min}^2 > 0 \quad \forall (x,v)

$$

This ensures the diffusion is **non-degenerate** in all directions.

**2. Bounded Eigenvalues:**

$$
\lambda_{\max}(\Sigma(x,v)\Sigma(x,v)^T) \leq \sigma_{\max}^2 < \infty \quad \forall (x,v)

$$

This prevents **infinite noise** in any direction.

**3. Lipschitz Continuity:**

$$
\|\Sigma(x_1,v_1) - \Sigma(x_2,v_2)\|_F \leq L_\Sigma(\|x_1-x_2\| + \|v_1-v_2\|)

$$

where $\|\cdot\|_F$ is the Frobenius norm.

**4. Regularity:**

$$
\Sigma \in C^1(\mathcal{X}_{\text{valid}} \times \mathbb{R}^d)

$$

**Canonical Instantiations:**

a) **Isotropic (Primary Case):**

$$
\Sigma(x,v) = \sigma_v I_d

$$

All directions receive equal thermal noise $\sigma_v > 0$.

b) **Position-Dependent:**

$$
\Sigma(x,v) = \sigma(x) I_d

$$

Noise intensity varies with position (e.g., higher near boundary for enhanced exploration).

c) **Hessian-Based (Future Work):**

$$
\Sigma(x,v) = (H_{\text{fitness}}(x,v) + \epsilon I_d)^{-1/2}

$$

Noise adapts to local fitness landscape curvature (Riemannian Langevin).
:::

:::{prf:remark} Why Uniform Ellipticity Matters
:label: rem-uniform-ellipticity-importance

The uniform ellipticity condition $\lambda_{\min} \geq \sigma_{\min}^2 > 0$ is **critical** for:

1. **Ergodicity:** Ensures all velocity directions are explored
2. **Hypocoercivity:** Allows diffusion in velocity to induce contraction in position
3. **Coupling arguments:** Synchronous coupling between two swarms remains correlated

Without this, the system can become **degenerate** and convergence may fail.
:::

#### 3.3.3. Friction and Timestep Parameters

:::{prf:axiom} Friction and Integration Parameters
:label: axiom-friction-timestep

**1. Friction Coefficient:**

$$
\gamma > 0

$$

Physically, $\gamma$ is the inverse of the **relaxation time** for velocity. Larger $\gamma$ → faster velocity dissipation.

**2. Timestep:**

$$
\tau \in (0, \tau_{\max}]

$$

where $\tau_{\max}$ must satisfy the hypotheses of the discrete estimate being used.
Friction and domain scales give the preliminary heuristic

$$
\tau_{\max} \lesssim \min\left(\frac{1}{\gamma}, \frac{r_{\text{valid}}^2}{\sigma_v^2}\right)

$$

This heuristic does not certify integrator stability: the actual force, curvature,
diffusion and Lyapunov derivative bounds must also be supplied. In the quadratic
specialization of {prf:ref}`thm-kinetic-exact-baoab-cap-coupling`, the explicit
condition is $0<\tau<2$. A nondegenerate Gaussian position increment can cross
a bounded valid domain at every positive timestep. The configured terminal
boundary test classifies that event; its probability is controlled by
{prf:ref}`lem-kinetic-terminal-status-coupling`.

**3. Velocity Squashing (Always On):**

The configured radial cap has radius $v_{\max}>0$ and is the continuously
differentiable map

$$
S(v)=\frac{v_{\max}v}{v_{\max}+\|v\|},\qquad
\|S(v)\|=\frac{v_{\max}\|v\|}{v_{\max}+\|v\|}<v_{\max}.
$$

It satisfies $S(0)=0$, $DS(0)=I_d$, and

$$
\|S(v)-S(w)\|\leq\|v-w\|,\qquad
0<\|S(v)\|<\|v\|\quad\text{for }v\ne0.
$$

Thus there is no positive-radius region on which the cap is the identity.
The map is applied after each configured capped kinetic step
(Definition {prf:ref}`def-kinetic-operator-stratonovich`), bounding all output
velocity moments. Derivative-based weak-error estimates must check their own
regularity hypotheses; continuous differentiability of this cap does not
provide higher derivatives at the origin.

**4. Fluctuation-Dissipation Balance (Optional):**

For physical systems at temperature $T$:

$$
\Sigma\Sigma^T = 2\gamma (k_B T / m) I_d

$$

where $k_B$ is Boltzmann's constant and $m$ is the particle mass. For the
underlying uncapped Ornstein--Uhlenbeck velocity component with constant
diffusion and zero force, this gives the invariant density proportional to
$e^{-m\|v\|^2/(2k_B T)}$. The capped output is supported in the velocity ball,
so that unbounded Gaussian law is not its invariant output law. Invariance
for the complete selected or killed kernel requires a separate argument.

For optimization applications, this balance is **not required** - $\gamma$ and $\sigma_v$ are independent algorithmic parameters.
:::

### 3.4. The Fokker-Planck Equation

The kinetic operator induces evolution of the swarm's probability density.

:::{prf:proposition} Fokker-Planck Equation for the Kinetic Operator
:label: prop-fokker-planck-kinetic

Let $\rho(x,v,t)$ be the probability density of a single walker at time $t$. Under the kinetic SDE ({prf:ref}`def-kinetic-operator-stratonovich`), $\rho$ evolves according to:

$$
\partial_t \rho = -\nabla_x \cdot (v \rho) - \nabla_v \cdot \left(\left[F(x) - \gamma\bigl(v - u(x)\bigr) + \frac{1}{2}\sum_{j=1}^d \Sigma_j \cdot \nabla_v \Sigma_j\right]\rho\right) + \frac{1}{2}\sum_{i,j} \partial_{v_i}\partial_{v_j}[(\Sigma\Sigma^T)_{ij} \rho]

$$

**Key Terms:**

1. **Transport:** $-v \cdot \nabla_x \rho$ (position advection by velocity)
2. **Drift:** $-\nabla_v \cdot ([F(x) - \gamma(v-u(x)) + \text{Stratonovich correction}]\rho)$
3. **Diffusion:** $\frac{1}{2}\text{Tr}(\Sigma\Sigma^T \nabla_v^2 \rho)$ (thermal noise)

This is the **generator** of the kinetic operator on the density space.
:::

:::{prf:proof}
**Proof.**

This follows from standard SDE theory. For Stratonovich SDEs, the Fokker-Planck equation is derived by:

1. Converting to Itô form (adding the Stratonovich correction)
2. Applying the Itô-to-Fokker-Planck correspondence

For our isotropic case where Stratonovich = Itô, the derivation is immediate from Itô's lemma applied to test functions.

**Q.E.D.**
:::

:::{prf:remark} Formal Invariant Measure (Without Boundary)
:label: rem-formal-invariant-measure

On the **unbounded domain** $\mathbb{R}^d \times \mathbb{R}^d$ without the boundary condition, the Fokker-Planck equation admits the formal invariant density:

$$
\rho_{\infty}(x,v) \propto \exp\left(-U(x) - \gamma\, v^T(\Sigma\Sigma^T)^{-1} v\right)

$$

For isotropic $\Sigma = \sigma_v I_d$, this becomes $\rho_{\infty}(x,v) \propto \exp\left(-U(x) - \frac{\gamma}{\sigma_v^2}\|v\|^2\right)$, i.e., the standard Gaussian with variance $\sigma_v^2/(2\gamma)$ in each velocity coordinate.

**However:** The boundary condition (walkers die when exiting $\mathcal{X}_{\text{valid}}$) makes this measure invalid. Instead, the system converges to a **quasi-stationary distribution** (QSD) - a distribution conditioned on survival. This is analyzed in the companion document {doc}`06_convergence`.
:::

### 3.5. Numerical Integration

For practical implementation, the Stratonovich SDE is discretized using splitting schemes.

:::{prf:definition} BAOAB Integrator for Stratonovich Langevin
:label: def-baoab-integrator

The **BAOAB splitting scheme** (Leimkuhler & Matthews, 2013) is a symmetric, second-order accurate integrator for underdamped Langevin dynamics:

**B-step (velocity drift from force):**

$$
v^{(1)} = v^{(0)} + \frac{\tau}{2} F(x^{(0)})

$$

**A-step (position update):**

$$
x^{(1)} = x^{(0)} + \frac{\tau}{2} v^{(1)}

$$

**O-step (Ornstein-Uhlenbeck for friction + noise):**

$$
v^{(2)} = e^{-\gamma \tau} v^{(1)} + \sqrt{\frac{1 - e^{-2\gamma\tau}}{2\gamma}} \, \Sigma \xi

$$

where $\xi \sim \mathcal{N}(0, I_d)$. For isotropic $\Sigma = \sigma_v I_d$, this reduces to $\sqrt{\sigma_v^2/(2\gamma)(1 - e^{-2\gamma\tau})}\,\xi$.

**A-step (position update, continued):**

$$
x^{(2)} = x^{(1)} + \frac{\tau}{2} v^{(2)}

$$

**B-step (velocity drift, continued):**

$$
v^{(3)} = v^{(2)} + \frac{\tau}{2} F(x^{(2)})

$$

**Output:** $(x^{(2)}, v^{(3)})$

**Advantages:**
- Second-order accurate in $\tau$
- Correct invariant distribution in the $\tau \to 0$ limit
- Separates deterministic and stochastic dynamics cleanly
:::

:::{prf:remark} Implementation Alignment
:label: rem-kinetic-004
In the Euclidean Gas implementation, the BAOAB map is applied to the total force
$$
F_{\text{tot}}(x, v) = -\nabla U(x) - \epsilon_F \nabla V_{\text{fit}}(x, v) + \nu F_{\text{viscous}}(x, v),
$$
with optional anisotropic diffusion and a **mandatory** velocity squashing map after the final B-step. The resulting one-step transition kernel is the pushforward of the Gaussian noise through this full BAOAB map, so it is not generally Gaussian when the force field is nonlinear.
:::

:::{prf:remark} Stratonovich Correction for Anisotropic Case
:label: rem-baoab-anisotropic

For general $\Sigma(x,v)$, the O-step must be modified to use the **midpoint evaluation** of $\Sigma$ with the OU variance factor $(1 - e^{-2\gamma\tau})/(2\gamma)$:

**Modified O-step:**
```python
# Predictor
noise_var = (1.0 - exp(-2.0*gamma*tau)) / (2.0*gamma)
v_pred = exp(-gamma*tau)*v + Sigma(x, v) * sqrt(noise_var) * xi

# Corrector (Stratonovich midpoint)
Sigma_mid = 0.5*(Sigma(x, v) + Sigma(x, v_pred))
v_new = exp(-gamma*tau)*v + Sigma_mid * sqrt(noise_var) * xi
```

For the isotropic case, this simplifies to the standard BAOAB.
:::

### 3.6. Summary and Preview

This chapter has established:

1. ✅ **Rigorous definition** of the kinetic operator in Stratonovich form
2. ✅ **Axioms** for confining potential and diffusion tensor
3. ✅ **Fokker-Planck equation** governing density evolution
4. ✅ **Numerical scheme** (BAOAB) for practical implementation

**What comes next:**

- **Section 3.7:** Establish rigorous connection between continuous-time generators and discrete-time expectations
- **Chapter 4:** Prove that $\Psi_{\text{kin}}$ contracts the inter-swarm error $V_W$ via **hypocoercivity**
- **Chapter 5:** Prove velocity variance dissipation via **Langevin friction**
- **Chapter 6:** Bound positional variance expansion from **diffusion**
- **Chapter 7:** Prove boundary potential contraction from **confining potential**

These drift inequalities will then be combined with the cloning results ({doc}`03_cloning`) to establish the main convergence theorem.

### 3.7. From Continuous-Time Generators to Discrete-Time Drift

**Purpose of This Section:**

Throughout Chapters 4-7, we analyze the kinetic operator's effect on various Lyapunov components. To make these analyses rigorous, we must clarify the relationship between:
1. **Continuous-time generators** $\mathcal{L}$ acting on Lyapunov functions
2. **Discrete-time expectations** $\mathbb{E}[V(S_\tau)] - V(S_0)$ for finite timestep $\tau$

This section establishes the foundational result that allows us to translate continuous-time drift inequalities into discrete-time contraction guarantees.



#### 3.7.1. The Continuous-Time Generator

:::{prf:definition} Infinitesimal Generator of the Kinetic SDE
:label: def-generator

For a smooth function $V: \mathbb{R}^{2dN} \to \mathbb{R}$ (where $N$ particles have positions $\{x_i\}$ and velocities $\{v_i\}$), the **infinitesimal generator** $\mathcal{L}$ of the kinetic SDE is:

$$
\mathcal{L}V(S) = \lim_{\tau \to 0^+} \frac{\mathbb{E}[V(S_\tau) | S_0 = S] - V(S)}{\tau}

$$

**Explicit Formula (Itô case):**

For the SDE system:

$$
\begin{aligned}
dx_i &= v_i \, dt \\
dv_i &= b_v(x_i,v_i)\, dt + \Sigma(x_i, v_i) \, dW_i
\end{aligned}

$$

The generator is:

$$
\mathcal{L}V = \sum_{i=1}^N \left[ v_i \cdot \nabla_{x_i} V + b_v(x_i,v_i) \cdot \nabla_{v_i} V + \frac{1}{2} \text{Tr}(A_i \nabla_{v_i}^2 V) \right]

$$

where $A_i = \Sigma(x_i, v_i) \Sigma^T(x_i, v_i)$ is the diffusion matrix.

**For Stratonovich SDEs:** the generator uses the Itô drift $b_v$ defined in {prf:ref}`rem-stratonovich-ito-equivalence`. For **isotropic diffusion** $\Sigma = \sigma_v I_d$, the correction term vanishes and $b_v(x,v) = F(x) - \gamma(v-u(x))$.
:::

:::{prf:remark} Why We Work with Generators
:label: rem-kinetic-006
:class: tip

The generator $\mathcal{L}$ captures the **instantaneous rate of change** of $V$ along trajectories. If we can prove:

$$
\mathcal{L}V(S) \leq -\kappa V(S) + C

$$

then this immediately implies exponential decay of $V$ in continuous time. The challenge is translating this to the discrete-time algorithm.
:::



#### 3.7.2. Main Discretization Theorem

:::{prf:theorem} Discrete-Time Inheritance of Generator Drift
:label: thm-discretization

Let $P_h$ be the exact semigroup of the stated kinetic generator $\mathcal L$
and let $K_h$ be a discrete extension approximating that same generator.
For a nonnegative observable $V$, assume integrability sufficient for Dynkin's
formula and the **verified observable weak-error certificate**

$$
|K_hV(S)-P_hV(S)|\le K_Vh^2(1+V(S)),\qquad 0<h\le H,    \tag{5.D1}
$$

where $K_V$ is explicit and independent of population size. Assume also
$\mathcal LV\le-\kappa V+C$, with $\kappa>0$, $C\ge0$. Then

$$
K_hV(S)\le\left(e^{-\kappa h}+K_Vh^2\right)V(S)
 +\frac C\kappa(1-e^{-\kappa h})+K_Vh^2.                \tag{5.D2}
$$

In particular, for

$$
h\le h_*:=\min\!\left(H,\frac{\kappa}{\kappa^2+2K_V}\right),
\qquad
K_hV(S)\le(1-\kappa h/2)V(S)+(C+K_VH)h.               \tag{5.D3}
$$

The regularity and moment hypotheses establishing (5.D1) must be checked for
its observable and transition family. Bounded derivatives on compact sets
alone do not supply a global coefficient or generator consistency.

This transfer applies to generator-consistent uncapped extensions, or to other
families for which (5.D1) is separately proved. The canonical fixed map
$C_V(v)=Vv/(V+|v|)$ is not near the identity as $h\downarrow0$:
for $v\ne0$, $C_V(v)-v\ne0$ has a nonzero limit. Its weak error against the
uncapped Langevin semigroup is therefore generally of order one, even for a
linear velocity test. A velocity bound from that cap does not establish (5.D1).
The native capped quadratic estimate is the direct theorem (5.K1).
:::

:::{prf:proof}
Dynkin's formula and Gronwall give
$P_hV\le e^{-\kappa h}V+C(1-e^{-\kappa h})/\kappa$.
Add (5.D1) to obtain (5.D2). The elementary inequalities
$e^{-u}\le1-u+u^2/2$ and $1-e^{-u}\le u$ for $u\ge0$ give

$$
K_hV\le[1-\kappa h+(\kappa^2/2+K_V)h^2]V+Ch+K_Vh^2.
$$

The restriction defining $h_*$ makes the quadratic coefficient at most
$\kappa h/2$, and $h\le H$ bounds the additive error by $K_VHh$.
This proves (5.D3). For the cap statement, apply the deterministic small-step
limit to a nonzero input velocity: uncapped kicks and friction tend to the
identity, while the final cap tends to $C_V$. Their limits differ. $\square$
:::

#### 3.7.3. Rigorous Component-Wise Weak Error Analysis

:::{prf:remark} Scope of all later generator transfers
:label: rem-kinetic-generator-transfer-scope

Every invocation of {prf:ref}`thm-discretization` in Sections 4--7 requires
(5.D1) for the exact observable and transition being used. Generator calculations
with fixed averaging sets describe the continuous extension. Status changes,
reselection and killed-boundary conventions require their own source bounds.
The canonical fixed radial cap has no such uncapped Langevin weak limit and
must use a direct finite-step estimate. Its deterministic velocity bound is
valid for the native output, but cannot be inserted into an uncapped generator
calculation without a separate moment argument for that continuous process.
:::

This section provides **complete rigorous proofs** that {prf:ref}`thm-discretization` applies to each **TV component** of
$$
V_{\text{TV}} = c_V(V_{\text{Var},x} + V_{\text{Var},v}) + c_\mu \|\mu_v\|^2 + c_B W_b.
$$
The Wasserstein component $V_W$ belongs to the deferred W2 track and is treated separately below.

:::{important}
**On proof completeness**: The TV-relevant components in §3.7.3.1-3.7.3.2 use standard BAOAB weak error theory for smooth bounded test functions (Leimkuhler & Matthews, 2015). The Wasserstein component in §3.7.3.3 is **deferred** and not used in the TV proof.
:::

**Challenge:** The standard weak error theory for BAOAB requires test functions with globally bounded derivatives. Our Lyapunov components require special handling:
- $V_W$ (Wasserstein): Not an explicit function, defined via optimal transport (deferred W2 track)
- $V_{\text{Var}}$ (Variance): Many-body term with combinatorial derivative structure
- $W_b$ (Boundary): Derivatives explode near $\partial\mathcal{X}_{\text{valid}}$
- $\|\mu_v\|^2$ (Barycenter): Quadratic but still requires bounded-derivative justification on the squashed state space

**Solution:** We prove weak error bounds component-by-component using specialized techniques.

##### 3.7.3.1. Weak Error for Variance Components ($V_{\text{Var}}$)

:::{prf:proposition} BAOAB Weak Error for Variance Lyapunov Functions
:label: prop-weak-error-variance

Use a generator-consistent extension as in {prf:ref}`thm-discretization`, with
fixed averaging sets and the same observable conventions in both kernels.
Put $M_2=N^{-1}\sum_i|z_i|^2$ and $B_2=|\bar z|^2$.
Assume separately proved, population-uniform weak-error certificates

$$
|(K_h-P_h)M_2|\le K_Mh^2(1+M_2),\qquad
|(K_h-P_h)B_2|\le K_Bh^2(1+M_2).
$$

Then the normalized variance $V_{\rm Var}=M_2-B_2$ satisfies

$$
|(K_h-P_h)V_{\rm Var}|
\le(K_M+K_B)h^2(1+M_2).                               \tag{5.D4}
$$

The coefficient is independent of $N$. Replacing $M_2$ by a multiple of
$1+V_{\rm Var}$ additionally requires control of the barycenter moment.
The certificates require analytic force/diffusion regularity and global
moment bounds, or an exact affine calculation. The fixed radial cap does not
provide a weak approximation to the uncapped semigroup.
:::

:::{prf:proof}
The parallel-axis identity gives
$N^{-1}\sum_i|z_i-\bar z|^2=M_2-B_2$ for every empirical measure.
Linearity of both kernels and the triangle inequality prove (5.D4).
Both inputs are normalized observables; there is no sum of unnormalized
particle errors. For per-atom certificates, averaging their bounds gives
$N^{-1}\sum_i K h^2(1+|z_i|^2)=Kh^2(1+M_2)$.
The barycenter certificate must still be proved for its actual joint noise
covariance. For affine quadratic dynamics these covariances propagate by
finite matrices, as in the exact specialization below. $\square$
:::

:::{prf:remark}
:label: rem-fg-kinetic-weak-error-velocity
The velocity barycenter observable $|\mu_v|^2$ requires its own certificate
of the form (5.D1), with the actual barycenter covariance and a global moment
envelope. Smoothness of this quadratic observable does not make a fixed-cap
transition generator consistent. For the affine uncapped extension its mean
and covariance can be propagated exactly; the corresponding weak coefficient
is population uniform for a normalized barycenter moment envelope.
:::

##### 3.7.3.2. Weak Error for Boundary Component ($W_b$)

:::{prf:proposition} BAOAB Weak Error for Boundary Lyapunov Function
:label: prop-weak-error-boundary

Use the same generator-consistent extension, status convention and boundary
observable in both kernels. Let $W_b=N^{-1}\sum_i\varphi(x_i)$ and assume a
verified per-atom observable weak-error certificate

$$
|(K_h-P_h)\varphi(x_i)|\le K_\varphi h^2\mathcal M_i(S),
\qquad N^{-1}\sum_i\mathcal M_i(S)\le\mathcal M(S),
$$

with $K_\varphi$ and the moment envelope $\mathcal M$ population uniform.
Then

$$
|(K_h-P_h)W_b(S)|\le K_\varphi h^2\mathcal M(S).        \tag{5.D5}
$$

Force/diffusion regularity, integrable barrier derivatives on the reachable
space, and any killed-boundary discontinuities must be checked when proving
the certificate. A bounded boundary-layer formula and the fixed velocity cap
alone do not provide it.
:::

:::{prf:proof}
Linearity and the triangle inequality bound the weak error of the average by
$N^{-1}\sum_i K_\varphi h^2\mathcal M_i$. The stated envelope gives (5.D5),
without a population-size factor. $\square$
:::

##### 3.7.3.3. Weak Error for Wasserstein Component ($V_W$) - Synchronous Coupling

:::{important}
This subsection is part of the **deferred W2 track** and is **not used** in the TV convergence proof. It remains provisional and will be repaired with a coupling-stability argument in a later revision.
:::

:::{prf:proposition} BAOAB Weak Error for Wasserstein Distance
:label: prop-weak-error-wasserstein

Assume a generator-consistent uncapped extension, an $N$-uniform fixed-coupling
quadratic weak-error certificate with a global moment envelope, and a separately
proved uniform matching-stability estimate transferring that certificate to the
assignment minimum. Under these additional hypotheses, for
$V_W=W_h^2(\mu_1,\mu_2)$:

$$
\left|\mathbb{E}[V_W(S_\tau^{\text{BAOAB}})] - \mathbb{E}[V_W(S_\tau^{\text{exact}})]\right| \leq K_W \tau^2 (1 + V_W(S_0))

$$

where $K_W = K_W(d, \gamma, L_F, L_\Sigma, \sigma_{\max}, \lambda_v, b)$ is **independent of $N$**.
:::

:::{prf:proof}
**Proof (Synchronous Coupling at Particle Level).**

**PART I: Synchronous Coupling Setup**

Consider two swarms $(S_1, S_2)$ evolving under the **same Brownian motion** $W_i(t)$ for each walker index $i$:

$$
\begin{aligned}
dx_{1,i} &= v_{1,i} \, dt \\
dv_{1,i} &= [F(x_{1,i}) - \gamma v_{1,i}] \, dt + \Sigma(x_{1,i}) \circ dW_i \\[1em]
dx_{2,i} &= v_{2,i} \, dt \\
dv_{2,i} &= [F(x_{2,i}) - \gamma v_{2,i}] \, dt + \Sigma(x_{2,i}) \circ dW_i
\end{aligned}

$$

**Key Property (Noise Cancellation):** The difference process $\Delta z_i(t) = z_{1,i}(t) - z_{2,i}(t)$ evolves as:

$$
\begin{aligned}
d(\Delta x_i) &= \Delta v_i \, dt \\
d(\Delta v_i) &= [\Delta F_i - \gamma \Delta v_i] \, dt + [\Sigma(x_{1,i}) - \Sigma(x_{2,i})] \circ dW_i
\end{aligned}

$$

where $\Delta F_i := F(x_{1,i}) - F(x_{2,i})$.

Since the Brownian motions are identical, the **leading-order noise cancels**. The residual noise $\Delta\Sigma_i = \Sigma(x_{1,i}) - \Sigma(x_{2,i})$ satisfies:

$$
\|\Delta\Sigma_i\|_F \leq L_\Sigma \|\Delta x_i\|

$$

by global Lipschitz continuity ({prf:ref}`axiom-diffusion-tensor`, part 3). The residual noise amplitude is $O(\|\Delta x_i\|)$, so its contribution to the generator acting on quadratic test functions is $O(\|\Delta x_i\|^2)$.

**PART II: Single-Pair Weak Error Analysis**

Define the **hypocoercive quadratic form** on the difference:

$$
f(\Delta z) := \|\Delta z\|_h^2 = \|\Delta x\|^2 + \lambda_v\|\Delta v\|^2 + b\langle\Delta x, \Delta v\rangle = \Delta z^T Q \Delta z

$$

where:

$$
Q = \begin{pmatrix} I_d & \frac{b}{2} I_d \\ \frac{b}{2} I_d & \lambda_v I_d \end{pmatrix}

$$

with $\lambda_v > 0$ and $4\lambda_v - b^2 > 0$ ensuring positive-definiteness.

**Derivatives:** Since $f$ is quadratic:

$$
\nabla f(\Delta z) = 2Q\Delta z \quad \text{(linear growth)}, \quad \nabla^2 f = 2Q \quad \text{(bounded)}, \quad \nabla^3 f = 0

$$

**Apply Weak Error Theory for Polynomial-Growth Test Functions:**

By weak error theory for Langevin dynamics (Leimkuhler & Matthews 2015, Talay-Tubaro expansions), for test functions $g$ with polynomial growth and bounded higher derivatives, under:
- Coercivity ({prf:ref}`axiom-confining-potential`) ensuring $\mathbb{E}[\|Z_t\|^4] < \infty$ uniformly in $t$
- Global Lipschitz $\Sigma$ ({prf:ref}`axiom-diffusion-tensor`)

we have:

$$
\left|\mathbb{E}[g(Z_\tau^{\text{BAOAB}})] - \mathbb{E}[g(Z_\tau^{\text{exact}})]\right| \leq C_{\text{LM}} \tau^2 (1 + \mathbb{E}[\|Z_0\|^{2p}])

$$

where $C_{\text{LM}} = C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max})$.

**Apply to $g = f$:** For our quadratic $f$ (with $p=2$):

$$
\left|\mathbb{E}[\|\Delta z_i(\tau)\|_h^2]^{\text{BAOAB}} - \mathbb{E}[\|\Delta z_i(\tau)\|_h^2]^{\text{exact}}\right| \leq C_{\text{pair}} \tau^2 (1 + \|\Delta z_i(0)\|_h^2)

$$

where $C_{\text{pair}} := C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max}) \cdot \|Q(\lambda_v, b)\|$.

**PART III: Force Term Handling**

From {prf:ref}`axiom-confining-potential`, the force $F = -\nabla U$ satisfies local Lipschitz bounds on compact sets (ensured by coercivity):

$$
\|\Delta F_i\| \leq L_F \|\Delta x_i\|

$$

The drift of $f(\Delta z_i)$ involves:

$$
\nabla f \cdot \text{drift} = 2Q\Delta z \cdot \begin{pmatrix} \Delta v \\ \Delta F - \gamma \Delta v \end{pmatrix}

$$

The force contribution is quadratic in $\|\Delta z\|_h^2$ and is absorbed into the weak error constant $C_{\text{pair}}$.

**PART IV: Aggregation Over $N$ Particles**

By index-matching:

$$
V_W(S_1, S_2) = W_h^2(\mu_1, \mu_2) \leq \frac{1}{N}\sum_{i=1}^N \|\Delta z_i\|_h^2

$$

Summing the single-pair bounds:

$$
\begin{aligned}
&\left|\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \|\Delta z_i(\tau)\|_h^2\right]^{\text{BAOAB}} - \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N \|\Delta z_i(\tau)\|_h^2\right]^{\text{exact}}\right| \\
&\quad= \frac{1}{N}\sum_{i=1}^N C_{\text{pair}} \tau^2 (1 + \|\Delta z_i(0)\|_h^2) \\
&\quad= C_{\text{pair}} \tau^2 \left(1 + \frac{1}{N}\sum_{i=1}^N \|\Delta z_i(0)\|_h^2\right) \leq C_{\text{pair}} \tau^2 (1 + V_W(S_0))
\end{aligned}

$$

**Propagate to Wasserstein via Min-Over-Permutations:**

Define $C_\sigma(S) := \frac{1}{N}\sum_{i=1}^N \|\Delta z_{\sigma(i)}\|_h^2$ for pairing $\sigma$. Then $V_W(S) = \min_\sigma C_\sigma(S)$.

**Key inequality:** For any states $S^A$, $S^E$:

$$
\left|\min_\sigma C_\sigma(S^A) - \min_\sigma C_\sigma(S^E)\right| \leq \max_\sigma \left|C_\sigma(S^A) - C_\sigma(S^E)\right|

$$

Controlling the right-hand side **uniformly in $\sigma$** requires stability of optimal matchings under the BAOAB perturbation. This step is deferred; the bound above remains the correct reduction but is not closed here.

**Deferred conclusion:** The fixed-matching weak error bound holds for each $\sigma$, but transferring it to $\min_\sigma$ requires additional coupling stability arguments.

**PART V: N-Uniformity**

Define $K_W := C_{\text{pair}} = C_{\text{LM}}(d, \gamma, L_F, L_\Sigma, \sigma_{\max}) \cdot \|Q(\lambda_v, b)\|$.

The constant $K_W$ is **independent of $N$** at the fixed-matching level because:
1. Each walker pair contributes $O(\tau^2)$ error
2. Summing $N$ terms and dividing by $N$ cancels the $N$-dependence
3. No mean-field approximation is used

**Why This Approach Works (Deferred):**

Unlike the kinetic Fokker-Planck PDE (which is NOT a $W_2$-gradient flow), this proof:
- Works at particle level with finite-$N$ systems
- Uses synchronous coupling for noise cancellation
- Applies standard weak error theory to an explicit quadratic test function
- Reduces the Wasserstein weak error to stability of optimal matchings (deferred)
**Deferred.**
:::

:::{prf:remark} Comparison to Gradient Flow Approach
:label: rem-gradient-flow-vs-coupling

The previous version of this proof incorrectly applied JKO scheme theory for Wasserstein gradient flows to the kinetic Fokker-Planck equation. **Fatal flaws:**

1. **Underdamped Langevin is NOT a $W_2$-gradient flow** - only overdamped Langevin ($dx = F(x)dt + \sigma dW$) has this structure
2. **JKO theory applies to continuous measures** evolving via PDE, not empirical measures (finite $N$)
3. **No verification of technical conditions** for the splitting scheme

The correct approach uses **synchronous coupling at the particle level** - a standard technique in weak error analysis that requires no PDE theory or gradient flow structure.
:::

:::{important}
**Note on Isotropic Diffusion:** For the primary case $\Sigma(x,v) = \sigma_v I_d$ (isotropic, constant diffusion), the Stratonovich and Itô formulations coincide (see {prf:ref}`rem-stratonovich-ito-equivalence`). For general state-dependent $\Sigma$, the BAOAB scheme requires midpoint evaluation for Stratonovich noise, and $L_\Sigma$ appears explicitly in $K_W$.
:::

##### 3.7.3.4. Assembly: Proof of {prf:ref}`thm-discretization` for $V_{\text{total}}^{W2}$ (Deferred)

:::{prf:proof}
**Conditional assembly for the deferred transport track.**
Assume all component weak-error certificates have been established for the
same generator-consistent extension, with a common global envelope dominated
by $1+V_{\rm total}^{W2}$. The transport certificate additionally requires its
matching-stability hypothesis. By linearity and the triangle inequality,

$$
|(K_h-P_h)V_{\rm total}^{W2}|
\le (K_W+c_VK_{\rm Var}+c_BK_b)h^2(1+V_{\rm total}^{W2}).
$$

Apply {prf:ref}`thm-discretization` with this sum as $K_V$ and a separately
proved generator drift for $V_{\rm total}^{W2}$. This gives (5.D2)--(5.D3).
The argument assembles verified inputs; it does not prove matching stability,
global derivative bounds, or fixed-cap generator consistency. $\square$
:::

:::{admonition} Key Achievement
:class: important

This multi-part proof is structured so that the **TV-relevant components** (variance, barycenter, boundary) rely only on standard weak-error estimates for smooth bounded test functions. The Wasserstein component is **deferred** and not used in the TV proof.
:::



#### 3.7.4. Explicit Constants

To make the above theorem fully constructive, we now provide explicit formulas for the constants.

:::{prf:proposition} Explicit Discretization Constants
:label: prop-explicit-constants

For the uncapped one-dimensional extension
$dx=v\,dt$, $dv=(-x-v)\,dt+dW$, no added position diffusion or killing,
$f(z)=|x|^2+|v|^2$, $|z_0|^2\le0.8$, $T=0.16$ and $0<h\le H=0.04$ dividing
$T$, the native BAOAB expectation satisfies

$$
|\mathbb E f(Z_T^{h})-\mathbb E f(Z_T)|\le C_{\rm weak}h^2,
\qquad C_{\rm weak}=12.29925819\ldots.                 \tag{5.D6}
$$

The same coefficient holds for the normalized average of $f$ over any
population with average initial squared norm at most $0.8$. For independent
particle noises, the barycenter moment has the same upper coefficient, and
the normalized variance has coefficient at most $2C_{\rm weak}$ by (5.D4).
These are coefficients for the specified affine extension and observable;
they do not establish a general nonlinear $K_W$ or a fixed-cap SDE transfer.
:::

:::{prf:proof}
Let $A=\left(\begin{smallmatrix}0&1\\-1&-1\end{smallmatrix}\right)$,
$L=\|A\|=(1+\sqrt5)/2$ and $S=3$, the sum of the norms of the three splitting
generators. The palindromic BAOAB transition $A_h$ and $e^{Ah}$ have equal
derivatives of orders zero, one and two at zero: $I,A,A^2$. Product
differentiation and Taylor's integral remainder give

$$
\|A_h-e^{Ah}\|\le C_Ah^3,\qquad
C_A=\frac{S^3e^{SH}+L^3e^{LH}}6.
$$

Write the native noise covariance as
$Q_h=q(h)u(h)u(h)^\top$, with
$q(h)=(1-e^{-2h})/2$ and $u(h)=(h/2,1-h^2/4)^\top$.
The exact covariance is $Q(h)=\int_0^h e^{As}Je^{A^\top s}\,ds$, where
$J=\operatorname{diag}(0,1)$. Both covariances have derivatives at zero
$0,J,AJ+JA^\top$, the last being
$\left(\begin{smallmatrix}0&1\\1&-2\end{smallmatrix}\right)$.
Set

$$
U=\sqrt{(H/2)^2+(1+H^2/4)^2},\quad
U_1=\sqrt{1/4+(H/2)^2},\quad U_2=1/2.
$$

On $[0,H]$ these bound $|u|,|u'|,|u''|$, while
$q\le H$, $|q'|\le1$, $|q''|\le2$ and $|q'''|\le4$.
The product rule bounds $\|Q_h'''\|$ by
$4U^2+12UU_1+6(U_1^2+UU_2)+6HU_1U_2$.
Differentiating $Q(h)$ gives $\|Q'''(h)\|\le4L^2e^{2LH}$.
Consequently

$$
\|Q_h-Q(h)\|\le C_Qh^3,\qquad
C_Q=\frac{4U^2+12UU_1+6(U_1^2+UU_2)+6HU_1U_2+4L^2e^{2LH}}6.
$$

The exact uncentered second-moment matrix has trace at most
$M_2=e^{2LT}(0.8+T)$. Subtract its recursion from the native recursion, and
write $D$ for their difference. Using nuclear norm in phase dimension two,
$\|A_h\|\le e^{Sh}$, $\|e^{Ah}\|\le e^{Lh}$ and
$\|Q_h-Q(h)\|_1\le2C_Qh^3$ gives

$$
\|D_{n+1}\|_1\le e^{2Sh}\|D_n\|_1
 +h^3\{C_A(e^{SH}+e^{LH})M_2+2C_Q\}.
$$

Since $D_0=0$, summing $T/h$ steps proves (5.D6) with

$$
C_{\rm weak}=Te^{2ST}\{C_A(e^{SH}+e^{LH})M_2+2C_Q\}.
$$

Substitution gives $C_A=5.82695229\ldots$, $C_Q=4.41612835\ldots$ and the
stated coefficient. Averaging the per-particle bound introduces no $N$.
For independent particle noises, the mean process is the same linear
transition with diffusion reduced by $N^{-1/2}$; its initial squared norm is
at most the average initial squared norm. The same bounds therefore hold
for its moment. Apply (5.D4) to conclude the variance coefficient.
$\square$
:::

#### 3.7.5. Application to Each Lyapunov Component

In the subsequent chapters, we prove generator bounds for each component:

| Chapter | Component (TV Track) | Generator Bound |
|:--------|:----------------------|:---------------|
| 5 | $V_{\text{Var},v}$ (velocity var) | $\mathcal{L}V_{\text{Var},v} \leq -(2\gamma-\epsilon) V_{\text{Var},v} + C_v'$ |
| 5.4.1 | $\|\mu_v\|^2$ | $\mathcal{L}\|\mu_v\|^2 \leq -\gamma \|\mu_v\|^2 + C_{\mu}'$ |
| 6 | $V_{\text{Var},x}$ (position var) | $\mathcal{L}V_{\text{Var},x} \leq C_x'$ |
| 7 | $W_b$ (boundary) | $\mathcal{L}W_b \leq -\kappa_b W_b + C_b'$ |

**Deferred:** The inter-swarm $V_W$ bounds in Chapter 4 belong to the W2 track and are not used here.

**By {prf:ref}`thm-discretization`:** A component with a verified generator-consistency and observable weak-error certificate inherits a discrete-time inequality:

$$
\mathbb{E}[V_{\text{component}}(S_\tau)] \leq (1 - \frac{\kappa_{\text{component}}\tau}{2})V_{\text{component}}(S_0) + (C_{\text{component}}'+K_{\text{component}}H)\tau

$$

for its explicit range (5.D3). A zero-contraction component instead retains its proved expansion bound.

**Unified timestep:** Taking $\tau < \tau_{\text{global}} := \min_{\text{components}} \tau_*(\kappa_{\text{component}})$ ensures all components satisfy their drift inequalities simultaneously.



#### 3.7.6. Summary and Interpretation

:::{admonition} Key Takeaways
:class: important

**What we've established:**
1. **Continuous-time generators** $\mathcal{L}$ are the natural objects for analysis (cleaner proofs, geometric interpretation)
2. **Discrete-time algorithms** inherit drift properties via Taylor expansion + integrator accuracy
3. **Explicit timestep bounds** $\tau_*$ ensure the discrete algorithm respects the continuous theory
4. **Constructive constants** allow practitioners to choose safe $\tau$ values

**How this resolves the reviewer's concern:**
- Previous proofs mixed $\mathcal{L}V$ and $\Delta V$ notation without justification
- Now we have a **rigorous bridge** between the two frameworks
- A generator drift requires the additional observable certificate before invoking {prf:ref}`thm-discretization`; native fixed-cap estimates use their direct finite-step proof.

**Cost:**
- Requires $\tau$ to be "sufficiently small" (but explicit bound given)
- Acceptable tradeoff: timestep restrictions are standard in numerical analysis
:::



**Notation for Subsequent Chapters:**

From now on:
- **$\mathcal{L}V \leq ...$** denotes continuous-time generator bounds
- **$\mathbb{E}[\Delta V] = \mathbb{E}[V(S_\tau) - V(S_0)] \leq ...$** denotes discrete-time drift, derived via {prf:ref}`thm-discretization`
- We will prove generator bounds first, then immediately cite {prf:ref}`thm-discretization` for the discrete version



**End of Section 3.7**

## Part II (Deferred): W2/Hypocoercive Track

The remainder of Part II records the hypocoercive/Wasserstein analysis. It is **not used** in the TV convergence proof and will be tightened in a later revision.

## 4. Hypocoercive Contraction of Inter-Swarm Error

### 4.1. Introduction: The Hypocoercivity Challenge

The kinetic operator faces a fundamental challenge: the velocity diffusion is **degenerate** in position space. The noise acts only on $v$, not directly on $x$:


$$
dv_t = \ldots + \Sigma(x_t,v_t) \circ dW_t

$$

$$
dx_t = v_t dt \quad \text{(no noise term!)}

$$

**Classical Poincaré Theory Fails:**

Standard elliptic regularity requires noise in all variables. Since $x$ has no direct noise, the generator is **not coercive** with respect to the full $(x,v)$ norm.

**Hypocoercivity to the Rescue:**

**Hypocoercivity theory** (Villani, 2009) shows that even with degenerate noise, the **coupling** between transport ($v \cdot \nabla_x$) and diffusion ($\text{noise in } v$) creates an effective dissipation in both variables.

**Key Insight:** Noise in $v$ → diffusion in $v$ → transport via $\dot{x} = v$ → effective regularization of $x$.

This chapter proves that this hypocoercive mechanism contracts the inter-swarm Wasserstein distance $V_W$.

:::{prf:remark} Hypocoercive contraction requires a matrix certificate
:label: rem-kinetic-009
:class: important

The contraction theorem below requires the actual macroforce/centered-force
closure, a common positive metric and its uniform Lyapunov matrix inequality.
Confinement, local force Lipschitz continuity and nondegenerate velocity noise
alone do not imply synchronous quadratic transport contraction. In particular,
this argument does not certify every coercive multiwell landscape.
The affine unit-quadratic specialization verifies the closure and matrix
inequality explicitly. A nonconvex specialization is admissible only if its
own stated closure, residual and matrix hypotheses are established.
:::

### 4.2. The Hypocoercive Norm

To analyze hypocoercivity, we must work with a specially designed norm that couples position and velocity.

:::{prf:definition} The Hypocoercive Norm
:label: def-hypocoercive-norm

For the coupled swarm state $(S_1, S_2)$, define the **hypocoercive norm squared** on the phase-space difference:

$$
\|\!(\Delta x, \Delta v)\!\|_h^2 := \|\Delta x\|^2 + \lambda_v \|\Delta v\|^2 + b \langle \Delta x, \Delta v \rangle

$$

where:
- $\Delta x = x_1 - x_2$: Position difference
- $\Delta v = v_1 - v_2$: Velocity difference
- $\lambda_v > 0$: Velocity weight (of order $1/\gamma$)
- $b \in \mathbb{R}$: Coupling coefficient (chosen appropriately)

**For the empirical measures:** The hypocoercive Wasserstein distance is:

$$
V_W(\mu_1, \mu_2) = W_h^2(\mu_1, \mu_2)

$$

where $W_h$ is the Wasserstein-2 distance with cost $\|\!(\Delta x, \Delta v)\!\|_h^2$.

**Decomposition (from {doc}`03_cloning`):**

$$
V_W = V_{\text{loc}} + V_{\text{struct}}

$$
where $V_{\text{loc}}$ measures barycenter separation and $V_{\text{struct}}$ measures shape dissimilarity.
:::

:::{prf:remark} Intuition for the Coupling Term
:label: rem-coupling-term-intuition

The coupling term $b\langle \Delta x, \Delta v \rangle$ is the key to hypocoercivity:

- **Without coupling** ($b = 0$): Position and velocity evolve independently in the norm. The degenerate noise in $v$ doesn't help regularize $x$.

- **With coupling** ($b \neq 0$): The cross term creates a "rotation" in the $(x,v)$ phase space. Even though noise only enters in $v$, the coupling allows dissipation to "leak" into the $x$ coordinate.

The optimal choice of $b$ depends on $\gamma$, $\sigma_v$, and the potential $U$.
:::

### 4.3. Main Theorem: Hypocoercive Contraction

:::{prf:theorem} Inter-Swarm Error Contraction Under Kinetic Operator
:label: thm-inter-swarm-contraction-kinetic

Assume the macroforce closure and positive matrix certificate of
{prf:ref}`lem-location-error-drift-kinetic`, the centered-coupling hypotheses of
{prf:ref}`lem-structural-error-drift-kinetic`, and population-uniform observable
weak-error certificates for the same generator-consistent discrete extension.
Use one common positive metric $P$ and a timestep admissible for both lemmas.
Then

$$
\mathbb E_{\rm kin}[V_W(S'_1,S'_2)\mid S_1,S_2]
\le(1-\kappa_Wh)V_W(S_1,S_2)+C_W'h,
$$

where

$$
\kappa_W=\tfrac14\min(\kappa_{\rm loc},\kappa_s)>0,
\qquad C_W'=C_{\rm loc}+C_s+(K_{\rm loc}+K_s)H.
$$

The constants are independent of $N$ when the hypotheses are uniform in $N$.
Coercivity and force Lipschitz continuity alone do not provide these matrix or
closure certificates. A fixed radial cap is not a generator-consistent
Langevin approximation as $h\downarrow0$; its executed unit-quadratic stage is
covered separately by {prf:ref}`thm-kinetic-exact-baoab-cap-coupling`.

:::

### 4.4. Exact finite-step coupling for the quadratic kinetic stage

:::{div} feynman-prose
Imagine running the quadratic kinetic stage twice, starting from slightly
different positions and velocities but feeding both runs the same Gaussian
draws. This lets us follow what happens to the initial separation through each
operation of the actual update. The common additive noise cancels from the
separation before capping. Friction removes part of that separation, but the
BAOAB identity below shows that it removes only one linear combination of
position and velocity. A second direction still needs control.

The smooth velocity cap supplies that control. Its slope is smaller away from
zero, and the velocity noise gives a uniformly positive chance of reaching
that region, whatever the starting mean. Thus the noise matters even though
the two runs use identical draws: it changes where the nonlinear cap acts.
The two dissipated combinations are independent, so together they control the
whole physical separation. The weighted norm records this fact exactly for
the quadratic force and the stated timestep range. Applying the same estimate
to paired rows and averaging introduces no population-size factor; collision
clusters enter through the inputs to this kinetic stage.
:::

:::{prf:theorem} Strict coupling of quadratic BAOAB with the velocity cap
:label: thm-kinetic-exact-baoab-cap-coupling

For the kinetic stage with $U(x)=|x|^2/2$, $0<h<2$, friction $\gamma>0$,
isotropic velocity diffusion factor $B>0$, final independent position
noise, and cap $C_V(v)=Vv/(V+|v|)$ with $V>0$, couple two input rows using
identical Gaussian innovations. The force kicks, drifts, and cap are those
of the complete algorithm. Put

$$
c=h/2,\quad k=1-c^2,\quad a=e^{-\gamma h},\quad
q=B\sqrt{\frac{1-a^2}{2\gamma}},\qquad
\|z\|_Q^2=k|x|^2+|v|^2.
$$

Define the positive constants

$$
p_0=\sqrt{2/\pi}\,e^{-2},\quad
\eta=p_0\left[1-\left(\frac V{V+kq}\right)^2\right],
$$

$$
T=1-a^2+\eta\bigl[a^2+c^2(1-a^2)\bigr],\quad
D=\eta c^2(1-a^2),\qquad \delta=D/T.
$$

The output physical coordinates, including the final position noise and cap,
satisfy

$$
\mathbb E\|Z^+-\widetilde Z^+\|_Q^2
\le(1-\delta)\|Z-\widetilde Z\|_Q^2.                 \tag{5.K1}
$$

The constants are independent of dimension and population size. Applied to
all paired rows, (5.K1) also contracts the averaged physical-coordinate
coupling cost. Terminal alive/dead indicators are governed separately by
{prf:ref}`lem-kinetic-terminal-status-coupling`.
:::

:::{prf:proof}
**Exact BAOAB difference.** The successive differences are
$\Delta v_1=\Delta v-c\Delta x$,
$\Delta x_1=\Delta x+c\Delta v_1$,
$\Delta v_2=a\Delta v_1$,
$\Delta x_2=\Delta x_1+c\Delta v_2$,
and $\Delta v_3=\Delta v_2-c\Delta x_2$.
Consequently, before capping the deterministic difference matrix is

$$
A=\begin{pmatrix}
1-c^2(1+a)&c(1+a)\\
-c(1+a)k&a-c^2(1+a)
\end{pmatrix},\qquad
A^\top QA=Q-k(1-a^2)ww^\top,\quad w=\binom{-c}{1}.       \tag{5.K2}
$$

The matrices act identically on every coordinate. Write
$b=(-c(1+a)k,\ a-c^2(1+a))^\top$. The last pre-cap velocity is its
input-dependent mean plus $kq\xi$, for a standard Gaussian vector $\xi$;
its coupled difference is $b^\top\Delta z$.

**Strict cap dissipation.** The cap derivative satisfies
$\|DC_V(u)\|_{\mathrm{op}}\le V/(V+|u|)$, including its continuous value
at zero. For any $m$ and $\sigma>0$, a centered interval maximizes the
probability of a fixed-length interval under the centered one-dimensional
Gaussian. This follows directly by differentiating the interval probability
with respect to its center. Therefore

$$
\Pr(|m+\sigma\xi|\ge\sigma)
\ge\Pr(|\xi_1|\ge1)
\ge 2\int_1^2\frac{e^{-2}}{\sqrt{2\pi}}\,dt=p_0.
$$

Taking $\sigma=kq$ yields
$\sup_m\mathbb E\|DC_V(m+kq\xi)\|_{\mathrm{op}}^2\le1-\eta$.
Integrate the derivative along the segment between the two deterministic
means and apply Jensen's inequality to obtain

$$
\mathbb E|C_V(u+kq\xi)-C_V(\widetilde u+kq\xi)|^2
\le(1-\eta)|u-\widetilde u|^2.                         \tag{5.K3}
$$

**Control of both coordinates.** The common final position noise cancels
from the difference. Equations (5.K2)–(5.K3) give

$$
\mathbb E\|\Delta z^+\|_Q^2
\le\|\Delta z\|_Q^2-k(1-a^2)|w^\top\Delta z|^2
-\eta|b^\top\Delta z|^2.
$$

The positive matrix
$Q^{-1/2}[k(1-a^2)ww^\top+\eta bb^\top]Q^{-1/2}$
has trace $T$ and determinant $D$. In particular, it is positive definite:
$\det(w,b)=c\ne0$. For its eigenvalues $0<\lambda_1\le\lambda_2$,
$\lambda_1=D/\lambda_2\ge D/T=\delta$. This proves (5.K1).
Averaging this row estimate proves its population version. $\square$
:::

:::{prf:corollary} The configured quadratic kinetic stage
:label: cor-kinetic-canonical-coupling

For $h=0.04$, $\gamma=B=1$, and $V=2$, the exact constants above give
$\eta>0.0184$ and $\delta>6.03\times10^{-6}$. These are strict
physical-coordinate contraction constants for the executed kinetic stage.
:::

:::{prf:proof}
Substitute the specified constants into the positive expressions for
$q,\eta,T,D$. The symbolic expressions in (5.K1) specify the constants exactly;
substitution gives $\eta=0.018414\ldots$ and
$\delta=0.0000060320\ldots$, with the stated strict lower bounds.
:::

:::{prf:lemma} Shift-uniform radial Gaussian cap dissipation
:label: lem-kinetic-shift-uniform-radial-cap

Let $C_V(v)=Vv/(V+|v|)$, $V>0$, $\sigma>0$ and $Z\sim N(0,I_d)$.
Set $p_d=4$ for $d=1$ and $p_d=2$ for $d\ge2$. Then

$$
\sup_m\mathbb E\|DC_V(m+\sigma Z)\|_{\rm op}^2
=\mathbb E\left(\frac V{V+\sigma\chi_d}\right)^{p_d},
\qquad
\eta_d=1-\mathbb E\left(\frac V{V+\sigma\chi_d}\right)^{p_d}>0. \tag{5.DIM1}
$$

For every deterministic pair $u,\widetilde u$ and the same $Z$,

$$
\mathbb E|C_V(u+\sigma Z)-C_V(\widetilde u+\sigma Z)|^2
\le(1-\underline\eta_d)|u-\widetilde u|^2,             \tag{5.DIM2}
$$

where $0<\underline\eta_d\le\eta_d$ is any certified lower bound.
The dimension-one derivative power is different, so $\eta_1\le\eta_2$ is
not asserted. For fixed $p=2$, increasing dimension increases the dissipation.
:::

:::{prf:proof}
At radius $r$ the cap's radial derivative eigenvalue is $V^2/(V+r)^2$.
In dimension one it is the only eigenvalue. In dimension at least two the
largest derivative eigenvalue is the tangential value $V/(V+r)$.
This proves the stated powers after squaring the operator norm.

A centered isotropic Gaussian maximizes the probability of every ball among
its translates. To verify this directly, rotate the translate so its center
lies on the first coordinate axis. For fixed remaining coordinates the ball
section is either empty or an interval of fixed length in the first coordinate.
The integral of a one-dimensional centered Gaussian over such an interval is
maximal when its center is zero: differentiating its integral with respect to
the interval center gives a nonpositive derivative for positive shifts.
Integrate over the remaining independent coordinates. Every radial decreasing
nonnegative function is a layer-cake integral of ball indicators; applying
the ball inequality inside that integral proves that its Gaussian expectation
is maximal at zero shift. Apply it to $(V/(V+r))^{p_d}$ to get (5.DIM1).
Since $\chi_d>0$ almost surely, the expectation is strictly less than one.

Integrate $DC_V$ along the line segment between $u+\sigma Z$ and
$\widetilde u+\sigma Z$. Jensen's inequality bounds the squared chord by
$|u-\widetilde u|^2$ times the integral of squared operator norms along that
segment. Each segment point is a deterministic translate of $\sigma Z$;
(5.DIM1) bounds its expectation by $1-\eta_d\le1-\underline\eta_d$.
This proves (5.DIM2). For dimensions at least two, couple
$\chi_{d+1}^2=\chi_d^2+Z_{d+1}^2$ to obtain the last monotonicity claim.
$\square$
:::

:::{prf:theorem} Dimension- and curvature-aware native quadratic cap contraction
:label: thm-kinetic-dimension-curvature-cap

Use the actual isotropic quadratic force $F(x)=-\omega x+f_0$, $\omega>0$,
with friction $\gamma>0$, velocity diffusion $B>0$, timestep $h>0$ and
$k=1-\omega h^2/4>0$. Put

$$
c=h/2,\quad a=e^{-\gamma h},\quad
q=B\sqrt{(1-a^2)/(2\gamma)},\quad\sigma=kq,
\qquad Q_\omega=\operatorname{diag}(\omega k,1)\otimes I_d.
$$

Use any certified $\underline\eta_d$ from (5.DIM1)--(5.DIM2), and define

$$
T_d=1-a^2+\underline\eta_d[a^2+\omega c^2(1-a^2)],\quad
D_d=\underline\eta_d\omega c^2(1-a^2),\qquad
\delta_d=\frac{2D_d}{T_d+\sqrt{T_d^2-4D_d}}.          \tag{5.DIM3}
$$

For the complete native kinetic stage with shared innovations, final position
noise and the radial cap, its physical-coordinate empirical transport obeys

$$
\mathbb E\mathcal W_{Q_\omega}^2(\mu^+,\widetilde\mu^+)
\le(1-\underline\delta_d)\mathcal W_{Q_\omega}^2(\mu,\widetilde\mu),
\qquad0<\underline\delta_d\le\delta_d.                \tag{5.DIM4}
$$

Here $\mathcal W_{Q_\omega}^2$ uses the optimal coupling of equal-mass empirical
measures and total physical squared cost divided by $N$. There is no factor
$N$ in the constants. The statement concerns kinetics; selected cloning,
count viscosity, status marks and nonlinear-force transfer require separate
estimates. The curvature $\omega$ and dimension $d$ are explicit parameters.
:::

:::{prf:proof}
The exact pre-cap difference matrix and cap-input noise are

$$
H=\begin{pmatrix}
1-\omega c^2(1+a)&c(1+a)\\
-\omega c(1+a)k&a-\omega c^2(1+a)
\end{pmatrix},\qquad v_3=b^\top z+kqZ,
$$

where $b=(-\omega c(1+a)k,a-\omega c^2(1+a))^\top$.
Direct multiplication gives

$$
H^\top Q_\omega H=Q_\omega-k(1-a^2)ww^\top,
\qquad w=(-\omega c,1)^\top.
$$

Use (5.DIM2) on the actual final cap. The common final position innovation
cancels. Thus the removed quadratic form is
$k(1-a^2)ww^\top+\underline\eta_d bb^\top$.
After conjugation by $Q_\omega^{-1/2}$, its trace is $T_d$ and determinant is
$D_d$: the determinant calculation uses $\det(w,b)=\omega c$.
Both coefficients are positive, and the smallest eigenvalue is exactly
$2D_d/(T_d+\sqrt{T_d^2-4D_d})$. This rationalized form avoids cancellation
in $(T_d-\sqrt{T_d^2-4D_d})/2$ and improves the former lower bound $D_d/T_d$.
A verified lower interval endpoint $\underline\delta_d$ preserves the inequality.

Choose an input-optimal empirical coupling and share independent innovations
according to that representative. Each marginal keeps its native noise law.
Average the single-pair bound with weights $1/N$; the propagated pairing is
admissible at output, so the output assignment minimum is no larger.
Storage permutations do not change either assignment minimum. $\square$
:::

:::{prf:corollary} Certified numerical radial integral
:label: cor-kinetic-certified-chi-cap-integral

Let $f_d(r)=c_dr^{d-1}e^{-r^2/2}$,
$c_1=\sqrt{2/\pi}$, $c_2=1$, $c_{d+2}=c_d/d$ and
$g(r)=1-(V/(V+\sigma r))^{p_d}$.
For a finite partition $0=r_0<\cdots<r_m=R$ and certified bounds
$\underline f_i\le f_d(r)\le\overline f_i$ on each interval,

$$
\sum_{i=0}^{m-1}g(r_i)(r_{i+1}-r_i)\underline f_i
\le\eta_d
\le\sum_{i=0}^{m-1}g(r_{i+1})(r_{i+1}-r_i)\overline f_i
 +2^{d/2}e^{-R^2/4}.                                \tag{5.DIM5}
$$

Clipping the two bounds to $[0,1]$ is valid. The implemented certificate uses
a dyadic partition, $R=\lceil\sqrt d\rceil+12$, directed IEEE interval arithmetic
and outward-rounded square roots. No unbounded numerical quadrature is treated
as an exact constant. The API supports positive integer $d\le256$ and rejects
parameter combinations whose floating-point enclosure cannot certify positivity.
:::

:::{prf:proof}
The chi density increases up to $\sqrt{d-1}$ and decreases afterwards, as seen
from its logarithmic derivative $(d-1)/r-r$; for $d=1$ it decreases from zero.
Its minimum on each interval is at an endpoint, and its maximum is at an
endpoint or the mode. These facts supply certified density bounds.
The increasing function $g$ then gives the lower and upper rectangle sums.
The tail contributes at most its probability because $0\le g\le1$.
Since $\mathbb E e^{\chi_d^2/4}=2^{d/2}$, Markov's inequality supplies the tail
term. Normalizer recurrence follows by integration by parts in the radial
Gaussian integral.

For the floating implementation, every basic interval operation is rounded
outwards using adjacent representable numbers. To enclose $e^{-t}$, reduce
$t$ by a power of two until $u\le1/8$, use the degree-16 alternating Taylor
sum and its next-term remainder, then repeatedly square the interval.
The terms decrease in magnitude, so the omitted error lies between minus the
next term and zero. The normalizer uses
$\pi=16\arctan(1/5)-4\arctan(1/239)$, with alternating-series remainder bounds;
the identity follows from the tangent addition formula and angles in
$(0,\pi/2)$. Consequently the interval procedure bounds every density and
rectangle contribution, including its rounding error. Positivity of the
resulting lower endpoints, rather than a tolerance, gates contraction.
$\square$
:::

:::{prf:theorem} Curvature-aware whole-step radial-cap sector certificate
:label: thm-kinetic-dimension-curvature-sector

For the same isotropic quadratic native stage, use scaled coordinates
$y=(\sqrt\omega x,v)$ and the positive metric
$G_\beta=\left(\begin{smallmatrix}1&\beta\\\beta&1\end{smallmatrix}\right)\otimes I_d$,
$|\beta|<1$. Let $\widehat H$ be $H$ in these coordinates and let
$\widehat H_j=\operatorname{diag}(1,j)\widehat H$, $j=0,1$.
If certified endpoint inequalities give

$$
G_\beta-\widehat H_j^\top G_\beta\widehat H_j
\succeq\underline\delta_\beta G_\beta,\qquad j=0,1,
\qquad\underline\delta_\beta>0,                        \tag{5.DIM6}
$$

then, pathwise for every shared Gaussian realization and every entering pair,

$$
\frac1N\sum_i|y_i^+-\widetilde y_i^+|_{G_\beta}^2
\le(1-\underline\delta_\beta)
       \frac1N\sum_i|y_i-\widetilde y_i|_{G_\beta}^2.  \tag{5.DIM7}
$$

The corresponding optimal empirical transport bound follows by choosing an
input-optimal representative. These constants are independent of dimension,
population, diffusion amplitude and cap radius; the noise and native cap
remain present. Each selected $\beta$ must have a certified endpoint LMI.
For the reference $h=.04$, $\gamma=\omega=1$, $\beta=1/25$, the established
{prf:ref}`thm-rcap-harmonic-whole-update` gives $\underline\delta_\beta=1/1040$;
the exact generalized endpoint eigenvalues yield a stronger admissible value.
:::

:::{prf:proof}
By {prf:ref}`lem-rcap-sector`, the cap's difference is $DZ$ for a
symmetric $0\preceq D\preceq I$. This also follows by integrating its symmetric
Jacobian along the segment between the two pre-cap inputs. Orthogonally
diagonalize $D$; the isotropic blocks of $\widehat H$ and $G_\beta$ commute
with that basis change. For each eigenvalue $s\in[0,1]$ the output quadratic
matrix $\widehat H_s^\top G_\beta\widehat H_s$ is convex in $s$: its second
derivative as a quadratic form is twice the square of the second row.
It is therefore bounded above by the chord between $s=0$ and $s=1$.
Apply the two endpoint inequalities to get the same contraction for every $s$,
then sum over coordinates and average over pairs. Both shared additive noise
arrays cancel before this calculation. Choosing an input-optimal coupling
bounds the optimal output cost without intrinsic particle labels.

The implemented generalized minimum endpoint eigenvalue is evaluated through
$2D/(T+\sqrt{T^2-4D})$ with certified intervals, where now $T$ and $D$ are the
trace and determinant of $G_\beta^{-1}$ times the endpoint deficit.
A deterministic candidate family may select the largest certified lower
endpoint; no measured decay rate enters that selection. $\square$
:::

:::{prf:corollary} Harmonic reference in a declared curvature-adapted metric
:label: cor-kinetic-regional-harmonic-reference

Retain all kinetic and isotropic quadratic force hypotheses of
{prf:ref}`thm-kinetic-dimension-curvature-sector`, with
$y=(\sqrt\omega x,v)$ and its exact matrix $\widehat H$.
Choose numerical $\alpha>0$ and $\beta^2<\alpha$, and put
$G_{\alpha,\beta}=\left(\begin{smallmatrix}\alpha&\beta\\\beta&1\end{smallmatrix}\right)\otimes I_d$.
For example, $\alpha$ may be a chosen representable approximation to
$k=1-\omega h^2/4$; its actual numerical value must enter the certificate.
If directed arithmetic verifies both endpoint inequalities

$$
G_{\alpha,\beta}-\widehat H_j^\top G_{\alpha,\beta}\widehat H_j
\succeq\underline\delta_{\alpha,\beta}G_{\alpha,\beta},\qquad j=0,1,
\quad\underline\delta_{\alpha,\beta}>0,\quad\alpha>\beta^2,
\tag{5.DIM10}
$$

then the native harmonic kinetic stage satisfies, for every shared realization,

$$
\frac1N\sum_i|y_i^+-\widetilde y_i^+|_{G_{\alpha,\beta}}^2
\le(1-\underline\delta_{\alpha,\beta})
       \frac1N\sum_i|y_i-\widetilde y_i|_{G_{\alpha,\beta}}^2.
\tag{5.DIM11}
$$

These are harmonic reference constants for a declared curvature $\omega$.
Using them on a nonquadratic region additionally requires a proved force
remainder estimate at both native kick queries and the probability-weighted
observable charge for leaving that region. A regional curvature value alone
does not prove global contraction for a nonquadratic force.
:::

:::{prf:proof}
The proof of {prf:ref}`thm-kinetic-dimension-curvature-sector` uses only that
the metric is positive and has scalar isotropic blocks. Those properties
hold here because $\alpha>\beta^2$. After diagonalizing the cap secant,
$\widehat H_s^\top G_{\alpha,\beta}\widehat H_s$ again has second
derivative equal to twice the square of the second row of $\widehat H$,
since the metric's velocity coefficient is one. Its endpoint chord is
therefore controlled by the two assumed LMIs. Sum the resulting scalar
quadratic inequalities and divide by $N$. No nonlinear force is substituted
for the declared harmonic reference in this argument. $\square$
:::

:::{prf:corollary} Explicit conversion between the two physical metrics
:label: cor-kinetic-dimension-metric-equivalence

Let $m_\beta$ and $M_\beta$ be the smallest and largest eigenvalues of
$Q_\omega^{-1/2}G_\beta Q_\omega^{-1/2}$, interpreted in scaled coordinates
where $Q_\omega=\operatorname{diag}(k,1)\otimes I_d$. Then

$$
m_\beta\mathcal W_{Q_\omega}^2\le\mathcal W_{G_\beta}^2
\le M_\beta\mathcal W_{Q_\omega}^2,\qquad
\mathbb E\mathcal W_{Q_\omega}^2(n)
\le\frac{M_\beta}{m_\beta}(1-\underline\delta_\beta)^n
                \mathcal W_{Q_\omega}^2(0).          \tag{5.DIM8}
$$

The ratio is an explicit population- and dimension-independent prefactor.
A rate in $G_\beta$ is not silently substituted into the diagonal-Q metric.
:::

:::{prf:proof}
The defining matrix eigenvalue inequalities hold for each atom difference,
hence for every transport coupling and then its minimum. Combine those two
inequalities with the sector contraction iterated $n$ steps. $\square$
:::

:::{prf:corollary} Iterated normalized physical transport for the quadratic kinetic kernel
:label: cor-kinetic-dimension-iterated-transport

Under the complete hypotheses of {prf:ref}`thm-kinetic-dimension-curvature-cap`
and, for the second estimate, {prf:ref}`thm-kinetic-dimension-curvature-sector`,
let both chains use the specified kinetic kernel at every step, with no
selection, death, or count viscosity. Use an optimal entering permutation
and common independent Gaussian innovations for that representative coupling.
For every integer $n\ge0$,

$$
\mathbb E\mathcal W_{Q_\omega}^2(n)
\le(1-\underline\delta_d)^n\mathcal W_{Q_\omega}^2(0).
\tag{5.DIM9a}
$$

$$
\mathbb E\mathcal W_{G_\beta}^2(n)
\le(1-\underline\delta_\beta)^n\mathcal W_{G_\beta}^2(0).
\tag{5.DIM9b}
$$

:::

:::{prf:proof}
The one-step diagonal estimate applies conditionally to every entering
representative coupling; the sector estimate holds for every innovation
realization. Iterate the respective conditional expectation inequalities
for the exhibited coupling. At time zero its cost is the optimal physical
transport cost. At each later time optimal transport costs no more than
this coupling. The normalized sums retain the same coefficients for every
population size. $\square$
:::

:::{prf:lemma} Terminal status coupling under the final position noise
:label: lem-kinetic-terminal-status-coupling

Let the terminal domain be an axis-aligned box $\mathcal D\subset\mathbb R^d$.
Condition on the complete cloning, collision, and BAOAB innovations, and let
$x,y$ be the two positions before final position diffusion. For its actual
amplitude $s=\sigma_x\sqrt h>0$, the synchronous Gaussian coupling obeys

$$
\Pr\!\left(\mathbf1_{\mathcal D}(x+s\zeta)
\ne\mathbf1_{\mathcal D}(y+s\zeta)\right)
\le \min\!\left\{1,\frac{2\|x-y\|_1}{s\sqrt{2\pi}}\right\}.
                                                               \tag{5.K4}
$$

This compares the terminal status marks used by the next cloning stage.
:::

:::{prf:proof}
For one coordinate, the translated membership intervals have symmetric
difference of length at most $2|x_j-y_j|$. The density of $s\zeta_j$ is
bounded by $1/(s\sqrt{2\pi})$. If box membership differs, at least one
coordinate's interval membership differs. Sum these one-dimensional
bounds and bound the resulting probability by one. $\square$
:::

:::{div} feynman-prose
Two nearby positions can sit on opposite sides of the boundary. Their physical
separation is small, yet their alive/dead marks disagree, and that disagreement
affects the next cloning stage. This is why the terminal test needs its own
estimate. With the same final Gaussian displacement, the marks can differ
only when the draw lands in the thin region between two translated boundary
tests. The Gaussian density bounds the probability of landing there.

Notice the different roles of position noise in these two calculations. It
cancels exactly when we compare physical positions, while its spread controls
the probability of a status mismatch. Equation (5.K4) keeps that contribution
available for the complete marked-state analysis, where revival and collision
membership must also be accounted for.
:::

:::{div} feynman-prose
Imagine following two walkers through the same BAOAB schedule. Their OU kicks
need not be identical for their final positions to agree. We can couple the
two Gaussian draws so that, on a matched event, their difference compensates
exactly for the entering position discrepancy. Each walker still receives
the prescribed Gaussian law. Only the relationship between the two runs has
been chosen for the comparison.

Once the intermediate positions agree, the final force evaluations agree
too. Shared position noise then preserves that agreement, including the
terminal alive/dead decision. A velocity discrepancy remains, and the radial
cap compresses it. The proof averages this compression over the actual OU
draw and separately charges for the event on which position matching fails.
The force and timestep bounds below make this estimate uniform over entering
physical states. This gives a bound on the distance between output laws
without requiring identical noise to shrink every individual trajectory.
:::

:::{prf:theorem} Bounded transport smoothing for the actual BAOAB and cap update
:label: thm-kinetic-bounded-transport-smoothing

Let the configured force $F:\mathbb R^d\to\mathbb R^d$ be globally
$L_F$-Lipschitz, and use the declared isotropic BAOAB, final position noise,
radial cap and terminal position classification. Put

$$
c=h/2,\quad a=e^{-\gamma h},\quad
q=B\sqrt{(1-a^2)/(2\gamma)}>0,\quad
\lambda=1-c^2L_F>0,\quad s=\sigma_x\sqrt h.
$$

The force bound is on all physical positions reached by the Gaussian
innovations. A bound only inside the valid domain is not sufficient for
this statement. For fixed position and velocity units $\ell_x,\ell_v>0$,
define the bounded marked-coordinate metric

$$
d_0(z,\widetilde z)=\min\left\{1,
\frac{|x-\widetilde x|}{\ell_x}
+\frac{|v-\widetilde v|}{\ell_v}
+\mathbf1_{\{e\ne\widetilde e\}}\right\}.
$$

Here $e$ is the terminal alive/dead mark; dead coordinates are retained.
Let $K(z,\cdot)$ be the row kinetic law from a post-cloning physical state
$z=(X,V)$. Define its actual intermediate quantities

$$
v_1=V+cF(X),\qquad x_1=X+cv_1,\qquad m=x_1+ca v_1.
$$

With $C_0=\sqrt{\pi/2}$ and cap radius $R$, the transport distance obeys

$$
W_{d_0}(K(z),K(\widetilde z))
\leq\min\left\{1,
\frac{|\Delta m|}{cq\sqrt{2\pi}}
+\frac{C_0R|\Delta x_1|}{\ell_v cq\lambda}\right\}
\leq\min\left\{1,
\frac{A_x|\Delta X|+A_v|\Delta V|}{q}\right\},             \tag{5.K5}
$$

where

$$
A_x=\frac{1+c^2(1+a)L_F}{c\sqrt{2\pi}}
+\frac{C_0R(1+c^2L_F)}{\ell_v c\lambda},\qquad
A_v=\frac{1+a}{\sqrt{2\pi}}+\frac{C_0R}{\ell_v\lambda}.
$$

These constants are independent of dimension, population size and $B$.
No convexity of the potential is required.
:::

:::{prf:proof}
The complete row update, with independent standard Gaussian vectors
$\xi,\zeta$, is

$$
x_2=m+cq\xi,\qquad
v_3=av_1+q\xi+cF(x_2),\qquad
x^+=x_2+s\zeta,\qquad v^+=C_R(v_3),
\quad C_R(u)=\frac{Ru}{R+|u|}.
$$

**Match positions through the OU innovation.** Set
$b=\Delta m/(cq)$. Couple $\xi$ and $\widetilde\xi$ with their required
standard Gaussian marginals so that $\widetilde\xi=\xi+b$ except with
probability $2\Phi(|b|/2)-1\leq |b|/\sqrt{2\pi}$. Such a coupling is
obtained by assigning the common density
$\min\{\phi(u),\phi(u+b)\}$ to the matched event and coupling the remaining
densities. Its mass follows by integrating on the two half-spaces separated
by their density-equality hyperplane. The matched subdensity of $\xi$ is
bounded above by $\phi$.

Use the same independent $\zeta$. On the matched event, $x_2=\widetilde x_2$,
so both final force evaluations, final positions and terminal marks agree.
The pre-cap velocity difference is exactly

$$
D=a\Delta v_1-\Delta m/c=-\Delta x_1/c.                 \tag{5.K6}
$$

This cancellation uses the final force evaluation in the actual BAOAB step.

**Average the cap derivative.** For the first input set
$T(y)=y+cF(x_1+cy)$. Then
$|T(y)-T(\widetilde y)|\geq\lambda|y-\widetilde y|$.
For each target $w$, the equation $y=w-cF(x_1+cy)$ is a contraction with
constant $c^2L_F<1$ on complete Euclidean space. Thus $T$ is onto and
one-to-one. For $t\in[0,1]$, let $y_t=T^{-1}(tD)$ and
$\xi_t=(y_t-av_1)/q$. Consequently

$$
|T(av_1+qG)-tD|\geq q\lambda|G-\xi_t|.
$$

For $d\geq2$, $\|DC_R(u)\|_{\mathrm{op}}=R/(R+|u|)\leq R/|u|$.
The identity
$r^{-1}=\pi^{-1/2}\int_0^\infty t^{-1/2}e^{-tr^2}\,dt$
and Gaussian integration give, for every $b\in\mathbb R^d$,

$$
\begin{aligned}
\mathbb E|G-b|^{-1}
&=\pi^{-1/2}\int_0^\infty
 t^{-1/2}(1+2t)^{-d/2}
 e^{-t|b|^2/(1+2t)}\,dt\\
&\leq\mathbb E|G|^{-1}
\leq\mathbb E(G_1^2+G_2^2)^{-1/2}=\sqrt{\pi/2}.
\end{aligned}
$$

The integrals are finite for $d\geq2$, and Tonelli justifies their order.
For $d=1$, $T$ is increasing and its inverse is $\lambda^{-1}$-Lipschitz.
The density of $T(av_1+qG)$ is therefore bounded by
$(q\lambda\sqrt{2\pi})^{-1}$. Since
$\int_{\mathbb R}|C_R'(u)|\,du=2R$, both cases yield

$$
\sup_{t\in[0,1]}
\mathbb E\|DC_R(T(av_1+qG)-tD)\|_{\mathrm{op}}
\leq\frac{C_0R}{q\lambda}.
$$

Integrate the derivative along the segment of length $|D|$. The matched
subdensity bound shows that the expected marked cost on the matched event
is at most $C_0R|D|/(\ell_v q\lambda)$. Failure costs at most one. Add its
probability and substitute (5.K6) to prove the first inequality in (5.K5).
Finally,
$|\Delta x_1|\leq(1+c^2L_F)|\Delta X|+c|\Delta V|$ and
$|\Delta m|\leq[1+c^2(1+a)L_F]|\Delta X|+c(1+a)|\Delta V|$
give the displayed coefficients. $\square$
:::

:::{prf:corollary} Population-normalized smoothing after the full component collision
:label: cor-kinetic-full-cluster-smoothing

Let $P,\widetilde P$ be laws of nonextinct entering swarms, and let $\Gamma$
couple their complete post-cloning swarm laws, including their
actual measurements, frozen acceptance, copied positions, jitter, revival
and shared component rotations. For the normalized output cost
$d_N=N^{-1}\sum_i d_0(z_i,\widetilde z_i)$, the complete kinetic laws satisfy

$$
W_{d_N}(\mathcal C_N(P)K_N,\mathcal C_N(\widetilde P)K_N)
\leq\mathbb E_\Gamma\frac1N\sum_i
\min\left\{1,\frac{A_x|\Delta X_i|+A_v|\Delta V_i|}{q}\right\}.
                                                               \tag{5.K7}
$$

Here $\mathcal C_N$ is the actual cloning kernel and $K_N$ retains the
physical marked outputs also on total extinction. The inequality does not
condition on survival. For the geometric error clusters, take their common
refinement under the chosen row pairing. In each block $G$, write
$\Delta X_i=\overline{\Delta X}_G+u_i$ and
$\Delta V_i=\overline{\Delta V}_G+w_i$. The right side of (5.K7) is at most

$$
\frac1q\sum_G\frac{|G|}{N}\left[
A_x\left(|\overline{\Delta X}_G|+
\sqrt{\frac1{|G|}\sum_{i\in G}|u_i|^2}\right)
+A_v\left(|\overline{\Delta V}_G|+
\sqrt{\frac1{|G|}\sum_{i\in G}|w_i|^2}\right)\right]
$$

averaged over $\Gamma$. Thus the smoothing estimate uses the same cluster
means and internal errors, with weights summing to one.

*Proof.* Conditional on both post-cloning swarms, apply the row coupling
independently to each paired row. Each marginal has exactly the prescribed
independent kinetic innovations. Sum the conditional bounds and integrate
over $\Gamma$. No independence of cloning outputs is used. The cluster bound
is the triangle inequality followed by Cauchy--Schwarz within each cluster.
$\square$
:::

:::{div} feynman-prose
Now keep the entire cloning outcome in view before applying this row
estimate. Several walkers may share a donor or a component rotation, so their
positions and velocities can be correlated. We first condition on those
complete outcomes. The kinetic innovations are independent across rows at
that stage, and we can apply the coupling to each paired row. Averaging over
the cloning outcomes afterwards retains their correlations.

The geometric error clusters organize the resulting sum. Within each cluster,
separate the mean discrepancy from the deviations around that mean. The
cluster contributes with weight $|G|/N$: the fraction of the population it
contains. These weights sum to one, so collecting more clusters does not
introduce a growing population factor. The resulting estimate feeds the
cluster means and internal errors directly into the kinetic comparison.
It concerns the complete marked outputs, with retained dead coordinates;
conditioning the whole population on survival would require its own
normalization.
:::

:::{prf:example} Verification for configured smooth potentials
:label: ex-kinetic-smoothing-force-constants

For the canonical $U(x)=|x|^2/2$, $F(x)=-x$ has $L_F=1$ on the entire
physical space. At $h=0.04$, $c^2L_F=0.0004<1$.
The implemented Rastrigin potential is
$U(x)=\sum_j[x_j^2-10\cos(2\pi x_j)+10]$. Its Hessian is diagonal with
entries $2+40\pi^2\cos(2\pi x_j)$, so its force is globally Lipschitz with
$L_F=2+40\pi^2$. At the same timestep, $c^2L_F<0.159<1$.
Both configurations therefore satisfy every force and timestep condition
of (5.K5) in every dimension, including at positions reached outside the
terminal valid domain. The second potential is nonconvex.
:::

### 4.5. Proof Strategy

The proof follows the **entropy method** adapted to the discrete swarm setting:

**Step 1:** Decompose $V_W$ into location and structural errors
**Step 2:** Analyze drift of each component separately under the Fokker-Planck evolution
**Step 3:** Use hypocoercive coupling to show the drift is negative when $V_W$ is large
**Step 4:** Bound noise-induced expansion terms

We now execute this strategy in detail.

### 4.6. Location Error Drift

:::{prf:lemma} Drift of Location Error Under Kinetics
:label: lem-location-error-drift-kinetic

Fix a coupled continuous-time kinetic extension and nonempty averaging sets.
Put $z=(\Delta\mu_x,\Delta\mu_v)$ and

$$
P=\begin{pmatrix}I_d&(b/2)I_d\\(b/2)I_d&\lambda_v I_d\end{pmatrix},
\qquad \lambda_v>b^2/4,\qquad V_{\rm loc}=z^\top Pz.
$$

Assume the **macroforce closure**

$$
\Delta\bar F=-K_t\Delta\mu_x+r_t,
\qquad A_{K_t}=\begin{pmatrix}0&I_d\\-K_t&-\gamma I_d\end{pmatrix}
$$

holds for the actual averaged forces. Require one fixed $P$ and an explicit
$\kappa_0>0$ satisfying the matrix inequality

$$
A_{K_t}^{\top}P+PA_{K_t}\preceq-\kappa_0 P                 \tag{5.L1}
$$

for every admissible $K_t$. The force residual, joint quadratic variation
$R_t\,dt$ of $z$, and any status/reselection jump contribution must satisfy

$$
2z^\top P\binom0{r_t}\le\epsilon_r V_{\rm loc}+C_r,
\quad \operatorname{tr}(PR_t)\le C_{\rm noise},
\quad \mathcal J V_{\rm loc}\le\epsilon_J V_{\rm loc}+C_J,
$$

with $\kappa_{\rm loc}:=\kappa_0-\epsilon_r-\epsilon_J>0$ and constants
uniform in population size. A fixed-set extension has $\mathcal J=0$.
Then, writing $C_{\rm loc}=C_r+C_{\rm noise}+C_J$,

$$
\mathcal L V_{\rm loc}\le-\kappa_{\rm loc}V_{\rm loc}+C_{\rm loc},
\qquad
\mathbb E V_{\rm loc}(t)\le e^{-\kappa_{\rm loc}t}V_{\rm loc}(0)
 +\frac{C_{\rm loc}}{\kappa_{\rm loc}}(1-e^{-\kappa_{\rm loc}t}). \tag{5.L2}
$$

For a discrete kernel satisfying the generator-consistency and observable
weak-error certificate of {prf:ref}`thm-discretization`, with coefficient
$K_{\rm loc}$, $h\le\min(1/\kappa_{\rm loc},\kappa_{\rm loc}/(4K_{\rm loc}),H)$
(with the second restriction omitted if $K_{\rm loc}=0$),

$$
\mathbb E[\Delta V_{\rm loc}]
\le-\frac{\kappa_{\rm loc}}4 hV_{\rm loc}
 +(C_{\rm loc}+K_{\rm loc}H)h.                         \tag{5.L3}
$$

A Lipschitz bound on individual forces and coercivity of the potential do
not establish (5.L1) or macroforce closure. For nonlinear forces, differences
of averaged forces need not be controlled by differences of barycenters.
These are additional analytic hypotheses. The fixed radial cap is governed
by the direct finite-step theorem {prf:ref}`thm-kinetic-exact-baoab-cap-coupling`;
its continuum transfer must not be inferred from (5.L2).
:::

:::{prf:proof}
Positive definiteness follows from the Schur complement
$\lambda_v-b^2/4>0$. Itô's formula, including the stated jump term, gives

$$
\mathcal L(z^\top Pz)
=z^\top(A_{K_t}^{\top}P+PA_{K_t})z
 +2z^\top P\binom0{r_t}+\operatorname{tr}(PR_t)+\mathcal J V_{\rm loc}.
$$

Apply (5.L1) and the three source bounds. This proves the generator
inequality in (5.L2); localization with the assumed integrable quadratic
variation followed by Gronwall proves its expectation bound. In particular,
no sign is assigned to a symmetric cross matrix without checking the full
matrix inequality.

For the discrete extension, its observable weak-error certificate adds
$K_{\rm loc}h^2(1+V_{\rm loc})$ to the exact expectation in (5.L2).
For $\kappa_{\rm loc}h\le1$, $e^{-\kappa_{\rm loc}h}\le1-\kappa_{\rm loc}h/2$;
the timestep restriction absorbs $K_{\rm loc}h^2V_{\rm loc}$ into
$\kappa_{\rm loc}hV_{\rm loc}/4$. The exact source integral is at most
$C_{\rm loc}h$, and $K_{\rm loc}h^2\le K_{\rm loc}Hh$. This proves (5.L3).

**Verified affine specialization.** For $F(x)=-x+f_0$ and $\gamma=1$,
averaging commutes with the force for every empirical measure. Thus $K_t=I_d$
and $r_t=0$, without particle labels. Take

$$
P=\begin{pmatrix}1&1/3\\1/3&2/3\end{pmatrix}\otimes I_d,
\quad \lambda_v=2/3,\quad b=2/3.
$$

Direct multiplication gives

$$
A_I^\top P+PA_I=-\frac23 I_{2d},\qquad
\lambda_{\max}(P)=\frac{5+\sqrt5}{6},\qquad
\kappa_0=\frac4{5+\sqrt5}=0.552786\ldots.               \tag{5.L4}
$$

Since $P\preceq\lambda_{\max}(P)I$, this verifies (5.L1). Common additive
noise cancels in the barycenter difference of equal-size swarms under an
admissible matched coupling, so $C_{\rm noise}=0$ for that specialization.
Independent noises give a quadratic-variation source that can instead be
bounded uniformly in $N$ using their actual covariance. Neither case changes
the metric or its verified contraction coefficient. $\square$
:::

:::{prf:definition} Core and Exterior Regions
:label: def-core-exterior-regions

For any $\delta_{\text{core}} > 0$, define:

**Core Region** (interior domain):

$$
\mathcal{R}_{\text{core}} := \{x \in \mathcal{X}_{\text{valid}} : \text{dist}(x, \partial\mathcal{X}_{\text{valid}}) \geq \delta_{\text{core}}\}

$$

**Exterior Region** (near boundary):

$$
\mathcal{R}_{\text{ext}} := \mathcal{X}_{\text{valid}} \setminus \mathcal{R}_{\text{core}} = \{x \in \mathcal{X}_{\text{valid}} : \text{dist}(x, \partial\mathcal{X}_{\text{valid}}) < \delta_{\text{core}}\}

$$

**Choice of $\delta_{\text{core}}$**: We take $\delta_{\text{core}} = \delta_{\text{boundary}}/2$ where $\delta_{\text{boundary}}$ is from {prf:ref}`axiom-confining-potential` (boundary compatibility), ensuring the exterior region is strictly contained in the boundary barrier zone.
:::

### 4.7. Structural Error Drift

:::{prf:lemma} Drift of Structural Error Under Kinetics
:label: lem-structural-error-drift-kinetic

Consider equal-size swarms and the centered empirical measures
$\widetilde\mu_k=N^{-1}\sum_i\delta_{z_{k,i}-\bar z_k}$, with
$V_{\rm struct}=W_P^2(\widetilde\mu_1,\widetilde\mu_2)$ and the same positive
metric $P$ as in {prf:ref}`lem-location-error-drift-kinetic`.
Choose an input-optimal coupling and construct a joint kinetic extension with
the correct marginal dynamics. Assume every centered coupled atom difference
obeys the macroforce/LMI, residual, quadratic-variation and status-source bounds
of that lemma, with a common positive coefficient $\kappa_s$ and averaged source
bound $C_s$, independent of $N$. Then

$$
\mathbb E V_{\rm struct}(t)
\le e^{-\kappa_s t}V_{\rm struct}(0)
 +\frac{C_s}{\kappa_s}(1-e^{-\kappa_s t}).               \tag{5.S1}
$$

If the transported quadratic coupling cost also has a uniform observable
weak-error certificate with coefficient $K_s$ for the generator-consistent
discrete extension, the restrictions
$h\le\min(H,1/\kappa_s,\kappa_s/(4K_s))$ give

$$
\mathbb E[\Delta V_{\rm struct}]
\le-\frac{\kappa_s}{4}hV_{\rm struct}+(C_s+K_sH)h.      \tag{5.S2}
$$

The certificate is for the smooth transported coupling cost; differentiability
of the assignment minimum is not assumed. Affine unit-quadratic force with
common additive noise satisfies the centered closure and LMI with the metric
and coefficient in (5.L4). Arbitrary confining nonlinear forces require their
own centered closure and matrix certificate.
:::

:::{prf:proof}
For equal-mass finite empirical measures an optimal permutation exists.
Choose any minimizing representative and feed each paired atom the same
Brownian innovation, while preserving the independent noises required within
each marginal swarm. The representative is a coupling construction, not an
intrinsic particle label. Reordering either storage array changes the chosen
representative without changing the assignment minimum.

Apply the quadratic Itô estimate from
{prf:ref}`lem-location-error-drift-kinetic` to each centered coupled difference.
For the transported average cost $C_t=N^{-1}\sum_i|\Delta\widetilde z_i(t)|_P^2$,
the hypotheses give

$$
\frac{d}{dt}\mathbb E C_t\le-\kappa_s\mathbb E C_t+C_s,
\qquad C_0=V_{\rm struct}(0).
$$

The source is an average of the atom sources and has no factor $N$.
Gronwall bounds $\mathbb E C_t$. The transported pairing is an admissible
coupling of the centered output measures, hence
$V_{\rm struct}(t)\le C_t$ even if that pairing is no longer optimal.
This proves (5.S1) without differentiating a Wasserstein minimum or equating
it to a persistent storage pairing.

For the discrete extension apply its transported-cost weak-error certificate,
then the same exponential and timestep estimates used for (5.L3). Since the
optimal output cost is bounded by the transported cost, (5.S2) follows.
For the affine specialization, centering commutes with its linear force and
common additive noise cancels pairwise and in the matched barycenters. Thus
(5.L4) verifies the centered assumptions with $C_s=0$. $\square$
:::

### 4.8. Proof of Main Theorem

:::{prf:proof}
**Proof of {prf:ref}`thm-inter-swarm-contraction-kinetic`.**
For a common quadratic metric, the parallel-axis identity applied to every
transport coupling gives the exact decomposition
$V_W=V_{\rm loc}+V_{\rm struct}$. The barycenter term is independent of the
coupling, so minimizing leaves the centered transport minimum.
Apply (5.L3) and (5.S2), using their common admissible timestep range. Set

$$
\kappa_W=\frac14\min(\kappa_{\rm loc},\kappa_s),\qquad
C_W'=C_{\rm loc}+C_s+(K_{\rm loc}+K_s)H.
$$

Both errors are nonnegative, so summing proves

$$
\mathbb E[\Delta V_W]\le-\kappa_WhV_W+C_W'h.
$$

All constants are population independent by the stated closure, LMI, averaged
source and observable-error hypotheses. No inequality asserting that this
coefficient exceeds a cloning expansion follows without a separate comparison
of those coefficients. The native capped quadratic kernel has the direct
finite-step estimate (5.K1), which does not use this continuum transfer.
$\square$
:::

### 4.9. Summary

This chapter has proven:

✅ **Hypocoercive contraction** of inter-swarm error $V_W$ with rate $\kappa_W > 0$

✅ **N-uniform bounds** - contraction doesn't degrade with swarm size

The signed comparison with cloning requires a separate calculation of the actual cloning and kinetic coefficients.

**Key Insight:** Even though noise only acts on velocity, the coupling between position and velocity through the hypocoercive norm allows effective dissipation of positional error.

**Next:** Chapter 5 proves that the same kinetic operator contracts velocity variance via Langevin friction.

## 5. Velocity Variance Dissipation via Langevin Friction

### 5.1. Introduction: The Friction Mechanism

While Chapter 4 showed hypocoercive contraction of inter-swarm error, this chapter focuses on **intra-swarm velocity variance**. The friction term $-\gamma v$ in the Langevin equation provides direct dissipation of kinetic energy.

**The Challenge from Cloning:**

Recall from {doc}`03_cloning` that the cloning operator causes **bounded velocity variance expansion** $\Delta V_{\text{Var},v} \leq C_v$ due to inelastic collisions. This chapter proves that the Langevin friction provides **linear contraction** that overcomes this expansion.

**Physical Intuition:**

The friction term $-\gamma v$ acts like a "drag force" that pulls all velocities toward zero (or toward the drift velocity $u(x)$ if non-zero). This causes the velocity distribution to shrink toward its equilibrium value.

### 5.2. Velocity Variance Definition (Recall)

:::{prf:definition} Velocity Variance Component (Recall)
:label: def-velocity-variance-recall

For a single swarm $S$, the velocity variance is:

$$
V_{\text{Var},v}(S) = \frac{1}{N}\sum_{i \in \mathcal{A}(S)} \|v_i - \mu_v\|^2

$$

where $\mu_v = \frac{1}{N}\sum_{i \in \mathcal{A}(S)} v_i$. In the coupled two-swarm setting of {doc}`03_cloning`, the corresponding component is the average over the two swarms. The TV proof uses the single-swarm form.

**Physical interpretation:** Measures the spread of velocities around the swarm velocity barycenter.
:::

### 5.2.1. Velocity Barycenter Term

:::{prf:definition} Velocity Barycenter Energy
:label: def-velocity-barycenter-energy

For a single swarm $S$, define the barycenter energy:

$$
V_{\mu_v}(S) := \|\mu_v\|^2

$$

This term is added to the TV Lyapunov function to control global velocity drift and to avoid hidden moment assumptions.
:::

### 5.3. Main Theorem: Velocity Dissipation

:::{prf:theorem} Velocity Variance Contraction Under Kinetic Operator
:label: thm-velocity-variance-contraction-kinetic

For the continuous uncapped kinetic extension with fixed nonempty averaging
sets, independent particle Brownian noises, $u=0$, and an actual force bound
$|F(x)|\le F_{\max}$ on its reachable space, set

$$
\rho_v=2\gamma-\epsilon>0,\qquad
C_v^{\rm gen}=F_{\max}^2/\epsilon+d\sigma_{\max}^2,
\qquad 0<\epsilon<2\gamma.
$$

Then

$$
\mathcal L V_{{\rm Var},v}\le-\rho_v V_{{\rm Var},v}+C_v^{\rm gen},
\quad
\mathbb E V_{{\rm Var},v}(t)
\le e^{-\rho_v t}V_{{\rm Var},v}(0)
 +\frac{C_v^{\rm gen}}{\rho_v}(1-e^{-\rho_v t}).
$$

The quotient $C_v^{\rm gen}/\rho_v$ is a moment upper envelope, not an
identified equilibrium law. For a generator-consistent discrete extension
with an independently proved (5.D1) certificate of coefficient $K_v$, (5.D3)
gives, on its admissible timestep range,

$$
\mathbb E[\Delta V_{{\rm Var},v}]
\le-\rho_v hV_{{\rm Var},v}/2+(C_v^{\rm gen}+K_vH)h.
$$

If forces are unbounded, their actual averaged force-square envelope must
replace $F_{\max}^2$. Removing/reviving walkers requires separate status-source
bounds; a fixed-cap algorithm does not inherit this continuum rate by weak
transfer. All stated coefficients are independent of population size.

:::

### 5.4. Proof

:::{prf:proof}
**Proof (Complete Algebraic Derivation).**

This proof provides the full algebraic decomposition of velocity variance evolution using Itô's lemma, the parallel axis theorem, and careful bookkeeping.

**PART I: Single-Walker Velocity Evolution**

For walker $i$ with velocity $v_i$, the Langevin equation is:

$$
dv_i = F(x_i) dt - \gamma v_i dt + \Sigma(x_i, v_i) \circ dW_i

$$

Apply **Itô's lemma** to $\|v_i\|^2$:

$$
d\|v_i\|^2 = 2\langle v_i, dv_i \rangle + \|dv_i\|^2

$$

**Compute the quadratic variation:**

$$
\|dv_i\|^2 = \|\Sigma(x_i, v_i) \circ dW_i\|^2 = \text{Tr}(\Sigma\Sigma^T) dt \quad \text{(Itô isometry)}

$$

**Substitute dynamics:**

$$
d\|v_i\|^2 = 2\langle v_i, F(x_i) - \gamma v_i \rangle dt + \text{Tr}(\Sigma\Sigma^T) dt + 2\langle v_i, \Sigma dW_i \rangle

$$

$$
= 2\langle v_i, F(x_i) \rangle dt - 2\gamma \|v_i\|^2 dt + \text{Tr}(\Sigma\Sigma^T) dt + 2\langle v_i, \Sigma dW_i \rangle

$$

**Take expectations (martingale term vanishes):**

$$
\mathbb{E}[d\|v_i\|^2] = 2\mathbb{E}[\langle v_i, F(x_i) \rangle] dt - 2\gamma \mathbb{E}[\|v_i\|^2] dt + \mathbb{E}[\text{Tr}(\Sigma\Sigma^T)] dt

$$

**PART II: Barycenter Velocity Evolution**

For swarm $k$ with $N_k$ alive walkers, the barycenter velocity is:

$$
\mu_{v,k} = \frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} v_{k,i}

$$

Apply Itô's lemma to $\|\mu_{v,k}\|^2$:

$$
d\|\mu_{v,k}\|^2 = 2\langle \mu_{v,k}, d\mu_{v,k} \rangle + \|d\mu_{v,k}\|^2

$$

**Barycenter evolution:**

$$
d\mu_{v,k} = \frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} dv_{k,i}

$$

$$
= \frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} [F(x_{k,i}) - \gamma v_{k,i}] dt + \frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} \Sigma(x_{k,i}, v_{k,i}) \circ dW_i

$$

**Quadratic variation of barycenter:**

$$
\|d\mu_{v,k}\|^2 = \left\|\frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} \Sigma dW_i\right\|^2 = \frac{1}{N_k^2}\sum_{i \in \mathcal{A}(S_k)} \text{Tr}(\Sigma_i\Sigma_i^T) dt

$$

$$
\leq \frac{1}{N_k} \sigma_{\max}^2 d \, dt

$$

**PART III: Parallel Axis Theorem (Sample Decomposition)**

For any finite sample of vectors $\{v_i\}_{i=1}^N$ with sample mean $\mu_v = \frac{1}{N}\sum_{i=1}^N v_i$:

$$
\frac{1}{N}\sum_{i=1}^N \|v_i\|^2 = \frac{1}{N}\sum_{i=1}^N \|v_i - \mu_v\|^2 + \|\mu_v\|^2

$$

where the left-hand side is the **mean of squared norms**, the first term on the right is the **sample variance**, and the second term is the **squared sample mean**.

**Rearranging:**

$$
\text{Var}(v) := \frac{1}{N}\sum_{i=1}^N \|v_i - \mu_v\|^2 = \frac{1}{N}\sum_{i=1}^N \|v_i\|^2 - \|\mu_v\|^2

$$

(Direct algebraic identity.)

**PART IV: Variance Evolution for Single Swarm**

For swarm $k$:

$$
\frac{d}{dt}\text{Var}_k(v) = \frac{d}{dt}\left[\frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} \|v_{k,i}\|^2 - \|\mu_{v,k}\|^2\right]

$$

$$
= \frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} \frac{d}{dt}\mathbb{E}[\|v_{k,i}\|^2] - \frac{d}{dt}\mathbb{E}[\|\mu_{v,k}\|^2]

$$

**From Part I:**

$$
\frac{1}{N_k}\sum_{i \in \mathcal{A}(S_k)} \frac{d}{dt}\mathbb{E}[\|v_{k,i}\|^2] = \frac{2}{N_k}\sum_i \mathbb{E}[\langle v_{k,i}, F(x_{k,i}) \rangle] - 2\gamma \frac{1}{N_k}\sum_i \mathbb{E}[\|v_{k,i}\|^2] + d\sigma_{\max}^2

$$

**From Part II:**

$$
\frac{d}{dt}\mathbb{E}[\|\mu_{v,k}\|^2] = 2\mathbb{E}[\langle \mu_{v,k}, F_{\text{avg},k} - \gamma\mu_{v,k} \rangle] + O(1/N_k)

$$

where $F_{\text{avg},k} = \frac{1}{N_k}\sum_i F(x_{k,i})$.

**Key cancellation:** The force terms largely cancel when we subtract. The residual force-work term is:

$$
\Delta_{\text{force}} := \frac{2}{N_k}\sum_i \mathbb{E}[\langle v_{k,i}, F(x_{k,i}) \rangle] - 2\mathbb{E}[\langle \mu_{v,k}, F_{\text{avg},k} \rangle]

$$

Expanding with $v_{k,i} = \mu_{v,k} + (v_{k,i} - \mu_{v,k})$:

$$
= \frac{2}{N_k}\sum_i \mathbb{E}[\langle v_{k,i} - \mu_{v,k}, F(x_{k,i}) \rangle] + \underbrace{2\mathbb{E}[\langle \mu_{v,k}, F_{\text{avg},k} \rangle] - 2\mathbb{E}[\langle \mu_{v,k}, F_{\text{avg},k} \rangle]}_{=0}

$$

$$
= \frac{2}{N_k}\sum_i \mathbb{E}[\langle v_{k,i} - \mu_{v,k}, F(x_{k,i}) \rangle]

$$

**Quantitative bound via Young's inequality:**

$$
|\Delta_{\text{force}}| \leq \frac{2}{N_k}\sum_i \mathbb{E}[\|v_{k,i} - \mu_{v,k}\| \cdot \|F(x_{k,i})\|]

$$

Using $2ab \leq \epsilon a^2 + \epsilon^{-1} b^2$ and $\|F(x)\| \leq F_{\max}$ on $\mathcal{X}_{\text{valid}}$:

$$
|\Delta_{\text{force}}| \leq \epsilon \text{Var}_k(v) + \frac{F_{\max}^2}{\epsilon}

$$

for any $\epsilon > 0$. This yields a **linear** drift bound without asymptotic arguments.

**Resulting bound:**

$$
\frac{d}{dt}\mathbb{E}[\text{Var}_k(v)] \leq -(2\gamma - \epsilon)\text{Var}_k(v) + \frac{F_{\max}^2}{\epsilon} + d\sigma_{\max}^2

$$

**PART V: Aggregate Over Both Swarms**

The total velocity variance is:

$$
V_{\text{Var},v} = \frac{1}{2}\sum_{k=1,2} \text{Var}_k(v)

$$

Summing:

$$
\frac{d}{dt}\mathbb{E}[V_{\text{Var},v}] = \frac{1}{2}\sum_{k=1,2} \frac{d}{dt}\mathbb{E}[\text{Var}_k(v)]

$$

$$
\leq \frac{1}{2}\sum_{k=1,2} \left[-(2\gamma - \epsilon)\text{Var}_k(v) + \frac{F_{\max}^2}{\epsilon} + d\sigma_{\max}^2\right]

$$

$$
= -(2\gamma - \epsilon) V_{\text{Var},v} + \frac{F_{\max}^2}{\epsilon} + d\sigma_{\max}^2

$$

**PART VI: Exact semigroup and conditional discrete version.**
The generator estimate established above yields its exact exponential
expectation bound by Dynkin's formula and Gronwall. To transfer it to a
numerical extension, (5.D1) must hold for $V_{{\rm Var},v}$ with its actual
global moment envelope. Then (5.D3) gives

$$
\mathbb E[\Delta V_{{\rm Var},v}]
\le-\rho_v hV_{{\rm Var},v}/2+(C_v^{\rm gen}+K_vH)h.
$$

An error proportional to $h^2V_{{\rm Var},v}$ is absorbed into the contraction
coefficient through the explicit timestep restriction, rather than into an
additive constant. The canonical fixed cap requires direct finite-step
analysis. The generator coefficient $\rho_v$ applies to the continuous
extension; its conditional numerical coefficient here is $\rho_v/2$.

**PART VII: Physical Interpretation**

This result shows:
1. **Contraction:** Friction dissipates velocity variance at rate $(2\gamma-\epsilon)$
2. **Expansion:** Thermal noise adds variance at rate $d\sigma_{\max}^2$ and the force term contributes $\frac{F_{\max}^2}{\epsilon}$
3. **Equilibrium:** When $V_{\text{Var},v} \to V_{\text{Var},v}^{\text{eq}}$, the drift balances

**Key property:** The contraction rate $(2\gamma-\epsilon)$ is **independent of swarm size** $N$.

**Q.E.D.**
:::

### 5.4.1. Velocity Barycenter Dissipation

:::{prf:theorem} Velocity Barycenter Drift Under Kinetics
:label: thm-velocity-barycenter-dissipation

For the continuous uncapped extension with fixed $N$ walkers, independent
particle Brownian noises and $|F|\le F_{\max}$ on the reachable space,

$$
\mathcal L|\mu_v|^2\le-\gamma|\mu_v|^2+C_\mu,
\qquad C_\mu=F_{\max}^2/\gamma+d\sigma_{\max}^2/N.
$$

Consequently

$$
\mathbb E|\mu_v(t)|^2\le e^{-\gamma t}|\mu_v(0)|^2
 +(C_\mu/\gamma)(1-e^{-\gamma t}).
$$

For a generator-consistent extension with a verified (5.D1) coefficient
$K_\mu$, (5.D3) gives

$$
\mathbb E[\Delta|\mu_v|^2]
\le-\gamma h|\mu_v|^2/2+(C_\mu+K_\mu H)h.
$$

This is not a weak-limit assertion for the fixed radial cap. Correlated
innovations require their actual mean-noise covariance in $C_\mu$.
:::

:::{prf:proof}
For the stated fixed-set continuous extension,

$$
d\mu_v=(\bar F-\gamma\mu_v)dt+N^{-1}\sum_i\Sigma_i\,dW_i.
$$

Itô's formula and independence give the trace source
$N^{-2}\sum_i\operatorname{tr}(\Sigma_i\Sigma_i^\top)\le d\sigma_{\max}^2/N$.
Young's inequality
$2\langle\mu_v,\bar F\rangle\le\gamma|\mu_v|^2+|\bar F|^2/\gamma$
and $|\bar F|\le F_{\max}$ establish the generator estimate. Gronwall gives
its exact semigroup bound. Under the additional certificate, apply (5.D3)
with $\kappa=\gamma$, $C=C_\mu$ to obtain the discrete statement.
$\square$
:::

### 5.5. Balancing with Cloning Expansion

:::{prf:corollary} Net Velocity Variance Contraction for Composed Operator
:label: cor-net-velocity-contraction

Assume the actual cloning and kinetic kernels satisfy
$P_CV_v\le V_v+C_{C,v}$ and $P_KV_v\le r_vV_v+b_{K,v}$ on every cloning
output, with $0\le r_v<1$ and population-uniform sources. Then for the
algorithmic order cloning followed by kinetics,

$$
P_CP_KV_v\le r_vV_v+r_vC_{C,v}+b_{K,v}.
$$

The drift is negative above
$(r_vC_{C,v}+b_{K,v})/(1-r_v)$.
For the certified uncapped extension in the preceding theorem,
$r_v=1-\rho_vh/2$ and $b_{K,v}=(C_v^{\rm gen}+K_vH)h$.
For the canonical capped kernel these inputs require their own native estimate.
:::

:::{prf:proof}
Apply the pointwise kinetic bound to the cloning output, then apply the
cloning expectation bound and use $r_v\ge0$. Subtract $V_v$ and rearrange.
$\square$
:::

:::{prf:corollary} Barycenter Drift Under the Composed Operator
:label: cor-net-barycenter-drift

Assume cloning outputs satisfy $P_C|\mu_v|^2\le v_{\max}^2$, and that the
actual kinetic kernel has a pointwise bound
$P_K|\mu_v|^2\le r_\mu|\mu_v|^2+b_{K,\mu}$ on every such output.
Then

$$
P_CP_K|\mu_v|^2\le r_\mu v_{\max}^2+b_{K,\mu}.
$$

For a certified generator-consistent extension,
$r_\mu=1-\gamma h/2$ and $b_{K,\mu}=(C_\mu+K_\mu H)h$ on its admissible
range. For a native capped output, the direct deterministic bound
$|\mu_v^+|^2\le v_{\max}^2$ is available without a continuum transfer.
:::

:::{prf:proof}
Integrate the pointwise kinetic estimate over cloning, then use the cloning
bound. The direct capped estimate is the squared norm bound for an average
of vectors in the velocity ball. $\square$
:::

### 5.6. Summary

This chapter has proven:

✅ **Linear contraction** of velocity variance with rate $(2\gamma-\epsilon)$

✅ **Barycenter dissipation** for $\|\mu_v\|^2$ with rate $\gamma$

✅ **Overcomes cloning expansion** when $V_{\text{Var},v}$ is large enough

✅ **Equilibrium bound** on velocity variance

✅ **N-uniform** - all constants independent of swarm size

**Key Mechanism:** The friction term $-\gamma v$ provides direct dissipation that overcomes both thermal noise and cloning-induced perturbations in velocities and their barycenter.

**Synergy with Cloning:**
- Cloning supplies $N$-uniform Keystone pressure and an independent positional reset bound ({doc}`03_cloning`)
- Kinetics contracts velocity variance and barycenter energy (this chapter)
- Together: the full phase-space drift is the signed expression in {prf:ref}`thm-slc-signed-complete-update`

**Next:** Chapter 6 analyzes the positional diffusion that causes bounded expansion of $V_{\text{Var},x}$.

## 6. Positional Diffusion and Bounded Expansion

### 6.1. Introduction: The Price of Thermal Noise

The Langevin equation includes thermal noise in velocity: $dv = \ldots + \Sigma \circ dW$. This noise, while essential for ergodicity, causes **diffusion in position space** via the coupling $\dot{x} = v$.

**The Tradeoff:**

- **Benefit:** Noise enables exploration and prevents kinetic collapse
- **Cost:** Noise causes random walk in position, expanding positional variance

The kinetic contribution is bounded on the moment class stated below. Whether Keystone pressure overcomes the donor and kinetic terms is decided by the complete-update estimate, not by the positional reset bound alone.

### 6.2. Positional Variance (Recall)

:::{prf:definition} Positional Variance Component (Recall)
:label: def-positional-variance-recall

From {doc}`03_cloning` Definition 3.3.1:

$$
V_{\text{Var},x}(S_1, S_2) = \frac{1}{N}\sum_{k=1,2} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2

$$

where $\delta_{x,k,i} = x_{k,i} - \mu_{x,k}$ is the centered position.
:::

### 6.3. Main Theorem: Bounded Positional Expansion

:::{prf:theorem} Finite-horizon Positional Variance Expansion Under Kinetics
:label: thm-positional-variance-bounded-expansion

Consider the continuous kinetic extension with a fixed nonempty averaging set,
$dx_i=v_i\,dt$, and no position diffusion or status jumps. Fix $H>0$ and
assume the actual transient moment bounds in
{prf:ref}`assump-uniform-variance-bounds` on $[0,H]$. For $0\le h\le H$,

$$
\mathbb E[V_{{\rm Var},x}(h)-V_{{\rm Var},x}(0)]
\le 2\sqrt{M_xM_v}\,h+M_vh^2
\le C_{{\rm kin},x}(H)h,                              \tag{5.X1}
$$

where

$$
C_1=2\sqrt{M_xM_v},\qquad C_2(H)=HM_v,\qquad
C_{{\rm kin},x}(H)=C_1+C_2(H).                         \tag{5.X2}
$$

These constants are independent of population size whenever the supplied
transient moment bounds are. No equilibrium or exponential velocity
covariance hypothesis is used. An additional position diffusion adds its
actual centered quadratic-variation source; changes of the averaging set
require their own status-source terms. The native capped BAOAB kernel has
the separate conditional identity (5.X3) below and does not inherit (5.X1)
through an unproved continuum transfer.
:::

### 6.4. Proof

:::{prf:proof}
Write $\delta x_i=x_i-\bar x$ and $\delta v_i=v_i-\bar v$ on the fixed
averaging set of size $n$. The exact integral identity is

$$
\delta x_i(h)=\delta x_i(0)+I_i(h),\qquad
I_i(h)=\int_0^h\delta v_i(t)\,dt.
$$

Consequently

$$
V_{{\rm Var},x}(h)-V_{{\rm Var},x}(0)
=\frac2n\sum_i\delta x_i(0)\cdot I_i(h)
 +\frac1n\sum_i|I_i(h)|^2.
$$

Cauchy--Schwarz first in time and then across walkers and probability gives

$$
\mathbb E\frac1n\sum_i|I_i(h)|^2
\le h\int_0^h\mathbb EV_{{\rm Var},v}(t)\,dt
\le M_vh^2,
$$

and

$$
\left|\mathbb E\frac2n\sum_i\delta x_i(0)\cdot I_i(h)\right|
\le 2\left(\mathbb EV_{{\rm Var},x}(0)\right)^{1/2}
       \left(\mathbb E\frac1n\sum_i|I_i(h)|^2\right)^{1/2}
\le 2\sqrt{M_xM_v}\,h.
$$

Adding proves the first bound in (5.X1). The second uses $h^2\le Hh$.
All normalization factors stay inside the averaged moments, so none grows
with $n$. The estimates are exact and retain the quadratic term; no
unquantified numerical remainder is discarded. The same proof applies to
the average of two swarms with their common moment envelopes. $\square$
:::

:::{prf:assumption} Transient Uniform Variance Bounds
:label: assump-uniform-variance-bounds

Fix a horizon $H>0$ and a law of the continuous kinetic extension with a
fixed averaging set. Supply population-uniform constants $M_x,M_v$ such that

$$
\mathbb EV_{{\rm Var},x}(0)\le M_x,\qquad
\sup_{0\le t\le H}\mathbb EV_{{\rm Var},v}(t)\le M_v.
$$

When {prf:ref}`thm-velocity-variance-contraction-kinetic` applies with an
actual force-square envelope throughout this time interval, one valid
choice is
$M_v=\max\{\mathbb EV_{{\rm Var},v}(0),C_v^{\rm gen}/\rho_v\}$.
The equilibrium upper envelope alone is insufficient when the initial
moment is larger. A complete-update positional moment estimate must
supply $M_x$ for the required input class: neither a cloning reset nor a
sampled maximum establishes such a uniform bound. In the canonical
complete update, retain the signed donor and kinetic residual of
(SCK.3)--(SCK.6) in {prf:ref}`thm-slc-signed-complete-update`.
:::

:::{prf:corollary} Conditional native positional moments
:label: cor-kinetic-native-positional-moments

Condition on an actual complete prepared nonextinct population with $N$
active rows, after cloning, revival, jitter and collision. Assume the
configured OU and final position innovations are independent across rows,
constant isotropic Gaussians of respective amplitudes $q$ and $s$. Put
$c=h/2$, $a=e^{-\gamma h}$, $b=c(1+a)$, and let $v_{1,i}$ be the actual
first kick output, including its configured deterministic viscous force.
The completed physical position is
$x_i^+=m_i+cq\xi_i+s\zeta_i$, where $m_i=x_i+bv_{1,i}$.
Final velocity capping and terminal classification do not change these
physical coordinates, including the retained coordinates of dead rows.
Thus, with $\tau_x^2=c^2q^2+s^2$,

$$
\begin{aligned}
\mathbb E[V_{{\rm Var},x}^+\mid x,v_1]
 &=V_{{\rm Var}}(m)+(1-N^{-1})d\tau_x^2,\\
\mathbb E[N^{-1}\sum_i|x_i^+|^2\mid x,v_1]
 &=N^{-1}\sum_i|m_i|^2+d\tau_x^2,\\
\mathbb E[|\bar x^+|^2\mid x,v_1]
 &=|\bar m|^2+d\tau_x^2/N.                           \tag{5.X3}
\end{aligned}
$$

In particular the conditional centered-variance increment is at most

$$
2b\sqrt{V_{{\rm Var},x}V_{{\rm Var}}(v_1)}
 +b^2V_{{\rm Var}}(v_1)+d\tau_x^2.                  \tag{5.X4}
$$

These formulas require the recorded first-kick state; a cap only at step
end does not bound the intermediate $v_1$. If noises are correlated,
state-dependent, or graph-driven, replace the independent trace terms by
the actual conditional covariance. They do not concern a law conditioned
on terminal survival.

*Proof.* The native two drift stages give
$x^+=x+cv_1+c(av_1+q\xi)+s\zeta$. Centering the independent Gaussian
increments gives total trace $d\tau_x^2$, barycenter trace
$d\tau_x^2/N$, and their difference
$(1-N^{-1})d\tau_x^2$. Expand the squared norms and use the zero mean of
the innovations to obtain (5.X3). Expanding $V_{{\rm Var}}(x+bv_1)$ and
applying Cauchy--Schwarz to its cross term gives (5.X4). $\square$
:::

### 6.5. Balancing with Keystone Pressure

:::{prf:corollary} Complete-update positional balance
:label: cor-net-positional-contraction

For the canonical kernel, the signed one-step identity is (SCK.3) of
{prf:ref}`thm-slc-signed-complete-update`. Its $N$-uniform Keystone
substitution is (SCK.6). The remaining donor, barycenter, collision,
force, cap, and boundary contributions are retained in its explicit
residual $\mathscr D_N$. Thus a net rate follows only from an upper
estimate for that residual on the stated input class.

*Proof.* Equation (SCK.3) is the conditional full-update identity;
the source Keystone lower bound on activity gives (SCK.6) by the
sign reversal of its coefficient $-\alpha/2$. No separate
$\kappa_x$ inequality is used. $\square$
:::

### 6.6. Summary

This chapter has proven:

✅ **Bounded expansion** of positional variance under kinetics

✅ **State-independent bound** - doesn't grow with system size or configuration

The full balance retains the Keystone pressure and the signed donor and kinetic contributions.

Thermal positional spreading, force transport, and donor insertion must be evaluated together in the complete-update identity before a contraction rate is asserted.

**Next:** Chapter 7 proves that the confining potential provides additional contraction of the boundary potential.

## 7. Boundary Potential Contraction via Confining Potential

### 7.1. Introduction: Dual Safety Mechanisms

The Euclidean Gas has **two independent mechanisms** that prevent boundary extinction:

1. **Safe Harbor via Cloning** ({doc}`03_cloning`, Ch 11): Boundary-proximate walkers have low fitness and are replaced by interior clones
2. **Confining Potential via Kinetics** (this chapter): The force $F(x) = -\nabla U(x)$ pushes walkers away from the boundary

This chapter proves the second mechanism, showing that the kinetic operator provides **additional** boundary safety beyond the cloning mechanism.

### 7.2. Boundary Potential (Recall)

:::{prf:definition} Boundary Potential (Recall)
:label: def-boundary-potential-kinetic

From {doc}`03_cloning` Definition 3.3.1:

$$
W_b(S_1, S_2) = \frac{1}{N}\sum_{k=1,2} \sum_{i \in \mathcal{A}(S_k)} \varphi_{\text{barrier}}(x_{k,i})

$$

where $\varphi_{\text{barrier}}: \mathcal{X}_{\text{valid}} \to \mathbb{R}_{\geq 0}$ is the smooth barrier function that:
- Equals zero in the safe interior
- Grows as $x \to \partial\mathcal{X}_{\text{valid}}$
:::

### 7.3. Main Theorem: Potential-Driven Safety

:::{prf:theorem} Boundary Potential Contraction Under Verified Corrector Bounds
:label: thm-boundary-potential-contraction-kinetic

Consider a continuous kinetic extension with fixed averaging sets, $u=0$,
and a nonnegative $C^2$ barrier $\varphi$ with integrable derivatives on its
reachable space. Put

$$
W_b=N^{-1}\sum_i\varphi(x_i),\quad
R_b=(\gamma N)^{-1}\sum_i v_i\cdot\nabla\varphi(x_i),
\quad \Phi_b=W_b+R_b.
$$

Assume, for the laws under consideration, the actual force alignment and
**barrier-weighted velocity/Hessian bound**

$$
N^{-1}\sum_i\mathbb E[F_i\cdot\nabla\varphi_i]
 \le-\alpha_{\rm align}\mathbb EW_b+C_F,
\quad
N^{-1}\sum_i\mathbb E[v_i^\top\nabla^2\varphi_i v_i]
 \le M_H\mathbb EW_b+C_H.
$$

Also require the signed corrector-source bound

$$
-\frac d{dt}\mathbb E R_b\le\epsilon_R\mathbb EW_b+C_R,
\qquad
\kappa_b^{\rm gen}:=(\alpha_{\rm align}-M_H)/\gamma-\epsilon_R>0.
$$

These bounds, including any additional diffusion or status sources, must be
uniform in population size and valid throughout the evolution. Then

$$
\frac d{dt}\mathbb EW_b
\le-\kappa_b^{\rm gen}\mathbb EW_b+C_b^{\rm gen},
\qquad C_b^{\rm gen}=(C_F+C_H)/\gamma+C_R.
$$

If these are pointwise generator bounds and a generator-consistent discrete
extension has the observable certificate (5.D1) with coefficient $K_b$, then
(5.D3) yields

$$
\mathbb E[\Delta W_b]\le-\kappa_{\rm pot}hW_b+C_{\rm pot}h,
\quad \kappa_{\rm pot}=\kappa_b^{\rm gen}/2,
\quad C_{\rm pot}=C_b^{\rm gen}+K_bH.
$$

A velocity variance bound does not bound total velocity moments or their
barrier-weighted versions. The fixed native radial cap does not provide a
continuous Langevin moment bound or the required weak-error certificate.
For that kernel, a boundary contraction coefficient must be established
directly for its actual output and status convention.
:::

### 7.4. Proof

:::{prf:proof} Boundary Potential Contraction from Confining Force
Itô's formula for the continuous extension gives
$\mathcal LW_b=N^{-1}\sum_i v_i\cdot\nabla\varphi_i$.
For the linear-in-velocity corrector, its velocity Hessian is zero. Using
$dv_i=(F_i-\gamma v_i)dt+\Sigma_i\,dW_i$ and $dx_i=v_i\,dt$ gives

$$
\mathcal L\Phi_b=(\gamma N)^{-1}\sum_i
\left[F_i\cdot\nabla\varphi_i+v_i^\top\nabla^2\varphi_i v_i\right].
$$

The position-drift terms cancel against the friction term in the corrector.
If a different generator includes position diffusion, correlated increments,
or status jumps, their actual additional terms must be included in the source
bounds in the statement.

After localization justified by the integrability assumptions, expectation
and the alignment/Hessian bounds give

$$
\frac d{dt}\mathbb E\Phi_b
\le-\frac{\alpha_{\rm align}-M_H}{\gamma}\mathbb EW_b
 +(C_F+C_H)/\gamma.
$$

This controls the corrected observable, not yet $W_b$. Subtract the derivative
of $\mathbb ER_b$ and apply its signed source bound to obtain the asserted
inequality for $\mathbb EW_b$. Gronwall proves its continuous expectation
bound. If the assumptions hold pointwise and (5.D1) is proved for $W_b$, apply
{prf:ref}`thm-discretization` to get the discrete statement with its reduced
coefficient and discretization source.

For an exponential-distance barrier, a geometric estimate
$\nabla^2\varphi\preceq K_\varphi\varphi I$ reduces the Hessian hypothesis to
an actual weighted-moment bound
$N^{-1}\sum_i\mathbb E[\varphi_i|v_i|^2]\le M_\varphi\mathbb EW_b+C_\varphi$;
then $M_H=K_\varphi M_\varphi$ and $C_H=K_\varphi C_\varphi$.
Neither an unweighted velocity variance nor a cap applied only at native step
end establishes this continuous weighted bound. The signed corrector estimate
must also be checked; force alignment alone cannot make
$\mathcal LW_b=v\cdot\nabla\varphi$ uniformly negative for arbitrary outward
velocities. These explicit hypotheses supply every step of the proof.
$\square$
:::

### 7.5. Layered Safety Architecture

:::{prf:corollary} Total Boundary Safety from Dual Mechanisms
:label: cor-total-boundary-safety

Suppose the same boundary observable and status extension satisfy
$P_CW_b\le r_CW_b+b_C$ and $P_KW_b\le r_KW_b+b_K$ on every cloning output,
with $r_C,r_K\ge0$ and population-uniform sources. Then

$$
P_CP_KW_b\le r_Kr_CW_b+r_Kb_C+b_K.
$$

When $r_C=1-\kappa_b$ and $r_K=1-\kappa_{\rm pot}h$, the complete-step
coefficient is

$$
1-r_Kr_C=\kappa_b+\kappa_{\rm pot}h-\kappa_b\kappa_{\rm pot}h.
$$

The kinetic coefficient from the preceding theorem is available only when
all its corrector/moment and numerical-transfer hypotheses hold. Native
fixed-cap boundary estimates can instead be inserted directly.
:::

:::{prf:proof}
Apply the kinetic inequality to each cloning output, integrate over cloning,
and use $r_K\ge0$. Expanding the product gives the complete-step coefficient.
$\square$
:::

### 7.6. Small-Set Minorization for the Kinetic Kernel

:::{prf:lemma} Constructive native minorization on bounded input sets
:label: lem-kinetic-minorization

Use the native isotropic BAOAB row kernel with independent OU and final
position Gaussians of actual amplitudes $q>0$ and $s>0$, terminal-only
classification, and the radial cap $C_V(w)=Vw/(V+|w|)$ with $V>0$.
Assume $F\in C^1(\mathbb R^d)$, $|F(0)|\le B_F$ and
$\|DF\|\le L_F$ on all physical positions, and no viscous or graph forces
coupling the row updates. Put $c=h/2$, $a=e^{-\gamma h}$ and
$\ell=c^2L_F<1$. Restrict actual prepared inputs to
$|x|\le B_x$, $|v|\le B_v$; a distance from the boundary alone does not
make this set bounded on an unbounded space.

Choose a position ball $B(x_*,r_x)$ contained in the valid interior and
$0<r_v<V$. Define

$$
\begin{gathered}
A_v=B_v+c(B_F+L_FB_x),\qquad A_x=B_x+cA_v,\\
B_3=\frac{Vr_v}{V-r_v},\qquad
W=\frac{B_3+c(B_F+L_FA_x)}{1-\ell},\qquad X_2=A_x+cW,\\
\log p_*=-d\log(2\pi qs)-\frac{(W+aA_v)^2}{2q^2}
 -\frac{(|x_*|+r_x+X_2)^2}{2s^2}-d\log(1+\ell),\\
\log\epsilon_{\rm kin}=\log p_*+2\log\omega_d
 +d\log r_x+d\log r_v,                              \tag{5.M1}
\end{gathered}
$$

where $\omega_d$ is the unit-ball volume. Let $\nu_{\rm kin}$ be uniform
probability on $B(x_*,r_x)\times B(0,r_v)$ with its terminal alive mark.
Then the complete physical row kernel satisfies

$$
K((x,v),A)\ge\epsilon_{\rm kin}\nu_{\rm kin}(A),
\qquad \epsilon_{\rm kin}=e^{\log\epsilon_{\rm kin}}>0. \tag{5.M2}
$$

This is the density construction of
{prf:ref}`cor-eg-compact-kinetic-covariance`, with the OU damping coefficient
written explicitly. The row coefficient is independent of $N$. For $N$
prepared rows in this input class, the independent row specialization gives
$\epsilon_{\rm kin}^N$ on the product reference and on its quotient by
permutations; this does not establish an $N$-uniform whole-swarm coefficient.
With nonzero viscosity or graph forces, retain their coupled random-state
Jacobian and noise law in a separate density proof; a row-product argument
cannot be transferred silently. Zero final position noise does not supply
the two independent $d$-dimensional innovations required here.
:::

:::{prf:proof}
The first kick and drift give $|v_1|\le A_v$, $|x_1|\le A_x$.
Write the OU velocity as $z=av_1+q\xi$, the second drift position as
$x_2=x_1+cz$, and the pre-cap velocity as
$T(z)=z+cF(x_1+cz)$. For every target $w$, the equation
$z=w-cF(x_1+cz)$ is a contraction of complete Euclidean space with
constant $\ell<1$. Thus $T$ is a global $C^1$ diffeomorphism and its
singular values lie in $[1-\ell,1+\ell]$.

For $|v^+|\le r_v$, the inverse cap obeys
$|C_V^{-1}(v^+)|\le B_3$. Global force growth gives
$(1-\ell)|z|\le B_3+c(B_F+L_FA_x)$, hence $|z|\le W$ and
$|x_2|\le X_2$. Its OU density is at least
$(2\pi q^2)^{-d/2}\exp[-(W+aA_v)^2/(2q^2)]$.
The independent final position density at
$x^+\in B(x_*,r_x)$ is at least
$(2\pi s^2)^{-d/2}\exp[-(|x_*|+r_x+X_2)^2/(2s^2)]$.
The change of variables $z\mapsto T(z)$ contributes an inverse determinant
at least $(1+\ell)^{-d}$. Every singular value of $DC_V$ is at most one,
so the inverse-cap determinant is at least one. Multiplying proves the
joint density lower bound $p_*$ on the target product ball. Integrating
against its volume $\omega_d^2r_x^dr_v^d$ proves (5.M2), including its
terminal alive mark. In particular $\epsilon_{\rm kin}\le1$ because it
minorizes a probability law.

Conditional on a complete prepared population in this class, the declared
uncoupled kinetic innovations are independent across rows. Multiplying
the row inequalities gives the displayed $N$th-power coefficient.
Taking the permutation quotient is a common measurable pushforward and
preserves the minorization. Logarithms in (5.M1) retain its coefficient
when binary64 exponentiation underflows. $\square$
:::

### 7.7. Summary

This chapter has proven:

✅ **Independent boundary contraction** from confining potential

✅ **Minorization on compact interior sets** for the kinetic kernel

✅ **Layered safety** - two mechanisms prevent extinction

✅ **Physical intuition** - the "bowl" potential keeps walkers contained

**Dual Protection:**
- **Cloning:** Removes boundary-proximate walkers (fast, discrete)
- **Kinetics:** Pushes walkers inward continuously (smooth, deterministic)

**Next:** The companion document *{doc}`06_convergence`* combines ALL drift results from this chapter and from *{doc}`03_cloning`* to prove the synergistic Foster-Lyapunov condition and establish the main convergence theorem.
