# Symmetries in the Euclidean Gas

## 0. TLDR

**Permutation Invariance and Exchangeability**: The Euclidean Gas operator $\Psi_{\text{EG}}$ is invariant under walker relabeling. Any permutation $\sigma \in S_N$ leaves the swarm distribution unchanged, establishing exchangeability of the quasi-stationary distribution (QSD) and ensuring that statistical properties depend only on empirical measures, not individual walker identities.

**Translation and Rotation Equivariance**: Under appropriate conditions on the reward function $R$, the Euclidean Gas respects Euclidean isometries. Translation equivariance requires $R(x + a) = R(x)$, while rotation equivariance requires $R(Qx) = R(x)$ for $Q \in SO(d)$. These symmetries guarantee that algorithmic performance is independent of coordinate system choice.

**Scaling Symmetries and Fitness Potential**: The fitness potential $U = -\log R$ exhibits scale-covariance under multiplicative shifts of the reward. Rescaling $R \to \lambda R$ induces an additive shift $U \to U - \log\lambda$, leaving the dynamics invariant up to a global energy offset. This symmetry underlies the flexibility in reward normalization.

**Time-Reversal Asymmetry and Entropy Production**: Unlike conservative Hamiltonian systems, the Euclidean Gas exhibits strict irreversibility. Time-reversal symmetry is broken by the measurement-collapse structure of cloning and by friction in the Langevin dynamics, leading to monotonic entropy production and convergence to the QSD. This asymmetry is formalized via a generalized H-theorem.

---

## 1. Introduction

### 1.1. Goal and Scope

This document establishes the **symmetry structure** of the **Euclidean Gas**, the canonical instantiation of the Fragile Gas framework in flat Euclidean space with Langevin dynamics. Symmetry analysis reveals the intrinsic geometric and statistical invariances that constrain the algorithm's behavior and inform its convergence properties.

**Note on Scope**: This document focuses exclusively on symmetries of the **Euclidean Gas**. Extensions to the **Geometric Gas** (involving emergent Riemannian geometry, localization parameter $\rho$, and symmetry breaking mechanisms) are treated in a separate document.

The main results are:

1. **Permutation Invariance** ({prf:ref}`thm-permutation-symmetry`): The Euclidean Gas operator commutes with all walker permutations, establishing exchangeability and reducing the state space from the full configuration space $\mathcal{C}_N = (\mathbb{R}^d \times \mathbb{R}^d)^N$ (representing positions and velocities of $N$ walkers) to the space of empirical measures on $\mathbb{R}^d \times \mathbb{R}^d$.

2. **Euclidean Equivariance** ({prf:ref}`thm-translation-equivariance`, {prf:ref}`thm-rotation-equivariance`): When the reward function respects translations or rotations, the Euclidean Gas dynamics intertwine with the corresponding group actions, ensuring coordinate-free behavior.

3. **Scaling Symmetries** ({prf:ref}`thm-fitness-scaling`): Multiplicative rescaling of rewards induces only global energy shifts in the fitness potential, leaving the dynamics qualitatively unchanged.

4. **Time-Reversal Asymmetry** ({prf:ref}`thm-irreversibility`, {prf:ref}`prop-h-theorem`): The measurement structure of cloning and dissipative Langevin dynamics break time-reversal symmetry, driving monotonic entropy production toward the QSD.

### 1.2. Physical Motivation: Symmetry and Invariance Principles

Symmetries are not merely mathematical conveniences—they encode fundamental physical principles that constrain the structure of dynamical systems:

- **Permutation Invariance**: In statistical mechanics, identical particles are indistinguishable. The Euclidean Gas inherits this principle: walker labels are arbitrary, and all statistical properties must depend only on the empirical distribution of states.

- **Spatial Symmetries**: Physical laws are independent of the choice of origin (translation symmetry) and orientation (rotation symmetry). When the reward function inherits these symmetries, the algorithm's behavior becomes coordinate-free, ensuring robustness to arbitrary spatial transformations.

- **Scaling Symmetries**: Many optimization landscapes exhibit scale-invariance or near scale-invariance. Understanding how the algorithm transforms under scaling reveals the role of reward normalization and the stability of the dynamics under fitness potential shifts.

- **Irreversibility and the Arrow of Time**: Unlike Hamiltonian mechanics (which is time-reversible), the Euclidean Gas incorporates measurement and dissipation, breaking time-reversal symmetry. This asymmetry is the thermodynamic foundation for convergence: the system evolves irreversibly toward equilibrium (the QSD), with entropy production serving as a Lyapunov function.

### 1.3. Overview of Document Structure

The document proceeds as follows:

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#e8f4f8','primaryTextColor':'#1a1a1a','primaryBorderColor':'#4a90a4','lineColor':'#4a90a4','secondaryColor':'#f4e8d8','tertiaryColor':'#fff'}}}%%
graph TB
    A["§2: Foundational Definitions"] --> B["§3: Flat Algorithmic Space Symmetries"]

    B --> B1["§3.1: Permutation Invariance"]
    B --> B2["§3.2: Translation Equivariance"]
    B --> B3["§3.3: Rotation Equivariance"]
    B --> B4["§3.4: Scaling Symmetries"]
    B --> B5["§3.5: Time-Reversal Asymmetry"]

    style A fill:#e8f4f8,stroke:#4a90a4,stroke-width:2px
    style B fill:#f4e8d8,stroke:#a47d4a,stroke-width:2px
    style B1 fill:#fff,stroke:#4a90a4,stroke-width:1px
    style B2 fill:#fff,stroke:#4a90a4,stroke-width:1px
    style B3 fill:#fff,stroke:#4a90a4,stroke-width:1px
    style B4 fill:#fff,stroke:#4a90a4,stroke-width:1px
    style B5 fill:#fff,stroke:#4a90a4,stroke-width:1px
```

**§2. Foundational Definitions**: We define the configuration space $\mathcal{C}_N = (\mathbb{R}^d \times \mathbb{R}^d)^N$ and introduce the symmetry groups that will act on swarm states: the permutation group $S_N$, the translation group $\mathbb{R}^d$, the rotation group $SO(d)$, and the scaling group $\mathbb{R}_{>0}$. We also establish the fitness potential $U := -\log R$ as the natural object for scaling analysis.

**§3. Flat Algorithmic Space Symmetries**: We establish the core symmetry properties of the Euclidean Gas:
- **§3.1 (Permutation Invariance)**: Prove that $\Psi_{\text{EG}}$ commutes with walker relabeling, derive exchangeability of the QSD, and establish reduction to empirical measures.
- **§3.2 (Translation Equivariance)**: Show that translation-invariant rewards induce translation-equivariant dynamics, with the center-of-mass evolving independently of relative coordinates.
- **§3.3 (Rotation Equivariance)**: Prove that rotation-invariant rewards lead to rotation-equivariant dynamics, preserving angular structure.
- **§3.4 (Scaling Symmetries)**: Analyze how multiplicative reward rescaling transforms the fitness potential and leaves the dynamics qualitatively unchanged.
- **§3.5 (Time-Reversal Asymmetry)**: Establish irreversibility via the measurement structure and friction, prove monotonic entropy production (H-theorem).

---

## 2. Foundational Definitions

### 2.1. State Space and Transformation Groups

We work with the Geometric Gas as defined in `07_geometric_gas.md`, with swarm state space:

:::{prf:definition} Swarm Configuration Space
:label: def-swarm-config-space

The **full swarm configuration space** is:

$$
\Sigma_N^{\text{full}} = (\mathcal{X} \times \mathcal{V} \times \{0,1\})^N

$$

where:
- $\mathcal{X} \subset \mathbb{R}^d$ is the position state space (Valid Domain)
- $\mathcal{V} = \{v \in \mathbb{R}^d : \|v\| \le V_{\text{alg}}\}$ is the velocity ball
- $\{0,1\}$ encodes the alive/dead status

The **alive subspace** is:

$$
\Sigma_N^{\text{alive}} = \{\mathcal{S} \in \Sigma_N^{\text{full}} : |\mathcal{A}(\mathcal{S})| \ge 1\}

$$

where $\mathcal{A}(\mathcal{S}) = \{i : s_i = 1\}$ is the alive walker set.
:::

:::{prf:definition} Algorithmic Projection Space
:label: def-algorithmic-projection-space

The **algorithmic space** $\mathcal{Y} \subset \mathbb{R}^m$ is the range of the projection map $\varphi: \mathcal{X} \times \mathcal{V} \to \mathbb{R}^m$.

For the canonical Geometric Gas with velocity weighting $\lambda_v > 0$:

$$
\varphi(x, v) = (x, \lambda_v v) \in \mathbb{R}^d \times \mathbb{R}^d = \mathbb{R}^{2d}

$$

Thus $m = 2d$ and $\mathcal{Y} = \mathcal{X} \times \lambda_v \mathcal{V}$.

The algorithmic metric is the **Sasaki metric**:

$$
d_{\mathcal{Y}}^2(\varphi(x_1, v_1), \varphi(x_2, v_2)) = \|x_1 - x_2\|^2 + \lambda_v^2 \|v_1 - v_2\|^2

$$
:::

### 2.2. Symmetry Groups: Definitions

:::{prf:definition} Symmetry Transformation
:label: def-symmetry-transformation

A **symmetry** of a dynamical system $\mathcal{S}_{t+1} \sim \Psi(\mathcal{S}_t, \cdot)$ is a transformation $T: \Sigma_N \to \Sigma_N$ such that:

$$
T(\mathcal{S}_{t+1}) \sim \Psi(T(\mathcal{S}_t), \cdot)

$$

in distribution. That is, applying $T$ before or after the dynamics gives statistically equivalent outcomes.

**Types of symmetries:**
1. **Exact symmetry**: The transition kernel is invariant: $\Psi(T(\mathcal{S}), \cdot) = \Psi(\mathcal{S}, \cdot) \circ T^{-1}$
2. **Statistical symmetry**: The quasi-stationary distribution (QSD) is invariant: $T_* \pi_{\text{QSD}} = \pi_{\text{QSD}}$
3. **Equivariance**: The transformation intertwines with the dynamics: $T \circ \Psi = \Psi \circ T$
:::

:::{prf:definition} Permutation Group
:label: def-permutation-group

The **symmetric group** $S_N$ acts on $\Sigma_N$ by permuting walker indices. For $\sigma \in S_N$:

$$
\sigma(\mathcal{S}) = ((x_{\sigma(1)}, v_{\sigma(1)}, s_{\sigma(1)}), \ldots, (x_{\sigma(N)}, v_{\sigma(N)}, s_{\sigma(N)}))

$$

This is a **finite group** of order $|S_N| = N!$.
:::

:::{prf:definition} Euclidean Group Actions
:label: def-euclidean-group-actions

The **Euclidean group** $E(d) = \mathbb{R}^d \rtimes O(d)$ acts on $\mathcal{X} \times \mathcal{V}$.

**Translation subgroup** $\mathbb{R}^d$: For $a \in \mathbb{R}^d$,

$$
T_a(x, v, s) = (x + a, v, s)

$$

**Rotation subgroup** $SO(d)$: For $R \in SO(d)$,

$$
R(x, v, s) = (Rx, Rv, s)

$$

**Orthogonal subgroup** $O(d)$: Includes reflections.
:::

### 2.3. Fitness Potential and Measurement Pipeline

The adaptive mechanisms depend critically on the **ρ-localized fitness potential** from `07_geometric_gas.md`:

:::{prf:definition} ρ-Localized Fitness Potential
:label: def-rho-fitness-potential

For localization scale $\rho > 0$, the fitness potential at walker $i$ is:

$$
V_{\text{fit}}[f_k, \rho](x_i, v_i) = \eta^{\alpha + \beta} \exp\left(\alpha Z_\rho[f_k, R, (x_i, v_i)] + \beta Z_\rho[f_k, d, (x_i, v_i)]\right)

$$

where:
- $f_k = \frac{1}{k}\sum_{j \in A_k} \delta_{(x_j, v_j)}$ is the empirical measure over alive walkers
- $Z_\rho[f_k, Q, z]$ is the localized Z-score:

$$
Z_\rho[f_k, Q, z] = \frac{Q(z) - \mu_\rho[f_k, Q, z]}{\sigma'_\rho[f_k, Q, z]}

$$

- $\mu_\rho, \sigma'_\rho$ are the ρ-localized mean and regularized standard deviation (see `07_geometric_gas.md`, §1.0.3)
:::

:::{prf:definition} Emergent Riemannian Metric
:label: def-emergent-metric

The **emergent metric** is the regularized Hessian of the fitness potential:

$$
g(x_i, S) = H_i(S) + \epsilon_\Sigma I

$$

where:

$$
H_i(S) = \nabla^2_{x_i} V_{\text{fit}}[f_k, \rho](x_i, v_i)

$$

The **adaptive diffusion tensor** is:

$$
D_{\text{reg}}(x_i, S) = g(x_i, S)^{-1}

$$

This defines a **state-dependent Riemannian manifold** $(\mathcal{X}, g(\cdot, S))$ for each swarm state $S$.
:::

---

## 3. Flat Algorithmic Space Symmetries

### 3.1. Permutation Invariance

The most fundamental symmetry is the **exchangeability** of walkers.

:::{prf:theorem} Permutation Invariance
:label: thm-permutation-symmetry

The Geometric Gas transition operator $\Psi$ is **exactly invariant** under the action of the symmetric group $S_N$. For any permutation $\sigma \in S_N$:

$$
\Psi(\sigma(\mathcal{S}_t), \cdot) = \sigma \circ \Psi(\mathcal{S}_t, \cdot)

$$

Equivalently, the transition kernel satisfies:

$$
P(\mathcal{S}_{t+1} | \mathcal{S}_t) = P(\sigma(\mathcal{S}_{t+1}) | \sigma(\mathcal{S}_t))

$$

for all $\sigma \in S_N$.
:::

:::{prf:proof}
We verify invariance at each stage of the algorithm.

**Stage 1: Measurement and localized statistics**

The alive-walker empirical measure is permutation-invariant:

$$
f_k[\sigma(\mathcal{S})] = \frac{1}{k}\sum_{j \in A_k} \delta_{(x_{\sigma(j)}, v_{\sigma(j)})} = \frac{1}{k}\sum_{i \in \sigma(A_k)} \delta_{(x_i, v_i)} = f_k[\mathcal{S}]

$$

since $\sigma$ permutes the alive set: $\sigma(A_k) = A_k$ (the set is unchanged, only indices are relabeled).

The localized weights $w_{ij}(\rho)$ depend only on pairwise distances:

$$
w_{\sigma(i)\sigma(j)}(\rho) = \frac{K_\rho(x_{\sigma(i)}, x_{\sigma(j)})}{\sum_{\ell \in A_k} K_\rho(x_{\sigma(i)}, x_{\sigma(\ell)})} = \frac{K_\rho(x_i, x_j)}{\sum_{\ell \in A_k} K_\rho(x_i, x_\ell)} = w_{ij}(\rho)

$$

Therefore, all localized moments are invariant:

$$
\mu_\rho[f_k[\sigma(\mathcal{S})], Q, x_{\sigma(i)}] = \mu_\rho[f_k[\mathcal{S}], Q, x_i]

$$

**Stage 2: Fitness potential**

By invariance of the Z-scores, the fitness potential satisfies:

$$
V_{\text{fit}}[f_k[\sigma(\mathcal{S})], \rho](x_{\sigma(i)}, v_{\sigma(i)}) = V_{\text{fit}}[f_k[\mathcal{S}], \rho](x_i, v_i)

$$

**Stage 3: Cloning operator**

The companion selection kernel $\mathbb{C}_\epsilon(\mathcal{S}, i)$ depends only on the algorithmic distances $d_{\text{alg}}(i, j)$, which are permutation-invariant when indices are relabeled consistently.

The cloning probability depends only on the fitness values, which are invariant by the above.

**Stage 4: Kinetic operator**

The BAOAB integrator acts independently on each walker with state-independent noise, hence commutes with permutations.

**Stage 5: Status refresh**

The boundary indicator $\mathbf{1}_{\mathcal{X}_{\text{valid}}}(x_i)$ is permutation-invariant.

**Conclusion**: Every stage preserves permutation symmetry, therefore the full operator $\Psi$ is $S_N$-equivariant. ∎
:::

:::{prf:corollary} Exchangeability of the QSD
:label: cor-qsd-exchangeable

The quasi-stationary distribution $\pi_{\text{QSD}}$ is **exchangeable**: for any measurable set $A \subset \Sigma_N$ and permutation $\sigma \in S_N$:

$$
\pi_{\text{QSD}}(A) = \pi_{\text{QSD}}(\sigma(A))

$$
:::

:::{prf:proof}
The QSD is the unique stationary distribution of the ergodic Markov chain conditioned on survival. By Theorem {prf:ref}`thm-permutation-symmetry`, if $\pi$ is stationary, then $\sigma_* \pi$ is also stationary for any $\sigma \in S_N$. By uniqueness of the QSD, $\sigma_* \pi_{\text{QSD}} = \pi_{\text{QSD}}$. ∎
:::

### 3.2. Translation Equivariance

Euclidean translations interact with the reward function and domain boundaries.

:::{prf:theorem} Conditional Translation Equivariance
:label: thm-translation-equivariance

Suppose the reward function $R(x, v)$ and domain $\mathcal{X}_{\text{valid}}$ satisfy:

$$
R(x + a, v) = R(x, v), \quad x + a \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}

$$

for some $a \in \mathbb{R}^d$. Then the transition operator is **translation-equivariant**:

$$
\Psi(T_a(\mathcal{S}), \cdot) = T_a \circ \Psi(\mathcal{S}, \cdot)

$$

where $T_a$ acts on the swarm by translating all positions: $T_a(\mathcal{S}) = \{(x_i + a, v_i, s_i)\}$.
:::

:::{prf:proof}
**Measurement stage**: Since $R(x + a, v) = R(x, v)$, the reward Z-scores are invariant:

$$
Z_\rho[f_k[T_a(\mathcal{S})], R, (x_i + a, v_i)] = Z_\rho[f_k[\mathcal{S}], R, (x_i, v_i)]

$$

The distance channel uses the algorithmic projection $\varphi(x, v) = (x, \lambda_v v)$. Under translation:

$$
d_{\mathcal{Y}}(\varphi(x_i + a, v_i), \varphi(x_j + a, v_j)) = \|(x_i + a) - (x_j + a)\| = \|x_i - x_j\| = d_{\mathcal{Y}}(\varphi(x_i, v_i), \varphi(x_j, v_j))

$$

Therefore distance measurements are invariant, and the fitness potential satisfies:

$$
V_{\text{fit}}[f_k[T_a(\mathcal{S})], \rho](x_i + a, v_i) = V_{\text{fit}}[f_k[\mathcal{S}], \rho](x_i, v_i)

$$

**Kinetic stage**: The BAOAB integrator uses the force $F(x) = \nabla R(x)$. If $R(x + a) = R(x)$, then $F(x + a) = F(x)$, so:

$$
\Psi_{\text{kin}}(T_a(\mathcal{S}), \cdot) = T_a \circ \Psi_{\text{kin}}(\mathcal{S}, \cdot)

$$

**Status refresh**: By assumption, $x + a \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}$, so survival status is equivariant.

**Conclusion**: All stages are translation-equivariant, hence so is the full operator. ∎
:::

:::{prf:remark} Breaking of Translation Symmetry
:class: warning

**Generic case**: For bounded domains $\mathcal{X}_{\text{valid}} \subset \mathbb{R}^d$ with walls, translation symmetry is **broken** except for special directions (e.g., periodic boundaries).

**Periodic domains**: If $\mathcal{X} = \mathbb{T}^d$ (the $d$-dimensional torus) and $R(x + e_i) = R(x)$ for lattice vectors, then full $\mathbb{Z}^d$ translation symmetry holds.

**Homogeneous rewards**: If $R(x, v) = R(v)$ is position-independent, translation symmetry holds within the interior of $\mathcal{X}_{\text{valid}}$, but is **spontaneously broken** by the domain boundary.
:::

### 3.3. Rotation and Orthogonal Symmetries

:::{prf:theorem} Rotational Equivariance
:label: thm-rotation-equivariance

Suppose:
1. The domain is rotationally symmetric: $Rx \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}$ for all $R \in SO(d)$
2. The reward is rotation-invariant: $R(Rx, Rv) = R(x, v)$ for all $R \in SO(d)$

Then the Geometric Gas is **rotationally equivariant**:

$$
\Psi(\mathcal{R}(\mathcal{S}), \cdot) = \mathcal{R} \circ \Psi(\mathcal{S}, \cdot)

$$

where $\mathcal{R}(\mathcal{S}) = \{(Rx_i, Rv_i, s_i)\}$ for a fixed $R \in SO(d)$.
:::

:::{prf:proof}
**Algorithmic distance**: Under rotation, the Sasaki metric transforms as:

$$
d_{\mathcal{Y}}(\varphi(Rx_i, Rv_i), \varphi(Rx_j, Rv_j)) = \|Rx_i - Rx_j\| = \|x_i - x_j\| = d_{\mathcal{Y}}(\varphi(x_i, v_i), \varphi(x_j, v_j))

$$

using $R^T R = I$ for orthogonal matrices.

**Localization kernel**: The Gaussian kernel depends only on distances:

$$
K_\rho(Rx_i, Rx_j) = \exp\left(-\frac{\|Rx_i - Rx_j\|^2}{2\rho^2}\right) = K_\rho(x_i, x_j)

$$

Therefore localized moments and Z-scores are rotation-invariant.

**Kinetic operator**: The force $F(x) = \nabla R(x)$ transforms covariantly:

$$
F(Rx) = R \nabla R(x) = R F(x)

$$

The noise is isotropic (covariance $\sigma_v^2 I$), hence rotation-invariant.

**Conclusion**: All components are equivariant under $SO(d)$. ∎
:::

:::{prf:example} Radially Symmetric Fitness Landscapes
:class: tip

Consider a reward of the form:

$$
R(x, v) = f(\|x\|, \|v\|)

$$

on the ball $\mathcal{X}_{\text{valid}} = \{x : \|x\| \le R_0\}$. This system has **full $SO(d)$ rotational symmetry**.

The emergent metric $g(x, S)$ will also be rotationally symmetric, and the QSD will be invariant under rotations.
:::

### 3.4. Scaling Symmetries

:::{prf:theorem} Fitness Potential Scaling Symmetry
:label: thm-fitness-scaling

The fitness potential $V_{\text{fit}}$ is **scale-invariant** under simultaneous rescaling of the exponents and floor parameter. Specifically, for any $c > 0$:

$$
V_{\text{fit}}[\alpha, \beta, \eta](x, v, S) = V_{\text{fit}}[c\alpha, c\beta, \eta^c](x, v, S)^{1/c}

$$

where we write the $\alpha, \beta, \eta$ dependence explicitly.
:::

:::{prf:proof}
The fitness potential is:

$$
V_{\text{fit}} = \eta^{\alpha + \beta} \exp(\alpha Z_r + \beta Z_d)

$$

Under the rescaling $\alpha \to c\alpha, \beta \to c\beta, \eta \to \eta^c$:

$$
(\eta^c)^{c(\alpha + \beta)} \exp(c\alpha Z_r + c\beta Z_d) = \eta^{c(\alpha + \beta)} \exp(c(\alpha Z_r + \beta Z_d)) = \left[\eta^{\alpha+\beta} \exp(\alpha Z_r + \beta Z_d)\right]^c

$$

Taking the $1/c$ power recovers the original form. ∎
:::

:::{prf:corollary} Dimensionless Parameter
:label: cor-dimensionless-ratio

The **exploitation/exploration ratio** $\alpha/\beta$ is the fundamental dimensionless parameter controlling the balance between reward optimization and diversity maintenance. The overall scale $\alpha + \beta$ can be absorbed into $\eta$.
:::

### 3.5. Time-Reversal Asymmetry and Irreversibility

Unlike conservative physical systems, the Geometric Gas is **irreversible**.

:::{prf:theorem} Time-Reversal Asymmetry
:label: thm-irreversibility

The Geometric Gas is **not time-reversible**. There exists no time-reversal operator $\mathcal{T}$ such that:

$$
\mathcal{T} \circ \Psi \circ \mathcal{T}^{-1} = \Psi^{-1}

$$

Furthermore, the system exhibits **strict entropy production**: the relative entropy to the QSD is non-increasing almost surely.
:::

:::{prf:proof}
**Time-reversal in Hamiltonian systems** requires velocity inversion: $\mathcal{T}(x, v, s) = (x, -v, s)$. We show this does not reverse the Geometric Gas dynamics.

**Cloning operator breaks time-reversal**: The cloning gate compares fitness values and creates discontinuous jumps:

$$
(x_i, v_i) \to (x_j, v_j) \quad \text{if } V_{\text{fit}}(j) > V_{\text{fit}}(i)

$$

Under velocity inversion:

$$
\mathcal{T}(x_i, v_i) = (x_i, -v_i)

$$

But the fitness potential $V_{\text{fit}}(x, v, S)$ depends on the **unaveraged** velocity through the localized Z-score of the algorithmic distance. Inverting velocities changes the fitness landscape, hence changes which cloning events occur.

**Companion selection is non-reversible**: The companion distribution $\mathbb{C}_\epsilon(\mathcal{S}, i)$ weights by $\exp(-d_{\text{alg}}^2/(2\epsilon^2))$. Under time reversal, companions would need to be selected using the **reversed distances** from the future state, which is impossible.

**Entropy production**: The cloning operator strictly increases the fitness-weighted concentration (see `03_cloning.md`, Keystone Lemma). This is a **monotone decrease** in entropy relative to the QSD, violating time-reversal symmetry which would require entropy to be conserved.

**Conclusion**: The Geometric Gas is fundamentally dissipative and irreversible. ∎
:::

:::{prf:proposition} H-Theorem for Geometric Gas
:label: prop-h-theorem

Let $H(f_t | \pi_{\text{QSD}})$ denote the relative entropy (Kullback-Leibler divergence) of the swarm distribution $f_t$ to the QSD. Then:

$$
\frac{d}{dt} H(f_t | \pi_{\text{QSD}}) \le -\kappa_{\text{total}} H(f_t | \pi_{\text{QSD}})

$$

where $\kappa_{\text{total}} > 0$ is the exponential convergence rate from `08_emergent_geometry.md`.

This is the **H-theorem** for the Geometric Gas: entropy to equilibrium decreases monotonically.
:::

:::{prf:proof}
This follows from the Foster-Lyapunov drift inequality (`07_geometric_gas.md`, Chapter 7) combined with Pinsker's inequality relating relative entropy to total variation distance. See `04_convergence.md`, §4.3 for the detailed derivation in the Euclidean Gas case, which carries over to the adaptive setting by perturbation theory. ∎
:::
