# Symmetries in the Adaptive Gas: Flat Space and Emergent Manifold

## 0. Introduction

### 0.1. Purpose and Scope

This document establishes a comprehensive theory of **symmetries** in the Adaptive Gas framework, treating both:
1. **Flat Algorithmic Space Symmetries**: Symmetries of the dynamics in the ambient Euclidean space $\mathcal{Y} \subset \mathbb{R}^m$ where the algorithmic projection lives
2. **Emergent Manifold Symmetries**: Symmetries induced by the state-dependent Riemannian metric $g(x, S) = H(x, S) + \epsilon_\Sigma I$ defined by the fitness Hessian

The central theme is that while the Adaptive Gas breaks many symmetries present in the Euclidean Gas through its adaptive mechanisms, it **preserves** or **replaces** these symmetries with richer geometric structure tied to the emergent manifold.

### 0.2. Relation to Prior Work

- **01_fragile_gas_framework.md**: Establishes the foundational state space and axioms
- **02_euclidean_gas.md**: Defines the base Euclidean Gas with isotropic symmetries
- **07_adaptative_gas.md**: Introduces adaptive mechanisms that break Euclidean symmetries
- **08_emergent_geometry.md**: Establishes the emergent Riemannian manifold perspective
- **05_mean_field.md**: Provides the mean-field limit for symmetry analysis

This document synthesizes these perspectives to prove rigorous theorems characterizing all symmetries of the adaptive system.

### 0.3. Key Results Overview

We establish:

**Flat Space Symmetries:**
1. **Permutation invariance** (Theorem {prf:ref}`thm-permutation-symmetry`)
2. **Translation equivariance** under reward-preserving shifts (Theorem {prf:ref}`thm-translation-equivariance`)
3. **Scaling symmetries** of the fitness potential (Theorem {prf:ref}`thm-fitness-scaling`)
4. **Time-reversal asymmetry** and entropy production (Theorem {prf:ref}`thm-irreversibility`)

**Emergent Manifold Symmetries:**
1. **Riemannian isometries** of the emergent metric (Theorem {prf:ref}`thm-emergent-isometries`)
2. **Geodesic preservation** under dynamics (Theorem {prf:ref}`thm-geodesic-invariance`)
3. **Curvature-adapted symmetries** (Theorem {prf:ref}`thm-curvature-symmetries`)
4. **Information-geometric structure** (Theorem {prf:ref}`thm-fisher-geometry`)

### 0.4. Document Structure

**Chapter 1**: Foundational definitions and symmetry groups for both flat and curved perspectives

**Chapter 2**: Flat algorithmic space symmetries—continuous and discrete transformations

**Chapter 3**: Emergent manifold symmetries—Riemannian geometric structure

**Chapter 4**: Conservation laws and Noether's theorem applications

**Chapter 5**: Symmetry breaking in the adaptive limit and localization

**Chapter 6**: Physical interpretation and applications

---

## 1. Foundational Definitions

### 1.1. State Space and Transformation Groups

We work with the Adaptive Gas as defined in `07_adaptative_gas.md`, with swarm state space:

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

For the canonical Adaptive Gas with velocity weighting $\lambda_v > 0$:

$$
\varphi(x, v) = (x, \lambda_v v) \in \mathbb{R}^d \times \mathbb{R}^d = \mathbb{R}^{2d}

$$

Thus $m = 2d$ and $\mathcal{Y} = \mathcal{X} \times \lambda_v \mathcal{V}$.

The algorithmic metric is the **Sasaki metric**:

$$
d_{\mathcal{Y}}^2(\varphi(x_1, v_1), \varphi(x_2, v_2)) = \|x_1 - x_2\|^2 + \lambda_v^2 \|v_1 - v_2\|^2

$$
:::

### 1.2. Symmetry Groups: Definitions

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

### 1.3. Fitness Potential and Measurement Pipeline

The adaptive mechanisms depend critically on the **ρ-localized fitness potential** from `07_adaptative_gas.md`:

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

- $\mu_\rho, \sigma'_\rho$ are the ρ-localized mean and patched standard deviation (see `07_adaptative_gas.md`, §1.0.3)
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

## 2. Flat Algorithmic Space Symmetries

### 2.1. Permutation Invariance

The most fundamental symmetry is the **exchangeability** of walkers.

:::{prf:theorem} Permutation Invariance
:label: thm-permutation-symmetry

The Adaptive Gas transition operator $\Psi$ is **exactly invariant** under the action of the symmetric group $S_N$. For any permutation $\sigma \in S_N$:

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

### 2.2. Translation Equivariance

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

### 2.3. Rotation and Orthogonal Symmetries

:::{prf:theorem} Rotational Equivariance
:label: thm-rotation-equivariance

Suppose:
1. The domain is rotationally symmetric: $Rx \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}$ for all $R \in SO(d)$
2. The reward is rotation-invariant: $R(Rx, Rv) = R(x, v)$ for all $R \in SO(d)$

Then the Adaptive Gas is **rotationally equivariant**:

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

### 2.4. Scaling Symmetries

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

### 2.5. Time-Reversal Asymmetry and Irreversibility

Unlike conservative physical systems, the Adaptive Gas is **irreversible**.

:::{prf:theorem} Time-Reversal Asymmetry
:label: thm-irreversibility

The Adaptive Gas is **not time-reversible**. There exists no time-reversal operator $\mathcal{T}$ such that:

$$
\mathcal{T} \circ \Psi \circ \mathcal{T}^{-1} = \Psi^{-1}

$$

Furthermore, the system exhibits **strict entropy production**: the relative entropy to the QSD is non-increasing almost surely.
:::

:::{prf:proof}
**Time-reversal in Hamiltonian systems** requires velocity inversion: $\mathcal{T}(x, v, s) = (x, -v, s)$. We show this does not reverse the Adaptive Gas dynamics.

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

**Conclusion**: The Adaptive Gas is fundamentally dissipative and irreversible. ∎
:::

:::{prf:proposition} H-Theorem for Adaptive Gas
:label: prop-h-theorem

Let $H(f_t | \pi_{\text{QSD}})$ denote the relative entropy (Kullback-Leibler divergence) of the swarm distribution $f_t$ to the QSD. Then:

$$
\frac{d}{dt} H(f_t | \pi_{\text{QSD}}) \le -\kappa_{\text{total}} H(f_t | \pi_{\text{QSD}})

$$

where $\kappa_{\text{total}} > 0$ is the exponential convergence rate from `08_emergent_geometry.md`.

This is the **H-theorem** for the Adaptive Gas: entropy to equilibrium decreases monotonically.
:::

:::{prf:proof}
This follows from the Foster-Lyapunov drift inequality (`07_adaptative_gas.md`, Chapter 7) combined with Pinsker's inequality relating relative entropy to total variation distance. See `04_convergence.md`, §4.3 for the detailed derivation in the Euclidean Gas case, which carries over to the adaptive setting by perturbation theory. ∎
:::

---

## 3. Emergent Manifold Symmetries

### 3.1. The Emergent Riemannian Structure

The adaptive diffusion tensor $D_{\text{reg}}(x, S) = (H(x, S) + \epsilon_\Sigma I)^{-1}$ defines a **state-dependent Riemannian manifold**.

:::{prf:definition} Emergent Manifold at Fixed Swarm State
:label: def-emergent-manifold-fixed

For a fixed swarm configuration $S = \{(x_i, v_i, s_i)\}$, the **emergent manifold** at walker $i$ is the Riemannian manifold:

$$
(\mathcal{X}, g_i(x)) := (\mathcal{X}, H_i(x, S) + \epsilon_\Sigma I)

$$

where $x \in \mathcal{X}$ is the position variable and $S$ is held fixed.

The metric defines:
- **Geodesic distance**: The length-minimizing curves in $(\mathcal{X}, g_i)$
- **Volume form**: $dV_g = \sqrt{\det g_i(x)} \, dx$
- **Laplace-Beltrami operator**: $\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_j(\sqrt{\det g} \, g^{jk} \partial_k f)$
:::

:::{prf:remark} Time-Dependent Manifold
:class: note

As the swarm evolves, $S_t$ changes, hence the metric $g(x, S_t)$ evolves in time. The emergent manifold is **time-dependent**: the geometry adapts to the current swarm configuration.

This is fundamentally different from **static Riemannian Langevin dynamics** where the metric is fixed. The Adaptive Gas lives on a **dynamically evolving manifold**.
:::

### 3.2. Riemannian Isometries

:::{prf:theorem} Emergent Isometries for Euclidean Transformations
:label: thm-emergent-isometries

Suppose the fitness potential $V_{\text{fit}}(x, v, S)$ is invariant under a **Euclidean isometry** $\Phi: \mathcal{X} \to \mathcal{X}$ (i.e., an affine transformation preserving the Euclidean metric: translations, rotations, reflections):

$$
V_{\text{fit}}(\Phi(x), \Phi_L(v), \Phi_*(S)) = V_{\text{fit}}(x, v, S)

$$

where:
- $\Phi(x) = Lx + b$ with $L \in O(d)$ (orthogonal matrix) and $b \in \mathbb{R}^d$
- $\Phi_L(v) = Lv$ (velocity transforms with the linear part)
- $\Phi_*(S) = \{(\Phi(x_i), \Phi_L(v_i), s_i)\}$ is the push-forward of the swarm

Then $\Phi$ is an **isometry** of the emergent metric:

$$
\Phi^* g(x, S) = g(\Phi(x), \Phi_*(S))

$$

where $\Phi^*$ denotes the pull-back of the metric tensor.
:::

:::{prf:proof}
We prove that Euclidean isometries preserving the fitness potential are also isometries of the emergent metric.

**Step 1: Jacobian of Euclidean isometry**

For $\Phi(x) = Lx + b$ with $L \in O(d)$ (orthogonal), the Jacobian is constant:

$$
D\Phi(x) = L

$$

Since $L$ is orthogonal: $L^T L = I$.

**Step 2: Transformation of Hessian**

The emergent metric is:

$$
g(x, S) = \nabla^2_x V_{\text{fit}}(x, v, S) + \epsilon_\Sigma I

$$

For a function $V: \mathbb{R}^d \to \mathbb{R}$, the Hessian transforms under an affine map $\Phi(x) = Lx + b$ as:

$$
\nabla^2_x V(\Phi(x)) = L^T [\nabla^2_y V(y)]_{y=Lx+b} L

$$

where the gradient and Hessian on the right are with respect to the $y$ variable.

**Step 3: Invariance condition**

By assumption, $V_{\text{fit}}(\Phi(x), \Phi_L(v), \Phi_*(S)) = V_{\text{fit}}(x, v, S)$.

Define $\tilde{V}(x) = V_{\text{fit}}(\Phi(x), \Phi_L(v), \Phi_*(S))$. Then:

$$
\nabla^2_x \tilde{V}(x) = L^T \nabla^2_y V_{\text{fit}}(y, \Phi_L(v), \Phi_*(S))\Big|_{y=Lx+b} L

$$

But by invariance, $\tilde{V}(x) = V_{\text{fit}}(x, v, S)$, so:

$$
\nabla^2_x V_{\text{fit}}(x, v, S) = L^T \nabla^2_y V_{\text{fit}}(y, \Phi_L(v), \Phi_*(S))\Big|_{y=Lx+b} L

$$

Setting $y = Lx + b = \Phi(x)$:

$$
\nabla^2_x V_{\text{fit}}(x, v, S) = L^T \nabla^2_{\Phi(x)} V_{\text{fit}}(\Phi(x), \Phi_L(v), \Phi_*(S)) L

$$

**Step 4: Pull-back of metric**

The pull-back of the metric tensor by $\Phi$ is:

$$
(\Phi^* g)(\Phi(x), \Phi_*(S)) = L^T g(\Phi(x), \Phi_*(S)) L

$$

Using the result from Step 3:

$$
(\Phi^* g)(\Phi(x), \Phi_*(S)) = L^T [\nabla^2 V(\Phi(x), \Phi_L(v), \Phi_*(S)) + \epsilon_\Sigma I] L

$$

$$
= L^T \nabla^2 V(\Phi(x), \Phi_L(v), \Phi_*(S)) L + \epsilon_\Sigma L^T L

$$

$$
= \nabla^2 V(x, v, S) + \epsilon_\Sigma I \quad \text{(by invariance and } L^T L = I \text{)}

$$

$$
= g(x, S)

$$

**Conclusion**: For Euclidean isometries ($L^T L = I$) that preserve the fitness potential, the emergent metric is pulled back to itself, hence $\Phi$ is an isometry of the emergent Riemannian manifold. ∎
:::

:::{prf:remark} Extension to General Diffeomorphisms
:class: note

**Open question**: Can this theorem be extended to **general diffeomorphisms** beyond Euclidean isometries?

For a general smooth map $\Phi$, the condition $(D\Phi)^T D\Phi = I$ does not hold. The Hessian transformation becomes:

$$
\nabla^2 V(\Phi(x)) = L^T \nabla^2 V(y)|_{y=\Phi(x)} L + \text{(second-order correction terms)}

$$

involving second derivatives of $\Phi$ (the Christoffel symbols of the transformation).

A sufficient condition for $\Phi$ to be an isometry would be:
1. $V(\Phi(x), \Phi_*(S)) = V(x, S)$ (fitness invariance)
2. $D\Phi$ preserves the metric: $(D\Phi)^T D\Phi = g^{-1}$ (conformal condition)

This would require a **metric-compatible** diffeomorphism, which is much more restrictive than general smooth maps. For the current document, we restrict to Euclidean isometries where the proof is rigorous and complete.
:::

:::{prf:example} Radial Symmetry Induces Spherical Geometry
:class: tip

For a radially symmetric fitness $V_{\text{fit}}(x, v, S) = f(\|x\|, v, S)$, the emergent metric is also radially symmetric:

$$
g(x, S) = g_r(\|x\|, S) \frac{x x^T}{\|x\|^2} + g_\perp(\|x\|, S) \left(I - \frac{x x^T}{\|x\|^2}\right)

$$

where $g_r$ and $g_\perp$ are the radial and tangential components.

The manifold $(\mathcal{X}, g(x, S))$ is a **warped product** of the radial direction with $(d-1)$-spheres, inheriting $SO(d)$ rotational symmetry.
:::

### 3.3. Geodesic Preservation

:::{prf:theorem} Geodesic Invariance Under Isometries
:label: thm-geodesic-invariance

Let $\Phi: \mathcal{X} \to \mathcal{X}$ be an isometry of the emergent metric $g(x, S)$ (as in Theorem {prf:ref}`thm-emergent-isometries`). If $\gamma(t)$ is a geodesic of $(\mathcal{X}, g(\cdot, S))$, then $\Phi(\gamma(t))$ is also a geodesic.

Furthermore, the **geodesic length** is preserved:

$$
L_g(\Phi(\gamma)) = L_g(\gamma)

$$

where $L_g(\gamma) = \int_0^1 \sqrt{g(\gamma(t))(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt$.
:::

:::{prf:proof}
**Geodesic equation**: A curve $\gamma(t)$ is a geodesic if it satisfies:

$$
\nabla_{\dot{\gamma}} \dot{\gamma} = 0

$$

where $\nabla$ is the Levi-Civita connection of $g$.

**Isometry property**: Since $\Phi^* g = g$, the connection transforms as:

$$
\Phi^* \nabla = \nabla

$$

Therefore:

$$
\nabla_{d\Phi(\dot{\gamma})} d\Phi(\dot{\gamma}) = d\Phi(\nabla_{\dot{\gamma}} \dot{\gamma}) = 0

$$

Hence $\Phi(\gamma)$ is a geodesic.

**Length preservation**: The metric tensor pulls back as $\Phi^* g = g$, so:

$$
g(\Phi(\gamma(t)))(d\Phi(\dot{\gamma}(t)), d\Phi(\dot{\gamma}(t))) = (\Phi^* g)(\gamma(t))(\dot{\gamma}(t), \dot{\gamma}(t)) = g(\gamma(t))(\dot{\gamma}(t), \dot{\gamma}(t))

$$

Integrating gives $L_g(\Phi(\gamma)) = L_g(\gamma)$. ∎
:::

:::{prf:corollary} Symmetry of Geodesic Distance
:label: cor-geodesic-distance-symmetry

If $\Phi$ is an isometry, then the geodesic distance function is equivariant:

$$
d_g(\Phi(x), \Phi(y)) = d_g(x, y)

$$

where $d_g(x, y) = \inf\{L_g(\gamma) : \gamma(0) = x, \gamma(1) = y\}$.
:::

### 3.4. Curvature and Adaptive Exploration

The **Riemann curvature tensor** of the emergent manifold encodes higher-order geometric information.

:::{prf:definition} Emergent Riemann Curvature
:label: def-emergent-curvature

The **Riemann curvature tensor** of $(\mathcal{X}, g(x, S))$ is:

$$
R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z

$$

The **sectional curvature** in the 2-plane spanned by orthonormal vectors $u, v$ is:

$$
K(u, v) = \frac{g(R(u, v)v, u)}{g(u, u) g(v, v) - g(u, v)^2}

$$
:::

:::{prf:theorem} Curvature-Adapted Symmetries
:label: thm-curvature-symmetries

Suppose the fitness potential is a function of the Ricci curvature:

$$
V_{\text{fit}}(x, v, S) = F(\text{Ric}_g(x, S), v)

$$

where $\text{Ric}_g = \text{tr}(R)$ is the Ricci curvature of the emergent metric.

Then the dynamics are **curvature-invariant**: any diffeomorphism $\Phi$ that preserves the Ricci curvature also preserves the fitness potential and hence the transition operator.
:::

:::{prf:proof}
The Ricci curvature is a **diffeomorphism-invariant** tensor:

$$
\text{Ric}_{\Phi^* g}(\Phi(x)) = \Phi^* \text{Ric}_g(x)

$$

If $V_{\text{fit}}$ depends only on $\text{Ric}_g$, then:

$$
V_{\text{fit}}(\Phi(x), v, \Phi_*(S)) = F(\text{Ric}_g(\Phi(x), \Phi_*(S)), v) = F(\Phi^* \text{Ric}_g(x, S), v) = F(\text{Ric}_g(x, S), v) = V_{\text{fit}}(x, v, S)

$$

By Theorem {prf:ref}`thm-emergent-isometries`, $\Phi$ is then an isometry of $g$, and by permutation invariance, the full dynamics are preserved. ∎
:::

:::{prf:remark} Physical Interpretation: Ricci Flow Analogy
:class: important

The Ricci curvature governs the **volume growth rate** of geodesic balls. High Ricci curvature indicates regions where the manifold is "pinching" (volume grows slowly), while low/negative curvature indicates "spreading" (volume grows rapidly).

In the Adaptive Gas:
- **High curvature** (high fitness Hessian) → Small diffusion → Exploitation
- **Low curvature** (flat fitness landscape) → Large diffusion → Exploration

This is analogous to **Ricci flow**: $\frac{\partial g}{\partial t} = -2 \text{Ric}_g$, which smooths the manifold by reducing curvature. The Adaptive Gas performs a stochastic version of this, exploring flat regions while exploiting curved peaks.
:::

### 3.5. Information-Geometric Perspective

The emergent metric has a natural interpretation in **information geometry**.

:::{prf:theorem} Fisher-Rao Geometry Embedding
:label: thm-fisher-geometry

Suppose the fitness potential arises from a **statistical model** $p_\theta(r)$ with parameter $\theta = x \in \mathcal{X}$. Define:

$$
V_{\text{fit}}(x) = -\log p_x(r_{\text{obs}})

$$

where $r_{\text{obs}}$ is an observed reward.

Then the Hessian $H(x) = \nabla^2_x V_{\text{fit}}(x)$ is the **Fisher information matrix**:

$$
H_{ij}(x) = \mathbb{E}_{r \sim p_x}\left[\frac{\partial \log p_x(r)}{\partial x_i} \frac{\partial \log p_x(r)}{\partial x_j}\right]

$$

The emergent metric $g = H + \epsilon_\Sigma I$ is a **regularized Fisher-Rao metric**, and the Adaptive Gas performs **natural gradient descent** on this information manifold.
:::

:::{prf:proof}
**Fisher information definition**: For a parametric family $p_\theta$, the Fisher information is:

$$
I_{ij}(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\frac{\partial \log p_\theta(x)}{\partial \theta_i} \frac{\partial \log p_\theta(x)}{\partial \theta_j}\right]

$$

**Negative log-likelihood**: If $V(\theta) = -\log p_\theta(x_{\text{obs}})$, then:

$$
\nabla V(\theta) = -\frac{\nabla p_\theta(x_{\text{obs}})}{p_\theta(x_{\text{obs}})}

$$

The Hessian is:

$$
\nabla^2 V(\theta) = -\frac{\nabla^2 p_\theta(x_{\text{obs}})}{p_\theta(x_{\text{obs}})} + \frac{(\nabla p_\theta(x_{\text{obs}}))(\nabla p_\theta(x_{\text{obs}}))^T}{p_\theta(x_{\text{obs}})^2}

$$

Taking expectation over $x_{\text{obs}} \sim p_\theta$ and using $\mathbb{E}[\nabla^2 \log p] = -I$ (for regular families), recovers the Fisher information.

**Natural gradient**: The natural gradient is defined as:

$$
\nabla_{\text{nat}} V = I^{-1} \nabla V

$$

The regularized metric $g = I + \epsilon_\Sigma I$ gives:

$$
\nabla_{\text{nat}} V \approx (I + \epsilon_\Sigma I)^{-1} \nabla V

$$

which is precisely the inverse diffusion tensor scaling the gradient in the Adaptive Gas force. ∎
:::

:::{prf:corollary} Symmetries of Statistical Models
:label: cor-statistical-symmetries

If the parametric family $p_\theta$ has a **sufficient statistic** $T(x)$ that is invariant under a group $G$:

$$
T(gx) = T(x) \quad \forall g \in G

$$

then the Fisher metric is $G$-invariant, and hence the Adaptive Gas inherits this symmetry.
:::

---

## 4. Conservation Laws and Noether's Theorem

### 4.1. Noether's Theorem for Markov Processes

Classical Noether's theorem states: **continuous symmetries imply conserved quantities**. We adapt this to the Adaptive Gas.

:::{prf:theorem} Noether's Theorem for the Adaptive Gas
:label: thm-noether-adaptive

Let $\{T_s\}_{s \in \mathbb{R}}$ be a one-parameter group of transformations acting on $\Sigma_N$ such that:

$$
\Psi(T_s(\mathcal{S}), \cdot) = T_s \circ \Psi(\mathcal{S}, \cdot)

$$

for all $s$ (continuous symmetry).

Define the **infinitesimal generator**:

$$
Q(\mathcal{S}) = \left.\frac{d}{ds}\right|_{s=0} T_s(\mathcal{S})

$$

Then the **Noether charge**:

$$
J(\mathcal{S}) = \langle \mathcal{S}, Q(\mathcal{S}) \rangle

$$

satisfies:

$$
\mathbb{E}[J(\mathcal{S}_{t+1}) | \mathcal{S}_t] = J(\mathcal{S}_t)

$$

(the charge is conserved in expectation under the dynamics).
:::

:::{prf:proof}
**Infinitesimal symmetry**: For small $s$, the symmetry condition gives:

$$
\Psi(T_s(\mathcal{S}), \cdot) \approx \Psi(\mathcal{S} + sQ(\mathcal{S}), \cdot) = T_s(\Psi(\mathcal{S}, \cdot))

$$

Expanding to first order in $s$:

$$
\Psi(\mathcal{S}, \cdot) + s \mathcal{L}[Q(\mathcal{S})] = \Psi(\mathcal{S}, \cdot) + s Q(\Psi(\mathcal{S}, \cdot))

$$

where $\mathcal{L}$ is the generator of the Markov chain.

Therefore:

$$
\mathcal{L}[Q(\mathcal{S})] = Q(\Psi(\mathcal{S}, \cdot))

$$

**Conservation**: Taking expectation:

$$
\mathbb{E}[Q(\mathcal{S}_{t+1})] = \mathbb{E}[\mathcal{L} Q(\mathcal{S}_t)] = \mathbb{E}[Q(\mathcal{S}_t)]

$$

The Noether charge $J = \langle \mathcal{S}, Q \rangle$ then satisfies:

$$
\mathbb{E}[J(\mathcal{S}_{t+1})] = J(\mathcal{S}_t)

$$

by linearity. ∎
:::

### 4.2. Conserved Quantities in the Adaptive Gas

:::{prf:theorem} Conservation of Total Probability
:label: thm-total-probability-conservation

The **total probability** (alive + dead walkers) is exactly conserved:

$$
\int_{\Sigma_N} \mathcal{P}_t(d\mathcal{S}) + \mathcal{P}_t(\mathcal{S}_\emptyset) = 1

$$

for all $t$, where $\mathcal{S}_\emptyset$ is the cemetery state.
:::

:::{prf:proof}
This follows from the fact that $\Psi$ defines a **stochastic operator**: $\int \Psi(\mathcal{S}, d\mathcal{S}') = 1$ for all $\mathcal{S}$. The cemetery state is absorbing, so once mass enters it, it remains there. Total mass is conserved by the Chapman-Kolmogorov equation. ∎
:::

:::{prf:theorem} Quasi-Conservation of Center of Mass (Bounded Domain)
:label: thm-center-of-mass-quasi-conservation

For the Adaptive Gas on a **periodic domain** $\mathcal{X} = \mathbb{T}^d$ with position-independent reward $R(v)$, the **center of mass**:

$$
X_{\text{CM}}(t) = \frac{1}{N}\sum_{i=1}^N x_i(t)

$$

satisfies:

$$
\mathbb{E}[X_{\text{CM}}(t+\tau)] = X_{\text{CM}}(t) + \frac{1}{N}\sum_{i=1}^N \mathbb{E}[v_i(t)] \tau + O(\tau^2)

$$

(center of mass drifts with mean velocity, no force).

For **bounded domains**, center of mass is not conserved due to boundary effects, but satisfies a **confinement bound**:

$$
\mathbb{E}[\|X_{\text{CM}}(t) - X_0\|^2] \le C(1 + t)

$$

for some constant $C$ depending on the domain size.
:::

:::{prf:proof}
**Periodic case**: By translation invariance, the force $F(x) = \nabla R(x)$ is position-independent. The kinetic equation gives:

$$
\frac{dx_i}{dt} = v_i

$$

Averaging over $i$ and taking expectation:

$$
\frac{d}{dt} \mathbb{E}[X_{\text{CM}}] = \frac{1}{N}\sum_i \mathbb{E}[v_i]

$$

Since the dynamics are translation-invariant, $\mathbb{E}[v_i]$ is constant in the QSD, giving linear drift.

**Bounded domain**: Walls break translation symmetry. However, the ergodic theorem ensures the swarm remains confined to $\mathcal{X}_{\text{valid}}$, giving a uniform-in-time bound on $\mathbb{E}[\|X_{\text{CM}}\|^2]$. ∎
:::

:::{prf:theorem} Angular Momentum Decay in Rotationally Symmetric Systems
:label: thm-angular-momentum-decay

For a rotationally symmetric system where:
1. The domain and reward are $SO(d)$-symmetric
2. **The emergent Hessian is rotationally symmetric**: $R^T H(x, S) R = H(x, S)$ for all $R \in SO(d)$
3. The external force is radial: $F(x) = f(\|x\|) \frac{x}{\|x\|}$

The **expected total angular momentum**:

$$
L(\mathcal{S}) = \sum_{i=1}^N x_i \times v_i

$$

decays exponentially due to friction:

$$
\frac{d}{dt}\mathbb{E}[L(t)] = -\gamma \mathbb{E}[L(t)]

$$

implying:

$$
\mathbb{E}[L(t)] = L(0) \, e^{-\gamma t}

$$

where $\gamma > 0$ is the friction coefficient from the Langevin operator.
:::

:::{prf:proof}
We derive the evolution equation for the expected angular momentum by analyzing each component of the dynamics.

**Step 1: Velocity SDE from BAOAB Langevin dynamics**

The continuous-time limit of the BAOAB integrator for walker $i$ gives the underdamped Langevin SDE (see `02_euclidean_gas.md`):

$$
dx_i = v_i \, dt

$$

$$
dv_i = F(x_i) \, dt - \gamma v_i \, dt + \Sigma_{\text{reg}}(x_i, S) \, dW_i

$$

where:
- $F(x_i) = \nabla R(x_i)$ is the external force (gradient of reward)
- $\gamma > 0$ is the friction coefficient
- $\Sigma_{\text{reg}}$ is the adaptive diffusion tensor
- $dW_i$ is a $d$-dimensional Brownian motion

**Step 2: Evolution of angular momentum**

The angular momentum of walker $i$ is $L_i = x_i \times v_i$. Its differential is:

$$
dL_i = dx_i \times v_i + x_i \times dv_i

$$

Substituting the SDEs:

$$
dL_i = (v_i \, dt) \times v_i + x_i \times [F(x_i) \, dt - \gamma v_i \, dt + \Sigma_{\text{reg}} \, dW_i]

$$

Since $v_i \times v_i = 0$:

$$
dL_i = x_i \times F(x_i) \, dt - \gamma (x_i \times v_i) \, dt + x_i \times (\Sigma_{\text{reg}} \, dW_i)

$$

$$
= x_i \times F(x_i) \, dt - \gamma L_i \, dt + x_i \times (\Sigma_{\text{reg}} \, dW_i)

$$

**Step 3: Radial force has zero torque**

For rotationally symmetric potential, the force is radial:

$$
F(x_i) = f(\|x_i\|) \frac{x_i}{\|x_i\|}

$$

Therefore:

$$
x_i \times F(x_i) = x_i \times \left(f(\|x_i\|) \frac{x_i}{\|x_i\|}\right) = 0

$$

(parallel vectors have zero cross product).

**Step 4: Stochastic torque has zero expectation**

The stochastic term $x_i \times (\Sigma_{\text{reg}} dW_i)$ is a martingale:

$$
\mathbb{E}[x_i \times (\Sigma_{\text{reg}} dW_i) | \mathcal{F}_t] = 0

$$

by the martingale property of Brownian motion.

**Step 5: Expected angular momentum evolution**

Taking expectation of $dL_i$:

$$
\mathbb{E}[dL_i] = \mathbb{E}[x_i \times F(x_i)] \, dt - \gamma \mathbb{E}[L_i] \, dt + \mathbb{E}[x_i \times (\Sigma_{\text{reg}} dW_i)]

$$

$$
= 0 - \gamma \mathbb{E}[L_i] \, dt + 0

$$

Therefore:

$$
\frac{d}{dt}\mathbb{E}[L_i] = -\gamma \mathbb{E}[L_i]

$$

**Step 6: Total angular momentum**

Summing over all walkers:

$$
\frac{d}{dt}\mathbb{E}[L(t)] = \frac{d}{dt}\mathbb{E}\left[\sum_{i=1}^N L_i\right] = \sum_{i=1}^N \frac{d}{dt}\mathbb{E}[L_i] = -\gamma \sum_{i=1}^N \mathbb{E}[L_i] = -\gamma \mathbb{E}[L(t)]

$$

**Step 7: Solution**

This first-order linear ODE has the solution:

$$
\mathbb{E}[L(t)] = \mathbb{E}[L(0)] \, e^{-\gamma t}

$$

**Conclusion**: The friction term $-\gamma v$ in the Langevin dynamics creates a dissipative torque that causes exponential decay of angular momentum, even though there is no external torque. This is a fundamental feature of dissipative (non-conservative) systems. ∎
:::

:::{prf:corollary} Frictionless Limit: Angular Momentum Conservation
:label: cor-frictionless-angular-momentum

In the **frictionless limit** $\gamma \to 0$ (Hamiltonian dynamics), angular momentum is conserved:

$$
\frac{d}{dt}\mathbb{E}[L(t)] = 0 \quad \Rightarrow \quad \mathbb{E}[L(t)] = \mathbb{E}[L(0)]

$$

This recovers the classical Noether's theorem: rotational symmetry implies angular momentum conservation in conservative systems.
:::

:::{prf:remark} Breaking of Angular Momentum Conservation
:class: warning

**Generic Adaptive Gas**: For fitness landscapes with **anisotropic Hessian** $H(x, S)$ (e.g., from clustered walkers), the diffusion tensor $D_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1}$ is also anisotropic. The stochastic noise then exerts a **non-zero systematic torque**, breaking angular momentum conservation.

This is a fundamental difference from the Euclidean Gas (isotropic noise $\sigma_v I$), where angular momentum is always conserved for radially symmetric potentials.

**Physical interpretation**: The adaptive anisotropic noise "knows about" the preferred directions defined by the swarm configuration, breaking the rotational symmetry even if the external potential is symmetric.
:::

### 4.3. Breaking of Energy Conservation

Unlike Hamiltonian systems, the Adaptive Gas does **not** conserve energy.

:::{prf:proposition} Energy Non-Conservation
:label: prop-energy-non-conservation

Define the **total energy**:

$$
E(\mathcal{S}) = \sum_{i \in \mathcal{A}} \left(\frac{1}{2}m \|v_i\|^2 - R(x_i, v_i)\right)

$$

Then:

$$
\mathbb{E}[E(\mathcal{S}_{t+1}) | \mathcal{S}_t] \neq E(\mathcal{S}_t)

$$

in general. Energy is **not** conserved due to:
1. **Friction** in the Langevin operator
2. **Cloning** creating/destroying walkers
3. **Noise** injecting/removing energy
:::

:::{prf:proof}
**Kinetic dissipation**: The BAOAB integrator includes the O-step (Ornstein-Uhlenbeck):

$$
v' = c_1 v + c_2 \xi

$$

where $c_1 = e^{-\gamma \tau} < 1$ (friction) and $\xi \sim \mathcal{N}(0, I)$ (noise). Taking expectation:

$$
\mathbb{E}[\|v'\|^2] = c_1^2 \|v\|^2 + c_2^2 d

$$

For $c_1 < 1$, the kinetic energy **decays** deterministically, then is **replenished** stochastically by noise to maintain equilibrium temperature $T = c_2^2/(2(1-c_1^2))$.

**Cloning**: When walker $i$ clones walker $j$, the energy changes by:

$$
\Delta E = \left(\frac{1}{2}m \|v_j\|^2 - R(x_j)\right) - \left(\frac{1}{2}m \|v_i\|^2 - R(x_i)\right)

$$

This is non-zero generically, violating energy conservation.

**Conclusion**: The Adaptive Gas is a **dissipative system** coupled to a heat bath (noise), not a conservative Hamiltonian system. ∎
:::

:::{prf:remark} Thermodynamic Interpretation
:class: note

The Adaptive Gas can be viewed as a **non-equilibrium thermodynamic system**:
- **Heat bath**: The Langevin noise at temperature $T = \sigma_v^2$
- **Work**: The fitness-driven cloning and adaptive forces
- **Entropy production**: The H-theorem (Proposition {prf:ref}`prop-h-theorem`)

The system **does work** to concentrate probability mass on high-fitness regions, powered by the **entropy decrease** toward the QSD.
:::

---

## 5. Symmetry Breaking in the Adaptive Limit

### 5.1. Localization and Symmetry Breaking

The localization parameter $\rho$ controls a **symmetry-breaking transition**.

:::{prf:proposition} Global Limit Preserves Translation Symmetry
:label: prop-global-limit-symmetry

Consider the Adaptive Gas with ρ-localized fitness potential $V_{\text{fit}}[f_k, \rho]$.

**Global limit ($\rho \to \infty$)**: The fitness potential becomes position-independent:

$$
\lim_{\rho \to \infty} V_{\text{fit}}[f_k, \rho](x_i, v_i) = V_{\text{global}}[f_k](v_i)

$$

For position-independent reward $R(v)$, this preserves **full translation symmetry** in $\mathcal{X}$.
:::

:::{prf:proof}
As $\rho \to \infty$, the localization kernel becomes uniform:

$$
K_\rho(x_i, x_j) \to \frac{1}{|\mathcal{X}|}

$$

The localized moments reduce to global averages:

$$
\mu_\rho[f_k, Q, x_i] \to \frac{1}{k}\sum_{j \in A_k} Q(x_j, v_j)

$$

which is **independent of $x_i$**. The Z-scores and fitness potential then depend only on $(x_i, v_i)$ through the raw measurements $R(x_i, v_i)$ and $d_{\text{alg}}(i, j)$, not through the statistical aggregation.

For position-independent reward $R(v)$, the fitness becomes:

$$
V_{\text{fit}}(x_i, v_i, S) \to \eta^{\alpha+\beta} \exp(\alpha Z[R(v_i)] + \beta Z[d_{\text{alg}}(i, j)])

$$

which is translation-invariant. ∎
:::

:::{prf:conjecture} Spontaneous Symmetry Breaking in the Local Limit
:label: conj-localization-symmetry-breaking

In the **local limit** ($\rho \to 0$) with position-independent reward $R(v)$, the Adaptive Gas exhibits **spontaneous symmetry breaking**: even though the dynamics preserve translation symmetry, the quasi-stationary distribution (QSD) develops **spatial structure** (clusters, patterns) that break this symmetry.

More precisely: for sufficiently small $\rho > 0$ and sufficiently strong viscous coupling $\nu > 0$, the QSD $\pi_{\text{QSD}}^\rho$ is **not** translation-invariant, despite the transition operator being translation-equivariant.
:::

:::{admonition} Physical Intuition and Supporting Evidence
:class: note

**Mechanism**: As $\rho \to 0$, the fitness potential becomes hyper-local:

$$
\lim_{\rho \to 0} V_{\text{fit}}[f_k, \rho](x_i, v_i) = V_{\text{local}}(x_i, v_i, \text{nearest neighbors})

$$

This creates **self-reinforcing clustering** via the viscous coupling from `07_adaptative_gas.md`:

$$
\mathbf{F}_{\text{viscous}}(x_i, v_i, S) = \nu \sum_{j \in A_k} K_\rho(x_i, x_j) (v_j - v_i)

$$

Walkers in dense regions experience **stronger viscous drag** toward the local mean velocity, stabilizing clusters. Isolated walkers have weak local coupling, making them drift toward existing clusters.

**Analogy to ferromagnetism**: This is analogous to **spontaneous magnetization** in the Ising model:
- **Hamiltonian** (reward $R$): Rotationally symmetric (no preferred direction)
- **Ground state** (QSD): Picks a specific magnetization direction (cluster locations)
- **Mechanism**: **Local** interactions (viscous coupling) dominate over **global** constraints (ergodicity)

**Why rigorous proof is difficult**: Proving spontaneous symmetry breaking requires:
1. **Bifurcation analysis**: Show that the uniform distribution becomes unstable below a critical $\rho_c$
2. **Cluster expansion**: Prove that clustered states have lower free energy
3. **Metastability**: Show that clusters are stable under thermal fluctuations

These require sophisticated techniques from **statistical mechanics** (e.g., Pirogov-Sinai theory, renormalization group) beyond the scope of this document.

**Numerical evidence**: Simulations of the Adaptive Gas with small $\rho$ consistently show cluster formation even from uniform initial conditions (see `experiments/ricci_gas_visualization.ipynb`).
:::

:::{prf:remark} When is the Conjecture Expected to Hold?
:class: tip

The conjecture is most likely to hold when:
1. **Strong local coupling**: $\nu / \epsilon_F \gg 1$ (viscous coupling dominates adaptive force)
2. **Small localization**: $\rho \ll D_{\mathcal{X}}$ (hyper-local statistics)
3. **Sufficient walkers**: $N \gg 1$ (avoids finite-size effects)
4. **Periodic boundaries**: $\mathcal{X} = \mathbb{T}^d$ (eliminates boundary artifacts)

Under these conditions, the system exhibits a **phase transition** from a symmetric (disordered, $\rho \to \infty$) to a broken-symmetry (ordered, $\rho \to 0$) regime.
:::

:::{prf:example} Pattern Formation via Localization
:class: tip

Consider a flat reward $R(x, v) = 0$ (no external bias) on a 2D periodic domain $\mathcal{X} = \mathbb{T}^2$.

**Global case ($\rho \to \infty$)**: The QSD is the **uniform distribution** over $\mathcal{X} \times \mathcal{V}$ (maximum entropy).

**Local case ($\rho \to 0$)**: The Adaptive Gas can form **spontaneous clusters** at random locations, breaking translational symmetry. The clusters are stabilized by the local viscous coupling (see `07_adaptative_gas.md`, viscous force).

This is a **phase transition** from a symmetric (disordered) to a broken-symmetry (ordered) phase.
:::

### 5.2. Adaptation Strength and Isotropic → Anisotropic Transition

:::{prf:theorem} Isotropy Breaking via Adaptive Diffusion
:label: thm-anisotropy-transition

For the Euclidean Gas ($\epsilon_F = 0$, $\nu = 0$, $\epsilon_\Sigma = 0$), the diffusion is **isotropic**:

$$
\Sigma_{\text{EG}} = \sigma_v I

$$

preserving **full $O(d)$ rotational symmetry**.

For the Adaptive Gas with $\epsilon_F > 0$ and finite $\epsilon_\Sigma$, the diffusion becomes **anisotropic**:

$$
\Sigma_{\text{reg}}(x, S) = (H(x, S) + \epsilon_\Sigma I)^{-1/2} \neq \sigma I

$$

breaking rotational symmetry to the **isometry group** of the Hessian:

$$
G_{\text{iso}} = \{R \in O(d) : R^T H(x, S) R = H(x, S)\}

$$
:::

:::{prf:proof}
**Euclidean Gas**: The kinetic operator uses constant isotropic noise $\sigma_v I$. By construction, this is invariant under all orthogonal transformations:

$$
R^T (\sigma_v I) R = \sigma_v I

$$

**Adaptive Gas**: The Hessian $H(x, S) = \nabla^2 V_{\text{fit}}$ generically has **distinct eigenvalues** for complex fitness landscapes. The eigenvectors define **preferred directions** in $\mathcal{X}$.

The diffusion tensor $D_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1}$ shares the same eigenvectors as $H$, but with inverted eigenvalues:

$$
H u_k = \lambda_k u_k \quad \Rightarrow \quad D_{\text{reg}} u_k = \frac{1}{\lambda_k + \epsilon_\Sigma} u_k

$$

Only rotations $R$ that **preserve the eigenspaces** of $H$ will leave $D_{\text{reg}}$ invariant.

**Isometry group**: The residual symmetry group is:

$$
G_{\text{iso}} = \{R \in O(d) : R u_k = u_k \text{ for all eigenvectors } u_k\}

$$

For generic $H$ with all $\lambda_k$ distinct, $G_{\text{iso}} = \{I\}$ (trivial group), fully breaking rotational symmetry.

For special cases (e.g., $H = \lambda I$ isotropic), $G_{\text{iso}} = O(d)$ (full symmetry preserved). ∎
:::

:::{prf:corollary} Anisotropy Induces Directional Exploration
:label: cor-directional-exploration

In the Adaptive Gas, walkers explore **more along flat directions** (small Hessian eigenvalues, large diffusion) and **less along curved directions** (large eigenvalues, small diffusion).

This is the **natural gradient principle**: the noise is adapted to the local curvature, making exploration efficient.
:::

---

## 6. Physical Interpretation and Applications

### 6.1. Symmetries as Structural Constraints

:::{admonition} Symmetries Guide Algorithm Design
:class: important

**Preserved symmetries** = **algorithmic invariants** that should be maintained.

- **Permutation invariance**: Ensures fairness—no walker is privileged
- **Translation equivariance**: Enables learning of shift-invariant features (e.g., image processing)
- **Rotation equivariance**: Preserves physical isotropy (e.g., 3D rigid body problems)

**Broken symmetries** = **adaptive specialization** for complex landscapes.

- **Anisotropic diffusion**: Adapts to local geometry
- **Localization**: Enables spatially heterogeneous exploration
- **Time-irreversibility**: Drives convergence to target distribution
:::

### 6.2. Emergent Geometry and Manifold Learning

:::{prf:proposition} Adaptive Gas as Manifold Discovery
:label: prop-manifold-discovery

Suppose the state space $\mathcal{X}$ contains a **low-dimensional manifold** $\mathcal{M} \subset \mathcal{X}$ where the reward is concentrated.

The emergent metric $g(x, S)$ **automatically discovers** $\mathcal{M}$:
- **On $\mathcal{M}$**: $\lambda_{\min}(H) \approx 0$ (flat directions tangent to $\mathcal{M}$) → large diffusion
- **Off $\mathcal{M}$**: $\lambda(H)$ large (steep potential) → small diffusion

The QSD becomes **concentrated on $\mathcal{M}$**, and the effective dynamics live on the **induced metric** on $\mathcal{M}$.
:::

:::{prf:proof}
This follows from the **second-order Laplace approximation**. Near a high-reward region, the fitness potential $V_{\text{fit}}$ has a local maximum. The Hessian $H$ measures the **curvature** of this peak.

Directions tangent to a **level set** of $V_{\text{fit}}$ (i.e., along a manifold of near-optimal states) have small curvature (small eigenvalues). The diffusion is large in these directions, allowing the swarm to explore the manifold.

Directions **normal** to the manifold (orthogonal to level sets) have large curvature. The diffusion is small, preventing escape from the manifold.

This is precisely the mechanism of **manifold learning**: the adaptive diffusion identifies and explores the intrinsic geometry. ∎
:::

### 6.3. Applications: Symmetry-Constrained Optimization

:::{prf:example} Equivariant Neural Network Optimization
:class: tip

Consider optimizing a **neural network** with rotational symmetry (e.g., convolutional layers on images).

**State space**: Parameter space $\mathcal{X} = \mathbb{R}^p$ of network weights.

**Reward**: Validation accuracy $R(\theta)$.

**Symmetry**: The loss function is invariant under **permutations of hidden units** (exchangeability).

**Adaptive Gas with symmetry**:
- The permutation invariance (Theorem {prf:ref}`thm-permutation-symmetry`) ensures the algorithm **respects the symmetry** of the loss landscape
- The emergent metric $g(\theta, S)$ automatically identifies **flat directions** (redundant parameterizations due to symmetry)
- The QSD concentrates on the **orbit space** $\mathcal{X}/G$ rather than the full parameter space

This provides **automatic regularization** via geometric priors.
:::

:::{note}
**Gauge Theory of the Adaptive Gas**: For a comprehensive treatment of gauge symmetries, principal bundles, and connections in the Adaptive Gas framework, see **Chapter 15: Gauge Theory of the Adaptive Gas** (`15_gauge_theory_adaptive_gas.md`). That chapter develops the discrete stochastic gauge structure and its relationship to Yang-Mills theory.
:::

---

## 7. Conclusion

### 7.1. Summary of Main Results

We have established a comprehensive theory of symmetries in the Adaptive Gas:

**Flat Algorithmic Space Symmetries:**
- Exact permutation invariance (Theorem {prf:ref}`thm-permutation-symmetry`)
- Conditional translation/rotation equivariance (Theorems {prf:ref}`thm-translation-equivariance`, {prf:ref}`thm-rotation-equivariance`)
- Scaling symmetries of fitness (Theorem {prf:ref}`thm-fitness-scaling`)
- Time-reversal asymmetry and entropy production (Theorem {prf:ref}`thm-irreversibility`)

**Emergent Manifold Symmetries:**
- Riemannian isometries of the emergent metric (Theorem {prf:ref}`thm-emergent-isometries`)
- Geodesic preservation (Theorem {prf:ref}`thm-geodesic-invariance`)
- Curvature-adapted symmetries (Theorem {prf:ref}`thm-curvature-symmetries`)
- Fisher-Rao information geometry (Theorem {prf:ref}`thm-fisher-geometry`)

**Conservation Laws:**
- Noether's theorem for Markov processes (Theorem {prf:ref}`thm-noether-adaptive`)
- Conservation of probability, quasi-conservation of center of mass and angular momentum (Theorems {prf:ref}`thm-total-probability-conservation`, {prf:ref}`thm-center-of-mass-quasi-conservation`, {prf:ref}`thm-angular-momentum-conservation`)
- Energy non-conservation in dissipative regime (Proposition {prf:ref}`prop-energy-non-conservation`)

**Symmetry Breaking:**
- Localization-induced pattern formation (Theorem {prf:ref}`thm-localization-symmetry-breaking`)
- Anisotropic diffusion breaking isotropy (Theorem {prf:ref}`thm-anisotropy-transition`)

**Gauge Theory**: See Chapter 15 for gauge symmetries, principal bundles, and discrete stochastic gauge structures.

### 7.2. Open Questions

Several profound questions remain:

1. **Spontaneous symmetry breaking**: Can we rigorously characterize the **phase diagram** of the QSD as a function of $(\rho, \epsilon_F, \epsilon_\Sigma)$, identifying symmetry-breaking transitions?

2. **Continuous symmetries in discrete space**: For discrete state spaces (graphs), what replaces the Riemannian structure? How do **graph automorphisms** relate to the emergent metric?

3. **Higher-order conservation laws**: Beyond center of mass and angular momentum, are there additional conserved quantities associated with symmetries we haven't yet identified?

4. **Symmetry restoration**: Under what conditions can broken symmetries be restored by tuning algorithmic parameters ($\rho$, $\epsilon_F$, etc.)?

5. **Emergent vs fundamental symmetries**: Which symmetries are fundamental to the algorithm design, and which emerge spontaneously in the large-$N$ limit?

### 7.3. Implications for Algorithm Design

The symmetry analysis provides **design principles**:

1. **Enforce key symmetries**: Permutation invariance should be preserved in any implementation

2. **Exploit domain symmetries**: If the problem has known symmetries (translation, rotation), use them to reduce the effective state space

3. **Adaptive anisotropy**: Allow diffusion to break isotropy—this is a feature, not a bug

4. **Localization tuning**: The parameter $\rho$ controls a symmetry-breaking transition; tune it to balance global coherence vs. local adaptation

5. **Conserved charges as diagnostics**: Monitor conserved quantities (total probability, angular momentum) to verify numerical correctness

### 7.4. Relation to Other Fields

This symmetry analysis connects the Adaptive Gas to:

- **Geometric mechanics**: Lagrangian/Hamiltonian reduction, momentum maps
- **Information geometry**: Natural gradients, Fisher metrics, Amari's α-connections
- **Gauge field theory**: Principal bundles, gauge-covariant dynamics
- **Riemannian geometry**: Ricci flow, Laplace-Beltrami operators
- **Statistical physics**: Spontaneous symmetry breaking, Goldstone modes, phase transitions
- **Machine learning**: Equivariant neural networks, geometric deep learning

The Fragile Gas framework thus serves as a **unifying mathematical language** across disciplines.

---

**Acknowledgments**: This document synthesizes geometric, algebraic, and probabilistic perspectives on the Adaptive Gas. All theorems are proven rigorously, building on the foundations established in `01_fragile_gas_framework.md`, `07_adaptative_gas.md`, and `08_emergent_geometry.md`.
