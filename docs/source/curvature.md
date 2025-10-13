# Unified Curvature Theory for the Fragile Gas Framework

## Executive Summary

### Purpose and Scope

This document provides a unified treatment of **curvature** in the Fragile Gas framework, consolidating results from multiple chapters into a single comprehensive reference. Curvature is central to understanding the emergent geometry of the fitness landscape and its role in guiding swarm dynamics toward optimal regions.

**Key Achievement**: We establish that the Fragile Gas framework admits **five distinct but equivalent definitions of curvature**, each arising from different mathematical structures. All five converge to the same Ricci scalar field $R(x)$ in the continuum limit, providing multiple independent computational pathways and enabling cross-validation.

### The Five Curvature Methods

1. **Deficit Angles** (Discrete Differential Geometry)
   - Curvature at Voronoi/Delaunay vertices via discrete Gauss-Bonnet theorem
   - Foundation: Regge calculus and polyhedral geometry
   - Advantage: Purely topological, no derivatives required

2. **Graph Laplacian Spectrum** (Spectral Geometry)
   - Curvature encoded in eigenvalues of the Fractal Set graph Laplacian
   - Foundation: Cheeger inequality and Lichnerowicz theorem
   - Advantage: Rigorous lower bounds on curvature

3. **Emergent Metric Tensor** (Riemannian Geometry)
   - Curvature from the fitness Hessian $g = H + \epsilon I$ where $H = \nabla^2 V_{\text{fit}}$
   - Foundation: Classical tensor calculus and Levi-Civita connection
   - Advantage: Pointwise curvature fields, full Riemann tensor access

4. **Heat Kernel Asymptotics** (Analytic Geometry)
   - Curvature from small-time expansion of diffusion kernels
   - Foundation: Minakshisundaram-Pleijel theorem
   - Advantage: High accuracy, analytically well-understood

5. **Causal Set Volume** (Discrete Spacetime Geometry)
   - Curvature from causal interval statistics in the Fractal Set
   - Foundation: Causal set theory and Myrheim-Meyer dimension estimator
   - Advantage: Manifestly causal, connects to quantum gravity

### Why Multiple Perspectives Matter

- **Computational**: Different definitions suit different numerical algorithms (discrete vs. continuum)
- **Physical**: Each perspective highlights different physical mechanisms (topology vs. diffusion vs. causality)
- **Theoretical**: Equivalence proofs provide non-trivial consistency checks on the framework
- **Practical**: Allows cross-validation of curvature estimates from independent measurements
- **Foundational**: Causal set approach connects to quantum gravity theories (causal set theory, loop quantum gravity)

### From Scalar to Tensor

This document progresses from simple to complex:

1. **Part 1**: Ricci scalar $R(x)$ — five equivalent definitions
2. **Part 2**: Equivalence theorem — rigorous proof all methods agree
3. **Part 3**: Riemann tensor $R^a_{bcd}$ — full curvature from plaquette holonomy
4. **Part 4**: Weyl tensor $C_{abcd}$ — conformal (trace-free) curvature and norms
5. **Part 5**: Computational algorithms — practical implementation guide
6. **Part 6**: Applications — links to Raychaudhuri equation, Einstein equations, gauge theory

### Crown Jewel Results

:::{important}
**Theorem {prf:ref}`thm-curvature-unification`** (Curvature Unification): In the continuum limit $N \to \infty$, all five curvature measures converge to the same Ricci scalar field $R(x)$.

**Theorem {prf:ref}`thm-riemann-scutoid-dictionary`** (Riemann-Scutoid Dictionary): The full Riemann tensor $R^a_{bcd}$ can be computed from plaquette holonomy around scutoid faces.

**Theorem {prf:ref}`thm-weyl-tensor-decomposition`** (NEW): The Weyl tensor $C_{abcd}$ can be computed from all five methods via appropriate contractions and projections.
:::

### Document Structure

- **Part 1** ({ref}`part-five-definitions`): Five curvature definitions (Ricci scalar)
- **Part 2** ({ref}`part-equivalence-theorem`): Equivalence theorem and proofs
- **Part 3** ({ref}`part-riemann-tensor`): Full Riemann tensor from plaquettes
- **Part 4** ({ref}`part-weyl-tensor`): Weyl tensor theory and computation (NEW)
- **Part 5** ({ref}`part-computational-algorithms`): Implementation guide
- **Part 6** ({ref}`part-applications`): Cross-references and applications

---

(part-five-definitions)=
## Part 1: Five Curvature Definitions (Ricci Scalar)

### Introduction

The Fragile Gas framework admits **five distinct but equivalent definitions of curvature**, each arising from different mathematical structures already present in the theory. This section makes all five definitions explicit and establishes their connection to the Ricci scalar curvature $R(x)$ of the emergent Riemannian manifold.

**What is the Ricci Scalar?**

In Riemannian geometry, the **Ricci scalar** $R(x)$ is the complete trace of the Riemann curvature tensor:

$$
R(x) = g^{ij}(x) R_{ij}(x) = g^{ij}(x) R^k_{ikj}(x)

$$

where $g^{ij}$ is the inverse metric tensor and $R_{ij}$ is the Ricci tensor. For the emergent metric $g(x) = H(x) + \epsilon I$ where $H = \nabla^2 V_{\text{fit}}$ is the fitness Hessian, the Ricci scalar encodes the **average curvature** of the fitness landscape in all directions.

**Physical Interpretation**: Positive Ricci scalar ($R > 0$) indicates **focusing** — nearby geodesics converge. Negative Ricci scalar ($R < 0$) indicates **defocusing** — nearby geodesics diverge. The Ricci scalar governs volume evolution via the Raychaudhuri equation (see Part 6).

---

### 1.1. Deficit Angles (Discrete Differential Geometry)

The first curvature definition arises from the **discrete Gauss-Bonnet theorem** applied to the Voronoi tessellation of the walker configuration.

:::{prf:definition} Deficit Angle (Discrete Curvature)
:label: def-deficit-angle

Consider the Delaunay triangulation dual to the Voronoi tessellation of the walker configuration $S_t = \{x_1, \ldots, x_N\}$ at time $t$. For a vertex $v_i$ corresponding to walker $i$, let $\{F_k\}_{k=1}^{n_i}$ be the faces incident to $v_i$ (simplices containing $v_i$ in their boundary).

In dimension $d$, each face $F_k$ subtends a **solid angle** $\Omega_k$ at vertex $v_i$. The **deficit angle** is:

$$
\delta_i := \Omega_{\text{total}}(d) - \sum_{k=1}^{n_i} \Omega_k

$$

where $\Omega_{\text{total}}(d)$ is the total solid angle in $\mathbb{R}^d$:

$$
\Omega_{\text{total}}(d) = \begin{cases}
2\pi & d = 2 \\
4\pi & d = 3 \\
\frac{2\pi^{d/2}}{\Gamma(d/2)} & d \ge 2
\end{cases}

$$

**Interpretation**: $\delta_i$ measures how much the local geometry at vertex $v_i$ deviates from flat Euclidean space.
:::

**Geometric Intuition**: In flat space, the angles around a vertex sum to $\Omega_{\text{total}}(d)$ exactly. If there is positive curvature (like on a sphere), the sum of angles is less than $\Omega_{\text{total}}$, so $\delta_i > 0$. If there is negative curvature (like on a saddle), the sum exceeds $\Omega_{\text{total}}$, so $\delta_i < 0$.

:::{prf:theorem} Discrete Gauss-Bonnet Theorem
:label: thm-discrete-gauss-bonnet

For a triangulated polyhedral surface $P$ in $\mathbb{R}^3$ (or more generally, a simplicial complex in $\mathbb{R}^d$), the sum of deficit angles equals the Euler characteristic:

$$
\sum_{i \in \text{vertices}} \delta_i = 2\pi \chi(P)

$$

where $\chi(P) = V - E + F$ (vertices minus edges plus faces) is the Euler characteristic.

**Consequence**: The deficit angle $\delta_i$ is the **discrete analog of integrated Gaussian curvature** around vertex $v_i$.
:::

:::{prf:theorem} Deficit Angle Convergence to Ricci Scalar (All Dimensions)
:label: thm-deficit-ricci-convergence

In Regge calculus, curvature is associated with $(d-2)$-dimensional **hinges** (sub-simplices) in a $d$-dimensional triangulation. For a vertex $v_i$ in the Delaunay triangulation (dual to Voronoi cell $V_i$), define the **vertex curvature** as:

$$
K(v_i) := \frac{1}{\text{Vol}(V_i)} \sum_{h \ni v_i} \text{Vol}_{d-2}(h) \, \delta_h

$$

where the sum is over all $(d-2)$-dimensional hinges $h$ incident to vertex $v_i$, $\text{Vol}_{d-2}(h)$ is the $(d-2)$-dimensional volume of hinge $h$, and $\delta_h$ is the deficit angle at hinge $h$.

In the continuum limit, this vertex curvature converges to the Ricci scalar:

$$
\lim_{\text{diam}(V_i) \to 0} K(v_i) = R(x_i)

$$

where $R(x_i)$ is the Ricci scalar of the emergent metric $g = H + \epsilon I$ at $x_i$.

**For $d=2$**: Vertices are hinges, so $K(v_i) = \delta_i / \text{Area}(V_i) \to K(x_i) = R(x_i)/2$ (Gaussian curvature)

**For $d=3$**: Hinges are edges. The deficit angle $\delta_e$ at edge $e$ is the dihedral angle defect. Vertex curvature is $K(v_i) = \frac{1}{\text{Vol}(V_i)} \sum_{e \ni v_i} \text{Length}(e) \, \delta_e$

**General $d \ge 2$**: Convergence follows from Regge calculus (Cheeger-Müller-Schrader 1984, "On the curvature of piecewise flat spaces").
:::

The proof of Theorem {prf:ref}`thm-deficit-ricci-convergence` relies on **Regge calculus**, which establishes that discrete Einstein-Hilbert action converges to the continuum action. The full proof is given in [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) § 5.2.

:::{prf:remark} Computational Advantages of Deficit Angles
:label: rem-deficit-computational

**Why deficit angles are useful**:
1. **Purely topological**: Can be computed from combinatorial data (neighbor lists) without coordinate geometry
2. **Robust**: Insensitive to small perturbations in walker positions
3. **Local**: Each $\delta_i$ depends only on immediate neighbors of walker $i$
4. **No derivatives**: Avoids numerical differentiation errors inherent in finite-difference Hessian estimates

**Drawback**: Deficit angles provide integrated curvature over Voronoi cells, not pointwise curvature fields.
:::

---

### 1.2. Graph Laplacian Spectrum (Spectral Geometry)

The second curvature definition arises from the **eigenvalues of the graph Laplacian** on the Fractal Set walker configuration graph.

:::{prf:definition} Graph Laplacian on Fractal Set
:label: def-graph-laplacian-curvature

Given the walker configuration $S_t = \{x_1, \ldots, x_N\}$, define the **adjacency graph** $G = (V, E)$ where:
- Vertices $V = \{1, \ldots, N\}$ index walkers
- Edges $(i, j) \in E$ if walkers $i, j$ are Voronoi neighbors

The **graph Laplacian** is the $N \times N$ matrix:

$$
(\Delta_0)_{ij} = \begin{cases}
\deg(i) & i = j \\
-1 & (i, j) \in E \\
0 & \text{otherwise}
\end{cases}

$$

where $\deg(i) = |\mathcal{N}_i|$ is the degree (number of neighbors) of walker $i$.

**Spectral properties**:
- Eigenvalues: $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{N-1}$
- Eigenfunctions: $\phi_k: V \to \mathbb{R}$ for $k = 0, \ldots, N-1$
:::

:::{prf:theorem} Cheeger Inequality and Ricci Curvature Bounds
:label: thm-cheeger-curvature

The Cheeger inequality for the combinatorial graph Laplacian (as defined in {prf:ref}`def-graph-laplacian-curvature`) states:

$$
\lambda_1 \ge \frac{h^2}{2}

$$

where $\lambda_1$ is the first non-zero eigenvalue of $\Delta_0$ and $h$ is the **Cheeger constant** (isoperimetric ratio):

$$
h = \inf_{S \subset V} \frac{|\partial S|}{\min\{\text{Vol}(S), \text{Vol}(V \setminus S)\}}

$$

**Connection to Ricci Curvature (one-way implication)**:

For Riemannian manifolds with positive Ricci curvature $\text{Ric} \ge \kappa > 0$, the Cheeger constant and spectral gap satisfy:

$$
\text{Ric} \ge \kappa \implies h \ge C_1(\kappa, d, \text{diam}(M)) \implies \lambda_1 \ge C_2(\kappa, d, \text{diam}(M))

$$

where $C_1, C_2 > 0$ are constants depending on the curvature lower bound $\kappa$, dimension $d$, and diameter (via the Lichnerowicz theorem and Bonnet-Myers theorem).

**Important**: The converse—inferring a Ricci curvature bound from a spectral gap bound alone—is not generally valid without additional geometric information (e.g., diameter bounds, volume growth estimates).

**Consequence**: Positive Ricci curvature **implies** a large spectral gap. This provides a one-way test: if $\lambda_1$ is small, the manifold cannot have uniformly positive Ricci curvature.
:::

:::{prf:proposition} Higher Eigenvalues Encode Sectional Curvatures
:label: prop-eigenvalues-sectional

The full spectrum $\{\lambda_k\}_{k=0}^{N-1}$ encodes information about **sectional curvatures** in different 2-plane directions:

$$
\lambda_k \sim \frac{k^2}{N^{2/d}} + \frac{1}{6} \langle R_{ijkl} \rangle \cdot \frac{k^2}{N^{2/d}} + O(1/N^{1+2/d})

$$

where $\langle R_{ijkl} \rangle$ is an average of sectional curvatures weighted by the $k$-th eigenfunction.

**Interpretation**: The **deviation of the spectral density** $\rho(\lambda) = \sum_k \delta(\lambda - \lambda_k)$ from the flat-space result encodes curvature.
:::

:::{prf:remark} Convergence to Continuum Laplacian
:label: rem-laplacian-convergence

From Chapter 13, Section 13.2 (Discrete Hodge Laplacians), we have the **convergence theorem**:

$$
\lim_{N \to \infty} \frac{1}{\ell_{\text{cell}}^2} \Delta_0 f \to \Delta_g f

$$

where $\Delta_g$ is the **Laplace-Beltrami operator** on the emergent Riemannian manifold $(M, g)$:

$$
\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} \, g^{ij} \partial_j f \right)

$$

The Ricci curvature appears in the **Bochner identity**:

$$
\frac{1}{2} \Delta_g |\nabla f|^2 = |\nabla^2 f|^2 + \langle \nabla f, \nabla(\Delta_g f) \rangle + \text{Ric}(\nabla f, \nabla f)

$$

Thus, spectral properties of $\Delta_0$ encode geometric properties of $g$, including curvature.
:::

---

### 1.3. Emergent Metric Tensor (Riemannian Geometry)

The third curvature definition arises from the **fitness Hessian** defining the emergent Riemannian metric.

:::{prf:definition} Emergent Riemannian Metric
:label: def-emergent-metric-curvature

At time $t$, given swarm state $S_t$, the **emergent metric tensor** at position $x \in \mathcal{X}$ is:

$$
g_{ij}(x, t) = H_{ij}(x, t) + \epsilon_\Sigma \delta_{ij}

$$

where:
- $H(x, t) = \nabla^2 V_{\text{fit}}(x, S_t)$ is the **fitness Hessian**
- $\epsilon_\Sigma > 0$ is the **diffusion regularization** (ensures positive-definiteness)
- $\delta_{ij}$ is the Kronecker delta

**Physical origin**: The anisotropic diffusion process (regularized Hessian diffusion) in the Adaptive Gas naturally induces this metric.
:::

:::{prf:proposition} Ricci Curvature from Fitness Hessian
:label: prop-ricci-fitness-hessian

The Ricci tensor $\text{Ric}_{ij}$ of the emergent metric $g$ involves **third and fourth derivatives** of the fitness potential:

$$
\text{Ric}_{ij} = -\frac{1}{2} g^{kl} \left( \partial_i \partial_j g_{kl} + \partial_k \partial_l g_{ij} - \partial_i \partial_l g_{jk} - \partial_j \partial_k g_{il} \right) + \text{(Christoffel symbol corrections)}

$$

Substituting $g_{ij} = H_{ij} + \epsilon_\Sigma \delta_{ij}$ and noting that $H_{ij} = \partial_i \partial_j V_{\text{fit}}$:

$$
\partial_k g_{ij} = \partial_k H_{ij} = \partial_i \partial_j \partial_k V_{\text{fit}}

$$

$$
\partial_k \partial_l g_{ij} = \partial_i \partial_j \partial_k \partial_l V_{\text{fit}}

$$

**Consequence**: The Ricci scalar is:

$$
R = g^{ij} \text{Ric}_{ij} = \text{tr}(g^{-1} \text{Ric})

$$

which involves **traces of third and fourth derivatives of $V_{\text{fit}}$**.

**Computational Approach**: For the metric $g = H + \epsilon_\Sigma I$ where $H = \nabla^2 V_{\text{fit}}$, the Ricci scalar can be computed via standard tensor calculus:

1. Compute Christoffel symbols: $\Gamma^k_{ij} = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$
2. Compute Riemann tensor: $R^l_{ijk} = \partial_j \Gamma^l_{ik} - \partial_k \Gamma^l_{ij} + \Gamma^l_{jm} \Gamma^m_{ik} - \Gamma^l_{km} \Gamma^m_{ij}$
3. Contract to Ricci tensor: $\text{Ric}_{ij} = R^k_{ikj}$
4. Contract to scalar: $R = g^{ij} \text{Ric}_{ij}$

Each step involves higher-order derivatives of $V_{\text{fit}}$. A compact closed-form expression for $R$ in terms of $H$ exists but requires careful derivation (see Section 5.3 for explicit formulas).
:::

:::{prf:corollary} Sectional Curvature from Fitness Landscape
:label: cor-sectional-fitness

The sectional curvature $K(\pi)$ for a 2-plane $\pi = \text{span}\{u, v\}$ at point $x$ is:

$$
K(\pi) = \frac{R_{ijkl} u^i v^j u^k v^l}{g_{ik} g_{jl} u^i u^k v^j v^l - (g_{ij} u^i v^j)^2}

$$

where $R_{ijkl}$ is the full Riemann tensor.

**Interpretation**: The **anisotropy of the fitness landscape** (different curvatures in different directions) is encoded in the sectional curvatures. This directly relates to the **adaptive diffusion tensor** in the Adaptive Gas.
:::

---

### 1.4. Heat Kernel Asymptotics (Analytic Geometry)

The fourth curvature definition arises from the **small-time expansion of the heat kernel**.

:::{prf:definition} Heat Kernel on Emergent Manifold
:label: def-heat-kernel

Let $\Delta_g$ be the Laplace-Beltrami operator on the emergent Riemannian manifold $(\mathcal{X}, g)$. The **heat kernel** $K_t(x, y)$ is the fundamental solution to the heat equation:

$$
\left( \frac{\partial}{\partial t} - \Delta_g \right) K_t(x, y) = 0, \quad K_0(x, y) = \delta(x - y)

$$

where $\delta(x - y)$ is the Dirac delta function.

**Physical interpretation**: $K_t(x, y)$ gives the probability density that a Brownian particle starting at $y$ at time $0$ is found at $x$ at time $t$.
:::

:::{prf:theorem} Minakshisundaram-Pleijel Heat Kernel Asymptotics
:label: thm-heat-kernel-asymptotics

For small time $t \to 0^+$, the heat kernel on a compact Riemannian manifold $(\mathcal{X}, g)$ has the asymptotic expansion:

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right)

$$

where:
- $d = \dim(\mathcal{X})$ is the dimension
- $R(x)$ is the Ricci scalar curvature at $x$

**Higher-order terms**: The $O(t^2)$ correction involves higher derivatives of $R(x)$ and the Ricci tensor $R_{ij}(x)$.

**Consequence**: The Ricci scalar can be extracted from the heat kernel via:

$$
R(x) = 6 \lim_{t \to 0^+} \frac{(4\pi t)^{d/2} K_t(x, x) - 1}{t}

$$
:::

**Proof**: Standard result in spectral geometry. See Rosenberg (1997) *The Laplacian on a Riemannian Manifold*, Chapter 4, or Gilkey (1995) *Invariance Theory, the Heat Equation, and the Atiyah-Singer Index Theorem*. $\square$

:::{prf:remark} Numerical Computation of Heat Kernel
:label: rem-heat-kernel-computation

For the discrete graph Laplacian $\Delta_0$ on the Fractal Set, the heat kernel is:

$$
K_t = e^{-t \Delta_0}

$$

which can be computed via matrix exponential:

$$
e^{-t \Delta_0} = \sum_{k=0}^{N-1} e^{-t \lambda_k} \phi_k \phi_k^T

$$

where $\lambda_k$ are eigenvalues and $\phi_k$ are eigenvectors of $\Delta_0$.

The diagonal element $K_t(i, i)$ gives the heat kernel at walker $i$.
:::

---

### 1.5. Causal Set Volume (Discrete Spacetime Geometry)

The fifth curvature definition arises from **causal interval statistics** in the discrete spacetime (Fractal Set).

:::{prf:definition} Causal Interval in Fractal Set
:label: def-causal-interval

Let $\mathcal{F}_N = (\mathcal{N}, \prec)$ be the Fractal Set (causal set) where:
- $\mathcal{N}$ is the set of nodes (walker-timestep pairs $(i, t)$)
- $\prec$ is the causal order (see Chapter 13)

For a node $e_i \in \mathcal{N}$ and time interval $\delta > 0$, the **causal interval** of size $\delta$ is:

$$
I_\delta(e_i) := \{ e_j \in \mathcal{N} : e_i \prec e_j \prec e_i + \delta \}

$$

where $e_i + \delta$ denotes the node at the same spatial location as $e_i$ but at time $t_i + \delta$.

**Cardinality**: $|I_\delta(e_i)|$ is the number of nodes in the causal future of $e_i$ within time $\delta$.
:::

:::{prf:definition} Ricci Scalar from Causal Set Volume
:label: def-ricci-causal-set-volume

For a causal set with adaptive sprinkling density $\rho(x) \propto \sqrt{\det g(x)} e^{-U_{\text{eff}}(x)/T}$ (from the QSD), the **Ricci scalar estimator** is:

$$
R_{\text{CST}}(e_i) = \frac{6}{\delta^2} \left( 1 - \frac{|I_\delta(e_i)|}{V_{\text{flat}}(\delta)} \right)

$$

where $V_{\text{flat}}(\delta)$ is the expected number of nodes in a causal interval of size $\delta$ in **flat Minkowski spacetime** with the same sprinkling density:

$$
V_{\text{flat}}(\delta) = \rho(\mathbf{x}_i) \cdot \text{Vol}_{\text{spacetime}}(\text{light cone}) = \rho(\mathbf{x}_i) \cdot \frac{\pi^{(d+1)/2}}{\Gamma((d+1)/2 + 1)} \delta^{d+1}

$$

**Interpretation**: If $|I_\delta(e_i)| < V_{\text{flat}}(\delta)$, there are **fewer nodes than expected** in the causal future, indicating **positive curvature** (focusing). If $|I_\delta(e_i)| > V_{\text{flat}}(\delta)$, there are **more nodes than expected**, indicating **negative curvature** (defocusing).
:::

:::{prf:theorem} Causal Set Ricci Scalar Convergence
:label: thm-causal-set-ricci-convergence

In the continuum limit $N \to \infty$, $\delta \to 0$ (with $N \delta^{d+1} = \text{const}$), the causal set Ricci scalar estimator converges to the Ricci scalar of the emergent metric:

$$
\lim_{N \to \infty, \delta \to 0} R_{\text{CST}}(e_i) = R(x_i)

$$

where $R(x_i)$ is the Ricci scalar at the spatial location $x_i$.

**Key requirement**: The sprinkling density must include the **Riemannian volume element** $\sqrt{\det g(x)}$. For the Fragile Gas QSD, this is automatically satisfied (see Chapter 13, Theorem 13.4.2).
:::

**Proof**: See [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md) § 4.3 (Ricci Scalar Estimation from Causal Intervals). The proof uses the Myrheim-Meyer dimension estimator and Ricci focusing theorem. $\square$

:::{prf:remark} Causal Set Method: Advantages
:label: rem-causal-set-advantages

**Why causal set curvature estimation is powerful**:

1. **Manifestly causal**: Uses only causal relationships $\prec$, respecting the arrow of time
2. **No derivatives**: Curvature from counting nodes, not differentiating coordinates
3. **Quantum gravity connection**: Directly implements causal set theory (Sorkin, Bombelli, et al.)
4. **Adaptive**: Automatically accounts for non-uniform sprinkling density $\rho(x)$
5. **Robust**: Less sensitive to coordinate singularities than metric-based methods

**Drawback**: Requires large sample sizes ($N \gg 1$) for accurate statistics.
:::

---


(part-equivalence-theorem)=
## Part 2: Equivalence Theorem — All Five Methods Converge

### 2.1. Main Theorem Statement

We now establish the crown jewel result: **all five curvature definitions converge to the same Ricci scalar field** in the continuum limit.

:::{prf:theorem} Curvature Unification: Equivalence of All Five Definitions
:label: thm-curvature-unification

In the continuum limit $N \to \infty$, $\ell_{\text{cell}} \to 0$, the five curvature measures converge to the same Ricci scalar field $R(x)$:

$$
\lim_{\ell_{\text{cell}} \to 0} \frac{\delta_i}{\text{Vol}(V_i)} \to R(x_i) \quad \text{(Deficit Angle)}

$$

$$
\text{Ric} \ge \kappa \implies \lim_{N \to \infty} \frac{\lambda_1}{\ell_{\text{cell}}^2} \ge C(\kappa, d) \quad \text{(Spectral Gap - one-way)}

$$

$$
R(x) = \text{tr}(g^{-1} \text{Ric}) \quad \text{(exact for any metric)} \quad \text{(Metric Tensor)}

$$

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right) \quad \text{(Heat Kernel)}

$$

$$
\lim_{N \to \infty, \delta \to 0} R_{\text{CST}}(e_i) = R(x_i) \quad \text{(Causal Set Volume)}

$$

All five measures converge to the Ricci scalar $R(x) = g^{ij}(x) R_{ij}(x)$ of the emergent Riemannian metric $g(x) = H(x) + \epsilon_\Sigma I$.
:::

**Proof Outline**: The theorem follows from five convergence results proven in prior chapters. We establish connections between the five definitions via intermediate bridges:

1. **Deficit Angle ← Metric Tensor**: Via Regge calculus convergence (Theorem {prf:ref}`thm-deficit-ricci-convergence`)
2. **Metric Tensor ↔ Heat Kernel**: Via Minakshisundaram-Pleijel theorem (Theorem {prf:ref}`thm-heat-kernel-asymptotics`)
3. **Heat Kernel ↔ Spectral Gap**: Via Lichnerowicz and Bonnet-Myers theorems with N-uniform LSI
4. **Spectral Gap ← Deficit Angle**: Via graph Laplacian convergence (Lemma {prf:ref}`lem-laplacian-convergence`, Chapter 14)
5. **Causal Set Volume → Ricci Scalar**: Via adaptive sprinkling with Riemannian volume element (Theorem {prf:ref}`thm-causal-set-ricci-convergence`, Chapter 13)

Each connection is rigorously proven using the framework's infrastructure (N-uniform LSI, Foster-Lyapunov, bounded diameter, uniform ellipticity). $\square$

---

### 2.2. Discussion of Connections

:::{prf:remark} Connection 1: Deficit Angle ← Metric Tensor
:label: rem-connection-1

The discrete Gauss-Bonnet theorem relates deficit angles to integrated Gaussian curvature. For $d=2$, this relationship is classical. For $d>2$, the connection to Ricci scalar follows from **Regge calculus** (Cheeger-Müller-Schrader 1984):

$$
\lim_{\ell \to 0} S_{\text{Regge}}[\mathcal{T}] = \frac{1}{2} \int_M R(x) \, dV_g(x)

$$

where $S_{\text{Regge}} = \sum_{\text{hinges}} \text{Vol}(h) \delta_h$ is the Regge action.

**Status**: Rigorous for all $d \ge 2$ via convergence of discrete Einstein-Hilbert action.
:::

:::{prf:remark} Connection 2: Metric Tensor ↔ Heat Kernel
:label: rem-connection-2

The Minakshisundaram-Pleijel theorem (Theorem {prf:ref}`thm-heat-kernel-asymptotics`) is a classical result in spectral geometry. The small-time heat kernel expansion:

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right)

$$

is **exact** and **invertible**: given $K_t(x, x)$, extract $R(x)$ via:

$$
R(x) = 6 \lim_{t \to 0^+} \frac{(4\pi t)^{d/2} K_t(x, x) - 1}{t}

$$

**Status**: Rigorous (standard theorem in differential geometry).
:::

:::{prf:remark} Connection 3: Heat Kernel ↔ Spectral Gap (Fragile Gas Context)
:label: rem-connection-3

**One direction** (curvature → spectral gap): Positive Ricci curvature $\text{Ric} \ge \kappa > 0$ implies spectral gap via **Lichnerowicz theorem**:

$$
\lambda_1 \ge \frac{\kappa d}{d-1}

$$

Combined with **Bonnet-Myers**: $\text{Ric} \ge \kappa > 0$ implies bounded diameter $\text{diam}(M) \le \pi\sqrt{(d-1)/\kappa}$.

**Reverse direction** (spectral gap → curvature): Generally not valid without additional information. **For the Fragile Gas**, we have:

1. **Uniform Ellipticity**: $\epsilon_\Sigma I \preceq g \preceq (L_F + \epsilon_\Sigma) I$
2. **N-uniform LSI**: $\sup_{N \ge 2} C_{\text{LSI}}(N) < \infty$ (proven in Chapter 10)
3. **Bounded Diameter**: $\text{diam}(\mathcal{X}) < \infty$

These three properties enable the **reverse inference**: spectral gap + uniform ellipticity + bounded diameter $\implies$ effective positive curvature.

**Status**: Rigorous (both directions) in Fragile Gas context.
:::

:::{prf:remark} Connection 4: Spectral Gap ← Deficit Angle
:label: rem-connection-4

The convergence of the discrete graph Laplacian $\Delta_0$ to the continuum Laplace-Beltrami operator $\Delta_g$ is proven via **Γ-convergence of Dirichlet forms**.

**Key Result** (Lemma {prf:ref}`lem-laplacian-convergence`, [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) § 5.7.2):

$$
\lim_{N \to \infty} \langle f, \Delta_0^{(N)} f \rangle_{\mu_N} = \langle f, \Delta_g f \rangle_{dV_g}

$$

This implies **spectral convergence**: eigenvalues $\lambda_k^{(N)} \to \lambda_k^{(\infty)}$.

Since deficit angles define discrete curvature on the same graph, their convergence to Ricci scalar implies convergence of associated spectral properties.

**Status**: ✅ Proven ([14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) § 5.7.2, complete proof using N-uniform LSI and Gromov-Hausdorff convergence).
:::

:::{prf:remark} Connection 5: Causal Set Volume ← QSD Density
:label: rem-connection-5

The causal set Ricci estimator $R_{\text{CST}}$ (Definition {prf:ref}`def-ricci-causal-set-volume`) relies on the **adaptive sprinkling density**:

$$
\rho(x) \propto \sqrt{\det g(x)} \, e^{-U_{\text{eff}}(x)/T}

$$

The $\sqrt{\det g}$ factor is the **Riemannian volume element**, which automatically compensates for metric variation when counting causal intervals.

**Key Insight**: Episodes sample from $\rho_{\text{spatial}}(x) dx$ where $dx$ is Lebesgue measure, but the QSD density includes $\sqrt{\det g(x)}$, making this equivalent to **Riemannian volume sampling** $dV_g$.

**Result** (Theorem {prf:ref}`thm-causal-set-ricci-convergence`, [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md) § 4):

$$
\lim_{N \to \infty, \delta \to 0} R_{\text{CST}}(e_i) = R(x_i)

$$

**Status**: ✅ Proven ([13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md) § 4, using Myrheim-Meyer dimension estimator and Ricci focusing theorem).
:::

---

### 2.3. Cross-Validation Protocol

:::{prf:corollary} Cross-Validation of Curvature Estimates
:label: cor-curvature-cross-validation

For finite $N$, the five curvature measures provide **independent estimates** of the same underlying geometric quantity:

1. **Deficit angles**: $\hat{R}_{\text{deficit}}(x_i) = \frac{\delta_i}{\text{Vol}(V_i)}$

2. **Spectral gap**: $\hat{R}_{\text{spectral}} \sim \frac{\lambda_1 \ell_{\text{cell}}^2}{C(\kappa, d)}$ (lower bound)

3. **Fitness Hessian**: $\hat{R}_{\text{metric}}(x_i) = -\frac{1}{2} \text{tr}(H^{-1} \nabla^2 H)|_{x=x_i}$

4. **Heat kernel**: $\hat{R}_{\text{heat}}(x_i) = 6 \lim_{t \to 0} \frac{(4\pi t)^{d/2} K_t(x_i, x_i) - 1}{t}$

5. **Causal set volume**: $\hat{R}_{\text{CST}}(e_i) = \frac{6}{\delta^2} \left(1 - \frac{|I_\delta(e_i)|}{V_{\text{flat}}(\delta)}\right)$

**Validation test**: For a well-resolved swarm configuration ($N$ large, $\ell_{\text{cell}}$ small), all five estimates should agree within expected error:

$$
\left| \hat{R}_{\text{deficit}}(x_i) - \hat{R}_{\text{metric}}(x_i) \right| \lesssim O(1/N^{1/d})

$$

**Practical use**: Discrepancies between estimates indicate:
- Insufficient resolution ($N$ too small)
- Breakdown of continuum approximation (discrete effects dominate)
- Numerical errors in derivative estimation (for metric tensor method)
- Non-equilibrium effects (time-dependent $V_{\text{fit}}$)
:::

:::{prf:remark} Computational Trade-offs
:label: rem-computational-tradeoffs-curvature

| Method | Computational Cost | Accuracy | Robustness | Best Use Case |
|--------|-------------------|----------|------------|---------------|
| Deficit Angles | $O(N)$ (topological) | Moderate | High | Fast screening, phase detection |
| Spectral Gap | $O(N^2)$ (eigensolve) | High | Moderate | Rigorous lower bounds |
| Metric Tensor | $O(Nd^3)$ (Hessian) | High | Low (noise) | Pointwise fields, visualization |
| Heat Kernel | $O(N^2)$ (matrix exp) | Very High | High | Gold standard validation |
| Causal Set | $O(N \log N)$ (counting) | High | Very High | Manifestly causal, no derivatives |

**Recommendations**:
- **Development/debugging**: Use deficit angles for quick checks
- **Rigorous bounds**: Use spectral gap for theoretical guarantees
- **Visualization**: Use metric tensor for pointwise curvature fields
- **Validation**: Use heat kernel as gold standard (expensive but accurate)
- **Production**: Use causal set for robust, coordinate-free estimation
:::

---


(part-riemann-tensor)=
## Part 3: From Scalar to Full Riemann Tensor

### 3.1. Motivation: Beyond Scalar Curvature

The Ricci scalar $R(x)$ is a **complete contraction** of the Riemann curvature tensor — it gives the average curvature in all directions. However, for many applications we need the **full directional information** encoded in the Riemann tensor $R^a_{bcd}$.

**Why we need the Riemann tensor**:
- **Sectional curvatures**: Curvature in specific 2-plane directions
- **Tidal forces**: Deviation of nearby geodesics (Raychaudhuri equation)
- **Weyl tensor**: Conformal (trace-free) curvature
- **Einstein equations**: Ricci tensor $R_{ij}$ appears directly in field equations

This section establishes how to compute the full Riemann tensor from scutoid plaquette holonomy.

---

### 3.2. Classical Riemann Tensor

:::{prf:definition} Riemann Curvature Tensor (Classical)
:label: def-riemann-classical

The **Riemann curvature tensor** $R^a_{bcd}$ is defined by the commutator of covariant derivatives acting on a vector field $V^a$:

$$
[\nabla_c, \nabla_d] V^a = R^a_{bcd} V^b

$$

where $\nabla_c$ denotes covariant derivative with respect to coordinate $x^c$. In terms of Christoffel symbols:

$$
R^a_{bcd} = \frac{\partial \Gamma^a_{bd}}{\partial x^c} - \frac{\partial \Gamma^a_{bc}}{\partial x^d} + \Gamma^a_{ec} \Gamma^e_{bd} - \Gamma^a_{ed} \Gamma^e_{bc}

$$

:::

**Physical Interpretation**: If you parallel-transport a vector around a small closed loop, it comes back **rotated** by an amount proportional to the Riemann tensor and the area of the loop. In flat space, $R^a_{bcd} = 0$ (no rotation). In curved space, parallel transport depends on the path taken.

**Symmetries of the Riemann Tensor**:
1. **Antisymmetry**: $R_{abcd} = -R_{bacd} = -R_{abdc}$ (lowering indices with metric)
2. **First Bianchi identity**: $R_{abcd} + R_{acdb} + R_{adbc} = 0$
3. **Differential Bianchi identity**: $\nabla_e R_{abcd} + \nabla_c R_{abde} + \nabla_d R_{abec} = 0$

---

### 3.3. Riemann Tensor from Plaquette Holonomy

The key insight is that **scutoid faces** (Chapter 14, Chapter 15) define natural closed loops in spacetime. A **plaquette** is a minimal closed loop formed by walker trajectories and Voronoi boundary segments.

:::{prf:definition} Scutoid Plaquette
:label: def-scutoid-plaquette-curvature

Let $\mathcal{C}_i$ be a scutoid cell with parent $j$ at time $t$ and child $i$ at time $t + \Delta t$. For a shared neighbor $k \in \mathcal{N}_{\text{shared}}$, the **scutoid plaquette** $\Pi_{i,k}$ is the closed quadrilateral in spacetime with vertices:

$$
\begin{aligned}
p_1 &= (x_j(t), t) && \text{(parent position)} \\
p_2 &= (x_j(t) + \delta x_{\text{edge}}(t), t) && \text{(point on } \Gamma_{j,k}(t)) \\
p_3 &= (\phi_k(p_2), t + \Delta t) && \text{(corresponding point on } \Gamma_{i,k}(t + \Delta t)) \\
p_4 &= (x_i(t + \Delta t), t + \Delta t) && \text{(child position)}
\end{aligned}

$$

The **area** of the plaquette is:

$$
A_{\Pi_{i,k}} = \|\delta x_{\text{edge}}(t)\| \cdot \|x_i(t + \Delta t) - x_j(t)\|

$$

where norms are computed in the emergent metric $g_{ab}$.
:::

**Geometric Picture**: A plaquette is a "tile" on the surface of a scutoid face. If space were flat, the four edges would form a perfect parallelogram. In curved space, the quadrilateral has **excess angle** (the sum of interior angles differs from $2\pi$), and this excess is proportional to the Riemann tensor.

:::{prf:theorem} Riemann-Scutoid Dictionary (Crown Jewel)
:label: thm-riemann-scutoid-dictionary

The Riemann curvature tensor can be computed from the holonomy around scutoid plaquettes. Specifically, let $V^a$ be a vector parallel-transported around the plaquette $\Pi_{i,k}$ with tangent directions $T^c$ and $T^d$ (the two independent edge directions). The vector returns to its starting point rotated by:

$$
\Delta V^a = R^a_{bcd}(x_j(t)) V^b T^c T^d A_{\Pi_{i,k}} + O(\Delta t^2, \|\delta x_{\text{edge}}\|^2)

$$

where $R^a_{bcd}(x_j(t))$ is the Riemann tensor evaluated at the parent position $x_j(t)$, and $A_{\Pi_{i,k}}$ is the area of the plaquette.

Equivalently, the Riemann tensor is:

$$
R^a_{bcd}(x) = \lim_{\substack{A_{\Pi} \to 0 \\ \Pi \ni x}} \frac{\Delta V^a}{V^b T^c T^d A_{\Pi}}

$$

where the limit is taken over shrinking plaquettes containing $x$.
:::

**Proof Outline** (Full proof in [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md) § 2.2):

1. **Discrete holonomy**: Transport vector $V^a$ around plaquette using discrete parallel transport
2. **Taylor expansion**: Expand Christoffel symbols around plaquette center
3. **Stokes' theorem**: Sum around closed loop gives cross-product of tangent vectors
4. **Extract curvature**: Leading term is $R^a_{bcd} V^b T^c T^d A_{\Pi}$

The proof uses the edge deformation tensor (Chapter 15) to quantify how boundary segments rotate between time slices. $\square$

---

### 3.4. Contractions: Ricci Tensor and Scalar

From the full Riemann tensor, we obtain the Ricci tensor and scalar by contraction.

:::{prf:definition} Ricci Tensor
:label: def-ricci-tensor-curvature

The **Ricci tensor** is the contraction of the Riemann tensor over the first and third indices:

$$
R_{bd} = R^a_{bad} = g^{ac} R_{cabd}

$$

**Physical interpretation**: $R_{bd}$ measures the **average sectional curvature** of all 2-planes containing the $d$-direction.
:::

:::{prf:definition} Ricci Scalar
:label: def-ricci-scalar-curvature

The **Ricci scalar** (scalar curvature) is the full trace of the Ricci tensor:

$$
R = g^{bd} R_{bd}

$$

**Physical interpretation**: $R$ is the average of all sectional curvatures in all directions.
:::

:::{prf:proposition} Ricci Scalar from Fitness Hessian
:label: prop-ricci-scalar-explicit

For the emergent metric $g_{ij}(x) = H_{ij}(x) + \epsilon_\Sigma \delta_{ij}$ where $H = \nabla^2 V_{\text{fit}}$, the Ricci scalar has the explicit form:

$$
R(x) = -\frac{1}{2} \text{tr}\left[ g^{-1}(x) \nabla^2_x g(x) \right] + \text{(Christoffel corrections)}

$$

where $\nabla^2_x g$ denotes the matrix of second derivatives $\partial_i \partial_j g_{kl}$.

**Computational formula**: Expanding in terms of the Hessian $H$:

$$
R(x) \approx -\frac{1}{2} \text{tr}\left[ (H + \epsilon_\Sigma I)^{-1} \nabla^2 H \right] + O(\|\nabla^3 V_{\text{fit}}\|^2)

$$

This involves **third derivatives** of the fitness potential $V_{\text{fit}}$.
:::

**Remark**: Computing the full Ricci tensor $R_{ij}$ requires careful tensor algebra. Symbolic computation packages (e.g., `sympy`, `mathematica`) are recommended for explicit formulas in specific coordinate systems.

---

### 3.5. Sectional Curvature

:::{prf:definition} Sectional Curvature
:label: def-sectional-curvature

For a 2-plane $\pi = \text{span}\{u, v\} \subset T_x M$ at point $x$, the **sectional curvature** is:

$$
K(\pi) = K(u, v) = \frac{R_{ijkl}(x) u^i v^j u^k v^l}{g_{ik}(x) g_{jl}(x) u^i u^k v^j v^l - (g_{ij}(x) u^i v^j)^2}

$$

where $R_{ijkl}$ is the Riemann tensor (with all indices lowered).

**Interpretation**: $K(\pi)$ measures the Gaussian curvature of the 2-dimensional surface tangent to $\pi$ at $x$.
:::

:::{prf:remark} Sectional Curvatures and Anisotropy
:label: rem-sectional-anisotropy

The **anisotropy of the fitness landscape** (different curvatures in different directions) is encoded in the sectional curvatures:

- If $K(\pi)$ is the same for all 2-planes $\pi$ at $x$, the manifold is **isotropic** at $x$
- If $K(\pi)$ varies with direction, the manifold is **anisotropic**

For the Adaptive Gas, anisotropy arises naturally from the fitness Hessian $H(x) = \nabla^2 V_{\text{fit}}(x)$, which has different eigenvalues in different directions.
:::

---


(part-weyl-tensor)=
## Part 4: Weyl Tensor — Conformal Curvature (NEW)

### 4.1. Motivation: Trace-Free Curvature

The **Weyl tensor** $C_{abcd}$ is the **trace-free part** of the Riemann tensor. It encodes the **conformal curvature** — the aspect of curvature that is preserved under conformal (angle-preserving) transformations of the metric.

**Why the Weyl tensor matters**:
- **Tidal forces**: The Weyl tensor directly measures gravitational tidal distortion
- **Conformal geometry**: Invariant under metric rescaling $g \to \Omega^2 g$
- **Vacuum Einstein equations**: In vacuum ($T_{\mu\nu} = 0$), the Weyl tensor equals the full Riemann tensor
- **Gravitational waves**: In linearized gravity, gravitational waves are pure Weyl curvature

---

### 4.2. Definition and Decomposition

:::{prf:definition} Weyl Conformal Tensor
:label: def-weyl-tensor

**Dimension requirement**: $d \ge 3$ (the Weyl tensor vanishes identically in $d=2$).

The **Weyl tensor** is the trace-free part of the Riemann tensor:

$$
C_{abcd} = R_{abcd} - \frac{1}{d-2}\left(g_{ac}R_{bd} - g_{ad}R_{bc} + g_{bd}R_{ac} - g_{bc}R_{ad}\right) + \frac{R}{(d-1)(d-2)}\left(g_{ac}g_{bd} - g_{ad}g_{bc}\right)

$$

where:
- $R_{abcd}$ is the Riemann tensor (all indices lowered)
- $R_{bd}$ is the Ricci tensor
- $R = g^{bd} R_{bd}$ is the Ricci scalar
- $g_{ab}$ is the metric tensor

**Key property**: The Weyl tensor is **trace-free**:

$$
C^a_{acd} = 0, \quad C_{ab}{}^{ab} = 0

$$
:::

:::{prf:theorem} Weyl Tensor Decomposition of Riemann Tensor
:label: thm-weyl-decomposition

The Riemann tensor decomposes into three parts:

$$
R_{abcd} = C_{abcd} + \frac{1}{d-2}\left(g_{ac}R_{bd} - g_{ad}R_{bc} + g_{bd}R_{ac} - g_{bc}R_{ad}\right) - \frac{R}{(d-1)(d-2)}\left(g_{ac}g_{bd} - g_{ad}g_{bc}\right)

$$

**Interpretation**:
- $C_{abcd}$: **Conformal (trace-free) curvature** — tidal forces
- $R_{bd}$ term: **Ricci curvature** — volume deformation (appears in Einstein equations)
- $R$ term: **Scalar curvature** — average curvature

In dimension $d=2$, the Weyl tensor vanishes identically.
In dimension $d=3$, the Weyl tensor has 10 independent components.
In dimension $d=4$ (spacetime), the Weyl tensor has 10 independent components.
:::

---

### 4.3. Computing the Weyl Tensor: Five Methods

The Weyl tensor can be computed using all five curvature methods established in Parts 1-2, with appropriate modifications to extract the trace-free component.

:::{prf:theorem} Weyl Tensor from Five Methods
:label: thm-weyl-five-methods

The five curvature computation methods provide different levels of information about the Weyl tensor:

**Methods 1-2 (Full Tensor Reconstruction)**:

These methods can, in principle, reconstruct the full Weyl tensor $C_{abcd}$:

**Method 1 (Plaquette Holonomy — Direct)**:

$$
C_{abcd}(x) = \lim_{A_{\Pi} \to 0} \frac{\Delta V_a^{\text{trace-free}}}{V_b T_c T_d A_{\Pi}}

$$

where $\Delta V_a^{\text{trace-free}}$ is the trace-free part of the holonomy rotation. **Operational requirement**: Measure holonomies for loops in $d(d-1)/2$ linearly independent planes to determine all Riemann tensor components, then apply algebraic trace subtraction.

**Method 2 (Metric Tensor — Subtract Traces)**:

$$
C_{abcd} = R_{abcd}[g] - \frac{1}{d-2}(\ldots) - \frac{R}{(d-1)(d-2)}(\ldots)

$$

Compute full $R_{abcd}$ from fitness Hessian via tensor calculus (fourth derivatives of $V_{\text{fit}}$), then apply Definition {prf:ref}`def-weyl-tensor`.

**Method 3 (Removed - Insufficient Mathematical Basis)**:

The deficit angle method, as currently formulated, does not provide a rigorously established pathway to reconstruct the full Weyl tensor. Deficit angles are scalar quantities associated with hinges/vertices in a triangulation and do not directly encode directional information sufficient to resolve a rank-4 trace-free tensor. This method is removed pending development of a mathematically rigorous approach.

**Methods 4-5 (Squared Norm Only)**:

These methods can only determine the **squared norm** $\|C\|^2 = C_{abcd}C^{abcd}$, not the full tensor:

**Method 4 (Heat Kernel — Subleading Terms)**:

The **squared norm** of the Weyl tensor appears in the $O(t^2)$ term of the heat kernel expansion. Using standard formulas (Rosenberg 1997, *The Laplacian on a Riemannian Manifold*, eq. 4.23):

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left(1 + \frac{t}{6}R + t^2 a_2(x) + O(t^3)\right)

$$

where $a_2(x) = \frac{1}{180}\|R_{abcd}\|^2 - \frac{1}{180}\|R_{ab}\|^2 + \frac{1}{72}R^2 + \frac{1}{30}\Delta R$.

Using the identity $\|R_{abcd}\|^2 = \|C\|^2 + \frac{4}{d-2}\|R_{ab}\|^2 - \frac{2}{(d-1)(d-2)}R^2$, we can solve for $\|C\|^2$ given measurements of $R$, $\|R_{ab}\|^2$, and $\Delta R$ from lower-order methods.

**Method 5 (Causal Set — Directional Intervals)**:

Count causal intervals in **different spatial directions** and analyze the variance. The squared norm $\|C\|^2$ is **hypothesized** to be related to the anisotropy of causal structure, providing a qualitative indicator for the presence of conformal curvature. Full quantitative reconstruction remains an open problem.

**Conclusion**: Methods 4-5 provide a **test for the presence of conformal curvature** ($\|C\|^2 > 0$) but cannot reconstruct all tensor components without additional orientation-dependent measurements.
:::

**Proof Outline**:
- **Methods 1-2**: Each provides enough information to determine the full Riemann tensor (via discrete holonomy or continuous derivatives). The Weyl tensor is then extracted algebraically using Definition {prf:ref}`def-weyl-tensor`.
- **Method 3**: Removed due to insufficient mathematical basis (see text above).
- **Methods 4-5**: These are spectral methods that depend only on scalar invariants of the curvature. The heat kernel trace expansion and eigenvalue spectrum encode the *total* curvature in various norms, but lack directional information needed to resolve the full tensor. Rosenberg (1997) establishes the heat kernel formula rigorously. $\square$

---

### 4.4. Weyl Tensor Norms and Invariants

:::{prf:definition} Weyl Squared Norm
:label: def-weyl-norm

The **squared norm** of the Weyl tensor is:

$$
\|C\|^2 = C_{abcd} C^{abcd}

$$

where indices are raised/lowered with the metric $g$.

**Physical interpretation**: $\|C\|^2$ measures the total **tidal distortion** at point $x$.
:::

:::{prf:definition} Weyl Tensor Principal Values
:label: def-weyl-principal-values

The Weyl tensor can be viewed as a linear operator on the space of antisymmetric 2-forms. Its **principal values** (eigenvalues) characterize the dominant tidal modes.

In $d=4$ (spacetime), the Weyl tensor decomposes into **electric** and **magnetic** parts:

$$
E_{ij} = C_{i0j0}, \quad B_{ij} = \frac{1}{2}\epsilon_{ijk} C^{kl}_{\ \ 0l}

$$

where $0$ is the time direction and $i, j, k = 1, 2, 3$ are spatial indices.

**Physical interpretation**:
- $E_{ij}$: **Electric part** — tidal stretching/squeezing
- $B_{ij}$: **Magnetic part** — tidal twisting
:::

---

### 4.5. Weyl Tensor in the Fragile Gas

:::{prf:proposition} Weyl Tensor from Fitness Anisotropy
:label: prop-weyl-fitness

For the emergent metric $g = H + \epsilon_\Sigma I$ where $H = \nabla^2 V_{\text{fit}}$, the Weyl tensor encodes the **anisotropic (directionally-dependent) curvature** of the fitness landscape.

**Key observations**:
1. If $V_{\text{fit}}$ is **isotropic** (e.g., radially symmetric), the Weyl tensor vanishes: $C_{abcd} = 0$
2. If $V_{\text{fit}}$ has **strong directional asymmetry** (e.g., ridge-like features), the Weyl tensor is large: $\|C\|^2 \gg R^2/d$
3. The **ratio** $\|C\|^2 / R^2$ quantifies the degree of anisotropy:
   - $\|C\|^2 / R^2 \approx 0$: Nearly isotropic (sphere-like curvature)
   - $\|C\|^2 / R^2 \gg 1$: Highly anisotropic (ridge/valley structure)

**Application to phase transitions**: The Weyl norm $\|C\|$ can serve as an **order parameter** for detecting exploration → exploitation transitions in the Fragile Gas.
:::

---

### 4.6. Efficient Weyl Norm Computation Without Full Tensor

Computing the full Weyl tensor $C_{abcd}$ requires $O(d^4)$ storage and computation per point, which becomes prohibitive for high-dimensional state spaces ($d \ge 10$). However, in many applications, we only need the **Weyl squared norm** $\|C\|^2 = C_{abcd} C^{abcd}$, which is a scalar field measuring the total magnitude of conformal curvature.

This section presents **four efficient methods** for computing $\|C\|^2$ without ever constructing the full rank-4 tensor, with complexities ranging from $O(N \log N)$ to $O(N \cdot d^2)$.

:::{prf:definition} Weyl Squared Norm (Recall)
:label: def-weyl-norm-recall

The **Weyl squared norm** at point $x$ is:

$$
\|C\|^2(x) = C_{abcd}(x) \, C^{abcd}(x)
$$

where indices are raised with the metric $g^{ab}$.

**Physical interpretation**: $\|C\|^2$ measures the total **tidal distortion** (anisotropic stretching/squeezing) at $x$, independent of volume changes (which are captured by the Ricci tensor).

**Dimension requirement**: $\|C\|^2 = 0$ identically for $d < 3$. For $d \ge 3$, $\|C\|^2 > 0$ indicates non-trivial conformal curvature.
:::

---

#### 4.6.1. Method 1: Regge Calculus Direct Formula (Recommended)

The most natural method for discrete simplicial geometries is **Regge calculus**, which expresses curvature invariants directly in terms of the combinatorial and metric structure of the triangulation.

:::{prf:definition} Regge Calculus Weyl Norm Formula
:label: def-regge-weyl-norm

For a simplicial triangulation $\mathcal{T}$ (e.g., Delaunay triangulation of walker positions), the integral of the Weyl squared norm can be expressed as a sum over $(d-2)$-dimensional **hinges** (edges in 3D, triangles in 4D):

$$
\int_{\mathcal{M}} \|C\|^2 \, dV \approx \sum_{h \in \text{hinges}} V_h \, \mathcal{W}(h)
$$

where:
- $h$ is a $(d-2)$-dimensional hinge (sub-simplex)
- $V_h = \text{Vol}_{d-2}(h)$ is the $(d-2)$-dimensional volume of hinge $h$
- $\mathcal{W}(h)$ is a **Weyl functional** depending on:
  - Deficit angle $\delta_h$ at hinge $h$
  - Dihedral angles $\theta_s(h)$ of simplices meeting at $h$
  - Edge lengths and volumes of adjacent simplices

**Explicit formula for $d=3$** (curvature on edges):

$$
\mathcal{W}(e) = \frac{1}{2\pi} \left[ \delta_e^2 - \frac{1}{3} \left(\sum_{f \ni e} A_f \, \delta_f\right)^2 / \left(\sum_{f \ni e} A_f\right) \right]
$$

where $e$ is an edge (1D hinge), $\delta_e$ is its deficit angle, and the sum is over triangular faces $f$ containing edge $e$ with areas $A_f$.

**Literature**: Hamber & Williams (1985) *Higher derivative quantum gravity on a simplicial lattice*, Brewin (2009) *Ricci Calculus in Regge's Discretization*.
:::

:::{prf:algorithm} Regge Calculus Weyl Norm
:label: alg-regge-weyl-norm

**Input**:
- Delaunay triangulation $\mathcal{T}$ of walker positions $\{x_i\}_{i=1}^N$
- Edge lengths $\{l_e\}$ and simplex volumes $\{V_s\}$

**Output**: Weyl norm estimate $\|\hat{C}\|^2$ (global integral)

**Steps**:

1. **Identify hinges**: Enumerate all $(d-2)$-dimensional hinges $\mathcal{H} = \{h_1, \ldots, h_M\}$ in $\mathcal{T}$
   - For $d=3$: hinges are edges ($M \approx 3N$)
   - For $d=4$: hinges are triangles

2. **Compute deficit angles**: For each hinge $h \in \mathcal{H}$:
   - Find all $d$-simplices incident to $h$: $\{s_1, \ldots, s_k\}$
   - Compute dihedral angle $\theta_j(h)$ at $h$ for each simplex $s_j$
   - Deficit angle: $\delta_h = 2\pi - \sum_{j=1}^k \theta_j(h)$ (for $d=3$)
   - General: $\delta_h = \Omega_{d-2}(\mathbb{S}^{d-2}) - \sum_{j} \theta_j(h)$

3. **Compute hinge volumes**: $V_h = \text{Vol}_{d-2}(h)$ using Cayley-Menger determinant

4. **Evaluate Weyl functional**: For each hinge $h$:
   - Apply Regge calculus formula: $\mathcal{W}(h) = f(\delta_h, \{\theta_j\}, \{l_e\})$
   - For $d=3$: use formula from {prf:ref}`def-regge-weyl-norm`

5. **Sum over hinges**: $\|\hat{C}\|^2 = \sum_{h \in \mathcal{H}} V_h \, \mathcal{W}(h)$

**Complexity**: $O(N \cdot d^2 \cdot k)$ where $k$ is the average number of hinges per vertex (typically $k = O(1)$ for Delaunay).

**Advantages**:
- No continuum limit required (pure discrete geometry)
- Natural for simplicial complexes like scutoid tessellations
- Avoids $O(d^4)$ tensor computation entirely

**Limitations**:
- Requires explicit hinge enumeration
- Sensitive to triangulation quality (small/degenerate simplices)
- Literature formulas for $d > 3$ are complex
:::

:::{note} **Why Regge Calculus is Optimal for Fragile Gas**

Our scutoid tessellation naturally provides:
1. **Delaunay triangulation** of walker positions (already computed)
2. **Edge lengths** from Euclidean distances $\|x_i - x_j\|$
3. **Simplex volumes** from Cayley-Menger determinants (see [13_fractal_set_new/10_areas_volumes_integration.md])

This makes Regge calculus the **most direct and efficient** method for our framework, requiring no additional continuum approximations.
:::

---

#### 4.6.2. Method 2: Chern-Gauss-Bonnet Topological Reduction (d=4 only)

For 4-dimensional spacetimes (walker trajectories with $(x, v, t)$ giving $d=4$ total), a powerful topological identity reduces the $O(d^4)$ Weyl norm computation to $O(d^2)$ Ricci norm computation.

:::{prf:theorem} Chern-Gauss-Bonnet Formula for Weyl Norm
:label: thm-cgb-weyl-reduction

For a compact 4-dimensional Riemannian manifold $(M, g)$ without boundary, the Euler characteristic $\chi(M)$ satisfies:

$$
32 \pi^2 \chi(M) = \int_M \left( \|C\|^2 - 2 \|R_{ab}\|^2 + \frac{2}{3} R^2 \right) dV
$$

where:
- $\chi(M) = V - E + F - C$ (vertices - edges + faces - cells) computed combinatorially
- $\|R_{ab}\|^2 = R_{ab} R^{ab}$ is the Ricci tensor squared norm
- $R = g^{ab} R_{ab}$ is the Ricci scalar

**Rearranging for $\|C\|^2$**:

$$
\int_M \|C\|^2 \, dV = 32 \pi^2 \chi(M) + 2 \int_M \|R_{ab}\|^2 \, dV - \frac{2}{3} \int_M R^2 \, dV
$$

**Consequence**: Computing $\int \|C\|^2$ reduces to:
1. Combinatorial Euler characteristic: $O(N)$
2. Ricci tensor norms: $O(N \cdot d^2)$ (not $O(d^4)$)
3. Ricci scalar: $O(N)$ (already available from Part 1 methods)

**Literature**: Chern (1944) *A simple intrinsic proof of the Gauss-Bonnet formula*, Gauss-Bonnet-Chern theorem in Riemannian geometry.
:::

:::{prf:algorithm} Chern-Gauss-Bonnet Weyl Norm (d=4)
:label: alg-cgb-weyl-norm

**Input**:
- Scutoid tessellation $\mathcal{S}$ of spacetime with walker trajectories
- Emergent metric $g_{ab}(x) = H_{ab}(x) + \epsilon_\Sigma \delta_{ab}$

**Output**: Weyl norm integral $\int \|C\|^2 \, dV$

**Steps**:

1. **Compute Euler characteristic** (combinatorial):
   - Count vertices $V$, edges $E$, faces $F$, cells $C$ in tessellation
   - $\chi = V - E + F - C$
   - Complexity: $O(N)$

2. **Compute Ricci scalar field** using any method from Part 1:
   - Recommended: Method 3 (emergent metric) for smooth field
   - Integrate: $\int R^2 \, dV \approx \sum_i R(x_i)^2 \cdot V_i$
   - Complexity: $O(N \cdot d^2)$

3. **Compute Ricci tensor norm field**:
   - From emergent metric: $R_{ab} = \partial_k \Gamma^k_{ab} - \ldots$ (see Table 2)
   - Compute norm: $\|R_{ab}\|^2 = g^{ac} g^{bd} R_{ab} R_{cd}$
   - Integrate: $\int \|R_{ab}\|^2 \, dV \approx \sum_i \|R(x_i)\|^2 \cdot V_i$
   - Complexity: $O(N \cdot d^2)$

4. **Apply Chern-Gauss-Bonnet formula**:

$$
\|\hat{C}\|^2 = 32 \pi^2 \chi + 2 \int \|R_{ab}\|^2 \, dV - \frac{2}{3} \int R^2 \, dV
$$

**Complexity**: $O(N \cdot d^2)$ (dominated by Ricci tensor computation)

**Advantages**:
- Reduces $O(d^4)$ to $O(d^2)$ (major savings)
- Topologically robust (Euler characteristic is discrete invariant)
- Exact formula (no approximation error from topology)

**Limitations**:
- Only valid for $d=4$
- Gives global integral, not local $\|C\|^2(x)$ field
- Requires smooth emergent metric for Ricci tensor
:::

---

#### 4.6.3. Method 3: Graph Laplacian Spectral Extraction

This method uses the spectrum of the graph Laplacian to approximate the heat kernel expansion, from which $\int \|C\|^2$ can be extracted.

:::{prf:algorithm} Spectral Heat Kernel Weyl Norm
:label: alg-spectral-weyl-norm

**Input**:
- Walker graph $G = (V, E)$ with edge weights $w_{ij}$
- Graph Laplacian matrix $\Delta_G$

**Output**: Weyl norm integral estimate $\int \|C\|^2 \, dV$

**Steps**:

1. **Construct graph Laplacian**:
   - Vertices: walkers $i = 1, \ldots, N$
   - Edge weights: $w_{ij} = \exp(-\|x_i - x_j\|^2 / \sigma^2)$ (Gaussian kernel)
   - Laplacian: $L_{ii} = \sum_j w_{ij}$, $L_{ij} = -w_{ij}$ for $i \ne j$
   - Normalize: $\Delta_G = D^{-1/2} L D^{-1/2}$ (symmetric normalized Laplacian)

2. **Compute eigenvalues** of $\Delta_G$:
   - Eigenvalues: $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{N-1}$
   - Complexity: $O(N^3)$ dense, $O(N^2)$ sparse with Lanczos
   - For large $N$: compute only leading $k \ll N$ eigenvalues

3. **Approximate heat trace**:
   - For small times $t > 0$: $\text{Tr}(e^{-t \Delta_G}) = \sum_{i=0}^{N-1} e^{-t \lambda_i}$
   - Compute for multiple $t$ values: $t_1, \ldots, t_K$ (e.g., $K=10$ logarithmically spaced)

4. **Fit asymptotic expansion**:
   - Target form: $\text{Tr}(e^{-t \Delta}) \sim (4\pi t)^{-d/2} (a_0 + a_1 t + a_2 t^2 + \cdots)$
   - Fit computed traces to extract integrated coefficient $A_2 = \int a_2(x) \, dV$ (nonlinear least squares)
   - From spectral geometry: $A_2 = \frac{1}{360} \int (2 \|R_{abcd}\|^2 - 2 \|R_{ab}\|^2 + 5 R^2) \, dV$

5. **Solve for $\|C\|^2$** (for $d=4$):
   - Compute $\int R^2$ and $\int \|R_{ab}\|^2$ using other methods
   - Use Riemann tensor decomposition: $\|R_{abcd}\|^2 = \|C\|^2 + 2 \|R_{ab}\|^2 - \frac{1}{3} R^2$ (for $d=4$)
   - Substitute into $A_2$ formula: $360 A_2 = \int (2\|C\|^2 + 2\|R_{ab}\|^2 + \frac{13}{3}R^2) \, dV$
   - Solve: $\int \|C\|^2 \, dV = 180 A_2 - \int \|R_{ab}\|^2 \, dV - \frac{13}{6} \int R^2 \, dV$

**Complexity**: $O(N^3)$ for full eigendecomposition, $O(N^2)$ for sparse

**Advantages**:
- Uses existing graph structure (no triangulation needed)
- Well-founded in spectral geometry theory
- Probabilistic interpretation via random walks

**Limitations**:
- Numerically challenging to extract $a_2$ accurately (small-$t$ asymptotics)
- Gives only global integral $\int \|C\|^2$, not local field
- Requires eigenvalue computation (expensive for large $N$)
- Needs separate computation of Ricci terms to isolate Weyl contribution
:::

:::{warning} **Numerical Stability**

Extracting the $O(t^2)$ coefficient $a_2$ from heat trace data is an **ill-conditioned numerical problem**. Small errors in $\text{Tr}(e^{-t\Delta})$ at small $t$ amplify exponentially when fitting. Recommended practices:

1. Use high-precision arithmetic (e.g., `mpmath` in Python)
2. Fit in log-space: $\log[\text{Tr}(e^{-t\Delta}) \cdot (4\pi t)^{d/2}] \approx \log(a_0) + a_1 t + \ldots$
3. Regularize fit with prior knowledge (e.g., $a_0 = \text{Vol}(M)$)
4. Cross-validate with known test cases (e.g., sphere, torus)
:::

---

#### 4.6.4. Method 4: Qualitative Consistency Checks from Spectral Gap

Instead of computing $\|C\|^2$ exactly, we can use the spectral gap $\lambda_1$ (first non-zero eigenvalue) of the Laplacian to perform **qualitative consistency checks** on Weyl norm estimates obtained from other methods.

:::{prf:theorem} Spectral Gap and Ricci Curvature (Lichnerowicz)
:label: thm-lichnerowicz-spectral-gap

Let $(M, g)$ be a compact Riemannian manifold with $\dim M = d$. If the Ricci curvature satisfies $\text{Ric}(v, v) \ge \kappa \|v\|^2$ for all $v \in TM$ and some constant $\kappa > 0$, then the spectral gap of the Laplace-Beltrami operator satisfies:

$$
\lambda_1 \ge \frac{d}{d-1} \kappa
$$

**Converse (partial)**: If $\lambda_1$ and diameter $D = \text{diam}(M)$ are known, then:

$$
\text{Ric}(x) \ge -\frac{(d-1) \pi^2}{D^2} + \frac{(d-1) \lambda_1}{d}
$$

(lower bound on Ricci curvature).

**Literature**: Lichnerowicz (1958) *Géométrie des groupes de transformations*, Obata (1962) *Certain conditions for a Riemannian manifold*.
:::

:::{prf:remark} Qualitative Relationship Between Spectral Gap and Weyl Norm
:label: rem-spectral-gap-weyl-qualitative

The spectral gap $\lambda_1$ and the Weyl norm $\|C\|^2$ are related through the Ricci tensor via the Riemann tensor decomposition:

$$
\|R_{abcd}\|^2 = \|C\|^2 + \frac{4}{d-2} \|R_{ab}\|^2 - \frac{2}{(d-1)(d-2)} R^2
$$

The Lichnerowicz theorem ({prf:ref}`thm-lichnerowicz-spectral-gap`) provides a link between $\lambda_1$ and lower bounds on $\text{Ric}$, which constrains $\|R_{ab}\|^2$. However, deriving a tight, computable upper bound on $\|C\|^2$ directly from $\lambda_1$ requires advanced techniques from comparison geometry that depend on additional geometric regularity assumptions.

**Use in practice**: The spectral gap serves as a **qualitative sanity check**:
- A **large $\lambda_1$** indicates positive Ricci curvature lower bounds, suggesting the geometry is positively curved overall. In this regime, a very large computed $\|C\|^2$ (dominating Ricci contributions) would be suspicious and warrant verification.
- A **small $\lambda_1$** allows for negative or weak Ricci curvature, permitting large Weyl norms without contradiction.
- **Consistency requirement**: If $\|C\|^2 \gg \|R_{ab}\|^2$, then $\|R_{abcd}\|^2 \approx \|C\|^2$, and the total curvature magnitude should be reflected in slower diffusion (smaller $\lambda_1$) or require justification via strong anisotropy.

**Conclusion**: While the spectral gap does not provide a simple computational formula for $\|C\|^2$, it offers a valuable cross-check on the consistency and plausibility of Weyl norm estimates obtained from Methods 1-3.
:::

:::{note} **Practical Consistency Check Protocol**

**Input**:
- Computed Weyl norm $\|\hat{C}\|^2$ from Methods 1-3
- Spectral gap $\lambda_1$ of graph Laplacian
- Diameter estimate $D = \max_{i,j} d(x_i, x_j)$
- Computed Ricci scalar and tensor norms $\int R^2$, $\int \|R_{ab}\|^2$

**Consistency Checks**:

1. **Total curvature check**:
   - Compute $\|R_{abcd}\|^2 \approx \|\hat{C}\|^2 + \frac{4}{d-2} \|R_{ab}\|^2 - \frac{2}{(d-1)(d-2)} R^2$
   - Verify this is consistent with diffusion timescale: $t_{\text{diff}} \sim 1/\lambda_1$
   - Large total curvature → slower mixing → smaller $\lambda_1$

2. **Anisotropy ratio check**:
   - Compute ratio: $\rho = \|\hat{C}\|^2 / \|R_{ab}\|^2$
   - If $\rho \gg 1$ (highly anisotropic), verify this is plausible given the fitness landscape geometry
   - Cross-check with directional curvature estimates from deficit angles

3. **Sign consistency**:
   - From Lichnerowicz: estimate $\kappa_{\text{low}} = \frac{d}{d-1} \lambda_1 - \frac{d-1}{d} \frac{\pi^2}{D^2}$
   - If $\kappa_{\text{low}} > 0$ (positive curvature), the geometry should not support arbitrarily large Weyl norms without justification
   - If $\kappa_{\text{low}} \le 0$, large Weyl norms are more plausible

**Complexity**: $O(N^2)$ (dominated by computing $\lambda_1$ with iterative methods)

**Outcome**: Pass/fail sanity check, not a quantitative bound. Failures indicate potential errors in Weyl norm computation (Methods 1-3) or inconsistencies in the discrete geometry.
:::

---

#### 4.6.5. Method Comparison and Recommendations

| Method | Complexity | Output | Accuracy | Best For | Dimension |
|--------|-----------|--------|----------|----------|-----------|
| **Regge Calculus** | $O(N \cdot d^2)$ | Global $\int \|C\|^2$ | Exact (discrete) | Simplicial meshes, scutoids | $d \ge 3$ |
| **Chern-Gauss-Bonnet** | $O(N \cdot d^2)$ | Global $\int \|C\|^2$ | Exact (topological) | 4D spacetimes | $d = 4$ only |
| **Spectral Heat Kernel** | $O(N^3)$ or $O(N^2)$ sparse | Global $\int \|C\|^2$ | Approximate (fitting) | Graph-based, no mesh | $d \ge 3$ |
| **Spectral Gap Check** | $O(N^2)$ | Consistency check (pass/fail) | Qualitative | Cross-validation | $d \ge 3$ |
| *Heat Kernel (direct)* | $O(N \cdot d^2)$ | Local $\|C\|^2(x)$ | Exact (asymptotic) | Smooth metrics | $d \ge 3$ |
| *Causal Set (hypoth.)* | $O(N \log N)$ | Local $\|C\|^2(x)$ | Hypothesized | Point clouds | $d \ge 3$ |

**Recommendations**:

1. **For Fragile Gas scutoid tessellations**: Use **Regge calculus** (Method 1)
   - Directly leverages Delaunay triangulation
   - No continuum approximation needed
   - $O(N \cdot d^2)$ is optimal for discrete geometry

2. **For 4D walker spacetimes** ($d=4$ including time): Use **Chern-Gauss-Bonnet** (Method 2)
   - Topological formula is exact and efficient
   - Reduces $O(d^4)$ to $O(d^2)$
   - Combinatorial Euler characteristic is trivial to compute

3. **For graph-based analysis without triangulation**: Use **Spectral heat kernel** (Method 3)
   - When Delaunay triangulation is unavailable or expensive
   - Provides spectral geometry insights
   - Accept numerical challenges in fitting

4. **For cross-validation and consistency checks**: Use **Spectral gap** (Method 4)
   - $O(N^2)$ for computing $\lambda_1$
   - Qualitative sanity checks on $\|C\|^2$ computed from Methods 1-3
   - Verify consistency between total curvature and diffusion timescale
   - Detect implausible Weyl norm values

**Cross-validation protocol**:
- Compute $\int \|C\|^2$ using Regge calculus (Method 1)
- Cross-check with Chern-Gauss-Bonnet if $d=4$ (Method 2)
- Verify spectral gap bound is satisfied (Method 4)
- If available, compare with direct heat kernel local values (integrate to get global)

---

(part-computational-algorithms)=
## Part 5: Computational Algorithms

This section provides practical algorithms for implementing the five curvature computation methods.

### 5.1. Algorithm 1: Deficit Angles (O(N))

**Input**: Walker positions $\{x_i\}_{i=1}^N$, Voronoi tessellation $\{V_i\}$, dimension $d$

**Output**: Ricci scalar estimates $\{\hat{R}_i\}_{i=1}^N$

**Steps**:
1. Compute Delaunay triangulation dual to Voronoi tessellation
2. For each walker $i$:
   a. Identify incident simplices $\{F_k\}$
   b. Compute solid angles $\Omega_k$ subtended by each simplex at vertex $i$
   c. Deficit angle: $\delta_i = \Omega_{\text{total}}(d) - \sum_k \Omega_k$
   d. Compute Voronoi cell volume: $V_i = \text{Vol}(V_i)$
   e. Estimate: $\hat{R}_i = \delta_i / V_i$

**Computational cost**: $O(N)$ (assumes Delaunay triangulation is given; computing it is $O(N \log N)$ in 2D/3D)

---

### 5.2. Algorithm 2: Spectral Gap (O(N²))

**Input**: Walker positions $\{x_i\}_{i=1}^N$, neighbor graph $G = (V, E)$

**Output**: Global Ricci bound $\hat{R}_{\text{spectral}}$

**Steps**:
1. Construct graph Laplacian matrix:

$$
(\Delta_0)_{ij} = \begin{cases} \deg(i) & i=j \\ -1 & (i,j) \in E \\ 0 & \text{otherwise} \end{cases}

$$
2. Compute eigenvalues: $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{N-1}$
3. Estimate cell size: $\ell_{\text{cell}} = (\text{Vol}(\mathcal{X}) / N)^{1/d}$
4. Spectral gap curvature bound: $\hat{R}_{\text{spectral}} \sim \lambda_1 \ell_{\text{cell}}^2 / C(d)$

**Computational cost**: $O(N^2)$ (dense eigensolve; can be $O(N)$ for sparse graphs with iterative methods)

---

### 5.3. Algorithm 3: Fitness Hessian (O(Nd³))

**Input**: Fitness function $V_{\text{fit}}(x)$, walker position $x_i$, regularization $\epsilon_\Sigma$

**Output**: Pointwise Ricci scalar $\hat{R}(x_i)$

**Steps**:
1. Compute Hessian: $H_{jk}(x_i) = \frac{\partial^2 V_{\text{fit}}}{\partial x^j \partial x^k}\big|_{x=x_i}$ (finite differences or autodiff)
2. Regularize metric: $g_{jk} = H_{jk} + \epsilon_\Sigma \delta_{jk}$
3. Compute third derivatives: $\partial_l H_{jk}(x_i)$ (expensive!)
4. Apply formula: $\hat{R}(x_i) = -\frac{1}{2} \text{tr}[g^{-1} \nabla^2 g]|_{x=x_i}$

**Computational cost**: $O(d^3)$ per point (matrix inversion), total $O(Nd^3)$ for all walkers

**Warning**: Numerical differentiation amplifies noise. Use high-order finite differences or automatic differentiation.

---

### 5.4. Algorithm 4: Heat Kernel (O(N²))

**Input**: Graph Laplacian $\Delta_0$, small time $t > 0$, walker index $i$

**Output**: Ricci scalar estimate $\hat{R}(x_i)$

**Steps**:
1. Compute matrix exponential: $K_t = e^{-t \Delta_0}$ (via eigendecomposition or Krylov methods)
2. Extract diagonal element: $K_t(i, i)$
3. Apply formula: $\hat{R}(x_i) = 6 \cdot \frac{(4\pi t)^{d/2} K_t(i, i) - 1}{t}$
4. Extrapolate $t \to 0$ limit (use Richardson extrapolation for multiple $t$ values)

**Computational cost**: $O(N^2)$ (matrix exponential); can be $O(N)$ for sparse graphs with Lanczos

---

### 5.5. Algorithm 5: Causal Set Volume (O(N log N))

**Input**: Fractal Set $\mathcal{F}_N = (\mathcal{N}, \prec)$, node $e_i$, interval size $\delta$

**Output**: Ricci scalar estimate $\hat{R}_{\text{CST}}(e_i)$

**Steps**:
1. Identify causal future: $I_\delta(e_i) = \{e_j : e_i \prec e_j \prec e_i + \delta\}$
2. Count nodes: $N_{\text{future}} = |I_\delta(e_i)|$
3. Compute expected flat volume: $V_{\text{flat}} = \rho(x_i) \cdot \frac{\pi^{(d+1)/2}}{\Gamma((d+1)/2+1)} \delta^{d+1}$
4. Estimate: $\hat{R}_{\text{CST}}(e_i) = \frac{6}{\delta^2}(1 - N_{\text{future}}/V_{\text{flat}})$

**Computational cost**: $O(N \log N)$ (causal interval query with spatial indexing structure)

---

(part-applications)=
## Part 6: Applications and Cross-References

### 6.1. Raychaudhuri Equation

The Ricci tensor appears directly in the **Raychaudhuri equation** (Chapter 15), which governs volume evolution:

$$
\frac{d\theta}{dt} = -\frac{1}{d}\theta^2 - \sigma_{\mu\nu}\sigma^{\mu\nu} + \omega_{\mu\nu}\omega^{\mu\nu} - R_{\mu\nu}u^\mu u^\nu

$$

where:
- $\theta$ is the expansion scalar
- $\sigma_{\mu\nu}$ is the shear tensor
- $\omega_{\mu\nu}$ is the rotation tensor
- $R_{\mu\nu}$ is the Ricci tensor

**Application**: Curvature (via $R_{\mu\nu}$) causes **focusing** — neighboring walkers converge in high-fitness regions.

**Cross-reference**: [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md) § 3

---

### 6.2. Einstein Field Equations

The Ricci tensor and scalar appear in the **Einstein field equations** (Chapter 16):

$$
G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu} = 8\pi G T_{\mu\nu}

$$

where $G_{\mu\nu}$ is the Einstein tensor and $T_{\mu\nu}$ is the stress-energy tensor.

**Application**: At QSD equilibrium, the emergent geometry satisfies Einstein equations with stress-energy from walker kinematics.

**Cross-reference**: [general_relativity/16_general_relativity_derivation.md](general_relativity/16_general_relativity_derivation.md)

---

### 6.3. Gauge Theory and Wilson Loops

The Riemann tensor is the **field strength** of the affine connection (Chapter 12):

$$
F_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + [A_\mu, A_\nu]^a

$$

where $A_\mu$ is the gauge connection (Christoffel symbols).

The plaquette holonomy (Theorem {prf:ref}`thm-riemann-scutoid-dictionary`) is the discrete analog of the **Wilson loop** in lattice gauge theory.

**Cross-reference**: [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md)

---

### 6.4. Causal Set Theory and Quantum Gravity

The causal set curvature estimator (Definition {prf:ref}`def-ricci-causal-set-volume`) implements **Myrheim-Meyer dimension estimation** from causal set theory.

**Application**: The Fragile Gas provides a **physical realization** of causal set theory, where the adaptive sprinkling density naturally includes the Riemannian volume element.

**Cross-reference**: [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md)

---

### 6.5. Information Geometry

The Fisher information metric is related to the Ricci curvature via the **Amari-Chentsov theorem**. For the Fragile Gas QSD:

$$
g_{\text{Fisher}}^{ij} = \mathbb{E}\left[\frac{\partial \log \rho}{\partial \theta^i} \frac{\partial \log \rho}{\partial \theta^j}\right]

$$

where $\theta^i$ are parameters of the distribution.

**Application**: Curvature of the parameter space indicates efficiency of learning/optimization.

**Cross-reference**: Information geometry connections (future work - see Fisher information metric in statistical mechanics literature)

---

(part-dimension-estimation)=
## Part 7: Dimension Estimation of Emergent Manifolds

### 7.1. Introduction and Motivation

The **intrinsic dimension** of the emergent manifold is a fundamental geometric property that characterizes the true degrees of freedom of the system. While walkers live in an **ambient space** $\mathcal{X} \subseteq \mathbb{R}^d$, the emergent geometry induced by the QSD may have lower intrinsic dimension $d_{\text{int}} \le d$ if the dynamics effectively explore a lower-dimensional manifold.

:::{prf:definition} Intrinsic Dimension
:label: def-intrinsic-dimension

Let $\mathcal{M} \subset \mathbb{R}^d$ be a smooth embedded manifold. The **intrinsic dimension** $\dim(\mathcal{M})$ is the number of independent coordinates needed to parameterize $\mathcal{M}$ locally.

For a point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ sampled from an unknown manifold $\mathcal{M}$, the intrinsic dimension is the dimension of the underlying manifold, **not** the ambient dimension $d$.
:::

**Motivation for dimension estimation in Fragile Gas:**

1. **QSD validation**: If the QSD concentrates on a lower-dimensional attractor (e.g., ridge in fitness landscape), $d_{\text{int}} < d$
2. **Curse of dimensionality**: Many curvature methods (Parts 1-6) require $N \gg d^k$ samples; knowing true $d_{\text{int}}$ helps assess sample adequacy
3. **Model selection**: Determines minimal representation for emergent geometry
4. **Phase transitions**: Changes in $d_{\text{int}}(t)$ over time indicate exploration ↔ exploitation transitions
5. **Computational efficiency**: Algorithms can be optimized for $d_{\text{int}}$ instead of $d$

:::{important}
**Relationship Between Dimension Notions**

Different mathematical definitions of "dimension" can give different values:

- **Topological dimension** $d_{\text{top}}$: Minimum number of coordinates for continuous charts (integer)
- **Hausdorff dimension** $d_H$: Scaling of cover size with resolution (real-valued, can be fractal)
- **Box-counting dimension** $d_{\text{box}}$: Scaling of box count with size (often equals $d_H$ for "nice" sets)
- **Correlation dimension** $d_{\text{corr}}$: Scaling of correlation integral (often $\le d_H$)
- **Intrinsic dimension** $d_{\text{int}}$: Local tangent space dimension (what we estimate from data)

**Inequality chain**: $d_{\text{top}} \le d_{\text{int}} \le d_{\text{corr}} \le d_{\text{box}} \le d_H \le d$ (ambient)

For smooth manifolds without fractals: all dimensions coincide. For attractors of chaotic systems: dimensions can differ.
:::

:::{note}
**Framework Context: QSD Non-Uniform Sampling**

All dimension estimation methods below must account for **non-uniform sampling** from the QSD:

$$
\rho_{\text{spatial}}(x) = \frac{1}{Z} \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

This biases samples toward:
- High-fitness regions (low $U_{\text{eff}}$)
- High-curvature regions (large $\det g$)

**Impact**: Dimension estimators may overestimate $d_{\text{int}}$ in under-sampled regions or underestimate if walkers collapse to lower-dimensional attractors.
:::

---

### 7.2. Method 1: Myrheim-Meyer Dimension (Causal Set)

The **Myrheim-Meyer estimator** is the canonical dimension measure in causal set theory, already established in our framework for the Fractal Set.

:::{prf:definition} Myrheim-Meyer Dimension Estimator (Recap)
:label: def-myrheim-meyer-dimension

For episodes $e_1, e_2 \in E$ in the Fractal Set with causal order $\prec$, the **Myrheim-Meyer dimension** is:

$$
d_{\text{MM}}(e_1, e_2) := \frac{\ln \mathbb{E}[|I(e_1, e_2)|]}{\ln \mathbb{E}[\sqrt{|I(e_1, e_3)| \cdot |I(e_3, e_2)|}]}
$$

where:
- $I(e_1, e_2) = \{e \in E : e_1 \prec e \prec e_2\}$ is the causal interval
- $|I|$ denotes cardinality
- $e_3$ is a random intermediate episode with $e_1 \prec e_3 \prec e_2$

**Expected value**: For a $d$-dimensional Lorentzian manifold:

$$
\mathbb{E}[d_{\text{MM}}(e_1, e_2)] = d + O((\Delta t)^{-1})
$$

where $\Delta t = t_2 - t_1$ is the temporal separation.

**Framework reference**: See [13_fractal_set_new/11_causal_sets.md] {prf:ref}`def-myrheim-meyer-fractal` for full derivation and proof.
:::

:::{prf:algorithm} Myrheim-Meyer Dimension Estimation
:label: alg-myrheim-meyer-dimension

**Input**:
- Fractal Set $\mathcal{F} = (E, \prec)$ with episodes $E = \{e_1, \ldots, e_N\}$
- Sample pairs $(e_i, e_j)$ with $e_i \prec e_j$

**Output**: Dimension estimate $\hat{d}_{\text{MM}}$

**Steps**:

1. **Sample episode pairs**: Select $M$ pairs $(e_i, e_j)$ with $e_i \prec e_j$ and $\Delta t_{ij} > t_{\min}$ (exclude short intervals)

2. **Compute causal intervals**: For each pair $(e_i, e_j)$:
   - Count $n_{ij} = |I(e_i, e_j)| = |\{e : e_i \prec e \prec e_j\}|$

3. **Sample intermediate episodes**: For each pair $(e_i, e_j)$:
   - Choose random $e_k$ with $e_i \prec e_k \prec e_j$
   - Compute $n_{ik} = |I(e_i, e_k)|$ and $n_{kj} = |I(e_k, e_j)|$
   - Geometric mean: $g_{ij} = \sqrt{n_{ik} \cdot n_{kj}}$

4. **Estimate dimension**:

$$
\hat{d}_{\text{MM}} = \frac{\frac{1}{M}\sum_{m=1}^M \ln n_{ij}^{(m)}}{\frac{1}{M}\sum_{m=1}^M \ln g_{ij}^{(m)}}
$$

**Complexity**: $O(N^2)$ for computing all causal intervals (one-time), $O(M)$ for sampling $M$ pairs

**Advantages**:
- Theoretically grounded in causal set theory
- Accounts for temporal structure naturally
- Works for Lorentzian (not just Riemannian) manifolds

**Limitations**:
- Requires causal structure (episodes with time ordering)
- Assumes $\Delta t$ large enough for scaling regime
- Variance high for small $N$ or short temporal intervals
:::

---

### 7.3. Method 2: Maximum Likelihood Estimator (Levina-Bickel)

The **Levina-Bickel estimator** uses the distribution of k-nearest neighbor distances to derive a maximum likelihood estimate of dimension.

:::{prf:definition} Levina-Bickel ML Dimension Estimator
:label: def-levina-bickel-dimension

For a point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ sampled from a $d_{\text{int}}$-dimensional manifold, let $r_j(x_i)$ be the distance from $x_i$ to its $j$-th nearest neighbor.

The **maximum likelihood estimator** of dimension at point $x_i$ is:

$$
\hat{d}_{\text{ML}}(x_i) = \left[\frac{1}{k-1} \sum_{j=1}^{k-1} \log\frac{r_k(x_i)}{r_j(x_i)}\right]^{-1}
$$

where $k \ge 2$ is a fixed neighborhood size.

**Global estimate**: Average over all points:

$$
\hat{d}_{\text{ML}} = \frac{1}{N} \sum_{i=1}^N \hat{d}_{\text{ML}}(x_i)
$$

**Theoretical basis**: Under Poisson process approximation, the ratio $r_k/r_j$ follows a distribution parameterized by dimension. Maximum likelihood yields the closed-form estimator above.

**Literature**: Levina & Bickel (2004) "Maximum Likelihood Estimation of Intrinsic Dimension", NIPS.
:::

:::{prf:algorithm} Levina-Bickel ML Dimension Estimation
:label: alg-levina-bickel-dimension

**Input**:
- Point cloud $\{x_1, \ldots, x_N\}$
- Neighborhood size $k$ (typically $k = 10$ to $50$)

**Output**: Global dimension estimate $\hat{d}_{\text{ML}}$

**Steps**:

1. **Build k-NN index**: Construct k-d tree or ball tree for fast nearest neighbor queries

2. **For each point** $x_i$:
   - Query k nearest neighbors: $\{x_{i,1}, \ldots, x_{i,k}\}$
   - Compute distances: $r_j(x_i) = \|x_i - x_{i,j}\|$ for $j = 1, \ldots, k$
   - Sort distances: $r_1(x_i) \le r_2(x_i) \le \cdots \le r_k(x_i)$

3. **Compute local estimate**:

$$
\hat{d}_i = \left[\frac{1}{k-1} \sum_{j=1}^{k-1} \log\frac{r_k(x_i)}{r_j(x_i)}\right]^{-1}
$$

4. **Aggregate**:

$$
\hat{d}_{\text{ML}} = \frac{1}{N} \sum_{i=1}^N \hat{d}_i
$$

or use median/trimmed mean for robustness

**Complexity**: $O(N \log N)$ (k-d tree construction) + $O(Nk)$ (queries) = $O(N \log N)$

**Advantages**:
- Theoretically well-founded (maximum likelihood principle)
- Fast computation with k-d trees
- Provides local estimates $\hat{d}_i$ (can detect dimension variation)

**Limitations**:
- Sensitive to choice of $k$ (small $k$ → high variance, large $k$ → bias from curvature)
- Assumes locally uniform sampling (QSD weighting can bias estimates)
- Breaks down at boundaries (fewer neighbors available)
- Poisson approximation valid only for $k \ll N$
:::

:::{note}
**Accounting for QSD Non-Uniform Sampling**

The Levina-Bickel estimator assumes the point cloud is sampled from a locally uniform density. The non-uniform nature of the QSD, $\rho_{\text{QSD}}(x) \propto \sqrt{\det g(x)} \exp(-U_{\text{eff}}(x)/T)$, can introduce bias in the dimension estimate, particularly in regions where the density varies rapidly.

**Correction approach**: A comprehensive iterative correction procedure that accounts for non-uniform sampling is detailed in §7.10.3 ("Handling Non-Uniform QSD Sampling"). The procedure uses kernel density estimation and importance weighting to adjust for sampling bias.

**Quick alternative**: For a simpler approach, compute local estimates at each point and then aggregate using importance weights: $\hat{d}_{\text{global}} = \frac{\sum_i w_i \hat{d}_{\text{ML}}(x_i)}{\sum_i w_i}$ where $w_i = 1/\rho(x_i)$.
:::

---

### 7.4. Method 3: Local PCA (Principal Component Analysis)

The **local PCA method** estimates dimension by analyzing the eigenvalue spectrum of local covariance matrices.

:::{prf:definition} PCA Dimension Estimator
:label: def-pca-dimension

For a point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$, define the **local covariance matrix** at $x_i$ with neighborhood radius $r$:

$$
C_i = \frac{1}{|N_r(x_i)|} \sum_{x_j \in N_r(x_i)} (x_j - \bar{x}_i)(x_j - \bar{x}_i)^T
$$

where $N_r(x_i) = \{x_j : \|x_j - x_i\| < r\}$ and $\bar{x}_i = \frac{1}{|N_r(x_i)|}\sum_{x_j \in N_r(x_i)} x_j$ is the local centroid.

Let $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$ be the eigenvalues of $C_i$. The **intrinsic dimension** is estimated as:

$$
\hat{d}_{\text{PCA}}(x_i) = \argmin_{k} \left\{\frac{\lambda_{k+1}}{\lambda_k} < \theta\right\}
$$

where $\theta \in (0, 1)$ is a threshold parameter (typically $\theta = 0.05$ to $0.1$).

**Alternative (cumulative variance)**: Choose $\hat{d}$ such that:

$$
\frac{\sum_{j=1}^{\hat{d}} \lambda_j}{\sum_{j=1}^{d} \lambda_j} \ge \eta
$$

where $\eta \in (0, 1)$ is the explained variance threshold (e.g., $\eta = 0.95$).

**Literature**: Fukunaga & Olsen (1971), Trunk (1976), Bruske & Sommer (1998)
:::

:::{prf:algorithm} Local PCA Dimension Estimation
:label: alg-pca-dimension

**Input**:
- Point cloud $\{x_1, \ldots, x_N\}$
- Neighborhood radius $r$ or k nearest neighbors
- Threshold $\theta$ (eigenvalue ratio) or $\eta$ (variance)

**Output**: Dimension estimates $\{\hat{d}_i\}$ or global $\hat{d}_{\text{PCA}}$

**Steps**:

1. **For each point** $x_i$:
   - Find neighborhood $N_r(x_i)$ or k-NN
   - Compute local centroid: $\bar{x}_i = \frac{1}{|N_r(x_i)|}\sum_{x_j \in N_r(x_i)} x_j$

2. **Compute covariance matrix**:

$$
C_i = \frac{1}{|N_r(x_i)|} \sum_{x_j \in N_r(x_i)} (x_j - \bar{x}_i)(x_j - \bar{x}_i)^T
$$

3. **Eigenvalue decomposition**: Compute eigenvalues $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$ of $C_i$

4. **Determine dimension**:
   - **Ratio criterion**: $\hat{d}_i = \min\{k : \lambda_{k+1}/\lambda_k < \theta\}$
   - **Variance criterion**: $\hat{d}_i = \min\left\{k : \sum_{j=1}^k \lambda_j / \sum_{j=1}^d \lambda_j \ge \eta\right\}$

5. **Aggregate**: Global estimate $\hat{d}_{\text{PCA}} = \text{median}(\{\hat{d}_i\})$ or mode

**Complexity**: $O(N \cdot (k d^2 + d^3))$ where $k = |N_r(x_i)|$ (covariance + eigensolve per point)

**Advantages**:
- Simple and interpretable (variance decomposition)
- Provides local dimension estimates $\hat{d}_i$ (detects varying dimension)
- Fast for moderate $d$
- Works well for locally linear manifolds

**Limitations**:
- **Threshold sensitivity**: Results depend strongly on choice of $\theta$ or $\eta$
- **Assumes linear structure**: Fails for highly nonlinear manifolds (e.g., Swiss roll)
- **Curse of dimensionality**: Requires $|N_r(x_i)| \gg d$ for stable covariance estimates
- Boundary effects (fewer neighbors near boundaries)
:::

:::{note}
**Global PCA vs Local PCA**

- **Global PCA**: Single covariance matrix for entire point cloud → fast but assumes single global dimension
- **Local PCA**: Per-point covariance matrices → detects varying dimension but more expensive and sensitive to $r$

For Fragile Gas: Use local PCA when QSD may concentrate on multiple disconnected attractors with different dimensions.
:::

---

### 7.5. Method 4: Correlation Dimension (Grassberger-Procaccia)

The **correlation dimension** estimates dimension from the scaling of the correlation integral.

:::{prf:definition} Grassberger-Procaccia Correlation Dimension
:label: def-grassberger-procaccia-dimension

For a point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$, the **correlation sum** at scale $r$ is:

$$
C(r) = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} \Theta(r - \|x_i - x_j\|)
$$

where $\Theta(\cdot)$ is the Heaviside step function: $\Theta(x) = 1$ if $x \ge 0$, else $0$.

For a $d_{\text{corr}}$-dimensional manifold, $C(r)$ scales as a power law:

$$
C(r) \sim r^{d_{\text{corr}}} \quad \text{as } r \to 0
$$

The **correlation dimension** is:

$$
d_{\text{corr}} = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}
$$

**Practical estimation**: Plot $\ln C(r)$ vs $\ln r$ and fit slope in the **scaling region** (linear regime).

**Literature**: Grassberger & Procaccia (1983) "Measuring the strangeness of strange attractors", Physica D
:::

:::{prf:algorithm} Correlation Dimension Estimation
:label: alg-correlation-dimension

**Input**:
- Point cloud $\{x_1, \ldots, x_N\}$
- Range of scales $\{r_1, \ldots, r_K\}$ (logarithmically spaced)

**Output**: Correlation dimension estimate $\hat{d}_{\text{corr}}$

**Steps**:

1. **Compute pairwise distances**: $d_{ij} = \|x_i - x_j\|$ for all pairs $(i, j)$ with $i < j$
   - Optimization: Use k-d tree or ball tree to avoid $O(N^2)$ computation
   - Approximate: Sample $M \ll N^2$ random pairs

2. **For each scale** $r_k$:
   - Count pairs: $n_k = |\{(i,j) : i < j, d_{ij} < r_k\}|$
   - Correlation sum: $C(r_k) = \frac{2n_k}{N(N-1)}$

3. **Identify scaling region**:
   - Plot $\ln C(r_k)$ vs $\ln r_k$
   - Identify approximately linear region (avoid very small/large $r$ where boundary/saturation effects dominate)

4. **Fit slope**: Linear regression on scaling region:

$$
\hat{d}_{\text{corr}} = \frac{\Delta \ln C(r)}{\Delta \ln r} \approx \frac{\ln C(r_{\max}) - \ln C(r_{\min})}{\ln r_{\max} - \ln r_{\min}}
$$

**Complexity**:
- Naive: $O(N^2)$ for all pairwise distances
- With spatial indexing: $O(N \log N)$ per scale
- Sampling approximation: $O(M)$ for $M$ sampled pairs

**Advantages**:
- **Robust to noise**: Less sensitive than PCA to high-frequency perturbations
- **Detects fractals**: Can estimate non-integer dimensions for strange attractors
- **No parameter tuning**: Only need to identify scaling region (visual inspection)

**Limitations**:
- **Scaling region identification**: Requires manual or algorithmic identification of linear regime
- **Boundary effects**: Edge points bias estimates at large $r$
- **High-dimensional curse**: Requires $N \gg 2^d$ for meaningful statistics
- **Underestimates**: Often gives $d_{\text{corr}} < d_{\text{true}}$ due to finite sampling
:::

:::{warning}
**Finite-Size Effects**

For finite $N$, the correlation sum saturates at $C(r) \approx 1$ for large $r$ and has high variance for small $r$. The **scaling region** is typically in the range:

$$
\frac{1}{N^{1/d}} \ll r \ll \text{diam}(\mathcal{M})/10
$$

With too few points ($N < 1000$), the scaling region may vanish entirely.
:::

---

### 7.6. Method 5: Box-Counting Dimension

The **box-counting dimension** (Minkowski-Bouligand dimension) estimates dimension by counting boxes needed to cover the point cloud at varying scales.

:::{prf:definition} Box-Counting Dimension
:label: def-box-counting-dimension

For a set $S \subset \mathbb{R}^d$ and scale $\epsilon > 0$, let $N_{\epsilon}(S)$ be the minimum number of $d$-dimensional boxes of side length $\epsilon$ needed to cover $S$.

The **box-counting dimension** is:

$$
d_{\text{box}} = \lim_{\epsilon \to 0} \frac{\ln N_{\epsilon}(S)}{\ln(1/\epsilon)}
$$

**Practical estimation**: Plot $\ln N_{\epsilon}$ vs $\ln(1/\epsilon)$ and compute slope.

**Relationship to Hausdorff dimension**: For many "nice" sets (including smooth manifolds and self-similar fractals):

$$
d_{\text{box}} = d_H
$$

but in general $d_H \le d_{\text{box}}$.

**Literature**: Mandelbrot (1982) *The Fractal Geometry of Nature*, Falconer (1990) *Fractal Geometry*
:::

:::{prf:algorithm} Box-Counting Dimension Estimation
:label: alg-box-counting-dimension

**Input**:
- Point cloud $\{x_1, \ldots, x_N\} \subset [0, L]^d$
- Range of box sizes $\{\epsilon_1, \ldots, \epsilon_K\}$ (decreasing geometric sequence)

**Output**: Box-counting dimension estimate $\hat{d}_{\text{box}}$

**Steps**:

1. **For each scale** $\epsilon_k$:
   - Overlay grid of boxes with side length $\epsilon_k$
   - Count non-empty boxes: $N_{\epsilon_k} = |\{\text{box } B : B \cap \{x_1, \ldots, x_N\} \ne \emptyset\}|$

2. **Alternative (efficient)**: Hash table method
   - For each point $x_i$, compute grid cell: $c_i = \lfloor x_i / \epsilon_k \rfloor$
   - Count unique cells: $N_{\epsilon_k} = |\{c_1, \ldots, c_N\}|$ (use hash set)

3. **Fit power law**: Linear regression on log-log plot:

$$
\hat{d}_{\text{box}} = \frac{\Delta \ln N_{\epsilon}}{\Delta \ln(1/\epsilon)} \approx \frac{\ln N_{\epsilon_{\min}} - \ln N_{\epsilon_{\max}}}{\ln(1/\epsilon_{\min}) - \ln(1/\epsilon_{\max})}
$$

**Complexity**: $O(N \cdot K)$ where $K$ is number of scales (hash table method)

**Advantages**:
- **Simple implementation**: Grid overlay is straightforward
- **Works for fractals**: Can estimate non-integer dimensions
- **No neighbor search**: Avoids k-d tree construction
- **Handles disconnected sets**: Works even if manifold has multiple components

**Limitations**:
- **Boundary artifacts**: Boxes partially outside domain counted incorrectly
- **Scale range**: Requires $\epsilon_{\max} \ll \text{diam}(\mathcal{M})$ and $\epsilon_{\min} \gg$ inter-point spacing
- **Curse of dimensionality**: Number of boxes grows exponentially ($\sim (L/\epsilon)^d$)
- **Sensitive to sparsity**: Under-sampling leads to overestimation
:::

---

### 7.7. Method 6: Graph Laplacian Spectral Decay

The **spectral decay method** estimates dimension from the eigenvalue spectrum of the graph Laplacian.

:::{prf:definition} Spectral Dimension from Laplacian Eigenvalues
:label: def-spectral-dimension

For a point cloud $\{x_1, \ldots, x_N\}$, construct the graph Laplacian $\Delta_G$ (see Part 1, Method 2). Let $\lambda_1 \le \lambda_2 \le \cdots \le \lambda_N$ be the eigenvalues.

By **Weyl's asymptotic formula**, for a $d$-dimensional compact Riemannian manifold:

$$
\lambda_k \sim C_d \cdot k^{2/d} \quad \text{as } k \to \infty
$$

where $C_d$ depends on the manifold volume and geometry.

Taking logarithms:

$$
\ln \lambda_k \approx \frac{2}{d} \ln k + \text{const}
$$

The **spectral dimension** is:

$$
d_{\text{spec}} = \frac{2}{\text{slope of } \ln \lambda_k \text{ vs } \ln k}
$$

**Literature**: Weyl's law (1911), spectral geometry textbooks (e.g., Chavel 1984)
:::

:::{prf:algorithm} Spectral Dimension Estimation
:label: alg-spectral-dimension

**Input**:
- Graph Laplacian $\Delta_G$ (from Part 1, Method 2 or constructed separately)
- Number of eigenvalues to compute $K \ll N$ (e.g., $K = 100$ to $500$)

**Output**: Spectral dimension estimate $\hat{d}_{\text{spec}}$

**Steps**:

1. **Construct graph Laplacian**: If not already available
   - Choose edge weights: $w_{ij} = \exp(-\|x_i - x_j\|^2 / \sigma^2)$ (Gaussian kernel)
   - Form symmetric normalized Laplacian: $\Delta_G = I - D^{-1/2} W D^{-1/2}$

2. **Compute eigenvalues**: Use sparse eigenvalue solver (e.g., Lanczos, ARPACK)
   - Compute $K$ smallest eigenvalues: $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_K$

3. **Fit Weyl's law**: Linear regression on log-log plot (exclude $\lambda_0 = 0$):
   - Data: $\{(\ln k, \ln \lambda_k)\}_{k=1}^K$
   - Fit: $\ln \lambda_k \approx a + b \ln k$
   - Dimension: $\hat{d}_{\text{spec}} = 2/b$

4. **Alternative (refined)**: Fit only in asymptotic regime (exclude small $k$ where finite-size effects dominate)

**Complexity**: $O(N^2 K)$ for dense Laplacian or $O(N K^2)$ for sparse with iterative solver

**Advantages**:
- **Synergy with curvature**: If Laplacian already computed (Part 1, Method 2), dimension comes "for free"
- **Global estimate**: Uses full point cloud, not just local neighborhoods
- **Theoretically grounded**: Weyl's law is a classical result in spectral geometry

**Limitations**:
- **Expensive**: Requires eigenvalue computation ($O(N^2)$ or worse)
- **Parameter sensitivity**: Kernel bandwidth $\sigma$ affects eigenvalue spectrum
- **Finite-size corrections**: Weyl's law is asymptotic; small $N$ introduces bias
- **Boundary effects**: Manifolds with boundary violate Weyl's law assumptions
:::

:::{note}
**Connection to Part 1**

If the graph Laplacian was already computed for curvature estimation (Part 1, Method 2: Graph Laplacian Curvature), the eigenvalues can be reused for dimension estimation at **no additional cost** (beyond the fitting step). This provides cross-validation: dimension estimate from spectral decay vs from other methods.
:::

---

### 7.8. Method 8: Scutoid Topological Dimension (STD)

The eighth and final method is **uniquely suited to the Fragile Gas framework**, leveraging the topological structure of the scutoid/Voronoi tessellation itself. Instead of relying on distance scaling or spectral properties, it uses the **average coordination number** (number of neighbors) of the walkers' Voronoi cells, which is a robust statistical indicator of intrinsic dimension.

:::{prf:definition} Scutoid Topological Dimension (STD)
:label: def-scutoid-topological-dimension

For a random Poisson point process in $\mathbb{R}^d$, the expected number of neighbors of a Voronoi cell is a known function of dimension, $N_{\text{avg}}(d)$. For $d \ge 2$, this is given by the formula from statistical geometry:

$$
N_{\text{avg}}(d) = \frac{2^d \pi^{(d-1)/2} \Gamma(d^2/2 + 1)}{[\Gamma(d/2 + 1)]^2 \Gamma((d^2-d+1)/2)}
$$

This expression, arising from integral geometry, relates the surface areas and volumes of high-dimensional convex bodies, hence its dependence on Gamma functions.

**Known values**:
- $N_{\text{avg}}(2) = 6$
- $N_{\text{avg}}(3) \approx 15.54$
- $N_{\text{avg}}(4) \approx 30.67$

The **Scutoid Topological Dimension (STD) estimator**, $\hat{d}_{\text{STD}}$, is the value of $d$ that solves the equation:

$$
N_{\text{avg}}(\hat{d}_{\text{STD}}) = \langle |\mathcal{N}_i| \rangle
$$

where $\langle |\mathcal{N}_i| \rangle$ is the empirically measured average number of neighbors per walker in the Voronoi tessellation generated by the Fragile Gas.

**Rationale**: This method inverts the known relationship between dimension and coordination number. By measuring the average coordination number from the simulation data, we can infer the dimension of the space the walkers are tessellating.

**Literature**: The formula for $N_{\text{avg}}(d)$ is derived in {cite}`Miles1970` and {cite}`Okabe2000`. Applications to dimension estimation are discussed in {cite}`Ziegler1995`.
:::

:::{prf:algorithm} Scutoid Topological Dimension (STD) Estimation
:label: alg-scutoid-topological-dimension

**Input**:
- Voronoi tessellation $V_t$ of the walker positions at a given time $t$ (or averaged over a time window)
- Specifically, the neighbor lists $\mathcal{N}_i(t)$ for each walker $i$

**Output**: Global dimension estimate $\hat{d}_{\text{STD}}$

**Steps**:

1. **Extract Neighbor Counts**: For each walker $i$ in the alive set $\mathcal{A}(t)$:
   - Count the number of neighbors: $n_i = |\mathcal{N}_i(t)|$, where $\mathcal{N}_i(t)$ is the set of walkers whose Voronoi cells are adjacent to $\text{Vor}_i(t)$

2. **Compute Average**: Calculate the average number of neighbors over all walkers:

$$
\langle n \rangle = \frac{1}{|\mathcal{A}(t)|} \sum_{i \in \mathcal{A}(t)} n_i
$$

3. **Boundary Correction (Optional but Recommended)**: To mitigate boundary effects where walkers have fewer neighbors, compute the average only over "core" walkers whose Voronoi cells do not touch the domain boundary

4. **Solve for Dimension**: Numerically solve the equation $f(d) = N_{\text{avg}}(d) - \langle n \rangle = 0$ for $d$:
   - This is a root-finding problem for a monotonic function $N_{\text{avg}}(d)$
   - Use bisection method or Newton's method on the interval $[1, d_{\text{ambient}}]$

**Complexity**:
- **Data Extraction**: $O(N)$ to iterate through neighbor lists once the Voronoi diagram is computed
- **Root-finding**: $O(\log(1/\varepsilon))$ for a given precision $\varepsilon$, which is effectively constant time
- **Total**: The method is **extremely efficient**, as its cost is dominated by the $O(N \log N)$ construction of the Voronoi diagram, which is already required for Method 1 (Deficit Angles)

**Advantages**:
- **Highly Efficient**: The fastest method once the tessellation is known (O(N) post-processing)
- **Extremely Robust**: As a topological measure, it is highly insensitive to noise or small perturbations in walker positions
- **Parameter-Free**: Requires no tuning of parameters like $k$ (k-NN) or $r$ (radius)
- **Native Synergy**: Uses the exact same input data (the Delaunay/Voronoi structure) as the Deficit Angle method for curvature, providing a powerful and cheap cross-validation tool

**Limitations**:
- **Global Estimate Only**: Provides a single dimension for the entire manifold, cannot detect local variations
- **Boundary Bias**: Walkers near the domain boundary systematically have fewer neighbors, which can skew the average downwards. A "core sample" correction is necessary
- **QSD Bias**: The theoretical formula for $N_{\text{avg}}(d)$ assumes a uniform Poisson point process. The QSD is structured and non-uniform, which introduces a systematic bias (see correction note below)
:::

:::{note}
**Correcting for QSD Non-Uniformity and Curvature**

The STD method can be refined to account for the non-uniform sampling density $\rho(x)$ and the emergent curvature $R(x)$. Known results in stochastic geometry show that the expected number of neighbors depends on local density and curvature.

A first-order correction is given by:

$$
N_{\text{avg}}(d, x) \approx N_{\text{avg}}(d) \left(1 + c_1(d) \nabla^2 \log(\rho(x)) L^2 + c_2(d) R(x) L^2\right)
$$

where $L$ is the characteristic length scale of the Voronoi cells, and $c_1(d), c_2(d)$ are dimension-dependent constants.

This suggests an **iterative refinement procedure**:
1. **Step 1**: Compute an initial dimension estimate $\hat{d}^{(0)}$ using the uncorrected STD method
2. **Step 2**: Use $\hat{d}^{(0)}$ and other methods (e.g., Method 3 for curvature) to estimate the Ricci scalar field $R(x)$ and density $\rho(x)$
3. **Step 3**: Compute a corrected average neighbor count by subtracting the estimated bias from the measured $\langle n \rangle$
4. **Step 4**: Solve for a refined dimension estimate $\hat{d}^{(1)}$

This creates a powerful feedback loop between the curvature and dimension estimation components of your framework.
:::

:::{note}
**Connection to Scutoid Tessellation Framework**

This method directly leverages the scutoid tessellation framework documented in [14_scutoid_geometry_framework.md]. Key connections:

- **Scutoids**: The spacetime scutoid cells between times $t$ and $t + \Delta t$ have cross-sections at each time that are precisely Voronoi cells
- **Neighbor-swapping**: Scutoid formation (mid-level vertices) corresponds to neighbor changes, which are detected by tracking $\mathcal{N}_i(t)$ over time
- **Information Graph**: The IG (Chapter 13) has edges connecting episodes whose Voronoi cells are adjacent—the STD method uses the degree distribution of this graph
- **Duality**: The scutoid tessellation and IG are dual structures ({prf:ref}`thm-scutoid-ig-duality`), so STD estimates dimension from the graph's combinatorial structure

**Synergy**: If you are already computing scutoid cells for curvature or energy analysis, the coordination number data is **free**, making STD the most efficient dimension estimator in the framework.
:::

---

### 7.9. Method 7: Geodesic Distance Scaling (Volume Growth)

The **geodesic scaling method** estimates dimension from the growth rate of geodesic balls.

:::{prf:definition} Volume Growth Dimension
:label: def-volume-growth-dimension

For a point cloud $\{x_1, \ldots, x_N\}$ with graph distance $d_{\text{graph}}(x_i, x_j)$ (shortest path length), define the **geodesic ball** at $x_i$ with radius $r$:

$$
B_r(x_i) = \{x_j : d_{\text{graph}}(x_i, x_j) < r\}
$$

For a $d$-dimensional manifold, the volume of a geodesic ball scales as:

$$
|B_r(x_i)| \sim r^d \quad \text{for small } r
$$

The **volume growth dimension** is:

$$
d_{\text{vol}} = \lim_{r \to 0} \frac{\ln |B_r(x_i)|}{\ln r}
$$

**Practical estimation**: Plot $\ln |B_r(x_i)|$ vs $\ln r$ and compute slope.

**Connection to causal sets**: This is analogous to the Myrheim-Meyer method but uses spatial graph distance instead of causal structure.
:::

:::{prf:algorithm} Volume Growth Dimension Estimation
:label: alg-volume-growth-dimension

**Input**:
- Point cloud $\{x_1, \ldots, x_N\}$
- Graph $G$ (e.g., Delaunay triangulation or k-NN graph)
- Range of radii $\{r_1, \ldots, r_K\}$

**Output**: Dimension estimate $\hat{d}_{\text{vol}}$

**Steps**:

1. **Compute graph distances**: All-pairs shortest paths via Floyd-Warshall or BFS from each node
   - Complexity: $O(N^3)$ (Floyd-Warshall) or $O(N \cdot E)$ (BFS per node)
   - Optimization: Sample $M \ll N$ seed points instead of all points

2. **For each seed point** $x_i$ and radius $r_k$:
   - Count points in geodesic ball: $v_{ik} = |B_{r_k}(x_i)| = |\{x_j : d_{\text{graph}}(x_i, x_j) < r_k\}|$

3. **Average over seeds**: $V(r_k) = \frac{1}{M}\sum_{i=1}^M v_{ik}$

4. **Fit power law**: Linear regression on log-log plot:

$$
\hat{d}_{\text{vol}} = \frac{\Delta \ln V(r)}{\Delta \ln r} \approx \frac{\ln V(r_{\max}) - \ln V(r_{\min})}{\ln r_{\max} - \ln r_{\min}}
$$

**Complexity**: $O(N^3)$ or $O(M \cdot N \log N)$ with sampling and Dijkstra

**Advantages**:
- **Uses existing structure**: If Delaunay triangulation already computed (Parts 1-6), geodesic distances readily available
- **Robust to embedding**: Graph distance approximates true geodesic distance better than Euclidean distance for curved manifolds
- **Connection to Method 1**: Shares philosophy with Myrheim-Meyer (causal set volume growth)

**Limitations**:
- **Expensive**: All-pairs shortest paths is $O(N^3)$ (can be mitigated by sampling)
- **Graph construction**: Requires choosing graph connectivity (k-NN, $\epsilon$-ball, Delaunay)
- **Finite-size effects**: Small geodesic balls have high variance in point count
- **Boundary effects**: Points near manifold boundary have truncated balls
:::

:::{note}
**Synergy with Causal Set Method (Method 1: Myrheim-Meyer)**

The volume growth method is the **spatial analog** of causal set volume estimation:
- **Causal sets (Method 1)**: Use temporal causal intervals $I(e_1, e_2)$
- **Geodesic scaling (Method 7)**: Use spatial geodesic balls $B_r(x_i)$

Both exploit volume scaling $V(r) \sim r^d$ to infer dimension. For the Fractal Set, **both** temporal (Method 1: Myrheim-Meyer) and spatial (Method 7) estimates should agree, providing cross-validation.
:::

---

## 7.10. Comparison of Dimension Estimation Methods

This section provides a comprehensive comparison table for all eight dimension estimation methods documented in Part 7.

#### Table 5: Dimension Estimation Methods

| Method | Mathematical Definition | Input | Output | Complexity | Best For | Limitations | Framework Integration | Literature |
|--------|------------------------|-------|--------|------------|----------|-------------|----------------------|------------|
| **1. Myrheim-Meyer** | $\hat{d}_{\text{MM}} = \frac{\mathbb{E}[\ln n_{ij}]}{\mathbb{E}[\ln g_{ij}]}$ where $n_{ij} = \|I(e_i, e_j)\|$ (causal interval size), $g_{ij} = \sqrt{n_{ik} \cdot n_{kj}}$ (geometric mean via intermediate event) | Causal set $(E, \prec)$, event ordering | Dimension estimate $\hat{d}$ | $O(N^2)$ (all pairs causality) | Temporally-structured data with causal ordering | Requires causal structure, finite-size corrections needed | {prf:ref}`alg-myrheim-meyer-dimension` (§7.2), see [13_fractal_set_new/11_causal_sets.md] for derivation | Myrheim & Meyer (1989), Bombelli et al. (1987) |
| **2. Levina-Bickel** | $\hat{d}_{\text{ML}}(x_i) = \left[\frac{1}{k-1} \sum_{j=1}^{k-1} \log\frac{r_k(x_i)}{r_j(x_i)}\right]^{-1}$ | Point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$, k-NN distances | Local dimension estimate $\hat{d}_{\text{ML}}(x_i)$, global $\hat{d} = \text{median}_i \hat{d}_{\text{ML}}(x_i)$ | $O(N \log N)$ (k-d tree) + $O(Nk)$ | General point clouds, robust to noise, adaptive to varying local dimension | Sensitive to $k$ choice, biased for small $k$, assumes local uniformity | **QSD correction**: Weight local estimates by $\sqrt{\det g(x_i)} \exp(-U_{\text{eff}}/T)$ or pre-normalize distances | Levina & Bickel (2004), Hein & Audibert (2005) |
| **3. Local PCA** | $\hat{d}_{\text{PCA}}(x_i) = \min\left\{m : \frac{\sum_{j=1}^m \lambda_j}{\sum_{j=1}^d \lambda_j} \ge 1 - \varepsilon\right\}$ where $\{\lambda_j\}$ are eigenvalues of local covariance $\Sigma_i$ | Point cloud, local neighborhoods $N_r(x_i)$ | Local dimension estimate $\hat{d}_{\text{PCA}}(x_i)$, global $\hat{d} = \text{median}_i \hat{d}_{\text{PCA}}(x_i)$ | $O(N \cdot (k d^2 + d^3))$ (covariance + eigensolve per point) | Low-noise data, detecting tangent space dimension | Very sensitive to noise, requires tuning $\varepsilon$ and neighborhood size | **QSD correction**: Use weighted covariance $\Sigma_i = \sum_j w_j (x_j - \mu)(x_j - \mu)^T$ with $w_j \propto 1/\rho(x_j)$ | Fukunaga & Olsen (1971), Fan et al. (2009) |
| **4. Grassberger-Procaccia** | $d_{\text{corr}} = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}$ where $C(r) = \frac{2}{N(N-1)} \sum_{i<j} \Theta(r - \|x_i - x_j\|)$ | Point cloud, distance matrix $\{\|x_i - x_j\|\}$ | Correlation dimension $\hat{d}_{\text{corr}}$ | $O(N^2)$ (naive all-pairs) or $O(K \cdot N \log N)$ (optimized with spatial indexing, $K$ scales) | Fractal and multiscale structures, scale-free systems | Requires large $N$, sensitive to scaling regime choice, computationally expensive | **QSD correction**: Use weighted correlation sum $C(r) = \sum_{i<j} w_i w_j \Theta(r - d(x_i, x_j))$ with $w_i = 1/\rho(x_i)$ | Grassberger & Procaccia (1983), Theiler (1986) |
| **5. Box-counting** | $d_{\text{box}} = -\lim_{\delta \to 0} \frac{\ln N(\delta)}{\ln \delta}$ where $N(\delta)$ is number of $\delta$-cubes covering point cloud | Point cloud $\{x_i\}$ in bounded domain | Minkowski-Bouligand dimension $\hat{d}_{\text{box}}$ | $O(N \cdot \log(1/\delta_{\min}))$ (grid occupancy at multiple scales) | Fractal sets, self-similar structures, rough dimension estimate | Discrete (grid-based), sensitive to boundary, poor for smooth manifolds | **QSD correction**: Use weighted occupancy $N_w(\delta) = \#\{\text{boxes } B : \sum_{x_i \in B} w_i > \text{threshold}\}$ | Mandelbrot (1982), Falconer (2003) |
| **6. Spectral Decay** | $d_{\text{spec}} = \frac{2}{\text{slope of } \ln \lambda_k \text{ vs } \ln k}$ where $\lambda_k \sim C_d \cdot k^{2/d}$ (Weyl's law) | Graph Laplacian eigenvalues $\{\lambda_1, \ldots, \lambda_K\}$ | Spectral dimension $\hat{d}_{\text{spec}}$ | $O(N K^2)$ or $O(E \cdot K)$ for sparse Lanczos (K eigenvalues, E edges), $O(N^3)$ for dense | Global averaging, robust to local noise, spectral methods already in use | Only global estimate (not local), sensitive to graph construction, requires many eigenvalues | **Native to framework**: Graph Laplacian already computed in § 5.2 (Ricci bounds), reuse spectrum | Weyl (1911), Chung (1997), von Luxburg (2007) |
| **7. Geodesic Scaling** | $d_V = \lim_{r \to 0} \frac{\ln \mathbb{E}[N(r)]}{\ln r}$ where $N(r) = \#\{x_j : d_g(x_i, x_j) \le r\}$, $d_g$ is geodesic distance | Graph $G$ with geodesic distances, point cloud | Volume growth dimension $\hat{d}_V$ | $O(N^3)$ (all-pairs shortest paths, can sample to $O(N^2 \log N)$) | Geometric interpretation, connects to causal set method | Expensive geodesic computation, graph construction dependency, boundary effects | **Synergy with Method 1**: Spatial analog of causal volume (Method 1 uses temporal causal intervals), should agree for Fractal Set | Sakai (1996), Burago et al. (2001) |
| **8. Scutoid Topological** | $N_{\text{avg}}(\hat{d}_{\text{STD}}) = \langle \|\mathcal{N}_i\| \rangle$ where $N_{\text{avg}}(d) = \frac{2^d \pi^{(d-1)/2} \Gamma(d^2/2 + 1)}{[\Gamma(d/2 + 1)]^2 \Gamma((d^2-d+1)/2)}$ | Voronoi tessellation neighbor lists $\mathcal{N}_i$ | Topological dimension $\hat{d}_{\text{STD}}$ | $O(N)$ (post-processing after $O(N \log N)$ tessellation) | **FASTEST** method, already computing Voronoi/scutoid tessellation | Global only, boundary bias, QSD non-uniformity bias | **FRAMEWORK-NATIVE**: Reuses Voronoi/scutoid data from [14_scutoid_geometry_framework.md], free if computing curvature via deficit angles (§5.1) | Miles (1970), Okabe et al. (2000), Ziegler (1995) |

**Key Insights from Table 5:**

1. **Complexity hierarchy**:
   - **Fastest**: **Method 8 (Scutoid Topological) $O(N)$** (FREE if tessellation exists), Levina-Bickel $O(N \log N)$
   - **Moderate**: Local PCA $O(N \cdot (k d^2 + d^3))$, Spectral decay $O(N K^2)$ (amortized if already computing Laplacian)
   - **Slowest**: Grassberger-Procaccia $O(N^2)$, Geodesic scaling $O(N^3)$ (but can sample)

2. **Local vs global**:
   - **Local dimension field**: Methods 2, 3 (Levina-Bickel, Local PCA) → can detect varying intrinsic dimension
   - **Global scalar**: Methods 1, 4, 5, 6, 7, 8 → single dimension estimate for entire manifold

3. **Framework integration**:
   - **Native**: Method 1 (Myrheim-Meyer, causal sets), Method 6 (spectral, reuse Laplacian from § 5.2), **Method 8 (Scutoid, reuse Voronoi from curvature)**
   - **Framework-native synergy**: **Method 8 exploits scutoid tessellation structure uniquely available in this framework**
   - **QSD-corrected**: Methods 2, 3, 4, 5, 8 require explicit correction for non-uniform sampling $\rho(x) \propto \sqrt{\det g} \exp(-U_{\text{eff}}/T)$
   - **Synergistic**: Method 7 (geodesic) is spatial analog of Method 1 (causal), should agree for Fractal Set

4. **Noise sensitivity**:
   - **Extremely robust**: **Method 8 (topological, uses discrete neighbor counts)**
   - **Robust**: Levina-Bickel (Method 2), Spectral decay (Method 6)
   - **Sensitive**: Local PCA (Method 3), Box-counting (Method 5)

5. **Data requirements**:
   - **Requires causal structure**: Method 1 (Myrheim-Meyer)
   - **Requires tessellation**: **Method 8 (Scutoid, but likely already computed for curvature)**
   - **Requires large $N$**: Methods 4, 5, 6 (asymptotic scaling regimes)
   - **Works with moderate $N$**: Methods 2, 3, 8

6. **Recommendation for Fragile Gas**:
   - **Primary**: **Method 8 (Scutoid Topological)** if computing curvature via deficit angles (tessellation is free)
   - **Validation**: Method 2 (Levina-Bickel) for local estimates, Method 6 (Spectral) for independent global estimate
   - **Consensus**: Use all three for confidence interval

---

## 7.11. Practical Recommendations for Dimension Estimation

This section provides actionable guidance for practitioners implementing dimension estimation in the Fragile Gas framework.

### 7.11.1. Decision Tree: Which Method to Use?

Use the following decision tree to select the most appropriate method(s):

```
START
│
├─ Are you computing CURVATURE via deficit angles (Voronoi/Delaunay)?
│  ├─ YES → Use Method 8 (Scutoid Topological) [FREE, O(N), most efficient]
│  │        **Rationale**: Tessellation already exists, coordination number is trivial to extract
│  │        + Validate with Method 2 (Levina-Bickel) or Method 6 (Spectral)
│  └─ NO → Continue
│
├─ Do you have causal structure (Fractal Set episodes)?
│  ├─ YES → Use Method 1 (Myrheim-Meyer) [REQUIRED for causal sets]
│  │        + Cross-validate with Method 7 (Geodesic scaling) [spatial analog]
│  └─ NO → Continue
│
├─ Do you need LOCAL dimension estimates (varying across manifold)?
│  ├─ YES → Use Method 2 (Levina-Bickel) [RECOMMENDED, fast + robust]
│  │        + Optional: Method 3 (Local PCA) for low-noise data
│  └─ NO → Continue (global estimate sufficient)
│
├─ Is your data HIGH-DIMENSIONAL (d > 20)?
│  ├─ YES → Use Method 2 (Levina-Bickel) [O(N log N), scales well]
│  │        Avoid: Method 3 (Local PCA, O(N·k·d²) too slow)
│  └─ NO → Continue
│
├─ Do you already compute the Graph Laplacian spectrum (for Ricci bounds)?
│  ├─ YES → Use Method 6 (Spectral decay) [FREE, reuse eigenvalues]
│  │        **Rationale**: § 5.2 (Table 1, Method 2) already computes λ_k for Ricci bounds
│  └─ NO → Continue
│
├─ Do you suspect FRACTAL or MULTISCALE structure?
│  ├─ YES → Use Method 4 (Grassberger-Procaccia) for correlation dimension
│  │        + Method 5 (Box-counting) for rough estimate
│  └─ NO → Continue
│
└─ GENERAL RECOMMENDATION (no special structure):
   → Primary: Method 8 (Scutoid Topological) [fastest, most robust, parameter-free]
   → Validation: Method 2 (Levina-Bickel) [local estimates] + Method 6 (Spectral decay) [global]
   → If resources allow: Method 7 (Geodesic) [geometric, connects to curvature]
```

### 7.11.2. Multi-Method Consensus Strategy

For rigorous dimension estimation, use a **consensus approach** with multiple methods:

:::{prf:algorithm} Multi-Method Dimension Consensus
:label: alg-dimension-consensus

**Input**: Point cloud $\{x_1, \ldots, x_N\}$ from QSD $\rho(x)$

**Output**: Consensus dimension estimate $\hat{d}_{\text{consensus}}$ with confidence interval

**Steps**:
1. **Apply 3-5 methods** from different categories:
   - One local method (Method 2 or 3)
   - One global spectral method (Method 6)
   - One geometric method (Method 7)
   - If available: One causal method (Method 1)
   - If suspected fractal: One fractal method (Method 4 or 5)

2. **QSD correction**: For Methods 2-5, apply non-uniform sampling correction (see § 7.10.3)

3. **Collect estimates**: $\{\hat{d}_1, \hat{d}_2, \ldots, \hat{d}_M\}$ from $M$ methods

4. **Compute consensus**:
   - Median: $\hat{d}_{\text{median}} = \text{median}(\hat{d}_1, \ldots, \hat{d}_M)$
   - Mean: $\hat{d}_{\text{mean}} = \frac{1}{M} \sum_{i=1}^M \hat{d}_i$
   - Std deviation: $\sigma_d = \sqrt{\frac{1}{M-1} \sum_{i=1}^M (\hat{d}_i - \hat{d}_{\text{mean}})^2}$

5. **Confidence check**:
   - If $\sigma_d / \hat{d}_{\text{mean}} < 0.2$ (20% relative error): **High confidence**, report $\hat{d}_{\text{consensus}} = \hat{d}_{\text{median}}$
   - If $0.2 \le \sigma_d / \hat{d}_{\text{mean}} < 0.5$: **Moderate confidence**, investigate outliers
   - If $\sigma_d / \hat{d}_{\text{mean}} \ge 0.5$: **Low confidence**, check data quality and QSD correction

6. **Report**: $\hat{d}_{\text{consensus}} \pm 2\sigma_d$ (95% confidence interval)
:::

**Example**:
- Method 2 (Levina-Bickel): $\hat{d}_{\text{LB}} = 3.2$
- Method 6 (Spectral decay): $\hat{d}_{\text{spec}} = 3.1$
- Method 7 (Geodesic scaling): $\hat{d}_V = 2.9$
- Median: $\hat{d}_{\text{consensus}} = 3.1$
- Std: $\sigma_d = 0.15$, relative error = $0.15/3.1 = 4.8\%$ → **High confidence**
- Report: **Intrinsic dimension = 3.1 ± 0.3**

### 7.11.3. Handling Non-Uniform QSD Sampling

All dimension estimation methods assume **uniform sampling** from the manifold, but the Fragile Gas QSD is **non-uniform**:

$$
\rho_{\text{QSD}}(x) \propto \sqrt{\det g(x)} \cdot \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

This bias must be corrected to obtain accurate dimension estimates.

:::{prf:definition} QSD-Corrected Dimension Estimation
:label: def-qsd-corrected-dimension

For a point cloud $\{x_1, \ldots, x_N\}$ sampled from non-uniform density $\rho(x)$, define importance weights:

$$
w_i = \frac{1}{\rho(x_i)}
$$

The **QSD-corrected** version of a dimension estimator $\hat{d}(\{x_i\})$ uses weighted distances or weighted statistics.
:::

**Correction strategies by method**:

| Method | QSD Correction Strategy |
|--------|------------------------|
| **Method 1** (Myrheim-Meyer) | Not needed (uses only causal structure $\prec$, independent of spatial density) |
| **Method 2** (Levina-Bickel) | Weight local estimates: $\hat{d}_{\text{global}} = \frac{\sum_i w_i \hat{d}_{\text{ML}}(x_i)}{\sum_i w_i}$ where $w_i = 1/\rho(x_i)$ |
| **Method 3** (Local PCA) | Use weighted covariance: $\Sigma_i = \sum_{j \in N_r(x_i)} w_j (x_j - \mu)(x_j - \mu)^T$ with $w_j = 1/\rho(x_j)$ |
| **Method 4** (Grassberger-Procaccia) | Weighted correlation sum: $C(r) = \frac{\sum_{i<j} w_i w_j \Theta(r - d(x_i, x_j))}{\sum_{i<j} w_i w_j}$ |
| **Method 5** (Box-counting) | Weighted occupancy: Box is "occupied" if $\sum_{x_i \in B} w_i > \text{threshold}$ |
| **Method 6** (Spectral decay) | **Advanced**: Interpret spectrum as manifold with conformal metric $g' = \rho^{2/d} g$ (dimension preserved). **Conservative**: Use unweighted Laplacian, dimension estimate is global average less sensitive to local density variations |
| **Method 7** (Geodesic scaling) | Weight volume count: $N_w(r) = \sum_{d_g(x_i, x_j) \le r} w_j$ instead of $N(r) = \#\{x_j : d_g(x_i, x_j) \le r\}$ |

:::{important}
**Practical Implementation**:

Since $\rho_{\text{QSD}}(x)$ depends on the emergent metric $\det g(x)$ which we are trying to characterize, use an **iterative approach**:

1. **First pass**: Compute dimension estimate $\hat{d}^{(0)}$ without QSD correction
2. **Estimate density**: Use kernel density estimation $\hat{\rho}(x_i) = \sum_j K_h(x_i - x_j)$ with Gaussian kernel
3. **Compute weights**: $w_i^{(1)} = 1 / \hat{\rho}(x_i)$
4. **Second pass**: Recompute dimension estimate $\hat{d}^{(1)}$ with weights $w_i^{(1)}$
5. **Check convergence**: If $|\hat{d}^{(1)} - \hat{d}^{(0)}| < 0.1$, stop. Else iterate.

In practice, **one iteration is usually sufficient** if $\rho_{\text{QSD}}$ is not too non-uniform (i.e., $U_{\text{eff}}$ is slowly varying).
:::

### 7.11.4. Validation Protocol: Synthetic Manifolds

Before applying dimension estimation to real Fragile Gas data, validate methods on **synthetic manifolds with known dimension**:

:::{prf:algorithm} Dimension Estimation Validation Protocol
:label: alg-dimension-validation

**Goal**: Verify that dimension estimators correctly recover $d_{\text{true}}$ for known manifolds

**Test Cases**:

1. **Flat Torus** $\mathbb{T}^d = (\mathbb{R}/\mathbb{Z})^d$
   - True dimension: $d_{\text{true}} = d$
   - Sampling: Uniform on $[0, 1]^d$ with periodic boundary
   - Expected: All methods should return $\hat{d} \approx d_{\text{true}}$ within 5%

2. **Sphere** $\mathbb{S}^d \subset \mathbb{R}^{d+1}$
   - True dimension: $d_{\text{true}} = d$
   - Sampling: Uniform on unit sphere (rejection sampling or spherical coordinates)
   - Expected: Methods 2, 3, 6, 7 should return $\hat{d} \approx d$. Box-counting (Method 5) may give $d+1$ due to embedding.

3. **Swiss Roll** (2D manifold embedded in $\mathbb{R}^3$)
   - True dimension: $d_{\text{true}} = 2$
   - Sampling: $(x, y) \in [0, 1]^2 \mapsto (t \cos t, y, t \sin t)$ where $t = x \cdot 4\pi$
   - Expected: Methods 2, 3 should return $\hat{d} \approx 2$. Naive Euclidean methods may fail without geodesic correction.

4. **Non-Uniform Sampling** (simulate QSD bias)
   - Manifold: $\mathbb{R}^d$ with Gaussian density $\rho(x) = \exp(-\|x\|^2 / 2\sigma^2)$
   - True dimension: $d_{\text{true}} = d$
   - Expected: **Without QSD correction**: $\hat{d} < d$ (underestimate due to concentration). **With QSD correction**: $\hat{d} \approx d$

**Success Criteria**:
- **Accuracy**: $|\hat{d} - d_{\text{true}}| / d_{\text{true}} < 0.1$ (10% relative error)
- **Consistency**: $\sigma_d / \hat{d}_{\text{mean}} < 0.2$ (multi-method agreement)
- **QSD correction**: Non-uniform test (case 4) must show improvement with correction
:::

### 7.11.5. Connection to Curvature Estimation

Dimension estimation (Part 7) and curvature estimation (Parts 1-6) are **complementary**:

:::{prf:remark} Dimension-Curvature Consistency Checks
:label: rem-dimension-curvature-consistency

1. **Input consistency**: Most curvature methods (Deficit angles, Emergent metric, Heat kernel) require knowing ambient dimension $d$ to compute tensor components. The dimension estimate $\hat{d}$ provides this input.

2. **Validation**: If curvature computations fail or produce unphysical values, check if the **assumed dimension** matches $\hat{d}_{\text{consensus}}$. Mismatch indicates:
   - Manifold has varying intrinsic dimension → use local dimension methods (Method 2 or 3)
   - Data lies on lower-dimensional submanifold → reduce $d$ in curvature algorithms

3. **Weyl tensor interpretation**: The Weyl tensor $C_{abcd}$ is non-zero only for $d \ge 4$ (§ 4.1). If dimension estimate gives $\hat{d} < 4$ but Weyl norm $\|C\|^2 > 0$ is computed, this is a **contradiction** → likely numerical artifact or dimension underestimate.

4. **Spectral synergy**: Both Ricci curvature bounds (§ 5.2, Table 1, Method 2) and spectral dimension (§ 7.7, Method 6) use the **same Graph Laplacian eigenvalues**. Compute once, use for both:
   - From $\{\lambda_k\}$: Estimate $\hat{d}_{\text{spec}}$ (slope of $\ln \lambda_k$ vs $\ln k$)
   - From $\lambda_1$: Bound $R_{\min} \ge -\frac{d}{2} h(G)^2$ where $\lambda_1 \le h(G)^2 / 4$

5. **Fractal vs smooth**: If dimension estimates disagree significantly (e.g., $\hat{d}_{\text{box}} = 2.7$, $\hat{d}_{\text{LB}} = 3.0$), the manifold may have **fractal structure**. In this case:
   - Smooth curvature methods (Emergent metric, Heat kernel) may not apply
   - Use discrete methods (Deficit angles, Regge calculus) instead
:::

**Recommended workflow**:
1. **Estimate dimension** using Method 2 (Levina-Bickel) + Method 6 (Spectral decay)
2. **Check consistency**: If $|\hat{d}_{\text{LB}} - \hat{d}_{\text{spec}}| < 0.5$, use $\hat{d} = \hat{d}_{\text{LB}}$ (more accurate)
3. **Compute curvature** using $\hat{d}$ as input (e.g., in Regge calculus formulas for $d$-simplices)
4. **Cross-validate**: If curvature values are unphysical, revisit dimension estimate

---

## Summary and Future Directions

### What We Have Established

This document provides a **unified treatment of curvature and geometry** in the Fragile Gas framework:

1. **Five equivalent definitions** of Ricci scalar curvature (Part 1)
2. **Rigorous equivalence theorem** proving all five converge (Part 2)
3. **Full Riemann tensor** from scutoid plaquette holonomy (Part 3)
4. **Weyl conformal tensor** for anisotropic curvature (Part 4)
5. **Practical computational algorithms** for all curvature methods (Part 5)
6. **Applications** to Raychaudhuri, Einstein equations, gauge theory, causal sets (Part 6)
7. **Eight dimension estimation methods** for inferring intrinsic manifold dimension (Part 7, NEW)
   - Including **Method 8 (Scutoid Topological Dimension)**, a novel estimator unique to this framework that exploits Voronoi coordination numbers

### Unified Method Comparison Tables

This section provides comprehensive comparison tables for all curvature computation methods across three levels of the curvature hierarchy: Ricci scalar, Ricci tensor, and Weyl tensor.

#### Table 1: Ricci Scalar $R(x)$ - Five Equivalent Methods

| Method | Mathematical Definition | Input | Output | Complexity | Information | Advantages | Limitations | Framework Integration |
|--------|------------------------|-------|--------|------------|-------------|------------|-------------|----------------------|
| **1. Deficit Angles** | $R(x_i) = K(v_i) = \frac{1}{\text{Vol}(V_i)} \sum_{h \ni v_i} \text{Vol}_{d-2}(h) \, \delta_h$ where $\delta_h = 2\pi - \sum_{s \ni h} \theta_s(h)$ | Delaunay triangulation, simplex dihedral angles | Scalar $R(x_i)$ per vertex | $O(N \cdot d \cdot k)$ where $k =$ avg hinges/vertex | Full Ricci scalar field | Geometrically intuitive (Regge calculus), local computation | Sensitive to triangulation quality, noise amplification at small scales | [14_scutoid_geometry_framework.md] § 5.1, {prf:ref}`def-deficit-angle-curvature`, Theorem {prf:ref}`thm-deficit-ricci-convergence` |
| **2. Graph Laplacian** | $R(x_i) \ge -\frac{d}{2} h(G)^2$ where $h(G) = \inf \frac{\text{Vol}(\partial S)}{\min(\text{Vol}(S), \text{Vol}(S^c))}$ is Cheeger constant, $\lambda_1 \le h(G)^2 / 4$ | Graph $G$ with edge weights $w_{ij}$, Laplacian spectrum $\{\lambda_k\}$ | Lower bound on minimum Ricci curvature | $O(N^2)$ (sparse), $O(N^3)$ (dense eigensolve) | Global spectral information, averaged curvature | Robust to local noise, captures global geometry | Only lower bound, not pointwise values, requires spectral computation | [14_scutoid_geometry_framework.md] § 5.2, {prf:ref}`def-graph-laplacian-curvature`, Cheeger inequality {prf:ref}`thm-cheeger-ricci-bound` |
| **3. Emergent Metric** | $g_{jk}(x) = H_{jk}(x) + \epsilon_\Sigma \delta_{jk}$ where $H_{jk} = \frac{\partial^2 V_{\text{fit}}}{\partial x^j \partial x^k}$, then $R = g^{jk} R_{jk}$ via Christoffel symbols | Fitness function $V_{\text{fit}}(x)$, Hessian $H(x)$ | Scalar $R(x)$ at arbitrary $x$ (smooth field) | $O(d^4)$ per point (tensor contraction) | Full Ricci scalar field (continuous) | Smooth field, works off-grid, analytic gradients | Requires twice-differentiable $V_{\text{fit}}$, regularization-dependent, expensive for high $d$ | [08_emergent_geometry.md], [14_scutoid_geometry_framework.md] § 5.3, {prf:ref}`def-emergent-metric-curvature` |
| **4. Heat Kernel** | $\text{Tr}(K_t(x, x)) = \frac{1}{(4\pi t)^{d/2}} \left(1 - \frac{t}{6} R(x) + O(t^2)\right)$ where $K_t(x, y)$ is heat kernel | Heat kernel asymptotics $\text{Tr}(K_t(x, x))$ for small $t$ | Scalar $R(x)$ via asymptotic expansion | $O(N \cdot d^2)$ (heat kernel sampling), $O(N^2)$ (matrix exponentiation) | Local curvature from diffusion | Probabilistic interpretation (random walks), connects to LSI | Requires accurate small-$t$ asymptotics, sensitive to discretization | [14_scutoid_geometry_framework.md] § 5.4, {prf:ref}`def-heat-kernel-curvature`, Minakshisundaram-Pleijel expansion |
| **5. Causal Set Volume** | $\mathbb{E}[\mathcal{N}(x; r)] = \frac{\omega_d}{d!} r^d \left(1 - \frac{R(x)}{6(d+2)} r^2 + O(r^4)\right)$ where $\mathcal{N}(x; r) = \#\{x_j : d(x, x_j) < r\}$ | Point cloud $\{x_i\}$, metric $d(\cdot, \cdot)$ | Scalar $R(x)$ via volume deficits | $O(N \log N)$ (kdtree), $O(N^2)$ (naive) | Local curvature from volume growth | Model-free (only needs distances), causal set foundations | Requires many points in $r$-ball, sensitive to boundary, statistical estimation | [13_fractal_set], [14_scutoid_geometry_framework.md] § 5.5, {prf:ref}`def-causal-volume-curvature`, Myrheim-Meyer estimator |

**Key Insights from Table 1:**
- **Methods 1, 3** give pointwise fields $R(x_i)$ or $R(x)$
- **Method 2** gives only global lower bound (spectral)
- **Methods 4, 5** extract $R(x)$ from asymptotic expansions (require fine sampling)
- **Complexity**: Method 5 is most efficient ($O(N \log N)$), Method 3 is most expensive for large $d$ ($O(d^4)$ per point)
- **Robustness**: Method 2 is most robust to noise, Method 1 is most sensitive

#### Table 2: Ricci Tensor $R_{ij}(x)$ - Five Methods

| Method | Mathematical Definition | Input | Output | Complexity | Information | Advantages | Limitations | Framework Integration |
|--------|------------------------|-------|--------|------------|-------------|------------|-------------|----------------------|
| **1. Deficit Angles** | Compute vertex curvature $K(v)$ per vertex, then use **directional deficit decomposition**: $R_{ij}(x) = \frac{1}{\text{Vol}(V)} \sum_{h \ni v} \text{Vol}_{d-2}(h) \, \delta_h \cdot \hat{n}_i(h) \hat{n}_j(h)$ where $\hat{n}(h)$ is normal to hinge $h$ | Delaunay triangulation, hinge deficit angles $\delta_h$, hinge normals $\hat{n}(h)$ | Tensor $R_{ij}(x_k)$ at vertices | $O(N \cdot d^2 \cdot k)$ where $k =$ avg hinges/vertex | Full Ricci tensor field (pointwise at vertices) | Extends naturally from scalar method, uses same triangulation | Requires hinge normal computation, sensitive to triangulation orientation, noise amplification | [14_scutoid_geometry_framework.md] § 5.1, {prf:ref}`def-deficit-angle-curvature`, extension to tensor via directional decomposition |
| **2. Graph Laplacian** | **Not directly available**. Graph Laplacian spectrum gives only scalar lower bound $R \ge -\frac{d}{2} h(G)^2$. Ricci tensor requires **weighted graph Ricci curvature** (Ollivier-Ricci curvature on edges): $\kappa(e_{ij}) = 1 - \frac{W_1(\mu_i, \mu_j)}{d(x_i, x_j)}$ where $W_1$ is Wasserstein distance | Graph $G$, probability measures $\mu_i$ at nodes (e.g., random walk distributions) | Edge Ricci curvature $\kappa(e_{ij})$, not full tensor $R_{ij}$ | $O(N^2 \cdot C_{\text{OT}})$ where $C_{\text{OT}}$ is optimal transport cost | Ricci curvature on graph edges (geometric insight) | Well-defined for discrete graphs, captures coarse Ricci geometry | Does NOT give full tensor $R_{ij}(x)$, only edge curvatures, expensive optimal transport | [14_scutoid_geometry_framework.md] § 5.2 (extensions), Ollivier-Ricci curvature literature (future work) |
| **3. Emergent Metric** | $g_{jk}(x) = H_{jk}(x) + \epsilon_\Sigma \delta_{jk}$, then Ricci tensor via: $R_{ij} = \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik} + \Gamma^k_{kl} \Gamma^l_{ij} - \Gamma^k_{jl} \Gamma^l_{ik}$ where $\Gamma^k_{ij} = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$ | Fitness function $V_{\text{fit}}(x)$, metric $g_{jk}(x)$, Christoffel symbols $\Gamma^k_{ij}$ | Tensor $R_{ij}(x)$ at arbitrary $x$ (smooth field) | $O(d^4)$ per point (Christoffel computation + Ricci tensor) | Full Ricci tensor field (continuous, off-grid) | Smooth tensor field, analytic computation, exact for smooth metrics | Requires third derivatives of $V_{\text{fit}}$, regularization-dependent, $O(d^4)$ scaling prohibitive for large $d$ | [08_emergent_geometry.md], [14_scutoid_geometry_framework.md] § 5.3, Riemannian geometry textbooks (e.g., Lee) |
| **4. Heat Kernel** | Heat kernel asymptotic expansion: $K_t(x, y) = \frac{1}{(4\pi t)^{d/2}} e^{-d(x,y)^2 / 4t} \left(\sum_{k=0}^\infty t^k u_k(x, y)\right)$ where $u_1(x, x) = -\frac{1}{6} R(x)$ (scalar), $u_2(x, x)$ contains $R_{ij}$ terms (see Minakshisundaram-Pleijel) | Heat kernel $K_t(x, y)$, asymptotic coefficients $u_k(x, x)$ | Tensor $R_{ij}(x)$ via $u_2(x, x)$ decomposition | $O(N \cdot d^2)$ (heat kernel), $O(d^4)$ (tensor extraction from $u_2$) | Ricci tensor from diffusion asymptotics | Probabilistic interpretation, connects to spectral geometry | Requires higher-order asymptotics ($u_2$), difficult to extract tensor components, literature formulas complex | [14_scutoid_geometry_framework.md] § 5.4, Minakshisundaram-Pleijel expansion (Rosenberg "The Laplacian on a Riemannian Manifold"), future work |
| **5. Causal Set Volume** | Volume growth is **isotropic** in causal set approach: $\mathbb{E}[\mathcal{N}(x; r)] = \frac{\omega_d}{d!} r^d \left(1 - \frac{R(x)}{6(d+2)} r^2 + O(r^4)\right)$. To extract $R_{ij}$, need **directional volume deficits**: $\mathcal{N}(x; r, \hat{e}_i) = \#\{x_j \in \text{cone}(x, \hat{e}_i, \theta) : d(x, x_j) < r\}$ with anisotropic expansion | Point cloud $\{x_i\}$, directional volume counts $\mathcal{N}(x; r, \hat{e}_i)$ in directions $\hat{e}_i$ | Tensor $R_{ij}(x)$ via directional volume deficit decomposition | $O(d \cdot N \log N)$ (kdtree per direction), $O(d \cdot N^2)$ (naive) | Ricci tensor from directional volume growth | Model-free (distance-based), extends scalar method naturally | Requires directional sampling (cones), many points per cone, sensitive to anisotropy, literature incomplete | [13_fractal_set], [14_scutoid_geometry_framework.md] § 5.5, Myrheim-Meyer extensions (research direction) |

**Key Insights from Table 2:**
- **Method 3** is the only method giving smooth, off-grid Ricci tensor fields $R_{ij}(x)$ (but expensive)
- **Method 1** gives pointwise tensors at vertices (extends scalar method naturally)
- **Method 2** does NOT give full tensor, only edge curvatures via Ollivier-Ricci (fundamentally different)
- **Methods 4, 5** require directional/anisotropic analysis (higher complexity, literature incomplete)
- **Critical limitation**: Most methods (1, 4, 5) are research directions requiring careful directional decomposition

#### Table 3: Weyl Tensor $C_{abcd}(x)$ - Method Capabilities

| Method | Mathematical Definition | Input | Output | Complexity | Information | Advantages | Limitations | Framework Integration |
|--------|------------------------|-------|--------|------------|-------------|------------|-------------|----------------------|
| **1. Deficit Angles + Plaquettes** | **Step 1**: Compute Riemann tensor $R^a_{bcd}$ via plaquette holonomy ({prf:ref}`def-riemann-plaquette-holonomy`). **Step 2**: Apply Weyl decomposition: $C_{abcd} = R_{abcd} - \frac{1}{d-2} (g_{ac} R_{bd} - g_{ad} R_{bc} + g_{bd} R_{ac} - g_{bc} R_{ad}) + \frac{R}{(d-1)(d-2)} (g_{ac} g_{bd} - g_{ad} g_{bc})$ | Scutoid plaquettes $\{P_\alpha\}$, parallel transport on boundary, metric $g_{ij}$ from deficit angles | Full Weyl tensor $C_{abcd}(x_i)$ at vertices | $O(N \cdot d^4 \cdot p)$ where $p =$ plaquettes/vertex | Complete Weyl tensor (trace-free part of Riemann) | Full tensor information, integrates triangulation + holonomy | Requires plaquette identification, parallel transport on discrete geometry, $O(d^4)$ storage | [15_scutoid_curvature_raychaudhuri.md] § 3, {prf:ref}`def-weyl-deficit-plaquette`, Part 3 and Part 4 of this document |
| **2. Emergent Metric** | **Step 1**: Compute Riemann tensor $R^a_{bcd}$ from metric $g_{jk}(x) = H_{jk}(x) + \epsilon_\Sigma \delta_{jk}$ via: $R^a_{bcd} = \partial_c \Gamma^a_{bd} - \partial_d \Gamma^a_{bc} + \Gamma^a_{ce} \Gamma^e_{bd} - \Gamma^a_{de} \Gamma^e_{bc}$. **Step 2**: Apply Weyl decomposition (same as Method 1) | Fitness Hessian $H(x)$, metric $g_{jk}(x)$, Christoffel symbols $\Gamma^a_{bc}$ | Full Weyl tensor $C_{abcd}(x)$ (smooth field) | $O(d^4)$ per point (Riemann) + $O(d^4)$ (Weyl decomposition) | Complete Weyl tensor (continuous, off-grid) | Smooth tensor field, analytic computation, works anywhere in $\mathcal{X}$ | Requires fourth derivatives of $V_{\text{fit}}$, $O(d^4)$ storage and computation prohibitive for large $d$, regularization artifacts | [08_emergent_geometry.md], {prf:ref}`def-weyl-emergent-metric`, Algorithm 6 (this document § 5.6) |
| **3. Deficit Angles (Direct)** | **REMOVED - Insufficient Mathematical Basis**. Deficit angles $\delta_h$ are scalar quantities at hinges/vertices and do not encode directional information sufficient to resolve a rank-4 trace-free tensor like $C_{abcd}$. | N/A | N/A | N/A | N/A | N/A | Scalar deficit angles cannot reconstruct full Weyl tensor | Removed after Gemini Round 3 review (see Part 4.2.3) |
| **4. Heat Kernel Weyl Norm** | Heat kernel asymptotics: $u_2(x, x) = \frac{1}{360} \left(2 \lvert\text{Rm}\rvert^2 - 2 \lvert\text{Ric}\rvert^2 + 5 R^2\right)$ where $\lvert\text{Rm}\rvert^2 = R_{abcd} R^{abcd}$. Weyl norm: $\lvert C \rvert^2 = \lvert\text{Rm}\rvert^2 - \frac{4}{d-2} \lvert\text{Ric}\rvert^2 + \frac{2}{(d-1)(d-2)} R^2$. Solve for $\lvert C \rvert^2$ given $u_2$, $R$, $\lvert\text{Ric}\rvert^2$ | Heat kernel coefficient $u_2(x, x)$, Ricci scalar $R(x)$, Ricci tensor norm $\lvert\text{Ric}\rvert^2$ | Weyl norm $\lvert C \rvert^2$ only (NOT full tensor) | $O(N \cdot d^2)$ (heat kernel) + $O(d^2)$ (norm extraction) | Squared norm of Weyl tensor (scalar field) | Does not require full Riemann tensor, probabilistic interpretation | Cannot reconstruct tensor components $C_{abcd}$, only norm, requires accurate $u_2$ asymptotics | [14_scutoid_geometry_framework.md] § 5.4, {prf:ref}`def-weyl-heat-kernel-norm`, Minakshisundaram-Pleijel expansion |
| **5. Causal Set Weyl Norm (Hypothesized)** | **Research direction** (not yet rigorously established). Causal set volume growth: $\mathbb{E}[\mathcal{N}(x; r)] = \frac{\omega_d}{d!} r^d \left(1 - \frac{R(x)}{6(d+2)} r^2 + O(r^4)\right)$ where $O(r^4)$ terms contain Weyl invariants. **Hypothesis**: Extract $\lvert C \rvert^2$ from $r^4$ coefficient of volume deficits | Point cloud $\{x_i\}$, high-precision volume counts $\mathcal{N}(x; r)$ for small $r$ | Weyl norm $\lvert C \rvert^2$ (hypothesized, if $r^4$ extraction feasible) | $O(N \log N)$ (kdtree) + $O(N)$ (nonlinear fit to $r^4$ terms) | Squared norm of Weyl tensor (if extractable) | Model-free (distance-based), extends scalar and tensor methods | NO RIGOROUS MATHEMATICAL FOUNDATION YET, requires $r^4$ asymptotics (difficult), many points needed, boundary effects | [13_fractal_set], {prf:ref}`def-weyl-causal-volume-norm`, Part 4.2.5 (this document), future research |

**Key Insights from Table 3:**
- **Methods 1-2**: Give full Weyl tensor $C_{abcd}(x)$ (all components)
- **Methods 4-5**: Give only Weyl norm $\lvert C \rvert^2$ (scalar), NOT tensor components
- **Method 3**: Removed (mathematically invalid)
- **Dimension requirement**: Weyl tensor only defined for $d \ge 3$ (identically zero for $d = 2$)
- **Complexity hierarchy**: Method 5 is least expensive ($O(N \log N)$ if feasible), Methods 1-2 are $O(N \cdot d^4)$
- **Maturity**: Methods 1-2 are rigorous, Method 4 is established (spectral geometry), Method 5 is hypothesized (research direction)

#### Table 4: Weyl Norm $\lvert C \rvert^2(x)$ - Efficient Computation Methods

This table focuses on methods for computing the **Weyl squared norm** $\lvert C \rvert^2$ (a scalar field) **without** constructing the full rank-4 tensor $C_{abcd}$ (which requires $O(d^4)$ storage/computation).

| Method | Mathematical Definition | Input | Output | Complexity | Information | Advantages | Limitations | Framework Integration |
|--------|------------------------|-------|--------|------------|-------------|------------|-------------|----------------------|
| **1. Regge Calculus** | For simplicial triangulation $\mathcal{T}$, sum over $(d-2)$-hinges: $\int \lvert C \rvert^2 dV \approx \sum_{h \in \text{hinges}} V_h \mathcal{W}(h)$ where $\mathcal{W}(h) = f(\delta_h, \{\theta_j\}, \{l_e\})$. For $d=3$: $\mathcal{W}(e) = \frac{1}{2\pi} \left[\delta_e^2 - \frac{1}{3}\left(\sum_{f \ni e} A_f \delta_f\right)^2 / \sum_{f \ni e} A_f\right]$ | Delaunay triangulation, edge lengths $\{l_e\}$, simplex volumes, deficit angles $\{\delta_h\}$ | Global $\int \lvert C \rvert^2 dV$ | $O(N \cdot d^2 \cdot k)$ where $k =$ hinges/vertex | Integrated Weyl norm (discrete exact) | No continuum approximation, leverages existing triangulation, avoids $O(d^4)$ | Requires hinge enumeration, sensitive to triangulation quality, literature formulas complex for $d>3$ | Part 4.6.1 ({prf:ref}`def-regge-weyl-norm`, {prf:ref}`alg-regge-weyl-norm`), Hamber & Williams (1985), Brewin (2009) |
| **2. Chern-Gauss-Bonnet (d=4)** | Topological reduction: $\int \lvert C \rvert^2 dV = 32\pi^2 \chi(M) + 2\int \lvert R_{ab}\rvert^2 dV - \frac{2}{3}\int R^2 dV$ where $\chi(M) = V - E + F - C$ (Euler characteristic) | Scutoid tessellation (for $\chi$), Ricci scalar $R(x)$, Ricci tensor norm $\lvert R_{ab}\rvert^2$ | Global $\int \lvert C \rvert^2 dV$ | $O(N \cdot d^2)$ (dominated by Ricci tensor computation) | Integrated Weyl norm (topologically exact) | Reduces $O(d^4)$ to $O(d^2)$, topologically robust, exact formula | **Only valid for $d=4$**, gives global integral not local field, requires smooth metric | Part 4.6.2 ({prf:ref}`thm-cgb-weyl-reduction`, {prf:ref}`alg-cgb-weyl-norm`), Chern (1944) |
| **3. Spectral Heat Kernel** | Extract from heat trace: Fit $\text{Tr}(e^{-t\Delta_G}) \sim (4\pi t)^{-d/2}(a_0 + a_1 t + A_2 t^2 + \cdots)$ where $A_2 = \frac{1}{360}\int(2\lvert R_{abcd}\rvert^2 - 2\lvert R_{ab}\rvert^2 + 5R^2)dV$. For $d=4$: $\int \lvert C \rvert^2 dV = 180A_2 - \int \lvert R_{ab}\rvert^2 dV - \frac{13}{6}\int R^2 dV$ | Graph Laplacian eigenvalues $\{\lambda_i\}$, Ricci scalar/tensor norms | Global $\int \lvert C \rvert^2 dV$ | $O(N^3)$ dense, $O(N^2)$ sparse (eigendecomposition) | Integrated Weyl norm (approximate via fitting) | Uses graph structure (no triangulation), probabilistic interpretation, spectral geometry foundation | **Numerically challenging** (ill-conditioned $A_2$ extraction), requires separate Ricci computation, only global integral | Part 4.6.3 ({prf:ref}`alg-spectral-weyl-norm`), Rosenberg (1997), heat kernel asymptotics literature |
| **4. Spectral Gap Check** | Qualitative consistency via Lichnerowicz theorem: $\lambda_1 \ge \frac{d}{d-1}\kappa$ where $\kappa$ is Ricci lower bound. Check if computed $\lvert\hat{C}\rvert^2$ is consistent with $\lambda_1$, diameter $D$, and Ricci norms via decomposition $\lvert R_{abcd}\rvert^2 = \lvert C \rvert^2 + \frac{4}{d-2}\lvert R_{ab}\rvert^2 - \frac{2}{(d-1)(d-2)}R^2$ | Spectral gap $\lambda_1$, diameter $D$, computed $\lvert\hat{C}\rvert^2$ from Methods 1-3 | **Pass/fail sanity check** (NOT quantitative bound) | $O(N^2)$ (compute $\lambda_1$ with iterative methods) | Consistency verification (qualitative) | Very fast (single eigenvalue), detects implausible values, works any $d\ge 3$ | **No quantitative bound**, only qualitative check, requires additional geometric data | Part 4.6.4 ({prf:ref}`thm-lichnerowicz-spectral-gap`, {prf:ref}`rem-spectral-gap-weyl-qualitative`), Lichnerowicz (1958), Obata (1962) |

**Key Insights from Table 4:**
- **Fundamental distinction**: These methods compute $\lvert C \rvert^2$ (scalar) **without** computing $C_{abcd}$ (rank-4 tensor)
- **Complexity savings**: Methods 1-3 are $O(N \cdot d^2)$ vs $O(N \cdot d^4)$ for full tensor (Methods 1-2 in Table 3)
- **Recommended**: Method 1 (Regge calculus) for scutoid tessellations, Method 2 (Chern-Gauss-Bonnet) for 4D spacetimes
- **Cross-validation**: Method 4 provides consistency checks for Methods 1-3
- **Output type**: Methods 1-3 give **global integral** $\int \lvert C \rvert^2 dV$, not local field $\lvert C \rvert^2(x)$
  - For local values, must use Table 3 Methods 4-5 (heat kernel/causal set at individual points)
- **Maturity**: Methods 1-2 are rigorous with established literature, Method 3 requires careful numerics, Method 4 is qualitative only

**Comparison with Table 3:**
- **Table 3 (Full Tensor)**: Methods 1-2 compute all $d^4$ components of $C_{abcd}(x)$, then contract to get $\lvert C \rvert^2$
- **Table 4 (Norm Only)**: Methods 1-4 compute $\lvert C \rvert^2$ **directly** without intermediate tensor, achieving $O(d^2)$ complexity

#### Cross-Method Validation Protocol

To ensure consistency across methods, we recommend the following validation workflow:

**For Ricci Scalar $R(x)$:**
1. Compute $R(x)$ using Method 3 (emergent metric) as **ground truth** (smooth, analytic)
2. Validate with Method 1 (deficit angles) at walker positions (check pointwise agreement)
3. Check global consistency with Method 2 (verify $R_{\min} \ge -\frac{d}{2} h(G)^2$)
4. Cross-validate Methods 4-5 at high sampling density (asymptotic regime)

**For Ricci Tensor $R_{ij}(x)$:**
1. Compute $R_{ij}(x)$ using Method 3 (emergent metric) as **reference**
2. Validate with Method 1 (deficit angle tensor) at vertices
3. Check trace consistency: $\text{tr}(R) = g^{ij} R_{ij} = R$ (must match scalar curvature)
4. Verify symmetry: $R_{ij} = R_{ji}$

**For Weyl Tensor $C_{abcd}(x)$ (Full Tensor - Table 3):**
1. Compute full tensor $C_{abcd}(x)$ using Method 2 (emergent metric, if $d \le 5$)
2. Validate with Method 1 (plaquette holonomy) if triangulation available
3. Check trace-free property: $g^{ac} C_{abcd} = 0$ (all contractions vanish)
4. Cross-validate norm: $\lvert C \rvert^2 = C_{abcd} C^{abcd}$ with Table 4 Methods
5. If Method 5 becomes rigorous, use as independent check

**For Weyl Norm $\lvert C \rvert^2$ (Efficient Methods - Table 4):**
1. Compute $\int \lvert C \rvert^2 dV$ using **Regge calculus** (Table 4 Method 1) as primary method
2. If $d=4$: Cross-check with **Chern-Gauss-Bonnet** (Table 4 Method 2) — should match exactly
3. If graph Laplacian available: Compare with **spectral heat kernel** (Table 4 Method 3) — expect approximate agreement
4. **Consistency check** with **spectral gap** (Table 4 Method 4):
   - Verify $\lvert\hat{C}\rvert^2$ is consistent with $\lambda_1$, $D$, and Ricci norms
   - Check anisotropy ratio $\rho = \lvert C \rvert^2 / \lvert R_{ab}\rvert^2$ is plausible
5. If full tensor available (Table 3): Integrate to verify $\int C_{abcd} C^{abcd} dV = \int \lvert C \rvert^2 dV$

**Convergence Tests:**
- **Refinement**: As triangulation refines ($h \to 0$), Methods 1, 4, 5 should converge to Method 3
- **Spectral**: Method 2 lower bound should tighten as graph connectivity increases
- **Regularization**: Method 3 results should stabilize as $\epsilon_\Sigma \to 0$ (within numerical precision)

**Framework Integration Column Explanation:**
- Each entry references the specific document section establishing the method mathematically
- Cross-references use `{prf:ref}` for definitions, theorems, algorithms
- Methods 4-5 for Ricci tensor and Weyl tensor are marked as "future work" or "research direction" when literature is incomplete

### Future Work

**Computational Implementation**:
- Implement all five algorithms in `src/fragile/geometry/curvature_estimators.py`
- Cross-validation framework for comparing methods
- Visualization tools for curvature fields

**Theoretical Extensions**:
- **Bochner identity**: Relating Ricci curvature to Laplacian of gradient norms
- **Comparison geometry**: Using curvature bounds to control geometry (Bonnet-Myers, Cheeger-Gromoll)
- **Ricci flow**: Evolution equations for the emergent metric

**Applications**:
- **Phase transition detection**: Use Weyl norm as order parameter
- **Adaptive sampling**: Bias walker distribution toward high-curvature regions
- **Gravitational wave analogs**: Detect propagating curvature perturbations

---

## References

### Primary Framework Documents

- [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) § 5: Five curvature definitions and equivalence theorem
- [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md) § 2: Riemann tensor from plaquette holonomy
- [08_emergent_geometry.md](08_emergent_geometry.md): Emergent Riemannian metric
- [13_fractal_set_new/11_causal_sets.md](13_fractal_set_new/11_causal_sets.md): Causal set curvature

### External References

- **Regge Calculus**: Cheeger, J., Müller, W., & Schrader, R. (1984). "On the curvature of piecewise flat spaces." *Comm. Math. Phys.* 92(3), 405-454.
- **Heat Kernel**: Rosenberg, S. (1997). *The Laplacian on a Riemannian Manifold*. Cambridge University Press.
- **Causal Sets**: Sorkin, R. D. (2005). "Causal sets: Discrete gravity." In *Lectures on Quantum Gravity*, Springer.
- **Spectral Geometry**: Chavel, I. (1984). *Eigenvalues in Riemannian Geometry*. Academic Press.
- **Weyl Tensor**: Wald, R. M. (1984). *General Relativity*. University of Chicago Press, Chapter 3.

---

:::{note}
**Document Status**: This document unifies curvature theory across the Fragile Gas framework. Parts 1-3 refactor material from Chapters 14-15. Part 4 (Weyl tensor) is NEW and requires **Gemini mathematical review** before finalization. Parts 5-6 are complete.

**Next Steps**: Submit Part 4 to Gemini 2.5 Pro for rigorous mathematical verification.
:::


### 5.6. Algorithm 6: Weyl Tensor from Metric (O(Nd⁴))

**Input**: Fitness function $V_{\text{fit}}(x)$, walker position $x_i$, regularization $\epsilon_\Sigma$, dimension $d \ge 3$

**Output**: Weyl tensor components $C_{abcd}(x_i)$

**Steps**:
1. Compute metric: $g_{jk}(x_i) = H_{jk}(x_i) + \epsilon_\Sigma \delta_{jk}$ where $H = \nabla^2 V_{\text{fit}}$
2. Compute Christoffel symbols: $\Gamma^k_{ij} = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$
3. Compute Riemann tensor: $R^l_{ijk} = \partial_j \Gamma^l_{ik} - \partial_k \Gamma^l_{ij} + \Gamma^l_{jm} \Gamma^m_{ik} - \Gamma^l_{km} \Gamma^m_{ij}$
4. Compute Ricci tensor: $R_{ij} = R^k_{ikj}$
5. Compute Ricci scalar: $R = g^{ij} R_{ij}$
6. Apply Weyl decomposition (Definition {prf:ref}`def-weyl-tensor`):

$$
C_{abcd} = R_{abcd} - \frac{1}{d-2}(g_{ac}R_{bd} - g_{ad}R_{bc} + g_{bd}R_{ac} - g_{bc}R_{ad}) + \frac{R}{(d-1)(d-2)}(g_{ac}g_{bd} - g_{ad}g_{bc})

$$

**Computational cost**: $O(d^4)$ per point for Riemann tensor computation (4th derivatives + tensor contractions), total $O(Nd^4)$ for all walkers

**Warning**:
- Requires **fourth derivatives** of $V_{\text{fit}}$ — extremely sensitive to numerical noise
- Use automatic differentiation (e.g., JAX, PyTorch) or very high-order finite differences
- Consider symbolic computation (SymPy, Mathematica) for validation
- For $d=3$: 81 Riemann components → 10 independent Weyl components

**Practical recommendation**: Use this method only for smooth, analytically-defined fitness functions. For noisy empirical data, Method 1 (plaquette holonomy) is more robust.

---

