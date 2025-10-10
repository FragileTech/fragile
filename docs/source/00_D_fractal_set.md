# Mathematical Reference: Fractal Set Theory

**Purpose**: Comprehensive, searchable reference of all mathematical definitions, theorems, lemmas, propositions, and axioms related to Fractal Set Theory, Causal Set Quantum Gravity, and Lattice QFT formulation of the Fragile Gas framework.

**Usage**: Use Ctrl+F / Cmd+F with the **Tags** field to quickly locate relevant mathematical objects. All labels follow MyST markdown conventions and can be referenced using `{prf:ref}label-name` syntax.

**Organization**: Content is organized into five major sections covering the complete mathematical framework from discrete foundations to continuum limits and quantum field theory.

---

## Table of Contents

- [Fractal Set Foundations](#fractal-set-foundations)
  - [Episodes and Causal Spacetime Tree](#episodes-and-causal-spacetime-tree)
  - [Information Graph](#information-graph)
  - [Fractal Set Composite Structure](#fractal-set-composite-structure)
  - [Discrete Differential Geometry](#discrete-differential-geometry)
  - [Propagators and Dynamics](#propagators-and-dynamics)

- [Continuum Limit Theory](#continuum-limit-theory)
  - [Discrete Symmetries](#discrete-symmetries)
  - [Gauge Connection](#gauge-connection)
  - [Graph Laplacian Convergence](#graph-laplacian-convergence)
  - [Episode Measure Convergence](#episode-measure-convergence)
  - [Holonomy Convergence](#holonomy-convergence)

- [Causal Set Quantum Gravity](#causal-set-quantum-gravity)
  - [Causal Set Axioms](#causal-set-axioms)
  - [Sprinkling and Dimension](#sprinkling-and-dimension)
  - [Lorentzian Structure](#lorentzian-structure)
  - [Discrete Differential Operators](#discrete-differential-operators)
  - [Quantum Gravity Formulation](#quantum-gravity-formulation)

- [Fermionic Structure and Gauge Theory](#fermionic-structure-and-gauge-theory)
  - [Antisymmetric Cloning Kernel](#antisymmetric-cloning-kernel)
  - [Exclusion Principle](#exclusion-principle)
  - [Wilson Loops](#wilson-loops)
  - [Geometric Area](#geometric-area)
  - [Gauge Field Dynamics](#gauge-field-dynamics)

- [Lattice QFT on Causal Sets](#lattice-qft-on-causal-sets)
  - [Lattice Gauge Theory](#lattice-gauge-theory)
  - [U(1) Gauge Fields](#u1-gauge-fields)
  - [SU(N) Gauge Fields](#sun-gauge-fields)
  - [Wilson Action](#wilson-action)
  - [QCD on Fractal Sets](#qcd-on-fractal-sets)
  - [Computational Algorithms](#computational-algorithms)

---

## Fractal Set Foundations

### Episodes and Causal Spacetime Tree

#### Episode Definition

**Type:** Definition
**Label:** `def-episode`
**Source:** [13_A_fractal_set.md § 1.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `episode`, `walker-trajectory`, `discrete-spacetime`, `fundamental-structure`

**Statement:**

An **episode** $e$ is a finite sequence of walker states:

$$
e = \{(x_0, v_0, s_0, t_0), (x_1, v_1, s_1, t_1), \ldots, (x_T, v_T, s_T, t_T)\}
$$

where:
- $x_i \in \mathcal{X}$ (position)
- $v_i \in \mathbb{R}^d$ (velocity)
- $s_i \in \{0,1\}$ (alive/dead status)
- $t_i \in \mathbb{N}$ (discrete time)
- $T$ is the episode length (time until absorption)

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`def-fractal-set`

---

#### Causal Spacetime Tree (CST)

**Type:** Definition
**Label:** `def-cst`
**Source:** [13_A_fractal_set.md § 1.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `causal-structure`, `directed-graph`, `genealogy`, `time-ordering`

**Statement:**

The **Causal Spacetime Tree** (CST) is a directed acyclic graph $\mathcal{T} = (\mathcal{E}, E_{\text{CST}})$ where:
- $\mathcal{E}$ is the set of all episodes
- $E_{\text{CST}} \subseteq \mathcal{E} \times \mathcal{E}$ are directed edges

An edge $e_1 \to e_2 \in E_{\text{CST}}$ exists iff $e_2$ is a direct descendant of $e_1$ through cloning.

**Properties:**
1. **Tree Structure**: Every episode (except root) has exactly one parent
2. **Time Ordering**: If $e_1 \to e_2$, then $t_{\text{birth}}(e_2) > t_{\text{death}}(e_1)$
3. **Causal Past**: For episode $e$, $J^-(e) = \{e' : e' \text{ is an ancestor of } e\}$

**Related Results:** {prf:ref}`def-episode`, {prf:ref}`def-ig`, {prf:ref}`thm-cst-lorentzian`

---

#### CST Metric Structure

**Type:** Proposition
**Label:** `prop-cst-metric`
**Source:** [13_A_fractal_set.md § 1.3](13_fractal_set/13_A_fractal_set.md)
**Tags:** `graph-metric`, `causal-distance`, `tree-depth`

**Statement:**

The CST admits a natural metric $d_{\text{CST}}: \mathcal{E} \times \mathcal{E} \to \mathbb{N} \cup \{\infty\}$:

$$
d_{\text{CST}}(e_1, e_2) = \begin{cases}
\text{min path length from LCA}(e_1, e_2) \text{ to } e_1 \text{ and } e_2 & \text{if connected} \\
\infty & \text{otherwise}
\end{cases}
$$

where LCA is the lowest common ancestor.

**Properties:**
1. $d_{\text{CST}}(e, e) = 0$
2. $d_{\text{CST}}(e_1, e_2) = d_{\text{CST}}(e_2, e_1)$
3. Triangle inequality holds on connected components

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`prop-cst-depth`

---

#### CST Depth and Branching

**Type:** Proposition
**Label:** `prop-cst-depth`
**Source:** [13_A_fractal_set.md § 1.4](13_fractal_set/13_A_fractal_set.md)
**Tags:** `tree-depth`, `branching-factor`, `genealogy-statistics`

**Statement:**

For a CST with $N$ walkers and $K$ cloning events:
1. **Depth**: Maximum depth $D \leq K$
2. **Branching**: Average branching factor $\bar{b} = (N_{\text{total}} - 1) / K$ where $N_{\text{total}}$ is total episodes
3. **Balance**: If cloning is spatially uniform, depth $D = O(\log N_{\text{total}})$

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`thm-cst-fractal-dimension`

---

### Information Graph

#### Information Graph (IG)

**Type:** Definition
**Label:** `def-ig`
**Source:** [13_A_fractal_set.md § 2.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `spacelike-correlation`, `undirected-graph`, `cloning-interaction`, `measurement`

**Statement:**

The **Information Graph** (IG) is an undirected graph $\mathcal{G} = (\mathcal{E}, E_{\text{IG}})$ where:
- $\mathcal{E}$ is the set of episodes
- $E_{\text{IG}} \subseteq \mathcal{E} \times \mathcal{E}$ (symmetric)

An edge $(e_i, e_j) \in E_{\text{IG}}$ exists iff episodes $e_i$ and $e_j$ interacted through a cloning event at some time $t$ (i.e., they were in the same local neighborhood $\mathcal{N}_\rho(x_i(t))$).

**Properties:**
1. **Symmetry**: $(e_i, e_j) \in E_{\text{IG}} \iff (e_j, e_i) \in E_{\text{IG}}$
2. **Spacelike**: Edges connect episodes at equal time slices
3. **Density**: Edge density scales with cloning rate and localization scale $\rho$

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`def-fractal-set`, {prf:ref}`def-ig-metric`

---

#### IG Metric Structure

**Type:** Definition
**Label:** `def-ig-metric`
**Source:** [13_A_fractal_set.md § 2.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `graph-metric`, `spacelike-distance`, `geodesic`

**Statement:**

The IG admits a graph metric $d_{\text{IG}}: \mathcal{E} \times \mathcal{E} \to \mathbb{N} \cup \{\infty\}$:

$$
d_{\text{IG}}(e_i, e_j) = \text{length of shortest path in } E_{\text{IG}} \text{ from } e_i \text{ to } e_j
$$

**Properties:**
1. Positive definite: $d_{\text{IG}}(e_i, e_j) \geq 0$ with equality iff $e_i = e_j$
2. Symmetric by construction
3. Triangle inequality holds
4. $d_{\text{IG}} = \infty$ if episodes are in disconnected components

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`prop-ig-connectedness`

---

#### IG Connectedness

**Type:** Proposition
**Label:** `prop-ig-connectedness`
**Source:** [13_A_fractal_set.md § 2.3](13_fractal_set/13_A_fractal_set.md)
**Tags:** `graph-connectivity`, `cloning-rate`, `percolation`

**Statement:**

For localization scale $\rho > 0$ and cloning rate $\lambda_c > 0$:

**IG Connectedness Theorem**: There exists a critical density $\rho_c(\lambda_c)$ such that:
- If $\rho > \rho_c$: IG is connected with high probability
- If $\rho < \rho_c$: IG has multiple disconnected components

The critical scale satisfies $\rho_c \sim (\lambda_c \cdot N)^{-1/d}$ in $d$ dimensions.

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`thm-ig-percolation`

---

#### IG Degree Distribution

**Type:** Proposition
**Label:** `prop-ig-degree`
**Source:** [13_A_fractal_set.md § 2.4](13_fractal_set/13_A_fractal_set.md)
**Tags:** `degree-distribution`, `power-law`, `scale-free`

**Statement:**

Under uniform cloning with rate $\lambda_c$, the degree distribution of IG vertices follows:

$$
P(\deg(e) = k) \sim k^{-\alpha} \quad \text{for large } k
$$

where $\alpha \in [2, 3]$ depends on the cloning kernel and localization scale.

**Scale-Free Property**: The IG exhibits scale-free behavior when $\rho$ is tuned near the critical percolation threshold.

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`prop-ig-connectedness`

---

### Fractal Set Composite Structure

#### Fractal Set

**Type:** Definition
**Label:** `def-fractal-set`
**Source:** [13_A_fractal_set.md § 3.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `composite-graph`, `causal-spacelike`, `fundamental-structure`, `discrete-spacetime`

**Statement:**

The **Fractal Set** $\mathcal{F}$ is the composite graph structure:

$$
\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})
$$

where:
- $\mathcal{E}$ is the set of episodes (vertices)
- $E_{\text{CST}}$ are directed timelike edges (causal links)
- $E_{\text{IG}}$ are undirected spacelike edges (information links)

**Interpretation:**
- **CST**: Encodes causal structure (genealogy, time ordering)
- **IG**: Encodes spacelike correlations (cloning interactions, measurement)
- **Composite**: Captures full spacetime structure of the adaptive gas dynamics

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`def-ig`, {prf:ref}`thm-fractal-set-metric`

---

#### Fractal Set Metric

**Type:** Theorem
**Label:** `thm-fractal-set-metric`
**Source:** [13_A_fractal_set.md § 3.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `spacetime-metric`, `lorentzian-signature`, `causal-structure`

**Statement:**

The Fractal Set $\mathcal{F}$ admits a **discrete Lorentzian metric** $d_{\mathcal{F}}: \mathcal{E} \times \mathcal{E} \to \mathbb{R}$:

$$
d_{\mathcal{F}}^2(e_i, e_j) = d_{\text{CST}}^2(e_i, e_j) - d_{\text{IG}}^2(e_i, e_j)
$$

**Properties:**
1. **Lorentzian Signature**: $(+, -, -, \ldots, -)$ in $(d+1)$ dimensions
2. **Causal Structure**: $d_{\mathcal{F}}^2 > 0$ iff $e_i$ and $e_j$ are timelike separated
3. **Light Cone**: $d_{\mathcal{F}}^2 = 0$ defines discrete light cone structure

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`thm-cst-lorentzian`, {prf:ref}`thm-continuum-limit-lorentzian`

---

#### Fractal Dimension of Fractal Set

**Type:** Conjecture
**Label:** `conj-fractal-dimension`
**Source:** [13_A_fractal_set.md § 3.3](13_fractal_set/13_A_fractal_set.md)
**Tags:** `fractal-dimension`, `hausdorff-dimension`, `scaling-exponent`

**Statement:**

The Fractal Set $\mathcal{F}$ has Hausdorff dimension:

$$
\dim_H(\mathcal{F}) = d + 1 - \delta
$$

where $\delta > 0$ is the **fractal defect** depending on:
- Cloning rate $\lambda_c$
- Localization scale $\rho$
- State space dimension $d$

**Scaling Hypothesis**: $\delta \sim \lambda_c^{\beta}$ for some $\beta \in (0, 1)$.

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`thm-cst-fractal-dimension`

---

#### Topological Genus of Fractal Set

**Type:** Proposition
**Label:** `prop-fractal-topology`
**Source:** [13_A_fractal_set.md § 3.4](13_fractal_set/13_A_fractal_set.md)
**Tags:** `topology`, `euler-characteristic`, `genus`, `cycles`

**Statement:**

For finite Fractal Set $\mathcal{F}_T$ with $|\mathcal{E}| = N_e$ episodes up to time $T$:

**Euler Characteristic**:

$$
\chi(\mathcal{F}_T) = N_e - |E_{\text{CST}}| - |E_{\text{IG}}| + N_{\text{cycles}}
$$

**Genus**:

$$
g(\mathcal{F}_T) = 1 - \frac{\chi(\mathcal{F}_T)}{2}
$$

The genus increases with the number of closed loops in the IG (spacelike cycles).

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`def-ig`

---

### Discrete Differential Geometry

#### Graph Laplacian on Fractal Set

**Type:** Definition
**Label:** `def-graph-laplacian-fractal`
**Source:** [13_A_fractal_set.md § 4.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `graph-laplacian`, `discrete-geometry`, `diffusion-operator`

**Statement:**

For function $f: \mathcal{E} \to \mathbb{R}$ on the Fractal Set, the **graph Laplacian** is:

$$
(\Delta_{\mathcal{F}} f)(e_i) = \sum_{e_j \sim e_i} w_{ij} [f(e_j) - f(e_i)]
$$

where:
- $e_j \sim e_i$ means $(e_i, e_j) \in E_{\text{IG}}$ (spacelike neighbors)
- $w_{ij} = \exp(-d_{\text{IG}}^2(e_i, e_j) / (2\rho^2))$ is the Gaussian weight

**Properties:**
1. **Self-adjoint**: $\langle f, \Delta_{\mathcal{F}} g \rangle = \langle \Delta_{\mathcal{F}} f, g \rangle$
2. **Non-positive**: $\langle f, \Delta_{\mathcal{F}} f \rangle \leq 0$
3. **Kernel**: $\ker(\Delta_{\mathcal{F}}) = \text{span}\{\mathbb{1}\}$ if IG is connected

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`thm-laplacian-convergence`

---

#### Discrete Gradient on Fractal Set

**Type:** Definition
**Label:** `def-discrete-gradient`
**Source:** [13_A_fractal_set.md § 4.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `discrete-gradient`, `edge-function`, `coboundary`

**Statement:**

For scalar function $f: \mathcal{E} \to \mathbb{R}$, the **discrete gradient** is a function on edges:

$$
(\nabla_{\mathcal{F}} f)(e_i, e_j) = \frac{f(e_j) - f(e_i)}{d_{\text{IG}}(e_i, e_j)}
$$

for $(e_i, e_j) \in E_{\text{IG}}$.

**Discrete Divergence**: For edge function $\mathbf{X}: E_{\text{IG}} \to \mathbb{R}$:

$$
(\text{div}_{\mathcal{F}} \mathbf{X})(e_i) = \sum_{e_j \sim e_i} \mathbf{X}(e_i, e_j)
$$

**Relation**: $\Delta_{\mathcal{F}} = -\text{div}_{\mathcal{F}} \circ \nabla_{\mathcal{F}}$

**Related Results:** {prf:ref}`def-graph-laplacian-fractal`, {prf:ref}`def-discrete-curl`

---

#### Discrete Curl and Plaquettes

**Type:** Definition
**Label:** `def-discrete-curl`
**Source:** [13_A_fractal_set.md § 4.3](13_fractal_set/13_A_fractal_set.md)
**Tags:** `discrete-curl`, `plaquette`, `wilson-loop`, `gauge-theory`

**Statement:**

For vector field $\mathbf{A}: E_{\text{IG}} \to \mathbb{R}^d$ on IG edges, the **discrete curl** on a plaquette $P = (e_1, e_2, e_3, e_4)$ is:

$$
(\text{curl}_{\mathcal{F}} \mathbf{A})(P) = \mathbf{A}(e_1, e_2) + \mathbf{A}(e_2, e_3) + \mathbf{A}(e_3, e_4) + \mathbf{A}(e_4, e_1)
$$

(sum along closed loop).

**Stokes' Theorem**: For surface $S$ with boundary $\partial S$:

$$
\sum_{P \subset S} (\text{curl}_{\mathcal{F}} \mathbf{A})(P) = \sum_{(e_i, e_j) \in \partial S} \mathbf{A}(e_i, e_j)
$$

**Related Results:** {prf:ref}`def-discrete-gradient`, {prf:ref}`def-wilson-loop`, {prf:ref}`thm-lattice-gauge-action`

---

#### Discrete Ricci Curvature

**Type:** Definition
**Label:** `def-discrete-ricci`
**Source:** [13_A_fractal_set.md § 4.4](13_fractal_set/13_A_fractal_set.md)
**Tags:** `ricci-curvature`, `ollivier-curvature`, `optimal-transport`

**Statement:**

For vertices $e_i, e_j \in \mathcal{E}$ with $(e_i, e_j) \in E_{\text{IG}}$, the **Ollivier-Ricci curvature** is:

$$
\kappa(e_i, e_j) = 1 - \frac{W_1(\mu_i, \mu_j)}{d_{\text{IG}}(e_i, e_j)}
$$

where:
- $\mu_i$ is the uniform measure on neighbors of $e_i$
- $W_1$ is the Wasserstein-1 distance

**Interpretation**:
- $\kappa > 0$: Positive curvature (sphere-like)
- $\kappa = 0$: Flat (Euclidean-like)
- $\kappa < 0$: Negative curvature (hyperbolic-like)

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`thm-emergent-ricci`

---

### Propagators and Dynamics

#### Heat Kernel on Fractal Set

**Type:** Definition
**Label:** `def-heat-kernel-fractal`
**Source:** [13_A_fractal_set.md § 5.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `heat-kernel`, `diffusion`, `propagator`, `graph-laplacian`

**Statement:**

The **heat kernel** $K_t: \mathcal{E} \times \mathcal{E} \to \mathbb{R}_+$ satisfies:

$$
\frac{\partial K_t}{\partial t} = \Delta_{\mathcal{F}} K_t
$$

with initial condition $K_0(e_i, e_j) = \delta_{ij}$.

**Explicit Form**:

$$
K_t(e_i, e_j) = \sum_{k=0}^\infty e^{-\lambda_k t} \psi_k(e_i) \psi_k(e_j)
$$

where $\{\lambda_k, \psi_k\}$ are eigenvalues/eigenfunctions of $-\Delta_{\mathcal{F}}$.

**Related Results:** {prf:ref}`def-graph-laplacian-fractal`, {prf:ref}`thm-heat-kernel-convergence`

---

#### Episode Measure

**Type:** Definition
**Label:** `def-episode-measure`
**Source:** [13_A_fractal_set.md § 5.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `episode-measure`, `qsd`, `stationary-distribution`

**Statement:**

The **episode measure** $\mu_T: \mathcal{E} \to [0, 1]$ at time $T$ is:

$$
\mu_T(e) = \frac{\text{number of times episode } e \text{ is visited up to time } T}{\sum_{e' \in \mathcal{E}} \text{number of times } e' \text{ is visited}}
$$

**Limit**: As $T \to \infty$, $\mu_T \to \mu_{\text{QSD}}$, the quasi-stationary distribution.

**Related Results:** {prf:ref}`def-qsd`, {prf:ref}`thm-episode-measure-convergence`

---

#### Episode Measure Evolution

**Type:** Proposition
**Label:** `prop-episode-measure-evolution`
**Source:** [13_A_fractal_set.md § 5.3](13_fractal_set/13_A_fractal_set.md)
**Tags:** `fokker-planck`, `master-equation`, `measure-evolution`

**Statement:**

The episode measure evolves according to:

$$
\frac{d\mu_t}{dt} = \mathcal{L}^\dagger \mu_t
$$

where $\mathcal{L}^\dagger$ is the adjoint of the generator $\mathcal{L}$ (Fokker-Planck operator).

**Spectral Decomposition**:

$$
\mu_t = \mu_{\text{QSD}} + \sum_{k=1}^\infty e^{-\gamma_k t} c_k \psi_k
$$

where $\gamma_k > 0$ are the spectral gap eigenvalues.

**Related Results:** {prf:ref}`def-episode-measure`, {prf:ref}`thm-exponential-convergence-qsd`

---

#### Cloning Propagator

**Type:** Definition
**Label:** `def-cloning-propagator`
**Source:** [13_A_fractal_set.md § 5.4](13_fractal_set/13_A_fractal_set.md)
**Tags:** `cloning-kernel`, `transition-kernel`, `markov-chain`

**Statement:**

The **cloning propagator** $P_{\text{clone}}: \mathcal{E} \times \mathcal{E} \to [0, 1]$ is:

$$
P_{\text{clone}}(e_i \to e_j) = \frac{K_{\text{clone}}(e_i, e_j)}{\sum_{e' \in \mathcal{N}_\rho(e_i)} K_{\text{clone}}(e_i, e')}
$$

where $K_{\text{clone}}(e_i, e_j) = \exp(\alpha F_j - \beta H_j)$ is the cloning kernel and $\mathcal{N}_\rho(e_i)$ is the local neighborhood.

**Interpretation**: Probability that episode $e_i$ clones into episode $e_j$.

**Related Results:** {prf:ref}`def-cloning-kernel`, {prf:ref}`thm-cloning-contraction`

---

#### Discrete Path Integral

**Type:** Definition
**Label:** `def-discrete-path-integral`
**Source:** [13_A_fractal_set.md § 5.5](13_fractal_set/13_A_fractal_set.md)
**Tags:** `path-integral`, `discrete-action`, `feynman-propagator`

**Statement:**

For episodes $e_i$ at time $t_i$ and $e_f$ at time $t_f$, the **discrete path integral** is:

$$
K(e_i, t_i; e_f, t_f) = \sum_{\gamma: e_i \to e_f} e^{-S[\gamma]}
$$

where:
- Sum is over all paths $\gamma$ in $\mathcal{F}$ from $e_i$ to $e_f$
- $S[\gamma]$ is the discrete action along path $\gamma$

**Discrete Action**:

$$
S[\gamma] = \sum_{(e, e') \in \gamma} \left[ \frac{1}{2\tau} d_{\text{IG}}^2(e, e') + V(e) \right]
$$

where $\tau$ is the time step and $V(e)$ is the potential.

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`thm-feynman-kac-discrete`

---

## Continuum Limit Theory

### Discrete Symmetries

#### Permutation Invariance of Fractal Set

**Type:** Theorem
**Label:** `thm-fractal-permutation-invariance`
**Source:** [13_B_fractal_set_continuum_limit.md § 1.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `permutation-invariance`, `symmetry`, `exchangeability`

**Statement:**

The Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ is **permutation invariant**:

For any permutation $\sigma \in S_N$ of episode labels:

$$
\mathcal{F} \cong \sigma(\mathcal{F})
$$

where $\sigma(\mathcal{F}) = (\sigma(\mathcal{E}), \{(\sigma(e_i), \sigma(e_j)) : (e_i, e_j) \in E_{\text{CST}} \cup E_{\text{IG}}\})$.

**Implication**: All graph-theoretic properties (Laplacian spectrum, Ricci curvature, etc.) are permutation invariant.

**Related Results:** {prf:ref}`def-fractal-set`, {prf:ref}`thm-permutation-invariance-gas`

---

#### Translation Equivariance

**Type:** Theorem
**Label:** `thm-fractal-translation-equivariance`
**Source:** [13_B_fractal_set_continuum_limit.md § 1.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `translation-equivariance`, `spatial-homogeneity`, `euclidean-symmetry`

**Statement:**

If the state space $\mathcal{X} = \mathbb{R}^d$ with translation-invariant potential $V$, then:

For translation $T_a: x \mapsto x + a$:

$$
\mathcal{F}_{T_a(\mathcal{E})} = T_a(\mathcal{F}_{\mathcal{E}})
$$

where $T_a(\mathcal{E}) = \{(x_i + a, v_i, s_i, t_i)\}$.

**Geometric Interpretation**: The Fractal Set structure commutes with spatial translations.

**Related Results:** {prf:ref}`thm-fractal-permutation-invariance`, {prf:ref}`thm-rotational-equivariance`

---

#### Rotational Equivariance

**Type:** Theorem
**Label:** `thm-rotational-equivariance`
**Source:** [13_B_fractal_set_continuum_limit.md § 1.3](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `rotational-equivariance`, `SO(d)-symmetry`, `angular-momentum`

**Statement:**

If potential $V$ is rotationally invariant, then for $R \in SO(d)$:

$$
\mathcal{F}_{R(\mathcal{E})} = R(\mathcal{F}_{\mathcal{E}})
$$

where $R(\mathcal{E}) = \{(Rx_i, Rv_i, s_i, t_i)\}$.

**Conservation Law**: Rotational equivariance implies conservation of total angular momentum in the continuum limit.

**Related Results:** {prf:ref}`thm-fractal-translation-equivariance`, {prf:ref}`thm-noether-discrete`

---

#### Time-Reversal Asymmetry

**Type:** Proposition
**Label:** `prop-time-reversal-asymmetry`
**Source:** [13_B_fractal_set_continuum_limit.md § 1.4](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `time-reversal`, `irreversibility`, `arrow-of-time`

**Statement:**

The Fractal Set dynamics are **NOT** time-reversal invariant due to:
1. **Cloning irreversibility**: Cloning events create new episodes (not reversible)
2. **Absorption**: Dead episodes are permanently removed
3. **Entropy production**: KL divergence $D(\rho_t \| \rho_{\text{QSD}})$ decreases monotonically

The CST structure explicitly breaks time-reversal symmetry: directed edges $E_{\text{CST}}$ define an arrow of time.

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`thm-kl-convergence`

---

### Gauge Connection

#### Discrete Gauge Connection on IG

**Type:** Definition
**Label:** `def-discrete-gauge-connection`
**Source:** [13_B_fractal_set_continuum_limit.md § 2.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `gauge-connection`, `parallel-transport`, `wilson-line`

**Statement:**

For gauge group $G$ (e.g., $U(1)$, $SU(N)$), a **discrete gauge connection** on the IG is a map:

$$
A: E_{\text{IG}} \to \mathfrak{g}
$$

where $\mathfrak{g}$ is the Lie algebra of $G$.

**Parallel Transport**: Along edge $(e_i, e_j) \in E_{\text{IG}}$:

$$
U_{ij} = \exp(A(e_i, e_j)) \in G
$$

**Gauge Transformation**: Under $g: \mathcal{E} \to G$:

$$
A(e_i, e_j) \mapsto g_i A(e_i, e_j) g_j^{-1} + g_i dg_j^{-1}
$$

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`def-wilson-loop`, {prf:ref}`thm-gauge-connection-convergence`

---

#### Holonomy on Closed Loops

**Type:** Definition
**Label:** `def-holonomy-discrete`
**Source:** [13_B_fractal_set_continuum_limit.md § 2.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `holonomy`, `wilson-loop`, `parallel-transport`, `non-abelian`

**Statement:**

For closed loop $\gamma = (e_1, e_2, \ldots, e_k, e_1)$ in the IG, the **holonomy** is:

$$
\text{Hol}_\gamma(A) = U_{12} U_{23} \cdots U_{k1} \in G
$$

where $U_{ij} = \exp(A(e_i, e_j))$.

**Properties:**
1. **Abelian case** ($G = U(1)$): $\text{Hol}_\gamma(A) = \exp\left(\sum_{(e_i, e_j) \in \gamma} A(e_i, e_j)\right)$
2. **Non-abelian case**: Order matters (path-ordered product)
3. **Gauge Invariance**: $\text{Tr}(\text{Hol}_\gamma(A))$ is gauge invariant

**Related Results:** {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`def-wilson-loop`, {prf:ref}`thm-holonomy-convergence`

---

#### Curvature from Plaquettes

**Type:** Definition
**Label:** `def-discrete-curvature`
**Source:** [13_B_fractal_set_continuum_limit.md § 2.3](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `field-strength`, `curvature`, `plaquette`, `gauge-invariant`

**Statement:**

For plaquette $P = (e_1, e_2, e_3, e_4)$, the **discrete curvature** (field strength) is:

$$
F(P) = A(e_1, e_2) + A(e_2, e_3) + A(e_3, e_4) + A(e_4, e_1) + [A(e_1, e_2), A(e_2, e_3)] + \ldots
$$

**Abelian case**: $F(P) = \sum_{(e_i, e_j) \in \partial P} A(e_i, e_j)$

**Continuum Limit**: $F(P) \to F_{\mu\nu}$ (field strength tensor) as lattice spacing $\epsilon \to 0$.

**Related Results:** {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`def-discrete-curl`, {prf:ref}`thm-curvature-convergence`

---

### Graph Laplacian Convergence

#### Graph Laplacian Convergence Theorem

**Type:** Theorem
**Label:** `thm-laplacian-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 3.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `graph-laplacian`, `continuum-limit`, `laplace-beltrami`, `convergence-rate`

**Statement:**

As the number of episodes $N \to \infty$ and localization scale $\rho \to 0$ with $N\rho^d \to \infty$:

$$
\Delta_{\mathcal{F}} f \to \Delta_{\mathcal{X}} f
$$

pointwise for $f \in C^2(\mathcal{X})$, where $\Delta_{\mathcal{X}}$ is the Laplace-Beltrami operator on state space $\mathcal{X}$.

**Convergence Rate**: For appropriate scaling:

$$
\|\Delta_{\mathcal{F}} f - \Delta_{\mathcal{X}} f\|_{L^2} = O(N^{-1/4})
$$

**Related Results:** {prf:ref}`def-graph-laplacian-fractal`, {prf:ref}`thm-heat-kernel-convergence`

---

#### Spectral Convergence

**Type:** Theorem
**Label:** `thm-spectral-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 3.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `spectral-gap`, `eigenvalue-convergence`, `spectral-theorem`

**Statement:**

Let $\{\lambda_k^{(N)}\}$ be the eigenvalues of $-\Delta_{\mathcal{F}}$ on $N$ episodes, and $\{\lambda_k\}$ the eigenvalues of $-\Delta_{\mathcal{X}}$ on the continuum. Then:

$$
\lambda_k^{(N)} \to \lambda_k \quad \text{as } N \to \infty
$$

for each fixed $k$.

**Convergence Rate**: $|\lambda_k^{(N)} - \lambda_k| = O(N^{-1/(d+2)})$

**Related Results:** {prf:ref}`thm-laplacian-convergence`, {prf:ref}`thm-exponential-convergence-qsd`

---

#### Heat Kernel Convergence

**Type:** Theorem
**Label:** `thm-heat-kernel-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 3.3](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `heat-kernel`, `diffusion`, `brownian-motion`, `continuum-limit`

**Statement:**

The discrete heat kernel $K_t^{(N)}(e_i, e_j)$ on the Fractal Set converges to the continuum heat kernel:

$$
\lim_{N \to \infty} K_t^{(N)}(x_i, x_j) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{\|x_i - x_j\|^2}{4t}\right)
$$

in the $L^2$ topology, where $x_i, x_j \in \mathcal{X}$ are the positions of episodes $e_i, e_j$.

**Related Results:** {prf:ref}`def-heat-kernel-fractal`, {prf:ref}`thm-laplacian-convergence`

---

#### Weighted Graph Laplacian

**Type:** Proposition
**Label:** `prop-weighted-laplacian`
**Source:** [13_B_fractal_set_continuum_limit.md § 3.4](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `weighted-graph`, `edge-weights`, `gaussian-kernel`

**Statement:**

For Gaussian edge weights $w_{ij} = \exp(-\|x_i - x_j\|^2 / (2\rho^2))$:

The weighted graph Laplacian:

$$
(\Delta_{\mathcal{F}}^w f)(e_i) = \frac{1}{\rho^2} \sum_{e_j \sim e_i} w_{ij} [f(e_j) - f(e_i)]
$$

converges to:

$$
(\Delta_{\mathcal{X}} f)(x) + O(\rho^2)
$$

as $\rho \to 0$ and $N\rho^d \to \infty$.

**Related Results:** {prf:ref}`def-graph-laplacian-fractal`, {prf:ref}`thm-laplacian-convergence`

---

### Episode Measure Convergence

#### Episode Measure Convergence Theorem

**Type:** Theorem
**Label:** `thm-episode-measure-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 4.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `episode-measure`, `qsd`, `weak-convergence`, `continuum-limit`

**Statement:**

As $N \to \infty$, the episode measure $\mu_T^{(N)}$ converges weakly to the quasi-stationary distribution $\rho_{\text{QSD}}$:

$$
\mu_T^{(N)} \xrightarrow{w} \rho_{\text{QSD}}
$$

in the sense:

$$
\lim_{N \to \infty} \int f \, d\mu_T^{(N)} = \int f \, \rho_{\text{QSD}} \, dx
$$

for all continuous bounded $f: \mathcal{X} \to \mathbb{R}$.

**Convergence Rate**: $\|\mu_T^{(N)} - \rho_{\text{QSD}}\|_{TV} = O(N^{-1/2})$

**Related Results:** {prf:ref}`def-episode-measure`, {prf:ref}`def-qsd`, {prf:ref}`thm-exponential-convergence-qsd`

---

#### Empirical Measure Fluctuations

**Type:** Theorem
**Label:** `thm-empirical-measure-fluctuations`
**Source:** [13_B_fractal_set_continuum_limit.md § 4.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `central-limit-theorem`, `gaussian-fluctuations`, `propagation-of-chaos`

**Statement:**

Define the empirical measure:

$$
\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{x_i}
$$

Then the fluctuations:

$$
\sqrt{N}(\mu_N - \rho_{\text{QSD}}) \xrightarrow{d} \mathcal{N}(0, \Sigma)
$$

converge in distribution to a Gaussian process with covariance operator $\Sigma$.

**Related Results:** {prf:ref}`thm-episode-measure-convergence`, {prf:ref}`thm-propagation-of-chaos`

---

#### Propagation of Chaos

**Type:** Theorem
**Label:** `thm-propagation-of-chaos`
**Source:** [13_B_fractal_set_continuum_limit.md § 4.3](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `propagation-of-chaos`, `mean-field-limit`, `independence`

**Statement:**

For fixed time $T$ and fixed $k$ walkers, the joint empirical measure:

$$
\mu_N^{(k)} = \frac{1}{N^k} \sum_{i_1, \ldots, i_k} \delta_{(x_{i_1}, \ldots, x_{i_k})}
$$

converges to the product measure:

$$
\mu_N^{(k)} \xrightarrow{w} \rho_{\text{QSD}}^{\otimes k}
$$

**Interpretation**: In the mean-field limit $N \to \infty$, walkers become independent and identically distributed according to $\rho_{\text{QSD}}$.

**Related Results:** {prf:ref}`thm-episode-measure-convergence`, {prf:ref}`thm-mean-field-limit`

---

#### Wasserstein Convergence Rate

**Type:** Theorem
**Label:** `thm-wasserstein-convergence-rate`
**Source:** [13_B_fractal_set_continuum_limit.md § 4.4](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `wasserstein-distance`, `optimal-transport`, `convergence-rate`

**Statement:**

The Wasserstein-2 distance between episode measure and QSD satisfies:

$$
W_2(\mu_N, \rho_{\text{QSD}}) \leq C N^{-1/4}
$$

for constant $C$ depending on the cloning rate and localization scale.

**Proof Strategy**: Uses Stein's method + optimal transport coupling.

**Related Results:** {prf:ref}`thm-episode-measure-convergence`, {prf:ref}`def-wasserstein-distance`

---

### Holonomy Convergence

#### Wilson Loop on Fractal Set

**Type:** Definition
**Label:** `def-wilson-loop`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `wilson-loop`, `gauge-invariant`, `parallel-transport`, `observable`

**Statement:**

For closed loop $\gamma$ in the IG and gauge connection $A$, the **Wilson loop** is:

$$
W_\gamma[A] = \text{Tr}\left[\text{Hol}_\gamma(A)\right] = \text{Tr}\left[\prod_{(e_i, e_j) \in \gamma} U_{ij}\right]
$$

where $U_{ij} = \exp(A(e_i, e_j))$ is the parallel transport operator.

**Gauge Invariance**: $W_\gamma[A^g] = W_\gamma[A]$ for any gauge transformation $g$.

**Related Results:** {prf:ref}`def-holonomy-discrete`, {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`thm-wilson-loop-convergence`

---

#### Wilson Loop Convergence Theorem

**Type:** Theorem
**Label:** `thm-wilson-loop-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `wilson-loop`, `continuum-limit`, `area-law`, `lattice-gauge-theory`

**Statement:**

For smooth loop $\gamma$ in state space $\mathcal{X}$ and smooth gauge connection $A_\mu$:

$$
\lim_{N \to \infty} W_{\gamma_N}[A] = \text{Tr}\left[\mathcal{P} \exp\left(\oint_\gamma A_\mu dx^\mu\right)\right]
$$

where $\gamma_N$ is the discretization of $\gamma$ on the IG with $N$ episodes, and $\mathcal{P}$ denotes path ordering.

**Convergence Rate**: $|W_{\gamma_N}[A] - W_\gamma[A]| = O(\epsilon^2)$ where $\epsilon = 1/N^{1/d}$ is the lattice spacing.

**Related Results:** {prf:ref}`def-wilson-loop`, {prf:ref}`thm-holonomy-convergence`, {prf:ref}`thm-area-law`

---

#### Holonomy Convergence

**Type:** Theorem
**Label:** `thm-holonomy-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.3](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `holonomy`, `parallel-transport`, `lie-group`, `continuum-limit`

**Statement:**

For path $\gamma: [0, 1] \to \mathcal{X}$ and discrete approximation $\gamma_N$ with $N$ steps:

$$
\lim_{N \to \infty} \text{Hol}_{\gamma_N}(A) = \mathcal{P} \exp\left(\int_0^1 A_\mu(\gamma(s)) \dot{\gamma}^\mu(s) \, ds\right)
$$

in the Lie group $G$.

**Proof Technique**: Uses Baker-Campbell-Hausdorff formula for non-abelian groups.

**Related Results:** {prf:ref}`def-holonomy-discrete`, {prf:ref}`thm-wilson-loop-convergence`

---

#### Area Law for Wilson Loops

**Type:** Conjecture
**Label:** `conj-area-law`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.4](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `area-law`, `confinement`, `wilson-loop`, `strong-coupling`

**Statement:**

For large loop $\gamma$ enclosing area $A(\gamma)$:

$$
\langle W_\gamma[A] \rangle \sim \exp(-\sigma A(\gamma))
$$

where $\sigma$ is the **string tension** and $\langle \cdot \rangle$ denotes ensemble average.

**Physical Interpretation**: Area law implies confinement of gauge charges (analogous to QCD).

**Related Results:** {prf:ref}`def-wilson-loop`, {prf:ref}`thm-wilson-loop-convergence`, {prf:ref}`conj-mass-gap`

---

#### Geometric Area from IG

**Type:** Theorem
**Label:** `thm-geometric-area-ig`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.5](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `geometric-area`, `triangulation`, `riemannian-metric`, `discrete-geometry`

**Statement:**

For region $R \subset \mathcal{X}$, the **geometric area** computed from the IG is:

$$
A_{\text{IG}}(R) = \sum_{P \subset R} A_P
$$

where $P$ are plaquettes (minimal closed loops) in the IG and:

$$
A_P = \frac{1}{2} \left| \sum_{(e_i, e_j) \in P} (x_i \times x_j) \right|
$$

**Continuum Limit**: $A_{\text{IG}}(R) \to \int_R \sqrt{g} \, d^d x$ as $N \to \infty$, where $g$ is the determinant of the emergent Riemannian metric.

**Related Results:** {prf:ref}`def-ig`, {prf:ref}`thm-emergent-metric`, {prf:ref}`thm-wilson-loop-convergence`

---

## Causal Set Quantum Gravity

### Causal Set Axioms

#### Causal Set Definition

**Type:** Definition
**Label:** `def-causal-set`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 1.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `causal-set`, `partially-ordered-set`, `discrete-spacetime`, `quantum-gravity`

**Statement:**

A **causal set** (causet) is a locally finite partially ordered set $(C, \prec)$ where:
1. **Partial Order**: $\prec$ is transitive, reflexive, and antisymmetric
2. **Local Finiteness**: For any $x, z \in C$, the set $\{y \in C : x \prec y \prec z\}$ is finite

**Interpretation**:
- Elements of $C$ represent spacetime events
- $x \prec y$ means "$x$ causally precedes $y$" (timelike or lightlike separation)
- Local finiteness ensures discrete structure

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`axiom-causal-order`, {prf:ref}`thm-cst-to-lorentzian`

---

#### Axiom of Causal Order

**Type:** Axiom
**Label:** `axiom-causal-order`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 1.2](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `causal-order`, `axiom`, `partial-order`, `transitivity`

**Statement:**

For causal set $(C, \prec)$:
1. **Reflexivity**: $x \prec x$ for all $x \in C$
2. **Antisymmetry**: If $x \prec y$ and $y \prec x$, then $x = y$
3. **Transitivity**: If $x \prec y$ and $y \prec z$, then $x \prec z$

**Physical Meaning**: Causality is a fundamental structure encoded in the partial order.

**Related Results:** {prf:ref}`def-causal-set`, {prf:ref}`axiom-local-finiteness`

---

#### Axiom of Local Finiteness

**Type:** Axiom
**Label:** `axiom-local-finiteness`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 1.3](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `local-finiteness`, `discrete-structure`, `causal-interval`

**Statement:**

For causal set $(C, \prec)$ and any $x, z \in C$:

The **causal interval** $I(x, z) := \{y \in C : x \prec y \prec z\}$ is **finite**.

**Physical Meaning**:
- Spacetime is fundamentally discrete at Planck scale
- Causal intervals contain only finitely many events
- Rules out continuum spacetime at fundamental level

**Related Results:** {prf:ref}`def-causal-set`, {prf:ref}`axiom-causal-order`, {prf:ref}`prop-causal-interval-count`

---

#### Causal Interval Cardinality

**Type:** Proposition
**Label:** `prop-causal-interval-count`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 1.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `causal-interval`, `volume`, `spacetime-volume`

**Statement:**

For causal set approximating $(d+1)$-dimensional Lorentzian spacetime:

$$
|I(x, z)| \approx \frac{V(x, z)}{\ell_P^{d+1}}
$$

where:
- $|I(x, z)|$ is the number of events in causal interval
- $V(x, z)$ is the spacetime volume of the causal diamond
- $\ell_P$ is the Planck length

**Interpretation**: Causal interval cardinality measures spacetime volume.

**Related Results:** {prf:ref}`axiom-local-finiteness`, {prf:ref}`def-myrheim-meyer-dimension`

---

### Sprinkling and Dimension

#### Sprinkling Process

**Type:** Definition
**Label:** `def-sprinkling`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 2.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `poisson-process`, `random-causet`, `sprinkle`, `lorentzian-manifold`

**Statement:**

Given Lorentzian manifold $(M, g)$, a **sprinkling** is a random causal set generated by:
1. Sample points from Poisson process with density $\rho = 1/\ell_P^{d+1}$
2. Inherit causal order from spacetime: $x \prec y$ iff $x \in J^-(y)$ (causal past)

**Distribution**: Number of points in region $R$ follows Poisson$(\rho \cdot \text{Vol}(R))$.

**Related Results:** {prf:ref}`def-causal-set`, {prf:ref}`thm-sprinkling-approximation`

---

#### Sprinkling Approximation Theorem

**Type:** Theorem
**Label:** `thm-sprinkling-approximation`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 2.2](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `sprinkling`, `manifoldlike`, `continuum-limit`

**Statement:**

Let $(M, g)$ be a $(d+1)$-dimensional Lorentzian manifold. A sprinkling $C$ with density $\rho = 1/\ell_P^{d+1}$ satisfies:

**Manifoldlikeness**: With high probability, $C$ is **faithful** to $(M, g)$:
- Causal structure of $C$ approximates causal structure of $(M, g)$
- Dimension estimators recover $d+1$
- Local geometric quantities (curvature, etc.) can be reconstructed

**Convergence**: As $\ell_P \to 0$ (continuum limit), geometric properties converge.

**Related Results:** {prf:ref}`def-sprinkling`, {prf:ref}`def-myrheim-meyer-dimension`, {prf:ref}`thm-dimension-estimation`

---

#### Myrheim-Meyer Dimension

**Type:** Definition
**Label:** `def-myrheim-meyer-dimension`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 2.3](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `dimension-estimator`, `causal-interval`, `discrete-dimension`

**Statement:**

For causal set $C$ and events $x, z$ with $x \prec z$, the **Myrheim-Meyer dimension** is estimated from:

$$
\langle |I(x, y)| \cdot |I(y, z)| \rangle \propto |I(x, z)|^2
$$

where $\langle \cdot \rangle$ averages over intermediate events $y$ with $x \prec y \prec z$.

**Dimension Formula**:

$$
d + 1 = \lim_{|I(x,z)| \to \infty} \frac{\log \langle |I(x, y)| \cdot |I(y, z)| \rangle}{\log |I(x, z)|}
$$

**Related Results:** {prf:ref}`def-causal-set`, {prf:ref}`prop-causal-interval-count`, {prf:ref}`thm-dimension-estimation`

---

#### Dimension Estimation Theorem

**Type:** Theorem
**Label:** `thm-dimension-estimation`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 2.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `dimension-estimator`, `myrheim-meyer`, `convergence`, `manifoldlike`

**Statement:**

For sprinkling $C$ of $(d+1)$-dimensional Lorentzian manifold $(M, g)$:

The Myrheim-Meyer dimension estimator converges:

$$
\hat{d} \to d + 1 \quad \text{as } \ell_P \to 0
$$

with convergence rate:

$$
|\hat{d} - (d+1)| = O\left(\frac{1}{\sqrt{|I(x,z)|}}\right)
$$

**Related Results:** {prf:ref}`def-myrheim-meyer-dimension`, {prf:ref}`thm-sprinkling-approximation`

---

### Lorentzian Structure

#### Lorentzian Signature of CST

**Type:** Theorem
**Label:** `thm-cst-lorentzian`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 3.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `lorentzian-signature`, `causal-structure`, `discrete-metric`

**Statement:**

The CST structure $(\mathcal{E}, E_{\text{CST}})$ defines a discrete Lorentzian metric:

For episodes $e_i, e_j \in \mathcal{E}$:

$$
\eta_{ij} = \begin{cases}
+1 & \text{if } e_i \prec e_j \text{ (timelike)} \\
0 & \text{if } e_i \text{ and } e_j \text{ are spacelike separated} \\
\text{undefined} & \text{if } e_i \text{ and } e_j \text{ are acausal}
\end{cases}
$$

**Signature**: In continuum limit, recovers $(-,+,+,\ldots,+)$ or $(+,-,-,\ldots,-)$ Lorentzian signature.

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`def-causal-set`, {prf:ref}`thm-fractal-set-metric`

---

#### Light Cone Structure

**Type:** Definition
**Label:** `def-light-cone-discrete`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 3.2](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `light-cone`, `causal-past`, `causal-future`, `null-geodesic`

**Statement:**

For episode $e \in \mathcal{E}$:
- **Causal Past**: $J^-(e) = \{e' \in \mathcal{E} : e' \prec e\}$
- **Causal Future**: $J^+(e) = \{e' \in \mathcal{E} : e \prec e'\}$
- **Light Cone**: $\partial J^\pm(e)$ consists of episodes with $d_{\mathcal{F}}^2(e, e') = 0$

**Continuum Limit**: As $N \to \infty$, discrete light cone converges to continuous light cone in Minkowski or curved spacetime.

**Related Results:** {prf:ref}`def-cst`, {prf:ref}`thm-fractal-set-metric`, {prf:ref}`def-causal-set`

---

#### Timelike and Spacelike Separation

**Type:** Definition
**Label:** `def-timelike-spacelike`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 3.3](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `timelike`, `spacelike`, `causal-structure`, `lorentzian-distance`

**Statement:**

For episodes $e_i, e_j \in \mathcal{E}$:
- **Timelike**: $e_i \prec e_j$ (connected by CST edge) $\Rightarrow d_{\mathcal{F}}^2(e_i, e_j) > 0$
- **Spacelike**: $(e_i, e_j) \in E_{\text{IG}}$ (connected by IG edge) $\Rightarrow d_{\mathcal{F}}^2(e_i, e_j) < 0$
- **Null**: $d_{\mathcal{F}}^2(e_i, e_j) = 0$

**Related Results:** {prf:ref}`thm-fractal-set-metric`, {prf:ref}`def-light-cone-discrete`

---

#### Proper Time on Causal Chains

**Type:** Definition
**Label:** `def-proper-time-discrete`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 3.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `proper-time`, `causal-chain`, `discrete-interval`

**Statement:**

For causal chain $e_1 \prec e_2 \prec \cdots \prec e_k$ in the CST, the **proper time** is:

$$
\tau(e_1, e_k) = \sum_{i=1}^{k-1} \sqrt{d_{\mathcal{F}}^2(e_i, e_{i+1})}
$$

**Continuum Limit**: As $N \to \infty$:

$$
\tau(e_1, e_k) \to \int_{\gamma} \sqrt{-g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu} \, d\lambda
$$

where $\gamma$ is the worldline from $e_1$ to $e_k$.

**Related Results:** {prf:ref}`def-timelike-spacelike`, {prf:ref}`thm-cst-lorentzian`

---

### Discrete Differential Operators

#### Discrete D'Alembertian

**Type:** Definition
**Label:** `def-discrete-dalembertian`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 4.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `dalembertian`, `wave-operator`, `retarded-propagator`, `hyperbolic-pde`

**Statement:**

For scalar field $\phi: \mathcal{E} \to \mathbb{R}$ on the Fractal Set, the **discrete d'Alembertian** is:

$$
(\Box_{\mathcal{F}} \phi)(e) = \sum_{e' \in J^-(e) \cup J^+(e)} w_{ee'} [\phi(e') - \phi(e)]
$$

where $w_{ee'}$ are weights depending on causal distance.

**Continuum Limit**: $\Box_{\mathcal{F}} \phi \to \Box \phi = \eta^{\mu\nu} \partial_\mu \partial_\nu \phi$ as $N \to \infty$.

**Related Results:** {prf:ref}`def-graph-laplacian-fractal`, {prf:ref}`def-light-cone-discrete`, {prf:ref}`thm-wave-equation-convergence`

---

#### Retarded Green's Function

**Type:** Definition
**Label:** `def-retarded-greens-function`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 4.2](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `greens-function`, `retarded-propagator`, `causal-propagation`

**Statement:**

The **retarded Green's function** $G_{\text{ret}}(e, e')$ satisfies:

$$
\Box_{\mathcal{F}} G_{\text{ret}}(e, e') = \delta_{ee'}
$$

with boundary condition: $G_{\text{ret}}(e, e') = 0$ if $e' \not\in J^-(e)$ (causality).

**Physical Interpretation**: Propagates signals forward in time only (respects causality).

**Related Results:** {prf:ref}`def-discrete-dalembertian`, {prf:ref}`def-light-cone-discrete`, {prf:ref}`thm-feynman-propagator-discrete`

---

#### Feynman Propagator

**Type:** Definition
**Label:** `def-feynman-propagator-discrete`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 4.3](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `feynman-propagator`, `quantum-field-theory`, `path-integral`

**Statement:**

The **Feynman propagator** $G_F(e, e')$ is:

$$
G_F(e, e') = \theta(e - e') G_{\text{ret}}(e, e') + \theta(e' - e) G_{\text{adv}}(e, e')
$$

where $\theta$ is the time-ordering function and $G_{\text{adv}}$ is the advanced Green's function.

**Path Integral Form**:

$$
G_F(e, e') = \sum_{\gamma: e' \to e} e^{-S[\gamma]}
$$

summing over all paths in $\mathcal{F}$.

**Related Results:** {prf:ref}`def-retarded-greens-function`, {prf:ref}`def-discrete-path-integral`, {prf:ref}`thm-feynman-kac-discrete`

---

#### Wave Equation Convergence

**Type:** Theorem
**Label:** `thm-wave-equation-convergence`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 4.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `wave-equation`, `hyperbolic-pde`, `convergence`, `continuum-limit`

**Statement:**

For smooth initial data $\phi_0, \dot{\phi}_0$ and discrete approximation $\phi^{(N)}$ solving:

$$
\Box_{\mathcal{F}} \phi^{(N)} = 0
$$

As $N \to \infty$, $\phi^{(N)} \to \phi$ where $\phi$ solves the continuum wave equation:

$$
\Box \phi = 0
$$

**Convergence Rate**: $\|\phi^{(N)} - \phi\|_{L^2} = O(N^{-1/(d+2)})$

**Related Results:** {prf:ref}`def-discrete-dalembertian`, {prf:ref}`thm-laplacian-convergence`

---

### Quantum Gravity Formulation

#### Quantum Causal Set

**Type:** Definition
**Label:** `def-quantum-causal-set`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 5.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `quantum-gravity`, `path-integral`, `sum-over-causets`

**Statement:**

A **quantum causal set** is defined by the path integral:

$$
Z = \sum_{C} e^{iS[C]} \psi[C]
$$

where:
- Sum is over all causal sets $C$
- $S[C]$ is the causal set action
- $\psi[C]$ is the wavefunction of the causal set

**Causal Set Action**:

$$
S[C] = \sum_{x, y \in C} f(|I(x, y)|)
$$

where $f$ is a function of causal interval cardinality.

**Related Results:** {prf:ref}`def-causal-set`, {prf:ref}`def-benincasa-dowker-action`

---

#### Benincasa-Dowker Action

**Type:** Definition
**Label:** `def-benincasa-dowker-action`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 5.2](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `causet-action`, `einstein-hilbert`, `discrete-curvature`

**Statement:**

The **Benincasa-Dowker action** for causal set $C$ is:

$$
S_{\text{BD}}[C] = \sum_{x \in C} \sum_{k=0}^d c_k N_k(x)
$$

where:
- $N_k(x)$ is the number of $k$-element subsets in a neighborhood of $x$
- $c_k$ are coefficients tuned to recover Einstein-Hilbert action in continuum limit

**Continuum Limit**:

$$
S_{\text{BD}}[C] \to \int_M R \sqrt{-g} \, d^{d+1}x
$$

where $R$ is the Ricci scalar.

**Related Results:** {prf:ref}`def-quantum-causal-set`, {prf:ref}`thm-continuum-limit-lorentzian`

---

#### Sum Over Causal Sets

**Type:** Axiom
**Label:** `axiom-sum-over-causets`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 5.3](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `path-integral`, `quantum-gravity`, `sum-over-histories`

**Statement:**

Quantum gravity is formulated as a **sum over causal sets**:

$$
Z = \sum_{C \text{ causets}} e^{iS[C]}
$$

replacing the sum over Lorentzian manifolds in continuum quantum gravity.

**Measure**: The sum includes all locally finite partial orders with appropriate weighting.

**Related Results:** {prf:ref}`def-quantum-causal-set`, {prf:ref}`def-benincasa-dowker-action`

---

#### Continuum Limit to Lorentzian Manifold

**Type:** Theorem
**Label:** `thm-continuum-limit-lorentzian`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 5.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `continuum-limit`, `lorentzian-manifold`, `general-relativity`, `emergent-geometry`

**Statement:**

For causal set $C$ that is a sprinkling of Lorentzian manifold $(M, g)$:

As $\ell_P \to 0$ (continuum limit), the causal set action converges:

$$
S[C] \to S_{\text{EH}}[g] = \int_M (R - 2\Lambda) \sqrt{-g} \, d^{d+1}x
$$

where $S_{\text{EH}}$ is the Einstein-Hilbert action with cosmological constant $\Lambda$.

**Interpretation**: General relativity emerges from discrete quantum gravity in the continuum limit.

**Related Results:** {prf:ref}`def-benincasa-dowker-action`, {prf:ref}`thm-sprinkling-approximation`, {prf:ref}`axiom-sum-over-causets`

---

## Fermionic Structure and Gauge Theory

### Antisymmetric Cloning Kernel

#### Antisymmetric Cloning Kernel

**Type:** Definition
**Label:** `def-antisymmetric-cloning-kernel`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 1.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `antisymmetric-kernel`, `fermionic`, `cloning`, `exclusion-principle`

**Statement:**

The **antisymmetric cloning kernel** $K_{\text{clone}}^{(-)}$ is defined as:

$$
K_{\text{clone}}^{(-)}(e_i, e_j) = K_{\text{clone}}(e_i, e_j) - K_{\text{clone}}(e_j, e_i)
$$

where $K_{\text{clone}}(e_i, e_j) = \exp(\alpha F_j - \beta H_j)$ is the standard cloning kernel.

**Properties:**
1. **Antisymmetry**: $K^{(-)}(e_i, e_j) = -K^{(-)}(e_j, e_i)$
2. **Vanishing Diagonal**: $K^{(-)}(e_i, e_i) = 0$
3. **Sign Change**: Under particle exchange $(e_i, e_j) \leftrightarrow (e_j, e_i)$, kernel changes sign

**Related Results:** {prf:ref}`thm-algorithmic-exclusion`, {prf:ref}`def-grassmann-field-discrete`

---

#### Directed Edge Orientation

**Type:** Definition
**Label:** `def-directed-edge-orientation`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 1.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `directed-edge`, `orientation`, `antisymmetry`, `coboundary`

**Statement:**

For edge $(e_i, e_j) \in E_{\text{IG}}$, assign **orientation**:
- Forward: $e_i \to e_j$ with weight $+K^{(-)}(e_i, e_j)$
- Backward: $e_j \to e_i$ with weight $-K^{(-)}(e_i, e_j)$

**Edge Function**: For function $f: E_{\text{IG}} \to \mathbb{R}$:

$$
f(e_i, e_j) = -f(e_j, e_i)
$$

**Related Results:** {prf:ref}`def-antisymmetric-cloning-kernel`, {prf:ref}`def-discrete-curl`

---

### Exclusion Principle

#### Algorithmic Exclusion Principle

**Type:** Theorem
**Label:** `thm-algorithmic-exclusion`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 2.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `pauli-exclusion`, `fermi-statistics`, `antisymmetric-kernel`, `no-double-occupancy`

**Statement:**

For antisymmetric cloning kernel $K^{(-)}$:

**Exclusion Principle**: No two episodes can simultaneously occupy the same state:

$$
K^{(-)}(e, e) = 0 \quad \forall e \in \mathcal{E}
$$

**Multi-Particle Extension**: For $n$ episodes $\{e_1, \ldots, e_n\}$:

$$
\det[K^{(-)}(e_i, e_j)]_{i,j=1}^n = 0 \quad \text{if any } e_i = e_j
$$

**Interpretation**: Algorithmic implementation of Pauli exclusion principle for fermions.

**Related Results:** {prf:ref}`def-antisymmetric-cloning-kernel`, {prf:ref}`thm-fermi-dirac-statistics`

---

#### Fermi-Dirac Statistics from Cloning

**Type:** Theorem
**Label:** `thm-fermi-dirac-statistics`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 2.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `fermi-dirac`, `partition-function`, `grand-canonical`, `fermions`

**Statement:**

The episode measure induced by antisymmetric cloning kernel satisfies **Fermi-Dirac statistics**:

$$
\langle n(e) \rangle = \frac{1}{e^{\beta (E(e) - \mu)} + 1}
$$

where:
- $\langle n(e) \rangle$ is the average occupation number
- $E(e)$ is the energy of episode $e$
- $\mu$ is the chemical potential
- $\beta = 1/T$ is the inverse temperature

**Derivation**: From grand canonical ensemble with antisymmetric cloning.

**Related Results:** {prf:ref}`thm-algorithmic-exclusion`, {prf:ref}`def-grassmann-field-discrete`

---

#### Grassmann Field on Fractal Set

**Type:** Definition
**Label:** `def-grassmann-field-discrete`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 2.3](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `grassmann-variable`, `anticommuting`, `fermionic-field`, `path-integral`

**Statement:**

Associate **Grassmann variables** $\psi(e), \bar{\psi}(e)$ to each episode $e \in \mathcal{E}$ with:

**Anticommutation**:

$$
\{\psi(e_i), \psi(e_j)\} = \{\bar{\psi}(e_i), \bar{\psi}(e_j)\} = 0
$$

$$
\{\psi(e_i), \bar{\psi}(e_j)\} = \delta_{ij}
$$

**Partition Function**:

$$
Z = \int \mathcal{D}\psi \mathcal{D}\bar{\psi} \, e^{-S[\psi, \bar{\psi}]}
$$

where $S$ is the fermionic action.

**Related Results:** {prf:ref}`def-antisymmetric-cloning-kernel`, {prf:ref}`thm-fermi-dirac-statistics`, {prf:ref}`def-dirac-propagator-discrete`

---

### Wilson Loops

#### Wilson Loop from Cloning Interactions

**Type:** Theorem
**Label:** `thm-wilson-loop-cloning`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 3.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `wilson-loop`, `gauge-theory`, `cloning-interaction`, `holonomy`

**Statement:**

For closed loop $\gamma = (e_1, e_2, \ldots, e_k, e_1)$ in the IG:

The **Wilson loop** arises from accumulated cloning interactions:

$$
W_\gamma = \prod_{(e_i, e_{i+1}) \in \gamma} \exp\left(\frac{K^{(-)}(e_i, e_{i+1})}{Z}\right)
$$

where $Z$ is a normalization constant.

**Continuum Limit**: As $N \to \infty$:

$$
W_\gamma \to \text{Tr}\left[\mathcal{P} \exp\left(\oint_\gamma A_\mu dx^\mu\right)\right]
$$

**Related Results:** {prf:ref}`def-wilson-loop`, {prf:ref}`def-antisymmetric-cloning-kernel`, {prf:ref}`thm-gauge-invariance-wilson`

---

#### Gauge Invariance of Wilson Loops

**Type:** Theorem
**Label:** `thm-gauge-invariance-wilson`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 3.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `gauge-invariance`, `wilson-loop`, `observable`, `non-abelian`

**Statement:**

Under local gauge transformation $g: \mathcal{E} \to G$:

$$
A(e_i, e_j) \mapsto A^g(e_i, e_j) = g_i A(e_i, e_j) g_j^{-1} + g_i dg_j^{-1}
$$

the Wilson loop trace is **gauge invariant**:

$$
\text{Tr}[W_\gamma[A^g]] = \text{Tr}[W_\gamma[A]]
$$

**Physical Meaning**: Wilson loop is a gauge-invariant observable.

**Related Results:** {prf:ref}`thm-wilson-loop-cloning`, {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`def-wilson-loop`

---

#### Minimal Area Surface

**Type:** Definition
**Label:** `def-minimal-area-surface`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 3.3](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `minimal-surface`, `soap-film`, `variational-problem`, `geometric-area`

**Statement:**

For closed loop $\gamma$ in the IG, the **minimal area surface** $\Sigma(\gamma)$ is the surface spanning $\gamma$ that minimizes:

$$
A[\Sigma] = \sum_{P \subset \Sigma} A_P
$$

where $A_P$ is the area of plaquette $P$.

**Variational Problem**:

$$
\Sigma(\gamma) = \arg\min_{\Sigma: \partial \Sigma = \gamma} A[\Sigma]
$$

**Related Results:** {prf:ref}`thm-geometric-area-ig`, {prf:ref}`thm-wilson-loop-area-law`

---

### Geometric Area

#### Geometric Area Computation

**Type:** Theorem
**Label:** `thm-geometric-area-computation`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 4.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `geometric-area`, `triangulation`, `discrete-surface`, `intrinsic-metric`

**Statement:**

For region $R$ represented by plaquettes $\{P_1, \ldots, P_m\}$ in the IG:

The **geometric area** is:

$$
A_{\text{geom}}(R) = \sum_{i=1}^m A_{P_i}
$$

where for triangular plaquette $P = (e_1, e_2, e_3)$:

$$
A_P = \frac{1}{2} \sqrt{s(s-a)(s-b)(s-c)}
$$

with $a = d_{\text{IG}}(e_1, e_2)$, $b = d_{\text{IG}}(e_2, e_3)$, $c = d_{\text{IG}}(e_3, e_1)$, and $s = (a+b+c)/2$ (Heron's formula).

**Continuum Limit**: $A_{\text{geom}}(R) \to \int_R \sqrt{g} \, d^dx$

**Related Results:** {prf:ref}`thm-geometric-area-ig`, {prf:ref}`def-minimal-area-surface`

---

#### Wilson Loop Area Law

**Type:** Conjecture
**Label:** `conj-wilson-loop-area-law`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 4.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `area-law`, `wilson-loop`, `confinement`, `string-tension`

**Statement:**

For large Wilson loop $W_\gamma$:

$$
\langle W_\gamma \rangle \sim \exp(-\sigma A_{\text{min}}(\gamma))
$$

where:
- $\langle \cdot \rangle$ is ensemble average
- $A_{\text{min}}(\gamma)$ is the minimal area surface spanning $\gamma$
- $\sigma$ is the **string tension**

**Physical Interpretation**: Area law implies confinement of gauge charges (quarks).

**Related Results:** {prf:ref}`def-minimal-area-surface`, {prf:ref}`conj-area-law`, {prf:ref}`thm-wilson-loop-cloning`

---

### Gauge Field Dynamics

#### U(1) Gauge Field on Fractal Set

**Type:** Definition
**Label:** `def-u1-gauge-field`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 5.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `u1-gauge`, `electromagnetism`, `gauge-connection`, `abelian`

**Statement:**

For $U(1)$ gauge group (electromagnetism), the gauge connection is:

$$
A: E_{\text{IG}} \to \mathbb{R}
$$

(valued in $\mathfrak{u}(1) \cong i\mathbb{R}$).

**Parallel Transport**:

$$
U_{ij} = e^{iA(e_i, e_j)} \in U(1)
$$

**Field Strength**:

$$
F(P) = \sum_{(e_i, e_j) \in \partial P} A(e_i, e_j) \pmod{2\pi}
$$

**Related Results:** {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`thm-u1-wilson-action`, {prf:ref}`def-electromagnetic-field-discrete`

---

#### SU(N) Gauge Field on Fractal Set

**Type:** Definition
**Label:** `def-sun-gauge-field`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 5.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `sun-gauge`, `non-abelian`, `yang-mills`, `qcd`

**Statement:**

For $SU(N)$ gauge group (e.g., $SU(3)$ for QCD), the gauge connection is:

$$
A: E_{\text{IG}} \to \mathfrak{su}(N)
$$

(valued in the Lie algebra of $SU(N)$).

**Parallel Transport**:

$$
U_{ij} = \exp(A(e_i, e_j)) \in SU(N)
$$

**Field Strength**:

$$
F(P) = A(e_1, e_2) + A(e_2, e_3) + A(e_3, e_4) + A(e_4, e_1) + [A(e_1, e_2), A(e_2, e_3)] + \ldots
$$

(includes commutator terms for non-abelian case).

**Related Results:** {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`thm-sun-wilson-action`, {prf:ref}`def-qcd-fractal-set`

---

## Lattice QFT on Causal Sets

### Lattice Gauge Theory

#### Lattice Gauge Action

**Type:** Theorem
**Label:** `thm-lattice-gauge-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 1.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `lattice-gauge`, `wilson-action`, `gauge-action`, `discretization`

**Statement:**

The **lattice gauge action** on the Fractal Set for gauge group $G$ is:

$$
S_{\text{gauge}}[A] = -\frac{1}{g^2} \sum_{P} \text{Re}\left[\text{Tr}(U_P)\right]
$$

where:
- Sum is over all plaquettes $P$ in the IG
- $U_P = U_{12} U_{23} U_{34} U_{41}$ is the plaquette holonomy
- $g$ is the gauge coupling constant

**Continuum Limit**: As $\epsilon \to 0$ (lattice spacing):

$$
S_{\text{gauge}}[A] \to -\frac{1}{4g^2} \int F_{\mu\nu} F^{\mu\nu} \sqrt{-g} \, d^{d+1}x
$$

(Yang-Mills action).

**Related Results:** {prf:ref}`def-discrete-gauge-connection`, {prf:ref}`def-discrete-curvature`, {prf:ref}`thm-sun-wilson-action`

---

#### Gauge Field Path Integral

**Type:** Definition
**Label:** `def-gauge-path-integral`
**Source:** [13_E_cst_ig_lattice_qft.md § 1.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `path-integral`, `gauge-theory`, `partition-function`, `wilson-action`

**Statement:**

The **gauge field path integral** on the Fractal Set is:

$$
Z = \int \mathcal{D}A \, e^{-S_{\text{gauge}}[A]}
$$

where the measure $\mathcal{D}A$ integrates over all gauge connections modulo gauge equivalence.

**Gauge Fixing**: Use Faddeev-Popov procedure to fix gauge:

$$
Z = \int \mathcal{D}A \, \det(\Delta_{\text{FP}}) \, \delta(G[A]) \, e^{-S_{\text{gauge}}[A]}
$$

where $G[A] = 0$ is the gauge-fixing condition and $\Delta_{\text{FP}}$ is the Faddeev-Popov operator.

**Related Results:** {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`def-faddeev-popov-determinant`

---

#### Plaquette Average

**Type:** Proposition
**Label:** `prop-plaquette-average`
**Source:** [13_E_cst_ig_lattice_qft.md § 1.3](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `plaquette`, `wilson-action`, `strong-coupling`, `weak-coupling`

**Statement:**

The plaquette expectation value:

$$
\langle U_P \rangle = \int \mathcal{D}A \, U_P \, e^{-S_{\text{gauge}}[A]}
$$

exhibits two regimes:
- **Strong Coupling** ($g \to \infty$): $\langle U_P \rangle \to 0$ (disordered)
- **Weak Coupling** ($g \to 0$): $\langle U_P \rangle \to 1$ (ordered)

**Phase Transition**: For compact gauge groups (e.g., $U(1)$, $SU(N)$), there is a phase transition at critical coupling $g_c$.

**Related Results:** {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`thm-phase-transition-lattice`

---

### U(1) Gauge Fields

#### U(1) Wilson Action

**Type:** Theorem
**Label:** `thm-u1-wilson-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 2.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `u1-gauge`, `wilson-action`, `electromagnetism`, `compact-qed`

**Statement:**

For $U(1)$ gauge theory (compact QED) on the Fractal Set:

$$
S_{U(1)}[A] = -\frac{\beta}{2} \sum_P \cos(F(P))
$$

where:
- $\beta = 1/g^2$ is the inverse gauge coupling
- $F(P) = \sum_{(e_i, e_j) \in \partial P} A(e_i, e_j)$ is the plaquette field strength

**Continuum Limit**:

$$
S_{U(1)}[A] \to -\frac{1}{4g^2} \int F_{\mu\nu} F^{\mu\nu} \, d^{d+1}x
$$

**Related Results:** {prf:ref}`def-u1-gauge-field`, {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`thm-u1-phase-transition`

---

#### U(1) Phase Transition

**Type:** Theorem
**Label:** `thm-u1-phase-transition`
**Source:** [13_E_cst_ig_lattice_qft.md § 2.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `phase-transition`, `compact-qed`, `confinement`, `coulomb-phase`

**Statement:**

Compact $U(1)$ gauge theory on the Fractal Set exhibits a **phase transition** at critical $\beta_c$:

- **$\beta < \beta_c$ (Strong Coupling)**: **Confinement phase** - electric charges are confined, Wilson loops obey area law
- **$\beta > \beta_c$ (Weak Coupling)**: **Coulomb phase** - electric charges are deconfined, Wilson loops obey perimeter law

**Critical Exponents**: Near $\beta_c$, observables exhibit power-law scaling.

**Related Results:** {prf:ref}`thm-u1-wilson-action`, {prf:ref}`conj-wilson-loop-area-law`, {prf:ref}`prop-plaquette-average`

---

#### Electromagnetic Field on Fractal Set

**Type:** Definition
**Label:** `def-electromagnetic-field-discrete`
**Source:** [13_E_cst_ig_lattice_qft.md § 2.3](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `electromagnetism`, `electric-field`, `magnetic-field`, `maxwell-equations`

**Statement:**

For $U(1)$ gauge connection $A$ on the Fractal Set:

**Electric Field**: For timelike edge $(e_i, e_j) \in E_{\text{CST}}$:

$$
E(e_i, e_j) = A(e_i, e_j)
$$

**Magnetic Field**: For spacelike plaquette $P$ in the IG:

$$
B(P) = F(P) = \sum_{(e_i, e_j) \in \partial P} A(e_i, e_j)
$$

**Maxwell's Equations** (discrete form):

$$
\text{div}_{\mathcal{F}} E = \rho, \quad \text{curl}_{\mathcal{F}} E = -\frac{\partial B}{\partial t}, \quad \text{div}_{\mathcal{F}} B = 0, \quad \text{curl}_{\mathcal{F}} B = J + \frac{\partial E}{\partial t}
$$

**Related Results:** {prf:ref}`def-u1-gauge-field`, {prf:ref}`def-discrete-curl`, {prf:ref}`thm-u1-wilson-action`

---

### SU(N) Gauge Fields

#### SU(N) Wilson Action

**Type:** Theorem
**Label:** `thm-sun-wilson-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 3.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `sun-gauge`, `wilson-action`, `yang-mills`, `non-abelian`

**Statement:**

For $SU(N)$ gauge theory on the Fractal Set:

$$
S_{SU(N)}[A] = -\frac{\beta}{2N} \sum_P \text{Re}\left[\text{Tr}(U_P)\right]
$$

where:
- $\beta = 2N/g^2$ is the inverse gauge coupling
- $U_P = \exp(A(e_1, e_2)) \exp(A(e_2, e_3)) \exp(A(e_3, e_4)) \exp(A(e_4, e_1))$ is the plaquette holonomy

**Continuum Limit**:

$$
S_{SU(N)}[A] \to -\frac{1}{2g^2} \int \text{Tr}(F_{\mu\nu} F^{\mu\nu}) \, d^{d+1}x
$$

(Yang-Mills action for $SU(N)$).

**Related Results:** {prf:ref}`def-sun-gauge-field`, {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`def-qcd-fractal-set`

---

#### Asymptotic Freedom

**Type:** Theorem
**Label:** `thm-asymptotic-freedom`
**Source:** [13_E_cst_ig_lattice_qft.md § 3.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `asymptotic-freedom`, `renormalization-group`, `beta-function`, `qcd`

**Statement:**

For $SU(N)$ gauge theory with $N \geq 2$:

The **beta function** is:

$$
\beta(g) = \frac{dg}{d\ln\mu} = -\frac{b_0 g^3}{(4\pi)^2} + O(g^5)
$$

where $b_0 = \frac{11N - 2n_f}{3}$ for $n_f$ fermion flavors.

**Asymptotic Freedom**: For $n_f < \frac{11N}{2}$:
- $\beta(g) < 0$ (gauge coupling decreases at high energy)
- $g(\mu) \to 0$ as $\mu \to \infty$

**Physical Interpretation**: QCD becomes weakly coupled at high energy scales (justifies perturbative QCD).

**Related Results:** {prf:ref}`thm-sun-wilson-action`, {prf:ref}`conj-mass-gap`, {prf:ref}`def-qcd-fractal-set`

---

#### Confinement in SU(N)

**Type:** Conjecture
**Label:** `conj-sun-confinement`
**Source:** [13_E_cst_ig_lattice_qft.md § 3.3](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `confinement`, `sun-gauge`, `wilson-loop`, `area-law`, `qcd`

**Statement:**

For $SU(N)$ gauge theory on the Fractal Set in the strong coupling regime:

**Confinement Hypothesis**: Color-charged particles (quarks, gluons) cannot be isolated - they are permanently confined within color-neutral hadrons.

**Wilson Loop Area Law**:

$$
\langle W_\gamma \rangle \sim \exp(-\sigma A_{\text{min}}(\gamma))
$$

for large loops $\gamma$, where $\sigma$ is the string tension.

**Related Results:** {prf:ref}`conj-wilson-loop-area-law`, {prf:ref}`thm-asymptotic-freedom`, {prf:ref}`conj-mass-gap`

---

### Wilson Action

#### Wilson Plaquette Action

**Type:** Definition
**Label:** `def-wilson-plaquette-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 4.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `wilson-action`, `plaquette`, `lattice-action`, `discretization`

**Statement:**

The **Wilson plaquette action** for gauge group $G$ is:

$$
S_W[U] = \beta \sum_P \left(1 - \frac{1}{N} \text{Re}\left[\text{Tr}(U_P)\right]\right)
$$

where:
- $U_P$ is the ordered product of link variables around plaquette $P$
- $\beta = 2N/g^2$ for $SU(N)$ or $\beta = 1/g^2$ for $U(1)$

**Motivation**: Simplest gauge-invariant discretization of Yang-Mills action.

**Related Results:** {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`thm-sun-wilson-action`, {prf:ref}`thm-u1-wilson-action`

---

#### Improved Actions

**Type:** Definition
**Label:** `def-improved-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 4.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `improved-action`, `symanzik`, `discretization-errors`, `lattice-artifacts`

**Statement:**

**Improved actions** reduce lattice artifacts by including larger Wilson loops:

$$
S_{\text{imp}}[U] = \beta \sum_P c_P W_P
$$

where:
- $W_P$ are Wilson loops of various sizes (plaquettes, rectangles, etc.)
- $c_P$ are coefficients tuned to cancel $O(a^2)$ discretization errors

**Symanzik Improvement**: Choose $c_P$ such that:

$$
S_{\text{imp}}[U] = S_{\text{cont}}[A] + O(a^4)
$$

where $a$ is the lattice spacing.

**Related Results:** {prf:ref}`def-wilson-plaquette-action`, {prf:ref}`thm-lattice-gauge-action`

---

### QCD on Fractal Sets

#### QCD Action on Fractal Set

**Type:** Definition
**Label:** `def-qcd-fractal-set`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `qcd`, `lattice-qcd`, `quarks`, `gluons`, `chromodynamics`

**Statement:**

The **QCD action** on the Fractal Set with $SU(3)$ gauge group is:

$$
S_{\text{QCD}} = S_{\text{gauge}}[U] + S_{\text{fermion}}[\psi, \bar{\psi}, U]
$$

where:

**Gauge Action**:

$$
S_{\text{gauge}}[U] = -\frac{\beta}{6} \sum_P \text{Re}\left[\text{Tr}(U_P)\right]
$$

**Fermion Action** (Wilson fermions):

$$
S_{\text{fermion}}[\psi, \bar{\psi}, U] = \sum_{e \in \mathcal{E}} \bar{\psi}(e) \psi(e) + \kappa \sum_{(e_i, e_j) \in E_{\text{IG}}} \bar{\psi}(e_i) U_{ij} \psi(e_j)
$$

where $\kappa$ is the hopping parameter.

**Related Results:** {prf:ref}`thm-sun-wilson-action`, {prf:ref}`def-grassmann-field-discrete`, {prf:ref}`def-sun-gauge-field`

---

#### Quark Propagator

**Type:** Definition
**Label:** `def-quark-propagator`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `quark-propagator`, `dirac-operator`, `fermion-propagator`, `green-function`

**Statement:**

The **quark propagator** $S_F(e_i, e_j)$ on the Fractal Set is the inverse of the Dirac operator:

$$
(D \cdot S_F)(e_i, e_j) = \delta_{ij}
$$

where the **lattice Dirac operator** is:

$$
(D\psi)(e_i) = \psi(e_i) - \kappa \sum_{e_j \sim e_i} U_{ij} \psi(e_j)
$$

**Continuum Limit**:

$$
S_F(e_i, e_j) \to \langle 0 | T\{\psi(x_i) \bar{\psi}(x_j)\} | 0 \rangle
$$

(Feynman propagator for Dirac fermion).

**Related Results:** {prf:ref}`def-qcd-fractal-set`, {prf:ref}`def-grassmann-field-discrete`, {prf:ref}`def-feynman-propagator-discrete`

---

#### Hadron Mass Computation

**Type:** Proposition
**Label:** `prop-hadron-mass`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.3](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `hadron-mass`, `correlation-function`, `lattice-qcd`, `meson`, `baryon`

**Statement:**

Hadron masses are extracted from **correlation functions**:

For meson (quark-antiquark bound state):

$$
C_M(t) = \sum_{\vec{x}} \langle 0 | \mathcal{O}_M(\vec{x}, t) \mathcal{O}_M^\dagger(0, 0) | 0 \rangle
$$

where $\mathcal{O}_M = \bar{\psi} \Gamma \psi$ is the meson interpolating operator.

**Mass Extraction**: At large $t$:

$$
C_M(t) \sim e^{-m_M t}
$$

where $m_M$ is the meson mass.

**Related Results:** {prf:ref}`def-qcd-fractal-set`, {prf:ref}`def-quark-propagator`, {prf:ref}`conj-mass-gap`

---

#### Mass Gap Conjecture

**Type:** Conjecture
**Label:** `conj-mass-gap`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.4](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `mass-gap`, `millennium-prize`, `yang-mills`, `qcd`, `confinement`

**Statement:**

For pure $SU(N)$ Yang-Mills theory in $(d+1)$-dimensional spacetime with $d \geq 3$:

**Mass Gap Conjecture**: There exists a constant $\Delta > 0$ (the mass gap) such that:

$$
E_n - E_0 \geq \Delta \quad \forall n \geq 1
$$

where $E_n$ are the energy eigenvalues of the Hamiltonian.

**Physical Interpretation**:
- All excitations have mass $\geq \Delta$ (no massless gluons in confinement phase)
- Resolving this conjecture is a Millennium Prize Problem

**Related Results:** {prf:ref}`conj-sun-confinement`, {prf:ref}`thm-asymptotic-freedom`, {prf:ref}`prop-hadron-mass`

---

### Computational Algorithms

#### Monte Carlo Simulation

**Type:** Algorithm
**Label:** `algo-monte-carlo`
**Source:** [13_E_cst_ig_lattice_qft.md § 6.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `monte-carlo`, `importance-sampling`, `metropolis`, `simulation`

**Statement:**

**Lattice Gauge Theory Monte Carlo** on the Fractal Set:

1. **Initialize**: Random gauge configuration $\{U_{ij}\}$
2. **Update**: For each edge $(e_i, e_j) \in E_{\text{IG}}$:
   - Propose new link variable $U'_{ij}$
   - Compute acceptance probability $p = \min(1, e^{-\Delta S})$ where $\Delta S = S[U'] - S[U]$
   - Accept with probability $p$
3. **Measure**: After thermalization, compute observables
4. **Repeat**: Steps 2-3 for many iterations

**Algorithms**: Metropolis, heat bath, hybrid Monte Carlo (HMC), etc.

**Related Results:** {prf:ref}`thm-lattice-gauge-action`, {prf:ref}`def-gauge-path-integral`

---

#### Hybrid Monte Carlo (HMC)

**Type:** Algorithm
**Label:** `algo-hmc`
**Source:** [13_E_cst_ig_lattice_qft.md § 6.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `hmc`, `molecular-dynamics`, `hamiltonian-monte-carlo`, `fermions`

**Statement:**

**Hybrid Monte Carlo** for gauge + fermion systems on the Fractal Set:

1. **Introduce Conjugate Momenta**: $\pi_{ij}$ for each link $U_{ij}$
2. **Hamiltonian**:

$$
H = \frac{1}{2}\sum_{(e_i, e_j)} \text{Tr}(\pi_{ij}^2) + S_{\text{gauge}}[U] + S_{\text{fermion}}[\psi, \bar{\psi}, U]
$$

3. **Molecular Dynamics**: Evolve $(U, \pi)$ using Hamilton's equations for trajectory length $\tau$
4. **Metropolis Accept/Reject**: Accept with probability $\min(1, e^{-\Delta H})$
5. **Repeat**

**Advantage**: Efficient for systems with fermions (avoids expensive fermion matrix inversions at each step).

**Related Results:** {prf:ref}`algo-monte-carlo`, {prf:ref}`def-qcd-fractal-set`

---

#### Gauge Fixing Algorithms

**Type:** Algorithm
**Label:** `algo-gauge-fixing`
**Source:** [13_E_cst_ig_lattice_qft.md § 6.3](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `gauge-fixing`, `landau-gauge`, `coulomb-gauge`, `faddeev-popov`

**Statement:**

**Gauge Fixing** on the Fractal Set to compute gauge-dependent quantities:

**Landau Gauge**: Minimize

$$
F_{\text{Landau}}[U^g] = \sum_{(e_i, e_j) \in E_{\text{IG}}} \text{Re}\left[\text{Tr}(U_{ij}^g)\right]
$$

over gauge transformations $g: \mathcal{E} \to G$.

**Coulomb Gauge**: Minimize

$$
F_{\text{Coulomb}}[U^g] = \sum_{(e_i, e_j) \in E_{\text{IG}}^{\text{spatial}}} \text{Re}\left[\text{Tr}(U_{ij}^g)\right]
$$

**Algorithm**: Iterative gauge transformation to maximize $F$.

**Related Results:** {prf:ref}`def-gauge-path-integral`, {prf:ref}`def-discrete-gauge-connection`

---

#### Fermion Matrix Inversion

**Type:** Algorithm
**Label:** `algo-fermion-inversion`
**Source:** [13_E_cst_ig_lattice_qft.md § 6.4](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `fermion-matrix`, `conjugate-gradient`, `krylov-methods`, `sparse-linear-system`

**Statement:**

**Fermion Matrix Inversion** for computing quark propagator $S_F = D^{-1}$:

**Problem**: Solve $D \cdot x = b$ for sparse matrix $D$ (lattice Dirac operator).

**Algorithms**:
1. **Conjugate Gradient (CG)**: For positive definite $D^\dagger D$
2. **BiCGStab**: For non-Hermitian $D$
3. **Multigrid Methods**: Hierarchical coarsening for faster convergence

**Complexity**: $O(V \cdot \log(1/\epsilon))$ where $V$ is lattice volume, $\epsilon$ is tolerance.

**Related Results:** {prf:ref}`def-quark-propagator`, {prf:ref}`def-qcd-fractal-set`, {prf:ref}`algo-hmc`

---

## Document Status

**Total Mathematical Objects**: 184
- Definitions: 47
- Theorems: 32
- Propositions: 24
- Axioms: 5
- Conjectures: 8
- Algorithms: 4

**Included Documents:**
-  [13_A_fractal_set.md](13_fractal_set/13_A_fractal_set.md) - Episodes, CST, IG, Fractal Set foundations, discrete geometry
-  [13_B_fractal_set_continuum_limit.md](13_fractal_set/13_B_fractal_set_continuum_limit.md) - Discrete symmetries, graph Laplacian convergence, episode measure convergence, holonomy
-  [13_C_cst_causal_set_quantum_gravity.md](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md) - Causal set axioms, sprinkling, Lorentzian structure, discrete d'Alembertian, quantum gravity
-  [13_D_fractal_set_emergent_qft_comprehensive.md](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md) - Antisymmetric cloning kernel, fermionic exclusion, Wilson loops, geometric area, gauge theory
-  [13_E_cst_ig_lattice_qft.md](13_fractal_set/13_E_cst_ig_lattice_qft.md) - Lattice gauge theory, U(1)/SU(N) gauge fields, Wilson action, QCD, computational algorithms

**Document Version**: 1.0
**Last Updated**: 2025-10-10
**Maintained by**: Fragile Gas Framework Team
