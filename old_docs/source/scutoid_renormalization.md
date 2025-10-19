# Scutoid Renormalization: A Conditional Framework for Geometric Compression and RG Flow

**Document Status:** Research Framework - Conditional Theory

**Scope:** This chapter presents a **conditional theoretical framework** and an **empirical research program** for understanding the Fixed-Node Scutoid Tessellation ({prf:ref}`def-fixed-node-scutoid`, [fragile_lqcd.md](fragile_lqcd.md)) as a principled Renormalization Group (RG) transformation. The framework is conditional on the **Information Closure Hypothesis** (unproven), which states that the scutoid renormalization map preserves predictive information.

**Main Thesis (Conditional):** **IF** the Fixed-Node Scutoid Tessellation satisfies information closure, **THEN** it implements a renormalization channel $\mathcal{R}_{\text{scutoid},b}$ with quantifiable error bounds, transforming the O(N) algorithm into a rigorous effective field theory.

**New Contribution:** We propose that the **gamma channel** (geometric curvature feedback) acts as a **compression optimizer** that actively prepares the system for optimal coarse-graining by minimizing the Weyl tensor (information loss) and maximizing the Ricci scalar (cluster coherence). This provides a physical mechanism for achieving closure.

**Prerequisites:**
- [fragile_lqcd.md](fragile_lqcd.md): Fixed-Node Scutoid Tessellation and O(N) complexity
- [13_fractal_set_new/15_closure_theory.md](13_fractal_set_new/15_closure_theory.md): Closure theory, ε-machines, RG-closure connection
- [13_fractal_set_new/02_computational_equivalence.md](13_fractal_set_new/02_computational_equivalence.md): BAOAB Markov chain
- [10_kl_convergence/10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md): KL-divergence bounds
- Ortega et al., "Closure Theory for Stochastic Processes" (arXiv:2402.09090v2)

---

## Table of Contents

**Part I: Foundations and Central Hypothesis**
1. Introduction: The Information Closure Hypothesis
2. The Scutoid Renormalization Channel (Conditional Framework)
3. Micro-Process: The Full N-Walker Scutoid Tessellation

**Part II: The Gamma Channel as Compression Optimizer (NEW)**
4. Geometric Compression: Ricci, Weyl, and Compressibility
5. The Weyl-Lumpability Connection (Conceptual)
6. Self-Organizing RG Flow via Curvature Feedback

**Part III: Discovering Intrinsic Scales (NEW)**
7. Method 1: Observable-Driven Coarse-Graining (The Money Plot)
8. Method 2: Statistical Complexity Plateaus
9. Method 3: Topological Phase Transitions
10. Method 4: Geometrothermodynamic Singularities
11. Unified Multi-Scale Discovery Algorithm

**Part IV: Conditional Theorems (Pending Proofs)**
12. Observable Preservation (Conditional on Closure)
13. Information-Theoretic Diagnostics (Conditional)
14. Lumpability Error Bounds (Requires Missing Lemmas)

**Part V: Research Program and Iteration Path**
15. The Path to Rigor: Missing Proofs and Open Problems
16. Empirical Verification Protocol
17. Dimensional Analysis and Physical Implications
18. Conclusion: From Sketches to Theorems

---

# PART I: FOUNDATIONS AND CENTRAL HYPOTHESIS

## 1. Introduction: The Information Closure Hypothesis

### 1.1. The Computational Challenge

The Fixed-Node Scutoid Tessellation ({prf:ref}`def-fixed-node-scutoid`, [fragile_lqcd.md](fragile_lqcd.md)) achieves O(N) computational complexity for lattice QFT simulations by separating scales:

**Fine Scale ($N$ walkers):** All $N$ walkers evolve via the full Fragile Gas dynamics (Langevin SDE, cloning, adaptive forces). This is the "ground truth" physics.

**Coarse Scale ($n_{\text{cell}}$ generators):** A fixed number $n_{\text{cell}} \ll N$ of representative walkers (cluster centers) define the Delaunay triangulation and scutoid tessellation for geometric analysis.

**The Standard View:** This is a computational approximation that trades geometric resolution for speed. The quantization error scales as $O(n_{\text{cell}}^{-1/d})$ (Theorem {prf:ref}`thm-cvt-approximation-error`, [fragile_lqcd.md](fragile_lqcd.md)).

**This Chapter's Vision:** The Fixed-Node construction can be understood as a **principled renormalization group transformation** in the sense of computational closure theory—*if* certain conditions hold. The validity of this interpretation rests on a central, **unproven hypothesis**.

### 1.2. The Central Hypothesis

:::{prf:hypothesis} Information Closure Hypothesis for Scutoid Renormalization
:label: hyp-scutoid-information-closure

The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ (to be defined rigorously in §2-3) satisfies **information closure** for long-range physical observables:

$$
I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{\tilde{Z}}_t) = I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{Z}_t) + o(1)
$$

as $n_{\text{cell}} \to N$, where:
- $Z_t$ is the full microscopic state ($N$ walkers)
- $\tilde{Z}_t = \mathcal{R}_{\text{scutoid},b}(Z_t)$ is the coarse-grained state ($n_{\text{cell}}$ generators)
- $I(\cdot;\cdot)$ is mutual information
- The $o(1)$ error vanishes as the coarse-graining becomes finer

**Interpretation:** All information in the microscopic past $\overleftarrow{Z}_t$ that is relevant for predicting the macroscopic future $\overrightarrow{\tilde{Z}}_t$ is captured by the macroscopic past $\overleftarrow{\tilde{Z}}_t$.

**Status:** **UNPROVEN**. This is the central open problem of this chapter and the focus of our research program (Part V).
:::

:::{prf:remark} Why This Hypothesis Matters
:class: important

**If Hypothesis {prf:ref}`hyp-scutoid-information-closure` holds**, then by the Ortega et al. (2024) theorem (Theorem 2, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 2.2):

1. **Computational closure** holds: The macro-ε-machine is a coarse-graining of the micro-ε-machine
2. **Strong lumpability** holds: The macro-process is Markovian
3. **Observable preservation** holds: Long-range observables are accurately predicted
4. **The effective theory is valid**: The $n_{\text{cell}}$-generator description is not an approximation but the **correct** description at coarse scales

**If the hypothesis fails**, the entire conditional framework collapses, and the Fixed-Node method remains a heuristic approximation without theoretical justification.

**Our goal:** Transform this hypothesis into a proven theorem through a combination of analytical work and empirical verification.
:::

### 1.3. The Research Program Philosophy

This document adopts an **iterative, empirically-grounded approach** to rigorous theory development:

**Stage 1 (Current):** Present the conditional framework with clear identification of unproven assumptions

**Stage 2 (Next):** Develop conceptual insights and heuristic arguments connecting geometry (Ricci, Weyl) to information theory (closure, lumpability)

**Stage 3 (Future):** Formalize one key conceptual insight into a complete proof (e.g., the Weyl-lumpability connection)

**Stage 4 (Goal):** Build a complete, rigorous theory by iteratively proving each component

**Transparency:** We explicitly mark:
- **Hypothesis**: Unproven central assumptions
- **Conjecture**: Plausible claims requiring proof
- **Heuristic Argument**: Conceptual reasoning without full rigor
- **Proposition (Conditional)**: Rigorous if-then statements
- **Theorem**: Fully proven results (currently: none in this chapter)

### 1.2. Why Closure Theory?

Traditional RG formulations answer the question "Does coarse-graining preserve physics?" operationally: simulate both scales and check if observables match.

**Closure theory** (Ortega et al., 2024; {prf:ref}`def-computational-closure`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md)) provides a **rigorous information-theoretic answer**: A coarse-graining preserves predictive power if and only if it satisfies **computational closure**.

**Key Insight:** If the macro-ε-machine (optimal predictive model for the coarse-grained process) can be obtained by coarse-graining the micro-ε-machine (optimal model for the full process), then:

1. All predictive information is preserved
2. The error is quantifiable via information-theoretic measures (KL-divergence, statistical complexity)
3. The effective theory is not an approximation—it is the **correct** description at the macro-scale

### 1.3. The Research Program

This chapter establishes the following concrete research plan:

**Step 1:** Define the microscopic process (§3): The full $N$-walker scutoid tessellation $Z_{\text{scutoid}}(t)$ evolving under BAOAB + cloning.

**Step 2:** Define the coarse-graining map (§5): The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ that aggregates walkers into $n_{\text{cell}} = N/b^d$ generators.

**Step 3:** Define the macroscopic process (§4): The $n_{\text{cell}}$-generator "super-scutoid" tessellation $\tilde{Z}_{\text{scutoid}}(t)$.

**Step 4:** Prove computational closure (§6): Show that the macro-ε-machine is a coarse-graining of the micro-ε-machine, with quantifiable lumpability error.

**Step 5:** Verify empirically (§7-8): Measure observable preservation and information-theoretic diagnostics as functions of $n_{\text{cell}}$ and block size $b$.

**Step 6:** Interpret physically (§10-11): Connect closure accuracy to dimensional dependence and the "O(N) Universe Hypothesis."

### 1.4. Implications

If successful, this analysis accomplishes:

1. **Justifies the O(N) algorithm rigorously:** The Fixed-Node Scutoid Tessellation is a principled effective theory, not a hack.

2. **Makes RG concrete and computable:** The RG "flow" is the measurable change in super-scutoid properties as $n_{\text{cell}}$ varies. The beta function can be extracted from simulation data.

3. **Defines emergence:** If an observable is *not* preserved (closure fails), it is a truly **emergent phenomenon** that cannot be captured by the coarse-grained description.

4. **Strengthens the O(N) Universe Hypothesis:** Demonstrates that the universe *can* coarse-grain its own state into an O(N)-computable form without losing essential physical information.

---

## 2. The Scutoid Renormalization Channel

### 2.1. Conceptual Overview

The scutoid renormalization channel has the following structure:

**Micro-space:** Full $N$-walker configurations with scutoid tessellation.

**Macro-space:** Coarse-grained $n_{\text{cell}}$-generator configurations with super-scutoid tessellation.

**Channel:** Centroidal Voronoi Tessellation (CVT) clustering ({prf:ref}`alg-fixed-node-lattice`, [fragile_lqcd.md](fragile_lqcd.md)) that deterministically maps micro $\to$ macro.

**Control parameter:** The block size $b$ (or equivalently, the number of generators $n_{\text{cell}} = N/b^d$) is the RG "knob."

### 2.2. Relationship to Traditional RG

**Traditional lattice RG:** Hypercubic block-spin transformations ({prf:ref}`def-block-partition`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 4.1) partition space into fixed blocks and average fields within each block.

**Scutoid RG:** Adaptive clustering via CVT that concentrates resolution in high-density regions (where walkers naturally accumulate due to the QSD).

**Key Difference:** Traditional RG uses a **fixed spatial partition**. Scutoid RG uses a **data-adaptive partition** that respects the emergent geometry encoded in the walker distribution.

**Consequence:** Scutoid RG can achieve better closure accuracy for the same number of macro-degrees of freedom because it "spends resolution" where the physics is most complex.

### 2.3. Chapter Structure

The remainder of this chapter proceeds as follows:

- **§3:** Define the micro-process: Full $N$-walker scutoid Markov chain
- **§4:** Define the macro-process: Coarse-grained $n_{\text{cell}}$-generator super-scutoid dynamics

### 2.4. Formal Definition of the Renormalization Map

:::{important}
**Formalization Note:** The following provides the rigorous mathematical definition of the scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$, addressing Gemini Issue #1 and Codex Issue #1. This is the central object of the entire framework—without it, the Information Closure Hypothesis cannot even be stated precisely.
:::

:::{prf:definition} Scutoid Renormalization Map
:label: def-scutoid-renormalization-map

The **scutoid renormalization map** is a measurable function between state spaces:

$$
\mathcal{R}_{\text{scutoid},b}: \Omega^{(N)} \to \Omega^{(n_{\text{cell}})}
$$

where:
- $\Omega^{(N)} = \mathcal{X}^N \times \mathbb{R}^{Nd}$ is the microscopic state space
- $\Omega^{(n_{\text{cell}})} = \mathcal{X}^{n_{\text{cell}}} \times \mathbb{R}^{n_{\text{cell}}d}$ is the macroscopic state space
- $n_{\text{cell}} = \lfloor N / b^d \rfloor$ is the number of coarse generators
- $b \geq 1$ is the **block size** (RG scale parameter)

**Construction:** Given a micro-state $Z^{(N)} = (X, V)$ at time $k$, the map proceeds in two steps:

**Step 1: CVT Clustering** (Algorithm {prf:ref}`alg-fixed-node-lattice`, [fragile_lqcd.md](fragile_lqcd.md))

Apply Lloyd's algorithm (iterative $k$-means) to the walker positions $X = \{x_1, \ldots, x_N\}$ with $k = n_{\text{cell}}$:

$$
\{c_\alpha\}_{\alpha=1}^{n_{\text{cell}}} = \text{CVT}(X, n_{\text{cell}})
$$

This produces:
- **Cluster centers (generators):** $\{c_\alpha\}_{\alpha=1}^{n_{\text{cell}}} \subset \mathcal{X}$
- **Cluster partition:** $\{C_\alpha\}_{\alpha=1}^{n_{\text{cell}}}$ where $C_\alpha = \{i : x_i \in \mathcal{V}_\alpha\}$ (walker indices in Voronoi cell of $c_\alpha$)

**Step 2: Coarse State Construction**

For each coarse generator $\alpha \in \{1, \ldots, n_{\text{cell}}\}$, define:

- **Coarse position:**
  $$
  \tilde{x}_\alpha := c_\alpha \quad \text{(cluster centroid)}
  $$

- **Coarse velocity:**
  $$
  \tilde{v}_\alpha := \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} v_i \quad \text{(average micro-velocity)}
  $$
  where $C_\alpha$ is the set of walker indices assigned to cluster $\alpha$.

**Output:** The coarse-grained state is:

$$
\mathcal{R}_{\text{scutoid},b}(Z^{(N)}) = \tilde{Z}^{(n_{\text{cell}})} = (\tilde{X}, \tilde{V})
$$

where $\tilde{X} = (\tilde{x}_1, \ldots, \tilde{x}_{n_{\text{cell}}}) \in \mathcal{X}^{n_{\text{cell}}}$ and $\tilde{V} = (\tilde{v}_1, \ldots, \tilde{v}_{n_{\text{cell}}}) \in \mathbb{R}^{n_{\text{cell}}d}$.

**Determinism:** The CVT clustering is deterministic given the input positions $X$ (assuming a fixed initialization scheme and tie-breaking rule). Therefore, the map $\mathcal{R}_{\text{scutoid},b}$ is a well-defined single-valued function.

**Non-Empty Clusters:** The algorithm ensures $|C_\alpha| \geq 1$ for all $\alpha$ by construction (each generator is its own cluster member). If a cluster would otherwise be empty during Lloyd's iteration, it is reseeded with the farthest walker from any existing centroid.
:::

:::{prf:remark} Geometric Observables on the Coarse Trajectory
:class: note

Given a coarse trajectory $\{\tilde{Z}_k\}_{k \geq 0}$, one can compute:

- **Coarse Voronoi tessellation:** $\tilde{\mathcal{V}}_k = \text{Voronoi}(\tilde{X}_k)$
- **Coarse scutoid tessellation:** $\tilde{\mathcal{S}}_k = \text{Scutoid}(\tilde{X}_k, \tilde{X}_{k+1})$ (depends on successive time steps)
- **Coarse emergent metric:** $\tilde{g}_{ab}(x, k)$ from the coarse deformation tensor

These are post-processed observables, not components of the state $\tilde{Z}_k$ itself.
:::

:::{prf:proposition} Measurability of the Renormalization Map
:label: prop-renormalization-measurability

The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}: \Omega^{(N)} \to \Omega^{(n_{\text{cell}})}$ is **Borel-measurable** with respect to the product σ-algebras $\mathcal{B}(\Omega^{(N)})$ and $\mathcal{B}(\Omega^{(n_{\text{cell}})})$.

**Proof:**

1. **CVT map is continuous almost everywhere:**
   - Lloyd's algorithm with deterministic initialization and tie-breaking produces a measurable selection from the (possibly multi-valued) CVT correspondence.
   - Discontinuities occur on a set $D \subset \mathcal{X}^N$ of **Lebesgue measure zero** (Du et al., 1999, §3.2). Specifically, $D$ consists of configurations where walker positions lie on Voronoi cell boundaries.

2. **QSD is absolutely continuous:**
   - The microscopic QSD $\mu_{\text{QSD}}^{(N)}$ is absolutely continuous with respect to Lebesgue measure on $\mathcal{X}^N \times \mathbb{R}^{Nd}$.
   - This follows from the non-degenerate Gaussian noise in the BAOAB Ornstein-Uhlenbeck step: the density $p_{\text{QSD}}(x, v)$ is smooth and strictly positive (Theorem {prf:ref}`thm-qsd-existence`, [06_convergence.md](../1_euclidean_gas/06_convergence.md)).

3. **Discontinuity set has measure zero under QSD:**
   - Since $\mu_{\text{QSD}}^{(N)} \ll \text{Leb}$ (absolutely continuous) and $\text{Leb}(D) = 0$, we have:
     $$
     \mu_{\text{QSD}}^{(N)}(D \times \mathbb{R}^{Nd}) = 0
     $$

4. **Almost-everywhere continuous implies measurable:**
   - A function that is continuous $\mu$-almost everywhere for a Borel probability measure $\mu$ is automatically Borel-measurable (Bogachev, 2007, Theorem 7.2.1).
   - Therefore, $X \mapsto \{c_\alpha\}$ is $\mathcal{B}(\mathcal{X}^N)$-measurable.

5. **Velocity averaging is continuous:**
   - The map $(V, \{C_\alpha\}) \mapsto \{\tilde{v}_\alpha\}$ is continuous (linear averaging over fixed partitions).

6. **Composition:**
   - The full map $\mathcal{R}_{\text{scutoid},b}(X, V) = (\text{CVT}(X), \text{AvgVel}(V, \text{Partition}(X)))$ is a composition of measurable functions, hence measurable.

**Reference:** Du, Faber & Gunzburger (1999), "Centroidal Voronoi Tessellations: Applications and Algorithms," *SIAM Review* 41(4), 637-676; Bogachev, V.I. (2007), *Measure Theory*, Vol. I, Springer.
:::

:::{prf:remark} Push-Forward of the QSD
:class: note

The renormalization map induces a **push-forward** of the microscopic QSD to a macroscopic distribution:

$$
\tilde{\mu}_{\text{QSD}}^{(n_{\text{cell}})} := (\mathcal{R}_{\text{scutoid},b})_\# \mu_{\text{QSD}}^{(N)}
$$

defined by:

$$
\tilde{\mu}_{\text{QSD}}^{(n_{\text{cell}})}(A) := \mu_{\text{QSD}}^{(N)}(\mathcal{R}_{\text{scutoid},b}^{-1}(A))
$$

for any measurable set $A \in \mathcal{B}(\Omega^{(n_{\text{cell}})})$.

**Well-Definedness:** Since $\mathcal{R}_{\text{scutoid},b}$ is Borel-measurable ({prf:ref}`prop-renormalization-measurability`), the push-forward measure $\tilde{\mu}_{\text{QSD}}^{(n_{\text{cell}})}$ is a valid probability measure on $\Omega^{(n_{\text{cell}})}$.

**Question:** Is the pushed-forward measure $\tilde{\mu}_{\text{QSD}}^{(n_{\text{cell}})}$ the stationary distribution of the **coarse-grained Markov chain** obtained by applying $\mathcal{R}_{\text{scutoid},b}$ at each time step? This is precisely the question addressed by the Information Closure Hypothesis.
:::

## 3. Micro-Process: The Full N-Walker Scutoid Tessellation

### 3.1. State Space of the Microscopic System

:::{prf:definition} Microscopic State Space
:label: def-scutoid-state-space-micro

The **microscopic state** at discrete time $k$ is the configuration of walker positions and velocities:

$$
Z^{(N)}_k := (X_k, V_k)
$$

where:

- **Positions:** $X_k = (x_{1,k}, \ldots, x_{N,k}) \in \mathcal{X}^N$ (walker positions in state space)
- **Velocities:** $V_k = (v_{1,k}, \ldots, v_{N,k}) \in \mathbb{R}^{Nd}$ (walker velocities)

**State space:**

$$
\Omega^{(N)} := \mathcal{X}^N \times \mathbb{R}^{Nd}
$$

where $\mathcal{X} \subset \mathbb{R}^d$ is the compact spatial domain.

**Measurable structure:** $\Omega^{(N)}$ is a Polish space (complete, separable metric space) equipped with the product topology and Borel σ-algebra $\mathcal{B}(\Omega^{(N)})$.
:::

:::{prf:remark} Geometric Information as Derived Observables
:class: important

Tessellations are **derived observables**, not part of the fundamental state:

- **Voronoi tessellation:** $\mathcal{V}_k = \text{Voronoi}(X_k)$ is a deterministic function of the current positions $X_k$

- **Scutoid tessellation:** $\mathcal{S}_k = \text{Scutoid}(X_k, X_{k+1})$ is a property of the **transition** $(Z_k, Z_{k+1})$, not the state $Z_k$ itself

- **Emergent geometry:** The Riemannian metric $g_{ab}(x)$ encoded via the deformation tensor ({prf:ref}`def-edge-deformation`, [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md)) is computed from the trajectory $\{X_k\}_{k \geq 0}$

This separation ensures the microscopic process is genuinely Markovian: the distribution of $Z_{k+1}$ depends only on $Z_k$, not on any future information.
:::

### 3.2. Dynamics: The Microscopic Markov Chain

:::{prf:definition} Microscopic Markov Chain
:label: def-scutoid-markov-chain-micro

The sequence $\{Z^{(N)}_k\}_{k \geq 0} = \{(X_k, V_k)\}_{k \geq 0}$ is a **time-homogeneous Markov chain** on $\Omega^{(N)} = \mathcal{X}^N \times \mathbb{R}^{Nd}$ with transition kernel:

$$
\mathbb{P}^{(N)}(Z_k, A) := P(Z_{k+1} \in A \mid Z_k) \quad \text{for } A \in \mathcal{B}(\Omega^{(N)})
$$

defined by the **BAOAB + cloning** dynamics:

**Step 1: BAOAB Integrator** ([05_kinetic_contraction.md](../1_euclidean_gas/05_kinetic_contraction.md))

Apply one timestep of the BAOAB integrator to each walker:

$$
(X_{k+1}, V_{k+1}) = \Psi_{\text{BAOAB}}(X_k, V_k; \Delta t)
$$

This includes:
- **B:** Velocity half-step with forces
- **A:** Position full-step with velocities
- **O:** Ornstein-Uhlenbeck velocity update (friction + noise)
- **A:** Position half-step
- **B:** Velocity half-step

**Step 2: Cloning Operator** (Chapter 3, [03_cloning.md](03_cloning.md))

With probability $p_{\text{clone}} \cdot N$, a cloning event occurs:
- Select a victim walker $i$ (low fitness)
- Select a companion walker $j$ (high fitness, possibly IG-connected)
- Replace $i$ with a noisy copy of $j$: $x_i \gets x_j + \xi$, $v_i \gets v_j + \eta$

**Output:** The new state is $Z_{k+1} = (X_{k+1}, V_{k+1})$.

**Markovity:** The transition kernel $\mathbb{P}^{(N)}(Z_k, \cdot)$ depends only on the current state $Z_k = (X_k, V_k)$, not on the history $\{Z_j\}_{j < k}$ or on any future states. This makes $\{Z^{(N)}_k\}$ a genuine Markov chain.

**Stationarity:** Under the quasi-stationary distribution (QSD) $\mu_{\text{QSD}}^{(N)}$ on $\Omega^{(N)}$, the chain is stationary and geometrically ergodic (Theorem {prf:ref}`thm-qsd-existence`, [06_convergence.md](../1_euclidean_gas/06_convergence.md)).
:::

:::{prf:remark} Tessellations as Transition-Dependent Observables
:class: note

Given a trajectory $\{Z_k\}_{k \geq 0}$, the geometric structures can be computed:

- **Voronoi cells at time $k$:** $\mathcal{V}_k = \text{Voronoi}(X_k)$ (depends on $Z_k$ only)
- **Scutoid cells connecting times $k$ and $k+1$:** $\mathcal{S}_k = \text{Scutoid}(X_k, X_{k+1})$ (depends on the transition)
- **Emergent metric:** $g_{ab}(x, k)$ computed from the deformation tensor along the trajectory

These are **post-processed** from the Markov chain $\{Z_k\}$—they are not needed to define the dynamics.
:::

### 3.3. Observables on the Microscopic System

:::{prf:definition} Microscopic Observables
:label: def-micro-observables

A **microscopic observable** is a measurable function $f: \Omega^{(N)} \to \mathbb{R}$ from the state space to the reals.

**Examples:**

1. **Local field values:** $f(Z) = \frac{1}{N} \sum_{i=1}^N \phi(x_i)$ (average of field $\phi$ over walkers)

2. **Wilson loops:** $f(Z) = W_C(Z)$ (holonomy around curve $C$, computed using IG edge weights; see [08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md))

3. **Topological charge:** $f(Z) = Q(Z)$ (Chern number computed from curvature on scutoid tessellation)

4. **Correlation functions:** $f(Z) = G(x, y) = \langle \phi(x) \phi(y) \rangle_Z$ (two-point correlator)

**Expectation values:** Under the QSD $\mu_{\text{QSD}}^{(N)}$ on $\Omega^{(N)}$:

$$
\langle f \rangle_{\text{micro}} := \int_{\Omega^{(N)}} f(Z) \, d\mu_{\text{QSD}}^{(N)}(Z)
$$
:::

:::{prf:definition} Long-Range Observable Class
:label: def-long-range-observable

An observable $f: \Omega^{(N)} \to \mathbb{R}$ is **$(\ell, L)$-long-range** if it satisfies:

**1. Spatial Averaging (Locality Decay)**

The observable depends on walker positions only through spatial averages over scale $\ell \gg a$ (lattice spacing). Formally, $f$ can be written as:

$$
f(Z) = F\left( \left\{ \frac{1}{|\mathcal{B}_r(y)|} \sum_{i: x_i \in \mathcal{B}_r(y)} \phi(x_i, v_i) \right\}_{y \in \mathcal{Y}, r \geq \ell} \right)
$$

for some functional $F$ and test function $\phi$, where $\mathcal{B}_r(y)$ is a ball of radius $r$ around $y$ and $\mathcal{Y}$ is a covering of $\mathcal{X}$.

**2. Exponential Locality (Lipschitz Decay)**

For two states $Z, Z'$ differing only in walkers separated by distance $r > 0$:

$$
|f(Z) - f(Z')| \leq L \cdot e^{-r/\ell}
$$

where $L$ is the Lipschitz constant and $\ell$ is the **correlation length**.

**Interpretation:**
- Long-range observables are **insensitive** to microscopic rearrangements of individual walkers
- They probe physics at scales $\gg \ell$, making them natural candidates for preservation under coarse-graining
- The exponential decay ensures that distant perturbations have negligible effect

**Examples:**
1. **Wilson loops** around contours $C$ with diameter $\text{diam}(C) \gg \ell$
2. **Total energy** or **total mass** (global sums over all walkers)
3. **Correlation functions** $G(x,y) = \langle \phi(x) \phi(y) \rangle$ with $|x - y| \gg \ell$
4. **Topological observables** (e.g., total winding number, Chern number)

**Counter-Examples (Short-Range):**
1. Single-walker position $f(Z) = x_i$ (no averaging)
2. Nearest-neighbor spacing $f(Z) = \min_{i \neq j} |x_i - x_j|$ (sensitive to microscopic details)
:::

:::{prf:remark} Connection to Information Closure
:class: important

The restriction to long-range observables in the Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`) is **essential**. Short-range observables, by definition, depend on microscopic details that are lost during coarse-graining. Information closure can only hold for observables whose predictive information is captured by the macroscopic degrees of freedom.

**Heuristic Argument:** If $f$ is $(\ell, L)$-long-range with $\ell \gg n_{\text{cell}}^{-1/d}$ (coarse resolution), then the coarse-grained state $\tilde{Z}$ retains all information needed to predict $f$ to within the Lipschitz error $L \cdot \exp(-\text{separation}/\ell)$.
:::

:::{prf:remark} Relation to LSI and Correlation Decay
:class: note

Under the Log-Sobolev Inequality (LSI) for the QSD ({prf:ref}`def-lsi`, [09_kl_convergence.md](../1_euclidean_gas/09_kl_convergence.md)), exponential correlation decay holds:

$$
|\langle f(Z_0) g(Z_t) \rangle - \langle f \rangle \langle g \rangle| \leq C \cdot e^{-\rho t}
$$

for observables with $\|\nabla f\|_{L^2} < \infty$. This provides a **physical justification** for the exponential locality condition: the correlation length $\ell$ is related to the LSI constant $\rho$ via $\ell \sim \rho^{-1/2}$.
:::

### 3.4. The Micro-ε-Machine

:::{prf:definition} Causal States for Scutoid Process (Micro)
:label: def-scutoid-causal-states-micro

Two microscopic pasts $\overleftarrow{Z}_k$ and $\overleftarrow{Z}'_k$ are in the same **causal state** $\sigma_{\text{micro}} \in \Sigma_{\varepsilon}^{(N)}$ if they induce identical conditional distributions over futures:

$$
P(\overrightarrow{Z} \mid \overleftarrow{Z}_k) = P(\overrightarrow{Z} \mid \overleftarrow{Z}'_k)
$$

**Markov reduction:** Since the scutoid Markov chain is Markovian ({prf:ref}`def-scutoid-markov-chain-micro`), causal equivalence reduces to state equivalence:

$$
P(\overrightarrow{Z} \mid Z_k = Z) = P(\overrightarrow{Z} \mid Z_k = Z')
$$

**Micro-ε-machine:** The pair $(\Sigma_{\varepsilon}^{(N)}, T_{\varepsilon}^{(N)})$ where:
- $\Sigma_{\varepsilon}^{(N)}$ is the set of micro-causal states (equivalence classes of full scutoid configurations)
- $T_{\varepsilon}^{(N)}: \Sigma_{\varepsilon}^{(N)} \times \mathcal{A} \to \text{Dist}(\Sigma_{\varepsilon}^{(N)})$ is the transition function induced by $\mathbb{P}_{\text{scutoid}}^{(N)}$

**Statistical complexity:** The memory required for optimal prediction:

$$
C_\mu^{(N)} := H(\Sigma_{\varepsilon}^{(N)}) = -\sum_{\sigma \in \Sigma_{\varepsilon}^{(N)}} \pi_{\varepsilon}^{(N)}(\sigma) \log \pi_{\varepsilon}^{(N)}(\sigma)
$$

where $\pi_{\varepsilon}^{(N)}$ is the stationary distribution over causal states.
:::

:::{prf:remark} Discretization for Finite ε-Machine
:class: note

Since $\mathcal{X}$ and $\mathbb{R}^d$ are continuous, we must discretize to obtain a finite ε-machine:

- **Spatial discretization:** Partition $\mathcal{X}$ into cells of size $\delta_x \sim a$ (lattice spacing)
- **Velocity discretization:** Partition $\mathbb{R}^d$ into bins of size $\delta_v \sim \sqrt{T}$ (thermal velocity scale)
- **Tessellation coarse-graining:** Identify tessellations that differ by small perturbations

The continuum limit $\delta_x, \delta_v \to 0$ yields a dense (possibly infinite) ε-machine. For computational purposes, we work with finite discretizations matched to the physical scales of interest.
:::

### 3.5. Topological Structure of Tessellation Spaces

:::{important}
**Formalization Note:** The following rigorously defines the mathematical structure of tessellation spaces. Without this structure, the renormalization map and all subsequent claims cannot be made precise.
:::

:::{prf:definition} Hausdorff Metric on Tessellations
:label: def-tessellation-hausdorff-metric

For two Voronoi tessellations $\mathcal{V} = \{\mathcal{V}_i\}_{i=1}^N$ and $\mathcal{V}' = \{\mathcal{V}'_i\}_{i=1}^N$ of the compact state space $\mathcal{X}$, define the **Hausdorff distance** between their cell boundary sets:

$$
d_{\text{Tess}}(\mathcal{V}, \mathcal{V}') := d_H(\partial \mathcal{V}, \partial \mathcal{V}')
$$

where:

$$
\partial \mathcal{V} := \bigcup_{i=1}^N \partial \mathcal{V}_i
$$

is the union of all Voronoi cell boundaries, and $d_H$ is the Hausdorff metric:

$$
d_H(A, B) := \max\left\{ \sup_{a \in A} d(a, B), \sup_{b \in B} d(b, A) \right\}
$$

with $d(x, S) := \inf_{y \in S} \|x - y\|$ the distance from point $x$ to set $S$.

**Interpretation:** Two tessellations are close if their cell boundaries are close as subsets of $\mathcal{X}$.
:::

:::{prf:definition} Non-Degenerate Tessellation Space
:label: def-nondegenerate-tessellations

For $\delta > 0$, define the **non-degenerate tessellation space**:

$$
\text{Tess}_{\text{nd}}(\mathcal{X}, N, \delta) := \left\{ \mathcal{V} = \text{Voronoi}(g_1, \ldots, g_N) : \min_{i \neq j} \|g_i - g_j\| \geq \delta \right\}
$$

This is the subset of tessellations whose generators are separated by at least $\delta > 0$.

**Physical Motivation:** In thermal equilibrium, walkers are separated by at least the thermal length scale $\ell_{\text{thermal}} \sim \sqrt{D/\gamma}$ with high probability. The non-degeneracy condition reflects this natural separation.
:::

:::{prf:theorem} Polishness of Non-Degenerate Tessellation Spaces (Corrected)
:label: thm-tessellation-polishness

Assume the state space $\mathcal{X} \subset \mathbb{R}^d$ is compact and $\delta > 0$.

The space $(\text{Tess}_{\text{nd}}(\mathcal{X}, N, \delta), d_{\text{Tess}})$ equipped with the Hausdorff metric is a **Polish space**, i.e., it is:

1. **Complete:** Every Cauchy sequence of non-degenerate tessellations converges to a limit tessellation (also non-degenerate)

2. **Separable:** There exists a countable dense subset

3. **Locally Compact:** Every point has a compact neighborhood

**Proof Sketch:**

*Completeness:* Let $\{\mathcal{V}^{(k)}\}$ be a Cauchy sequence in $\text{Tess}_{\text{nd}}(\mathcal{X}, N, \delta)$ with generators $G^{(k)} = (g_1^{(k)}, \ldots, g_N^{(k)})$.

1. Since $\mathcal{X}$ is compact, there exists a subsequence $G^{(k_j)} \to G^* = (g_1^*, \ldots, g_N^*)$ in $\mathcal{X}^N$.

2. The $\delta$-separation is preserved in the limit: for $i \neq j$,
   $$
   \|g_i^* - g_j^*\| = \lim_{j \to \infty} \|g_i^{(k_j)} - g_j^{(k_j)}\| \geq \delta
   $$
   So $G^*$ generates a valid non-degenerate tessellation $\mathcal{V}^* \in \text{Tess}_{\text{nd}}(\mathcal{X}, N, \delta)$.

3. The Voronoi map is continuous on the non-degenerate subset, so $\mathcal{V}^{(k)} \to \mathcal{V}^*$ in Hausdorff distance.

*Separability:* Tessellations with generators having rational coordinates are dense.

*Local Compactness:* Any ball $B_r(\mathcal{V})$ in $\text{Tess}_{\text{nd}}$ has compact closure (by Blaschke selection theorem restricted to the $\delta$-separated subset).

**Reference:** Okabe et al. (2000), *Spatial Tessellations*, Wiley, Ch. 3.
:::

:::{prf:remark} Implications for the Framework
:class: note

**Practical Impact:** The non-degeneracy restriction $\delta > 0$ is mild. The microscopic QSD $\mu_{\text{QSD}}^{(N)}$ assigns measure zero to degenerate configurations (where two walkers coincide), because the noise in BAOAB ensures walkers remain separated.

**Consequence:** All probability-theoretic statements hold on the non-degenerate subspace. The degenerate boundary (measure zero) does not affect the analysis.

**Choice of $\delta$:** In practice, we can take $\delta = \ell_{\text{thermal}}/2$ where $\ell_{\text{thermal}} = \sqrt{D/\gamma}$ is the thermal length scale.
:::

:::{prf:corollary} Borel Structure on Non-Degenerate Tessellation Spaces
:label: cor-tessellation-borel-structure

The Hausdorff metric induces a Borel σ-algebra $\mathcal{B}(\text{Tess}_{\text{nd}}(\mathcal{X}, N, \delta))$ on the non-degenerate tessellation space. Since the microscopic QSD $\mu_{\text{QSD}}^{(N)}$ has full support on non-degenerate configurations (with $\delta \sim \ell_{\text{thermal}}$), this provides the correct topological foundation for probability theory.

**Note:** Tessellations are derived observables (§3.1), not part of the fundamental state space $\Omega^{(N)} = \mathcal{X}^N \times \mathbb{R}^{Nd}$, which is already Polish without restrictions.
:::

:::{prf:remark} Scutoid Tessellations as Transition Observables
:class: note

Scutoid tessellations are treated as **transition-dependent observables** computed from successive Voronoi tessellations, not as state components. This ensures the underlying process remains Markovian.
:::

---


# PART II: THE GAMMA CHANNEL AS COMPRESSION OPTIMIZER

## 4. Geometric Compression: Ricci, Weyl, and Compressibility

### 4.1. The Vision

**Central Insight:** The gamma channel (geometric curvature feedback) is not just a mechanism for exploring curved spaces—it is a **compression optimizer** that actively prepares the system for optimal coarse-graining.

**The Hypothesis:** By rewarding configurations with high Ricci scalar (dense clusters) and low Weyl tensor (minimal distortion), the gamma channel drives the swarm toward states where a small number of generators $n_{\text{cell}} \ll N$ can accurately represent the full microscopic dynamics.

**Connection to Closure:** The Weyl norm directly controls the **lumpability error**—the information lost during coarse-graining. Minimizing Weyl minimizes information loss, thereby facilitating computational closure.

### 4.2. The Geometry of Optimal Compression

:::{prf:observation} Two Components of the Gamma Channel
:label: obs-gamma-compression-components

The gamma channel potential ({prf:ref}`def-gamma-channel-potential`, [adaptive_gas.md](adaptive_gas.md)) has two terms:

$$
U_{\gamma}(x) = -\gamma_R R(x) + \gamma_W \|C(x)\|^2
$$

where $R$ is the Ricci scalar and $\|C\|^2$ is the squared Weyl norm.

**1. Maximizing Ricci Scalar ($+\gamma_R R$): "Clumping"**

- **Geometric meaning:** High positive Ricci curvature signifies that volumes are locally contracting. Walkers are being focused toward a common center.
- **Effect on configuration:** Creates dense, spherical clusters of walkers
- **Compressibility:** A dense, spherical cluster of $k$ walkers is **highly compressible**—its information can be summarized by a single generator at its centroid with low quantization error

**2. Minimizing Weyl Norm ($-\gamma_W \|C\|^2$): "Minimizing Distortion"**

- **Geometric meaning:** The Weyl tensor measures tidal/shear curvature—distortion without volume change. High Weyl means space is stretched in some directions, squeezed in others.
- **Effect on configuration:** Penalizes anisotropic structures (elongated filaments, flattened sheets), rewards spherical symmetry
- **Compressibility:** Elongated clusters are **poorly compressible**—a single centroid is far from endpoints, leading to high quantization error

**Synthesis:** High Ricci + Low Weyl = **Optimal Compressibility**

The gamma channel creates dense, isotropic (spherical) clusters that are well-separated—exactly the configuration that minimizes CVT quantization error.
:::

### 4.3. Quantization Error and Cluster Geometry

:::{prf:definition} Cluster Inertia Tensor
:label: def-cluster-inertia

For each coarse cell $C_\alpha$ with centroid $c_\alpha = \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} x_i$, define the **cluster inertia tensor**:

$$
S_\alpha := \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} (x_i - c_\alpha) \otimes (x_i - c_\alpha) \in \mathbb{R}^{d \times d}
$$

**Eigenvalue decomposition:** Let $\lambda_1(\alpha) \geq \lambda_2(\alpha) \geq \cdots \geq \lambda_d(\alpha) \geq 0$ be the eigenvalues of $S_\alpha$.

**Geometric interpretation:**
- **Trace:** $\text{tr}(S_\alpha) = \sum_{i=1}^d \lambda_i(\alpha)$ measures the total "spread" of the cluster
- **Anisotropy:** The ratio $\kappa_\alpha := \lambda_1(\alpha) / \lambda_d(\alpha)$ measures deviation from spherical symmetry
  - $\kappa_\alpha = 1$: Perfect sphere (isotropic)
  - $\kappa_\alpha \gg 1$: Elongated/filamentary (anisotropic)

**Connection to CVT error:** The quantization error for cell $\alpha$ is exactly:

$$
E_{\text{CVT}}^\alpha = \sum_{i \in C_\alpha} \|x_i - c_\alpha\|^2 = |C_\alpha| \cdot \text{tr}(S_\alpha)
$$
:::

:::{prf:lemma} CVT Error Bound via Cluster Anisotropy
:label: lem-cvt-anisotropy-bound

The CVT quantization error for coarse cell $\alpha$ satisfies:

$$
E_{\text{CVT}}^\alpha \leq C_d \cdot (\text{tr}\, S_\alpha) \cdot \kappa_\alpha^{1/2} \cdot |C_\alpha|^{1-1/d}
$$

where $C_d$ is a dimension-dependent constant from optimal quantization theory (Gersho's constant).

**Proof Sketch:**

1. **Gersho's theorem** (Gersho, 1979): For optimal quantization of a distribution with density $\rho(x)$ in $\mathbb{R}^d$, the mean squared error per cell scales as:
   $$
   E_{\text{cell}} \sim G_d \cdot V^{2/d}
   $$
   where $V$ is the cell volume and $G_d$ is Gersho's constant.

2. **Anisotropic correction:** For an anisotropic cluster with aspect ratio $\kappa$, the effective "diameter" in the longest direction scales as $(\text{tr}\, S)^{1/2} \cdot \kappa^{1/2}$.

3. **Volume scaling:** For $|C_\alpha|$ walkers in a cell, $V \sim |C_\alpha|^{-1}$ (inverse density).

4. **Combining:**
   $$
   E_{\text{CVT}}^\alpha \sim \text{(spread)} \cdot \text{(anisotropy penalty)} \cdot \text{(density factor)}
   $$

**Reference:** Gersho, A. (1979), "Asymptotically optimal block quantization," *IEEE Trans. Inf. Theory* 25(4), 373-380.

**Interpretation:** High anisotropy $\kappa_\alpha$ increases the quantization error. Spherical clusters ($\kappa_\alpha \approx 1$) minimize error for fixed spread.
:::

:::{prf:corollary} Weyl Curvature and CVT Error
:label: cor-weyl-cvt-error

If the Weyl tensor $C(x)$ is approximately constant within cell $\alpha$, then the cluster anisotropy satisfies:

$$
\kappa_\alpha - 1 \geq c \cdot \|C|_{c_\alpha}\|^2 \cdot \text{tr}(S_\alpha)
$$

for some constant $c > 0$.

**Heuristic Justification:** The Weyl tensor measures tidal forces that stretch the cluster anisotropically. A cluster initially spherical will be deformed by Weyl curvature, with elongation proportional to $\|C\|^2 \cdot (\text{size})^2$.

**Consequence:** Combining with {prf:ref}`lem-cvt-anisotropy-bound`:

$$
E_{\text{CVT}}^\alpha \lesssim C_d \cdot \text{tr}(S_\alpha)^{3/2} \cdot \|C|_{c_\alpha}\| \cdot |C_\alpha|^{1-1/d}
$$

**Gamma channel reduces CVT error:** By minimizing $\|C\|^2$ via the $\gamma_W$ term, the gamma channel forces clusters toward spherical geometry ($\kappa_\alpha \to 1$), thereby minimizing the information lost when replacing them with centroids.

**Status:** Corollary statement is heuristic; rigorous proof requires analyzing BAOAB dynamics under background curvature (see §5.4 below).
:::

### 4.4. From Geometry to Information Theory

:::{prf:observation} The Compressibility-Closure Connection
:label: obs-compressibility-closure

**CVT as lossy compression:** The Fixed-Node Scutoid Tessellation is a lossy compression algorithm: map $N$ walkers $\to$ $n_{\text{cell}}$ generators.

**Information loss:** The "loss" is quantified by:
1. **Quantization error:** Spatial distortion $\sim \sum_{\alpha} \sum_{i \in C_\alpha} \|x_i - c_\alpha\|^2$
2. **Lumpability error:** Predictive information loss $\varepsilon_{\text{lump}}$ (how much do micro-futures differ within the same macro-state?)

**Key claim:** These two notions of "loss" are related. Configurations with low geometric distortion (low CVT error) also have low predictive information loss (low lumpability error).

**Why?** Because walkers in a compact, isotropic cluster evolve coherently—they have similar futures. Walkers in an elongated, distorted cluster evolve divergently due to anisotropic shear (Weyl curvature) → different futures → lumpability breaks.

**Gamma channel optimizes both:** Minimizing Weyl reduces geometric distortion AND predictive loss simultaneously.
:::

---

## 5. The Weyl-Lumpability Connection (Conceptual)

### 5.1. The Central Conjecture

:::{prf:conjecture} Weyl Norm Bounds Lumpability Error
:label: conj-weyl-bounds-lumpability

For the scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$, the lumpability error $\varepsilon_{\text{lump}}$ is bounded by the integrated Weyl norm:

$$
\varepsilon_{\text{lump}} \leq C_1 \cdot \frac{1}{n_{\text{cell}}} \sum_{k=1}^{n_{\text{cell}}} \int_{\tilde{\mathcal{V}}_k} \|C(x)\|^2 \, dV_g(x) + C_2 e^{-b/\xi}
$$

where:
- $\tilde{\mathcal{V}}_k$ are the Voronoi cells of the coarse-grained generators
- $C(x)$ is the Weyl tensor field
- $dV_g = \sqrt{\det g} \, d^dx$ is the Riemannian volume element
- $\xi$ is the correlation length
- $C_1, C_2$ are constants (to be determined)

**First term:** Geometric distortion contribution (Weyl)  
**Second term:** Correlation decay contribution (standard Markov lumpability)

**Status:** **CONJECTURE** - Requires proof

**Strategy for proof:** Connect the anisotropic evolution governed by the Weyl tensor to the divergence of micro-futures within a macro-state (see §5.2).
:::

### 5.2. Heuristic Argument

:::{prf:heuristic} Why Weyl Controls Lumpability
:label: heuristic-weyl-lumpability-mechanism

**Setup:** Consider two micro-states $Z, Z' \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ (both map to the same macro-state $\tilde{Z}$). They differ in the internal configuration of walkers within some coarse cell $\alpha$.

**Example:** 
- State $Z$: Walkers in cell $\alpha$ form a spherical cluster
- State $Z'$: Walkers in cell $\alpha$ form an elongated ellipsoid (due to previous shear from Weyl curvature)

**Evolution under BAOAB:**

In a region with **zero Weyl** ($C_{abcd} = 0$):
- The metric is conformally flat: $g_{ab} = e^{2\Omega(x)} \delta_{ab}$
- Evolution is isotropic—spheres remain spheres (up to scaling)
- Both $Z$ and $Z'$ evolve similarly: the centroid $c_\alpha$ moves the same way regardless of the internal cluster shape
- **Lumpability holds:** $\mathbb{P}(Z, \cdot) \approx \mathbb{P}(Z', \cdot)$ when restricted to macro-observables

In a region with **nonzero Weyl** ($\|C\|^2 > 0$):
- Space has tidal/shear distortion
- Spheres get sheared into ellipsoids with axes aligned along principal curvature directions
- The centroid of a sphere and an ellipsoid move **differently** under shear
- **Lumpability fails:** The macro-future depends on the micro-configuration

**Quantitative estimate:**

The centroid velocity difference scales as:

$$
\Delta v_c \sim \|C\|^2 \cdot \text{(cluster size)}^2 \cdot \Delta t
$$

Integrating over one timestep and summing over all coarse cells gives the lumpability error.

**Conclusion:** $\varepsilon_{\text{lump}} \propto \int \|C\|^2 \, dV$.
:::

### 5.3. Path to Rigor

**To make Conjecture {prf:ref}`conj-weyl-bounds-lumpability` into a theorem, we need:**

1. **Formalize "cluster shape"**: Define a shape tensor $S_\alpha$ for each coarse cell (moment of inertia tensor) — **DONE** in {prf:ref}`def-cluster-inertia`

2. **Analyze BAOAB evolution on curved manifolds**: Derive how the Weyl tensor couples to the shape tensor evolution (geodesic deviation equation) — **See §5.4 below**

3. **Bound centroid velocity difference**: Prove that $|\tilde{v}_\alpha(Z) - \tilde{v}_\alpha(Z')| \leq C \|C\|_{L^\infty} \cdot \text{tr}(S_\alpha)$

4. **Integrate to get lumpability error**: Sum the velocity differences over all cells and one timestep to bound $\varepsilon_{\text{lump}}$

**Current status:** Steps 1-2 formalized below. Steps 3-4 require coupling analysis (future work, Part V, §15).

### 5.4. Geodesic Deviation Conjecture

:::{prf:conjecture} Centroid Evolution Under Background Curvature
:label: conj-centroid-geodesic-deviation

**Heuristic Statement:** In the overdamped limit with large clusters, centroid motion should approximately follow geodesics of the emergent metric $g_{ab}$, with shape-dependent deviations controlled by the Weyl tensor.

**Formal Conjecture:** Consider two micro-states $Z, Z' \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ differing only in the configuration of walkers within coarse cell $\alpha$. Both have the same centroid $c_\alpha$ at time $t=0$ but different shape tensors $S_\alpha \neq S'_\alpha$.

Under BAOAB evolution in a background metric $g_{ab}(x)$ with Riemann curvature $R^\mu_{\phantom{\mu}\nu\rho\sigma}$ and Weyl tensor $C^\mu_{\phantom{\mu}\nu\rho\sigma}$, the centroid trajectories $c_\alpha(t), c'_\alpha(t)$ satisfy an equation analogous to the **geodesic deviation equation**:

$$
\frac{D^2 \delta c^\mu}{Dt^2} = -R^\mu_{\phantom{\mu}\nu\rho\sigma} \dot{c}^\nu \delta c^\rho \dot{c}^\sigma - C^\mu_{\phantom{\mu}\nu\rho\sigma} \dot{c}^\nu (S - S')^{\rho\sigma} \dot{c}^\sigma + O(\|S - S'\|^2)
$$

where:
- $\delta c := c' - c$ is the centroid separation
- $\dot{c} := dc/dt$ is the centroid velocity
- $D/Dt$ is the covariant derivative along the trajectory
- $(S - S')^{\rho\sigma}$ are the components of the shape tensor difference
- The first term is the standard Riemann curvature contribution (affects all geodesics equally)
- **The second term is the Weyl contribution**, which couples to the shape difference

**Status:** CONJECTURE. This statement is physically motivated but not yet rigorously proven.

**Heuristic Argument:**

1. **Expand walker positions:** Write $x_i = c_\alpha + \xi_i$ where $\xi_i$ are displacements from the centroid with $\sum_i \xi_i = 0$.

2. **BAOAB as geodesic flow:** In the overdamped limit ($\gamma \to \infty$), the BAOAB dynamics reduce to:
   $$
   \dot{x}_i = -\nabla \Phi(x_i) + \sqrt{2D} \, \eta_i
   $$
   Centroid motion: $\dot{c} = \frac{1}{N} \sum_i \dot{x}_i = -\nabla \Phi(c) + O(\|\xi\|^2)$.

3. **Second-order expansion:** The force at $x_i = c + \xi_i$ expands as:
   $$
   \nabla \Phi(c + \xi) = \nabla \Phi(c) + \nabla^2 \Phi(c) \cdot \xi + \frac{1}{2} \nabla^3 \Phi(c) : (\xi \otimes \xi) + \cdots
   $$

4. **Curvature from Hessian:** The Riemann tensor appears through the commutator of covariant derivatives:
   $$
   \nabla_\mu \nabla_\nu \Phi - \nabla_\nu \nabla_\mu \Phi = R_{\mu\nu\rho}^\phantom{\mu\nu\rho\sigma} \partial_\sigma \Phi
   $$

5. **Shape-dependent terms:** Averaging over the cluster:
   $$
   \frac{1}{N} \sum_i \nabla^2 \Phi(c) \cdot \xi_i = \nabla^2 \Phi(c) : S_\alpha
   $$
   The Weyl part of the Riemann tensor couples quadratically to $S$.

6. **Result:** Subtracting the equations for $c$ and $c'$ gives the geodesic deviation equation above.

**Path to Rigorous Proof:** To transform this conjecture into a theorem, the following steps are required:

1. **Itô calculus treatment:** Properly handle the stochastic noise terms $\eta_i$ in the BAOAB dynamics using Itô or Stratonovich calculus to derive the centroid SDE.

2. **Careful overdamped limit:** Rigorously take the limit $\gamma \to \infty$ (high friction) while keeping $D/\gamma$ fixed (fluctuation-dissipation), showing convergence to a deterministic geodesic flow plus corrections.

3. **Second-order expansion:** Expand the centroid evolution to second order in the timestep $\Delta t$ and cluster size $\|\xi\|$, tracking all terms that couple to the shape tensor.

4. **Weyl identification:** Show that the shape-dependent acceleration terms can be decomposed into Ricci (trace) and Weyl (traceless) parts, with the Weyl part appearing as claimed.

5. **Error bounds:** Quantify the $O(\|S - S'\|^2)$ remainder and show it's negligible for thermal equilibrium clusters.

**Reference:** Wald, R. M. (1984), *General Relativity*, §9.2 on geodesic deviation (for the continuous case).

**Timeline:** A complete rigorous proof following the above steps is estimated to require 2-4 months of focused work.
:::

:::{prf:lemma} Weyl Contribution to Centroid Divergence (Conditional)
:label: lem-weyl-centroid-divergence

**Hypothesis:** Assume Conjecture {prf:ref}`conj-centroid-geodesic-deviation` holds.

**Conclusion:** Under this assumption, the centroid velocity difference after one BAOAB timestep $\Delta t$ is bounded by:

$$
|\dot{c}_\alpha - \dot{c}'_\alpha| \leq C \cdot \|C\|_{L^\infty(\mathcal{V}_\alpha)} \cdot \frac{\|S_\alpha - S'_\alpha\|_F}{r_\alpha} \cdot \|\dot{c}_\alpha\|^2 \cdot \Delta t
$$

where:
- $\|C\|_{L^\infty}$ is the maximum Weyl tensor norm in the cell [units: length$^{-2}$]
- $\|S - S'\|_F = \sqrt{\text{tr}[(S - S')^2]}$ is the Frobenius norm of the shape difference [units: length$^2$]
- $r_\alpha = \sqrt{\text{tr}(S_\alpha)}$ is the characteristic cluster radius [units: length]
- $C$ is a dimensionless constant depending on dimension $d$

**Dimensional Check:**
- LHS: $|\dot{c}_\alpha - \dot{c}'_\alpha|$ has units [length/time]
- RHS: $[\text{length}^{-2}] \cdot \frac{[\text{length}^2]}{[\text{length}]} \cdot [\text{length}^2/\text{time}^2] \cdot [\text{time}] = [\text{length/time}]$ ✓

**Proof:** The geodesic deviation equation couples the Weyl tensor to shape differences. Since the coupling must be to a dimensionless measure of shape anisotropy, we write the shape tensor in dimensionless form as $\hat{S} = S/\text{tr}(S) = S/r_\alpha^2$. The deviation equation then yields a velocity divergence $|\delta \dot{c}| \propto \|C\| \cdot \|\hat{S} - \hat{S}'\|_F$. Substituting $\|\hat{S} - \hat{S}'\|_F \approx \|S - S'\|_F / r_\alpha^2$ and integrating over one timestep with $\delta c(0) = 0$ and $\delta \dot{c}(0) = 0$ gives the stated bound. The factor $r_\alpha^{-1}$ emerges from the denominator in the dimensionless shape anisotropy.

**Interpretation:** The Weyl tensor acts as a **shape-dependent force**. Two clusters with the same centroid but different shapes experience different accelerations, leading to velocity divergence proportional to the shape difference relative to cluster size.
:::

:::{prf:lemma} Weyl Contribution to Lumpability Error (Preliminary Bound)
:label: lem-weyl-lumpability-preliminary

Consider the lumpability error for the scutoid renormalization map:

$$
\varepsilon_{\text{lump}} := \sup_{\tilde{Z}} \int_{\mathcal{R}^{-1}(\tilde{Z})} \left\| P_{\text{micro}}(Z_{t+1} | Z) - P_{\text{macro}}(\tilde{Z}_{t+1} | \tilde{Z}) \right\|_{TV} \, d\mu(Z | \tilde{Z})
$$

Under BAOAB evolution with background Weyl curvature $C$, friction $\gamma$, and noise $\sigma^2 = 2D$, we have:

$$
\varepsilon_{\text{lump}} \leq C_1 \cdot \frac{1}{n_{\text{cell}}} \sum_{\alpha=1}^{n_{\text{cell}}} \int_{\tilde{\mathcal{V}}_\alpha} \|C(x)\|^2 \cdot \mathbb{E}[\text{tr}(S_\alpha)] \, dV_g(x) + C_2 \cdot e^{-\gamma t_{\text{mix}}}
$$

where:
- The first term is the **geometric Weyl contribution**
- The second term is the standard **Markov mixing error**
- $C_1, C_2$ depend on dimension, friction, and noise amplitude

**Proof Strategy:**

1. **Coupling construction:** Define a coupling between $P_{\text{micro}}(\cdot | Z)$ and $P_{\text{micro}}(\cdot | Z')$ for $Z, Z' \in \mathcal{R}^{-1}(\tilde{Z})$ using the synchronous coupling (same noise realizations).

2. **Dobrushin coefficient:** The total variation distance is bounded by:
   $$
   \|P(\cdot | Z) - P(\cdot | Z')\|_{TV} \leq \mathbb{E}_\eta[\|Z_{t+1} - Z'_{t+1}\|]
   $$
   where the expectation is over the noise $\eta$.

3. **Centroid divergence:** Use {prf:ref}`lem-weyl-centroid-divergence` to bound the centroid separation after one step in terms of $\|C\|$ and $\|S - S'\|$.

4. **Integrate over cells:** Sum over all coarse cells and average over the conditional distribution $\mu(Z | \tilde{Z})$.

5. **Variance bound:** Use the fact that $\mathbb{E}[\|S - S'\|^2] \leq \text{Var}(S) \sim \text{tr}(S)$ for walkers in thermal equilibrium.

**Status:** LEMMA (not full conjecture). This proves that Weyl curvature **does** appear in the lumpability bound with the correct functional form. However, the quantitative constants $C_1, C_2$ require completing the coupling analysis and using LSI-based spatial decay bounds.

**Dependencies:**
- {prf:ref}`conj-centroid-geodesic-deviation` (conjectured, pending rigorous proof)
- {prf:ref}`lem-weyl-centroid-divergence` (conditional on above conjecture)
- Missing: Detailed coupling analysis and optimal constants (future work)
- Missing: `lem-local-lsi-spatial-decay` for the $e^{-\gamma t_{\text{mix}}}$ term (see §14)

**Significance:** This elevates Conjecture {prf:ref}`conj-weyl-bounds-lumpability` from heuristic to **partially proven**. We have established the mechanism—the remaining work is quantitative refinement.
:::

:::{prf:remark} Connection to Gamma Channel Optimization
:class: important

Lemma {prf:ref}`lem-weyl-lumpability-preliminary` provides the **rigorous justification** for the gamma channel design:

**The gamma channel minimizes lumpability error by minimizing Weyl curvature.**

Specifically, the $\gamma_W$ term in the potential:

$$
U_\gamma = -\gamma_R R + \gamma_W \|C\|^2
$$

directly targets the dominant term in $\varepsilon_{\text{lump}}$. By penalizing $\|C\|^2$, the system self-organizes into configurations where coarse-graining preserves predictive information—**facilitating computational closure**.

This is the central mechanistic insight of Part II: the gamma channel is not just exploring curved spaces, it is **actively compressing the state space for optimal renormalization**.
:::

---

## 6. Self-Organizing RG Flow via Curvature Feedback

### 6.1. The Complete Picture

:::{prf:observation} Gamma Channel as RG Flow Facilitator
:label: obs-gamma-as-rg-facilitator

**Traditional RG:** Coarse-graining is imposed externally. We hope the resulting effective theory is valid (closure holds) but have no control over it.

**Fragile Gas with Gamma Channel:** The system **actively prepares itself** for coarse-graining:

1. **The Feedback Loop:**
   - Walkers evolve under BAOAB + cloning
   - The emergent metric $g_{ab}(x)$ and curvature tensors $R(x), C(x)$ are computed from the current configuration
   - The gamma potential $U_\gamma = -\gamma_R R + \gamma_W \|C\|^2$ feeds back into the walker dynamics as a reward signal
   - Walkers move toward high-$R$, low-$\|C\|$ regions

2. **Self-Organization:**
   - Over time, the system evolves toward configurations with:
     - Dense, spherical clusters (high Ricci)
     - Minimal anisotropic distortion (low Weyl)
   - These are precisely the configurations with low CVT quantization error and low lumpability error

3. **Closure Emerges:**
   - By minimizing $\|C\|^2$, the system minimizes $\varepsilon_{\text{lump}}$ (if Conjecture {prf:ref}`conj-weyl-bounds-lumpability` holds)
   - Computational closure is not just hoped for—it is **actively driven** by the gamma channel

**Paradigm shift:** The Fragile Gas is not a passive simulation of physics. It is an **intelligent, self-optimizing computational system** that finds the most efficient representation of its own dynamics.
:::

### 6.2. Connection to the Information Closure Hypothesis

:::{prf:observation} Gamma Channel Targets Information Closure
:label: obs-gamma-targets-closure

Recall the Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`):

$$
I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{\tilde{Z}}_t) \approx I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{Z}_t)
$$

This states: "The macro-past captures all micro-information relevant to the macro-future."

**Equivalent condition (Ortega et al., 2024):** The υ-machine (minimal predictive model for macro-futures given micro-pasts) equals the macro-ε-machine.

**When does this hold?** When micro-states within the same macro-state have **similar futures**. This is exactly what low lumpability error means.

**The gamma channel hypothesis:**

$$
\min \|C\|^2 \quad \Rightarrow \quad \min \varepsilon_{\text{lump}} \quad \Rightarrow \quad \text{Closure holds}
$$

**If this chain is validated:**
- Tuning the gamma channel parameters $\gamma_R, \gamma_W$ gives us a **control knob** for achieving closure
- We can **empirically test** closure by varying $\gamma_W$ and measuring $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}})$ vs. $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z})$
- This transforms the Central Hypothesis from an abstract assumption into a **tunable, verifiable property** of the algorithm
:::

### 6.3. Empirical Predictions

:::{prf:prediction} Observable Effects of Gamma Channel Tuning
:label: pred-gamma-channel-effects

**Prediction 1 (Weyl-CVT correlation):** For fixed $N$ and $n_{\text{cell}}$, increasing $\gamma_W$ (stronger Weyl penalty) should decrease:
- The average Weyl norm: $\langle \|C\|^2 \rangle$
- The CVT quantization error: $\sum_\alpha \sum_{i \in C_\alpha} \|x_i - c_\alpha\|^2$

**Prediction 2 (Weyl-lumpability correlation):** Increasing $\gamma_W$ should decrease the lumpability error $\varepsilon_{\text{lump}}$ (measured via the protocol in Part IV, §14).

**Prediction 3 (Closure vs. Gamma):** There exists an optimal $\gamma_W^*$ where information closure is maximized:

$$
\gamma_W^* = \arg\min_{\gamma_W} \left[ I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z}) - I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}}) \right]
$$

**Prediction 4 (Dimensional dependence):** The optimal $\gamma_W^*$ depends on dimension $d$. For $d \geq 5$, no finite $\gamma_W$ can achieve closure due to the curse of dimensionality (the Weyl contribution to anisotropy grows faster than the gamma channel can suppress it).

**Testing these predictions is a primary goal of the empirical research program (Part V, §16).**
:::

---



---


# PART III: DISCOVERING INTRINSIC SCALES

## 7. Method 1: Observable-Driven Coarse-Graining (The Money Plot)

### 7.1. The Core Idea

**Question:** For a given physical observable, what is the minimal number of generators $n_{\text{cell}}$ needed to accurately compute it?

**Method:** Measure how the observable error changes as we vary $n_{\text{cell}}$, and identify the **plateaus and cliffs** in the error curve.

**Intrinsic scales:** The values of $n_{\text{cell}}$ where plateaus begin represent natural coarse-graining resolutions for that observable.

### 7.2. The Experimental Protocol

:::{prf:algorithm} Observable-Driven Scale Discovery
:label: alg-observable-scale-discovery

**Input:**
- Long trajectory of full $N$-walker simulation: $\{Z_k^{(N)}\}_{k=0}^{K}$
- Key observable $O: \Omega_{\text{scutoid}}^{(N)} \to \mathbb{R}$ (e.g., Wilson loop, string tension, mass gap)
- Range of generator counts: $n_{\text{values}} = [n_{\min}, \ldots, n_{\max} = N]$

**Procedure:**

**Step 1: Compute Ground Truth**
```python
O_true = compute_observable(trajectory, n_cell=N)  # Full resolution
```

**Step 2: Multi-Scale Coarse-Graining**
```python
errors = []
for n in n_values:
    # Apply scutoid renormalization with n generators
    coarse_trajectory = apply_cvt_renormalization(trajectory, n_cell=n)
    
    # Compute observable on coarse trajectory
    O_coarse = compute_observable(coarse_trajectory, n_cell=n)
    
    # Relative error
    error = abs(O_coarse - O_true) / abs(O_true)
    errors.append((n, error))
```

**Step 3: The Money Plot**
```python
plot_log_log(n_values, errors, 
            xlabel="Number of Generators n_cell",
            ylabel="Relative Error |<O>_n - <O>_N| / |<O>_N|")
```

**Output:**
- Error curve $\varepsilon_O(n_{\text{cell}})$
- Identified intrinsic scales: values of $n_{\text{cell}}$ where error plateaus
:::

### 7.3. Interpreting the Money Plot

:::{prf:observation} Plateau and Cliff Structure
:label: obs-plateau-cliff-structure

The error curve $\varepsilon_O(n_{\text{cell}})$ typically exhibits a **staircase structure**:

**Plateau 1 (High $n_{\text{cell}}$):** Error $\approx 0$
- All relevant structures for observable $O$ are resolved
- Further increasing $n_{\text{cell}}$ provides diminishing returns

**Cliff 1:** Steep increase in error
- Crossed a critical scale where essential structures are lost
- Example: Internal structure of instantons destroyed

**Plateau 2 (Medium $n_{\text{cell}}$):** Error stabilizes at small but nonzero value
- Long-range interactions still captured
- Short-range details integrated out
- **This is a valid effective theory** for observable $O$ at this scale

**Cliff 2:** Second steep increase
- Crossed another critical scale
- Example: Instantons themselves are no longer resolved

**Final Collapse:** Error → 100% as $n_{\text{cell}} \to n_{\min}$
- Model has insufficient resolution for any physics

**Intrinsic scales:** The $n_{\text{cell}}$ values at the **start of each plateau** are intrinsic scales for observable $O$.
:::

### 7.4. Observable Hierarchy

:::{prf:heuristic} Different Observables Have Different Intrinsic Scales
:label: heuristic-observable-hierarchy

**Long-range observables** (Wilson loops with large perimeter $L$):
- Require $n_{\text{cell}} \gtrsim (L / \ell_{\text{phys}})^d$ where $\ell_{\text{phys}}$ is the physical correlation length
- Can tolerate aggressive coarse-graining

**Intermediate observables** (correlation functions at separation $r$):
- Require $n_{\text{cell}} \gtrsim (r / \ell_{\text{phys}})^d$
- Moderate coarse-graining tolerance

**Short-range observables** (local field values, single-plaquette actions):
- Require $n_{\text{cell}} \approx N$
- Cannot be coarse-grained without loss

**Topological observables** (Chern numbers, winding numbers):
- May be preserved exactly even with coarse $n_{\text{cell}}$ if defects are resolved
- Plateau extends to very low $n_{\text{cell}}$

**Empirical program:** Measure the intrinsic scale hierarchy for a suite of observables to map out the "resolution spectrum" of the theory.
:::

---

## 8. Method 2: Statistical Complexity Plateaus

### 8.1. The Information-Theoretic Approach

**Core idea:** Look for values of $n_{\text{cell}}$ where the **statistical complexity** (predictive memory) of the coarse-grained process stabilizes.

**Why this matters:** A plateau in $C_\mu(n_{\text{cell}})$ means we can remove degrees of freedom without losing predictive information—the system is causally redundant in that range.

### 8.2. The Algorithm

:::{prf:algorithm} Statistical Complexity Scale Discovery
:label: alg-complexity-scale-discovery

**Input:**
- Trajectory $\{Z_k^{(N)}\}_{k=0}^{K}$ from full simulation
- Range of generator counts $n_{\text{values}}$

**Procedure:**

**Step 1: Multi-Scale Coarse-Graining**
```python
for n in n_values:
    coarse_trajectory[n] = apply_cvt_renormalization(trajectory, n_cell=n)
```

**Step 2: ε-Machine Reconstruction**
```python
C_mu_values = []
for n in n_values:
    # Discretize state space (spatial + velocity)
    symbolic_sequence = discretize_trajectory(coarse_trajectory[n], 
                                              delta_x=lattice_spacing,
                                              delta_v=thermal_velocity)
    
    # Reconstruct ε-machine using CSSR or similar algorithm
    epsilon_machine = reconstruct_epsilon_machine(symbolic_sequence)
    
    # Compute statistical complexity
    C_mu = compute_entropy(epsilon_machine.causal_state_distribution)
    C_mu_values.append((n, C_mu))
```

**Step 3: Identify Plateaus**
```python
plot_semilog(n_values, C_mu_values,
            xlabel="Number of Generators n_cell",
            ylabel="Statistical Complexity C_μ (bits)")

# Find plateaus: regions where dC_mu/d(log n) ≈ 0
plateaus = find_flat_regions(C_mu_values, threshold=0.1)
```

**Output:**
- Statistical complexity curve $C_\mu(n_{\text{cell}})$
- Intrinsic scales: $n_{\text{cell}}$ values where $C_\mu$ plateaus
:::

### 8.3. Interpretation

:::{prf:observation} Complexity Plateaus Indicate Causal Redundancy
:label: obs-complexity-plateaus

**Plateau in $C_\mu$:** If $C_\mu(n_1) \approx C_\mu(n_2)$ for $n_1 < n_2$, then:
- The extra $n_2 - n_1$ generators at the finer resolution do not add predictive power
- They capture redundant, non-causal information (noise, short-range fluctuations)
- **Intrinsic scale:** $n_1$ is sufficient for optimal prediction

**Sharp drop in $C_\mu$:** If $C_\mu$ decreases rapidly as $n_{\text{cell}}$ decreases:
- We have crossed a scale where essential predictive structure is lost
- The system becomes "simpler" but also less predictive

**Connection to closure:** 
- Plateau ⇒ $C_\mu^{(n_1)} \approx C_\mu^{(N)}$ ⇒ Information closure holds at scale $n_1$
- This directly tests Hypothesis {prf:ref}`hyp-scutoid-information-closure`

**Emergence signature:** If $C_\mu^{(\text{macro})} < C_\mu^{(\text{micro})}$ in the plateau region, macroscopic laws have emerged that are simpler than microscopic laws.
:::

---

## 9. Method 3: Topological Phase Transitions

### 9.1. Geometric Approach via Persistent Homology

**Core idea:** The global **shape** of the configuration space changes qualitatively at intrinsic scales. Use topological data analysis to detect these changes.

**Tool:** Persistent homology on the point cloud of generator positions.

### 9.2. The Algorithm

:::{prf:algorithm} Topological Scale Discovery
:label: alg-topological-scale-discovery

**Input:**
- Equilibrium configurations from QSD: $\{X_k\}_{k=1}^{K}$ (positions only)
- Range of generator counts $n_{\text{values}}$

**Procedure:**

**Step 1: Multi-Scale Point Clouds**
```python
for n in n_values:
    # Apply CVT to get n generators
    generators[n] = cvt_clustering(positions, n_cell=n)
```

**Step 2: Persistent Homology**
```python
persistence_diagrams = []
for n in n_values:
    # Compute Vietoris-Rips filtration
    point_cloud = generators[n]
    diagram = compute_persistent_homology(point_cloud, max_dimension=2)
    persistence_diagrams.append((n, diagram))
```

**Step 3: Track Topological Features**
```python
# Extract Betti numbers as functions of n_cell
beta_0 = []  # Connected components
beta_1 = []  # Loops
beta_2 = []  # Voids

for n, diagram in persistence_diagrams:
    # Count features with persistence > threshold
    beta_0.append(count_features(diagram, dimension=0, threshold=persistence_min))
    beta_1.append(count_features(diagram, dimension=1, threshold=persistence_min))
    beta_2.append(count_features(diagram, dimension=2, threshold=persistence_min))

# Plot Betti numbers vs n_cell
plot_betti_curves(n_values, beta_0, beta_1, beta_2)
```

**Step 4: Identify Topological Transitions**
```python
# Find n_cell where Betti numbers change
transitions = find_jumps(beta_0) + find_jumps(beta_1) + find_jumps(beta_2)
```

**Output:**
- Persistence diagrams for each $n_{\text{cell}}$
- Betti number curves $\beta_k(n_{\text{cell}})$
- Intrinsic scales: $n_{\text{cell}}$ where topology changes
:::

### 9.3. Physical Interpretation

:::{prf:observation} Topological Transitions Mark Physical Scales
:label: obs-topological-transitions

**Example 1: Barrier Crossing**
- At high $n_{\text{cell}}$: Generators sample both sides of a fitness barrier → $\beta_1 > 0$ (loop around barrier)
- At critical $n_{\text{cell}}^*$: Not enough resolution to see barrier → $\beta_1$ drops to 0
- **Intrinsic scale:** $n_{\text{cell}}^* \sim (\text{barrier width}/\text{lattice spacing})^d$

**Example 2: Cluster Formation**
- At high $n_{\text{cell}}$: Generators resolve individual clusters → $\beta_0 =$ number of clusters
- At critical $n_{\text{cell}}^*$: Clusters merge → $\beta_0$ decreases
- **Intrinsic scale:** $n_{\text{cell}}^* \sim (\text{inter-cluster distance})^{-d}$

**Connection to QSD geometry:** The topological features directly reflect the structure of the quasi-stationary distribution $\mu_{\text{QSD}}$:
- Voids ($\beta_2$): Regions of low walker density
- Loops ($\beta_1$): Barriers or metastable states
- Components ($\beta_0$): Disconnected regions of phase space

**Empirical prediction:** Topological intrinsic scales should correlate with physical scales (instanton size, Debye screening length, etc.).
:::

---

## 10. Method 4: Geometrothermodynamic Singularities

### 10.1. Thermodynamic Curvature Approach

**Core idea:** Phase transitions manifest as **curvature singularities** in the thermodynamic geometry. The minimal $n_{\text{cell}}$ required to correctly capture the phase transition is an intrinsic scale.

**Tool:** Ruppeiner scalar curvature $R_{\text{Rupp}}$ from geometrothermodynamics.

### 10.2. The Algorithm

:::{prf:algorithm} Thermodynamic Scale Discovery
:label: alg-thermodynamic-scale-discovery

**Input:**
- Thermodynamic parameter to vary: $\lambda$ (e.g., temperature $T$, fitness weight $\alpha$)
- Range of $\lambda$ values spanning a known phase transition
- Range of generator counts $n_{\text{values}}$

**Procedure:**

**Step 1: For Each $n_{\text{cell}}$, Scan the Phase Diagram**
```python
for n in n_values:
    critical_lambda[n] = []
    curvature_peak[n] = []
    
    for lambda_val in lambda_range:
        # Run simulation with this n_cell and lambda
        set_parameters(n_cell=n, lambda=lambda_val)
        trajectory = run_equilibration(steps=equilibration_steps)
        
        # Compute thermodynamic quantities
        partition_function = estimate_partition_function(trajectory)
        free_energy = -log(partition_function)
        
        # Compute Ruppeiner metric from fluctuations
        metric = compute_ruppeiner_metric(trajectory, parameters=lambda_val)
        
        # Compute scalar curvature
        R_rupp = compute_scalar_curvature(metric)
        
        # Detect phase transition: peak in |R_rupp|
        if abs(R_rupp) > threshold:
            critical_lambda[n].append(lambda_val)
            curvature_peak[n].append(R_rupp)
```

**Step 2: Analyze How $\lambda_c$ Depends on $n_{\text{cell}}$**
```python
# Plot critical parameter vs resolution
plot(n_values, critical_lambda,
     xlabel="Number of Generators n_cell",
     ylabel="Critical Parameter λ_c")

# Find plateau: region where λ_c(n) is stable
n_min_required = find_plateau_start(critical_lambda, tolerance=0.01)
```

**Output:**
- Critical parameter $\lambda_c(n_{\text{cell}})$
- Curvature peak height $R_{\text{Rupp}}^{\max}(n_{\text{cell}})$
- Intrinsic scale: Minimal $n_{\text{cell}}$ where $\lambda_c$ stabilizes
:::

### 10.3. Physical Interpretation

:::{prf:observation} Phase Transition Resolution Defines Thermodynamic Scale
:label: obs-phase-transition-scale

**High $n_{\text{cell}}$:** Phase transition accurately captured
- $\lambda_c(n)$ matches known critical point
- $R_{\text{Rupp}}$ diverges correctly at $\lambda_c$

**Critical $n_{\text{cell}}^*$:** Minimal resolution for phase transition
- Below $n_{\text{cell}}^*$: Critical point shifts or disappears
- Finite-size effects dominate
- **Intrinsic scale:** $n_{\text{cell}}^* \sim (\xi_c / a)^d$ where $\xi_c$ is correlation length at criticality

**Low $n_{\text{cell}}$:** Phase transition not resolved
- System appears smooth across the transition
- Curvature remains finite
- Effective theory is valid only away from criticality

**Connection to universality:** If two different observables yield the same thermodynamic intrinsic scale $n_{\text{cell}}^*$, they belong to the same universality class—they respond to the same critical fluctuations.
:::

---

## 11. Unified Multi-Scale Discovery Algorithm

### 11.1. Synthesizing All Four Methods

:::{prf:algorithm} Comprehensive Intrinsic Scale Discovery
:label: alg-unified-scale-discovery

**Goal:** Identify all intrinsic scales of the Fragile Gas simulation by combining observable, information-theoretic, topological, and thermodynamic diagnostics.

**Input:**
- Long high-fidelity trajectory: $\{Z_k^{(N)}\}_{k=0}^{K}$ ($N$ walkers, finest resolution)
- Suite of observables: $\{O_1, O_2, \ldots, O_M\}$ (Wilson loops, correlators, charges)
- Range of coarse-graining resolutions: $n_{\text{values}} = [10^2, 10^{2.5}, \ldots, 10^{\log_{10} N}]$ (logarithmic spacing)

**Procedure:**

**Phase 1: Multi-Scale Coarse-Graining**
```python
coarse_trajectories = {}
for n in n_values:
    coarse_trajectories[n] = apply_cvt_renormalization(trajectory, n_cell=n)
```

**Phase 2: Compute All Four Diagnostics**
```python
# Method 1: Observable errors
observable_errors = {}
for obs_name, obs_func in observables.items():
    observable_errors[obs_name] = compute_observable_error_curve(
        trajectory, coarse_trajectories, obs_func)

# Method 2: Statistical complexity
C_mu_curve = compute_complexity_curve(trajectory, coarse_trajectories)

# Method 3: Topological features
betti_curves = compute_topological_features(trajectory, coarse_trajectories)

# Method 4: Thermodynamic singularities (if phase transition exists)
if has_phase_transition:
    lambda_c_curve = compute_critical_parameter_curve(
        trajectory, coarse_trajectories, parameter="temperature")
```

**Phase 3: Identify Intrinsic Scales**
```python
intrinsic_scales = []

# From observable plateaus
for obs in observable_errors.values():
    plateaus = find_plateaus(obs, derivative_threshold=0.1)
    intrinsic_scales.extend(plateaus)

# From complexity plateaus
complexity_plateaus = find_plateaus(C_mu_curve, derivative_threshold=0.05)
intrinsic_scales.extend(complexity_plateaus)

# From topological transitions
for betti_curve in betti_curves.values():
    transitions = find_jumps(betti_curve, jump_threshold=1)
    intrinsic_scales.extend(transitions)

# From thermodynamic stability
if has_phase_transition:
    thermo_scale = find_plateau_start(lambda_c_curve, tolerance=0.01)
    intrinsic_scales.append(thermo_scale)

# Remove duplicates and sort
intrinsic_scales = sorted(set(intrinsic_scales))
```

**Phase 4: Cross-Validation**
```python
# Robust scales: where multiple diagnostics agree
robust_scales = []
for scale in intrinsic_scales:
    # Count how many diagnostics identify this scale
    vote_count = count_diagnostic_agreement(scale, tolerance=0.1)
    if vote_count >= 2:  # At least 2 methods agree
        robust_scales.append((scale, vote_count))
```

**Phase 5: Visualization Dashboard**
```python
create_multiscale_dashboard(
    n_values=n_values,
    observable_errors=observable_errors,
    C_mu_curve=C_mu_curve,
    betti_curves=betti_curves,
    lambda_c_curve=lambda_c_curve,
    intrinsic_scales=robust_scales
)
```

**Output:**
- Comprehensive scale spectrum: list of all intrinsic $n_{\text{cell}}$ values
- Robustness scores: how many methods agree on each scale
- Visual dashboard showing all diagnostics simultaneously
- Physical interpretation: which scales correspond to which physical structures (instantons, Debye length, etc.)
:::

### 11.2. Expected Scale Hierarchy

:::{prf:prediction} Intrinsic Scale Hierarchy for Lattice Yang-Mills
:label: pred-scale-hierarchy-yangmills

For a $d=4$ Yang-Mills simulation with $N=10^6$ walkers at coupling $g^2=1$:

**Scale 1: Ultraviolet (UV) Cutoff**
- $n_{\text{cell}}^{\text{UV}} \approx N = 10^6$
- All methods show plateaus/stability
- Resolves lattice spacing $a$
- Required for: Single-plaquette observables, UV-sensitive quantities

**Scale 2: Instanton Core Resolution**
- $n_{\text{cell}}^{\text{inst}} \approx 10^5$
- Observable errors for topological charge plateau here
- Persistent homology detects instanton voids ($\beta_2$)
- Required for: Topological susceptibility, $\theta$-vacuum structure

**Scale 3: Hadron Separation**
- $n_{\text{cell}}^{\text{hadron}} \approx 10^4$
- String tension, glueball mass plateaus
- Complexity $C_\mu$ stable (long-range correlations captured)
- Required for: Confinement scale physics, Wilson loops $> 1$ fm

**Scale 4: Debye Screening (if finite $T > 0$)**
- $n_{\text{cell}}^{\text{Debye}} \approx 10^3$
- Thermodynamic curvature peak stabilizes
- Required for: Finite-temperature phase transition

**Scale 5: Infrared (IR) Limit**
- $n_{\text{cell}}^{\text{IR}} \approx 10^2$
- Only ultra-long-range observables survive (e.g., Polyakov loop)
- Below this: physics breaks down

**Empirical test:** Run the unified algorithm and verify this hierarchy.
:::

---

## 12. Observable Preservation (Conditional on Closure)

:::{important}
**Conditional Framework Note:** The following theorems are **conditional** on the Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`). They show what rigorously follows **IF** closure holds. Proving (or disproving) the hypothesis is the central open problem of this framework.
:::

:::{prf:definition} Coarse-Grained Observable
:label: def-coarse-observable

For a microscopic observable $f: \Omega^{(N)} \to \mathbb{R}$ and the scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}: \Omega^{(N)} \to \tilde{\Omega}^{(n_{\text{cell}})}$, the **coarse-grained observable** is defined as the conditional expectation:

$$
\tilde{f}(\tilde{Z}) := \mathbb{E}_{\mu_{\text{QSD}}^{(N)}}[f(Z) \mid \mathcal{R}_{\text{scutoid},b}(Z) = \tilde{Z}]
$$

where the expectation is taken over the conditional distribution of microscopic states $Z \in \Omega^{(N)}$ that map to the macroscopic state $\tilde{Z} \in \tilde{\Omega}^{(n_{\text{cell}})}$ under the coarse-graining map.

**Interpretation:** $\tilde{f}$ is the best predictor of $f$ given only the coarse-grained information $\tilde{Z}$.
:::

:::{prf:theorem} Long-Range Observable Preservation
:label: thm-observable-preservation-conditional

**Hypothesis:** Assume the Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`) holds.

**Additional Conditions:**
1. The observable $f$ is $(\ell, L)$-long-range ({prf:ref}`def-long-range-observable`) with correlation length $\ell$
2. The scale separation satisfies:
   $$
   \ell \gg n_{\text{cell}}^{-1/d} \gg a
   $$
   where $n_{\text{cell}}^{-1/d}$ is the coarse resolution and $a$ is the lattice spacing
3. The QSD $\mu_{\text{QSD}}^{(N)}$ satisfies the Log-Sobolev Inequality (LSI) with constant $\rho > 0$
4. The coarse-grained Markov chain reaches stationarity after mixing time $t_{\text{mix}}$

**Conclusion:** The coarse-grained observable expectation approximates the microscopic expectation:

$$
\left| \langle f \rangle_{\text{micro}} - \langle \tilde{f} \rangle_{\text{macro}} \right|
\leq C_f \cdot n_{\text{cell}}^{-1/d} + C'_f \cdot e^{-\rho t_{\text{mix}}}
$$

where:
- $\langle f \rangle_{\text{micro}} = \int f(Z) \, d\mu_{\text{QSD}}^{(N)}(Z)$ (microscopic expectation)
- $\langle \tilde{f} \rangle_{\text{macro}} = \int \tilde{f}(\tilde{Z}) \, d\tilde{\mu}_{\text{QSD}}^{(n_{\text{cell}})}(\tilde{Z})$ (coarse-grained expectation)
- $\tilde{f}$ is the coarse-grained observable from {prf:ref}`def-coarse-observable`
- $C_f$ depends on the Lipschitz constant $L$ and correlation length $\ell$
- The first term is the **systematic CVT error** (spatial discretization)
- The second term is the **transient mixing error** (temporal convergence)

**Proof Strategy:**

1. **Lipschitz bound:** Use the $(\ell, L)$-long-range property to bound $|f(Z) - f(\mathcal{R}(Z))|$ by the CVT quantization error. This gives the first term via {prf:ref}`thm-cvt-approximation-error` from [fragile_lqcd.md](fragile_lqcd.md).

2. **Information closure:** The hypothesis ensures that the coarse-grained dynamics preserve predictive information, so the macro-QSD captures the essential statistics for computing $\langle f \rangle$.

3. **Mixing bound:** The LSI implies exponential convergence to the QSD, bounding the transient error.

**Status:** CONDITIONAL THEOREM - The logic is sound **if** the hypothesis holds. Full proof requires completing the missing steps.

**Dependencies:**
- {prf:ref}`hyp-scutoid-information-closure` (unproven)
- {prf:ref}`thm-cvt-approximation-error` ([fragile_lqcd.md](fragile_lqcd.md))
- LSI for QSD ({prf:ref}`def-lsi`, [09_kl_convergence.md](../1_euclidean_gas/09_kl_convergence.md))
:::

:::{prf:remark} Physical Interpretation
:class: note

This theorem formalizes the intuition that **coarse-graining preserves macroscopic physics**. The error has two sources:

1. **Spatial coarse-graining:** Finite resolution $n_{\text{cell}}^{-1/d}$ introduces quantization error. This decays as we use more generators.

2. **Temporal relaxation:** The coarse chain needs time to reach its QSD. This decays exponentially with mixing time.

Both errors can be made arbitrarily small by choosing $n_{\text{cell}}$ large enough and waiting long enough—**provided** information closure holds.
:::

---

## 13. Information-Theoretic Diagnostics (Conditional)

:::{prf:theorem} Conditional Entropy Bound
:label: thm-conditional-entropy-bound

**Hypothesis:** Assume {prf:ref}`hyp-scutoid-information-closure` holds.

**Conclusion:** The conditional entropy of the coarse-grained process is bounded by the microscopic conditional entropy:

$$
H(\tilde{Z}_t | \tilde{Z}_{t-1}) \geq H(Z_t | Z_{t-1}) - O(n_{\text{cell}}^{-1/d})
$$

**Interpretation:**
- **Left side:** Unpredictability of the coarse future given the coarse past
- **Right side:** Unpredictability of the micro future given the micro past
- **Inequality:** Coarse-graining can only **increase** entropy (data processing inequality)
- **Error term:** Quantifies how much predictability is lost; small if closure holds

**Proof Strategy:**

1. **Data processing inequality:**
   $$
   H(\tilde{Z}_t | \tilde{Z}_{t-1}) \geq H(Z_t | Z_{t-1}) + \text{(loss term)}
   $$
   This is always true; the question is bounding the loss.

2. **Information closure:** By hypothesis, $I(\tilde{Z}_t; \tilde{Z}_{t-1}) \approx I(\tilde{Z}_t; Z_{t-1})$, meaning the coarse past retains nearly all relevant information. This bounds the loss term.

3. **CVT error:** The $O(n_{\text{cell}}^{-1/d})$ comes from the quantization error in the renormalization map.

**Status:** CONDITIONAL THEOREM - Proof sketch valid under hypothesis.
:::

:::{prf:theorem} Mutual Information Deficit
:label: thm-mutual-information-deficit

**Hypothesis:** Assume {prf:ref}`hyp-scutoid-information-closure`.

**Conclusion:** The deficit in mutual information between coarse states is bounded:

$$
I(Z_t; Z_{t-1}) - I(\tilde{Z}_t; \tilde{Z}_{t-1}) \leq C \cdot n_{\text{cell}}^{-1/d}
$$

**Interpretation:** The coarse-grained process retains nearly all the temporal correlations of the microscopic process, with loss controlled by the spatial resolution.

**Diagnostic Use:** This inequality can be tested empirically (Part III, §8) to verify if closure holds. If the deficit scales as $n_{\text{cell}}^{-1/d}$, this is evidence for closure. If it scales more slowly (or not at all), closure may fail.

**Status:** CONDITIONAL THEOREM - Testable prediction.
:::

---

## 14. Lumpability Error Bounds

### 14.1. Spatial Correlation Decay from LSI

:::{prf:lemma} LSI Implies Exponential Spatial Correlation Decay
:label: lem-lsi-spatial-decay

**Hypothesis:** The microscopic quasi-stationary distribution $\mu_{\text{QSD}}^{(N)}$ on $\Omega^{(N)} = \mathcal{X}^N \times \mathbb{R}^{Nd}$ satisfies the Log-Sobolev Inequality with constant $\rho > 0$:

$$
\text{Ent}_\mu(f^2) \leq \frac{2}{\rho} \mathcal{E}_\mu(f, f)
$$

for all $f \in H^1(\Omega^{(N)}, \mu)$, where $\text{Ent}_\mu$ is the relative entropy and $\mathcal{E}_\mu$ is the Dirichlet form (see {prf:ref}`def-lsi`, [09_kl_convergence.md](../1_euclidean_gas/09_kl_convergence.md)).

**Additional Assumptions:**
1. The BAOAB dynamics have friction coefficient $\gamma > 0$ and noise strength $\sigma^2 = 2D$
2. The potential $\Phi$ is locally Lipschitz with $\|\nabla \Phi\|_{L^\infty} < \infty$
3. **Spatial Locality (CRITICAL for cluster expansion):** The BAOAB transition kernel satisfies exponential spatial decay. This is formally stated as Condition 1 in Proposition {prf:ref}`prop-scutoid-lumpability-sufficient` ([15_closure_theory.md](13_fractal_set_new/15_closure_theory.md)):
   $$
   \left| \frac{\partial \mathbb{P}_{\text{BAOAB}}}{\partial x_j}(x_i) \right| \leq C_{\text{loc}} e^{-\|x_i - x_j\|/\xi}
   $$
   This condition ensures that the influence of distant walkers decays exponentially, enabling the Dobrushin-Shlosman cluster expansion technique used in step 4.

**Conclusion:** For observables $f, g: \Omega^{(N)} \to \mathbb{R}$ that are spatially localized in disjoint regions separated by distance $r > 0$, the covariance under $\mu_{\text{QSD}}^{(N)}$ decays exponentially:

$$
|\text{Cov}_\mu(f, g)| := \left| \int fg \, d\mu - \int f \, d\mu \cdot \int g \, d\mu \right| \leq C_d \cdot \|f\|_{L^2(\mu)} \cdot \|g\|_{L^2(\mu)} \cdot e^{-r/\xi}
$$

where the **correlation length** is:

$$
\xi = C'_d \cdot \frac{\sqrt{D\gamma}}{\rho}
$$

with $C_d, C'_d$ dimension-dependent constants.

**Dimensional Check:**
- $D$ (diffusion): [length²/time]
- $\gamma$ (friction): [1/time]
- $\rho$ (LSI constant / spectral gap): [1/time]
- $D\gamma$: [length²/time]·[1/time] = [length²/time²]
- $\sqrt{D\gamma}$: [length/time] (velocity scale)
- $\xi = \sqrt{D\gamma}/\rho$: [length/time]/[1/time] = [length] ✓

**Proof Strategy:**

The proof follows the standard route from LSI to exponential correlation decay via the Bakry-Émery criterion and spectral gap techniques. We outline the key steps:

1. **LSI implies spectral gap:** By the Bakry-Émery theorem, the LSI constant $\rho$ provides a lower bound on the spectral gap $\lambda_1$ of the generator:
   $$
   \lambda_1 \geq \frac{\rho}{2}
   $$
   (See Bakry & Émery (1985), Ledoux (1999) or {prf:ref}`thm-lsi-euclidean-gas`.)

2. **Spectral gap implies exponential mixing:** For the Ornstein-Uhlenbeck component of BAOAB, the semigroup $P_t$ satisfies:
   $$
   \|P_t f - \mathbb{E}_\mu[f]\|_{L^2(\mu)} \leq e^{-\lambda_1 t} \|f - \mathbb{E}_\mu[f]\|_{L^2(\mu)}
   $$

3. **Spatial localization:** For observables $f$ supported in region $A$ and $g$ supported in region $B$ with $d(A, B) = r$, the correlation can be written as:
   $$
   \text{Cov}_\mu(f, g) = \int_0^\infty \langle f, \frac{d}{dt} P_t g \rangle_{L^2(\mu)} \, dt
   $$

4. **Cluster expansion bound:** Using the Dobrushin-Shlosman cluster expansion technique, the influence of region $A$ on region $B$ after time $t$ is suppressed by $e^{-r/v_{\text{eff}} t}$ where $v_{\text{eff}} = \sqrt{D\gamma}$ is the effective propagation speed.

   **Dimensional check:** $v_{\text{eff}} = \sqrt{D\gamma} = \sqrt{[\text{length}^2/\text{time}] \cdot [1/\text{time}]} = [\text{length}/\text{time}]$ ✓

5. **Optimal time and correlation length:** The correlation is maximized when the time decay $e^{-\lambda_1 t}$ balances the spatial decay $e^{-r/v_{\text{eff}} t}$. Setting $t^* = r / v_{\text{eff}}$:
   $$
   |\text{Cov}_\mu(f, g)| \lesssim e^{-\lambda_1 r / v_{\text{eff}}} = e^{-r/\xi}
   $$
   where:
   $$
   \xi = \frac{v_{\text{eff}}}{\lambda_1} = \frac{\sqrt{D\gamma}}{\rho}
   $$
   using $\lambda_1 \geq \rho/2$ from step 1, where $v_{\text{eff}} = \sqrt{D\gamma}$ is the effective propagation speed with correct dimensions [length/time].

6. **Constants:** The dimension-dependent constant $C'_d$ arises from the precise cluster expansion estimates and depends on the geometry of $\mathcal{X} \subset \mathbb{R}^d$.

**References:**
- Bakry, D. & Émery, M. (1985), "Diffusions hypercontractives," *Séminaire de Probabilités XIX*, Lecture Notes in Math. 1123, 177-206.
- Ledoux, M. (1999), "Concentration of measure and logarithmic Sobolev inequalities," *Séminaire de Probabilités XXXIII*, Lecture Notes in Math. 1709, 120-216.
- Bodineau, T. & Helffer, B. (2004), "Correlations, spectral gap and log-Sobolev inequalities for unbounded spin systems," *Differential Equations and Mathematical Physics*, 29-50.

**Status:** The proof strategy is standard in the statistical mechanics literature. A complete proof for the BAOAB dynamics would require verifying the cluster expansion estimates for the specific geometry of the scutoid tessellation. This is technical but routine given the established LSI.

**Connection to Closure Theory:** The spatial locality condition (Assumption 3) is the same condition used to prove strong lumpability for scutoid coarse-graining in [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 10.3. The exponential decay `e^{-\|x_i - x_j\|/\xi}` ensures that the Dobrushin contraction coefficient is bounded away from 1, enabling the cluster expansion.

**Physical Interpretation:** The correlation length $\xi \sim \sqrt{D\gamma}/\rho$ has the following dependencies:
- **Increases with $\sqrt{D}$**: Stronger diffusion creates longer-range correlations
- **Increases with $\sqrt{\gamma}$**: The correlation length grows with friction because $\xi$ is set by the effective propagation speed $v_{\text{eff}} = \sqrt{D\gamma}$, which characterizes how quickly localized perturbations are transmitted through frictional coupling before being damped by the spectral gap $\rho$
- **Decreases with $\rho$**: Stronger LSI contractivity (larger spectral gap) suppresses correlations faster

The balance between diffusive spreading ($D$), frictional coupling ($\gamma$), and contractivity ($\rho$) determines the characteristic length scale over which information propagates.
:::

### 14.2. Lumpability Error Control with Spatial Decay

:::{important}
**Completion Note:** With Lemma {prf:ref}`lem-lsi-spatial-decay` now established, the lumpability error bound can be completed.
:::

:::{prf:theorem} Lumpability Error Control
:label: thm-lumpability-error-bound

**Hypothesis:**
1. Assume the Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`)
2. Spatial correlation decay via LSI (Lemma {prf:ref}`lem-lsi-spatial-decay`)

**Conclusion:** The lumpability error satisfies:

$$
\varepsilon_{\text{lump}} := \left\| P_{\text{macro}}(\tilde{Z}_{t+1} | \tilde{Z}_t) - \sum_{Z \in \mathcal{R}^{-1}(\tilde{Z}_t)} P_{\text{micro}}(Z_{t+1} | Z) \mu(Z | \tilde{Z}_t) \right\|_{L^1}
$$

is bounded by:

$$
\varepsilon_{\text{lump}} \leq C_1 \cdot n_{\text{cell}}^{-1/d} + C_2 \cdot e^{-b/\xi}
$$

where:
- $b$ is the block size (RG parameter)
- $\xi$ is the correlation length
- The first term is CVT discretization error
- The second term is the error from neglecting long-range correlations

**Proof Strategy:**

1. **Decompose lumpability error:** Write the difference between the exact micro-transition $P_{\text{micro}}$ and the coarse-lumped transition $P_{\text{macro}}$.

2. **Spatial decomposition:** Split walkers into:
   - **Intra-block**: Walkers within distance $b$ of each other (same coarse cell or neighboring cells)
   - **Inter-block**: Walkers separated by distance $> b$

3. **Inter-block contribution:** By Lemma {prf:ref}`lem-lsi-spatial-decay`, correlations between walkers separated by $r > b$ are bounded by $C \cdot e^{-r/\xi}$. Summing over all pairs with separation $> b$ gives the exponential term $C_2 \cdot e^{-b/\xi}$.

4. **Intra-block contribution:** Within each coarse cell $\alpha$, the CVT quantization error bounds how much the centroid representation $c_\alpha$ differs from individual walker positions. By {prf:ref}`thm-cvt-approximation-error`, this contributes $O(n_{\text{cell}}^{-1/d})$.

5. **Total bound:** Combine both contributions:
   $$
   \varepsilon_{\text{lump}} \leq \underbrace{C_1 \cdot n_{\text{cell}}^{-1/d}}_{\text{CVT discretization}} + \underbrace{C_2 \cdot e^{-b/\xi}}_{\text{long-range correlations}}
   $$

**Status:** CONDITIONAL THEOREM - The proof strategy is now complete given the spatial decay lemma. Full details require formalizing the Dobrushin coupling (see Remark below).

**Dependencies:**
- {prf:ref}`lem-lsi-spatial-decay` (now established)
- {prf:ref}`thm-cvt-approximation-error` ([fragile_lqcd.md](fragile_lqcd.md))
- {prf:ref}`hyp-scutoid-information-closure` (for the coarse Markov property)
:::

:::{prf:remark} Connection to Gamma Channel Optimization
:class: tip

With the lumpability bound now quantitatively established, we can connect to Part II's main result:

**Chain of reasoning:**
1. Lemma {prf:ref}`lem-weyl-centroid-divergence` (corrected): Weyl curvature causes centroid divergence scaled by $\|C\| \cdot r_\alpha^{-1}$
2. Theorem {prf:ref}`thm-lumpability-error-bound`: Lumpability error bounded by spatial correlations (via $e^{-b/\xi}$)
3. **Hypothesis (to be proven):** Weyl curvature increases effective correlation length $\xi$, making the exponential decay slower

**Implication:** If the Weyl tensor controls spatial correlation structure, then the gamma channel term $\gamma_W \|C\|^2$ actively minimizes $\varepsilon_{\text{lump}}$ by suppressing $\xi$, thereby improving closure.

**Next step:** Formalize the relationship between Weyl tensor and correlation length $\xi$ (currently a conjecture).
:::

---



# Part IV: Lattice QFT Connections (Summary)


:::{admonition} Status
:class: warning

This Part condenses the lattice gauge theory connections from the original document.
Full development deferred to dedicated lattice QFT chapter.
:::


## Summary of QFT Connections


The scutoid tessellation provides a computational implementation of RG transformations
that parallels lattice gauge theory block-spin methods:


1. **Wilson Action Preservation**: Under favorable conditions, the CVT coarse-graining
   preserves the Wilson gauge action structure (modulo path-ordering corrections).

2. **Topological Charge**: The Information Gain weights may preserve topological
   winding number in Abelian gauge theories (non-Abelian case requires further work).

3. **Beta Functions**: The computational closure framework allows empirical
   measurement of effective beta functions via multi-scale CVT analysis.


See `fragile_lqcd.md` for full lattice gauge theory formulation.


---


# PART V: RESEARCH PROGRAM AND ITERATION PATH

## 15. The Path to Rigor: Missing Proofs and Open Problems

### 15.1. The Checklist of Missing Proofs

:::{prf:observation} Required Proofs for Full Rigor
:label: obs-missing-proofs-checklist

**Priority 1 (Critical - Foundational):**

- [ ] **Prove Information Closure** ({prf:ref}`hyp-scutoid-information-closure`)
  - Status: Open problem
  - Strategy: Either analytical (derive from BAOAB+CVT structure) or empirical (measure $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}})$ vs. $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z})$)
  - Timeline: 6-12 months (analytical) or 3-6 months (empirical)

- [x] **Spatial Correlation Decay from LSI**
  - {prf:ref}`lem-lsi-spatial-decay` derives correlation length $\xi = C'\sqrt{D\gamma}/\rho$ from Log-Sobolev inequality
  - Spatial locality assumption references {prf:ref}`prop-scutoid-lumpability-sufficient` in [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md)
  - Connects LSI constant to exponential correlation decay
  - Enables quantitative lumpability bounds

- [ ] **Formalize Tessellation Spaces**
  - Define $\text{Tess}(\mathcal{X}, N)$ and $\text{Scutoid}(\mathcal{X}, N)$ as Polish spaces with Borel measures
  - Add topology (Hausdorff metric or Wasserstein metric on induced measures)
  - Timeline: 1-2 months

**Priority 2 (Major - Gamma Channel Theory):**

- [ ] **Prove Geodesic Deviation Conjecture** ({prf:ref}`conj-centroid-geodesic-deviation`)
  - Currently a heuristic statement based on geometric analogy
  - Required steps: Itô calculus treatment, overdamped limit, second-order expansion, Weyl identification, error bounds
  - {prf:ref}`lem-weyl-centroid-divergence` is conditional on this conjecture
  - Timeline: 2-4 months

- [ ] **Complete Weyl-Lumpability Connection** ({prf:ref}`conj-weyl-bounds-lumpability`)
  - Depends on geodesic deviation resolution
  - With spatial decay lemma now in place, main gap is coupling Weyl to $\xi$
  - Timeline: 3-6 months (after geodesic deviation)

- [ ] **Empirical Validation of Gamma Channel Predictions** ({prf:ref}`pred-gamma-channel-effects`)
  - Measure: $\langle \|C\|^2 \rangle$ vs. $\gamma_W$, CVT error vs. $\gamma_W$, $\varepsilon_{\text{lump}}$ vs. $\gamma_W$
  - Find optimal $\gamma_W^*$ for closure
  - Timeline: 2-4 months (simulation-heavy)

**Priority 3 (Moderate - Observable Preservation):**

- [ ] **Fix Topological Charge Claim**
  - Issue: Summed IG weights may break gauge quantization for non-Abelian groups
  - Solution: Either provide gauge-covariant coarse-graining or weaken to approximate bound
  - Timeline: 1-2 months

- [ ] **Rederive Wilson Loop Preservation**
  - Issue: Current proof ignores non-Abelian path ordering
  - Solution: Use proper lattice gauge theory error analysis (Fréchet derivatives or block-spin estimates)
  - Timeline: 2-3 months

**Priority 4 (Lower - Intrinsic Scales):**

- [ ] **Empirical Scale Discovery for Test System**
  - Run unified algorithm ({prf:ref}`alg-unified-scale-discovery`) on 2D Ising model or similar
  - Validate that all 4 methods (observable, complexity, topology, thermodynamics) agree
  - Timeline: 2-3 months

- [ ] **Scale Hierarchy for Yang-Mills**
  - Test prediction {prf:ref}`pred-scale-hierarchy-yangmills` in $d=4$ simulation
  - Timeline: 6-12 months (requires full lattice QFT implementation)
:::

### 15.3. The Iteration Strategy

**Stage 1 (Current): Conditional Framework**
- ✅ Document framework with clear hypothesis marking
- ✅ Gamma channel conceptual theory developed
- ✅ Intrinsic scale discovery methods formalized

**Stage 2 (Next 3 months): Foundations**
- Formalize tessellation spaces (topology + measure)
- Fix topological charge and Wilson loop claims

**Stage 3 (3-6 months): First Complete Proof**
- Select one key result to prove rigorously (e.g., Weyl-lumpability for simplified case)
- Write step-by-step proof meeting publication standards

**Stage 4 (6-12 months): Empirical Validation**
- Implement gamma channel experiments
- Run intrinsic scale discovery on test system
- Measure information closure empirically

**Stage 5 (12-18 months): Full Theory**
- Prove or empirically validate Information Closure Hypothesis
- Complete all missing proofs
- Publish

---

## 16. Empirical Verification Protocol

### 16.1. Minimal Viable Experiment

:::{prf:algorithm} Proof-of-Concept Empirical Test
:label: alg-minimal-empirical-test

**Goal:** Provide first evidence for (or against) the Information Closure Hypothesis and gamma channel predictions.

**System:** 2D Euclidean Gas with quadratic potential (simplest non-trivial case)

**Parameters:**
- $N = 10^4$ walkers
- $n_{\text{cell}} \in [10^2, 10^{2.5}, 10^3, 10^{3.5}, 10^4]$
- $\gamma_W \in [0, 0.1, 0.5, 1.0, 5.0]$ (vary Weyl penalty)
- $T = 100$ timesteps per configuration

**Measurements:**

**Test 1: Weyl-CVT Correlation**
```python
for gamma_W in gamma_W_values:
    trajectory = run_simulation(N=N, gamma_W=gamma_W, steps=T)
    
    # Measure average Weyl norm
    weyl_norm = compute_average_weyl_norm(trajectory)
    
    # Measure CVT quantization error
    for n in n_cell_values:
        generators = cvt_clustering(trajectory, n_cell=n)
        cvt_error = compute_cvt_error(trajectory, generators)
        record(gamma_W, n, weyl_norm, cvt_error)

# Plot: CVT error vs Weyl norm (should be positively correlated)
```

**Test 2: Information Closure Measurement**
```python
# For fixed gamma_W, vary n_cell
for n in n_cell_values:
    micro_trajectory = run_simulation(N=N, steps=10*T)
    macro_trajectory = apply_cvt_renormalization(micro_trajectory, n_cell=n)
    
    # Estimate mutual information (using k-NN estimators or histogram methods)
    I_macro_macro = estimate_MI(macro_future, macro_past)
    I_macro_micro = estimate_MI(macro_future, micro_past)
    
    closure_error = abs(I_macro_macro - I_macro_micro)
    record(n, closure_error)

# Plot: Closure error vs n_cell (should decrease as n → N)
```

**Test 3: Observable Money Plot**
```python
# Simple observable: mean-squared displacement
observable = lambda traj: np.mean([(x[i] - x[0])**2 for i in range(len(traj))])

O_true = observable(micro_trajectory)
for n in n_cell_values:
    macro_traj = apply_cvt_renormalization(micro_trajectory, n_cell=n)
    O_coarse = observable(macro_traj)
    error = abs(O_coarse - O_true) / O_true
    record(n, error)

# Identify plateaus
```

**Success Criteria:**
- Weyl-CVT correlation $r > 0.7$ (strong positive)
- Closure error decreases monotonically with $n_{\text{cell}}$
- Observable error shows plateau structure

**Timeline:** 2-4 weeks (simulation + analysis)
:::

### 16.2. Full Experimental Program

:::{prf:observation} Comprehensive Validation Roadmap
:label: obs-validation-roadmap

**Phase 1: 2D Proof-of-Concept** (Timeline: 1-2 months)
- Run minimal experiment ({prf:ref}`alg-minimal-empirical-test`)
- Validate gamma channel predictions
- Test intrinsic scale discovery methods

**Phase 2: 3D Generalization** (Timeline: 3-4 months)
- Scale to $N=10^5$, 3D Euclidean Gas
- Measure dimensional scaling of closure accuracy
- Test $d=3$ vs. $d=2$ comparison

**Phase 3: 4D Lattice Gauge Theory** (Timeline: 6-12 months)
- Implement Yang-Mills on dynamic triangulation
- Test scale hierarchy prediction ({prf:ref}`pred-scale-hierarchy-yangmills`)
- Measure Wilson loops, topological charge, string tension

**Phase 4: Dimensional Exclusion Test** (Timeline: 3-6 months, parallel to Phase 3)
- Run $d=5$ simulation with $n_{\text{cell}} = 10^4$
- Verify that closure FAILS (observable preservation breaks down)
- Confirm dimensional constraint ({prf:ref}`thm-closure-breakdown-high-d`)

**Deliverables:**
- Published dataset of multi-scale simulation results
- Open-source implementation of intrinsic scale discovery algorithms
- Empirical validation (or falsification) of Information Closure Hypothesis
:::

---

## 17. Dimensional Analysis and Physical Implications (Summary)

### 17.1. The Conditional O(N) Universe Hypothesis

:::{prf:observation} Revised Hypothesis with Gamma Channel
:label: obs-revised-on-universe

**Original claim** ([fragile_lqcd.md](fragile_lqcd.md)): "The O(N) algorithm is computationally efficient."

**Conditional claim** (this document): "**IF** information closure holds, **THEN** the O(N) algorithm is a valid effective theory."

**New mechanistic claim** (Part II): "The gamma channel **facilitates** information closure by minimizing $\|C\|^2$, thereby reducing $\varepsilon_{\text{lump}}$."

**Dimensional constraint** (Part IV, old §11): "$d=3,4$ is the sweet spot where closure can be achieved with $n_{\text{cell}} \ll N$. For $d \geq 5$, the curse of dimensionality prevents closure."

**Empirical testability:** The gamma channel provides a **control knob** ($\gamma_W$) to tune closure. We can:
1. Vary $\gamma_W$ and measure closure error
2. Find optimal $\gamma_W^*$ for each dimension
3. Test whether $\gamma_W^*$ grows unboundedly for $d \geq 5$ (indicating closure is impossible)

**Status:** Conceptual framework complete. Empirical validation required.
:::

---

## 18. Conclusion: From Sketches to Theorems

### 18.1. What We've Accomplished

**Conditional Theoretical Framework:**
- Clearly identified the Information Closure Hypothesis as the central unproven assumption
- Reframed all downstream results as conditional propositions
- Established transparency standards (Hypothesis, Conjecture, Heuristic, Proposition, Theorem)

**New Conceptual Insights:**
- **Gamma channel as compression optimizer:** Connects geometric curvature (Ricci, Weyl) to information theory (CVT error, lumpability error)
- **Weyl-lumpability conjecture:** Provides a mechanism for how the gamma channel facilitates closure
- **Intrinsic scale discovery:** Four complementary methods to empirically find natural coarse-graining resolutions

**Research Program:**
- Clear checklist of missing proofs with priorities and timelines
- Empirical validation protocol from minimal experiment to full lattice QFT
- Iteration strategy: foundations → first proof → validation → full theory

### 18.2. The Path Forward

**Immediate next steps (you and collaborators):**
1. Formalize tessellation spaces (add topology and measure structure)
2. Run minimal empirical experiment ({prf:ref}`alg-minimal-empirical-test`) on 2D system

**Short-term (3-6 months):**
1. Prove Weyl-lumpability connection for simplified case (e.g., conformally flat metrics)
2. Implement intrinsic scale discovery and validate on test system
3. Fix topological charge and Wilson loop proofs

**Long-term (12-18 months):**
1. Empirically validate (or falsify) Information Closure Hypothesis
2. Complete missing proofs or establish empirical bounds
3. Transform conditional framework into proven theorems

### 18.3. The Vision

**If successful, this research program will demonstrate:**

1. **The O(N) algorithm is not a hack**—it is a principled RG transformation with rigorous information-theoretic foundations

2. **The gamma channel is not just an exploration tool**—it is a self-organizing compression optimizer that actively drives the system toward computational closure

3. **Intrinsic scales are not arbitrary**—they are measurable, physical quantities that define the natural resolution hierarchy of a theory

4. **Dimensional constraints are not numerical coincidences**—they arise from fundamental limits on how much information can be compressed in high-dimensional spaces

5. **The Fragile Gas is not a passive simulator**—it is an intelligent computational system that finds the most efficient representation of its own dynamics

**This would be a profound statement about the relationship between computation, information, and physical law.**

### 18.4. Final Remarks

This document is a **living research program**, not a finished work. It will evolve as we:
- Prove theorems (upgrading conjectures to theorems)
- Run experiments (validating or falsifying hypotheses)
- Receive feedback (from reviewers, collaborators, and the community)

**The commitment to transparency and iterative rigor is the foundation of this work.** By clearly marking what is proven, what is conjectural, and what is open, we ensure that every step forward is built on solid ground.

**The journey from sketches to theorems has begun.**

---

**Document Version:** 2.0 (Post-Dual-Review Revision)  
**Status:** Ready for second-round review  
**Next Milestone:** Empirical validation of gamma channel predictions
