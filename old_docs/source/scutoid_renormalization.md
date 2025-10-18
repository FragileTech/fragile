# Scutoid Renormalization: A Conditional Framework for Geometric Compression and RG Flow

**Document Status:** 🚧 Research Program - Iterative Development

**Dual Review Status:**
- ✅ Reviewed by Gemini 2.5 Pro (2025-10-19)
- ✅ Reviewed by Codex (2025-10-19)
- ⚠️ Major revisions required: Reframe as conditional framework, add missing proofs

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

## 3. Micro-Process: The Full N-Walker Scutoid Tessellation

### 3.1. State Space of the Microscopic System

:::{prf:definition} Scutoid State Space (Micro)
:label: def-scutoid-state-space-micro

The **microscopic state** at discrete time $k$ is the full scutoid tessellation of $N$ walkers:

$$
Z_{\text{scutoid}}^{(N)}(k) := (X_k, V_k, \mathcal{V}_k, \mathcal{S}_k)
$$

where:

- **Positions:** $X_k = (x_{1,k}, \ldots, x_{N,k}) \in \mathcal{X}^N$ (walker positions in state space)
- **Velocities:** $V_k = (v_{1,k}, \ldots, v_{N,k}) \in \mathbb{R}^{Nd}$ (walker velocities)
- **Voronoi tessellation:** $\mathcal{V}_k = \{\mathcal{V}_{1,k}, \ldots, \mathcal{V}_{N,k}\}$ (Voronoi cells at time $k$)
- **Scutoid tessellation:** $\mathcal{S}_k = \{S_{1,k}, \ldots, S_{N,k}\}$ (scutoid cells connecting time slices $k$ and $k+1$)

**State space:**

$$
\Omega_{\text{scutoid}}^{(N)} := \mathcal{X}^N \times \mathbb{R}^{Nd} \times \text{Tess}(\mathcal{X}, N) \times \text{Scutoid}(\mathcal{X}, N)
$$

where $\text{Tess}(\mathcal{X}, N)$ is the space of $N$-cell Voronoi tessellations of $\mathcal{X}$ and $\text{Scutoid}(\mathcal{X}, N)$ is the space of $N$-cell scutoid tessellations.

**Geometry encoding:** The scutoid tessellation encodes the emergent Riemannian metric $g_{ab}(x)$ via the deformation tensor ({prf:ref}`def-edge-deformation`, [15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md)).
:::

:::{prf:remark} Deterministic Tessellation
:class: note

The Voronoi and scutoid tessellations are **deterministic functions** of the walker positions:

$$
\mathcal{V}_k = \text{Voronoi}(X_k), \quad \mathcal{S}_k = \text{Scutoid}(X_k, X_{k+1})
$$

Therefore, the full state is redundantly specified. We could work with just $(X_k, V_k)$ and reconstruct $(\mathcal{V}_k, \mathcal{S}_k)$ on demand. However, we include the tessellations explicitly to emphasize that **geometric information** is part of the microscopic state.
:::

### 3.2. Dynamics: The Scutoid Markov Chain

:::{prf:definition} Scutoid Markov Chain (Micro)
:label: def-scutoid-markov-chain-micro

The sequence $\{Z_{\text{scutoid}}^{(N)}(k)\}_{k \geq 0}$ is a **time-homogeneous Markov chain** on $\Omega_{\text{scutoid}}^{(N)}$ with transition kernel:

$$
\mathbb{P}_{\text{scutoid}}^{(N)}(Z_k, A) := P(Z_{k+1} \in A \mid Z_k)
$$

defined by the **BAOAB + cloning** dynamics:

**Step 1: BAOAB Integrator** (Chapter 4, [04_convergence.md](04_convergence.md))

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

**Step 3: Tessellation Update**

Recompute the Voronoi and scutoid tessellations:

$$
\mathcal{V}_{k+1} = \text{Voronoi}(X_{k+1}), \quad \mathcal{S}_{k+1} = \text{Scutoid}(X_k, X_{k+1})
$$

**Markovity:** The transition kernel depends only on the current state $Z_k$, not on the history.

**Stationarity:** Under the quasi-stationary distribution (QSD) $\mu_{\text{QSD}}^{(N)}$, the chain is stationary and ergodic (Theorem {prf:ref}`thm-qsd-existence`, [04_convergence.md](04_convergence.md)).
:::

### 3.3. Observables on the Microscopic System

:::{prf:definition} Microscopic Observables
:label: def-micro-observables

A **microscopic observable** is a measurable function $f: \Omega_{\text{scutoid}}^{(N)} \to \mathbb{R}$.

**Examples:**

1. **Local field values:** $f(Z) = \frac{1}{N} \sum_{i=1}^N \phi(x_i)$ (average of field $\phi$ over walkers)

2. **Wilson loops:** $f(Z) = W_C(Z)$ (holonomy around curve $C$, computed using IG edge weights; see [08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md))

3. **Topological charge:** $f(Z) = Q(Z)$ (Chern number computed from curvature on scutoid tessellation)

4. **Correlation functions:** $f(Z) = G(x, y) = \langle \phi(x) \phi(y) \rangle_Z$ (two-point correlator)

**Long-range vs. short-range observables:**

- **Long-range:** Observables that average over many walkers or large spatial regions (e.g., global Wilson loops, total energy). These are the observables we expect to be preserved under coarse-graining.

- **Short-range:** Observables sensitive to individual walker positions or small-scale fluctuations (e.g., single-walker momentum). These may *not* be preserved.

**Expectation values:** Under the QSD:

$$
\langle f \rangle_{\text{micro}} := \int_{\Omega_{\text{scutoid}}^{(N)}} f(Z) \, d\mu_{\text{QSD}}^{(N)}(Z)
$$
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

:::{prf:heuristic} CVT Error Depends on Cluster Anisotropy
:label: heuristic-cvt-anisotropy

For a cluster of walkers $\{x_1, \ldots, x_k\}$ with centroid $c = \frac{1}{k}\sum_i x_i$, the CVT quantization error is:

$$
E_{\text{CVT}} = \sum_{i=1}^k \|x_i - c\|^2
$$

**Spherical cluster:** If walkers are uniformly distributed on a sphere of radius $r$, then $E_{\text{CVT}} \sim kr^2$.

**Elongated cluster:** If walkers are distributed along a filament of length $L$ (aspect ratio $L/r \gg 1$), then $E_{\text{CVT}} \sim kL^2 \gg kr^2$.

**Weyl measures distortion:** The Weyl tensor quantifies how much a cluster deviates from spherical symmetry. High Weyl $\Rightarrow$ high aspect ratio $\Rightarrow$ high CVT error.

**Gamma channel reduces CVT error:** By minimizing $\|C\|^2$, the gamma channel forces clusters toward spherical geometry, thereby minimizing the information lost when replacing them with centroids.
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

1. **Formalize "cluster shape"**: Define a shape tensor $S_\alpha$ for each coarse cell (e.g., moment of inertia tensor of walker positions)

2. **Analyze BAOAB evolution on curved manifolds**: Derive how the Weyl tensor couples to the shape tensor evolution (geodesic deviation equation)

3. **Bound centroid velocity difference**: Prove that $|\tilde{v}_\alpha(Z) - \tilde{v}_\alpha(Z')| \leq C \|C\|_{L^\infty} \cdot \text{tr}(S_\alpha)$

4. **Integrate to get lumpability error**: Sum the velocity differences over all cells and one timestep to bound $\varepsilon_{\text{lump}}$

**Current status:** Conceptual framework established. Rigorous proof is future work (Part V, §15).

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

### 15.1. Critical Review Summary

**Dual Review Status (2025-10-19):**
- ✅ Gemini 2.5 Pro: Identified central hypothesis as unproven
- ✅ Codex: Identified topological charge claim as potentially incorrect, missing lemmas

**Consensus Critical Issues:**
1. Information Closure Hypothesis ({prf:ref}`hyp-scutoid-information-closure`) is UNPROVEN
2. All downstream "theorems" are conditional on this hypothesis
3. Missing formal definitions (tessellation spaces need topology)
4. Proofs are sketches, not rigorous derivations

**Codex-Specific Issues (Verified and Accepted):**
5. Topological charge "exact preservation" may be wrong for non-Abelian gauge theory
6. Wilson loop proof ignores path ordering (non-commutative holonomies)
7. Missing correlation-length lemma `lem-local-lsi-spatial-decay`

### 15.2. The Checklist of Missing Proofs

:::{prf:observation} Required Proofs for Full Rigor
:label: obs-missing-proofs-checklist

**Priority 1 (Critical - Foundational):**

- [ ] **Prove Information Closure** ({prf:ref}`hyp-scutoid-information-closure`)
  - Status: Open problem
  - Strategy: Either analytical (derive from BAOAB+CVT structure) or empirical (measure $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}})$ vs. $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z})$)
  - Timeline: 6-12 months (analytical) or 3-6 months (empirical)

- [ ] **Formalize Tessellation Spaces**
  - Define $\text{Tess}(\mathcal{X}, N)$ and $\text{Scutoid}(\mathcal{X}, N)$ as Polish spaces with Borel measures
  - Add topology (Hausdorff metric or Wasserstein metric on induced measures)
  - Timeline: 1-2 months

**Priority 2 (Major - Gamma Channel Theory):**

- [ ] **Prove Weyl-Lumpability Connection** ({prf:ref}`conj-weyl-bounds-lumpability`)
  - Current status: Heuristic argument only
  - Required steps:
    1. Formalize cluster shape tensor $S_\alpha$
    2. Derive BAOAB evolution on curved manifolds (geodesic deviation)
    3. Bound centroid velocity difference via Weyl tensor
    4. Integrate to get $\varepsilon_{\text{lump}}$ bound
  - Timeline: 3-6 months

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

- [ ] **Add Missing Correlation-Length Lemma**
  - Define `lem-local-lsi-spatial-decay` relating LSI constant to spatial correlation decay
  - Prove or cite from existing KL-convergence literature
  - Timeline: 1 month

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
- ✅ Document reframed with clear hypothesis marking
- ✅ Gamma channel conceptual theory developed
- ✅ Intrinsic scale discovery methods formalized
- 🚧 Awaiting second-round dual review

**Stage 2 (Next 3 months): Foundations**
- Formalize tessellation spaces (topology + measure)
- Add missing correlation-length lemma
- Fix topological charge and Wilson loop claims

**Stage 3 (3-6 months): First Complete Proof**
- Select one key result to prove rigorously (e.g., Weyl-lumpability for simplified case)
- Write step-by-step proof meeting publication standards
- Submit to dual review for validation

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
1. Second-round dual review of this revised document (especially Part II gamma channel theory)
2. Formalize tessellation spaces (add topology and measure structure)
3. Run minimal empirical experiment ({prf:ref}`alg-minimal-empirical-test`) on 2D system

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
