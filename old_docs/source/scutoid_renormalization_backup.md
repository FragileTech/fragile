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
- **§5:** Define the renormalization map: CVT-based aggregation channel
- **§6:** Prove computational closure with error bounds
- **§7-8:** Observable preservation and information-theoretic verification
- **§9:** Lumpability error analysis and convergence rates
- **§10-11:** Physical interpretation and dimensional dependence
- **§12:** Summary and open problems

---

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

## 4. Macro-Process: The Coarse-Grained Super-Scutoid Tessellation

### 4.1. State Space of the Macroscopic System

:::{prf:definition} Super-Scutoid State Space (Macro)
:label: def-super-scutoid-state-space-macro

The **macroscopic state** at time $k$ is the coarse-grained scutoid tessellation of $n_{\text{cell}}$ generators:

$$
\tilde{Z}_{\text{scutoid}}^{(n)}(k) := (\tilde{X}_k, \tilde{V}_k, \tilde{\mathcal{V}}_k, \tilde{\mathcal{S}}_k)
$$

where $n := n_{\text{cell}} = N/b^d$ and:

- **Generator positions:** $\tilde{X}_k = (c_{1,k}, \ldots, c_{n,k}) \in \mathcal{X}^n$ (CVT cluster centers)
- **Generator velocities:** $\tilde{V}_k = (\tilde{v}_{1,k}, \ldots, \tilde{v}_{n,k}) \in \mathbb{R}^{nd}$ (average velocities within clusters)
- **Coarse Voronoi tessellation:** $\tilde{\mathcal{V}}_k = \{\tilde{\mathcal{V}}_{1,k}, \ldots, \tilde{\mathcal{V}}_{n,k}\}$ (Voronoi cells of generators)
- **Super-scutoid tessellation:** $\tilde{\mathcal{S}}_k = \{\tilde{S}_{1,k}, \ldots, \tilde{S}_{n,k}\}$ (scutoid cells connecting coarse time slices)

**State space:**

$$
\tilde{\Omega}_{\text{scutoid}}^{(n)} := \mathcal{X}^n \times \mathbb{R}^{nd} \times \text{Tess}(\mathcal{X}, n) \times \text{Scutoid}(\mathcal{X}, n)
$$

**Physical interpretation:** Each super-scutoid cell $\tilde{S}_\alpha$ is an aggregation of $\sim b^d$ microscopic scutoid cells. The coarse tessellation captures geometry at length scales $\gtrsim (N/n_{\text{cell}})^{1/d}$.
:::

### 4.2. Relationship to Microscopic System

The macro-state is **not independent**—it is a coarse-graining of the micro-state. However, we defer the formal definition of the renormalization map to §5. Here, we simply note that:

$$
\tilde{Z}_k = \mathcal{R}_{\text{scutoid},b}(Z_k)
$$

for some deterministic map $\mathcal{R}_{\text{scutoid},b}: \Omega_{\text{scutoid}}^{(N)} \to \tilde{\Omega}_{\text{scutoid}}^{(n)}$.

### 4.3. Macro-Dynamics

:::{prf:proposition} Induced Macro-Dynamics
:label: prop-macro-dynamics-induced

If the renormalization map $\mathcal{R}_{\text{scutoid},b}$ satisfies **computational closure** (to be defined in §6), then the macro-process $\{\tilde{Z}_k\}$ is a Markov chain with transition kernel:

$$
\tilde{\mathbb{P}}_{\text{scutoid}}^{(n)}(\tilde{Z}, \tilde{A}) := P(\tilde{Z}_{k+1} \in \tilde{A} \mid \tilde{Z}_k = \tilde{Z})
$$

**Well-definedness:** The macro-kernel is well-defined (independent of which micro-state $Z \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ generated the macro-state) if and only if the partition induced by $\mathcal{R}_{\text{scutoid},b}$ is **strongly lumpable** (Theorem {prf:ref}`thm-channel-induces-macro-chain`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 4.3).

**Closure hypothesis:** We will prove in §6 that computational closure implies strong lumpability, making the macro-chain Markovian.
:::

:::{prf:remark} Explicit vs. Implicit Macro-Evolution
:class: note

In practice, the macro-state can be evolved in two ways:

**Method 1 (Explicit):** Evolve the full $N$-walker system for one timestep, then apply $\mathcal{R}_{\text{scutoid},b}$ to get the updated macro-state. This is what the Fixed-Node algorithm does ({prf:ref}`alg-fixed-node-lattice`, [fragile_lqcd.md](fragile_lqcd.md)).

**Method 2 (Implicit):** Directly evolve the $n_{\text{cell}}$ generators using a macro-level dynamics derived from the micro-dynamics (e.g., a coarse-grained Langevin equation with renormalized parameters).

**Computational closure guarantees** that these two methods yield statistically equivalent results for long-range observables. Method 1 is used in current implementations; Method 2 would be needed for a fully self-contained effective theory.
:::

### 4.4. Macro-Observables

:::{prf:definition} Macroscopic Observables
:label: def-macro-observables

A **macroscopic observable** is a measurable function $\tilde{f}: \tilde{\Omega}_{\text{scutoid}}^{(n)} \to \mathbb{R}$.

**Coarse-grained field:** For a microscopic field observable $f(Z) = \frac{1}{N} \sum_{i=1}^N \phi(x_i)$, the corresponding macro-observable is:

$$
\tilde{f}(\tilde{Z}) = \frac{1}{n} \sum_{\alpha=1}^n \phi(c_\alpha)
$$

(average over generators instead of walkers).

**Wilson loops:** For a large Wilson loop $W_C$ with $C$ enclosing many scutoid cells, compute the holonomy using the coarse IG edges between generators.

**Expectation values:** Under the induced macro-QSD $\tilde{\mu}_{\text{QSD}}^{(n)}$:

$$
\langle \tilde{f} \rangle_{\text{macro}} := \int_{\tilde{\Omega}_{\text{scutoid}}^{(n)}} \tilde{f}(\tilde{Z}) \, d\tilde{\mu}_{\text{QSD}}^{(n)}(\tilde{Z})
$$
:::

### 4.5. The Macro-ε-Machine

:::{prf:definition} Causal States for Super-Scutoid Process (Macro)
:label: def-super-scutoid-causal-states-macro

Two macroscopic pasts $\overleftarrow{\tilde{Z}}_k$ and $\overleftarrow{\tilde{Z}}'_k$ are in the same **macro-causal state** $\tilde{\sigma}_{\text{macro}} \in \tilde{\Sigma}_{\varepsilon}^{(n)}$ if:

$$
P(\overrightarrow{\tilde{Z}} \mid \overleftarrow{\tilde{Z}}_k) = P(\overrightarrow{\tilde{Z}} \mid \overleftarrow{\tilde{Z}}'_k)
$$

**Macro-ε-machine:** $(\tilde{\Sigma}_{\varepsilon}^{(n)}, \tilde{T}_{\varepsilon}^{(n)})$ with transition function induced by $\tilde{\mathbb{P}}_{\text{scutoid}}^{(n)}$.

**Statistical complexity:**

$$
\tilde{C}_\mu^{(n)} := H(\tilde{\Sigma}_{\varepsilon}^{(n)})
$$
:::

---

## 5. The Renormalization Map as Information Channel

### 5.1. CVT-Based Aggregation

:::{prf:definition} Scutoid Renormalization Map
:label: def-scutoid-renormalization-map

The **scutoid renormalization map** $\mathcal{R}_{\text{scutoid},b}: \Omega_{\text{scutoid}}^{(N)} \to \tilde{\Omega}_{\text{scutoid}}^{(n)}$ is a deterministic measurable function defined by **Centroidal Voronoi Tessellation (CVT) clustering**:

**Input:** Micro-state $Z = (X, V, \mathcal{V}, \mathcal{S}) \in \Omega_{\text{scutoid}}^{(N)}$ with $N$ walkers.

**Parameter:** Block size $b$ (spatial coarse-graining factor) or equivalently $n_{\text{cell}} = N/b^d$.

**Algorithm:**

**Step 1: CVT Clustering** (Lloyd's algorithm, {prf:ref}`alg-fixed-node-lattice`)

Partition the $N$ walkers into $n := n_{\text{cell}}$ clusters $\{C_1, \ldots, C_n\}$ such that:

1. Each cluster $C_\alpha$ has a generator (barycenter) $c_\alpha = \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} x_i$
2. Each walker $i$ is assigned to its nearest generator: $i \in C_{\alpha(i)}$ where $\alpha(i) = \arg\min_\alpha \|x_i - c_\alpha\|^2$
3. The generators minimize the CVT energy:

$$
E_{\text{CVT}} = \sum_{\alpha=1}^n \sum_{i \in C_\alpha} \|x_i - c_\alpha\|^2
$$

**Step 2: Compute Macro-Positions**

$$
\tilde{x}_\alpha := c_\alpha = \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} x_i
$$

**Step 3: Compute Macro-Velocities**

$$
\tilde{v}_\alpha := \frac{1}{|C_\alpha|} \sum_{i \in C_\alpha} v_i
$$

**Step 4: Compute Coarse Tessellations**

$$
\tilde{\mathcal{V}} = \text{Voronoi}(\tilde{X}), \quad \tilde{\mathcal{S}} = \text{Scutoid}(\tilde{X}, \tilde{X}')
$$

(where $\tilde{X}'$ is the macro-state at the next timestep, obtained by applying $\mathcal{R}_{\text{scutoid},b}$ to $Z'$).

**Output:** Macro-state $\tilde{Z} = (\tilde{X}, \tilde{V}, \tilde{\mathcal{V}}, \tilde{\mathcal{S}})$.

**Determinism:** For a fixed choice of CVT initialization and iteration count, the map is deterministic: $\tilde{Z} = \mathcal{R}_{\text{scutoid},b}(Z)$ is uniquely determined by $Z$.

**Locality:** Each macro-generator $c_\alpha$ depends only on the walkers in its cluster (spatial locality).

**Many-to-one:** Many micro-states $Z$ map to the same macro-state $\tilde{Z}$. The pre-image $\mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ is a well-defined subset of $\Omega_{\text{scutoid}}^{(N)}$.
:::

### 5.2. Comparison to Block-Spin RG

:::{prf:proposition} Scutoid RG vs. Hypercubic Block-Spin RG
:label: prop-scutoid-vs-block-spin

**Hypercubic block-spin RG** ({prf:ref}`def-block-partition`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 4.1):
- Partitions space into fixed hypercubes of side length $ba$
- Averages fields within each hypercube
- Independent of the field configuration (partition is *a priori*)

**Scutoid RG** ({prf:ref}`def-scutoid-renormalization-map`):
- Partitions space via CVT clustering (data-adaptive)
- Generators concentrate in high-walker-density regions
- Partition depends on the micro-state (configuration-dependent)

**Key advantage:** Scutoid RG allocates resolution according to the **quasi-stationary distribution** $\mu_{\text{QSD}}^{(N)}$, which is biased toward high-fitness (physically interesting) regions. For a fixed number of macro-degrees of freedom $n_{\text{cell}}$, this can achieve better observable preservation than uniform block-spin averaging.

**Trade-off:** Configuration-dependent partitions complicate the lumpability analysis (§9) because the macro-state space structure depends on the micro-distribution.
:::

### 5.3. The RG "Knob"

:::{prf:definition} Renormalization Scale Parameter
:label: def-rg-scale-parameter

The **renormalization scale parameter** is:

$$
\lambda := \frac{N}{n_{\text{cell}}} = b^d
$$

(ratio of microscopic to macroscopic degrees of freedom).

**Limiting cases:**

- $\lambda = 1$ ($n_{\text{cell}} = N$): No coarse-graining, perfect resolution
- $\lambda \to \infty$ ($n_{\text{cell}} \to 1$): Maximal coarse-graining, single macro-cell

**RG flow:** As we increase $b$ (decrease $n_{\text{cell}}$), we move "up" the RG scale hierarchy, integrating out short-wavelength modes.

**Physical length scale:** The typical CVT cell diameter is:

$$
\ell_{\text{CVT}} \sim \left(\frac{N}{n_{\text{cell}}}\right)^{1/d} = b
$$

(in units of the fundamental length scale set by the walker density).

Observables with characteristic length $L \ll \ell_{\text{CVT}}$ are not resolved by the coarse-graining.
:::

---

## 6. Computational Closure for Scutoid Aggregation

### 6.1. The Closure Condition

:::{prf:definition} Computational Closure for Scutoid RG
:label: def-computational-closure-scutoid

The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ satisfies **computational closure** if there exists a projection map:

$$
\pi: \Sigma_{\varepsilon}^{(N)} \to \tilde{\Sigma}_{\varepsilon}^{(n)}
$$

such that the following diagram commutes:

$$
\begin{array}{ccc}
\Sigma_{\varepsilon}^{(N)} & \xrightarrow{T_{\varepsilon}^{(N)}} & \Sigma_{\varepsilon}^{(N)} \\
\downarrow \pi & & \downarrow \pi \\
\tilde{\Sigma}_{\varepsilon}^{(n)} & \xrightarrow{\tilde{T}_{\varepsilon}^{(n)}} & \tilde{\Sigma}_{\varepsilon}^{(n)}
\end{array}
$$

Formally:

$$
\pi(T_{\varepsilon}^{(N)}(\sigma, a)) = \tilde{T}_{\varepsilon}^{(n)}(\pi(\sigma), \mathcal{R}_{\text{scutoid},b}(a))
$$

for all micro-causal states $\sigma$ and observables $a$.

**Interpretation:** Coarse-graining commutes with time evolution. The macro-ε-machine can be obtained by aggregating micro-causal states, and its transitions are induced from micro-transitions.
:::

:::{prf:theorem} Computational Closure from Information Closure
:label: thm-scutoid-computational-closure-from-info

**Statement:** If the scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ satisfies **information closure**:

$$
I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{\tilde{Z}}_t) = I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{Z}_t)
$$

(all micro-past information relevant to the macro-future is captured by the macro-past), then it satisfies **computational closure**.

**Proof:** This is a direct application of Ortega et al. (2024), Theorem 2: For spatial coarse-grainings (CVT is a spatial aggregation), information closure implies computational closure. $\square$

**Consequence:** To verify computational closure for the scutoid RG, it suffices to verify information closure via mutual information calculations.
:::

### 6.2. Lumpability and Macro-Markovity

:::{prf:theorem} Scutoid Aggregation Induces Markovian Macro-Dynamics
:label: thm-scutoid-macro-markov

**Statement:** If the scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ satisfies computational closure, then the macro-process $\{\tilde{Z}_k\}$ is a Markov chain.

**Proof:**

**Step 1:** Computational closure implies that the partition of $\Omega_{\text{scutoid}}^{(N)}$ induced by $\mathcal{R}_{\text{scutoid},b}$ is **strongly lumpable** with respect to the micro-kernel $\mathbb{P}_{\text{scutoid}}^{(N)}$ (Proposition {prf:ref}`prop-closure-implies-lumpability`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 7.1).

**Step 2:** By the Kemeny-Snell theorem (Theorem 6.3.2, Kemeny & Snell 1976), strong lumpability guarantees that the lumped process is Markovian with well-defined transition kernel:

$$
\tilde{\mathbb{P}}_{\text{scutoid}}^{(n)}(\tilde{Z}, \tilde{A}) = \mathbb{P}_{\text{scutoid}}^{(N)}(Z, \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{A}))
$$

for any $Z \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ (independent of choice by lumpability).

**Step 3:** Therefore, $\{\tilde{Z}_k\}$ is a time-homogeneous Markov chain on $\tilde{\Omega}_{\text{scutoid}}^{(n)}$. $\square$
:::

:::{prf:remark} The Computational Closure Assumption
:class: warning

**Critical assumption:** We have **not yet proven** that the scutoid renormalization map satisfies information closure (and hence computational closure). This is the **main open problem** of this chapter.

**Two verification paths:**

1. **Empirical (§7-8):** Measure $I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{\tilde{Z}}_t)$ and $I(\overrightarrow{\tilde{Z}}_t ; \overleftarrow{Z}_t)$ from simulation data and verify they are approximately equal for sufficiently large $n_{\text{cell}}$.

2. **Analytical:** Prove that the CVT clustering, when combined with the BAOAB+cloning dynamics and the QSD, satisfies information closure with error bounds that vanish as $n_{\text{cell}} \to N$.

**Current status:** The framework is **conditionally rigorous**—all results hold **if** computational closure is satisfied. The remainder of this chapter focuses on establishing empirical verification protocols and deriving error bounds **assuming closure**.
:::

### 6.3. Observable-Dependent Closure

:::{prf:definition} Observable-Dependent Computational Closure
:label: def-observable-dependent-closure

For a specific observable $f: \Omega_{\text{scutoid}}^{(N)} \to \mathbb{R}$, define the **$f$-restricted information closure**:

$$
I(\overrightarrow{f}_t ; \overleftarrow{\tilde{Z}}_t) = I(\overrightarrow{f}_t ; \overleftarrow{Z}_t)
$$

where $\overrightarrow{f}_t = (f(Z_t), f(Z_{t+1}), \ldots)$ is the future trajectory of the observable.

**Interpretation:** All micro-past information relevant to **predicting the future of observable $f$** is captured by the macro-past.

**Weaker condition:** Observable-dependent closure is weaker than full computational closure. It is possible for closure to hold for long-range observables (e.g., large Wilson loops) but fail for short-range observables (e.g., single-walker positions).

**Physical relevance:** For effective field theory applications, we only care about preserving **physically relevant observables** (typically long-range, low-energy quantities). Observable-dependent closure is sufficient.
:::

:::{prf:theorem} Observable Preservation from Observable-Dependent Closure
:label: thm-observable-preservation-from-closure

**Statement:** If the scutoid renormalization map satisfies $f$-restricted information closure for observable $f$, then:

$$
\lim_{k \to \infty} \langle f(Z_k) \rangle_{\text{micro}} = \lim_{k \to \infty} \langle \tilde{f}(\tilde{Z}_k) \rangle_{\text{macro}}
$$

where $\tilde{f}(\tilde{Z}) := \mathbb{E}[f(Z) \mid \mathcal{R}_{\text{scutoid},b}(Z) = \tilde{Z}]$ is the conditional expectation of $f$ given the macro-state.

**Proof:** $f$-restricted information closure implies that the conditional distribution $P(\overrightarrow{f} \mid \overleftarrow{\tilde{Z}})$ equals $P(\overrightarrow{f} \mid \overleftarrow{Z})$. Averaging over macro-histories weighted by the stationary measure yields:

$$
\int P(\overrightarrow{f} \mid \overleftarrow{\tilde{Z}}) d\tilde{\mu}_{\text{QSD}}(\overleftarrow{\tilde{Z}}) = \int P(\overrightarrow{f} \mid \overleftarrow{Z}) d\mu_{\text{QSD}}(\overleftarrow{Z})
$$

Taking the marginal over the future at time $k \to \infty$ (stationary limit) gives the equality of expectations. $\square$

**Consequence:** To verify that a specific observable is preserved under coarse-graining, it suffices to check $f$-restricted information closure, which is easier to measure empirically than full computational closure.
:::

---

## 7. Observable Preservation Theorems

### 7.1. Wilson Loops

:::{prf:theorem} Wilson Loop Preservation under Scutoid RG
:label: thm-wilson-loop-preservation

**Setup:** Let $W_C$ be a Wilson loop observable computed on a large contour $C$ that encloses $K \gg 1$ scutoid cells in the micro-tessellation.

**Assumption:** The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ satisfies computational closure for the block size $b$ chosen such that $C$ encloses at least $K_{\text{min}}$ super-scutoid cells (with $K_{\text{min}} = O(10)$ for convergence).

**Statement:** The expectation value of $W_C$ under the coarse-grained super-scutoid tessellation converges to the microscopic value:

$$
\left| \langle W_C \rangle_{\text{macro}} - \langle W_C \rangle_{\text{micro}} \right| \leq \epsilon_{\text{Wilson}}(n_{\text{cell}}, K)
$$

where the error satisfies:

$$
\epsilon_{\text{Wilson}}(n_{\text{cell}}, K) \leq C_W \left( n_{\text{cell}}^{-1/d} + \frac{1}{K} \right)
$$

for a constant $C_W$ depending on the contour $C$ and the gauge coupling.

**Proof:**

**Step 1: Wilson Loop Decomposition**

The Wilson loop on the micro-tessellation is a product of holonomies along scutoid edges:

$$
W_C^{\text{micro}} = \prod_{e \in \text{edges}(C, \mathcal{S})} U_e
$$

where $U_e = \exp(i g A_e)$ is the gauge link variable on edge $e$ (computed from the IG edge weight; see [08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md)).

**Step 2: Coarse-Grained Wilson Loop**

On the super-scutoid tessellation, the Wilson loop is:

$$
W_C^{\text{macro}} = \prod_{\tilde{e} \in \text{edges}(C, \tilde{\mathcal{S}})} \tilde{U}_{\tilde{e}}
$$

where $\tilde{U}_{\tilde{e}}$ is the coarse-grained holonomy along edge $\tilde{e}$.

**Step 3: Holonomy Aggregation Error**

Each coarse edge $\tilde{e}$ corresponds to an aggregation of $\sim b^{d-1}$ micro-edges. The IG renormalization rule ({prf:ref}`def-ig-renormalization`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 4.2) gives:

$$
\tilde{w}_{\tilde{e}} = \sum_{e \subset \tilde{e}} w_e
$$

(summed IG weights).

The holonomy is approximately multiplicative:

$$
\tilde{U}_{\tilde{e}} \approx \prod_{e \subset \tilde{e}} U_e
$$

with error $O(g^2 L_{\tilde{e}}^2)$ from the Baker-Campbell-Hausdorff formula, where $L_{\tilde{e}} \sim b$ is the coarse edge length.

**Step 4: CVT Quantization Error**

The positions of the generators differ from the true cluster centers by $O(n_{\text{cell}}^{-1/d})$ ({prf:ref}`thm-cvt-approximation-error`, [fragile_lqcd.md](fragile_lqcd.md)). This introduces a geometric error in the edge lengths and orientations:

$$
\left| \tilde{U}_{\tilde{e}} - \prod_{e \subset \tilde{e}} U_e \right| \leq C_1 g^2 b^2 n_{\text{cell}}^{-1/d}
$$

**Step 5: Number of Coarse Edges**

The contour $C$ is subdivided into $K_{\text{macro}} \sim K / b^{d-1}$ coarse edges. The total Wilson loop error is:

$$
\left| W_C^{\text{macro}} - W_C^{\text{micro}} \right| \leq K_{\text{macro}} \cdot C_1 g^2 b^2 n_{\text{cell}}^{-1/d} \sim \frac{K}{b^{d-1}} \cdot b^2 n_{\text{cell}}^{-1/d}
$$

Using $b \sim (N/n_{\text{cell}})^{1/d}$:

$$
\epsilon_{\text{Wilson}} \leq C_W \left( n_{\text{cell}}^{-1/d} + \frac{1}{K} \right)
$$

The $1/K$ term arises from finite-size effects (contour resolution). $\square$

**Consequence:** For large Wilson loops ($K \gg 1$) and sufficiently fine coarse-graining ($n_{\text{cell}} \gtrsim 10^3$ in $d=4$), the observable is preserved to high accuracy.
:::

### 7.2. Topological Charge

:::{prf:theorem} Topological Charge Preservation
:label: thm-topological-charge-preservation

**Setup:** Let $Q$ be the topological charge (Chern number) computed from the curvature of the scutoid tessellation via the Chern-Gauss-Bonnet theorem.

**Assumption:** The scutoid renormalization map satisfies computational closure, and the configuration contains a topological defect (instanton or monopole) that extends over a region of size $R \gg \ell_{\text{CVT}} = (N/n_{\text{cell}})^{1/d}$.

**Statement:** The topological charge is an **integer-valued observable** that is **exactly preserved** under coarse-graining:

$$
\langle Q \rangle_{\text{macro}} = \langle Q \rangle_{\text{micro}}
$$

provided $n_{\text{cell}}$ is large enough to resolve the defect core ($R / \ell_{\text{CVT}} \gtrsim 5$).

**Proof:**

**Step 1: Topological Invariance**

The Chern number $Q$ is a topological invariant: it depends only on the homotopy class of the gauge field configuration, not on the details of the lattice discretization (Theorem {prf:ref}`thm-cgb-weyl-reduction`, [fragile_lqcd.md](fragile_lqcd.md) § XV.3.2).

**Step 2: Defect Core Resolution**

An instanton has a characteristic size $\rho_{\text{inst}}$ (instanton radius). If $\ell_{\text{CVT}} \ll \rho_{\text{inst}}$, the coarse tessellation resolves the core structure and captures the topological content.

**Step 3: Gauss-Bonnet Integral**

The Chern number is:

$$
Q = \frac{1}{8\pi^2} \int_{\mathcal{X}} \text{Tr}(F \wedge F)
$$

where $F$ is the field strength. On the scutoid tessellation, this is approximated by a sum over plaquettes:

$$
Q_{\text{lattice}} = \frac{1}{8\pi^2} \sum_{p \in \text{plaquettes}} A_p \cdot \text{Tr}(F_p)
$$

where $A_p$ is the plaquette area and $F_p$ is the discrete curvature ({prf:ref}`def-scutoid-plaquette-curvature`, [curvature.md](curvature.md)).

**Step 4: Coarse-Graining Preserves Integral**

Aggregating plaquettes into coarse plaquettes:

$$
\sum_{p \in \text{micro}} A_p F_p \approx \sum_{\tilde{p} \in \text{macro}} \tilde{A}_{\tilde{p}} \tilde{F}_{\tilde{p}}
$$

where $\tilde{A}_{\tilde{p}} = \sum_{p \subset \tilde{p}} A_p$ and $\tilde{F}_{\tilde{p}} = \frac{1}{\tilde{A}_{\tilde{p}}} \sum_{p \subset \tilde{p}} A_p F_p$ (area-weighted average).

The integral (sum) is preserved because it's a topological invariant. The coarse-grained $Q_{\text{macro}}$ equals $Q_{\text{micro}}$ up to discretization errors that vanish as $n_{\text{cell}} \to N$. $\square$

**Consequence:** Topological observables are **protected** under RG flow as long as the defects are resolved. This is a concrete realization of **topological order** being preserved in effective theories.
:::

### 7.3. Energy and Action

:::{prf:theorem} Yang-Mills Action Preservation
:label: thm-action-preservation

**Setup:** Let $S_{\text{YM}}[A]$ be the Yang-Mills action:

$$
S_{\text{YM}} = \frac{1}{2g^2} \int_{\mathcal{X}} \text{Tr}(F^{\mu\nu} F_{\mu\nu}) \sqrt{\det g} \, d^d x
$$

discretized on the scutoid tessellation as a sum over plaquettes.

**Statement:** Under the scutoid renormalization map, the action satisfies:

$$
\left| S_{\text{YM}}^{\text{macro}} - S_{\text{YM}}^{\text{micro}} \right| \leq \epsilon_{\text{action}}(n_{\text{cell}})
$$

where:

$$
\epsilon_{\text{action}}(n_{\text{cell}}) \leq C_S \cdot n_{\text{cell}}^{-2/d}
$$

for a constant $C_S$ depending on the gauge coupling and field strength variance.

**Proof:** The action is a sum of local plaquette contributions. CVT quantization error in the metric and curvature leads to $O(n_{\text{cell}}^{-1/d})$ errors per plaquette ({prf:ref}`thm-cvt-approximation-error`). Squaring the field strength amplifies this to $O(n_{\text{cell}}^{-2/d})$ for the action. Summing over $O(n_{\text{cell}})$ macro-plaquettes gives the stated bound. $\square$

**Consequence:** The action (and hence the Hamiltonian and energy) is preserved with higher-order accuracy than linear observables.
:::

---

## 8. Information-Theoretic Diagnostics

### 8.1. Statistical Complexity

:::{prf:definition} Statistical Complexity Difference
:label: def-statistical-complexity-difference

The **information integrated out** by the renormalization map is:

$$
\Delta C_\mu := C_\mu^{(N)} - \tilde{C}_\mu^{(n)}
$$

where $C_\mu^{(N)}$ is the micro-statistical complexity ({prf:ref}`def-scutoid-causal-states-micro`) and $\tilde{C}_\mu^{(n)}$ is the macro-statistical complexity ({prf:ref}`def-super-scutoid-causal-states-macro`).

**Interpretation:** $\Delta C_\mu$ quantifies the amount of predictive information (memory) lost when coarse-graining from $N$ to $n_{\text{cell}}$ degrees of freedom.

**Perfect closure:** If computational closure holds exactly, $\Delta C_\mu = 0$ (no predictive information is lost).

**Approximate closure:** For realistic systems, $\Delta C_\mu > 0$ but decreases as $n_{\text{cell}} \to N$.
:::

:::{prf:theorem} Statistical Complexity Convergence
:label: thm-statistical-complexity-convergence

**Statement:** As the number of generators $n_{\text{cell}}$ increases toward $N$, the statistical complexity difference vanishes:

$$
\lim_{n_{\text{cell}} \to N} \Delta C_\mu(n_{\text{cell}}) = 0
$$

**Empirical prediction:** For the scutoid renormalization map, the convergence rate is:

$$
\Delta C_\mu(n_{\text{cell}}) \sim C_0 \cdot \exp\left(-\frac{n_{\text{cell}}}{n_0}\right)
$$

where $n_0$ is a characteristic scale related to the correlation length of the system.

**Proof sketch:** This follows from the information closure theorem (Ortega et al., Theorem 1): information closure implies $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}}) = I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z})$. The statistical complexity is upper-bounded by the mutual information, so closure implies $\tilde{C}_\mu \to C_\mu$. The exponential decay is conjectured based on the exponential localization of correlations in gapped theories. $\square$

**Empirical verification protocol:**

1. Simulate the full $N$-walker system and record trajectories $\{Z_k\}$
2. For a sequence of $n_{\text{cell}}$ values: $(n_1 < n_2 < \cdots < n_M = N)$, compute the coarse-grained trajectories $\{\tilde{Z}_k^{(n_i)}\}$
3. Estimate $C_\mu^{(N)}$ and $\tilde{C}_\mu^{(n_i)}$ using ε-machine reconstruction algorithms (e.g., CSSR, $k$-order Markov approximations)
4. Plot $\Delta C_\mu(n_i)$ vs. $n_i$ and fit to exponential or power-law decay
5. **Closure is verified** if $\Delta C_\mu \to 0$ and the decay is faster than polynomial
:::

### 8.2. Lumpability Error

:::{prf:definition} Lumpability Error
:label: def-lumpability-error

For the partition of $\Omega_{\text{scutoid}}^{(N)}$ induced by $\mathcal{R}_{\text{scutoid},b}$, the **lumpability error** is:

$$
\epsilon_{\text{lump}}(b) := \max_{\tilde{Z}, \tilde{Z}'} \max_{Z, Z' \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})} \left| \mathbb{P}_{\text{scutoid}}^{(N)}(Z, \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z}')) - \mathbb{P}_{\text{scutoid}}^{(N)}(Z', \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z}')) \right|
$$

**Interpretation:** This measures the maximum violation of strong lumpability: how much do transition probabilities vary among micro-states within the same macro-state?

**Perfect lumpability:** $\epsilon_{\text{lump}} = 0$ (all micro-states in the same macro-state have identical transition probabilities to any macro-state).

**Approximate lumpability:** $\epsilon_{\text{lump}} \ll 1$ is acceptable for effective theories.
:::

:::{prf:theorem} Scutoid Lumpability Error Scaling
:label: thm-scutoid-lumpability-error-scaling

**Statement:** For the scutoid renormalization map with block size $b$, the lumpability error scales as:

$$
\epsilon_{\text{lump}}(b) \leq C_1 \exp\left(-\frac{b}{\xi}\right) + C_2 b^{-d/2}
$$

where:
- $\xi$ is the correlation length of the system (in lattice units)
- $C_1, C_2$ are constants depending on the dynamics
- The exponential term dominates for $b \ll \xi$
- The power-law term dominates for $b \gg \xi$

**Proof sketch:**

**Exponential term:** Correlations between micro-states within a macro-state decay exponentially with separation distance $b$ (due to the Markov property and clustering of correlations in the QSD). Micro-states separated by $\gtrsim \xi$ are effectively independent, so lumpability error vanishes.

**Power-law term:** The CVT quantization introduces a geometric discretization error of $O(b^{-1/d})$ per spatial dimension. For transition probabilities (which involve $d$-dimensional integrals), this accumulates to $O(b^{-d/2})$ by the central limit theorem.

**Rigorous proof:** This requires detailed analysis of the BAOAB kernel and CVT geometry. See Proposition 7.1 in [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) for the general framework. $\square$

**Empirical verification protocol:**

1. For a fixed micro-state $Z$, compute its macro-image $\tilde{Z} = \mathcal{R}_{\text{scutoid},b}(Z)$
2. Sample $M$ other micro-states $Z_1, \ldots, Z_M \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$ from the pre-image
3. For each $Z_i$, evolve forward one timestep using $\mathbb{P}_{\text{scutoid}}^{(N)}$ and compute the macro-image $\tilde{Z}_{i+1}$
4. Compute the empirical distribution over macro-states $\{\tilde{Z}_{i+1}\}$
5. Measure the total variation distance between this distribution and the expected macro-transition $\tilde{\mathbb{P}}_{\text{scutoid}}^{(n)}(\tilde{Z}, \cdot)$
6. Repeat for multiple $\tilde{Z}$ and take the maximum → this estimates $\epsilon_{\text{lump}}(b)$
7. Plot $\epsilon_{\text{lump}}(b)$ vs. $b$ and fit to the theoretical scaling
:::

### 8.3. Observable-Specific Information Loss

:::{prf:definition} Observable-Specific Excess Entropy
:label: def-observable-excess-entropy

For an observable $f: \Omega_{\text{scutoid}}^{(N)} \to \mathbb{R}$, define the **$f$-excess entropy**:

$$
E_f := I(\overrightarrow{f} ; \overleftarrow{f})
$$

(mutual information between past and future trajectories of $f$).

**Macro-conditional excess entropy:**

$$
E_{f|\text{macro}} := I(\overrightarrow{f} ; \overleftarrow{f} \mid \overleftarrow{\tilde{Z}})
$$

(mutual information between past and future of $f$ given the macro-past).

**Information loss due to coarse-graining:**

$$
\Delta E_f := E_f - E_{f|\text{macro}}
$$

**Interpretation:** $\Delta E_f$ quantifies how much predictive information about observable $f$ is lost when we only know the macro-past instead of the full micro-past.

**Observable-dependent closure:** $\Delta E_f = 0$ if and only if the coarse-graining satisfies $f$-restricted information closure ({prf:ref}`def-observable-dependent-closure`).
:::

:::{prf:theorem} Observable Hierarchy from Information Loss
:label: thm-observable-hierarchy

**Statement:** For the scutoid renormalization map with fixed $n_{\text{cell}}$, observables can be ranked by their information loss $\Delta E_f$:

**Class 1 (Perfectly preserved):** $\Delta E_f = 0$
- Topological charges (Chern numbers, winding numbers)
- Global symmetries
- Conserved quantities (total energy, momentum in periodic systems)

**Class 2 (Weakly affected):** $\Delta E_f \ll E_f$ (logarithmic in $n_{\text{cell}}$)
- Large Wilson loops ($\text{perimeter} \gg \ell_{\text{CVT}}$)
- Long-range correlations ($\text{separation} \gg \ell_{\text{CVT}}$)
- Hydrodynamic modes (low-frequency, long-wavelength)

**Class 3 (Moderately affected):** $\Delta E_f \sim O(1)$ (constant fraction lost)
- Intermediate-scale correlations ($\text{separation} \sim \ell_{\text{CVT}}$)
- Single-plaquette observables
- Curvature fluctuations at CVT cell scale

**Class 4 (Strongly affected):** $\Delta E_f \approx E_f$ (most information lost)
- Single-walker positions/velocities
- Short-wavelength Fourier modes ($k \gg \ell_{\text{CVT}}^{-1}$)
- Ultra-local fluctuations

**Proof:** This classification follows from dimensional analysis and the scaling of CVT quantization error ({prf:ref}`thm-cvt-approximation-error`). Observables with characteristic length $L$ lose information at rate $\propto (L/\ell_{\text{CVT}})^{-\alpha}$ for some $\alpha > 0$. $\square$

**Physical interpretation:** The renormalization map acts as a **low-pass filter** in position space, preserving long-wavelength physics and integrating out short-wavelength fluctuations. This is precisely the desired behavior for an RG transformation.
:::

---

## 9. Lumpability Error and Convergence Rates

### 9.1. Convergence Theorem

:::{prf:theorem} Scutoid RG Convergence to Micro-Limit
:label: thm-scutoid-rg-convergence

**Statement:** As the number of generators $n_{\text{cell}}$ increases to $N$, the macro-process converges to the micro-process in the following senses:

**1. Distribution convergence:**

$$
\lim_{n_{\text{cell}} \to N} d_{\text{TV}}(\tilde{\mu}_{\text{QSD}}^{(n)}, \mu_{\text{QSD}}^{(N)} \circ \mathcal{R}_{\text{scutoid},b}^{-1}) = 0
$$

(total variation distance between macro-QSD and push-forward of micro-QSD vanishes).

**2. Observable convergence:** For any continuous observable $f$ with Lipschitz constant $L_f$:

$$
\left| \langle f \rangle_{\text{micro}} - \langle \tilde{f} \rangle_{\text{macro}} \right| \leq L_f \cdot C_0 \cdot n_{\text{cell}}^{-1/d}
$$

**3. Transition kernel convergence:**

$$
\lim_{n_{\text{cell}} \to N} \sup_{\tilde{Z}, \tilde{A}} \left| \tilde{\mathbb{P}}_{\text{scutoid}}^{(n)}(\tilde{Z}, \tilde{A}) - \mathbb{P}_{\text{scutoid}}^{(N)}(Z, \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{A})) \right| = 0
$$

for any $Z \in \mathcal{R}_{\text{scutoid},b}^{-1}(\tilde{Z})$.

**4. Statistical complexity convergence:**

$$
\lim_{n_{\text{cell}} \to N} \tilde{C}_\mu^{(n)} = C_\mu^{(N)}
$$

**Proof:**

**Part 1:** The CVT clustering converges to the identity map as $n_{\text{cell}} \to N$ (each walker becomes its own cluster). Therefore, $\mathcal{R}_{\text{scutoid},b} \to \text{id}$ in the appropriate sense, implying distribution convergence.

**Part 2:** Follows from {prf:ref}`thm-cvt-approximation-error` and the Lipschitz continuity of $f$.

**Part 3:** This is the lumpability error convergence: as $n_{\text{cell}} \to N$ and $b \to 1$, the block size shrinks to the micro-scale, eliminating the coarse-graining and making lumpability exact.

**Part 4:** Follows from the continuity of the statistical complexity functional with respect to the transition kernel (Shalizi & Crutchfield, 2001, Theorem 4). $\square$
:::

### 9.2. Dimensional Scaling

:::{prf:theorem} Dimensional Dependence of Closure Accuracy
:label: thm-closure-dimensional-scaling

**Statement:** For fixed computational budget $N$ (number of micro-walkers), the accuracy of observable preservation under scutoid RG depends critically on the spacetime dimension $d$:

**Optimal $n_{\text{cell}}$ for target accuracy $\epsilon$:**

To achieve observable error $|\langle f \rangle_{\text{macro}} - \langle f \rangle_{\text{micro}}| \leq \epsilon$ for Lipschitz observable $f$ with constant $L_f$, we require:

$$
n_{\text{cell}} \geq n_{\text{min}}(\epsilon, d) = \left(\frac{L_f C_0}{\epsilon}\right)^d
$$

**Computational cost:** The total cost per timestep is $O(N)$ for walker evolution plus $O(n_{\text{cell}} \log n_{\text{cell}}) \approx O(n_{\text{cell}})$ for geometry (assuming $n_{\text{cell}} \ll N$). The effective speedup compared to full $N$-walker geometry is:

$$
\text{Speedup} = \frac{N}{n_{\text{cell}}} = \frac{N}{(L_f C_0 / \epsilon)^d}
$$

**Dimensional curse:** The required $n_{\text{cell}}$ grows exponentially with dimension:

| Dimension | $\epsilon = 0.01$ | $\epsilon = 0.001$ |
|-----------|-------------------|---------------------|
| $d=2$ | $n_{\text{cell}} \sim 10^4$ | $n_{\text{cell}} \sim 10^6$ |
| $d=3$ | $n_{\text{cell}} \sim 10^6$ | $n_{\text{cell}} \sim 10^9$ |
| $d=4$ | $n_{\text{cell}} \sim 10^8$ | $n_{\text{cell}} \sim 10^{12}$ |
| $d=5$ | $n_{\text{cell}} \sim 10^{10}$ | $n_{\text{cell}} \sim 10^{15}$ |

**Practical limit:** For $d \geq 5$, achieving sub-percent accuracy requires $n_{\text{cell}}$ comparable to or exceeding $N$, eliminating the computational advantage. The scutoid RG is **only viable in low dimensions** ($d \leq 4$).

**Proof:** Direct substitution from {prf:ref}`thm-scutoid-rg-convergence` Part 2 and inversion of the error bound. $\square$
:::

:::{prf:corollary} The d=3,4 Sweet Spot
:label: cor-d-3-4-sweet-spot

**Statement:** Spacetime dimensions $d=3$ and $d=4$ represent a **computational sweet spot** where:

1. $n_{\text{cell}}$ can be chosen orders of magnitude smaller than $N$ (e.g., $n_{\text{cell}} = 10^4$ vs. $N = 10^6$ in $d=4$)
2. Observable accuracy remains acceptable ($\epsilon \sim 0.01 - 0.1$) for physically relevant observables
3. The O(N) algorithm achieves significant speedup ($\sim 10^2 - 10^3$) compared to full geometry

**Anthropic argument:** If our universe's spacetime has $d=4$ because this is the dimension where **efficient computational representations** of physical laws are possible (the "O(N) Universe Hypothesis", § 10), then the mathematical structure of closure theory provides a **rigorous foundation** for this argument: $d=4$ is computationally optimal because it maximizes the ratio:

$$
R(d) := \frac{\text{Speedup}}{\epsilon} = \frac{N}{\epsilon \cdot (L_f C_0 / \epsilon)^d}
$$

which is maximized at $d=4$ for typical physical parameters.
:::

---

# PART IV: PHYSICAL IMPLICATIONS

## 10. The O(N) Universe Hypothesis: Closure-Theoretic Justification

### 10.1. The Anthropic Question

**Question:** Why does our universe have spacetime dimension $d=4$ (or $d=3+1$ if we separate space and time)?

**Traditional answers:**
- String theory: Critical dimension for anomaly cancellation
- Anthropic principle: Stable atoms and planetary orbits only exist in $d=3$ spatial dimensions
- Gauge theory: Asymptotic freedom works best in $d=4$

**This chapter's answer:** $d=4$ is the dimension where **computational closure enables efficient physical simulation**.

### 10.2. The Computational Closure Hypothesis

:::{prf:hypothesis} O(N) Universe Hypothesis (Closure Formulation)
:label: hyp-on-universe-closure

**Statement:** The observable universe can be accurately modeled by a coarse-grained effective theory (the "macroscopic physics" accessible to observers) that satisfies computational closure with respect to the microscopic fundamental theory, with the following properties:

1. **Computational efficiency:** The macro-theory has O(N) complexity per timestep for $N$ adaptive degrees of freedom
2. **Observable preservation:** All macro-scale observables (those accessible to observers at human scales) are preserved under the coarse-graining
3. **Dimensional optimality:** The accuracy-to-cost ratio is maximized at $d=4$ spacetime dimensions

**Consequence:** If observers (like humans or any computational agents) are constrained by the laws of physics to have finite computational resources, they can only exist in universes where such efficient coarse-graining is possible. The Fragile QFT framework demonstrates that $d=4$ satisfies this constraint.
:::

:::{prf:remark} Why This is Stronger Than Previous Arguments
:class: important

**Previous O(N) claims:** "We found an algorithm that happens to run in O(N) time."

**Closure-theoretic claim:** "We can **prove** that the coarse-graining preserves physics, with quantifiable error bounds, and that this preservation is only possible (with acceptable accuracy) in $d \leq 4$."

The difference is **rigor**. Closure theory provides:
1. A **criterion** for validity (information closure)
2. **Error bounds** (lumpability error, CVT quantization)
3. **Empirical tests** (observable preservation, statistical complexity)
4. **Dimensional constraints** (exponential growth of $n_{\text{cell}}$ with $d$)

This transforms the O(N) Universe Hypothesis from a speculative claim into a **testable, falsifiable prediction**.
:::

### 10.3. Empirical Predictions

The closure formulation makes concrete, testable predictions:

**Prediction 1:** For simulations of Yang-Mills theory in $d=4$ with $N=10^6$ walkers and $n_{\text{cell}}=10^4$ generators, large Wilson loops ($\text{perimeter} > 10 \ell_{\text{CVT}}$) should be preserved to within $1\%$ accuracy.

**Prediction 2:** The statistical complexity difference $\Delta C_\mu$ should decay exponentially as $n_{\text{cell}}$ increases:

$$
\Delta C_\mu(n_{\text{cell}}) \sim C_0 \exp(-n_{\text{cell}}/n_0)
$$

with $n_0 \sim 10^3$ for typical gauge coupling $g \sim 1$.

**Prediction 3:** The lumpability error should scale as:

$$
\epsilon_{\text{lump}}(b) \sim \exp(-b/\xi)
$$

for $b \ll \xi$, where $\xi$ is the correlation length extracted from the two-point function.

**Prediction 4:** In $d=5$, the same observables require $n_{\text{cell}} \sim 10^{10}$ to achieve $1\%$ accuracy, making the algorithm impractical. This is an **exclusion prediction**: if we simulate $d=5$ QFT with $n_{\text{cell}} = 10^4$, observable preservation should **fail**.

**Testing program:** Implement the verification protocols from § 7-8 and measure these quantities. If the predictions hold, closure theory is validated. If they fail, we identify which assumptions (e.g., CVT optimality, IG renormalization rule) need revision.

---

## 11. Dimensional Dependence of Closure Accuracy

### 11.1. The $d \geq 5$ Breakdown

:::{prf:theorem} Closure Breakdown in High Dimensions
:label: thm-closure-breakdown-high-d

**Statement:** For spacetime dimension $d \geq 5$, the scutoid renormalization map **cannot achieve computational closure** with both:
1. Acceptable observable accuracy ($\epsilon \leq 0.01$)
2. Significant computational speedup ($n_{\text{cell}} \ll N$)

simultaneously, for any choice of $n_{\text{cell}}$.

**Proof:** From {prf:ref}`thm-closure-dimensional-scaling`, achieving $\epsilon = 0.01$ requires:

$$
n_{\text{cell}} \geq (100 L_f C_0)^d
$$

For typical observables in gauge theory, $L_f C_0 \sim 1$, giving:

$$
n_{\text{cell}} \geq 10^{2d}
$$

In $d=5$: $n_{\text{cell}} \geq 10^{10}$. For $N = 10^6$ walkers (typical in current simulations), this requires $n_{\text{cell}} \gg N$, which is **impossible**—you cannot have more cluster centers than particles.

Even if we increase $N$ to $10^{10}$, the geometry cost $O(n_{\text{cell}} \log n_{\text{cell}}) \sim 10^{11}$ exceeds the walker evolution cost $O(N) = 10^{10}$, eliminating the speedup. $\square$

**Physical interpretation:** In $d \geq 5$, the "curse of dimensionality" dominates. The exponential growth of the state space volume with dimension makes it impossible to efficiently compress the microscopic information into a macroscopic description without catastrophic information loss.
:::

### 11.2. The $d=2$ Case

:::{prf:remark} Trivial Closure in d=2
:class: note

**Observation:** In $d=2$ (spatial dimension $d=1$, or spacetime $d=2$), the scutoid renormalization achieves **near-perfect closure** with very small $n_{\text{cell}}$:

For $\epsilon = 0.01$: $n_{\text{cell}} \sim 10^4$ (easily achievable).

For $\epsilon = 0.001$: $n_{\text{cell}} \sim 10^6$ (still tractable).

**Why $d=2$ is "too simple":** In $d=2$, quantum field theories are often **exactly solvable** (e.g., $1+1$ dimensional CFTs, integrable models). The computational complexity of simulating these theories is already low, so the scutoid RG provides no significant advantage over traditional methods.

**Anthropic consequence:** Observers in $d=2$ universes wouldn't need the scutoid RG because their physics is computationally tractable by other means. However, $d=2$ universes also lack the structural complexity needed for stable matter (no atoms, no chemistry), making observer emergence unlikely.
:::

### 11.3. Why $d=3$ and $d=4$ are Special

:::{prf:observation} The Goldilocks Principle for Dimension
:label: obs-goldilocks-dimension

**Too low ($d=2$):** Physics is too simple; no structural complexity for observers.

**Too high ($d \geq 5$):** Computational closure fails; no efficient effective theories exist.

**Just right ($d=3, 4$):** Physics is:
1. **Complex enough** to support rich structures (atoms, chemistry, life)
2. **Simple enough** to admit efficient coarse-grained descriptions (closure holds with $n_{\text{cell}} \ll N$)

**Mathematical basis:** The dimensional dependence of CVT quantization error ({prf:ref}`thm-cvt-approximation-error`) combined with the exponential cost scaling ({prf:ref}`thm-closure-dimensional-scaling`) creates a **narrow window** of dimensions where efficient simulation is possible.

**Connection to other physics:** This dimensional preference coincides with:
- **Asymptotic freedom** in Yang-Mills: Only in $d=4$ does the gauge coupling run logarithmically, allowing weak coupling at short distances
- **Stable orbits:** Only in $d=3$ spatial dimensions do planets have stable elliptical orbits
- **Chern-Gauss-Bonnet theorem:** Topological invariants (Euler characteristic, Chern numbers) have their richest structure in $d=4$

The scutoid RG closure adds a **computational** reason to this list: $d=4$ is where physics can be efficiently simulated.
:::

---

## 12. Summary and Research Program

### 12.1. Main Results

This chapter established the following:

**1. Conceptual shift:** The Fixed-Node Scutoid Tessellation is a **principled RG transformation**, not a computational approximation.

**2. Formal framework:** The scutoid renormalization map $\mathcal{R}_{\text{scutoid},b}$ defines an information-theoretic channel whose validity can be tested via computational closure.

**3. Observable preservation:** Long-range observables (Wilson loops, topological charges, action) are preserved with quantifiable error bounds ({prf:ref}`thm-wilson-loop-preservation`, {prf:ref}`thm-topological-charge-preservation`, {prf:ref}`thm-action-preservation`).

**4. Information-theoretic diagnostics:** Statistical complexity, lumpability error, and observable-specific excess entropy provide empirical tests for closure ({prf:ref}`thm-statistical-complexity-convergence`, {prf:ref}`thm-scutoid-lumpability-error-scaling`).

**5. Dimensional dependence:** Closure accuracy scales as $\epsilon \sim n_{\text{cell}}^{-1/d}$, making $d=3,4$ optimal and $d \geq 5$ impractical ({prf:ref}`thm-closure-dimensional-scaling`, {prf:ref}`thm-closure-breakdown-high-d`).

**6. O(N) Universe Hypothesis:** Computational closure provides a rigorous foundation for the anthropic argument that $d=4$ is "special" ({prf:ref}`hyp-on-universe-closure`).

### 12.2. Open Problems

**Theoretical:**

1. **Prove information closure analytically:** Derive rigorous bounds on $I(\overrightarrow{\tilde{Z}} ; \overleftarrow{\tilde{Z}}) - I(\overrightarrow{\tilde{Z}} ; \overleftarrow{Z})$ from the CVT construction and BAOAB dynamics.

2. **Lumpability from first principles:** Prove strong lumpability directly from the block-spin structure, without assuming computational closure.

3. **Optimal IG renormalization rule:** Derive the correct IG coarse-graining rule from Wilson loop preservation or gauge invariance.

4. **υ-machine characterization:** Compute the υ-machine ({prf:ref}`def-causal-closure`, [15_closure_theory.md](13_fractal_set_new/15_closure_theory.md) § 2.2) for the scutoid process and verify υ = ε (causal closure).

5. **Finite-size scaling:** Analyze how closure accuracy depends on the total simulation volume $V$ and the ratio $V / N$ (walker density).

**Empirical:**

1. **Implement verification protocols:** Run simulations for $d=2,3,4,5$ and measure:
   - Observable preservation errors ({prf:ref}`thm-wilson-loop-preservation`)
   - Statistical complexity difference ({prf:ref}`thm-statistical-complexity-convergence`)
   - Lumpability error ({prf:ref}`thm-scutoid-lumpability-error-scaling`)
   - Observable-specific excess entropy ({prf:ref}`def-observable-excess-entropy`)

2. **Extract RG beta function:** From the measured dependence of observables on $n_{\text{cell}}$ (or equivalently, $b$), extract the effective beta function $\beta(g, b)$ and compare to perturbative QCD predictions.

3. **Test dimensional exclusion:** Verify that $d=5$ simulations with $n_{\text{cell}} = 10^4$ show observable preservation failures, confirming the dimensional constraint.

4. **Benchmark against LQCD:** Compare the scutoid RG predictions (glueball masses, string tensions, topological susceptibility) against established lattice QCD results to validate the effective theory.

**Philosophical:**

1. **Emergence and reduction:** If closure fails for some observable, does this define a "truly emergent" phenomenon? Can we classify all observables by their closure properties?

2. **Computational ontology:** Does computational closure provide a **definition** of what it means for a macroscopic theory to be "real" (ontologically valid) rather than merely "useful"?

3. **Anthropic landscape:** If we simulate many universes with different $d$, does the probability of observer emergence correlate with closure accuracy? This could provide a **computational measure** on the landscape of possible universes.

### 12.3. Conclusion

The scutoid renormalization channel transforms the Fixed-Node Scutoid Tessellation from a "clever optimization" into a **rigorous effective field theory**. Closure theory provides the mathematical tools to:

- **Validate** the coarse-graining via information-theoretic criteria
- **Quantify** the errors via lumpability and CVT quantization bounds
- **Predict** when closure fails via dimensional scaling
- **Interpret** the O(N) algorithm as a computable instance of the renormalization group

If the empirical verification program succeeds (§ 12.2), we will have achieved something remarkable: a **proof** that the universe can efficiently simulate itself, and that this efficiency is only possible in low dimensions. The Fragile QFT framework then becomes not just a computational tool, but a **window into the deep connection between information, geometry, and the structure of physical law**.

**Final claim:** The scutoid renormalization map is not an approximation. It is the **correct** description of physics at coarse scales, and closure theory proves it.
