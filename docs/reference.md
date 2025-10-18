# Mathematical Reference for the Fragile Gas Framework

**Version:** 2.1 (Enhanced with critical theorems and proofs)
**Last Updated:** 2025-10-18
**Total Entries:** 101 (curated from 723 total in glossary)
**Coverage:** Chapter 1: ~60 entries, Chapter 2: ~41 entries

---

## Purpose and Usage

This document provides a **curated TLDR reference** for the most important mathematical results in the Fragile Gas framework. It complements the comprehensive glossary with detailed explanations, plain-English interpretations, and practical context.

**Target Audience:** Language models, researchers, and engineers needing quick understanding of key theorems and proofs.

**Relationship with Other Documents:**
- **This document (reference.md)**: 101 curated entries with TLDR format, detailed explanations, and "why it matters" context
- **docs/glossary.md**: Exhaustive index of all 723 mathematical entries with basic metadata (type, label, tags, source)
- **Source documents**: Full rigorous proofs and detailed mathematical development

**Design Philosophy:**
- **TLDR Format:** Each entry includes plain-English interpretation, key formula, and impact statement
- **Conceptual Organization:** Results grouped by theme, not document order
- **LLM-Optimized:** Balance between completeness and token efficiency
- **Cross-Linked:** Extensive tagging and references for navigation

**How to Use:**
1. **Start here** for understanding key results with context and motivation
2. **Search by tag:** Use tags like `cloning`, `kinetic`, `lsi`, `qsd`, `convergence`, etc.
3. **Browse by section:** Navigate conceptual hierarchy (foundations → operators → convergence → geometry)
4. **Follow references:** Use labels and cross-references to explore related results
5. **For exhaustive search:** Use **docs/glossary.md** to find all 723 entries by label/tag
6. **For full proofs:** Consult source documents listed in each entry

**Document Coverage:**
- **Chapter 1: Euclidean Gas** (~60 curated entries) - Core framework, operators, convergence theory
- **Chapter 2: Geometric Gas** (~41 curated entries) - Adaptive mechanisms, emergent Riemannian geometry, anisotropic diffusion
- **Total**: 101 entries carefully selected from 723 total mathematical results in the framework

---

## Table of Contents

### Chapter 1: Euclidean Gas
1. [Foundational Framework](#1-foundational-framework)
2. [State Space and Metrics](#2-state-space-and-metrics)
3. [Core Axioms](#3-core-axioms)
4. [Operators: Measurement and Fitness](#4-operators-measurement-and-fitness)
5. [Operators: Kinetic (Langevin Dynamics)](#5-operators-kinetic-langevin-dynamics)
6. [Operators: Cloning and Keystone Principle](#6-operators-cloning-and-keystone-principle)
7. [Wasserstein Contraction](#7-wasserstein-contraction)
8. [Hypocoercivity and Coupled Lyapunov Functions](#8-hypocoercivity-and-coupled-lyapunov-functions)
9. [Convergence to QSD](#9-convergence-to-qsd)
10. [Mean-Field Limit and McKean-Vlasov PDE](#10-mean-field-limit-and-mckean-vlasov-pde)
11. [Propagation of Chaos](#11-propagation-of-chaos)
12. [KL-Divergence Convergence and LSI Theory](#12-kl-divergence-convergence-and-lsi-theory)

### Chapter 2: Geometric Gas
13. [Geometric Gas: Adaptive Mechanisms](#13-geometric-gas-adaptive-mechanisms)
14. [Emergent Riemannian Geometry](#14-emergent-riemannian-geometry)
15. [Anisotropic Diffusion and Hypocoercivity](#15-anisotropic-diffusion-and-hypocoercivity)
16. [Geometric Gas Convergence](#16-geometric-gas-convergence)
17. [C³ and C⁴ Regularity Theory](#17-c3-and-c4-regularity-theory)
18. [N-Uniform LSI for Geometric Gas](#18-n-uniform-lsi-for-geometric-gas)

### Cross-Cutting Themes
19. [Key Inequalities and Bounds](#19-key-inequalities-and-bounds)
20. [Critical Parameters and Constants](#20-critical-parameters-and-constants)

---

# Chapter 1: Euclidean Gas

## 1. Foundational Framework

### Walker
**Label:** `def-walker`
**Source:** [01_fragile_gas_framework.md § 1.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `foundation`, `state-space`

**What it says:** A walker is a tuple $(x, s)$ where $x \in \mathcal{X}$ is position and $s \in \{0,1\}$ is survival status.

**Math:**
$$w := (x, s), \quad x \in \mathcal{X}, \quad s \in \{0,1\}$$

Extended form for Euclidean Gas: $w := (x, v, s)$ with velocity $v \in \mathbb{R}^d$.

**Why it matters:** Fundamental unit of computation. Status bit enables viability constraints and cloning dynamics.

---

### Swarm and Swarm State Space
**Label:** `def-swarm-and-state-space`
**Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `foundation`, `state-space`, `swarm`

**What it says:** A swarm is an N-tuple of walkers living in product space.

**Math:**
$$\mathcal{S} := (w_1, \ldots, w_N), \quad \Sigma_N := (\mathcal{X} \times \{0,1\})^N$$

**Why it matters:** All algorithms operate on swarms, not individual walkers. Enables mean-field limits and propagation of chaos.

**Related:** `def-alive-dead-sets`, `def-valid-state-space`

---

### Alive and Dead Sets
**Label:** `def-alive-dead-sets`
**Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `foundation`, `partition`, `viability`

**What it says:** Partition swarm into alive and dead walkers based on status bit.

**Math:**
$$\mathcal{A}(\mathcal{S}) := \{i : s_i = 1\}, \quad \mathcal{D}(\mathcal{S}) := \{i : s_i = 0\}$$

**Why it matters:** Cloning operator acts on this partition. Dead walkers are candidates for revival, alive walkers are potential companions.

**Related:** `def-axiom-guaranteed-revival`, `def-cloning-operator-formal`

---

### Valid State Space
**Label:** `def-valid-state-space`
**Source:** [01_fragile_gas_framework.md § 1.4](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `foundation`, `topology`, `measure-theory`, `polish-space`

**What it says:** State space must be Polish (complete, separable metric) with reference measure and regular boundary.

**Math:** Triple $(\mathcal{X}, d_\mathcal{X}, \mu_\mathcal{X})$ where:
- $(\mathcal{X}, d_\mathcal{X})$ is Polish
- $\mu_\mathcal{X}$ is reference measure (Lebesgue or Riemannian volume)
- $\partial\mathcal{X}_{\text{valid}}$ is $C^1$ $(d-1)$-dimensional submanifold

**Why it matters:** Ensures well-posedness of probability measures, Wasserstein metrics, and stochastic operators.

**Related:** `def-axiom-boundary-smoothness`, `lem-polishness-and-w2`

---

## 2. State Space and Metrics

### N-Particle Displacement Metric
**Label:** `def-n-particle-displacement-metric`
**Source:** [01_fragile_gas_framework.md § 1.6](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `metric`, `displacement`, `wasserstein`

**What it says:** Root-mean-square distance between paired swarms in algorithmic space.

**Math:**
$$d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) := \sqrt{\frac{1}{N}\sum_i d_\mathcal{Y}(\varphi(x_{1,i}), \varphi(x_{2,i}))^2 + \frac{\lambda_{\text{status}}}{N}\sum_i (s_{1,i} - s_{2,i})^2}$$

Squared form: $d^2 = \frac{1}{N}\Delta^2_{\text{pos}} + \frac{\lambda_{\text{status}}}{N}n^c$ where $n^c$ counts status changes.

**Why it matters:** Foundation for Wasserstein-2 metrics on swarm probability distributions. Enables contraction proofs.

**Related:** `def-metric-quotient`, `lem-polishness-and-w2`

---

### Kolmogorov Quotient
**Label:** `def-metric-quotient`
**Source:** [01_fragile_gas_framework.md § 1.6.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `metric`, `topology`, `quotient-space`

**What it says:** Mod out by zero-distance swarms to get true metric space.

**Math:**
$$\mathcal{S}_1 \sim \mathcal{S}_2 \iff d_{\text{Disp}}(\mathcal{S}_1, \mathcal{S}_2) = 0$$
$$\bar{\Sigma}_N := \Sigma_N / \sim, \quad \bar{d}_{\text{Disp}}([\mathcal{S}_1], [\mathcal{S}_2]) := d_{\text{Disp}}(\mathcal{S}_1, \mathcal{S}_2)$$

**Why it matters:** Ensures displacement pseudometric becomes genuine metric. Critical for Wasserstein theory.

**Related:** `lem-polishness-and-w2`

---

### Polishness and W₂ Well-Posedness
**Label:** `lem-polishness-and-w2`
**Source:** [01_fragile_gas_framework.md § 1.7.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `topology`, `wasserstein`, `polish-space`, `lemma`

**What it says:** Quotient swarm space is Polish, so Wasserstein-2 metric is well-defined.

**Math:** If $(\mathcal{Y}, d_\mathcal{Y})$ Polish and $N < \infty$, then $(\bar{\Sigma}_N, \bar{d}_{\text{Disp}})$ is Polish.

**Why it matters:** Wasserstein-2 on $\mathcal{P}(\bar{\Sigma}_N)$ is finite on measures with finite second moment. Enables all contraction proofs.

**Proof idea:** Metric quotient of Polish space by closed equivalence is Polish (Kechris).

**Related:** `def-metric-quotient`, `def-n-particle-displacement-metric`

---

### Algorithmic Distance
**Label:** `def-alg-distance`
**Source:** [01_fragile_gas_framework.md § 5.3](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `metric`, `algorithmic-space`

**What it says:** Distance between swarms measured in projected algorithmic space (position + status).

**Math:**
$$d_{\text{alg}}(\mathcal{S}_1, \mathcal{S}_2) := d_{\text{Disp},\mathcal{Y}}(\varphi(\mathcal{S}_1), \varphi(\mathcal{S}_2))$$

where $\varphi$ projects swarms to algorithmic space (e.g., position only, or position+velocity).

**Why it matters:** Defines geometry that cloning and measurement operators respect. Different from kinetic metric.

**Related:** `def-algorithmic-space-generic`, `def-distance-positional-measures`

---

## 3. Core Axioms

### Axiom of Guaranteed Revival
**Label:** `def-axiom-guaranteed-revival`
**Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `viability`, `cloning`, `revival`

**What it says:** Dead walker's cloning score must exceed maximum threshold, ensuring deterministic revival.

**Math:**
$$\kappa_{\text{revival}} := \frac{\eta^{\alpha+\beta}}{\epsilon_{\text{clone}} \cdot p_{\max}} > 1$$

**Why it matters:** Prevents gradual extinction. Without this, swarms die out with positive probability.

**Consequence:** `thm-revival-guarantee` - dead walkers are revived almost surely.

**Related:** `thm-revival-guarantee`, `def-cloning-measure`

---

### Almost-Sure Revival Theorem
**Label:** `thm-revival-guarantee`
**Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `theorem`, `viability`, `revival`

**What it says:** Under revival axiom, every dead walker is cloned with probability 1.

**Math:** If $\epsilon_{\text{clone}} p_{\max} < \eta^{\alpha+\beta}$, $|\mathcal{A}| \geq 1$, $i \in \mathcal{D}$, then $\mathbb{P}[\text{walker } i \text{ revived}] = 1$.

**Why it matters:** Guarantees long-term viability. Swarm never goes extinct.

**Proof idea:** Any alive companion has $V_{\text{fit}} \geq \eta^{\alpha+\beta}$, so dead walker's score exceeds $p_{\max}$.

**Related:** `def-axiom-guaranteed-revival`

---

### Axiom of Boundary Regularity
**Label:** `def-axiom-boundary-regularity`
**Source:** [01_fragile_gas_framework.md § 2.1.2](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `viability`, `geometry`, `holder`

**What it says:** Death probability after perturbation is Hölder continuous in swarm state.

**Math:**
$$|P(s_{\text{out},i}=0 | \mathcal{S}_1) - P(s_{\text{out},i}=0 | \mathcal{S}_2)| \leq L_{\text{death}} \cdot d_{\text{Disp}}(\mathcal{S}_1, \mathcal{S}_2)^{\alpha_B}$$

Typical: $L_{\text{death}} \lesssim \text{Per}(\mathcal{X}_{\text{invalid}}) / \sigma$, $\alpha_B = 1$ (Lipschitz).

**Why it matters:** Enables Wasserstein contraction proofs. Death operator becomes Lipschitz.

**Related:** `lem-boundary-uniform-ball`, `lem-boundary-heat-kernel`

---

### Axiom of Environmental Richness
**Label:** `def-axiom-environmental-richness`
**Source:** [01_fragile_gas_framework.md § 2.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `environment`, `learning`, `variance`

**What it says:** Environment provides sufficient variation in rewards for meaningful fitness discrimination.

**Math:** Requires non-trivial variance in reward measurements across state space.

**Why it matters:** Without environmental richness, all states look identical → no learning. Ensures fitness landscape is informative.

**Related:** `def-axiom-reward-regularity`, `def-reward-measurement`

---

### Axiom of Reward Regularity
**Label:** `def-axiom-reward-regularity`
**Source:** [01_fragile_gas_framework.md § 2.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `fitness`, `lipschitz`

**What it says:** Reward function is Lipschitz continuous in position.

**Math:**
$$|R(x_1) - R(x_2)| \leq L_R \cdot d_\mathcal{X}(x_1, x_2)$$

**Why it matters:** Ensures smooth fitness landscapes. Enables gradient-based convergence arguments.

**Related:** `def-reward-measurement`, `ax:lipschitz-fields`

---

### Axiom of Bounded Algorithmic Diameter
**Label:** `def-axiom-bounded-algorithmic-diameter`
**Source:** [01_fragile_gas_framework.md § 2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `metric`

**What it says:** Algorithmic space has finite diameter.

**Math:**
$$\text{diam}(\mathcal{Y}) := \sup_{y_1, y_2 \in \mathcal{Y}} d_\mathcal{Y}(y_1, y_2) < \infty$$

**Why it matters:** Ensures bounded Wasserstein distances. Critical for finite-time convergence bounds.

**Related:** `def-alg-distance`, `def-algorithmic-space-generic`

---

### Axiom of Non-Degenerate Noise
**Label:** `def-axiom-non-degenerate-noise`
**Source:** [01_fragile_gas_framework.md § 2.3.2](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `perturbation`, `noise`

**What it says:** Noise kernels have full support (can reach any state from any other).

**Math:** For heat kernel: $p_\sigma(x, y) > 0$ for all $x, y \in \mathcal{X}$.

**Why it matters:** Ensures irreducibility. Without this, swarm could get trapped in subsets.

**Related:** `def-perturbation-measure`, `def-cloning-measure`

---

### Axiom of Geometric Consistency
**Label:** `def-axiom-geometric-consistency`
**Source:** [01_fragile_gas_framework.md § 2.4.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `axiom`, `geometry`, `metric`

**What it says:** Algorithmic projection preserves distances (non-expansive).

**Math:**
$$d_\mathcal{Y}(\varphi(w_1), \varphi(w_2)) \leq d_\mathcal{X}(x_1, x_2)$$

**Why it matters:** Ensures projection doesn't create artificial separations. Algorithmic distance is meaningful.

**Related:** `def-alg-distance`, `def-algorithmic-space-generic`

---

## 4. Operators: Measurement and Fitness

### Reward Measurement
**Label:** `def-reward-measurement`
**Source:** [01_fragile_gas_framework.md § 3.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `fitness`, `operator`

**What it says:** Measure environment reward at walker position, possibly with noise.

**Math:**
$$R_i := R(x_i) + \xi_i, \quad \xi_i \sim \mathcal{N}(0, \sigma_R^2)$$

**Why it matters:** Converts spatial exploration to fitness signal. Foundation for virtual rewards.

**Related:** `def-axiom-reward-regularity`, `def-raw-value-operator`

---

### Raw Value Operator
**Label:** `def-raw-value-operator`
**Source:** [01_fragile_gas_framework.md § 9.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `fitness`, `operator`

**What it says:** Combine reward and diversity signals into scalar fitness value.

**Math:**
$$V_{\text{raw},i} := \alpha R_i + \beta D_i$$

where $R_i$ is reward, $D_i$ is diversity (distance to companions).

**Why it matters:** Balances exploitation ($\alpha$) and exploration ($\beta$). Tunable via hyperparameters.

**Related:** `def-reward-measurement`, `def-distance-to-companion-measurement`

---

### Distance-to-Companion Measurement
**Label:** `def-distance-to-companion-measurement`
**Source:** [01_fragile_gas_framework.md § 10.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `metric`, `diversity`, `operator`

**What it says:** Measure average distance from walker to companion set.

**Math:**
$$D_i := \frac{1}{k} \sum_{j \in C_i} d_\mathcal{Y}(\varphi(w_i), \varphi(w_j))$$

where $C_i$ is companion set (typically alive walkers).

**Why it matters:** Diversity term encourages exploration. Prevents swarm collapse to single point.

**Related:** `def-companion-selection-measure`, `def-raw-value-operator`

---

### N-Dimensional Standardization Operator
**Label:** `def-standardization-operator-n-dimensional`
**Source:** [01_fragile_gas_framework.md § 11.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `operator`, `standardization`

**What it says:** Convert raw values to z-scores (zero mean, unit variance) across swarm.

**Math:**
$$z_i := \frac{V_{\text{raw},i} - \bar{V}}{\sigma_{\text{reg}}(\{V_{\text{raw},j}\})}$$

where $\sigma_{\text{reg}}$ is regularized standard deviation (lower-bounded to avoid division by zero).

**Why it matters:** Scale-invariant fitness. Enables comparison across different reward ranges.

**Related:** `def-asymmetric-rescale-function`, `thm-z-score-norm-bound`

---

### Smooth Piecewise Rescale Function
**Label:** `def-asymmetric-rescale-function`
**Source:** [01_fragile_gas_framework.md § 8.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `metric`, `rescale`, `lipschitz`

**What it says:** Monotone smooth map from raw values to $[0, 1]$ interval via cubic interpolation.

**Math:** Piecewise function with:
- Linear segment for $z < -\eta$: $g(z) = 0$
- Cubic patch for $|z| \leq \eta$: smooth interpolation
- Linear segment for $z > \eta$: $g(z) = 1$

Lipschitz constant: $L_P \approx 1.0054$ (proven in framework).

**Why it matters:** Ensures fitness values are bounded and Lipschitz. Critical for contraction proofs.

**Related:** `thm-rescale-function-lipschitz`, `lem-cubic-patch-coefficients`

---

### Rescaled Potential Operator for Alive Set
**Label:** `def-alive-set-potential-operator`
**Source:** [01_fragile_gas_framework.md § 12.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `fitness`, `viability`, `operator`

**What it says:** Compute fitness potential for alive walkers only (exclude dead from statistics).

**Math:**
$$\Phi_i := g_{\text{rescale}}(z_i(\{V_{\text{raw},j} : j \in \mathcal{A}\}))$$

**Why it matters:** Prevents dead walkers from contaminating fitness statistics. Improves signal quality.

**Related:** `def-alive-dead-sets`, `def-standardization-operator-n-dimensional`

---

## 5. Operators: Kinetic (Langevin Dynamics)

### BAOAB Integrator
**Label:** (described in `02_euclidean_gas.md`)
**Source:** [02_euclidean_gas.md § 1.1](source/1_euclidean_gas/02_euclidean_gas)
**Tags:** `kinetic`, `langevin`, `integrator`

**What it says:** Splitting integrator for Langevin dynamics: B(drift) - A(half-kick) - O(Ornstein-Uhlenbeck) - A(half-kick) - B(drift).

**Math:**
$$\begin{aligned}
\text{B: } & x \gets x + \frac{\tau}{2} v \\
\text{A: } & v \gets v - \frac{\tau}{2\lambda_v} \nabla V(x) \\
\text{O: } & v \gets e^{-\gamma \tau} v + \sqrt{T(1 - e^{-2\gamma\tau})} \xi, \quad \xi \sim \mathcal{N}(0, I) \\
\text{A: } & v \gets v - \frac{\tau}{2\lambda_v} \nabla V(x) \\
\text{B: } & x \gets x + \frac{\tau}{2} v
\end{aligned}$$

**Why it matters:** Second-order accurate, preserves detailed balance, time-reversible. Gold standard for molecular dynamics.

**Related:** `def-langevin-sde`, `thm-baoab-accuracy`

---

### Kinetic Operator $\Psi_{\text{kin}}$
**Label:** (described throughout kinetic sections)
**Source:** [02_euclidean_gas.md § 1](source/1_euclidean_gas/02_euclidean_gas)
**Tags:** `kinetic`, `operator`, `langevin`

**What it says:** Discrete-time Markov operator implementing BAOAB on all walkers.

**Math:**
$$\Psi_{\text{kin}}: \mathcal{P}(\Sigma_N) \to \mathcal{P}(\Sigma_N), \quad \mu \mapsto \Psi_{\text{kin}}^* \mu$$

**Why it matters:** Enables velocity thermalization and spatial exploration. Foundation for hypocoercivity.

**Related:** `lem-kinetic-lsi-hypocoercive`, `thm-velocity-variance-contraction`

---

### Sasaki Metric on Tangent Bundle
**Label:** (described in `02_euclidean_gas.md`)
**Source:** [02_euclidean_gas.md § 2.1](source/1_euclidean_gas/02_euclidean_gas)
**Tags:** `metric`, `kinetic`, `phase-space`

**What it says:** Natural Riemannian metric on phase space $(x, v)$ treating positions and velocities on equal footing.

**Math:**
$$d_{\text{Sasaki}}^2((x_1, v_1), (x_2, v_2)) := d_\mathcal{X}^2(x_1, x_2) + \lambda_v^2 \|v_1 - v_2\|^2$$

Parameter $\lambda_v > 0$ sets relative weight of velocity component.

**Why it matters:** Enables hypocoercive analysis. Couples position and velocity errors.

**Related:** `def-full-synergistic-lyapunov-function`, `thm-sasaki-standardization-composite-sq`

---

## 6. Operators: Cloning and Keystone Principle

### Cloning Measure
**Label:** `def-cloning-measure`
**Source:** [01_fragile_gas_framework.md § 4.1](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `cloning`, `operator`

**What it says:** Kernel for selecting companion walker to clone from, weighted by fitness.

**Math:**
$$K_{\text{clone}}(i, j | \mathcal{S}) \propto \exp(\beta V_{\text{fit},j}) \cdot \mathbb{1}_{j \in \mathcal{A}}$$

Typically: softmax over alive walkers with inverse temperature $\beta$.

**Why it matters:** Implements fitness-proportional selection. High-fitness walkers have higher cloning probability.

**Related:** `def-companion-selection-measure`, `def-cloning-operator-formal`

---

### Cloning Operator $\Psi_{\text{clone}}$
**Label:** `def-cloning-operator-formal`
**Source:** [03_cloning.md § 9.2](source/1_euclidean_gas/03_cloning)
**Tags:** `cloning`, `operator`

**What it says:** Composite operator: measure → fitness → decision → state update.

**Math:**
$$\Psi_{\text{clone}} = \Psi_{\text{update}} \circ \Psi_{\text{decision}} \circ \Psi_{\text{fitness}} \circ \Psi_{\text{measure}}$$

**Substeps:**
1. **Measure:** Perturb positions, update status (alive/dead)
2. **Fitness:** Compute virtual rewards from reward + diversity
3. **Decision:** Select companions, compute cloning scores
4. **Update:** Clone walkers (copy position/velocity) with inelastic collision noise

**Why it matters:** Core operator driving fitness selection and exploration. Implements evolutionary pressure.

**Related:** `thm-cloning-operator-composition`, `def-measurement-operator`

---

### Keystone Principle (Quantitative)
**Label:** `lem-quantitative-keystone`
**Source:** [03_cloning.md § 8.1](source/1_euclidean_gas/03_cloning)
**Tags:** `cloning`, `keystone`, `lemma`, `N-uniform`

**What it says:** If swarm has high internal variance, cloning concentrates it in low-fitness region, enabling targeted contraction.

**Math:** If $\text{Var}_x \geq \kappa_{\text{keystone}} \cdot \mathbb{E}[V^2]$, then cloning reduces variance by factor $1 - c_{\text{keystone}}$ where $c > 0$ is **N-uniform**.

**Why it matters:** Heart of convergence proof. Shows cloning isn't just random resampling—it's targeted variance reduction.

**Proof strategy:** Define critical target set $H_k := \{i : V_i < V_{\text{thresh}}\}$. Show:
1. Target set has substantial mass
2. Variance concentrates in target set
3. Cloning pressure acts preferentially on target set

**Related:** `def-critical-target-set`, `lem-variance-concentration-Hk`, `thm-positional-variance-contraction`

---

### Positional Variance Contraction Under Cloning
**Label:** `thm-positional-variance-contraction`
**Source:** [03_cloning.md § 10.3.1](source/1_euclidean_gas/03_cloning)
**Tags:** `cloning`, `theorem`, `contraction`, `N-uniform`

**What it says:** Cloning reduces swarm positional variance via fitness-weighted resampling.

**Math:**
$$\mathbb{E}[\text{Var}_x^{\text{post}}] \leq (1 - c_{\text{keystone}}) \text{Var}_x^{\text{pre}} + C_{\text{revival}} \cdot D^2$$

where $c_{\text{keystone}} > 0$ is **N-uniform** and $C_{\text{revival}}$ handles dead walker revival.

**Why it matters:** Quantifies exploitation. Swarm contracts toward high-fitness regions.

**Related:** `lem-quantitative-keystone`, `lem-dead-walker-revival-bounded`

---

### Velocity Variance Expansion (Bounded)
**Label:** `thm-velocity-variance-bounded-expansion`
**Source:** [03_cloning.md § 10.4](source/1_euclidean_gas/03_cloning)
**Tags:** `cloning`, `kinetic`, `theorem`

**What it says:** Cloning increases velocity variance (inelastic collisions add noise), but expansion is bounded.

**Math:**
$$\mathbb{E}[\text{Var}_v^{\text{post}}] \leq (1 + c_v) \text{Var}_v^{\text{pre}} + C_v \cdot \delta^2$$

where $\delta$ is collision noise scale.

**Why it matters:** Velocity expansion is controlled. Kinetic operator must dissipate this energy → hypocoercive balance.

**Related:** `rem-synergistic-velocity-dissipation`, `thm-complete-variance-drift`

---

### Complete Variance Drift Characterization
**Label:** `thm-complete-variance-drift`
**Source:** [03_cloning.md § 10.6](source/1_euclidean_gas/03_cloning)
**Tags:** `cloning`, `theorem`, `drift`, `lyapunov`

**What it says:** Full drift inequality for coupled Lyapunov function under cloning.

**Math:**
$$\mathbb{E}[V^{\text{post}}] \leq (1 - c_{\text{clone}}) V^{\text{pre}} + C_{\text{clone}}$$

where $V := \text{Var}_x + \lambda_v^2 \text{Var}_v + \lambda_B B$ is synergistic Lyapunov function.

Constants $c_{\text{clone}}, C_{\text{clone}}$ are **N-uniform**.

**Why it matters:** Establishes Foster-Lyapunov drift for cloning. Half of the convergence proof.

**Related:** `def-full-synergistic-lyapunov-function`, `thm-foster-lyapunov-adaptive`

---

## 7. Wasserstein Contraction

### Wasserstein-2 Metric
**Label:** (standard definition, used throughout)
**Tags:** `wasserstein`, `metric`, `optimal-transport`

**What it says:** Optimal transport distance between probability measures.

**Math:**
$$W_2(\mu, \nu) := \inf_{\pi \in \Pi(\mu, \nu)} \left( \int d^2(x, y) \, d\pi(x, y) \right)^{1/2}$$

where $\Pi(\mu, \nu)$ is set of couplings with marginals $\mu, \nu$.

**Why it matters:** Natural metric for probability measures. Equivalent to displacement under optimal coupling.

**Related:** `lem-polishness-and-w2`, `def-n-particle-displacement-metric`

---

### Decomposition of Hypocoercive Wasserstein Distance
**Label:** `lem-wasserstein-decomposition`
**Source:** [03_cloning.md § 3.2.3](source/1_euclidean_gas/03_cloning)
**Tags:** `wasserstein`, `metric`, `lemma`, `hypocoercivity`

**What it says:** Wasserstein distance decomposes into location error and structural error.

**Math:**
$$W_2^2(\mu_1, \mu_2) \leq V_{\text{loc}} + V_{\text{struct}}$$

- $V_{\text{loc}} := \|\bar{x}_1 - \bar{x}_2\|^2$ (barycenter displacement)
- $V_{\text{struct}} := \mathbb{E}_\pi[\|x_1 - \bar{x}_1 - (x_2 - \bar{x}_2)\|^2]$ (internal variance mismatch)

**Why it matters:** Enables separate analysis of mean drift and variance contraction. Foundation for hypocoercive proofs.

**Related:** `def-full-synergistic-lyapunov-function`, `lem-sx-implies-variance`

---

### Variance Decomposition by Clusters
**Label:** `lem-variance-decomposition`
**Source:** [04_wasserstein_contraction.md § 2.1](source/1_euclidean_gas/04_wasserstein_contraction)
**Tags:** `variance`, `lemma`, `cloning`, `decomposition`

**What it says:** Swarm variance decomposes into within-cluster and between-cluster components.

**Math:**
For a swarm $S_k$ partitioned into $I_k$ (target) and $J_k$ (complement) with population fractions $f_I = |I_k|/k$ and $f_J = |J_k|/k$:

$$V_{\text{total}} = V_{\text{within}} + V_{\text{between}}$$

where $V_{\text{within}} = f_I V_I + f_J V_J$ and $V_{\text{between}} = f_I f_J \|\bar{x}_I - \bar{x}_J\|^2$.

**Why it matters:** Separates intra-cluster variance from inter-cluster separation. Key for analyzing cloning's effect on different population segments.

**Related:** `cor-between-group-dominance`, `lem-cross-swarm-distance`

---

### Between-Group Variance Dominance
**Label:** `cor-between-group-dominance`
**Source:** [04_wasserstein_contraction.md § 2.1.1](source/1_euclidean_gas/04_wasserstein_contraction)
**Tags:** `variance`, `corollary`, `clustering`

**What it says:** For high-error swarms, between-group variance dominates.

**Math:**
When $V_{\text{struct}} > R^2_{\text{spread}}$:

$$V_{\text{between}} \geq \frac{R^2_{\text{spread}}}{2}$$

**Why it matters:** Ensures cloning pressure concentrates on cross-cluster alignment, not just internal compression.

**Related:** `lem-variance-decomposition`, `lem-target-cloning-pressure`

---

### Cluster-Level Outlier Alignment
**Label:** `lem-cluster-alignment`
**Source:** [04_wasserstein_contraction.md § 2.2](source/1_euclidean_gas/04_wasserstein_contraction)
**Tags:** `lemma`, `alignment`, `cloning`

**What it says:** Cloning preferentially aligns outlier clusters (target sets) toward barycenter.

**Math:**
For two swarms $S_1, S_2$ satisfying cluster-preserving conditions, outliers in $J_k$ (complement) clone toward centroids $\bar{x}_{I_k}$ (target).

**Why it matters:** Explains why cloning reduces Wasserstein distance: it systematically eliminates outliers.

**Related:** `lem-expected-distance-change`, `cor-average-cloning`

---

### Cloning Pressure on Target Set
**Label:** `lem-target-cloning-pressure`
**Source:** [04_wasserstein_contraction.md § 2.4](source/1_euclidean_gas/04_wasserstein_contraction)
**Tags:** `lemma`, `cloning`, `fitness`

**What it says:** Walkers in the target set (high fitness) have higher cloning probability.

**Math:**
For any walker $i \in I_k$ (target set):

$$\mathbb{P}[\text{clone}_i] \geq \frac{1}{2} + \frac{\epsilon_F}{4\sqrt{2\pi}}$$

where $\epsilon_F$ is the fitness gap.

**Why it matters:** Quantifies selective pressure. Larger $\epsilon_F$ means stronger concentration on high-fitness walkers.

**Related:** `cor-average-cloning`, `lem-wasserstein-population-bound`

---

### Wasserstein-2 Contraction (Cluster-Based)
**Label:** `thm-main-contraction-full`
**Source:** [04_wasserstein_contraction.md § 2.5](source/1_euclidean_gas/04_wasserstein_contraction)
**Tags:** `theorem`, `wasserstein`, `contraction`, `cloning`

**What it says:** Cloning operator contracts Wasserstein-2 distance between swarm distributions.

**Math:**
Under cluster-preserving conditions with fitness gap $\epsilon_F > 0$:

$$\mathbb{E}[W_2^2(S_1', S_2') \mid S_1, S_2] \leq (1 - c\epsilon_F) W_2^2(S_1, S_2)$$

where $c > 0$ is a universal constant and $S'$ denotes post-cloning state.

**Why it matters:** **Main contraction theorem for cloning operator.** Proves exponential convergence to QSD in Wasserstein metric. Foundation for all finite-N convergence theory.

**Related:** `thm-main-convergence`, `thm-foster-lyapunov-adaptive`, `lem-wasserstein-decomposition`

---

## 8. Hypocoercivity and Coupled Lyapunov Functions

### Full Synergistic Hypocoercive Lyapunov Function
**Label:** `def-full-synergistic-lyapunov-function`
**Source:** [03_cloning.md § 3.3](source/1_euclidean_gas/03_cloning)
**Tags:** `lyapunov`, `hypocoercivity`, `coupling`

**What it says:** Combine position variance, velocity variance, and boundary potential with optimal weights.

**Math:**
$$\mathcal{V} := \text{Var}_x + \lambda_v^2 \text{Var}_v + \lambda_B B$$

where:
- $\text{Var}_x := \frac{1}{N}\sum_i \|x_i - \bar{x}\|^2$ (position variance)
- $\text{Var}_v := \frac{1}{N}\sum_i \|v_i - \bar{v}\|^2$ (velocity variance)
- $B := \frac{1}{N}\sum_i \Phi_{\text{boundary}}(x_i)$ (boundary potential)

Weights $\lambda_v, \lambda_B > 0$ balance terms.

**Why it matters:** No single term contracts under all operators. Synergistic combination enables total contraction.

**Analogy:** Like Villani's hypocoercivity but for discrete-time particle system with boundaries.

**Related:** `thm-complete-variance-drift`, `prop-lyapunov-necessity`

---

### Necessity of Augmented Lyapunov Structure
**Label:** `prop-lyapunov-necessity`
**Source:** [03_cloning.md § 3.3.2](source/1_euclidean_gas/03_cloning)
**Tags:** `lyapunov`, `proposition`, `hypocoercivity`

**What it says:** No single component (position or velocity variance alone) contracts under composition. Must couple them.

**Math:**
- Kinetic operator: $\mathbb{E}[\text{Var}_x^{\text{post}}] > \text{Var}_x^{\text{pre}}$ (expands position)
- Cloning operator: $\mathbb{E}[\text{Var}_v^{\text{post}}] > \text{Var}_v^{\text{pre}}$ (expands velocity)

But weighted sum contracts: $\mathbb{E}[\mathcal{V}^{\text{post}}] < \mathcal{V}^{\text{pre}}$.

**Why it matters:** Justifies complex Lyapunov structure. Simple Lyapunov functions don't work.

**Related:** `def-full-synergistic-lyapunov-function`, `rem-synergistic-velocity-dissipation`

---

### Drift Matrix Analysis
**Label:** (implicitly in `05_kinetic_contraction.md`)
**Source:** [05_kinetic_contraction.md](source/1_euclidean_gas/05_kinetic_contraction)
**Tags:** `hypocoercivity`, `drift-matrix`, `spectral-analysis`

**What it says:** Hypocoercive drift can be analyzed via drift matrix with negative eigenvalues.

**Math:**
$$\mathbb{E}[\mathcal{V}^{\text{post}} - \mathcal{V}^{\text{pre}}] \leq -\langle v, M v \rangle$$

where $v := (\text{Var}_x, \text{Var}_v, B)^\top$ and $M$ has negative real eigenvalues.

**Why it matters:** Linear algebra perspective on hypocoercivity. Eigenvalues determine convergence rate.

**Related:** `thm-hypocoercive-main`, `thm-main-convergence`

---

## 9. Convergence to QSD

### Quasi-Stationary Distribution (QSD)
**Label:** `def-qsd-adaptive`
**Source:** [15_geometric_gas_lsi_proof.md § 3.3](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `qsd`, `stationary`, `definition`

**What it says:** Stationary distribution of the algorithm conditioned on staying alive.

**Math:**
$$\nu_{\text{QSD}}(\cdot) := \lim_{t \to \infty} \mathbb{P}[S_t \in \cdot | \text{alive at time } t]$$

Satisfies: $\Psi_{\text{total}}^* \nu_{\text{QSD}} = \nu_{\text{QSD}}$ (fixed point).

**Why it matters:** Long-term behavior of algorithm. Target distribution for optimization.

**Related:** `thm-qsd-existence-corrected`, `thm-qsd-stability`

---

### QSD Existence and Uniqueness
**Label:** `thm-qsd-existence-corrected`
**Source:** [16_convergence_mean_field.md § 1.4](source/2_geometric_gas/16_convergence_mean_field)
**Tags:** `qsd`, `theorem`, `existence`

**What it says:** Under framework axioms, QSD exists and is unique.

**Math:** There exists unique $\nu_{\text{QSD}} \in \mathcal{P}(\Sigma_N)$ satisfying:
1. $\Psi_{\text{total}}^* \nu_{\text{QSD}} = \nu_{\text{QSD}}$ (stationarity)
2. $\nu_{\text{QSD}}(\mathcal{A}^N) = 1$ (alive set has full measure)
3. $\nu_{\text{QSD}}$ has smooth density (hypoelliptic regularity)

**Proof ingredients:** Irreducibility + Foster-Lyapunov + hypocoercivity.

**Why it matters:** Guarantees algorithm has well-defined target. No cyclic or chaotic behavior.

**Related:** `lem-irreducibility`, `thm-qsd-stability`, `thm-qsd-smoothness`

---

### Geometric Ergodicity
**Label:** `thm-main-convergence`
**Source:** [18_emergent_geometry.md § 2.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `convergence`, `geometric-ergodicity`, `theorem`, `main-result`, `N-uniform`

**What it says:** Algorithm converges exponentially fast to QSD at **N-uniform** rate.

**Math:**
$$W_2(\Psi_{\text{total}}^t \mu_0, \nu_{\text{QSD}}) \leq C \cdot e^{-\lambda t} W_2(\mu_0, \nu_{\text{QSD}})$$

where $\lambda > 0$ is **independent of N** (swarm size).

**Why it matters:** Finite-time convergence guarantees. Scaling to large populations.

**Related:** `thm-foster-lyapunov-adaptive`, `thm-explicit-total-rate`

---

### Foster-Lyapunov Condition
**Label:** `thm-foster-lyapunov-adaptive`
**Source:** [18_emergent_geometry.md § 4.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `foster-lyapunov`, `theorem`, `drift`, `convergence`

**What it says:** Synergistic Lyapunov function satisfies drift inequality: expected decrease outside compact set.

**Math:**
$$\mathbb{E}[\mathcal{V}(\Psi \mathcal{S}) | \mathcal{S}] \leq (1 - \lambda) \mathcal{V}(\mathcal{S}) + C$$

for some $\lambda > 0$, $C < \infty$ **independent of N**.

**Why it matters:** Standard machinery for proving geometric ergodicity (Meyn-Tweedie).

**Related:** `def-full-synergistic-lyapunov-function`, `thm-complete-variance-drift`

---

### Explicit Convergence Time
**Label:** `cor-explicit-convergence-time`
**Source:** [18_emergent_geometry.md § 5.5](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `convergence-time`, `mixing-time`, `corollary`, `explicit`

**What it says:** Number of iterations to reach $\epsilon$-accuracy.

**Math:**
$$t_{\text{mix}}(\epsilon) = O\left( \frac{1}{\lambda_{\text{total}}} \log \frac{1}{\epsilon} \right)$$

where $\lambda_{\text{total}} = \min(\lambda_{\text{clone}}, \lambda_{\text{kin}}, \lambda_B) / (1 + \kappa)$ with $\kappa$ measuring operator coupling strength.

**Why it matters:** Practical iteration budget. Algorithmic design via $\lambda$ maximization.

**Related:** `thm-explicit-total-rate`, `obs-three-regimes`

---

### Three Bottleneck Regimes
**Label:** `obs-three-regimes`
**Source:** [18_emergent_geometry.md § 5.6](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `convergence`, `regimes`, `bottleneck`

**What it says:** Convergence rate limited by slowest operator.

**Regimes:**
1. **Cloning-limited:** $\lambda_{\text{clone}} \ll \lambda_{\text{kin}}, \lambda_B$ → increase $\alpha, \beta$ (fitness weights)
2. **Hypocoercivity-limited:** $\lambda_{\text{kin}} \ll \lambda_{\text{clone}}, \lambda_B$ → increase $\gamma$ (friction) or decrease $\tau$ (timestep)
3. **Boundary-limited:** $\lambda_B \ll \lambda_{\text{clone}}, \lambda_{\text{kin}}$ → increase $\lambda_{\Phi}$ (boundary penalty)

**Why it matters:** Diagnostics for parameter tuning. Identify and address performance bottlenecks.

**Related:** `cor-explicit-convergence-time`, `obs-regularization-tradeoff`

---

## 10. Mean-Field Limit and McKean-Vlasov PDE

### McKean-Vlasov PDE (Mean-Field Limit)
**Label:** (described in `07_mean_field.md`)
**Source:** [07_mean_field.md](source/1_euclidean_gas/07_mean_field)
**Tags:** `mean-field`, `pde`, `mckean-vlasov`

**What it says:** As $N \to \infty$, empirical distribution converges to deterministic PDE.

**Math:**
$$\partial_t \rho = \nabla \cdot (\rho \nabla V_{\text{eff}}[\rho]) + T \Delta \rho$$

where $V_{\text{eff}}[\rho](x)$ is fitness potential depending on full distribution $\rho$.

**Why it matters:** Continuum limit enables PDE analysis tools. Understand infinite-population behavior.

**Related:** `thm-mean-field-convergence`, `def-mean-field-fitness-potential`

---

### Mean-Field Convergence Rate
**Label:** (described in `07_mean_field.md`)
**Source:** [07_mean_field.md](source/1_euclidean_gas/07_mean_field)
**Tags:** `mean-field`, `convergence`, `theorem`

**What it says:** Empirical distribution converges to mean-field limit at rate $O(1/\sqrt{N})$.

**Math:**
$$W_2(\mu_N, \rho_{\text{MF}}) \leq \frac{C}{\sqrt{N}}$$

**Why it matters:** Quantifies finite-$N$ effects. Large swarms approximate continuum well.

**Related:** `thm-propagation-chaos`, `thm-correlation-decay`

---

## 11. Propagation of Chaos

### Propagation of Chaos
**Label:** `thm-propagation-chaos-qsd`
**Source:** [10_qsd_exchangeability_theory.md § A1.2.1](source/1_euclidean_gas/10_qsd_exchangeability_theory)
**Tags:** `propagation-chaos`, `qsd`, `theorem`, `mean-field`

**What it says:** QSD factorizes asymptotically as $N \to \infty$.

**Math:**
$$\nu_{\text{QSD}}^{(N)}(dx_1, \ldots, dx_N) \to \rho(dx_1) \otimes \cdots \otimes \rho(dx_N)$$

where $\rho$ is single-particle marginal.

**Why it matters:** Walkers become asymptotically independent. Mean-field approximation is exact in large-$N$ limit.

**Related:** `thm-exchangeability-qsd`, `thm-correlation-decay`

---

### Quantitative Decorrelation
**Label:** `thm-correlation-decay`
**Source:** [10_qsd_exchangeability_theory.md § A1.2.2](source/1_euclidean_gas/10_qsd_exchangeability_theory)
**Tags:** `theorem`, `correlation`, `decorrelation`

**What it says:** Correlation between walkers decays as $O(1/N)$.

**Math:**
$$|\mathbb{E}_{\nu_{\text{QSD}}}[f(x_i) g(x_j)] - \mathbb{E}[f] \mathbb{E}[g]| \leq \frac{C}{N} \|f\| \|g\|$$

for $i \neq j$.

**Why it matters:** Quantifies chaos propagation. Finite-$N$ corrections are small.

**Related:** `thm-propagation-chaos-qsd`, `cor-mean-field-lsi`

---

### Exchangeability of QSD
**Label:** `thm-qsd-exchangeability`
**Source:** [10_qsd_exchangeability_theory.md § A1.1.1](source/1_euclidean_gas/10_qsd_exchangeability_theory)
**Tags:** `qsd`, `theorem`, `exchangeability`

**What it says:** QSD is invariant under permutation of walker indices.

**Math:**
$$\nu_{\text{QSD}}(dx_{\sigma(1)}, \ldots, dx_{\sigma(N)}) = \nu_{\text{QSD}}(dx_1, \ldots, dx_N)$$

for any permutation $\sigma \in S_N$.

**Why it matters:** No special walkers. Enables Hewitt-Savage representation and propagation of chaos.

**Related:** `thm-hewitt-savage-representation`, `thm-propagation-chaos-qsd`

---

### Tightness of QSD Marginals
**Label:** `thm-qsd-marginals-are-tight`
**Source:** [08_propagation_chaos.md § 2.1](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `qsd`, `tightness`, `mean-field`

**What it says:** The sequence of single-particle marginal measures $\{\mu_N\}_{N=2}^\infty$ is tight in $\mathcal{P}(\Omega)$.

**Math:**
For any $\epsilon > 0$, there exists compact $K_\epsilon \subset \Omega$ such that:

$$\mu_N(K_\epsilon) \geq 1 - \epsilon \quad \forall N \geq 2$$

**Why it matters:** Ensures existence of convergent subsequences. Prerequisite for all mean-field limit theorems.

**Related:** `thm-limit-is-weak-solution`, `thm-uniqueness-of-qsd`

---

### Extinction Rate Vanishes in Mean-Field Limit
**Label:** `thm-extinction-rate-vanishes`
**Source:** [08_propagation_chaos.md § 2.2](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `extinction`, `mean-field`

**What it says:** The extinction rate $\lambda_N$ of the N-particle QSD vanishes as $N \to \infty$.

**Math:**
$$\lim_{N \to \infty} \lambda_N = 0$$

**Why it matters:** In the thermodynamic limit, the system becomes stable—no mass loss through boundaries. The QSD becomes a true stationary distribution.

**Related:** `thm-limit-is-weak-solution`, `def-qsd-adaptive`

---

### Weak Solution to Stationary Mean-Field PDE
**Label:** `thm-limit-is-weak-solution`
**Source:** [08_propagation_chaos.md § 2.3](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `mean-field`, `pde`, `existence`

**What it says:** Any limit point $\mu_\infty$ of the marginal sequence is a weak solution to the stationary mean-field coupled system.

**Math:**
$$L^\dagger \rho_0 + S[\rho_0] + B[\rho_0] = 0$$

where $L^\dagger$ is kinetic adjoint, $S[\rho_0]$ is cloning source, $B[\rho_0]$ is boundary absorption.

**Why it matters:** **Existence of mean-field stationary distribution.** Connects finite-N QSDs to continuum PDE.

**Related:** `thm-uniqueness-of-qsd`, `thm-extinction-rate-vanishes`

---

### Hörmander's Theorem for Kinetic Operators
**Label:** `thm-uniqueness-hormander`
**Source:** [08_propagation_chaos.md § 3.2](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `hypoellipticity`, `hormander`

**What it says:** Kinetic operator $L$ satisfies Hörmander's condition, implying hypoellipticity.

**Math:**
The Lie algebra generated by drift and diffusion vector fields spans $\mathbb{R}^{2d}$ at every point.

**Why it matters:** Guarantees regularity of solutions. Enables strong maximum principle and uniqueness arguments.

**Related:** `thm-uniqueness-hypoelliptic-regularity`, `thm-uniqueness-of-qsd`

---

### Hypoelliptic Regularity
**Label:** `thm-uniqueness-hypoelliptic-regularity`
**Source:** [08_propagation_chaos.md § 3.3](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `regularity`, `hypoellipticity`

**What it says:** Solutions to $\mathcal{L}_{\text{lin}} u = f$ are $C^\infty$ in the interior.

**Math:**
If $f \in L^2_w(\Omega)$ and $\mathcal{L}_{\text{lin}} u = f$ in the weak sense, then $u \in C^\infty(\text{int}(\Omega))$.

**Why it matters:** Elevates weak solutions to classical solutions. Enables pointwise estimates.

**Related:** `thm-uniqueness-hormander`, `thm-uniqueness-uniqueness-stationary-solution`

---

### Uniqueness of Stationary Solution
**Label:** `thm-uniqueness-of-qsd`
**Source:** [08_propagation_chaos.md § 3.5](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `uniqueness`, `qsd`, `mean-field`

**What it says:** There is at most one probability density $\rho \in \mathcal{P}(\Omega)$ that is a weak solution to the stationary mean-field equation.

**Math:**
If $\rho_1, \rho_2$ are two weak solutions, then $\rho_1 = \rho_2$ almost everywhere.

**Why it matters:** **Uniqueness of mean-field QSD.** Combined with existence, proves all subsequences converge to the same limit.

**Related:** `thm-limit-is-weak-solution`, `thm-thermodynamic-limit`

---

### Thermodynamic Limit
**Label:** `thm-thermodynamic-limit`
**Source:** [08_propagation_chaos.md § 4.1](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `theorem`, `convergence`, `mean-field`, `thermodynamic-limit`

**What it says:** Macroscopic observables in N-particle QSD converge to mean-field expectations.

**Math:**
Let $\phi: \Omega \to \mathbb{R}$ be bounded and continuous. Then:

$$\lim_{N \to \infty} \mathbb{E}_{\nu_{\text{QSD}}^{(N)}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(x_i, v_i) \right] = \int_\Omega \phi(x, v) \, \rho_0(x, v) \, dx \, dv$$

where $\rho_0$ is the unique mean-field stationary density.

**Why it matters:** **Law of large numbers for QSD.** Justifies mean-field approximation for physical observables.

**Related:** `cor-w2-convergence-thermodynamic-limit`, `thm-propagation-chaos-qsd`

---

### Wasserstein-2 Convergence in Thermodynamic Limit
**Label:** `cor-w2-convergence-thermodynamic-limit`
**Source:** [08_propagation_chaos.md § 4.2](source/1_euclidean_gas/08_propagation_chaos)
**Tags:** `corollary`, `convergence`, `wasserstein`

**What it says:** Empirical measure of N-particle QSD converges to $\rho_0$ in Wasserstein-2 metric.

**Math:**
$$\lim_{N \to \infty} W_2\left( \frac{1}{N} \sum_{i=1}^N \delta_{(x_i, v_i)}, \rho_0 \right) = 0$$

in probability under $\nu_{\text{QSD}}^{(N)}$.

**Why it matters:** Stronger than weak convergence. Guarantees convergence of all moments.

**Related:** `thm-thermodynamic-limit`, `thm-uniqueness-of-qsd`

---

## 12. KL-Divergence Convergence and LSI Theory

### Logarithmic Sobolev Inequality (LSI)
**Label:** `def-lsi-adaptive`
**Source:** [15_geometric_gas_lsi_proof.md § 3.4](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `lsi`, `definition`, `entropy`

**What it says:** Entropy production controls KL-divergence to equilibrium.

**Math:**
$$\text{Ent}_\nu[\rho] \leq \frac{1}{2\lambda_{\text{LSI}}} \mathcal{I}_\nu[\rho]$$

where:
- $\text{Ent}[\rho] := \int \rho \log(\rho/\nu) \, d\mu$ (relative entropy)
- $\mathcal{I}[\rho] := \int \|\nabla \sqrt{\rho}\|^2 \, d\mu$ (Fisher information)

**Why it matters:** Implies exponential KL-convergence with rate $\lambda_{\text{LSI}}$.

**Related:** `thm-adaptive-lsi-main`, `thm-exp-convergence-standalone`

---

### N-Uniform LSI for Geometric Gas
**Label:** `thm-adaptive-lsi-main`
**Source:** [15_geometric_gas_lsi_proof.md § 9.1](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `lsi`, `theorem`, `N-uniform`, `main-result`, `geometric-gas`

**What it says:** QSD of Geometric Gas satisfies LSI with constant **independent of N**.

**Math:**
$$\lambda_{\text{LSI}} = c \cdot \min(\gamma, \lambda_{\text{adapt}}, \lambda_{\text{visc}})$$

where $c > 0$ is **N-uniform**.

**Why it matters:** Breakthrough result. LSI with N-uniform constant is rare for interacting particle systems.

**Proof strategy:** Cattiaux-Guillin hypocoercivity framework + N-uniform Poincaré for velocities + regularized Hessian ellipticity.

**Related:** `thm-cattiaux-guillin-verification`, `thm-qsd-poincare-rigorous`

---

### Hypocoercive LSI for Kinetic Operator
**Label:** `lem-kinetic-lsi-hypocoercive`
**Source:** [09_kl_convergence.md § 2.2.2](source/1_euclidean_gas/09_kl_convergence)
**Tags:** `kinetic`, `lsi`, `lemma`, `hypocoercivity`

**What it says:** Discrete-time BAOAB satisfies LSI via Villani hypocoercivity.

**Math:**
$$\text{Ent}_{\nu_{\text{kin}}}[\rho] \leq \frac{1}{2\lambda_{\text{kin}}} \mathcal{I}_{\nu_{\text{kin}}}[\rho]$$

where $\lambda_{\text{kin}} \sim \gamma \wedge (T/\tau^2)$ depends on friction, temperature, timestep.

**Why it matters:** Kinetic operator alone has exponential KL-convergence (before cloning).

**Related:** `thm-villani-hypocoercivity`, `prop-hypocoercivity-piecewise`

---

### Dobrushin Contraction for Euclidean Gas
**Label:** `thm-dobrushin-contraction`
**Source:** [09_kl_convergence.md § 3.4](source/1_euclidean_gas/09_kl_convergence)
**Tags:** `theorem`, `dobrushin`, `contraction`, `cloning`

**What it says:** Cloning operator is Dobrushin-contractive in status-change metric.

**Math:**
$$\|\Psi_{\text{clone}}^* \mu - \Psi_{\text{clone}}^* \nu\|_{\text{TV}} \leq (1 - \delta) \|\mu - \nu\|_{\text{TV}}$$

for some $\delta > 0$ **independent of N**.

**Why it matters:** Total variation contraction weaker than LSI but sufficient for certain convergence results.

**Related:** `thm-exponential-convergence-status`, `lem-softmax-lipschitz-status`

---

### Exponential KL Convergence for Non-Convex Fitness
**Label:** `thm-nonconvex-main`
**Source:** [09_kl_convergence.md § 4.1](source/1_euclidean_gas/09_kl_convergence)
**Tags:** `convergence`, `fitness`, `theorem`, `non-convex`, `kl-divergence`

**What it says:** Under confining potential (not necessarily convex), algorithm has exponential KL-convergence.

**Math:**
$$\text{KL}(\mu_t \| \nu_{\text{QSD}}) \leq e^{-\lambda t} \text{KL}(\mu_0 \| \nu_{\text{QSD}})$$

**Assumptions:**
- Confining: $\nabla V \cdot x \geq c \|x\|^2$ outside large ball
- Bounded Hessian: $\|\nabla^2 V\| \leq C$

**Why it matters:** Handles realistic fitness landscapes (multi-modal, non-convex). Not limited to log-concave QSD.

**Related:** `ax-confining-complete`, `thm-fl-recap`

---

# Chapter 2: Geometric Gas

## 13. Geometric Gas: Adaptive Mechanisms

### Localization Kernel
**Label:** `def-localization-kernel`
**Source:** [11_geometric_gas.md § 1.0.2](source/2_geometric_gas/11_geometric_gas)
**Tags:** `localization`, `geometric-gas`, `kernel`

**What it says:** Weight function for computing local statistics around each walker.

**Math:**
$$w_\rho(x_i, x_j) := \frac{K_\rho(d(x_i, x_j))}{\sum_k K_\rho(d(x_i, x_k))}$$

Typical: Gaussian kernel $K_\rho(r) = \exp(-r^2 / 2\rho^2)$ with bandwidth $\rho > 0$.

**Why it matters:** Enables local mean-field approximation. Each walker responds to nearby neighbors, not global swarm.

**Related:** `def-localized-mean-field-moments`, `def-unified-z-score`

---

### Localized Mean-Field Moments
**Label:** `def-localized-mean-field-moments`
**Source:** [11_geometric_gas.md § 1.0.3](source/2_geometric_gas/11_geometric_gas)
**Tags:** `mean-field`, `localization`, `moments`

**What it says:** Compute mean and variance using weighted neighbors, not full swarm.

**Math:**
$$\bar{V}_i^\rho := \sum_j w_\rho(x_i, x_j) V_j, \quad \sigma_i^2(\rho) := \sum_j w_\rho(x_i, x_j) (V_j - \bar{V}_i^\rho)^2$$

**Why it matters:** Spatially heterogeneous fitness landscape. Walker's fitness depends on local context.

**Related:** `def-unified-z-score`, `def-localized-mean-field-fitness`

---

### Unified Localized Z-Score
**Label:** `def-unified-z-score`
**Source:** [11_geometric_gas.md § 1.0.4](source/2_geometric_gas/11_geometric_gas)
**Tags:** `z-score`, `localization`, `fitness`

**What it says:** Standardize fitness using local statistics.

**Math:**
$$z_i(\rho) := \frac{V_i - \bar{V}_i^\rho}{\sigma_{\text{reg}}(\sigma_i(\rho))}$$

**Why it matters:** Local fitness relative to neighborhood. Enables adaptive force computation.

**Related:** `def-localized-mean-field-fitness`, `def-localized-mean-field-moments`

---

### Localized Mean-Field Fitness Potential
**Label:** `def-localized-mean-field-fitness`
**Source:** [11_geometric_gas.md § 2.1](source/2_geometric_gas/11_geometric_gas)
**Tags:** `fitness`, `mean-field`, `localization`, `adaptive-gas`

**What it says:** Fitness potential computed from local z-score.

**Math:**
$$V_{\text{MF}}(x_i) := g_{\text{rescale}}(z_i(\rho))$$

**Why it matters:** Drives adaptive force $F_{\text{adapt}} = -\nabla V_{\text{MF}}$. Pulls walkers toward locally fit regions.

**Related:** `def-unified-z-score`, `prop:bounded-adaptive-force`

---

### Adaptive Viscous Fluid SDE
**Label:** `def-hybrid-sde`
**Source:** [11_geometric_gas.md § 2](source/2_geometric_gas/11_geometric_gas)
**Tags:** `sde`, `langevin`, `adaptive-gas`, `viscous-coupling`

**What it says:** Extended Langevin dynamics with three adaptive mechanisms.

**Math:**
$$\begin{aligned}
dx_i &= v_i \, dt \\
dv_i &= -\frac{1}{\lambda_v}\nabla V(x_i) dt - \gamma v_i dt + F_{\text{adapt},i} dt + F_{\text{visc},i} dt + \sqrt{2\gamma T} D_i^{1/2} dW_i
\end{aligned}$$

**Three mechanisms:**
1. **Adaptive force:** $F_{\text{adapt}} = -\nabla V_{\text{MF}}$ (mean-field fitness gradient)
2. **Viscous coupling:** $F_{\text{visc}} = \sum_j K_{\text{visc}}(x_i, x_j)(v_j - v_i)$ (velocity alignment)
3. **Anisotropic diffusion:** $D_i = \epsilon_\Sigma I + (1 - \epsilon_\Sigma) H_i$ (regularized Hessian)

**Why it matters:** Full Geometric Gas dynamics. Three synergistic mechanisms enhance convergence.

**Related:** `def-regularized-hessian-tensor`, `ax:viscous-kernel`

---

### Regularized Hessian Diffusion Tensor
**Label:** `def-regularized-hessian-tensor`
**Source:** [11_geometric_gas.md § 2](source/2_geometric_gas/11_geometric_gas)
**Tags:** `diffusion`, `anisotropic`, `hessian`, `regularization`

**What it says:** Diffusion tensor adapts to fitness curvature with spectral floor.

**Math:**
$$D_i := \epsilon_\Sigma I + (1 - \epsilon_\Sigma) H_i, \quad H_i := \nabla^2 V(x_i)$$

Regularization $\epsilon_\Sigma \in (0, 1)$ ensures uniform ellipticity.

**Why it matters:** Anisotropic noise explores more along low-curvature directions. Accelerates escape from saddles.

**Related:** `thm-uniform-ellipticity`, `def-d-adaptive-diffusion`

---

### Uniform Ellipticity of Regularized Hessian (UEPH)
**Label:** `thm-ueph`
**Source:** [11_geometric_gas.md § 3.2](source/2_geometric_gas/11_geometric_gas)
**Tags:** `theorem`, `ellipticity`, `hessian`, `regularity`

**What it says:** Regularized Hessian tensor is uniformly elliptic with N-independent bounds.

**Math:**
For all $i$ and all $\xi \in \mathbb{R}^d$:

$$\epsilon_\Sigma \|\xi\|^2 \leq \xi^\top D_i \xi \leq (1 + C_V \epsilon_\Sigma) \|\xi\|^2$$

where $C_V$ depends on $\|\nabla^2 V\|_\infty$ but not on $N$ or localization radius $\rho$.

**Why it matters:** **Cornerstone of geometric gas theory.** Guarantees non-degenerate diffusion and enables hypocoercivity. Uniform bounds allow N-uniform LSI.

**Related:** `def-regularized-hessian-tensor`, `thm-adaptive-lsi-main`, `prop-lipschitz-diffusion`

---

### Foster-Lyapunov Drift for Geometric Gas
**Label:** `thm-fl-drift-adaptive`
**Source:** [11_geometric_gas.md § 4.3](source/2_geometric_gas/11_geometric_gas)
**Tags:** `theorem`, `convergence`, `foster-lyapunov`, `drift`

**What it says:** Geometric Gas satisfies Foster-Lyapunov condition with critical fitness threshold $\epsilon_F^*(\rho)$.

**Math:**
There exist constants $\kappa > 0$, $C < \infty$ such that:

$$\mathbb{E}[V_{\text{Lyap}}(S_{k+1}) \mid S_k] \leq (1 - \kappa \cdot \mathbb{1}_{\epsilon_F < \epsilon_F^*(\rho)}) V_{\text{Lyap}}(S_k) + C$$

where $\epsilon_F^*(\rho) = O(\rho^{-3})$ is the critical threshold.

**Why it matters:** Establishes exponential convergence to QSD neighborhood. Reveals fundamental tradeoff: smaller $\rho$ (more local) requires larger $\epsilon_F$ (stronger fitness signal).

**Related:** `thm-main-convergence`, `thm-backbone-convergence`, `obs-regularization-tradeoff`

---

### Backbone Geometric Ergodicity
**Label:** `thm-backbone-convergence`
**Source:** [11_geometric_gas.md § 4.1](source/2_geometric_gas/11_geometric_gas)
**Tags:** `theorem`, `ergodicity`, `convergence`

**What it says:** The backbone system (Euclidean Gas without adaptive mechanisms) is geometrically ergodic.

**Math:**
$$\mathbb{E}[V_{\text{Lyap}}(S_{k+1}) \mid S_k] \leq (1 - \kappa_{\text{backbone}}) V_{\text{Lyap}}(S_k) + C_{\text{backbone}}$$

for constants $\kappa_{\text{backbone}} > 0$ and $C_{\text{backbone}} < \infty$.

**Why it matters:** Validates "stable backbone + adaptive perturbation" philosophy. Adaptive mechanisms perturb a provably convergent base system.

**Related:** `thm-fl-drift-adaptive`, `def-hybrid-sde`

---

### Stratonovich Chain Rule for Lyapunov Functions
**Label:** `thm-strat-chain`
**Source:** [11_geometric_gas.md § 4.2](source/2_geometric_gas/11_geometric_gas)
**Tags:** `theorem`, `stochastic-calculus`, `stratonovich`

**What it says:** Lyapunov drift for Stratonovich SDE follows classical chain rule plus correction term.

**Math:**
$$d V(S_t) = \nabla V \cdot b_{\text{tot}} \, dt + \frac{1}{2} \text{Tr}(\nabla^2 V \cdot \Sigma_{\text{tot}}) dt + \nabla V \cdot \sigma_{\text{tot}} \circ dW_t$$

where $\Sigma_{\text{tot}} = \sigma_{\text{tot}} \sigma_{\text{tot}}^\top$ is total diffusion matrix.

**Why it matters:** Enables clean computation of expected drift. Stratonovich form avoids Itô correction clutter when working with geometric objects.

**Related:** `thm-fl-drift-adaptive`, `lem-ito-correction-bound`

---

### Velocity Variance Contraction (Anisotropic)
**Label:** `thm-velocity-variance-anisotropic`
**Source:** [11_geometric_gas.md § 5.1](source/2_geometric_gas/11_geometric_gas)
**Tags:** `theorem`, `variance`, `contraction`, `anisotropic`

**What it says:** Anisotropic diffusion maintains velocity variance contraction.

**Math:**
$$\mathbb{E}[\text{Var}(v_{k+1}) \mid S_k] \leq (1 - \kappa_v) \text{Var}(v_k) + C_v$$

where $\kappa_v$ depends on friction $\gamma$ and $C_v$ depends on diffusion ellipticity bounds.

**Why it matters:** Proves adaptive mechanisms don't破坏 kinetic operator's dissipation. Anisotropic noise compatible with hypocoercivity.

**Related:** `thm-hypocoercive-main`, `thm-ueph`

---

## 14. Emergent Riemannian Geometry

### Adaptive Diffusion Tensor (as Inverse Metric)
**Label:** `def-d-adaptive-diffusion`
**Source:** [18_emergent_geometry.md § 1.1](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `diffusion`, `anisotropic`, `riemannian-metric`, `emergent-geometry`

**What it says:** Diffusion tensor defines Riemannian metric via $g = D^{-1}$.

**Math:**
$$D(x) := \epsilon_\Sigma I + (1 - \epsilon_\Sigma) \nabla^2 V(x), \quad g(x) := D(x)^{-1}$$

**Why it says Riemannian metric:**
- Symmetric positive-definite by construction
- Defines inner product: $\langle u, v \rangle_g := u^\top g v$
- Geodesic distance: $d_g(x, y) := \inf_\gamma \int \sqrt{\dot{\gamma}^\top g(\gamma) \dot{\gamma}} \, dt$

**Why it matters:** **Emergent geometry from algorithmic parameters.** Walkers move on curved space shaped by fitness landscape.

**Connection to information geometry:** $g \sim$ Fisher information metric when $V = -\log \rho$.

**Related:** `obs-emergent-metric`, `def-emergent-manifold`

---

### Uniform Ellipticity by Construction
**Label:** `thm-uniform-ellipticity`
**Source:** [18_emergent_geometry.md § 1.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `ellipticity`, `diffusion-bounds`, `regularization`, `N-uniform`, `theorem`

**What it says:** Regularization ensures diffusion eigenvalues bounded away from zero.

**Math:**
$$\epsilon_\Sigma I \preceq D(x) \preceq (1 + (1 - \epsilon_\Sigma) C_H) I$$

uniformly over $x \in \mathcal{X}$, **independent of N**.

**Why it matters:** Prevents degeneracy. All directions have positive diffusion → irreducibility.

**Related:** `assump-spectral-floor`, `thm-uniform-ellipticity-explicit`

---

### Lipschitz Continuity of Adaptive Diffusion
**Label:** `prop-lipschitz-diffusion`
**Source:** [18_emergent_geometry.md § 1.3](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `lipschitz`, `continuity`, `diffusion`, `N-uniform`, `proposition`

**What it says:** Diffusion tensor is Lipschitz in position.

**Math:**
$$\|D(x_1) - D(x_2)\| \leq L_D \cdot d(x_1, x_2)$$

where $L_D = (1 - \epsilon_\Sigma) \cdot \|\nabla^3 V\|_\infty$ is **N-uniform** (proven via C³ regularity).

**Why it matters:** Regularity critical for SDE well-posedness and Wasserstein contraction.

**Related:** `thm-c4-regularity`, `thm-fitness-third-deriv-proven`

---

### Kinetic Operator with Adaptive Diffusion
**Label:** `def-d-kinetic-operator-adaptive`
**Source:** [18_emergent_geometry.md § 1.4](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `kinetic-operator`, `langevin`, `stratonovich`, `anisotropic`, `sde`

**What it says:** BAOAB integrator adapted for state-dependent diffusion.

**Math:** Stratonovich interpretation:
$$dv_i = -\frac{1}{\lambda_v}\nabla V(x_i) dt - \gamma v_i dt + \sqrt{2\gamma T} D_i^{1/2} \circ dW_i$$

**Key difference from isotropic:** Itô correction term $\frac{1}{2}\nabla \cdot D$ appears.

**Why it matters:** Preserves geometric structure under coordinate changes. Natural for Riemannian SDEs.

**Related:** `thm-coordinate-invariance`, `lem-ito-correction-bound`

---

### Coordinate Invariance (Refined)
**Label:** `thm-coordinate-invariance`
**Source:** [18_emergent_geometry.md § 1.6](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `coordinate-invariance`, `stratonovich`, `geometric-invariance`, `riemannian`, `theorem`

**What it says:** Stratonovich SDE transforms correctly under coordinate changes.

**Math:** If $\phi: \mathcal{X} \to \mathcal{X}'$ is diffeomorphism, then SDE in new coordinates:
$$dy = \phi_*(dx), \quad dy_i = \cdots + \sqrt{2\gamma T} (D\phi \cdot D_i \cdot D\phi^\top)^{1/2} \circ dW_i'$$

pulls back to original SDE.

**Why it matters:** Geometric Gas dynamics are intrinsic, not coordinate-dependent. Algorithm doesn't depend on parameterization choice.

**Related:** `obs-two-formulations`, `def-emergent-manifold`

---

### Emergent Riemannian Manifold
**Label:** `def-emergent-manifold`
**Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `riemannian-manifold`, `metric-tensor`, `geodesics`, `christoffel-symbols`, `volume-element`

**What it says:** State space becomes Riemannian manifold $(\mathcal{X}, g)$ with induced geometry.

**Geometric objects:**
- **Metric tensor:** $g_{ij}(x) := (D^{-1})_{ij}(x)$
- **Christoffel symbols:** $\Gamma_{ij}^k = \frac{1}{2} g^{k\ell} (\partial_i g_{j\ell} + \partial_j g_{i\ell} - \partial_\ell g_{ij})$
- **Geodesics:** $\ddot{\gamma}^k + \Gamma_{ij}^k \dot{\gamma}^i \dot{\gamma}^j = 0$
- **Volume element:** $d\text{Vol}_g = \sqrt{\det g} \, dx$

**Why it matters:** Full differential geometry machinery applies. Can compute curvature, parallel transport, etc.

**Related:** `def-d-adaptive-diffusion`, `prop-geodesics-fitness`

---

### Geodesics Favor High-Fitness Regions
**Label:** `prop-geodesics-fitness`
**Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `geodesics`, `fitness`, `natural-gradient`, `riemannian-distance`, `proposition`

**What it says:** Geodesic paths naturally route through high-fitness regions (short distance).

**Math:** If $V(x_1) < V(x_2)$ (higher fitness at $x_2$), then geodesic from $x_1$ to $x_2$ bends toward high-fitness regions.

**Intuition:** Metric $g = (\nabla^2 V + \epsilon I)^{-1}$ contracts in high-curvature (high-fitness) directions → shorter geodesics.

**Why it matters:** Natural gradient descent follows geodesics. Emergent geometry guides optimization.

**Related:** `obs-emergent-metric`, `prop-rate-metric-ellipticity`

---

### QSD Spatial Marginal is Riemannian Volume Measure
**Label:** `thm-qsd-spatial-riemannian-volume`
**Source:** [18_emergent_geometry.md § A.1](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `qsd`, `spatial-marginal`, `riemannian-volume`, `stratonovich`, `theorem`

**What it says:** Position marginal of QSD is proportional to Riemannian volume measure.

**Math:**
$$\rho_{\text{QSD}}^{\text{pos}}(x) \propto \sqrt{\det g(x)} = \sqrt{\det D(x)^{-1}}$$

**Why it matters:** QSD "fills" emergent Riemannian manifold uniformly (volume-preserving). Geometry determines stationary distribution.

**Proof idea:** Stratonovich SDE preserves volume form under appropriate gauge.

**Related:** `lem-companion-bias-riemannian`, `lem-velocity-marginalization`

---

### Algorithmic Tunability of Emergent Geometry
**Label:** `thm-algorithmic-tunability`
**Source:** [18_emergent_geometry.md § 9.6.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `algorithmic-tunability`, `parameters`, `geometric-control`, `design`, `theorem`

**What it says:** Algorithm designer controls emergent metric via hyperparameters.

**Tuning knobs:**
- $\epsilon_\Sigma$ (regularization): trade-off between isotropy and adaptation
- $\rho$ (localization bandwidth): spatial scale of mean-field interactions
- $\alpha, \beta$ (fitness weights): relative importance of reward vs diversity
- $\gamma, T$ (friction, temperature): thermalization timescale

**Why it matters:** **Programmable geometry.** Can engineer metric to favor certain convergence properties.

**Related:** `obs-regularization-tradeoff`, `obs-three-regimes`

---

## 15. Anisotropic Diffusion and Hypocoercivity

### Itô Correction Term Bound
**Label:** `lem-ito-correction-bound`
**Source:** [18_emergent_geometry.md § 3.1](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `ito-correction`, `stratonovich`, `state-dependent`, `N-uniform`, `lemma`

**What it says:** Itô vs Stratonovich difference is small under regularization.

**Math:**
$$\|\nabla \cdot D\| \leq C_{\text{Itô}} \cdot (1 - \epsilon_\Sigma) \cdot \|\nabla^3 V\|$$

where $C_{\text{Itô}}$ is **N-uniform**.

**Why it matters:** Perturbative correction. Main dynamics captured by Stratonovich formulation.

**Related:** `def-d-kinetic-operator-adaptive`, `thm-coordinate-invariance`

---

### Velocity Variance Contraction (Anisotropic)
**Label:** `thm-velocity-variance-anisotropic`
**Source:** [18_emergent_geometry.md § 3.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `velocity-variance`, `friction`, `contraction`, `anisotropic`, `N-uniform`, `theorem`

**What it says:** Friction dissipates velocity variance despite anisotropic noise.

**Math:**
$$\mathbb{E}[\text{Var}_v^{\text{post}}] \leq (1 - \gamma \tau) \text{Var}_v^{\text{pre}} + C_{\text{therm}}$$

where constant $C_{\text{therm}}$ depends on $T, \epsilon_\Sigma$ but is **N-uniform**.

**Why it matters:** Velocity still thermalizes under state-dependent diffusion. Hypocoercivity survives anisotropy.

**Related:** `def-d-hypocoercive-norm`, `rem-coupling-essential`

---

### Hypocoercive Norm
**Label:** `def-d-hypocoercive-norm`
**Source:** [18_emergent_geometry.md § 3.2.1](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `hypocoercivity`, `coupling`, `phase-space`, `norm`, `weighted`

**What it says:** Weighted combination of position and velocity errors.

**Math:**
$$\mathcal{H} := V_{\text{loc}} + \lambda_v^2 \text{Var}_v$$

Hypocoercivity: neither term contracts alone, but weighted sum does.

**Why it matters:** Canonical coupling for degenerate diffusion (position-only noise).

**Related:** `rem-coupling-essential`, `thm-hypocoercive-main`

---

### Hypocoercive Contraction for Adaptive Gas
**Label:** `thm-hypocoercive-main`
**Source:** [18_emergent_geometry.md § 3.2.5](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `hypocoercivity`, `anisotropic`, `main-result`, `wasserstein`, `N-uniform`, `theorem`

**What it says:** Kinetic operator contracts hypocoercive norm at **N-uniform** rate.

**Math:**
$$\mathbb{E}[\mathcal{H}^{\text{post}}] \leq (1 - \lambda_{\text{hypo}}) \mathcal{H}^{\text{pre}} + C_{\text{hypo}}$$

where $\lambda_{\text{hypo}} = c \cdot \min(\gamma, \epsilon_\Sigma)$ with $c > 0$ **independent of N**.

**Why it matters:** Extends classical hypocoercivity to anisotropic setting. Main technical achievement.

**Related:** `thm-velocity-variance-anisotropic`, `thm-location-error-anisotropic`

---

## 16. Geometric Gas Convergence

### Geometric Ergodicity of Adaptive Gas
**Label:** `thm-main-convergence`
**Source:** [18_emergent_geometry.md § 2.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `geometric-ergodicity`, `qsd`, `convergence`, `anisotropic`, `N-uniform`, `main-result`, `theorem`

**What it says:** Full Geometric Gas (kinetic + cloning + adaptive mechanisms) converges exponentially to QSD.

**Math:**
$$W_2(\Psi_{\text{Geo}}^t \mu_0, \nu_{\text{QSD}}) \leq C e^{-\lambda_{\text{total}} t}$$

where $\lambda_{\text{total}} > 0$ is **N-uniform**.

**Why it matters:** Main convergence theorem for Geometric Gas. All three adaptive mechanisms (force, viscosity, diffusion) preserve N-uniform convergence.

**Related:** `thm-foster-lyapunov-adaptive`, `thm-explicit-total-rate`

---

### Explicit Convergence Rate with Full Parameter Dependence
**Label:** `thm-explicit-total-rate`
**Source:** [18_emergent_geometry.md § 5.2](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `convergence-rate`, `explicit`, `parameters`, `N-uniform`, `theorem`

**What it says:** Total convergence rate is minimum of operator rates, modulated by coupling strength.

**Math:**
$$\lambda_{\text{total}} = \frac{\min(\lambda_{\text{clone}}, \lambda_{\text{kin}}, \lambda_B)}{1 + \kappa_{\text{coupling}}}$$

where coupling $\kappa$ measures inter-operator dependence.

**Why it matters:** Design formula. Optimize parameters to maximize slowest rate.

**Related:** `cor-explicit-convergence-time`, `obs-three-regimes`

---

### Regularization Trade-Off
**Label:** `obs-regularization-tradeoff`
**Source:** [18_emergent_geometry.md § 5.7](source/2_geometric_gas/18_emergent_geometry)
**Tags:** `regularization`, `trade-off`, `ellipticity`, `adaptation`, `robustness`

**What it says:** Regularization parameter $\epsilon_\Sigma$ trades adaptation speed for robustness.

**Trade-off:**
- **Low $\epsilon_\Sigma$:** More anisotropic → faster adaptation to curvature → faster convergence
- **High $\epsilon_\Sigma$:** More isotropic → slower adaptation → more robust to noise

**Optimal:** $\epsilon_\Sigma \sim 0.01$ - 0.1 (empirical sweet spot).

**Why it matters:** Practical tuning guidance. Balance exploration and exploitation.

**Related:** `thm-algorithmic-tunability`, `thm-uniform-ellipticity-explicit`

---

## 17. C³ and C⁴ Regularity Theory

### C³ Regularity of Fitness Potential
**Label:** (results throughout `13_geometric_gas_c3_regularity.md`)
**Source:** [13_geometric_gas_c3_regularity.md](source/2_geometric_gas/13_geometric_gas_c3_regularity)
**Tags:** `regularity`, `c3`, `fitness`, `lipschitz-hessian`

**What it says:** Under reasonable assumptions, fitness potential $V(x)$ has Lipschitz Hessian (C³).

**Math:** $\nabla^2 V$ is Lipschitz with constant **independent of N**.

**Why it matters:**
- Ensures diffusion tensor $D = \epsilon I + (1-\epsilon)\nabla^2 V$ is Lipschitz
- Enables Wasserstein contraction proofs for anisotropic kinetic operator
- Required for hypocoercivity analysis

**Related:** `thm-continuity-third-derivatives`, `prop-timestep-constraint`

---

### C⁴ Regularity of Fitness Potential
**Label:** `thm-c4-regularity`
**Source:** [14_geometric_gas_c4_regularity.md § 8](source/2_geometric_gas/14_geometric_gas_c4_regularity)
**Tags:** `regularity`, `c4`, `fitness`, `theorem`

**What it says:** With smooth primitives (C⁴ measurement, kernel, rescale), fitness potential is C⁴.

**Math:** Fourth derivatives exist and satisfy $\|\nabla^4 V\| \leq K_{V,4}(\rho)$ where $K_{V,4}$ is **N-uniform**.

**Why it matters:**
- Enables Bakry-Émery criterion for LSI (requires C² Hessian, i.e., C⁴ potential)
- Permits fourth-order integrators
- Conditional Brascamp-Lieb inequality

**Related:** `cor-hessian-lipschitz`, `prop-bakry-emery-gamma2`

---

### Hessian Lipschitz Continuity
**Label:** `cor-hessian-lipschitz`
**Source:** [14_geometric_gas_c4_regularity.md § 9](source/2_geometric_gas/14_geometric_gas_c4_regularity)
**Tags:** `lipschitz`, `hessian`, `corollary`, `regularity`

**What it says:** C⁴ regularity implies Lipschitz Hessian.

**Math:**
$$\|\nabla^2 V(x_1) - \nabla^2 V(x_2)\| \leq K_{V,4} \cdot d(x_1, x_2)$$

**Why it matters:** Critical for diffusion tensor Lipschitz continuity and SDE well-posedness.

**Related:** `prop-lipschitz-diffusion`, `thm-c4-regularity`

---

## 18. N-Uniform LSI for Geometric Gas

### N-Uniform Poincaré Inequality for QSD Velocities
**Label:** `thm-qsd-poincare-rigorous`
**Source:** [15_geometric_gas_lsi_proof.md § 8.3](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `qsd`, `poincare`, `theorem`, `N-uniform`, `velocity`

**What it says:** Velocity marginal of QSD satisfies Poincaré inequality with **N-uniform** constant.

**Math:**
$$\text{Var}_{\nu_{\text{QSD}}^v}[f] \leq \frac{C_P}{T} \mathbb{E}[\|\nabla f\|^2]$$

where $C_P$ is **independent of N**.

**Why it matters:** Key ingredient for LSI via Cattiaux-Guillin framework. Shows velocities thermalize uniformly.

**Related:** `lem-conditional-gaussian-qsd`, `thm-cattiaux-guillin-verification`

---

### Verification of Cattiaux-Guillin Hypotheses
**Label:** `thm-cattiaux-guillin-verification`
**Source:** [15_geometric_gas_lsi_proof.md § 8.5](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `theorem`, `hypocoercivity`, `lsi`, `cattiaux-guillin`

**What it says:** All hypotheses of Cattiaux-Guillin hypocoercive LSI framework are satisfied with **N-uniform** constants.

**Hypotheses verified:**
1. Velocity Poincaré with N-uniform constant ✓
2. Drift bounded perturbation of linear operator ✓
3. Commutator bounds from propagation of chaos ✓
4. Uniform ellipticity of diffusion ✓

**Why it matters:** Rigorous application of existing theory. Shows Geometric Gas fits into established framework.

**Related:** `thm-qsd-poincare-rigorous`, `thm-adaptive-lsi-main`

---

### N-Uniform LSI for Geometric Gas (Main Result)
**Label:** `thm-adaptive-lsi-main`
**Source:** [15_geometric_gas_lsi_proof.md § 9.1](source/2_geometric_gas/15_geometric_gas_lsi_proof)
**Tags:** `lsi`, `theorem`, `N-uniform`, `main-result`, `geometric-gas`, `geometry`, `metric`

**What it says:** QSD of full Geometric Gas satisfies LSI with constant **independent of N**.

**Math:**
$$\text{Ent}_{\nu_{\text{QSD}}}[\rho] \leq \frac{1}{2\lambda_{\text{LSI}}} \mathcal{I}_{\nu_{\text{QSD}}}[\rho]$$

where $\lambda_{\text{LSI}} = c \cdot \min(\gamma, T/\epsilon_\Sigma, \lambda_{\text{adapt}})$ with $c > 0$ **N-uniform**.

**Why it matters:**
- **Exponential KL-convergence** to QSD at N-uniform rate
- **Concentration of measure** for QSD (Talagrand, Bobkov-Ledoux)
- **Mean-field LSI** passes to continuum limit

**Proof ingredients:** Cattiaux-Guillin + N-uniform Poincaré + C³/C⁴ regularity + uniform ellipticity.

**Related:** `thm-cattiaux-guillin-verification`, `cor-mean-field-lsi`

---

# Cross-Cutting Themes

## 19. Key Inequalities and Bounds

### Triangle Inequality for Displacement
**Tags:** `inequality`, `metric`, `displacement`

**Math:** $d_{\text{Disp}}(\mathcal{S}_1, \mathcal{S}_3) \leq d_{\text{Disp}}(\mathcal{S}_1, \mathcal{S}_2) + d_{\text{Disp}}(\mathcal{S}_2, \mathcal{S}_3)$

---

### Cauchy-Schwarz for Variance Decomposition
**Tags:** `inequality`, `variance`, `decomposition`

**Math:** $\text{Var}[X + Y] \leq 2(\text{Var}[X] + \text{Var}[Y])$

Used ubiquitously in error decomposition proofs.

---

### Z-Score Norm Bound
**Label:** `thm-z-score-norm-bound`
**Source:** [01_fragile_gas_framework.md § 11.1.4](source/1_euclidean_gas/01_fragile_gas_framework)
**Tags:** `inequality`, `standardization`, `theorem`

**Math:** $\|z\|_2 \leq \sqrt{N}$ for standardized vector $z$ with $\bar{z} = 0$, $\text{Var}[z] = 1$.

**Why it matters:** Bounds propagation of standardization errors.

---

### Lipschitz Continuity of Empirical Moments
**Tags:** `inequality`, `lipschitz`, `moments`

**Math:** $|\bar{V}(\mathcal{S}_1) - \bar{V}(\mathcal{S}_2)| \leq \frac{1}{\sqrt{N}} \|V(\mathcal{S}_1) - V(\mathcal{S}_2)\|_2$

**Related:** `lem-empirical-aggregator-properties`

---

### Poincaré Inequality
**Tags:** `inequality`, `poincare`, `spectral-gap`

**Math:** $\text{Var}_\nu[f] \leq \frac{1}{\lambda_P} \int \|\nabla f\|^2 \, d\nu$

Constant $\lambda_P$ is spectral gap of generator.

**Related:** `thm-qsd-poincare-rigorous`

---

### Brascamp-Lieb Inequality (Conditional)
**Label:** `cor-brascamp-lieb`
**Source:** [14_geometric_gas_c4_regularity.md § 9](source/2_geometric_gas/14_geometric_gas_c4_regularity)
**Tags:** `inequality`, `covariance-bound`, `corollary`

**Math:** If $\nu \propto e^{-V}$ with $\nabla^2 V \succeq m I$, then $\text{Cov}_\nu[f, g] \leq \frac{1}{m} \mathbb{E}[\nabla f \cdot \nabla g]$.

**Condition:** Requires C² Hessian (C⁴ potential).

---

## 20. Critical Parameters and Constants

### Friction Coefficient $\gamma$
**Tags:** `parameter`, `kinetic`, `friction`

**Role:** Controls velocity thermalization rate in Langevin dynamics.

**Typical range:** $\gamma \in [0.1, 10]$. Higher $\gamma$ → faster thermalization.

**Impact:** Appears in kinetic contraction rate: $\lambda_{\text{kin}} \sim \gamma \wedge (T/\tau^2)$.

---

### Temperature $T$
**Tags:** `parameter`, `kinetic`, `thermalization`

**Role:** Sets equilibrium kinetic energy scale.

**Typical range:** $T \in [0.1, 1]$.

**Impact:** Higher $T$ → more exploration, slower convergence. Appears in noise amplitude $\sqrt{2\gamma T}$.

---

### Timestep $\tau$
**Tags:** `parameter`, `integrator`, `discretization`

**Role:** BAOAB integrator timestep.

**Typical range:** $\tau \in [0.01, 0.1]$.

**Constraint:** Must satisfy $\tau \lesssim \frac{\epsilon_\Sigma}{K_{V,3}(\rho)}$ for stability (from C³ regularity).

**Related:** `prop-timestep-constraint`

---

### Regularization $\epsilon_\Sigma$
**Tags:** `parameter`, `regularization`, `anisotropic`, `geometric-gas`

**Role:** Spectral floor for diffusion tensor: $D = \epsilon_\Sigma I + (1-\epsilon_\Sigma) H$.

**Typical range:** $\epsilon_\Sigma \in [0.01, 0.2]$.

**Trade-off:** Lower → more anisotropic (faster) but less stable. Higher → more isotropic (robust) but slower.

**Related:** `obs-regularization-tradeoff`, `thm-uniform-ellipticity`

---

### Localization Bandwidth $\rho$
**Tags:** `parameter`, `localization`, `geometric-gas`, `kernel`

**Role:** Spatial scale for local mean-field interactions.

**Typical range:** $\rho \in [0.1 \cdot \text{diam}(\mathcal{X}), 0.5 \cdot \text{diam}(\mathcal{X})]$.

**Impact:** Larger $\rho$ → more global (closer to full mean-field). Smaller $\rho$ → more local (spatially heterogeneous).

**Related:** `def-localization-kernel`, `def-localized-mean-field-moments`

---

### Fitness Weights $\alpha, \beta$
**Tags:** `parameter`, `fitness`, `exploration-exploitation`

**Role:** Balance reward ($\alpha$) vs diversity ($\beta$).

**Typical range:** $\alpha \in [0.5, 2]$, $\beta \in [0, 1]$.

**Impact:** High $\alpha$ → exploitation (converge fast to local optima). High $\beta$ → exploration (maintain diversity).

**Related:** `def-raw-value-operator`

---

### Revival Parameter $\kappa_{\text{revival}}$
**Label:** `def-axiom-guaranteed-revival`
**Tags:** `parameter`, `viability`, `revival`

**Formula:** $\kappa_{\text{revival}} = \frac{\eta^{\alpha+\beta}}{\epsilon_{\text{clone}} p_{\max}}$

**Constraint:** Must satisfy $\kappa_{\text{revival}} > 1$ for guaranteed revival.

**Related:** `thm-revival-guarantee`

---

### Keystone Constant $c_{\text{keystone}}$
**Tags:** `constant`, `cloning`, `N-uniform`, `keystone`

**Role:** Contraction rate for positional variance under cloning.

**Key property:** **Independent of N** (proven via quantitative Keystone Lemma).

**Appears in:** $\mathbb{E}[\text{Var}_x^{\text{post}}] \leq (1 - c_{\text{keystone}}) \text{Var}_x^{\text{pre}} + \cdots$

**Related:** `lem-quantitative-keystone`, `thm-positional-variance-contraction`

---

### LSI Constant $\lambda_{\text{LSI}}$
**Tags:** `constant`, `lsi`, `convergence-rate`, `N-uniform`

**Role:** Rate of KL-divergence convergence.

**Formula:** $\lambda_{\text{LSI}} = c \cdot \min(\gamma, T/\epsilon_\Sigma, \lambda_{\text{adapt}})$ where $c > 0$ is **N-uniform**.

**Why it matters:** Determines mixing time: $t_{\text{mix}} \sim \frac{1}{\lambda_{\text{LSI}}} \log \frac{1}{\epsilon}$.

**Related:** `thm-adaptive-lsi-main`, `thm-exp-convergence-standalone`

---

## Notation Conventions

### Greek Letters
- $\alpha$ (alpha): Exploitation weight for reward
- $\beta$ (beta): Exploitation weight for diversity (also: inverse temperature for companion selection)
- $\gamma$ (gamma): Friction coefficient
- $\delta$ (delta): Cloning noise scale
- $\epsilon$ (epsilon): Regularization parameters (e.g., $\epsilon_\Sigma$ for spectral floor, $\epsilon_{\text{clone}}$ for cloning threshold)
- $\eta$ (eta): Rescale lower bound
- $\lambda$ (lambda): Weight parameters (e.g., $\lambda_v$ for velocity, $\lambda_B$ for boundary)
- $\rho$ (rho): Localization bandwidth / probability density
- $\sigma$ (sigma): Noise scale / standard deviation
- $\tau$ (tau): Timestep
- $\nu$ (nu): Probability measure (typically QSD)
- $\mu$ (mu): Reference measure / probability measure
- $\Phi$ (Phi): Potential function / projection map
- $\Psi$ (Psi): Operator (e.g., $\Psi_{\text{clone}}$, $\Psi_{\text{kin}}$)

### Calligraphic Letters
- $\mathcal{A}$ (A): Alive set
- $\mathcal{D}$ (D): Dead set
- $\mathcal{S}$ (S): Swarm configuration
- $\mathcal{X}$ (X): State space
- $\mathcal{Y}$ (Y): Algorithmic space
- $\mathcal{V}$ (V): Lyapunov function
- $\mathcal{H}$ (H): Hypocoercive norm
- $\mathcal{I}$ (I): Fisher information

### Common Abbreviations
- **QSD:** Quasi-Stationary Distribution
- **LSI:** Logarithmic Sobolev Inequality
- **BAOAB:** Position-Velocity splitting integrator (B-drift, A-kick, O-Ornstein-Uhlenbeck, A-kick, B-drift)
- **W₂:** Wasserstein-2 metric
- **KL:** Kullback-Leibler divergence
- **SDE:** Stochastic Differential Equation
- **PDE:** Partial Differential Equation
- **N-uniform:** Constant independent of swarm size N

---

## Quick Lookup by Tag

**Foundational:** `def-walker`, `def-swarm-and-state-space`, `def-valid-state-space`, `def-alive-dead-sets`

**Axioms:** `def-axiom-guaranteed-revival`, `def-axiom-boundary-regularity`, `def-axiom-environmental-richness`, `def-axiom-reward-regularity`, `ax:lipschitz-fields`, `ax:safe-harbor`, `ax:non-deceptive-landscape`

**Metrics:** `def-n-particle-displacement-metric`, `def-metric-quotient`, `def-alg-distance`, `lem-polishness-and-w2`

**Operators - Kinetic:** `def-d-kinetic-operator-adaptive`, BAOAB integrator

**Operators - Cloning:** `def-cloning-operator-formal`, `lem-quantitative-keystone`, `thm-positional-variance-contraction`, `thm-velocity-variance-bounded-expansion`, `thm-complete-variance-drift`

**Lyapunov & Hypocoercivity:** `def-full-synergistic-lyapunov-function`, `prop-lyapunov-necessity`, `def-d-hypocoercive-norm`, `thm-hypocoercive-main`

**Convergence:** `thm-main-convergence`, `thm-foster-lyapunov-adaptive`, `thm-explicit-total-rate`, `cor-explicit-convergence-time`

**QSD:** `def-qsd-adaptive`, `thm-qsd-existence-corrected`, `thm-qsd-stability`, `thm-qsd-spatial-riemannian-volume`

**LSI:** `def-lsi-adaptive`, `thm-adaptive-lsi-main`, `lem-kinetic-lsi-hypocoercive`, `thm-cattiaux-guillin-verification`

**Mean-Field:** `thm-propagation-chaos-qsd`, `thm-correlation-decay`, `thm-exchangeability-qsd`, McKean-Vlasov PDE

**Geometric Gas:** `def-localization-kernel`, `def-unified-z-score`, `def-hybrid-sde`, `def-regularized-hessian-tensor`, `def-d-adaptive-diffusion`

**Emergent Geometry:** `def-emergent-manifold`, `obs-emergent-metric`, `prop-geodesics-fitness`, `thm-coordinate-invariance`, `thm-algorithmic-tunability`

**Regularity:** `thm-c4-regularity`, `cor-hessian-lipschitz`, `prop-lipschitz-diffusion`, `thm-fitness-third-deriv-proven`

**N-Uniform Results:** `lem-quantitative-keystone`, `thm-uniform-ellipticity`, `thm-adaptive-lsi-main`, `thm-qsd-poincare-rigorous`, `thm-main-convergence`

---

## Document Change Log

**v2.1 (2025-10-18):**
- **Major expansion**: Added 18 critical theorems and proofs from Chapter 1 and Chapter 2
- **Wasserstein Contraction (§7)**: Added main contraction theorem, variance decomposition, cloning pressure lemmas
- **Propagation of Chaos (§11)**: Added tightness, uniqueness, thermodynamic limit theorems
- **Geometric Gas (§13)**: Added UEPH, Foster-Lyapunov drift, backbone ergodicity, Stratonovich chain rule
- **Total entries**: 101 (up from 83), carefully curated from 723 in comprehensive glossary
- Clarified purpose: This is a **curated TLDR reference** with detailed explanations for key results
- For exhaustive index of all 723 entries, see **docs/glossary.md**

**v2.0 (2025-10-17):**
- Updated for new document structure (1_euclidean_gas/, 2_geometric_gas/)
- Added 18_emergent_geometry.md content (40 entries)
- Improved TLDR formatting for LLM comprehension
- Added cross-cutting themes section
- Enhanced parameter and constant documentation
- Removed references to deleted documents (old_docs/)
- Consolidated tag system

**v1.0 (Previous):**
- Initial reference covering algorithm/, docs/source/ structure

---

## For Further Reading

**Foundational Documents:**
- [01_fragile_gas_framework.md](source/1_euclidean_gas/01_fragile_gas_framework) - Axioms, state space, metrics (142 entries)
- [02_euclidean_gas.md](source/1_euclidean_gas/02_euclidean_gas) - BAOAB, Sasaki metric, operator composition (31 entries)

**Core Operators:**
- [03_cloning.md](source/1_euclidean_gas/03_cloning) - Cloning operator, Keystone Principle (124 entries)
- [05_kinetic_contraction.md](source/1_euclidean_gas/05_kinetic_contraction) - Langevin dynamics, hypocoercivity (33 entries)

**Convergence Theory:**
- [06_convergence.md](source/1_euclidean_gas/06_convergence) - Foster-Lyapunov, QSD existence (38 entries)
- [09_kl_convergence.md](source/1_euclidean_gas/09_kl_convergence) - LSI, exponential convergence, non-convex landscapes (77 entries)

**Geometric Gas:**
- [11_geometric_gas.md](source/2_geometric_gas/11_geometric_gas) - Localization, adaptive mechanisms (60 entries)
- [18_emergent_geometry.md](source/2_geometric_gas/18_emergent_geometry) - Riemannian geometry, anisotropic diffusion (40 entries)
- [15_geometric_gas_lsi_proof.md](source/2_geometric_gas/15_geometric_gas_lsi_proof) - N-uniform LSI proof (7 entries)

**Mean-Field Theory:**
- [07_mean_field.md](source/1_euclidean_gas/07_mean_field) - McKean-Vlasov PDE (21 entries)
- [08_propagation_chaos.md](source/1_euclidean_gas/08_propagation_chaos) - Thermodynamic limit (34 entries)
- [16_convergence_mean_field.md](source/2_geometric_gas/16_convergence_mean_field) - QSD regularity, entropy production (35 entries)

---

**END OF MATHEMATICAL REFERENCE**
