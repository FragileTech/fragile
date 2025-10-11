# Mathematical Reference for the Fragile Gas Framework

This document provides a comprehensive, searchable reference of all mathematical definitions, theorems, lemmas, propositions, axioms, and key formulas from the Fragile Gas framework. It is designed to enable LLMs to quickly locate and reference mathematical results without reading through the full framework documents.

**Usage:** Search for tags, mathematical objects, or result types to find relevant definitions and theorems. Each entry includes:
- The mathematical statement
- Tags for searchability
- Cross-reference to the source document and section
- Related results and dependencies

**Document Status:** Currently includes results from:
- [01_fragile_gas_framework.md](01_fragile_gas_framework.md) - Core axioms, foundational definitions, viability conditions
- [03_cloning.md](03_cloning.md) - Cloning operator and Keystone Principle
- [03_B__wasserstein_contraction.md](03_B__wasserstein_contraction.md) - Wasserstein-2 contraction proof
- [04_convergence.md](04_convergence.md) - Kinetic operator and complete convergence to QSD
- [05_mean_field.md](05_mean_field.md) - Mean-field limit and McKean-Vlasov PDE
- [06_propagation_chaos.md](06_propagation_chaos.md) - Propagation of chaos and thermodynamic limit
- [07_adaptative_gas.md](07_adaptative_gas.md) - Adaptive Viscous Fluid Model with ρ-localization
- [08_emergent_geometry.md](08_emergent_geometry.md) - Anisotropic diffusion and emergent Riemannian geometry
- [09_symmetries_adaptive_gas.md](09_symmetries_adaptive_gas.md) - Symmetry structure and conservation laws
- [10_kl_convergence/](10_kl_convergence/) - KL-divergence convergence, LSI theory, hypocoercivity, entropy-transport Lyapunov functions
- [11_mean_field_convergence/](11_mean_field_convergence/) - Mean-field entropy production, explicit constants, QSD regularity, parameter analysis
- [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md) - Gauge theory and braid group topology
- [13_fractal_set/](13_fractal_set/) - Episodes, CST, Information Graph, Fractal Set, discrete spacetime, causal set quantum gravity, lattice gauge theory, fermionic exclusion, QCD (condensed - see [00_D_fractal_set.md](00_D_fractal_set.md) for complete reference with 184 mathematical objects)
- [18_hk_convergence.md](18_hk_convergence.md) - Hellinger-Kantorovich metric convergence, mass contraction, LSI-based structural variance contraction, kinetic Hellinger analysis

Complete coverage from foundational axioms through N-particle and mean-field KL-convergence to discrete spacetime formulation.

---

## Table of Contents

- [Foundational Framework and Axioms](#foundational-framework-and-axioms)
- [State Space and Metrics](#state-space-and-metrics)
- [Lyapunov Functions](#lyapunov-functions)
- [Axioms and Foundational Assumptions](#axioms-and-foundational-assumptions)
- [Measurement and Fitness Operators](#measurement-and-fitness-operators)
- [Geometric Properties](#geometric-properties)
- [Contraction and Stability Results](#contraction-and-stability-results)
- [Wasserstein-2 Contraction](#wasserstein-2-contraction)
- [Kinetic Operator and Hypocoercivity](#kinetic-operator-and-hypocoercivity)
- [Drift Inequalities](#drift-inequalities)
- [Complete Convergence to QSD](#complete-convergence-to-qsd)
- [Mean-Field Limit and McKean-Vlasov PDE](#mean-field-limit-and-mckean-vlasov-pde)
- [Propagation of Chaos and Thermodynamic Limit](#propagation-of-chaos-and-thermodynamic-limit)
- [Adaptive Viscous Fluid Model](#adaptive-viscous-fluid-model)
- [Emergent Riemannian Geometry](#emergent-riemannian-geometry)
- [Symmetries of the Adaptive Gas](#symmetries-of-the-adaptive-gas)
- [Gauge Theory Formulation](#gauge-theory-formulation)
- [KL-Divergence Convergence and Logarithmic Sobolev Inequalities](#kl-divergence-convergence-and-logarithmic-sobolev-inequalities)
- [Mean-Field Entropy Production and Explicit Constants](#mean-field-entropy-production-and-explicit-constants)
- [Hellinger-Kantorovich Metric Convergence](#hellinger-kantorovich-metric-convergence)
- [Fractal Set Theory and Discrete Spacetime](#fractal-set-theory-and-discrete-spacetime)
- [Key Inequalities and Bounds](#key-inequalities-and-bounds)

---

## Foundational Framework and Axioms

This section contains the core axiomatic foundations from the framework document that establish the abstract mathematical structure on which all operator-specific results build. This includes state space definitions, viability axioms, environmental assumptions, and fundamental operator properties.

### Walker (Foundation)

**Type:** Definition
**Label:** `def-walker`
**Source:** [01_fragile_gas_framework.md § 1.1](01_fragile_gas_framework.md)
**Tags:** `foundation`, `state-space`, `walker`

**Statement:**
A **walker** $w$ is a tuple:

$$w := (x, s)$$

where $x \in \mathcal{X}$ is the position in state space, $s \in \{0, 1\}$ is the survival status (1=alive, 0=dead).

Extended to $(x, v, s)$ in Euclidean Gas variants where $v \in \mathcal{V}$ is velocity.

**Related Results:** `def-swarm-and-state-space`, `def-alive-dead-sets`

---

### Swarm and State Space

**Type:** Definition
**Label:** `def-swarm-and-state-space`
**Source:** [01_fragile_gas_framework.md § 1.2](01_fragile_gas_framework.md)
**Tags:** `foundation`, `state-space`, `swarm`

**Statement:**
Swarm: $\mathcal{S} := (w_1, w_2, \ldots, w_N)$

Swarm State Space: $\Sigma_N := (\mathcal{X} \times \{0,1\})^N$

**Related Results:** Foundation for all swarm dynamics

---

### Alive and Dead Sets

**Type:** Definition
**Label:** `def-alive-dead-sets`
**Source:** [01_fragile_gas_framework.md § 1.2](01_fragile_gas_framework.md)
**Tags:** `foundation`, `partition`, `alive-set`

**Statement:**
For swarm $\mathcal{S} = ((x_1,s_1), \ldots, (x_N,s_N))$:

$$\mathcal{A}(\mathcal{S}) := \{i \in \{1,\ldots,N\} : s_i = 1\}$$ (alive set)
$$\mathcal{D}(\mathcal{S}) := \{i \in \{1,\ldots,N\} : s_i = 0\}$$ (dead set)

**Related Results:** Used in all operator definitions

---

### Valid State Space

**Type:** Definition
**Label:** `def-valid-state-space`
**Source:** [01_fragile_gas_framework.md § 1.4](01_fragile_gas_framework.md)
**Tags:** `foundation`, `topology`, `measure-theory`, `polish-space`

**Statement:**
A **valid state space** is a triple $(\mathcal{X}, d_\mathcal{X}, \mu_\mathcal{X})$ with:

1. **Topological**: $(\mathcal{X}, d_\mathcal{X})$ is Polish (complete, separable metric space)
2. **Measure**: $\mu_\mathcal{X}$ is a reference measure (e.g., Lebesgue, Riemannian volume)
3. **Noise Support**: Supports valid noise measures satisfying axioms
4. **Boundary Regularity**: $\partial\mathcal{X}_{\text{valid}}$ is C¹ $(d-1)$-dimensional submanifold

Examples: Bounded Euclidean domains, compact Riemannian manifolds, finite/countable graphs.

**Related Results:** `def-axiom-boundary-smoothness`

---

### N-Particle Displacement Metric

**Type:** Definition
**Label:** `def-n-particle-displacement-metric`
**Source:** [01_fragile_gas_framework.md § 1.6](01_fragile_gas_framework.md)
**Tags:** `metric`, `displacement`, `wasserstein`

**Statement:**
N-Particle Displacement Pseudometric:

$$d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) := \sqrt{\frac{1}{N}\sum_i d_\mathcal{Y}(\varphi(x_{1,i}), \varphi(x_{2,i}))^2 + \frac{\lambda_{\text{status}}}{N}\sum_i (s_{1,i} - s_{2,i})^2}$$

Squared form: $d^2_{\text{Disp},\mathcal{Y}} = \frac{1}{N}\Delta^2_{\text{pos}} + \frac{\lambda_{\text{status}}}{N}n^c$

Parameter $\lambda_{\text{status}} > 0$ weights status changes.

**Related Results:** Foundation for Wasserstein metrics

---

### Kolmogorov Quotient

**Type:** Definition
**Label:** `def-metric-quotient`
**Source:** [01_fragile_gas_framework.md § 1.6.1](01_fragile_gas_framework.md)
**Tags:** `metric`, `topology`, `quotient-space`

**Statement:**
Define equivalence: $\mathcal{S}_1 \sim \mathcal{S}_2$ iff $d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2) = 0$

Quotient space: $\bar{\Sigma}_N := \Sigma_N / \sim$

Quotient metric: $\bar{d}_{\text{Disp},\mathcal{Y}}([\mathcal{S}_1], [\mathcal{S}_2]) := d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)$

**Related Results:** `lem-polishness-and-w2`

---

### Polishness and W₂ Well-Posedness

**Type:** Lemma
**Label:** `lem-polishness-and-w2`
**Source:** [01_fragile_gas_framework.md § 1.7.1](01_fragile_gas_framework.md)
**Tags:** `topology`, `wasserstein`, `polish-space`

**Statement:**
If $(\mathcal{Y}, d_\mathcal{Y})$ is Polish and $N < \infty$, then $(\bar{\Sigma}_N, \bar{d}_{\text{Disp},\mathcal{Y}})$ is Polish.

Consequently, $W_2$ on $\mathcal{P}(\bar{\Sigma}_N)$ is well-posed and finite on measures with finite second moment.

Proof: Metric quotient of Polish space by closed equivalence is Polish (Kechris, Classical Descriptive Set Theory, Thm 5.5).

**Related Results:** Foundations for Wasserstein convergence theory

---

### Axiom of Guaranteed Revival

**Type:** Axiom
**Label:** `def-axiom-guaranteed-revival`
**Source:** [01_fragile_gas_framework.md § 2.1.1](01_fragile_gas_framework.md)
**Tags:** `viability`, `cloning`, `revival`

**Statement:**
Dead walker's cloning score must exceed $p_{\max}$.

Parameter: $\kappa_{\text{revival}} := \eta^{\alpha+\beta} / (\epsilon_{\text{clone}} \cdot p_{\max})$

Condition: $\kappa_{\text{revival}} > 1$

Failure: If $\kappa_{\text{revival}} \leq 1$, revival becomes probabilistic → gradual attrition → extinction.

**Related Results:** `thm-revival-guarantee`

---

### Almost-Sure Revival Theorem

**Type:** Theorem
**Label:** `thm-revival-guarantee`
**Source:** [01_fragile_gas_framework.md § 2.1.1](01_fragile_gas_framework.md)
**Tags:** `viability`, `probability`, `revival`

**Statement:**
Assumptions: $\epsilon_{\text{clone}} \cdot p_{\max} < \eta^{\alpha+\beta}$, $|\mathcal{A}(\mathcal{S})| \geq 1$, $i \in \mathcal{D}(\mathcal{S})$

Conclusion: $\mathbb{P}[\text{walker } i \text{ revived in cloning}] = 1$

Proof: Any alive companion $j$ has $V_{\text{fit},j} \geq \eta^{\alpha+\beta}$. Dead walker's score $S_i \geq V_{\text{fit},j} / \epsilon_{\text{clone}} \geq \eta^{\alpha+\beta} / \epsilon_{\text{clone}} > p_{\max}$. Since threshold $T_{\text{clone}} \in [0, p_{\max}]$, have $S_i > T_{\text{clone}}$ surely.

**Related Results:** `def-axiom-guaranteed-revival`

---

### Axiom of Boundary Regularity

**Type:** Axiom
**Label:** `def-axiom-boundary-regularity`
**Source:** [01_fragile_gas_framework.md § 2.1.2](01_fragile_gas_framework.md)
**Tags:** `viability`, `geometry`, `continuity`, `holder`

**Statement:**
Marginal death probability after perturbation+status is Hölder continuous in swarm state.

Parameters: $L_{\text{death}} > 0$ (Boundary Instability Factor), $\alpha_B \in (0, 1]$ (Smoothing Exponent)

Condition:

$$|P(s_{\text{out},i}=0 | \mathcal{S}_1) - P(s_{\text{out},i}=0 | \mathcal{S}_2)| \leq L_{\text{death}} \cdot d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1, \mathcal{S}_2)^{\alpha_B}$$

Canonical bounds for uniform ball/Gaussian kernels: $L_{\text{death}} \leq C_d \cdot \text{Per}(\mathcal{X}_{\text{invalid}}) / \sigma$, $\alpha_B = 1$

**Related Results:** Critical for death operator continuity

---

### Axiom of Environmental Richness

**Type:** Axiom
**Label:** `def-axiom-environmental-richness`
**Source:** [01_fragile_gas_framework.md § 2.2.1](01_fragile_gas_framework.md)
**Tags:** `environment`, `learning`, `variance`

**Statement:**
Reward function not pathologically flat at minimum length scale.

Parameters: $r_{\min} > 0$ (Minimum Richness Scale), $\kappa_{\text{richness}}$ (Environmental Richness Floor)

Condition:

$$\kappa_{\text{richness}} \leq \inf_{y\in\varphi(\mathcal{X}_{\text{valid}}), r\geq r_{\min}} \text{Var}_{y'\in B(y,r)\cap\varphi(\mathcal{X}_{\text{valid}})}[R_\mathcal{Y}(y')]$$

Required: $\kappa_{\text{richness}} > 0$

Failure: $\kappa_{\text{richness}} \approx 0$ → flat landscape → learning stalls.

**Related Results:** `thm-forced-activity`

---

### Axiom of Reward Regularity

**Type:** Axiom
**Label:** `def-axiom-reward-regularity`
**Source:** [01_fragile_gas_framework.md § 2.2.2](01_fragile_gas_framework.md)
**Tags:** `environment`, `continuity`, `holder`

**Statement:**
Reward function is Hölder continuous in algorithmic space.

Parameters: $L_{R,\mathcal{Y}} > 0$ (Reward Volatility), $\alpha_R \in (0, 1]$ (Smoothing Exponent)

Condition: $|R_\mathcal{Y}(y_1) - R_\mathcal{Y}(y_2)| \leq L_{R,\mathcal{Y}} \cdot d_\mathcal{Y}(y_1, y_2)^{\alpha_R}$

Failure: Large $L_{R,\mathcal{Y}}$ → bumpy landscape → noisy exploitation → unstable cloning.

**Related Results:** Feeds into fitness potential Lipschitz constants

---

### Axiom of Sufficient Amplification

**Type:** Axiom
**Label:** `def-axiom-sufficient-amplification`
**Source:** [01_fragile_gas_framework.md § 2.3.1](01_fragile_gas_framework.md)
**Tags:** `dynamics`, `fitness`, `amplification`

**Statement:**
Parameter: $\kappa_{\text{amplification}} := \alpha + \beta$

Condition: $\kappa_{\text{amplification}} > 0$

Failure: If $\kappa_{\text{amplification}} = 0$, then $V_i = 1$ for all alive walkers → no cloning pressure → independent random walkers.

**Related Results:** Core requirement for adaptive dynamics

---

### Axiom of Non-Degenerate Noise

**Type:** Axiom
**Label:** `def-axiom-non-degenerate-noise`
**Source:** [01_fragile_gas_framework.md § 2.3.2](01_fragile_gas_framework.md)
**Tags:** `noise`, `exploration`, `perturbation`

**Statement:**
Parameters: $\sigma > 0$ (perturbation), $\delta > 0$ (cloning)

Condition: $\sigma > 0$ AND $\delta > 0$

Failure: If $\sigma = \delta = 0$ → no new positions → exploration loss → collapse to few points.

**Related Results:** Required for irreducibility

---

### Theorem of Forced Activity

**Type:** Theorem
**Label:** `thm-forced-activity`
**Source:** [01_fragile_gas_framework.md § 2.4.2](01_fragile_gas_framework.md)
**Tags:** `dynamics`, `cloning`, `activity`

**Statement:**
Sufficiently spread swarm in rich environment generates non-zero cloning probability.

Parameter: $p_{\text{clone,min}} > 0$ (Minimum Average Cloning Probability)

Emergent from: Environmental Richness + Non-Degenerate Noise + Sufficient Amplification

Condition: Swarm non-degenerate (diameter $> r_{\min}$ to experience $\kappa_{\text{richness}}$)

Required: $p_{\text{clone,min}} > 0$

Failure: $p_{\text{clone,min}} = 0$ → adaptation stops → stagnation.

**Related Results:** `def-axiom-environmental-richness`, `def-axiom-sufficient-amplification`

---

### Algorithmic Distance

**Type:** Definition
**Label:** `def-alg-distance`
**Source:** [01_fragile_gas_framework.md § 5.3](01_fragile_gas_framework.md)
**Tags:** `distance`, `metric`, `projection`

**Statement:**

$$d_{\text{alg}}(x_1, x_2) := d_\mathcal{Y}(\varphi(x_1), \varphi(x_2))$$

Practical implementation of Wasserstein distance between projected Dirac measures. Ground distance for all subsequent calculations.

**Related Results:** Used throughout measurement pipeline

---

### Companion Selection Measure

**Type:** Definition
**Label:** `def-companion-selection-measure`
**Source:** [01_fragile_gas_framework.md § 7.1](01_fragile_gas_framework.md)
**Tags:** `companion`, `measure`, `sampling`

**Statement:**
For walker $i$ in swarm $\mathcal{S}$ with alive set $\mathcal{A}$, uniform measure $\mathcal{C}_i(\mathcal{S})$ over support $S_i$:

- **Alive, multiple companions** ($i \in \mathcal{A}$, $|\mathcal{A}| \geq 2$): $S_i := \mathcal{A} \setminus \{i\}$
- **Dead, swarm alive** ($i \notin \mathcal{A}$, $|\mathcal{A}| \geq 1$): $S_i := \mathcal{A}$
- **Single survivor** ($|\mathcal{A}| = 1$, $\mathcal{A} = \{i\}$): $S_i := \{i\}$ (self-companion)
- **Empty swarm** ($|\mathcal{A}| = 0$): $S_i := \emptyset$

Sampling: $\mathcal{C}_i(\mathcal{S})(\{j\}) = 1/|S_i|$ if $j \in S_i$ and $|S_i| > 0$

Expectation: $\mathbb{E}_{j\sim\mathcal{C}_i}[f(j)] = \frac{1}{|S_i|} \sum_{j\in S_i} f(j)$

Sampling **with replacement**, independent across walkers.

**Related Results:** Core for distance-to-companion measurement

---

### Axiom of Well-Behaved Rescale Function

**Type:** Axiom
**Label:** `def-axiom-rescale-function`
**Source:** [01_fragile_gas_framework.md § 8.1](01_fragile_gas_framework.md)
**Tags:** `rescale`, `fitness`, `smoothness`

**Statement:**
Function $g_A: \mathbb{R} \to \mathbb{R}_{>0}$ must satisfy:

1. **C¹ Smoothness**: Continuously differentiable on $\mathbb{R}$
2. **Monotonicity**: $g'_A(z) \geq 0$ for all $z \in \mathbb{R}$
3. **Uniform Boundedness**: Range $\subseteq (0, g_{A,\max}]$ for finite $g_{A,\max} > 0$
4. **Global Lipschitz**: $\sup_z |g'_A(z)| = L_{g_A} < \infty$

Purpose: Ensures fitness potential smooth, prevents infinite values, bounds error amplification.

**Related Results:** Core for fitness potential construction

---

### Swarm Update Procedure

**Type:** Definition
**Label:** `def-swarm-update-procedure`
**Source:** [01_fragile_gas_framework.md § 17.1](01_fragile_gas_framework.md)
**Tags:** `operator`, `markov`, `transition`

**Statement:**
Signature: $\Psi: \Sigma_N \to \mathcal{P}(\Sigma_N)$

One-step transition measure: $\mathcal{S}_t \to$ distribution over $\mathcal{S}_{t+1}$

Stages:
1. **Cemetery Absorption**: If $|\mathcal{A}(\mathcal{S}_t)| = 0$, return $\delta_{\mathcal{S}_t}$ (terminate)
2. **Measurement & Potential**: Sample raw measurements $(r_\mathcal{A}, d_\mathcal{A})$, compute fitness $V_{\text{fit}}$
3. **Cloning Transition**: Sample companions, determine clone/persist actions, sample intermediate positions
4. **Perturbation & Status**: Apply perturbation kernel, update survival status

**Related Results:** Composition of all operators

---

## State Space and Metrics

### Single-Walker and Swarm State Spaces

**Type:** Definition
**Label:** `def-single-swarm-space`
**Source:** [03_cloning.md § 2.1](03_cloning.md#21-the-single-swarm-state-space)
**Tags:** `state-space`, `swarm`, `walker`, `configuration`, `survival-status`

**Statement:**

1. A **walker** is a tuple $(x, s)$ where:
   - $x \in \mathcal{X}$ is the position
   - $s \in \{0, 1\}$ is the survival status

2. For the Euclidean Gas, a walker includes velocity: $(x, v, s) \in \mathbb{R}^d \times \mathbb{R}^d \times \{0, 1\}$

3. A **swarm configuration** with $N$ walkers is:

$$
S := \left( (x_1, v_1, s_1), (x_2, v_2, s_2), \dots, (x_N, v_N, s_N) \right)
$$

4. The **single-swarm state space** is:

$$
\Sigma_N := \left( \mathbb{R}^d \times \mathbb{R}^d \times \{0, 1\} \right)^N
$$

**Related Results:**
- Used in: Coupled state space ({prf:ref}`def-coupled-state-space`)
- Defines basis for: State difference vectors ({prf:ref}`def-state-difference-vectors`)

---

### The Coupled State Space

**Type:** Definition
**Label:** `def-coupled-state-space`
**Source:** [03_cloning.md § 2.2](03_cloning.md#22-the-coupled-state-space)
**Tags:** `coupling`, `convergence-analysis`, `product-space`, `paired-swarms`

**Statement:**

The **coupled state space** is the product space $\Sigma_N \times \Sigma_N$, containing pairs of swarm configurations $(S_1, S_2)$ used for convergence analysis via coupling arguments.

**Key Formula:**

$$
S_k = \left( (x_{k,1}, v_{k,1}, s_{k,1}), \dots, (x_{k,N}, v_{k,N}, s_{k,N}) \right) \in \Sigma_N
$$

where $k \in \{1, 2\}$ indexes the two coupled swarms.

**Related Results:**
- Requires: Single-swarm space ({prf:ref}`def-single-swarm-space`)
- Used in: All Lyapunov functions and drift analysis
- Enables: Geometric ergodicity proofs via coupling

---

### State Difference Vectors

**Type:** Definition
**Label:** `def-state-difference-vectors`
**Source:** [03_cloning.md § 2.3](03_cloning.md#23-state-difference-vectors)
**Tags:** `difference-vector`, `coupling-error`, `position-error`, `velocity-error`

**Statement:**

For any $(S_1, S_2) \in \Sigma_N \times \Sigma_N$ and walker index $i \in \{1, \ldots, N\}$:

1. **Position difference vector:**

$$
\Delta x_i := x_{1,i} - x_{2,i} \in \mathbb{R}^d
$$

2. **Velocity difference vector:**

$$
\Delta v_i := v_{1,i} - v_{2,i} \in \mathbb{R}^d
$$

**Related Results:**
- Used in: Structural error component ({prf:ref}`def-structural-error-component`)
- Used in: Coercivity lemma ({prf:ref}`lem-V-coercive`)
- Quantifies: Inter-swarm walker-wise error

---

### Algorithmic Distance for Companion Selection

**Type:** Definition
**Label:** `def-algorithmic-distance-metric`
**Source:** [03_cloning.md § 5.0.1](03_cloning.md#501-algorithmic-distance)
**Tags:** `distance`, `metric`, `phase-space`, `companion-selection`, `hypocoercive`

**Statement:**

For walkers $i, j$ with kinematic states $(x_i, v_i)$ and $(x_j, v_j)$, the **algorithmic distance** is:

$$
d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2
$$

where $\lambda_{\text{alg}} \geq 0$ controls the importance of velocity differences.

**Regimes:**
- $\lambda_{\text{alg}} = 0$: Position-only model (purely spatial)
- $\lambda_{\text{alg}} > 0$: Fluid dynamics model (phase-space aware)
- $\lambda_{\text{alg}} = 1$: Balanced phase-space model

**Related Results:**
- Used in: Companion selection operators ({prf:ref}`def-spatial-pairing-operator-diversity`, {prf:ref}`def-cloning-companion-operator`)
- Used in: Geometric separation lemma ({prf:ref}`lem-geometric-separation-of-partition`)
- Analogous to: Hypocoercive metric in Lyapunov function

---

## Lyapunov Functions

### Barycentres and Centered Vectors

**Type:** Definition
**Label:** `def-barycentres-and-centered-vectors`
**Source:** [03_cloning.md § 3.1.1](03_cloning.md#311-barycentres-and-centered-vectors)
**Tags:** `center-of-mass`, `barycenter`, `centered-coordinates`, `alive-only`

**Statement:**

For swarm $k$ with alive set $\mathcal{A}(S_k)$ of size $k_{\text{alive}}$:

1. **Positional center of mass:**

$$
\mu_{x,k} := \frac{1}{k_{\text{alive}}}\sum_{i \in \mathcal{A}(S_k)} x_{k,i}
$$

2. **Velocity center of mass:**

$$
\mu_{v,k} := \frac{1}{k_{\text{alive}}}\sum_{i \in \mathcal{A}(S_k)} v_{k,i}
$$

3. **Centered position vector** for alive walker $i \in \mathcal{A}(S_k)$:

$$
\delta_{x,k,i} := x_{k,i} - \mu_{x,k}
$$

4. **Centered velocity vector** for alive walker $i \in \mathcal{A}(S_k)$:

$$
\delta_{v,k,i} := v_{k,i} - \mu_{v,k}
$$

**Convention:** Dead walkers do not contribute to any statistics.

**Related Results:**
- Used in: All variance and internal error measures
- Used in: Structural error component ({prf:ref}`def-structural-error-component`)
- Foundation for: Variance decomposition

---

### The Location Error Component ($V_{\text{loc}}$)

**Type:** Definition
**Label:** `def-location-error-component`
**Source:** [03_cloning.md § 3.2.1.1](03_cloning.md#3211-the-location-error-component)
**Tags:** `wasserstein`, `inter-swarm-error`, `location-error`, `barycenter-distance`

**Statement:**

$$
V_{\text{loc}} := \|\Delta\mu_x\|^2 + \lambda_v\|\Delta\mu_v\|^2 + b\langle\Delta\mu_x, \Delta\mu_v\rangle
$$

where:
- $\Delta\mu_x = \mu_{x,1} - \mu_{x,2}$ is the barycenter position difference
- $\Delta\mu_v = \mu_{v,1} - \mu_{v,2}$ is the barycenter velocity difference
- $\lambda_v > 0$ weights velocity error
- $b \in \mathbb{R}$ is the hypocoercive coupling parameter

**Related Results:**
- Part of: Wasserstein decomposition ({prf:ref}`lem-wasserstein-decomposition`)
- Coercivity: Proven in {prf:ref}`lem-V-coercive`
- Measures: Inter-swarm center-of-mass separation

---

### The Structural Error Component ($V_{\text{struct}}$)

**Type:** Definition
**Label:** `def-structural-error-component`
**Source:** [03_cloning.md § 3.2.1.2](03_cloning.md#3212-the-structural-error-component)
**Tags:** `wasserstein`, `structural-error`, `optimal-transport`, `internal-error`

**Statement:**

Let $\tilde{\mu}_k$ be the **centered empirical measure** of swarm $k$ (alive walkers only):

$$
\tilde{\mu}_k := \frac{1}{k_{\text{alive}}} \sum_{i \in \mathcal{A}(S_k)} \delta_{(\delta_{x,k,i}, \delta_{v,k,i})}
$$

where $\delta_z$ denotes the Dirac measure at point $z$.

The **structural error** is the hypocoercive Wasserstein distance between centered measures:

$$
V_{\text{struct}} := W_h^2(\tilde{\mu}_1, \tilde{\mu}_2) = \inf_{\gamma \in \Gamma(\tilde{\mu}_1, \tilde{\mu}_2)} \int c(\delta_{z,1}, \delta_{z,2}) \, d\gamma
$$

where $\Gamma(\tilde{\mu}_1, \tilde{\mu}_2)$ is the set of couplings and $c(\cdot, \cdot)$ is the hypocoercive cost.

**Related Results:**
- Part of: Wasserstein decomposition ({prf:ref}`lem-wasserstein-decomposition`)
- Bounded by: Internal variance ({prf:ref}`lem-sx-implies-variance`)
- Contracts under: Cloning operator ({prf:ref}`cor-structural-error-contraction`)

---

### Decomposition of the Hypocoercive Wasserstein Distance

**Type:** Lemma
**Label:** `lem-wasserstein-decomposition`
**Source:** [03_cloning.md § 3.2.2](03_cloning.md#322-decomposition)
**Tags:** `wasserstein`, `decomposition`, `location-structure-split`, `exact-decomposition`

**Statement:**

$$
W_h^2(\mu_1, \mu_2) = V_{\text{loc}} + V_{\text{struct}}
$$

This is an **exact decomposition** of the total inter-swarm hypocoercive Wasserstein distance into:
- $V_{\text{loc}}$: Error in swarm centers of mass
- $V_{\text{struct}}$: Error in internal swarm structure (shape)

**Proof Technique:** Optimal transport with barycenter translation.

**Related Results:**
- Components defined in: {prf:ref}`def-location-error-component`, {prf:ref}`def-structural-error-component`
- Used in: All drift analysis of inter-swarm error
- Key property: Enables separate analysis of location vs. shape convergence

---

### Structural Positional Error and Internal Variance

**Type:** Lemma
**Label:** `lem-sx-implies-variance`
**Source:** [03_cloning.md § 3.2.4](03_cloning.md#324-structural-error-bound)
**Tags:** `variance-bound`, `structural-error`, `internal-variance`, `spread`

**Statement:**

Let $V_{\text{x,struct}}$ be the positional component of structural error and $\text{Var}_k(x)$ be the physical internal positional variance (k-normalized) of alive walkers in swarm $k$. Then:

$$
V_{\text{x,struct}} \le 2(\text{Var}_1(x) + \text{Var}_2(x))
$$

**Consequence:** If the structural error is large:

$$
V_{\text{x,struct}} > R^2_{\text{spread}} \quad \Longrightarrow \quad \exists k: \text{Var}_k(x) > \frac{R^2_{\text{spread}}}{4}
$$

At least one swarm has large internal positional variance.

**Related Results:**
- Links: Structural error to single-swarm variance
- Used in: Geometric analysis (Chapter 6)
- Foundation for: High-error set definitions

---

### The Full Synergistic Hypocoercive Lyapunov Function

**Type:** Definition
**Label:** `def-full-synergistic-lyapunov-function`
**Source:** [03_cloning.md § 3.3.1](03_cloning.md#331-complete-lyapunov-function)
**Tags:** `lyapunov`, `synergistic`, `variance`, `boundary`, `N-normalized`

**Statement:**

$$
V_{\mathrm{total}}(S_1, S_2) := W_h^2(\mu_1, \mu_2) + c_V V_{Var}(S_1, S_2) + c_B W_b(S_1, S_2)
$$

where:

**Internal Variance Component:**

$$
V_{Var}(S_1, S_2) = V_{Var,x}(S_1, S_2) + \lambda_v V_{Var,v}(S_1, S_2)
$$

with **N-normalized** components:

$$
\begin{align*}
V_{Var,x}(S_1, S_2) &:= \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{x,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{x,2,i}\|^2 \\
V_{Var,v}(S_1, S_2) &:= \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \|\delta_{v,1,i}\|^2 + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \|\delta_{v,2,i}\|^2
\end{align*}
$$

**Boundary Potential:**

$$
W_b(S_1, S_2) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \varphi_{\text{barrier}}(x_{1,i}) + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \varphi_{\text{barrier}}(x_{2,i})
$$

where $\varphi_{\text{barrier}}(x) \to \infty$ as $x \to \partial \mathcal{X}_{\text{valid}}$.

**Coupling Constants:**
- $c_V > 0$: Weight for variance component
- $c_B > 0$: Weight for boundary component

**Related Results:**
- Components contract separately under different operators
- Complete drift analysis: {prf:ref}`thm-complete-cloning-drift`
- Synergy with kinetic operator: {prf:ref}`thm-synergistic-foster-lyapunov-preview`

---

### Variance Notation Conversion Formulas

**Type:** Definition
**Label:** `def-variance-conversions`
**Source:** [03_cloning.md § 3.3.1.1](03_cloning.md#3311-variance-notation-conversions)
**Tags:** `variance`, `normalization`, `k-normalized`, `N-normalized`, `conversion`

**Statement:**

For swarm $k$ with $k_{\text{alive}}$ alive walkers out of $N$ total slots:

1. **Un-normalized sum:**

$$
S_k := \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2
$$

2. **Physical internal variance (k-normalized):**

$$
\text{Var}_k(x) := \frac{S_k}{k_{\text{alive}}}
$$

3. **Lyapunov variance component (N-normalized):**

$$
V_{\text{Var},x}(S_k) := \frac{S_k}{N}
$$

**Conversion Formulas:**

$$
\begin{aligned}
S_k &= k_{\text{alive}} \cdot \text{Var}_k(x) = N \cdot V_{\text{Var},x}(S_k) \\
V_{\text{Var},x}(S_k) &= \frac{k_{\text{alive}}}{N} \cdot \text{Var}_k(x) \\
\text{Var}_k(x) &= \frac{N}{k_{\text{alive}}} \cdot V_{\text{Var},x}(S_k)
\end{aligned}
$$

**Critical Property:** N-normalization ensures uniform bounds independent of $N$ (scalability).

**Related Results:**
- Essential for: N-uniform contraction rates
- Used in: All variance drift inequalities
- Connects: Physical quantities to Lyapunov components

---

### Coercivity of the Hypocoercive Lyapunov Components

**Type:** Lemma
**Label:** `lem-V-coercive`
**Source:** [03_cloning.md § 3.4.1](03_cloning.md#341-coercivity)
**Tags:** `coercivity`, `lower-bound`, `positive-definite`, `elliptic`

**Statement:**

If the hypocoercive parameters satisfy:

$$
b^2 < 4\lambda_v
$$

then there exist constants $\lambda_1, \lambda_2 > 0$ such that:

**Location Coercivity:**

$$
V_{\text{loc}} \ge \lambda_1 (\|\Delta\mu_x\|^2 + \|\Delta\mu_v\|^2)
$$

**Structural Coercivity:**

$$
V_{\text{struct}} \ge \lambda_2 \frac{1}{N}\sum_{i=1}^N (\|\Delta\delta_{x,i}\|^2 + \|\Delta\delta_{v,i}\|^2)
$$

**Explicit Coercivity Constant:**

$$
\lambda_{\min} = \frac{1 + \lambda_v - \sqrt{(1 - \lambda_v)^2 + b^2}}{2} > 0
$$

**Related Results:**
- Ensures: Wasserstein distance is a valid norm
- Used in: Convergence rate analysis
- Requires: Hypocoercive parameter constraint

---

## Axioms and Foundational Assumptions

### Axiom EG-0: Regularity of the Domain

**Type:** Axiom
**Label:** `ax:domain-regularity`
**Source:** [03_cloning.md § 2.4](03_cloning.md#24-domain-regularity)
**Tags:** `domain`, `regularity`, `boundary`, `smooth-manifold`

**Statement:**

The valid domain $\mathcal{X}_{\text{valid}}$ is an **open, bounded, connected** subset of $\mathbb{R}^d$ with boundary $\partial \mathcal{X}_{\text{valid}}$ being a **$C^{\infty}$-smooth compact manifold** without boundary.

**Consequences:**
- Enables construction of smooth barrier functions
- Ensures well-defined signed distance to boundary
- Guarantees existence of smooth cutoff functions

**Related Results:**
- Enables: Barrier function existence ({prf:ref}`prop-barrier-existence`)
- Required for: Boundary potential analysis (Chapter 11)

---

### Existence of a Global Smooth Barrier Function

**Type:** Proposition
**Label:** `prop-barrier-existence`
**Source:** [03_cloning.md § 2.4.2](03_cloning.md#242-barrier-function-construction)
**Tags:** `barrier-function`, `boundary-divergence`, `smooth`, `confining-potential`

**Statement:**

Under Axiom EG-0, there exists a function $\varphi: \mathcal{X}_{\text{valid}} \to \mathbb{R}$ with:

1. **Smoothness:** $\varphi(x) \in C^{\infty}(\mathcal{X}_{\text{valid}})$
2. **Positivity:** $\varphi(x) > 0$ for all $x \in \mathcal{X}_{\text{valid}}$
3. **Boundary Divergence:** $\lim_{x \to \partial \mathcal{X}_{\text{valid}}} \varphi(x) = \infty$

**Explicit Construction:**

$$
\varphi(x) := \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right)\left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right)
$$

where:
- $\rho(x)$ is the signed distance to the boundary
- $\psi: \mathbb{R} \to [0,1]$ is a smooth cutoff function
- $\delta > 0$ is a small smoothing parameter

**Related Results:**
- Required by: Axiom EG-0 ({prf:ref}`ax:domain-regularity`)
- Used in: Boundary potential ({prf:ref}`def-boundary-potential-component`)
- Enables: Boundary contraction analysis ({prf:ref}`thm-boundary-potential-contraction`)

---

### Axiom EG-1: Lipschitz Regularity of Environmental Fields

**Type:** Axiom
**Label:** `ax:lipschitz-fields`
**Source:** [03_cloning.md § 4.1](03_cloning.md#41-lipschitz-regularity)
**Tags:** `lipschitz`, `force-field`, `flow-field`, `regularity`

**Statement:**

There exist constants $L_F, L_u < \infty$ such that for all $x_1, x_2 \in \mathcal X_{\mathrm{valid}}$:

1. **Force Field Lipschitz Continuity:**

$$
\|F(x_1) - F(x_2)\| \leq L_F \|x_1 - x_2\|
$$

2. **Steady Flow Field Lipschitz Continuity:**

$$
\|u(x_1) - u(x_2)\| \leq L_u \|x_1 - x_2\|
$$

**Related Results:**
- Ensures: Bounded error propagation through environmental fields
- Required for: Kinetic operator analysis
- Enables: Wasserstein contraction under drift

---

### Axiom EG-2: Existence of a Safe Harbor

**Type:** Axiom
**Label:** `ax:safe-harbor`
**Source:** [03_cloning.md § 4.2](03_cloning.md#42-safe-harbor)
**Tags:** `safe-harbor`, `high-reward`, `boundary-separation`, `target-region`

**Statement:**

There exists a compact set $C_{\mathrm{safe}} \subset \mathcal X_{\mathrm{valid}}$ and reward threshold $R_{\mathrm{safe}}$ such that:

1. **Boundary Separation:** For all $x \in C_{\mathrm{safe}}$:

$$
d(x, \partial X_{\mathrm{valid}}) \geq \delta_{\mathrm{safe}} > 0
$$

2. **High Reward Region:**

$$
\max_{y \in C_{\mathrm{safe}}} R_{\mathrm{pos}}(y) \geq R_{\mathrm{safe}}
$$

3. **Uniqueness:** For all $x \notin C_{\mathrm{safe}}$:

$$
R_{\mathrm{pos}}(x) < R_{\mathrm{safe}}
$$

**Physical Interpretation:** The Safe Harbor is a well-defined, high-reward region away from the boundary that attracts the swarm.

**Related Results:**
- Enables: Boundary contraction ({prf:ref}`thm-boundary-potential-contraction`)
- Ensures: Walkers near boundary have systematically lower fitness
- Foundation for: Fitness gradient lemma ({prf:ref}`lem-fitness-gradient-boundary`)

---

### Axiom EG-3: Non-Deceptive Landscape

**Type:** Axiom
**Label:** `ax:non-deceptive-landscape`
**Source:** [03_cloning.md § 4.3](03_cloning.md#43-non-deceptive-landscape)
**Tags:** `non-deceptive`, `reward-landscape`, `detectability`, `gradient-bound`

**Statement:**

There exist constants $L_{\text{grad}} > 0$ and $\kappa_{\text{raw},r} > 0$ such that:

$$
\|x - y\| \geq L_{\text{grad}} \quad \Longrightarrow \quad |R_{\mathrm{pos}}(y) - R_{\mathrm{pos}}(x)| \geq \kappa_{\text{raw},r}
$$

**Interpretation:** Spatially separated positions have detectably different rewards. The landscape is not "deceptively flat."

**Related Results:**
- Ensures: Reward signal can detect high-variance configurations
- Required for: Corrective nature of fitness (Chapter 7)
- Prevents: Complete reward homogeneity across distant regions

---

### Axiom EG-4: Velocity Regularization via Reward

**Type:** Axiom
**Label:** `ax:velocity-regularization`
**Source:** [03_cloning.md § 4.4](03_cloning.md#44-velocity-regularization)
**Tags:** `velocity-penalty`, `kinetic-energy`, `regularization`, `reward-structure`

**Statement:**

The total reward explicitly penalizes high kinetic energy:

$$
R_{\text{total}}(x, v) := R_{\text{pos}}(x) - c_{v\_reg} \|v\|^2
$$

where $c_{v\_reg} > 0$ is a strictly positive constant.

**Interpretation:** Walkers with high velocities are systematically penalized, preventing unbounded kinetic energy accumulation.

**Related Results:**
- Ensures: Bounded velocity variance in equilibrium
- Complements: Langevin friction in kinetic operator
- Required for: Velocity variance contraction analysis

---

### Axiom EG-5: Active Diversity Signal

**Type:** Axiom
**Label:** `ax:active-diversity`
**Source:** [03_cloning.md § 4.5](03_cloning.md#45-active-diversity)
**Tags:** `diversity`, `exploration`, `non-zero-beta`

**Statement:**

The diversity channel is active:

$$
\beta > 0
$$

**Interpretation:** The fitness function includes a non-trivial diversity component, preventing pure exploitation.

**Related Results:**
- Required for: Stability condition ({prf:ref}`thm-stability-condition-intelligent-adaptation`)
- Ensures: Unfit set overlaps with high-error set
- Foundation for: Keystone Principle

---

## Measurement and Fitness Operators

### Spatially-Aware Pairing Operator (Idealized Model)

**Type:** Definition
**Label:** `def-spatial-pairing-operator-diversity`
**Source:** [03_cloning.md § 5.1.1](03_cloning.md#511-idealized-pairing)
**Tags:** `pairing`, `diversity`, `softmax`, `maximum-weight-matching`

**Statement:**

For alive set $\mathcal{A}_t$ of size $k$, edge weights between walkers $i, j$ are:

$$
w_{ij} := \exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon_d^2}\right)
$$

where $\epsilon_d > 0$ is the diversity interaction range.

**Matching Quality:**

$$
W(M) := \prod_{(i,j) \in M} w_{ij}
$$

for matching $M \in \mathcal{M}_k$ (set of all perfect matchings).

**Selection Probability (Gibbs Distribution):**

$$
P(M) = \frac{W(M)}{\sum_{M' \in \mathcal{M}_k} W(M')}
$$

**Computational Complexity:** $O(k^3)$ via Hungarian algorithm or $O(k^4)$ for exact Gibbs sampling.

**Related Results:**
- Approximated by: Greedy pairing ({prf:ref}`def-greedy-pairing-algorithm`)
- Used for: Diversity measurement
- Properties preserved by: Greedy approximation ({prf:ref}`lem-greedy-preserves-signal`)

---

### Sequential Stochastic Greedy Pairing Operator

**Type:** Definition
**Label:** `def-greedy-pairing-algorithm`
**Source:** [03_cloning.md § 5.1.2](03_cloning.md#512-greedy-pairing)
**Tags:** `greedy`, `pairing`, `stochastic`, `O(k^2)`, `practical`

**Statement:**

A practical $O(k^2)$ algorithm that builds a matching iteratively.

**Algorithm:**
1. Initialize unpaired set $U := \mathcal{A}_t$
2. While $|U| \geq 2$:
   - Select walker $i \in U$ (uniformly at random or sequentially)
   - For each remaining walker $j \in U \setminus \{i\}$, compute pairing probability:

$$
P(\text{choose } j \mid i) = \frac{w_{ij}}{\sum_{l \in U \setminus \{i\}} w_{il}} = \frac{\exp(-d_{\text{alg}}(i, j)^2 / 2\epsilon_d^2)}{\sum_{l \in U \setminus \{i\}} \exp(-d_{\text{alg}}(i, l)^2 / 2\epsilon_d^2)}
$$

   - Sample $j$ from this distribution
   - Add $(i,j)$ to matching, remove both from $U$

**Related Results:**
- Approximates: Idealized pairing ({prf:ref}`def-spatial-pairing-operator-diversity`)
- Signal preservation: {prf:ref}`lem-greedy-preserves-signal`
- Used in: Diversity measurement pipeline

---

### Greedy Pairing Guarantees Signal Separation

**Type:** Lemma
**Label:** `lem-greedy-preserves-signal`
**Source:** [03_cloning.md § 5.1.3](03_cloning.md#513-signal-preservation)
**Tags:** `greedy-guarantee`, `signal-preservation`, `high-low-separation`

**Statement:**

For a high-variance swarm partitioned into:
- High-error set $H_k$ with typical distances $D_H(\epsilon)$
- Low-error set $L_k$ with typical distances $R_L(\epsilon) < D_H(\epsilon)$

The greedy pairing guarantees:

**1. High-Error Walkers Paired with Distant Companions:**

$$
\mathbb{E}[d_i \mid S_t, i \in H_k] \ge D_H(\epsilon) - O(\epsilon)
$$

**2. Low-Error Walkers Paired with Close Companions:**

$$
\mathbb{E}[d_j \mid S_t, j \in L_k] \le R_L(\epsilon) + D_{\mathrm{valid}} \cdot c_k \exp\left(-\frac{D_H(\epsilon)^2 - R_L(\epsilon)^2}{2\epsilon^2}\right)
$$

where the exponential term is negligible when $D_H(\epsilon) \gg R_L(\epsilon)$.

**Interpretation:** The greedy algorithm preserves the geometric signal despite not computing the optimal matching.

**Related Results:**
- Enables: Measurement variance detection (Chapter 6)
- Foundation for: Keystone Principle
- N-uniform: All constants independent of $N$

---

### Raw Value Operators

**Type:** Definition
**Label:** `def-raw-value-operators`
**Source:** [03_cloning.md § 5.2.1](03_cloning.md#521-raw-measurements)
**Tags:** `measurement`, `reward`, `diversity`, `raw-values`

**Statement:**

**1. Reward Measurement:**

$$
r_i := R(x_i, v_i) = R_{\text{pos}}(x_i) - c_{v\_reg} \|v_i\|^2
$$

**2. Paired Distance Measurement:**

$$
d_i := d_{\text{alg}}(i, c(i))
$$

where $c(i)$ is walker $i$'s companion from the pairing operator.

**Related Results:**
- Inputs to: Rescale operators (§ 5.5)
- Used in: Fitness potential ({prf:ref}`def-fitness-potential-operator`)
- Foundation for: Z-score normalization

---

### Canonical Logistic Rescale Function

**Type:** Definition
**Label:** `def-logistic-rescale`
**Source:** [03_cloning.md § 5.5.1](03_cloning.md#551-logistic-rescale)
**Tags:** `rescale`, `logistic`, `bounded`, `lipschitz`

**Statement:**

$$
g_A(z) := \frac{2}{1 + e^{-z}}
$$

**Properties:**
- **Range:** $(0, 2)$
- **Lipschitz:** Globally Lipschitz with constant $\leq 1/2$
- **Monotone:** Strictly increasing
- **Asymptotic:** $g_A(z) \to 0$ as $z \to -\infty$ and $g_A(z) \to 2$ as $z \to \infty$

**Related Results:**
- Used in: Fitness potential ({prf:ref}`def-fitness-potential-operator`)
- Ensures: Bounded fitness values
- Property used in: Raw-to-rescaled gap lemma ({prf:ref}`lem-raw-to-rescaled-gap`)

---

### Fitness Potential Operator

**Type:** Definition
**Label:** `def-fitness-potential-operator`
**Source:** [03_cloning.md § 5.6.1](03_cloning.md#561-fitness-potential)
**Tags:** `fitness`, `potential`, `diversity`, `reward`, `cobb-douglas`

**Statement:**

For alive walker $i \in \mathcal{A}_k$:

**Step 1: Compute Z-scores**

$$
z_{r,i} = \frac{r_i - \mu_r}{\sigma_r}, \quad z_{d,i} = \frac{d_i - \mu_d}{\sigma_d}
$$

where $\mu_r, \sigma_r$ are mean and std of rewards, and $\mu_d, \sigma_d$ are mean and std of distances.

**Step 2: Rescale with Floor**

$$
r'_i := g_A(z_{r,i}) + \eta, \quad d'_i := g_A(z_{d,i}) + \eta
$$

where $\eta > 0$ is a small floor parameter.

**Step 3: Cobb-Douglas Combination**

$$
V_i := (d'_i)^\beta \cdot (r'_i)^\alpha
$$

where $\alpha, \beta > 0$ are exploitation and exploration weights.

**Related Results:**
- Uniform bounds: {prf:ref}`lem-potential-bounds`
- Used in: Cloning score ({prf:ref}`def-cloning-score`)
- Foundation for: Unfit set definition ({prf:ref}`def-unfit-set`)

---

### Uniform Bounds of the Fitness Potential

**Type:** Lemma
**Label:** `lem-potential-bounds`
**Source:** [03_cloning.md § 5.6.2](03_cloning.md#562-uniform-bounds)
**Tags:** `fitness-bounds`, `uniform`, `bounded-potential`

**Statement:**

Any non-zero fitness potential is uniformly bounded:

$$
V_{\text{pot,min}} := \eta^{\alpha+\beta} \le V_i \le (g_{A,\max} + \eta)^{\alpha+\beta} =: V_{\text{pot,max}}
$$

where $g_{A,\max} = 2$ is the maximum value of the logistic rescale function.

**Explicit Values:**
- $V_{\text{pot,min}} = \eta^{\alpha+\beta}$
- $V_{\text{pot,max}} = (2 + \eta)^{\alpha+\beta}$

**Related Results:**
- Ensures: Bounded fitness ratios
- Prevents: Division by zero in cloning scores
- Required for: N-uniform analysis

---

### Companion Selection Operator for Cloning

**Type:** Definition
**Label:** `def-cloning-companion-operator`
**Source:** [03_cloning.md § 5.7.1](03_cloning.md#571-companion-selection-for-cloning)
**Tags:** `companion-selection`, `cloning`, `phase-space-aware`, `softmax`

**Statement:**

The **Companion Selection Operator** $\mathcal{C}_i(S)$ defines a probability measure for selecting walker $i$'s companion:

**If $i$ is ALIVE** ($i \in \mathcal{A}_k$):

$$
P(c_i=j \mid i \in \mathcal{A}_k) := \frac{\exp\left(-\frac{d_{\text{alg}}(i, j)^2}{2\epsilon_c^2}\right)}{\sum_{l \in \mathcal{A}_k \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}(i, l)^2}{2\epsilon_c^2}\right)}
$$

where $\epsilon_c > 0$ is the cloning interaction range.

**If $i$ is DEAD** ($i \in \mathcal{D}_k$):

$$
P(c_i=j \mid i \in \mathcal{D}_k) := \frac{1}{k_{\text{alive}}}
$$

(uniform selection from alive walkers)

**Related Results:**
- Used in: Cloning probability ({prf:ref}`def-cloning-probability`)
- Foundation for: Companion fitness gap ({prf:ref}`lem-mean-companion-fitness-gap`)
- Defines: Spatial locality of cloning

---

### The Canonical Cloning Score

**Type:** Definition
**Label:** `def-cloning-score`
**Source:** [03_cloning.md § 5.7.2](03_cloning.md#572-cloning-score)
**Tags:** `cloning-score`, `fitness-ratio`, `companion-comparison`

**Statement:**

$$
S_i(c_i) := \frac{V_{\text{fit},{c_i}} - V_{\text{fit},i}}{V_{\text{fit},i} + \varepsilon_{\mathrm{clone}}}
$$

where:
- $V_{\text{fit},i}$ is walker $i$'s fitness potential
- $V_{\text{fit},{c_i}}$ is the companion's fitness potential
- $\varepsilon_{\mathrm{clone}} > 0$ is a small regularization parameter

**Interpretation:**
- $S_i(c_i) > 0$: Companion is fitter (positive cloning pressure)
- $S_i(c_i) \le 0$: Walker is at least as fit as companion (no cloning)

**Related Results:**
- Used in: Total cloning probability ({prf:ref}`def-cloning-probability`)
- Foundation for: Cloning pressure analysis (Chapter 8)

---

### Total Cloning Probability

**Type:** Definition
**Label:** `def-cloning-probability`
**Source:** [03_cloning.md § 5.7.2.1](03_cloning.md#5721-cloning-probability)
**Tags:** `cloning-probability`, `expected-cloning`, `stochastic-threshold`

**Statement:**

$$
p_i := \mathbb{E}_{c_i \sim \mathcal{C}_i(S)} \left[ \mathbb{P}_{T_i \sim U(0,p_{\max})} \left( S_i(c_i) > T_i \right) \right]
$$

**Simplified Form:**

$$
p_i = \mathbb{E}_{c_i \sim \mathcal{C}_i(S)}\left[\min\left(1, \max\left(0, \frac{S_i(c_i)}{p_{\max}}\right)\right)\right]
$$

where $p_{\max}$ is the maximum cloning probability per step.

**Interpretation:** $p_i$ is the probability that walker $i$ will clone in a single step, averaging over all possible companion selections and threshold draws.

**Related Results:**
- Lower bound for unfit walkers: {prf:ref}`lem-unfit-cloning-pressure`
- Enhanced near boundary: {prf:ref}`lem-enhanced-cloning-near-boundary`
- Foundation for: Keystone Principle

---

### The Inelastic Collision State Update

**Type:** Definition
**Label:** `def-inelastic-collision-update`
**Source:** [03_cloning.md § 5.7.4](03_cloning.md#574-inelastic-collision)
**Tags:** `inelastic-collision`, `momentum-conservation`, `cloning-update`, `position-jitter`

**Statement:**

For $M$ cloners $I_c = \{i_1, \ldots, i_M\}$ selecting companion $c \in \mathcal{A}_k$:

**Position Updates:**

$$
\begin{aligned}
x'_j &:= x_c + \sigma_x \zeta_j^x \quad \text{for } j \in I_c \\
x'_c &:= x_c
\end{aligned}
$$

where $\zeta_j^x \sim \mathcal{N}(0, I_d)$ are independent Gaussian jitters with scale $\sigma_x > 0$.

**Velocity Updates (Inelastic Collapse):**

**Step 1:** Compute center-of-mass velocity in lab frame:

$$
V_{COM, c} := \frac{1}{M+1} \left( v_c + \sum_{j \in I_c} v_j \right)
$$

**Step 2:** Transform to COM frame and apply restitution with random rotation:

$$
u'_k := \alpha_{\text{restitution}} \cdot R_k(u_k)
$$

where:
- $u_k = v_k - V_{COM,c}$ is relative velocity
- $R_k$ is a random orthogonal matrix (random rotation)
- $\alpha_{\text{restitution}} \in [0, 1]$ is the restitution coefficient

**Step 3:** Transform back to lab frame:

$$
v'_k := V_{COM, c} + u'_k \quad \text{for all } k \in I_c \cup \{c\}
$$

**Properties:**
- **Momentum Conservation:** Total momentum $\sum_k m_k v_k$ is preserved exactly
- **Energy Dissipation:** Kinetic energy decreases by factor $\alpha_{\text{restitution}}^2$ in COM frame
- **Spatial Collapse:** All cloners positioned near companion

**Related Results:**
- Bounded velocity expansion: {prf:ref}`prop-bounded-velocity-expansion`
- Enables: Momentum-conserving cloning dynamics
- Foundation for: Velocity variance analysis

---

### Bounded Velocity Variance Expansion from Cloning

**Type:** Proposition
**Label:** `prop-bounded-velocity-expansion`
**Source:** [03_cloning.md § 5.7.5](03_cloning.md#575-velocity-variance-bound)
**Tags:** `velocity-expansion`, `bounded-growth`, `cloning-noise`

**Statement:**

For cloning with fraction $f_{\text{clone}}$ (expected fraction of walkers that clone per step) and restitution coefficient $\alpha_{\text{restitution}}$:

$$
\Delta V_{Var,v} \leq f_{\text{clone}} \cdot C_{\text{reset}} \cdot V_{\max,\text{KE}}
$$

where:

$$
C_{\text{reset}} := 8(\alpha_{\text{restitution}}^2 + 4), \quad V_{\max,\text{KE}} := V_{\max}^2
$$

and $V_{\max}$ is the maximum velocity magnitude in the domain.

**Interpretation:** Velocity variance can increase under cloning, but the expansion per step is bounded by a state-independent constant.

**Related Results:**
- Key result for: Complete variance drift ({prf:ref}`thm-complete-variance-drift-cloning`)
- Compensated by: Langevin friction in kinetic operator
- Ensures: Bounded long-term velocity variance

---

## Geometric Properties

### Large $V_{\text{Var},x}$ Implies Large Single-Swarm Positional Variance

**Type:** Lemma
**Label:** `lem-V_Varx-implies-variance`
**Source:** [03_cloning.md § 6.2](03_cloning.md#62-variance-localization)
**Tags:** `variance-localization`, `pigeonhole`, `single-swarm`

**Statement:**

If the coupled positional variance exceeds a threshold:

$$
V_{Var,x} > R_{total\_var,x}^2
$$

then at least one swarm $k \in \{1, 2\}$ satisfies:

$$
\frac{1}{N} \sum_{i \in \mathcal{A}(S_k)} \|\delta_{x,k,i}\|^2 > \frac{R_{total\_var,x}^2}{2}
$$

**Proof:** Pigeonhole principle.

**Related Results:**
- Reduces: Coupled analysis to single-swarm analysis
- Foundation for: High-error set definitions (Chapter 6)
- Enables: Geometric partition construction

---

### The Unified High-Error and Low-Error Sets

**Type:** Definition
**Label:** `def-unified-high-low-error-sets`
**Source:** [03_cloning.md § 6.3.1](03_cloning.md#631-clustering-based-partition)
**Tags:** `high-error-set`, `low-error-set`, `clustering`, `outliers`, `partition`

**Statement:**

**Step 1: Cluster alive walkers** into clusters $\{G_1, \ldots, G_M\}$ such that:

$$
\text{diam}(G_m) := \max_{i,j \in G_m} d_{\text{alg}}(i, j) \le D_{\text{diam}}(\epsilon) = c_d \cdot \epsilon
$$

for some constant $c_d > 0$ and interaction range $\epsilon > 0$.

**Step 2: Compute hypocoercive variance contribution** of each cluster:

$$
\text{Contrib}(G_m) := |G_m| \left(\|\mu_{x,m} - \mu_x\|^2 + \lambda_v \|\mu_{v,m} - \mu_v\|^2\right)
$$

where $\mu_{x,m}, \mu_{v,m}$ are cluster $m$'s barycenters.

**Step 3: Identify outlier clusters** $O_M$ whose cumulative contribution is substantial:

$$
\sum_{m \in O_M} \text{Contrib}(G_m) \ge (1-\varepsilon_O) \sum_{\substack{m: |G_m| \ge k_{\min}}} \text{Contrib}(G_m)
$$

where $\varepsilon_O > 0$ is a small tolerance and $k_{\min}$ is the minimum cluster size.

**Step 4: Define high-error and low-error sets:**

$$
\begin{aligned}
H_k(\epsilon) &:= \left(\bigcup_{m \in O_M} G_m\right) \cup \left(\bigcup_{m: |G_m| < k_{\min}} G_m\right) \\
L_k(\epsilon) &:= \mathcal{A}_k \setminus H_k(\epsilon)
\end{aligned}
$$

**Interpretation:**
- $H_k(\epsilon)$: Walkers in outlier clusters or small clusters (high error)
- $L_k(\epsilon)$: Walkers in large, central clusters (low error)

**Related Results:**
- Foundation for: Geometric separation ({prf:ref}`lem-geometric-separation-of-partition`)
- N-uniform size: {prf:ref}`cor-vvarx-to-high-error-fraction`
- Used in: Fitness correctness analysis (Chapter 7)

---

### The Phase-Space Packing Lemma

**Type:** Lemma
**Label:** `lem-phase-space-packing`
**Source:** [03_cloning.md § 6.4.1](03_cloning.md#641-packing-lemma)
**Tags:** `packing`, `variance-implies-spread`, `close-pairs`, `geometric`

**Statement:**

Define total **hypocoercive variance**:

$$
\mathrm{Var}_h(S_k) := \mathrm{Var}_x(S_k) + \lambda_v \mathrm{Var}_v(S_k)
$$

For proximity threshold $d_{\text{close}}$, let:

$$
f_{\text{close}} = \frac{N_{\text{close}}}{\binom{k}{2}}
$$

be the fraction of walker pairs with $d_{\text{alg}}(i, j) < d_{\text{close}}$.

**Then:**

$$
f_{\text{close}} \le g(\mathrm{Var}_h(S_k)) := \frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h(S_k)}{D_{\text{valid}}^2 - d_{\text{close}}^2}
$$

where $D_{\text{valid}}^2 := D_x^2 + \lambda_{\text{alg}} D_v^2$ is the hypocoercive diameter of the domain.

**Consequence:** If $\mathrm{Var}_h(S_k) > d_{\text{close}}^2 / 2$, then:

$$
g(\mathrm{Var}_h) < 1
$$

Not all pairs can be close simultaneously.

**Related Results:**
- Foundation for: Outlier fraction bounds ({prf:ref}`lem-outlier-fraction-lower-bound`)
- Geometric constraint: High variance implies spatial spread
- N-uniform: Fraction bounds independent of $N$

---

### Positional Variance as a Lower Bound for Hypocoercive Variance

**Type:** Lemma
**Label:** `lem-var-x-implies-var-h`
**Source:** [03_cloning.md § 6.4.2.1](03_cloning.md#6421-positional-lower-bound)
**Tags:** `variance-relation`, `lower-bound`, `positional-hypocoercive`

**Statement:**

$$
\mathrm{Var}_h(S_k) \ge \mathrm{Var}_x(S_k)
$$

**Consequence:** If positional variance is large:

$$
\mathrm{Var}_x(S_k) > R^2_{\text{var}} \quad \Longrightarrow \quad \mathrm{Var}_h(S_k) > R^2_{\text{var}}
$$

**Proof:** Direct from definition since $\mathrm{Var}_v \geq 0$.

**Related Results:**
- Simplifies: Variance threshold analysis
- Used in: Packing lemma applications
- Enables: Position-based variance detection

---

### N-Uniform Lower Bound on the Outlier Fraction

**Type:** Lemma
**Label:** `lem-outlier-fraction-lower-bound`
**Source:** [03_cloning.md § 6.4.2.2](03_cloning.md#6422-outlier-fraction)
**Tags:** `outlier-fraction`, `N-uniform`, `variance-to-population`

**Statement:**

If the hypocoercive variance exceeds a threshold:

$$
\mathrm{Var}_h(S_k) > R^2_h
$$

then the outlier set $O_k$ (walkers contributing to variance) satisfies:

$$
\frac{|O_k|}{k} \ge \frac{(1-\varepsilon_O) R^2_h}{D_h^2} =: f_O > 0
$$

where:
- $D_h^2 := D_x^2 + \lambda_v D_v^2$ is the hypocoercive diameter
- $\varepsilon_O > 0$ is a small tolerance
- $f_O$ is **N-uniform** (independent of $N$ and $k$)

**Related Results:**
- Used in: High-error fraction bounds ({prf:ref}`cor-vvarx-to-high-error-fraction`)
- Foundation for: N-uniform Keystone Principle
- Ensures: Substantial outlier population for high variance

---

### N-Uniform Lower Bound on the Outlier-Cluster Fraction

**Type:** Lemma
**Label:** `lem-outlier-cluster-fraction-lower-bound`
**Source:** [03_cloning.md § 6.4.3](03_cloning.md#643-cluster-outlier-fraction)
**Tags:** `clustering`, `outlier-clusters`, `N-uniform`, `high-error-fraction`

**Statement:**

For clustering-based high-error set with cluster diameter bound:

$$
c_d \cdot \epsilon < 2\sqrt{R^2_{\text{var}}}
$$

if $\mathrm{Var}_x(S_k) > R^2_{\text{var}}$, then:

$$
\frac{|H_k(\epsilon)|}{k} \ge f_H(\epsilon) > 0
$$

where:

$$
f_H(\epsilon) = \frac{(1-\varepsilon_O) \left(R^2_{\mathrm{var}} - (D_{\mathrm{diam}}(\epsilon)/2)^2\right)}{D_{\mathrm{valid}}^2}
$$

is **N-uniform** (independent of $N$ and $k$) and **$\epsilon$-dependent**.

**Related Results:**
- Foundation for: Keystone Principle
- Used in: Corrective fitness analysis (Chapter 7)
- Ensures: Substantial high-error population

---

### Large Intra-Swarm Positional Variance Guarantees Non-Vanishing High-Error Fraction

**Type:** Corollary
**Label:** `cor-vvarx-to-high-error-fraction`
**Source:** [03_cloning.md § 6.4.4](03_cloning.md#644-variance-to-high-error)
**Tags:** `variance-to-fraction`, `N-uniform`, `high-error-guarantee`

**Statement:**

If the coupled positional variance is large:

$$
V_{\text{Var},x} > R^2_{\text{total\_var},x}
$$

then at least one swarm $k \in \{1, 2\}$ satisfies:

$$
\frac{|H_k(\epsilon)|}{k} \ge f_H(\epsilon) > 0
$$

where:

$$
f_H(\epsilon) := \min(f_O, f_{H,\text{cluster}}(\epsilon))
$$

is **N-uniform** and **$\epsilon$-dependent**.

**Interpretation:** Large system-level positional variance guarantees a substantial high-error population in at least one swarm.

**Related Results:**
- Combines: Variance localization ({prf:ref}`lem-V_Varx-implies-variance`) and outlier bounds
- Foundation for: Keystone Principle
- Key N-uniform result: Enables scalable analysis

---

### Geometric Separation of the Partition

**Type:** Lemma
**Label:** `lem-geometric-separation-of-partition`
**Source:** [03_cloning.md § 6.5.1](03_cloning.md#651-geometric-separation)
**Tags:** `separation`, `clustering`, `high-low-distance`, `N-uniform`

**Statement:**

If $\mathrm{Var}(x) > R^2_{\mathrm{var}}$, then there exist **N-uniform** constants $D_H(\epsilon) > R_L(\epsilon) > 0$ and $f_c > 0$ such that:

**Part 1 (Separation):** For any $i \in H_k(\epsilon)$ and $j \in L_k(\epsilon)$:

$$
d_{\text{alg}}(i, j) \ge D_H(\epsilon)
$$

High-error walkers are far from low-error walkers.

**Part 2 (Low-Error Clustering):** For any $j \in L_k(\epsilon)$, there exists a cluster $C_j \subset L_k(\epsilon)$ with $|C_j| \ge f_c k$ such that:

$$
d_{\text{alg}}(j, \ell) \le R_L(\epsilon) \quad \forall \ell \in C_j
$$

Each low-error walker has many nearby low-error companions.

**Typical Values:**
- $R_L(\epsilon) = c_d \cdot \epsilon$ (cluster diameter)
- $D_H(\epsilon) \gg R_L(\epsilon)$ (well-separated)

**Related Results:**
- Foundation for: Measurement variance ({prf:ref}`thm-geometric-structure-guarantees-measurement-variance`)
- Used in: Greedy pairing signal preservation ({prf:ref}`lem-greedy-preserves-signal`)
- Key geometric property: Enables fitness signal detection

---

## Contraction and Stability Results

### Geometric Structure Guarantees Measurement Variance

**Type:** Theorem
**Label:** `thm-geometric-structure-guarantees-measurement-variance`
**Source:** [03_cloning.md § 7.2](03_cloning.md#72-measurement-variance)
**Tags:** `measurement-variance`, `geometric-signal`, `distance-variance`, `N-uniform`

**Statement:**

For a high-variance swarm with partition $(H_k, L_k)$, the variance of raw distance measurements satisfies:

$$
\text{Var}(d) \ge \kappa_d(\epsilon) \cdot f_H(\epsilon) \cdot [D_H(\epsilon) - R_L(\epsilon)]^2
$$

where:
- $\kappa_d(\epsilon) > 0$ is a geometric constant
- $f_H(\epsilon) > 0$ is the high-error fraction
- $D_H(\epsilon) - R_L(\epsilon)$ is the separation gap

**All constants are N-uniform and $\epsilon$-dependent.**

**Interpretation:** Large positional variance creates detectable variance in diversity measurements.

**Related Results:**
- Uses: Geometric separation ({prf:ref}`lem-geometric-separation-of-partition`)
- Foundation for: Corrective fitness (Chapter 7)
- Key step in: Keystone Principle proof chain

---

### From Raw Measurement Gap to Rescaled Value Gap

**Type:** Lemma
**Label:** `lem-raw-to-rescaled-gap`
**Source:** [03_cloning.md § 7.3.3](03_cloning.md#733-rescaling-preserves-gap)
**Tags:** `rescaling`, `gap-preservation`, `z-score`, `lipschitz`

**Statement:**

If raw diversity measurements satisfy:

$$
\mu_{d,H} - \mu_{d,L} \ge \Delta_{\text{raw},d}
$$

then rescaled values satisfy:

$$
\mu_{d'_H} - \mu_{d'_L} \ge g'_{\min} \cdot \frac{\Delta_{\text{raw},d}}{\sigma'_{\max}}
$$

where:
- $g'_{\min} > 0$ is the minimum derivative of the rescale function $g_A$
- $\sigma'_{\max}$ is the maximum patched standard deviation

**Interpretation:** The rescaling and normalization steps preserve a quantifiable gap between high-error and low-error fitness contributions.

**Related Results:**
- Uses: Logistic rescale properties ({prf:ref}`def-logistic-rescale`)
- Foundation for: Fitness gap analysis
- Ensures: Signal preservation through preprocessing

---

### Derivation of the Stability Condition for Intelligent Adaptation

**Type:** Theorem
**Label:** `thm-stability-condition-intelligent-adaptation`
**Source:** [03_cloning.md § 7.5](03_cloning.md#75-stability-condition)
**Tags:** `stability-condition`, `diversity-dominance`, `intelligent-adaptation`, `beta-alpha-balance`

**Statement:**

For the diversity signal to dominate reward noise and ensure the unfit set $U_k$ has substantial overlap with the high-error set $H_k$, the following **Stability Condition** must hold:

$$
\beta \cdot \ln\left(1 + \frac{\Delta_{d'}}{\eta}\right) > \alpha \cdot \ln\left(\frac{\mu_F}{\mu_U}\right)
$$

where:
- $\beta$ is the diversity exponent
- $\alpha$ is the reward exponent
- $\Delta_{d'}$ is the rescaled diversity gap
- $\eta$ is the rescale floor parameter
- $\mu_F / \mu_U$ is the fit-to-unfit reward ratio

**Interpretation:** The diversity channel must be strong enough (high $\beta$) relative to the reward channel ($\alpha$) to overcome potential reward variations.

**Related Results:**
- Required for: Unfit-high-error overlap ({prf:ref}`thm-unfit-high-error-overlap-fraction`)
- Foundation for: Keystone Principle
- Design constraint: Guides parameter selection ($\alpha, \beta, \eta$)

---

### The Unfit Set

**Type:** Definition
**Label:** `def-unfit-set`
**Source:** [03_cloning.md § 7.6.1](03_cloning.md#761-unfit-set-definition)
**Tags:** `unfit-set`, `below-median`, `cloning-targets`

**Statement:**

$$
U_k := \{i \in \mathcal{A}_k : V_{k,i} \le \mu_{V,k}\}
$$

where:
- $V_{k,i}$ is walker $i$'s fitness potential
- $\mu_{V,k} = \frac{1}{k} \sum_{j \in \mathcal{A}_k} V_{k,j}$ is the mean fitness

**Interpretation:** Walkers with below-average fitness are candidates for cloning.

**Related Results:**
- Minimum size: {prf:ref}`lem-unfit-fraction-lower-bound`
- Overlap with high-error set: {prf:ref}`thm-unfit-high-error-overlap-fraction`
- Cloning pressure: {prf:ref}`lem-unfit-cloning-pressure`

---

### N-Uniform Lower Bound on the Unfit Fraction

**Type:** Lemma
**Label:** `lem-unfit-fraction-lower-bound`
**Source:** [03_cloning.md § 7.6.2](03_cloning.md#762-unfit-fraction)
**Tags:** `unfit-fraction`, `median-bound`, `N-uniform`

**Statement:**

$$
\frac{|U_k|}{k} \ge \frac{1}{2}
$$

**Proof:** By definition, at least half the walkers have fitness below or equal to the mean.

**Interpretation:** At least half the alive walkers are unfit.

**Related Results:**
- Trivial but essential: Ensures substantial unfit population
- Used in: Overlap fraction analysis
- N-uniform: Holds for any $N, k$

---

### N-Uniform Lower Bound on the Unfit-High-Error Overlap Fraction

**Type:** Theorem
**Label:** `thm-unfit-high-error-overlap-fraction`
**Source:** [03_cloning.md § 7.6.3](03_cloning.md#763-overlap-fraction)
**Tags:** `overlap-fraction`, `unfit-high-error`, `N-uniform`, `stability-condition`

**Statement:**

Under the Stability Condition ({prf:ref}`thm-stability-condition-intelligent-adaptation`), the overlap fraction satisfies:

$$
\frac{|U_k \cap H_k(\epsilon)|}{k} \ge f_{UH}(\epsilon) := \frac{1}{2}f_H(\epsilon) > 0
$$

where $f_H(\epsilon)$ is the high-error fraction.

**All constants are N-uniform and $\epsilon$-dependent.**

**Interpretation:** A substantial fraction of walkers are simultaneously unfit AND high-error, meaning the fitness function correctly identifies problematic walkers.

**Related Results:**
- Requires: Stability Condition ({prf:ref}`thm-stability-condition-intelligent-adaptation`)
- Foundation for: Keystone Principle ({prf:ref}`lem-quantitative-keystone`)
- Key correctness result: Fitness aligns with geometric error

---

### The N-Uniform Quantitative Keystone Lemma

**Type:** Lemma
**Label:** `lem-quantitative-keystone`
**Source:** [03_cloning.md § 8.1](03_cloning.md#81-keystone-lemma-statement)
**Tags:** `keystone-principle`, `error-weighted-cloning`, `N-uniform`, `linear-scaling`

**Statement:**

There exist **N-uniform, $\epsilon$-dependent** constants:
- $R^2_{\text{spread}} > 0$ (variance threshold)
- $\chi(\epsilon) > 0$ (contraction rate)
- $g_{\max}(\epsilon) \ge 0$ (bounded error term)

such that:

$$
\frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2 \ge \chi(\epsilon) V_{\text{struct}} - g_{\max}(\epsilon)
$$

where:
- $I_{11}$ is the stably alive set (walkers alive in both swarms)
- $p_{k,i}$ is walker $i$'s cloning probability in swarm $k$
- $\|\Delta\delta_{x,i}\|^2$ is the squared position error for walker $i$

**Interpretation:** The error-weighted cloning pressure scales linearly with the structural error. Walkers with high positional error experience proportionally high cloning pressure.

**Related Results:**
- **Central result of Chapter 8**
- Foundation for: Positional variance contraction ({prf:ref}`thm-positional-variance-contraction`)
- Proof uses: Overlap fraction ({prf:ref}`thm-unfit-high-error-overlap-fraction`) and error concentration

---

### The Critical Target Set

**Type:** Definition
**Label:** `def-critical-target-set`
**Source:** [03_cloning.md § 8.2](03_cloning.md#82-target-set)
**Tags:** `target-set`, `triple-intersection`, `cloning-targets`

**Statement:**

$$
I_{\text{target}} := I_{11} \cap U_k \cap H_k(\epsilon)
$$

**Interpretation:** Walkers that are simultaneously:
1. **Stably alive** ($I_{11}$): Alive in both coupled swarms
2. **Unfit** ($U_k$): Below-average fitness
3. **High-error** ($H_k(\epsilon)$): In geometric outlier set

These are the walkers that will be targeted by the corrective cloning mechanism.

**Related Results:**
- Population bound: $|I_{\text{target}}| / k \ge f_{UH}(\epsilon)$
- Cloning pressure: {prf:ref}`cor-cloning-pressure-target-set`
- Error concentration: {prf:ref}`lem-error-concentration-target-set`

---

### Lower Bound on Mean Companion Fitness Gap

**Type:** Lemma
**Label:** `lem-mean-companion-fitness-gap`
**Source:** [03_cloning.md § 8.3.1](03_cloning.md#831-companion-fitness-gap)
**Tags:** `fitness-gap`, `companion-selection`, `unfit-walker`, `N-uniform`

**Statement:**

For any unfit walker $i \in U_k$, the expected fitness gap with its companion satisfies:

$$
\mu_{\text{comp},i} - V_{k,i} \geq \frac{f_F}{k-1} (\mu_F - \mu_U)
$$

where:
- $\mu_{\text{comp},i} = \mathbb{E}_{c \sim \mathcal{C}_i(S_k)}[V_{k,c}]$ is the expected companion fitness
- $f_F = |F_k|/k$ is the fit fraction
- $\mu_F, \mu_U$ are mean fitness of fit and unfit sets

**Further, the fit-unfit gap satisfies:**

$$
\mu_F - \mu_U \geq \frac{f_U}{f_F + f_U^2/f_F} \kappa_{V,\text{gap}}(\epsilon)
$$

**Combined bound:**

$$
\mu_{\text{comp},i} - V_{k,i} \geq \frac{f_F f_U}{(k-1)(f_F + f_U^2/f_F)} \kappa_{V,\text{gap}}(\epsilon) =: \Delta_{\min}(\epsilon, f_U, f_F, k) > 0
$$

**All constants are N-uniform and $\epsilon$-dependent.**

**Related Results:**
- Uses: Companion selection operator ({prf:ref}`def-cloning-companion-operator`)
- Foundation for: Cloning pressure bound ({prf:ref}`lem-unfit-cloning-pressure`)
- Key mechanism: Spatial proximity biases selection toward fitter walkers

---

### Guaranteed Cloning Pressure on the Unfit Set

**Type:** Lemma
**Label:** `lem-unfit-cloning-pressure`
**Source:** [03_cloning.md § 8.3.2](03_cloning.md#832-cloning-pressure-unfit)
**Tags:** `cloning-pressure`, `unfit-walkers`, `N-uniform`, `lower-bound`

**Statement:**

For any walker $i \in U_k$:

$$
p_{k,i} = \mathbb{E}_{c \sim \mathbb{C}_i(S_k)}[\pi(S(V_{k,c}, V_{k,i}))] \ge p_u(\epsilon) > 0
$$

where:
- $p_{k,i}$ is walker $i$'s total cloning probability
- $p_u(\epsilon)$ is an **N-uniform, $\epsilon$-dependent** constant
- $\pi(\cdot)$ is the stochastic threshold function

**Explicit Lower Bound:**

$$
p_u(\epsilon) := \frac{\Delta_{\min}(\epsilon)}{p_{\max}} \cdot P(\text{select fit companion}) > 0
$$

**Interpretation:** Every unfit walker experiences guaranteed positive cloning pressure.

**Related Results:**
- Uses: Fitness gap ({prf:ref}`lem-mean-companion-fitness-gap`)
- Foundation for: Keystone Lemma ({prf:ref}`lem-quantitative-keystone`)
- Key scalability result: N-uniform lower bound

---

### Cloning Pressure on the Target Set

**Type:** Corollary
**Label:** `cor-cloning-pressure-target-set`
**Source:** [03_cloning.md § 8.3.3](03_cloning.md#833-target-set-pressure)
**Tags:** `target-set`, `cloning-pressure`, `N-uniform`

**Statement:**

For all $i \in I_{\text{target}}$:

$$
p_{k,i} \ge p_u(\epsilon) > 0
$$

**Interpretation:** The N-uniform cloning pressure guarantee extends from the unfit set to the critical target set (stably alive, unfit, high-error walkers).

**Related Results:**
- Direct consequence of: {prf:ref}`lem-unfit-cloning-pressure`
- Used in: Error concentration ({prf:ref}`lem-error-concentration-target-set`)
- Foundation for: Keystone Lemma proof

---

### Variance Concentration in the High-Error Set

**Type:** Lemma
**Label:** `lem-variance-concentration-high-error`
**Source:** [03_cloning.md § 8.4.1](03_cloning.md#841-variance-concentration)
**Tags:** `variance-concentration`, `high-error-set`, `structural-error`, `geometric`

**Statement:**

The target set accounts for a substantial fraction of the structural error:

$$
\frac{1}{N}\sum_{i \in I_{\text{target}}} \|\Delta\delta_{x,i}\|^2 \ge f_{\text{target}} \cdot C_{\text{struct}} \cdot V_{\text{struct}}
$$

where:
- $f_{\text{target}} = f_{UH}(\epsilon) > 0$ is the **N-uniform** target fraction
- $C_{\text{struct}} > 0$ is a geometric constant

**Interpretation:** The error is not uniformly distributed—it concentrates in the target set.

**Related Results:**
- Foundation for: Error-weighted cloning pressure ({prf:ref}`lem-error-concentration-target-set`)
- Key geometric property: Enables linear scaling in Keystone Lemma
- N-uniform: All constants independent of $N$

---

### Error Concentration in the Target Set

**Type:** Lemma
**Label:** `lem-error-concentration-target-set`
**Source:** [03_cloning.md § 8.4.2](03_cloning.md#842-error-weighted-pressure)
**Tags:** `error-weighted-pressure`, `keystone-core`, `N-uniform`

**Statement:**

Combining cloning pressure and error concentration:

$$
\frac{1}{N}\sum_{i \in I_{\text{target}}} p_{k,i} \|\Delta\delta_{x,i}\|^2 \ge p_u(\epsilon) \cdot f_{\text{target}} \cdot C_{\text{struct}} \cdot V_{\text{struct}}
$$

where all constants are **N-uniform** and **$\epsilon$-dependent**.

**Interpretation:** This is the **core quantitative result** enabling the Keystone Lemma. The error-weighted cloning pressure scales linearly with structural error.

**Related Results:**
- Combines: Cloning pressure ({prf:ref}`cor-cloning-pressure-target-set`) and variance concentration ({prf:ref}`lem-variance-concentration-high-error`)
- **Direct proof** of: Keystone Lemma ({prf:ref}`lem-quantitative-keystone`)
- Key scalability result: N-uniform constants

---

## Wasserstein-2 Contraction

### Wasserstein-2 Contraction for Cloning Operator

**Type:** Theorem
**Label:** `thm-w2-cloning-contraction`
**Source:** [03_B__wasserstein_contraction.md § 0.1](03_B__wasserstein_contraction.md)
**Tags:** `wasserstein-2`, `contraction`, `cloning`, `N-uniform`, `coupling`

**Statement:**

$$
\mathbb{E}[W_2^2(\mu_{S_1'}, \mu_{S_2'})] \leq (1 - \kappa_W) W_2^2(\mu_{S_1}, \mu_{S_2}) + C_W
$$

where $\mu_{S_k} = \frac{1}{N}\sum_{i=1}^N \delta_{x_{k,i}}$ is the empirical measure.

**Constants:**
- $\kappa_W = \frac{p_u \eta}{2} > 0$ (contraction rate, N-uniform)
- $C_W = 4d\delta^2$ (jitter noise, N-uniform)
- Typical: $\kappa_W \geq 0.0125$, so $1 - \kappa_W \leq 0.9875 < 1$

**Related Results:**
- Required for: LSI-based KL convergence
- Proven via: Synchronous coupling + Outlier Alignment
- Cases: A (bounded expansion) + B (strong contraction)

---

### Synchronous Cloning Coupling

**Type:** Definition
**Label:** `def-synchronous-cloning-coupling`
**Source:** [03_B__wasserstein_contraction.md § 1.2](03_B__wasserstein_contraction.md)
**Tags:** `coupling`, `synchronous`, `shared-randomness`, `jitter-cancellation`

**Statement:**

Evolve $(S_1, S_2)$ using shared randomness:
1. Sample matching $M \sim P(\cdot | S_1)$ once
2. Use same permutation $\pi$ for both swarms
3. Share cloning thresholds: $T_i \sim U(0, p_{\max})$
4. Share jitter: $\zeta_i \sim \mathcal{N}(0, \delta^2 I_d)$

$$
x'_{k,i} = \begin{cases}
x_{k,\pi(i)} + \zeta_i & \text{if } T_i < p_{k,i} \\
x_{k,i} & \text{otherwise}
\end{cases}
$$

**Key Property:** In Clone-Clone case, jitter cancels: $\|x'_{1,i} - x'_{2,i}\|^2 = \|x_{1,\pi(i)} - x_{2,\pi(i)}\|^2$

**Related Results:**
- Optimal: Minimizes expected W₂ distance
- Enables: Case analysis by fitness ordering
- Foundation for: All contraction proofs

---

### Outlier Alignment Lemma

**Type:** Lemma
**Label:** `lem-outlier-alignment`
**Source:** [03_B__wasserstein_contraction.md § 2.1](03_B__wasserstein_contraction.md)
**Tags:** `outlier-alignment`, `emergent-property`, `geometric`, `N-uniform`

**Statement:**

For swarms separated by $L = \|\bar{x}_1 - \bar{x}_2\| > D_{\min}$, outliers $x_{1,i} \in H_1$ satisfy:

$$
\langle x_{1,i} - \bar{x}_1, \bar{x}_1 - \bar{x}_2 \rangle \geq \eta \|x_{1,i} - \bar{x}_1\| \|\bar{x}_1 - \bar{x}_2\|
$$

with $\eta \geq 1/4$ (conservative) or $\eta = 2/3$ (improved).

**Interpretation:** Outliers point away from the other swarm.

**Key Innovation:** This is **emergent** from cloning dynamics (not axiomatic), proven via fitness valley argument.

**Related Results:**
- Proven via: H-theorem contradiction + survival probability analysis
- Critical for: Case B contraction
- N-uniform: $\eta$ independent of $N$

---

### Case A: Consistent Fitness Ordering

**Type:** Definition + Lemma
**Label:** `def-case-a`, `lem-case-a-bounded-expansion`
**Source:** [03_B__wasserstein_contraction.md § 3](03_B__wasserstein_contraction.md)
**Tags:** `case-a`, `consistent-fitness`, `bounded-expansion`, `jitter-cancellation`

**Definition:** Pair $(i,j)$ has consistent fitness if $V_{1,i} \leq V_{1,j}$ AND $V_{2,i} \leq V_{2,j}$.

**Result:**

$$
\mathbb{E}[D'_{ii} + D'_{jj}] \leq (1 + \epsilon_A) (D_{ii} + D_{jj}) + 2d\delta^2
$$

with $\epsilon_A = O(R_H/L) \ll 1$ for separated swarms.

**Subcases:** PP (persist-persist), CC (clone-clone with jitter cancellation), CP, PC

**Related Results:**
- Jitter cancellation: Key in CC subcase
- Bounded expansion: Not contraction
- Requires Case B: For overall contraction

---

### Case B: Mixed Fitness Ordering

**Type:** Definition + Lemma
**Label:** `def-case-b`, `lem-case-b-contraction`
**Source:** [03_B__wasserstein_contraction.md § 4](03_B__wasserstein_contraction.md)
**Tags:** `case-b`, `mixed-fitness`, `strong-contraction`, `outlier-alignment`

**Definition:** Pair $(i,j)$ has mixed fitness if $V_{1,i} \leq V_{1,j}$ BUT $V_{2,i} > V_{2,j}$.

**Result:**

$$
\mathbb{E}[D'_{ii} + D'_{jj}] \leq \gamma_B (D_{ii} + D_{jj}) + 4d\delta^2
$$

with $\gamma_B = 1 - \frac{p_u \eta}{2} \leq 0.9875 < 1$ (strong contraction).

**Key Bound:** Using Outlier Alignment:

$$
D_{ii} - D_{ji} \geq \eta R_H L
$$

**Related Results:**
- Uses: Outlier Alignment Lemma critically
- Walker roles: $i$ outlier in $S_1$, companion in $S_2$; $j$ opposite
- Provides: Main contraction for W₂

---

### Contraction Constants

**Type:** Definition
**Label:** `def-w2-contraction-constants`
**Source:** [03_B__wasserstein_contraction.md § 7.2](03_B__wasserstein_contraction.md)
**Tags:** `constants`, `N-uniform`, `contraction-rate`

**Contraction Rate:**

$$
\kappa_W = 1 - \max(\gamma_A, \gamma_B) = \frac{p_u \eta}{2}
$$

with $p_u \geq 0.1$ (unfit cloning prob), $\eta \geq 1/4$ (outlier alignment).

**Additive Constant:**

$$
C_W = 4d\delta^2
$$

**Explicit Bounds:** $\kappa_W \geq 0.0125$ for typical parameters.

**N-Uniformity:** Both constants independent of $N$.

**Related Results:**
- Used in: Main theorem
- Required for: Mean-field analysis
- Part of: LSI-based convergence

---

## Kinetic Operator and Hypocoercivity

### The Kinetic Operator (Stratonovich Form)

**Type:** Definition
**Label:** `def-kinetic-operator-stratonovich`
**Source:** [04_convergence.md § 1.2](04_convergence.md)
**Tags:** `kinetic`, `langevin`, `stratonovich`, `underdamped`

**Statement:**

$$
\begin{aligned}
dx_t &= v_t \, dt \\
dv_t &= F(x_t) \, dt - \gamma(v_t - u(x_t)) \, dt + \Sigma(x_t, v_t) \circ dW_t
\end{aligned}
$$

**Components:**
- Force field: $F(x) = -\nabla U(x)$ from confining potential
- Friction: $\gamma > 0$
- Diffusion tensor: $\Sigma(x,v) \in \mathbb{R}^{d \times d}$
- Stratonovich product: $\circ$ (geometric invariance)

**Boundary:** $s_i^{(t+1)} = \mathbf{1}_{\mathcal{X}_{\text{valid}}}(x_i(t+\tau))$

**Related Results:**
- Axioms: Confining potential, diffusion tensor, friction
- Numerical: BAOAB integrator
- Complements: Cloning operator

---

### Axiom: Globally Confining Potential

**Type:** Axiom
**Label:** `axiom-confining-potential`
**Source:** [04_convergence.md § 1.3](04_convergence.md)
**Tags:** `confining-potential`, `coercivity`, `boundary-inward`

**Key Properties:**

1. **Smoothness:** $U \in C^2(\mathcal{X}_{\text{valid}})$
2. **Coercivity:** $\langle x, \nabla U(x) \rangle \geq \alpha_U \|x\|^2 - R_U$
3. **Bounded interior force:** $\|F(x)\| \leq F_{\max}$ for $x \in B(0, r_{\text{interior}})$
4. **Boundary-inward:** $\langle \vec{n}(x), F(x) \rangle < 0$ near $\partial \mathcal{X}_{\text{valid}}$

**Related Results:**
- Enables: Hypocoercive contraction
- Ensures: Bounded velocity equilibrium
- Foundation for: Boundary contraction

---

### Hypocoercive Contraction (No Convexity Required)

**Type:** Theorem
**Label:** `thm-inter-swarm-contraction-kinetic`
**Source:** [04_convergence.md § 2.3.1](04_convergence.md)
**Tags:** `hypocoercivity`, `wasserstein-contraction`, `no-convexity`, `N-uniform`

**Statement:**

$$
\mathbb{E}_{\text{kin}}[V_W(S'_1, S'_2)] \leq (1 - \kappa_W \tau) V_W(S_1, S_2) + C_W' \tau
$$

**Constants:**
- $\kappa_W \sim \min(\gamma, \alpha_U, \sigma_{\min}^2)$
- $C_W' = O(\sigma_v^2/\gamma)$
- Both N-uniform

**Key Innovation:** Uses only coercivity, Lipschitz, and friction-transport coupling—NOT convexity.

**Related Results:**
- Compensates: Cloning expansion of $V_W$
- Requires: Hypocoercive coupling ($b \neq 0$)
- Foundation for: Synergistic contraction

---

### Velocity Variance Dissipation

**Type:** Theorem
**Label:** `thm-velocity-variance-contraction-kinetic`
**Source:** [04_convergence.md § 3.3.1](04_convergence.md)
**Tags:** `velocity-dissipation`, `langevin-friction`, `bounded-equilibrium`

**Statement:**

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v} \tau + \sigma_{\max}^2 d \tau
$$

**Equilibrium:** When $V_{\text{Var},v} > \frac{\sigma_{\max}^2 d}{2\gamma}$, drift is strictly negative.

**Steady State:** $V_{\text{Var},v}^{\text{eq}} \approx \frac{d\sigma_{\max}^2}{2\gamma}$

**Related Results:**
- Compensates: Cloning expansion $C_v$
- Pure dissipation: Langevin friction
- Enables: Gibbs distribution for velocities

---

### BAOAB Integrator

**Type:** Definition
**Label:** `def-baoab-integrator`
**Source:** [04_convergence.md § 1.5](04_convergence.md)
**Tags:** `baoab`, `numerical-integration`, `second-order`, `splitting`

**Five-Step Scheme:**

$$
\begin{aligned}
\text{B:} \quad & v_{t+\tau/2} = v_t + \frac{\tau}{2} F(x_t) \\
\text{A:} \quad & x_{t+\tau/2} = x_t + \frac{\tau}{2} v_{t+\tau/2} \\
\text{O:} \quad & v_{t+\tau/2}' = e^{-\gamma\tau} v_{t+\tau/2} + \sqrt{1 - e^{-2\gamma\tau}} \sigma_v \xi \\
\text{A:} \quad & x_{t+\tau} = x_{t+\tau/2} + \frac{\tau}{2} v_{t+\tau/2}' \\
\text{B:} \quad & v_{t+\tau} = v_{t+\tau/2}' + \frac{\tau}{2} F(x_{t+\tau})
\end{aligned}
$$

where $\xi \sim \mathcal{N}(0, I_d)$.

**Properties:**
- Second-order weak accuracy
- Preserves Gibbs measure exactly (O-step)
- Stratonovich-consistent for isotropic diffusion

**Related Results:**
- Inherits generator drift (Theorem 1.7.2)
- Weak error bounds for Lyapunov functions
- Timestep constraint: $\tau < \tau_*$

---

### Foster-Lyapunov for Composed Operator

**Type:** Theorem
**Label:** `thm-foster-lyapunov-main`
**Source:** [04_convergence.md § 6.4.1](04_convergence.md)
**Tags:** `foster-lyapunov`, `composed-operator`, `synergistic`, `N-uniform`

**Statement:**

$$
\mathbb{E}_{\text{total}}[V_{\text{total}}(S')] \leq (1 - \kappa_{\text{total}}\tau) V_{\text{total}}(S) + C_{\text{total}}
$$

**Total Rate:**

$$
\kappa_{\text{total}} := \min\left(\frac{\kappa_W}{2}, \frac{c_V^* \kappa_x}{2}, \frac{c_V^* \gamma}{2}, \frac{c_B^*(\kappa_b + \kappa_{\text{pot}}\tau)}{2}\right) > 0
$$

**Key Property:** All constants N-uniform.

**Related Results:**
- **Main convergence result**
- Combines: Cloning + kinetic drifts
- Enables: Geometric ergodicity

---

## Drift Inequalities

### Positional Variance Contraction Under Cloning

**Type:** Theorem
**Label:** `thm-positional-variance-contraction`
**Source:** [03_cloning.md § 10.3.1](03_cloning.md#1031-main-result)
**Tags:** `positional-contraction`, `foster-lyapunov`, `cloning-operator`, `N-uniform`

**Statement:**

There exist **N-uniform** constants $\kappa_x > 0$, $C_x < \infty$, $R^2_{\text{spread}} > 0$ such that:

**Foster-Lyapunov Drift Inequality:**

$$
\mathbb{E}_{\text{clone}}[V_{\text{Var},x}(S'_1, S'_2) \mid S_1, S_2] \leq (1 - \kappa_x) V_{\text{Var},x}(S_1, S_2) + C_x
$$

**Equivalent Form:**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x
$$

**Contraction Regime:** When $V_{\text{Var},x} > \tilde{C}_x := C_x / \kappa_x$ (sufficiently large):

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] < 0
$$

Strict expected contraction.

**Related Results:**
- **Main result of Chapter 10**
- Proven using: Keystone Lemma ({prf:ref}`lem-quantitative-keystone`)
- Part of: Complete variance drift ({prf:ref}`thm-complete-variance-drift-cloning`)

---

### Variance Change Decomposition

**Type:** Lemma
**Label:** `lem-variance-change-decomposition`
**Source:** [03_cloning.md § 10.3.3](03_cloning.md#1033-variance-decomposition)
**Tags:** `variance-decomposition`, `alive-vs-status`, `bookkeeping`

**Statement:**

$$
\Delta V_{\text{Var},x} = \sum_{k=1}^{2} \left[\underbrace{\Delta V_{\text{Var},x}^{(k,\text{alive})}}_{\text{alive walkers}} + \underbrace{\Delta V_{\text{Var},x}^{(k,\text{status})}}_{\text{status changes}}\right]
$$

where:

**Alive Walker Contribution:**

$$
\Delta V_{\text{Var},x}^{(k,\text{alive})} = \frac{1}{N}\sum_{i \in \mathcal{A}(S_k)} \left[\|\delta'_{x,k,i}\|^2 - \|\delta_{x,k,i}\|^2\right]
$$

**Status Change Contribution:**

$$
\Delta V_{\text{Var},x}^{(k,\text{status})} = \frac{1}{N}\sum_{i \in \mathcal{D}(S_k)} \|\delta'_{x,k,i}\|^2
$$

(newly revived walkers contribute their new variance)

**Related Results:**
- Enables: Separate analysis of alive and revived contributions
- Used in: Keystone contraction proof ({prf:ref}`lem-keystone-contraction-alive`)
- Accounting tool: Tracks all variance sources

---

### Keystone-Driven Contraction for Stably Alive Walkers

**Type:** Lemma
**Label:** `lem-keystone-contraction-alive`
**Source:** [03_cloning.md § 10.3.4](03_cloning.md#1034-keystone-application)
**Tags:** `keystone-application`, `stably-alive`, `contraction-core`

**Statement:**

$$
\mathbb{E}_{\text{clone}}\left[\sum_{i \in I_{11}} \sum_{k=1,2} \left(\|\delta'_{x,k,i}\|^2 - \|\delta_{x,k,i}\|^2\right)\right] \leq -\frac{\chi(\epsilon)}{2N} \cdot V_{\text{struct}} + \frac{g_{\max}(\epsilon)}{N} + C_{\text{pers}}
$$

where:
- $\chi(\epsilon), g_{\max}(\epsilon)$ are Keystone constants ({prf:ref}`lem-quantitative-keystone`)
- $C_{\text{pers}}$ accounts for persisting walkers (no cloning)

**Interpretation:** The Keystone Lemma directly implies contraction for stably alive walkers.

**Related Results:**
- Direct application of: Keystone Lemma ({prf:ref}`lem-quantitative-keystone`)
- Foundation for: Complete positional contraction ({prf:ref}`thm-positional-variance-contraction`)
- Key step: Links Keystone to drift inequality

---

### Bounded Velocity Variance Expansion from Cloning

**Type:** Theorem
**Label:** `thm-bounded-velocity-expansion-cloning`
**Source:** [03_cloning.md § 10.4.1](03_cloning.md#1041-velocity-expansion)
**Tags:** `velocity-expansion`, `bounded-growth`, `state-independent`

**Statement:**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

where $C_v$ is a **state-independent** constant bounded by:

$$
C_v = f_{\text{clone}} \cdot 8(\alpha_{\text{restitution}}^2 + 4) V_{\max}^2
$$

**Interpretation:** Velocity variance can increase under cloning (due to inelastic collision noise), but the expansion per step is uniformly bounded.

**Related Results:**
- Proven in: {prf:ref}`prop-bounded-velocity-expansion`
- Part of: Complete variance drift ({prf:ref}`thm-complete-variance-drift-cloning`)
- Compensated by: Langevin friction in kinetic operator

---

### Structural Error Contraction

**Type:** Corollary
**Label:** `cor-structural-error-contraction`
**Source:** [03_cloning.md § 10.5](03_cloning.md#105-structural-contraction)
**Tags:** `structural-contraction`, `wasserstein-component`

**Statement:**

Since $V_{\text{struct}} \le V_{\text{Var},x}$ (by {prf:ref}`lem-sx-implies-variance`), the structural error also contracts:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{struct}}] \leq -\kappa_x V_{\text{struct}} + C_x
$$

**Interpretation:** The cloning operator contracts the shape component of the Wasserstein distance.

**Related Results:**
- Consequence of: Positional variance contraction ({prf:ref}`thm-positional-variance-contraction`)
- Part of: Complete Wasserstein drift analysis
- Complements: Location error expansion

---

### Complete Variance Drift Characterization for Cloning

**Type:** Theorem
**Label:** `thm-complete-variance-drift-cloning`
**Source:** [03_cloning.md § 10.6](03_cloning.md#106-complete-variance-drift)
**Tags:** `complete-variance-drift`, `position-velocity`, `cloning-summary`

**Statement:**

The cloning operator induces:

**Positional Variance (Strong Contraction):**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] \leq -\kappa_x V_{\text{Var},x} + C_x
$$

**Velocity Variance (Bounded Expansion):**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

**Combined Hypocoercive Variance:**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var}}] \leq -\kappa_x V_{\text{Var},x} + (C_x + C_v)
$$

where $V_{\text{Var}} = V_{\text{Var},x} + \lambda_v V_{\text{Var},v}$.

**Related Results:**
- Combines: {prf:ref}`thm-positional-variance-contraction` and {prf:ref}`thm-bounded-velocity-expansion-cloning`
- Part of: Complete Lyapunov drift ({prf:ref}`thm-complete-cloning-drift`)
- **Summary of Chapter 10**

---

### Complete Wasserstein Decomposition Drift

**Type:** Theorem
**Label:** `thm-complete-wasserstein-drift`
**Source:** [03_cloning.md § 11.1.3](03_cloning.md)
**Tags:** `wasserstein-drift`, `location-error`, `structural-error`, `decomposition`

**Statement:**

The total inter-swarm Wasserstein distance $V_W = V_{\text{loc}} + V_{\text{struct}}$ satisfies a combined drift inequality under the cloning operator:

$$
\mathbb{E}_{\text{clone}}[\Delta V_W] \leq C_W
$$

where $C_W < \infty$ is a state-independent constant satisfying:

$$
C_W = C_{\text{loc}} + C_{\text{struct}}
$$

**Component Bounds:**

From the decomposition $V_W^2 = V_{\text{loc}}^2 + V_{\text{struct}}^2$:

**Location Error (Center-of-Mass):**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{loc}}] \leq C_{\text{loc}}
$$

where $C_{\text{loc}} = O(\delta^2)$ from cloning jitter.

**Structural Error (Internal Geometry):**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{struct}}] \leq C_{\text{struct}}
$$

where $C_{\text{struct}} = O(\delta^2)$ from positional jitter.

**N-Uniformity:** All constants are independent of swarm size $N$.

**Physical Interpretation:**
- Cloning alone provides **bounded drift** (no contraction) for Wasserstein distance
- Contraction requires kinetic operator to reduce both location and structural errors
- The decomposition allows separate analysis of center-of-mass vs internal geometry dynamics

**Related Results:**
- Combines: Location error drift ({prf:ref}`thm-location-drift`) and structural error drift ({prf:ref}`thm-structural-drift`)
- Requires: Kinetic operator for contraction via {prf:ref}`thm-w2-kinetic-contraction`
- Part of: Synergistic Lyapunov function ({prf:ref}`def-full-synergistic-lyapunov-function`)
- Complements: Variance drift ({prf:ref}`thm-complete-variance-drift-cloning`)

---

### Boundary Potential Component (Recall)

**Type:** Definition
**Label:** `def-boundary-potential-component`
**Source:** [03_cloning.md § 11.2.1](03_cloning.md#1121-boundary-potential-recall)
**Tags:** `boundary-potential`, `barrier-function`, `confinement`

**Statement:**

$$
W_b(S_1, S_2) := \frac{1}{N} \sum_{i \in \mathcal{A}(S_1)} \varphi_{\text{barrier}}(x_{1,i}) + \frac{1}{N} \sum_{i \in \mathcal{A}(S_2)} \varphi_{\text{barrier}}(x_{2,i})
$$

where $\varphi_{\text{barrier}}: \mathcal{X}_{\text{valid}} \to \mathbb{R}_+$ is the smooth barrier function with:

$$
\varphi_{\text{barrier}}(x) \to \infty \quad \text{as} \quad x \to \partial \mathcal{X}_{\text{valid}}
$$

**Interpretation:** Measures average proximity to the boundary across both swarms.

**Related Results:**
- Construction: {prf:ref}`prop-barrier-existence`
- Contraction: {prf:ref}`thm-boundary-potential-contraction`
- Part of: Synergistic Lyapunov function ({prf:ref}`def-full-synergistic-lyapunov-function`)

---

### Fitness Gradient from Boundary Proximity

**Type:** Lemma
**Label:** `lem-fitness-gradient-boundary`
**Source:** [03_cloning.md § 11.2.2](03_cloning.md#1122-fitness-boundary-gradient)
**Tags:** `fitness-gradient`, `boundary-penalty`, `safe-harbor-mechanism`

**Statement:**

For walker $i$ at position $x_i$ with $\varphi_{\text{barrier}}(x_i) > \varphi_{\text{thresh}}$ (near boundary), its fitness is systematically lower than the swarm average:

$$
V_{k,i} < \mu_{V,k} - \Delta_{\text{barrier}}(\varphi(x_i))
$$

where $\Delta_{\text{barrier}}(\varphi) > 0$ is an **increasing function** of the barrier value.

**Mechanism:** Walkers near the boundary have:
1. Lower positional reward (Safe Harbor axiom)
2. Higher velocity penalty (if bouncing off walls)

**Related Results:**
- Requires: Safe Harbor axiom ({prf:ref}`ax:safe-harbor`)
- Foundation for: Boundary-exposed set analysis ({prf:ref}`def-boundary-exposed-set`)
- Enables: Enhanced cloning near boundary ({prf:ref}`lem-enhanced-cloning-near-boundary`)

---

### The Boundary-Exposed Set

**Type:** Definition
**Label:** `def-boundary-exposed-set`
**Source:** [03_cloning.md § 11.2.3](03_cloning.md#1123-boundary-exposed-set)
**Tags:** `boundary-exposed`, `high-barrier`, `near-boundary`

**Statement:**

$$
B_k(\varphi_{\text{thresh}}) := \{i \in \mathcal{A}(S_k) : \varphi_{\text{barrier}}(x_{k,i}) > \varphi_{\text{thresh}}\}
$$

**Interpretation:** Walkers with high barrier values (close to the boundary).

**Related Results:**
- Elements have: Low fitness ({prf:ref}`lem-fitness-gradient-boundary`)
- Experience: Enhanced cloning pressure ({prf:ref}`lem-enhanced-cloning-near-boundary`)
- Targeted by: Safe Harbor mechanism

---

### Boundary Potential Contraction Under Cloning

**Type:** Theorem
**Label:** `thm-boundary-potential-contraction`
**Source:** [03_cloning.md § 11.3](03_cloning.md#113-boundary-contraction-statement)
**Tags:** `boundary-contraction`, `foster-lyapunov`, `N-uniform`, `safe-harbor`

**Statement:**

There exist **N-uniform** constants $\kappa_b > 0$ and $C_b < \infty$ such that:

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

**Interpretation:** The cloning operator provides strong, systematic contraction of boundary exposure via the Safe Harbor mechanism.

**Related Results:**
- **Main result of Chapter 11**
- Requires: Safe Harbor axiom ({prf:ref}`ax:safe-harbor`)
- Part of: Complete Lyapunov drift ({prf:ref}`thm-complete-cloning-drift`)

---

### Enhanced Cloning Probability Near Boundary

**Type:** Lemma
**Label:** `lem-enhanced-cloning-near-boundary`
**Source:** [03_cloning.md § 11.4.1](03_cloning.md#1141-enhanced-cloning-pressure)
**Tags:** `boundary-cloning`, `enhanced-pressure`, `fitness-gradient`

**Statement:**

For walker $i \in B_k(\varphi_{\text{thresh}})$ (boundary-exposed):

$$
p_{k,i} \ge p_b(\varphi(x_i)) > 0
$$

where $p_b(\varphi)$ is an **increasing function** of the barrier value, bounded below by:

$$
p_b(\varphi) \ge p_u(\epsilon)
$$

the unfit cloning probability.

**Interpretation:** Walkers near the boundary experience systematically higher cloning pressure due to their low fitness.

**Related Results:**
- Uses: Fitness gradient ({prf:ref}`lem-fitness-gradient-boundary`)
- Foundation for: Boundary potential contraction ({prf:ref}`thm-boundary-potential-contraction`)
- Mechanism: Safe Harbor-induced selection pressure

---

### Expected Barrier Reduction for Cloned Walker

**Type:** Lemma
**Label:** `lem-expected-barrier-reduction-cloning`
**Source:** [03_cloning.md § 11.4.2](03_cloning.md#1142-barrier-reduction)
**Tags:** `barrier-reduction`, `cloning-destination`, `safe-harbor-pull`

**Statement:**

When walker $i \in B_k$ (boundary-exposed) clones, its expected post-cloning barrier value satisfies:

$$
\mathbb{E}[\varphi(x'_i) \mid \text{clone}] \le \varphi_{\text{safe}} + C_{\text{jitter}}
$$

where:
- $\varphi_{\text{safe}} < \varphi_{\text{thresh}}$ is the barrier value in the Safe Harbor region
- $C_{\text{jitter}}$ accounts for position jitter ($\sigma_x$)

**Mechanism:** Cloning moves the walker to a companion's location, which is biased toward the Safe Harbor due to higher fitness there.

**Related Results:**
- Uses: Companion selection bias toward high-fitness regions
- Foundation for: Boundary contraction ({prf:ref}`thm-boundary-potential-contraction`)
- Key property: Cloning as corrective mechanism

---

### Bounded Boundary Exposure in Equilibrium

**Type:** Corollary
**Label:** `cor-bounded-boundary-exposure-equilibrium`
**Source:** [03_cloning.md § 11.5.1](03_cloning.md#1151-equilibrium-bound)
**Tags:** `equilibrium`, `stationary-bound`, `qsd`

**Statement:**

In the quasi-stationary regime (when drift balances noise):

$$
\mathbb{E}[W_b] \le \frac{C_b}{\kappa_b}
$$

**Interpretation:** Boundary exposure is uniformly bounded in the QSD.

**Proof:** Set $\mathbb{E}[\Delta W_b] = 0$ in the drift inequality.

**Related Results:**
- Consequence of: Boundary contraction ({prf:ref}`thm-boundary-potential-contraction`)
- Implies: Bounded mean time to extinction
- Part of: QSD characterization

---

### Exponentially Suppressed Extinction Probability

**Type:** Corollary
**Label:** `cor-exponentially-suppressed-extinction`
**Source:** [03_cloning.md § 11.5.2](03_cloning.md#1152-extinction-suppression)
**Tags:** `extinction`, `qsd`, `exponential-suppression`

**Statement:**

The probability of total swarm extinction in a single step is:

$$
P(\text{extinction}) \le \exp\left(-c_{\text{extinct}} \cdot \frac{C_b}{\kappa_b}\right)
$$

where $c_{\text{extinct}} > 0$ depends on the barrier growth rate near $\partial \mathcal{X}_{\text{valid}}$.

**Interpretation:** In the QSD regime, extinction probability is exponentially small.

**Related Results:**
- Consequence of: Bounded boundary exposure ({prf:ref}`cor-bounded-boundary-exposure-equilibrium`)
- Justifies: QSD as meaningful long-term description
- Key practical result: Enables long simulation runs

---

### Complete Boundary Potential Drift Characterization

**Type:** Theorem
**Label:** `thm-complete-boundary-drift-cloning`
**Source:** [03_cloning.md § 11.6](03_cloning.md#116-complete-boundary-drift)
**Tags:** `boundary-drift`, `explicit-constants`, `N-uniform`

**Statement:**

$$
\mathbb{E}_{\text{clone}}[\Delta W_b] \leq -\kappa_b W_b + C_b
$$

with explicit bounds:

**Contraction Rate:**

$$
\kappa_b = \frac{p_b(\varphi_{\text{thresh}}) (\varphi_{\text{thresh}} - \varphi_{\text{safe}})}{2\varphi_{\max}}
$$

**Expansion Constant:**

$$
C_b = \frac{N \cdot C_{\text{jitter}}}{\varphi_{\max}}
$$

where $\varphi_{\max}$ is a large upper bound on barrier values in the relevant domain.

**Related Results:**
- **Summary of Chapter 11**
- Part of: Complete Lyapunov drift ({prf:ref}`thm-complete-cloning-drift`)
- Provides: Explicit parameter dependencies

---

### Bounded Expansion of Inter-Swarm Wasserstein Distance

**Type:** Theorem
**Label:** `thm-bounded-expansion-inter-swarm-error`
**Source:** [03_cloning.md § 12.2.1](03_cloning.md#1221-wasserstein-expansion)
**Tags:** `wasserstein-expansion`, `inter-swarm-error`, `bounded-growth`

**Statement:**

$$
\mathbb{E}_{\text{clone}}[\Delta W_h^2] \leq C_W
$$

where $C_W < \infty$ is **state-independent**.

**Decomposition:**

$$
\begin{aligned}
\mathbb{E}_{\text{clone}}[\Delta V_{\text{loc}}] &\leq C_{\text{loc}} \\
\mathbb{E}_{\text{clone}}[\Delta V_{\text{struct}}] &\leq C_{\text{struct}}
\end{aligned}
$$

**Interpretation:** The cloning operator allows inter-swarm error to expand, but expansion is bounded (not explosive). This expansion will be compensated by the kinetic operator.

**Related Results:**
- Complements: Variance contraction ({prf:ref}`thm-complete-variance-drift-cloning`)
- Requires: Kinetic operator to contract $W_h^2$
- Part of: Synergistic drift analysis ({prf:ref}`thm-complete-cloning-drift`)

---

### Complete Drift Inequality for the Cloning Operator

**Type:** Theorem
**Label:** `thm-complete-cloning-drift`
**Source:** [03_cloning.md § 12.3.1](03_cloning.md#1231-complete-cloning-drift)
**Tags:** `complete-drift`, `synergistic-lyapunov`, `cloning-summary`, `N-uniform`

**Statement:**

For the synergistic Lyapunov function:

$$
V_{\text{total}}(S_1, S_2) = V_W(S_1, S_2) + c_V V_{\text{Var}}(S_1, S_2) + c_B W_b(S_1, S_2)
$$

**Individual Component Drifts:**

$$
\begin{aligned}
\mathbb{E}_{\text{clone}}[\Delta V_W] &\leq C_W \quad \text{(bounded expansion)} \\
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] &\leq -\kappa_x V_{\text{Var},x} + C_x \quad \text{(strong contraction)} \\
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] &\leq C_v \quad \text{(bounded expansion)} \\
\mathbb{E}_{\text{clone}}[\Delta W_b] &\leq -\kappa_b W_b + C_b \quad \text{(strong contraction)}
\end{aligned}
$$

**Combined Drift:**

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{total}}] \leq C_W + c_V(-\kappa_x V_{\text{Var},x} + C_v + C_x) + c_B(-\kappa_b W_b + C_b)
$$

**Partial Contraction:** When $c_V V_{\text{Var},x} + c_B W_b$ is sufficiently large:

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{total}}] < 0
$$

**Related Results:**
- **Main result of Chapter 12 and the entire cloning analysis**
- Requires synergy with: Kinetic operator to contract $V_W$ and $V_{\text{Var},v}$
- Preview of: {prf:ref}`thm-synergistic-foster-lyapunov-preview`

---

### Necessity of the Kinetic Operator

**Type:** Proposition
**Label:** `prop-kinetic-necessity`
**Source:** [03_cloning.md § 12.3.3](03_cloning.md#1233-kinetic-necessity)
**Tags:** `kinetic-necessity`, `cloning-limitations`, `complementary-operators`

**Statement:**

The cloning operator alone cannot guarantee convergence because:

1. **Velocity variance accumulation:** $+C_v$ per step accumulates without bound
2. **Inter-swarm divergence:** $+C_W$ allows unbounded separation
3. **No velocity equilibrium:** Cloning only redistributes kinetic energy

**Therefore, the kinetic operator is essential to:**
- Contract $V_{\text{Var},v}$ via Langevin friction
- Contract $V_W$ via hypocoercive drift
- Establish velocity equilibrium with Gibbs distribution

**Related Results:**
- Motivates: Composition $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$
- Foundation for: Synergistic drift analysis
- Key architectural insight: Complementary contraction mechanisms

---

### Synergistic Foster-Lyapunov Condition (Preview)

**Type:** Theorem
**Label:** `thm-synergistic-foster-lyapunov-preview`
**Source:** [03_cloning.md § 12.4.2](03_cloning.md#1242-synergistic-preview)
**Tags:** `synergistic-drift`, `foster-lyapunov`, `complete-convergence`, `qsd`

**Statement:**

When coupling constants $c_V, c_B$ are chosen appropriately, the composed operator:

$$
\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}
$$

satisfies the **Foster-Lyapunov condition**:

$$
\mathbb{E}_{\text{total}}[V_{\text{total}}(S')] \leq (1 - \kappa_{\text{total}}) V_{\text{total}}(S) + C_{\text{total}}
$$

for **N-uniform** constants $\kappa_{\text{total}} > 0$ and $C_{\text{total}} < \infty$.

**Consequences:**

1. **Geometric ergodicity** of the Markov chain
2. **Exponential convergence** to quasi-stationary distribution (QSD)
3. **Exponentially suppressed extinction probability** in QSD regime

**Related Results:**
- Combines: Cloning drift ({prf:ref}`thm-complete-cloning-drift`) and kinetic drift (deferred to companion document)
- **Ultimate goal** of the convergence analysis
- Proven in detail: Companion document on hypocoercivity

---

## Complete Convergence to QSD

### Quasi-Stationary Distribution (QSD)

**Type:** Definition
**Label:** `def-qsd`
**Source:** [04_convergence.md § 7.3](04_convergence.md)
**Tags:** `qsd`, `absorption`, `conditional-stationarity`

**Statement:**

A measure $\nu_{\text{QSD}}$ on $\Sigma_N^{\text{alive}}$ is a QSD if:

$$
P(S_{t+1} \in A \mid S_t \sim \nu_{\text{QSD}}, \text{not absorbed}) = \nu_{\text{QSD}}(A)
$$

**Interpretation:** Stationary distribution conditioned on survival.

**Related Results:**
- Exists: Via Foster-Lyapunov + irreducibility
- Unique: For Euclidean Gas
- Absorption inevitable: Unbounded Gaussian noise

---

### Geometric Ergodicity and Convergence to QSD

**Type:** Theorem (Main Convergence Theorem)
**Label:** `thm-main-convergence`
**Source:** [04_convergence.md § 7.5](04_convergence.md)
**Tags:** `geometric-ergodicity`, `qsd-convergence`, `exponential-convergence`, `main-result`

**Four Main Results:**

**1. Existence and Uniqueness:**
Unique QSD $\nu_{\text{QSD}}$ exists.

**2. Exponential Convergence:**

$$
\|\mu_t - \nu_{\text{QSD}}\|_{\text{TV}} \leq C_{\text{conv}} e^{-\kappa_{\text{QSD}} t}
$$

where $\kappa_{\text{QSD}} = \Theta(\kappa_{\text{total}}\tau) > 0$.

**3. Exponentially Long Survival:**

$$
\mathbb{E}_{\nu_{\text{QSD}}}[\tau_{\dagger}] = e^{\Theta(N)}
$$

Mean time to extinction exponential in $N$.

**4. Concentration:**

$$
P(V_{\text{total}}(S_t) > (1+\epsilon) V_{\text{total}}^{\text{QSD}}) \leq e^{-\Theta(N)}
$$

**Related Results:**
- **Ultimate convergence guarantee**
- Requires: Foster-Lyapunov + irreducibility + aperiodicity
- Practical implication: QSD is meaningful for all relevant timescales

---

### φ-Irreducibility

**Type:** Theorem
**Label:** `thm-phi-irreducibility`
**Source:** [04_convergence.md § 7.4.1](04_convergence.md)
**Tags:** `irreducibility`, `reachability`, `hormander`

**Statement:**

For any $S_A \in \Sigma_N^{\text{alive}}$ and open set $O_B \subseteq \Sigma_N^{\text{alive}}$, $\exists M \in \mathbb{N}$ such that:

$$
P^M(S_A, O_B) > 0
$$

**Proof Method:** Two-stage construction:
1. **Gathering via cloning:** Concentrate walkers
2. **Spreading via kinetics:** Hörmander's theorem for hypoellipticity

**Related Results:**
- Essential for: Unique QSD
- Uses: Uniform ellipticity + non-degenerate noise
- Key property: Full state space exploration

---

### QSD Properties

**Type:** Proposition
**Label:** `prop-qsd-properties`
**Source:** [04_convergence.md § 7.6](04_convergence.md)
**Tags:** `qsd-properties`, `equilibrium-distribution`, `gibbs-like`

**Four Key Properties:**

**1. Position Distribution:**

$$
\rho_{\text{pos}}(x) \propto e^{-U(x) - \varphi_{\text{barrier}}(x)}
$$

Gibbs with combined potential.

**2. Velocity Distribution:**

$$
\rho_{\text{vel}}(v) \propto e^{-\frac{\|v\|^2}{2\sigma_v^2/\gamma}}
$$

Maxwellian with temperature $\sigma_v^2/\gamma$.

**3. Decorrelation:**

$$
\mathbb{E}_{\nu_{\text{QSD}}}[\langle x - \bar{x}, v - \bar{v}\rangle] = O(e^{-\gamma \Delta t})
$$

Position-velocity correlations decay exponentially.

**4. Internal Variance:**

$$
V_{\text{Var},x}^{\text{QSD}} = O(C_x/\kappa_x), \quad V_{\text{Var},v}^{\text{QSD}} = O(\sigma_v^2/\gamma)
$$

**Related Results:**
- Equilibrium: Balances cloning + kinetics
- Realistic: Resembles physical systems
- Predictable: Explicit parameter dependence

---

### Equilibrium Variance Bounds from Drift Inequalities

**Type:** Theorem
**Label:** `thm-equilibrium-variance-bounds`
**Source:** [04_convergence.md § 8.4.1](04_convergence.md)
**Tags:** `equilibrium-bounds`, `qsd-variance`, `foster-lyapunov`, `parameter-explicit`

**Statement:**

The quasi-stationary distribution satisfies explicit variance bounds derived from component drift inequalities.

**Positional Variance Equilibrium:**

From {prf:ref}`thm-positional-variance-contraction` (cloning contraction) and kinetic drift, setting $\mathbb{E}[\Delta V_{\text{Var},x}] = 0$ yields:

$$
V_{\text{Var},x}^{\text{QSD}} \leq \frac{C_x}{\kappa_x}
$$

where:
- $C_x = O(\sigma_v^2 \tau^2 / (\gamma \lambda))$ is the expansion from kinetic operator
- $\kappa_x = \lambda$ is the cloning contraction rate

**Velocity Variance Equilibrium:**

From {prf:ref}`thm-velocity-variance-contraction` (kinetic contraction) and cloning expansion:

$$
V_{\text{Var},v}^{\text{QSD}} \leq \frac{C_v + \sigma_{\max}^2 d \tau}{2\gamma\tau}
$$

where:
- $C_v = O(\delta^2)$ is cloning jitter
- $\sigma_{\max}^2 d \tau$ is kinetic noise injection
- $2\gamma\tau$ is friction dissipation rate

**Wasserstein Distance Equilibrium:**

From {prf:ref}`thm-complete-wasserstein-drift` (cloning bounded drift) and {prf:ref}`thm-w2-kinetic-contraction`:

$$
V_W^{\text{QSD}} \leq \frac{C_W}{\kappa_W}
$$

where $\kappa_W = O(\gamma)$ is hypocoercive contraction rate.

**Boundary Potential Equilibrium:**

From {prf:ref}`thm-boundary-potential-contraction`:

$$
W_b^{\text{QSD}} \leq \frac{C_b}{\kappa_b}
$$

**Physical Interpretation:**
- Each equilibrium variance is determined by **balance between expansion and contraction**
- Setting $\mathbb{E}[\Delta V] = 0$ in drift inequality yields $V^{\text{QSD}} = C/\kappa$
- Positional variance: Cloning contracts, kinetics expands
- Velocity variance: Friction dissipates, noise injects
- Wasserstein: Hypocoercivity contracts, jitter expands
- Boundary: Fitness gradient repels, diffusion pushes

**Related Results:**
- Foundation: Foster-Lyapunov drift inequalities for each component
- Uses: {prf:ref}`thm-positional-variance-contraction`, {prf:ref}`thm-velocity-variance-contraction`
- Refines: {prf:ref}`prop-qsd-properties` with explicit parameter dependence
- Part of: Complete equilibrium characterization

---

### Synergistic Rate Derivation from Component Drifts

**Type:** Theorem
**Label:** `thm-synergistic-rate-derivation`
**Source:** [04_convergence.md § 8.5](04_convergence.md)
**Tags:** `synergistic-convergence`, `hypocoercivity`, `component-coupling`, `weight-balancing`

**Statement:**

The total drift inequality combines component-wise drift bounds from cloning and kinetic operators to yield explicit synergistic convergence.

**Component Drift Structure:**

From cloning and kinetic operators, each Lyapunov component satisfies:

$$
\begin{aligned}
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},x}] &\leq -\kappa_x V_{\text{Var},x} + C_x + C_{xv} V_{\text{Var},v} + C_{xW} V_W \\
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] &\leq -\kappa_v V_{\text{Var},v} + C_v + C_{vx} V_{\text{Var},x} \\
\mathbb{E}_{\text{clone}}[\Delta V_W] &\leq -\kappa_W V_W + C_W \\
\mathbb{E}_{\text{clone}}[\Delta W_b] &\leq -\kappa_b W_b + C_b
\end{aligned}
$$

where cross-component coupling terms $C_{xv}, C_{xW}, C_{vx}$ arise from expansion by the complementary operator.

**Weighted Combination:**

Define the weighted Lyapunov function:

$$
V_{\text{total}} = V_{\text{Var},x} + \alpha_v V_{\text{Var},v} + \alpha_W V_W + \alpha_b W_b
$$

**Weight Selection for Coupling Domination:**

Choose weights to ensure coupling terms are dominated by contraction:

$$
\alpha_v \geq \frac{C_{xv}}{\kappa_v V_{\text{Var},v}^{\text{eq}}}, \quad
\alpha_W \geq \frac{C_{xW}}{\kappa_W V_W^{\text{eq}}}, \quad
\alpha_v \kappa_v \geq C_{vx} / V_{\text{Var},x}^{\text{eq}}
$$

With these weights, coupling terms satisfy:

$$
C_{xv} V_{\text{Var},v} - \alpha_v \kappa_v V_{\text{Var},v} \leq -\epsilon_v \alpha_v \kappa_v V_{\text{Var},v}
$$

where $\epsilon_v, \epsilon_W \ll 1$ are small positive fractions.

**Synergistic Rate:**

After cancellation of dominated coupling terms:

$$
\mathbb{E}[\Delta V_{\text{total}}] \leq -\kappa_{\text{total}} V_{\text{total}} + C_{\text{total}}
$$

where:

$$
\kappa_{\text{total}} = \min(\kappa_x, \alpha_v \kappa_v, \alpha_W \kappa_W, \alpha_b \kappa_b) \cdot (1 - \epsilon_{\text{coupling}})
$$

$$
C_{\text{total}} = C_x + \alpha_v C_v + \alpha_W C_W + \alpha_b C_b
$$

and $\epsilon_{\text{coupling}} = \max(\epsilon_v, \epsilon_W, \ldots)$ is the residual coupling ratio.

**Physical Interpretation:**
- **Bottleneck principle**: The weakest contraction rate dominates (min over components)
- **Coupling penalty**: $\epsilon_{\text{coupling}}$ reduces effective rate due to energy transfer between components
- **Weight balancing**: Optimal $\alpha_i$ maximize $\alpha_i \kappa_i$ subject to coupling domination
- **Hypocoercivity**: No single component contracts alone, but weighted combination does

**Related Results:**
- Foundation: Component drift inequalities {prf:ref}`thm-positional-variance-contraction`, {prf:ref}`thm-velocity-variance-contraction`
- Leads to: {prf:ref}`thm-total-rate-explicit` with parameter-explicit formulas
- Uses: Foster-Lyapunov weight balancing technique
- Part of: Complete convergence theory

---

### Mixing Time

**Type:** Proposition
**Label:** `prop-mixing-time-explicit`
**Source:** [04_convergence.md § 8.6](04_convergence.md)
**Tags:** `mixing-time`, `convergence-rate`, `practical`

**Statement:**

$$
T_{\text{mix}}(\epsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{total}}^{\text{init}}}{\epsilon C_{\text{total}}}\right)
$$

**Typical Case:** $T_{\text{mix}} \sim \frac{5}{\kappa_{\text{total}}}$ for $\epsilon = 0.01$.

**Scaling:** $\kappa_{\text{total}} \sim \min(\lambda, 2\gamma, \kappa_W, \kappa_b)$

**Related Results:**
- Practical bound: Number of steps to reach QSD
- Parameter-dependent: Can be optimized
- N-uniform: Mixing time independent of swarm size

---

## Mean-Field Limit and McKean-Vlasov PDE

### Phase-Space Density

**Type:** Definition
**Label:** `def-phase-space-density`
**Source:** [05_mean_field.md § 1.1](05_mean_field.md)
**Tags:** `mean-field`, `phase-space`, `probability-density`, `continuum-limit`

**Statement:**

The **phase-space sub-probability density** $f: [0, \infty) \times \Omega \to [0, \infty)$ describes the alive population:

$$
m_a(t) := \int_{\Omega} f(t,x,v)\,\mathrm{d}x\,\mathrm{d}v \le 1
$$

Dead mass: $m_d(t) = 1 - m_a(t)$

**Regularity:** $f \in C([0, \infty); L^1(\Omega))$

**Related Results:**
- Evolves via: McKean-Vlasov PDE
- Conservation: $m_a(t) + m_d(t) = 1$
- Limit of: N-particle empirical measures

---

### Mean-Field Fitness Potential

**Type:** Definition
**Label:** `def-mean-field-fitness-potential`
**Source:** [05_mean_field.md § 1.3](05_mean_field.md)
**Tags:** `fitness`, `non-local`, `non-linear`, `self-consistent`

**Statement:**

$$
V[f](z,z_c,t) := \left(g_A(\widetilde{d}[f](z,z_c,t)) + \eta\right)^{\beta} \cdot \left(g_A(\widetilde{r}[f](z,t)) + \eta\right)^{\alpha}
$$

where:
- $\widetilde{r}[f](z,t) = \frac{R(z) - \mu_R[f](t)}{\widehat{\sigma}_R[f](t)}$ (reward Z-score)
- $\widetilde{d}[f](z,z_c,t) = \frac{d_{\mathcal{Y}}(\varphi(z),\varphi(z_c)) - \mu_D[f](t)}{\widehat{\sigma}_D[f](t)}$ (distance Z-score)

**Key Properties:**
- **Non-local:** Depends on global moments $\mu_R[f], \sigma_R[f], \mu_D[f], \sigma_D[f]$
- **Non-linear:** Squared in variance functionals
- **Self-consistent:** Fitness depends on entire population state

**Related Results:**
- Drives: Cloning operator $S[f]$
- Analogous to: Discrete fitness potential $V_{\text{fit},i}$

---

### The Mean-Field Equations

**Type:** Theorem
**Label:** `thm-mean-field-equation`
**Source:** [05_mean_field.md § 3.3](05_mean_field.md)
**Tags:** `mckean-vlasov`, `pde`, `coupled-system`, `mass-conservation`

**Statement:**

$$
\boxed{\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]}
$$

$$
\boxed{\frac{\mathrm{d}}{\mathrm{d}t} m_d(t) = \int_{\Omega} c(z)f(t,z)\,\mathrm{d}z - \lambda_{\text{rev}} m_d(t)}
$$

**Operators:**
- $L^\dagger$: Kinetic transport (Fokker-Planck)
- $c(z)$: Killing rate near boundary
- $B[f, m_d]$: Revival operator (non-local source)
- $S[f]$: Cloning operator (mass-neutral redistribution)

**Conservation:** $m_a(t) + m_d(t) = 1$ for all $t$

**Related Results:**
- Limit of: N-particle dynamics as $N \to \infty$
- Stationary: $\rho_0$ solves $0 = L^\dagger \rho_0 - c\rho_0 + S[\rho_0] + B[\rho_0, m_{d,\infty}]$

---

### Killing Rate Consistency

**Type:** Theorem
**Label:** `thm-killing-rate-consistency`
**Source:** [05_mean_field.md § 4.4.2](05_mean_field.md)
**Tags:** `killing-rate`, `boundary`, `consistency`, `ballistic-limit`

**Statement:**

**Pointwise convergence:**

$$
\lim_{\tau \to 0} \frac{1}{\tau} p_{\text{exit}}(x,v,\tau) = c(x,v)
$$

where:

$$
c(x,v) = \begin{cases}\frac{(v \cdot n_x(x))^+}{d(x)} \cdot \mathbf{1}_{d(x) < \delta} & \text{if } x \in \mathcal{T}_\delta \\ 0 & \text{otherwise}\end{cases}
$$

**Uniform error bound:**

$$
\left|\frac{1}{\tau} K_{\text{discrete}}(\tau) - K_{\text{continuous}}\right| \le C \left(\sqrt{\tau} + \|f^\tau - f\|_{L^1}\right)
$$

**Related Results:**
- Justifies: Interior killing rate approximation
- Ballistic regime: Drift dominates diffusion near boundary
- Foundation for: Rigorous $N \to \infty$ limit

---

## Propagation of Chaos and Thermodynamic Limit

### Tightness of QSD Marginals

**Type:** Theorem
**Label:** `thm-qsd-marginals-are-tight`
**Source:** [06_propagation_chaos.md § 2](06_propagation_chaos.md)
**Tags:** `tightness`, `compactness`, `prokhorov`, `marginals`

**Statement:**

The sequence $\{\mu_N\}$ of single-particle marginals from N-particle QSDs is **tight** in $\mathcal{P}(\Omega)$.

**Proof Method:**
1. N-uniform Foster-Lyapunov: $\mathbb{E}_{\nu_N^{QSD}}[V_{\text{total}}] \le C$
2. Single-particle bound: $\mathbb{E}_{\mu_N}[\|x\|^2 + \|v\|^2] \le C'$
3. Prokhorov: Tight sequences have convergent subsequences

**Related Results:**
- Enables: Subsequence selection with $\mu_{N_k} \rightharpoonup \mu_\infty$
- Uses: N-uniform moment bounds from 04_convergence.md
- Foundation for: Propagation of chaos

---

### Extinction Rate Vanishes

**Type:** Theorem
**Label:** `thm-extinction-rate-vanishes`
**Source:** [06_propagation_chaos.md § 3, Part C.5](06_propagation_chaos.md)
**Tags:** `extinction`, `large-deviations`, `N-scaling`

**Statement:**

$$
\lim_{N \to \infty} \lambda_N = 0
$$

where $\lambda_N$ is the extinction rate of the N-particle QSD.

**Key Bounds:**
- Expected hitting time: $\mathbb{E}_{\nu_N^{QSD}}[\tau_{\text{ext}}] \geq C e^{\beta N}$
- Large deviation: $\mathbb{P}_{\nu_N^{QSD}}(\tau_{\text{ext}} \leq T) \leq e^{-c N}$

**Interpretation:** In the limit $N \to \infty$, the system becomes truly stationary (no absorption).

**Related Results:**
- Uses: N-uniform Foster-Lyapunov + Cramér's theorem
- Consequence: Limit point satisfies true stationarity (not QSD)
- Critical for: Mean-field PDE stationarity

---

### Limit Points are Weak Solutions

**Type:** Theorem
**Label:** `thm-limit-is-weak-solution`
**Source:** [06_propagation_chaos.md § 3, Part C](06_propagation_chaos.md)
**Tags:** `weak-solution`, `stationary-pde`, `limit-point`

**Statement:**

Any limit point $\mu_\infty$ of convergent subsequence $\mu_{N_k} \rightharpoonup \mu_\infty$ satisfies:

$$
\int_\Omega \left(L^\dagger \rho_0(z) - c(z)\rho_0(z) + S[\rho_0](z) + B[\rho_0, m_{d,\infty}](z)\right) \phi(z) \, dz = 0
$$

for all $\phi \in C_c^\infty(\Omega)$, with equilibrium $k_{\text{killed}}[\rho_0] = \lambda_{\text{rev}} m_{d,\infty}$.

**Proof Method:**
- Take limit of N-particle stationarity
- Term-by-term convergence of functionals
- Extinction rate vanishes

**Related Results:**
- Proves: Existence of weak solutions
- Uniqueness: Via contraction mapping
- Complete: With tightness theorem

---

### Uniqueness via Contraction Mapping

**Type:** Theorem
**Label:** `thm-uniqueness-contraction-solution-operator`
**Source:** [06_propagation_chaos.md § 4, Part D](06_propagation_chaos.md)
**Tags:** `uniqueness`, `contraction`, `banach-fixed-point`, `hypoelliptic`

**Statement:**

For sufficiently large $\sigma_v^2$, the solution operator $\mathcal{T}: \mathcal{P}_R \to \mathcal{P}_R$ is a **strict contraction**:

$$
\|\mathcal{T}[\rho_1] - \mathcal{T}[\rho_2]\|_{H^1_w} \le \kappa(R^*) \|\rho_1 - \rho_2\|_{H^1_w}
$$

with:

$$
\kappa(R^*) = \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\sigma_v^2 \gamma} < 1
$$

**Sufficient Condition:**

$$
\sigma_v^2 > \frac{(C_S + C_B)(1 + R^*) + L_c + C}{\gamma}
$$

**Related Results:**
- Uses: Hypoelliptic regularity (Hörmander)
- Scaling: $C_{\text{hypo}} \sim (\sigma_v^2 \gamma)^{-1}$
- Conclusion: Banach Fixed-Point Theorem → unique solution

---

### Thermodynamic Limit

**Type:** Theorem
**Label:** `thm-thermodynamic-limit`
**Source:** [06_propagation_chaos.md § 5.6](06_propagation_chaos.md)
**Tags:** `thermodynamic-limit`, `law-of-large-numbers`, `macroscopic-observables`

**Statement:**

For any bounded, continuous $\phi: \Omega \to \mathbb{R}$:

$$
\lim_{N \to \infty} \mathbb{E}_{\nu_N^{QSD}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] = \int_\Omega \phi(z) \rho_0(z) dz
$$

**Interpretation:** Macroscopic observables (empirical averages) converge to mean-field expectations.

**Proof Method:** Exchangeability + weak convergence of marginals

**Related Results:**
- **Complete propagation of chaos**
- Combines: Tightness + weak solution + uniqueness
- Enables: Law of large numbers for swarm statistics

---

## Adaptive Viscous Fluid Model

This section contains mathematical results from the Adaptive Viscous Fluid Model with ρ-localized fitness potential, which extends the backbone Euclidean Gas with three adaptive mechanisms: adaptive force from mean-field potential, viscous coupling between walkers, and regularized Hessian diffusion.

### Localization Kernel

**Type:** Definition
**Label:** `def-localization-kernel`
**Source:** [07_adaptative_gas.md § 1.0.2](07_adaptative_gas.md)
**Tags:** `localization`, `kernel`, `ρ-parameterization`, `measurement-pipeline`

**Statement:**
For localization scale ρ > 0, the localization kernel $K_\rho: \mathcal{X} \times \mathcal{X} \to [0, 1]$ satisfies:
1. Normalization: $\int_{\mathcal{X}} K_\rho(x, x') dx' = 1$
2. Locality: $K_\rho(x, x') \to 0$ rapidly as $\|x - x'\| \gg \rho$
3. Symmetry: $K_\rho(x, x') = K_\rho(x', x)$
4. Limit behavior: As ρ → 0: $K_\rho(x, x') \to \delta(x - x')$; As ρ → ∞: $K_\rho(x, x') \to 1/|\mathcal{X}|$

Standard example (Gaussian): $K_\rho(x, x') := \frac{1}{Z_\rho(x)} \exp\left(-\frac{\|x - x'\|^2}{2\rho^2}\right)$

**Related Results:** `def-localized-mean-field-moments`, `def-unified-z-score`

---

### Localized Mean-Field Moments

**Type:** Definition
**Label:** `def-localized-mean-field-moments`
**Source:** [07_adaptative_gas.md § 1.0.3](07_adaptative_gas.md)
**Tags:** `statistics`, `mean-field`, `ρ-localization`, `alive-walkers`, `N-uniform`

**Statement:**
For distribution $f \in \mathcal{P}(\mathcal{X} \times \mathbb{R}^d)$, measurement $d: \mathcal{X} \to \mathbb{R}$, reference point $x \in \mathcal{X}$:

Localized mean: $\mu_\rho[f, d, x] := \int_{\mathcal{X} \times \mathbb{R}^d} K_\rho(x, x') d(x') f(x', v) dx' dv$

Localized variance: $\sigma^2_\rho[f, d, x] := \int_{\mathcal{X} \times \mathbb{R}^d} K_\rho(x, x') [d(x') - \mu_\rho[f, d, x]]^2 f(x', v) dx' dv$

N-particle form (alive walkers): $\mu_\rho[f_k, d, x_i] = \sum_{j \in A_k} w_{ij}(\rho) d(x_j)$ where $w_{ij}(\rho) := \frac{K_\rho(x_i, x_j)}{\sum_{\ell \in A_k} K_\rho(x_i, x_\ell)}$

**Related Results:** `def-unified-z-score`, `prop-limiting-regimes`

---

### Unified Localized Z-Score

**Type:** Definition
**Label:** `def-unified-z-score`
**Source:** [07_adaptative_gas.md § 1.0.4](07_adaptative_gas.md)
**Tags:** `Z-score`, `regularization`, `C¹-smoothing`, `numerical-stability`

**Statement:**
The unified ρ-dependent Z-score:

$$Z_\rho[f, d, x] := \frac{d(x) - \mu_\rho[f, d, x]}{\sigma'_\rho[f, d, x]}$$

where $\sigma'_\rho[f, d, x] := \sigma'_{\text{patch}}(\sigma^2_\rho[f, d, x])$ is C¹-regularized standard deviation.

Properties: bounded when $d$ bounded, well-posed everywhere, localizes to ρ-neighborhood, recovers global Z-score as ρ → ∞.

**Related Results:** `def-localized-mean-field-fitness`

---

### The Adaptive Viscous Fluid SDE

**Type:** Definition
**Label:** `def-hybrid-sde`
**Source:** [07_adaptative_gas.md § 2](07_adaptative_gas.md)
**Tags:** `SDE`, `Stratonovich`, `adaptive-force`, `viscous-coupling`, `Hessian-diffusion`

**Statement:**
Each walker $i \in \{1, \dots, N\}$ evolves according to Stratonovich SDE:

$$\begin{aligned}
dx_i &= v_i dt \\
dv_i &= \left[ \mathbf{F}_{\text{stable}}(x_i) + \mathbf{F}_{\text{adapt}}(x_i, S) + \mathbf{F}_{\text{viscous}}(x_i, S) - \gamma v_i \right] dt + \Sigma_{\text{reg}}(x_i, S) \circ dW_i
\end{aligned}$$

Force components:
1. Stability: $\mathbf{F}_{\text{stable}}(x_i) := -\nabla U(x_i)$
2. Adaptive: $\mathbf{F}_{\text{adapt}}(x_i, S) := \epsilon_F \nabla_{x_i} V_{\text{fit}}[f_k, \rho](x_i)$
3. Viscous: $\mathbf{F}_{\text{viscous}}(x_i, S) := \nu \sum_{j \neq i} K(x_i - x_j)(v_j - v_i)$
4. Friction: $-\gamma v_i$ with $\gamma > 0$
5. Regularized diffusion: $\Sigma_{\text{reg}}(x_i, S) := (\nabla^2_{x_i} V_{\text{fit}}[f_k, \rho](x_i) + \epsilon_\Sigma I)^{-1/2}$

**Related Results:** `def-localized-mean-field-fitness`, `def-regularized-hessian-tensor`

---

### Localized Mean-Field Fitness Potential

**Type:** Definition
**Label:** `def-localized-mean-field-fitness`
**Source:** [07_adaptative_gas.md § 2.1](07_adaptative_gas.md)
**Tags:** `fitness-potential`, `mean-field`, `ρ-dependence`, `nonlocal`, `nonlinear`

**Statement:**
The ρ-localized fitness potential:

$$V_{\text{fit}}[f, \rho](x) := g_A(Z_\rho[f, d, x])$$

where $g_A: \mathbb{R} \to [0, A]$ is smooth, bounded, monotone increasing rescale function, and $Z_\rho$ is unified localized Z-score.

Properties: ρ-dependent through localization kernel, nonlocal (depends on ρ-neighborhood), nonlinear functional in $f$, bounded $0 \le V_{\text{fit}} \le A$, $C^\infty$ in $x$ when $f$ sufficiently regular.

**Related Results:** `thm-c1-regularity`, `thm-c2-regularity`

---

### Regularized Hessian Diffusion Tensor

**Type:** Definition
**Label:** `def-regularized-hessian-tensor`
**Source:** [07_adaptative_gas.md § 2.1](07_adaptative_gas.md)
**Tags:** `diffusion`, `Hessian`, `regularization`, `information-geometry`, `uniform-ellipticity`

**Statement:**
The regularized adaptive diffusion tensor:

$$\Sigma_{\text{reg}}(x_i, S) := (H_i(S) + \epsilon_\Sigma I)^{-1/2}$$

where $H_i(S) = \nabla^2_{x_i} V_{\text{fit}}[f_k, \rho](x_i)$ and $\epsilon_\Sigma > 0$.

Induced Riemannian metric: $G_{\text{reg}}(x_i, S) := (H_i(S) + \epsilon_\Sigma I)^{-1}$

Role: ensures $(H_i + \epsilon_\Sigma I)$ strictly positive definite, eigenvalues bounded below by $\epsilon_\Sigma$, transforms uniform ellipticity from conjecture to trivial property.

**Related Results:** `thm-ueph`

---

### k-Uniform Ellipticity of the Regularized Metric

**Type:** Theorem
**Label:** `thm-ueph`
**Source:** [07_adaptative_gas.md § 4.1](07_adaptative_gas.md)
**Tags:** `uniform-ellipticity`, `eigenvalue-bounds`, `N-uniform`, `UEPH`

**Statement:**
For $\epsilon_\Sigma > H_{\max}(\rho)$ (k-uniform bound from Theorem A.2), the regularized metric

$$G_{\text{reg}}(S) = (H(S) + \epsilon_\Sigma I)^{-1}$$

is uniformly elliptic with k-uniform ellipticity constants:

$$c_{\min}(\rho) = \frac{1}{H_{\max}(\rho) + \epsilon_\Sigma}, \quad c_{\max}(\rho) = \frac{1}{\epsilon_\Sigma - H_{\max}(\rho)}$$

such that $c_{\min}(\rho) I \preceq G_{\text{reg}}(S) \preceq c_{\max}(\rho) I$ for all $S \in \Sigma_N$, all $k$, all $N$.

Critical property: Since $H_{\max}(\rho)$ independent of $k$ and $N$, ellipticity constants depend only on ρ and $\epsilon_\Sigma$.

**Related Results:** `thm-c2-regularity`, `cor-wellposed`

---

### Foster-Lyapunov Drift for Adaptive Viscous Fluid

**Type:** Theorem
**Label:** `thm-fl-drift-adaptive`
**Source:** [07_adaptative_gas.md § 7.1](07_adaptative_gas.md)
**Tags:** `Foster-Lyapunov`, `geometric-ergodicity`, `ρ-dependent`, `stability-threshold`

**Statement:**
For localization scale ρ > 0, there exist ρ-dependent critical parameters $\epsilon_F^*(\rho) > 0$ and $\nu^*(\rho) > 0$ such that for all $0 \le \epsilon_F < \epsilon_F^*(\rho)$ and $0 \le \nu < \nu^*(\rho)$:

$$\mathbb{E}[V_{\text{total}}(S_{k+1}) \mid S_k] \le (1 - \kappa_{\text{total}}(\rho)) V_{\text{total}}(S_k) + C_{\text{total}}(\rho)$$

where:

$$\kappa_{\text{total}}(\rho) = \kappa_{\text{backbone}} - \epsilon_F K_F(\rho) - C_{\text{diff,1}}(\rho) > 0$$

Critical stability threshold: $\epsilon_F^*(\rho) := \frac{\kappa_{\text{backbone}} - C_{\text{diff,1}}(\rho)}{2 K_F(\rho)}$

Interpretation: For any fixed ρ > 0, $\epsilon_F^*(\rho) > 0$. Smaller ρ → larger $K_F(\rho)$ → smaller $\epsilon_F^*(\rho)$ (more restrictive). As ρ → ∞, recovers backbone.

Consequently, full adaptive system is geometrically ergodic with exponential convergence rate $\lambda(\rho) = 1 - \kappa_{\text{total}}(\rho)$.

**Related Results:** `cor-exp-convergence`, `thm-c1-regularity`

---

### C¹ Regularity and k-Uniform Gradient Bound

**Type:** Theorem
**Label:** `thm-c1-regularity`
**Source:** [07_adaptative_gas.md § A.3](07_adaptative_gas.md)
**Tags:** `C¹-regularity`, `gradient-bound`, `N-uniform`, `ρ-dependent`

**Statement:**
The ρ-localized fitness potential $V_{\text{fit}}[f_k, \rho](x_i) = g_A(Z_\rho[f_k, d, x_i])$ is C¹ in $x_i$ with gradient satisfying:

$$\|\nabla_{x_i} V_{\text{fit}}[f_k, \rho](x_i)\| \le F_{\text{adapt,max}}(\rho)$$

where:

$$F_{\text{adapt,max}}(\rho) = L_{g_A} \cdot \left[ \frac{2d'_{\max}}{\sigma'_{\min,\text{bound}}} \left(1 + \frac{2d_{\max} C_{\nabla K}(\rho)}{\rho d'_{\max}}\right) + \frac{4d_{\max}^2 L_{\sigma'_{\text{patch}}}}{\sigma'^2_{\min,\text{bound}}} \cdot C_{\mu,V}(\rho) \right]$$

with N-uniform bound on variance derivative: $C_{\mu,V}(\rho) = 2d'_{\max}(d_{\max} + d'_{\max}) + 4d_{\max}^2 \frac{C_{\nabla K}(\rho)}{\rho}$

k-Uniformity: Bound independent of $k$ (thus $N$) due to telescoping property + only $k_{\text{eff}}(\rho) = O(1)$ walkers contributing effectively.

Implication: Adaptive force remains bounded, scalable perturbation for all alive walker counts and swarm sizes.

**Related Results:** `lem-adaptive-force-bounded`, `thm-fl-drift-adaptive`

---

### C² Regularity and k-Uniform Hessian Bound

**Type:** Theorem
**Label:** `thm-c2-regularity`
**Source:** [07_adaptative_gas.md § A.4](07_adaptative_gas.md)
**Tags:** `C²-regularity`, `Hessian-bound`, `N-uniform`, `ρ-dependent`

**Statement:**
The ρ-localized fitness potential $V_{\text{fit}}[f_k, \rho](x_i)$ is C² in $x_i$ with Hessian satisfying:

$$\|\nabla^2_{x_i} V_{\text{fit}}[f_k, \rho](x_i)\| \le H_{\max}(\rho)$$

where $H_{\max}(\rho)$ is k-uniform (thus N-uniform) ρ-dependent constant:

$$H_{\max}(\rho) = L_{g''_A} \|\nabla Z_\rho\|^2_{\max}(\rho) + L_{g_A} \|\nabla^2 Z_\rho\|_{\max}(\rho)$$

For Gaussian kernel with bounded measurements, using telescoping property: $H_{\max}(\rho) = O(1/\rho^2)$ and independent of $k$ (thus $N$).

Implication: Since $H_{\max}(\rho)$ N-uniform, choice $\epsilon_\Sigma = C \cdot (1/\rho^2)$ ensures uniform ellipticity for all $k$ and $N$. Entire adaptive framework maintains mathematical validity as N → ∞.

**Related Results:** `thm-ueph`, `def-regularized-hessian-tensor`

---

### Keystone Lemma for ρ-Localized Adaptive Model

**Type:** Theorem
**Label:** `thm-keystone-adaptive`
**Source:** [07_adaptative_gas.md § B.5](07_adaptative_gas.md)
**Tags:** `Keystone-Principle`, `N-uniform`, `ρ-dependent`, `structural-reduction`

**Statement:**
For adaptive model with localization scale ρ > 0 satisfying ρ-Dependent Stability Condition, the N-Uniform Quantitative Keystone Lemma from 03_cloning.md holds:

$$\frac{1}{N} \sum_{i \in I_{11}} (p_{1,i} + p_{2,i}) \|\Delta \delta_{x,i}\|^2 \ge \chi(\epsilon, \rho) \cdot V_{\text{struct}}(S) - g_{\max}(\epsilon, \rho)$$

where $\chi(\epsilon, \rho) > 0$ is ρ-dependent structural reduction coefficient, $g_{\max}(\epsilon, \rho)$ is ρ-dependent geometric negligibility bound, both constants uniform in N and depend continuously on ρ.

Significance: Critical bridge between backbone and adaptive models. Establishes cloning mechanism remains stable under ρ-localized measurements with all stability constants becoming functions of ρ but remaining positive and N-uniform for any fixed ρ > 0.

**Related Results:** `thm-signal-generation-adaptive`, `thm-fl-drift-adaptive`

---

## Emergent Riemannian Geometry

This section contains mathematical results for the Adaptive Gas with anisotropic, state-dependent diffusion, establishing convergence through emergent Riemannian geometric structure.

### Adaptive Diffusion Tensor

**Type:** Definition
**Label:** `def-d-adaptive-diffusion`
**Source:** [08_emergent_geometry.md § 1.1](08_emergent_geometry.md)
**Tags:** `anisotropic-diffusion`, `regularization`, `matrix-square-root`, `emergent-geometry`

**Statement:**
For swarm state $S = \{(x_i, v_i, s_i)\}_{i=1}^N$, the adaptive diffusion tensor for walker $i$:

$$\Sigma_{\text{reg}}(x_i, S) = \left( H_i(S) + \epsilon_\Sigma I \right)^{-1/2}$$

where $H_i(S) = \nabla^2_{x_i} V_{\text{fit}}(S)$ is Hessian of fitness potential, $\epsilon_\Sigma > 0$ is regularization parameter.

Induced diffusion matrix: $D_{\text{reg}}(x_i, S) = \left( H_i(S) + \epsilon_\Sigma I \right)^{-1}$

**Related Results:** `thm-uniform-ellipticity-geom`, `prop-lipschitz-diffusion`

---

### Hypocoercive Norm

**Type:** Definition
**Label:** `def-d-hypocoercive-norm`
**Source:** [08_emergent_geometry.md § 3.2.1](08_emergent_geometry.md)
**Tags:** `hypocoercivity`, `position-velocity-coupling`, `wasserstein`

**Statement:**
For phase-space differences $(\Delta x, \Delta v) \in \mathbb{R}^{2d}$:

$$\|(\Delta x, \Delta v)\|_h^2 = \|\Delta x\|^2 + \lambda_v \|\Delta v\|^2 + b \langle \Delta x, \Delta v \rangle$$

where $\lambda_v > 0$ is velocity weight (typically $\lambda_v \sim 1/\gamma$), $b \in \mathbb{R}$ is coupling coefficient. Positive definiteness requires $\lambda_v > b^2/4$. Optimal choice: $\lambda_v = 1/\gamma$, $b = 2/\sqrt{\gamma}$.

**Related Results:** `thm-hypocoercive-main`, `thm-location-error-anisotropic`

---

### Coupled Lyapunov Function for Geometric Ergodicity

**Type:** Definition
**Label:** `def-d-coupled-lyapunov`
**Source:** [08_emergent_geometry.md § 2.1](08_emergent_geometry.md)
**Tags:** `lyapunov-function`, `wasserstein`, `variance`, `boundary-potential`

**Statement:**
The total Lyapunov function:

$$V_{\text{total}}(S_1, S_2) = c_V V_{\text{inter}}(S_1, S_2) + c_B V_{\text{boundary}}(S_1, S_2)$$

Inter-swarm component:
$$V_{\text{inter}}(S_1, S_2) = V_W(S_1, S_2) + V_{\text{Var},x}(S_1, S_2) + V_{\text{Var},v}(S_1, S_2)$$

where $V_W(S_1, S_2) = W_h^2(\mu_1, \mu_2)$ is Wasserstein-2 distance with hypocoercive cost, position variance sum $V_{\text{Var},x}$, velocity variance sum $V_{\text{Var},v}$.

Boundary component: $V_{\text{boundary}}(S_1, S_2) = W_b(S_1) + W_b(S_2)$

**Related Results:** `thm-foster-lyapunov-adaptive-geom`, `thm-main-convergence-geom`

---

### Emergent Riemannian Metric

**Type:** Definition
**Label:** `def-metric-explicit`
**Source:** [08_emergent_geometry.md § 9.4](08_emergent_geometry.md)
**Tags:** `riemannian-geometry`, `metric-tensor`, `diffusion-tensor`

**Statement:**
For walker at position $x$ in swarm state $S$:

$$g(x, S) = H(x, S) + \epsilon_\Sigma I$$

where $H(x, S) = \nabla^2_x V_{\text{fit}}[f_k, \rho](x)$.

Diffusion tensor (inverse of metric): $D_{\text{reg}}(x, S) = g(x, S)^{-1}$

Diffusion coefficient matrix: $\Sigma_{\text{reg}}(x, S) = g(x, S)^{-1/2}$

**Related Results:** `thm-uniform-ellipticity-explicit-geom`, `def-emergent-manifold`

---

### Emergent Riemannian Manifold

**Type:** Definition
**Label:** `def-emergent-manifold`
**Source:** [08_emergent_geometry.md § 9.5](08_emergent_geometry.md)
**Tags:** `geodesics`, `christoffel-symbols`, `volume-element`, `curvature`

**Statement:**
The metric $g(x, S)$ endows state space with Riemannian structure $(\mathcal{X}, g_S)$.

Geometric quantities:
1. Metric tensor: $g_{ab}(x, S) = [\nabla^2_x V_{\text{fit}}]_{ab} + \epsilon_\Sigma \delta_{ab}$
2. Inverse metric: $g^{ab}(x, S) = [(\nabla^2_x V_{\text{fit}} + \epsilon_\Sigma I)^{-1}]_{ab}$
3. Volume element: $d\text{Vol}_g = \sqrt{\det g(x, S)} \, dx$
4. Geodesic equation: $\frac{d^2 \gamma^a}{dt^2} + \Gamma^a_{bc}(x, S) \frac{d\gamma^b}{dt} \frac{d\gamma^c}{dt} = 0$
5. Christoffel symbols: $\Gamma^a_{bc} = \frac{1}{2} g^{ad} \left(\frac{\partial g_{db}}{\partial x^c} + \frac{\partial g_{dc}}{\partial x^b} - \frac{\partial g_{bc}}{\partial x^d}\right)$

**Related Results:** `prop-geodesics-fitness`, `thm-algorithmic-tunability`

---

### Uniform Ellipticity by Construction

**Type:** Theorem
**Label:** `thm-uniform-ellipticity-geom`
**Source:** [08_emergent_geometry.md § 1.2](08_emergent_geometry.md)
**Tags:** `ellipticity`, `regularization`, `eigenvalue-bounds`, `n-uniform`

**Statement:**
For all swarm states $S$ and walkers $i$:

$$c_{\min} I \preceq D_{\text{reg}}(x_i, S) \preceq c_{\max} I$$

where:
$$c_{\min} = \frac{1}{\lambda_{\max}(H) + \epsilon_\Sigma}, \quad c_{\max} = \frac{1}{\epsilon_\Sigma - \Lambda_-}$$

Simplified form ($H \succeq 0$): $c_{\min} = \frac{1}{\lambda_{\max}(H) + \epsilon_\Sigma}$, $c_{\max} = \frac{1}{\epsilon_\Sigma}$

Proof: Eigenvalue analysis of regularized inverse diffusion matrix.

**Related Results:** `thm-uniform-ellipticity-explicit-geom`, `assump-spectral-floor`

---

### Geometric Ergodicity of the Adaptive Gas

**Type:** Theorem
**Label:** `thm-main-convergence-geom`
**Source:** [08_emergent_geometry.md § 2.2](08_emergent_geometry.md)
**Tags:** `foster-lyapunov`, `convergence`, `exponential-rate`, `qsd`

**Statement:**
Under uniform ellipticity and coercivity assumptions, there exist coupling constants $c_V, c_B > 0$ and N-uniform constants $\kappa_{\text{total}} > 0$, $C_{\text{total}} < \infty$ such that:

Foster-Lyapunov Condition:
$$\mathbb{E}[V_{\text{total}}(S_1', S_2') \mid S_1, S_2] \le (1 - \kappa_{\text{total}}) V_{\text{total}}(S_1, S_2) + C_{\text{total}}$$

Geometric Ergodicity:
$$\left\| \mathcal{L}(S_t \mid S_0) - \pi_{\text{QSD}} \right\|_{\text{TV}} \le C_\pi (1 + V_{\text{total}}(S_0, S_0)) \rho^t$$

where $\rho = 1 - \kappa_{\text{total}} < 1$.

Explicit Rate: $\kappa_{\text{total}} = O\left(\min\left\{\gamma \tau, \, \kappa_x^{\text{clone}}, \, c_{\min}\right\}\right)$

**Related Results:** `thm-foster-lyapunov-adaptive-geom`, `thm-explicit-total-rate-geom`

---

### Invariance Under Coordinate Changes

**Type:** Theorem
**Label:** `thm-coordinate-invariance`
**Source:** [08_emergent_geometry.md § 1.6.3](08_emergent_geometry.md)
**Tags:** `geometric-invariance`, `coordinate-transformation`, `tv-distance`

**Statement:**
Let $\Psi: (\mathbb{R}^d, D_{\text{flat}}) \to (M, g)$ be $C^2$ diffeomorphism with bounded Jacobian norms and push-forward relation $D_{\text{flat}}(x) = (d\Psi_x^{-1})^T g(\Psi(x))^{-1} (d\Psi_x^{-1})$.

Then:
1. TV distances match exactly: $\|\mathcal{L}^{\text{flat}}(X_t) - \pi^{\text{flat}}\|_{\text{TV}} = \|\mathcal{L}^{\text{curved}}(Y_t) - \pi^{\text{curved}}\|_{\text{TV}}$
2. Lyapunov drift preserved up to condition number: $\mathbb{E}[\Delta V_{\text{curved}}(Y)] \le -\kappa' V_{\text{curved}}(Y) + C'$ where $\kappa' \asymp \kappa/\text{cond}(d\Psi)^2$

**Related Results:** `obs-two-formulations`, `prop-rate-metric-ellipticity`

---

### Hypocoercive Contraction for Adaptive Gas

**Type:** Theorem
**Label:** `thm-hypocoercive-main`
**Source:** [08_emergent_geometry.md § 3.2.5](08_emergent_geometry.md)
**Tags:** `main-result`, `wasserstein-contraction`, `anisotropic-diffusion`

**Statement:**
The inter-swarm Wasserstein distance $V_W(S_1, S_2) = V_{\text{loc}} + V_{\text{struct}}$ satisfies:

$$\mathbb{E}[\Delta V_W] \le -\kappa'_W \tau V_W + C'_W \tau$$

where:
- $\kappa'_W = \min\{\kappa_{\text{loc}}, \kappa_{\text{struct}}\} = O(\min\{\gamma, c_{\min}\}) > 0$
- $C'_W = C_{\text{loc}} + C_{\text{struct}} = O(c_{\max}^2)$
- Both constants N-uniform

Proof: Direct combination of location and structural error theorems.

**Related Results:** `thm-location-error-anisotropic`, `thm-structural-error-anisotropic`

---

### Total Convergence Rate with Full Parameter Dependence

**Type:** Theorem
**Label:** `thm-explicit-total-rate-geom`
**Source:** [08_emergent_geometry.md § 5.2](08_emergent_geometry.md)
**Tags:** `explicit-parameters`, `algorithmic-tunability`, `convergence-rate`

**Statement:**
The total convergence rate has explicit form:

$$\kappa_{\text{total}} = \min\left\{ \kappa_x, \quad \min\left\{\gamma, \frac{\epsilon_\Sigma}{\lambda_{\max}(H) + \epsilon_\Sigma}\right\} \tau, \quad \kappa_b + O(\alpha_U) \tau \right\}$$

All constants independent of swarm size $N$.

**Related Results:** `thm-explicit-total-constant-geom`, `cor-explicit-convergence-time`

---

### Explicit Hessian Formula

**Type:** Theorem
**Label:** `thm-explicit-hessian`
**Source:** [08_emergent_geometry.md § 9.3](08_emergent_geometry.md)
**Tags:** `chain-rule`, `curvature`, `z-score`, `localization`

**Statement:**
The Hessian of fitness potential is:

$$H(x, S) = \nabla^2_x V_{\text{fit}}[f_k, \rho](x) = g''_A(Z) \, \nabla_x Z \otimes \nabla_x Z + g'_A(Z) \, \nabla^2_x Z$$

Expanded form:
$$H(x, S) = \frac{g''_A(Z)}{\sigma'^2_\rho} \nabla_x d \otimes \nabla_x d + \frac{g'_A(Z)}{\sigma'_\rho} \nabla^2_x d + \text{(moment correction terms)}$$

N-uniform bound: $\|H(x, S)\| \le H_{\max}(\rho) = \frac{g''_{\max} (d'_{\max})^2}{\kappa^2_{\text{var,min}}} + \frac{g'_{\max} d''_{\max}}{\kappa_{\text{var,min}}} + O(1/\rho)$

**Related Results:** `def-fitness-algorithmic`, `thm-uniform-ellipticity-explicit-geom`

---

### Uniform Ellipticity from Regularization

**Type:** Theorem
**Label:** `thm-uniform-ellipticity-explicit-geom`
**Source:** [08_emergent_geometry.md § 9.4](08_emergent_geometry.md)
**Tags:** `spectral-bounds`, `positive-definiteness`, `algorithmic-control`

**Statement:**
The metric $g(x, S)$ is uniformly elliptic:

$$c_{\min}(\rho) I \preceq g(x, S) \preceq c_{\max} I$$

where:
$$c_{\min}(\rho) = \epsilon_\Sigma - \Lambda_-(\rho), \quad c_{\max} = H_{\max}(\rho) + \epsilon_\Sigma$$

Sufficient condition: $\epsilon_\Sigma > \Lambda_-(\rho)$ ensures $g \succ 0$.

Inverse bounds: $\frac{1}{c_{\max}} I \preceq D_{\text{reg}}(x, S) \preceq \frac{1}{c_{\min}(\rho)} I$

**Related Results:** `thm-uniform-ellipticity-geom`, `thm-explicit-hessian`

---

### Algorithmic Tunability of Emergent Geometry

**Type:** Theorem
**Label:** `thm-algorithmic-tunability`
**Source:** [08_emergent_geometry.md § 9.6.2](08_emergent_geometry.md)
**Tags:** `parameter-control`, `geometric-design`, `information-geometry`

**Statement:**
The emergent Riemannian geometry is completely determined by algorithmic parameters:

1. Localization scale ρ: Controls spatial extent (small ρ → hyper-local, large ρ → global)
2. Regularization $\epsilon_\Sigma$: Controls deviation from Euclidean (small → strong adaptation $g \approx H$, large → weak adaptation $g \approx \epsilon_\Sigma I$)
3. Variance regularization $\kappa_{\text{var,min}}$: Controls Z-score conditioning
4. Measurement function $d$: Determines emergent structure
5. Rescale function $g_A$: Controls curvature amplification

**Related Results:** `def-emergent-manifold`, `thm-explicit-hessian`

---

### Lipschitz Continuity of Adaptive Diffusion

**Type:** Proposition
**Label:** `prop-lipschitz-diffusion`
**Source:** [08_emergent_geometry.md § 1.3](08_emergent_geometry.md)
**Tags:** `smoothness`, `n-uniform`, `state-dependence`

**Statement:**
The adaptive diffusion tensor is Lipschitz continuous:

$$\|\Sigma_{\text{reg}}(x_1, S_1) - \Sigma_{\text{reg}}(x_2, S_2)\|_F \le L_\Sigma \cdot d_{\text{state}}((x_1, S_1), (x_2, S_2))$$

where $L_\Sigma = K_{\text{sqrt}} \cdot L_H$ is independent of $N$, $K_{\text{sqrt}}$ depends on $\epsilon_\Sigma, H_{\max}$, and $L_H = L_{\phi}^{(3)}$ from fitness potential third derivatives.

State-space metric: $d_{\text{state}}((x_i, S_1), (x_i, S_2)) = \|x_{1,i} - x_{2,i}\| + \frac{1}{N}\sum_{j=1}^N \|x_{1,j} - x_{2,j}\|$

**Related Results:** `thm-structural-error-anisotropic`, `thm-explicit-hessian`

---

### Geodesics Favor High-Fitness Regions

**Type:** Proposition
**Label:** `prop-geodesics-fitness`
**Source:** [08_emergent_geometry.md § 9.5](08_emergent_geometry.md)
**Tags:** `geodesics`, `natural-gradient`, `riemannian-distance`

**Statement:**
1. Shorter distances in high-fitness regions: Riemannian distance $d_g(x_1, x_2) = \inf_{\gamma: x_1 \to x_2} \int_0^1 \sqrt{g_{ab}(\gamma(t), S) \dot{\gamma}^a(t) \dot{\gamma}^b(t)} \, dt$ is smaller when path passes through high $V_{\text{fit}}$ regions.

2. Geodesics avoid high-curvature regions: Metric eigenvalues largest where Hessian has large positive eigenvalues.

3. Natural gradient connection: $\nabla^{\text{nat}} V_{\text{fit}} = g^{-1} \nabla V_{\text{fit}} = D_{\text{reg}} \nabla V_{\text{fit}}$

**Related Results:** `def-emergent-manifold`, `thm-algorithmic-tunability`

---

## Symmetries of the Adaptive Gas

This section contains mathematical results characterizing the symmetry structure of the Adaptive Gas, including permutation invariance, Euclidean symmetries, emergent geometric isometries, and conservation laws from Noether's theorem.

### Swarm Configuration Space

**Type:** Definition
**Label:** `def-swarm-config-space`
**Source:** [09_symmetries_adaptive_gas.md § 1.1](09_symmetries_adaptive_gas.md)
**Tags:** `swarm-state`, `configuration-space`, `state-space`, `alive-dead-status`

**Statement:**
The **full swarm configuration space** is:

$$\Sigma_N^{\text{full}} = (\mathcal{X} \times \mathcal{V} \times \{0,1\})^N$$

where $\mathcal{X} \subset \mathbb{R}^d$ is position space, $\mathcal{V} = \{v \in \mathbb{R}^d : \|v\| \le V_{\text{alg}}\}$ is velocity ball, $\{0,1\}$ encodes alive/dead status.

The **alive subspace**: $\Sigma_N^{\text{alive}} = \{\mathcal{S} \in \Sigma_N^{\text{full}} : |\mathcal{A}(\mathcal{S})| \ge 1\}$ where $\mathcal{A}(\mathcal{S}) = \{i : s_i = 1\}$.

**Related Results:** Foundation for all symmetry definitions

---

### Permutation Group Action

**Type:** Definition
**Label:** `def-permutation-group`
**Source:** [09_symmetries_adaptive_gas.md § 1.2](09_symmetries_adaptive_gas.md)
**Tags:** `permutation`, `symmetric-group`, `walker-exchangeability`, `S_N`

**Statement:**
The **symmetric group** $S_N$ acts on $\Sigma_N$ by permuting walker indices. For $\sigma \in S_N$:

$$\sigma(\mathcal{S}) = ((x_{\sigma(1)}, v_{\sigma(1)}, s_{\sigma(1)}), \ldots, (x_{\sigma(N)}, v_{\sigma(N)}, s_{\sigma(N)}))$$

This is a **finite group** of order $|S_N| = N!$.

**Related Results:** `thm-permutation-symmetry`

---

### Permutation Invariance Theorem

**Type:** Theorem
**Label:** `thm-permutation-symmetry`
**Source:** [09_symmetries_adaptive_gas.md § 2.1](09_symmetries_adaptive_gas.md)
**Tags:** `permutation`, `invariance`, `symmetric-group`, `exchangeability`

**Statement:**
The Adaptive Gas transition operator $\Psi$ is **exactly invariant** under $S_N$. For any $\sigma \in S_N$:

$$\Psi(\sigma(\mathcal{S}_t), \cdot) = \sigma \circ \Psi(\mathcal{S}_t, \cdot)$$

Equivalently: $P(\mathcal{S}_{t+1} | \mathcal{S}_t) = P(\sigma(\mathcal{S}_{t+1}) | \sigma(\mathcal{S}_t))$

Proof: Verified invariance at each algorithm stage (measurement, fitness, cloning, kinetic, status refresh).

**Related Results:** `cor-qsd-exchangeable`

---

### Conditional Translation Equivariance

**Type:** Theorem
**Label:** `thm-translation-equivariance`
**Source:** [09_symmetries_adaptive_gas.md § 2.2](09_symmetries_adaptive_gas.md)
**Tags:** `translation`, `equivariance`, `reward-invariance`, `domain-invariance`

**Statement:**
Suppose reward and domain satisfy $R(x + a, v) = R(x, v)$ and $x + a \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}$ for some $a \in \mathbb{R}^d$.

Then: $\Psi(T_a(\mathcal{S}), \cdot) = T_a \circ \Psi(\mathcal{S}, \cdot)$ where $T_a$ translates all positions.

Note: Breaks in bounded domains.

**Related Results:** `prop-global-limit-symmetry`

---

### Rotational Equivariance

**Type:** Theorem
**Label:** `thm-rotation-equivariance`
**Source:** [09_symmetries_adaptive_gas.md § 2.3](09_symmetries_adaptive_gas.md)
**Tags:** `rotation`, `equivariance`, `SO(d)`, `rotational-symmetry`

**Statement:**
Suppose:
1. Domain rotationally symmetric: $Rx \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}$ for all $R \in SO(d)$
2. Reward rotation-invariant: $R(Rx, Rv) = R(x, v)$

Then: $\Psi(\mathcal{R}(\mathcal{S}), \cdot) = \mathcal{R} \circ \Psi(\mathcal{S}, \cdot)$ where $\mathcal{R}(\mathcal{S}) = \{(Rx_i, Rv_i, s_i)\}$.

**Related Results:** Algorithmic distance, localization kernel, kinetic operator all rotation-invariant

---

### Time-Reversal Asymmetry

**Type:** Theorem
**Label:** `thm-irreversibility`
**Source:** [09_symmetries_adaptive_gas.md § 2.5](09_symmetries_adaptive_gas.md)
**Tags:** `time-reversal`, `irreversibility`, `entropy-production`, `dissipative`

**Statement:**
The Adaptive Gas is **not time-reversible**. There exists no time-reversal operator $\mathcal{T}$ such that:

$$\mathcal{T} \circ \Psi \circ \mathcal{T}^{-1} = \Psi^{-1}$$

Furthermore, the system exhibits **strict entropy production**.

Proof: Cloning breaks time-reversal via fitness-dependent jumps; companion selection non-reversible; entropy production monotone.

**Related Results:** `prop-h-theorem`

---

### Emergent Isometries Theorem

**Type:** Theorem
**Label:** `thm-emergent-isometries`
**Source:** [09_symmetries_adaptive_gas.md § 3.2](09_symmetries_adaptive_gas.md)
**Tags:** `isometry`, `riemannian-metric`, `euclidean-transformation`, `pull-back`

**Statement:**
Suppose fitness potential invariant under Euclidean isometry $\Phi(x) = Lx + b$ with $L \in O(d)$:

$$V_{\text{fit}}(\Phi(x), \Phi_L(v), \Phi_*(S)) = V_{\text{fit}}(x, v, S)$$

Then $\Phi$ is an **isometry** of the emergent metric: $\Phi^* g(x, S) = g(\Phi(x), \Phi_*(S))$

Proof: Detailed computation of Hessian transformation under affine maps using orthogonality $L^T L = I$.

**Related Results:** `thm-geodesic-invariance`

---

### Geodesic Invariance Under Isometries

**Type:** Theorem
**Label:** `thm-geodesic-invariance`
**Source:** [09_symmetries_adaptive_gas.md § 3.3](09_symmetries_adaptive_gas.md)
**Tags:** `geodesic`, `isometry`, `length-preservation`, `levi-civita`

**Statement:**
Let $\Phi$ be an isometry of $g(x, S)$. If $\gamma(t)$ is a geodesic, then $\Phi(\gamma(t))$ is also a geodesic.

Furthermore: $L_g(\Phi(\gamma)) = L_g(\gamma)$ where $L_g(\gamma) = \int_0^1 \sqrt{g(\gamma(t))(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt$.

**Related Results:** `cor-geodesic-distance-symmetry`

---

### Fisher-Rao Geometry Embedding

**Type:** Theorem
**Label:** `thm-fisher-geometry`
**Source:** [09_symmetries_adaptive_gas.md § 3.5](09_symmetries_adaptive_gas.md)
**Tags:** `fisher-information`, `information-geometry`, `natural-gradient`, `statistical-model`

**Statement:**
Suppose fitness arises from statistical model $p_\theta(r)$ with $\theta = x$:

$$V_{\text{fit}}(x) = -\log p_x(r_{\text{obs}})$$

Then Hessian is **Fisher information matrix**:

$$H_{ij}(x) = \mathbb{E}_{r \sim p_x}\left[\frac{\partial \log p_x(r)}{\partial x_i} \frac{\partial \log p_x(r)}{\partial x_j}\right]$$

The emergent metric $g = H + \epsilon_\Sigma I$ is **regularized Fisher-Rao metric**, and Adaptive Gas performs **natural gradient descent**.

**Related Results:** `cor-statistical-symmetries`

---

### Noether's Theorem for Adaptive Gas

**Type:** Theorem
**Label:** `thm-noether-adaptive`
**Source:** [09_symmetries_adaptive_gas.md § 4.1](09_symmetries_adaptive_gas.md)
**Tags:** `noether`, `conservation`, `continuous-symmetry`, `conserved-charge`

**Statement:**
Let $\{T_s\}_{s \in \mathbb{R}}$ be one-parameter symmetry group: $\Psi(T_s(\mathcal{S}), \cdot) = T_s \circ \Psi(\mathcal{S}, \cdot)$

Define infinitesimal generator: $Q(\mathcal{S}) = \left.\frac{d}{ds}\right|_{s=0} T_s(\mathcal{S})$

Then **Noether charge** $J(\mathcal{S}) = \langle \mathcal{S}, Q(\mathcal{S}) \rangle$ satisfies:

$$\mathbb{E}[J(\mathcal{S}_{t+1}) | \mathcal{S}_t] = J(\mathcal{S}_t)$$

Proof: Infinitesimal symmetry expansion; generator commutes with Markov operator.

**Related Results:** Classical Noether theorem adapted to stochastic setting

---

### Angular Momentum Decay

**Type:** Theorem
**Label:** `thm-angular-momentum-decay`
**Source:** [09_symmetries_adaptive_gas.md § 4.2](09_symmetries_adaptive_gas.md)
**Tags:** `angular-momentum`, `friction`, `exponential-decay`, `dissipative`

**Statement:**
For rotationally symmetric system with radial force $F(x) = f(\|x\|) \frac{x}{\|x\|}$, the **expected total angular momentum** $L(\mathcal{S}) = \sum_{i=1}^N x_i \times v_i$ decays exponentially:

$$\frac{d}{dt}\mathbb{E}[L(t)] = -\gamma \mathbb{E}[L(t)]$$

implying $\mathbb{E}[L(t)] = L(0) \, e^{-\gamma t}$

Proof: Radial force has zero torque; friction creates dissipative torque.

**Related Results:** `cor-frictionless-angular-momentum`

---

### Isotropy Breaking via Adaptive Diffusion

**Type:** Theorem
**Label:** `thm-anisotropy-transition`
**Source:** [09_symmetries_adaptive_gas.md § 5.2](09_symmetries_adaptive_gas.md)
**Tags:** `anisotropy`, `adaptive-diffusion`, `symmetry-breaking`, `isometry-group`

**Statement:**
Euclidean Gas has **isotropic** diffusion: $\Sigma_{\text{EG}} = \sigma_v I$.

Adaptive Gas has **anisotropic** diffusion: $\Sigma_{\text{reg}}(x, S) = (H(x, S) + \epsilon_\Sigma I)^{-1/2} \neq \sigma I$ breaking rotational symmetry to the **isometry group** of the Hessian:

$$G_{\text{iso}} = \{R \in O(d) : R^T H(x, S) R = H(x, S)\}$$

Proof: Hessian eigenspaces define preferred directions; only rotations preserving eigenspaces leave diffusion invariant.

**Related Results:** `cor-directional-exploration`

---

### Spontaneous Symmetry Breaking Conjecture

**Type:** Conjecture
**Label:** `conj-localization-symmetry-breaking`
**Source:** [09_symmetries_adaptive_gas.md § 5.1](09_symmetries_adaptive_gas.md)
**Tags:** `spontaneous-symmetry-breaking`, `local-limit`, `clustering`, `pattern-formation`, `phase-transition`

**Statement:**
In the **local limit** ($\rho \to 0$) with position-independent reward $R(v)$, the Adaptive Gas exhibits **spontaneous symmetry breaking**: even though dynamics preserve translation symmetry, the QSD develops **spatial structure** (clusters, patterns) breaking this symmetry.

For sufficiently small $\rho > 0$ and strong viscous coupling $\nu > 0$, the QSD $\pi_{\text{QSD}}^\rho$ is **not** translation-invariant, despite transition operator being translation-equivariant.

Supporting evidence: Physical mechanism via viscous coupling; analogy to ferromagnetism/Ising model; numerical simulations show cluster formation.

**Related Results:** `prop-global-limit-symmetry` (contrast)

---

## Gauge Theory Formulation

This section develops the rigorous gauge-theoretic formulation of the Adaptive Gas using braid group topology and principal orbifold bundles, establishing mathematical foundations for publication in top-tier mathematics journals.

### Gauge Group

**Type:** Definition
**Label:** `def-gauge-group-rigorous`
**Source:** [12_gauge_theory_adaptive_gas.md § 1.1](12_gauge_theory_adaptive_gas.md)
**Tags:** `gauge-group`, `symmetric-group`, `principal-bundle`, `permutation-action`

**Statement:**
The **gauge group** for the Adaptive Gas is the symmetric group:

$$G = S_N$$

acting on swarm state space $\Sigma_N = (\mathcal{X} \times \mathcal{V} \times \{0,1\})^N$ by:

$$\sigma \cdot (w_1, \ldots, w_N) = (w_{\sigma(1)}, \ldots, w_{\sigma(N)})$$

for $\sigma \in S_N$ and $w_i = (x_i, v_i, s_i)$.

**Related Results:** `def-config-space-rigorous`

---

### Configuration Space as Orbifold

**Type:** Definition
**Label:** `def-config-space-rigorous`
**Source:** [12_gauge_theory_adaptive_gas.md § 1.1](12_gauge_theory_adaptive_gas.md)
**Tags:** `configuration-space`, `gauge-orbit`, `quotient-topology`, `orbifold`

**Statement:**
For $\mathcal{S} \in \Sigma_N$, the **gauge orbit** is:

$$[\mathcal{S}]_G = \{\sigma \cdot \mathcal{S} : \sigma \in S_N\}$$

The **configuration space** is the orbit space:

$$\mathcal{M}_{\text{config}} = \Sigma_N / S_N = \{[\mathcal{S}]_G : \mathcal{S} \in \Sigma_N\}$$

equipped with quotient topology induced by projection $\pi_G: \Sigma_N \to \mathcal{M}_{\text{config}}$.

**Related Results:** `thm-config-orbifold`

---

### Stabilizer Subgroup

**Type:** Definition
**Label:** `def-stabilizer`
**Source:** [12_gauge_theory_adaptive_gas.md § 1.2](12_gauge_theory_adaptive_gas.md)
**Tags:** `stabilizer`, `isotropy-group`, `orbifold-structure`

**Statement:**
For $\mathcal{S} = (w_1, \ldots, w_N) \in \Sigma_N$, the **stabilizer subgroup** is:

$$\text{Stab}_{S_N}(\mathcal{S}) = \{\sigma \in S_N : \sigma \cdot \mathcal{S} = \mathcal{S}\}$$

Equivalently, consists of all permutations $\sigma$ such that $w_{\sigma(i)} = w_i$ for all $i$.

Structure: If walkers partition into equivalence classes $I_1, \ldots, I_m$ (same state within class), then:

$$\text{Stab}_{S_N}(\mathcal{S}) \cong S_{|I_1|} \times S_{|I_2|} \times \cdots \times S_{|I_m|}$$

**Related Results:** `prop-stabilizer-structure`

---

### Configuration Space Orbifold Structure

**Type:** Theorem
**Label:** `thm-config-orbifold`
**Source:** [12_gauge_theory_adaptive_gas.md § 1.4](12_gauge_theory_adaptive_gas.md)
**Tags:** `orbifold`, `smooth-manifold`, `covering-space`, `dimension`

**Statement:**
The configuration space $\mathcal{M}_{\text{config}} = \Sigma_N / S_N$ is a **smooth orbifold** of dimension $2Nd$, with:

1. **Generic part**: $\mathcal{M}_{\text{config}}^{\text{gen}} = \Sigma_N^{\text{gen}} / S_N$ is a smooth manifold
2. **Singular part**: Points in $\mathcal{M}_{\text{config}}^{\text{sing}}$ have non-trivial **orbifold groups** (stabilizers)
3. **Covering**: Projection $\pi_G|_{\Sigma_N^{\text{gen}}}: \Sigma_N^{\text{gen}} \to \mathcal{M}_{\text{config}}^{\text{gen}}$ is an $N!$-sheeted covering map

Proof: Discrete group action gives orbifold; free action on generic locus gives manifold; covering degree equals $|S_N| = N!$.

**Related Results:** `def-generic-locus`

---

### Braid Group

**Type:** Definition
**Label:** `def-braid-group`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.2](12_gauge_theory_adaptive_gas.md)
**Tags:** `braid-group`, `algebraic-topology`, `configuration-space`, `fundamental-group`

**Statement:**
The **$N$-strand braid group** $B_N$ is the group of isotopy classes of braids on $N$ strands.

**Algebraic presentation**: Generated by $\{\sigma_1, \ldots, \sigma_{N-1}\}$ (elementary braids) subject to:
1. **Far commutativity**: $\sigma_i \sigma_j = \sigma_j \sigma_i$ for $|i - j| \geq 2$
2. **Braid relation**: $\sigma_i \sigma_{i+1} \sigma_i = \sigma_{i+1} \sigma_i \sigma_{i+1}$

**Geometric interpretation**: Each $\sigma_i$ represents elementary crossing where particle $i$ passes over particle $i+1$.

**Related Results:** `thm-braid-to-permutation`

---

### Fundamental Group is Braid Group

**Type:** Theorem
**Label:** `thm-fundamental-group-isomorphism`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.1](12_gauge_theory_adaptive_gas.md)
**Tags:** `fundamental-group`, `braid-group`, `fiber-bundle`, `contractible-fiber`

**Statement:**
The projection $p: \mathcal{M}'^{\text{state}}_{\text{config}} \to \mathcal{M}'^{\text{spatial}}_{\text{config}}$ induces an isomorphism on fundamental groups:

$$p_*: \pi_1(\mathcal{M}'^{\text{state}}_{\text{config}}) \xrightarrow{\cong} \pi_1(\mathcal{M}'^{\text{spatial}}_{\text{config}}) \cong B_N(\mathcal{X})$$

Therefore, fundamental group of full state configuration space is the braid group.

Proof: Long exact sequence of homotopy groups; fiber $(V \times \{0,1\})^N$ contractible; exactness gives isomorphism.

**Related Results:** `prop-spatial-config-topology`

---

### Canonical Homomorphism to Permutations

**Type:** Theorem
**Label:** `thm-braid-to-permutation`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.2](12_gauge_theory_adaptive_gas.md)
**Tags:** `braid-to-permutation`, `homomorphism`, `pure-braid-group`, `short-exact-sequence`

**Statement:**
There exists canonical surjective group homomorphism:

$$\rho: B_N \to S_N$$

mapping each braid to **net permutation** it induces on strands.

On generators: $\rho(\sigma_i) = \tau_i = (i \, i+1)$ (transposition).

The **kernel** is the **pure braid group** $P_N$:

$$1 \to P_N \to B_N \xrightarrow{\rho} S_N \to 1$$

(short exact sequence).

Proof: Assignment $\rho(\sigma_i) = (i \, i+1)$ extends to homomorphism by verifying far commutativity and braid relation; surjective since any permutation is product of adjacent transpositions.

**Related Results:** `def-parallel-transport-braid`

---

### Parallel Transport via Braid Holonomy

**Type:** Definition
**Label:** `def-parallel-transport-braid`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.3](12_gauge_theory_adaptive_gas.md)
**Tags:** `parallel-transport`, `gauge-connection`, `holonomy`, `flat-connection`

**Statement:**
Let $\gamma: [0,1] \to \mathcal{M}'^{\text{spatial}}_{\text{config}}$ be path in non-singular spatial configuration space.

The **parallel transport map** acts on fibers of state configuration space:

$$\mathcal{T}_\gamma: p^{-1}([x_1^0, \ldots, x_N^0]) \to p^{-1}([x_1^1, \ldots, x_N^1])$$

defined by:

$$\mathcal{T}_\gamma(\mathcal{S}) = \rho([\gamma]) \cdot \mathcal{S}$$

where $[\gamma] \in B_N(\mathcal{X})$ is braid class represented by spatial path, $\rho([\gamma]) \in S_N$ is induced permutation.

**Geometric meaning**: To parallel transport full state along spatial path, relabel walkers according to braid permutation induced by how positions braid in physical space.

**Related Results:** `thm-parallel-transport-well-defined`

---

### Well-Definedness of Parallel Transport

**Type:** Theorem
**Label:** `thm-parallel-transport-well-defined`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.3](12_gauge_theory_adaptive_gas.md)
**Tags:** `well-definedness`, `flat-connection`, `holonomy`, `principal-bundle`

**Statement:**
The parallel transport map $\mathcal{T}_\gamma$ is **well-defined** on fibers and defines a **flat $S_N$-connection** on the principal bundle over spatial configuration space.

Proof: Well-definedness on fibers by $S_N$-action; flatness from holonomy composition via group homomorphism $\rho: B_N \to S_N$; curvature 2-form vanishes.

**Related Results:** `def-holonomy-braid`

---

### Holonomy for Closed Loops

**Type:** Definition
**Label:** `def-holonomy-braid`
**Source:** [12_gauge_theory_adaptive_gas.md § 3.4](12_gauge_theory_adaptive_gas.md)
**Tags:** `holonomy`, `gauge-transformation`, `braid-class`, `closed-loop`

**Statement:**
For closed loop $\gamma: [0,1] \to \mathcal{M}'_{\text{config}}$ with $\gamma(0) = \gamma(1) = [\mathcal{S}_0]$, the **holonomy** is the permutation:

$$\text{Hol}(\gamma) = \rho([\gamma]) \in S_N$$

This is net relabeling of walkers induced by continuously following path through configuration space.

Holonomy depends only on **homotopy class**: $\text{Hol}: \pi_1(\mathcal{M}'_{\text{config}}) \to S_N$

**Related Results:** `thm-holonomy-topological`

---

### Dynamics Generate Braids

**Type:** Proposition
**Label:** `prop-dynamics-generate-braids`
**Source:** [12_gauge_theory_adaptive_gas.md § 4.1](12_gauge_theory_adaptive_gas.md)
**Tags:** `dynamics`, `braid-realization`, `spacetime-trajectory`, `kinetic-cloning`

**Statement:**
Under Adaptive Gas dynamics, whenever swarm configuration $[\mathcal{S}(t)]$ completes closed loop in $\mathcal{M}'_{\text{config}}$ (returns to same unordered set of positions/velocities), the trajectory traces a braid in spacetime.

The **braid class** depends on specific path taken, determined by:
1. Kinetic operator (Langevin dynamics)
2. Cloning operator (companion selection and walker replacement)
3. Emergent metric $g(x, S) = H(x, S) + \epsilon_\Sigma I$

Proof: Construct continuous paths $x_i(t)$ for each walker; returning configuration forms braid in $\mathcal{X} \times [0,T]$; homotopy class determines holonomy.

**Related Results:** `thm-accessible-braids`

---

### Accessible Non-Trivial Braids

**Type:** Theorem
**Label:** `thm-accessible-braids`
**Source:** [12_gauge_theory_adaptive_gas.md § 4.2](12_gauge_theory_adaptive_gas.md)
**Tags:** `anisotropic-dynamics`, `non-trivial-holonomy`, `positive-probability`, `elementary-braid`

**Statement:**
Consider Adaptive Gas evolving under **anisotropic emergent metric** $g(x, S) = H(x, S) + \epsilon_\Sigma I$ where $H$ is non-constant.

Then with positive probability, dynamics generate **non-trivial braids**:

$$\mathbb{P}(\exists T > 0: [\mathcal{S}(T)] = [\mathcal{S}_0] \text{ and } \text{Hol}(\gamma) \neq e) > 0$$

Proof: Explicit construction for $N=2$, $d=2$: define tubular neighborhood around elementary braid $\sigma_1$; Langevin dynamics has positive probability to remain in tube via support theorem for diffusions; path returns to initial configuration with net transposition $(1 \, 2)$.

**Related Results:** `thm-curvature-information-flow`

---

### Transition Operator Descends to Configuration Space

**Type:** Theorem
**Label:** `thm-transition-descends`
**Source:** [12_gauge_theory_adaptive_gas.md § 5](12_gauge_theory_adaptive_gas.md)
**Tags:** `gauge-invariance`, `descent`, `quotient-map`, `well-defined-dynamics`

**Statement:**
The Adaptive Gas transition operator $\Psi: \Sigma_N \to \mathcal{P}(\Sigma_N)$ is gauge-invariant:

$$\Psi(\sigma \cdot \mathcal{S}, \cdot) = (\sigma \cdot)_* \Psi(\mathcal{S}, \cdot)$$

for all $\sigma \in S_N$, where $(\sigma \cdot)_*$ is push-forward map on probability measures.

Therefore, $\Psi$ descends to well-defined operator:

$$\bar{\Psi}: \mathcal{M}_{\text{config}} \to \mathcal{P}(\mathcal{M}_{\text{config}})$$

Proof: Follows from equivariance of each algorithm stage (measurement, cloning, kinetics, status refresh).

**Related Results:** Reference to Theorem 6.4.4 in symmetries document

---

## KL-Divergence Convergence and Logarithmic Sobolev Inequalities

This section contains results on exponential KL-divergence convergence, which is stronger than the total variation (TV) convergence proven in Section "Complete Convergence to QSD". The theory uses logarithmic Sobolev inequalities (LSI), hypocoercivity, and entropy-transport Lyapunov functions to establish exponential convergence in relative entropy.

### Relative Entropy (KL-Divergence)

**Type:** Definition
**Label:** `def-relative-entropy`
**Source:** [10_kl_convergence.md § 1.1](10_kl_convergence/10_kl_convergence.md)
**Tags:** `kl-divergence`, `relative-entropy`, `entropy`, `kullback-leibler`

**Statement:**
For probability measures $\mu, \nu$ on $(\mathcal{X} \times \mathbb{R}^d)^N$ with $\mu \ll \nu$:

$$D_{\text{KL}}(\mu \| \nu) := \int \log\left(\frac{d\mu}{d\nu}\right) d\mu = \mathbb{E}_\mu\left[\log\left(\frac{d\mu}{d\nu}\right)\right]$$

Properties:
- Non-negative: $D_{\text{KL}}(\mu \| \nu) \geq 0$ with equality iff $\mu = \nu$
- Not symmetric: $D_{\text{KL}}(\mu \| \nu) \neq D_{\text{KL}}(\nu \| \mu)$
- Convex in first argument

**Related Results:** Fundamental to all KL-convergence results, stronger than TV-distance

---

### Relative Fisher Information

**Type:** Definition
**Label:** `def-relative-fisher`
**Source:** [10_kl_convergence.md § 1.1](10_kl_convergence/10_kl_convergence.md)
**Tags:** `fisher-information`, `entropy-dissipation`, `score-function`

**Statement:**
For $\mu \ll \nu$ with density $h = d\mu/d\nu$:

$$I(\mu \| \nu) := \int \left\|\nabla \log h\right\|^2 d\mu = \int \frac{\|\nabla h\|^2}{h} d\nu$$

Physical interpretation: Rate of entropy dissipation under diffusion.

Connection to entropy: $\frac{d}{dt}D_{\text{KL}}(\mu_t \| \nu) = -I(\mu_t \| \nu)$ for Fokker-Planck evolution.

**Related Results:** `def-lsi-continuous`, `thm-hwi-inequality`

---

### Logarithmic Sobolev Inequality (Continuous)

**Type:** Definition
**Label:** `def-lsi-continuous`
**Source:** [10_kl_convergence.md § 1.2](10_kl_convergence/10_kl_convergence.md)
**Tags:** `lsi`, `functional-inequality`, `entropy-fisher`

**Statement:**
A probability measure $\pi$ on $\mathbb{R}^m$ satisfies an **LSI with constant $C_{\text{LSI}} > 0$** if for all smooth densities $f$:

$$\text{Ent}_\pi(f^2) \leq C_{\text{LSI}} \cdot I(f^2 \pi \| \pi)$$

where $\text{Ent}_\pi(f^2) := \int f^2 \log f^2 \, d\pi - \left(\int f^2 d\pi\right) \log\left(\int f^2 d\pi\right)$.

Equivalent formulation (KL-Fisher):

$$D_{\text{KL}}(\mu \| \pi) \leq C_{\text{LSI}} \cdot I(\mu \| \pi)$$

for $\mu = f^2 \pi$.

**Implications:**
- Exponential convergence: $D_{\text{KL}}(\mu_t \| \pi) \leq e^{-t/C_{\text{LSI}}} D_{\text{KL}}(\mu_0 \| \pi)$
- Concentration of measure (Herbst's argument)
- Poincaré inequality

**Related Results:** `thm-main-kl-convergence`, `thm-hypocoercive-lsi`

---

### Logarithmic Sobolev Inequality (Discrete-Time)

**Type:** Definition
**Label:** `def-lsi-discrete`
**Source:** [10_kl_convergence.md § 1.2](10_kl_convergence/10_kl_convergence.md)
**Tags:** `lsi`, `discrete-time`, `markov-chain`

**Statement:**
A Markov transition operator $P: \mathcal{P}(\Omega) \to \mathcal{P}(\Omega)$ satisfies a **discrete-time LSI with constant $C > 0$** if:

$$D_{\text{KL}}(P\mu \| \pi) \leq (1 - 1/C) D_{\text{KL}}(\mu \| \pi)$$

for all $\mu \in \mathcal{P}(\Omega)$, where $\pi$ is the invariant measure.

Equivalent contraction form:

$$D_{\text{KL}}(\mu_t \| \pi) \leq \left(1 - \frac{1}{C}\right)^t D_{\text{KL}}(\mu_0 \| \pi)$$

**Related Results:** `thm-discrete-lsi`, `thm-main-kl-convergence`

---

### Hypocoercive Metric for LSI

**Type:** Definition
**Label:** `def-hypocoercive-metric-lsi`
**Source:** [10_kl_convergence.md § 2.1](10_kl_convergence/10_kl_convergence.md)
**Tags:** `hypocoercivity`, `auxiliary-metric`, `villani`, `position-velocity-coupling`

**Statement:**
For phase space $\mathbb{R}^{2d} = \mathbb{R}^d_x \times \mathbb{R}^d_v$, the **hypocoercive metric** with parameters $\lambda, \mu > 0$:

$$\|\nabla f\|^2_{\text{hypo}} := \|\nabla_v f\|^2 + \lambda \|\nabla_x f\|^2 + 2\mu \langle \nabla_v f, \nabla_x f \rangle$$

Hypocoercive Dirichlet form:

$$\mathcal{E}_{\text{hypo}}(f, f) := \int \|\nabla f\|^2_{\text{hypo}} f^2 d\pi$$

This captures position-velocity coupling essential for hypoelliptic systems.

**Optimal parameters:**
- $\lambda = O(1/\gamma)$
- $\mu = O(1/\sqrt{\gamma})$
- Ensures positive-definiteness and optimal contraction

**Related Results:** `thm-hypocoercive-lsi`, `lem-hypocoercive-dissipation`

---

### Entropy-Transport Lyapunov Function

**Type:** Definition
**Label:** `def-entropy-transport-lyapunov`
**Source:** [10_kl_convergence.md § 5.1](10_kl_convergence/10_kl_convergence.md)
**Tags:** `lyapunov`, `entropy`, `wasserstein`, `seesaw-mechanism`

**Statement:**
The **entropy-transport Lyapunov function** combines KL-divergence and Wasserstein distance:

$$\mathcal{L}_{\text{ET}}(\mu) := D_{\text{KL}}(\mu \| \pi_{\text{QSD}}) + \eta \cdot W_2^2(\mu, \pi_{\text{QSD}})$$

where $\eta > 0$ is a coupling weight.

**Seesaw mechanism:** Captures complementary dissipation:
- Kinetic operator: Primarily dissipates $W_2^2$ (spatial contraction)
- Cloning operator: Primarily dissipates $D_{\text{KL}}$ (fitness selection)

**Contraction condition:** Choose $\eta$ such that:

$$\mathbb{E}[\mathcal{L}_{\text{ET}}(\mu_{t+1})] \leq (1 - \kappa_{\text{ET}}) \mathcal{L}_{\text{ET}}(\mu_t)$$

for some $\kappa_{\text{ET}} > 0$.

**Related Results:** `thm-entropy-transport-contraction`, `lem-entropy-transport-dissipation`

---

### Exponential KL-Convergence for Euclidean Gas

**Type:** Theorem
**Label:** `thm-main-kl-convergence`
**Source:** [10_kl_convergence.md § 0.1](10_kl_convergence/10_kl_convergence.md)
**Tags:** `main-result`, `kl-convergence`, `exponential-decay`, `qsd`, `n-uniform`

**Statement:**
Under Axiom `ax-qsd-log-concave` (log-concavity of QSD), for N-particle Euclidean Gas with Foster-Lyapunov parameters and cloning noise variance $\delta^2$ satisfying:

$$\delta > \delta_* = e^{-\alpha\tau/(2C_0)} \cdot C_{\text{HWI}} \sqrt{\frac{2(1 - \kappa_W)}{\kappa_{\text{conf}}}}$$

the discrete-time Markov chain $S_{t+1} = \Psi_{\text{total}}(S_t) := (\Psi_{\text{kin}}(\tau) \circ \Psi_{\text{clone}})(S_t)$ satisfies a discrete-time LSI with constant $C_{\text{LSI}} > 0$.

Consequently, for any initial distribution $\mu_0$ with finite entropy:

$$D_{\text{KL}}(\mu_t \| \pi_{\text{QSD}}) \leq e^{-t/C_{\text{LSI}}} \cdot D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}})$$

**Explicit constant:** $C_{\text{LSI}} = O(1/(\gamma \kappa_{\text{conf}} \kappa_W \delta^2))$

**Parameter condition:** Noise parameter $\delta$ must be large enough to regularize Fisher information but not destroy convergence rate.

**Proof technique:** Three-stage composition:
1. Hypocoercive LSI for kinetic operator
2. HWI inequality for cloning operator
3. Entropy-transport Lyapunov function for composition

**Related Results:** `thm-hypocoercive-lsi`, `thm-hwi-inequality`, `thm-entropy-transport-contraction`

---

### Hypocoercive LSI for Kinetic Operator

**Type:** Theorem
**Label:** `thm-hypocoercive-lsi`
**Source:** [10_kl_convergence.md § 2.3](10_kl_convergence/10_kl_convergence.md)
**Tags:** `hypocoercivity`, `kinetic-operator`, `lsi`, `langevin`, `villani`

**Statement:**
The kinetic operator $\Psi_{\text{kin}}(\tau)$ for underdamped Langevin dynamics with confining potential $U(x)$ satisfying $\nabla^2 U \geq \kappa_{\text{conf}} I$ satisfies a hypocoercive LSI:

$$D_{\text{KL}}(\Psi_{\text{kin}}(\tau)\mu \| \pi_{\text{kin}}) \leq (1 - \kappa_{\text{kin}}\tau) D_{\text{KL}}(\mu \| \pi_{\text{kin}})$$

where $\pi_{\text{kin}}$ is the Gibbs measure and:

$$\kappa_{\text{kin}} = O(\min\{\gamma, \kappa_{\text{conf}}\})$$

**Proof technique:** Villani's hypocoercivity framework with auxiliary metric $\|\nabla f\|^2_{\text{hypo}}$ combining velocity dissipation and position-velocity coupling.

**Key steps:**
1. Microscopic coercivity: Velocity variance contracts by friction
2. Macroscopic transport: Position transport via $\dot{x} = v$
3. Coupling term: Rotates position error into velocity space

**Related Results:** `lem-hypocoercive-dissipation`, `def-hypocoercive-metric-lsi`

---

### HWI Inequality for Cloning Operator

**Type:** Theorem
**Label:** `thm-hwi-inequality`
**Source:** [10_kl_convergence.md § 4.2](10_kl_convergence/10_kl_convergence.md)
**Tags:** `hwi`, `otto-villani`, `optimal-transport`, `entropy`, `wasserstein`, `fisher`

**Statement:**
For the cloning operator $\Psi_{\text{clone}}$, distributions $\mu, \nu$ satisfy the **HWI inequality**:

$$D_{\text{KL}}(\Psi_{\text{clone}}\mu \| \Psi_{\text{clone}}\nu) \leq W_2(\Psi_{\text{clone}}\mu, \Psi_{\text{clone}}\nu) \cdot \sqrt{I(\Psi_{\text{clone}}\mu \| \Psi_{\text{clone}}\nu)}$$

Combined with Wasserstein contraction $W_2(\Psi_{\text{clone}}\mu, \Psi_{\text{clone}}\nu) \leq (1 - \kappa_W) W_2(\mu, \nu)$ and Fisher information regularization from cloning noise $\delta^2$:

$$D_{\text{KL}}(\Psi_{\text{clone}}\mu \| \pi) \leq (1 - \kappa_W) W_2(\mu, \pi) \cdot \sqrt{\frac{D_{\text{KL}}(\mu \| \pi)}{\delta^2}} + O(\delta^2)$$

**Proof technique:** Otto-Villani calculus on Wasserstein space, viewing cloning as gradient flow with jumps.

**Related Results:** `lem-wasserstein-contraction`, `lem-fisher-information-bound`

---

### Entropy-Transport Contraction

**Type:** Theorem
**Label:** `thm-entropy-transport-contraction`
**Source:** [10_kl_convergence.md § 5.3](10_kl_convergence/10_kl_convergence.md)
**Tags:** `entropy-transport`, `lyapunov`, `contraction`, `seesaw`

**Statement:**
The entropy-transport Lyapunov function $\mathcal{L}_{\text{ET}} = D_{\text{KL}} + \eta W_2^2$ satisfies:

$$\mathbb{E}[\mathcal{L}_{\text{ET}}(\mu_{t+1})] \leq (1 - \kappa_{\text{ET}}) \mathcal{L}_{\text{ET}}(\mu_t)$$

where:

$$\kappa_{\text{ET}} = \min\left\{\frac{\kappa_{\text{kin}} - C_{\text{HWI}}\sqrt{\eta}}{1 + \eta}, \, \frac{\kappa_W - C_{\text{LSI,kin}}\sqrt{\eta}}{\eta}\right\}$$

**Seesaw condition:** Choose $\eta$ to balance:
- Kinetic dissipation: $\Delta D_{\text{KL}} \approx -\kappa_{\text{kin}} D_{\text{KL}} + C_{\text{expand}} W_2^2$
- Cloning dissipation: $\Delta W_2^2 \approx -\kappa_W W_2^2 + C_{\text{expand}} D_{\text{KL}}$

Optimal: $\eta = \frac{\kappa_{\text{kin}}}{\kappa_W}$

**Related Results:** `lem-entropy-transport-dissipation`, `thm-main-kl-convergence`

---

### N-Uniform LSI Constant

**Type:** Theorem
**Label:** `thm-n-uniform-lsi`
**Source:** [10_Q_complete_resolution_summary.md § 4](10_kl_convergence/10_Q_complete_resolution_summary.md)
**Tags:** `n-uniform`, `scalability`, `mean-field-limit`

**Statement:**
Under appropriate regularity conditions, the LSI constant $C_{\text{LSI}}$ can be chosen **independent of N** (number of particles).

**Key requirements:**
1. **Permutation symmetry** (Gap #1 resolution): QSD is exchangeable
2. **Conditional independence** (Tensorization): Cloning preserves product structure conditionally
3. **Fisher information regularization** (Gap #3 resolution): Cloning noise prevents Fisher blow-up

**Explicit bound:**

$$C_{\text{LSI}} \leq C_0 \cdot \frac{1}{\gamma \kappa_{\text{conf}} \kappa_W \delta^2}$$

where $C_0$ is a universal constant independent of N.

**Implications:**
- Mean-field limit well-defined
- Propagation of chaos at KL-level
- Uniform concentration for all N

**Related Results:** `thm-gap1-resolution`, `thm-gap3-resolution`

---

### Axiom of Log-Concave QSD

**Type:** Axiom
**Label:** `ax-qsd-log-concave`
**Source:** [10_kl_convergence.md § 1.3](10_kl_convergence/10_kl_convergence.md)
**Tags:** `qsd`, `log-concavity`, `assumption`, `structural`

**Statement:**
The quasi-stationary distribution $\pi_{\text{QSD}}$ has a log-concave density with respect to Lebesgue measure on $(\mathcal{X} \times \mathbb{R}^d)^N$:

$$\pi_{\text{QSD}}(dx, dv) = h_{\text{QSD}}(x, v) \, dx \, dv$$

where $\log h_{\text{QSD}}$ is concave (i.e., $h_{\text{QSD}}$ is log-concave).

**Justification:**
- Kinetic part: Gibbs measure $\pi_{\text{kin}} \propto e^{-U(x) - \|v\|^2/2}$ is log-concave if $U$ convex
- Cloning part: Fitness-dependent selection preserves log-concavity under mild conditions
- Numerically verified for standard test cases

**Implications:**
- LSI holds for log-concave measures (Bakry-Émery)
- Fisher information bounds
- Concentration inequalities

**Related Results:** `thm-main-kl-convergence`, `thm-hypocoercive-lsi`

---

## Mean-Field Entropy Production and Explicit Constants

This section contains mean-field convergence results with explicit constants and parameter analysis. The mean-field limit removes finite-N fluctuations, enabling cleaner entropy production analysis and fully computable convergence rates.

### Revival Operator is KL-Expansive

**Type:** Theorem
**Label:** `thm-revival-kl-expansive`
**Source:** [11_stage0_revival_kl.md § 7.1](11_mean_field_convergence/11_stage0_revival_kl.md)
**Tags:** `revival`, `kl-divergence`, `expansive`, `verified`

**Statement:**
The mean-field revival operator $\mathcal{R}[\rho, m_d]$ **increases** the KL-divergence to the invariant measure $\pi$:

$$\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} > 0 \quad \text{for all } \rho \neq \pi, \, m_d > 0$$

Explicitly:

$$\frac{d}{dt} D_{\text{KL}}(\rho \| \pi) \Big|_{\text{revival}} = \lambda m_d \left( 1 + \frac{D_{\text{KL}}(\rho \| \pi)}{\|\rho\|_{L^1}} \right)$$

**Status**: PROVEN (verified by Gemini 2025-01-08)

**Implication:** KL-convergence requires kinetic dissipation to dominate revival expansion.

**Related Results:** `thm-corrected-kl-convergence-meanfield`

---

### QSD Regularity Framework (R1-R6)

**Type:** Framework
**Label:** `framework-qsd-regularity`
**Source:** [11_stage05_qsd_regularity.md](11_mean_field_convergence/11_stage05_qsd_regularity.md)
**Tags:** `regularity`, `qsd`, `smoothness`, `bounds`

**Statement:**
The QSD $\rho_\infty$ satisfies six regularity properties:

**R1 (Existence and Uniqueness)**: Via Schauder fixed-point theorem for nonlinear operator

**R2 (C² Smoothness)**: $\rho_\infty \in C^2(\Omega)$ via Hörmander hypoellipticity

**R3 (Strict Positivity)**: $\rho_\infty(x,v) > 0$ for all $(x,v) \in \Omega$ via irreducibility

**R4 (Bounded Spatial Log-Gradient)**:

$$C_{\nabla x} := \|\nabla_x \log \rho_\infty\|_{L^\infty} < \infty$$

**R5 (Bounded Velocity Log-Derivatives)**:

$$\begin{aligned}
C_{\nabla v} &:= \|\nabla_v \log \rho_\infty\|_{L^\infty} < \infty \\
C_{\Delta v} &:= \|\Delta_v \log \rho_\infty\|_{L^\infty} < \infty
\end{aligned}$$

**R6 (Exponential Concentration)**:

$$\rho_\infty(x,v) \leq C_{\exp} e^{-\alpha_{\exp}(|x|^2 + |v|^2)}$$

for some $C_{\exp}, \alpha_{\exp} > 0$

**Status**: All six properties proven (roadmap provided in Stage 0.5 document)

**Related Results:** `thm-lsi-constant-explicit-meanfield`, `thm-main-explicit-rate-meanfield`

---

### Explicit LSI Constant for Mean-Field QSD

**Type:** Theorem
**Label:** `thm-lsi-constant-explicit-meanfield`
**Source:** [11_stage2_explicit_constants.md § 2.2](11_mean_field_convergence/11_stage2_explicit_constants.md)
**Tags:** `lsi`, `explicit-constant`, `bakry-emery`

**Statement:**

$$\boxed{\lambda_{\text{LSI}} \geq \frac{\alpha_{\exp}}{1 + C_{\Delta v}/\alpha_{\exp}}}$$

where:
- $\alpha_{\exp}$ is the exponential concentration rate from (R6)
- $C_{\Delta v} = \|\Delta_v \log \rho_\infty\|_{L^\infty}$ from (R5)

**Simplified** (when $C_{\Delta v} \ll \alpha_{\exp}$):

$$\lambda_{\text{LSI}} \approx \alpha_{\exp} \left(1 - \frac{C_{\Delta v}}{\alpha_{\exp}}\right)$$

**Proof method**: Holley-Stroock perturbation theorem from Gaussian reference measure

**Related Results:** `framework-qsd-regularity`, `thm-main-explicit-rate-meanfield`

---

### Mean-Field Convergence Rate (Explicit)

**Type:** Theorem
**Label:** `thm-main-explicit-rate-meanfield`
**Source:** [11_stage2_explicit_constants.md § 5](11_mean_field_convergence/11_stage2_explicit_constants.md)
**Tags:** `explicit-constants`, `main-result`, `convergence-rate`

**Statement:**
Under QSD regularity (R1-R6) and:

$$\sigma^2 > \sigma_{\text{crit}}^2 := \frac{2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} + C_{\text{KL}}^{\text{coup}} + A_{\text{jump}}}{\lambda_{\text{LSI}}}$$

the mean-field Euclidean Gas converges exponentially with rate:

$$\boxed{\alpha_{\text{net}} = \frac{1}{2}\left(\lambda_{\text{LSI}} \sigma^2 - 2\lambda_{\text{LSI}} C_{\text{Fisher}}^{\text{coup}} - C_{\text{KL}}^{\text{coup}} - A_{\text{jump}}\right)}$$

where all constants are explicit in $(\gamma, \sigma, L_U, \kappa_{\max}, \lambda_{\text{revive}})$ and $(C_{\nabla x}, C_{\nabla v}, C_{\Delta v}, \alpha_{\exp})$.

**Significance:**
1. Fully computable
2. Numerically verifiable
3. Tunable via physical parameters
4. Completes mean-field convergence proof

**Related Results:** `thm-revival-kl-expansive`, `framework-qsd-regularity`

---

### KL-Convergence for Mean-Field Euclidean Gas

**Type:** Theorem
**Label:** `thm-corrected-kl-convergence-meanfield`
**Source:** [11_stage1_entropy_production.md § 4](11_mean_field_convergence/11_stage1_entropy_production.md)
**Tags:** `main-theorem`, `kl-convergence`, `framework`

**Statement:**
If the kinetic dominance condition holds:

$$\sigma^2 \gamma \kappa_{\text{conf}} > C_0 \max\left(\frac{\lambda}{M_\infty}, \bar{\kappa}\right)$$

then the mean-field Euclidean Gas converges exponentially to its QSD:

$$D_{\text{KL}}(\rho_t \| \rho_\infty) \leq e^{-\alpha_{\text{net}} t} D_{\text{KL}}(\rho_0 \| \rho_\infty) + \frac{B_{\text{jump}}}{\alpha_{\text{net}}} (1 - e^{-\alpha_{\text{net}} t})$$

where $\alpha_{\text{net}} = \alpha_{\text{kin}} - A_{\text{jump}} > 0$.

**Status**: Framework established, technical details in Stage 2-3

**Related Results:** `thm-revival-kl-expansive`, `thm-main-explicit-rate-meanfield`

---

### Optimal Parameter Scaling

**Type:** Theorem
**Label:** `thm-optimal-parameter-scaling`
**Source:** [11_stage3_parameter_analysis.md § 2.3](11_mean_field_convergence/11_stage3_parameter_analysis.md)
**Tags:** `optimal-scaling`, `parameter-tuning`

**Statement:**
For landscape with Lipschitz constant $L_U$ and minimum Hessian eigenvalue $\lambda_{\min}$:

$$\begin{aligned}
\gamma^* &\sim L_U^{3/7} \\
\sigma^* &\sim L_U^{9/14} \\
\tau^* &\sim L_U^{-12/7} \\
\lambda_{\text{revive}}^* &\sim \kappa_{\max}
\end{aligned}$$

yielding convergence rate:

$$\alpha_{\text{net}}^* \sim \gamma^* \sim L_U^{3/7}$$

**Interpretation**: Provides scaling laws for parameter selection as landscape roughness $L_U$ varies.

**Related Results:** `thm-main-explicit-rate-meanfield`

---

### Critical Diffusion Threshold

**Type:** Formula
**Label:** `formula-critical-diffusion`
**Source:** [11_stage3_parameter_analysis.md § 2.2](11_mean_field_convergence/11_stage3_parameter_analysis.md)
**Tags:** `critical-threshold`, `diffusion`, `convergence-condition`

**Statement:**
For $\alpha_{\text{net}} > 0$, dominant balance gives:

$$\boxed{\sigma_{\text{crit}} \gtrsim \left(\frac{2L_U^3}{\gamma}\right)^{1/4}}$$

**Interpretation**: Diffusion must scale as $L_U^{3/4}$ to overcome landscape roughness.

**Related Results:** `thm-main-explicit-rate-meanfield`, `thm-optimal-parameter-scaling`

---

### Finite-N Corrections to Mean-Field Rate

**Type:** Formula
**Label:** `formula-finite-n-corrections`
**Source:** [11_stage3_parameter_analysis.md § 5](11_mean_field_convergence/11_stage3_parameter_analysis.md)
**Tags:** `finite-n`, `discretization`, `corrections`

**Statement:**

$$\boxed{\alpha_N \approx \alpha_{\text{net}} \left(1 - \frac{c_{\text{clone}}}{\delta^2 N}\right) \left(1 - \frac{\tau \alpha_{\text{net}}}{2\gamma}\right)}$$

where:
- First factor: Cloning fluctuations ($c_{\text{clone}} \sim 1$, $\delta$ = cloning noise variance)
- Second factor: Time-discretization error

**Guideline**: To stay within 5% of mean-field rate:

$$\frac{c_{\text{clone}}}{\delta^2 N} + \frac{\tau \alpha_{\text{net}}}{2\gamma} < 0.05$$

**Related Results:** `thm-main-explicit-rate-meanfield`

---

## Hellinger-Kantorovich Metric Convergence

This section contains results on the exponential convergence of the Fragile Gas in the Hellinger-Kantorovich (HK) metric, which naturally handles hybrid continuous-discrete dynamics involving diffusion with birth/death processes. The theory establishes mass contraction via revival mechanisms, structural variance contraction via LSI machinery, and kinetic Hellinger contraction via hypocoercivity.

### Hellinger-Kantorovich Metric

**Type:** Definition
**Label:** `def-hk-metric`
**Source:** [18_hk_convergence.md § 0](18_hk_convergence.md)
**Tags:** `hellinger-kantorovich`, `metric`, `optimal-transport`, `birth-death`

**Statement:**
For two sub-probability measures $\mu_1, \mu_2$ on a metric space $(\mathcal{X}, d)$, the **Hellinger-Kantorovich metric** is:

$$
d_{HK}^2(\mu_1, \mu_2) := d_H^2(\mu_1, \mu_2) + W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)
$$

where:
- $d_H^2(\mu_1, \mu_2) = \int \left( \sqrt{\frac{d\mu_1}{d\lambda}} - \sqrt{\frac{d\mu_2}{d\lambda}} \right)^2 d\lambda$ is the **Hellinger distance**
- $W_2^2(\tilde{\mu}_1, \tilde{\mu}_2)$ is the **Wasserstein-2 distance** between normalized measures $\tilde{\mu}_i = \mu_i / \|\mu_i\|$
- $\lambda$ is a common reference measure (e.g., Lebesgue measure)

**Alternative form (Bhattacharyya):**

$$
d_H^2(\mu_1, \mu_2) = \|\mu_1\| + \|\mu_2\| - 2\int \sqrt{f_1 f_2} \, d\lambda
$$

where $f_i = d\mu_i / d\lambda$ and the integral is the **Bhattacharyya coefficient** $BC(\mu_1, \mu_2)$.

**Related Results:** `thm-hk-convergence-main`, `lem-mass-contraction-revival-death`

---

### Exponential HK-Convergence of the Fragile Gas

**Type:** Theorem
**Label:** `thm-hk-convergence-main`
**Source:** [18_hk_convergence.md § 1](18_hk_convergence.md)
**Tags:** `hellinger-kantorovich`, `convergence`, `exponential`, `contraction`

**Statement:**
Let the Fragile Gas evolve under the composed operator $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$ on the space of sub-probability measures representing the alive swarm. Let $\mu_t$ denote the empirical measure at time $t$ and $\pi_{\text{QSD}}$ denote the quasi-stationary distribution.

Then $\Psi_{\text{total}}$ is a strict contraction in the Hellinger-Kantorovich metric. Specifically, there exist constants $\kappa_{HK} > 0$ and $C_{HK} < \infty$ such that:

$$
\mathbb{E}[d_{HK}^2(\mu_{t+1}, \pi_{\text{QSD}})] \leq (1 - \kappa_{HK}) d_{HK}^2(\mu_t, \pi_{\text{QSD}}) + C_{HK}
$$

**Implication (Exponential Convergence):**

$$
d_{HK}(\mu_t, \pi_{\text{QSD}}) \leq e^{-\kappa_{HK} t/2} \cdot d_{HK}(\mu_0, \pi_{\text{QSD}}) + \sqrt{\frac{C_{HK}}{\kappa_{HK}}}
$$

**Proof Dependency Structure:**
- Part I: Wasserstein Component (PROVEN in existing framework via hypocoercivity and W₂ cloning contraction)
- Part II: Hellinger Component (Lemmas A, B, C below)

**Related Results:** `lem-mass-contraction-revival-death`, `lem-structural-variance-contraction`, `lem-kinetic-hellinger-contraction`

---

### Mass Contraction via Revival and Death

**Type:** Lemma
**Label:** `lem-mass-contraction-revival-death`
**Source:** [18_hk_convergence.md § 2](18_hk_convergence.md)
**Tags:** `mass-contraction`, `revival`, `boundary-death`, `lyapunov`

**Statement:**
Let $k_t = \|\mu_t\|$ denote the number of alive walkers at time $t$ (the total mass of the empirical measure). Let $k_* = \|\pi_{\text{QSD}}\|$ denote the equilibrium alive count under the QSD.

Assume:
1. **Birth Mechanism**: Total births $B_t = (N - k_t) + C_t$ where $\mathbb{E}[C_t | k_t] = \lambda_{\text{clone}}(k_t) k_t$
2. **Death Mechanism**: $\mathbb{E}[D_t | k_t] = \bar{p}_{\text{kill}}(k_t) k_t$
3. **QSD Equilibrium**: $(N - k_*) + \lambda_{\text{clone}}^* k_* = \bar{p}_{\text{kill}}^* k_*$
4. **Lipschitz Continuity**: Both $\lambda_{\text{clone}}(k)$ and $\bar{p}_{\text{kill}}(k)$ are Lipschitz continuous

Then there exist constants $\kappa_{\text{mass}} > 0$ and $C_{\text{mass}} < \infty$ such that:

$$
\mathbb{E}[(k_{t+1} - k_*)^2] \leq (1 - 2\kappa_{\text{mass}}) \mathbb{E}[(k_t - k_*)^2] + C_{\text{mass}}
$$

where:
- $\kappa_{\text{mass}} = \frac{1 - \epsilon - \epsilon^2}{2}$ with $\epsilon = (1 + 2L_p N + \bar{p}_{\text{kill}}^*)(L_\lambda N + \lambda_{\text{clone}}^*)$
- $C_{\text{mass}} = C_N \cdot N$ where $C_N = C_{\text{var}} + O(1/N)$
- $C_{\text{var}} = \bar{p}_{\max}(1 + \lambda_{\max}) + 2(1 + L_g^{(1)})^2 \lambda_{\max} + \frac{1}{2}(L_g^{(2)})^2 \lambda_{\max}$

**Assumptions:**
1. $\epsilon^2 + \epsilon < 1$, requiring $\epsilon < \frac{\sqrt{5} - 1}{2} \approx 0.618$ (achieved when $L_p L_\lambda = O(1/N^2)$ for large $N$)
2. $\bar{p}_{\text{kill}}(k')$ is twice continuously differentiable with $L_g^{(2)} = O(N^{-1})$

**Related Results:** `thm-hk-convergence-main`, `lem-structural-variance-contraction`

---

### Exponential Contraction of Structural Variance

**Type:** Lemma
**Label:** `lem-structural-variance-contraction`
**Source:** [18_hk_convergence.md § 3](18_hk_convergence.md)
**Tags:** `structural-variance`, `wasserstein`, `kl-divergence`, `lsi`, `reverse-talagrand`, `path-dependent`

**Statement:**
Let $\mu_t$ be the law of the empirical measure at time $t$ and let $\pi_{\text{QSD}}$ be the quasi-stationary distribution. Assume $\pi_{\text{QSD}}$ is log-concave.

Then for any initial measure $\mu_0$, the structural variance $V_{\text{struct}}(\mu_t, \pi_{\text{QSD}})$ contracts exponentially to zero:

$$
V_{\text{struct}}(\mu_t, \pi_{\text{QSD}}) \leq \frac{2}{\kappa_{\text{conf}}} \cdot D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}}) \cdot e^{-\lambda t}
$$

where:
- $\lambda > 0$ is the exponential convergence rate from the LSI (see `thm-main-kl-convergence`)
- $\kappa_{\text{conf}} > 0$ is the strong convexity constant of the confining potential
- $D_{\text{KL}}(\mu_0 \| \pi_{\text{QSD}})$ is the initial KL-divergence from equilibrium

**Explicit LSI constant:**

$$
\lambda = \frac{1}{C_{\text{LSI}}} = O(\gamma \kappa_{\text{conf}} \kappa_W \delta^2)
$$

where:
- $\gamma > 0$ is the friction coefficient (kinetic operator parameter)
- $\kappa_W > 0$ is the Wasserstein contraction rate of the cloning operator
- $\delta^2 > 0$ is the cloning noise variance

**Path-Dependence Note:** This is a **path-dependent convergence result**. The pre-factor depends on the initial measure $\mu_0$. For any initial measure $\mu_0$ with finite KL-divergence to $\pi_{\text{QSD}}$, the structural variance contracts exponentially at rate $\lambda$. The exponential rate $\lambda > 0$ is uniform and system-dependent, not path-dependent.

**Proof Strategy:**
1. KL-divergence contraction via LSI: $D_{\text{KL}}(\mu_t \| \pi) \leq e^{-\lambda t} D_{\text{KL}}(\mu_0 \| \pi)$
2. Reverse Talagrand: $W_2^2(\mu, \pi) \leq (2/\kappa_{\text{conf}}) D_{\text{KL}}(\mu \| \pi)$
3. Variance decomposition: $V_{\text{struct}} = W_2^2(\tilde{\mu}, \tilde{\pi}) \leq W_2^2(\mu, \pi)$

**Related Results:** `thm-main-kl-convergence`, `prop-wasserstein-variance-decomposition`, `lem-kinetic-hellinger-contraction`

---

### Wasserstein Variance Decomposition

**Type:** Proposition
**Label:** `prop-wasserstein-variance-decomposition`
**Source:** [18_hk_convergence.md § 3](18_hk_convergence.md)
**Tags:** `wasserstein`, `variance-decomposition`, `optimal-transport`

**Statement:**
For any two probability measures $\mu$ and $\pi$ on $\mathbb{R}^d$ with finite second moments:

$$
W_2^2(\mu, \pi) = W_2^2(\tilde{\mu}, \tilde{\pi}) + \|m_\mu - m_\pi\|^2
$$

where:
- $\tilde{\mu}$ is the centered version of $\mu$ (mean translated to origin)
- $\tilde{\pi}$ is the centered version of $\pi$ (mean translated to origin)
- $m_\mu := \mathbb{E}_{X \sim \mu}[X]$ is the mean of $\mu$
- $m_\pi := \mathbb{E}_{Y \sim \pi}[Y]$ is the mean of $\pi$

**Reference:** This is a standard result in optimal transport theory. For a proof, see Villani (2009), *Optimal Transport: Old and New*, Theorem 7.17, or Peyré & Cuturi (2019), *Computational Optimal Transport*, Section 2.3.

**Related Results:** `lem-structural-variance-contraction`

---

### Kinetic Operator Hellinger Contraction

**Type:** Lemma
**Label:** `lem-kinetic-hellinger-contraction`
**Source:** [18_hk_convergence.md § 4](18_hk_convergence.md)
**Tags:** `kinetic-operator`, `hellinger`, `hypocoercivity`, `bounded-density`, `mass-shape-decomposition`

**Statement:**
Let $\mu_t$ be the empirical measure of alive walkers at time $t$ and let $\pi_{\text{QSD}}$ be the quasi-stationary distribution.

**Assumption:** The normalized density ratio is uniformly bounded:

$$
\sup_{t \geq 0} \sup_{x \in \mathcal{X}_{\text{valid}}} \frac{d\tilde{\mu}_t}{d\tilde{\pi}_{\text{QSD}}}(x) \leq M < \infty
$$

where $\tilde{\mu}_t = \mu_t / \|\mu_t\|$ and $\tilde{\pi}_{\text{QSD}} = \pi_{\text{QSD}} / \|\pi_{\text{QSD}}\|$ are the normalized probability measures.

Under this assumption and the kinetic operator $\Psi_{\text{kin}}$ (BAOAB + boundary killing), there exist constants $\kappa_{\text{kin}}(M) > 0$ and $C_{\text{kin}} < \infty$ such that:

$$
\mathbb{E}[d_H^2(\mu_{t+1}, \pi_{\text{QSD}}) | \mu_t] \leq (1 - \kappa_{\text{kin}}(M) \tau) d_H^2(\mu_t, \pi_{\text{QSD}}) + C_{\text{kin}} \tau^2
$$

where $\tau$ is the time step size.

**Explicit constants:**

**Contraction rate:**

$$
\kappa_{\text{kin}} = \min\left(2\lambda_{\text{mass}}, \alpha_{\text{shape}}\right) = \min\left(2(r_* + c_*), \frac{\alpha_{\text{eff}}}{C_{\text{rev}}(M)}\right)
$$

where:

*Mass equilibration rate:*
- $\lambda_{\text{mass}} = r_* + c_*$ combines:
  - $r_* > 0$: equilibrium revival rate per empty slot
  - $c_* = \bar{c}_{\text{kill}}(\pi_{\text{QSD}}) > 0$: equilibrium death rate at QSD

*Shape contraction rate:*
- $\alpha_{\text{shape}} = \alpha_{\text{eff}} / C_{\text{rev}}(M)$ where:
  - $\alpha_{\text{eff}} = \min(\kappa_{\text{hypo}}, \alpha_U)$ is the effective hypocoercive rate
    - $\kappa_{\text{hypo}} \sim \gamma$: hypocoercive coupling rate (proportional to friction)
    - $\alpha_U > 0$: coercivity constant of potential $U$ in exterior region
  - $C_{\text{rev}}(M) = O(M)$: reverse Pinsker constant for density bound $M$

**Expansion constant:**

$$
C_{\text{kin}} = k_* K_H + C_{\text{cross}}
$$

where:
- $k_* = \|\pi_{\text{QSD}}\|$: equilibrium alive mass
- $K_H > 0$: BAOAB weak error constant (depends on potential smoothness, friction $\gamma$, noise strength $\sigma$)
- $C_{\text{cross}} > 0$: bounds cross-terms from $O(\tau^2 d_H^2)$ remainder

**Justification of Bounded Density Assumption:** The bounded density ratio is automatically satisfied when:
1. The initial measure has bounded density: $d\mu_0/d\pi_{\text{QSD}} \leq M_0 < \infty$
2. The cloning operator with Gaussian noise ($\delta^2 > 0$) provides immediate $L^\infty$ regularization
3. The diffusive Langevin dynamics prevents singularity formation
4. The confining potential ensures mass concentration where $\pi_{\text{QSD}} > 0$

**Proof Strategy:**
1. **Exact Hellinger decomposition** into mass and shape: $d_H^2(\mu_t, \pi) = (\sqrt{k_t} - \sqrt{k_*})^2 + \sqrt{k_t k_*} \cdot d_H^2(\tilde{\mu}_t, \tilde{\pi})$
2. **Mass contraction** via boundary killing and revival: $\mathbb{E}[(\sqrt{k_{t+1}} - \sqrt{k_*})^2] \leq (1 - 2\tau \lambda_{\text{mass}}) (\sqrt{k_t} - \sqrt{k_*})^2$
3. **Shape contraction** via hypocoercivity: $d_H^2(\tilde{\mu}_{t+1}, \tilde{\pi}) \leq (1 - \tau \alpha_{\text{shape}}) d_H^2(\tilde{\mu}_t, \tilde{\pi})$
4. **BAOAB discretization error**: $O(\tau^2)$ weak error in Hellinger distance

**Related Results:** `thm-hk-convergence-main`, `thm-hypocoercive-main`, `lem-mass-contraction-revival-death`

---

## Fractal Set Theory and Discrete Spacetime

This section contains the mathematical framework for viewing the Fragile Gas dynamics as a discrete spacetime structure ("Fractal Set") that emerges from episode trajectories. The theory connects to causal set quantum gravity, lattice gauge theory, and provides a discrete formulation of QFT on the walker genealogy graph.

**Note**: This is a condensed reference focusing on key theoretical structures. See [00_D_fractal_set.md](00_D_fractal_set.md) for the complete reference with 184 mathematical objects.

### Episode and Causal Spacetime Tree

**Type:** Definition
**Label:** `def-episode`
**Source:** [13_A_fractal_set.md § 1.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `episode`, `walker-trajectory`, `discrete-spacetime`

**Statement:**
An **episode** $e$ is a finite sequence of walker states:

$$e = \{(x_0, v_0, s_0, t_0), (x_1, v_1, s_1, t_1), \ldots, (x_T, v_T, s_T, t_T)\}$$

where $x_i \in \mathcal{X}$ (position), $v_i \in \mathbb{R}^d$ (velocity), $s_i \in \{0,1\}$ (alive/dead status), $t_i \in \mathbb{N}$ (discrete time), and $T$ is the episode length (time until absorption).

**Related Results:** `def-cst`, `def-fractal-set`

---

### Causal Spacetime Tree (CST)

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

**Related Results:** `def-ig`, `thm-cst-lorentzian`

---

### Information Graph (IG)

**Type:** Definition
**Label:** `def-ig`
**Source:** [13_A_fractal_set.md § 2.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `spacelike-correlation`, `undirected-graph`, `cloning-interaction`

**Statement:**
The **Information Graph** (IG) is an undirected graph $\mathcal{G} = (\mathcal{E}, E_{\text{IG}})$ where:
- $\mathcal{E}$ is the set of episodes
- $E_{\text{IG}} \subseteq \mathcal{E} \times \mathcal{E}$ (symmetric)

An edge $(e_i, e_j) \in E_{\text{IG}}$ exists iff episodes $e_i$ and $e_j$ interacted through a cloning event at some time $t$ (i.e., they were in the same local neighborhood $\mathcal{N}_\rho(x_i(t))$).

**Properties:**
1. **Symmetry**: $(e_i, e_j) \in E_{\text{IG}} \iff (e_j, e_i) \in E_{\text{IG}}$
2. **Spacelike**: Edges connect episodes at equal time slices
3. **Density**: Edge density scales with cloning rate and localization scale $\rho$

**Related Results:** `def-fractal-set`, `prop-ig-connectedness`

---

### Fractal Set

**Type:** Definition
**Label:** `def-fractal-set`
**Source:** [13_A_fractal_set.md § 3.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `composite-graph`, `causal-spacelike`, `discrete-spacetime`

**Statement:**
The **Fractal Set** $\mathcal{F}$ is the composite graph structure:

$$\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$$

where:
- $\mathcal{E}$ is the set of episodes (vertices)
- $E_{\text{CST}}$ are directed timelike edges (causal links)
- $E_{\text{IG}}$ are undirected spacelike edges (information links)

**Interpretation:**
- **CST**: Encodes causal structure (genealogy, time ordering)
- **IG**: Encodes spacelike correlations (cloning interactions, measurement)
- **Composite**: Captures full spacetime structure of the adaptive gas dynamics

**Related Results:** `thm-fractal-set-metric`, `thm-continuum-limit-lorentzian`

---

### Fractal Set Lorentzian Metric

**Type:** Theorem
**Label:** `thm-fractal-set-metric`
**Source:** [13_A_fractal_set.md § 3.2](13_fractal_set/13_A_fractal_set.md)
**Tags:** `spacetime-metric`, `lorentzian-signature`, `causal-structure`

**Statement:**
The Fractal Set $\mathcal{F}$ admits a **discrete Lorentzian metric** $d_{\mathcal{F}}: \mathcal{E} \times \mathcal{E} \to \mathbb{R}$:

$$d_{\mathcal{F}}^2(e_i, e_j) = d_{\text{CST}}^2(e_i, e_j) - d_{\text{IG}}^2(e_i, e_j)$$

**Properties:**
1. **Lorentzian Signature**: $(+, -, -, \ldots, -)$ in $(d+1)$ dimensions
2. **Causal Structure**: $d_{\mathcal{F}}^2 > 0$ iff $e_i$ and $e_j$ are timelike separated
3. **Light Cone**: $d_{\mathcal{F}}^2 = 0$ defines discrete light cone structure

**Related Results:** `thm-cst-lorentzian`, `def-causal-set`

---

### Graph Laplacian on Fractal Set

**Type:** Definition
**Label:** `def-graph-laplacian-fractal`
**Source:** [13_A_fractal_set.md § 4.1](13_fractal_set/13_A_fractal_set.md)
**Tags:** `graph-laplacian`, `discrete-geometry`, `diffusion-operator`

**Statement:**
For function $f: \mathcal{E} \to \mathbb{R}$ on the Fractal Set, the **graph Laplacian** is:

$$(\Delta_{\mathcal{F}} f)(e_i) = \sum_{e_j \sim e_i} w_{ij} [f(e_j) - f(e_i)]$$

where:
- $e_j \sim e_i$ means $(e_i, e_j) \in E_{\text{IG}}$ (spacelike neighbors)
- $w_{ij} = \exp(-d_{\text{IG}}^2(e_i, e_j) / (2\rho^2))$ is the Gaussian weight

**Properties:**
1. **Self-adjoint**: $\langle f, \Delta_{\mathcal{F}} g \rangle = \langle \Delta_{\mathcal{F}} f, g \rangle$
2. **Non-positive**: $\langle f, \Delta_{\mathcal{F}} f \rangle \leq 0$
3. **Kernel**: $\ker(\Delta_{\mathcal{F}}) = \text{span}\{\mathbb{1}\}$ if IG is connected

**Related Results:** `thm-laplacian-convergence`

---

### Graph Laplacian Convergence

**Type:** Theorem
**Label:** `thm-laplacian-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 3.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `continuum-limit`, `laplace-beltrami`, `convergence-rate`

**Statement:**
As the number of episodes $N \to \infty$ and localization scale $\rho \to 0$ with $N\rho^d \to \infty$:

$$\Delta_{\mathcal{F}} f \to \Delta_{\mathcal{X}} f$$

pointwise for $f \in C^2(\mathcal{X})$, where $\Delta_{\mathcal{X}}$ is the Laplace-Beltrami operator on state space $\mathcal{X}$.

**Convergence Rate**: For appropriate scaling:

$$\|\Delta_{\mathcal{F}} f - \Delta_{\mathcal{X}} f\|_{L^2} = O(N^{-1/4})$$

**Related Results:** `def-graph-laplacian-fractal`, `thm-heat-kernel-convergence`

---

### Episode Measure Convergence

**Type:** Theorem
**Label:** `thm-episode-measure-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 4.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `episode-measure`, `qsd`, `weak-convergence`, `continuum-limit`

**Statement:**
As $N \to \infty$, the episode measure $\mu_T^{(N)}$ converges weakly to the quasi-stationary distribution $\rho_{\text{QSD}}$:

$$\mu_T^{(N)} \xrightarrow{w} \rho_{\text{QSD}}$$

in the sense:

$$\lim_{N \to \infty} \int f \, d\mu_T^{(N)} = \int f \, \rho_{\text{QSD}} \, dx$$

for all continuous bounded $f: \mathcal{X} \to \mathbb{R}$.

**Convergence Rate**: $\|\mu_T^{(N)} - \rho_{\text{QSD}}\|_{TV} = O(N^{-1/2})$

**Related Results:** `def-episode-measure`, `thm-propagation-of-chaos`

---

### Causal Set Definition

**Type:** Definition
**Label:** `def-causal-set`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 1.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `causal-set`, `partially-ordered-set`, `discrete-spacetime`, `quantum-gravity`

**Statement:**
A **causal set** (causet) is a locally finite partially ordered set $(C, \prec)$ where:
1. **Partial Order**: $\prec$ is transitive, reflexive, and antisymmetric
2. **Local Finiteness**: For any $x, z \in C$, the set $\{y \in C : x \prec y \prec z\}$ is finite

**Interpretation:**
- Elements of $C$ represent spacetime events
- $x \prec y$ means "$x$ causally precedes $y$" (timelike or lightlike separation)
- Local finiteness ensures discrete structure

**Related Results:** `def-cst`, `thm-cst-lorentzian`

---

### Sprinkling Process

**Type:** Definition
**Label:** `def-sprinkling`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 2.1](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `poisson-process`, `random-causet`, `lorentzian-manifold`

**Statement:**
Given Lorentzian manifold $(M, g)$, a **sprinkling** is a random causal set generated by:
1. Sample points from Poisson process with density $\rho = 1/\ell_P^{d+1}$
2. Inherit causal order from spacetime: $x \prec y$ iff $x \in J^-(y)$ (causal past)

**Distribution**: Number of points in region $R$ follows Poisson$(\rho \cdot \text{Vol}(R))$.

**Related Results:** `thm-sprinkling-approximation`, `thm-dimension-estimation`

---

### Discrete Gauge Connection on IG

**Type:** Definition
**Label:** `def-discrete-gauge-connection`
**Source:** [13_B_fractal_set_continuum_limit.md § 2.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `gauge-connection`, `parallel-transport`, `wilson-line`

**Statement:**
For gauge group $G$ (e.g., $U(1)$, $SU(N)$), a **discrete gauge connection** on the IG is a map:

$$A: E_{\text{IG}} \to \mathfrak{g}$$

where $\mathfrak{g}$ is the Lie algebra of $G$.

**Parallel Transport**: Along edge $(e_i, e_j) \in E_{\text{IG}}$:

$$U_{ij} = \exp(A(e_i, e_j)) \in G$$

**Gauge Transformation**: Under $g: \mathcal{E} \to G$:

$$A(e_i, e_j) \mapsto g_i A(e_i, e_j) g_j^{-1} + g_i dg_j^{-1}$$

**Related Results:** `def-wilson-loop`, `thm-gauge-connection-convergence`

---

### Wilson Loop on Fractal Set

**Type:** Definition
**Label:** `def-wilson-loop`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.1](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `wilson-loop`, `gauge-invariant`, `parallel-transport`, `observable`

**Statement:**
For closed loop $\gamma$ in the IG and gauge connection $A$, the **Wilson loop** is:

$$W_\gamma[A] = \text{Tr}\left[\text{Hol}_\gamma(A)\right] = \text{Tr}\left[\prod_{(e_i, e_j) \in \gamma} U_{ij}\right]$$

where $U_{ij} = \exp(A(e_i, e_j))$ is the parallel transport operator.

**Gauge Invariance**: $W_\gamma[A^g] = W_\gamma[A]$ for any gauge transformation $g$.

**Related Results:** `thm-wilson-loop-convergence`, `conj-area-law`

---

### Wilson Loop Convergence

**Type:** Theorem
**Label:** `thm-wilson-loop-convergence`
**Source:** [13_B_fractal_set_continuum_limit.md § 5.2](13_fractal_set/13_B_fractal_set_continuum_limit.md)
**Tags:** `wilson-loop`, `continuum-limit`, `area-law`

**Statement:**
For smooth loop $\gamma$ in state space $\mathcal{X}$ and smooth gauge connection $A_\mu$:

$$\lim_{N \to \infty} W_{\gamma_N}[A] = \text{Tr}\left[\mathcal{P} \exp\left(\oint_\gamma A_\mu dx^\mu\right)\right]$$

where $\gamma_N$ is the discretization of $\gamma$ on the IG with $N$ episodes, and $\mathcal{P}$ denotes path ordering.

**Convergence Rate**: $|W_{\gamma_N}[A] - W_\gamma[A]| = O(\epsilon^2)$ where $\epsilon = 1/N^{1/d}$ is the lattice spacing.

**Related Results:** `def-wilson-loop`, `thm-holonomy-convergence`

---

### Antisymmetric Cloning Kernel

**Type:** Definition
**Label:** `def-antisymmetric-cloning-kernel`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 1.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `antisymmetric-kernel`, `fermionic`, `cloning`, `exclusion-principle`

**Statement:**
The **antisymmetric cloning kernel** $K_{\text{clone}}^{(-)}$ is defined as:

$$K_{\text{clone}}^{(-)}(e_i, e_j) = K_{\text{clone}}(e_i, e_j) - K_{\text{clone}}(e_j, e_i)$$

where $K_{\text{clone}}(e_i, e_j) = \exp(\alpha F_j - \beta H_j)$ is the standard cloning kernel.

**Properties:**
1. **Antisymmetry**: $K^{(-)}(e_i, e_j) = -K^{(-)}(e_j, e_i)$
2. **Vanishing Diagonal**: $K^{(-)}(e_i, e_i) = 0$
3. **Sign Change**: Under particle exchange $(e_i, e_j) \leftrightarrow (e_j, e_i)$, kernel changes sign

**Related Results:** `thm-algorithmic-exclusion`, `thm-fermi-dirac-statistics`

---

### Algorithmic Exclusion Principle

**Type:** Theorem
**Label:** `thm-algorithmic-exclusion`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 2.1](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `pauli-exclusion`, `fermi-statistics`, `antisymmetric-kernel`

**Statement:**
For antisymmetric cloning kernel $K^{(-)}$:

**Exclusion Principle**: No two episodes can simultaneously occupy the same state:

$$K^{(-)}(e, e) = 0 \quad \forall e \in \mathcal{E}$$

**Multi-Particle Extension**: For $n$ episodes $\{e_1, \ldots, e_n\}$:

$$\det[K^{(-)}(e_i, e_j)]_{i,j=1}^n = 0 \quad \text{if any } e_i = e_j$$

**Interpretation**: Algorithmic implementation of Pauli exclusion principle for fermions.

**Related Results:** `def-antisymmetric-cloning-kernel`, `thm-fermi-dirac-statistics`

---

### Fermi-Dirac Statistics from Cloning

**Type:** Theorem
**Label:** `thm-fermi-dirac-statistics`
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 2.2](13_fractal_set/13_D_fractal_set_emergent_qft_comprehensive.md)
**Tags:** `fermi-dirac`, `partition-function`, `grand-canonical`, `fermions`

**Statement:**
The episode measure induced by antisymmetric cloning kernel satisfies **Fermi-Dirac statistics**:

$$\langle n(e) \rangle = \frac{1}{e^{\beta (E(e) - \mu)} + 1}$$

where:
- $\langle n(e) \rangle$ is the average occupation number
- $E(e)$ is the energy of episode $e$
- $\mu$ is the chemical potential
- $\beta = 1/T$ is the inverse temperature

**Derivation**: From grand canonical ensemble with antisymmetric cloning.

**Related Results:** `thm-algorithmic-exclusion`, `def-grassmann-field-discrete`

---

### Lattice Gauge Action

**Type:** Theorem
**Label:** `thm-lattice-gauge-action`
**Source:** [13_E_cst_ig_lattice_qft.md § 1.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `lattice-gauge`, `wilson-action`, `discretization`

**Statement:**
The **lattice gauge action** on the Fractal Set for gauge group $G$ is:

$$S_{\text{gauge}}[A] = -\frac{1}{g^2} \sum_{P} \text{Re}\left[\text{Tr}(U_P)\right]$$

where:
- Sum is over all plaquettes $P$ in the IG
- $U_P = U_{12} U_{23} U_{34} U_{41}$ is the plaquette holonomy
- $g$ is the gauge coupling constant

**Continuum Limit**: As $\epsilon \to 0$ (lattice spacing):

$$S_{\text{gauge}}[A] \to -\frac{1}{4g^2} \int F_{\mu\nu} F^{\mu\nu} \sqrt{-g} \, d^{d+1}x$$

(Yang-Mills action).

**Related Results:** `def-discrete-gauge-connection`, `thm-sun-wilson-action`

---

### QCD Action on Fractal Set

**Type:** Definition
**Label:** `def-qcd-fractal-set`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.1](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `qcd`, `lattice-qcd`, `quarks`, `gluons`

**Statement:**
The **QCD action** on the Fractal Set with $SU(3)$ gauge group is:

$$S_{\text{QCD}} = S_{\text{gauge}}[U] + S_{\text{fermion}}[\psi, \bar{\psi}, U]$$

where:

**Gauge Action**:

$$S_{\text{gauge}}[U] = -\frac{\beta}{6} \sum_P \text{Re}\left[\text{Tr}(U_P)\right]$$

**Fermion Action** (Wilson fermions):

$$S_{\text{fermion}}[\psi, \bar{\psi}, U] = \sum_{e \in \mathcal{E}} \bar{\psi}(e) \psi(e) + \kappa \sum_{(e_i, e_j) \in E_{\text{IG}}} \bar{\psi}(e_i) U_{ij} \psi(e_j)$$

where $\kappa$ is the hopping parameter.

**Related Results:** `thm-sun-wilson-action`, `def-grassmann-field-discrete`

---

### Asymptotic Freedom

**Type:** Theorem
**Label:** `thm-asymptotic-freedom`
**Source:** [13_E_cst_ig_lattice_qft.md § 3.2](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `asymptotic-freedom`, `renormalization-group`, `beta-function`, `qcd`

**Statement:**
For $SU(N)$ gauge theory with $N \geq 2$:

The **beta function** is:

$$\beta(g) = \frac{dg}{d\ln\mu} = -\frac{b_0 g^3}{(4\pi)^2} + O(g^5)$$

where $b_0 = \frac{11N - 2n_f}{3}$ for $n_f$ fermion flavors.

**Asymptotic Freedom**: For $n_f < \frac{11N}{2}$:
- $\beta(g) < 0$ (gauge coupling decreases at high energy)
- $g(\mu) \to 0$ as $\mu \to \infty$

**Physical Interpretation**: QCD becomes weakly coupled at high energy scales (justifies perturbative QCD).

**Related Results:** `thm-sun-wilson-action`, `conj-mass-gap`

---

### Mass Gap Conjecture

**Type:** Conjecture
**Label:** `conj-mass-gap`
**Source:** [13_E_cst_ig_lattice_qft.md § 5.4](13_fractal_set/13_E_cst_ig_lattice_qft.md)
**Tags:** `mass-gap`, `millennium-prize`, `yang-mills`, `qcd`, `confinement`

**Statement:**
For pure $SU(N)$ Yang-Mills theory in $(d+1)$-dimensional spacetime with $d \geq 3$:

**Mass Gap Conjecture**: There exists a constant $\Delta > 0$ (the mass gap) such that:

$$E_n - E_0 \geq \Delta \quad \forall n \geq 1$$

where $E_n$ are the energy eigenvalues of the Hamiltonian.

**Physical Interpretation**:
- All excitations have mass $\geq \Delta$ (no massless gluons in confinement phase)
- Resolving this conjecture is a Millennium Prize Problem

**Related Results:** `conj-sun-confinement`, `thm-asymptotic-freedom`

---

### Continuum Limit to Lorentzian Manifold

**Type:** Theorem
**Label:** `thm-continuum-limit-lorentzian`
**Source:** [13_C_cst_causal_set_quantum_gravity.md § 5.4](13_fractal_set/13_C_cst_causal_set_quantum_gravity.md)
**Tags:** `continuum-limit`, `lorentzian-manifold`, `general-relativity`, `emergent-geometry`

**Statement:**
For causal set $C$ that is a sprinkling of Lorentzian manifold $(M, g)$:

As $\ell_P \to 0$ (continuum limit), the causal set action converges:

$$S[C] \to S_{\text{EH}}[g] = \int_M (R - 2\Lambda) \sqrt{-g} \, d^{d+1}x$$

where $S_{\text{EH}}$ is the Einstein-Hilbert action with cosmological constant $\Lambda$.

**Interpretation**: General relativity emerges from discrete quantum gravity in the continuum limit.

**Related Results:** `def-benincasa-dowker-action`, `thm-sprinkling-approximation`

---

## Key Inequalities and Bounds

### Summary of N-Uniform Constants

**Source:** [03_cloning.md § 12](03_cloning.md#12-complete-lyapunov-drift)
**Tags:** `constants`, `N-uniform`, `scalability`, `summary`

#### Contraction Rates (All N-uniform, $\epsilon$-dependent)

- **Positional variance:** $\kappa_x(\epsilon) > 0$
- **Boundary potential:** $\kappa_b(\epsilon) > 0$
- **Total Lyapunov (with kinetic):** $\kappa_{\text{total}}(\epsilon) > 0$

#### Expansion Bounds (All state-independent)

- **Velocity variance per step:** $C_v = f_{\text{clone}} \cdot 8(\alpha_{\text{restitution}}^2 + 4) V_{\max}^2$
- **Inter-swarm error per step:** $C_W < \infty$
- **Positional noise:** $C_x < \infty$
- **Boundary noise:** $C_b < \infty$

#### Population Fractions (All N-uniform, $\epsilon$-dependent)

- **High-error fraction:** $f_H(\epsilon) > 0$
- **Unfit fraction:** $\geq 1/2$ (N-uniform, $\epsilon$-independent)
- **Unfit-high-error overlap:** $f_{UH}(\epsilon) = \frac{1}{2}f_H(\epsilon) > 0$

#### Cloning Pressures (All N-uniform, $\epsilon$-dependent)

- **Unfit walker:** $p_u(\epsilon) > 0$
- **Boundary-exposed walker:** $p_b(\varphi) \ge p_u(\epsilon)$ (increasing with $\varphi$)

#### Geometric Separation (All N-uniform, $\epsilon$-dependent)

- **High-error isolation distance:** $D_H(\epsilon)$
- **Low-error clustering radius:** $R_L(\epsilon) = c_d \cdot \epsilon$
- **Separation condition:** $D_H(\epsilon) > R_L(\epsilon)$

---

## Index of Mathematical Objects

**State Space Objects:**
- Swarm configuration: $S \in \Sigma_N$
- Coupled state: $(S_1, S_2) \in \Sigma_N \times \Sigma_N$
- Walker state: $(x_i, v_i, s_i)$
- Alive set: $\mathcal{A}(S_k)$
- Dead set: $\mathcal{D}(S_k)$

**Centers and Deviations:**
- Position barycenter: $\mu_{x,k}$
- Velocity barycenter: $\mu_{v,k}$
- Centered position: $\delta_{x,k,i} = x_{k,i} - \mu_{x,k}$
- Centered velocity: $\delta_{v,k,i} = v_{k,i} - \mu_{v,k}$
- Difference vectors: $\Delta x_i$, $\Delta v_i$, $\Delta \delta_{x,i}$, $\Delta \delta_{v,i}$

**Lyapunov Components:**
- Total Lyapunov: $V_{\text{total}} = V_W + c_V V_{\text{Var}} + c_B W_b$
- Wasserstein distance: $W_h^2(\mu_1, \mu_2) = V_{\text{loc}} + V_{\text{struct}}$
- Location error: $V_{\text{loc}}$
- Structural error: $V_{\text{struct}}$
- Positional variance: $V_{\text{Var},x}$
- Velocity variance: $V_{\text{Var},v}$
- Boundary potential: $W_b$

**Distances and Metrics:**
- Algorithmic distance: $d_{\text{alg}}(i, j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$
- Hypocoercive cost: $c(z_1, z_2) = \|x_1 - x_2\|^2 + \lambda_v \|v_1 - v_2\|^2 + b\langle x_1 - x_2, v_1 - v_2 \rangle$

**Fitness and Measurements:**
- Raw reward: $r_i = R_{\text{pos}}(x_i) - c_{v\_reg} \|v_i\|^2$
- Paired distance: $d_i = d_{\text{alg}}(i, c(i))$
- Rescaled values: $r'_i, d'_i$
- Fitness potential: $V_i = (d'_i)^\beta (r'_i)^\alpha$
- Cloning score: $S_i(c_i)$
- Cloning probability: $p_i$

**Sets and Partitions:**
- High-error set: $H_k(\epsilon)$
- Low-error set: $L_k(\epsilon)$
- Unfit set: $U_k$
- Fit set: $F_k$
- Target set: $I_{\text{target}} = I_{11} \cap U_k \cap H_k(\epsilon)$
- Boundary-exposed set: $B_k(\varphi_{\text{thresh}})$

**Operators:**
- Cloning operator: $\Psi_{\text{clone}}$
- Kinetic operator: $\Psi_{\text{kin}}$
- Total operator: $\Psi_{\text{total}} = \Psi_{\text{kin}} \circ \Psi_{\text{clone}}$
- Companion selection: $\mathcal{C}_i(S)$
- Pairing operator: Greedy or idealized matching

---

## Cross-Reference Map

This map shows the logical dependency structure:

**Foundations → Geometry → Fitness → Keystone → Drift**

```
Axioms (Ch 4) → State Space (Ch 2) → Lyapunov (Ch 3)
                        ↓
              Measurement Pipeline (Ch 5)
                        ↓
              Geometric Partition (Ch 6) → Measurement Variance (Ch 7.2)
                        ↓
              Fitness Correctness (Ch 7) → Stability Condition (Ch 7.5)
                        ↓
              Keystone Lemma (Ch 8) → Cloning Pressure + Error Concentration
                        ↓
              Positional Drift (Ch 10) → V_Var,x contraction
                        ↓
              Boundary Drift (Ch 11) → W_b contraction
                        ↓
              Complete Drift (Ch 12) → Synergy with Kinetic Operator
```

---

---

**End of Mathematical Reference Document**

**Included Documents:**
- ✅ [03_cloning.md](03_cloning.md) - Cloning operator, Keystone Principle, variance/boundary contraction
- ✅ [03_B__wasserstein_contraction.md](03_B__wasserstein_contraction.md) - W₂ contraction via synchronous coupling
- ✅ [04_convergence.md](04_convergence.md) - Kinetic operator, hypocoercivity, complete QSD convergence
- ✅ [05_mean_field.md](05_mean_field.md) - Mean-field limit, McKean-Vlasov PDE, killing rate consistency
- ✅ [06_propagation_chaos.md](06_propagation_chaos.md) - Propagation of chaos, thermodynamic limit, uniqueness
- ✅ [07_adaptative_gas.md](07_adaptative_gas.md) - Adaptive Viscous Fluid Model with ρ-localization, unified measurement pipeline, N-uniform regularity
- ✅ [08_emergent_geometry.md](08_emergent_geometry.md) - Anisotropic diffusion, emergent Riemannian geometry, hypocoercivity with state-dependent diffusion
- ✅ [09_symmetries_adaptive_gas.md](09_symmetries_adaptive_gas.md) - Permutation invariance, Euclidean symmetries, emergent isometries, Noether's theorem, conservation laws
- ✅ [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md) - Gauge group, orbifold structure, braid group topology, holonomy, principal bundles
- ✅ [18_hk_convergence.md](18_hk_convergence.md) - Hellinger-Kantorovich metric convergence with explicit constants, mass contraction via Lyapunov analysis, LSI-based structural variance contraction (path-dependent), kinetic Hellinger contraction via exact mass-shape decomposition and hypocoercivity

**Coverage:** Complete convergence proof chain from N-particle dynamics to mean-field limit, including adaptive extensions with ρ-localized fitness, emergent geometric structure, symmetry analysis, rigorous gauge-theoretic formulation, and Hellinger-Kantorovich metric convergence for birth-death dynamics

**Future Additions:**
- [01_fragile_gas_framework.md](01_fragile_gas_framework.md) - Core framework axioms and definitions
- [02_euclidean_gas.md](02_euclidean_gas.md) - Euclidean Gas specification
- Other specialized documents as needed
