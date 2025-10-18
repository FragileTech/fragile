# Mathematical Glossary
Comprehensive index of all mathematical entries from the Fragile Gas framework.

**Total Entries:** 723 (Chapter 1: 523, Chapter 2: 200)

---

## Table of Contents

### Chapter 1: Euclidean Gas
- [01_fragile_gas_framework.md](#01_fragile_gas_framework) (142 entries)
- [02_euclidean_gas.md](#02_euclidean_gas) (31 entries)
- [03_cloning.md](#03_cloning) (124 entries)
- [04_wasserstein_contraction.md](#04_wasserstein_contraction) (14 entries)
- [05_kinetic_contraction.md](#05_kinetic_contraction) (33 entries)
- [06_convergence.md](#06_convergence) (38 entries)
- [07_mean_field.md](#07_mean_field) (21 entries)
- [08_propagation_chaos.md](#08_propagation_chaos) (34 entries)
- [09_kl_convergence.md](#09_kl_convergence) (77 entries)
- [10_qsd_exchangeability_theory.md](#10_qsd_exchangeability_theory) (9 entries)

### Chapter 2: Geometric Gas
- [11_geometric_gas.md](#11_geometric_gas) (60 entries)
- [12_symmetries_geometric_gas.md](#12_symmetries_geometric_gas) (17 entries)
- [13_geometric_gas_c3_regularity.md](#13_geometric_gas_c3_regularity) (22 entries)
- [14_geometric_gas_c4_regularity.md](#14_geometric_gas_c4_regularity) (19 entries)
- [15_geometric_gas_lsi_proof.md](#15_geometric_gas_lsi_proof) (7 entries)
- [16_convergence_mean_field.md](#16_convergence_mean_field) (35 entries)
- [18_emergent_geometry.md](#18_emergent_geometry) (40 entries)

---

## Chapter 1: Euclidean Gas

**Entries:** 523

---

### Source: 01_fragile_gas_framework.md {#01_fragile_gas_framework}

### Walker
- **Type:** Definition
- **Label:** `def-walker`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Swarm and Swarm State Space
- **Type:** Definition
- **Label:** `def-swarm-and-state-space`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Alive and Dead Sets
- **Type:** Definition
- **Label:** `def-alive-dead-sets`
- **Tags:** viability
- **Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Valid State Space
- **Type:** Definition
- **Label:** `def-valid-state-space`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Ambient Euclidean Structure and Reference Measures
- **Type:** Assumption
- **Label:** `def-ambient-euclidean`
- **Tags:** assumption
- **Source:** [01_fragile_gas_framework.md § A. Foundational & Environmental Parameters](source/1_euclidean_gas/01_fragile_gas_framework)

### Reference Noise and Kernel Families
- **Type:** Definition
- **Label:** `def-reference-measures`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § A. Foundational & Environmental Parameters](source/1_euclidean_gas/01_fragile_gas_framework)

### N-Particle Displacement Pseudometric ($d_{\text{Disp},\mathcal{Y}}$)
- **Type:** Definition
- **Label:** `def-n-particle-displacement-metric`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 1.6](source/1_euclidean_gas/01_fragile_gas_framework)

### Metric quotient of $(\Sigma_N, d_{\text{Disp},\mathcal{Y}})$
- **Type:** Definition
- **Label:** `def-metric-quotient`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 1.6.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Borel image of the projected swarm space
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 1.6.2](source/1_euclidean_gas/01_fragile_gas_framework)

### If $\widehat{\Phi}(\Sigma_N)$ is not closed, replacing it by its closure in $(\mathcal Y\times\{0,1\})^N$ yields a closed (hence complete) subspace. All probability measures considered are supported on $\widehat{\Phi}(\Sigma_N)$, and optimal couplings for costs continuous in $D$ concentrate on the product of supports, so no generality is lost by completing.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.6.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Polishness of the quotient state space and $W_2$
- **Type:** Lemma
- **Label:** `lem-polishness-and-w2`
- **Tags:** lemma, wasserstein
- **Source:** [01_fragile_gas_framework.md § 1.7.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Components of Swarm Displacement
- **Type:** Definition
- **Label:** `def-displacement-components`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.7.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Conditional product structure within a step
- **Type:** Axiom
- **Label:** `def-assumption-instep-independence`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § Assumption A (In‑Step Independence)](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Guaranteed Revival
- **Type:** Axiom
- **Label:** `def-axiom-guaranteed-revival`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Almost‑sure revival under the global constraint
- **Type:** Theorem
- **Label:** `thm-revival-guarantee`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Boundary Regularity
- **Type:** Axiom
- **Label:** `def-axiom-boundary-regularity`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Boundary Smoothness
- **Type:** Axiom
- **Label:** `def-axiom-boundary-smoothness`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Environmental Richness
- **Type:** Axiom
- **Label:** `def-axiom-environmental-richness`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Reward Regularity
- **Type:** Axiom
- **Label:** `def-axiom-reward-regularity`
- **Tags:** axiom, fitness
- **Source:** [01_fragile_gas_framework.md § 2.2.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Projection compatibility
- **Type:** Axiom
- **Label:** `unlabeled`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Algorithmic Diameter
- **Type:** Axiom
- **Label:** `def-axiom-bounded-algorithmic-diameter`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Range‑Respecting Mean
- **Type:** Axiom
- **Label:** `def-axiom-range-respecting-mean`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Valid Noise Measure
- **Type:** Definition
- **Label:** `def-valid-noise-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Sufficient Amplification
- **Type:** Axiom
- **Label:** `def-axiom-sufficient-amplification`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Non-Degenerate Noise
- **Type:** Axiom
- **Label:** `def-axiom-non-degenerate-noise`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Components of Mean-Square Standardization Error
- **Type:** Definition
- **Label:** `def-components-mean-square-standardization-error`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 2.3.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Asymptotic Behavior of the Mean-Square Standardization Error
- **Type:** Theorem
- **Label:** `thm-mean-square-standardization-error`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.3.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Relative Collapse
- **Type:** Axiom
- **Label:** `def-axiom-bounded-relative-collapse`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Deviation from Aggregated Variance
- **Type:** Axiom
- **Label:** `def-axiom-bounded-deviation-variance`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Variance Production
- **Type:** Axiom
- **Label:** `def-axiom-bounded-variance-production`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Geometric Consistency
- **Type:** Axiom
- **Label:** `def-axiom-geometric-consistency`
- **Tags:** axiom, geometry, metric
- **Source:** [01_fragile_gas_framework.md § 2.4.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Theorem of Forced Activity
- **Type:** Theorem
- **Label:** `thm-forced-activity`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.4.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Position‑Only Status Margin
- **Type:** Axiom
- **Label:** `def-axiom-margin-stability`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.4.3](source/1_euclidean_gas/01_fragile_gas_framework)

### This axiom expresses a deterministic stability of the status update in terms of the positional component alone. It is strictly stronger than the trivial consequence of the identity
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 2.4.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Reward Measurement
- **Type:** Definition
- **Label:** `def-reward-measurement`
- **Tags:** fitness
- **Source:** [01_fragile_gas_framework.md § 3.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Perturbation Measure
- **Type:** Definition
- **Label:** `def-perturbation-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 4.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Cloning Measure
- **Type:** Definition
- **Label:** `def-cloning-measure`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 4.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Validation of the Heat Kernel
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Validation of the Uniform Ball Measure
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Uniform‑ball death probability is Lipschitz under finite perimeter
- **Type:** Lemma
- **Label:** `lem-boundary-uniform-ball`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 4.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Projection choice
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 4.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Heat‑kernel death probability is Lipschitz with constant $\lesssim 1/\sigma$
- **Type:** Lemma
- **Label:** `lem-boundary-heat-kernel`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 4.2.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Algorithmic Space
- **Type:** Definition
- **Label:** `def-algorithmic-space-generic`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 5.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Distance Between Positional Measures
- **Type:** Definition
- **Label:** `def-distance-positional-measures`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 5.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Algorithmic Distance
- **Type:** Definition
- **Label:** `def-alg-distance`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 5.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Swarm Aggregation Operator
- **Type:** Definition
- **Label:** `def-swarm-aggregation-operator-axiomatic`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Empirical moments are Lipschitz in L2
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 6.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiomatic Properties of the Empirical Measure Aggregator
- **Type:** Lemma
- **Label:** `lem-empirical-aggregator-properties`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 6.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Smoothed Gaussian Measure
- **Type:** Definition
- **Label:** `def-smoothed-gaussian-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Algorithmic space with cemetery point
- **Type:** Definition
- **Label:** `def-algorithmic-cemetery-extension`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Maximal cemetery distance (design choice)
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Cemetery State Measure
- **Type:** Definition
- **Label:** `def-cemetery-state-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Distance to the Cemetery State
- **Type:** Definition
- **Label:** `def-distance-to-cemetery-state`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Companion Selection Measure
- **Type:** Definition
- **Label:** `def-companion-selection-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 7.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Error from Companion Set Change
- **Type:** Lemma
- **Label:** `lem-set-difference-bound`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 7.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Error from Normalization Change
- **Type:** Lemma
- **Label:** `lem-normalization-difference-bound`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 7.2.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Total Error Bound in Terms of Status Changes
- **Type:** Theorem
- **Label:** `thm-total-error-status-bound`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 7.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of a Well-Behaved Rescale Function
- **Type:** Axiom
- **Label:** `def-axiom-rescale-function`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 8.1.](source/1_euclidean_gas/01_fragile_gas_framework)

### Smooth Piecewise Rescale Function
- **Type:** Definition
- **Label:** `def-asymmetric-rescale-function`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 8.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Existence and Uniqueness of the Smooth Rescale Patch
- **Type:** Lemma
- **Label:** `lem-cubic-patch-uniqueness`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Explicit Coefficients of the Smooth Rescale Patch
- **Type:** Lemma
- **Label:** `lem-cubic-patch-coefficients`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Explicit Form of the Polynomial Patch Derivative
- **Type:** Lemma
- **Label:** `lem-cubic-patch-derivative`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Monotonicity of the Polynomial Patch
- **Type:** Lemma
- **Label:** `lem-polynomial-patch-monotonicity`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)

### This construction is the standard monotone cubic Hermite approach (PCHIP/PCHIM). The global derivative bound $L_P\approx 1.0054$ from §8.2.2.5 provides an explicit Lipschitz constant for the rescale segment.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** lipschitz
- **Source:** [01_fragile_gas_framework.md § 8.2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounds on the Polynomial Patch Derivative
- **Type:** Lemma
- **Label:** `lem-cubic-patch-derivative-bounds`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.5](source/1_euclidean_gas/01_fragile_gas_framework)

### Monotonicity of the Smooth Rescale Function
- **Type:** Lemma
- **Label:** `lem-rescale-monotonicity`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.6](source/1_euclidean_gas/01_fragile_gas_framework)

### Global Lipschitz Continuity of the Smooth Rescale Function
- **Type:** Theorem
- **Label:** `thm-rescale-function-lipschitz`
- **Tags:** lipschitz, theorem
- **Source:** [01_fragile_gas_framework.md § 8.2.2.7](source/1_euclidean_gas/01_fragile_gas_framework)

### Lipschitz constant of the patched standardization
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 8.2.2.8](source/1_euclidean_gas/01_fragile_gas_framework)

### Derivative bound for \sigma\'_{\text{reg}}
- **Type:** Lemma
- **Label:** `lem-sigma-patch-derivative-bound`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 8.2.2.9](source/1_euclidean_gas/01_fragile_gas_framework)

### Lipschitz bound for the variance functional
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)

### Chain‑rule bound for \sigma\'_{\text{reg}}\circ \mathrm{Var}
- **Type:** Corollary
- **Label:** `unlabeled`
- **Tags:** corollary
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)

### Closed‑form bound for $L_{g_A\circ z}$ (empirical aggregator)
- **Type:** Corollary
- **Label:** `unlabeled`
- **Tags:** corollary
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)

### Canonical Logistic Rescale Function
- **Type:** Definition
- **Label:** `def-canonical-logistic-rescale-function-example`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 8.3.1.](source/1_euclidean_gas/01_fragile_gas_framework)

### The Canonical Logistic Function is a Valid Rescale Function
- **Type:** Theorem
- **Label:** `thm-canonical-logistic-validity`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 8.3.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Raw Value Operator
- **Type:** Definition
- **Label:** `def-raw-value-operator`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 9.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Mean-Square Continuity for Raw Values
- **Type:** Axiom
- **Label:** `axiom-raw-value-mean-square-continuity`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 9.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Measurement Variance
- **Type:** Axiom
- **Label:** `axiom-bounded-measurement-variance`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 9.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Distance-to-Companion Measurement
- **Type:** Definition
- **Label:** `def-distance-to-companion-measurement`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 10.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on Single-Walker Error from Positional Change
- **Type:** Lemma
- **Label:** `lem-single-walker-positional-error`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on Single-Walker Error from Structural Change
- **Type:** Lemma
- **Label:** `lem-single-walker-structural-error`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on Single-Walker Error from Own Status Change
- **Type:** Lemma
- **Label:** `lem-single-walker-own-status-error`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Total Squared Error for Unstable Walkers
- **Type:** Lemma
- **Label:** `lem-total-squared-error-unstable`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.5](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Total Squared Error for Stable Walkers
- **Type:** Lemma
- **Label:** `lem-total-squared-error-stable`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.6.](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Total Squared Error for Unstable Walkers
- **Type:** Lemma
- **Label:** `lem-total-squared-error-unstable`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.5](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Total Squared Error for Stable Walkers
- **Type:** Lemma
- **Label:** `lem-total-squared-error-stable`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.6](source/1_euclidean_gas/01_fragile_gas_framework)

### Decomposition of Stable Walker Error
- **Type:** Lemma
- **Label:** `sub-lem-stable-walker-error-decomposition`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.6.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Positional Error Component
- **Type:** Lemma
- **Label:** `sub-lem-stable-positional-error-bound`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.6.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Structural Error Component for Stable Walkers
- **Type:** Lemma
- **Label:** `sub-lem-stable-structural-error-bound`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 10.2.6.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Bound on the Expected Raw Distance Vector Change
- **Type:** Theorem
- **Label:** `thm-expected-raw-distance-bound`
- **Tags:** metric, theorem
- **Source:** [01_fragile_gas_framework.md § 10.2.7](source/1_euclidean_gas/01_fragile_gas_framework)

### Deterministic Behavior of the Expected Raw Distance Vector at $k=1$
- **Type:** Theorem
- **Label:** `thm-expected-raw-distance-k1`
- **Tags:** metric, theorem
- **Source:** [01_fragile_gas_framework.md § 10.2.8](source/1_euclidean_gas/01_fragile_gas_framework)

### The Distance Operator Satisfies the Bounded Variance Axiom
- **Type:** Theorem
- **Label:** `thm-distance-operator-satisfies-bounded-variance-axiom`
- **Tags:** metric, theorem
- **Source:** [01_fragile_gas_framework.md § 10.3.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Mean-Square Continuity of the Distance Operator
- **Type:** Theorem
- **Label:** `thm-distance-operator-mean-square-continuity`
- **Tags:** metric, theorem
- **Source:** [01_fragile_gas_framework.md § 10.3.2](source/1_euclidean_gas/01_fragile_gas_framework)

### N-Dimensional Standardization Operator
- **Type:** Definition
- **Label:** `def-standardization-operator-n-dimensional`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 11.1.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Statistical Properties Measurement
- **Type:** Definition
- **Label:** `def-statistical-properties-measurement`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 11.1.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Value Continuity of Statistical Properties
- **Type:** Lemma
- **Label:** `lem-stats-value-continuity`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 11.1.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Structural Continuity of Statistical Properties
- **Type:** Lemma
- **Label:** `lem-stats-structural-continuity`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 11.1.3](source/1_euclidean_gas/01_fragile_gas_framework)

### General Bound on the Norm of the Standardized Vector
- **Type:** Theorem
- **Label:** `thm-z-score-norm-bound`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.1.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Asymptotic Behavior of the Structural Continuity for the Regularized Standard Deviation
- **Type:** Theorem
- **Label:** `thm-asymptotic-std-dev-structural-continuity`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.1.5](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Expected Squared Value Error
- **Type:** Theorem
- **Label:** `thm-standardization-value-error-mean-square`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Expected Squared Value Error
- **Type:** Theorem
- **Label:** `thm-standardization-value-error-mean-square`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.2.2.6.](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Expected Squared Structural Error
- **Type:** Theorem
- **Label:** `thm-standardization-structural-error-mean-square`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Expected Squared Structural Error
- **Type:** Theorem
- **Label:** `thm-standardization-structural-error-mean-square`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.2.3.5.](source/1_euclidean_gas/01_fragile_gas_framework)

### Decomposition of the Total Standardization Error
- **Type:** Theorem
- **Label:** `thm-deterministic-error-decomposition`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.3.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Algebraic Decomposition of the Value Error
- **Type:** Lemma
- **Label:** `sub-lem-lipschitz-value-error-decomposition`
- **Tags:** lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 11.3.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Squared Value Error
- **Type:** Theorem
- **Label:** `thm-lipschitz-value-error-bound`
- **Tags:** lipschitz, theorem
- **Source:** [01_fragile_gas_framework.md § 11.3.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Value Error Coefficients
- **Type:** Definition
- **Label:** `def-lipschitz-value-error-coefficients`
- **Tags:** lipschitz
- **Source:** [01_fragile_gas_framework.md § 11.3.4](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Squared Structural Error
- **Type:** Theorem
- **Label:** `thm-lipschitz-structural-error-bound`
- **Tags:** lipschitz, theorem
- **Source:** [01_fragile_gas_framework.md § 11.3.5](source/1_euclidean_gas/01_fragile_gas_framework)

### Structural Error Coefficients
- **Type:** Definition
- **Label:** `def-lipschitz-structural-error-coefficients`
- **Tags:** lipschitz
- **Source:** [01_fragile_gas_framework.md § 11.3.6](source/1_euclidean_gas/01_fragile_gas_framework)

### Global Continuity of the Patched Standardization Operator
- **Type:** Theorem
- **Label:** `thm-global-continuity-patched-standardization`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 11.3.7](source/1_euclidean_gas/01_fragile_gas_framework)

### Rescaled Potential Operator for the Alive Set
- **Type:** Definition
- **Label:** `def-alive-set-potential-operator`
- **Tags:** fitness, viability
- **Source:** [01_fragile_gas_framework.md § 12.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Swarm Potential Assembly Operator
- **Type:** Definition
- **Label:** `def-swarm-potential-assembly-operator`
- **Tags:** fitness
- **Source:** [01_fragile_gas_framework.md § 12.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Assume the **Axiom of Margin-Based Status Stability** ([](#def-axiom-margin-stability)). Then for all inputs
- **Type:** Corollary
- **Label:** `unlabeled`
- **Tags:** corollary
- **Source:** [01_fragile_gas_framework.md § 12.3.3](source/1_euclidean_gas/01_fragile_gas_framework)

### Perturbation Operator
- **Type:** Definition
- **Label:** `def-perturbation-operator`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 13.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Axiom of Bounded Second Moment of Perturbation
- **Type:** Axiom
- **Label:** `def-axiom-bounded-second-moment-perturbation`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 13.2.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounding the Output Positional Displacement
- **Type:** Lemma
- **Label:** `sub-lem-perturbation-positional-bound-reproof`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 13.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Bounded differences for $f_{\text{avg}}$
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 13.2.3.0.](source/1_euclidean_gas/01_fragile_gas_framework)

### McDiarmid's Inequality (Bounded Differences Inequality) (Boucheron–Lugosi–Massart)
- **Type:** Theorem
- **Label:** `thm-mcdiarmids-inequality`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 13.2.3.1.](source/1_euclidean_gas/01_fragile_gas_framework)

### Probabilistic Bound on Total Perturbation-Induced Displacement
- **Type:** Lemma
- **Label:** `sub-lem-probabilistic-bound-perturbation-displacement-reproof`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 13.2.3.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Perturbation Fluctuation Bounds
- **Type:** Definition
- **Label:** `def-perturbation-fluctuation-bounds-reproof`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 13.2.4.](source/1_euclidean_gas/01_fragile_gas_framework)

### Probabilistic Continuity of the Perturbation Operator
- **Type:** Theorem
- **Label:** `thm-perturbation-operator-continuity-reproof`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 13.2.5.](source/1_euclidean_gas/01_fragile_gas_framework)

### Status Update Operator
- **Type:** Definition
- **Label:** `def-status-update-operator`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 14.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Probabilistic Continuity of the Post-Perturbation Status Update
- **Type:** Theorem
- **Label:** `thm-post-perturbation-status-update-continuity`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 14.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Cloning Score Function
- **Type:** Definition
- **Label:** `def-cloning-score-function`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 15.1.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Stochastic Threshold Cloning
- **Type:** Definition
- **Label:** `def-stochastic-threshold-cloning`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 15.1.2](source/1_euclidean_gas/01_fragile_gas_framework)

### Total Expected Cloning Action
- **Type:** Definition
- **Label:** `def-total-expected-cloning-action`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 15.2.1.](source/1_euclidean_gas/01_fragile_gas_framework)

### The Conditional Cloning Probability Function
- **Type:** Definition
- **Label:** `def-cloning-probability-function`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 15.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Lipschitz Continuity of the Conditional Cloning Probability Function (case split)
- **Type:** Lemma
- **Label:** `lem-cloning-probability-lipschitz`
- **Tags:** cloning, lemma, lipschitz
- **Source:** [01_fragile_gas_framework.md § 15.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)

### Conditional Expected Cloning Action
- **Type:** Definition
- **Label:** `def-expected-cloning-action`
- **Tags:** cloning
- **Source:** [01_fragile_gas_framework.md § 15.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)

### Continuity of the Conditional Expected Cloning Action
- **Type:** Theorem
- **Label:** `thm-expected-cloning-action-continuity`
- **Tags:** cloning, theorem
- **Source:** [01_fragile_gas_framework.md § 15.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)

### Continuity of the Total Expected Cloning Action
- **Type:** Theorem
- **Label:** `thm-total-expected-cloning-action-continuity`
- **Tags:** cloning, theorem
- **Source:** [01_fragile_gas_framework.md § 15.2.4.](source/1_euclidean_gas/01_fragile_gas_framework)

### Theorem of Guaranteed Revival from a Single Survivor
- **Type:** Theorem
- **Label:** `thm-k1-revival-state`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 16.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Swarm Update Procedure
- **Type:** Definition
- **Label:** `def-swarm-update-procedure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 17.1](source/1_euclidean_gas/01_fragile_gas_framework)

### Local vs global: for $V\in[0,1]$ all sub-linear powers are $\le 1$ and can be absorbed in a constant; for $V\ge 1$ every sub-linear term is bounded above by the term with exponent $p_{\max}$. This is the only global (uniform in $V\ge 0$) way to replace a sum of distinct powers by a single power, and it justifies using $\alpha_H^{\mathrm{global}}=\max(\tfrac12,\alpha_B)$ when aggregating sub-linear exponents.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 17.2.4.3.](source/1_euclidean_gas/01_fragile_gas_framework)

### Wasserstein-2 on the output space (quotient)
- **Type:** Definition
- **Label:** `def-w2-output-metric`
- **Tags:** metric, wasserstein
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)

### W2 continuity bound without offset (for $k\ge 2$)
- **Type:** Proposition
- **Label:** `prop-w2-bound-no-offset`
- **Tags:** proposition
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)

### The offset $K_{\Psi}$ appearing in the expectation-based bound corresponds to allowing arbitrary (e.g., independent) couplings of the output randomness. When the comparison is made in $W_2$—or, operationally, under synchronous coupling—the artificial offset vanishes at zero input distance, yielding a cleaner continuity statement. The composite constants $C_{\Psi,L}$ and $C_{\Psi,H}$ are exactly those defined in [](#def-composite-continuity-coeffs-recorrected) and inherit boundedness/continuity from [](#subsec-coefficient-regularity).
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** metric, wasserstein
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)

### The Swarm Update defines a Markov kernel
- **Type:** Proposition
- **Label:** `prop-psi-markov-kernel`
- **Tags:** proposition
- **Source:** [01_fragile_gas_framework.md § 17.2.4.5.](source/1_euclidean_gas/01_fragile_gas_framework)

### Feller-type (continuity-preserving) properties for $\Psi$ follow from the stagewise measurability and continuity assumptions stated in Section 2 for the operators and aggregators; on compact (or sublevel) sets these imply boundedness and continuity of the induced kernel maps.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 17.2.4.5.](source/1_euclidean_gas/01_fragile_gas_framework)

### Boundedness and continuity of composite coefficients
- **Type:** Proposition
- **Label:** `prop-coefficient-regularity`
- **Tags:** proposition
- **Source:** [01_fragile_gas_framework.md § 17.2.4.6.](source/1_euclidean_gas/01_fragile_gas_framework)

### In particular, on such sublevel sets the $W_2$ continuity bound and the deterministic standardization bounds promote to genuine continuity statements for the composite operators since the constants do not blow up along admissible sequences.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** wasserstein
- **Source:** [01_fragile_gas_framework.md § 17.2.4.6.](source/1_euclidean_gas/01_fragile_gas_framework)

### Fragile Swarm Instantiation
- **Type:** Definition
- **Label:** `def-fragile-swarm-instantiation`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 18.1](source/1_euclidean_gas/01_fragile_gas_framework)

### The Fragile Gas Algorithm
- **Type:** Definition
- **Label:** `def-fragile-gas-algorithm`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 18.2](source/1_euclidean_gas/01_fragile_gas_framework)

---

### Source: 02_euclidean_gas.md {#02_euclidean_gas}

### Euclidean Gas Update
- **Type:** Algorithm
- **Label:** `alg-euclidean-gas`
- **Tags:** algorithm
- **Source:** [02_euclidean_gas.md § **3.1 Euclidean Gas algorithm (canonical pipeline)](source/1_euclidean_gas/02_euclidean_gas)

### Properties of smooth radial squashing maps
- **Type:** Lemma
- **Label:** `lem-squashing-properties-generic`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 3.3](source/1_euclidean_gas/02_euclidean_gas)

### Lipschitz continuity of the projection $\varphi$
- **Type:** Lemma
- **Label:** `lem-projection-lipschitz`
- **Tags:** lemma, lipschitz
- **Source:** [02_euclidean_gas.md § 3.3](source/1_euclidean_gas/02_euclidean_gas)

### Lipschitz property of the kinetic flow
- **Type:** Lemma
- **Label:** `lem-sasaki-kinetic-lipschitz`
- **Tags:** kinetic, lemma, lipschitz
- **Source:** [02_euclidean_gas.md § 4.1](source/1_euclidean_gas/02_euclidean_gas)

### Hölder continuity of the death probability
- **Type:** Lemma
- **Label:** `lem-euclidean-boundary-holder`
- **Tags:** holder, lemma
- **Source:** [02_euclidean_gas.md § 4.1](source/1_euclidean_gas/02_euclidean_gas)

### Reward regularity in the Sasaki metric
- **Type:** Lemma
- **Label:** `lem-euclidean-reward-regularity`
- **Tags:** fitness, lemma, metric
- **Source:** [02_euclidean_gas.md § 4.2](source/1_euclidean_gas/02_euclidean_gas)

### Environmental richness with a kinetic regularizer
- **Type:** Lemma
- **Label:** `lem-euclidean-richness`
- **Tags:** kinetic, lemma
- **Source:** [02_euclidean_gas.md § 4.2](source/1_euclidean_gas/02_euclidean_gas)

### Perturbation second moment in the Sasaki metric
- **Type:** Lemma
- **Label:** `lem-euclidean-perturb-moment`
- **Tags:** lemma, metric
- **Source:** [02_euclidean_gas.md § 4.3](source/1_euclidean_gas/02_euclidean_gas)

### Geometric consistency under the capped kinetic kernel
- **Type:** Lemma
- **Label:** `lem-euclidean-geometric-consistency`
- **Tags:** geometry, kinetic, lemma, metric
- **Source:** [02_euclidean_gas.md § 4.3](source/1_euclidean_gas/02_euclidean_gas)

### Single-walker positional error bound in the Sasaki metric
- **Type:** Lemma
- **Label:** `lem-sasaki-single-walker-positional-error`
- **Tags:** lemma, metric
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Single-walker structural error bound in the Sasaki metric
- **Type:** Lemma
- **Label:** `lem-sasaki-single-walker-structural-error`
- **Tags:** lemma, metric
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Mean-square error on stable walkers (Sasaki)
- **Type:** Lemma
- **Label:** `lem-sasaki-total-squared-error-stable`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Mean-square continuity of the distance measurement (Sasaki)
- **Type:** Theorem
- **Label:** `thm-sasaki-distance-ms`
- **Tags:** metric, theorem
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Value continuity of the empirical moments
- **Type:** Lemma
- **Label:** `lem-sasaki-aggregator-value`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Structural continuity of the empirical moments
- **Type:** Lemma
- **Label:** `lem-sasaki-aggregator-structural`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Lipschitz data for the Sasaki empirical aggregators
- **Type:** Lemma
- **Label:** `lem-sasaki-aggregator-lipschitz`
- **Tags:** lemma, lipschitz
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Standardization constants (Sasaki geometry)
- **Type:** Definition
- **Label:** `def-sasaki-standardization-constants`
- **Tags:** geometry
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)

### Value continuity of patched standardization (Sasaki)
- **Type:** Theorem
- **Label:** `unlabeled`
- **Tags:** theorem
- **Source:** [02_euclidean_gas.md § 2.3.4.](source/1_euclidean_gas/02_euclidean_gas)

### Decomposition of the Value Error
- **Type:** Lemma
- **Label:** `lem-sasaki-value-error-decomposition`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.4.1.](source/1_euclidean_gas/02_euclidean_gas)

### Bound on the Squared Direct Shift Component
- **Type:** Lemma
- **Label:** `lem-sasaki-direct-shift-bound-sq`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.4.2.](source/1_euclidean_gas/02_euclidean_gas)

### Bound on the Squared Mean Shift Component
- **Type:** Lemma
- **Label:** `lem-sasaki-mean-shift-bound-sq`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.4.3.](source/1_euclidean_gas/02_euclidean_gas)

### Bounding the Squared Denominator Shift Component
- **Type:** Lemma
- **Label:** `lem-sasaki-denom-shift-bound-sq`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.4.4.](source/1_euclidean_gas/02_euclidean_gas)

### Value Error Coefficients (Squared Form)
- **Type:** Definition
- **Label:** `def-sasaki-standardization-constants-sq`
- **Tags:** general
- **Source:** [02_euclidean_gas.md § 2.3.5.](source/1_euclidean_gas/02_euclidean_gas)

### Structural Continuity of Patched Standardization (Sasaki)
- **Type:** Theorem
- **Label:** `thm-sasaki-standardization-structural-sq`
- **Tags:** theorem
- **Source:** [02_euclidean_gas.md § 2.3.6.](source/1_euclidean_gas/02_euclidean_gas)

### Decomposition of the Structural Error
- **Type:** Lemma
- **Label:** `lem-sasaki-structural-error-decomposition`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.6.1.](source/1_euclidean_gas/02_euclidean_gas)

### Bound on the Squared Direct Structural Error
- **Type:** Lemma
- **Label:** `lem-sasaki-direct-structural-error-sq`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.6.2.](source/1_euclidean_gas/02_euclidean_gas)

### Bound on the Squared Indirect Structural Error
- **Type:** Lemma
- **Label:** `lem-sasaki-indirect-structural-error-sq`
- **Tags:** lemma
- **Source:** [02_euclidean_gas.md § 2.3.6.3.](source/1_euclidean_gas/02_euclidean_gas)

### Composite Continuity of the Patched Standardization Operator (Sasaki)
- **Type:** Theorem
- **Label:** `thm-sasaki-standardization-composite-sq`
- **Tags:** theorem
- **Source:** [02_euclidean_gas.md § 2.3.8.](source/1_euclidean_gas/02_euclidean_gas)

### Lipschitz continuity of patched standardization (Sasaki)
- **Type:** Lemma
- **Label:** `lem-sasaki-standardization-lipschitz`
- **Tags:** lemma, lipschitz
- **Source:** [02_euclidean_gas.md § 2.3.8.](source/1_euclidean_gas/02_euclidean_gas)

### Axiom of Non-Deceptive Landscapes
- **Type:** Axiom
- **Label:** `def-axiom-non-deceptive`
- **Tags:** axiom
- **Source:** [02_euclidean_gas.md § 2.6](source/1_euclidean_gas/02_euclidean_gas)

### Feller continuity of $\Psi_{\mathcal F_{\mathrm{EG}}}$
- **Type:** Theorem
- **Label:** `thm-euclidean-feller`
- **Tags:** theorem
- **Source:** [02_euclidean_gas.md § Appendix A. Proof of the Feller Property for the E](source/1_euclidean_gas/02_euclidean_gas)

---

### Source: 03_cloning.md {#03_cloning}

### Single-Walker and Swarm State Spaces
- **Type:** Definition
- **Label:** `def-single-swarm-space`
- **Tags:** general
- **Source:** [03_cloning.md § 2.1.](source/1_euclidean_gas/03_cloning)

### The Coupled State Space
- **Type:** Definition
- **Label:** `def-coupled-state-space`
- **Tags:** general
- **Source:** [03_cloning.md § 2.2.](source/1_euclidean_gas/03_cloning)

### State Difference Vectors
- **Type:** Definition
- **Label:** `def-state-difference-vectors`
- **Tags:** general
- **Source:** [03_cloning.md § 2.3.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-0): Regularity of the Domain**
- **Type:** Axiom
- **Label:** `ax:domain-regularity`
- **Tags:** axiom
- **Source:** [03_cloning.md § 2.4.1.](source/1_euclidean_gas/03_cloning)

### Existence of a Global Smooth Barrier Function
- **Type:** Proposition
- **Label:** `prop-barrier-existence`
- **Tags:** proposition
- **Source:** [03_cloning.md § 2.4.2.](source/1_euclidean_gas/03_cloning)

### Barycentres and Centered Vectors (Alive Walkers Only)
- **Type:** Definition
- **Label:** `def-barycentres-and-centered-vectors`
- **Tags:** viability
- **Source:** [03_cloning.md § 3.1.](source/1_euclidean_gas/03_cloning)

### The Location Error Component ($V_{\text{loc}}$)
- **Type:** Definition
- **Label:** `def-location-error-component`
- **Tags:** general
- **Source:** [03_cloning.md § 3.2.1.](source/1_euclidean_gas/03_cloning)

### The Structural Error Component ($V_{\text{struct}}$)
- **Type:** Definition
- **Label:** `def-structural-error-component`
- **Tags:** general
- **Source:** [03_cloning.md § 3.2.2.](source/1_euclidean_gas/03_cloning)

### Decomposition of the Hypocoercive Wasserstein Distance
- **Type:** Lemma
- **Label:** `lem-wasserstein-decomposition`
- **Tags:** lemma, metric, wasserstein
- **Source:** [03_cloning.md § 3.2.3.](source/1_euclidean_gas/03_cloning)

### Structural Positional Error and Internal Variance
- **Type:** Lemma
- **Label:** `lem-sx-implies-variance`
- **Tags:** lemma
- **Source:** [03_cloning.md § 3.2.4](source/1_euclidean_gas/03_cloning)

### The Full Synergistic Hypocoercive Lyapunov Function
- **Type:** Definition
- **Label:** `def-full-synergistic-lyapunov-function`
- **Tags:** general
- **Source:** [03_cloning.md § 3.3.](source/1_euclidean_gas/03_cloning)

### Variance Notation Conversion Formulas
- **Type:** Definition
- **Label:** `def-variance-conversions`
- **Tags:** general
- **Source:** [03_cloning.md § 3.3.1.](source/1_euclidean_gas/03_cloning)

### Necessity of the Augmented Lyapunov Structure
- **Type:** Proposition
- **Label:** `prop-lyapunov-necessity`
- **Tags:** proposition
- **Source:** [03_cloning.md § 3.3.2.](source/1_euclidean_gas/03_cloning)

### Analogy to Classical Hypocoercivity Theory
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [03_cloning.md § 3.3.2.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-1): Lipschitz Regularity of Environmental Fields**
- **Type:** Axiom
- **Label:** `ax:lipschitz-fields`
- **Tags:** axiom, lipschitz
- **Source:** [03_cloning.md § 4.1.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-2): Existence of a Safe Harbor**
- **Type:** Axiom
- **Label:** `ax:safe-harbor`
- **Tags:** axiom
- **Source:** [03_cloning.md § 4.1.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-3): Non-Deceptive Landscape**
- **Type:** Axiom
- **Label:** `ax:non-deceptive-landscape`
- **Tags:** axiom
- **Source:** [03_cloning.md § 4.2.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-4): Velocity Regularization via Reward**
- **Type:** Axiom
- **Label:** `ax:velocity-regularization`
- **Tags:** axiom, fitness, kinetic
- **Source:** [03_cloning.md § 4.3.](source/1_euclidean_gas/03_cloning)

### **(Axiom EG-5): Active Diversity Signal**
- **Type:** Axiom
- **Label:** `ax:active-diversity`
- **Tags:** axiom
- **Source:** [03_cloning.md § 4.3.](source/1_euclidean_gas/03_cloning)

### Algorithmic Distance for Companion Selection
- **Type:** Definition
- **Label:** `def-algorithmic-distance-metric`
- **Tags:** metric
- **Source:** [03_cloning.md § 5.0.](source/1_euclidean_gas/03_cloning)

### Spatially-Aware Pairing Operator (Idealized Model)
- **Type:** Definition
- **Label:** `def-spatial-pairing-operator-diversity`
- **Tags:** general
- **Source:** [03_cloning.md § 5.1.1.](source/1_euclidean_gas/03_cloning)

### Sequential Stochastic Greedy Pairing Operator
- **Type:** Definition
- **Label:** `def-greedy-pairing-algorithm`
- **Tags:** general
- **Source:** [03_cloning.md § 5.1.2.](source/1_euclidean_gas/03_cloning)

### Sequential Stochastic Greedy Pairing Algorithm
- **Type:** Algorithm
- **Label:** `unlabeled`
- **Tags:** algorithm
- **Source:** [03_cloning.md § 5.1.2.](source/1_euclidean_gas/03_cloning)

### Geometric Partitioning of High-Variance Swarms
- **Type:** Definition
- **Label:** `def-geometric-partition`
- **Tags:** geometry, metric
- **Source:** [03_cloning.md § 5.1.3.](source/1_euclidean_gas/03_cloning)

### Greedy Pairing Guarantees Signal Separation
- **Type:** Lemma
- **Label:** `lem-greedy-preserves-signal`
- **Tags:** lemma
- **Source:** [03_cloning.md § 5.1.3.](source/1_euclidean_gas/03_cloning)

### Raw Value Operators
- **Type:** Definition
- **Label:** `def-raw-value-operators`
- **Tags:** general
- **Source:** [03_cloning.md § 5.2.](source/1_euclidean_gas/03_cloning)

### Swarm Aggregation Operator
- **Type:** Definition
- **Label:** `def-swarm-aggregation-operator`
- **Tags:** general
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)

### Patched Standard Deviation Function
- **Type:** Definition
- **Label:** `def-patched-std-dev-function`
- **Tags:** general
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)

### Properties of the Patching Function
- **Type:** Lemma
- **Label:** `lem-patching-properties`
- **Tags:** lemma
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)

### N-Dimensional Standardization Operator
- **Type:** Definition
- **Label:** `def-standardization-operator`
- **Tags:** general
- **Source:** [03_cloning.md § 5.4.](source/1_euclidean_gas/03_cloning)

### Compact Support of Standardized Scores
- **Type:** Lemma
- **Label:** `lem-compact-support-z-scores`
- **Tags:** lemma
- **Source:** [03_cloning.md § 5.4.](source/1_euclidean_gas/03_cloning)

### Canonical Logistic Rescale Function
- **Type:** Definition
- **Label:** `def-logistic-rescale`
- **Tags:** general
- **Source:** [03_cloning.md § 5.5.](source/1_euclidean_gas/03_cloning)

### Verification of Axiomatic Properties
- **Type:** Lemma
- **Label:** `lem-logistic-properties`
- **Tags:** lemma
- **Source:** [03_cloning.md § 5.5.](source/1_euclidean_gas/03_cloning)

### Fitness Potential Operator
- **Type:** Definition
- **Label:** `def-fitness-potential-operator`
- **Tags:** fitness
- **Source:** [03_cloning.md § 5.6.](source/1_euclidean_gas/03_cloning)

### Uniform Bounds of the Fitness Potential
- **Type:** Lemma
- **Label:** `lem-potential-bounds`
- **Tags:** fitness, lemma
- **Source:** [03_cloning.md § 5.6.](source/1_euclidean_gas/03_cloning)

### Companion Selection Operator for Cloning
- **Type:** Definition
- **Label:** `def-cloning-companion-operator`
- **Tags:** cloning
- **Source:** [03_cloning.md § 5.7.1](source/1_euclidean_gas/03_cloning)

### The Canonical Cloning Score
- **Type:** Definition
- **Label:** `def-cloning-score`
- **Tags:** cloning
- **Source:** [03_cloning.md § 5.7.2](source/1_euclidean_gas/03_cloning)

### Total Cloning Probability
- **Type:** Definition
- **Label:** `def-cloning-probability`
- **Tags:** cloning
- **Source:** [03_cloning.md § 5.7.2](source/1_euclidean_gas/03_cloning)

### The Stochastic Cloning Decision
- **Type:** Definition
- **Label:** `def-cloning-decision`
- **Tags:** cloning
- **Source:** [03_cloning.md § 5.7.3](source/1_euclidean_gas/03_cloning)

### The Inelastic Collision State Update
- **Type:** Definition
- **Label:** `def-inelastic-collision-update`
- **Tags:** general
- **Source:** [03_cloning.md § 5.7.4.](source/1_euclidean_gas/03_cloning)

### Bounded Velocity Variance Expansion from Cloning
- **Type:** Proposition
- **Label:** `prop-bounded-velocity-expansion`
- **Tags:** cloning, kinetic, proposition
- **Source:** [03_cloning.md § 5.7.5.](source/1_euclidean_gas/03_cloning)

### Large $V_{\text{Var},x}$ Implies Large Single-Swarm Positional Variance
- **Type:** Lemma
- **Label:** `lem-V_Varx-implies-variance`
- **Tags:** lemma
- **Source:** [03_cloning.md § 6.2.](source/1_euclidean_gas/03_cloning)

### The Unified High-Error and Low-Error Sets
- **Type:** Definition
- **Label:** `def-unified-high-low-error-sets`
- **Tags:** general
- **Source:** [03_cloning.md § 6.3](source/1_euclidean_gas/03_cloning)

### The Phase-Space Packing Lemma
- **Type:** Lemma
- **Label:** `lem-phase-space-packing`
- **Tags:** lemma
- **Source:** [03_cloning.md § 6.4.1.](source/1_euclidean_gas/03_cloning)

### Positional Variance as a Lower Bound for Hypocoercive Variance
- **Type:** Lemma
- **Label:** `lem-var-x-implies-var-h`
- **Tags:** lemma
- **Source:** [03_cloning.md § 6.4.2](source/1_euclidean_gas/03_cloning)

### N-Uniform Lower Bound on the Outlier Fraction
- **Type:** Lemma
- **Label:** `lem-outlier-fraction-lower-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 6.4.2](source/1_euclidean_gas/03_cloning)

### N-Uniform Lower Bound on the Outlier-Cluster Fraction
- **Type:** Lemma
- **Label:** `lem-outlier-cluster-fraction-lower-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 6.4.3.](source/1_euclidean_gas/03_cloning)

### A Large Intra-Swarm Positional Variance Guarantees a Non-Vanishing High-Error Fraction
- **Type:** Corollary
- **Label:** `cor-vvarx-to-high-error-fraction`
- **Tags:** corollary
- **Source:** [03_cloning.md § 6.4.4](source/1_euclidean_gas/03_cloning)

### Geometric Separation of the Partition
- **Type:** Lemma
- **Label:** `lem-geometric-separation-of-partition`
- **Tags:** geometry, lemma, metric
- **Source:** [03_cloning.md § 6.5.1.](source/1_euclidean_gas/03_cloning)

### Geometric Structure Guarantees Measurement Variance
- **Type:** Theorem
- **Label:** `thm-geometry-guarantees-variance`
- **Tags:** geometry, metric, theorem
- **Source:** [03_cloning.md § 7.2.1](source/1_euclidean_gas/03_cloning)

### **(Satisfiability of the Signal-to-Noise Condition via Signal Gain)**
- **Type:** Proposition
- **Label:** `prop-satisfiability-of-snr-gamma`
- **Tags:** proposition
- **Source:** [03_cloning.md § 7.2.2.](source/1_euclidean_gas/03_cloning)

### From Bounded Variance to a Guaranteed Gap
- **Type:** Lemma
- **Label:** `lem-variance-to-gap`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.3.1.](source/1_euclidean_gas/03_cloning)

### Maximum Patched Standard Deviation
- **Type:** Definition
- **Label:** `def-max-patched-std`
- **Tags:** general
- **Source:** [03_cloning.md § 7.3.2.1.](source/1_euclidean_gas/03_cloning)

### Positive Derivative Bound for the Rescale Function
- **Type:** Lemma
- **Label:** `lem-rescale-derivative-lower-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.3.2.1.](source/1_euclidean_gas/03_cloning)

### From Raw Measurement Gap to Rescaled Value Gap
- **Type:** Lemma
- **Label:** `lem-raw-gap-to-rescaled-gap`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.3.2.2.](source/1_euclidean_gas/03_cloning)

### **(From Total Variance to Mean Separation)**
- **Type:** Lemma
- **Label:** `lem-variance-to-mean-separation`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.4.1](source/1_euclidean_gas/03_cloning)

### Derivation of the Stability Condition for Intelligent Adaptation
- **Type:** Theorem
- **Label:** `thm-derivation-of-stability-condition`
- **Tags:** theorem
- **Source:** [03_cloning.md § 7.4.2](source/1_euclidean_gas/03_cloning)

### Lower Bound on Logarithmic Mean Gap
- **Type:** Lemma
- **Label:** `lem-log-gap-lower-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.5.1.1.](source/1_euclidean_gas/03_cloning)

### On the Tightness of the Bound
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [03_cloning.md § 7.5.1.1.](source/1_euclidean_gas/03_cloning)

### Upper Bound on Logarithmic Mean Gap
- **Type:** Lemma
- **Label:** `lem-log-gap-upper-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.5.1.2.](source/1_euclidean_gas/03_cloning)

### Why the Bound is Tight at $V_{\min}$
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [03_cloning.md § 7.5.1.2.](source/1_euclidean_gas/03_cloning)

### **(Lower Bound on the Corrective Diversity Signal)**
- **Type:** Proposition
- **Label:** `prop-corrective-signal-bound`
- **Tags:** proposition
- **Source:** [03_cloning.md § 7.5.2.1.](source/1_euclidean_gas/03_cloning)

### **(Worst-Case Upper Bound on the Adversarial Reward Signal)**
- **Type:** Proposition
- **Label:** `prop-adversarial-signal-bound-naive`
- **Tags:** fitness, proposition
- **Source:** [03_cloning.md § 7.5.2.2](source/1_euclidean_gas/03_cloning)

### **(Lipschitz Bound on the Raw Reward Mean Gap)**
- **Type:** Proposition
- **Label:** `prop-raw-reward-mean-gap-bound`
- **Tags:** fitness, lipschitz, proposition
- **Source:** [03_cloning.md § 7.5.2.3.](source/1_euclidean_gas/03_cloning)

### **(Axiom-Based Bound on the Logarithmic Reward Gap)**
- **Type:** Proposition
- **Label:** `prop-log-reward-gap-axiom-bound`
- **Tags:** fitness, proposition
- **Source:** [03_cloning.md § 7.5.2.3.](source/1_euclidean_gas/03_cloning)

### **(The Corrected Stability Condition for Intelligent Adaptation)**
- **Type:** Theorem
- **Label:** `thm-stability-condition-final-corrected`
- **Tags:** theorem
- **Source:** [03_cloning.md § 7.5.2.4.](source/1_euclidean_gas/03_cloning)

### The Unfit Set
- **Type:** Definition
- **Label:** `def-unfit-set`
- **Tags:** general
- **Source:** [03_cloning.md § 7.6.1](source/1_euclidean_gas/03_cloning)

### N-Uniform Lower Bound on the Unfit Fraction
- **Type:** Lemma
- **Label:** `lem-unfit-fraction-lower-bound`
- **Tags:** lemma
- **Source:** [03_cloning.md § 7.6.1](source/1_euclidean_gas/03_cloning)

### N-Uniform Lower Bound on the Unfit-High-Error Overlap Fraction
- **Type:** Theorem
- **Label:** `thm-unfit-high-error-overlap-fraction`
- **Tags:** theorem
- **Source:** [03_cloning.md § 7.6.2](source/1_euclidean_gas/03_cloning)

### The N-Uniform Quantitative Keystone Lemma
- **Type:** Lemma
- **Label:** `lem-quantitative-keystone`
- **Tags:** lemma
- **Source:** [03_cloning.md § 8.1](source/1_euclidean_gas/03_cloning)

### The Critical Target Set
- **Type:** Definition
- **Label:** `def-critical-target-set`
- **Tags:** general
- **Source:** [03_cloning.md § 8.2](source/1_euclidean_gas/03_cloning)

### Lower Bound on Mean Companion Fitness Gap
- **Type:** Lemma
- **Label:** `lem-mean-companion-fitness-gap`
- **Tags:** fitness, lemma
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)

### N-Uniformity of the Bound
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)

### Guaranteed Cloning Pressure on the Unfit Set
- **Type:** Lemma
- **Label:** `lem-unfit-cloning-pressure`
- **Tags:** cloning, lemma
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)

### Cloning Pressure on the Target Set
- **Type:** Corollary
- **Label:** `cor-cloning-pressure-target-set`
- **Tags:** cloning, corollary
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)

### **(Variance Concentration in the High-Error Set)**
- **Type:** Lemma
- **Label:** `lem-variance-concentration-Hk`
- **Tags:** lemma
- **Source:** [03_cloning.md § 8.4](source/1_euclidean_gas/03_cloning)

### Error Concentration in the Target Set
- **Type:** Lemma
- **Label:** `lem-error-concentration-target-set`
- **Tags:** lemma
- **Source:** [03_cloning.md § 8.4](source/1_euclidean_gas/03_cloning)

### N-Uniformity of Keystone Constants
- **Type:** Proposition
- **Label:** `prop-n-uniformity-keystone`
- **Tags:** proposition
- **Source:** [03_cloning.md § 8.6.3](source/1_euclidean_gas/03_cloning)

### The Cloning Operator $\Psi_{\text{clone}}$
- **Type:** Definition
- **Label:** `def-cloning-operator-formal`
- **Tags:** cloning
- **Source:** [03_cloning.md § 9.2.](source/1_euclidean_gas/03_cloning)

### The Measurement Operator
- **Type:** Definition
- **Label:** `def-measurement-operator`
- **Tags:** general
- **Source:** [03_cloning.md § 9.3.1.](source/1_euclidean_gas/03_cloning)

### Stochastic Coupling for Drift Analysis
- **Type:** Remark
- **Label:** `rem-measurement-coupling`
- **Tags:** general
- **Source:** [03_cloning.md § 9.3.1.](source/1_euclidean_gas/03_cloning)

### The Fitness Evaluation Operator
- **Type:** Definition
- **Label:** `def-fitness-operator`
- **Tags:** fitness
- **Source:** [03_cloning.md § 9.3.2.](source/1_euclidean_gas/03_cloning)

### The Cloning Decision Operator
- **Type:** Definition
- **Label:** `def-decision-operator`
- **Tags:** cloning
- **Source:** [03_cloning.md § 9.3.3.](source/1_euclidean_gas/03_cloning)

### Total Cloning Probability for Dead Walkers
- **Type:** Lemma
- **Label:** `lem-dead-walker-clone-prob`
- **Tags:** cloning, lemma
- **Source:** [03_cloning.md § 9.3.3.](source/1_euclidean_gas/03_cloning)

### The State Update Operator
- **Type:** Definition
- **Label:** `def-update-operator`
- **Tags:** general
- **Source:** [03_cloning.md § 9.3.4.](source/1_euclidean_gas/03_cloning)

### Position Jitter vs. Velocity Collision Model
- **Type:** Remark
- **Label:** `rem-position-velocity-update-difference`
- **Tags:** kinetic
- **Source:** [03_cloning.md § 9.3.4.](source/1_euclidean_gas/03_cloning)

### Compositional Structure of $\Psi_{\text{clone}}$
- **Type:** Theorem
- **Label:** `thm-cloning-operator-composition`
- **Tags:** cloning, theorem
- **Source:** [03_cloning.md § 9.4.](source/1_euclidean_gas/03_cloning)

### Key Operator Outputs
- **Type:** Definition
- **Label:** `def-key-operator-outputs`
- **Tags:** general
- **Source:** [03_cloning.md § 9.5.](source/1_euclidean_gas/03_cloning)

### Expected Displacement Under Cloning
- **Type:** Proposition
- **Label:** `prop-expected-displacement-cloning`
- **Tags:** cloning, proposition
- **Source:** [03_cloning.md § 9.5.](source/1_euclidean_gas/03_cloning)

### Coupled Cloning Expectation
- **Type:** Definition
- **Label:** `def-coupled-cloning-expectation`
- **Tags:** cloning
- **Source:** [03_cloning.md § 10.2.](source/1_euclidean_gas/03_cloning)

### Synchronous Coupling Benefits
- **Type:** Remark
- **Label:** `rem-coupling-benefits`
- **Tags:** general
- **Source:** [03_cloning.md § 10.2.](source/1_euclidean_gas/03_cloning)

### Positional Variance Contraction Under Cloning
- **Type:** Theorem
- **Label:** `thm-positional-variance-contraction`
- **Tags:** cloning, theorem
- **Source:** [03_cloning.md § 10.3.1.](source/1_euclidean_gas/03_cloning)

### Variance Change Decomposition
- **Type:** Lemma
- **Label:** `lem-variance-change-decomposition`
- **Tags:** lemma
- **Source:** [03_cloning.md § 10.3.3.](source/1_euclidean_gas/03_cloning)

### Keystone-Driven Contraction for Stably Alive Walkers
- **Type:** Lemma
- **Label:** `lem-keystone-contraction-alive`
- **Tags:** lemma, viability
- **Source:** [03_cloning.md § 10.3.4.](source/1_euclidean_gas/03_cloning)

### Bounded Contribution from Dead Walker Revival
- **Type:** Lemma
- **Label:** `lem-dead-walker-revival-bounded`
- **Tags:** lemma
- **Source:** [03_cloning.md § 10.3.5.](source/1_euclidean_gas/03_cloning)

### Bounded Velocity Variance Expansion from Cloning
- **Type:** Theorem
- **Label:** `thm-velocity-variance-bounded-expansion`
- **Tags:** cloning, kinetic, theorem
- **Source:** [03_cloning.md § 10.4.](source/1_euclidean_gas/03_cloning)

### Synergistic Dissipation Enables Net Contraction
- **Type:** Remark
- **Label:** `rem-synergistic-velocity-dissipation`
- **Tags:** kinetic
- **Source:** [03_cloning.md § 10.4.1.](source/1_euclidean_gas/03_cloning)

### Structural Error Contraction
- **Type:** Corollary
- **Label:** `cor-structural-error-contraction`
- **Tags:** corollary
- **Source:** [03_cloning.md § 10.5.](source/1_euclidean_gas/03_cloning)

### Complete Variance Drift Characterization for Cloning
- **Type:** Theorem
- **Label:** `thm-complete-variance-drift`
- **Tags:** cloning, theorem
- **Source:** [03_cloning.md § 10.6.](source/1_euclidean_gas/03_cloning)

### Constants and Parameter Dependencies
- **Type:** Remark
- **Label:** `rem-drift-constants-dependencies`
- **Tags:** general
- **Source:** [03_cloning.md § 10.6.](source/1_euclidean_gas/03_cloning)

### Boundary Potential Component (Recall)
- **Type:** Definition
- **Label:** `def-boundary-potential-recall`
- **Tags:** fitness
- **Source:** [03_cloning.md § 11.2.1.](source/1_euclidean_gas/03_cloning)

### Barrier Function as Geometric Penalty
- **Type:** Remark
- **Label:** `rem-barrier-geometric-penalty`
- **Tags:** geometry, metric
- **Source:** [03_cloning.md § 11.2.1.](source/1_euclidean_gas/03_cloning)

### Fitness Gradient from Boundary Proximity
- **Type:** Lemma
- **Label:** `lem-fitness-gradient-boundary`
- **Tags:** fitness, lemma
- **Source:** [03_cloning.md § 11.2.2.](source/1_euclidean_gas/03_cloning)

### The Boundary-Exposed Set
- **Type:** Definition
- **Label:** `def-boundary-exposed-set`
- **Tags:** general
- **Source:** [03_cloning.md § 11.2.3.](source/1_euclidean_gas/03_cloning)

### Relationship to Total Boundary Potential
- **Type:** Remark
- **Label:** `rem-boundary-mass-relationship`
- **Tags:** fitness
- **Source:** [03_cloning.md § 11.2.3.](source/1_euclidean_gas/03_cloning)

### Boundary Potential Contraction Under Cloning
- **Type:** Theorem
- **Label:** `thm-boundary-potential-contraction`
- **Tags:** cloning, fitness, theorem
- **Source:** [03_cloning.md § 11.3.](source/1_euclidean_gas/03_cloning)

### Interpretation: Progressive Safety Enhancement
- **Type:** Remark
- **Label:** `rem-progressive-safety`
- **Tags:** general
- **Source:** [03_cloning.md § 11.3.](source/1_euclidean_gas/03_cloning)

### Enhanced Cloning Probability Near Boundary
- **Type:** Lemma
- **Label:** `lem-boundary-enhanced-cloning`
- **Tags:** cloning, lemma
- **Source:** [03_cloning.md § 11.4.1.](source/1_euclidean_gas/03_cloning)

### Expected Barrier Reduction for Cloned Walker
- **Type:** Lemma
- **Label:** `lem-barrier-reduction-cloning`
- **Tags:** cloning, lemma
- **Source:** [03_cloning.md § 11.4.2.](source/1_euclidean_gas/03_cloning)

### Bounded Boundary Exposure in Equilibrium
- **Type:** Corollary
- **Label:** `cor-bounded-boundary-exposure`
- **Tags:** corollary
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)

### Exponentially Suppressed Extinction Probability
- **Type:** Corollary
- **Label:** `cor-extinction-suppression`
- **Tags:** corollary
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)

### Safety Margin and Parameter Tuning
- **Type:** Remark
- **Label:** `rem-safety-margin-tuning`
- **Tags:** general
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)

### Complete Boundary Potential Drift Characterization
- **Type:** Theorem
- **Label:** `thm-complete-boundary-drift`
- **Tags:** fitness, theorem
- **Source:** [03_cloning.md § 11.6.](source/1_euclidean_gas/03_cloning)

### Bounded Expansion of Inter-Swarm Wasserstein Distance
- **Type:** Theorem
- **Label:** `thm-inter-swarm-bounded-expansion`
- **Tags:** metric, theorem, wasserstein
- **Source:** [03_cloning.md § 12.2.1.](source/1_euclidean_gas/03_cloning)

### Why Inter-Swarm Error Doesn't Contract Under Cloning
- **Type:** Remark
- **Label:** `rem-why-vw-expands`
- **Tags:** cloning
- **Source:** [03_cloning.md § 12.2.1.](source/1_euclidean_gas/03_cloning)

### Component-Wise Bounds on Inter-Swarm Error
- **Type:** Corollary
- **Label:** `cor-component-bounds-vw`
- **Tags:** corollary
- **Source:** [03_cloning.md § 12.2.2.](source/1_euclidean_gas/03_cloning)

### Complete Wasserstein Decomposition Drift
- **Type:** Theorem
- **Label:** `thm-complete-wasserstein-drift`
- **Tags:** theorem, wasserstein
- **Source:** [03_cloning.md § 12.2.2.](source/1_euclidean_gas/03_cloning)

### Complete Drift Inequality for the Cloning Operator
- **Type:** Theorem
- **Label:** `thm-complete-cloning-drift`
- **Tags:** cloning, theorem
- **Source:** [03_cloning.md § 12.3.1.](source/1_euclidean_gas/03_cloning)

### Necessity of the Kinetic Operator
- **Type:** Proposition
- **Label:** `prop-kinetic-necessity`
- **Tags:** kinetic, proposition
- **Source:** [03_cloning.md § 12.3.3.](source/1_euclidean_gas/03_cloning)

### Perfect Complementarity
- **Type:** Remark
- **Label:** `rem-perfect-complementarity`
- **Tags:** general
- **Source:** [03_cloning.md § 12.4.1.](source/1_euclidean_gas/03_cloning)

### Synergistic Foster-Lyapunov Condition (Preview)
- **Type:** Theorem
- **Label:** `thm-synergistic-foster-lyapunov-preview`
- **Tags:** theorem
- **Source:** [03_cloning.md § 12.4.2.](source/1_euclidean_gas/03_cloning)

### Existence of Valid Coupling Constants
- **Type:** Proposition
- **Label:** `prop-coupling-constant-existence`
- **Tags:** proposition
- **Source:** [03_cloning.md § 12.4.3.](source/1_euclidean_gas/03_cloning)

### Tuning Guidance
- **Type:** Remark
- **Label:** `rem-tuning-guidance`
- **Tags:** general
- **Source:** [03_cloning.md § 12.4.3.](source/1_euclidean_gas/03_cloning)

### Main Results of the Cloning Analysis (Summary)
- **Type:** Theorem
- **Label:** `thm-main-results-summary`
- **Tags:** cloning, theorem
- **Source:** [03_cloning.md § 12.5.1.](source/1_euclidean_gas/03_cloning)

---

### Source: 04_wasserstein_contraction.md {#04_wasserstein_contraction}

### Target Set and Complement
- **Type:** Definition
- **Label:** `def-target-complement`
- **Tags:** general
- **Source:** [04_wasserstein_contraction.md § 2.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Why These Sets?
- **Type:** Remark
- **Label:** `rem-why-target-sets`
- **Tags:** general
- **Source:** [04_wasserstein_contraction.md § 2.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Cluster-Preserving Coupling
- **Type:** Definition
- **Label:** `def-cluster-coupling`
- **Tags:** general
- **Source:** [04_wasserstein_contraction.md § 2.2.](source/1_euclidean_gas/04_wasserstein_contraction)

### Why This Coupling Works
- **Type:** Remark
- **Label:** `rem-coupling-advantages`
- **Tags:** general
- **Source:** [04_wasserstein_contraction.md § 2.2.](source/1_euclidean_gas/04_wasserstein_contraction)

### Variance Decomposition by Clusters
- **Type:** Lemma
- **Label:** `lem-variance-decomposition`
- **Tags:** lemma
- **Source:** [04_wasserstein_contraction.md § 3.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Between-Group Variance Dominance
- **Type:** Corollary
- **Label:** `cor-between-group-dominance`
- **Tags:** corollary
- **Source:** [04_wasserstein_contraction.md § 3.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Cross-Swarm Distance Decomposition
- **Type:** Lemma
- **Label:** `lem-cross-swarm-distance`
- **Tags:** lemma, metric
- **Source:** [04_wasserstein_contraction.md § 3.2.](source/1_euclidean_gas/04_wasserstein_contraction)

### Cluster-Level Outlier Alignment
- **Type:** Lemma
- **Label:** `lem-cluster-alignment`
- **Tags:** lemma
- **Source:** [04_wasserstein_contraction.md § 4.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Why This Proof is Static and Robust
- **Type:** Remark
- **Label:** `rem-static-robust`
- **Tags:** general
- **Source:** [04_wasserstein_contraction.md § 4.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Expected Cross-Distance Change
- **Type:** Lemma
- **Label:** `lem-expected-distance-change`
- **Tags:** lemma, metric
- **Source:** [04_wasserstein_contraction.md § 5.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Cloning Pressure on Target Set
- **Type:** Lemma
- **Label:** `lem-target-cloning-pressure`
- **Tags:** cloning, lemma
- **Source:** [04_wasserstein_contraction.md § 5.2.](source/1_euclidean_gas/04_wasserstein_contraction)

### Average Cloning Pressure Bound
- **Type:** Corollary
- **Label:** `cor-average-cloning`
- **Tags:** cloning, corollary
- **Source:** [04_wasserstein_contraction.md § 5.2.](source/1_euclidean_gas/04_wasserstein_contraction)

### Wasserstein Distance and Population Cross-Distances
- **Type:** Lemma
- **Label:** `lem-wasserstein-population-bound`
- **Tags:** lemma, metric, wasserstein
- **Source:** [04_wasserstein_contraction.md § 6.1.](source/1_euclidean_gas/04_wasserstein_contraction)

### Wasserstein-2 Contraction (Cluster-Based)
- **Type:** Theorem
- **Label:** `thm-main-contraction-full`
- **Tags:** theorem, wasserstein
- **Source:** [04_wasserstein_contraction.md § 6.2.](source/1_euclidean_gas/04_wasserstein_contraction)

---

### Source: 05_kinetic_contraction.md {#05_kinetic_contraction}

### The Kinetic Operator (Stratonovich Form)
- **Type:** Definition
- **Label:** `def-kinetic-operator-stratonovich`
- **Tags:** kinetic
- **Source:** [05_kinetic_contraction.md § 5.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Relationship to Itô Formulation
- **Type:** Remark
- **Label:** `rem-stratonovich-ito-equivalence`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 5.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Globally Confining Potential
- **Type:** Axiom
- **Label:** `axiom-confining-potential`
- **Tags:** axiom, fitness
- **Source:** [05_kinetic_contraction.md § 5.3.1.](source/1_euclidean_gas/05_kinetic_contraction)

### Canonical Confining Potential
- **Type:** Example
- **Label:** `ex-canonical-confining-potential`
- **Tags:** fitness
- **Source:** [05_kinetic_contraction.md § 5.3.1.](source/1_euclidean_gas/05_kinetic_contraction)

### Anisotropic Diffusion Tensor
- **Type:** Axiom
- **Label:** `axiom-diffusion-tensor`
- **Tags:** axiom, langevin
- **Source:** [05_kinetic_contraction.md § 5.3.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Why Uniform Ellipticity Matters
- **Type:** Remark
- **Label:** `rem-uniform-ellipticity-importance`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 5.3.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Friction and Integration Parameters
- **Type:** Axiom
- **Label:** `axiom-friction-timestep`
- **Tags:** axiom
- **Source:** [05_kinetic_contraction.md § 5.3.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Fokker-Planck Equation for the Kinetic Operator
- **Type:** Proposition
- **Label:** `prop-fokker-planck-kinetic`
- **Tags:** kinetic, proposition
- **Source:** [05_kinetic_contraction.md § 5.4.](source/1_euclidean_gas/05_kinetic_contraction)

### Formal Invariant Measure (Without Boundary)
- **Type:** Remark
- **Label:** `rem-formal-invariant-measure`
- **Tags:** symmetry
- **Source:** [05_kinetic_contraction.md § 5.4.](source/1_euclidean_gas/05_kinetic_contraction)

### BAOAB Integrator for Stratonovich Langevin
- **Type:** Definition
- **Label:** `def-baoab-integrator`
- **Tags:** langevin
- **Source:** [05_kinetic_contraction.md § 5.5.](source/1_euclidean_gas/05_kinetic_contraction)

### Stratonovich Correction for Anisotropic Case
- **Type:** Remark
- **Label:** `rem-baoab-anisotropic`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 5.5.](source/1_euclidean_gas/05_kinetic_contraction)

### Infinitesimal Generator of the Kinetic SDE
- **Type:** Definition
- **Label:** `def-generator`
- **Tags:** kinetic, langevin
- **Source:** [05_kinetic_contraction.md § 5.7.1.](source/1_euclidean_gas/05_kinetic_contraction)

### Why We Work with Generators
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 5.7.1.](source/1_euclidean_gas/05_kinetic_contraction)

### Discrete-Time Inheritance of Generator Drift
- **Type:** Theorem
- **Label:** `thm-discretization`
- **Tags:** theorem
- **Source:** [05_kinetic_contraction.md § 5.7.2.](source/1_euclidean_gas/05_kinetic_contraction)

### BAOAB Weak Error for Variance Lyapunov Functions
- **Type:** Proposition
- **Label:** `prop-weak-error-variance`
- **Tags:** proposition
- **Source:** [05_kinetic_contraction.md § 1.7.3.1.](source/1_euclidean_gas/05_kinetic_contraction)

### BAOAB Weak Error for Boundary Lyapunov Function
- **Type:** Proposition
- **Label:** `prop-weak-error-boundary`
- **Tags:** proposition
- **Source:** [05_kinetic_contraction.md § 1.7.3.2.](source/1_euclidean_gas/05_kinetic_contraction)

### BAOAB Weak Error for Wasserstein Distance
- **Type:** Proposition
- **Label:** `prop-weak-error-wasserstein`
- **Tags:** metric, proposition, wasserstein
- **Source:** [05_kinetic_contraction.md § 1.7.3.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Explicit Discretization Constants
- **Type:** Proposition
- **Label:** `prop-explicit-constants`
- **Tags:** proposition
- **Source:** [05_kinetic_contraction.md § 5.7.4.](source/1_euclidean_gas/05_kinetic_contraction)

### No Convexity Required
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 6.1.](source/1_euclidean_gas/05_kinetic_contraction)

### The Hypocoercive Norm
- **Type:** Definition
- **Label:** `def-hypocoercive-norm`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Intuition for the Coupling Term
- **Type:** Remark
- **Label:** `rem-coupling-term-intuition`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Inter-Swarm Error Contraction Under Kinetic Operator
- **Type:** Theorem
- **Label:** `thm-inter-swarm-contraction-kinetic`
- **Tags:** kinetic, theorem
- **Source:** [05_kinetic_contraction.md § 6.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Drift of Location Error Under Kinetics
- **Type:** Lemma
- **Label:** `lem-location-error-drift-kinetic`
- **Tags:** kinetic, lemma
- **Source:** [05_kinetic_contraction.md § 6.5.](source/1_euclidean_gas/05_kinetic_contraction)

### Drift of Structural Error Under Kinetics
- **Type:** Lemma
- **Label:** `lem-structural-error-drift-kinetic`
- **Tags:** kinetic, lemma
- **Source:** [05_kinetic_contraction.md § 6.6.](source/1_euclidean_gas/05_kinetic_contraction)

### Velocity Variance Component (Recall)
- **Type:** Definition
- **Label:** `def-velocity-variance-recall`
- **Tags:** kinetic
- **Source:** [05_kinetic_contraction.md § 7.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Velocity Variance Contraction Under Kinetic Operator
- **Type:** Theorem
- **Label:** `thm-velocity-variance-contraction-kinetic`
- **Tags:** kinetic, theorem
- **Source:** [05_kinetic_contraction.md § 7.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Net Velocity Variance Contraction for Composed Operator
- **Type:** Corollary
- **Label:** `cor-net-velocity-contraction`
- **Tags:** corollary, kinetic
- **Source:** [05_kinetic_contraction.md § 7.5.](source/1_euclidean_gas/05_kinetic_contraction)

### Positional Variance Component (Recall)
- **Type:** Definition
- **Label:** `def-positional-variance-recall`
- **Tags:** general
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Bounded Positional Variance Expansion Under Kinetics
- **Type:** Theorem
- **Label:** `thm-positional-variance-bounded-expansion`
- **Tags:** kinetic, theorem
- **Source:** [05_kinetic_contraction.md § 6.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Net Positional Variance Contraction for Composed Operator
- **Type:** Corollary
- **Label:** `cor-net-positional-contraction`
- **Tags:** corollary
- **Source:** [05_kinetic_contraction.md § 6.5.](source/1_euclidean_gas/05_kinetic_contraction)

### Boundary Potential (Recall)
- **Type:** Definition
- **Label:** `def-boundary-potential-recall`
- **Tags:** fitness
- **Source:** [05_kinetic_contraction.md § 7.2.](source/1_euclidean_gas/05_kinetic_contraction)

### Boundary Potential Contraction Under Kinetic Operator
- **Type:** Theorem
- **Label:** `thm-boundary-potential-contraction-kinetic`
- **Tags:** fitness, kinetic, theorem
- **Source:** [05_kinetic_contraction.md § 7.3.](source/1_euclidean_gas/05_kinetic_contraction)

### Total Boundary Safety from Dual Mechanisms
- **Type:** Corollary
- **Label:** `cor-total-boundary-safety`
- **Tags:** corollary
- **Source:** [05_kinetic_contraction.md § 7.5.](source/1_euclidean_gas/05_kinetic_contraction)

---

### Source: 06_convergence.md {#06_convergence}

### Summary of Required Operator Drifts
- **Type:** Remark
- **Label:** `rem-prerequisite-drifts`
- **Tags:** general
- **Source:** [06_convergence.md § 2.3.](source/1_euclidean_gas/06_convergence)

### Synergistic Lyapunov Function (Recall)
- **Type:** Definition
- **Label:** `def-full-lyapunov-recall`
- **Tags:** general
- **Source:** [06_convergence.md § 3.2.](source/1_euclidean_gas/06_convergence)

### Complete Drift Characterization
- **Type:** Proposition
- **Label:** `prop-complete-drift-summary`
- **Tags:** proposition
- **Source:** [06_convergence.md § 3.3.](source/1_euclidean_gas/06_convergence)

### Foster-Lyapunov Drift for the Composed Operator
- **Type:** Theorem
- **Label:** `thm-foster-lyapunov-main`
- **Tags:** theorem
- **Source:** [06_convergence.md § 3.4.](source/1_euclidean_gas/06_convergence)

### The Cemetery State
- **Type:** Definition
- **Label:** `def-cemetery-state`
- **Tags:** general
- **Source:** [06_convergence.md § 4.2.](source/1_euclidean_gas/06_convergence)

### Why Extinction is Inevitable (Eventually)
- **Type:** Remark
- **Label:** `rem-extinction-inevitable`
- **Tags:** general
- **Source:** [06_convergence.md § 4.2.](source/1_euclidean_gas/06_convergence)

### Quasi-Stationary Distribution (QSD)
- **Type:** Definition
- **Label:** `def-qsd`
- **Tags:** qsd
- **Source:** [06_convergence.md § 4.3.](source/1_euclidean_gas/06_convergence)

### φ-Irreducibility of the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-phi-irreducibility`
- **Tags:** theorem
- **Source:** [06_convergence.md § 4.4.1.](source/1_euclidean_gas/06_convergence)

### Aperiodicity of the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-aperiodicity`
- **Tags:** theorem
- **Source:** [06_convergence.md § 4.4.2.](source/1_euclidean_gas/06_convergence)

### Geometric Ergodicity and Convergence to QSD
- **Type:** Theorem
- **Label:** `thm-main-convergence`
- **Tags:** convergence, geometry, metric, qsd, theorem
- **Source:** [06_convergence.md § 4.5.](source/1_euclidean_gas/06_convergence)

### Properties of the Quasi-Stationary Distribution
- **Type:** Proposition
- **Label:** `prop-qsd-properties`
- **Tags:** proposition, qsd
- **Source:** [06_convergence.md § 4.6.](source/1_euclidean_gas/06_convergence)

### Equilibrium Variance Bounds from Drift Inequalities
- **Type:** Theorem
- **Label:** `thm-equilibrium-variance-bounds`
- **Tags:** theorem
- **Source:** [06_convergence.md § 4.6.](source/1_euclidean_gas/06_convergence)

### Velocity Dissipation Rate (Parameter-Explicit)
- **Type:** Proposition
- **Label:** `prop-velocity-rate-explicit`
- **Tags:** kinetic, proposition
- **Source:** [06_convergence.md § 5.1.](source/1_euclidean_gas/06_convergence)

### Positional Contraction Rate (Parameter-Explicit)
- **Type:** Proposition
- **Label:** `prop-position-rate-explicit`
- **Tags:** proposition
- **Source:** [06_convergence.md § 5.2.](source/1_euclidean_gas/06_convergence)

### Wasserstein Contraction Rate (Parameter-Explicit)
- **Type:** Proposition
- **Label:** `prop-wasserstein-rate-explicit`
- **Tags:** proposition, wasserstein
- **Source:** [06_convergence.md § 5.3.](source/1_euclidean_gas/06_convergence)

### Boundary Contraction Rate (Parameter-Explicit)
- **Type:** Proposition
- **Label:** `prop-boundary-rate-explicit`
- **Tags:** proposition
- **Source:** [06_convergence.md § 5.4.](source/1_euclidean_gas/06_convergence)

### Synergistic Rate Derivation from Component Drifts
- **Type:** Theorem
- **Label:** `thm-synergistic-rate-derivation`
- **Tags:** theorem
- **Source:** [06_convergence.md § 5.5.](source/1_euclidean_gas/06_convergence)

### Total Convergence Rate (Parameter-Explicit)
- **Type:** Theorem
- **Label:** `thm-total-rate-explicit`
- **Tags:** convergence, theorem
- **Source:** [06_convergence.md § 5.5.](source/1_euclidean_gas/06_convergence)

### Mixing Time (Parameter-Explicit)
- **Type:** Proposition
- **Label:** `prop-mixing-time-explicit`
- **Tags:** proposition
- **Source:** [06_convergence.md § 5.6.](source/1_euclidean_gas/06_convergence)

### Parameter Selection for Optimal Convergence
- **Type:** Algorithm
- **Label:** `alg-param-selection`
- **Tags:** algorithm, convergence
- **Source:** [06_convergence.md § 5.7.](source/1_euclidean_gas/06_convergence)

### Complete Parameter Space
- **Type:** Definition
- **Label:** `def-complete-parameter-space`
- **Tags:** general
- **Source:** [06_convergence.md § 6.1.](source/1_euclidean_gas/06_convergence)

### Parameter Classification
- **Type:** Proposition
- **Label:** `prop-parameter-classification`
- **Tags:** proposition
- **Source:** [06_convergence.md § 6.2.](source/1_euclidean_gas/06_convergence)

### Log-Sensitivity Matrix for Convergence Rates
- **Type:** Definition
- **Label:** `def-rate-sensitivity-matrix`
- **Tags:** convergence
- **Source:** [06_convergence.md § 6.3.1.](source/1_euclidean_gas/06_convergence)

### Explicit Rate Sensitivity Matrix
- **Type:** Theorem
- **Label:** `thm-explicit-rate-sensitivity`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.3.1.](source/1_euclidean_gas/06_convergence)

### Equilibrium Constant Sensitivity Matrix
- **Type:** Definition
- **Label:** `def-equilibrium-sensitivity-matrix`
- **Tags:** general
- **Source:** [06_convergence.md § 6.3.2.](source/1_euclidean_gas/06_convergence)

### SVD of Rate Sensitivity Matrix
- **Type:** Theorem
- **Label:** `thm-svd-rate-matrix`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.4.](source/1_euclidean_gas/06_convergence)

### Condition Number of Rate Sensitivity
- **Type:** Proposition
- **Label:** `prop-condition-number-rate`
- **Tags:** proposition
- **Source:** [06_convergence.md § 6.4.](source/1_euclidean_gas/06_convergence)

### Parameter Optimization Problem
- **Type:** Definition
- **Label:** `def-parameter-optimization`
- **Tags:** general
- **Source:** [06_convergence.md § 6.5.1.](source/1_euclidean_gas/06_convergence)

### Subgradient of min() Function
- **Type:** Theorem
- **Label:** `thm-subgradient-min`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.5.2.](source/1_euclidean_gas/06_convergence)

### Necessity of Balanced Rates at Optimum
- **Type:** Theorem
- **Label:** `thm-balanced-optimality`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.5.3.](source/1_euclidean_gas/06_convergence)

### Restitution-Friction Coupling
- **Type:** Proposition
- **Label:** `prop-restitution-friction-coupling`
- **Tags:** proposition
- **Source:** [06_convergence.md § 6.6.1.](source/1_euclidean_gas/06_convergence)

### Position Jitter - Cloning Rate Coupling
- **Type:** Proposition
- **Label:** `prop-jitter-cloning-coupling`
- **Tags:** cloning, proposition
- **Source:** [06_convergence.md § 6.6.2.](source/1_euclidean_gas/06_convergence)

### Phase-Space Pairing Quality
- **Type:** Proposition
- **Label:** `prop-phase-space-pairing`
- **Tags:** proposition
- **Source:** [06_convergence.md § 6.6.3.](source/1_euclidean_gas/06_convergence)

### Parameter Error Propagation Bound
- **Type:** Theorem
- **Label:** `thm-error-propagation`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.7.](source/1_euclidean_gas/06_convergence)

### Closed-Form Balanced Optimum
- **Type:** Theorem
- **Label:** `thm-closed-form-optimum`
- **Tags:** theorem
- **Source:** [06_convergence.md § 6.10.1.](source/1_euclidean_gas/06_convergence)

### Projected Gradient Ascent for Parameter Optimization
- **Type:** Algorithm
- **Label:** `alg-projected-gradient-ascent`
- **Tags:** algorithm
- **Source:** [06_convergence.md § 6.10.2.](source/1_euclidean_gas/06_convergence)

### Pareto Optimality in Parameter Space
- **Type:** Definition
- **Label:** `def-pareto-optimality`
- **Tags:** general
- **Source:** [06_convergence.md § 6.10.3.](source/1_euclidean_gas/06_convergence)

### Adaptive Parameter Tuning
- **Type:** Algorithm
- **Label:** `alg-adaptive-tuning`
- **Tags:** algorithm
- **Source:** [06_convergence.md § 6.10.4.](source/1_euclidean_gas/06_convergence)

---

### Source: 07_mean_field.md {#07_mean_field}

### Phase Space
- **Type:** Definition
- **Label:** `def-mean-field-phase-space`
- **Tags:** mean-field
- **Source:** [07_mean_field.md § **1.1. Phase Space and Probability Density**](source/1_euclidean_gas/07_mean_field)

### Phase-Space Density
- **Type:** Definition
- **Label:** `def-phase-space-density`
- **Tags:** general
- **Source:** [07_mean_field.md § **1.1. Phase Space and Probability Density**](source/1_euclidean_gas/07_mean_field)

### Mean-Field Statistical Moments
- **Type:** Definition
- **Label:** `def-mean-field-moments`
- **Tags:** mean-field
- **Source:** [07_mean_field.md § **1.2. Mean-Field Measurement Pipeline**](source/1_euclidean_gas/07_mean_field)

### Mean-Field Regularized Standard Deviation
- **Type:** Definition
- **Label:** `def-mean-field-patched-std`
- **Tags:** mean-field
- **Source:** [07_mean_field.md § **1.2. Mean-Field Measurement Pipeline**](source/1_euclidean_gas/07_mean_field)

### Mean-Field Z-Scores
- **Type:** Definition
- **Label:** `def-mean-field-z-scores`
- **Tags:** mean-field
- **Source:** [07_mean_field.md § **1.3. Density-Dependent Fitness Potential**](source/1_euclidean_gas/07_mean_field)

### Mean-Field Fitness Potential
- **Type:** Definition
- **Label:** `def-mean-field-fitness-potential`
- **Tags:** fitness, mean-field
- **Source:** [07_mean_field.md § **1.3. Density-Dependent Fitness Potential**](source/1_euclidean_gas/07_mean_field)

### The BAOAB Update Rule
- **Type:** Definition
- **Label:** `def-baoab-update-rule`
- **Tags:** general
- **Source:** [07_mean_field.md § **2.1. The Underlying Discrete-Time Integrator: BA](source/1_euclidean_gas/07_mean_field)

### Kinetic Transport Operator
- **Type:** Definition
- **Label:** `def-kinetic-generator`
- **Tags:** kinetic
- **Source:** [07_mean_field.md § **2.2. The Kinetic Transport Operator ($L^\dagger$](source/1_euclidean_gas/07_mean_field)

### Interior Killing Operator
- **Type:** Definition
- **Label:** `def-killing-operator`
- **Tags:** general
- **Source:** [07_mean_field.md § **2.3. The Reaction Operators (Killing, Revival, a](source/1_euclidean_gas/07_mean_field)

### Revival Operator
- **Type:** Definition
- **Label:** `def-revival-operator`
- **Tags:** general
- **Source:** [07_mean_field.md § **2.3. The Reaction Operators (Killing, Revival, a](source/1_euclidean_gas/07_mean_field)

### Internal Cloning Operator (Derived Form)
- **Type:** Definition
- **Label:** `def-cloning-generator`
- **Tags:** cloning
- **Source:** [07_mean_field.md § **2.3.3. Derivation of the Internal Cloning Operat](source/1_euclidean_gas/07_mean_field)

### Transport Operator and Probability Flux
- **Type:** Definition
- **Label:** `def-transport-operator`
- **Tags:** general
- **Source:** [07_mean_field.md § **3.1. The Transport Operator ($L^\dagger$) is Mas](source/1_euclidean_gas/07_mean_field)

### Mass Conservation of Transport
- **Type:** Lemma
- **Label:** `lem-mass-conservation-transport`
- **Tags:** lemma
- **Source:** [07_mean_field.md § **3.1. The Transport Operator ($L^\dagger$) is Mas](source/1_euclidean_gas/07_mean_field)

### The Mean-Field Equations for the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-mean-field-equation`
- **Tags:** mean-field, theorem
- **Source:** [07_mean_field.md § **3.3. The Coupled Mean-Field Equations**](source/1_euclidean_gas/07_mean_field)

### Total Mass Conservation and Population Dynamics
- **Type:** Theorem
- **Label:** `thm-mass-conservation`
- **Tags:** theorem
- **Source:** [07_mean_field.md § **3.3. The Coupled Mean-Field Equations**](source/1_euclidean_gas/07_mean_field)

### Summary of Regularity Assumptions
- **Type:** Assumption
- **Label:** `asmp-regularity-summary`
- **Tags:** assumption
- **Source:** [07_mean_field.md § **4.2. Well-Posedness and Future Work**](source/1_euclidean_gas/07_mean_field)

### Regularity of the Valid Domain
- **Type:** Assumption
- **Label:** `asmp-domain-regularity`
- **Tags:** assumption
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)

### Regularity of the Discrete Integrator
- **Type:** Assumption
- **Label:** `asmp-integrator-regularity`
- **Tags:** assumption
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)

### Density Regularity
- **Type:** Assumption
- **Label:** `asmp-density-regularity-killing`
- **Tags:** assumption
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)

### Consistency of the Interior Killing Rate Approximation
- **Type:** Theorem
- **Label:** `thm-killing-rate-consistency`
- **Tags:** theorem
- **Source:** [07_mean_field.md § **4.4.2. Main Theorem: Rigorous Killing Rate Appro](source/1_euclidean_gas/07_mean_field)

### Mean-Field Limit (Informal Statement)
- **Type:** Theorem
- **Label:** `thm-mean-field-limit-informal`
- **Tags:** convergence, mean-field, theorem
- **Source:** [07_mean_field.md § **4.4.4. Connection to the Main Mean-Field Result*](source/1_euclidean_gas/07_mean_field)

---

### Source: 08_propagation_chaos.md {#08_propagation_chaos}

### Sequence of N-Particle QSDs and their Marginals
- **Type:** Definition
- **Label:** `def-sequence-of-qsds`
- **Tags:** qsd
- **Source:** [08_propagation_chaos.md § 2.](source/1_euclidean_gas/08_propagation_chaos)

### The Sequence of Marginals $\{\mu_N\}$ is Tight
- **Type:** Theorem
- **Label:** `thm-qsd-marginals-are-tight`
- **Tags:** qsd, theorem
- **Source:** [08_propagation_chaos.md § **Introduction**](source/1_euclidean_gas/08_propagation_chaos)

### Exchangeability of the N-Particle QSD
- **Type:** Lemma
- **Label:** `lem-exchangeability`
- **Tags:** lemma, qsd
- **Source:** [08_propagation_chaos.md § **Lemma A.1: Exchangeability of the N-Particle QSD](source/1_euclidean_gas/08_propagation_chaos)

### Weak Convergence of the Empirical Companion Measure
- **Type:** Lemma
- **Label:** `lem-empirical-convergence`
- **Tags:** convergence, lemma
- **Source:** [08_propagation_chaos.md § **Lemma A.2: Weak Convergence of the Empirical Com](source/1_euclidean_gas/08_propagation_chaos)

### Continuity of the Reward Moments
- **Type:** Lemma
- **Label:** `lem-reward-continuity`
- **Tags:** fitness, lemma
- **Source:** [08_propagation_chaos.md § **Lemma B.1: Continuity of the Reward Moments**](source/1_euclidean_gas/08_propagation_chaos)

### Continuity of the Distance Moments
- **Type:** Lemma
- **Label:** `lem-distance-continuity`
- **Tags:** lemma, metric
- **Source:** [08_propagation_chaos.md § **Lemma B.2: Continuity of the Distance Moments**](source/1_euclidean_gas/08_propagation_chaos)

### Uniform Integrability and Interchange of Limits
- **Type:** Lemma
- **Label:** `lem-uniform-integrability`
- **Tags:** convergence, lemma
- **Source:** [08_propagation_chaos.md § **Lemma C.1: Uniform Integrability and Interchange](source/1_euclidean_gas/08_propagation_chaos)

### Convergence of Boundary Death and Revival
- **Type:** Lemma
- **Label:** `lem-boundary-convergence`
- **Tags:** convergence, lemma
- **Source:** [08_propagation_chaos.md § **Lemma C.2: Convergence of the Boundary Death and](source/1_euclidean_gas/08_propagation_chaos)

### QSD Stationarity vs. True Stationarity
- **Type:** Remark
- **Label:** `rem-qsd-vs-true-stationarity`
- **Tags:** qsd
- **Source:** [08_propagation_chaos.md § **The QSD Stationarity Condition with Extinction R](source/1_euclidean_gas/08_propagation_chaos)

### Extinction Rate Vanishes in the Mean-Field Limit
- **Type:** Theorem
- **Label:** `thm-extinction-rate-vanishes`
- **Tags:** convergence, mean-field, theorem
- **Source:** [08_propagation_chaos.md § **Proof of Vanishing Extinction Rate**](source/1_euclidean_gas/08_propagation_chaos)

### Physical Interpretation
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [08_propagation_chaos.md § **Proof of Vanishing Extinction Rate**](source/1_euclidean_gas/08_propagation_chaos)

### Limit Points are Weak Solutions to the Stationary Mean-Field PDE
- **Type:** Theorem
- **Label:** `thm-limit-is-weak-solution`
- **Tags:** convergence, mean-field, theorem
- **Source:** [08_propagation_chaos.md § **Theorem C.2: Limit Points are Weak Solutions to ](source/1_euclidean_gas/08_propagation_chaos)

### Weighted Sobolev Space $H^1_w(\Omega)$
- **Type:** Definition
- **Label:** `def-uniqueness-weighted-sobolev-h1w`
- **Tags:** general
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)

### Completeness of $H^1_w(\Omega)$
- **Type:** Theorem
- **Label:** `thm-uniqueness-completeness-h1w-omega`
- **Tags:** theorem
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)

### Completeness of the Constraint Set $\mathcal{P}$
- **Type:** Remark
- **Label:** `rem-uniqueness-completeness-constraint-set`
- **Tags:** general
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)

### Self-Mapping Property of the Solution Operator
- **Type:** Lemma
- **Label:** `lem-uniqueness-self-mapping`
- **Tags:** lemma
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)

### Lipschitz Continuity of Moment Functionals
- **Type:** Lemma
- **Label:** `lem-uniqueness-lipschitz-moments`
- **Tags:** lemma, lipschitz
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)

### Fixed Points Lie in a Bounded Ball
- **Type:** Lemma
- **Label:** `lem-uniqueness-fixed-point-bounded`
- **Tags:** lemma
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)

### Lipschitz Continuity of the Fitness Potential
- **Type:** Lemma
- **Label:** `lem-uniqueness-lipschitz-fitness-potential`
- **Tags:** fitness, lemma, lipschitz
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)

### Local Lipschitz Continuity of the Cloning Operator
- **Type:** Lemma
- **Label:** `lem-uniqueness-lipschitz-cloning-operator`
- **Tags:** cloning, lemma, lipschitz
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)

### Hörmander's Theorem for Kinetic Operators
- **Type:** Theorem
- **Label:** `thm-uniqueness-hormander`
- **Tags:** kinetic, theorem
- **Source:** [08_propagation_chaos.md § **C.2. Hörmander's Hypoellipticity Condition**](source/1_euclidean_gas/08_propagation_chaos)

### Verification of Hörmander's Condition for the Kinetic Operator
- **Type:** Lemma
- **Label:** `lem-uniqueness-hormander-verification`
- **Tags:** kinetic, lemma
- **Source:** [08_propagation_chaos.md § **C.2. Hörmander's Hypoellipticity Condition**](source/1_euclidean_gas/08_propagation_chaos)

### Hypoelliptic Regularity for the Kinetic Operator
- **Type:** Theorem
- **Label:** `thm-uniqueness-hypoelliptic-regularity`
- **Tags:** kinetic, theorem
- **Source:** [08_propagation_chaos.md § **C.3. Hypoelliptic Regularity Estimates**](source/1_euclidean_gas/08_propagation_chaos)

### Scaling of $C_{\text{hypo}}$ with Diffusion Strength
- **Type:** Lemma
- **Label:** `lem-uniqueness-scaling-hypoelliptic-constant`
- **Tags:** langevin, lemma
- **Source:** [08_propagation_chaos.md § **C.4. Scaling of the Hypoelliptic Constant**](source/1_euclidean_gas/08_propagation_chaos)

### Contraction Property of the Solution Operator on an Invariant Ball
- **Type:** Theorem
- **Label:** `thm-uniqueness-contraction-solution-operator`
- **Tags:** symmetry, theorem
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)

### Uniqueness of the Stationary Solution
- **Type:** Theorem
- **Label:** `thm-uniqueness-uniqueness-stationary-solution`
- **Tags:** theorem
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)

### The proof structure demonstrates a powerful technique in nonlinear analysis: when global Lipschitz continuity fails, we can still prove uniqueness by:
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** lipschitz
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)

### This uniqueness proof reveals a deep connection between the algorithm's design parameters and the mathematical well-posedness of the model. The condition
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** gauge
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)

### Sequence of N-Particle QSDs and their Marginals
- **Type:** Definition
- **Label:** `def-sequence-of-qsds`
- **Tags:** qsd
- **Source:** [08_propagation_chaos.md § **6.2. The Sequence of N-Particle Stationary Measu](source/1_euclidean_gas/08_propagation_chaos)

### The Sequence of Marginals $\{\mu_N\}$ is Tight
- **Type:** Theorem
- **Label:** `thm-qsd-marginals-are-tight`
- **Tags:** qsd, theorem
- **Source:** [08_propagation_chaos.md § **6.3. Step 1: Tightness of the Marginal Sequence*](source/1_euclidean_gas/08_propagation_chaos)

### Limit Points are Weak Solutions to the Stationary Mean-Field PDE
- **Type:** Theorem
- **Label:** `thm-limit-is-weak-solution`
- **Tags:** convergence, mean-field, theorem
- **Source:** [08_propagation_chaos.md § **6.4. Step 2: Identification of the Limit Point**](source/1_euclidean_gas/08_propagation_chaos)

### Uniqueness of the Stationary Solution
- **Type:** Theorem
- **Label:** `thm-uniqueness-of-qsd`
- **Tags:** qsd, theorem
- **Source:** [08_propagation_chaos.md § **6.5. Step 3: Uniqueness of the Weak Solution**](source/1_euclidean_gas/08_propagation_chaos)

### Convergence of Macroscopic Observables (The Thermodynamic Limit)
- **Type:** Theorem
- **Label:** `thm-thermodynamic-limit`
- **Tags:** convergence, theorem
- **Source:** [08_propagation_chaos.md § **6.6. The Thermodynamic Limit**](source/1_euclidean_gas/08_propagation_chaos)

### Wasserstein-2 Convergence in the Thermodynamic Limit
- **Type:** Corollary
- **Label:** `cor-w2-convergence-thermodynamic-limit`
- **Tags:** convergence, corollary, wasserstein
- **Source:** [08_propagation_chaos.md § **6.6. The Thermodynamic Limit**](source/1_euclidean_gas/08_propagation_chaos)

---

### Source: 09_kl_convergence.md {#09_kl_convergence}

### Exponential KL-Convergence for the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-main-kl-convergence-consolidated`
- **Tags:** convergence, theorem
- **Source:** [09_kl_convergence.md § I.1.1. Primary Theorem (Log-Concave Case)](source/1_euclidean_gas/09_kl_convergence)

### KL-Convergence Without Log-Concavity
- **Type:** Theorem
- **Label:** `thm-nonconvex-kl-convergence`
- **Tags:** convergence, theorem
- **Source:** [09_kl_convergence.md § I.1.2. Extended Theorem (Non-Convex Case)](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL-Convergence for the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-main-kl-convergence`
- **Tags:** convergence, theorem
- **Source:** [09_kl_convergence.md § 0.1.](source/1_euclidean_gas/09_kl_convergence)

### Relative Entropy and Fisher Information
- **Type:** Definition
- **Label:** `def-relative-entropy`
- **Tags:** entropy
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)

### Logarithmic Sobolev Inequality (LSI)
- **Type:** Definition
- **Label:** `def-lsi-continuous`
- **Tags:** lsi
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)

### Discrete-Time LSI
- **Type:** Definition
- **Label:** `def-discrete-lsi`
- **Tags:** lsi
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)

### Bakry-Émery Criterion for LSI
- **Type:** Theorem
- **Label:** `thm-bakry-emery`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § 1.2.](source/1_euclidean_gas/09_kl_convergence)

### Target Gibbs Measure for Kinetic Dynamics
- **Type:** Definition
- **Label:** `def-gibbs-kinetic`
- **Tags:** kinetic
- **Source:** [09_kl_convergence.md § 2.1.](source/1_euclidean_gas/09_kl_convergence)

### The generator $\mathcal{L}_{\text{kin}}$ is **not self-adjoint** with respect to $\pi_{\text{kin}}$. This non-reversibility is a fundamental barrier to applying classical LSI theory.
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** lsi
- **Source:** [09_kl_convergence.md § 2.1.](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive Metric and Modified Dirichlet Form
- **Type:** Definition
- **Label:** `def-hypocoercive-metric`
- **Tags:** metric
- **Source:** [09_kl_convergence.md § 2.2.](source/1_euclidean_gas/09_kl_convergence)

### Dissipation of the Hypocoercive Norm
- **Type:** Lemma
- **Label:** `lem-hypocoercive-dissipation`
- **Tags:** lemma
- **Source:** [09_kl_convergence.md § 2.2.](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive LSI for the Kinetic Flow Map
- **Type:** Theorem
- **Label:** `thm-kinetic-lsi`
- **Tags:** kinetic, lsi, theorem
- **Source:** [09_kl_convergence.md § 2.3.](source/1_euclidean_gas/09_kl_convergence)

### Tensorization of LSI
- **Type:** Theorem
- **Label:** `thm-tensorization`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § 3.1.](source/1_euclidean_gas/09_kl_convergence)

### LSI for N-Particle Kinetic Operator
- **Type:** Corollary
- **Label:** `cor-n-particle-kinetic-lsi`
- **Tags:** corollary, kinetic, lsi
- **Source:** [09_kl_convergence.md § 3.1.](source/1_euclidean_gas/09_kl_convergence)

### Log-Concavity of the Quasi-Stationary Distribution
- **Type:** Axiom
- **Label:** `ax-qsd-log-concave`
- **Tags:** axiom, qsd
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)

### Motivation and Justification
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)

### Explicit Log-Concavity Condition
- **Type:** Definition
- **Label:** `def-log-concavity-condition`
- **Tags:** general
- **Source:** [09_kl_convergence.md § From Axiom to Verifiable Condition](source/1_euclidean_gas/09_kl_convergence)

### Log-Concavity for Pure Yang-Mills Vacuum
- **Type:** Lemma
- **Label:** `lem-log-concave-yang-mills`
- **Tags:** lemma
- **Source:** [09_kl_convergence.md § Verification for Specific Physical Systems](source/1_euclidean_gas/09_kl_convergence)

### Implications for Millennium Prize
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [09_kl_convergence.md § Verification for Specific Physical Systems](source/1_euclidean_gas/09_kl_convergence)

### Conditional Independence of Cloning
- **Type:** Lemma
- **Label:** `lem-cloning-conditional-independence`
- **Tags:** cloning, lemma
- **Source:** [09_kl_convergence.md § 4.1.](source/1_euclidean_gas/09_kl_convergence)

### The HWI Inequality (Otto-Villani)
- **Type:** Theorem
- **Label:** `thm-hwi-inequality`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 4.2.](source/1_euclidean_gas/09_kl_convergence)

### The HWI inequality provides a **bridge** between:
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [09_kl_convergence.md § 4.2.](source/1_euclidean_gas/09_kl_convergence)

### Wasserstein-2 Contraction for Cloning
- **Type:** Lemma
- **Label:** `lem-cloning-wasserstein-contraction`
- **Tags:** cloning, lemma, wasserstein
- **Source:** [09_kl_convergence.md § 4.3.](source/1_euclidean_gas/09_kl_convergence)

### Fisher Information Bound After Cloning
- **Type:** Lemma
- **Label:** `lem-cloning-fisher-info`
- **Tags:** cloning, lemma
- **Source:** [09_kl_convergence.md § 4.4.](source/1_euclidean_gas/09_kl_convergence)

### Entropy Contraction for the Cloning Operator
- **Type:** Theorem
- **Label:** `thm-cloning-entropy-contraction`
- **Tags:** cloning, entropy, theorem
- **Source:** [09_kl_convergence.md § 4.5.](source/1_euclidean_gas/09_kl_convergence)

### Interpretation
- **Type:** Remark
- **Label:** `rem-cloning-sublinear`
- **Tags:** cloning
- **Source:** [09_kl_convergence.md § 4.5.](source/1_euclidean_gas/09_kl_convergence)

### Entropy-Transport Lyapunov Function
- **Type:** Definition
- **Label:** `def-entropy-transport-lyapunov`
- **Tags:** entropy
- **Source:** [09_kl_convergence.md § 5.1.](source/1_euclidean_gas/09_kl_convergence)

### Entropy-Transport Dissipation Inequality
- **Type:** Lemma
- **Label:** `lem-entropy-transport-dissipation`
- **Tags:** entropy, lemma
- **Source:** [09_kl_convergence.md § 5.2.](source/1_euclidean_gas/09_kl_convergence)

### This lemma is the **key technical innovation**. It shows that the geometric contraction in Wasserstein space (already proven in [04_convergence.md](04_convergence.md)) drives entropy dissipation. The constant $\alpha$ depends on:
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** convergence, entropy, geometry, metric, wasserstein
- **Source:** [09_kl_convergence.md § 5.2.](source/1_euclidean_gas/09_kl_convergence)

### Kinetic Evolution Bounds
- **Type:** Lemma
- **Label:** `lem-kinetic-evolution-bounds`
- **Tags:** kinetic, lemma
- **Source:** [09_kl_convergence.md § 5.3.](source/1_euclidean_gas/09_kl_convergence)

### Linear Contraction of the Entropy-Transport Lyapunov Function
- **Type:** Theorem
- **Label:** `thm-entropy-transport-contraction`
- **Tags:** entropy, theorem
- **Source:** [09_kl_convergence.md § 5.4.](source/1_euclidean_gas/09_kl_convergence)

### Discrete-Time LSI for the Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-main-lsi-composition`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § 5.5.](source/1_euclidean_gas/09_kl_convergence)

### Quantitative LSI Constant
- **Type:** Corollary
- **Label:** `cor-quantitative-lsi-final`
- **Tags:** corollary, lsi
- **Source:** [09_kl_convergence.md § 5.6.](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL-Convergence via LSI
- **Type:** Theorem
- **Label:** `thm-lsi-implies-kl-convergence`
- **Tags:** convergence, lsi, theorem
- **Source:** [09_kl_convergence.md § 6.1.](source/1_euclidean_gas/09_kl_convergence)

### KL-Convergence of the Euclidean Gas (Main Result)
- **Type:** Theorem
- **Label:** `thm-main-kl-final`
- **Tags:** convergence, theorem
- **Source:** [09_kl_convergence.md § 6.2.](source/1_euclidean_gas/09_kl_convergence)

### Relationship Between KL and TV Convergence Rates
- **Type:** Remark
- **Label:** `rem-kl-tv-comparison`
- **Tags:** convergence
- **Source:** [09_kl_convergence.md § 6.3.](source/1_euclidean_gas/09_kl_convergence)

### LSI Stability Under Bounded Perturbations
- **Type:** Theorem
- **Label:** `thm-lsi-perturbation`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § 7.1.](source/1_euclidean_gas/09_kl_convergence)

### LSI for the ρ-Localized Geometric Gas
- **Type:** Corollary
- **Label:** `cor-adaptive-lsi`
- **Tags:** corollary, geometry, lsi, metric
- **Source:** [09_kl_convergence.md § 7.2.](source/1_euclidean_gas/09_kl_convergence)

### N-Uniform Logarithmic Sobolev Inequality
- **Type:** Corollary
- **Label:** `cor-n-uniform-lsi`
- **Tags:** corollary, lsi
- **Source:** [09_kl_convergence.md § 9.6.](source/1_euclidean_gas/09_kl_convergence)

### Entropy Dissipation Under Cloning (Mean-Field Sketch)
- **Type:** Lemma
- **Label:** `lem-mean-field-cloning-sketch`
- **Tags:** cloning, entropy, lemma, mean-field
- **Source:** [09_kl_convergence.md § Lemma Statement](source/1_euclidean_gas/09_kl_convergence)

### Sinh Inequality
- **Type:** Lemma
- **Label:** `lem-sinh-bound-global`
- **Tags:** lemma
- **Source:** [09_kl_convergence.md § A.4: ✅ Contraction Inequality via Permutation Symm](source/1_euclidean_gas/09_kl_convergence)

### Entropy Bound via De Bruijn Identity
- **Type:** Theorem
- **Label:** `thm-entropy-bound-debruijn`
- **Tags:** entropy, theorem
- **Source:** [09_kl_convergence.md § Rigorous Formulation](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL-Convergence via Mean-Field Analysis
- **Type:** Theorem
- **Label:** `thm-meanfield-kl-convergence-hybrid`
- **Tags:** convergence, mean-field, theorem
- **Source:** [09_kl_convergence.md § Main Result](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive LSI for Kinetic Operator (Reference)
- **Type:** Theorem
- **Label:** `thm-kinetic-lsi-reference`
- **Tags:** kinetic, lsi, theorem
- **Source:** [09_kl_convergence.md § Step 1: Kinetic Operator LSI (Existing Result)](source/1_euclidean_gas/09_kl_convergence)

### Mean-Field Cloning Entropy Dissipation
- **Type:** Lemma
- **Label:** `lem-meanfield-cloning-dissipation-hybrid`
- **Tags:** cloning, entropy, lemma, mean-field
- **Source:** [09_kl_convergence.md § Step 2: Cloning Operator Contraction (Mean-Field P](source/1_euclidean_gas/09_kl_convergence)

### Composition of LSI Operators (Reference)
- **Type:** Theorem
- **Label:** `thm-composition-reference`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § Step 3: Composition via Existing Theorem](source/1_euclidean_gas/09_kl_convergence)

### Discrete Dirichlet Form
- **Type:** Definition
- **Label:** `def-discrete-dirichlet`
- **Tags:** general
- **Source:** [09_kl_convergence.md § Step 4: Discrete-Time LSI Formulation](source/1_euclidean_gas/09_kl_convergence)

### Discrete-Time LSI
- **Type:** Theorem
- **Label:** `thm-discrete-lsi-hybrid`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § Step 4: Discrete-Time LSI Formulation](source/1_euclidean_gas/09_kl_convergence)

### Exponential Convergence from LSI
- **Type:** Theorem
- **Label:** `thm-exp-convergence-hybrid`
- **Tags:** convergence, lsi, theorem
- **Source:** [09_kl_convergence.md § Step 5: Exponential KL Convergence](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL-Convergence via Mean-Field Generator Analysis
- **Type:** Theorem
- **Label:** `thm-meanfield-lsi-standalone`
- **Tags:** convergence, lsi, mean-field, theorem
- **Source:** [09_kl_convergence.md § Main Result](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive LSI for Kinetic Operator
- **Type:** Theorem
- **Label:** `thm-kinetic-lsi-standalone`
- **Tags:** kinetic, lsi, theorem
- **Source:** [09_kl_convergence.md § 1.4.](source/1_euclidean_gas/09_kl_convergence)

### Mean-Field Cloning Contraction
- **Type:** Lemma
- **Label:** `lem-cloning-contraction-standalone`
- **Tags:** cloning, lemma, mean-field
- **Source:** [09_kl_convergence.md § 2.3.](source/1_euclidean_gas/09_kl_convergence)

### Composition of Kinetic and Cloning Operators
- **Type:** Theorem
- **Label:** `thm-composition-standalone`
- **Tags:** cloning, kinetic, theorem
- **Source:** [09_kl_convergence.md § Section 3: Composition](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL Convergence
- **Type:** Theorem
- **Label:** `thm-exp-convergence-standalone`
- **Tags:** convergence, theorem
- **Source:** [09_kl_convergence.md § Section 4: Exponential Convergence](source/1_euclidean_gas/09_kl_convergence)

### Log-Concavity of the Quasi-Stationary Distribution (Current Requirement)
- **Type:** Axiom
- **Label:** `ax-qsd-log-concave-recap`
- **Tags:** axiom, qsd
- **Source:** [09_kl_convergence.md § 0.1.](source/1_euclidean_gas/09_kl_convergence)

### Confining Potential (from 04_convergence.md, Axiom 1.3.1)
- **Type:** Axiom
- **Label:** `ax-confining-recap`
- **Tags:** axiom, convergence, fitness
- **Source:** [09_kl_convergence.md § 0.2.](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL Convergence for Non-Convex Fitness (Informal)
- **Type:** Theorem
- **Label:** `thm-nonconvex-informal`
- **Tags:** convergence, fitness, theorem
- **Source:** [09_kl_convergence.md § 0.3.](source/1_euclidean_gas/09_kl_convergence)

### Confining Potential (Complete Statement)
- **Type:** Axiom
- **Label:** `ax-confining-complete`
- **Tags:** axiom, fitness
- **Source:** [09_kl_convergence.md § 1.1.1.](source/1_euclidean_gas/09_kl_convergence)

### Villani's Hypocoercivity (Simplified)
- **Type:** Theorem
- **Label:** `thm-villani-hypocoercivity`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 2.1.1.](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercivity for Piecewise Smooth Confining Potentials
- **Type:** Proposition
- **Label:** `prop-hypocoercivity-piecewise`
- **Tags:** fitness, proposition
- **Source:** [09_kl_convergence.md § 2.1.3.](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive LSI for Discrete-Time Kinetic Operator
- **Type:** Lemma
- **Label:** `lem-kinetic-lsi-hypocoercive`
- **Tags:** kinetic, lemma, lsi
- **Source:** [09_kl_convergence.md § 2.2.2.](source/1_euclidean_gas/09_kl_convergence)

### N-Particle Hypocoercive LSI
- **Type:** Corollary
- **Label:** `cor-n-particle-hypocoercive`
- **Tags:** corollary, lsi
- **Source:** [09_kl_convergence.md § 2.3.1.](source/1_euclidean_gas/09_kl_convergence)

### Discrete Status-Change Metric
- **Type:** Definition
- **Label:** `def-status-metric`
- **Tags:** metric
- **Source:** [09_kl_convergence.md § 3.2.](source/1_euclidean_gas/09_kl_convergence)

### Lipschitz Continuity of Softmax-Weighted Companion Selection
- **Type:** Lemma
- **Label:** `lem-softmax-lipschitz-status`
- **Tags:** lemma, lipschitz
- **Source:** [09_kl_convergence.md § 3.3.](source/1_euclidean_gas/09_kl_convergence)

### Dobrushin Contraction for Euclidean Gas
- **Type:** Theorem
- **Label:** `thm-dobrushin-contraction`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 3.4.](source/1_euclidean_gas/09_kl_convergence)

### Exponential Convergence in $d_{\text{status}}$ Metric
- **Type:** Theorem
- **Label:** `thm-exponential-convergence-status`
- **Tags:** convergence, metric, theorem
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)

### Exponential KL Convergence for Non-Convex Fitness Landscapes
- **Type:** Theorem
- **Label:** `thm-nonconvex-main`
- **Tags:** convergence, fitness, theorem
- **Source:** [09_kl_convergence.md § 4.1.](source/1_euclidean_gas/09_kl_convergence)

### Why Composition Fails
- **Type:** Observation
- **Label:** `obs-composition-failure`
- **Tags:** observation
- **Source:** [09_kl_convergence.md § 4.5.1.](source/1_euclidean_gas/09_kl_convergence)

### Foster-Lyapunov Drift (Unconditional)
- **Type:** Theorem
- **Label:** `thm-fl-recap`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § What We Have (Unconditional)](source/1_euclidean_gas/09_kl_convergence)

### Logarithmic Sobolev Inequality (Target)
- **Type:** Theorem
- **Label:** `thm-lsi-target`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § What We Want](source/1_euclidean_gas/09_kl_convergence)

### Random Walk on ℤ
- **Type:** Example
- **Label:** `ex-fl-no-lsi`
- **Tags:** lsi
- **Source:** [09_kl_convergence.md § The Gap](source/1_euclidean_gas/09_kl_convergence)

### Classical Bakry-Émery Criterion
- **Type:** Theorem
- **Label:** `thm-bakry-emery-classical`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 0.2](source/1_euclidean_gas/09_kl_convergence)

### Villani's Hypocoercivity (Informal)
- **Type:** Theorem
- **Label:** `thm-villani-hypocoercivity-recap`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § Villani's Hypocoercivity (2009)](source/1_euclidean_gas/09_kl_convergence)

### Synergistic Foster-Lyapunov (Established)
- **Type:** Theorem
- **Label:** `thm-fl-established`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 1.1](source/1_euclidean_gas/09_kl_convergence)

### Hypocoercive LSI for Ψ_kin (Established)
- **Type:** Lemma
- **Label:** `lem-kinetic-lsi-established`
- **Tags:** kinetic, lemma, lsi
- **Source:** [09_kl_convergence.md § 1.2](source/1_euclidean_gas/09_kl_convergence)

### Dobrushin Contraction (Established)
- **Type:** Theorem
- **Label:** `thm-dobrushin-established`
- **Tags:** theorem
- **Source:** [09_kl_convergence.md § 1.3](source/1_euclidean_gas/09_kl_convergence)

### Unconditional LSI for Euclidean Gas (TARGET)
- **Type:** Theorem
- **Label:** `thm-unconditional-lsi`
- **Tags:** lsi, theorem
- **Source:** [09_kl_convergence.md § 3.1](source/1_euclidean_gas/09_kl_convergence)

---

### Source: 10_qsd_exchangeability_theory.md {#10_qsd_exchangeability_theory}

### Exchangeability of the QSD
- **Type:** Theorem
- **Label:** `thm-qsd-exchangeability`
- **Tags:** qsd, theorem
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.1 Main Result](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Mixture Representation (Hewitt-Savage)
- **Type:** Theorem
- **Label:** `thm-hewitt-savage-representation`
- **Tags:** theorem
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.2 Hewitt-Savage Representation](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Single-Particle Marginal
- **Type:** Definition
- **Label:** `def-single-particle-marginal`
- **Tags:** general
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.3 Single-Particle Marginal](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Marginal as Mixture Average
- **Type:** Proposition
- **Label:** `prop-marginal-mixture`
- **Tags:** proposition
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.3 Single-Particle Marginal](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Propagation of Chaos
- **Type:** Theorem
- **Label:** `thm-propagation-chaos-qsd`
- **Tags:** propagation-chaos, qsd, theorem
- **Source:** [10_qsd_exchangeability_theory.md § A1.2.1 Main Convergence Result](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Quantitative Decorrelation
- **Type:** Theorem
- **Label:** `thm-correlation-decay`
- **Tags:** theorem
- **Source:** [10_qsd_exchangeability_theory.md § A1.2.2 Correlation Decay](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### N-Uniform LSI via Hypocoercivity
- **Type:** Theorem
- **Label:** `thm-n-uniform-lsi-exchangeable`
- **Tags:** lsi, propagation-chaos, theorem
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.1 LSI for Exchangeable Measures](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Conditional Gaussian Structure
- **Type:** Lemma
- **Label:** `lem-conditional-gaussian-qsd`
- **Tags:** lemma, qsd
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.1 LSI for Exchangeable Measures](source/1_euclidean_gas/10_qsd_exchangeability_theory)

### Mean-Field LSI from N-Uniform Bounds
- **Type:** Corollary
- **Label:** `cor-mean-field-lsi`
- **Tags:** corollary, lsi, mean-field
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.2 Implications for Mean-Field LSI](source/1_euclidean_gas/10_qsd_exchangeability_theory)

## Chapter 2: Geometric Gas

**Entries:** 200

---

### Source: 11_geometric_gas.md {#11_geometric_gas}

### Localization Kernel
- **Type:** Definition
- **Label:** `def-localization-kernel`
- **Tags:** general
- **Source:** [11_geometric_gas.md § 1.0.2.](source/2_geometric_gas/11_geometric_gas)

### Localized Mean-Field Moments
- **Type:** Definition
- **Label:** `def-localized-mean-field-moments`
- **Tags:** mean-field
- **Source:** [11_geometric_gas.md § 1.0.3.](source/2_geometric_gas/11_geometric_gas)

### Unified Localized Z-Score
- **Type:** Definition
- **Label:** `def-unified-z-score`
- **Tags:** general
- **Source:** [11_geometric_gas.md § 1.0.4.](source/2_geometric_gas/11_geometric_gas)

### Limiting Behavior of the Unified Pipeline
- **Type:** Proposition
- **Label:** `prop-limiting-regimes`
- **Tags:** convergence, proposition
- **Source:** [11_geometric_gas.md § 1.0.5.](source/2_geometric_gas/11_geometric_gas)

### The Adaptive Viscous Fluid SDE
- **Type:** Definition
- **Label:** `def-hybrid-sde`
- **Tags:** langevin
- **Source:** [11_geometric_gas.md § 2.](source/2_geometric_gas/11_geometric_gas)

### Regularized Hessian Diffusion Tensor
- **Type:** Definition
- **Label:** `def-regularized-hessian-tensor`
- **Tags:** langevin
- **Source:** [11_geometric_gas.md § 2.](source/2_geometric_gas/11_geometric_gas)

### Localized Mean-Field Fitness Potential
- **Type:** Definition
- **Label:** `def-localized-mean-field-fitness`
- **Tags:** fitness, mean-field
- **Source:** [11_geometric_gas.md § 2.1.](source/2_geometric_gas/11_geometric_gas)

### Axiom of a Globally Confining Potential
- **Type:** Axiom
- **Label:** `ax:confining-potential-hybrid`
- **Tags:** axiom, fitness
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)

### Axiom of Positive Friction
- **Type:** Axiom
- **Label:** `ax:positive-friction-hybrid`
- **Tags:** axiom
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)

### Foundational Cloning and Environmental Axioms
- **Type:** Axiom
- **Label:** `ax:cloning-env-hybrid`
- **Tags:** axiom, cloning
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)

### k-Uniform Boundedness of the Adaptive Force (ρ-Dependent)
- **Type:** Proposition
- **Label:** `prop:bounded-adaptive-force`
- **Tags:** proposition
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)

### Axiom of a Well-Behaved Viscous Kernel
- **Type:** Axiom
- **Label:** `ax:viscous-kernel`
- **Tags:** axiom
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)

### k-Uniform Ellipticity by Construction (Proven in Chapter 4)
- **Type:** Proposition
- **Label:** `prop:ueph-by-construction`
- **Tags:** proposition
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)

### k-Uniform Ellipticity of the Regularized Metric
- **Type:** Theorem
- **Label:** `thm-ueph`
- **Tags:** metric, theorem
- **Source:** [11_geometric_gas.md § 4.1.](source/2_geometric_gas/11_geometric_gas)

### N-Uniform Boundedness of the Pure Hessian
- **Type:** Lemma
- **Label:** `lem-hessian-bounded`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)

### Rigorous Boundedness of the Hessian
- **Type:** Lemma
- **Label:** `lem-hessian-bounded-rigorous`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)

### Failure of Uniformity Without Regularization
- **Type:** Lemma
- **Label:** `lem-hessian-explosion`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)

### Existence and Uniqueness of Solutions
- **Type:** Corollary
- **Label:** `cor-wellposed`
- **Tags:** corollary
- **Source:** [11_geometric_gas.md § 4.3.](source/2_geometric_gas/11_geometric_gas)

### The Backbone SDE
- **Type:** Definition
- **Label:** `def-backbone-sde`
- **Tags:** langevin
- **Source:** [11_geometric_gas.md § 5.1.](source/2_geometric_gas/11_geometric_gas)

### Geometric Ergodicity of the Backbone
- **Type:** Theorem
- **Label:** `thm-backbone-convergence`
- **Tags:** convergence, geometry, metric, theorem
- **Source:** [11_geometric_gas.md § 5.2.](source/2_geometric_gas/11_geometric_gas)

### Stratonovich Chain Rule for Lyapunov Functions
- **Type:** Theorem
- **Label:** `thm-strat-chain`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § 5.4.](source/2_geometric_gas/11_geometric_gas)

### Stratonovich Drift for the Hybrid System
- **Type:** Definition
- **Label:** `def-strat-drift`
- **Tags:** general
- **Source:** [11_geometric_gas.md § 5.4.](source/2_geometric_gas/11_geometric_gas)

### N-Uniform Bounded Perturbation from Adaptive Force
- **Type:** Lemma
- **Label:** `lem-adaptive-force-bounded`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 6.2.](source/2_geometric_gas/11_geometric_gas)

### Dissipative Contribution from Viscous Force
- **Type:** Lemma
- **Label:** `lem-viscous-dissipative`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 6.3.](source/2_geometric_gas/11_geometric_gas)

### Bounded Change from Adaptive Diffusion
- **Type:** Lemma
- **Label:** `lem-diffusion-bounded`
- **Tags:** langevin, lemma
- **Source:** [11_geometric_gas.md § 6.4.](source/2_geometric_gas/11_geometric_gas)

### Total Perturbation Bound (ρ-Dependent)
- **Type:** Corollary
- **Label:** `cor-total-perturbation`
- **Tags:** corollary
- **Source:** [11_geometric_gas.md § 6.5.](source/2_geometric_gas/11_geometric_gas)

### Foster-Lyapunov Drift for the ρ-Localized Geometric Viscous Fluid Model
- **Type:** Theorem
- **Label:** `thm-fl-drift-adaptive`
- **Tags:** geometry, metric, theorem
- **Source:** [11_geometric_gas.md § 7.1.](source/2_geometric_gas/11_geometric_gas)

### Exponential Convergence
- **Type:** Corollary
- **Label:** `cor-exp-convergence`
- **Tags:** convergence, corollary
- **Source:** [11_geometric_gas.md § 7.3.](source/2_geometric_gas/11_geometric_gas)

### The N-Particle Generator for the Adaptive System
- **Type:** Definition
- **Label:** `def-n-particle-generator-lsi`
- **Tags:** lsi
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)

### Relative Entropy and Fisher Information
- **Type:** Definition
- **Label:** `def-entropy-fisher-lsi`
- **Tags:** entropy, lsi
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)

### Logarithmic Sobolev Inequality (LSI)
- **Type:** Definition
- **Label:** `def-lsi`
- **Tags:** lsi
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)

### N-Uniform Log-Sobolev Inequality for the Geometric Viscous Fluid Model
- **Type:** Theorem
- **Label:** `thm-lsi-adaptive-gas`
- **Tags:** geometry, lsi, metric, theorem
- **Source:** [11_geometric_gas.md § 8.3.](source/2_geometric_gas/11_geometric_gas)

### Exponential Convergence in Relative Entropy
- **Type:** Corollary
- **Label:** `cor-entropy-convergence-lsi`
- **Tags:** convergence, corollary, entropy, lsi
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)

### Geometric Ergodicity via LSI
- **Type:** Corollary
- **Label:** `cor-geometric-ergodicity-lsi`
- **Tags:** corollary, geometry, lsi, metric
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)

### Concentration of Measure
- **Type:** Remark
- **Label:** `rem-concentration-lsi`
- **Tags:** lsi
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)

### Existence and Uniqueness of the QSD
- **Type:** Theorem
- **Label:** `thm-qsd-existence`
- **Tags:** qsd, theorem
- **Source:** [11_geometric_gas.md § 9.1.](source/2_geometric_gas/11_geometric_gas)

### Formal Analogy and Evidence
- **Type:** Remark
- **Label:** `rem-wfr-analogy`
- **Tags:** general
- **Source:** [11_geometric_gas.md § 9.2.](source/2_geometric_gas/11_geometric_gas)

### Logarithmic Sobolev Inequality for the Mean-Field Generator
- **Type:** Theorem
- **Label:** `thm-lsi-mean-field`
- **Tags:** lsi, mean-field, theorem
- **Source:** [11_geometric_gas.md § 9.3.](source/2_geometric_gas/11_geometric_gas)

### Decomposition of Entropy Dissipation
- **Type:** Lemma
- **Label:** `lem-dissipation-decomp`
- **Tags:** entropy, lemma
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)

### Microlocal Decomposition
- **Type:** Definition
- **Label:** `def-microlocal`
- **Tags:** general
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)

### Microscopic Coercivity (Step A)
- **Type:** Lemma
- **Label:** `lem-micro-coercivity`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)

### Macroscopic Transport (Step B)
- **Type:** Lemma
- **Label:** `lem-macro-transport`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)

### Microscopic Regularization (Step C)
- **Type:** Lemma
- **Label:** `lem-micro-reg`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)

### Derivatives of Localization Weights
- **Type:** Lemma
- **Label:** `lem-weight-derivatives`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)

### First Derivative of Localized Mean
- **Type:** Lemma
- **Label:** `lem-mean-first-derivative`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)

### Second Derivative of Localized Mean
- **Type:** Lemma
- **Label:** `lem-mean-second-derivative`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)

### k-Uniform Gradient of Localized Variance
- **Type:** Lemma
- **Label:** `lem-variance-gradient`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)

### k-Uniform Hessian of Localized Variance
- **Type:** Lemma
- **Label:** `lem-variance-hessian`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)

### C¹ Regularity and k-Uniform Gradient Bound
- **Type:** Theorem
- **Label:** `thm-c1-regularity`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § A.3. Theorem A.1: Uniform C¹ Bound on the Fitness ](source/2_geometric_gas/11_geometric_gas)

### C² Regularity and k-Uniform Hessian Bound
- **Type:** Theorem
- **Label:** `thm-c2-regularity`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § A.4. Theorem A.2: Uniform C² Bound on the Fitness ](source/2_geometric_gas/11_geometric_gas)

### Verification of Axioms 3.2.1 and 3.2.3
- **Type:** Corollary
- **Label:** `cor-axioms-verified`
- **Tags:** corollary
- **Source:** [11_geometric_gas.md § A.5. Corollary: Implications for the Main Text](source/2_geometric_gas/11_geometric_gas)

### Signal Generation for the Adaptive Model
- **Type:** Theorem
- **Label:** `thm-signal-generation-adaptive`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § B.2. Hypothesis 1: Signal Generation (Geometry-Bas](source/2_geometric_gas/11_geometric_gas)

### Variance-to-Gap (from 03_cloning.md, Lemma 7.3.1)
- **Type:** Lemma
- **Label:** `lem-variance-to-gap-adaptive`
- **Tags:** cloning, lemma
- **Source:** [11_geometric_gas.md § B.3.1. The Variance-to-Gap Lemma (Universal)](source/2_geometric_gas/11_geometric_gas)

### Uniform Bounds on the ρ-Localized Pipeline
- **Type:** Lemma
- **Label:** `lem-rho-pipeline-bounds`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § B.3.2. Uniform Bounds on ρ-Dependent Pipeline Comp](source/2_geometric_gas/11_geometric_gas)

### Raw-Gap to Rescaled-Gap for ρ-Localized Pipeline
- **Type:** Lemma
- **Label:** `lem-raw-to-rescaled-gap-rho`
- **Tags:** lemma
- **Source:** [11_geometric_gas.md § B.3.3. Raw-Gap to Rescaled-Gap Propagation (ρ-Depe](source/2_geometric_gas/11_geometric_gas)

### Logarithmic Gap Bounds (from 03_cloning.md, Lemma 7.5.1)
- **Type:** Lemma
- **Label:** `lem-log-gap-bounds-adaptive`
- **Tags:** cloning, lemma
- **Source:** [11_geometric_gas.md § B.4.1. Foundational Statistical Lemmas (ρ-Independ](source/2_geometric_gas/11_geometric_gas)

### Lower Bound on Corrective Diversity Signal (ρ-Dependent)
- **Type:** Proposition
- **Label:** `prop-diversity-signal-rho`
- **Tags:** proposition
- **Source:** [11_geometric_gas.md § B.4.2. ρ-Dependent Lower Bound on Corrective Diver](source/2_geometric_gas/11_geometric_gas)

### Axiom-Based Bound on Logarithmic Reward Gap (ρ-Dependent)
- **Type:** Proposition
- **Label:** `prop-reward-bias-rho`
- **Tags:** fitness, proposition
- **Source:** [11_geometric_gas.md § B.4.3. ρ-Dependent Upper Bound on Adversarial Rewa](source/2_geometric_gas/11_geometric_gas)

### ρ-Dependent Stability Condition for Intelligent Targeting
- **Type:** Theorem
- **Label:** `thm-stability-condition-rho`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § B.4.4. The ρ-Dependent Stability Condition](source/2_geometric_gas/11_geometric_gas)

### Keystone Lemma for the ρ-Localized Adaptive Model
- **Type:** Theorem
- **Label:** `thm-keystone-adaptive`
- **Tags:** theorem
- **Source:** [11_geometric_gas.md § B.5. Conclusion: The Keystone Lemma Holds for the ](source/2_geometric_gas/11_geometric_gas)

---

### Source: 12_symmetries_geometric_gas.md {#12_symmetries_geometric_gas}

### Swarm Configuration Space
- **Type:** Definition
- **Label:** `def-swarm-config-space`
- **Tags:** general
- **Source:** [12_symmetries_geometric_gas.md § 2.1.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Algorithmic Projection Space
- **Type:** Definition
- **Label:** `def-algorithmic-projection-space`
- **Tags:** general
- **Source:** [12_symmetries_geometric_gas.md § 2.1.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Symmetry Transformation
- **Type:** Definition
- **Label:** `def-symmetry-transformation`
- **Tags:** symmetry
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Permutation Group
- **Type:** Definition
- **Label:** `def-permutation-group`
- **Tags:** general
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Euclidean Group Actions
- **Type:** Definition
- **Label:** `def-euclidean-group-actions`
- **Tags:** general
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)

### ρ-Localized Fitness Potential
- **Type:** Definition
- **Label:** `def-rho-fitness-potential`
- **Tags:** fitness
- **Source:** [12_symmetries_geometric_gas.md § 2.3.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Emergent Riemannian Metric
- **Type:** Definition
- **Label:** `def-emergent-metric`
- **Tags:** geometry, metric
- **Source:** [12_symmetries_geometric_gas.md § 2.3.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Permutation Invariance
- **Type:** Theorem
- **Label:** `thm-permutation-symmetry`
- **Tags:** symmetry, theorem
- **Source:** [12_symmetries_geometric_gas.md § 3.1.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Exchangeability of the QSD
- **Type:** Corollary
- **Label:** `cor-qsd-exchangeable`
- **Tags:** corollary, propagation-chaos, qsd
- **Source:** [12_symmetries_geometric_gas.md § 3.1.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Conditional Translation Equivariance
- **Type:** Theorem
- **Label:** `thm-translation-equivariance`
- **Tags:** theorem
- **Source:** [12_symmetries_geometric_gas.md § 3.2.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Breaking of Translation Symmetry
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** symmetry
- **Source:** [12_symmetries_geometric_gas.md § 3.2.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Rotational Equivariance
- **Type:** Theorem
- **Label:** `thm-rotation-equivariance`
- **Tags:** theorem
- **Source:** [12_symmetries_geometric_gas.md § 3.3.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Radially Symmetric Fitness Landscapes
- **Type:** Example
- **Label:** `unlabeled`
- **Tags:** fitness, metric
- **Source:** [12_symmetries_geometric_gas.md § 3.3.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Fitness Potential Scaling Symmetry
- **Type:** Theorem
- **Label:** `thm-fitness-scaling`
- **Tags:** fitness, symmetry, theorem
- **Source:** [12_symmetries_geometric_gas.md § 3.4.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Dimensionless Parameter
- **Type:** Corollary
- **Label:** `cor-dimensionless-ratio`
- **Tags:** corollary
- **Source:** [12_symmetries_geometric_gas.md § 3.4.](source/2_geometric_gas/12_symmetries_geometric_gas)

### Time-Reversal Asymmetry
- **Type:** Theorem
- **Label:** `thm-irreversibility`
- **Tags:** symmetry, theorem
- **Source:** [12_symmetries_geometric_gas.md § 3.5.](source/2_geometric_gas/12_symmetries_geometric_gas)

### H-Theorem for Geometric Gas
- **Type:** Proposition
- **Label:** `prop-h-theorem`
- **Tags:** geometry, metric, proposition
- **Source:** [12_symmetries_geometric_gas.md § 3.5.](source/2_geometric_gas/12_symmetries_geometric_gas)

---

### Source: 13_geometric_gas_c3_regularity.md {#13_geometric_gas_c3_regularity}

### Telescoping Identity for Derivatives
- **Type:** Lemma
- **Label:** `lem-telescoping-derivatives`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 2.5.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Measurement Function $C^3$ Regularity
- **Type:** Assumption
- **Label:** `assump-c3-measurement`
- **Tags:** assumption
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Localization Kernel $C^3$ Regularity
- **Type:** Assumption
- **Label:** `assump-c3-kernel`
- **Tags:** assumption
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Rescale Function $C^3$ Regularity
- **Type:** Assumption
- **Label:** `assump-c3-rescale`
- **Tags:** assumption
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Regularized Standard Deviation $C^\infty$ Regularity
- **Type:** Assumption
- **Label:** `assump-c3-patch`
- **Tags:** assumption
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Third Derivative of Localization Weights
- **Type:** Lemma
- **Label:** `lem-weight-third-derivative`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 4.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### k-Uniform Third Derivative of Localized Mean
- **Type:** Lemma
- **Label:** `lem-mean-third-derivative`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 5.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### k-Uniform Third Derivative of Localized Variance
- **Type:** Lemma
- **Label:** `lem-variance-third-derivative`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 5.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Chain Rule for Regularized Standard Deviation
- **Type:** Lemma
- **Label:** `lem-patch-chain-rule`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 6.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Third Derivative Bound for Regularized Standard Deviation
- **Type:** Lemma
- **Label:** `lem-patch-third-derivative`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 6.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### k-Uniform Third Derivative of Z-Score
- **Type:** Lemma
- **Label:** `lem-zscore-third-derivative`
- **Tags:** lemma
- **Source:** [13_geometric_gas_c3_regularity.md § 7.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### $C^3$ Regularity of the �-Localized Fitness Potential
- **Type:** Theorem
- **Label:** `thm-c3-regularity`
- **Tags:** fitness, theorem
- **Source:** [13_geometric_gas_c3_regularity.md § 8.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### ρ-Scaling of Third Derivative Bound
- **Type:** Proposition
- **Label:** `prop-scaling-kv3`
- **Tags:** proposition
- **Source:** [13_geometric_gas_c3_regularity.md § 8.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### BAOAB Discretization Validity
- **Type:** Corollary
- **Label:** `cor-baoab-validity`
- **Tags:** corollary
- **Source:** [13_geometric_gas_c3_regularity.md § 9.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### $C^3$ Regularity of Total Lyapunov Function
- **Type:** Corollary
- **Label:** `cor-lyapunov-c3`
- **Tags:** corollary
- **Source:** [13_geometric_gas_c3_regularity.md § 9.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### $C^3$ Perturbation Structure
- **Type:** Corollary
- **Label:** `cor-smooth-perturbation`
- **Tags:** corollary
- **Source:** [13_geometric_gas_c3_regularity.md § 9.3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Regularity Hierarchy Complete
- **Type:** Corollary
- **Label:** `cor-regularity-hierarchy`
- **Tags:** corollary
- **Source:** [13_geometric_gas_c3_regularity.md § 9.4.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Scaling of Third-Derivative Bound
- **Type:** Proposition
- **Label:** `prop-scaling-k-v-3`
- **Tags:** proposition
- **Source:** [13_geometric_gas_c3_regularity.md § 10.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Time Step Constraint from $C^3$ Regularity
- **Type:** Proposition
- **Label:** `prop-timestep-constraint`
- **Tags:** proposition
- **Source:** [13_geometric_gas_c3_regularity.md § 10.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Explicit Formula for $K_{V,3}(\rho)$
- **Type:** Proposition
- **Label:** `prop-explicit-k-v-3`
- **Tags:** proposition
- **Source:** [13_geometric_gas_c3_regularity.md § 10.3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Continuity of Third Derivatives
- **Type:** Theorem
- **Label:** `thm-continuity-third-derivatives`
- **Tags:** theorem
- **Source:** [13_geometric_gas_c3_regularity.md § 11.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

### Regularized Standard Deviation (Implementation)
- **Type:** Definition
- **Label:** `def-reg-std-implementation`
- **Tags:** general
- **Source:** [13_geometric_gas_c3_regularity.md § 12.5.](source/2_geometric_gas/13_geometric_gas_c3_regularity)

---

### Source: 14_geometric_gas_c4_regularity.md {#14_geometric_gas_c4_regularity}

### C⁴ Measurement Function
- **Type:** Assumption
- **Label:** `assump-c4-measurement`
- **Tags:** assumption
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### C⁴ Localization Kernel
- **Type:** Assumption
- **Label:** `assump-c4-kernel`
- **Tags:** assumption
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### C⁴ Rescale Function
- **Type:** Assumption
- **Label:** `assump-c4-rescale`
- **Tags:** assumption
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### C^∞ Regularized Standard Deviation
- **Type:** Assumption
- **Label:** `assump-c4-regularized-std`
- **Tags:** assumption
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Bounded Measurement Range
- **Type:** Assumption
- **Label:** `assump-c4-bounded-measurement`
- **Tags:** assumption
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### QSD Bounded Density (Regularity Condition R2)
- **Type:** Assumption
- **Label:** `assump-c4-qsd-bounded-density`
- **Tags:** assumption, qsd
- **Source:** [14_geometric_gas_c4_regularity.md § 3.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth Derivative of Localization Weights
- **Type:** Lemma
- **Label:** `lem-weight-fourth-derivative`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 4.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Telescoping Property for Fourth Derivative
- **Type:** Lemma
- **Label:** `lem-weight-telescoping-fourth`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 4.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth Derivative of Localized Mean
- **Type:** Lemma
- **Label:** `lem-mean-fourth-derivative`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 5.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth Derivative of Localized Variance
- **Type:** Lemma
- **Label:** `lem-variance-fourth-derivative`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 5.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Chain Rule for $\sigma'_{\text{reg}}$
- **Type:** Lemma
- **Label:** `lem-reg-fourth-chain`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 6.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth Derivative of Z-Score
- **Type:** Lemma
- **Label:** `lem-zscore-fourth-derivative`
- **Tags:** lemma
- **Source:** [14_geometric_gas_c4_regularity.md § 7.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### C⁴ Regularity of Fitness Potential
- **Type:** Theorem
- **Label:** `thm-c4-regularity`
- **Tags:** fitness, theorem
- **Source:** [14_geometric_gas_c4_regularity.md § 8.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Hessian Lipschitz Continuity
- **Type:** Corollary
- **Label:** `cor-hessian-lipschitz`
- **Tags:** corollary, lipschitz
- **Source:** [14_geometric_gas_c4_regularity.md § 9.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth-Order Integrator Compatibility
- **Type:** Corollary
- **Label:** `cor-fourth-order-integrators`
- **Tags:** corollary
- **Source:** [14_geometric_gas_c4_regularity.md § 9.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Brascamp-Lieb Inequality (Conditional)
- **Type:** Corollary
- **Label:** `cor-brascamp-lieb`
- **Tags:** corollary
- **Source:** [14_geometric_gas_c4_regularity.md § 9.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Bakry-Émery Γ₂ Criterion (Conditional)
- **Type:** Proposition
- **Label:** `prop-bakry-emery-gamma2`
- **Tags:** proposition
- **Source:** [14_geometric_gas_c4_regularity.md § 9.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Fourth-Derivative Scaling
- **Type:** Proposition
- **Label:** `prop-scaling-k-v-4`
- **Tags:** proposition
- **Source:** [14_geometric_gas_c4_regularity.md § 10.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

### Time Step Constraint (Corrected)
- **Type:** Proposition
- **Label:** `prop-timestep-c4`
- **Tags:** proposition
- **Source:** [14_geometric_gas_c4_regularity.md § 10.](source/2_geometric_gas/14_geometric_gas_c4_regularity)

---

### Source: 15_geometric_gas_lsi_proof.md {#15_geometric_gas_lsi_proof}

### Quasi-Stationary Distribution (QSD)
- **Type:** Definition
- **Label:** `def-qsd-adaptive`
- **Tags:** qsd
- **Source:** [15_geometric_gas_lsi_proof.md § 3.3.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### Log-Sobolev Inequality
- **Type:** Definition
- **Label:** `def-lsi-adaptive`
- **Tags:** lsi
- **Source:** [15_geometric_gas_lsi_proof.md § 3.4.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### N-Uniform Third Derivative Bound for Fitness (PROVEN)
- **Type:** Theorem
- **Label:** `thm-fitness-third-deriv-proven`
- **Tags:** fitness, theorem
- **Source:** [15_geometric_gas_lsi_proof.md § 8.2.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### N-Uniform Poincaré Inequality for QSD Velocities (CORRECTED PROOF)
- **Type:** Theorem
- **Label:** `thm-qsd-poincare-rigorous`
- **Tags:** qsd, theorem
- **Source:** [15_geometric_gas_lsi_proof.md § 8.3.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### N-Uniform Drift Perturbation Bounds
- **Type:** Theorem
- **Label:** `thm-drift-perturbation-bounds`
- **Tags:** theorem
- **Source:** [15_geometric_gas_lsi_proof.md § 8.5.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### Verification of Cattiaux-Guillin Hypotheses
- **Type:** Theorem
- **Label:** `thm-cattiaux-guillin-verification`
- **Tags:** theorem
- **Source:** [15_geometric_gas_lsi_proof.md § 8.5.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

### N-Uniform Log-Sobolev Inequality for Geometric Viscous Fluid Model
- **Type:** Theorem
- **Label:** `thm-adaptive-lsi-main`
- **Tags:** geometry, lsi, metric, theorem
- **Source:** [15_geometric_gas_lsi_proof.md § 9.1.](source/2_geometric_gas/15_geometric_gas_lsi_proof)

---

### Source: 16_convergence_mean_field.md {#16_convergence_mean_field}

### Mean-Field Revival Operator (Formal)
- **Type:** Definition
- **Label:** `def-revival-operator-formal`
- **Tags:** mean-field
- **Source:** [16_convergence_mean_field.md § 1.2.](source/2_geometric_gas/16_convergence_mean_field)

### Combined Jump Operator
- **Type:** Definition
- **Label:** `def-combined-jump-operator`
- **Tags:** general
- **Source:** [16_convergence_mean_field.md § 1.3.](source/2_geometric_gas/16_convergence_mean_field)

### Finite-N LSI Preservation (Proven)
- **Type:** Theorem
- **Label:** `thm-finite-n-lsi-preservation`
- **Tags:** lsi, theorem
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)

### Data Processing Inequality (Standard Result)
- **Type:** Theorem
- **Label:** `thm-data-processing`
- **Tags:** theorem
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)

### Wasserstein Contraction for Proportional Resampling (Conjecture)
- **Type:** Lemma
- **Label:** `lem-wasserstein-revival`
- **Tags:** cloning, lemma, wasserstein
- **Source:** [16_convergence_mean_field.md § 3.2.](source/2_geometric_gas/16_convergence_mean_field)

### Revival Rate Constraint
- **Type:** Observation
- **Label:** `obs-revival-rate-constraint`
- **Tags:** observation
- **Source:** [16_convergence_mean_field.md § 4.1.](source/2_geometric_gas/16_convergence_mean_field)

### Revival Operator is KL-Expansive (VERIFIED)
- **Type:** Theorem
- **Label:** `thm-revival-kl-expansive`
- **Tags:** theorem
- **Source:** [16_convergence_mean_field.md § 7.1.](source/2_geometric_gas/16_convergence_mean_field)

### Joint Jump Operator NOT Unconditionally Contractive (VERIFIED)
- **Type:** Theorem
- **Label:** `thm-joint-not-contractive`
- **Tags:** theorem
- **Source:** [16_convergence_mean_field.md § 7.2.](source/2_geometric_gas/16_convergence_mean_field)

### Stage 0 COMPLETE (VERIFIED)
- **Type:** Theorem
- **Label:** `thm-stage0-complete`
- **Tags:** theorem
- **Source:** [16_convergence_mean_field.md § 8.1.](source/2_geometric_gas/16_convergence_mean_field)

### Quasi-Stationary Distribution (QSD)
- **Type:** Definition
- **Label:** `def-qsd-mean-field`
- **Tags:** mean-field, qsd
- **Source:** [16_convergence_mean_field.md § 0.2.](source/2_geometric_gas/16_convergence_mean_field)

### Framework Assumptions
- **Type:** Assumption
- **Label:** `assump-qsd-existence`
- **Tags:** assumption, qsd
- **Source:** [16_convergence_mean_field.md § 1.2.](source/2_geometric_gas/16_convergence_mean_field)

### QSD Existence via Nonlinear Fixed-Point
- **Type:** Theorem
- **Label:** `thm-qsd-existence-corrected`
- **Tags:** qsd, theorem
- **Source:** [16_convergence_mean_field.md § 1.4.](source/2_geometric_gas/16_convergence_mean_field)

### QSD Stability (Champagnat-Villemonais)
- **Type:** Theorem
- **Label:** `thm-qsd-stability`
- **Tags:** qsd, theorem
- **Source:** [16_convergence_mean_field.md § Step 3c: QSD Stability](source/2_geometric_gas/16_convergence_mean_field)

### Hörmander's Condition
- **Type:** Lemma
- **Label:** `lem-hormander`
- **Tags:** lemma
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)

### Hypoelliptic Regularity
- **Type:** Corollary
- **Label:** `cor-hypoelliptic-regularity`
- **Tags:** corollary
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)

### QSD Smoothness
- **Type:** Theorem
- **Label:** `thm-qsd-smoothness`
- **Tags:** qsd, theorem
- **Source:** [16_convergence_mean_field.md § 2.2.](source/2_geometric_gas/16_convergence_mean_field)

### QSD Strict Positivity
- **Type:** Theorem
- **Label:** `thm-qsd-positivity`
- **Tags:** qsd, theorem
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)

### Irreducibility
- **Type:** Lemma
- **Label:** `lem-irreducibility`
- **Tags:** lemma
- **Source:** [16_convergence_mean_field.md § Step 1: Irreducibility of the Process](source/2_geometric_gas/16_convergence_mean_field)

### Strong Maximum Principle
- **Type:** Lemma
- **Label:** `lem-strong-max-principle`
- **Tags:** lemma
- **Source:** [16_convergence_mean_field.md § Step 2: Strong Maximum Principle for Irreducible P](source/2_geometric_gas/16_convergence_mean_field)

### Uniform Velocity Gradient Bound
- **Type:** Proposition
- **Label:** `prop-velocity-gradient-uniform`
- **Tags:** kinetic, proposition
- **Source:** [16_convergence_mean_field.md § 3.2.](source/2_geometric_gas/16_convergence_mean_field)

### Complete Gradient and Laplacian Bounds
- **Type:** Proposition
- **Label:** `prop-complete-gradient-bounds`
- **Tags:** proposition
- **Source:** [16_convergence_mean_field.md § 3.3.](source/2_geometric_gas/16_convergence_mean_field)

### Drift Condition with Quadratic Lyapunov
- **Type:** Lemma
- **Label:** `lem-drift-condition-corrected`
- **Tags:** lemma
- **Source:** [16_convergence_mean_field.md § 4.2.](source/2_geometric_gas/16_convergence_mean_field)

### Exponential Tails for QSD
- **Type:** Theorem
- **Label:** `thm-exponential-tails`
- **Tags:** qsd, theorem
- **Source:** [16_convergence_mean_field.md § 4.3.](source/2_geometric_gas/16_convergence_mean_field)

### KL-Convergence for Mean-Field Euclidean Gas (CORRECTED)
- **Type:** Theorem
- **Label:** `thm-corrected-kl-convergence`
- **Tags:** convergence, mean-field, theorem
- **Source:** [16_convergence_mean_field.md § 4.](source/2_geometric_gas/16_convergence_mean_field)

### Modified Fisher Information
- **Type:** Definition
- **Label:** `def-modified-fisher`
- **Tags:** general
- **Source:** [16_convergence_mean_field.md § 1.3.](source/2_geometric_gas/16_convergence_mean_field)

### Log-Sobolev Inequality (LSI) for QSD
- **Type:** Theorem
- **Label:** `thm-lsi-qsd`
- **Tags:** lsi, qsd, theorem
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)

### Explicit LSI Constant
- **Type:** Theorem
- **Label:** `thm-lsi-constant-explicit`
- **Tags:** lsi, theorem
- **Source:** [16_convergence_mean_field.md § 2.2.](source/2_geometric_gas/16_convergence_mean_field)

### Fisher Information Bound
- **Type:** Lemma
- **Label:** `lem-fisher-bound`
- **Tags:** lemma
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)

### Kinetic Energy Control
- **Type:** Lemma
- **Label:** `lem-kinetic-energy-bound`
- **Tags:** kinetic, lemma
- **Source:** [16_convergence_mean_field.md § 3.2.1.](source/2_geometric_gas/16_convergence_mean_field)

### Entropy $L^1$ Bound
- **Type:** Lemma
- **Label:** `lem-entropy-l1-bound`
- **Tags:** entropy, lemma
- **Source:** [16_convergence_mean_field.md § 4.2.](source/2_geometric_gas/16_convergence_mean_field)

### Exponential Convergence (Local)
- **Type:** Theorem
- **Label:** `thm-exponential-convergence-local`
- **Tags:** convergence, theorem
- **Source:** [16_convergence_mean_field.md § 5.4.](source/2_geometric_gas/16_convergence_mean_field)

### Main Result: Explicit Convergence Rate
- **Type:** Theorem
- **Label:** `thm-main-explicit-rate`
- **Tags:** convergence, theorem
- **Source:** [16_convergence_mean_field.md § 10.](source/2_geometric_gas/16_convergence_mean_field)

### Mean-Field Convergence Rate (Explicit)
- **Type:** Theorem
- **Label:** `thm-alpha-net-explicit`
- **Tags:** convergence, mean-field, theorem
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)

### Optimal Parameter Scaling
- **Type:** Theorem
- **Label:** `thm-optimal-parameter-scaling`
- **Tags:** theorem
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)

### Exponential KL-Convergence in the Mean-Field Limit
- **Type:** Theorem
- **Label:** `thm-mean-field-lsi-main`
- **Tags:** convergence, lsi, mean-field, theorem
- **Source:** [16_convergence_mean_field.md § 1.](source/2_geometric_gas/16_convergence_mean_field)

---

### Source: 18_emergent_geometry.md {#18_emergent_geometry}

### Main Theorem (Informal)
- **Type:** Theorem
- **Label:** `thm-main-informal`
- **Tags:** convergence, adaptive-gas, anisotropic, QSD, geometric-ergodicity, main-result
- **Source:** [18_emergent_geometry.md § 0.5](source/2_geometric_gas/18_emergent_geometry)

### Adaptive Diffusion Tensor
- **Type:** Definition
- **Label:** `def-d-adaptive-diffusion`
- **Tags:** adaptive-gas, diffusion, anisotropic, regularization, Riemannian-metric
- **Source:** [18_emergent_geometry.md § 1.1](source/2_geometric_gas/18_emergent_geometry)

### Why This is a Riemannian Metric
- **Type:** Remark
- **Label:** (inline remark)
- **Tags:** Riemannian-metric, information-geometry, natural-gradient
- **Source:** [18_emergent_geometry.md § 1.1](source/2_geometric_gas/18_emergent_geometry)

### Spectral Floor (Standing Assumption)
- **Type:** Assumption
- **Label:** `assump-spectral-floor`
- **Tags:** regularization, positive-definiteness, spectral-bounds, technical-assumption
- **Source:** [18_emergent_geometry.md § 1.2](source/2_geometric_gas/18_emergent_geometry)

### Uniform Ellipticity by Construction
- **Type:** Theorem
- **Label:** `thm-uniform-ellipticity`
- **Tags:** ellipticity, diffusion-bounds, regularization, N-uniform, anisotropic
- **Source:** [18_emergent_geometry.md § 1.2](source/2_geometric_gas/18_emergent_geometry)

### Lipschitz Continuity of Adaptive Diffusion
- **Type:** Proposition
- **Label:** `prop-lipschitz-diffusion`
- **Tags:** Lipschitz, continuity, diffusion, N-uniform, regularity
- **Source:** [18_emergent_geometry.md § 1.3](source/2_geometric_gas/18_emergent_geometry)

### Kinetic Operator with Adaptive Diffusion
- **Type:** Definition
- **Label:** `def-d-kinetic-operator-adaptive`
- **Tags:** kinetic-operator, Langevin-dynamics, Stratonovich, anisotropic, SDE
- **Source:** [18_emergent_geometry.md § 1.4](source/2_geometric_gas/18_emergent_geometry)

### Comparison to Isotropic Case
- **Type:** Remark
- **Label:** `rem-comparison-isotropic`
- **Tags:** isotropic, comparison, challenges, state-dependent
- **Source:** [18_emergent_geometry.md § 1.4](source/2_geometric_gas/18_emergent_geometry)

### Two Equivalent Formulations
- **Type:** Observation
- **Label:** `obs-two-formulations`
- **Tags:** flat-space, curved-space, equivalence, geometric-perspective
- **Source:** [18_emergent_geometry.md § 1.6](source/2_geometric_gas/18_emergent_geometry)

### Invariance Under Coordinate Changes (Refined)
- **Type:** Theorem
- **Label:** `thm-coordinate-invariance`
- **Tags:** coordinate-invariance, Stratonovich, geometric-invariance, Riemannian
- **Source:** [18_emergent_geometry.md § 1.6](source/2_geometric_gas/18_emergent_geometry)

### Coupled Swarm State
- **Type:** Definition
- **Label:** `def-d-coupled-state`
- **Tags:** coupling, geometric-ergodicity, Lyapunov, two-swarms
- **Source:** [18_emergent_geometry.md § 2.1](source/2_geometric_gas/18_emergent_geometry)

### Coupled Lyapunov Function
- **Type:** Definition
- **Label:** `def-d-coupled-lyapunov`
- **Tags:** Lyapunov, coupling, Wasserstein, variance, boundary, Foster-Lyapunov
- **Source:** [18_emergent_geometry.md § 2.1](source/2_geometric_gas/18_emergent_geometry)

### Geometric Ergodicity of the Adaptive Gas
- **Type:** Theorem
- **Label:** `thm-main-convergence`
- **Tags:** geometric-ergodicity, QSD, convergence, anisotropic, N-uniform, main-result
- **Source:** [18_emergent_geometry.md § 2.2](source/2_geometric_gas/18_emergent_geometry)

### Itô Correction Term Bound
- **Type:** Lemma
- **Label:** `lem-ito-correction-bound`
- **Tags:** Itô-correction, Stratonovich, state-dependent, N-uniform, technical
- **Source:** [18_emergent_geometry.md § 3.1](source/2_geometric_gas/18_emergent_geometry)

### Velocity Variance Contraction (Anisotropic)
- **Type:** Theorem
- **Label:** `thm-velocity-variance-anisotropic`
- **Tags:** velocity-variance, friction, contraction, anisotropic, N-uniform
- **Source:** [18_emergent_geometry.md § 3.2](source/2_geometric_gas/18_emergent_geometry)

### Hypocoercive Norm
- **Type:** Definition
- **Label:** `def-d-hypocoercive-norm`
- **Tags:** hypocoercivity, coupling, phase-space, norm, weighted
- **Source:** [18_emergent_geometry.md § 3.2.1](source/2_geometric_gas/18_emergent_geometry)

### Why Coupling is Essential
- **Type:** Remark
- **Label:** `rem-coupling-essential`
- **Tags:** hypocoercivity, coupling, position-velocity, degenerate-diffusion
- **Source:** [18_emergent_geometry.md § 3.2.1](source/2_geometric_gas/18_emergent_geometry)

### Location Error Contraction (Anisotropic)
- **Type:** Theorem
- **Label:** `thm-location-error-anisotropic`
- **Tags:** location-error, hypocoercivity, anisotropic, contraction, drift-matrix
- **Source:** [18_emergent_geometry.md § 3.2.3](source/2_geometric_gas/18_emergent_geometry)

### Structural Error Contraction (Anisotropic)
- **Type:** Theorem
- **Label:** `thm-structural-error-anisotropic`
- **Tags:** structural-error, Wasserstein, synchronous-coupling, anisotropic, contraction
- **Source:** [18_emergent_geometry.md § 3.2.4](source/2_geometric_gas/18_emergent_geometry)

### Hypocoercive Contraction for Adaptive Gas
- **Type:** Theorem
- **Label:** `thm-hypocoercive-main`
- **Tags:** hypocoercivity, anisotropic, main-result, Wasserstein, N-uniform
- **Source:** [18_emergent_geometry.md § 3.2.5](source/2_geometric_gas/18_emergent_geometry)

### Position Variance Expansion
- **Type:** Theorem
- **Label:** `thm-position-variance-expansion`
- **Tags:** position-variance, expansion, kinetic-operator, bounded
- **Source:** [18_emergent_geometry.md § 3.3](source/2_geometric_gas/18_emergent_geometry)

### Boundary Potential Contraction
- **Type:** Theorem
- **Label:** `thm-boundary-contraction`
- **Tags:** boundary, confining-potential, contraction, coercivity
- **Source:** [18_emergent_geometry.md § 3.4](source/2_geometric_gas/18_emergent_geometry)

### Foster-Lyapunov Condition for Adaptive Gas
- **Type:** Theorem
- **Label:** `thm-foster-lyapunov-adaptive`
- **Tags:** Foster-Lyapunov, operator-composition, synergy, convergence, N-uniform
- **Source:** [18_emergent_geometry.md § 4.2](source/2_geometric_gas/18_emergent_geometry)

### Total Convergence Rate with Full Parameter Dependence
- **Type:** Theorem
- **Label:** `thm-explicit-total-rate`
- **Tags:** convergence-rate, explicit, parameters, N-uniform, algorithmic-tunability
- **Source:** [18_emergent_geometry.md § 5.2](source/2_geometric_gas/18_emergent_geometry)

### Total Expansion Constant with Full Parameter Dependence
- **Type:** Theorem
- **Label:** `thm-explicit-total-constant`
- **Tags:** expansion-constant, explicit, parameters, N-uniform, Foster-Lyapunov
- **Source:** [18_emergent_geometry.md § 5.3](source/2_geometric_gas/18_emergent_geometry)

### Explicit Convergence Time
- **Type:** Corollary
- **Label:** `cor-explicit-convergence-time`
- **Tags:** mixing-time, convergence-time, explicit, iterations, complexity
- **Source:** [18_emergent_geometry.md § 5.5](source/2_geometric_gas/18_emergent_geometry)

### Three Bottleneck Regimes
- **Type:** Observation
- **Label:** `obs-three-regimes`
- **Tags:** regimes, bottleneck, cloning-limited, hypocoercivity-limited, boundary-limited
- **Source:** [18_emergent_geometry.md § 5.6](source/2_geometric_gas/18_emergent_geometry)

### Regularization Trade-Off
- **Type:** Observation
- **Label:** `obs-regularization-tradeoff`
- **Tags:** regularization, trade-off, ellipticity, adaptation, robustness
- **Source:** [18_emergent_geometry.md § 5.7](source/2_geometric_gas/18_emergent_geometry)

### The Emergent Metric
- **Type:** Observation
- **Label:** `obs-emergent-metric`
- **Tags:** Riemannian-metric, emergent-geometry, geodesic, diffusion, inverse-metric
- **Source:** [18_emergent_geometry.md § 6.1](source/2_geometric_gas/18_emergent_geometry)

### Convergence Rate Depends on Metric Ellipticity
- **Type:** Proposition
- **Label:** `prop-rate-metric-ellipticity`
- **Tags:** convergence-rate, ellipticity, metric, geometric-interpretation, conditioning
- **Source:** [18_emergent_geometry.md § 6.1](source/2_geometric_gas/18_emergent_geometry)

### Fitness Potential Construction (Algorithmic Specification)
- **Type:** Definition
- **Label:** `def-fitness-algorithmic`
- **Tags:** fitness-potential, algorithmic, localization, Z-score, rescale, pipeline
- **Source:** [18_emergent_geometry.md § 9.2](source/2_geometric_gas/18_emergent_geometry)

### Explicit Hessian Formula
- **Type:** Theorem
- **Label:** `thm-explicit-hessian`
- **Tags:** Hessian, chain-rule, curvature, explicit, fitness-landscape, N-uniform
- **Source:** [18_emergent_geometry.md § 9.3](source/2_geometric_gas/18_emergent_geometry)

### Emergent Riemannian Metric (Explicit Construction)
- **Type:** Definition
- **Label:** `def-metric-explicit`
- **Tags:** Riemannian-metric, emergent-geometry, regularization, diffusion-tensor, explicit
- **Source:** [18_emergent_geometry.md § 9.4](source/2_geometric_gas/18_emergent_geometry)

### Uniform Ellipticity from Regularization
- **Type:** Theorem
- **Label:** `thm-uniform-ellipticity-explicit`
- **Tags:** ellipticity, regularization, spectral-bounds, explicit, algorithmic-control
- **Source:** [18_emergent_geometry.md § 9.4](source/2_geometric_gas/18_emergent_geometry)

### Emergent Riemannian Manifold
- **Type:** Definition
- **Label:** `def-emergent-manifold`
- **Tags:** Riemannian-manifold, metric-tensor, geodesics, Christoffel-symbols, volume-element
- **Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)

### Geodesics Favor High-Fitness Regions
- **Type:** Proposition
- **Label:** `prop-geodesics-fitness`
- **Tags:** geodesics, fitness, natural-gradient, Riemannian-distance, curvature
- **Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)

### Algorithmic Tunability of the Emergent Geometry
- **Type:** Theorem
- **Label:** `thm-algorithmic-tunability`
- **Tags:** algorithmic-tunability, parameters, geometric-control, design, localization
- **Source:** [18_emergent_geometry.md § 9.6.2](source/2_geometric_gas/18_emergent_geometry)

### Companion Flux Balance at QSD
- **Type:** Lemma
- **Label:** `lem-companion-flux-balance`
- **Tags:** flux-balance, QSD, companion-selection, Riemannian-volume, stationarity
- **Source:** [18_emergent_geometry.md § 10](source/2_geometric_gas/18_emergent_geometry)

### QSD Spatial Marginal is Riemannian Volume Measure
- **Type:** Theorem
- **Label:** `thm-qsd-spatial-riemannian-volume`
- **Tags:** QSD, spatial-marginal, Riemannian-volume, Stratonovich, stationary-distribution
- **Source:** [18_emergent_geometry.md § A.1](source/2_geometric_gas/18_emergent_geometry)

### Fast Velocity Thermalization Justifies Annealed Approximation
- **Type:** Lemma
- **Label:** `lem-velocity-marginalization`
- **Tags:** velocity-thermalization, timescale-separation, annealed-approximation, Maxwell-Boltzmann
- **Source:** [18_emergent_geometry.md § A.2](source/2_geometric_gas/18_emergent_geometry)

### Continuum Limit via Saddle-Point Approximation
- **Type:** Lemma
- **Label:** `lem-companion-bias-riemannian`
- **Tags:** continuum-limit, saddle-point, companion-selection, Riemannian-Gibbs, mean-field
- **Source:** [18_emergent_geometry.md § A.3](source/2_geometric_gas/18_emergent_geometry)
