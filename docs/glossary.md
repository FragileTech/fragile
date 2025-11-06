# Mathematical Glossary
Comprehensive index of all mathematical entries from the Fragile Gas framework.

**Version:** 2.2
**Last Updated:** 2025-10-24
**Total Entries:** 741 (Chapter 1: 541, Chapter 2: 200)
**Format:** Each entry includes Type, Label, Tags, Source, and concise Description (<15 words)
**Description Style:** TLDR context WITHOUT repeating title (avoids token waste)

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
- [12_quantitative_error_bounds.md](#12_quantitative_error_bounds) (18 entries)

### Chapter 2: Geometric Gas
- [11_geometric_gas.md](#11_geometric_gas) (60 entries)
- [12_symmetries_geometric_gas.md](#12_symmetries_geometric_gas) (17 entries)
- [13_geometric_gas_c3_regularity.md](#13_geometric_gas_c3_regularity) (22 entries)
- [14_geometric_gas_cinf_regularity_full.md](#14_geometric_gas_cinf_regularity_full) (19 entries)
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
- **Description:** Tuple w=(x,v,s) with position, velocity, and survival status

### Swarm and Swarm State Space
- **Type:** Definition
- **Label:** `def-swarm-and-state-space`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Product space Σ_N containing N-tuples of agents

### Alive and Dead Sets
- **Type:** Definition
- **Label:** `def-alive-dead-sets`
- **Tags:** viability
- **Source:** [01_fragile_gas_framework.md § 1.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Partition A and D based on survival status bit

### Valid State Space
- **Type:** Definition
- **Label:** `def-valid-state-space`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Polish metric space X_valid with Borel reference measure

### Ambient Euclidean Structure and Reference Measures
- **Type:** Assumption
- **Label:** `def-ambient-euclidean`
- **Tags:** assumption
- **Source:** [01_fragile_gas_framework.md § A. Foundational & Environmental Parameters](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Assumes R^d embedding with heat kernel and uniform ball

### Reference Noise and Kernel Families
- **Type:** Definition
- **Label:** `def-reference-measures`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § A. Foundational & Environmental Parameters](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Heat kernels P_σ and uniform balls Q_δ for stochastic operations

### N-Particle Displacement Pseudometric ($d_{\text{Disp},\mathcal{Y}}$)
- **Type:** Definition
- **Label:** `def-n-particle-displacement-metric`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 1.6](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Measures positional and status differences across population

### Metric quotient of $(\Sigma_N, d_{\text{Disp},\mathcal{Y}})$
- **Type:** Definition
- **Label:** `def-metric-quotient`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 1.6.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Kolmogorov construction identifying permutation-equivalent configurations

### Borel image of the projected swarm space
- **Type:** Lemma
- **Label:** `unlabeled`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 1.6.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Projection Φ maps to Borel measurable subset

### If $\widehat{\Phi}(\Sigma_N)$ is not closed, replacing it by its closure in $(\mathcal Y\times\{0,1\})^N$ yields a closed (hence complete) subspace. All probability measures considered are supported on $\widehat{\Phi}(\Sigma_N)$, and optimal couplings for costs continuous in $D$ concentrate on the product of supports, so no generality is lost by completing.
- **Source:** [01_fragile_gas_framework.md § 1.6.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** If  is not closed, replacing it by its closure in  yields a closed (hence com...

### Polishness of the quotient state space and $W_2$
- **Type:** Lemma
- **Label:** `lem-polishness-and-w2`
- **Tags:** lemma, wasserstein
- **Source:** [01_fragile_gas_framework.md § 1.7.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Enables Wasserstein-2 metric on complete separable quotient

### Components of Swarm Displacement
- **Type:** Definition
- **Label:** `def-displacement-components`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 1.7.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Splits into position and status change contributions

### Conditional product structure within a step
- **Type:** Axiom
- **Label:** `def-assumption-instep-independence`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § Assumption A (In‑Step Independence)](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Operators act independently conditioned on current configuration

### Axiom of Guaranteed Revival
- **Type:** Axiom
- **Label:** `def-axiom-guaranteed-revival`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Dead agents clone with certainty preventing extinction

### Almost‑sure revival under the global constraint
- **Type:** Theorem
- **Label:** `thm-revival-guarantee`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Proves probability of resurrection equals one

### Axiom of Boundary Regularity
- **Type:** Axiom
- **Label:** `def-axiom-boundary-regularity`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Death probability Hölder continuous in configuration

### Axiom of Boundary Smoothness
- **Type:** Axiom
- **Label:** `def-axiom-boundary-smoothness`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.1.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Domain has finite perimeter enabling integration

### Axiom of Environmental Richness
- **Type:** Axiom
- **Label:** `def-axiom-environmental-richness`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Requires sufficient variation in rewards for discrimination

### Axiom of Reward Regularity
- **Type:** Axiom
- **Label:** `def-axiom-reward-regularity`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** R(x) is L_R-Lipschitz continuous

### Projection compatibility
- **Type:** Axiom
- **Label:** `def-axiom-projection-compatibility`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Map φ compatible with metric structure

### Axiom of Bounded Algorithmic Diameter
- **Type:** Axiom
- **Label:** `def-axiom-bounded-algorithmic-diameter`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Projected space Y has finite diameter D_Y

### Range‑Respecting Mean
- **Type:** Axiom
- **Label:** `def-axiom-range-respecting-mean`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Aggregator g_A maps to interval [inf V, sup V]

### Valid Noise Measure
- **Type:** Definition
- **Label:** `def-valid-noise-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Feller kernel with bounded second moment

### Axiom of Sufficient Amplification
- **Type:** Axiom
- **Label:** `def-axiom-sufficient-amplification`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Rescaling amplifies fitness differences adequately

### Axiom of Non-Degenerate Noise
- **Type:** Axiom
- **Label:** `def-axiom-non-degenerate-noise`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Perturbations have full-dimensional support

### Components of Mean-Square Standardization Error
- **Type:** Definition
- **Label:** `def-components-mean-square-standardization-error`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 2.3.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Splits into value and structural contributions

### Asymptotic Behavior of the Mean-Square Standardization Error
- **Type:** Theorem
- **Label:** `thm-mean-square-standardization-error`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.3.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Growth rates O(1/N) for both components

### Axiom of Bounded Relative Collapse
- **Type:** Axiom
- **Label:** `def-axiom-bounded-relative-collapse`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Local variance doesn't collapse faster than global

### Axiom of Bounded Deviation from Aggregated Variance
- **Type:** Axiom
- **Label:** `def-axiom-bounded-deviation-variance`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Individual variances close to population aggregate

### Axiom of Bounded Variance Production
- **Type:** Axiom
- **Label:** `def-axiom-bounded-variance-production`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.3.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Distance operator generates limited spread

### Axiom of Geometric Consistency
- **Type:** Axiom
- **Label:** `def-axiom-geometric-consistency`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.4.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Projection φ respects viability constraints

### Theorem of Forced Activity
- **Type:** Theorem
- **Label:** `thm-forced-activity`
- **Tags:** theorem
- **Source:** [01_fragile_gas_framework.md § 2.4.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Guarantees non-stagnation via fitness signal

### Axiom of Position‑Only Status Margin
- **Type:** Axiom
- **Label:** `def-axiom-margin-stability`
- **Tags:** axiom
- **Source:** [01_fragile_gas_framework.md § 2.4.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Survival determined by position with margin ε_margin

### This axiom expresses a deterministic stability of the status update in terms of the positional component alone. It is strictly stronger than the trivial consequence of the identity
- **Source:** [01_fragile_gas_framework.md § 2.4.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** This axiom expresses a deterministic stability of the status update in terms ...

### Reward Measurement
- **Type:** Definition
- **Label:** `def-reward-measurement`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 3.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Integrates R(x) against positional measure

### Perturbation Measure
- **Type:** Definition
- **Label:** `def-perturbation-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 4.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Kernel P_σ for random walk exploration

### Cloning Measure
- **Type:** Definition
- **Label:** `def-cloning-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 4.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Kernel Q_δ for displacement during replication

### Validation of the Heat Kernel
- **Type:** Lemma
- **Label:** `lem-validation-heat-kernel`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Gaussian P_σ satisfies Feller and moment conditions

### Validation of the Uniform Ball Measure
- **Type:** Lemma
- **Label:** `lem-validation-uniform-ball`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Uniform ball Q_δ satisfies all axioms

### Uniform‑ball death probability is Lipschitz under finite perimeter
- **Type:** Lemma
- **Label:** `lem-boundary-uniform-ball`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Boundary exit has Lipschitz constant proportional to perimeter

### Projection choice
- **Type:** Remark
- **Label:** `unlabeled`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 4.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Map φ can be identity or custom embedding

### Heat‑kernel death probability is Lipschitz with constant $\lesssim 1/\sigma$
- **Type:** Lemma
- **Label:** `lem-boundary-heat-kernel`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 4.2.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Gaussian exit probability has constant ∼ 1/σ

### Algorithmic Space
- **Type:** Definition
- **Label:** `def-algorithmic-space-generic`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 5.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Pair (Y, d_Y) where agents project and compare

### Distance Between Positional Measures
- **Type:** Definition
- **Label:** `def-distance-positional-measures`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 5.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** 1-Wasserstein on projected empirical distributions

### Algorithmic Distance
- **Type:** Definition
- **Label:** `def-alg-distance`
- **Tags:** metric
- **Source:** [01_fragile_gas_framework.md § 5.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Metric d_alg(y_1, y_2) on projected points

### Swarm Aggregation Operator
- **Type:** Definition
- **Label:** `def-swarm-aggregation-operator-axiomatic`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Maps N fitnesses to probability on R

### Empirical moments are Lipschitz in L2
- **Type:** Lemma
- **Label:** `lem-empirical-moments-lipschitz`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 6.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Mean and variance continuous in W_2 metric

### Axiomatic Properties of the Empirical Measure Aggregator
- **Type:** Lemma
- **Label:** `lem-empirical-aggregator-properties`
- **Tags:** lemma
- **Source:** [01_fragile_gas_framework.md § 6.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Sample mean satisfies range-respecting property

### Smoothed Gaussian Measure
- **Type:** Definition
- **Label:** `def-smoothed-gaussian-measure`
- **Tags:** general
- **Source:** [01_fragile_gas_framework.md § 6.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Smoothed Gaussian Measure

### Algorithmic space with cemetery point
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Algorithmic space with cemetery point

### Maximal cemetery distance (design choice)
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Maximal cemetery distance (design choice)

### Cemetery State Measure
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Cemetery State Measure

### Distance to the Cemetery State
- **Source:** [01_fragile_gas_framework.md § 6.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Distance to the Cemetery State

### Companion Selection Measure
- **Source:** [01_fragile_gas_framework.md § 7.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Companion Selection Measure

### Bound on the Error from Companion Set Change
- **Source:** [01_fragile_gas_framework.md § 7.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Error from Companion Set Change

### Bound on the Error from Normalization Change
- **Source:** [01_fragile_gas_framework.md § 7.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Error from Normalization Change

### Total Error Bound in Terms of Status Changes
- **Source:** [01_fragile_gas_framework.md § 7.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Total Error Bound in Terms of Status Changes

### Axiom of a Well-Behaved Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.1.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Axiom of a Well-Behaved Rescale Function

### Smooth Piecewise Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Smooth Piecewise Rescale Function

### Existence and Uniqueness of the Smooth Rescale Patch
- **Source:** [01_fragile_gas_framework.md § 8.2.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Existence and Uniqueness of the Smooth Rescale Patch

### Explicit Coefficients of the Smooth Rescale Patch
- **Source:** [01_fragile_gas_framework.md § 8.2.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Explicit Coefficients of the Smooth Rescale Patch

### Explicit Form of the Polynomial Patch Derivative
- **Source:** [01_fragile_gas_framework.md § 8.2.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Explicit Form of the Polynomial Patch Derivative

### Monotonicity of the Polynomial Patch
- **Source:** [01_fragile_gas_framework.md § 8.2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Monotonicity of the Polynomial Patch

### This construction is the standard monotone cubic Hermite approach (PCHIP/PCHIM). The global derivative bound $L_P\approx 1.0054$ from §8.2.2.5 provides an explicit Lipschitz constant for the rescale segment.
- **Source:** [01_fragile_gas_framework.md § 8.2.2.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** This construction is the standard monotone cubic Hermite approach (PCHIP/PCHI...

### Bounds on the Polynomial Patch Derivative
- **Source:** [01_fragile_gas_framework.md § 8.2.2.5](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounds on the Polynomial Patch Derivative

### Monotonicity of the Smooth Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.2.2.6](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Monotonicity of the Smooth Rescale Function

### Global Lipschitz Continuity of the Smooth Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.2.2.7](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Global Lipschitz Continuity of the Smooth Rescale Function

### Lipschitz constant of the patched standardization
- **Source:** [01_fragile_gas_framework.md § 8.2.2.8](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Lipschitz constant of the patched standardization

### Derivative bound for \sigma\'_{\text{reg}}
- **Source:** [01_fragile_gas_framework.md § 8.2.2.9](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Derivative bound for \sigma\'\textreg

### Lipschitz bound for the variance functional
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Lipschitz bound for the variance functional

### Chain‑rule bound for \sigma\'_{\text{reg}}\circ \mathrm{Var}
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Chain‑rule bound for \sigma\'\textreg\circ \mathrmVar

### Closed‑form bound for $L_{g_A\circ z}$ (empirical aggregator)
- **Source:** [01_fragile_gas_framework.md § 8.2.2.10](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Closed‑form bound for  (empirical aggregator)

### Canonical Logistic Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.3.1.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Canonical Logistic Rescale Function

### The Canonical Logistic Function is a Valid Rescale Function
- **Source:** [01_fragile_gas_framework.md § 8.3.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The Canonical Logistic Function is a Valid Rescale Function

### Raw Value Operator
- **Source:** [01_fragile_gas_framework.md § 9.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Raw Value Operator

### Axiom of Mean-Square Continuity for Raw Values
- **Source:** [01_fragile_gas_framework.md § 9.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Axiom of Mean-Square Continuity for Raw Values

### Axiom of Bounded Measurement Variance
- **Source:** [01_fragile_gas_framework.md § 9.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Axiom of Bounded Measurement Variance

### Distance-to-Companion Measurement
- **Source:** [01_fragile_gas_framework.md § 10.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Distance-to-Companion Measurement

### Bound on Single-Walker Error from Positional Change
- **Source:** [01_fragile_gas_framework.md § 10.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on Single-Walker Error from Positional Change

### Bound on Single-Walker Error from Structural Change
- **Source:** [01_fragile_gas_framework.md § 10.2.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on Single-Walker Error from Structural Change

### Bound on Single-Walker Error from Own Status Change
- **Source:** [01_fragile_gas_framework.md § 10.2.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on Single-Walker Error from Own Status Change

### Bound on the Total Squared Error for Unstable Walkers
- **Source:** [01_fragile_gas_framework.md § 10.2.5](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Total Squared Error for Unstable Walkers

### Bound on the Total Squared Error for Stable Walkers
- **Source:** [01_fragile_gas_framework.md § 10.2.6.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Total Squared Error for Stable Walkers

### Bound on the Total Squared Error for Unstable Walkers
- **Source:** [01_fragile_gas_framework.md § 10.2.5](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Total Squared Error for Unstable Walkers

### Bound on the Total Squared Error for Stable Walkers
- **Source:** [01_fragile_gas_framework.md § 10.2.6](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Total Squared Error for Stable Walkers

### Decomposition of Stable Walker Error
- **Source:** [01_fragile_gas_framework.md § 10.2.6.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Decomposition of Stable Walker Error

### Bounding the Positional Error Component
- **Source:** [01_fragile_gas_framework.md § 10.2.6.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Positional Error Component

### Bounding the Structural Error Component for Stable Walkers
- **Source:** [01_fragile_gas_framework.md § 10.2.6.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Structural Error Component for Stable Walkers

### Bound on the Expected Raw Distance Vector Change
- **Source:** [01_fragile_gas_framework.md § 10.2.7](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bound on the Expected Raw Distance Vector Change

### Deterministic Behavior of the Expected Raw Distance Vector at $k=1$
- **Source:** [01_fragile_gas_framework.md § 10.2.8](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Deterministic Behavior of the Expected Raw Distance Vector at

### The Distance Operator Satisfies the Bounded Variance Axiom
- **Source:** [01_fragile_gas_framework.md § 10.3.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The Distance Operator Satisfies the Bounded Variance Axiom

### Mean-Square Continuity of the Distance Operator
- **Source:** [01_fragile_gas_framework.md § 10.3.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Mean-Square Continuity of the Distance Operator

### N-Dimensional Standardization Operator
- **Source:** [01_fragile_gas_framework.md § 11.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** N-Dimensional Standardization Operator

### Statistical Properties Measurement
- **Source:** [01_fragile_gas_framework.md § 11.1.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Statistical Properties Measurement

### Value Continuity of Statistical Properties
- **Source:** [01_fragile_gas_framework.md § 11.1.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Value Continuity of Statistical Properties

### Structural Continuity of Statistical Properties
- **Source:** [01_fragile_gas_framework.md § 11.1.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Structural Continuity of Statistical Properties

### General Bound on the Norm of the Standardized Vector
- **Source:** [01_fragile_gas_framework.md § 11.1.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** General Bound on the Norm of the Standardized Vector

### Asymptotic Behavior of the Structural Continuity for the Regularized Standard Deviation
- **Source:** [01_fragile_gas_framework.md § 11.1.5](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Asymptotic Behavior of the Structural Continuity for the Regularized Standard...

### Bounding the Expected Squared Value Error
- **Source:** [01_fragile_gas_framework.md § 11.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Expected Squared Value Error

### Bounding the Expected Squared Value Error
- **Source:** [01_fragile_gas_framework.md § 11.2.2.6.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Expected Squared Value Error

### Bounding the Expected Squared Structural Error
- **Source:** [01_fragile_gas_framework.md § 11.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Expected Squared Structural Error

### Bounding the Expected Squared Structural Error
- **Source:** [01_fragile_gas_framework.md § 11.2.3.5.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Expected Squared Structural Error

### Decomposition of the Total Standardization Error
- **Source:** [01_fragile_gas_framework.md § 11.3.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Decomposition of the Total Standardization Error

### Algebraic Decomposition of the Value Error
- **Source:** [01_fragile_gas_framework.md § 11.3.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Algebraic Decomposition of the Value Error

### Bounding the Squared Value Error
- **Source:** [01_fragile_gas_framework.md § 11.3.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Squared Value Error

### Value Error Coefficients
- **Source:** [01_fragile_gas_framework.md § 11.3.4](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Value Error Coefficients

### Bounding the Squared Structural Error
- **Source:** [01_fragile_gas_framework.md § 11.3.5](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Squared Structural Error

### Structural Error Coefficients
- **Source:** [01_fragile_gas_framework.md § 11.3.6](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Structural Error Coefficients

### Global Continuity of the Patched Standardization Operator
- **Source:** [01_fragile_gas_framework.md § 11.3.7](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Global Continuity of the Patched Standardization Operator

### Rescaled Potential Operator for the Alive Set
- **Source:** [01_fragile_gas_framework.md § 12.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Rescaled Potential Operator for the Alive Set

### Swarm Potential Assembly Operator
- **Source:** [01_fragile_gas_framework.md § 12.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Swarm Potential Assembly Operator

### Assume the **Axiom of Margin-Based Status Stability** ([](#def-axiom-margin-stability)). Then for all inputs
- **Source:** [01_fragile_gas_framework.md § 12.3.3](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Assume the **Axiom of Margin-Based Status Stability** ([](#def-axiom-margin-s...

### Perturbation Operator
- **Source:** [01_fragile_gas_framework.md § 13.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Perturbation Operator

### Axiom of Bounded Second Moment of Perturbation
- **Source:** [01_fragile_gas_framework.md § 13.2.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Axiom of Bounded Second Moment of Perturbation

### Bounding the Output Positional Displacement
- **Source:** [01_fragile_gas_framework.md § 13.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounding the Output Positional Displacement

### Bounded differences for $f_{\text{avg}}$
- **Source:** [01_fragile_gas_framework.md § 13.2.3.0.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Bounded differences for

### McDiarmid's Inequality (Bounded Differences Inequality) (Boucheron–Lugosi–Massart)
- **Source:** [01_fragile_gas_framework.md § 13.2.3.1.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** McDiarmid's Inequality (Bounded Differences Inequality) (Boucheron–Lugosi–Mas...

### Probabilistic Bound on Total Perturbation-Induced Displacement
- **Source:** [01_fragile_gas_framework.md § 13.2.3.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Probabilistic Bound on Total Perturbation-Induced Displacement

### Perturbation Fluctuation Bounds
- **Source:** [01_fragile_gas_framework.md § 13.2.4.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Perturbation Fluctuation Bounds

### Probabilistic Continuity of the Perturbation Operator
- **Source:** [01_fragile_gas_framework.md § 13.2.5.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Probabilistic Continuity of the Perturbation Operator

### Status Update Operator
- **Source:** [01_fragile_gas_framework.md § 14.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Status Update Operator

### Probabilistic Continuity of the Post-Perturbation Status Update
- **Source:** [01_fragile_gas_framework.md § 14.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Probabilistic Continuity of the Post-Perturbation Status Update

### Cloning Score Function
- **Source:** [01_fragile_gas_framework.md § 15.1.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Cloning Score Function

### Stochastic Threshold Cloning
- **Source:** [01_fragile_gas_framework.md § 15.1.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Stochastic Threshold Cloning

### Total Expected Cloning Action
- **Source:** [01_fragile_gas_framework.md § 15.2.1.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Total Expected Cloning Action

### The Conditional Cloning Probability Function
- **Source:** [01_fragile_gas_framework.md § 15.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The Conditional Cloning Probability Function

### Lipschitz Continuity of the Conditional Cloning Probability Function (case split)
- **Source:** [01_fragile_gas_framework.md § 15.2.2.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Lipschitz Continuity of the Conditional Cloning Probability Function (case sp...

### Conditional Expected Cloning Action
- **Source:** [01_fragile_gas_framework.md § 15.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Conditional Expected Cloning Action

### Continuity of the Conditional Expected Cloning Action
- **Source:** [01_fragile_gas_framework.md § 15.2.3.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Continuity of the Conditional Expected Cloning Action

### Continuity of the Total Expected Cloning Action
- **Source:** [01_fragile_gas_framework.md § 15.2.4.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Continuity of the Total Expected Cloning Action

### Theorem of Guaranteed Revival from a Single Survivor
- **Source:** [01_fragile_gas_framework.md § 16.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Theorem of Guaranteed Revival from a Single Survivor

### Swarm Update Procedure
- **Source:** [01_fragile_gas_framework.md § 17.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Swarm Update Procedure

### Local vs global: for $V\in[0,1]$ all sub-linear powers are $\le 1$ and can be absorbed in a constant; for $V\ge 1$ every sub-linear term is bounded above by the term with exponent $p_{\max}$. This is the only global (uniform in $V\ge 0$) way to replace a sum of distinct powers by a single power, and it justifies using $\alpha_H^{\mathrm{global}}=\max(\tfrac12,\alpha_B)$ when aggregating sub-linear exponents.
- **Source:** [01_fragile_gas_framework.md § 17.2.4.3.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Local vs global: for  all sub-linear powers are  and can be absorbed in a con...

### Wasserstein-2 on the output space (quotient)
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Wasserstein-2 on the output space (quotient)

### W2 continuity bound without offset (for $k\ge 2$)
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** W2 continuity bound without offset (for )

### The offset $K_{\Psi}$ appearing in the expectation-based bound corresponds to allowing arbitrary (e.g., independent) couplings of the output randomness. When the comparison is made in $W_2$—or, operationally, under synchronous coupling—the artificial offset vanishes at zero input distance, yielding a cleaner continuity statement. The composite constants $C_{\Psi,L}$ and $C_{\Psi,H}$ are exactly those defined in [](#def-composite-continuity-coeffs-recorrected) and inherit boundedness/continuity from [](#subsec-coefficient-regularity).
- **Source:** [01_fragile_gas_framework.md § 17.2.4.4.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The offset  appearing in the expectation-based bound corresponds to allowing ...

### The Swarm Update defines a Markov kernel
- **Source:** [01_fragile_gas_framework.md § 17.2.4.5.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The Swarm Update defines a Markov kernel

### Feller-type (continuity-preserving) properties for $\Psi$ follow from the stagewise measurability and continuity assumptions stated in Section 2 for the operators and aggregators; on compact (or sublevel) sets these imply boundedness and continuity of the induced kernel maps.
- **Source:** [01_fragile_gas_framework.md § 17.2.4.5.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Feller-type (continuity-preserving) properties for  follow from the stagewise...

### Boundedness and continuity of composite coefficients
- **Source:** [01_fragile_gas_framework.md § 17.2.4.6.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Boundedness and continuity of composite coefficients

### In particular, on such sublevel sets the $W_2$ continuity bound and the deterministic standardization bounds promote to genuine continuity statements for the composite operators since the constants do not blow up along admissible sequences.
- **Source:** [01_fragile_gas_framework.md § 17.2.4.6.](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** In particular, on such sublevel sets the  continuity bound and the determinis...

### Fragile Swarm Instantiation
- **Source:** [01_fragile_gas_framework.md § 18.1](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** Fragile Swarm Instantiation

### The Fragile Gas Algorithm
- **Source:** [01_fragile_gas_framework.md § 18.2](source/1_euclidean_gas/01_fragile_gas_framework)
- **Description:** The Fragile Gas Algorithm

---

### Source: 02_euclidean_gas.md {#02_euclidean_gas}


### Euclidean Gas Update
- **Source:** [02_euclidean_gas.md § **3.1 Euclidean Gas algorithm (canonical pipeline)](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Euclidean Gas Update

### Properties of smooth radial squashing maps
- **Source:** [02_euclidean_gas.md § 3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Properties of smooth radial squashing maps

### Lipschitz continuity of the projection $\varphi$
- **Source:** [02_euclidean_gas.md § 3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Lipschitz continuity of the projection

### Lipschitz property of the kinetic flow
- **Source:** [02_euclidean_gas.md § 4.1](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Lipschitz property of the kinetic flow

### Hölder continuity of the death probability
- **Source:** [02_euclidean_gas.md § 4.1](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Hölder continuity of the death probability

### Reward regularity in the Sasaki metric
- **Source:** [02_euclidean_gas.md § 4.2](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Reward regularity in the Sasaki metric

### Environmental richness with a kinetic regularizer
- **Source:** [02_euclidean_gas.md § 4.2](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Environmental richness with a kinetic regularizer

### Perturbation second moment in the Sasaki metric
- **Source:** [02_euclidean_gas.md § 4.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Perturbation second moment in the Sasaki metric

### Geometric consistency under the capped kinetic kernel
- **Source:** [02_euclidean_gas.md § 4.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Geometric consistency under the capped kinetic kernel

### Single-walker positional error bound in the Sasaki metric
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Single-walker positional error bound in the Sasaki metric

### Single-walker structural error bound in the Sasaki metric
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Single-walker structural error bound in the Sasaki metric

### Mean-square error on stable walkers (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Mean-square error on stable walkers (Sasaki)

### Mean-square continuity of the distance measurement (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Mean-square continuity of the distance measurement (Sasaki)

### Value continuity of the empirical moments
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Value continuity of the empirical moments

### Structural continuity of the empirical moments
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Structural continuity of the empirical moments

### Lipschitz data for the Sasaki empirical aggregators
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Lipschitz data for the Sasaki empirical aggregators

### Standardization constants (Sasaki geometry)
- **Source:** [02_euclidean_gas.md § 2.3.3](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Standardization constants (Sasaki geometry)

### Value continuity of patched standardization (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.4.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Value continuity of patched standardization (Sasaki)

### Decomposition of the Value Error
- **Source:** [02_euclidean_gas.md § 2.3.4.1.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Decomposition of the Value Error

### Bound on the Squared Direct Shift Component
- **Source:** [02_euclidean_gas.md § 2.3.4.2.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Bound on the Squared Direct Shift Component

### Bound on the Squared Mean Shift Component
- **Source:** [02_euclidean_gas.md § 2.3.4.3.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Bound on the Squared Mean Shift Component

### Bounding the Squared Denominator Shift Component
- **Source:** [02_euclidean_gas.md § 2.3.4.4.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Bounding the Squared Denominator Shift Component

### Value Error Coefficients (Squared Form)
- **Source:** [02_euclidean_gas.md § 2.3.5.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Value Error Coefficients (Squared Form)

### Structural Continuity of Patched Standardization (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.6.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Structural Continuity of Patched Standardization (Sasaki)

### Decomposition of the Structural Error
- **Source:** [02_euclidean_gas.md § 2.3.6.1.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Decomposition of the Structural Error

### Bound on the Squared Direct Structural Error
- **Source:** [02_euclidean_gas.md § 2.3.6.2.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Bound on the Squared Direct Structural Error

### Bound on the Squared Indirect Structural Error
- **Source:** [02_euclidean_gas.md § 2.3.6.3.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Bound on the Squared Indirect Structural Error

### Composite Continuity of the Patched Standardization Operator (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.8.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Composite Continuity of the Patched Standardization Operator (Sasaki)

### Lipschitz continuity of patched standardization (Sasaki)
- **Source:** [02_euclidean_gas.md § 2.3.8.](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Lipschitz continuity of patched standardization (Sasaki)

### Axiom of Non-Deceptive Landscapes
- **Source:** [02_euclidean_gas.md § 2.6](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Axiom of Non-Deceptive Landscapes

### Feller continuity of $\Psi_{\mathcal F_{\mathrm{EG}}}$
- **Source:** [02_euclidean_gas.md § Appendix A. Proof of the Feller Property for the E](source/1_euclidean_gas/02_euclidean_gas)
- **Description:** Feller continuity of

---

### Source: 03_cloning.md {#03_cloning}


### Single-Walker and Swarm State Spaces
- **Source:** [03_cloning.md § 2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Single-Walker and Swarm State Spaces

### The Coupled State Space
- **Source:** [03_cloning.md § 2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** The Coupled State Space

### State Difference Vectors
- **Source:** [03_cloning.md § 2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** State Difference Vectors

### **(Axiom EG-0): Regularity of the Domain**
- **Source:** [03_cloning.md § 2.4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-0): Regularity of the Domain**

### Existence of a Global Smooth Barrier Function
- **Source:** [03_cloning.md § 2.4.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Existence of a Global Smooth Barrier Function

### Barycentres and Centered Vectors (Alive Walkers Only)
- **Source:** [03_cloning.md § 3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Barycentres and Centered Vectors (Alive Walkers Only)

### The Location Error Component ($V_{\text{loc}}$)
- **Source:** [03_cloning.md § 3.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** The Location Error Component ()

### The Structural Error Component ($V_{\text{struct}}$)
- **Source:** [03_cloning.md § 3.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** The Structural Error Component ()

### Decomposition of the Hypocoercive Wasserstein Distance
- **Source:** [03_cloning.md § 3.2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Decomposition of the Hypocoercive Wasserstein Distance

### Structural Positional Error and Internal Variance
- **Source:** [03_cloning.md § 3.2.4](source/1_euclidean_gas/03_cloning)
- **Description:** Structural Positional Error and Internal Variance

### The Full Synergistic Hypocoercive Lyapunov Function
- **Source:** [03_cloning.md § 3.3.](source/1_euclidean_gas/03_cloning)
- **Description:** The Full Synergistic Hypocoercive Lyapunov Function

### Variance Notation Conversion Formulas
- **Source:** [03_cloning.md § 3.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Variance Notation Conversion Formulas

### Necessity of the Augmented Lyapunov Structure
- **Source:** [03_cloning.md § 3.3.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Necessity of the Augmented Lyapunov Structure

### Analogy to Classical Hypocoercivity Theory
- **Source:** [03_cloning.md § 3.3.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Analogy to Classical Hypocoercivity Theory

### **(Axiom EG-1): Lipschitz Regularity of Environmental Fields**
- **Source:** [03_cloning.md § 4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-1): Lipschitz Regularity of Environmental Fields**

### **(Axiom EG-2): Existence of a Safe Harbor**
- **Source:** [03_cloning.md § 4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-2): Existence of a Safe Harbor**

### **(Axiom EG-3): Non-Deceptive Landscape**
- **Source:** [03_cloning.md § 4.2.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-3): Non-Deceptive Landscape**

### **(Axiom EG-4): Velocity Regularization via Reward**
- **Source:** [03_cloning.md § 4.3.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-4): Velocity Regularization via Reward**

### **(Axiom EG-5): Active Diversity Signal**
- **Source:** [03_cloning.md § 4.3.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom EG-5): Active Diversity Signal**

### Algorithmic Distance for Companion Selection
- **Source:** [03_cloning.md § 5.0.](source/1_euclidean_gas/03_cloning)
- **Description:** Algorithmic Distance for Companion Selection

### Spatially-Aware Pairing Operator (Idealized Model)
- **Source:** [03_cloning.md § 5.1.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Spatially-Aware Pairing Operator (Idealized Model)

### Sequential Stochastic Greedy Pairing Operator
- **Source:** [03_cloning.md § 5.1.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Sequential Stochastic Greedy Pairing Operator

### Sequential Stochastic Greedy Pairing Algorithm
- **Source:** [03_cloning.md § 5.1.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Sequential Stochastic Greedy Pairing Algorithm

### Geometric Partitioning of High-Variance Swarms
- **Source:** [03_cloning.md § 5.1.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Geometric Partitioning of High-Variance Swarms

### Greedy Pairing Guarantees Signal Separation
- **Source:** [03_cloning.md § 5.1.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Greedy Pairing Guarantees Signal Separation

### Raw Value Operators
- **Source:** [03_cloning.md § 5.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Raw Value Operators

### Swarm Aggregation Operator
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Swarm Aggregation Operator

### Patched Standard Deviation Function
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Patched Standard Deviation Function

### Properties of the Patching Function
- **Source:** [03_cloning.md § 5.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Properties of the Patching Function

### N-Dimensional Standardization Operator
- **Source:** [03_cloning.md § 5.4.](source/1_euclidean_gas/03_cloning)
- **Description:** N-Dimensional Standardization Operator

### Compact Support of Standardized Scores
- **Source:** [03_cloning.md § 5.4.](source/1_euclidean_gas/03_cloning)
- **Description:** Compact Support of Standardized Scores

### Canonical Logistic Rescale Function
- **Source:** [03_cloning.md § 5.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Canonical Logistic Rescale Function

### Verification of Axiomatic Properties
- **Source:** [03_cloning.md § 5.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Verification of Axiomatic Properties

### Fitness Potential Operator
- **Source:** [03_cloning.md § 5.6.](source/1_euclidean_gas/03_cloning)
- **Description:** Fitness Potential Operator

### Uniform Bounds of the Fitness Potential
- **Source:** [03_cloning.md § 5.6.](source/1_euclidean_gas/03_cloning)
- **Description:** Uniform Bounds of the Fitness Potential

### Companion Selection Operator for Cloning
- **Source:** [03_cloning.md § 5.7.1](source/1_euclidean_gas/03_cloning)
- **Description:** Companion Selection Operator for Cloning

### The Canonical Cloning Score
- **Source:** [03_cloning.md § 5.7.2](source/1_euclidean_gas/03_cloning)
- **Description:** The Canonical Cloning Score

### Total Cloning Probability
- **Source:** [03_cloning.md § 5.7.2](source/1_euclidean_gas/03_cloning)
- **Description:** Total Cloning Probability

### The Stochastic Cloning Decision
- **Source:** [03_cloning.md § 5.7.3](source/1_euclidean_gas/03_cloning)
- **Description:** The Stochastic Cloning Decision

### The Inelastic Collision State Update
- **Source:** [03_cloning.md § 5.7.4.](source/1_euclidean_gas/03_cloning)
- **Description:** The Inelastic Collision State Update

### Bounded Velocity Variance Expansion from Cloning
- **Source:** [03_cloning.md § 5.7.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Bounded Velocity Variance Expansion from Cloning

### Large $V_{\text{Var},x}$ Implies Large Single-Swarm Positional Variance
- **Source:** [03_cloning.md § 6.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Large  Implies Large Single-Swarm Positional Variance

### The Unified High-Error and Low-Error Sets
- **Source:** [03_cloning.md § 6.3](source/1_euclidean_gas/03_cloning)
- **Description:** The Unified High-Error and Low-Error Sets

### The Phase-Space Packing Lemma
- **Source:** [03_cloning.md § 6.4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** The Phase-Space Packing Lemma

### Positional Variance as a Lower Bound for Hypocoercive Variance
- **Source:** [03_cloning.md § 6.4.2](source/1_euclidean_gas/03_cloning)
- **Description:** Positional Variance as a Lower Bound for Hypocoercive Variance

### N-Uniform Lower Bound on the Outlier Fraction
- **Source:** [03_cloning.md § 6.4.2](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniform Lower Bound on the Outlier Fraction

### N-Uniform Lower Bound on the Outlier-Cluster Fraction
- **Source:** [03_cloning.md § 6.4.3.](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniform Lower Bound on the Outlier-Cluster Fraction

### A Large Intra-Swarm Positional Variance Guarantees a Non-Vanishing High-Error Fraction
- **Source:** [03_cloning.md § 6.4.4](source/1_euclidean_gas/03_cloning)
- **Description:** A Large Intra-Swarm Positional Variance Guarantees a Non-Vanishing High-Error...

### Geometric Separation of the Partition
- **Source:** [03_cloning.md § 6.5.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Geometric Separation of the Partition

### Geometric Structure Guarantees Measurement Variance
- **Source:** [03_cloning.md § 7.2.1](source/1_euclidean_gas/03_cloning)
- **Description:** Geometric Structure Guarantees Measurement Variance

### **(Satisfiability of the Signal-to-Noise Condition via Signal Gain)**
- **Source:** [03_cloning.md § 7.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Satisfiability of the Signal-to-Noise Condition via Signal Gain)**

### From Bounded Variance to a Guaranteed Gap
- **Source:** [03_cloning.md § 7.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** From Bounded Variance to a Guaranteed Gap

### Maximum Patched Standard Deviation
- **Source:** [03_cloning.md § 7.3.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Maximum Patched Standard Deviation

### Positive Derivative Bound for the Rescale Function
- **Source:** [03_cloning.md § 7.3.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Positive Derivative Bound for the Rescale Function

### From Raw Measurement Gap to Rescaled Value Gap
- **Source:** [03_cloning.md § 7.3.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** From Raw Measurement Gap to Rescaled Value Gap

### **(From Total Variance to Mean Separation)**
- **Source:** [03_cloning.md § 7.4.1](source/1_euclidean_gas/03_cloning)
- **Description:** **(From Total Variance to Mean Separation)**

### Derivation of the Stability Condition for Intelligent Adaptation
- **Source:** [03_cloning.md § 7.4.2](source/1_euclidean_gas/03_cloning)
- **Description:** Derivation of the Stability Condition for Intelligent Adaptation

### Lower Bound on Logarithmic Mean Gap
- **Source:** [03_cloning.md § 7.5.1.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Lower Bound on Logarithmic Mean Gap

### On the Tightness of the Bound
- **Source:** [03_cloning.md § 7.5.1.1.](source/1_euclidean_gas/03_cloning)
- **Description:** On the Tightness of the Bound

### Upper Bound on Logarithmic Mean Gap
- **Source:** [03_cloning.md § 7.5.1.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Upper Bound on Logarithmic Mean Gap

### Why the Bound is Tight at $V_{\min}$
- **Source:** [03_cloning.md § 7.5.1.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Why the Bound is Tight at

### **(Lower Bound on the Corrective Diversity Signal)**
- **Source:** [03_cloning.md § 7.5.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Lower Bound on the Corrective Diversity Signal)**

### **(Worst-Case Upper Bound on the Adversarial Reward Signal)**
- **Source:** [03_cloning.md § 7.5.2.2](source/1_euclidean_gas/03_cloning)
- **Description:** **(Worst-Case Upper Bound on the Adversarial Reward Signal)**

### **(Lipschitz Bound on the Raw Reward Mean Gap)**
- **Source:** [03_cloning.md § 7.5.2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Lipschitz Bound on the Raw Reward Mean Gap)**

### **(Axiom-Based Bound on the Logarithmic Reward Gap)**
- **Source:** [03_cloning.md § 7.5.2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** **(Axiom-Based Bound on the Logarithmic Reward Gap)**

### **(The Corrected Stability Condition for Intelligent Adaptation)**
- **Source:** [03_cloning.md § 7.5.2.4.](source/1_euclidean_gas/03_cloning)
- **Description:** **(The Corrected Stability Condition for Intelligent Adaptation)**

### The Unfit Set
- **Source:** [03_cloning.md § 7.6.1](source/1_euclidean_gas/03_cloning)
- **Description:** The Unfit Set

### N-Uniform Lower Bound on the Unfit Fraction
- **Source:** [03_cloning.md § 7.6.1](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniform Lower Bound on the Unfit Fraction

### N-Uniform Lower Bound on the Unfit-High-Error Overlap Fraction
- **Source:** [03_cloning.md § 7.6.2](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniform Lower Bound on the Unfit-High-Error Overlap Fraction

### The N-Uniform Quantitative Keystone Lemma
- **Source:** [03_cloning.md § 8.1](source/1_euclidean_gas/03_cloning)
- **Description:** The N-Uniform Quantitative Keystone Lemma

### The Critical Target Set
- **Source:** [03_cloning.md § 8.2](source/1_euclidean_gas/03_cloning)
- **Description:** The Critical Target Set

### Lower Bound on Mean Companion Fitness Gap
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Lower Bound on Mean Companion Fitness Gap

### N-Uniformity of the Bound
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniformity of the Bound

### Guaranteed Cloning Pressure on the Unfit Set
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Guaranteed Cloning Pressure on the Unfit Set

### Cloning Pressure on the Target Set
- **Source:** [03_cloning.md § 8.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Cloning Pressure on the Target Set

### **(Variance Concentration in the High-Error Set)**
- **Source:** [03_cloning.md § 8.4](source/1_euclidean_gas/03_cloning)
- **Description:** **(Variance Concentration in the High-Error Set)**

### Error Concentration in the Target Set
- **Source:** [03_cloning.md § 8.4](source/1_euclidean_gas/03_cloning)
- **Description:** Error Concentration in the Target Set

### N-Uniformity of Keystone Constants
- **Source:** [03_cloning.md § 8.6.3](source/1_euclidean_gas/03_cloning)
- **Description:** N-Uniformity of Keystone Constants

### The Cloning Operator $\Psi_{\text{clone}}$
- **Source:** [03_cloning.md § 9.2.](source/1_euclidean_gas/03_cloning)
- **Description:** The Cloning Operator

### The Measurement Operator
- **Source:** [03_cloning.md § 9.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** The Measurement Operator

### Stochastic Coupling for Drift Analysis
- **Source:** [03_cloning.md § 9.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Stochastic Coupling for Drift Analysis

### The Fitness Evaluation Operator
- **Source:** [03_cloning.md § 9.3.2.](source/1_euclidean_gas/03_cloning)
- **Description:** The Fitness Evaluation Operator

### The Cloning Decision Operator
- **Source:** [03_cloning.md § 9.3.3.](source/1_euclidean_gas/03_cloning)
- **Description:** The Cloning Decision Operator

### Total Cloning Probability for Dead Walkers
- **Source:** [03_cloning.md § 9.3.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Total Cloning Probability for Dead Walkers

### The State Update Operator
- **Source:** [03_cloning.md § 9.3.4.](source/1_euclidean_gas/03_cloning)
- **Description:** The State Update Operator

### Position Jitter vs. Velocity Collision Model
- **Source:** [03_cloning.md § 9.3.4.](source/1_euclidean_gas/03_cloning)
- **Description:** Position Jitter vs. Velocity Collision Model

### Compositional Structure of $\Psi_{\text{clone}}$
- **Source:** [03_cloning.md § 9.4.](source/1_euclidean_gas/03_cloning)
- **Description:** Compositional Structure of

### Key Operator Outputs
- **Source:** [03_cloning.md § 9.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Key Operator Outputs

### Expected Displacement Under Cloning
- **Source:** [03_cloning.md § 9.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Expected Displacement Under Cloning

### Coupled Cloning Expectation
- **Source:** [03_cloning.md § 10.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Coupled Cloning Expectation

### Synchronous Coupling Benefits
- **Source:** [03_cloning.md § 10.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Synchronous Coupling Benefits

### Positional Variance Contraction Under Cloning
- **Source:** [03_cloning.md § 10.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Positional Variance Contraction Under Cloning

### Variance Change Decomposition
- **Source:** [03_cloning.md § 10.3.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Variance Change Decomposition

### Keystone-Driven Contraction for Stably Alive Walkers
- **Source:** [03_cloning.md § 10.3.4.](source/1_euclidean_gas/03_cloning)
- **Description:** Keystone-Driven Contraction for Stably Alive Walkers

### Bounded Contribution from Dead Walker Revival
- **Source:** [03_cloning.md § 10.3.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Bounded Contribution from Dead Walker Revival

### Bounded Velocity Variance Expansion from Cloning
- **Source:** [03_cloning.md § 10.4.](source/1_euclidean_gas/03_cloning)
- **Description:** Bounded Velocity Variance Expansion from Cloning

### Synergistic Dissipation Enables Net Contraction
- **Source:** [03_cloning.md § 10.4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Synergistic Dissipation Enables Net Contraction

### Structural Error Contraction
- **Source:** [03_cloning.md § 10.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Structural Error Contraction

### Complete Variance Drift Characterization for Cloning
- **Source:** [03_cloning.md § 10.6.](source/1_euclidean_gas/03_cloning)
- **Description:** Complete Variance Drift Characterization for Cloning

### Constants and Parameter Dependencies
- **Source:** [03_cloning.md § 10.6.](source/1_euclidean_gas/03_cloning)
- **Description:** Constants and Parameter Dependencies

### Boundary Potential Component (Recall)
- **Source:** [03_cloning.md § 11.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Boundary Potential Component (Recall)

### Barrier Function as Geometric Penalty
- **Source:** [03_cloning.md § 11.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Barrier Function as Geometric Penalty

### Fitness Gradient from Boundary Proximity
- **Source:** [03_cloning.md § 11.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Fitness Gradient from Boundary Proximity

### The Boundary-Exposed Set
- **Source:** [03_cloning.md § 11.2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** The Boundary-Exposed Set

### Relationship to Total Boundary Potential
- **Source:** [03_cloning.md § 11.2.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Relationship to Total Boundary Potential

### Boundary Potential Contraction Under Cloning
- **Source:** [03_cloning.md § 11.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Boundary Potential Contraction Under Cloning

### Interpretation: Progressive Safety Enhancement
- **Source:** [03_cloning.md § 11.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Interpretation: Progressive Safety Enhancement

### Enhanced Cloning Probability Near Boundary
- **Source:** [03_cloning.md § 11.4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Enhanced Cloning Probability Near Boundary

### Expected Barrier Reduction for Cloned Walker
- **Source:** [03_cloning.md § 11.4.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Expected Barrier Reduction for Cloned Walker

### Bounded Boundary Exposure in Equilibrium
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Bounded Boundary Exposure in Equilibrium

### Exponentially Suppressed Extinction Probability
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Exponentially Suppressed Extinction Probability

### Safety Margin and Parameter Tuning
- **Source:** [03_cloning.md § 11.5.](source/1_euclidean_gas/03_cloning)
- **Description:** Safety Margin and Parameter Tuning

### Complete Boundary Potential Drift Characterization
- **Source:** [03_cloning.md § 11.6.](source/1_euclidean_gas/03_cloning)
- **Description:** Complete Boundary Potential Drift Characterization

### Bounded Expansion of Inter-Swarm Wasserstein Distance
- **Source:** [03_cloning.md § 12.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Bounded Expansion of Inter-Swarm Wasserstein Distance

### Why Inter-Swarm Error Doesn't Contract Under Cloning
- **Source:** [03_cloning.md § 12.2.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Why Inter-Swarm Error Doesn't Contract Under Cloning

### Component-Wise Bounds on Inter-Swarm Error
- **Source:** [03_cloning.md § 12.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Component-Wise Bounds on Inter-Swarm Error

### Complete Wasserstein Decomposition Drift
- **Source:** [03_cloning.md § 12.2.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Complete Wasserstein Decomposition Drift

### Complete Drift Inequality for the Cloning Operator
- **Source:** [03_cloning.md § 12.3.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Complete Drift Inequality for the Cloning Operator

### Necessity of the Kinetic Operator
- **Source:** [03_cloning.md § 12.3.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Necessity of the Kinetic Operator

### Perfect Complementarity
- **Source:** [03_cloning.md § 12.4.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Perfect Complementarity

### Synergistic Foster-Lyapunov Condition (Preview)
- **Source:** [03_cloning.md § 12.4.2.](source/1_euclidean_gas/03_cloning)
- **Description:** Synergistic Foster-Lyapunov Condition (Preview)

### Existence of Valid Coupling Constants
- **Source:** [03_cloning.md § 12.4.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Existence of Valid Coupling Constants

### Tuning Guidance
- **Source:** [03_cloning.md § 12.4.3.](source/1_euclidean_gas/03_cloning)
- **Description:** Tuning Guidance

### Main Results of the Cloning Analysis (Summary)
- **Source:** [03_cloning.md § 12.5.1.](source/1_euclidean_gas/03_cloning)
- **Description:** Main Results of the Cloning Analysis (Summary)

---

### Source: 04_wasserstein_contraction.md {#04_wasserstein_contraction}


### Target Set and Complement
- **Source:** [04_wasserstein_contraction.md § 2.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Target Set and Complement

### Why These Sets?
- **Source:** [04_wasserstein_contraction.md § 2.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Why These Sets?

### Cluster-Preserving Coupling
- **Source:** [04_wasserstein_contraction.md § 2.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Cluster-Preserving Coupling

### Why This Coupling Works
- **Source:** [04_wasserstein_contraction.md § 2.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Why This Coupling Works

### Variance Decomposition by Clusters
- **Source:** [04_wasserstein_contraction.md § 3.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Variance Decomposition by Clusters

### Between-Group Variance Dominance
- **Source:** [04_wasserstein_contraction.md § 3.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Between-Group Variance Dominance

### Cross-Swarm Distance Decomposition
- **Source:** [04_wasserstein_contraction.md § 3.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Cross-Swarm Distance Decomposition

### Cluster-Level Outlier Alignment
- **Source:** [04_wasserstein_contraction.md § 4.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Cluster-Level Outlier Alignment

### Why This Proof is Static and Robust
- **Source:** [04_wasserstein_contraction.md § 4.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Why This Proof is Static and Robust

### Expected Cross-Distance Change
- **Source:** [04_wasserstein_contraction.md § 5.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Expected Cross-Distance Change

### Cloning Pressure on Target Set
- **Source:** [04_wasserstein_contraction.md § 5.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Cloning Pressure on Target Set

### Average Cloning Pressure Bound
- **Source:** [04_wasserstein_contraction.md § 5.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Average Cloning Pressure Bound

### Wasserstein Distance and Population Cross-Distances
- **Source:** [04_wasserstein_contraction.md § 6.1.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Wasserstein Distance and Population Cross-Distances

### Wasserstein-2 Contraction (Cluster-Based)
- **Source:** [04_wasserstein_contraction.md § 6.2.](source/1_euclidean_gas/04_wasserstein_contraction)
- **Description:** Wasserstein-2 Contraction (Cluster-Based)

---

### Source: 05_kinetic_contraction.md {#05_kinetic_contraction}


### The Kinetic Operator (Stratonovich Form)
- **Source:** [05_kinetic_contraction.md § 5.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** The Kinetic Operator (Stratonovich Form)

### Relationship to Itô Formulation
- **Source:** [05_kinetic_contraction.md § 5.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Relationship to Itô Formulation

### Globally Confining Potential
- **Source:** [05_kinetic_contraction.md § 5.3.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Globally Confining Potential

### Canonical Confining Potential
- **Source:** [05_kinetic_contraction.md § 5.3.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Canonical Confining Potential

### Anisotropic Diffusion Tensor
- **Source:** [05_kinetic_contraction.md § 5.3.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Anisotropic Diffusion Tensor

### Why Uniform Ellipticity Matters
- **Source:** [05_kinetic_contraction.md § 5.3.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Why Uniform Ellipticity Matters

### Friction and Integration Parameters
- **Source:** [05_kinetic_contraction.md § 5.3.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Friction and Integration Parameters

### Fokker-Planck Equation for the Kinetic Operator
- **Source:** [05_kinetic_contraction.md § 5.4.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Fokker-Planck Equation for the Kinetic Operator

### Formal Invariant Measure (Without Boundary)
- **Source:** [05_kinetic_contraction.md § 5.4.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Formal Invariant Measure (Without Boundary)

### BAOAB Integrator for Stratonovich Langevin
- **Source:** [05_kinetic_contraction.md § 5.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** BAOAB Integrator for Stratonovich Langevin

### Stratonovich Correction for Anisotropic Case
- **Source:** [05_kinetic_contraction.md § 5.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Stratonovich Correction for Anisotropic Case

### Infinitesimal Generator of the Kinetic SDE
- **Source:** [05_kinetic_contraction.md § 5.7.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Infinitesimal Generator of the Kinetic SDE

### Why We Work with Generators
- **Source:** [05_kinetic_contraction.md § 5.7.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Why We Work with Generators

### Discrete-Time Inheritance of Generator Drift
- **Source:** [05_kinetic_contraction.md § 5.7.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Discrete-Time Inheritance of Generator Drift

### BAOAB Weak Error for Variance Lyapunov Functions
- **Source:** [05_kinetic_contraction.md § 1.7.3.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** BAOAB Weak Error for Variance Lyapunov Functions

### BAOAB Weak Error for Boundary Lyapunov Function
- **Source:** [05_kinetic_contraction.md § 1.7.3.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** BAOAB Weak Error for Boundary Lyapunov Function

### BAOAB Weak Error for Wasserstein Distance
- **Source:** [05_kinetic_contraction.md § 1.7.3.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** BAOAB Weak Error for Wasserstein Distance

### Explicit Discretization Constants
- **Source:** [05_kinetic_contraction.md § 5.7.4.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Explicit Discretization Constants

### No Convexity Required
- **Source:** [05_kinetic_contraction.md § 6.1.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** No Convexity Required

### The Hypocoercive Norm
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** The Hypocoercive Norm

### Intuition for the Coupling Term
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Intuition for the Coupling Term

### Inter-Swarm Error Contraction Under Kinetic Operator
- **Source:** [05_kinetic_contraction.md § 6.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Inter-Swarm Error Contraction Under Kinetic Operator

### Drift of Location Error Under Kinetics
- **Source:** [05_kinetic_contraction.md § 6.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Drift of Location Error Under Kinetics

### Drift of Structural Error Under Kinetics
- **Source:** [05_kinetic_contraction.md § 6.6.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Drift of Structural Error Under Kinetics

### Velocity Variance Component (Recall)
- **Source:** [05_kinetic_contraction.md § 7.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Velocity Variance Component (Recall)

### Velocity Variance Contraction Under Kinetic Operator
- **Source:** [05_kinetic_contraction.md § 7.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Velocity Variance Contraction Under Kinetic Operator

### Net Velocity Variance Contraction for Composed Operator
- **Source:** [05_kinetic_contraction.md § 7.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Net Velocity Variance Contraction for Composed Operator

### Positional Variance Component (Recall)
- **Source:** [05_kinetic_contraction.md § 6.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Positional Variance Component (Recall)

### Bounded Positional Variance Expansion Under Kinetics
- **Source:** [05_kinetic_contraction.md § 6.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Bounded Positional Variance Expansion Under Kinetics

### Net Positional Variance Contraction for Composed Operator
- **Source:** [05_kinetic_contraction.md § 6.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Net Positional Variance Contraction for Composed Operator

### Boundary Potential (Recall)
- **Source:** [05_kinetic_contraction.md § 7.2.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Boundary Potential (Recall)

### Boundary Potential Contraction Under Kinetic Operator
- **Source:** [05_kinetic_contraction.md § 7.3.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Boundary Potential Contraction Under Kinetic Operator

### Total Boundary Safety from Dual Mechanisms
- **Source:** [05_kinetic_contraction.md § 7.5.](source/1_euclidean_gas/05_kinetic_contraction)
- **Description:** Total Boundary Safety from Dual Mechanisms

---

### Source: 06_convergence.md {#06_convergence}


### Summary of Required Operator Drifts
- **Source:** [06_convergence.md § 2.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Summary of Required Operator Drifts

### Synergistic Lyapunov Function (Recall)
- **Source:** [06_convergence.md § 3.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Synergistic Lyapunov Function (Recall)

### Complete Drift Characterization
- **Source:** [06_convergence.md § 3.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Complete Drift Characterization

### Foster-Lyapunov Drift for the Composed Operator
- **Source:** [06_convergence.md § 3.4.](source/1_euclidean_gas/06_convergence)
- **Description:** Foster-Lyapunov Drift for the Composed Operator

### The Cemetery State
- **Source:** [06_convergence.md § 4.2.](source/1_euclidean_gas/06_convergence)
- **Description:** The Cemetery State

### Why Extinction is Inevitable (Eventually)
- **Source:** [06_convergence.md § 4.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Why Extinction is Inevitable (Eventually)

### Quasi-Stationary Distribution (QSD)
- **Source:** [06_convergence.md § 4.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Quasi-Stationary Distribution (QSD)

### φ-Irreducibility of the Euclidean Gas
- **Source:** [06_convergence.md § 4.4.1.](source/1_euclidean_gas/06_convergence)
- **Description:** φ-Irreducibility of the Euclidean Gas

### Aperiodicity of the Euclidean Gas
- **Source:** [06_convergence.md § 4.4.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Aperiodicity of the Euclidean Gas

### Geometric Ergodicity and Convergence to QSD
- **Source:** [06_convergence.md § 4.5.](source/1_euclidean_gas/06_convergence)
- **Description:** Geometric Ergodicity and Convergence to QSD

### Properties of the Quasi-Stationary Distribution
- **Source:** [06_convergence.md § 4.6.](source/1_euclidean_gas/06_convergence)
- **Description:** Properties of the Quasi-Stationary Distribution

### Equilibrium Variance Bounds from Drift Inequalities
- **Source:** [06_convergence.md § 4.6.](source/1_euclidean_gas/06_convergence)
- **Description:** Equilibrium Variance Bounds from Drift Inequalities

### Velocity Dissipation Rate (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Velocity Dissipation Rate (Parameter-Explicit)

### Positional Contraction Rate (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Positional Contraction Rate (Parameter-Explicit)

### Wasserstein Contraction Rate (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Wasserstein Contraction Rate (Parameter-Explicit)

### Boundary Contraction Rate (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.4.](source/1_euclidean_gas/06_convergence)
- **Description:** Boundary Contraction Rate (Parameter-Explicit)

### Synergistic Rate Derivation from Component Drifts
- **Source:** [06_convergence.md § 5.5.](source/1_euclidean_gas/06_convergence)
- **Description:** Synergistic Rate Derivation from Component Drifts

### Total Convergence Rate (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.5.](source/1_euclidean_gas/06_convergence)
- **Description:** Total Convergence Rate (Parameter-Explicit)

### Mixing Time (Parameter-Explicit)
- **Source:** [06_convergence.md § 5.6.](source/1_euclidean_gas/06_convergence)
- **Description:** Mixing Time (Parameter-Explicit)

### Parameter Selection for Optimal Convergence
- **Source:** [06_convergence.md § 5.7.](source/1_euclidean_gas/06_convergence)
- **Description:** Parameter Selection for Optimal Convergence

### Complete Parameter Space
- **Source:** [06_convergence.md § 6.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Complete Parameter Space

### Parameter Classification
- **Source:** [06_convergence.md § 6.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Parameter Classification

### Log-Sensitivity Matrix for Convergence Rates
- **Source:** [06_convergence.md § 6.3.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Log-Sensitivity Matrix for Convergence Rates

### Explicit Rate Sensitivity Matrix
- **Source:** [06_convergence.md § 6.3.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Explicit Rate Sensitivity Matrix

### Equilibrium Constant Sensitivity Matrix
- **Source:** [06_convergence.md § 6.3.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Equilibrium Constant Sensitivity Matrix

### SVD of Rate Sensitivity Matrix
- **Source:** [06_convergence.md § 6.4.](source/1_euclidean_gas/06_convergence)
- **Description:** SVD of Rate Sensitivity Matrix

### Condition Number of Rate Sensitivity
- **Source:** [06_convergence.md § 6.4.](source/1_euclidean_gas/06_convergence)
- **Description:** Condition Number of Rate Sensitivity

### Parameter Optimization Problem
- **Source:** [06_convergence.md § 6.5.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Parameter Optimization Problem

### Subgradient of min() Function
- **Source:** [06_convergence.md § 6.5.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Subgradient of min() Function

### Necessity of Balanced Rates at Optimum
- **Source:** [06_convergence.md § 6.5.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Necessity of Balanced Rates at Optimum

### Restitution-Friction Coupling
- **Source:** [06_convergence.md § 6.6.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Restitution-Friction Coupling

### Position Jitter - Cloning Rate Coupling
- **Source:** [06_convergence.md § 6.6.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Position Jitter - Cloning Rate Coupling

### Phase-Space Pairing Quality
- **Source:** [06_convergence.md § 6.6.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Phase-Space Pairing Quality

### Parameter Error Propagation Bound
- **Source:** [06_convergence.md § 6.7.](source/1_euclidean_gas/06_convergence)
- **Description:** Parameter Error Propagation Bound

### Closed-Form Balanced Optimum
- **Source:** [06_convergence.md § 6.10.1.](source/1_euclidean_gas/06_convergence)
- **Description:** Closed-Form Balanced Optimum

### Projected Gradient Ascent for Parameter Optimization
- **Source:** [06_convergence.md § 6.10.2.](source/1_euclidean_gas/06_convergence)
- **Description:** Projected Gradient Ascent for Parameter Optimization

### Pareto Optimality in Parameter Space
- **Source:** [06_convergence.md § 6.10.3.](source/1_euclidean_gas/06_convergence)
- **Description:** Pareto Optimality in Parameter Space

### Adaptive Parameter Tuning
- **Source:** [06_convergence.md § 6.10.4.](source/1_euclidean_gas/06_convergence)
- **Description:** Adaptive Parameter Tuning

---

### Source: 07_mean_field.md {#07_mean_field}


### Phase Space
- **Source:** [07_mean_field.md § **1.1. Phase Space and Probability Density**](source/1_euclidean_gas/07_mean_field)
- **Description:** Phase Space

### Phase-Space Density
- **Source:** [07_mean_field.md § **1.1. Phase Space and Probability Density**](source/1_euclidean_gas/07_mean_field)
- **Description:** Phase-Space Density

### Mean-Field Statistical Moments
- **Source:** [07_mean_field.md § **1.2. Mean-Field Measurement Pipeline**](source/1_euclidean_gas/07_mean_field)
- **Description:** Mean-Field Statistical Moments

### Mean-Field Regularized Standard Deviation
- **Source:** [07_mean_field.md § **1.2. Mean-Field Measurement Pipeline**](source/1_euclidean_gas/07_mean_field)
- **Description:** Mean-Field Regularized Standard Deviation

### Mean-Field Z-Scores
- **Source:** [07_mean_field.md § **1.3. Density-Dependent Fitness Potential**](source/1_euclidean_gas/07_mean_field)
- **Description:** Mean-Field Z-Scores

### Mean-Field Fitness Potential
- **Source:** [07_mean_field.md § **1.3. Density-Dependent Fitness Potential**](source/1_euclidean_gas/07_mean_field)
- **Description:** Mean-Field Fitness Potential

### The BAOAB Update Rule
- **Source:** [07_mean_field.md § **2.1. The Underlying Discrete-Time Integrator: BA](source/1_euclidean_gas/07_mean_field)
- **Description:** The BAOAB Update Rule

### Kinetic Transport Operator
- **Source:** [07_mean_field.md § **2.2. The Kinetic Transport Operator ($L^\dagger$](source/1_euclidean_gas/07_mean_field)
- **Description:** Kinetic Transport Operator

### Interior Killing Operator
- **Source:** [07_mean_field.md § **2.3. The Reaction Operators (Killing, Revival, a](source/1_euclidean_gas/07_mean_field)
- **Description:** Interior Killing Operator

### Revival Operator
- **Source:** [07_mean_field.md § **2.3. The Reaction Operators (Killing, Revival, a](source/1_euclidean_gas/07_mean_field)
- **Description:** Revival Operator

### Internal Cloning Operator (Derived Form)
- **Source:** [07_mean_field.md § **2.3.3. Derivation of the Internal Cloning Operat](source/1_euclidean_gas/07_mean_field)
- **Description:** Internal Cloning Operator (Derived Form)

### Transport Operator and Probability Flux
- **Source:** [07_mean_field.md § **3.1. The Transport Operator ($L^\dagger$) is Mas](source/1_euclidean_gas/07_mean_field)
- **Description:** Transport Operator and Probability Flux

### Mass Conservation of Transport
- **Source:** [07_mean_field.md § **3.1. The Transport Operator ($L^\dagger$) is Mas](source/1_euclidean_gas/07_mean_field)
- **Description:** Mass Conservation of Transport

### The Mean-Field Equations for the Euclidean Gas
- **Source:** [07_mean_field.md § **3.3. The Coupled Mean-Field Equations**](source/1_euclidean_gas/07_mean_field)
- **Description:** The Mean-Field Equations for the Euclidean Gas

### Total Mass Conservation and Population Dynamics
- **Source:** [07_mean_field.md § **3.3. The Coupled Mean-Field Equations**](source/1_euclidean_gas/07_mean_field)
- **Description:** Total Mass Conservation and Population Dynamics

### Summary of Regularity Assumptions
- **Source:** [07_mean_field.md § **4.2. Well-Posedness and Future Work**](source/1_euclidean_gas/07_mean_field)
- **Description:** Summary of Regularity Assumptions

### Regularity of the Valid Domain
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)
- **Description:** Regularity of the Valid Domain

### Regularity of the Discrete Integrator
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)
- **Description:** Regularity of the Discrete Integrator

### Density Regularity
- **Source:** [07_mean_field.md § **4.4.1. Mathematical Setup and Regularity Assumpt](source/1_euclidean_gas/07_mean_field)
- **Description:** Density Regularity

### Consistency of the Interior Killing Rate Approximation
- **Source:** [07_mean_field.md § **4.4.2. Main Theorem: Rigorous Killing Rate Appro](source/1_euclidean_gas/07_mean_field)
- **Description:** Consistency of the Interior Killing Rate Approximation

### Mean-Field Limit (Informal Statement)
- **Source:** [07_mean_field.md § **4.4.4. Connection to the Main Mean-Field Result*](source/1_euclidean_gas/07_mean_field)
- **Description:** Mean-Field Limit (Informal Statement)

---

### Source: 08_propagation_chaos.md {#08_propagation_chaos}


### Sequence of N-Particle QSDs and their Marginals
- **Source:** [08_propagation_chaos.md § 2.](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Sequence of N-Particle QSDs and their Marginals

### The Sequence of Marginals $\{\mu_N\}$ is Tight
- **Source:** [08_propagation_chaos.md § **Introduction**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** The Sequence of Marginals  is Tight

### Exchangeability of the N-Particle QSD
- **Source:** [08_propagation_chaos.md § **Lemma A.1: Exchangeability of the N-Particle QSD](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Exchangeability of the N-Particle QSD

### Weak Convergence of the Empirical Companion Measure
- **Source:** [08_propagation_chaos.md § **Lemma A.2: Weak Convergence of the Empirical Com](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Weak Convergence of the Empirical Companion Measure

### Continuity of the Reward Moments
- **Source:** [08_propagation_chaos.md § **Lemma B.1: Continuity of the Reward Moments**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Continuity of the Reward Moments

### Continuity of the Distance Moments
- **Source:** [08_propagation_chaos.md § **Lemma B.2: Continuity of the Distance Moments**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Continuity of the Distance Moments

### Uniform Integrability and Interchange of Limits
- **Source:** [08_propagation_chaos.md § **Lemma C.1: Uniform Integrability and Interchange](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Uniform Integrability and Interchange of Limits

### Convergence of Boundary Death and Revival
- **Source:** [08_propagation_chaos.md § **Lemma C.2: Convergence of the Boundary Death and](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Convergence of Boundary Death and Revival

### QSD Stationarity vs. True Stationarity
- **Source:** [08_propagation_chaos.md § **The QSD Stationarity Condition with Extinction R](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** QSD Stationarity vs. True Stationarity

### Extinction Rate Vanishes in the Mean-Field Limit
- **Source:** [08_propagation_chaos.md § **Proof of Vanishing Extinction Rate**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Extinction Rate Vanishes in the Mean-Field Limit

### Physical Interpretation
- **Source:** [08_propagation_chaos.md § **Proof of Vanishing Extinction Rate**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Physical Interpretation

### Limit Points are Weak Solutions to the Stationary Mean-Field PDE
- **Source:** [08_propagation_chaos.md § **Theorem C.2: Limit Points are Weak Solutions to ](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Limit Points are Weak Solutions to the Stationary Mean-Field PDE

### Weighted Sobolev Space $H^1_w(\Omega)$
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Weighted Sobolev Space

### Completeness of $H^1_w(\Omega)$
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Completeness of

### Completeness of the Constraint Set $\mathcal{P}$
- **Source:** [08_propagation_chaos.md § **Part A: The Weighted Function Space**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Completeness of the Constraint Set

### Self-Mapping Property of the Solution Operator
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Self-Mapping Property of the Solution Operator

### Lipschitz Continuity of Moment Functionals
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Lipschitz Continuity of Moment Functionals

### Fixed Points Lie in a Bounded Ball
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Fixed Points Lie in a Bounded Ball

### Lipschitz Continuity of the Fitness Potential
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Lipschitz Continuity of the Fitness Potential

### Local Lipschitz Continuity of the Cloning Operator
- **Source:** [08_propagation_chaos.md § **Reformulation as a Fixed-Point Problem**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Local Lipschitz Continuity of the Cloning Operator

### Hörmander's Theorem for Kinetic Operators
- **Source:** [08_propagation_chaos.md § **C.2. Hörmander's Hypoellipticity Condition**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Hörmander's Theorem for Kinetic Operators

### Verification of Hörmander's Condition for the Kinetic Operator
- **Source:** [08_propagation_chaos.md § **C.2. Hörmander's Hypoellipticity Condition**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Verification of Hörmander's Condition for the Kinetic Operator

### Hypoelliptic Regularity for the Kinetic Operator
- **Source:** [08_propagation_chaos.md § **C.3. Hypoelliptic Regularity Estimates**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Hypoelliptic Regularity for the Kinetic Operator

### Scaling of $C_{\text{hypo}}$ with Diffusion Strength
- **Source:** [08_propagation_chaos.md § **C.4. Scaling of the Hypoelliptic Constant**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Scaling of  with Diffusion Strength

### Contraction Property of the Solution Operator on an Invariant Ball
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Contraction Property of the Solution Operator on an Invariant Ball

### Uniqueness of the Stationary Solution
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Uniqueness of the Stationary Solution

### The proof structure demonstrates a powerful technique in nonlinear analysis: when global Lipschitz continuity fails, we can still prove uniqueness by:
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** The proof structure demonstrates a powerful technique in nonlinear analysis: ...

### This uniqueness proof reveals a deep connection between the algorithm's design parameters and the mathematical well-posedness of the model. The condition
- **Source:** [08_propagation_chaos.md § **Part D: Assembly of the Contraction Argument**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** This uniqueness proof reveals a deep connection between the algorithm's desig...

### Sequence of N-Particle QSDs and their Marginals
- **Source:** [08_propagation_chaos.md § **6.2. The Sequence of N-Particle Stationary Measu](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Sequence of N-Particle QSDs and their Marginals

### The Sequence of Marginals $\{\mu_N\}$ is Tight
- **Source:** [08_propagation_chaos.md § **6.3. Step 1: Tightness of the Marginal Sequence*](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** The Sequence of Marginals  is Tight

### Limit Points are Weak Solutions to the Stationary Mean-Field PDE
- **Source:** [08_propagation_chaos.md § **6.4. Step 2: Identification of the Limit Point**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Limit Points are Weak Solutions to the Stationary Mean-Field PDE

### Uniqueness of the Stationary Solution
- **Source:** [08_propagation_chaos.md § **6.5. Step 3: Uniqueness of the Weak Solution**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Uniqueness of the Stationary Solution

### Convergence of Macroscopic Observables (The Thermodynamic Limit)
- **Source:** [08_propagation_chaos.md § **6.6. The Thermodynamic Limit**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Convergence of Macroscopic Observables (The Thermodynamic Limit)

### Wasserstein-2 Convergence in the Thermodynamic Limit
- **Source:** [08_propagation_chaos.md § **6.6. The Thermodynamic Limit**](source/1_euclidean_gas/08_propagation_chaos)
- **Description:** Wasserstein-2 Convergence in the Thermodynamic Limit

---

### Source: 09_kl_convergence.md {#09_kl_convergence}


### Exponential KL-Convergence for the Euclidean Gas
- **Source:** [09_kl_convergence.md § I.1.1. Primary Theorem (Log-Concave Case)](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL-Convergence for the Euclidean Gas

### KL-Convergence Without Log-Concavity
- **Source:** [09_kl_convergence.md § I.1.2. Extended Theorem (Non-Convex Case)](source/1_euclidean_gas/09_kl_convergence)
- **Description:** KL-Convergence Without Log-Concavity

### Exponential KL-Convergence for the Euclidean Gas
- **Source:** [09_kl_convergence.md § 0.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL-Convergence for the Euclidean Gas

### Relative Entropy and Fisher Information
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Relative Entropy and Fisher Information

### Logarithmic Sobolev Inequality (LSI)
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Logarithmic Sobolev Inequality (LSI)

### Discrete-Time LSI
- **Source:** [09_kl_convergence.md § 1.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Discrete-Time LSI

### Bakry-Émery Criterion for LSI
- **Source:** [09_kl_convergence.md § 1.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Bakry-Émery Criterion for LSI

### Target Gibbs Measure for Kinetic Dynamics
- **Source:** [09_kl_convergence.md § 2.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Target Gibbs Measure for Kinetic Dynamics

### The generator $\mathcal{L}_{\text{kin}}$ is **not self-adjoint** with respect to $\pi_{\text{kin}}$. This non-reversibility is a fundamental barrier to applying classical LSI theory.
- **Source:** [09_kl_convergence.md § 2.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** The generator  is **not self-adjoint** with respect to . This non-reversibili...

### Hypocoercive Metric and Modified Dirichlet Form
- **Source:** [09_kl_convergence.md § 2.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive Metric and Modified Dirichlet Form

### Dissipation of the Hypocoercive Norm
- **Source:** [09_kl_convergence.md § 2.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Dissipation of the Hypocoercive Norm

### Hypocoercive LSI for the Kinetic Flow Map
- **Source:** [09_kl_convergence.md § 2.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive LSI for the Kinetic Flow Map

### Tensorization of LSI
- **Source:** [09_kl_convergence.md § 3.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Tensorization of LSI

### LSI for N-Particle Kinetic Operator
- **Source:** [09_kl_convergence.md § 3.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** LSI for N-Particle Kinetic Operator

### Log-Concavity of the Quasi-Stationary Distribution
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Log-Concavity of the Quasi-Stationary Distribution

### Motivation and Justification
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Motivation and Justification

### Explicit Log-Concavity Condition
- **Source:** [09_kl_convergence.md § From Axiom to Verifiable Condition](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Explicit Log-Concavity Condition

### Log-Concavity for Pure Yang-Mills Vacuum
- **Source:** [09_kl_convergence.md § Verification for Specific Physical Systems](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Log-Concavity for Pure Yang-Mills Vacuum

### Implications for Millennium Prize
- **Source:** [09_kl_convergence.md § Verification for Specific Physical Systems](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Implications for Millennium Prize

### Conditional Independence of Cloning
- **Source:** [09_kl_convergence.md § 4.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Conditional Independence of Cloning

### The HWI Inequality (Otto-Villani)
- **Source:** [09_kl_convergence.md § 4.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** The HWI Inequality (Otto-Villani)

### The HWI inequality provides a **bridge** between:
- **Source:** [09_kl_convergence.md § 4.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** The HWI inequality provides a **bridge** between:

### Wasserstein-2 Contraction for Cloning
- **Source:** [09_kl_convergence.md § 4.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Wasserstein-2 Contraction for Cloning

### Fisher Information Bound After Cloning
- **Source:** [09_kl_convergence.md § 4.4.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Fisher Information Bound After Cloning

### Entropy Contraction for the Cloning Operator
- **Source:** [09_kl_convergence.md § 4.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Entropy Contraction for the Cloning Operator

### Interpretation
- **Source:** [09_kl_convergence.md § 4.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Interpretation

### Entropy-Transport Lyapunov Function
- **Source:** [09_kl_convergence.md § 5.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Entropy-Transport Lyapunov Function

### Entropy-Transport Dissipation Inequality
- **Source:** [09_kl_convergence.md § 5.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Entropy-Transport Dissipation Inequality

### This lemma is the **key technical innovation**. It shows that the geometric contraction in Wasserstein space (already proven in [04_convergence.md](04_convergence.md)) drives entropy dissipation. The constant $\alpha$ depends on:
- **Source:** [09_kl_convergence.md § 5.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** This lemma is the **key technical innovation**. It shows that the geometric c...

### Kinetic Evolution Bounds
- **Source:** [09_kl_convergence.md § 5.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Kinetic Evolution Bounds

### Linear Contraction of the Entropy-Transport Lyapunov Function
- **Source:** [09_kl_convergence.md § 5.4.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Linear Contraction of the Entropy-Transport Lyapunov Function

### Discrete-Time LSI for the Euclidean Gas
- **Source:** [09_kl_convergence.md § 5.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Discrete-Time LSI for the Euclidean Gas

### Quantitative LSI Constant
- **Source:** [09_kl_convergence.md § 5.6.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Quantitative LSI Constant

### Exponential KL-Convergence via LSI
- **Source:** [09_kl_convergence.md § 6.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL-Convergence via LSI

### KL-Convergence of the Euclidean Gas (Main Result)
- **Source:** [09_kl_convergence.md § 6.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** KL-Convergence of the Euclidean Gas (Main Result)

### Relationship Between KL and TV Convergence Rates
- **Source:** [09_kl_convergence.md § 6.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Relationship Between KL and TV Convergence Rates

### LSI Stability Under Bounded Perturbations
- **Source:** [09_kl_convergence.md § 7.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** LSI Stability Under Bounded Perturbations

### LSI for the ρ-Localized Geometric Gas
- **Source:** [09_kl_convergence.md § 7.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** LSI for the ρ-Localized Geometric Gas

### N-Uniform Logarithmic Sobolev Inequality
- **Source:** [09_kl_convergence.md § 9.6.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** N-Uniform Logarithmic Sobolev Inequality

### Entropy Dissipation Under Cloning (Mean-Field Sketch)
- **Source:** [09_kl_convergence.md § Lemma Statement](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Entropy Dissipation Under Cloning (Mean-Field Sketch)

### Sinh Inequality
- **Source:** [09_kl_convergence.md § A.4: ✅ Contraction Inequality via Permutation Symm](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Sinh Inequality

### Entropy Bound via De Bruijn Identity
- **Source:** [09_kl_convergence.md § Rigorous Formulation](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Entropy Bound via De Bruijn Identity

### Exponential KL-Convergence via Mean-Field Analysis
- **Source:** [09_kl_convergence.md § Main Result](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL-Convergence via Mean-Field Analysis

### Hypocoercive LSI for Kinetic Operator (Reference)
- **Source:** [09_kl_convergence.md § Step 1: Kinetic Operator LSI (Existing Result)](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive LSI for Kinetic Operator (Reference)

### Mean-Field Cloning Entropy Dissipation
- **Source:** [09_kl_convergence.md § Step 2: Cloning Operator Contraction (Mean-Field P](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Mean-Field Cloning Entropy Dissipation

### Composition of LSI Operators (Reference)
- **Source:** [09_kl_convergence.md § Step 3: Composition via Existing Theorem](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Composition of LSI Operators (Reference)

### Discrete Dirichlet Form
- **Source:** [09_kl_convergence.md § Step 4: Discrete-Time LSI Formulation](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Discrete Dirichlet Form

### Discrete-Time LSI
- **Source:** [09_kl_convergence.md § Step 4: Discrete-Time LSI Formulation](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Discrete-Time LSI

### Exponential Convergence from LSI
- **Source:** [09_kl_convergence.md § Step 5: Exponential KL Convergence](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential Convergence from LSI

### Exponential KL-Convergence via Mean-Field Generator Analysis
- **Source:** [09_kl_convergence.md § Main Result](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL-Convergence via Mean-Field Generator Analysis

### Hypocoercive LSI for Kinetic Operator
- **Source:** [09_kl_convergence.md § 1.4.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive LSI for Kinetic Operator

### Mean-Field Cloning Contraction
- **Source:** [09_kl_convergence.md § 2.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Mean-Field Cloning Contraction

### Composition of Kinetic and Cloning Operators
- **Source:** [09_kl_convergence.md § Section 3: Composition](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Composition of Kinetic and Cloning Operators

### Exponential KL Convergence
- **Source:** [09_kl_convergence.md § Section 4: Exponential Convergence](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL Convergence

### Log-Concavity of the Quasi-Stationary Distribution (Current Requirement)
- **Source:** [09_kl_convergence.md § 0.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Log-Concavity of the Quasi-Stationary Distribution (Current Requirement)

### Confining Potential (from 04_convergence.md, Axiom 1.3.1)
- **Source:** [09_kl_convergence.md § 0.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Confining Potential (from 04convergence.md, Axiom 1.3.1)

### Exponential KL Convergence for Non-Convex Fitness (Informal)
- **Source:** [09_kl_convergence.md § 0.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL Convergence for Non-Convex Fitness (Informal)

### Confining Potential (Complete Statement)
- **Source:** [09_kl_convergence.md § 1.1.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Confining Potential (Complete Statement)

### Villani's Hypocoercivity (Simplified)
- **Source:** [09_kl_convergence.md § 2.1.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Villani's Hypocoercivity (Simplified)

### Hypocoercivity for Piecewise Smooth Confining Potentials
- **Source:** [09_kl_convergence.md § 2.1.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercivity for Piecewise Smooth Confining Potentials

### Hypocoercive LSI for Discrete-Time Kinetic Operator
- **Source:** [09_kl_convergence.md § 2.2.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive LSI for Discrete-Time Kinetic Operator

### N-Particle Hypocoercive LSI
- **Source:** [09_kl_convergence.md § 2.3.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** N-Particle Hypocoercive LSI

### Discrete Status-Change Metric
- **Source:** [09_kl_convergence.md § 3.2.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Discrete Status-Change Metric

### Lipschitz Continuity of Softmax-Weighted Companion Selection
- **Source:** [09_kl_convergence.md § 3.3.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Lipschitz Continuity of Softmax-Weighted Companion Selection

### Dobrushin Contraction for Euclidean Gas
- **Source:** [09_kl_convergence.md § 3.4.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Dobrushin Contraction for Euclidean Gas

### Exponential Convergence in $d_{\text{status}}$ Metric
- **Source:** [09_kl_convergence.md § 3.5.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential Convergence in  Metric

### Exponential KL Convergence for Non-Convex Fitness Landscapes
- **Source:** [09_kl_convergence.md § 4.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Exponential KL Convergence for Non-Convex Fitness Landscapes

### Why Composition Fails
- **Source:** [09_kl_convergence.md § 4.5.1.](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Why Composition Fails

### Foster-Lyapunov Drift (Unconditional)
- **Source:** [09_kl_convergence.md § What We Have (Unconditional)](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Foster-Lyapunov Drift (Unconditional)

### Logarithmic Sobolev Inequality (Target)
- **Source:** [09_kl_convergence.md § What We Want](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Logarithmic Sobolev Inequality (Target)

### Random Walk on ℤ
- **Source:** [09_kl_convergence.md § The Gap](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Random Walk on ℤ

### Classical Bakry-Émery Criterion
- **Source:** [09_kl_convergence.md § 0.2](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Classical Bakry-Émery Criterion

### Villani's Hypocoercivity (Informal)
- **Source:** [09_kl_convergence.md § Villani's Hypocoercivity (2009)](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Villani's Hypocoercivity (Informal)

### Synergistic Foster-Lyapunov (Established)
- **Source:** [09_kl_convergence.md § 1.1](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Synergistic Foster-Lyapunov (Established)

### Hypocoercive LSI for Ψ_kin (Established)
- **Source:** [09_kl_convergence.md § 1.2](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Hypocoercive LSI for Ψkin (Established)

### Dobrushin Contraction (Established)
- **Source:** [09_kl_convergence.md § 1.3](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Dobrushin Contraction (Established)

### Unconditional LSI for Euclidean Gas (TARGET)
- **Source:** [09_kl_convergence.md § 3.1](source/1_euclidean_gas/09_kl_convergence)
- **Description:** Unconditional LSI for Euclidean Gas (TARGET)

---

### Source: 10_qsd_exchangeability_theory.md {#10_qsd_exchangeability_theory}


### Exchangeability of the QSD
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.1 Main Result](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Exchangeability of the QSD

### Mixture Representation (Hewitt-Savage)
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.2 Hewitt-Savage Representation](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Mixture Representation (Hewitt-Savage)

### Single-Particle Marginal
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.3 Single-Particle Marginal](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Single-Particle Marginal

### Marginal as Mixture Average
- **Source:** [10_qsd_exchangeability_theory.md § A1.1.3 Single-Particle Marginal](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Marginal as Mixture Average

### Propagation of Chaos
- **Source:** [10_qsd_exchangeability_theory.md § A1.2.1 Main Convergence Result](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Propagation of Chaos

### Quantitative Decorrelation
- **Source:** [10_qsd_exchangeability_theory.md § A1.2.2 Correlation Decay](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Quantitative Decorrelation

### N-Uniform LSI via Hypocoercivity
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.1 LSI for Exchangeable Measures](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** N-Uniform LSI via Hypocoercivity

### Conditional Gaussian Structure
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.1 LSI for Exchangeable Measures](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Conditional Gaussian Structure

### Mean-Field LSI from N-Uniform Bounds
- **Source:** [10_qsd_exchangeability_theory.md § A1.3.2 Implications for Mean-Field LSI](source/1_euclidean_gas/10_qsd_exchangeability_theory)
- **Description:** Mean-Field LSI from N-Uniform Bounds

---

### Source: 12_quantitative_error_bounds.md {#12_quantitative_error_bounds}

### Wasserstein-Entropy Inequality
- **Type:** Lemma
- **Label:** `lem-wasserstein-entropy`
- **Tags:** wasserstein, lsi, talagrand, quantitative
- **Source:** [12_quantitative_error_bounds.md § 1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** W_2² ≤ (2/λ_LSI)·D_KL from Otto-Villani Talagrand inequality

### Quantitative KL Bound
- **Type:** Lemma
- **Label:** `lem-quantitative-kl-bound`
- **Tags:** kl-divergence, mean-field, quantitative, interaction
- **Source:** [12_quantitative_error_bounds.md § 2](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** D_KL(ν_N^QSD ‖ ρ_0^⊗N) ≤ C_int/N via modulated free energy

### Boundedness of Interaction Complexity Constant
- **Type:** Proposition
- **Label:** `prop-interaction-complexity-bound`
- **Tags:** interaction, complexity, mean-field, lipschitz
- **Source:** [12_quantitative_error_bounds.md § 2.1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** C_int ≤ λ·L_log·diam(Ω), finite and N-independent via Jabin-Wang

### Empirical Measure Observable Error
- **Type:** Lemma
- **Label:** `lem-lipschitz-observable-error`
- **Tags:** observable, lipschitz, wasserstein, kantorovich
- **Source:** [12_quantitative_error_bounds.md § 3](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Observable error ≤ L_φ·W_1 via Kantorovich-Rubinstein duality

### Empirical Measure Concentration
- **Type:** Proposition
- **Label:** `prop-empirical-wasserstein-concentration`
- **Tags:** empirical, wasserstein, concentration, fournier-guillin
- **Source:** [12_quantitative_error_bounds.md § 3.1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** E[W_2²(μ̄_N,ρ_0)] ≤ C_var/N + C'·D_KL for exchangeable particles

### Finite Second Moment of Mean-Field QSD
- **Type:** Proposition
- **Label:** `prop-finite-second-moment-meanfield`
- **Tags:** mean-field, qsd, moment-bounds, lyapunov
- **Source:** [12_quantitative_error_bounds.md § 3.2](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** C_var < ∞ from energy dissipation and confinement

### Quantitative Propagation of Chaos
- **Type:** Theorem
- **Label:** `thm-quantitative-propagation-chaos`
- **Tags:** propagation-chaos, quantitative, observable, rate, wasserstein
- **Source:** [12_quantitative_error_bounds.md § 4](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Observable error O(1/√N) for Lipschitz functions, explicit constants

### Fourth-Moment Uniform Bounds for BAOAB
- **Type:** Proposition
- **Label:** `prop-fourth-moment-baoab`
- **Tags:** baoab, discretization, moment-bounds
- **Source:** [12_quantitative_error_bounds.md § Part II.1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** E[|Z_k|⁴] ≤ M_4 uniform in Δt for BAOAB chain

### BAOAB Second-Order Weak Convergence
- **Type:** Lemma
- **Label:** `lem-baoab-weak-error`
- **Tags:** baoab, weak-convergence, discretization, second-order
- **Source:** [12_quantitative_error_bounds.md § Part II.2](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Weak order 2 for observables: |E[φ(Z_t)]-E[φ(Z̃_t)]| ≤ C·Δt²

### BAOAB Invariant Measure Error
- **Type:** Lemma
- **Label:** `lem-baoab-invariant-measure-error`
- **Tags:** baoab, invariant-measure, discretization-error
- **Source:** [12_quantitative_error_bounds.md § Part II.3](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Invariant measure error O(Δt) from weak order 2 via Talay

### Langevin-BAOAB Time Discretization Error
- **Type:** Theorem
- **Label:** `thm-langevin-baoab-discretization-error`
- **Tags:** discretization, baoab, langevin, observable-error
- **Source:** [12_quantitative_error_bounds.md § Part II.4](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Observable error O(Δt) for kinetic operator discretization

### Full System Time Discretization Error
- **Type:** Theorem
- **Label:** `thm-full-system-discretization-error`
- **Tags:** discretization, lie-splitting, total-error
- **Source:** [12_quantitative_error_bounds.md § Part II.5](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Combined BAOAB + Lie splitting error O(Δt) for full system

### One-Step Weak Error for Lie Splitting
- **Type:** Lemma
- **Label:** `lem-lie-splitting-weak-error`
- **Tags:** lie-splitting, weak-error, operator-splitting
- **Source:** [12_quantitative_error_bounds.md § Part II.5.1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** One-step weak error O(Δt²) for Ψ_kin ∘ Ψ_clone splitting

### Uniform Geometric Ergodicity
- **Type:** Lemma
- **Label:** `lem-uniform-geometric-ergodicity`
- **Tags:** ergodicity, geometric, discrete-time, drift
- **Source:** [12_quantitative_error_bounds.md § Part III.1](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Discrete Fragile Gas satisfies drift-minorization with explicit constants

### Relationship Between Continuous and Discrete Mixing Rates
- **Type:** Proposition
- **Label:** `prop-mixing-rate-relationship`
- **Tags:** mixing-time, continuous, discrete, ergodicity
- **Source:** [12_quantitative_error_bounds.md § Part III.2](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** τ_mix^discrete ≤ τ_mix^continuous/Δt + O(1/κΔt)

### Error Propagation for Ergodic Chains
- **Type:** Theorem
- **Label:** `thm-quantitative-error-propagation`
- **Tags:** error-propagation, ergodic, total-variation
- **Source:** [12_quantitative_error_bounds.md § Part III.3](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** Total error bounds via perturbation theory for ergodic chains

### Total Error Bound for Discrete Fragile Gas
- **Type:** Theorem
- **Label:** `thm-total-error-bound`
- **Tags:** total-error, mean-field, discretization, observable
- **Source:** [12_quantitative_error_bounds.md § Part IV](source/1_euclidean_gas/12_quantitative_error_bounds)
- **Description:** |E[φ_N,Δt]-∫φdρ_0| ≤ C_obs/√N + C_disc·Δt combined bound

## Chapter 2: Geometric Gas

**Entries:** 200

---

### Source: 11_geometric_gas.md {#11_geometric_gas}


### Localization Kernel
- **Source:** [11_geometric_gas.md § 1.0.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Localization Kernel

### Localized Mean-Field Moments
- **Source:** [11_geometric_gas.md § 1.0.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Localized Mean-Field Moments

### Unified Localized Z-Score
- **Source:** [11_geometric_gas.md § 1.0.4.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Unified Localized Z-Score

### Limiting Behavior of the Unified Pipeline
- **Source:** [11_geometric_gas.md § 1.0.5.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Limiting Behavior of the Unified Pipeline

### The Adaptive Viscous Fluid SDE
- **Source:** [11_geometric_gas.md § 2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** The Adaptive Viscous Fluid SDE

### Regularized Hessian Diffusion Tensor
- **Source:** [11_geometric_gas.md § 2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Regularized Hessian Diffusion Tensor

### Localized Mean-Field Fitness Potential
- **Source:** [11_geometric_gas.md § 2.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Localized Mean-Field Fitness Potential

### Axiom of a Globally Confining Potential
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Axiom of a Globally Confining Potential

### Axiom of Positive Friction
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Axiom of Positive Friction

### Foundational Cloning and Environmental Axioms
- **Source:** [11_geometric_gas.md § 3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Foundational Cloning and Environmental Axioms

### k-Uniform Boundedness of the Adaptive Force (ρ-Dependent)
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** k-Uniform Boundedness of the Adaptive Force (ρ-Dependent)

### Axiom of a Well-Behaved Viscous Kernel
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Axiom of a Well-Behaved Viscous Kernel

### k-Uniform Ellipticity by Construction (Proven in Chapter 4)
- **Source:** [11_geometric_gas.md § 3.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** k-Uniform Ellipticity by Construction (Proven in Chapter 4)

### k-Uniform Ellipticity of the Regularized Metric
- **Source:** [11_geometric_gas.md § 4.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** k-Uniform Ellipticity of the Regularized Metric

### N-Uniform Boundedness of the Pure Hessian
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** N-Uniform Boundedness of the Pure Hessian

### Rigorous Boundedness of the Hessian
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Rigorous Boundedness of the Hessian

### Failure of Uniformity Without Regularization
- **Source:** [11_geometric_gas.md § 4.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Failure of Uniformity Without Regularization

### Existence and Uniqueness of Solutions
- **Source:** [11_geometric_gas.md § 4.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Existence and Uniqueness of Solutions

### The Backbone SDE
- **Source:** [11_geometric_gas.md § 5.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** The Backbone SDE

### Geometric Ergodicity of the Backbone
- **Source:** [11_geometric_gas.md § 5.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Geometric Ergodicity of the Backbone

### Stratonovich Chain Rule for Lyapunov Functions
- **Source:** [11_geometric_gas.md § 5.4.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Stratonovich Chain Rule for Lyapunov Functions

### Stratonovich Drift for the Hybrid System
- **Source:** [11_geometric_gas.md § 5.4.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Stratonovich Drift for the Hybrid System

### N-Uniform Bounded Perturbation from Adaptive Force
- **Source:** [11_geometric_gas.md § 6.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** N-Uniform Bounded Perturbation from Adaptive Force

### Dissipative Contribution from Viscous Force
- **Source:** [11_geometric_gas.md § 6.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Dissipative Contribution from Viscous Force

### Bounded Change from Adaptive Diffusion
- **Source:** [11_geometric_gas.md § 6.4.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Bounded Change from Adaptive Diffusion

### Total Perturbation Bound (ρ-Dependent)
- **Source:** [11_geometric_gas.md § 6.5.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Total Perturbation Bound (ρ-Dependent)

### Foster-Lyapunov Drift for the ρ-Localized Geometric Viscous Fluid Model
- **Source:** [11_geometric_gas.md § 7.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Foster-Lyapunov Drift for the ρ-Localized Geometric Viscous Fluid Model

### Exponential Convergence
- **Source:** [11_geometric_gas.md § 7.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Exponential Convergence

### The N-Particle Generator for the Adaptive System
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** The N-Particle Generator for the Adaptive System

### Relative Entropy and Fisher Information
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Relative Entropy and Fisher Information

### Logarithmic Sobolev Inequality (LSI)
- **Source:** [11_geometric_gas.md § 8.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Logarithmic Sobolev Inequality (LSI)

### N-Uniform Log-Sobolev Inequality for the Geometric Viscous Fluid Model
- **Source:** [11_geometric_gas.md § 8.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** N-Uniform Log-Sobolev Inequality for the Geometric Viscous Fluid Model

### Exponential Convergence in Relative Entropy
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Exponential Convergence in Relative Entropy

### Geometric Ergodicity via LSI
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Geometric Ergodicity via LSI

### Concentration of Measure
- **Source:** [11_geometric_gas.md § 8.5.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Concentration of Measure

### Existence and Uniqueness of the QSD
- **Source:** [11_geometric_gas.md § 9.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Existence and Uniqueness of the QSD

### Formal Analogy and Evidence
- **Source:** [11_geometric_gas.md § 9.2.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Formal Analogy and Evidence

### Logarithmic Sobolev Inequality for the Mean-Field Generator
- **Source:** [11_geometric_gas.md § 9.3.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Logarithmic Sobolev Inequality for the Mean-Field Generator

### Decomposition of Entropy Dissipation
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Decomposition of Entropy Dissipation

### Microlocal Decomposition
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Microlocal Decomposition

### Microscopic Coercivity (Step A)
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Microscopic Coercivity (Step A)

### Macroscopic Transport (Step B)
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Macroscopic Transport (Step B)

### Microscopic Regularization (Step C)
- **Source:** [11_geometric_gas.md § 9.3.1.](source/2_geometric_gas/11_geometric_gas)
- **Description:** Microscopic Regularization (Step C)

### Derivatives of Localization Weights
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)
- **Description:** Derivatives of Localization Weights

### First Derivative of Localized Mean
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)
- **Description:** First Derivative of Localized Mean

### Second Derivative of Localized Mean
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)
- **Description:** Second Derivative of Localized Mean

### k-Uniform Gradient of Localized Variance
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)
- **Description:** k-Uniform Gradient of Localized Variance

### k-Uniform Hessian of Localized Variance
- **Source:** [11_geometric_gas.md § A.2. Preliminary Lemmas on Weighted Sums](source/2_geometric_gas/11_geometric_gas)
- **Description:** k-Uniform Hessian of Localized Variance

### C¹ Regularity and k-Uniform Gradient Bound
- **Source:** [11_geometric_gas.md § A.3. Theorem A.1: Uniform C¹ Bound on the Fitness ](source/2_geometric_gas/11_geometric_gas)
- **Description:** C¹ Regularity and k-Uniform Gradient Bound

### C² Regularity and k-Uniform Hessian Bound
- **Source:** [11_geometric_gas.md § A.4. Theorem A.2: Uniform C² Bound on the Fitness ](source/2_geometric_gas/11_geometric_gas)
- **Description:** C² Regularity and k-Uniform Hessian Bound

### Verification of Axioms 3.2.1 and 3.2.3
- **Source:** [11_geometric_gas.md § A.5. Corollary: Implications for the Main Text](source/2_geometric_gas/11_geometric_gas)
- **Description:** Verification of Axioms 3.2.1 and 3.2.3

### Signal Generation for the Adaptive Model
- **Source:** [11_geometric_gas.md § B.2. Hypothesis 1: Signal Generation (Geometry-Bas](source/2_geometric_gas/11_geometric_gas)
- **Description:** Signal Generation for the Adaptive Model

### Variance-to-Gap (from 03_cloning.md, Lemma 7.3.1)
- **Source:** [11_geometric_gas.md § B.3.1. The Variance-to-Gap Lemma (Universal)](source/2_geometric_gas/11_geometric_gas)
- **Description:** Variance-to-Gap (from 03cloning.md, Lemma 7.3.1)

### Uniform Bounds on the ρ-Localized Pipeline
- **Source:** [11_geometric_gas.md § B.3.2. Uniform Bounds on ρ-Dependent Pipeline Comp](source/2_geometric_gas/11_geometric_gas)
- **Description:** Uniform Bounds on the ρ-Localized Pipeline

### Raw-Gap to Rescaled-Gap for ρ-Localized Pipeline
- **Source:** [11_geometric_gas.md § B.3.3. Raw-Gap to Rescaled-Gap Propagation (ρ-Depe](source/2_geometric_gas/11_geometric_gas)
- **Description:** Raw-Gap to Rescaled-Gap for ρ-Localized Pipeline

### Logarithmic Gap Bounds (from 03_cloning.md, Lemma 7.5.1)
- **Source:** [11_geometric_gas.md § B.4.1. Foundational Statistical Lemmas (ρ-Independ](source/2_geometric_gas/11_geometric_gas)
- **Description:** Logarithmic Gap Bounds (from 03cloning.md, Lemma 7.5.1)

### Lower Bound on Corrective Diversity Signal (ρ-Dependent)
- **Source:** [11_geometric_gas.md § B.4.2. ρ-Dependent Lower Bound on Corrective Diver](source/2_geometric_gas/11_geometric_gas)
- **Description:** Lower Bound on Corrective Diversity Signal (ρ-Dependent)

### Axiom-Based Bound on Logarithmic Reward Gap (ρ-Dependent)
- **Source:** [11_geometric_gas.md § B.4.3. ρ-Dependent Upper Bound on Adversarial Rewa](source/2_geometric_gas/11_geometric_gas)
- **Description:** Axiom-Based Bound on Logarithmic Reward Gap (ρ-Dependent)

### ρ-Dependent Stability Condition for Intelligent Targeting
- **Source:** [11_geometric_gas.md § B.4.4. The ρ-Dependent Stability Condition](source/2_geometric_gas/11_geometric_gas)
- **Description:** ρ-Dependent Stability Condition for Intelligent Targeting

### Keystone Lemma for the ρ-Localized Adaptive Model
- **Source:** [11_geometric_gas.md § B.5. Conclusion: The Keystone Lemma Holds for the ](source/2_geometric_gas/11_geometric_gas)
- **Description:** Keystone Lemma for the ρ-Localized Adaptive Model

---

### Source: 12_symmetries_geometric_gas.md {#12_symmetries_geometric_gas}


### Swarm Configuration Space
- **Source:** [12_symmetries_geometric_gas.md § 2.1.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Swarm Configuration Space

### Algorithmic Projection Space
- **Source:** [12_symmetries_geometric_gas.md § 2.1.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Algorithmic Projection Space

### Symmetry Transformation
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Symmetry Transformation

### Permutation Group
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Permutation Group

### Euclidean Group Actions
- **Source:** [12_symmetries_geometric_gas.md § 2.2.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Euclidean Group Actions

### ρ-Localized Fitness Potential
- **Source:** [12_symmetries_geometric_gas.md § 2.3.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** ρ-Localized Fitness Potential

### Emergent Riemannian Metric
- **Source:** [12_symmetries_geometric_gas.md § 2.3.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Emergent Riemannian Metric

### Permutation Invariance
- **Source:** [12_symmetries_geometric_gas.md § 3.1.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Permutation Invariance

### Exchangeability of the QSD
- **Source:** [12_symmetries_geometric_gas.md § 3.1.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Exchangeability of the QSD

### Conditional Translation Equivariance
- **Source:** [12_symmetries_geometric_gas.md § 3.2.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Conditional Translation Equivariance

### Breaking of Translation Symmetry
- **Source:** [12_symmetries_geometric_gas.md § 3.2.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Breaking of Translation Symmetry

### Rotational Equivariance
- **Source:** [12_symmetries_geometric_gas.md § 3.3.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Rotational Equivariance

### Radially Symmetric Fitness Landscapes
- **Source:** [12_symmetries_geometric_gas.md § 3.3.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Radially Symmetric Fitness Landscapes

### Fitness Potential Scaling Symmetry
- **Source:** [12_symmetries_geometric_gas.md § 3.4.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Fitness Potential Scaling Symmetry

### Dimensionless Parameter
- **Source:** [12_symmetries_geometric_gas.md § 3.4.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Dimensionless Parameter

### Time-Reversal Asymmetry
- **Source:** [12_symmetries_geometric_gas.md § 3.5.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** Time-Reversal Asymmetry

### H-Theorem for Geometric Gas
- **Source:** [12_symmetries_geometric_gas.md § 3.5.](source/2_geometric_gas/12_symmetries_geometric_gas)
- **Description:** H-Theorem for Geometric Gas

---

### Source: 13_geometric_gas_c3_regularity.md {#13_geometric_gas_c3_regularity}


### Telescoping Identity for Derivatives
- **Source:** [13_geometric_gas_c3_regularity.md § 2.5.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Telescoping Identity for Derivatives

### Measurement Function $C^3$ Regularity
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Measurement Function  Regularity

### Localization Kernel $C^3$ Regularity
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Localization Kernel  Regularity

### Rescale Function $C^3$ Regularity
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Rescale Function  Regularity

### Regularized Standard Deviation $C^\infty$ Regularity
- **Source:** [13_geometric_gas_c3_regularity.md § 3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Regularized Standard Deviation  Regularity

### Third Derivative of Localization Weights
- **Source:** [13_geometric_gas_c3_regularity.md § 4.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Third Derivative of Localization Weights

### k-Uniform Third Derivative of Localized Mean
- **Source:** [13_geometric_gas_c3_regularity.md § 5.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** k-Uniform Third Derivative of Localized Mean

### k-Uniform Third Derivative of Localized Variance
- **Source:** [13_geometric_gas_c3_regularity.md § 5.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** k-Uniform Third Derivative of Localized Variance

### Chain Rule for Regularized Standard Deviation
- **Source:** [13_geometric_gas_c3_regularity.md § 6.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Chain Rule for Regularized Standard Deviation

### Third Derivative Bound for Regularized Standard Deviation
- **Source:** [13_geometric_gas_c3_regularity.md § 6.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Third Derivative Bound for Regularized Standard Deviation

### k-Uniform Third Derivative of Z-Score
- **Source:** [13_geometric_gas_c3_regularity.md § 7.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** k-Uniform Third Derivative of Z-Score

### $C^3$ Regularity of the �-Localized Fitness Potential
- **Source:** [13_geometric_gas_c3_regularity.md § 8.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Regularity of the �-Localized Fitness Potential

### ρ-Scaling of Third Derivative Bound
- **Source:** [13_geometric_gas_c3_regularity.md § 8.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** ρ-Scaling of Third Derivative Bound

### BAOAB Discretization Validity
- **Source:** [13_geometric_gas_c3_regularity.md § 9.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** BAOAB Discretization Validity

### $C^3$ Regularity of Total Lyapunov Function
- **Source:** [13_geometric_gas_c3_regularity.md § 9.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Regularity of Total Lyapunov Function

### $C^3$ Perturbation Structure
- **Source:** [13_geometric_gas_c3_regularity.md § 9.3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Perturbation Structure

### Regularity Hierarchy Complete
- **Source:** [13_geometric_gas_c3_regularity.md § 9.4.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Regularity Hierarchy Complete

### Scaling of Third-Derivative Bound
- **Source:** [13_geometric_gas_c3_regularity.md § 10.1.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Scaling of Third-Derivative Bound

### Time Step Constraint from $C^3$ Regularity
- **Source:** [13_geometric_gas_c3_regularity.md § 10.2.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Time Step Constraint from  Regularity

### Explicit Formula for $K_{V,3}(\rho)$
- **Source:** [13_geometric_gas_c3_regularity.md § 10.3.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Explicit Formula for

### Continuity of Third Derivatives
- **Source:** [13_geometric_gas_c3_regularity.md § 11.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Continuity of Third Derivatives

### Regularized Standard Deviation (Implementation)
- **Source:** [13_geometric_gas_c3_regularity.md § 12.5.](source/2_geometric_gas/13_geometric_gas_c3_regularity)
- **Description:** Regularized Standard Deviation (Implementation)

---

### Source: 14_geometric_gas_cinf_regularity_full.md {#14_geometric_gas_cinf_regularity_full}


### C⁴ Measurement Function
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** C⁴ Measurement Function

### C⁴ Localization Kernel
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** C⁴ Localization Kernel

### C⁴ Rescale Function
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** C⁴ Rescale Function

### C^∞ Regularized Standard Deviation
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** C^∞ Regularized Standard Deviation

### Bounded Measurement Range
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Bounded Measurement Range

### QSD Bounded Density (Regularity Condition R2)
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 3.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** QSD Bounded Density (Regularity Condition R2)

### Fourth Derivative of Localization Weights
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 4.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth Derivative of Localization Weights

### Telescoping Property for Fourth Derivative
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 4.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Telescoping Property for Fourth Derivative

### Fourth Derivative of Localized Mean
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 5.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth Derivative of Localized Mean

### Fourth Derivative of Localized Variance
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 5.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth Derivative of Localized Variance

### Chain Rule for $\sigma'_{\text{reg}}$
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 6.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Chain Rule for

### Fourth Derivative of Z-Score
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 7.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth Derivative of Z-Score

### C⁴ Regularity of Fitness Potential
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 8.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** C⁴ Regularity of Fitness Potential

### Hessian Lipschitz Continuity
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 9.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Hessian Lipschitz Continuity

### Fourth-Order Integrator Compatibility
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 9.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth-Order Integrator Compatibility

### Brascamp-Lieb Inequality (Conditional)
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 9.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Brascamp-Lieb Inequality (Conditional)

### Bakry-Émery Γ₂ Criterion (Conditional)
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 9.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Bakry-Émery Γ₂ Criterion (Conditional)

### Fourth-Derivative Scaling
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 10.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Fourth-Derivative Scaling

### Time Step Constraint (Corrected)
- **Source:** [14_geometric_gas_cinf_regularity_full.md § 10.](source/2_geometric_gas/14_geometric_gas_cinf_regularity_full)
- **Description:** Time Step Constraint (Corrected)

---

### Source: 15_geometric_gas_lsi_proof.md {#15_geometric_gas_lsi_proof}


### Quasi-Stationary Distribution (QSD)
- **Source:** [15_geometric_gas_lsi_proof.md § 3.3.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** Quasi-Stationary Distribution (QSD)

### Log-Sobolev Inequality
- **Source:** [15_geometric_gas_lsi_proof.md § 3.4.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** Log-Sobolev Inequality

### N-Uniform Third Derivative Bound for Fitness (PROVEN)
- **Source:** [15_geometric_gas_lsi_proof.md § 8.2.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** N-Uniform Third Derivative Bound for Fitness (PROVEN)

### N-Uniform Poincaré Inequality for QSD Velocities (CORRECTED PROOF)
- **Source:** [15_geometric_gas_lsi_proof.md § 8.3.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** N-Uniform Poincaré Inequality for QSD Velocities (CORRECTED PROOF)

### N-Uniform Drift Perturbation Bounds
- **Source:** [15_geometric_gas_lsi_proof.md § 8.5.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** N-Uniform Drift Perturbation Bounds

### Verification of Cattiaux-Guillin Hypotheses
- **Source:** [15_geometric_gas_lsi_proof.md § 8.5.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** Verification of Cattiaux-Guillin Hypotheses

### N-Uniform Log-Sobolev Inequality for Geometric Viscous Fluid Model
- **Source:** [15_geometric_gas_lsi_proof.md § 9.1.](source/2_geometric_gas/15_geometric_gas_lsi_proof)
- **Description:** N-Uniform Log-Sobolev Inequality for Geometric Viscous Fluid Model

---

### Source: 16_convergence_mean_field.md {#16_convergence_mean_field}


### Mean-Field Revival Operator (Formal)
- **Source:** [16_convergence_mean_field.md § 1.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Mean-Field Revival Operator (Formal)

### Combined Jump Operator
- **Source:** [16_convergence_mean_field.md § 1.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Combined Jump Operator

### Finite-N LSI Preservation (Proven)
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Finite-N LSI Preservation (Proven)

### Data Processing Inequality (Standard Result)
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Data Processing Inequality (Standard Result)

### Wasserstein Contraction for Proportional Resampling (Conjecture)
- **Source:** [16_convergence_mean_field.md § 3.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Wasserstein Contraction for Proportional Resampling (Conjecture)

### Revival Rate Constraint
- **Source:** [16_convergence_mean_field.md § 4.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Revival Rate Constraint

### Revival Operator is KL-Expansive (VERIFIED)
- **Source:** [16_convergence_mean_field.md § 7.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Revival Operator is KL-Expansive (VERIFIED)

### Joint Jump Operator NOT Unconditionally Contractive (VERIFIED)
- **Source:** [16_convergence_mean_field.md § 7.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Joint Jump Operator NOT Unconditionally Contractive (VERIFIED)

### Stage 0 COMPLETE (VERIFIED)
- **Source:** [16_convergence_mean_field.md § 8.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Stage 0 COMPLETE (VERIFIED)

### Quasi-Stationary Distribution (QSD)
- **Source:** [16_convergence_mean_field.md § 0.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Quasi-Stationary Distribution (QSD)

### Framework Assumptions
- **Source:** [16_convergence_mean_field.md § 1.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Framework Assumptions

### QSD Existence via Nonlinear Fixed-Point
- **Source:** [16_convergence_mean_field.md § 1.4.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** QSD Existence via Nonlinear Fixed-Point

### QSD Stability (Champagnat-Villemonais)
- **Source:** [16_convergence_mean_field.md § Step 3c: QSD Stability](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** QSD Stability (Champagnat-Villemonais)

### Hörmander's Condition
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Hörmander's Condition

### Hypoelliptic Regularity
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Hypoelliptic Regularity

### QSD Smoothness
- **Source:** [16_convergence_mean_field.md § 2.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** QSD Smoothness

### QSD Strict Positivity
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** QSD Strict Positivity

### Irreducibility
- **Source:** [16_convergence_mean_field.md § Step 1: Irreducibility of the Process](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Irreducibility

### Strong Maximum Principle
- **Source:** [16_convergence_mean_field.md § Step 2: Strong Maximum Principle for Irreducible P](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Strong Maximum Principle

### Uniform Velocity Gradient Bound
- **Source:** [16_convergence_mean_field.md § 3.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Uniform Velocity Gradient Bound

### Complete Gradient and Laplacian Bounds
- **Source:** [16_convergence_mean_field.md § 3.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Complete Gradient and Laplacian Bounds

### Drift Condition with Quadratic Lyapunov
- **Source:** [16_convergence_mean_field.md § 4.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Drift Condition with Quadratic Lyapunov

### Exponential Tails for QSD
- **Source:** [16_convergence_mean_field.md § 4.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Exponential Tails for QSD

### KL-Convergence for Mean-Field Euclidean Gas (CORRECTED)
- **Source:** [16_convergence_mean_field.md § 4.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** KL-Convergence for Mean-Field Euclidean Gas (CORRECTED)

### Modified Fisher Information
- **Source:** [16_convergence_mean_field.md § 1.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Modified Fisher Information

### Log-Sobolev Inequality (LSI) for QSD
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Log-Sobolev Inequality (LSI) for QSD

### Explicit LSI Constant
- **Source:** [16_convergence_mean_field.md § 2.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Explicit LSI Constant

### Fisher Information Bound
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Fisher Information Bound

### Kinetic Energy Control
- **Source:** [16_convergence_mean_field.md § 3.2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Kinetic Energy Control

### Entropy $L^1$ Bound
- **Source:** [16_convergence_mean_field.md § 4.2.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Entropy  Bound

### Exponential Convergence (Local)
- **Source:** [16_convergence_mean_field.md § 5.4.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Exponential Convergence (Local)

### Main Result: Explicit Convergence Rate
- **Source:** [16_convergence_mean_field.md § 10.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Main Result: Explicit Convergence Rate

### Mean-Field Convergence Rate (Explicit)
- **Source:** [16_convergence_mean_field.md § 2.1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Mean-Field Convergence Rate (Explicit)

### Optimal Parameter Scaling
- **Source:** [16_convergence_mean_field.md § 2.3.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Optimal Parameter Scaling

### Exponential KL-Convergence in the Mean-Field Limit
- **Source:** [16_convergence_mean_field.md § 1.](source/2_geometric_gas/16_convergence_mean_field)
- **Description:** Exponential KL-Convergence in the Mean-Field Limit

---

### Source: 18_emergent_geometry.md {#18_emergent_geometry}


### Main Theorem (Informal)
- **Source:** [18_emergent_geometry.md § 0.5](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Main Theorem (Informal)

### Adaptive Diffusion Tensor
- **Source:** [18_emergent_geometry.md § 1.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Adaptive Diffusion Tensor

### Why This is a Riemannian Metric
- **Source:** [18_emergent_geometry.md § 1.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Why This is a Riemannian Metric

### Spectral Floor (Standing Assumption)
- **Source:** [18_emergent_geometry.md § 1.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Spectral Floor (Standing Assumption)

### Uniform Ellipticity by Construction
- **Source:** [18_emergent_geometry.md § 1.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Uniform Ellipticity by Construction

### Lipschitz Continuity of Adaptive Diffusion
- **Source:** [18_emergent_geometry.md § 1.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Lipschitz Continuity of Adaptive Diffusion

### Kinetic Operator with Adaptive Diffusion
- **Source:** [18_emergent_geometry.md § 1.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Kinetic Operator with Adaptive Diffusion

### Comparison to Isotropic Case
- **Source:** [18_emergent_geometry.md § 1.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Comparison to Isotropic Case

### Two Equivalent Formulations
- **Source:** [18_emergent_geometry.md § 1.6](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Two Equivalent Formulations

### Invariance Under Coordinate Changes (Refined)
- **Source:** [18_emergent_geometry.md § 1.6](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Invariance Under Coordinate Changes (Refined)

### Coupled Swarm State
- **Source:** [18_emergent_geometry.md § 2.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Coupled Swarm State

### Coupled Lyapunov Function
- **Source:** [18_emergent_geometry.md § 2.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Coupled Lyapunov Function

### Geometric Ergodicity of the Adaptive Gas
- **Source:** [18_emergent_geometry.md § 2.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Geometric Ergodicity of the Adaptive Gas

### Itô Correction Term Bound
- **Source:** [18_emergent_geometry.md § 3.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Itô Correction Term Bound

### Velocity Variance Contraction (Anisotropic)
- **Source:** [18_emergent_geometry.md § 3.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Velocity Variance Contraction (Anisotropic)

### Hypocoercive Norm
- **Source:** [18_emergent_geometry.md § 3.2.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Hypocoercive Norm

### Why Coupling is Essential
- **Source:** [18_emergent_geometry.md § 3.2.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Why Coupling is Essential

### Location Error Contraction (Anisotropic)
- **Source:** [18_emergent_geometry.md § 3.2.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Location Error Contraction (Anisotropic)

### Structural Error Contraction (Anisotropic)
- **Source:** [18_emergent_geometry.md § 3.2.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Structural Error Contraction (Anisotropic)

### Hypocoercive Contraction for Adaptive Gas
- **Source:** [18_emergent_geometry.md § 3.2.5](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Hypocoercive Contraction for Adaptive Gas

### Position Variance Expansion
- **Source:** [18_emergent_geometry.md § 3.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Position Variance Expansion

### Boundary Potential Contraction
- **Source:** [18_emergent_geometry.md § 3.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Boundary Potential Contraction

### Foster-Lyapunov Condition for Adaptive Gas
- **Source:** [18_emergent_geometry.md § 4.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Foster-Lyapunov Condition for Adaptive Gas

### Total Convergence Rate with Full Parameter Dependence
- **Source:** [18_emergent_geometry.md § 5.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Total Convergence Rate with Full Parameter Dependence

### Total Expansion Constant with Full Parameter Dependence
- **Source:** [18_emergent_geometry.md § 5.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Total Expansion Constant with Full Parameter Dependence

### Explicit Convergence Time
- **Source:** [18_emergent_geometry.md § 5.5](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Explicit Convergence Time

### Three Bottleneck Regimes
- **Source:** [18_emergent_geometry.md § 5.6](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Three Bottleneck Regimes

### Regularization Trade-Off
- **Source:** [18_emergent_geometry.md § 5.7](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Regularization Trade-Off

### The Emergent Metric
- **Source:** [18_emergent_geometry.md § 6.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** The Emergent Metric

### Convergence Rate Depends on Metric Ellipticity
- **Source:** [18_emergent_geometry.md § 6.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Convergence Rate Depends on Metric Ellipticity

### Fitness Potential Construction (Algorithmic Specification)
- **Source:** [18_emergent_geometry.md § 9.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Fitness Potential Construction (Algorithmic Specification)

### Explicit Hessian Formula
- **Source:** [18_emergent_geometry.md § 9.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Explicit Hessian Formula

### Emergent Riemannian Metric (Explicit Construction)
- **Source:** [18_emergent_geometry.md § 9.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Emergent Riemannian Metric (Explicit Construction)

### Uniform Ellipticity from Regularization
- **Source:** [18_emergent_geometry.md § 9.4](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Uniform Ellipticity from Regularization

### Emergent Riemannian Manifold
- **Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Emergent Riemannian Manifold

### Geodesics Favor High-Fitness Regions
- **Source:** [18_emergent_geometry.md § 9.5](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Geodesics Favor High-Fitness Regions

### Algorithmic Tunability of the Emergent Geometry
- **Source:** [18_emergent_geometry.md § 9.6.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Algorithmic Tunability of the Emergent Geometry

### Companion Flux Balance at QSD
- **Source:** [18_emergent_geometry.md § 10](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Companion Flux Balance at QSD

### QSD Spatial Marginal is Riemannian Volume Measure
- **Source:** [18_emergent_geometry.md § A.1](source/2_geometric_gas/18_emergent_geometry)
- **Description:** QSD Spatial Marginal is Riemannian Volume Measure

### Fast Velocity Thermalization Justifies Annealed Approximation
- **Source:** [18_emergent_geometry.md § A.2](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Fast Velocity Thermalization Justifies Annealed Approximation

### Continuum Limit via Saddle-Point Approximation
- **Source:** [18_emergent_geometry.md § A.3](source/2_geometric_gas/18_emergent_geometry)
- **Description:** Continuum Limit via Saddle-Point Approximation
