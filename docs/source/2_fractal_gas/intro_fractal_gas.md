---
title: "The Fractal Gas"
subtitle: "Interacting Particles, Convergence, and Emergent Geometry"
author: "Guillem Duran-Ballester and Sergio Hernández Cerezo"
---

(sec-fractal-gas-combined-intro)=
# The Fractal Gas

**Volume II: Interacting Particles, Convergence, and Emergent Geometry**

by *Guillem Duran-Ballester and Sergio Hernández Cerezo*

:::{div} feynman-prose
A Fractal Gas distributes a search across a population of interacting walkers.
Each walker moves, measures its surroundings, and compares itself with selected
companions. Cloning reallocates walkers toward higher-fitness states; noise and
motion let them explore further. The mathematical problem is to understand the
population produced by these repeated operations.

This volume begins with the update rule and develops its probability theory:
operator estimates, finite-particle convergence, mean-field limits, and entropy
bounds. It then studies the geometric record of a run, called the **Fractal Set**,
and the field models constructed from that record. Computational chapters explain
how to measure the resulting observables and compare simulations.

Keep three questions separate as you read. What happens when a fixed swarm runs
for a long time? What happens when its population increases? What happens when a
discrete geometric construction is refined toward a continuum? Each question has
its own hypotheses and its own notion of convergence.
:::

(sec-fg-combined-executive-summary)=
(sec-fg-combined-core-mechanisms)=
## The algorithm and its variants

:::{div} feynman-prose
Imagine watching a cloud of walkers on a landscape. Motion changes their
positions. Cloning changes where the population spends its computational effort:
a walker can be replaced by a perturbed copy of a companion. The population
therefore changes through both transport and selection.

Reward and diversity enter the fitness calculation. Reward describes how well a
walker performs the task; diversity measures separation according to the chosen
comparison rule. Their normalization and relative weights matter. A diversity
term can oppose concentration, but its presence alone does not establish a lower
bound on the population's spread.

The common sequence is:

1. **Select companions.** Sample other walkers according to the specified
   interaction rule.
2. **Measure and compute fitness.** Combine regularized reward and diversity
   measurements.
3. **Clone and revive.** Apply the replacement rule, including its treatment of
   walkers that have left the alive region.
4. **Advance the kinetic dynamics.** Apply motion, friction, noise, and any
   interaction forces included in the chosen model.

The order of these operations, the noise law, and the boundary rule define the
transition kernel. They are part of the mathematical model.
:::

The {doc}`algorithm introduction <1_the_algorithm/01_algorithm_intuition>` gives
the operational description. The
{doc}`general framework <convergence_program/01_fragile_gas_framework>` specifies
the operators and their assumptions. The
{doc}`Euclidean model <convergence_program/02_euclidean_gas>` provides the main
setting for the convergence analysis, and the
{doc}`single-particle chapter <convergence_program/04_single_particle>` isolates
the underlying kinetic behavior. The
{doc}`latent-space model <1_the_algorithm/02_fractal_gas_latent>` introduces the
geometric choices needed when positions and velocities live on a manifold.

:::{note}
:class: feynman-added

An estimate for one variant applies to another when the latter satisfies the
estimate's hypotheses. For example, momentum conservation during cloning depends
on the specified velocity update, and revival depends on the replacement rule
and the presence of a surviving companion. These properties must be checked for
the actual transition kernel.
:::

(sec-fg-combined-proof-strategy)=
(sec-fg-combined-skeptical)=
## What convergence means here

:::{div} feynman-prose
A swarm that settles into a stable statistical pattern can still lose every
walker through an extinction event. To describe the runs that remain alive, we
condition on survival. This is why quasi-stationary distributions appear in the
analysis. A model with revival or a prescribed source can instead have a
stationary balance between loss and replacement. Its normalized alive profile
and its alive fraction are determined together; that source balance is distinct
from conditioning a killed process on survival.

There is another distinction to keep in view. The distribution of the entire
swarm contains correlations between walkers. The distribution of one selected
walker contains less information. A smooth density used in a mean-field equation
is a third object, obtained only after justifying a limit. Confusing these objects
can turn an estimate about one particle into an unsupported claim about the
whole population.
:::

:::{prf:definition} Finite-particle quasi-stationarity
:label: def-fg-intro-finite-particle-qsd

Let $S_k$ be the state of the $N$-walker swarm after $k$ updates, let
$\tau_{\dagger}$ be its extinction time, and let $\nu_N$ be a probability measure
on surviving swarm configurations with
$\mathbb{P}_{\nu_N}(\tau_{\dagger}>k)>0$ for every integer $k\geq 0$.
The measure $\nu_N$ is a
**quasi-stationary distribution** (QSD) if, when $S_0$ has law $\nu_N$,

$$
\mathbb{P}_{\nu_N}\!\left(S_k\in A\mid \tau_{\dagger}>k\right)
=\nu_N(A)
$$

for every measurable set $A$ and every integer $k\geq 0$.
:::

:::{prf:remark} Proved results and their regimes
:label: rem-fg-intro-proved-results

The volume develops the following complete analytical arguments.

1. **Composition and finite-particle relaxation.** The comparison-matrix theorem
   {prf:ref}`thm-synergistic-rate-derivation` constructs positive Lyapunov weights
   when the nonnegative full-step comparison matrix has spectral radius below
   one. The conservative Harris argument and the killed-block QSD argument are
   proved separately in {prf:ref}`thm-convergence-conservative-harris` and
   {prf:ref}`thm-main-convergence`. Their full-kernel hypotheses give existence,
   uniqueness, and geometric relaxation to the appropriate invariant law or QSD.

2. **Positive mean-field evolution and a stationary limit.** For the specified
   mass-conserving gain-loss model, bounded loss rates and a globally Lipschitz
   reaction give a unique global positive mild solution in
   {prf:ref}`thm-chaos-mild-wellposedness`. If the kinetic semigroup contracts
   zero-mass differences by $Ke^{-at}$ and the reaction Lipschitz constant is
   $L_{\mathcal R}$, the regime $a-KL_{\mathcal R}>0$ gives a unique stationary
   density and global attraction at rate $a-KL_{\mathcal R}$ in
   {prf:ref}`thm-uniqueness-uniqueness-stationary-solution`. The finite-time
   consistency, tightness, and stationary-chaos proofs then identify particle
   limits under their stated concentration or global-attraction hypotheses.

3. **Population-uniform logarithmic Sobolev inequalities.**
   {prf:ref}`cor-n-uniform-lsi` proves four routes for specified continuous joint
   laws: tensorized kinetic references, uniformly bounded whole-law density
   tilts, uniform joint curvature, and contractive additive-noise invariant
   flows. These control the full position–velocity gradient. A law with discrete
   alive/dead strata also requires its status entropy form. The criteria apply
   to an actual QSD when that law meets the stated structural condition.

4. **Hypocoercive entropy convergence.** The complete kinetic commutator and
   modified-Fisher calculation proves {prf:ref}`thm-villani-hypocoercivity`,
   including nonconvex product references with the required LSI and bounded
   Hessian. The exact cloning, killing-normalization, and boundary identities
   feed {prf:ref}`thm-kl-convergence-euclidean`: an actual-law LSI constant $C_N$
   and full dissipation bound $\dot\Phi_G\leq-\delta_N I$ give the rate
   $\delta_N/(C_N/2+g_{+,N})$. Uniform input constants give a uniform rate.

5. **All-orders regularity of the actual companion laws.** Normalized derivative
   calculus controls sampled fitness, fitness formed from expected measurements,
   and expected sampled fitness as separate objects. For bounded pair distances,
   positive scales and regularizers, and uniform analytic reward bounds,
   {prf:ref}`thm-main-complete-cinf-geometric-gas-full` proves block-derivative
   bounds $C B^n n!$ independent of population size. The actual sequential greedy
   pairing law has its own full-history proof in
   {prf:ref}`lem-greedy-ideal-equivalence`; it is not replaced by an idealized
   matching law. Smooth inputs give $C^\infty$ on the specified strata without
   automatically giving analytic bounds.

6. **Observable errors and Gaussian boundary limits.** A uniform total entropy
   $D_{\mathrm{KL}}(\pi_N\Vert\rho^{\otimes N})$ gives $O(N^{-1})$ mean-squared
   error for bounded empirical observables in
   {prf:ref}`thm-mixing-variance-corrected`; full-gradient LSI gives corresponding
   Lipschitz variance bounds. The Gaussian perimeter and sampling proofs
   {prf:ref}`thm-gamma-convergence` and {prf:ref}`thm-ig-cut-scaling` recover
   $N^{(d-1)/d}$ raw cut scaling at range $\varepsilon\propto N^{-1/d}$ for
   independent samples in dimension $d>1$, under their geometric hypotheses.
   {prf:ref}`cor-holo-dependent-cut-limit` transfers the area limit to dependent
   laws when total relative entropy is $o(\log N)$. Identifying a genealogical
   count with that boundary uses the separate boundary-cell correspondence.
:::

:::{div} feynman-prose
These results connect because each supplies a concrete input to another proof.
For example, a uniform joint entropy bound controls empirical observables and
also transfers the Gaussian cut limit to a dependent population. The stationary
mean-field theorem identifies the limiting density when its contraction and
consistency conditions hold. The conditions specify where the results apply;
they are part of the mathematical conclusions, not a substitute for their proofs.

A physical interpretation adds an identification of variables and observables.
The geometric chapters prove their area, curvature, and variation formulas, and
state separately the additional conditions needed to interpret a cut as quantum
entropy or a constructed field as the continuum limit of a sampled metric.
:::

(sec-fg-combined-volume-structure)=
## The structure of the volume

:::{div} feynman-prose
The chapter order follows the objects being studied. First define the swarm,
then estimate its motion, then consider large populations. The geometric and
field chapters come after these analytic foundations so that their additional
assumptions can be stated precisely. The computational chapters provide a place
to examine what a finite simulation actually measures.
:::

### Part I: Algorithms and foundations

{doc}`Part I <parts/01_foundations>` contains the algorithm intuition, general
framework, Euclidean model, single-particle analysis, and latent-space model.
Read it to identify the state space, the update order, and the assumptions attached
to each variant.

### Part II: Finite-particle convergence

{doc}`Part II <parts/02_convergence>` develops the
{doc}`cloning estimates <convergence_program/03_cloning>`,
{doc}`Wasserstein control <convergence_program/04_wasserstein_contraction>`,
{doc}`kinetic estimates <convergence_program/05_kinetic_contraction>`, and
{doc}`composed convergence analysis <convergence_program/06_convergence>`.
The central task is to combine component inequalities into a statement about the
full swarm transition.

### Part III: Mean-field limits and quasi-stationarity

{doc}`Part III <parts/03_mean_field>` develops the
{doc}`mean-field equation <convergence_program/08_mean_field>`,
{doc}`propagation-of-chaos analysis <convergence_program/09_propagation_chaos>`,
{doc}`discrete QSD structure <convergence_program/07_discrete_qsd>`, and
{doc}`exchangeability theory <convergence_program/12_qsd_exchangeability_theory>`.
It proves positive mean-field well-posedness, stationary existence and attraction
in the stated contractive regime, and the consistency and concentration arguments
for stationary particle limits. The equilibrium-profile chapter also turns the
residual of a proposed stationary density into a quantitative error bound.

### Part IV: Entropy, regularity, and quantitative bounds

{doc}`Part IV <parts/04_entropy_regularity>` studies
{doc}`hypocoercive entropy estimates <convergence_program/10_kl_hypocoercive>`,
{doc}`Hellinger–Kantorovich convergence <convergence_program/11_hk_convergence>`,
{doc}`third-order regularity <convergence_program/14_a_geometric_gas_c3_regularity>`,
{doc}`smooth regularity <convergence_program/14_b_geometric_gas_cinf_regularity_full>`,
and {doc}`LSI-based entropy estimates <convergence_program/15_kl_convergence>`.
It also contains the {doc}`geometric gas <convergence_program/17_geometric_gas>`,
{doc}`finite-population error bounds <convergence_program/13_quantitative_error_bounds>`,
and {doc}`parameter constraints <1_the_algorithm/03_parameter_constraints>`.
The complete LSI and kinetic hypocoercivity proofs supply explicit sufficient
regimes. The regularity chapters preserve probability derivatives for each actual
companion law, and the error chapter separates population bias, fluctuations,
time-step error, and transient mixing. Their constants retain the dependence on
population size, dimension, and regularization scales.

### Part V: The Fractal Set and continuum geometry

{doc}`Part V <parts/05_continuum>` starts with the
{doc}`Fractal Set <2_fractal_set/01_fractal_set>`: the record of walker histories,
cloning events, and interactions. It develops
{doc}`emergent geometry <3_fitness_manifold/01_emergent_geometry>` and
{doc}`scutoid constructions <3_fitness_manifold/02_scutoid_spacetime>`, then examines
{doc}`continuum conditions <convergence_program/16_continuum_discharge>` and
{doc}`causal-set constructions <2_fractal_set/02_causal_set_theory>`.
The existence of a discrete record and convergence to a smooth geometric model
are separate results.

### Part VI: Fields and emergent physics

{doc}`Part VI <parts/06_fields>` develops the proposed field descriptions:
{doc}`lattice QFT <2_fractal_set/03_lattice_qft>`,
{doc}`Standard Model correspondences <2_fractal_set/04_standard_model>`,
{doc}`Yang–Mills and Noether analysis <2_fractal_set/05_yang_mills_noether>`,
and the {doc}`twistor formulation <2_fractal_set/08_twistor_formulation>`.
The geometric developments continue through
{doc}`curvature <3_fitness_manifold/03_curvature_gravity>`,
{doc}`field equations <3_fitness_manifold/04_field_equations>`,
{doc}`graph cuts and boundary thermodynamics <3_fitness_manifold/05_holography>`, and
{doc}`cosmology <3_fitness_manifold/06_cosmology>`.
Read each physical correspondence together with its hypotheses and its stated
mathematical status.

### Part VII: Computation and experiments

{doc}`Part VII <parts/07_experiments>` connects definitions to measurements through
{doc}`computational proxies <3_fitness_manifold/07_computational_proxies>`,
{doc}`Voronoi Wilson loops <3_fitness_manifold/08_voronoi_wilson_loops>`, and
{doc}`empirical studies <2_fractal_set/06_empirical_validation>`.
The {doc}`calibration notebook <2_fractal_set/08_qft_calibration_notebook>`,
{doc}`calibration report <2_fractal_set/07_qft_calibration_report>`, and
{doc}`calibration guide <2_fractal_set/09_qft_calibration>` present the computational
workflow and its interpretation. Calibration choices belong to the specification
of a numerical comparison.

### Reference material

The {doc}`reference section <parts/08_reference>` contains the
{doc}`FAQ <reference/faq>` and {doc}`literature guide <reference/literature>`.
The analytic arguments appear in the chapters where their results are developed.

(sec-fg-combined-novel-vs-repackaged)=
## How the arguments fit together

:::{div} feynman-prose
The first estimates concern individual operations. Cloning changes positions and
correlations; kinetic motion changes velocities and transports the population.
An estimate for one operation can contain a term that the other operation must
control. This is the reason for studying their composition explicitly.

Beyond that composition, the argument branches. The positive mean-field proof
controls the nonlinear evolution. Joint-law entropy and concentration estimates
control fluctuations and stationary limits. The regularity and perimeter proofs
then supply specific geometric limits. The diagram shows how those completed
arguments enter the subsequent constructions.
:::

::::{div} feynman-added
```{mermaid}
flowchart TD
    A["I: Define the transition kernel"] --> B["II: Cloning and kinetic estimates"]
    B --> C["II: Composed drift and full-kernel QSD criteria"]
    C --> D["III: Positive mean-field flow and stationary chaos"]
    A --> E["IV: Joint-law LSI, regularity, and entropy proofs"]
    E --> D
    A --> F["V: Construct the Fractal Set"]
    D --> G["V–VI: Sampling and continuum limits"]
    E --> G
    F --> G
    G --> H["VI: Fields with stated observable identifications"]
    F --> I["VII: Discrete observables and experiments"]
    H --> I
```
::::

The arrows connect proof inputs; the hypotheses of individual theorems give the
precise implications. In particular:

- A Lyapunov drift estimate supplies moment control; a convergence theorem also
  needs the appropriate accessibility, mixing, and survival estimates.
- Uniform moment bounds support compactness as $N$ grows. Identifying a
  deterministic mean-field limit additionally requires control of correlations
  and passage through the nonlinear interaction terms.
- A rate for a fixed observable, a marginal law, and an empirical measure in
  Wasserstein distance are different estimates. Population-size rates must be
  quoted with their metric, moment assumptions, and dimension dependence.
- A continuum identification requires the complete set of geometric and scaling
  hypotheses used in its proof. The preceding probabilistic estimates establish
  only the inputs that their statements explicitly cover.

(sec-fg-combined-quick-navigation)=
(sec-fg-combined-volume-connections)=
## Reading guide

:::{div} feynman-prose
For a first pass, read the algorithm introduction and the Euclidean model before
following the convergence chapters. Keep one question beside each theorem:
which part of the update rule does this hypothesis control? That question makes
the constants easier to interpret.

The probabilistic development starts from the definitions in this volume. The
latent-space variant connects to the learned representations of Volume I;
readers working in Euclidean space can begin directly with the Euclidean model.
:::

::::{div} feynman-added
| Your aim | Suggested route | What to track |
|---|---|---|
| Implement a swarm | Part I, then parameter constraints in Part IV | Update order, boundary rules, regularization, and noise |
| Follow the finite-particle proofs | Foundations in Part I, then Part II | The full transition kernel and the hypotheses used in composition |
| Understand the mean-field model | Parts II–III, then finite-population bounds in Part IV | Marginals, correlations, uniformity in time, and the order of limits |
| Study geometric observables | The Fractal Set in Part V, then Part VII | Which quantities are defined on a finite record |
| Study continuum and field claims | Relevant estimates in Parts III–IV, then Parts V–VI | Geometric assumptions, scaling regimes, and conditional statements |
| Assess numerical evidence | Part VII and the reference material | Estimators, sampling dependence, uncertainty, and calibration choices |
::::

(sec-fg-combined-historical-context)=
(sec-fg-combined-empirical-theoretical-bridge)=
## Historical context and empirical interpretation

The historical starting points are *General Algorithmic Search* (2017)
{cite}`hernandez2017gas` and *Fractal AI: A fragile theory of intelligence* (2018)
{cite}`hernandez2018fractal`. The {doc}`literature guide <reference/literature>`
collects these works and subsequent studies. This volume develops explicit
particle models and their mathematical analysis; the transition rule and
hypotheses of each chapter determine which results apply to an implementation.

:::{div} feynman-prose
A successful run gives information about the configuration that was run. To
compare it with a theorem, identify the same observable on both sides: a
population variance, a conditional distribution, a correlation function, or a
geometric quantity. Then check the sampling procedure and the parameter regime.

The same care applies to calibration. If a measured quantity fixes the unit of
mass or length, that choice is an input to the comparison. Predictions concern
the other quantities computed after that input is fixed. The computational
chapters provide the definitions and records needed to inspect these comparisons.
:::

(sec-fg-combined-references)=
## References

```{bibliography}
:filter: docname in docnames
```
