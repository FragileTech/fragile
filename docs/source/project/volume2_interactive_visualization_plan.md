# Interactive experiments for Volume 2, Parts I–IV

(sec-v2-demo-overview)=
## Purpose, scope, and reading map

:::{div} feynman-prose
All 42 experiments in the first four parts execute the Rust Euclidean Gas. Their probabilities, geometry, derivatives, histograms, moments and predictions are computed in Rust from executed configurations and recorded states. The browser initializes the compiled module once per worker and retains it while the reader changes experiments. Every change that starts a new seeded run remains an execution of the same library.

Each card below states the implemented observable and its lecture placement. A controlled component configuration can disable copying through constant fitness, but that choice is part of its executed configuration. Comparisons with an exact numerical transition use that same configuration. An interacting run retains its actual cloning and kinetic contributions.

The experiment catalog and admitted controls are maintained in the shared Rust registry. The browser renders its outputs; it does not provide a separate simulation or probability implementation. The [lecture implementation guide](../../../fractal-gas-web/web/euclidean-gas/lecture/README.md) describes the session interface, and the [validation guide](../../../fractal-gas-web/web/euclidean-gas/lecture/VALIDATION.md) identifies the checks and their scientific meaning.
:::

:::{div} feynman-added
| Part | Experiments | Main measured object |
|---|---|---|
| [I — Algorithms and Foundations](../2_fractal_gas/parts/01_foundations.md) | I-01–I-10 | Stages, donor choices, fitness, cloning and innovations |
| [II — Finite-Particle Convergence](../2_fractal_gas/parts/02_convergence.md) | II-01–II-08 | Conditional increments, empirical transport and survival |
| [III — Mean-Field Limits and Equilibrium](../2_fractal_gas/parts/03_mean_field.md) | III-01–III-08 | Population ledgers, independent-run variation and donor collisions |
| [IV — Entropy, Regularity, and Bounds](../2_fractal_gas/parts/04_entropy_regularity.md) | IV-01–IV-16 | Recorded distributions, conditional derivatives and finite-step predictions |
:::

(sec-v2-demo-part-i)=
## Part I — Algorithms and Foundations: make one update visible

### I-01 — Follow one walker through a complete step

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/01_algorithm_intuition.md), at `Minimal Pseudocode`. Which operation moved this walker?

**Executed experiment.** Recorded stages expose the selected position and velocity coordinate; mechanical budgets identify stage transfers. Advance, inspect the latest committed update and replay the same seeded protocol.

**Controls.** Walkers (`walkers`, default `32`); Objective (`benchmark`, default `sphere`); Selected slot (`walker`, default `0`)
:::

### I-02 — Two donor networks, two different questions

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/01_algorithm_intuition.md), at `sec-companion-selection`. Does the walker measuring diversity have to be the one copied?

**Executed experiment.** The result contains distance companions and source references, the clone plan, fitness, conditional acceptance and realized decisions. Independent distance sampling additionally supplies exact normalized donor rows. Change the two kernel widths separately.

**Controls.** Walkers (`walkers`, default `6`); Distance donor law (`law`, default `independent`); Distance Gaussian width (`width`, default `1`); Cloning Gaussian width (`cloneWidth`, default `0.7`); Distance rounds / donors (`count`, default `1`)
:::

### I-03 — The fitness calculation as an instrument panel

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/01_fragile_gas_framework.md), at `12. Standardization pipeline`. Why can fitness change while a raw reward stays fixed?

**Executed experiment.** Inspect oriented reward, separation, normalized reward and fitness with their recorded standardizers. Change the exponents, positive floor, outlier eligibility and local/global convention and inspect the executed pipeline.

**Controls.** Reward exponent α (`alpha`, default `1`); Diversity exponent β (`beta`, default `1`); Standard deviation floor (`sigma`, default `0.01`); Positive-map floor (`floor`, default `0.01`); Outlier radius (`outlier`, default `1`); Outlier eligible (`exclude`, default `yes`); Statistics (`stats`, default `global`); Local self weight (`self`, default `include`)
:::

### I-04 — Revival, singleton continuation, and extinction

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/01_fragile_gas_framework.md), at `17. The Revival State`. What does fixed population mean when slots lose eligibility?

**Executed experiment.** The eligibility ledger distinguishes the pre-clone population, revivals and final survivors; the final histogram uses eligible walkers. Select four, one or zero initial survivors. Extinction is an explicit recorded terminal outcome, including when no update can commit.

**Controls.** Initial eligible slots (`survivors`, default `4`); Boundary (`boundary`, default `absorbing`); Copy jitter (`jitter`, default `0.05`)
:::

### I-05 — BAOAB under a microscope

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/02_euclidean_gas.md), at `3.1 Euclidean Gas algorithm`. At which positions are the two force evaluations made?

**Executed experiment.** Plot the selected position and velocity through actual recorded substages. The first and final force kicks use their corresponding stage coordinates. Constant fitness disables copying for this operator experiment.

**Controls.** Time step h (`dt`, default `0.04`); Friction γ (`gamma`, default `1`); Thermal parameter T (`temperature`, default `0.4`); Selected slot (`walker`, default `0`)
:::

### I-06 — Compare reward and mechanical motion

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/02_euclidean_gas.md), at `3.4 Swarm distance`. How do separate reward and force choices affect independent gas runs?

**Executed experiment.** Compare recorded mean reward, position and velocity in the first run with endpoint means from independently seeded complete runs. Aligned, shifted and multiwell controls configure actual benchmark/reward choices.

**Controls.** Reward / force preset (`landscape`, default `aligned`); Walkers per run (`walkers`, default `128`); Reward exponent α (`alpha`, default `1`); Diversity exponent β (`beta`, default `0`); Time step (`dt`, default `0.025`)
:::

### I-07 — Conditional copy probabilities and realized events

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/04_single_particle.md), at `5. Conditional Cloning Probability Field`. How do recorded acceptance probabilities compare with actual copy decisions?

**Executed experiment.** Compare selected-donor acceptance probabilities with Bernoulli decisions and the cumulative acceptance-minus-probability residual. Independent donor sampling also exposes its exact normalized kernel rows. A single event need not equal its probability.

**Controls.** Frozen population N (`walkers`, default `5`); Diversity assignment law (`law`, default `independent`); Diversity width (`width`, default `1`); Cloning width (`cloneWidth`, default `1`); Acceptance saturation scale (`saturation`, default `1`)
:::

### I-08 — Track the actual cloning displacement

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/04_single_particle.md), at `6. Post-Cloning Position Distribution`. How do copy and jitter operations move the selected walker?

**Executed experiment.** Read actual positions and velocities through the recorded stages, with the mechanical ledger and clone plan. The jitter and absorbing-boundary controls affect the executed update; the card does not draw a separate mixture sampler.

**Controls.** Voluntary copy jitter σ (`jitter`, default `0.1`); Viable half-width (`boundary`, default `1.1`)
:::

### I-09 — What an inelastic collision really conserves

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md), at `2.2. What the collision conserves`. How can relative motion shrink while pair momentum stays fixed?

**Executed experiment.** The configured mutual-pair restitution acts inside the actual clone transform. Inspect clone energy and event counts, keeping literal replacement, restitution and subsequent kinetics distinct. Pair-level conservation refers to the restitution operation and is not total conservation of the full step.

**Controls.** Restitution a (`restitution`, default `0.5`); First velocity x₁ (`vx`, default `1.5`); First velocity x₂ (`vy`, default `0.5`)
:::

### I-10 — Fitness geometry and executed anisotropic noise

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md), at `sec-latent-fractal-gas-kinetics`. Do the recorded noise factors predict the coordinate second moments?

**Executed experiment.** The actual adaptive metric provider records its Hessian and noise factor. Compare eligible raw innovation coordinate second moments with the factor-product diagonal. This comparison uses the factors evaluated during the executed O stage.

**Controls.** The card uses its declared fixed protocol and seed.
:::

(sec-v2-demo-part-ii)=
## Part II — Finite-Particle Convergence: measure the actual transition

### II-01 — Fitness and copying in a displaced population

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/03_cloning.md), at `sec-cloning-keystone`. How does displacing part of the population change its fitness and copying?

**Executed experiment.** Displace a fraction of initial walkers and inspect the resulting fitness, selected-donor probabilities and realized copies. The current readout measures these local ingredients; it does not measure the complete theorem’s overlap lower bound.

**Controls.** Walkers (`walkers`, default `64`); Requested outer-cluster fraction (`fraction`, default `0.15`); Outer-cluster distance (`separation`, default `2`); Reward exponent (`alpha`, default `1`); Diversity exponent (`beta`, default `0`)
:::

### II-02 — Conditional drift across independent gas continuations

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/03_cloning.md), at `10.3. Positional Variance Contraction`. How much does one population change under independent future random choices?

**Executed experiment.** Replicas start from the same explicit positions and zero velocities with different simulation seeds. The latest stage trace and endpoint means expose the different futures. The across-run endpoint variance describes this finite-horizon ensemble, not a fitted stationary floor.

**Controls.** Clone jitter / kinetic noise amplitude (`jitter`, default `0.05`); Replicates per spread (`replicas`, default `24`); Walkers (`walkers`, default `32`)
:::

### II-03 — Optimal transport of observed coordinate marginals

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/04_wasserstein_contraction.md), at `3.2. Centered Positional Wasserstein Bound`. How far apart are the coordinate distributions of two executed gas runs?

**Executed experiment.** Two executed runs have a controlled initial translation and common seed. Sort their measured coordinate marginals and compute the exact equal-mass one-dimensional quadratic matching cost. The result also keeps initial/final marginal comparisons from the first run.

**Controls.** Equal masses (`walkers`, default `8`); Cloud translation (`translation`, default `0`)
:::

### II-04 — Compare measured transport with an independent coupling

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/04_wasserstein_contraction.md), at `5. From Variance Contraction`. How loose is the independent coupling cost for two measured clouds?

**Executed experiment.** Compare the one-dimensional optimal matching cost with the independent coupling cost formed from the measured means and variances. The latter is an admissible coupling and therefore an upper bound. No full phase-space optimality is claimed.

**Controls.** Walkers (`walkers`, default `16`); Initial translation (`translation`, default `1`)
:::

### II-05 — Friction and noise in executed BAOAB

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/05_kinetic_contraction.md), at `5. Velocity Variance Dissipation`. Does the exact finite-step law predict the measured velocity observable?

**Executed experiment.** Plot velocity mean, empirical variance, mean cos(v), and realized versus conditional O-stage energy. The exact harmonic BAOAB matrix propagates the actual initial population to a Gaussian-convolution expectation for cos(v). Copying is disabled by constant fitness.

**Controls.** Friction γ (`gamma`, default `1`); Temperature / diffusion parameter (`temperature`, default `0.5`); Timestep (`h`, default `0.02`); Independent velocities (`walkers`, default `256`)
:::

### II-06 — Confinement and absorption in the gas

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/05_kinetic_contraction.md), at `7. Boundary Potential Contraction`. How does a displaced population respond to a confining force and absorbing boundary?

**Executed experiment.** Run the confining quadratic dynamics with the configured absorbing box and displaced initialization. Inspect eligibility, revival, final mass and a final coordinate histogram. The view does not estimate the phase-space controllability rank.

**Controls.** Absorbing half-width b (`box`, default `1`); Start as fraction of b (`start`, default `0.8`); BAOAB step (`h`, default `0.1`); Temperature (`temperature`, default `1`); Transition block (`steps`, default `2`)
:::

### II-07 — Measure the finite-step moment generator

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/06_convergence.md), at `sec-convergence-composition`. What drift does the complete update produce in the squared position?

**Executed experiment.** Compute the difference between final and before-step empirical position second moments divided by the actual step duration. Plot this finite-step generator observation together with mean and second moment. An individual increment is not its conditional expectation.

**Controls.** Walkers (`walkers`, default `64`); Time step (`h`, default `0.04`)
:::

### II-08 — Survival and conditional shape in a killed gas

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/06_convergence.md), at `4.2. A complete killed-block criterion`. How do viable population size and its spatial shape evolve together?

**Executed experiment.** The eligibility ledger and final histogram come from the killed, reviving gas. Compare mass and conditional shape without inserting a finite-state killed-chain eigenvector as the gas’s quasi-stationary law.

**Controls.** Walkers (`walkers`, default `64`); Time step (`h`, default `0.04`); Absorbing half-width (`box`, default `1.1`)
:::

(sec-v2-demo-part-iii)=
## Part III — Mean-Field Limits and Equilibrium: distinguish population statistics

### III-01 — The executed population eligibility ledger

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/08_mean_field.md), at `sec-mean-field-population`. How many walkers are eligible before cloning and after the full update?

**Executed experiment.** Start with a selected number of eligible walkers. Record eligibility before cloning, revival counts and eligibility after the full update. Replacement and survival remain separately visible in the same executed population ledger.

**Controls.** Walkers (`walkers`, default `64`); Time step (`h`, default `0.04`); Absorbing half-width (`box`, default `1.1`); Initially eligible walkers (`survivors`, default `16`)
:::

### III-02 — Measure cloning frequency at a finite timestep

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/08_mean_field.md), at `sec-mean-field-reactions`. How does the observed number of copies compare with the sum of their probabilities?

**Executed experiment.** Plot actual copy counts and the sum of ordinary-copy conditional probabilities, excluding forced revival from that comparison. Change timestep and saturation to inspect the finite-step event law. Also inspect the executed clone-energy ledger.

**Controls.** Timeline timestep (`h`, default `0.025`); Acceptance saturation (`saturation`, default `1.0`)
:::

### III-03 — Collective fluctuations across executed runs

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/09_propagation_chaos.md), at `sec-chaos-population-laws`. How do independently seeded population means fluctuate under selection?

**Executed experiment.** Run the chosen independent or shared initialization and selection strength on separate seeds. The first trajectory supplies within-population spread and donor-label collisions; all replicas supply endpoint means and their empirical across-run variance. Change selection to make a separate comparison run.

**Controls.** Observation update (`updates`, default `4`); Initial population (`initial`, default `independent`); Active reward exponent (`selection`, default `1`); Replicas per N and rule (`replicas`, default `8`)
:::

### III-04 — Finite-step generator and moment evolution

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/09_propagation_chaos.md), at `sec-chaos-identification`. Does the actual numerical process have stationary position moments?

**Executed experiment.** With constant fitness, measure increments of the position second moment divided by timestep and plot the position moments. This readout measures the executed finite-step process; zero stationary generator expectation requires a stationary law and averaging.

**Controls.** OU temperature T (`temperature`, default `1`)
:::

### III-05 — Measure fitness variation in the evolving support

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/07_discrete_qsd.md), at `sec-equilibrium-selection`. Does the measured population settle into a narrow fitness range?

**Executed experiment.** Change the actual reward-to-diversity exponent ratio and inspect oriented reward, separation, normalized reward, fitness and population fitness variance. The measured profile is generated by the gas; no spectral equilibrium solver supplies its target.

**Controls.** α / β (`ratio`, default `1`)
:::

### III-06 — Surviving shape of the Brownian gas

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/07_discrete_qsd.md), at `sec-equilibrium-kinetic`. How does the spatial distribution change under diffusion, selection and absorption?

**Executed experiment.** Execute Brownian position updates with selection, revival and the specified absorbing interval in each spatial coordinate. Inspect eligibility and the final surviving coordinate histogram. A sine or source-balanced parabola is not imposed as its equilibrium.

**Controls.** Interval length L (`length`, default `1`); Diffusivity D₀ (`diffusivity`, default `0.5`)
:::

### III-07 — Collective fluctuations with correlated initial states

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/12_qsd_exchangeability_theory.md), at `sec-exchangeability-qsd`. How does shared initial information affect independent runs of the interacting gas?

**Executed experiment.** A controlled shared component modifies the initialized position coordinate within each independent run. Compare within-population variance with separate-run endpoint means. Their different behavior identifies collective fluctuations without treating all walkers as independent samples.

**Controls.** Walkers N (`walkers`, default `64`); Shared-state strength ρ (`correlation`, default `1`); Independent runs (`replicas`, default `8`)
:::

### III-08 — Repeated labels in the actual donor sampler

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/12_qsd_exchangeability_theory.md), at `sec-exchangeability-finite-mixtures`. How often do executed donor selections point to the same source?

**Executed experiment.** Count repeated actual donor indices among the selected prefix of draws. For independent sampling, compute the conditional pair-collision expectation from the recorded normalized donor rows. This expectation follows the sampler’s row-dependent probabilities.

**Controls.** Population labels N (`walkers`, default `64`); Tuple length k (capped at N) (`tuple`, default `8`)
:::

(sec-v2-demo-part-iv)=
## Part IV — Entropy, Regularity, and Bounds: inspect the measured quantities

### IV-01 — Transport of a displaced population

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/10_kl_hypocoercive.md), at `2.2. Entropy and the three Fisher terms`. How does a position displacement affect the later velocity law?

**Executed experiment.** Displace the initial population in the constant-fitness quadratic BAOAB experiment. Inspect velocity moments, mean cos(v) with its exact finite-step expectation, and conditional thermostat energy. The current plots measure those observables rather than entropy or Fisher information.

**Controls.** Friction γ (`gamma`, default `1`); Initial displacement (`displacement`, default `2`); Temperature θ (`theta`, default `1`)
:::

### IV-02 — Measure entropy along an executed trajectory

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/10_kl_hypocoercive.md), at `sec-kl-hypocoercive-full-evolution`. How does a declared spatial histogram change as the gas evolves?

**Executed experiment.** Compute a 48-bin coordinate histogram in the declared window at every recorded step and its discrete binned entropy. Initial and final density curves show which redistribution accompanies the entropy change. This is not relative entropy to a supplied invariant law.

**Controls.** Walkers (`walkers`, default `64`); Time step (`h`, default `0.04`); Histogram half-width (`window`, default `4`)
:::

### IV-03 — Surviving mass and measured conditional shape

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/11_hk_convergence.md), at `sec-hk-metrics`. How do eligibility and position distribution respond to a displaced population?

**Executed experiment.** Displace an actual population inside a configured absorbing box. Display the eligible fraction, eligibility/revival ledger and coordinate histogram. The eligible population supplies the conditional shape; it is not a quasi-stationary eigenvector chosen externally.

**Controls.** Second center (`center`, default `1`); Absorbing half-width (`box`, default `1.1`)
:::

### IV-04 — Density estimates in a declared region

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/11_hk_convergence.md), at `sec-hk-density`. How does the observation window affect the mass represented by the density?

**Executed experiment.** The window control determines the histogram’s coordinate region. The mode control configures an actual absorbing or unbounded run. Inspect retained mass as well as density; the mode name does not select a source dataset or a separate Gaussian sampler.

**Controls.** Boundary (`boundary`, default `unbounded`); Tail window / interior reach (`window`, default `3`)
:::

### IV-05 — Differentiate the recorded normalized fitness

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/14_a_geometric_gas_c3_regularity.md), at `sec-normalized-measurement-calculus`. Which conditional derivative comes from the executed normalization?

**Executed experiment.** Differentiate the actual conditional fitness with recorded donors and other measured rows fixed. Compare its direct scalar evaluation with a Taylor polynomial, and its first derivative with an independent central difference. Local normalization weights remain part of the differentiation.

**Controls.** Localization width ρ (`rho`, default `0.7`); Standard-deviation floor (`sigma`, default `0.15`); Distance floor δ (`delta`, default `0.1`); Walkers (`N`, default `4`)
:::

### IV-06 — Local normalization in dense and sparse gas populations

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/14_a_geometric_gas_c3_regularity.md), at `sec-c3-counts-scaling`. How many effective neighbors enter the local normalization?

**Executed experiment.** Generate clustered or dispersed initial positions in Rust, execute the local-statistics gas and inspect effective neighbor counts from normalized locality weights. The result also exposes the actual fitness pipeline. Effective count is the inverse sum of squared normalized weights.

**Controls.** Walkers (`N`, default `64`); Localization width ρ (`rho`, default `0.7`); Cloud geometry (`geometry`, default `clustered`)
:::

### IV-07 — Taylor coefficients of a recorded fitness field

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full.md), at `sec-gg-cinf-regularity`. How accurately do conditional fitness derivatives predict a local displacement?

**Executed experiment.** Fit no polynomial coefficients: obtain them by automatic differentiation of the recorded conditional field. Compare the resulting Taylor polynomial with direct field evaluations over the selected radius. Order and radius control approximation of this fixed conditional field.

**Controls.** Taylor order (`order`, default `6`); Expansion radius (`radius`, default `0.2`); Standard-deviation floor (`sigma`, default `0.15`)
:::

### IV-08 — Conditional acceptance in a localized gas

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full.md), at `sec-cinf-companion-laws`. How do local fitness and donor width affect actual copying?

**Executed experiment.** Run localized standardization and donor sampling, then inspect fitness, conditional acceptance, actual copying and the accumulated acceptance residual. The view does not enumerate a separate greedy assignment model or differentiate an averaged update.

**Controls.** Companion width (`width`, default `0.7`); Localization width (`rho`, default `0.7`)
:::

### IV-09 — Empirical density in a nonconvex reward landscape

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/15_kl_convergence.md), at `2.3. Bounded perturbations`. What spatial distribution does the actual Rastrigin gas produce?

**Executed experiment.** Execute the actual Rastrigin gas. Plot its initial/final coordinate histogram and binned entropy in the declared window. The resulting density includes cloning and kinetics and is not generated by Metropolis sampling or prescribed as Gibbs.

**Controls.** Temperature θ (`theta`, default `1`)
:::

### IV-10 — Compare the numerical BAOAB velocity law

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/15_kl_convergence.md), at `sec-fg-kl-conv-discrete`. Does the finite-step harmonic prediction match an executed run?

**Executed experiment.** Execute the harmonic constant-fitness BAOAB component configuration. Compare mean cos(v) against the exact numerical transition prediction and inspect velocity moments and thermostat energy. Its finite-step observable is not a plotted Gaussian KL distance.

**Controls.** Timestep h (`h`, default `0.08`); Friction γ (`gamma`, default `1`); Temperature θ (`theta`, default `1`)
:::

### IV-11 — Fitness Hessian and adaptive innovation covariance

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/17_geometric_gas.md), at `sec-gg-uniform-ellipticity`. Which directions receive the largest innovation variance?

**Executed experiment.** Inspect field evaluations from the actual adaptive metric provider and raw innovation second moments versus their factor-product prediction. Metric shift changes the executed metric. Eligibility masks exclude noise rows that did not receive the operation.

**Controls.** Spectral shift ε (`shift`, default `1`)
:::

### IV-12 — Inspect viscous alignment inside the full update

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/17_geometric_gas.md), at `3.1. Exact component estimates`. Which recorded stages change velocity when row-normalized viscosity acts?

**Executed experiment.** Initialize separated groups with opposing velocities and execute row-normalized viscosity inside the full kinetic force. Read actual stage coordinates and mechanical budgets. Other forces and cloning remain in this experiment; the weighted invariant of an isolated frozen graph is not a total-run conservation law.

**Controls.** Neighbor width (`width`, default `0.8`); Viscosity ν (`viscosity`, default `1`); Gap between clusters (`gap`, default `1.5`)
:::

### IV-13 — Measure bounded observables at a fixed physical duration

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/13_quantitative_error_bounds.md), at `sec-quantitative-total-error`. How does cos(v) behave in the actual finite-step gas?

**Executed experiment.** Run to the declared physical horizon and measure velocity moments, cos(v) and conditional thermostat energy. Compare cos(v) with the exact configured finite-step expectation. This is a direct bounded-observable experiment, not an assumed sum of unrelated error formulas.

**Controls.** Walkers (`N`, default `64`); Requested timestep h (`h`, default `0.5`); Physical duration (`T`, default `15`)
:::

### IV-14 — Compare actual timestep refinements

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/convergence_program/13_quantitative_error_bounds.md), at `sec-quantitative-local-errors`. How does the endpoint observable change under matched-duration timestep refinement?

**Executed experiment.** Execute timestep 0.2, 0.1 and 0.05 to the same requested duration. Plot endpoint mean cos(v) beside the exact expectation for each numerical transition. Shared seeds do not supply Brownian-bridge coupling, so sampling fluctuations remain in differences.

**Controls.** Fixed physical duration (`T`, default `5`); Friction γ (`gamma`, default `1`)
:::

### IV-15 — Companion reach and cloning saturation in an executed gas

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/03_parameter_constraints.md), at `sec-master-constraints`. How do population size and saturation change the conditional copy decisions?

**Executed experiment.** Change population size and the clone saturation denominator. Inspect executed conditional probabilities, realized copies, fitness and the acceptance residual; independent donor rows supply the measured selection law. No separate feasibility diagram supplies the data.

**Controls.** Alive count (`N`, default `64`); Cloning saturation denominator (`cap`, default `1`)
:::

### IV-16 — Independent seeded gas experiments

:::{div} feynman-prose
**Where and why.** [Lecture chapter](../2_fractal_gas/1_the_algorithm/03_parameter_constraints.md), at `sec-parameter-selection`. How do endpoint population means vary across complete seeded runs?

**Executed experiment.** Execute independent seeded replicas with the selected configuration. Plot the first run’s density and binned entropy plus every run’s endpoint mean and empirical across-run variance. The result records run seeds and completed steps; this variance is not a confidence interval.

**Controls.** Base walkers (`N`, default `64`); Timestep h (`h`, default `0.04`); Friction γ (`gamma`, default `1`); Independent replicas per group (`replicas`, default `4`)
:::

(sec-v2-demo-runtime)=
## Rust sessions and browser delivery

:::{div} feynman-prose
`LectureRequest` contains `id`, `seed`, `parameters` and a requested step budget. The registry supplies declared control defaults and rejects unknown controls or values outside their admitted ranges. The protocol resolves the actual run configurations, independent seeds and horizons. In Parts I–IV, duration or update controls can determine the protocol horizon; the resulting completed and required steps are returned for every run.

`LectureSession` creates the gas, applies the explicit initialization protocol in Rust, starts bounded recording and advances complete updates. The compiled `LectureExperiment` exposes creation, advancement, snapshots, evidence, checkpoints and restoration. `lecture_analyze` recomputes results from validated evidence containing the request, configurations and actual archives. Imported evidence must include the run data needed by the calculation.

Scientific fields come from the recorded stage used by the observable. Cloning decisions refer to their actual pre-clone fitness and donor realization; thermostat moments use the corresponding input and noise factor; derivative queries freeze their declared recorded context. The analyzer never draws a replacement trajectory while describing it as the recorded one.

The browser worker retains its initialized CPU/WASM module across resets. The lecture adapter requests eight updates per advance and renders Rust-provided arrays. Initialization collects the first bounded batch, and the displayed completed-step counts expose that work. Plot animation and scene projection do not change the scientific time or calculation. A closed experiment releases its owned session.

Open `<site-prefix>/euclidean-gas/lecture.html?demo=I-01`; append `&embed=1` for the compact lecture display. The deployment prefix is part of the published URL. Static posters and manifests are generated from these executed lecture sessions. A documentation-only build can use the committed generated assets.
:::

(sec-v2-demo-delivery)=
## Measurement contracts and publication checks

:::{div} feynman-prose
Every result identifies its request, executed configuration, seed, run counts, stage-qualified records and numerical readouts. A sampling statistic must retain its normalization. Histograms use a declared window and binning; their missing mass is distinct from extinction. Empirical population spread uses the measured interacting population and is not assigned an independent-walker standard error.

Independent seeded runs provide the sampling unit for ensemble variation. An across-run variance is a descriptive statistic, not automatically a standard error. Shared seeds between timestep refinements are not a constructed path coupling. Exact finite-step expectations, realized increments, tensor identities and fitted predictions retain distinct labels.

The native checks exercise all early experiment analyzers on actual archives, source identity, exact conditional derivatives, harmonic predictions and matched-duration refinements. The compiled and browser suites check the shared delivery path, controls, evidence recomputation and lecture embedding. Each check must report what it actually established. A successful scene render does not establish a convergence rate or stationary law.

Run the registered native experiment from `algorithmic-gas/` with the command below. The result contains a snapshot and its evidence. Reanalysis recomputes the scientific result from that evidence; stress cases enumerate declared control endpoints, categories and pairwise combinations.
:::

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- I-01 7 96 --output /tmp/I-01.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- analyze /tmp/I-01.json --output /tmp/I-01-reanalysis.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- stress I-01 0 10 --output /tmp/I-01-stress.json
```

:::{div} feynman-prose
The teaching sequence stays attached to the executed object. First inspect the operation, then name the statistic, then identify the prediction that applies to it. Finally change a parameter and repeat. This lets the reader discover both agreement and unresolved behavior without substituting a simpler process for the one being studied.
:::
