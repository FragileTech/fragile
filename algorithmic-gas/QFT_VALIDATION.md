# Algorithm-derived fields: experiments and validation

All 128 Volume II demos execute the Rust Euclidean Gas through the shared experiment registry. Their observations come from executed trajectories, paired runs, or complete-checkpoint continuations. Rust computes the scientific readouts and predictions; the browser renders those results. Part VI contains 66 registered experiments, each with a [machine-readable measurement contract](crates/algorithmic-gas/src/physics/partvi_contracts.json).

The state used by a full transition includes the population, eligibility, donor memory, provider configuration and the context required by its readouts. Companion sampling, historical rescoring, literal clone writes, viscous kicks, BAOAB noise, clipping and boundaries all belong to that transition. Component experiments identify their configured restrictions explicitly.

## Execution and evidence

The [Rust registry](crates/algorithmic-gas/src/lecture_experiments.json) supplies IDs, controls and defaults. `LectureSession` advances the actual populations in bounded batches and retains the resolved configurations. Ordinary measurements use archived transitions. VI-15, VI-30 and VI-31 use paired complete runs for parity, translation and local intervention. VI-19, VI-22 and VI-45 use independent complete-checkpoint continuations for source response, stochastic balance and material metric evolution.

A run result contains its request, configurations, archives and computed snapshot. Reanalysis validates these inputs and recomputes the measurement; continuation experiments retain the checkpoint information needed by their protocol. Scientific outcomes such as extinction, missing geometric support and inconclusive fits are distinguished from configuration and execution errors. Reference calculations used as test oracles do not generate the lecture observations.

```sh
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  VI-18 7 96 --output /tmp/lecture-evidence.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  analyze /tmp/lecture-evidence.json --output /tmp/lecture-analysis.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  stress VI-18 0 8 --output /tmp/lecture-stress.json
```

`gas-physics` and `algorithmic-gas-qft` expose the same commands. `run REQUEST.json` accepts a complete registered request. Stress output records each case, its configuration and its completion or error; a failed case produces a failing process status.

## Exact identities and predictive claims

For a complete-state transition kernel $K$, define

$$D_F=KF-F,\qquad Q_{FG}=K(FG)-(KF)(KG).$$

For the actual state law $\mu_n$,

$$\mu_{n+1}F=\mu_nF+\mu_nD_F,$$

$$\operatorname{Cov}_{\mu_{n+1}}(F,G)
=\operatorname{Cov}_{\mu_n}(F+D_F,G+D_G)+\mu_nQ_{FG}.$$

These are conditional-expectation identities. A reduced field model becomes predictive only when it estimates the relevant conditional quantities accurately on separate trajectories or complete-state continuations. A descriptor can discard donor and population memory, so the fitted one-step descriptor channel does not automatically compose into the correct multi-step evolution.

A full-step increment is the sum of its recorded stage increments. Its variance includes cross terms between stages and walkers. Cloning, common donors, killing and movement of a material observation point contribute to these measured increments. The stated normalization and cemetery extension are part of each observable.

### Explicit field equations of the executed algorithm

The [field-equations chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md) derives the finite-step transition by composing the actual donor laws, historical rescoring, clone gates, restitution, BAOAB primitives, boundaries and history admission. Its complete marked empirical field retains every row and its source identity; spatial moments inherit explicit source and flux equations from that transition.

For a primitive row change $(x,v,a)\mapsto(y,w,b)$, with eligibility marks $a,b$, its density and momentum contributions satisfy

$$\Delta\rho=(b-a)\delta_x-\nabla\!\cdot\!\left[b(y-x)\int_0^1\delta_{x+t(y-x)}dt\right],$$

$$\Delta j=(bw-av)\delta_x-\nabla\!\cdot\!\left[bw\otimes(y-x)\int_0^1\delta_{x+t(y-x)}dt\right].$$

Summing over rows with normalization $1/N$ and over executed stages gives the full field equations. The velocity second moment supplies an anisotropic stress. Clone transfers, row-normalized viscosity, thermal injection and eligibility changes supply their computed momentum sources. The metric is the configured spectral function of the differentiated conditional fitness; its next-stage evolution includes changing population statistics, immutable donor context and the next A1 query position. Projection onto fewer fields produces the derived transient memory recurrence, without a stationarity assumption.

VI-51 evaluates these laws as Fourier fields for density, momentum, stress, energy or a phase-space characteristic. Rust integrates acceptance gates conditional on the actual donor candidates and rescored fitness, resolves exact historical sources, and computes conditional thermostat means and covariances for the executed Gaussian or standardized-uniform innovation law. Independent kick/transport reconstruction tests the deterministic primitives; the full stage sum retains boundary and eligibility sources.

The [field-equation regression](crates/benchmarks/tests/algorithmic_field_equations.rs) uses 48 independent eight-step gas trajectories per innovation law, with dense correlated noise, source shifts, both viscosity normalizations and historical cloning. The successive stages of a trajectory remain dependent.

| Check | Result |
|---|---:|
| Independent literal-copy and A/B residuals | below $2\times10^{-13}$ |
| Recorded field-accounting residual | below $2\times10^{-13}$ |
| Thermostat maximum absolute standardized accumulated innovation | 2.0881 |
| Thermostat realized/predicted quadratic variation | 0.8570–1.2287 |
| Clone maximum absolute standardized accumulated innovation | 1.9401 |
| Clone realized/predicted quadratic variation | 0.9114–1.1564 |
| Accepted historical copies, Gaussian / uniform | 706 / 719 |

The [validation record](validation/field-equations.json) preserves the protocol and reported results. All 85 combinations of five field observables and the 17 quarter-step wave numbers from 0 through 4 pass archive recomputation, and all five browser views report both computed equations. Additional actual gas runs cover conditional-fitness metric noise, periodic boundaries, absorbing loss and revival. The checks reject modified clocks, noise samples/factors, donor incarnations, clone probabilities and missing required stages. Predictable fluctuation scales are martingale scales, not confidence bands for full-step forecasts. The clone calculation conditions on donor selection rather than integrating its outer randomness.

The default seed-7, 96-update momentum-field run reports both equations available. Its recorded field-accounting residual is 4.16e-17, independent copy residual 2.50e-16 and independent kick/transport residual 2.78e-17. Accumulated clone and thermostat fluctuations divided by their predicted scales are −1.35 and −1.04.

The field-accounting residual checks a telescoping sum of realized increments. Its conditional thermostat mean cancels algebraically against the defined innovation; that residual therefore does not validate the mean prediction. Independent copy and A/B checks, moment quadrature and trajectory ensembles provide the prediction tests. Kick reconstruction conditions on the recorded potential gradient and independently recomputes viscosity. Availability requires the actual ordered stage sequence and its provenance; recorded boundary extinction permits only the stages genuinely executed.

Independent polynomial quadrature agrees with conditional means and all four real/imaginary covariance entries within $2\times10^{-12}$ for Gaussian and uniform innovations, low-rank and actual metric factors, source shifts, and friction $0$, $10^{-12}$ and $1.3$. Enumeration of all 256 gate patterns at actual eight-walker historical contexts independently verifies clone means and full covariance to the same tolerance. Both f32 and f64 execution are covered.

Reproduce the metric ensemble with `ALGORITHMIC_GAS_FIELD_REPORT=/tmp/metric-field-review.json cargo test -p algorithmic-gas-benchmarks --test algorithmic_metric_field_predictions -- --nocapture`.

The [metric-active ensemble report](validation/metric-field-review.json) retains 896 six-update trajectories: 128 Gaussian runs and three disjoint uniform batches of 128, 128 and 512 runs. Each run uses eight walkers, historical donors, viscosity and the actual fitness-metric provider. Recorded factor matrices change across updates. Independently reconstructing the metric from recorded fitness Hessians and checking $g(BB^\top)=2\gamma T I$ gives a maximum normalized residual $1.27\times10^{-12}$. Real, imaginary, sum and difference projections test the full complex covariance of clone and thermostat increments. Standard errors use complete trajectory sampling units.

Finite ensembles produce visible discrepancies, retained here with their independent follow-ups:

| Measurement | 128-run batch | Disjoint follow-up |
|---|---:|---:|
| Uniform thermostat energy imaginary mean, estimated-SE units | −3.120 | −0.041, 128 runs |
| Uniform thermostat stress01 real+imag quadratic variation, measured/predicted | 0.8367 | 0.9591, 128 runs |
| Uniform clone energy real+imag quadratic variation, measured/predicted | 0.7137 | 1.037736, 512 runs |
| Uniform clone stress00 real+imag terminal variance, measured/predicted | 0.5485 | 0.956527, 512 runs |

The notable deviations do not persist in those disjoint batches. In the 512-run batch, the largest absolute standardized discrepancy is 2.244, for clone quadratic variation; the thermostat mean, quadratic-variation and terminal-variance comparisons are each below 1.95 estimated standard errors. The fixed regression threshold is 4.5 estimated standard errors, a detection rule rather than a simultaneous confidence guarantee. The report preserves all 192 projection comparisons, the configurations, sample sizes, variances and standard errors. It supports agreement of the tested conditional equations without equating every finite-sample covariance estimate with its expectation.

### Correlations and spectral diagnostics

A fixed-normalization frame mean and a source-frozen valid-pair statistic are different observables. For $F_t=N^{-1}\sum_i O_i(t)$, its correlation includes cross-walker products. The source-pair statistic retains particular relationships and a lag-dependent valid-pair denominator. VI-35 and VI-36 keep these definitions separate.

VI-36 fits signed/complex exponential and damped-oscillation candidates. Training data determine centering, whitening and fits. Candidate rates are retained separately from accepted empirical fit parameters. Window stability and chronological held-out prediction error are required; an inconclusive candidate does not identify growth, decay or mass. Independent runs are needed to attach statistical uncertainty to these temporally dependent estimates.

The seed-516 diagnostic with donor memory two and viscosity illustrates the distinction. The rate **−1.78624**, frequency **−18.08976** and held-out complex RMSE **0.392445** belong to the **phase-space frame mean $x+i v$**, not the twistor readout. Its normalized correlation magnitudes are approximately

$$1,\;0.77,\;0.38,\;0.12,\;0.48,\;0.68,\;0.68.$$

The correlation changes sign near the dip. Fitting lag windows 1–3, 1–6 and 4–9 gives candidate decay rates **+22.83, −1.79 and +6.53**. This is a window-sensitive fit to a cancellation and rebound, not evidence of an unstable growing mode. The actual twistor readout has a sharp one-step correlation decrease followed by a small unresolved tail. Neither readout supplies a particle mass from this diagnostic.

The fit acceptance rules are numerical diagnostics: exponential candidates require small log/phase residuals, bounded trimmed-window rate variation and held-out improvement over the zero predictor. Damped-oscillation candidates additionally require stable fitted rate and frequency after dropping the first lag. These rules do not establish a positive self-adjoint transfer representation.

### Historical and material observables

Immutable sources are identified by frame, version, slot and generation. A historical slot is never resolved against the present population. The conditional metric uses the actual frozen source position and velocity. Transform influences can refer to intermediate operator-stage versions; the Fractal Set indexes those exact recorded states as well as anchors and completed populations.

Metric reconstruction agrees with executed derivatives within **3.55×10⁻¹⁵** for 26 global historical queries and **3.11×10⁻¹⁵** for 25 local distance-memory queries. An absorbing-boundary check exercises 36 revivals, including 29 historical distance contexts, with zero reconstruction discrepancy. Programmed revival selects an eligible current-frame donor; that donor's distance context can be historical. Historical cloning requires the supported global standardization configuration.

VI-45 evaluates a material metric at the resulting slot position. Its field-change, clone-replacement and subsequent-motion decomposition retains the complete cross-covariance. Summing that covariance reconstructs the total increment covariance within **8.17×10⁻¹⁴** in the reported seed-11/29/43 checks. This validates the measured decomposition; it does not supply a metric-only evolution closure.

### Conditional noise, source response and mechanics

For a shifted innovation $Y=B(\xi+s)$,

$$\mathbb E[Y]=Bs,\qquad\operatorname{Cov}(Y)=BB^\mathsf T,$$

while the uncentered second moment also contains $(Bs)(Bs)^\mathsf T$. VI-18 uses the actual factor rank and innovation law and excludes ineligible zero-filled rows. Standardized uniform innovations have fourth cumulant $-6/5$. Raw innovation moments and physically scaled thermostat moments are distinct readouts.

Reported comparisons include:

| Measurement | Executed result | Prediction or independent uncertainty |
|---|---:|---:|
| Rank-one uniform innovation second moment | 1.74313 | 1.75 |
| Rank-one uniform innovation fourth moment | 10.30073 | 10.2375 |
| Thermostat martingale variance, 384 independent continuations | measured/predicted = 0.97501 | ratio 1 |
| Gaussian source: direct-minus-reweighted residual sum | 0.05749 | standard error 0.13110 |
| Absorbing source: direct-minus-reweighted residual sum | −0.000465 | standard error 0.077357 |

The absorbing source comparison retains three extinct outcomes among 192 direct replicas. The quantities and sampling units are recorded in [the numerical evidence](validation/partvi-evidence.json). The noise moments come from dependent trajectory observations; the source and continuation uncertainties use independent runs. These comparisons support the tested conditional laws without assigning independent-sample errors to interacting walkers.

Actual force kicks include the provider force and viscosity. Mechanical ledgers include literal cloning, restitution, jitter, kinetic updates, boundaries and eligibility transfers. Deterministic copy maps and low-rank innovations do not acquire fictitious full-dimensional Gaussian densities: path calculations state the primitive carrier and its coverage. Unknown external providers and some dependent donor-sampling likelihoods remain explicit coverage limitations.

### The tested curvature–stress closure fails

The tested common two-coefficient spatial relation is

$$\overline{\operatorname{Ric}-\tfrac12 Rg}
=\kappa\,\text{centered kinetic moment}+\lambda\,\overline g.$$

Coefficients fitted on separate training seeds give these held-out errors:

| Study | Proposed relation RMSE | Constant baseline | Shuffled-training control |
|---|---:|---:|---:|
| Global | 13,340,129.4 | 13,414,635.6 | 13,384,667.1 |
| Local | 51,363.2 | 31,599.7 | 32,372.4 |

The global improvement is only 0.56%, nearly matched by shuffled data; the local fit is substantially worse than a constant. **This particular reduced spatial closure fails the measured prediction test.** Its algebraic form is not repaired by refitting coefficients.

Curvature depends on differentiated normalized fitness, donor context and clipping. A centered velocity second moment is only one candidate stress ingredient; volume normalization and force, transport, cloning, thermal and boundary contributions require explicit treatment. Their omission is a possible explanation of the failed reduction, not a demonstrated causal diagnosis. [The research protocol](VERIFICATION.md) states the configurations, train/test split and measured quantities.

## Tessellation geometry and the Einstein–Hilbert gas

The tessellation estimators are checked against three kinds of reference
(`crates/algorithmic-gas/tests/tessellation_{geometry,parity,engine}.rs`,
`crates/benchmarks/tests/einstein_hilbert_gas.rs`).

**Analytic.** A flat metric gives zero curvature for every estimator. For a
conformally flat metric `e^{2u}δ` with quadratic `u` the quadratic-fit scalar
and the trace of its Ricci tensor match the closed form to `1e-5` in 2D and 3D
(ridge far below the squared spacing). Voronoi volumes sum to the box volume to
`1e-10` in clip and periodic boxes in 2D and 3D, with reciprocal facet areas.
Regge deficits of a flat triangulation vanish up to the conditioning of angles
computed from lengths (`< 1e-4` in curvature); with exact great-circle lengths
on the unit sphere the Regge action per area is `2` within `1%` at 1500 sites.
With endpoint-metric lengths on the same sphere it is `4.8`: the documented
`O(1)` bias of that length model. The 3D Delaunay complex passes brute-force
empty-circumsphere, orientation, Euler and hull checks on generic clouds, a
cubic lattice, a Kronecker lattice with exactly coplanar quadruples, and sites
on a sphere.

**Python reference estimators** (`fragile.physics.geometry`, float64 fixtures
from `tools/export_tessellation_fixtures.py`, 60 sites in 2D and 80 in 3D).
Delaunay edges agree exactly, including coincident, collinear, coplanar and
duplicate-group swarms. Emergent metric, determinant, volume element, diffusion
factor and geodesic lengths agree to `1e-8` relative when the run is given the
reference's own absolute ridge (`RidgeScale::Absolute`, the convention the
fixtures were measured with; the shipped scale-covariant default departs from
them by the amounts the `Q30 (tessellation fixtures)` row below records); all
shared edge-weight modes and the conformal Laplacian curvature to `1e-7`; the
quadratic-fit scalar and Ricci tensor to `1e-5` (normal equations with an
absolute ridge); viscous force and one Boris B step to `1e-8`, curl and
rotation angle to `1e-6`. The reference `inverse_volume` weights have unit
cells and equal the uniform weights; here that mode uses the Voronoi volumes.

**Dynamics.** 500 walkers, three dimensions, 750 steps, single precision, all
walkers starting at the origin at rest, cloning every 20 steps. Under
`GeometrySchedule::PostClone { every: 1, on_clone: true }`, which is the
reference run's schedule, eight Rust seeds and three Python seeds give

| Quantity (final state) | Rust, 8 seeds | Python, 3 seeds |
|---|---|---|
| Mean squared speed | 0.281 ± 0.007 | 0.289 ± 0.018 |
| Action `Σ R_i sqrt(det g_i)` | (1.10 ± 0.15) × 10⁶ | 0.78, 1.13, 1.03 × 10⁶ |
| Mean curvature | 0.074 ± 0.010 | 0.070, 0.083, 0.055 |
| Mean volume element | 2490 ± 320 | 1756, 2781, 2488 |
| Directed neighbor edges | 2960–2976 | 2962–2974 |
| Clones per cloning step | 157 | 159 |

The two engines use different random number generators, so agreement is
statistical; the spreads are seed-to-seed standard deviations. The mean squared
speed is below `3T = 0.99` because the row-normalized viscous coupling adds a
relaxation rate `ν = 3` toward the neighbor mean without a matching noise. A
run takes 1.5–3 s natively and 20–45 s in the Python engine on the same
machine. Under the default `EveryStage` schedule the same configuration gives
an action of (0.78 ± 0.13) × 10⁶ over eight seeds: rewarding current rather
than one-step-old geometry changes the selection pressure, not the estimators.

The Python Einstein-equation analyzer
(`fragile.physics.app.qft.einstein_equations.compute_einstein_test`) runs
unchanged on a `RunHistory` converted from a recorded `RunArchive`
(`tools/eh_archive_to_history.py`). These runs validate the implementation of
the estimators and of the sampler; they make no claim that the conformal
curvature proxy converges to a continuum curvature of the walker density, nor
that the gas satisfies a field equation.

## What the geometric and information experiments establish

Recorded ray-overlap loops provide an explicit U(1) connection and exact orientation/local-phase identities. They do not establish an SU(3) transporter or identify the loop action with the gas path action. Paired parity and translation experiments transform the complete configured dynamics rather than inferring symmetry from a color conjugation identity alone.

Spatial metric transport integrates the reconstructed conditional connection along recorded-point paths. Cloud expansion uses a declared regularized covariance volume. These are measured geometric constructions; they do not supply a Lorentzian congruence, Raychaudhuri equation, spacetime caustic or Einstein law.

Graph cuts, maximum flow, ancestry depth and region-to-region information use executed edges and immutable events. Their exact combinatorial identities do not identify graph capacity with geometric area or quantum entropy. A velocity-derived null bispinor is an explicit algebraic embedding, not a mass measurement. Likewise, a scalar-derived curvature length is not an externally supplied AdS radius.

The energy-distribution experiment fits an exponential comparison on the training prefix and displays held-out discrepancies. Stationarity and descriptor closure experiments measure temporal drift and predictive errors. Empirical exponential tilting differentiates a finite measured distribution; it does not replace the physical Gaussian-source intervention in VI-19.

## Measured agreement and discrepancies in the chapter runs

The comparisons below use seed 7 and 96 baseline updates. Continuation experiments default to 32 replicas per independent group; the eight-replica VI-22 entry is a small-sample precision diagnostic. The following comparisons retain their actual sampling units:

| Experiment | Measured comparison | Interpretation |
|---|---|---|
| VI-18 | Innovation second moment 0.629888 versus 0.640000; fourth moment 1.178321 versus 1.228800 | Differences of about 1.6% and 4.1%. These trajectory moments do not have independent-walker error bars. |
| VI-19, 32 replicas per group | Direct-minus-reweighted response −0.076182, independent standard error 0.069352 | Agreement within 1.10 standard errors. The effective importance sample size is 29.78. The measured source change is 0.018499 ± 0.002765 standard error. |
| VI-22 | Eight-replica group residual −2.650312, estimated standard error 0.889854 | A visible discrepancy of about 2.98 estimated standard errors, requiring more independent continuations. |
| VI-45, 32 replicas per group | Material metric increment difference −0.546006, standard error 1.714137; component covariance residual 1.42×10⁻¹⁴ | Conditional comparison within 0.32 standard errors; the full cross-covariance decomposition agrees to numerical precision. |
| VI-28 | Smallest temporal-reflection eigenvalue −4.6954×10⁻⁵ from 90 overlapping windows | The empirical reflected matrix is not positive. Dependent windows provide no independent-run confidence interval. |
| VI-12/13/32 | Two-step projected memory defect 0.11353; held-out transition-matrix RMSE 0.42603 at lag one and 0.35301 at lag two | The compressed field descriptor loses information required to compose the actual dynamics, despite improving on a constant Brier predictor. |
| VI-36 | Default twistor candidate rates +0.371, +0.989 and −1.908 across trimmed windows; held-out complex RMSE 0.371 versus zero baseline 0.378 | The fit is inconclusive and identifies no rate or mass. This readout differs from the phase-space diagnostic discussed above. |
| VI-57 | Exponential energy-histogram total variation 0.136251 | The fitted exponential leaves a measurable distribution mismatch; it is not the established equilibrium law of the gas. |
| VI-65 | Held-out one-step Brier loss 0.082015 versus constant 0.125539 | The descriptor helps one-step prediction on this run; this does not establish multi-step closure. |

For VI-22, increasing to 32 continuations per independent group gives residuals and standard errors of −0.334481 ± 0.510271 at seed 7, −0.683903 ± 0.564662 at seed 0, and −0.372993 ± 0.591304 at seed 516. Their absolute standardized residuals are 0.66, 1.21 and 0.63. The analytic thermostat momentum and energy residuals are within 1.33 predicted standard errors in each of these runs. The seed-7 extension includes the smaller sample, so it is a precision extension rather than an independent replication of that sample. The results are consistent with sampling variation and do not show a persistent drift-law discrepancy. [Requests and measured follow-up results](validation/lecture-review.json) preserve this check.

VI-51 reports executed mechanical sources together with the conditional clone and thermostat field predictions above. Its archive regression excludes prescribed pair examples and prescribed Ricci contractions from the run result.

The VI-39 scalar-curvature cross-check is under-resolved at spacing 0.002: the packed value is −2564.930957, while the coarse independent difference gives −3573.082250, a 39.3% discrepancy. That coarse stencil also crosses a spectral clipping surface. Refining the spatial probe on the same archived conditional field gives −2895.936228 at 0.001, −2649.973203 at 0.0005, −2568.362734 at 0.0001, −2564.965283 at 0.00001, and −2564.930585 at 0.000001. The error decreases approximately quadratically in the resolved range and reaches 1.45×10⁻⁷ relative error. This identifies a finite-difference resolution/support error rather than a discrepancy in the packed curvature formula for this state. The [refinement evidence](validation/lecture-review.json) preserves the measured sequence.

VI-39 automatically refines from 0.002 through at most twelve halvings. It rejects stencils with a changed number of positive Hessian eigenvalues and retries locally. Convergence requires two successive numerical scalar/Ricci estimates to satisfy relative tolerance 10⁻⁵ and absolute tolerance 10⁻⁶. The packed analytic result plays no role in choosing the spacing. The demo displays the selected spacing, refinement curve and converged, underresolved or unsupported status.

The default 96-step regression selects spacing 1.953125×10⁻⁶ and reports converged numerical curvature −2564.931964 against packed −2564.930957, a relative discrepancy of 3.92×10⁻⁷. It retains the actual step-94 donor source throughout the probes.

Reproduce the independent spatial-refinement check from the Rust workspace:

```sh
cargo run --release -p algorithmic-gas-benchmarks --example curvature_refinement
```

## Verification coverage

The following checks have completed for this implementation:

- Parts I–IV: all 42 IDs at seeds 0, 7 and 516, giving 126 native four-step executions; every declared control endpoint; derivative reconstruction and finite differences; transport upper bounds; exact finite-step harmonic predictions checked against independent populations; replay without observational disturbance; matched-duration refinement.
- Compiled Parts I–IV: 52 passing WASM tests, including archive reanalysis, deterministic replay, explicit extinction, twelfth-order derivatives and an independently derived 64-replica BAOAB moment comparison.
- Core numerical expansion: order-12 exponential and inverse-square-root coefficients, with an admission bound before multi-index allocation.
- Tracking: the intermediate-source identity regression and all 14 existing Part V tracking checks; source identities retain donor memory and actual viscous-stage versions.
- Field and QFT work: targeted native recorded-observable, geometry, complete paired-run and checkpoint-evidence checks exercise the new readouts. Their exact cases remain in the named test suites below.
- Benchmark all-targets Clippy with warnings denied.

The workspace sweep passed 301 tests. The strengthened field checks pass nine field-equation tests, four stage-coverage tests and one metric-ensemble test. The native control sweep passed 258 Part V–VI seeded defaults and 1,455 individual control endpoints; the continuation follow-up passed 72 cases. Compiled and browser checks passed all 128 demos, evidence replay and all chapter embeds. The Theory build and portal assembly passed. The [lecture validation record](../fractal-gas-web/web/euclidean-gas/lecture/VALIDATION.md) gives the coverage. No claim of accelerator runtime parity follows from CPU/WASM checks.

The applicable suites are `lecture_early`, `lecture_source_identity`, `lecture_field_geometry`, `lecture_fractal_observables`, `partvi_run_observables`, `lecture_qft_protocols`, `lecture_qft_evidence` and `lecture_registry`. Exact identities use numerical residuals; fitted physical hypotheses retain held-out errors and explicit interpretation limits.

## Per-experiment measurement map

Each entry links the chapter's concept to an observable of an actual run. The machine contract additionally preserves the theory labels, mathematical identity, sampling definition and limitations.

| ID | Executed measurement | Execution |
|---|---|---|
| VI-01 | Oriented transport on recorded triangles | Run archive |
| VI-02 | Executed clone gates and their innovations | Run archive |
| VI-03 | Exterior algebra of measured frame observables | Run archive |
| VI-04 | Measured transitions and CAR channels | Run archive |
| VI-05 | Regional covariance and locality | Run archive |
| VI-06 | Products and empirical replica wedges | Run archive |
| VI-07 | Finite-step weak-generator samples | Run archive |
| VI-08 | Force-based colors from executed stages | Run archive |
| VI-09 | Invariant coordinates of measured colors | Run archive |
| VI-10 | Companion doublets with immutable sources | Run archive |
| VI-11 | Phase-space ray holonomy | Run archive |
| VI-12 | Compressed channels and memory | Run archive |
| VI-13 | Predictive partition refinement | Run archive |
| VI-14 | Literal clone writes and population balances | Run archive |
| VI-15 | Complete-engine parity and color conjugation | Paired complete runs |
| VI-16 | Generating functions of recorded observables | Run archive |
| VI-17 | Executed path-action accounting | Run archive |
| VI-18 | Noise moments from actual factors | Run archive |
| VI-19 | Gaussian source response through complete updates | Checkpoint continuations |
| VI-20 | Recorded transition information | Run archive |
| VI-21 | Empirical geometry-fiber disintegration | Run archive |
| VI-22 | Conditional stochastic balance | Checkpoint continuations |
| VI-23 | Local phase Ward identity on recorded loops | Run archive |
| VI-24 | Measured loop action and phase variation | Run archive |
| VI-25 | Empirical descriptor contraction | Run archive |
| VI-26 | Cloning and restitution mechanical ledger | Run archive |
| VI-27 | Eligibility and conditional population moments | Run archive |
| VI-28 | Empirical temporal reflection spectrum | Run archive |
| VI-29 | Oriented boundaries of recorded interaction faces | Run archive |
| VI-30 | Complete-engine translation covariance and response | Paired complete runs |
| VI-31 | Propagation of a local intervention | Paired complete runs |
| VI-32 | Predictive algorithmic field channels | Run archive |
| VI-33 | Velocity-derived null bispinors | Run archive |
| VI-34 | Twistor observables from recorded triplets | Run archive |
| VI-35 | Source-frozen twistor lag tracking | Run archive |
| VI-36 | Signed correlations and spectral fit diagnostics | Run archive |
| VI-37 | Metric transport along executed displacements | Run archive |
| VI-38 | Holonomy along a recorded-point loop | Run archive |
| VI-39 | Curvature of the executed fitness metric | Run archive |
| VI-40 | Connection of a measured conditional metric | Run archive |
| VI-41 | Finite-step cloud expansion balance | Run archive |
| VI-42 | Population volume and its stage budget | Run archive |
| VI-43 | Measured cloud focusing | Run archive |
| VI-44 | Topology of the recorded Fractal Set | Run archive |
| VI-45 | Conditional material metric drift and covariance | Checkpoint continuations |
| VI-46 | Executed virial and dilation work | Run archive |
| VI-47 | Density modes of the executed population | Run archive |
| VI-48 | Conditional BAOAB diffusion and heating | Run archive |
| VI-49 | Velocity-mode energy and Parseval balance | Run archive |
| VI-50 | Measured cloud and kinetic scales | Run archive |
| VI-51 | Algorithm-derived Fourier field equations, conditional clone/noise predictions and mechanical sources | Run archive |
| VI-52 | Cuts by executed interaction channel | Run archive |
| VI-53 | Empirical transition time asymmetry | Run archive |
| VI-54 | Recorded interaction perimeter scan | Run archive |
| VI-55 | Observed companion diversity and support | Run archive |
| VI-56 | First variation of an executed graph cut | Run archive |
| VI-57 | Measured energy distribution and an exponential hypothesis | Run archive |
| VI-58 | Empirical source susceptibility | Run archive |
| VI-59 | Recorded ancestry depth | Run archive |
| VI-60 | Maximum flow and minimum cut of selected interactions | Run archive |
| VI-61 | Recorded distribution stationarity | Run archive |
| VI-62 | Measured fitness curvature scale | Run archive |
| VI-63 | Scale evolution of the executed cloud | Run archive |
| VI-64 | Information in executed region-to-region interactions | Run archive |
| VI-65 | Held-out empirical field closure | Run archive |
| VI-66 | Dimensionless scales measured from the gas | Run archive |

## Lecture execution checks

All 66 registered Part VI demos execute the same Rust sessions in native and
WASM builds. Archive import recomputes their measurements. VI-19, VI-22 and VI-45
condition on the displayed session's exact completed state; tests compare its
population and step with the exported baseline. A 72-case follow-up covers three
seeds and every individual control endpoint for these continuation experiments.

A full 96-step comparison verifies that retaining the required conditioning
context instead of duplicating the entire audit recording changes no plotted
value, metric or baseline archive. Complete donor history and addressed random
schedules remain in the checkpoint. Default native evidence bundles are below
64 MiB. The same evidence survives JavaScript's integral-number JSON encoding
and reproduces the measured results in Rust.

## Algorithmic spectroscopy: scope, parity exclusions and open conventions

The spectroscopy subsystem (`crates/algorithmic-gas/src/physics/spectroscopy.rs` and its
module files) measures gauge-theory operators on the recorded companion topology of a gas run,
correlates them in algorithm time, fits decay rates and compares the rates with a reference
table. It is split in two halves that can be run independently: a *measurement* streams recorded
steps through an `Accumulator` and retains operator series, source-frozen propagator moments and
coverage counters; an *analysis* turns one or more measurements into a `SpectroscopyReport` and
can be repeated with a different `AnalysisConfig` without running the gas again. Both halves are
f64 only: an archive of an `f32` run is refused with `GasError::Capability` rather than promoted,
so an Einstein–Hilbert session, whose constructor inherits `Precision::F32`, must select F64
explicitly before it can be measured.

This section states what the subsystem measures, which variant can support which measurement
family, what it does where the estimator cancels identically, which Python behaviours are
deliberately not reproduced — the eleven D-numbers of the first audit and the eighty deduplicated
defects Q1-Q80 of the second — what a later port of the subsystems that were left out must
implement instead, and which conventions are still open. It records contracts,
availability rules and exclusions; it reports no measured spectrum, and no number below is a
result of a gas run. The measurement and the analysis run end to end on recorded
archives: `crates/algorithmic-gas/tests/spectroscopy_pipeline.rs` measures Einstein-Hilbert,
viscous Euclidean and Euclidean runs with the standard channel set and fits rates on them, and
the `algorithmic_gas_benchmarks::spectroscopy` runner, the `gas-spectroscopy` binary and the
wasm bindings drive the same code from a session, a command line and a browser
(`crates/benchmarks/tests/spectroscopy_session.rs`, `crates/benchmarks/tests/spectroscopy_cli.rs`,
`fractal-gas-web/tests/euclidean-gas/qft-simulator-browser.mjs`). Statements below about what
the subsystem does are statements about the frozen contract unless they cite a test or a fixture.

### What is fitted, and when it may be called a mass

The fitted quantity is the decay rate of the algorithm-time autocorrelation of a frame
observable, in the sense of `def-qft-channel-decay-rate` of
`docs/source/2_fractal_gas/2_fractal_set/09_qft_calibration.md`. A *channel* is the whole
bookkeeping — local operator, element selection, masks, weights, colour alignment and frame
normalization — and the channel correlator is the connected autocorrelation contracted over the
operator's components,

$$C_\chi(\ell)=\sum_{k}\operatorname{Cov}_\pi\bigl(f_\chi^{k}(R_0),f_\chi^{k}(R_\ell)\bigr).$$

Under no further hypothesis (item 1 of `prop-qft-decay-rate-scope`) a fitted rate is a
rate of the semigroup of the executed algorithm on the cyclic subspace of that frame observable,
and nothing more. It becomes a **channel mass** only under
`assm-qft-positive-transfer`, the existence of a positive representing measure
$C_\chi(\ell)=\int\lambda^{\ell}d\nu_\chi(\lambda)$, which holds when the transfer operator is
self-adjoint and positive on $L^2(\pi)$ — in particular under the hypotheses of
`cor-effective-twistor-positive-transfer` of `08_twistor_formulation.md`. No result of
Volume II establishes that assumption for any gas variant. The constant `INTERPRETATION_NOTE` is
prepended to the `notes` of every report and says exactly this.

The executed chain is not reversible. Item 3 of `prop-qft-decay-rate-scope` exhibits the
failure mode explicitly: for a rotation-like kernel the correlator is
$C(\ell)=|\lambda|^{\ell}\cos(\ell\varphi)$ with $0<\varphi<\pi$, so it is negative at some lags,
no positive representing measure exists, and the effective rate
$m_\chi(\ell)=-\Delta\tau^{-1}\log[C_\chi(\ell+1)/C_\chi(\ell)]$ is undefined there although the
envelope still decays. A complex or oscillating mode is therefore the expected behaviour of this
chain, not an anomaly. The subsystem treats it as a refutation of the model for that channel and
not as a small or zero rate: `FitOutcome::mass` is `None` whenever `FitDiagnostics` carries
`model_rejected` (oscillating or sign-changing correlator), `no_signal` (rate significance below
`WindowScanConfig::min_rate_snr`, default 2) or a dominated prior
(`PriorDominance::dominated`). The report never encodes an undefined number as a placeholder zero:
`SpectroscopyReport::validate` rejects a non-finite number, and an undefined value serializes as
`null`.

Three estimators are available, and each labels its rate with a different quantity string, because
they are different observables:

| Estimator | Quantity reported | Transfer-matrix reading |
|---|---|---|
| `FrameMean` | decay rate of the algorithm-time autocorrelation (`RATE_QUANTITY`) | yes, under `assm-qft-positive-transfer`; requires the fixed-$N$ normalization |
| `SourceFrozen` | decay rate of the source-frozen pair correlator (`RATE_QUANTITY_SOURCE_FROZEN`) | no: `prop-qft-decay-rate-scope` is explicitly not asserted for it |
| `EuclideanTime` | decay rate of the Euclidean-coordinate slab correlator (`RATE_QUANTITY_EUCLIDEAN`) | no; it is a spatial correlation inside a frame |

The source-frozen propagator is a ratio of sums with a lag-dependent valid-pair denominator that
pairs a source observable with a different sink observable
(`def-effective-twistor-correlators`), so its plateau is not a spectral reading even when
it is clean. Errors are resampling errors whose sampling unit every `SamplesMeta` states; with
fewer than four independently seeded replicas a note says that they are block-resampling errors
inside a run and not replica standard errors.

### Capability model: what a variant can support

`Capabilities::of(gas_config, recording, dimension)` decides, before any step is ingested, which
kinds of record the run will produce. Each absent record carries the reason it is absent;
`Requirements::check` turns that reason into the `Availability::Unavailable { reason }` of every
dependent channel. A missing record therefore makes channels unavailable with an explanation and
never fails the run.

| Record | Supplied by | Absent when |
|---|---|---|
| `Velocities` | a BAOAB kinetic integrator | any other integrator |
| `Color` | a recorded viscous force with a nonzero coefficient (`qft.viscosity` dense, or `qft.graph_viscosity` on the neighbour graph) | no velocities, or "viscous force is identically zero" — both coefficients absent or zero |
| `Graph` | a geometry stage whose graph is recorded | `geometry` is `None`, or `RecordingConfig::graph` is false |
| `EuclideanTime` | a geometry projection that drops a position coordinate (`Projection::DropLast`, ambient $d\ge3$) | no component of the tuple distinguishes a coordinate |
| `PeriodicBox` | `BoundaryPolicy::PeriodicBox`, searched through composed policies | any other boundary: "momentum projection needs a periodic box" |
| `DistanceCompanions`, `CloningCompanions`, `ClonePlan` | the donor and cloning stages | — |

This reproduces `rem-variants-measurability` of
`docs/source/2_fractal_gas/1_the_algorithm/04_gas_variants.md`. The three shipped variants sit as
follows.

| Variant | Colour channels | Graph and multiscale | Euclidean-time axis | Momentum projection |
|---|---|---|---|---|
| Euclidean Gas (`GasConfig::euclidean`) | unavailable: $F^{\mathrm{visc}}\equiv0$, the encoding $F/\lVert F\rVert$ is undefined on every row at every step | unavailable: no geometry stage | unavailable: every coordinate is treated identically | unavailable: absorbing box |
| Viscous Euclidean Gas (`GasConfig::viscous_euclidean`) | available for $\nu>0$, dense viscosity | unavailable: no geometry stage | unavailable | unavailable: absorbing box |
| Einstein–Hilbert Gas (`GasConfig::einstein_hilbert`) | available, graph viscous force $\nu=3$ | available: Delaunay graph with its weights and metric edge lengths | available for $d\ge3$: the dropped last coordinate | unavailable at the reference configuration: unbounded boundary |

The Geometric, Latent and Environment Gases are specified in the book and not implemented by the
engine: `Variant::reference` returns `None` for them, and `Variant::config` (and
`Variant::default_config`) refuses them with a `GasError::Capability` naming the variant's title
and its book label.

Four points of this table are worth stating plainly for a reader who expects colour to be a
property of the theory rather than of the variant.

- **The canonical Euclidean Gas has no colour at all.** The viscous force is identically zero by
  definition of the variant, not small, so there is no direction to encode and no row on which the
  colour state exists. It cannot be rescued by substituting $-\nabla U$ (a gradient that knows
  nothing about the other walkers) or by softening the denominator (which returns the zero vector
  at zero force). The only two routes to colour are $\nu>0$ (Viscous Euclidean, which costs every
  convergence theorem) or the graph viscous force (Einstein–Hilbert, which costs confinement and
  time homogeneity). A `ColorSource::RecordedField` may name two recorded fields explicitly and is
  the only colour route for a variant whose integrator is not BAOAB.
- **Availability is a property of the variant, validity a property of the sample.** Even where the
  coupling is nonzero, the graph viscous force $\nu\sum_j w_{ij}(v_j-v_i)$ vanishes on a population
  with equal velocities — exactly the reference initial condition, all walkers at the origin at
  rest — and on rows the tessellation left without neighbours. Those rows are masked by the
  absolute threshold $\delta_c$ (default `1e-12`, in force units) and counted, never treated as
  zero colour.
- **Dense and graph viscosity are different measurements.** Dense viscosity records $2N(N-1)$
  pair influences per step and is flagged by `Capabilities::dense_viscosity`; the graph force reads
  the recorded neighbour graph instead, and the multiscale and smoothing channels exist only when
  that graph is recorded.
- **A Euclidean-time axis is a property of the variant only when the tuple distinguishes a
  coordinate.** `TimeAxis::Euclidean { axis: Some(k) }` lets an analysis declare one on any run;
  the resulting numbers are then reproducible measurements of that choice and not of the gas, and
  `Capabilities::refine` records the declaration. Momentum weights
  $\cos/\sin(2\pi n\,x_{\text{axis}}/L)$ require `Record::PeriodicBox`; the mode-0 sine, which
  vanishes identically, is rejected by `Momentum::validate` instead of being offered.

### The exchange-odd cancellation on mutual pairings

`prop-exchange-odd-cancellation` (04_gas_variants) is an algebraic identity, not a
statistical statement. For any involutive companion map $c$ with $c(c(i))=i$, any exchange-odd
operator $O_{ji}=-O_{ij}$ and any weight $w$,

$$\sum_{i}w_iO_{i\,c(i)}=\tfrac12\sum_{i}\bigl(w_i-w_{c(i)}\bigr)O_{i\,c(i)},$$

so a pair-symmetric weight — an ordinary unweighted frame average — gives exactly zero, for every
realization, at every step, whatever the gas is doing. The proof uses no probability, no largeness
of $N$ and no property of the dynamics. Fixed points contribute $O_{ii}=0$.

The hypothesis holds where the companion law matches walkers two by two:
`SamplingLaw::FisherYates` and `SamplingLaw::GaussianGreedy`, recorded as
`Capabilities::mutual_distance` and `mutual_cloning`. The Einstein–Hilbert preset uses Fisher–Yates
for both roles. It does not hold for independent companion draws.

The channels affected are the ones the operator declares `ExchangeParity::Odd`: the pseudoscalar
$\operatorname{Im}q_{ij}$, the vector $\operatorname{Re}q_{ij}\,r_{ij}$, the complex baryon and the
tensor components. Scalar and axial are `Even` and survive. The complex U(1) and SU(2) phase
channels are `Mixed`: their real part is exchange-even and survives while their imaginary part
cancels, and because the declared parity is not `Odd` the estimator fallback below does not fire
for them — the cancellation of the imaginary component is stated in the channel notes instead.
Two consequences beyond the plain statement were measured on the parity
fixtures: score *direction* repairs only the pseudoscalar, because conjugating $q$ on downhill
pairs leaves $\operatorname{Re}q\cdot r$ odd and makes $\operatorname{Im}q\cdot r$ odd, so
`axial_score_directed` cancels although `axial` survives; score *weighting* keeps the orientation
of the pair, so `pseudoscalar_score_weighted` cancels too.

What the subsystem does instead of reporting a fitted zero:

- The decision is taken on the **built topology**, per frame, not on the law's flag:
  `topology::is_involutive` inspects the element set after masking (self companions, historical
  companions, ineligible rows, out-of-range indices), and `ChannelSeries::weight` counts the
  unmirrored elements of each frame. A single unmirrored frame — an odd population size leaves one
  self companion per frame and role, masks break mirror pairs — keeps the frame mean alive.
- `select_estimator` declines only when the weight is zero in every frame of every replica. With
  `EstimatorChoice::Auto` it then falls back to the source-frozen propagator if one was measured,
  with a note saying why; `PropagatorConfig::default` enables propagators for exactly the four
  exchange-odd specifications of the standard set (`meson/pseudoscalar/standard`,
  `vector/vector/full/raw`, `baryon/complex`, `tensor/components`). With no propagator the channel
  is `Unavailable` with the reason `exchange-odd operator cancels on a mutual pairing`, carries no
  correlator, no effective mass and no rate.
- A relabelling-odd *triplet* operator has a frame mean of zero expectation under exchangeable
  companion roles; `Auto` prefers its propagator when one was measured.
- A GEVP basis is a frame-mean object, so members that fell back to the propagator are dropped
  from it; fewer than two remaining members make the basis unavailable with that reason.
- The escape hatches of `rem-exchange-odd-scope` remain configurable: a weight that is not
  pair-symmetric (role masks, score direction) retains the signal, and the normalized cloning score
  is not exchange-odd because its two denominators differ.

On the `mutual_pairing_odd.json` fixture (Fisher–Yates, $N=9$) the cancelled float32 series reach
at most `1.5e-8` while the surviving ones are of order `0.2`–`0.31`; in float64 the frame sum of
$\operatorname{Im}q$ over valid pairs is at most `3.3e-16`, inside the identity bound
`1e-14·N`. The audit measured the same effect on a recorded run: pseudoscalar `1.3e-8` and vector
`1.5e-8` against scalar `0.14` and axial `0.35`.

### QFT_PARITY_EXCLUSIONS

The Python pipeline in `src/fragile/physics` was audited operator by operator against the book.
Where it is correct, the Rust implementation is tested against exported fixtures; where the audit
confirmed a defect, Rust pins the corrected behaviour and parity is deliberately excluded. The
fixtures and their schema are described in
[`crates/algorithmic-gas/tests/fixtures/qft/README.md`](crates/algorithmic-gas/tests/fixtures/qft/README.md),
whose "Parity exclusions" section carries the list row by row, keyed to the Python file and line
that produces each behaviour.

The table below is the first audit, D1-D11: its confirmed defects and the behaviour Rust pins for
them, with one row of the second audit appended, `Q30 (tessellation fixtures)`, which excludes a
tessellation fixture rather than a QFT one and therefore has no counterpart in that README. The deduplicated ledger of the second audit, Q1-Q80, follows it and subsumes it. Not every row is a *fixture* exclusion. The rows that section excludes from the exported
fixtures are D1, D2, D3, D4, D6, D7, D8, D10, D11b, D11c, D11d, D11e, D11g and D11j. D5, D9,
D11a, D11f, D11h and D11k concern behaviour outside the fixture set, and D11i is a tolerance
rather than an exclusion: the fixtures do export Python's float32 series and parity compares them
to `1e-6` absolute, Rust computing the same quantity in f64.

| Audit | Python behaviour | Observable consequence | Rust pins |
|---|---|---|---|
| D1 | Fisher–Yates pairing gives $c(i)=j$, $c(j)=i$; exchange-odd frame means are averaged over both orientations | pseudoscalar, vector and the imaginary part of every U(1) channel are float32 roundoff (`1e-8`) while scalar and axial are `0.14`–`0.35`; a fit on them returns a rate of the noise | `LocalOperator::exchange()`; an odd channel with no unmirrored element in any frame falls back to its source-frozen propagator or is `Unavailable`, never fitted |
| D2 | per-channel `dE_ground` priors (π 0.024, σ 0.09, ρ 0.13, N 0.16, G 0.29) tuned to PDG | PDG ratios are reproduced by construction (π/N 0.150 against 0.149); the fit cannot fail | channel-agnostic scale-free priors (Gaussian on $\ln dE_n$, identical for every channel); every rate carries `PriorDominance` (posterior/prior width ratio, mean shift in prior widths); a dominated ground state reports no rate; the reference table is a `hypothesis mapping` and no reference value enters a prior, a window or a channel selection |
| D3 | GEVP bootstrap shuffles time points independently | on AR(1) with $\rho=0.9$ the resampled $C(1)/C(0)$ is 0.028 instead of 0.898: errors of a decorrelated series | origin-block resampling of the cross-lag products; every matrix entry shares one block length and one set of draws, so the entries stay jointly consistent |
| D4 | `epsilon_clone` doubles as the Gaussian range of the pair amplitude | with `1e-8` the amplitude $e^{-D^2/4\varepsilon^2}$ underflows and `su2_component`, `su2_doublet`, `su2_doublet_diff`, `ew_mixed` are identically zero | `ElectroweakScales` separates $\hbar_{\mathrm{eff}}$, $h_S$, $\varepsilon_d$ and $\varepsilon_c$; a `Range` is the gas's Gaussian companion kernel width or a fixed value, never the clone regulariser; any other kernel makes the dependent channels unavailable |
| D5 | colour built from a force evaluated on post-clone velocities, paired with the pre-clone companions, scores and positions of the frame | the phase velocity and the velocity inside the force are different velocities; cloned rows carry a colour of a population that no longer exists | four explicit `ColorAlignment` arms; the default `PrecedingKick` uses the B2 force of step $t-1$ with its own input velocity, so one velocity enters both the force and the phase as `thm-sm-su3-emergence` requires; `MatchedKick` masks the rows cloned at $t$ |
| D6 | block size hard-coded to 10; the jackknife reuses the full-sample mean for the disconnected part | on AR(1) with $\rho=0.95$, $T=1000$, the nominal 68 % interval covers 0.41 of the time (block 10) and 0.57 (block 100) | `BlockSize::Auto` $=\max(2\tau_{\mathrm{int}},(2\tau_{\mathrm{int}})^{2/3}T^{1/3})$; each resample recomputes its own disconnected part; $\tau_{\mathrm{int}}$, the effective block, the covariance rank against the number of lags and the SVD cut are reported |
| D7 | `cloning_frames_only` correlates a gappy series as if it were uniform | the lag axis is wrong by the gate period; with cloning every 20 steps a lag of 1 frame is 20 algorithm steps | not ported: full-frame series with per-frame masks and weights |
| D8 | vector, axial and $\sigma_{\mu\nu}$ components averaged before correlating | the mean over components is not a rotation scalar and cancels for an isotropic system | components are kept and the correlator contracts $\sum_k C_{kk}(\tau)$, as `def-qft-channel-decay-rate` prescribes; the RMS envelope is `correlatable: false`; the documented non-invariant scalar survives only as the explicitly labelled `tensor_mean` twistor observable |
| D9 | $J^{PC}$ labels assigned by hand ($\gamma_5\gamma_k$ as $1^{+-}$, $\sigma_{\mu\nu}$ as "$2^{++}$", "$\psi_L$ = upper components" in the Dirac representation) | channels are named after states they have not been shown to create | channels are labelled by algebraic definition plus the measured exchange parity and spatial parity; physical names appear only through `AnalysisConfig::assignments`, an input hypothesis |
| D10 | "Wilson flow": convex neighbour averaging on a degree-$\le2$ companion graph with a non-standard $w_0$ | a length scale is read off a curve that defines none | ported as a graph smoothing diagnostic with no $w_0$ and no length claim; the module states why no threshold crossing of the roughness curve sets a scale |
| D11a | the tensor RMS envelope can be FFT-correlated | a correlator of an amplitude is reported as a channel | `correlatable: false`; the report notes "diagnostic envelope: it never enters a correlator" |
| D11b | frames with zero valid pairs written as `0.0` | a placeholder zero enters the mean and the correlator | weight 0; fixtures export `count = 0`; an undefined number is `null`, never a zero |
| D11c | `gvar(0, 0)` for undefined effective-mass points | a point with zero error dominates any fit through it | `None` |
| D11d | electroweak operators do not mask self companions | a self companion contributes the phasor $1+0i$ with amplitude 1 | self companions are masked everywhere and counted as `masked_self` |
| D11e | `projection_mode` silently ignored in standard mode | the configured projection is not the one measured | the projection applies in every displacement mode |
| D11f | independent amplitudes $a\ne b$ fitted to an autocorrelation | a correlator of one observable with itself is given two amplitudes | $a=b$ |
| D11g | momentum basis $k_n=2\pi n/L$ on a non-periodic domain with $L$ taken from the data range; `sin_0` identically zero | a Fourier mode of a box that does not exist; a channel that is zero by construction | momentum weights require `Record::PeriodicBox`; the mode-0 sine is rejected by validation |
| D11h | fastfit-seeded and data-scaled priors | the data are used twice and the prior is not independent | the amplitude prior is a single wide Gaussian about $\ln\lvert C(t_{\min})\rvert$ and is reported as reading the data; the gap prior is scale free |
| D11i | operator series computed in float32 | frame means of order `0.1` carry about `1e-7` absolute error, up to `1.7e-5` relative | f64 throughout; an f32 archive is refused, with no precision fallback |
| D11j | `mass > 0` filter and `min_mass` in the AIC scan | noise is biased positive: only positive rates can be averaged | a rate whose significance is below `min_rate_snr` (default 2) is reported as "no signal" |
| D11k | dead API `build_channel_group_models` with a hard-coded `max_lag` | — | not ported |
| Q30 (tessellation fixtures) | the emergent metric regularizes the displacement covariance with a hard-coded absolute `epsilon_numerical = 1e-5` identity (`geometry/hessian_estimation.py:605-607`), which is not a multiple of the covariance scale | the exported tessellation fixtures carry that convention, so under the shipped Rust default they no longer agree: `metric[0]` is `353.1274750061289` against `351.2450676772849`, and the largest relative departure is `6.7e-2` in `cloud_2d` and `4.2e-3` in `cloud_3d` (`6.7e-2` and `2.1e-3` on `det g`). The cause is measurable on the same clouds: under $x\to\lambda x$ at $\lambda=10^{-3}$ the fixtures' convention leaves `det g` short of its covariant value by `4.4e3` to `5.6e6` (2D) and `9.5e3` to `2.0e8` (3D), while the default holds the $\lambda^{-2d}$ law to `1e-12` | the ridge is `ridge * tr(C)/d` and the eigenvalue bounds are divided by the same scale (`RidgeScale::RelativeToTrace`, the default). `tessellation_parity.rs` selects the reference's `RidgeScale::Absolute` and keeps full `1e-8` parity under it — the `1e-8` agreement quoted under "Tessellation geometry and the Einstein–Hilbert gas" above is that convention — and `the_default_ridge_scale_departs_from_the_reference_metric` pins the numbers above for the default, together with the fact that the topology, the edge distances and the `uniform`, `inverse_distance`, `inverse_volume` and `kernel` weights are bit identical between the two |

What parity *does* cover, from the same fixture set (three cases, $T=12$ frames, $N=9$–10,
$d=3$, f64): the per-pair $q_{ij}$ and per-triplet $b_{ijk}$, $\Pi_{ijk}$ with their validity
masks to `1e-12`; the scalar, axial, `vector_unit`, baryon and glueball frame series (f32 series to
`1e-6` absolute); the U(1) and SU(2) phase series to `1e-6` absolute, Python evaluating them in
complex64; the origin statistics, their FFT estimator and the contracted vector correlator to
`1e-10`; and the AIC window-scan table and its averages on a fixed synthetic correlator to `1e-9`.
Only the records the audit's operator table judged correct carry `status: audited` and are parity
required. One exported record per file is `non_book` (`baryon_det_abs`, the $\lvert\det\rvert$
Python default), and the rest are `extra` — ten in `non_involutive.json` and `masked.json`, seven
in `mutual_pairing_odd.json`, namely the score-directed and score-weighted meson and vector
variants, `baryon_score_signed`, `baryon_flux_action`, `fitness_phase` and `clone_indicator`. A
mismatch on a `non_book` or `extra` record calls for investigation, not for a change in Rust.
Walker-type splits, `velocity_norm_*`, tensor, Dirac, twistor, chirality and multiscale channels
are outside this fixture set entirely.

### The full defect ledger: Q1–Q80

The D-numbers above are the first audit. The second pass audited the same Python pipeline
operator by operator, statistic by statistic, and deduplicated 217 filed findings and the eleven
D-numbers onto **80 distinct defects by root cause**; the D-numbers survive as aliases. Twenty-one
of the 80 are *critical* in the sense that a number produced, plotted or exported under a
reachable shipped configuration is wrong, forty-one are wrong under a reachable non-default
setting or whenever a public estimator is called, and eighteen have no numeric consequence and are
labelling, dead code or an unreachable configuration.

The table is the ledger of what the Rust does. A row states the Python behaviour with the
consequence measured for it, and the rule the port implements instead.

Two marks qualify the right-hand column, and both mean that it is **not** a statement about code
that runs today. Rows marked ‡ are defects the first port **inherited** from the reference
implementation: for those rows the right-hand column is the correction the audit requires of the
module that owns the behaviour, not a behaviour this document certifies as present. Before relying
on a ‡ row, read the module it names and the test that pins it; where neither exists yet, the port
still has the Python behaviour with the consequence in the left-hand column. Rows marked § have no
counterpart in the port at all: their rules are in "Rules for the subsystems that are not ported"
below, so that a later port inherits them rather than the code. An unmarked row is a correction
that is in the tree and carries a test named in the audit's register.

| id | aliases | Python behaviour and its measured consequence | What the port does instead |
|---|---|---|---|
| Q1 | D8 | a multi-component series is reduced (mean, norm or `Re`) before the lag product: `C(0)` is the variance of one reduced scalar, five of nine default electroweak channels lose `Im` (effective-mass shift 1.07 %), and a maximally anti-correlated vector series has identically zero autocorrelation | components are kept and the correlator contracts them, $C(\tau)=\sum_k C_{kk}(\tau)$ |
| Q2 | D1 | exchange-odd operators cancel identically on the mutual pairing: pseudoscalar and vector series are float32 roundoff (`1.3e-8`, `1.5e-8`) against scalar `0.14` and axial `0.35`, and `su2_doublet_diff` $\equiv0$ while `su2_doublet` $=2\,$`su2_component`, all shipped as independent channels | an `Odd` operator has no frame mean on an involution: weight 0 on every frame, `GasError::Capability` with the algebraic reason, and the source-frozen propagator where one was measured |
| Q3 | correlators-1 | the source-frozen propagator subtracts $\bar O^2$ instead of centring both legs: lag 1 has the opposite sign to the book (`-0.6436` against `+0.5129`), and two glueball modes that are affine images get different propagators | the global-mean form $\sum ab/n-\bar O(\bar a+\bar b)+\bar O^2$, or per-leg lag means |
| Q4 | correlators-2 | the propagator correlates each walker's mean over its pairs, a product of means; the element population is invisible | the accumulator sums per element, $ab[\ell] \mathrel{+}= w\sum_k \mathrm{src}_k\,\mathrm{snk}_k$, and reports the element counts |
| Q5 ‡ | aic-gevp-flow-10 | the GEVP takes the largest eigenvalue at each lag independently: the fitted ground rate is `0.18349` against a truth of `0.2` at $\sigma=0.002$ (−8.3 %) and `0.03497` (−82.5 %) at ten times that noise | one eigenvector solved at a reference lag, $\lambda_n(t)=v_n^\mathsf{T}C(t)v_n/v_n^\mathsf{T}C(t_0)v_n$, which returns `0.20004` on the same data |
| Q6 ‡ | correlators-11 | the score-derived orientation of a *frozen* element is re-derived from sink-time scores: the fitted rate is $m_{\text{colour}}+m_{\text{score}}$, a frozen colour field whose true correlator is flat drops by 176× at $\tau=1$, with a sign flip | the orientation is frozen with the element at the source evaluation, or the score-sign autocorrelation travels with every rate of a fitness channel |
| Q7 ‡ | baryon-glueball-10 | the momentum projection multiplies the *raw* operator, so it publishes the normalized walker-density Fourier mode: `re_plaquette` and `one_minus_re`, one channel of the book, give rates `0.16658` and `0.07284`, a factor 2.29 | the projection is applied to the connected element $O_I-\bar O_t$; both modes then give `0.18233` |
| Q8 | D2 | PDG-tuned `dE_ground` priors reproduce the PDG ratios by construction ($\pi/N$ `0.150` against `0.149`) | one channel-agnostic scale-free gap prior and a `PriorDominance` diagnostic on every level |
| Q9 | mass-extraction-1 | the fitted covariance is rank deficient by construction and `svdcut` fabricates its dominant constraint: `scatter/quoted = 17.58`, 1σ coverage `0.1` | the rank is reported against the number of lags, an active SVD floor is stated to decalibrate $\chi^2$, and a window the resample count cannot support is refused |
| Q10 | D3 | error bars bootstrap individual time points i.i.d.: AR(1) with $\rho=0.9$ gives `C(1)/C(0) = 0.028` instead of `0.898` | whole origin blocks of cross-lag products are resampled from one addressed stream |
| Q11 | D6 | block size hard-coded to 10 and silently clamped, and the jackknife reuses the full-sample disconnected part: 1σ coverage `0.41` at $\rho=0.95$ | per-block first moments, each resample subtracting its own disconnected part, `BlockSize::Auto` $\ge 2\tau_{\mathrm{int}}$, with $\tau_{\mathrm{int}}$, the effective block and the covariance rank reported |
| Q12 ‡ | mass-extraction-19 | the block jackknife deletes time *origins*, not observations, so at lags beyond the block nothing is deleted: quoted/true `0.67`–`0.85` at $T=200$ | origins stay — excising observations overshoots (`1.13`–`1.22`) — and the straddling pairs and the lag-against-block relation are reported |
| Q13 | D4 | `epsilon_clone` doubles as the Gaussian interaction range: at `1e-8` four SU(2) and mixed channels are identically zero, and `epsilon_d` falls back to `fitness.std()`, a fitness scale used as a length | `ElectroweakScales` keeps $\varepsilon_c$, $\varepsilon_d$ and $\varepsilon_{\mathrm{clone}}$ apart; a range is a Gaussian companion width or a fixed value, or the channel is unavailable |
| Q14 | D5 | post-clone fitness beside pre-clone companions and scores: 9 of 10 cloners have the fitness of their companion, 28 % of all SU(2) phases are identically zero, and reconstructing the cloning score from the fitness agrees on 42.9 % of entries | four explicit `ColorAlignment` arms, each with its own mask, so one velocity enters both the force and the phase |
| Q15 | wiring-22 | the advertised ensemble pair $\nu=0.948271$, $\ell_{\mathrm{visc}}=0.00976705$ realizes $\alpha_s=1.9459\times10^{-6}$ against the target `0.1179` — 60 000× too small — because $\nu$ presupposes $\langle K^2\rangle=0.823813$ while the kernel at that width measures `1.3597e-5`; nothing states or checks $\langle K^2\rangle$ | $g_3^2$ is computed from the **measured** $\langle K_{\mathrm{visc}}^2\rangle$ of the warm-up, the calibration check prints the moment the configured $\nu$ presupposes beside the measured one, and a factor beyond two is flagged as an inconsistent calibration |
| Q16 | geometry-N1 | the recorded "fitness manifold" geometry contains no fitness: the gas passes `fitness_values = zeros`, so the metric is the inverse neighbour second moment, bit-identical for three wildly different fitness fields | the metric estimator is named in the coupling report, `MetricKind::HessianFd { scalar_field }` is the arm that reads a field, and no row claims a fitness-manifold distance |
| Q17 | geometry-N2 | at the production length scale a third of the kernel weights underflow to exactly 0 in float32 (3099 of 9234 edges of a recorded $N=48$ run); one walker's whole kernel row sums to 0 | f64 throughout, with the squared geodesic length floored at `1e-12` so an affinity is never exactly zero |
| Q18 | geometry-1 | recorded edge *weights* are handed to the shortest-path solver as edge *lengths*: $r(\text{weight},\text{true length})=-0.63$, half the pairs at distance exactly 0, the whole ladder on the `min_scale` clamp and 98 % of pairs admitted at every scale | `GraphSnapshot` carries `euclidean_length` and `geodesic_length`; an affinity is never a length |
| Q19 § | electroweak-2 | the chirality connected correlator subtracts $\bar\mu^2$ instead of $\langle\mu_i^2\rangle$ and drops constant-chirality walkers: the plateau is the un-subtracted disconnected piece and `fermion_mass` is off by a factor `16.70` | no per-walker chirality autocorrelation is ported; the rule is recorded below |
| Q20 | electroweak-4 | `ew_mixed` is a product of four neighbour averages, not the triplet product | the triplet product is evaluated per element |
| Q21 | electroweak-6 | `_fit_exponential_decay` selects only positive-`C` lags and clamps the mass, biasing every rate positive and hiding a rejected model | fits go through the real covariance and report "model rejected" instead of clamping |
| Q22 | electroweak-10 | the emergent Weinberg angle uses the wrong two scales and drops the group factor | $\sin^2\theta=g_1^2/(g_1^2+g_2^2)$ on the proxies with the $2C_2(2)/C_2(d)$ Casimir factor |
| Q23 | electroweak-11 | validity and alive masks are absent or applied *after* the reduction: dead walkers become cloners, `avg_alive_walkers` is `N` unconditionally, a walker with no neighbours enters a frame mean as an exact 0, and one non-finite value poisons a whole frame | masking precedes every reduction; an empty reduction is missing data with weight 0, never a value |
| Q24 | inputs-7 | invalid companion indices are clamped or wrapped onto real walkers and self companions are not masked: `-1` aliases walker $N-1$, and self companions enter the $\ell_0$ median as exact zeros (`1.6174` instead of `1.7159`) | an out-of-range or self companion is a `CompanionMask`, never a substitute slot |
| Q25 | inputs-1 | the colour state is normalized in $\mathbb C^d$ and then truncated to $\mathbb C^3$ without renormalizing: at $d=4$ the norms spread over `[0.137, 1.0]` and $\Pi_{iii}$ falls to `6.7e-6` where the book has 1 | there is no projection step; the $\mathbb C^3$ channels are refused unless $d=3$ |
| Q26 | meson-vector-1 | `score_directed` orients $q$ but not $r$: on an involution the axial channel goes from `[0.107, 0.048, 0.129]` to zero and the vector one is not repaired. This refutes D1's claim that score direction is the repair | both factors carry the orientation, or the arm does not exist: the J=1 family has no `ScoreDirected` displacement arm |
| Q27 ‡ | meson-vector-2 | `score_weighted` does not lift an odd parity and collapses to exact zeros when every score is tied: `scalar_raw = [0,0,0,0]` with 51 valid source pairs, presented as a healthy channel | a frame whose score-weight sum vanishes has no value: weight 0, with $\sum_I w_I\lvert\Delta S_I\rvert$ published as coverage |
| Q28 ‡ | meson-vector-3 | the "score gradient" is missing the $1/\lvert r\rvert$ of the finite difference: it has the units of a score, is invariant under $x\to\lambda x$, and is `30.96°` off the true direction on a linear-field oracle — and it is also the projection axis | $\nabla S\approx\operatorname{mean}_p(\Delta S_p/\lvert r_p\rvert)\hat r_p$ |
| Q29 | D11i | single precision throughout, including positions before the displacement subtraction: the vector series loses translation invariance, 16.7 % error at an offset of `1e5` | f64 end to end; an f32 archive is refused, with no precision fallback |
| Q30 | geometry-10 | absolute regularizers where a scale-covariant one is required: the ridge does not scale with the covariance, so $g$ stops scaling as $\lambda^{-2}$ — a factor ~370 at $\lambda=10^{-3}$ — and `h_eff`, `mass` and the momentum length are clamped rather than refused | every regularizer is relative to a trace or a median, and a non-positive `h_eff`, `mass` or box length is refused |
| Q31 | geometry-20(a) | an indefinite metric is silently made positive definite ($1/v$ for $v<0$, then a floor), and is reported with a finite positive determinant and no flag | the clipped eigenvalues are recorded, and a strict policy makes an indefinite metric a numerical error |
| Q32 | geometry-N13 | `neighbor_graph_update_every > 1` records a stale tessellation for several frames with no flag, so a scale gate compares moving pairs against a frozen geometry | the snapshot carries its staleness, and the gated channels either reuse the recorded table with that provenance or decline |
| Q33 | D11b | a frame with no valid entry is written as a real `0.0`: a ~30 % shift at the fitted lags in the measured instance | weight 0 and no value; an undefined lag is `null` |
| Q34 | D7 | invalid or empty frames are deleted and the survivors correlated as if contiguous, so every lag is shifted | the full time axis is kept with per-frame weights, and a gap opens a segment that no lag spans |
| Q35 | tensor-dirac-4 | the source-pair-frozen correlator re-applies the multiscale gate at the *sink* time: the valid-pair population shrinks with the lag and is selected on the observable (`[72, 53, 34, 15, 0, 0]`) | the gate fires once, when the frame becomes a source; the stored element list is re-evaluated on the sink state |
| Q36 | tensor-dirac-1 | the 5-component traceless tensor basis is not orthonormal — the three shear components are missing $\sqrt2$ — so $\sum_\alpha$ is a coordinate-dependent bilinear form, not $\operatorname{Tr}(QQ')$ | no traceless symmetric basis is ported; the tensor channel is the $\operatorname{Im}[c^\dagger\sigma_{\mu\nu}c]$ wedge |
| Q37 | tensor-dirac-13 | channels that are bit-identical are shipped as independent observables, so a joint fit or a GEVP basis holding both inherits an exact null direction | one id per observable, duplicate ids rejected, and an arm that cannot exist is a `Capability` error |
| Q38 | tensor-dirac-14 | every Dirac baryon channel changes under a colour rotation and the nucleon changes sign; `parity_projection` sums four spinor components and calls it a scalar | channels are labelled by algebraic definition and verified invariance; the ported lift declares its non-equivariance |
| Q39 | D9 | $J^{PC}$ labels asserted by hand that the operator does not have | algebraic labels plus the measured exchange and spatial parity; physical names only as an explicit assignment |
| Q40 | D10 | "Wilson flow" is convex neighbour averaging on a degree-$\le2$ graph with a non-standard $w_0$, reported in length units it does not have | a graph-smoothing diagnostic with `steps` and `roughness` and no length claim |
| Q41 | mass-extraction-5 | a non-converged fit is reported as an ordinary number with an error bar | a reported rate requires a converged minimizer |
| Q42 | mass-extraction-8 | `excited_masses` can hold a level below the ground state, so a published spectrum can be inverted | $E_n=\sum_{m\le n}\exp(p_m)$, strictly increasing by construction |
| Q43 | mass-extraction-11 | prior handling: any gap prior narrower than 30 % is widened, a `0.5(5)` amplitude prior is `7e3`× off at an `O(1e-9)` data scale, the prior is round-tripped through `str(gvar)`, and a successful `fastfit` discards the configured prior | the prior is the configuration verbatim; the amplitude prior is one wide scale prior that states that it reads the data |
| Q44 | mass-extraction-3 | auto-detected groups force one shared gap on provably different rates, and the aggregate $\chi^2$, dof and $Q$ sum per-group fits of the same frames as if independent | explicit groups only, and one joint $\chi^2$/dof/$Q$ from one joint resample table, so a duplicated member drops the rank instead of adding information |
| Q45 | D11c | `gvar(0,0)` written at an emitted $t$, and an `arccosh` guard that is discontinuous at ratio $\to1$ | an undefined point is `None` |
| Q46 | D11j | the AIC scan's `mass > 0` filter and `min_mass` bias the estimate positive and refuse most clean inputs: 110 of 300 clean single-exponential inputs produce a mass at all, mean `0.2412` against a truth of `0.30`, 1σ coverage `0.600` | no window is dropped for the rate it happens to give; below a signal-to-noise threshold the channel reports "no signal" instead of a rate |
| Q47 | aic-gevp-flow-2 | `_select_best_scale` compares AIC across different data sets | each geodesic scale is its own report with its own rate |
| Q48 | aic-gevp-flow-11 | the GEVP mass is fitted with unit weights and `mass_error = NaN`, an indefinite symmetrized $C(t_0)$ is swallowed by the caller, and the whitener leaves $\lambda(t_0)\ne1$ | the cross-lag moments are resampled over shared origin blocks and fitted on the real covariance; $t_0$ is solved against itself; an indefinite metric is a reported numerical failure |
| Q49 | inputs-10 | configuration parameters are accepted and silently ignored, and dead code paths raise at call time | one enum arm per documented option and `serde(deny_unknown_fields)` |
| Q50 | correlators-12 | one unusable channel aborts the whole mass extraction | a declined channel becomes `Availability::unavailable(reason)` and the rest of the report is produced |
| Q51 § | wiring-1 | the coupling-diagnostics and ensemble-statistics subsystem: a Gaussian fit for an exponential $\xi$, failed fits entering the mean as $\xi=0$, a one-hop "holonomy" $u_iu_j\bar u_i\equiv u_j$ that is not gauge invariant, a linear standard deviation of wrapped phases, `polyakov_spread` $\equiv0$, a unit-dependent regime score, and the anchor channel plotted as a prediction with exactly zero error | not ported, except that the anchor is excluded from its own prediction table and the U(1) frame mean is the circular resultant; the rules are recorded below |
| Q52 | inputs-4 | `bin_by_euclidean_time` no longer bins by Euclidean time and raises against its only caller, and the fallback frame mean divides by $N$ instead of the valid count: the momentum projection divides by a frame-varying denominator that is neither book normalization | both normalizations are arms and the Euclidean-time axis has its own estimator with its own bins; the presentation prints the normalization the report carries on every frame-mean rate row and two rates under different normalizations are never divided, but nothing yet fills that field from the measurement (see below) |
| Q53 | wiring-29 | the SU(2) operator divides by $\lvert F_i\rvert+\varepsilon$ where the gas and `def-cloning-score` divide by $F_i+\varepsilon$; the two differ on every negative fitness | chapter 04 wins by the precedence rule; the disagreement is an open convention below |
| Q54 | wiring-14 | `_downsample_history` pairs per-step arrays of one step with strided positions, and an irregular cloning comb is collapsed to one median spacing, so lag products are formed across gaps | every off-stride gap breaks the series, `time_step = stride·dt`, and a gap opens a segment no lag spans |
| Q55 | inputs-12 | `pair_selection="both"` double-counts a pair when the two companion maps coincide, and the `distinct` flag is computed and discarded | the two companion laws are separate element kinds and separate channels; within one role an anchor contributes total weight 1 |
| Q56 | baryon-glueball-8 | `det_abs` names two different observables in two code paths, so a channel's series and its propagator are different observables (a factor 2.6 in the rate) | two distinct ids with two distinct signatures; $\lvert b\rvert$ is labelled non-book |
| Q57 | baryon-glueball-11 | a numerical zero-guard is reused as a cut on the operator *value*, and `eps = 0` publishes $\arg 0=0$ as "maximally coherent" | a `const` link floor that no configuration reaches, tested on every link and never on the product |
| Q58 | D11g | momentum basis $k_n=2\pi n/L$ on a non-periodic domain with $L$ from the data range, floored at 1.0 | a periodic box is required and $L$ is the box; the mode-0 sine is rejected at validation |
| Q59 | D11a | the tensor RMS envelope can be FFT-correlated standalone | an envelope is non-correlatable by type |
| Q60 | D11f | independent $a$ and $b$ amplitudes for an autocorrelation, the reported amplitude being $a$ while only $ab$ is identifiable, and `fit_parameters` storing $\log(dE)$ | one amplitude per channel and level, and levels reported in the rate unit |
| Q61 | aic-gevp-flow-17 | a population standard deviation, the jackknife error and the walker-bootstrap spread added in quadrature, a walker bootstrap that drops ~38 % of companion pairs, and a remap that relies on undefined scatter ordering | one resample table over time-origin blocks, jackknife $(n-1)/n$ and bootstrap $1/(n-1)$; walkers are never resampled |
| Q62 | correlators-8 | the trailing $T \bmod B$ origins are silently dropped, the FFT estimator drops `Im` and one NaN destroys every lag, and `N_cut` is counted from all valid points anywhere | `div_ceil` keeps the short tail, lag sums are direct with per-element masks, and `usable` is the contiguous run from $t_{\min}$ |
| Q63 § | geometry-3 | the Hessian, gradient and Ricci estimator family: no $1/\ell^2$ in the Ricci proxy, a central second difference with unequal spacings whose contamination *diverges* as $h\to0$, a strictly diagonal "full" Hessian, a directional derivative off by $\lVert\Delta x\rVert/d$, a double volume factor and an unguarded rank-deficient local quadratic | these estimators are not in the spectroscopy port; `physics/geometry.rs` and `physics/fields/gravity.rs` are a separate subsystem with their own contracts |
| Q64 | geometry-2 | min-symmetrization makes an edge length depend on its endpoints' degrees, `symmetrize_metric = False` makes $d_g$ asymmetric, and a recorded weight of exactly 0 is a valid path cost | the midpoint metric $(g_i+g_j)/2$, symmetric by construction, and a smeared result renormalized to unit norm |
| Q65 | geometry-13 | `select_scales` emits duplicate rungs through a dead float32 guard, fabricates $n$ identical scales on an empty sample, and samples quantiles on a stride that aliases with the row length | equal rungs are merged and never nudged, an empty sample is an explicit error, and the sample is the upper triangle taken once |
| Q66 | inputs-9 | three $\ell_0$ estimators use three windows and two reductions with silent fallbacks, and the calibrated $\rho$ serves both as the viscous kernel width and as $\ell_0$ | `LengthScale` arms pooled over all warm-up frames, a missing input being a `Capability` error, and $\nu$, $\langle K^2\rangle$, $\varepsilon_c$, $\varepsilon_d$ and $\rho$ kept as distinct rows |
| Q67 | electroweak-26 | the three-role `frame_global` partition is degenerate — the persister cell is one walker or none — and two spinor entry points with the same name pair walkers by different companions | the book's four roles, which partition the living set, and the element kind in the channel id |
| Q68 | electroweak-25 | `_compute_electroweak_series` raises when no frame has cloning and disagrees with the chirality route about the frame set, and `cloning_frames_only` correlates a gappy series as uniform | full-frame series with masks; with gated cloning the roles come from the ungated score sign, and no available channel is an explicit refusal |
| Q69 | electroweak-20 | `lr_fraction` and `lr_coupling_mag` turn "undefined" into 0, plotting as a measurement a quantity the book proves is identically zero | a `Capability` decline whose reason says it is identically zero |
| Q70 | electroweak-7 | the SU(2) "gauge link" saturates to the constant phase $i$ | the book form of the SU(2) phase from the cloning score |
| Q71 | aic-gevp-flow-7 | two copies of the pipeline compute different pseudoscalars and build the per-scale correlator differently, and the dashboard's two multiscale buttons use different forks | one implementation; two definitions are two arms with two ids and two declared parities |
| Q72 | D11e | `projection_mode` is a dead parameter in two of three displacement modes | the projection applies in every displacement mode |
| Q73 | inputs-2 | `color_valid` is the mask of the untruncated vector, an infinite viscous force is reported valid, and the colour phase is never wrapped | one vector and one mask, validity requiring a finite norm above threshold and a finite force, with the aliasing fraction published |
| Q74 | inputs-new-1 | the `h_eff` *selection* and the mass *measurement* read two different colour fields, and only the selection path is correct | one colour producer and one frame state per variant, read by every operator |
| Q75 | baryon-glueball-6 | `flux_exp_alpha` is clamped at 0 in one path and hard-coded to 1 in another, so two exponents share one channel name | $\alpha$ is validated and a non-unit $\alpha$ is part of the channel id |
| Q76 § | baryon-glueball-5 | triplet coherence averages a *ratio* of determinants with an unguarded denominator: $E[1/\lvert\det_0\rvert]$ diverges logarithmically, running means `3.81` → `11.56` from $n=10^3$ to $10^6$ against a correct answer of 1 | the channel is not ported; the rule is recorded below |
| Q77 | wiring-15 | the calibration script applies the $\hbar_{\mathrm{eff}}=c=1$ shorthand to an arbitrary `--hbar-eff`, and presents an inversion as a measurement | $h_{\mathrm{eff}}$ is carried explicitly in every proxy and in the inversion, and the inversion table is labelled as inputs of a calibration |
| Q78 | wiring-25 | the ensemble recomputes the fitness instead of reading the recorded one, and `lambda_alg` is recorded and never resolved, so the algorithmic distance loses its $\lambda\lVert\Delta v\rVert^2$ term | the recorded fitness is read as recorded, and an unset $\lambda$ reads the role's donor-module distance |
| Q79 | wiring-11 | the AIC quality filter passes channels whose uncertainty is unknown and rejects channels whose uncertainty is measured and large, so the published table is selected against the honest channels | a rate without a finite positive error is never produced, and every exclusion is noted |
| Q80 | wiring-19 | "AIC from Correlators" pairs a pair-based propagator with a frame-mean series and the tensor channel with an unrelated envelope: `C(0) = 4.1625` against `0.5272` for the correctly paired channel | exactly one estimator per channel, recorded with the rate; an envelope is not correlatable |

The normalization of Q52 travels on `ChannelReport::normalization`: the presentation prints it
beside the estimator of every frame-mean and Euclidean-time rate, `reference::compare` refuses to
divide two rates that do not share it, and a `ReferenceRow` carries the one behind its measured
number. A channel whose measurement recorded no denominator prints "not stated by the
measurement"; nothing is assumed for it, and no such rate is divided by one that states a
denominator.

That last sentence is, at the time of writing, the whole behaviour of a real report. The
denominator a frame mean actually used is resolved inside the accumulator
(`Signature::normalization`, else `MeasurementConfig::normalization`) and is not retained on
`ChannelSeries`, so nothing fills `ChannelReport::normalization` and every measured channel prints
"not stated by the measurement". The wire field, the printed line and the comparison rule are in
place and tested against hand-built reports; carrying the resolved denominator out of the
accumulator onto the series, and copying it onto the report in `channel_report`, is the remaining
step. Until it is taken, no rate row of a real run states its normalization, which is the half of
Q52 that `09_qft_calibration` asks for.

### Rules for the subsystems that are not ported

The coupling diagnostics and ensemble statistics of `run_qft_validation_ensemble.py` and
`app/coupling_diagnostics.py` (Q51), the per-walker chirality autocorrelation of
`electroweak/chirality.py` (Q19) and the triplet coherence of `baryon_triplet_channels.py` (Q76)
have no counterpart in the port. The audit confirmed a defect in each; the rules below are what a
later port must implement, and they are recorded here so that the port inherits them rather than
the code.

- A correlation length $\xi$ of an exponential decay comes from a log-linear regression of
  $\ln C$ on $r$, never from a Gaussian fit.
- A failed fit is `None`. It never enters a mean as $\xi=0$, which pulls every ensemble average
  toward zero by the failure rate.
- A holonomy is a closed loop. The one-hop product $u_i u_j\bar u_i\equiv u_j$ up to a positive
  real factor is not gauge invariant and carries no information; use $\Pi_{ijk}$, and let one
  invalid transport step invalidate the whole loop rather than acting as the identity.
- The dispersion of wrapped phases is $\sqrt{-2\ln R}$ with $R=\lvert\langle e^{i\theta}\rangle\rvert$,
  never a linear standard deviation of the angles, and $\lvert e^{i\varphi}\rvert$ has no spread
  at all.
- Any regime score must be dimensionless: a score that changes when the length unit changes
  measures the unit.
- An anchor is an input. The channel that sets the scale is excluded from its own prediction
  table, and is never plotted as a prediction with zero error.
- A chirality connected correlator subtracts $\langle\mu_i^2\rangle$ per walker, not the square
  of the global mean, and a constant-chirality walker is a zero-variance datum, not an exclusion.
  The Python form plateaus at the un-subtracted disconnected piece and gives a fermion mass
  `16.70` times too small.
- A triplet coherence is the **ratio of means** $\sum\lvert\det_h\rvert/\sum\lvert\det_0\rvert$
  (or $\exp(\operatorname{mean}\log\cdot)$), with a **relative** degeneracy cut reported as a mask
  count. The mean of the ratio has no finite expectation.

### The strong sector of the published ensemble

One number of the published validation ensemble is wrong by more than any statistical effect in
this document, and it is a calibration, not an estimator. The advertised pair
$\nu=0.948271$, $\ell_{\mathrm{visc}}=0.00976705$ was obtained by inverting $\alpha_s=0.1179$
through $\nu=\hbar_{\mathrm{eff}}g_3/\sqrt{d(d^2-1)/12\cdot\langle K_{\mathrm{visc}}^2\rangle}$ at
$\langle K_{\mathrm{visc}}^2\rangle=0.823813178557852$, while the Gaussian kernel at that width
has $\langle K_{\mathrm{visc}}^2\rangle=1.3596989447251\times10^{-5}$. The realized coupling is
therefore $g_3=0.004945029050169188$ and $\alpha_s=1.945932764315832\times10^{-6}$, a factor
`60587.9` below the target; at $\ell_{\mathrm{visc}}=1$ the same $\nu$ gives
$\alpha_s=0.13414950641749082$. **Every SU(3)-sector conclusion drawn from that ensemble
describes a nearly free gas, not a confining one.**

`prop-qft-report-inversion` fixes $\nu$ only together with the moment of the kernel that will
actually run, so a $(\nu,\rho)$ pair carried over from an inversion made at another moment
describes another coupling. The coupling report states both numbers: `kernel_second_moment` is the
warm-up measurement, and the calibration check prints the moment the configured $\nu$ presupposes,
the measured one, their factor and the realized $\alpha_d$. A factor beyond two is reported as an
inconsistent calibration, and a run with no warm-up moment says that nothing checks the
presupposition instead of substituting a placeholder. A calibration must be iterated at the
measured moment and its convergence checked (`sec-qft-calibration-qsd`).

### Open conventions awaiting a decision

Each item is a genuine ambiguity that the book, the frozen contracts and the reference
implementation do not jointly settle. None of them is guessed silently: the current default is
stated, and where it is configured.

**Measurement conventions.**

| # | Question | Current default | Configured in |
|---|---|---|---|
| OC-1 | Which pairing does the book sentence "the selected `v_before_clone` frame is paired with its preceding `force_viscous` entry" (04_standard_model) mean? The reference implementation's $-1$ is an array-storage offset, so it computes the B1 force of step $t$ with the pre-clone velocity of step $t$; the frozen doc comment reads it as the B2 force of step $t-1$ | `PrecedingKick` (B2 force of $t-1$ with its own input velocity). All four readings exist as arms: `PrecedingKick`, `MatchedKick{B1,B2}`, `PrecedingForce`, `ReferenceOffset` | `MeasurementConfig::color = ColorSource::ViscousForce { alignment }` |
| OC-2 | Under `MatchedKick`, should the mask cover only the rows cloned at $t$, or also the uncloned partners of an inelastic collision, whose velocity also changed? | cloned rows only | `ColorAlignment::MatchedKick` |
| OC-3 | Which graph do scale gating, smearing and the `WarmupEdgeMean` length use? The only archived graph is the post-clone force graph, which on a cloning step disagrees with the frame's positions and has zero-length edges | the archived graph as it is | `ScaleConfig`, `LengthScale::WarmupEdgeMean` |
| OC-4 | Under the offset arms the colour phase uses the B2 input velocity of step $t-1$ while `FrameState::v` is the pre-clone velocity of step $t$. Operators that mix colour with velocity (Dirac, twistor, tensor) therefore mix two velocities. Should the frame state carry a velocity choice? | one `v` in the contract; the difference is stated in the channel notes | `contract::FrameState` |
| OC-5 | Is the colour mask threshold $\delta_c$ absolute in force units (the book's reading) or relative? The dense force is normalized by $N_{\mathrm{alive}}$ in Rust and unnormalized in the book, so an absolute threshold fires earlier at large $N$ | absolute, `1e-12` | `ColorSource::{ViscousForce,RecordedField}::threshold` |
| OC-6 | Is the frame normalization the valid-element count (chapter 04) or the fixed $N$ (chapter 08)? Only the fixed-$N$ observable has the transfer-matrix reading that licenses the word "mass" | `ValidCount` | `MeasurementConfig::normalization` |
| OC-7 | Which velocity and which walkers enter the reported colour phase wrapping $\kappa v$, and should a post-warm-up counter be reported as well? Only $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$ is observable; $m$, $\ell_0$ and $\hbar_{\mathrm{eff}}$ are not separately meaningful for the colour sector | warm-up only; $\kappa$ reported | `PhaseScale`, `Calibration` notes |
| OC-8 | What exactly is `WarmupCompanionMedian`: pooled or median of medians, which of $K>1$ companions, minimum image on a periodic box, and which rows excluded? | pooled median over the warm-up frames | `LengthScale::WarmupCompanionMedian` |
| OC-9 | `WarmupEdgeMean` on a run with no recorded graph: unavailable, or a silent fallback to the companion median as in Python? | unavailable | `LengthScale::WarmupEdgeMean` |
| OC-10 | A revived row is masked as a propagator source. As a sink at lag $\tau\ge1$ the slot holds a new incarnation: follow the slot worldline or mask it? The choice silently changes pair counts when deaths are frequent | `IdentityPolicy::Slot` | `MeasurementConfig::identity` |
| OC-11 | The first frame after a gap or an epoch change under an offset arm: invalidate the colours of that frame only, or drop the frame from every channel? | colours of that frame invalid | `fields::color` |
| OC-12 | Is the colour encoding offered at $d\ne3$? The reference implementation normalizes over $d$ and keeps three components, an undocumented map into $\mathbb C^3$; the book requires a separately specified map | not offered | — (`ColorSource` has no `color_dims` field) |
| OC-13 | `PrecedingKick` assumes nothing moves positions after the B2 stage. `GasConfig::viscous_euclidean` inherits `position_diffusion = 0.1` from the Euclidean Gas, so its colour force belongs to positions displaced by $0.1\sqrt{h}$ rms per coordinate. Accept and document, or give the viscous variant zero position diffusion? | accepted and documented | `GasConfig::viscous_euclidean` |
| OC-14 | Graph viscosity records the first quarter-kick evaluation only, at the edge weights of the last geometry refresh — possibly a graph cached several steps earlier. The colour of a graph-viscosity run, including the Einstein–Hilbert preset, is the force of a stale graph | documented, not corrected | `GeometrySchedule` |

**Book reconciliation.** The audit rule was to implement every documented definition as an arm,
default to chapter `04_standard_model.md`, and list the disagreement here rather than choose
silently.

| Question | Current default |
|---|---|
| Chapters 04 and 09 define different pseudoscalar, vector and glueball operators ($\operatorname{Im}q_{ij}$ against $\sum_a c^*(\gamma_5)_{aa}c$; $\operatorname{Re}q_{ij}r_{ij}$ against $d^{-1}\sum_\mu c^\dagger\gamma_\mu c$; $\operatorname{Re}\Pi_{ijk}$ against $\sum_i\lVert F^{\mathrm{visc}}\rVert^2$). Which is normative? | chapter 04; the chapter-09 forms are additional arms |
| Is $\lvert\det b_{ijk}\rvert$, the reference implementation's default baryon mode, intended? It is not a book mode; the book gives $\operatorname{Re}b$, $\operatorname{Im}b$ and $\lvert b\rvert^2$ | offered and labelled non-book |
| One action scale or two: is $\hbar_{\mathrm{eff}}$ of the U(1) fitness phase also the $h_S$ of the SU(2) cloning phase, as the reference implementation assumes? `thm-sm-su2-emergence` states a separate scale | `h_s: None`, i.e. $h_S=\hbar_{\mathrm{eff}}$ |
| Is the $d^{-1}\sum_\mu$ component averaging of chapter 09 intended, given that it is not rotation invariant (D8)? | components kept and contracted |
| Which estimator does the word "mass" refer to in the calibration chapter? | the frame correlator; the other two are labelled with their own quantity strings |
| What role do the PDG targets $R_{\rho\pi}=5.5$, $R_{N\pi}=6.7$ of `09_qft_calibration.md` play? | none in any fit; the default `assignments` map (pion, f0_500, rho, a1, nucleon, glueball_0pp) and the `nucleon` anchor are an input hypothesis, labelled `hypothesis mapping` in every comparison |
| The book text at 04:376-379 describes the reference implementation's array offset and is in tension with `thm-sm-su3-emergence` (one $v$ in force and phase) and with the same-stage viscous feature map used later in the same chapter | `PrecedingKick`; the tension is unresolved in the book |
| The role-swap proposition R04-5 is proven but has not been confirmed on a recorded run; the provenance of the numbers in the "Empirical calibration status" passage (R09-12) is unknown | open |
| `04_gas_variants.md` cites `src/canonical.rs` and `src/tessellation/presets.rs`, which are now `src/variants/euclidean.rs` and `src/variants/einstein_hilbert.rs`; `GasConfig::viscous_euclidean` has no cited location | open |

**Conventions of the 80-defect ledger.** Six of the twelve open conventions the second audit
raised are the ones already listed above: the chapter 04 against chapter 09 operator definitions,
the meaning of the word "mass", $\lvert\det b\rvert$ as a baryon mode, one action scale or two,
the $d^{-1}\sum_\mu$ component averaging and the role of the PDG targets are all in **Book
reconciliation**; the colour time alignment is OC-1 and the frame normalization is OC-6. The
remaining ones are new.

| # | Question | Current default | Configured in |
|---|---|---|---|
| OC-15 | The SU(2) denominator: $\lvert F_i\rvert+\varepsilon$ (`SM.U1` of 04_standard_model) or $F_i+\varepsilon$ (`def-cloning-score` of 03, and the gas itself)? They differ on every negative fitness entry, and the signed denominator can flip the phase sign. This is the sharpest disagreement in the set | chapter 04 by the precedence rule; `fields::score` keeps both formulas for their two roles, and the SU(2) phase uses the absolute one | `fields::score` |
| OC-16 | Does `baryon/score_ordered` survive as a propagator channel? With the ordering frozen at the source evaluation it is bit-identical to `baryon/real`, so it would belong among the frame-mean channels only. That is a channel-list change, not a code change | open | `PropagatorConfig::enabled_for`, the channel list |
| OC-17 | The jackknife deletion geometry. Deleting time origins understates the error at lags beyond the block (quoted/true `0.67`–`0.85` at $T=200$); excising the observations themselves overshoots (`1.13`–`1.22`). Neither is unbiased | origins are deleted, which is the lesser bias, and the shortfall is now published: `Estimated::notes` states the straddling-pair count and the lag-against-block relation, and `channel_report` puts it in `ChannelReport::notes` of every analysed channel | `BlockSize`; `estimators::DELETION_GEOMETRY_NOTE` |
| OC-18 | Under an offset colour alignment, pre-clone companions and positions are combined with post-clone colours. Is that combination intended, or should the frame be dropped? | the combination is kept and stated in the channel notes (see OC-2 and OC-4) | `ColorAlignment` |

The comparison notes that accompany every reference table state the remaining interpretation
limits directly: tensions carry no look-elsewhere correction for the number of rows or for the
choice among operator variants, fit windows and assignments; channel rates estimated on the same
frames are treated as uncorrelated because the report carries no cross-channel covariance; ratios
of rates are invariant under the time unit assigned to a lag
(`thm-qft-ratio-rescale`) but not under the integrator step, the recording stride, the
estimator, the frame normalization or the smearing scale, so only channels sharing the time unit,
the estimator, the scale and the frame normalization are compared; and the
electroweak assignments follow the legacy dashboard mapping, which the book itself records as not
reproducing electroweak mass ratios and treats as phase-coherence diagnostics. The Standard Model
coupling map is an inversion from reference inputs to gas parameters, not a measurement, and is
presented as such.

**OC-16 resolved (Task A2 of the 80-defect ledger).** The score-derived factor of a frozen
element — the orientation of `meson/*/score_directed`, the column order of `baryon/score_ordered`,
the dispersion of `meson/*/score_weighted` — is now evaluated once, at the source frame, and
reapplied at every sink, so no rate mixes colour decorrelation with the score crossings of the
sink frame. The stated consequence is accepted: a frozen order enters a lag product squared, so
the source-frozen propagator of `baryon/score_ordered` is the one of `baryon/real` and that arm is
**no longer propagated** (`Signature::propagatable = false`); it survives as a frame-mean channel,
where the canonical order is what stops `Re b` cancelling under relabelling. The same identity
holds for `meson/*/score_directed` against `meson/*/standard` — measured equal lag by lag to
`1e-12` on an eight-walker, forty-step record while the arm still published a propagator — so the
remediation merge resolved that half the same way: both directed meson arms now carry
`Signature::propagatable = false` and survive as frame-mean channels only. The rejected alternative was to
keep the sink re-derivation and publish the contamination — a diagnostic channel of
`E[sgn(ΔS_t) sgn(ΔS_{t+τ})]` with its `τ_int` attached to every rate of a `Record::Fitness`
channel — which would have left every such rate equal to `m_colour + m_score` and asked the reader
to subtract it.

**Momentum mode 0 is refused (verification of Task A1).** The momentum projection is taken of the
element minus the frame mean, `Ã_t = Σ_I w_I (O_I − Ō_t) e_I / Σ_I w_I`. The cosine of mode 0
weights every element by `e_I = 1`, so that sum is zero by construction, exactly as the sine of
mode 0 is. Both phases of mode 0 are therefore refused: the sine by `Momentum::validate`, the
cosine by `glueball::signature` with the reason *the connected projection on momentum mode 0
vanishes identically; measure the unprojected arm for the zero mode*. The unprojected arm is the
zero-momentum observable, and its source-frozen propagator already subtracts the lag means. Before
the connected projection, the cosine of mode 0 reproduced the unprojected frame mean and was
recorded as the same observable; after it, an unrefused arm would have published a series of exact
zeros at full weight — the failure `meson/*/score_weighted` is refused for.

### Schema versions after the remediation merge

The remediation of the 80-defect ledger changed what several channels measure without changing
the name of any of them. House rule 14 checks a versioned container by equality and migrates
nothing, so the only instrument that can say "the same configuration now means different numbers"
is the version constant. One rule was applied to every container:

> A version moves when the current code would write different bytes for the same inputs, or when
> a field's absence in an older payload is indistinguishable from a value the current writer can
> produce. It does not move for a change that only makes a future trajectory differ, because a
> run's provenance already carries the configuration and the code that produced it.

| container | version | decision |
|---|---|---|
| `SPECTROSCOPY_VERSION` (`Measurement`, `AccumulatorState`, `SpectroscopyReport`, cached evidence) | **1 → 2** | Both halves of the rule fire. `Signature` gained `propagatable` and `auxiliary`, which have no defaults inside a `deny_unknown_fields` struct, so a version-1 measurement cannot decode at all and should fail as a version mismatch rather than as a serde error. More importantly the numbers moved under an unchanged configuration — a momentum mode projects the connected observable, a score orientation or dispersion is frozen with its element, a score gradient carries `1/\|r\|²`, a tied frame has no weight, a generalized eigenvalue follows a fixed vector, a floored fit reports the rank as its dof — and `MeasurementConfig::fingerprint` hashes this constant beside the configuration, so bumping it is exactly what stops a version-1 accumulator being resumed, a version-1 measurement being pooled, or version-1 evidence being reused, under a fingerprint that would otherwise match |
| `RunArchive::schema_version` | **2, unchanged** | Additive with a checkable default. `GraphSnapshot::stale_steps` defaults to `0`, which is the truth under every schedule that tessellates every step — every archive in the tree and the shipped default. Under `GeometrySchedule::PostClone{every > 1}` a `0` would be a lie, and the archive decides that itself: `RunArchive::validate` now refuses an archive in which two contiguously recorded steps hold the same tessellation while the later one still claims to have measured its own. The check is one-sided, so a clone-triggered refresh and a trimmed step both keep their legitimate stamps |
| `CHECKPOINT_VERSION`, `RNG_VERSION` | **5 and 1, unchanged** | No field moved and no byte in a checkpoint changed meaning: it holds raw state, not derived numbers. A resumed run continues with the corrected geometry, which is a code change the configuration and provenance already record, not a format change |
| the QFT parity fixture schema (`tools/export_qft_fixtures.py`) | **1, unchanged** | A separate contract for the Python-side fixture files; none of its fields moved |

### Residuals of the remediation merge

Each of these was measured and left in place deliberately; none is a silent number in a shipped
default path, and each states where it would bite.

1. **A rank-deficient neighbour set gets a ridge-dominated metric that is not flagged.** With
   `MetricKind::NeighborCovariance`, a walker whose neighbour displacements span fewer than $d$
   directions has a covariance eigenvalue of exactly zero, and the ridge is what is inverted
   there. For a walker in three dimensions whose neighbours are coplanar with unit spread in the
   plane, $\tau=\mathrm{tr}(C)/d=2/3$ and the default `ridge = 1e-5` give
   $g=\mathrm{diag}(0.99999, 0.99999, 150000)$ and $\det g = 149997$ — an inflation of $1.5\times
   10^5$ along the unresolved axis and a diffusion suppressed there by $387$ — while
   `MetricField::clipped` is `[false, false, false]`, because `pinv_clamped` sees a strictly
   positive eigenvalue and no bound moved it. The repair is not a one-line one: flagging it turns
   every such walker into a hard error under `MetricPolicy::Strict`, and a walker with two
   neighbours in three dimensions is ordinary. **Not fixed; the number is the regulariser, and
   `repaired()` does not say so.**
2. **The diffusion floor is absolute where the ridge is relative.** `MetricKind::estimate` floors
   the metric eigenvalues at `min_eig.unwrap_or(1e-6).max(1e-6)` before forming
   $g^{-1/2}$, in inverse squared length units, while the ridge and the eigenvalue bounds are
   multiples of $\tau$ under the default `RidgeScale::RelativeToTrace`. The floor binds for a
   metric eigenvalue below `1e-6`, i.e. a neighbour spacing above $10^3$ ambient units, and caps
   the diffusion at $10^3$ there: a factor $10$ low at spacing $10^4$, $10^3$ low at spacing
   $10^6$. The floor is one scalar for the whole field while $\tau$ is per walker, so making it
   covariant is a structural change, not a division. **Not fixed; it does not bind on any cloud in
   the tree.**
3. **A stale tessellation refuses the whole measurement, not only the scale-gated channels.**
   `scales::distances` returns `GasError::Capability("scale gating needs a tessellation refreshed
   every step")` and `Accumulator::ingest_step` propagates it, so a run configured with
   `MeasurementConfig::scales` and `GeometrySchedule::PostClone{every > 1}` fails at its first
   carried-over frame instead of declining the multiscale copies and measuring the rest. The
   failure is loud and produces no number. **Not fixed; the refusal is correct and its scope is
   wider than the ledger asked.**
4. **A small but non-zero score dispersion is invisible.** `meson/*/score_weighted` gives a frame
   whose weighted dispersion $\sum_I w_I\lvert\Delta S_I\rvert$ vanishes the weight `0`, which is
   the half of Q27 that produced exact zeros at full weight. The ledger also asked for the
   dispersion itself as a published coverage quantity, so that a nearly-degenerate frame is
   visible as well. `Coverage` is a struct of element counters merged by addition across
   replicas; an $f_{64}$ dispersion is a different kind of quantity and belongs to a design
   decision about that struct. **Not fixed; only the vanishing case is signalled.**
5. **The staleness check covers contiguous recorded steps only.** An archive written before
   `stale_steps` existed, recorded with a stride so that no two recorded steps are consecutive,
   and run under `PostClone{every > 1}` with graph recording, still decodes with every stamp `0`.
   The complementary rule — the stamp is `(step − 1) mod every` at a gap — cannot be checked
   one-sidedly, because a clone-triggered refresh and a trimmed leading step both produce
   legitimate stamps that the arithmetic alone would reject. **Not fixed; the combination needs a
   strided recording, a non-default schedule and a pre-change archive at once.**
6. **An archive older than `RidgeScale` claims the corrected convention.** `MetricKind` omits
   `scale` when it is the default `RelativeToTrace`, exactly as it omits `policy` when it is
   `Clipped`, so the `gas_config` of an archive recorded before the arm existed decodes as
   relative although its lengths were measured with the absolute ridge. The recorded numbers are
   unaffected — they are data of the run that produced them — but a re-measurement that intends to
   reproduce them must select `RidgeScale::Absolute`, which is what
   `tests/fixtures/tessellation/*.json` and `tests/tessellation_parity.rs` already do. **Not
   fixed; making the decode default `Absolute` would hand the Q30 defect back to any configuration
   that simply omits the field.**
7. **`window_scan` and `gaps` count degrees of freedom differently under an active SVD floor.**
   The multi-exponential fit reports `min(points, rank)`; `window_scan` still reports
   `rows − parameters` from `linear_fit`. Only `presentation.rs` reads `FitDiagnostics::dof`, for
   the displayed $\chi^2/\mathrm{dof}$, but the shipped $q$ of a floored group fit moves (`0.982
   → 0.377` on the reference channel at `svd_cut 0.5`). **Not fixed; the convention belongs to
   `numerics::least_squares`.**
8. **The Q9 and Q46 coverage studies are pinned by exact seeded counts.** The two ensemble tests
   assert `kept == 65` and `kept == 164` and golden means to `1e-4`, not the ledger's property
   triple, which the Python behaviours also satisfy on those rows. They are legitimate
   deterministic pins, but any change to `AnalysisConfig::default()` or to the resampling geometry
   moves them, and the failure lands in a file that change does not own. Both now run in CI: they
   take `0.87 s` and `0.86 s` in a debug build, so the `#[ignore]` the ledger prescribed for them
   was removed, in line with checklist item 41 and with the same decision already taken for
   `numerics_statistics.rs`.
9. **Q35 has no test.** The source-frozen propagator cannot re-gate at a sink by construction —
   `commit` stores the gated element list in the `SourceFrame` and `push_ring` re-evaluates that
   list — but the pooled `n[lag]` sums a different set of origins per lag even under correct
   gating, so the ledger's "equal at every lag" oracle does not hold. A workable oracle would
   compare `n[lag]` against the prefix sums of the per-frame element counts, which the public
   surface does not expose. **Correct, unpinned.**
