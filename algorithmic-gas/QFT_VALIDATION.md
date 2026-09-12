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

The exact full recorded algebra also has a stronger obstruction than the noisy temporal-reflection diagnostic: the [localized labeled-color proposition](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md#prop-ym-native-labeled-color-reflection-sign) gives a negative reflected expectation −(u+v)²/4 whenever the retained labeled localization has nonzero support. A fixed vertex cannot occupy both reflected half-spaces. Thus this particular full labeled algebra cannot supply the proposed positive physical reconstruction. Summed observables include cross-label terms and require their own analysis; the temporal VI-28 matrix tests a different reflected observable.

VI-51 reports only executed mechanical stage ledgers and conditional thermostat moments. Its regression compares the ledger with the archive and rejects prescribed two-particle restitution values, prescribed Ricci contractions and constitutive-example residuals from the run result.

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

The complete workspace run passed 293 tests. The native control sweep passed 258 Part V–VI seeded defaults and 1,455 individual control endpoints; the continuation follow-up passed 72 cases. Compiled and browser checks passed all 128 demos, evidence replay and all chapter embeds. The Theory build and portal assembly passed. The [lecture validation record](../fractal-gas-web/web/euclidean-gas/lecture/VALIDATION.md) gives the coverage. No claim of accelerator runtime parity follows from CPU/WASM checks.

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
| VI-51 | Complete mechanical balances and stress ingredients | Run archive |
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
