# Algorithm-derived QFT: measurements and validation

The field theory describes observables of the executed Fractal Gas transition. Its state includes the population, validity, donor memory, provider configuration and the recorded context needed by a field readout. Companion sampling, historical rescoring, literal clone writes, viscous kicks, BAOAB noise, spectral clipping and boundary operations all belong to that transition.

The 66 workbenches contain **23 archive measurements, three independent complete-checkpoint continuation experiments, and 40 finite-model calculations**. The latter calculate stated algebraic or analytic models; their successful tests do not establish those models as a law of the gas. Every exported result carries its calculation origin, the executed gas configuration when available, and the experiment's derivation contract. Passing an archive to an unsupported workbench produces an error instead of returning an unrelated fixture.

The [lecture guide](../docs/source/2_fractal_gas/partvi_experiments.md) describes placements and controls. The [machine-readable contracts](crates/algorithmic-gas/src/physics/partvi_contracts.json) cover all 66 workbenches. [Numerical evidence](validation/partvi-evidence.json) records the independent sampling units and configurations.

## Exact field evolution

For the complete-state transition kernel $K$, define

$$D_F=KF-F,\qquad Q_{FG}=K(FG)-(KF)(KG).$$

At any time, with $\mu_n$ the actual state law,

$$\mu_{n+1}F=\mu_nF+\mu_nD_F,$$

$$\operatorname{Cov}_{\mu_{n+1}}(F,G)
=\operatorname{Cov}_{\mu_n}(F+D_F,G+D_G)+\mu_nQ_{FG}.$$

These identities retain transient behavior. The lag correlation is $\mu_n[\overline F K^\ell G]$. They follow directly from conditional expectation and total covariance; the complete proof is `prop-ym-transient-algorithm-fields` in the Yang–Mills chapter. A reduced field predictor must approximate these conditional quantities and account for information omitted by its descriptors. Its accuracy is measured on separate trajectories or complete-state continuations.

A full-step increment is the sum of its recorded stage increments. Its variance contains all cross terms between walkers and stages. Cloning and shared donor choices therefore contribute their actual joint increments, while killing contributes through eligibility changes and the specified cemetery extension. The empirical field law and characteristic functional are pushforwards of this executed path law.

## Mathematical and implementation findings

### Correlator normalization and spectral interpretation

A source-frozen, valid-pair-normalized twistor statistic and a fixed-normalization frame observable have different correlation functions. The frame average contains cross-walker products. The valid-pair denominator also varies with lag. A transfer calculation for one does not establish a spectral representation for the other.

VI-36 uses a fixed $1/N$ sum of zero-extended local readouts in `frame_mean` mode. `source_pairs` supplies the separately named source-frozen statistic. The chapter proves their appropriate complete-kernel identity and gives the positive self-adjoint transfer representation as a separate conditional result. General complex fits retain phase, signed decay, and held-out error. A positive fitted rate alone does not assign a particle mass.

For seed 516 with donor memory two and viscosity active, direct and compiled frame Gram values both equal **0.8500623473105489**. The fitted finite-window mode has decay **−1.78624**, frequency **−18.08976**, and held-out complex RMSE **0.392445**. Its fitted magnitude grows over that window. The exported mass is null. This trajectory does not support a positive-decay interpretation for that fit.

### Historical sources and the conditional metric

A historical donor is identified by its immutable frame, version, slot and generation. Its slot number cannot be used to read the present population. Twistor and companion-of-companion observables resolve covered historical records explicitly. The metric provider differentiates the target against the actual frozen donor-pool position and velocity.

Historical metric reconstruction was checked against executed derivatives: **26 global queries** agreed within **3.55×10⁻¹⁵**, and **25 local distance-memory queries** within **3.11×10⁻¹⁵**. An absorbing-boundary test exercised **36 revivals**, including **29 historical distance contexts**, with zero reconstruction discrepancy.

The programmed revival sampler chooses an eligible current-frame source. That source's distance donor can be historical. Ordinary accepted cloning can use historical donors. These different rules are preserved. Local standardization with historical *cloning* remains an unsupported engine configuration; local standardization with historical *distance* donors is measured and tested.

### Material metric evolution

A field readout attached to the last pre-clone context alone omits the current step's clone replacement and subsequent movement of the observation point. VI-45's material readout evaluates the field at the resulting slot position and decomposes its change into field change, literal-clone probe replacement, and subsequent motion. The fixed-probe readout remains available as a different observable.

Independent calibration and validation groups retain all component means and their full cross-covariance. Summing that covariance reproduces the total increment covariance within **8.17×10⁻¹⁴** across seeds 11, 29 and 43. Missing reconstruction data produce a diagnostic; they are not silently converted to a defined zero metric.

### Noise, source means and conditional uncertainty

For an executed innovation $Y=B(\xi+s)$, the source mean is $\mu=Bs$ and the covariance is $BB^\mathsf T$. Its uncentered second moment contains an additional $\mu\mu^\mathsf T$. VI-18 reports these separately, uses the actual factor rank, and excludes ineligible zero-filled rows. Its fourth moment includes the innovation law's fourth cumulant, which is **−6/5** for standardized uniform innovations.

A recorded rank-one uniform test measured second moment **1.74313** against **1.75**, and fourth moment **10.30073** against **10.2375**. VI-22 and VI-48 additionally test conditional momentum and energy moments at the actual pre-O state. The full run retains every preceding clone, force, memory and boundary effect. Entire continuations are the independent sampling units; predictable variances are summed along each continuation.

Across 384 independent continuations, the measured/predicted thermostat martingale variance ratio was **0.97501**. Separate 64-run checks gave standardized cumulative energy residuals **−0.460** for Gaussian noise and **0.674** for uniform noise, with second-moment ratios **1.015** and **0.870**. If no continuation reaches a covered O stage, the check is reported as unexercised.

### Primitive path carriers and source response

A singular or low-rank factor does not supply a full-dimensional Lebesgue density for its output. The path calculation retains the actual latent innovation carrier when needed. Deterministic cloning and boundary maps have their own carrier; they are not assigned fictitious Gaussian densities. Every component states its carrier and whether its law is covered.

For an addressed Gaussian source, direct reruns and exponential likelihood reweighting test the same complete executed map, including threshold crossings, cloning and killing. With memory and viscosity active, the summed direct-minus-reweighted residual was **0.05749 ± 0.13110** (one independent standard error). An absorbing case retained three extinct outcomes among 192 direct replicas and gave **−0.000465 ± 0.077357**. Finite-source identities and derivatives at zero are distinct comparisons.

### Kicks, cloning and stress measurements

The executed kick is $\delta v=(h/2)f_{\rm total}$, where $f_{\rm total}=-D_{\rm provider}+f_{\rm viscous}$. A formula using only the provider gradient omits viscosity. The chapter's kick-work proposition uses total force; VI-51 checks it against raw stage velocities and keeps transport, boundary and eligibility terms in the weak momentum balance.

VI-02 reconstructs actual clipped gate probabilities and their predictable variance. VI-14 reconstructs every literal clone write from immutable donor records and retains later transformation and boundary budgets. Ordinary copying can change total momentum; conservation is assessed for the configured operation that supplies it.

### Gram spaces, channels and measured graphs

The measured Fock construction uses the positive quotient of the whole-record Gram matrix. Null directions do not become fermionic modes. A measured two-time contraction uses separate source and target marginal Gram matrices, with later held-out cross moments. It does not assume that two transient empirical marginals coincide.

VI-52 measures actual selected-interaction cuts by edge channel. A Gaussian graph reconstructed from projected positions is separately identified because it is a different graph. Its cut identity does not establish a cut law for the recorded interaction network.

The reference density-mode solver also controls its step using the fastest selected mode. All advertised combined control endpoints preserve positive decay and numerical stability.

## Independent large-run evidence and a failed closure

The [full physics verification](VERIFICATION.md) covers **544 trajectories**, **483,328 complete updates**, **78,381,056 walker updates**, **36,864 independent replica updates**, and **1,632 audited archives**. Those research configurations use donor history zero; the memory-dependent tests above exercise separate configurations. Archive reconstruction reproduces every saved balance and curvature summary. Kinetic telescoping residuals are zero.

Across 14 global configurations, the thermostat standardized residual ranges from **−2.7127 to 1.8343**. Independent metric comparisons have median absolute standardized residual **0.4831** and 95th percentile **1.9131**. Uncertainty is calculated within each configuration; repeated seeds across configurations are not treated as independent draws.

The tested common two-coefficient spatial curvature/stress predictor is not an adequate constitutive closure. Its global held-out RMSE is **13,340,129**, versus **13,414,636** for a constant baseline and **13,384,667** for shuffled training: only about **0.56%** improvement over the constant. Locally its RMSE is **51,363**, versus **31,600** for the constant and **32,372** for shuffled training. That local prediction fails the held-out comparison.

The algebraic Ricci contraction of an assumed Einstein relation does not derive that relation from the algorithm. The supported description is the actual conditional metric drift and covariance, together with the recorded mechanical transfers. A proposed stress closure must improve those predictions on held-out data while retaining relevant state and source dependence.

## Reproduction

Run the native suites from `algorithmic-gas/`:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The focused suites are `partvi_algorithm_channels`, `partvi_algorithm_paths`, `partvi_algorithm_fields`, `partvi_metric_replicas`, and `partvi_cli`. Browser tests use the same compiled Rust implementation. The lecture supports replayable JSON export, raw archive replay, and import of native result bundles. Large-run reproduction commands and figures are linked from `VERIFICATION.md`.

## Verification results

| Check | Result |
|---|---:|
| Rust workspace tests | 252 passed |
| Compiled Part VI checks | 76 passed |
| Engine, tracking and lecture-host regressions | 44 passed |
| Native browser-default configurations | 26 passed; 1,582 finite plotted points |
| Additional mechanism configurations | 5 passed |
| Browser controls, archive replay and mobile layout | Passed |
| Lecture embeds and Expert Mode | 128 figures passed |
| Rust formatting and Clippy | Passed |
| Theory book build | Passed |
| CPU and WebGPU WASM compilation | Passed |

Runtime validation uses CPU execution. WebGPU hardware execution is untested in this environment. The book build has offline DOI lookup warnings; the edited Part VI guide and formal statements build successfully.

## Per-workbench derivation and coverage

### VI-01 — Oriented diagrams and transport

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Oriented transport words supplied as unitary links

**Exact identity:** Word reversal is adjoint; rebasing conjugates a closed word and preserves its trace.

**Measurement:** Finite supplied SU(2) links; no executed gas connection is available.

**Validation:** Independent matrix products check adjoint and trace identities.

**Scope:** Native archive route unsupported: an actual transport connection must be constructed and recorded before this characterizes gas holonomy. Color rank-one projectors are not substitute SU(3) links.

**Theory anchors:** `thm-lqft-oriented-word-algebra`.

### VI-02 — Cloning antisymmetry and exclusion

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Actual directed cloning scores, configured epsilon/saturation, donor rescoring, clipped acceptance and revival

**Exact identity:** (Vi+eps)Sij+(Vj+eps)Sji=0; Pij=clip(Sij/saturation,0,1); E[Aij|recorded pre-gate inputs]=Pij.

**Measurement:** All ordinary recorded gates, exact pool-aligned donor fitness including historical rescoring, selected SourceRefs, realized gates, deterministic revivals separately.

**Validation:** Reconstruct probabilities independently of stored CloneChoice.probability; compute realized gate innovations and predictable sum p(1-p).

**Scope:** Custom clone-decision providers may differ; a nonzero stored-probability residual is a substantive mismatch, not silently forced to zero. Conditional gate variance does not assume independent temporal states.

**Theory anchors:** `thm-cloning-antisymmetry-lqft`.

### VI-03 — Exterior algebra from recorded observables

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Whole-swarm pre-clone frame observables and their empirical centered Gram quotient

**Exact identity:** {a(fi),a†(fj)}=Gram(fi,fj)I after quotienting null covariance directions; Fock dimension=2^retained_rank.

**Measurement:** Eligible swarm coordinate means per complete frame, centered with equal frame weights; all contributing epoch/step/version/slot identities retained.

**Validation:** Independently multiply explicit creation/annihilation matrices in measured positive eigenspace and compare projected Gram; independently sum replica determinant squared.

**Scope:** Requested truncation discards measured modes, reported by discarded covariance norm. Empirical frames are not iid population samples.

**Theory anchors:** `def-lqft-record-fock-space`, `thm-lqft-record-fock-reconstruction`, `prop-lqft-finite-record-transfer`.

### VI-04 — Recorded transitions and CAR evolution

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Actual two-time joint law of eligible whole-swarm velocity means

**Exact identity:** The conditional transfer between separately centered source/target marginal Hilbert spaces is a contraction. Its canonical singular-value vacuum dilation is UCP with defect I-C*C.

**Measurement:** Strict chronological training and held-out frame pairs; separate source/target covariances and whitening, empirical transfer singular values, covariance drift and held-out cross moments.

**Validation:** Explicit exterior-power fermionic Kraus operators and Choi construction validate CP and CAR; disjoint later pairs independently assess cross-moment drift.

**Scope:** A two-time measured transfer does not imply a compressed-state semigroup. Singular-vector bases are canonical coordinates; no stationarity is imposed. Insufficient or zero-rank data gives unavailable output.

**Theory anchors:** `prop-lqft-finite-record-transfer`, `thm-lqft-record-car-channel`.

### VI-05 — Regional covariance and locality

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Two spatial-region averages on the same full swarm record

**Exact identity:** Their CAR anticommutator coefficient equals measured centered covariance, and normalized overlap equals whitened cross covariance.

**Measurement:** Left/right position split and chosen scalar feature; paired nonempty frame regional means and validity counts.

**Validation:** Explicit two-mode CAR matrices independently reproduce measured normalized covariance.

**Scope:** Frames with an empty region are excluded and counts retained; no continuum locality or independence follows from spatial separation alone.

**Theory anchors:** `thm-lqft-record-locality-defect`.

### VI-06 — Products and replica wedges

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Same-record products versus independent replicas of the empirical whole-record law

**Exact identity:** Squared norm of det[fi(Yj)]/sqrt(2!) under independent record replicas equals det Gram.

**Measurement:** Same-frame product and all pairs of centered full-frame observables from actual records.

**Validation:** Direct double empirical-replica sum compared with separately evaluated 2x2 Gram determinant.

**Scope:** Independent replicas here are the product of the selected finite empirical law, not independence of walkers in one interacting swarm.

**Theory anchors:** `thm-lqft-replica-isomorphism`.

### VI-07 — Density-aware continuum operators

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md).

**Algorithmic object:** Finite point-cloud quadrature of the declared continuum density-weighted integral operator

**Exact identity:** Sample mean approximates the density-weighted kernel integral with the declared normalization.

**Measurement:** Independent generated point cloud and deterministic quadrature under a supplied density.

**Validation:** Existing regression uses independent Monte Carlo points versus quadrature and tests seed variation and sample-size error.

**Scope:** Native archive route unsupported: this is a declared continuum reference sampling law, not a fitted density/operator derived from the full interacting gas.

**Theory anchors:** `thm-lqft-record-locality-defect`.

### VI-08 — Recorded color states

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Actual three-dimensional viscous force and its matching input velocity

**Exact identity:** c_a=Fvisc_a exp(i kappa v_a)/max(||Fvisc||,delta), mask=1[||Fvisc||>delta].

**Measurement:** Latest executed B1 viscous force with exact B1 input velocity/version, per-slot availability, raw normalized vector and separate valid mask.

**Validation:** Independent raw norm, masked norm and strict stage/version validation; existing archive tests tamper viscous source generation/version.

**Scope:** This explicitly declared matched-B1 convention differs from the book Python preceding-force/current-preclone pairing; it must be named as its own observable map. Zero viscous support yields invalid colors, not normalized physical states.

**Theory anchors:** `def-sm-direct-observable-law`.

### VI-09 — SU(3) invariant coordinates

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Pair, baryon, triangle and rank-one projector contractions of actual valid color states

**Exact identity:** Common SU(3) preserves q,b,triangle; independent ray phases preserve triangle; triangle=Tr(PiPjPk); |b|²=det Gram.

**Measurement:** Consecutive valid triples from actual B1 color records, with complex values, masks and original source-slot mapping.

**Validation:** Independent dense projector matrix multiplication and Gram determinant, plus non-diagonal common SU(3) rotation and independent slot phases.

**Scope:** Triple selection is explicit and finite; no actual transport connection is inferred from projectors. Fewer than three valid colors gives unavailable triple result.

**Theory anchors:** `thm-sm-direct-color-invariants`, `prop-sm-direct-triangle-projectors`.

### VI-10 — Companion doublets

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Executed directed cloning companion scalar and companion-of-companion doublet

**Exact identity:** a_i=exp(-D_i²/(4ell²)) exp(i(Fk-Fi)/((|Fi|+eps)hS)); doublet=(a_i,a_k(i)); conditional scalar variance equals pairwise half-sum under the actual categorical law.

**Measurement:** Actual configured distance, rescored donor fitness, selected SourceRefs, own-companion map and immutable historical source scalar lookup. Exact normalized configured-kernel probabilities for Independent count-one cloning.

**Validation:** Conditional variance computed both about the complex mean and by pairwise differences; actual selected scalar innovations compared with conditional means; tests verify valid doublets from engine records.

**Scope:** For dependent mutual/greedy samplers actual doublets are measured but their joint-law variance is not replaced by independent-row variance. Historical companion scalar is masked only outside covered immutable source records. A full dependent-sampler conditional-doublet fluctuation estimator remains missing.

**Theory anchors:** `def-sm-direct-companion-doublet`, `thm-sm-native-doublet-fluctuations`.

### VI-11 — Attribution holonomy

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Holonomy from supplied unitary transport links

**Exact identity:** Closed word transforms by basepoint conjugation; trace is invariant.

**Measurement:** Finite explicit SU(2) transport fixture.

**Validation:** Independent ordered matrix products versus gauge transformed products.

**Scope:** Native archive route unsupported: executable transport must be defined from actual records. Scalar phases or rank-one color projectors do not establish the supplied SU(2)/SU(3) bundle connection.

**Theory anchors:** `thm-lqft-oriented-word-algebra`.

### VI-12 — Channel compression and memory

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Actual compressed whole-record descriptor and measured memory of its fitted transition

**Exact identity:** Projected two-step difference Pi P² Pi-(Pi P Pi)² equals the finite block memory term; a fitted compressed chain need not be Markov for actual evolution.

**Measurement:** Training-only quantile partition, actual chronological transition counts, coarse projection, held-out multi-lag matrices and Brier scores.

**Validation:** Compare later empirical conditional distributions with powers fitted on strictly earlier frames; finite projection matrix multiplication independently measures algebraic memory.

**Scope:** One quantile partition is not a nested prediction-completed sequence. Temporal rows remain correlated; no iid confidence bands. Fitted algebraic memory differs from full population memory.

**Theory anchors:** `thm-sm-direct-channel-memory`, `thm-sm-prediction-complete-descriptors`.

### VI-13 — Predictive partition refinement

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Actual finite descriptor predictor and held-out multitime law

**Exact identity:** Training transition powers define a proposed finite closure; held-out lagged laws measure its error.

**Measurement:** Strict chronological training/guard/test split, fitted partition edges, transition counts, multi-lag held-out conditional matrices, occupancy baseline.

**Validation:** Independent later records evaluate Brier scores and matrix RMSE; unknown transition rows remain unavailable rather than fabricated.

**Scope:** No claim of predictive partition convergence follows from one nonnested empirical quantile fit. Stationarity and sufficiency are measured/required, not inferred from training fit.

**Theory anchors:** `thm-sm-predictive-partition-convergence`.

### VI-14 — Complete updates and pair cancellation

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Actual immutable clone writes and subsequent transform/reconciliation/boundary stages

**Exact identity:** For accepted recipient i, literal velocity equals the weighted immutable donor source row. Net copying momentum is incoming donor contributions minus overwritten recipient momenta.

**Measurement:** All source references including covered historical records; literal-clone velocity and successive stage momentum/eligibility totals at unit mass.

**Validation:** Independently reconstruct every copied velocity from source populations and compare recorded literal_clone; retain measured later-stage increments.

**Scope:** All-slot momentum and eligibility counts are separate. General directed copying is not momentum-conserving; restitution conservation requires the actually configured reciprocal disjoint-pair transform. Out-of-coverage historical sources are a capability error.

**Theory anchors:** `prop-sm-implemented-collision-increments`, `cor-sm-physics-paired-cloning`.

### VI-15 — Coupling statistics and CP

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Declared charge-conjugation orbit fixture and symmetry-breaking proxy

**Exact identity:** An invariant ensemble with a declared odd observable has zero mean under its full involution.

**Measurement:** Finite conjugate-color orbit fixture; no full algorithmic CP action or invariant ensemble has been measured.

**Validation:** Fixture symmetry and a biased two-orbit mean are checked algebraically.

**Scope:** Native route explicitly unsupported. Conjugating a color or a zero empirical mean does not establish full CP invariance of positions, velocities, companion law, noise, clipping and history. No complete actual CP involution has been derived here.

**Theory anchors:** `def-sm-cp-transformation`, `thm-sm-cp-violation`.

### VI-16 — Effective action and generating functions

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/04_standard_model.md).

**Algorithmic object:** Empirical generating function of a bounded recorded-step descriptor, together with separately reported primitive path carriers

**Exact identity:** Derivatives of the logarithm of a finite empirical exponential sum equal its tilted mean and cumulants.

**Measurement:** Uniform retained-step exponential sum for tanh(mean terminal position components), with analytic derivative and independent finite difference; complete available primitive path densities are reported separately.

**Validation:** Finite differences against the directly summed tilted derivative; primitive carrier and density tests apply to the accompanying path report.

**Scope:** Dependent retained steps define the displayed empirical measure. Its exponential sum is not an independently sampled full-trajectory partition function.

**Theory anchors:** `def-sm-direct-observable-law`.

### VI-17 — Executed path-action accounting

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Complete staged execution with current/historical companion pools, clone gates, revival, and actual innovation carriers

**Exact identity:** Conditional primitive densities multiply; negative logarithms add. Deterministic stage maps retain their own carrier.

**Measurement:** Per-step component action, cumulative action, Gaussian quadratic and log determinant; singular/uniform stages retain exact latent innovation density.

**Validation:** suite: partvi_path: 9 passing tests; checks: uniform independent ordered assignments normalize; without-replacement assignment normalization; Fisher–Yates matching shuffle multiplicity; configured live-permutation versus death fallback; Gaussian weights reconstructed from actual historical coordinates; archive replay/serde and independent hand-evaluated Gaussian density; singular latent Gaussian law; low-rank uniform latent density -N*r*log(2*sqrt(3)); uniform latent density tolerance: 1e-12

**Scope:** Weighted multi-donor sampling without replacement retains unsorted reservoir slots; exact slot law is unavailable.; Gaussian-greedy joint likelihood requires its unretained shuffle path.; Unknown custom providers and unrecorded external-environment kernels remain explicitly unavailable.; The displayed primitive density is not a final-state Lebesgue density or a likelihood ratio to a separately sampled normalized reference process.

**Theory anchors:** `sec-ym-first-principles`, `thm-action-from-path-integral`, `thm-ym-recorded-action-emergence`.

### VI-18 — Kinetic metric from one record

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Actual eligible A1 rows and executed O-stage B xi, including low-rank factors and source shifts

**Exact identity:** E[Y_j^2]=mu_j^2+C_jj; E[Y_j^4]=mu_j^4+6mu_j^2 C_jj+3C_jj^2+kappa4 sum_k B_jk^4.

**Measurement:** Recorded second/fourth moments against predictions from the factor and innovation law fixed before sampling.

**Validation:** innovation: standardized uniform; rank: 1; dimensions: 3; history window: 2; viscosity coefficient: 0.15; second measured: 1.7431301050375065; second predicted: 1.75; fourth measured: 10.30072817957915; fourth predicted: 10.2375; steps: 96; walkers: 8

**Scope:** Conditional-moment averages use dependent trajectory data; no independent-component standard error is asserted.; These are raw B xi moments before thermostat scaling; the exact physical O-stage moments are separately evaluated in VI22/48.

**Theory anchors:** `thm-ym-kinetic-metric-correspondence`, `thm-ym-kinetic-metric-correspondence`.

### VI-19 — Gaussian source response

**Data source:** independent continuations. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Complete checkpoint continuation map with independent future seeds and an addressed future Gaussian innovation shift

**Exact identity:** E[F(Gamma(Xi+theta,V))]=E[F(Gamma(Xi,V)) exp(theta Xi-theta^2/2)] for the unselected zero-extended observable.

**Measurement:** Independent direct-shift and reweighted groups; Hermite derivatives at zero and paired finite differences are separate outputs.

**Validation:** unbounded: initial seeds: 31; 407; 9021; replicas per seed: 96; horizon: 3; warmup: 3; theta: 0.3; history window: 2; viscosity: True; residual sum: 0.05749065239085449; independent standard error: 0.13109824944390772; absorbing boundary: initial seeds: 159; 9183; direct replicas: 192; extinct direct replicas: 3; horizon: 4; theta: 0.6; observable: alive_fraction; residual sum: -0.0004651478233969053; independent standard error: 0.07735672237046183

**Scope:** Hermite estimates are derivatives at zero; finite-theta finite differences can cross clone/mask thresholds and retain finite-shift bias.; Survival-selected normalized response is not the displayed unselected cemetery-extended law.; Physical force-source KL and reduced descriptor scores in VI20 are a separate calculation.

**Theory anchors:** `thm-ym-native-source-response`, `thm-ym-native-source-response`.

### VI-20 — Force sources and information

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite scalar Gaussian source fixture; no native archive source-score projection

**Exact identity:** Gaussian mean shift KL equals theta^2/2 and conditional score projection reduces Fisher information.

**Measurement:** Gaussian samples and exact conditional descriptor probabilities.

**Validation:** suite: partvi_qft: scalar source/quadrature and finite serialization tests pass

**Scope:** This reference experiment does not reconstruct the actual metric force-source score from gas archives.; The actual Gaussian innovation-source engine identity is tested in VI19.

**Theory anchors:** `thm-ym-metric-force-sources`, `thm-ym-metric-force-sources`.

### VI-21 — Geometry-fiber disintegration

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Empirical joint law of bounded retained-step position and accepted-clone-fraction descriptors

**Exact identity:** A joint empirical mass equals its row marginal times its conditional mass; -log masses add on positive cells.

**Measurement:** Integer-count joint/marginal/conditional histogram plus separate actual primitive likelihood components.

**Validation:** suite: partvi_path empirical disintegration normalization passes

**Scope:** Empirical retained-step frequencies are not the reference-conditional path-likelihood expectation in the full geometry/fiber action theorem.; Empty geometry bins have no conditional law.; Temporal dependence remains in the retained sample.

**Theory anchors:** `thm-ym-native-geometry-fiber-action`, `thm-ym-native-geometry-fiber-action`.

### VI-22 — Stochastic Noether balance

**Data source:** independent continuations. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Actual complete-checkpoint finite-horizon increments and exact conditional O-stage martingales

**Exact identity:** Q(S_h)-Q(S_0) has conditional drift K^h Q-Q. At each actual A1 state the Gaussian/uniform BAOAB conditional momentum/energy moments supply a known martingale compensator.

**Measurement:** Independent calibration/validation of complete increments; analytic O-stage residuals and predictable quadratic variations over independently seeded complete continuations.

**Validation:** initial seeds: 71; 809; 1217; replicas total: 384; horizon: 3; history window: 2; viscosity: True; momentum residual mean sum: 0.03093718833664627; momentum mean standard error: 0.11833943636680205; energy residual mean sum: -0.4902668678494803; energy mean standard error: 0.5157135201574125; realized over predicted martingale variance: 0.9750085228096965

**Scope:** Full-step conditional drift is estimated from independent replicas; it is not supplied by a closed macroscopic model.; The analytic substage test isolates O before boundary handling; complete-engine samples and stage ledgers retain clone, force, boundary and eligibility transfers.

**Theory anchors:** `sec-ym-noether`, `thm-u1-noether-current`, `thm-ym-kinetic-metric-correspondence`.

### VI-23 — Wilson variation and Ward identity

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite supplied noncommuting SU(2) loop

**Exact identity:** Trace is invariant under conjugation; direct Wilson action derivative equals analytic sine/cosine expression.

**Measurement:** Finite-difference action derivative and basepoint conjugation residual.

**Validation:** suite: partvi_qft finite reference endpoint/serialization checks

**Scope:** No archive gauge-link variation or node-by-node Ward divergence is measured.

**Theory anchors:** `lem-ym-discrete-ward`.

### VI-24 — Continuum Yang-Mills consistency

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite constant noncommuting connection

**Exact identity:** (U_square-I)/h^2 approaches the matrix commutator.

**Measurement:** Direct ordered products versus commutator and matrix-valued refinement residual.

**Validation:** suite: partvi_qft reference checks

**Scope:** The connection is supplied by the reference controls, rather than derived from actual gas transport.

**Theory anchors:** `sec-ym-continuum`.

### VI-25 — Units, discrete time, and gaps

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Exact sampled two-state semigroup and Euler discretization

**Exact identity:** Exact lambda=exp(-2qh) gives -log(lambda)/h=2q; Euler lambda=1-2qh converges under refinement.

**Measurement:** Repeated transition powers, decay curves and recovered-rate refinement.

**Validation:** test: euler_gap_bias_refines_while_exact_semigroup_rate_is_fixed: passes

**Scope:** These are supplied finite generators, not a native gas spectral-gap estimate.

**Theory anchors:** `sec-ym-constants`.

### VI-26 — Fluctuation and collision budgets

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite joint two-walker swap/collision fixture

**Exact identity:** Delta(XY)=X DeltaY+Y DeltaX+DeltaX DeltaY; joint cross-covariance is -p(1-p)(y-x)^2.

**Measurement:** Exact enumeration of coupled outcomes and finite-step product terms.

**Validation:** test: joint_collision_covariance_and_product_terms_survive: passes

**Scope:** The reference correlated swap law is not the complete configured native clone/restitution law. Native full-write measurements belong to VI14/51.

**Theory anchors:** `prop-ym-complete-fluctuation-equations`.

### VI-27 — QSD windows and burn-in

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Exactly solved two-state killed chain

**Exact identity:** QSD left eigenmeasure and survival-conditioned two-step path window use the same survival eigenvalue.

**Measurement:** Enumerated path TV against QSD-window bound under burn-in.

**Validation:** test: qsd_window_comparison_enumerates_the_conditioned_path_law: passes

**Scope:** No empirical QSD identification or quantitative mixing bound for the configured gas is inferred from this fixture.

**Theory anchors:** `thm-ym-qsd-window-relaxation`.

### VI-28 — Reflection matrices

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite labeled-word reflection matrix with specified off-diagonal moments

**Exact identity:** For c=-(u+v)/2, (c,1)Q(c,1)^T=-(u+v)^2/4.

**Measurement:** Direct matrix contraction, Hermitian defect and Hermitian-part eigenvalues.

**Validation:** test: reflection_counterexample_has_the_predicted_negative_sign: passes

**Scope:** Moments u and v are supplied reference values; they are not estimated from the native reflected geometry/color law.; This reflected pairing is distinct from the ordinary covariance pairing used in VI36.

**Theory anchors:** `prop-ym-native-labeled-color-reflection-sign`.

### VI-29 — Scalar triangles and outer plaquettes

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite scalar potential-difference links

**Exact identity:** Closed outer edge potential differences telescope; the defined triangle readout retains its phase difference.

**Measurement:** Direct complex products and scalar defects.

**Validation:** test: scalar_outer_faces_telescope_while_triangles_do_not: passes

**Scope:** This experiment evaluates supplied scalar links rather than recording full native geometric/gauge face construction.

**Theory anchors:** `prop-ym-native-scalar-face-evaluation`.

### VI-30 — Translation and localization

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite point cloud with anchored Gaussian regulator

**Exact identity:** Co-translating coordinates and the regulator preserves its readout; moving only coordinates changes the anchored measurement.

**Measurement:** Direct sums and translated difference curve.

**Validation:** suite: partvi_qft reference checks

**Scope:** No native law-level translated QSD or boundary-domain comparison is estimated.

**Theory anchors:** `prop-ym-anchored-regulator-translation-test`.

### VI-31 — Regional algebras and interventions

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Finite multiplication operators and CAR covariance fixture

**Exact identity:** Multiplication operators commute; covariance overlap controls the CAR regional anticommutator.

**Measurement:** Explicit matrix commutators and local/nonlocal kernel fixture.

**Validation:** suite: partvi_qft CAR/operator checks

**Scope:** Actual intervention-recomputed regional transition kernels are not reconstructed here.

**Theory anchors:** `sec-qft-axioms-verification`.

### VI-32 — Connected algorithmic QFT

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md).

**Algorithmic object:** Actual descriptor sequence with chronological train/guard/heldout finite partitions

**Exact identity:** Conditional probabilities define the finite fitted predictor; weighted coarsening measures its two-step memory defect.

**Measurement:** Training-only quantiles and transition matrix; later-block Brier scores and per-lag transition errors.

**Validation:** suite: Chronological predictive checks compare training-only partitions with disjoint held-out transition and Brier scores.

**Scope:** A finite fitted descriptor transition is not automatically the exact projected Markov semigroup of the gas.; The empirical predictive pipeline does not itself verify all CAR/source-response identities listed in the synthesis chapter.

**Theory anchors:** `sec-ym-algorithmic-qft-synthesis`.

### VI-33 — Spinor and mass identities

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/08_twistor_formulation.md).

**Algorithmic object:** Finite real Lorentz vector and complex two-spinor algebra

**Exact identity:** Determinant of sigma map equals Lorentz norm; null and two-spinor factorization identities.

**Measurement:** Independent determinant and epsilon-bracket calculations.

**Validation:** tests: spinor_tie_rule_and_null_mass_identity passes; effective_twistor_incidence_matches_sigma_matrix_action passes

**Scope:** This verifies the algebra of supplied vectors; it does not reconstruct a physical on-shell particle momentum from the algorithm.

**Theory anchors:** `sec-twistor-spinor-conventions`.

### VI-34 — Recorded triplet twistor operators

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/08_twistor_formulation.md).

**Algorithmic object:** Actual pre-clone recipient and immutable current/historical distance and cloning donor snapshots

**Exact identity:** Effective edge spinors and bounded local channels are deterministic functions of the exact source coordinates and configured velocity scale.

**Measurement:** Normalized-column spinors, scalar/vector/tensor readouts, masks, source refs and source ages.

**Validation:** seed: 516; history window: 2; walkers: 8; recorded steps: 96; valid readouts: 8; historical donors: 10; total selected donors: 16

**Scope:** Donor coordinates preceding retained archive/anchor coverage remain explicitly unavailable.; The effective edge uses the specified algorithmic Delta t; donor age is a separate recorded descriptor.

**Theory anchors:** `sec-fractal-set-twistorization`, `def-effective-twistor-triplet`, `def-effective-twistor-edge-data`, `def-effective-edge-twistor`.

### VI-35 — Source-frozen lag tracking

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/08_twistor_formulation.md).

**Algorithmic object:** Source-frozen triplet lag diagnostic on recorded states

**Exact identity:** Current numerical source slots advance to the sink; historical source snapshots stay immutable; each curve retains its valid-pair normalization.

**Measurement:** Complex fixed-source and reselected-source lag correlations with exact counts.

**Validation:** tests: fixed_source_slots_give_distinct_lag_readouts passes; archive source identity and epoch validation tests pass

**Scope:** Numerical slots are distinct from walker ancestry identities.; Pair-conditioned correlations have lag-dependent denominators and differ from fixed frame-observable covariance.

**Theory anchors:** `sec-twistor-vs-euclidean`, `def-effective-twistor-correlators`.

### VI-36 — Spectral decay and twistor amplitudes

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/2_fractal_set/08_twistor_formulation.md).

**Algorithmic object:** Actual complete-record fixed-normalization frame observable; optional separately named source-pair diagnostic

**Exact identity:** C_frame_ab(n,l)=mu K^n[conj(f_a)K^l f_b]-conj(mu K^n f_a)mu K^(n+l)f_b, with actual K retaining memory, clone, kinetics, clipping, boundaries and killing.

**Measurement:** Training-only means/whitening, general complex roots, phase/decay fits and held-out error; fixed 1/N zero extension by default.

**Validation:** seed: 516; history window: 2; viscosity: True; frame gram direct: 0.8500623473105489; frame gram measured: 0.8500623473105489; fit: decay rate: -1.7862384364191546; frequency: -18.089762813984343; log fit rmse: 0.6110975403956461; phase fit rmse: 0.5911784247409881; heldout complex rmse: 0.3924447584217878; mass: None; interpretation: This measured finite-window mode is complex and grows in magnitude over the fit window; it does not supply a positive particle-mass estimate.

**Scope:** General complex finite-window fits characterize the retained observable; they do not certify a positive transfer representation.; Stationarity is not assumed in the exact actual-K identity; a single trajectory can exhibit nonstationary finite-window behavior.; Chronological held-out data retain temporal dependence.

**Theory anchors:** `thm-effective-twistor-spectral-meaning`, `def-effective-twistor-correlators`, `thm-effective-twistor-spectral-meaning`, `cor-effective-twistor-positive-transfer`.

### VI-37 — Transport in a changing geometry

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Specified time-dependent metric slab, not an executed walker transport

**Exact identity:** Slice and ambient Levi-Civita connections differ by time variation; Lorentzian parallel transport preserves norm.

**Measurement:** RK4 transports a vector along the slab spatial path; compares with hyperbolic-function solution.

**Validation:** Norm and refinement tests.

**Scope:** Reference-only: the engine does not supply a Lorentzian slab or a differentiable material map.

**Theory anchors:** `assump-curvature-geometric-setting`, `lem-curvature-slab-connection`, `def-parallel-transport`.

### VI-38 — Curvature per loop area

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Specified unit-sphere connection

**Exact identity:** Small-loop holonomy divided by controlled area tends to Gaussian curvature.

**Measurement:** Integrates transport around four sides and computes loop angle and area independently.

**Validation:** Integrated sphere holonomy converges to K=1.

**Scope:** Reference-only; no recorded scutoid connection is inferred from edge displacements.

**Theory anchors:** `lem-holonomy-small-loops`, `lem-regge-holonomy-approx`, `thm-riemann-scutoid`.

### VI-39 — Conditional fitness curvature

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Actual recorded conditional fitness query, immutable donor source and global/local normalizer; fixture alternatives

**Exact identity:** g=epsilon I+H in strict domain, or epsilon I+H_positive for clipped provider; differentiate the configured query field through fourth order and contract curvature.

**Measurement:** Archive reconstruction matches recorded O-stage Hessian at the identical query. Analytic packed curvature is checked against finite differences of the metric.

**Validation:** global phase space history: historical queries: 26; maximum hessian error: 3.552713678800501e-15; local phase space distance history: historical queries: 25; maximum hessian error: 3.1086244689504383e-15; cloning history: current-frame as explicitly configured, because native historical clone rescoring requires global standardization; absorbing killing and memory: checked revivals: 36; revival fields with historical distance sources: 29; maximum hessian error: 0.0; other: Existing same-frame global/local reconstruction and independent curvature tests retained.

**Scope:** Clipped curvature needs a spectral neighborhood away from zero crossings. Unsupported derivative providers or missing historical coordinates are explicit errors. The programmed metric construction is a defining rule, not an inferred force law.

**Theory anchors:** `def-curvature-conditional-fitness-field`, `lem-curvature-hessian-cancellation`, `lem-curvature-whitened-contractions`, `cor-curvature-packed-three-dimensional`, `lem-curvature-query-moment-cache`, `rem-curvature-clipped-computation`.

### VI-40 — Connection identifiability

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Specified symmetric connection identification experiment

**Exact identity:** Integrated transport contraction determines coefficients only for a full-rank design.

**Measurement:** Independent integrated transports feed least squares; measured Gram eigenvalues expose missing directions.

**Validation:** Rank-deficient fixture retains a zero eigenvalue and unavailable coefficients; full design converges.

**Scope:** Reference-only transport data; no claim that a single walker velocity identifies a connection.

**Theory anchors:** `def-edge-deformation`, `prop-connection-least-squares`.

### VI-41 — Raychaudhuri term balance

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Bianchi-I and rotating Minkowski congruences

**Exact identity:** Raychaudhuri includes expansion, shear, twist, Ricci contraction and acceleration divergence.

**Measurement:** Bianchi expansion from volume differentiation; rotating velocity gradients are finite-differenced and acceleration divergence independently differentiated.

**Validation:** Dimensions2/3; speeds0,.4,.8; zero expansion/shear and twist/acceleration cancellation.

**Scope:** Reference-only explicitly specified congruences. No congruence is extracted automatically from cloning.

**Theory anchors:** `def-kinematic-decomposition`, `thm-raychaudhuri`.

### VI-42 — Material and reconstructed volumes

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Explicit nonlinear one-dimensional flow with transverse product

**Exact identity:** Derivative of material log-volume tends to point expansion; reconstructed-cell flux is separate.

**Measurement:** Finite material-cell widths and redrawn Voronoi intervals are differentiated.

**Validation:** Cell refinement reduces error quadratically.

**Scope:** Reference-only; a reconstructed Voronoi volume is not automatically a material cell.

**Theory anchors:** `def-regularity-conditions`, `thm-discrete-raychaudhuri`, `cor-discrete-relaxation`.

### VI-43 — Focusing and caustics

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Contracting Raychaudhuri comparison ODE

**Exact identity:** theta_prime=-theta_squared/d-shear_squared-Ricci; nonnegative focusing terms imply focusing no later than d/abs(theta0).

**Measurement:** RK4 solution and Jacobian compared with exact constant-forcing solution and zero-forcing bound.

**Validation:** Nonzero shear shortens exact focusing time; integrated curve stays below comparison.

**Scope:** Prescribed constant invariants are an ODE comparison, not a metric reconstructed from the run.

**Theory anchors:** `def-sec`, `thm-focusing`.

### VI-44 — Curvature and topology

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/03_curvature_gravity.md).

**Algorithmic object:** Triangulated disk, closed polyhedral sphere and independent three-dimensional hinge

**Exact identity:** 2D Gauss-Bonnet sums deficits plus boundary turning to2pi chi; 3D integrated hinge curvature scales with edge length.

**Measurement:** Triangle angles and mesh incidences computed directly at each sphere refinement.

**Validation:** Sphere chi2, no boundary, total4pi at refinements0..3 and scales.1,4; 3D linear scaling.

**Scope:** Reference meshes; arbitrary recorded cloning does not specify a piecewise-flat manifold.

**Theory anchors:** `def-regge-curvature`, `thm-curvature-topology`, `prop-curvature-change-2d`, `rem-higher-dim-curvature`.

### VI-45 — Fluctuation energy and metric evolution

**Data source:** independent continuations. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Complete executed checkpoint including population, donor memory, providers and fresh future innovation keys; material metric readout

**Exact identity:** Conditional increment b=E[A_next-A], covariance Gamma; two independent groups have discrepancy covariance Gamma/R+Gamma/S. Field/probe decomposition telescopes with all cross covariances.

**Measurement:** Native replicas evaluate the programmed conditional metric at final slot position. Exact decomposition uses new field at old probe, literal-clone probe and final probe. Fixed-probe lagged readout remains explicitly selectable.

**Validation:** benchmark tests passed: 10; material component covariance telescoping: seed11: 8.171241461241152e-14; seed29: 2.220446049250313e-16; seed43: 2.7755575615628914e-17; memory replica test: history_window2, warmup2, horizon2, four replicas per independent group; both groups have four defined metrics and historical sources are retained before continuation recording

**Scope:** The metric is programmed; conditional drift/covariance characterize its actual evolution. Same-state replica calibration does not establish a metric-only closed equation. At horizons>1 the middle probe term includes earlier motion. Free-energy plots in reference mode are separate comparison models. Built-in revival selects an eligible current-frame donor; historical accepted cloning is retained. The revived source field can itself depend on a historical distance donor. This source convention follows cloning.rs, rather than introducing a different revival law.

**Theory anchors:** `def-algorithmic-transition-state`, `thm-algorithmic-observable-increment`, `prop-algorithmic-material-metric`, `prop-algorithmic-independent-replicas`, `lem-algorithmic-quadratic-finite-step-moments`, `lem-ig-rate-function`, `def-ig-free-energy`.

### VI-46 — Pressure and deformation

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Declared Gaussian pair-energy functional

**Exact identity:** Dilation derivative gives pressure; sign and stiffness depend on energy sign and kernel normalization.

**Measurement:** Finite-difference dilation pressure compared with pair-integral derivative and scaling curves.

**Validation:** Signed and nonnegative density fixtures, attractive/repulsive signs.

**Scope:** Reference-only: pair energy is not inferred from the actual cloning law.

**Theory anchors:** `def-boost-perturbation`, `thm-elastic-pressure`, `rem-elastic-interpretation`.

### VI-47 — Density-wave dispersion

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Specified homogeneous Gaussian redistribution closure

**Exact identity:** Fourier decay D k²+lambda(1-exp(-epsilon² k²/2)); constant mode conserved.

**Measurement:** Grid convolution/Laplacian multiplier measured separately from analytic Fourier multiplier; explicit Euler mode evolution.

**Validation:** Refinement and fastest combined control endpoint maintain positive monotone decay.

**Scope:** Reference closure, not the interacting donor/acceptance kernel. Negative high-frequency truncated rates diagnose breakdown of the long-wave truncation.

**Theory anchors:** `def-mckean-vlasov`, `def-uniform-qsd-linearization`, `thm-dispersion-relation`, `thm-qsd-stability`, `cor-exponential-relaxation`.

### VI-48 — BAOAB diffusion and thermostat heating

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Executed raw BAOAB O stages, actual diffusion factors, Gaussian/uniform laws and addressed innovation shifts

**Exact identity:** Conditional kinetic mean and variance follow from v_plus=c v+s B(xi+shift); uniform fourth moment9/5 changes energy variance. Predictable variances add for sequential martingale differences.

**Measurement:** Archive mode measures raw velocity increments and computes source-aware conditional moments from recorded B and law. Cumulative centered energy compared with square-root predictable variation. Scalar reference separately samples the native innovation generator.

**Validation:** independent runs per law: 64; raw O stages per run: 2; rectangular factor shape: 3; 2; nonzero source shift: 0.7; Gaussian: centered sum: -1.846554308133356; predictable variance sum: 16.130137994548914; standardized sum: -0.4597725550340293; measured to predicted second moment ratio: 1.0151134825335426; StandardizedUniform: centered sum: 2.6660553839577825; predictable variance sum: 15.634784650200292; standardized sum: 0.6742535088461272; measured to predicted second moment ratio: 0.8695021899002708; method: Predictions use source-aware analytic conditional moments at each actual stage. Independent full-run seeds are the sampling units. The standardized sums are diagnostics, not exact Gaussian p-values.

**Scope:** Predictable variation is a fluctuation scale, not an exact confidence band. Programmed B=sqrt(2gamma T)g_inverse_sqrt determines noise geometry; measured raw increments independently test its consequence.

**Theory anchors:** `thm-algorithmic-thermostat-moments`, `cor-algorithmic-geometric-heating`, `lem-field-discrete-ou-diffusion`.

### VI-49 — Gaussian mode thermodynamics

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Specified independent Gaussian mode state

**Exact identity:** Gaussian variance and fixed-mode-set partition derivative give the declared mode pressure; mobility controls relaxation separately.

**Measurement:** Sampled variance, partition pressure derivative and OU relaxation comparison.

**Validation:** Mode-pressure finite differences and variance tests.

**Scope:** Reference-only specified mode state; no interacting QSD Gibbs law is inferred.

**Theory anchors:** `ass-thermal-equilibrium`, `prop-radiation-pressure`, `rem-pressure-comparison`.

### VI-50 — Pressure crossover

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Declared thermal/correlation pressure comparison

**Exact identity:** Crossover solves equality of the two explicitly supplied pressure scales.

**Measurement:** Independent bisection compared with analytic crossover.

**Validation:** Root residual test.

**Scope:** Reference-only parameters require common normalization; no actual algorithm equation of state estimated here.

**Theory anchors:** `def-thermal-correlation-length`, `thm-pressure-regimes`.

### VI-51 — Mechanical balances and gravitational stress

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/04_field_equations.md).

**Algorithmic object:** Recorded clone, transform, raw kinetic, boundary and eligibility stages; separate constitutive algebra

**Exact identity:** Exact kinetic/momentum differences telescope. Kick work=v dot delta-v+norm(delta-v)²/2. Weak local momentum splits impulse/transport/eligibility. Restitution overwrites pre-clone pairs and must subtract literal-clone increment.

**Measurement:** Archive source-aware heating, full stage budgets, centered kinetic moment, raw total-force kick work, and local test-function momentum measured directly.

**Validation:** Executed viscosity/source-shift tests check two kicks, weak identity and telescoping; no potential inferred from reward.

**Scope:** Centered kinetic moment requires volume normalization and additional contributions before serving as stress. Constitutive reference assumes a Lorentzian Einstein equation. Reported local96 closure Ebar=kappa centered_kinetic+lambda gbar failed: held-out RMSE51363 versus constant31599 and shuffled32372; this particular fitted spatial relation is rejected by that data, not repaired by fitting.

**Theory anchors:** `def-algorithmic-mechanical-observables`, `prop-algorithmic-stage-telescoping`, `prop-algorithmic-kick-work`, `prop-algorithmic-clone-balances`, `prop-algorithmic-weak-density`, `thm-structural-correspondence`, `rem-correspondence-meaning`.

### VI-52 — Genealogical separators and cuts

**Data source:** archive. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Actual recorded episode ancestry and selected directed interactions; separately reconstructed complete Gaussian graph

**Exact identity:** Separator must be an antichain intersecting every chosen ancestral chain once. Cut capacities use actual edges/weights; auxiliary symmetric Gaussian graph obeys max-flow/min-cut.

**Measurement:** Episode parent forest resolved by epoch/slot/generation. Actual selected channels counted across fixed slot partition, missing weights retained as unavailable. Gaussian mincut on explicitly projected archived points computed separately.

**Validation:** Recorded channel counts compared directly with FractalSet; network-flow/partition identity; living ancestor fixture rejects terminal episodes as antichain.

**Scope:** Missing historical sources yield partial ancestry coverage. Gaussian reconstruction is not the actual sparse/directed IG. Slot-partition cuts are structural observables, not geometric area.

**Theory anchors:** `def-cst-structure`, `def-ig-structure`, `def-separating-antichain`, `def-ig-entanglement-entropy`, `prop-holo-cut-identities`.

### VI-53 — Jump, modular, and path entropy

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Specified qubit state and exact finite Markov path reference

**Exact identity:** Density-matrix relative entropy controls modular difference; exact path KL and exponential identity depend on normalized reverse law and support.

**Measurement:** Eigenvalue entropy and enumerated finite paths with endpoint likelihoods; explicit support failure.

**Validation:** Noncommuting state and finite path identities.

**Scope:** Reference-only. Recorded walker covariance does not specify a quantum state or exact reverse path likelihood.

**Theory anchors:** `prop-modular-connection`, `def-algorithmic-path-irreversibility`, `prop-algorithmic-path-kl-identities`.

### VI-54 — Gaussian cuts and perimeter

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Declared density on torus with periodized Gaussian cut kernel

**Exact identity:** Scaled nonlocal cut approaches constant times integral of rho² over both boundary components.

**Measurement:** Direct pair quadrature compared with short-range perimeter constant.

**Validation:** Range refinement and density contrast tests.

**Scope:** Reference-only geometric sampling hypotheses; no automatic actual IG perimeter limit.

**Theory anchors:** `def-nonlocal-perimeter`, `lem-holo-bv-differences`, `thm-gamma-convergence`.

### VI-55 — Finite-population area scaling

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Independent uniform reference populations

**Exact identity:** Mean pair cut=N(N-1) times one-sided cross integral; sampling unit is an independent population.

**Measurement:** Complete torus pair graph sampled in independent groups; mean and population-level SE compared with quadrature expectation.

**Validation:** Independent-population expectation test.

**Scope:** Finite quadrature error remains; these independent samples are not interacting walkers from one gas population.

**Theory anchors:** `thm-ig-cut-scaling`, `cor-holo-dependent-cut-limit`, `thm-antichain-surface`, `thm-informational-area-law`.

### VI-56 — Three first variations

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Declared density, shape and density-matrix perturbations

**Exact identity:** Cut functional density derivative, geometric shape variation and quantum-state first variation are separate directional derivatives.

**Measurement:** Finite differences compared with analytic expressions in each state space.

**Validation:** All three variations implemented, including noncommuting matrix-state variation.

**Scope:** Reference-only variation protocols; no universal first-law identification among the three state spaces.

**Theory anchors:** `def-swarm-energy-variation`, `def-ig-entropy-variation`, `prop-holo-cut-first-law-condition`, `thm-first-law-entanglement`, `thm-holographic-pressure`.

### VI-57 — Gibbs QSD candidates

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Exact killed two-state comparison kernel

**Exact identity:** Gibbs candidate and QSD are distinct; residual and contraction bound control their discrepancy.

**Measurement:** Numerical eigenmeasure compared with chosen Gibbs law and bound.

**Validation:** TV residual bound test.

**Scope:** Reference-only killed kernel; not an invariant Gibbs claim for actual cloning.

**Theory anchors:** `thm-qsd-gibbs`.

### VI-58 — Response and Fisher information

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Interior two-state controlled Markov reference

**Exact identity:** Stationary response pi D (I-K)_centered_inverse A; stationary, transition and path Fisher are distinct.

**Measurement:** Poisson solve compared with independently differentiated stationary law and exact formulas; static exponential tilt comparison.

**Validation:** Response finite differences and Fisher formulas.

**Scope:** Reference-only. Actual stationary response requires the actual transition derivative, mixing and common-support hypotheses; actual source finite-horizon response lives in native19/22.

**Theory anchors:** `prop-fluctuation-dissipation`, `thm-algorithmic-stationary-response`, `prop-algorithmic-controlled-fisher`, `rem-algorithmic-three-geometries`.

### VI-59 — Euclidean horizons and AdS

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Specified Euclidean cone and constant-curvature AdS metric

**Exact identity:** Smooth cone requires kappa beta=2pi; AdS curvature solves assumed vacuum Einstein equation.

**Measurement:** Circumference/periodicity and algebraic curvature comparison.

**Validation:** Cone smoothness and AdS contraction identities.

**Scope:** Reference-only spacetime/thermal state. No gas temperature or AdS radius is inferred from a graph cut.

**Theory anchors:** `prop-unruh-hawking-connection`, `thm-ads-uv-regime`, `rem-holo-cosmological-sign`.

### VI-60 — Minimum cuts and limiting surfaces

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/05_holography.md).

**Algorithmic object:** Nearest-neighbor weighted perimeter reference network

**Exact identity:** Terminal mincut equals the minimum of a specified local discrete surface functional.

**Measurement:** Max-flow partition compared with independently enumerated straight interfaces.

**Validation:** Constrained minimum versus exact candidates.

**Scope:** Reference-only local network, distinct from the complete Gaussian graph and actual recorded IG.

**Theory anchors:** `thm-ryu-takayanagi`.

### VI-61 — Conditional stationarity and survival

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Two-state QSD reference with independent constant killing

**Exact identity:** Conditioned stationary expectations are constant; unnormalized survival decays, and an energy offset does not affect increment.

**Measurement:** Matrix evolution and survival curve.

**Validation:** Stationary/QSD distinction and shift-invariance tests.

**Scope:** Reference-only; actual full-state conditional increments are characterized by native45 and22, not a claimed observed QSD.

**Theory anchors:** `def-bulk-qsd-vacuum`, `thm-vanishing-bulk-vacuum`, `prop-geometric-distinction`.

### VI-62 — Interaction range and curvature scale

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Circle chord-distance Gaussian integral and separately declared AdS radius

**Exact identity:** Local chord integral has leading relative epsilon²/(8R²); AdS radius relation assumes constant-curvature vacuum equation.

**Measurement:** Independent scaled-variable quadrature checks locality series.

**Validation:** Small-range quadrature converges without under-resolving narrow kernels.

**Scope:** The positive-curvature circle reference is explicitly independent from the negative-curvature AdS algebra. Kernel range alone does not determine actual curvature.

**Theory anchors:** `def-uv-regime`, `thm-ads-boundary-uv`.

### VI-63 — Expansion histories

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Specified Riccati comparison and de Sitter/Milne geometries

**Exact identity:** Complete forcing lower bound gives tanh expansion bound; positive expansion alone does not determine curvature.

**Measurement:** RK4 comparison plus distinct exact de Sitter and flat Milne expansion.

**Validation:** Lower-bound comparison tests.

**Scope:** Reference-only. No freely fitted Ricci forcing is advertised as an actual metric evolution equation.

**Theory anchors:** `thm-exploration-expansion`, `thm-bulk-can-be-ds`.

### VI-64 — Predictive information

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Declared future-bit reveal channel

**Exact identity:** Discarded predictive information equals conditional mutual information; reveal probability gives p log2.

**Measurement:** Independent samples estimate entropy/information against exact reference.

**Validation:** Information endpoints and sample agreement.

**Scope:** Reference-only controlled information process; actual donor memory may carry predictive information but is not measured by this bit fixture.

**Theory anchors:** `def-information-closure-cosmo`, `def-computational-closure-cosmo`, `def-causal-closure-cosmo`, `thm-closure-equivalence-cosmo`.

### VI-65 — Dynamical closure

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Exact four-state CTMC projection reference

**Exact identity:** Dynkin generator closure error accumulated over time is bounded by t times uniform generator residual.

**Measurement:** Microscopic semigroup evolution compared with projected macro model and residual.

**Validation:** Lumpable case zero residual; nonlumpable error bounded.

**Scope:** Reference-only. No Markov closure is inferred from a histogram of actual walkers.

**Theory anchors:** `prop-cosmo-generator-closure`.

### VI-66 — Vacuum units and observables

**Data source:** finite model. [Chapter](../docs/source/2_fractal_gas/3_fitness_manifold/06_cosmology.md).

**Algorithmic object:** Explicit physical-unit conversion and specified cosmic parameters

**Exact identity:** Lambda=3 Omega_Lambda H0²/c² and stated pressure conventions.

**Measurement:** Numerical unit conversion retains c² and length/time factors.

**Validation:** SI conversion regression.

**Scope:** Parameter calculator, not an estimate of physical cosmology from the gas.

**Theory anchors:** `prop-observed-lambda`, `prop-dark-energy-exploration`, `thm-de-sitter-resolution`.
