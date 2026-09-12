# Interactive experiments for Part VI: fields and algorithmic QFT

:::{div} feynman-prose
A calculation becomes easier to understand when we can change one input and watch the consequence. A loop trace can stay fixed while every link matrix changes. A color vector can cross a validity threshold. A compressed channel can predict one step correctly and miss the next because information returns from a discarded coordinate. The workbenches in this guide make those mechanisms visible beside the corresponding chapter arguments.

There are 66 workbenches across the eight chapters of Part VI: 23 expose archive measurements, three run independent complete-state continuations, and 40 inspect supplied finite models or geometric inputs. Each entry below states its placement, the quantities actually computed, the controls, and the comparison to make. Some use complete finite probability models or supplied analytic geometries, where the reference can be calculated independently. Others read a gas archive or run independent continuations of the compiled engine. The displayed model and exported details identify which calculation supplies the data. A formula link identifies the chapter passage to read with the experiment; the numerical comparison concerns the particular quantities and inputs described here.
:::

(sec-partvi-experiments-use)=
## Reading and running a workbench

:::{div} feynman-prose
Begin with a recorded workbench and identify the complete update that generated its data. Change an engine parameter, keep the observable definition fixed, and compare its measured increment with the conditional law computed from that update. Use a finite reference calculation to inspect the relevant algebra or approximation separately. A residual measures the difference between those two calculations. For sampling experiments, increase the independent sample count and inspect the uncertainty as well as the mean. For a refinement experiment, change the numerical resolution while holding the model fixed. Those operations answer different questions and have separate controls.

The seed selects reproducible random draws. In a reference sampling workbench, another step selects another seeded sample. In an archive workbench, stepping advances the gas by four complete updates before measuring again. Source-response, stochastic-balance, and metric-evolution workbenches run independent continuation groups from a complete frozen checkpoint. Their replica count controls the number of independent continuations, and their horizon controls the number of subsequent gas steps.
:::

**Source selection.** Recorded mode is the default for VI-02–06, VI-08–10, VI-12–14, VI-16–19, VI-21–22, VI-32, VI-34–36, VI-39, VI-45, VI-48, VI-51, and VI-52. Each also offers its finite reference calculation. The other workbenches accept their explicitly supplied reference inputs. An archive request requires an implemented native observable; unsupported requests return a capability error.

**Recorded gas profile.** The browser starts a CPU f64 gas with three spatial components, 32 walkers, a quadratic objective, BAOAB timestep $0.04$, friction one, Gaussian innovations, and Gaussian graph viscosity of coefficient one. Both companion roles use independent uniform selection from the configured donor pool, with donor history window two. The pool therefore includes immutable historical data consumed by the actual update. Engine controls change population, memory, timestep, friction, viscosity, boundary rule, and innovation law. Bounded runs use the position box $[-4,4]^3$. Periodic runs use that same domain for both donor-distance calculations and position wrapping. VI-39, VI-45, and VI-48 offer unbounded or absorbing boundaries: their conditional fitness metric provider supports these domains and rejects periodic minimum-image seams. Other recorded workbenches also offer periodic boundaries. The isotropic noise factor is $\sqrt{0.8}$; VI-39, VI-45, and VI-48 install the state-dependent fitness metric described below. VI-19 selects Gaussian innovations for its Gaussian likelihood identity.

**Recording interval.** VI-03–06, VI-12–13, VI-32, and VI-36 begin with 96 complete recorded updates. The other trajectory workbenches begin with at least four, extended to cover the chosen donor history. A live step adds four updates within the 128-step and 128-MiB budgets. The population view displays executed positions with slot and generation identities. Imported archives retain their own population, history, and configured law.

**Archive readouts.** The Gram, regional covariance, predictive, path-action, color, twistor, spectral, and curvature workbenches read the recorded stages described in their entries. VI-02 reconstructs the executed gate probabilities and their innovations. VI-18 compares executed raw noise moments with the recorded factor and innovation law. VI-51 measures full stage budgets, total-force work, and weak momentum transport. VI-52 reads selected interaction edges and episode ancestry, and separately constructs an auxiliary Gaussian spatial graph. Archive import uses the same native readout routines. Temporal analyses retain their frame, epoch, mask, and normalization conventions in the result.

**Metric profile.** Recorded VI-39, VI-45, and VI-48 install the conditional fitness metric provider. Their browser engine uses metric regularizer one, temperature $0.4$, the clipped Hessian policy, and clipping threshold $10^{-8}$. Global reward and diversity standardizers use minimum scale $0.1$, and the distance floor is $0.15$. VI-39's **Epsilon** control changes the analyzed metric at the fixed recorded field; VI-45 measures the metric defined by the engine configuration. The continuation experiment's reference interaction range has a separate meaning.

**Continuation readouts.** Recorded VI-19, VI-22, and VI-45 use full engine reruns from complete checkpoints carrying population state, donor history, and algorithm configuration. VI-19 and VI-22 offer 8, 16, 32, or 64 replicas per group and 1, 2, 4, or 8 continuation steps, initially 16 replicas and four steps. VI-45 offers 8, 16, or 32 replicas and 1, 2, or 4 steps, initially 16 replicas and one step. Its default material readout evaluates the metric along a selected walker slot through cloning and motion. The `fixed_probe` choice evaluates an unchanged chart point. The native interface also exposes warmup and readout-specific coordinates. A future innovation shift acts at its addressed O step, and subsequent dynamics are recomputed.

**Results and replay.** Exported results include the request, model, metrics, plots, notes, calculation details, precision, and source information. Live trajectory workbenches export their archive and checkpoint. Continuation workbenches export replica results and their recorded contexts. **Export JSON** writes a `fragile-partvi-results-v1` result bundle. **Open native sweep** imports that bundle or a native multi-result sweep into the lecture viewer. The plots display stored native arrays. Small real square Gram, covariance, kernel, transition, Hessian, Ricci, and Choi arrays in the details also appear as matrix heatmaps when present. Unavailable scalar metrics carry a null value and a diagnostic. VI-39 reports the selected curvature backend and any backend availability failure separately from its host calculation.

**Controls.** Each workbench lists the actual browser defaults, ranges, and choices, including source and continuation controls. Seed and run controls are shared. In a workbench with both sources, fixture parameters change the reference calculation; recorded calculations use the additional readout controls identified in the prose. Recorded walker count configures a new live run. An imported archive supplies its own executed population. Exact arrays, masks, matrix ranks, and normalization counts remain in the exported details.

(sec-partvi-experiments-algorithm-law)=
## The field theory follows the complete update

:::{div} feynman-prose
Watch one complete update. Companion selection reads a population and its retained donor records. Fitness rescoring sets the cloning probabilities. Accepted recipients receive immutable donor values, then the configured transforms, force kicks, position moves, thermostat, clipping, and boundary rules act in their stated order. A field describes this algorithm when its value can be calculated from that state and its change can be traced through those operations. Donor memory, shared clone writes, and state-dependent noise belong in that calculation: they are mechanisms whose effects we want to explain.

The complete state $R_n$ is the common starting point in {prf:ref}`prop-ym-transient-algorithm-fields`. The executed kernel $K$ predicts a chosen field through $KF$, and its conditional drift is $D_F=KF-F$. The conditional covariance $Q_{FG}=K(FG)-(KF)(KG)$ includes correlations between walkers and between operations. For a swarm average, the sum includes every pair of walker increments. Cloning can make those cross terms large. Replacing them by independent single-walker variances would change the predicted fluctuations.

This gives a practical route through the experiments. Measure the field from an archived state, fork the complete checkpoint into independent continuations, and compare the measured future values with the conditional prediction. VI-02 tests the clone gate law; VI-14 resolves the actual writes; VI-19 tests a source intervention through subsequent dynamics; VI-22 and VI-48 test the thermostat's conditional moments; VI-45 measures material metric drift and covariance; VI-51 resolves mechanical changes by stage. In VI-36 the same complete kernel controls multitime twistor observables. These calculations apply during transient evolution, including recorded killing and eligibility changes.

A smaller set of fields can be convenient for prediction. Its accuracy is an experimental question: does the retained information determine the next conditional increment? VI-12, VI-13, and VI-32 compare compressed predictors with later data and expose memory carried by discarded coordinates. A failed predictor identifies information the proposed description has lost. The full transition and its exact field hierarchy remain the objects being measured.

Several workbenches supply a finite matrix, density, connection, or spacetime geometry as their input. They explain an algebraic identity or a proposed approximation and make its numerical implementation inspectable. Connecting such an input to this gas requires its extraction from the complete algorithm and a comparison with measured behavior. The per-experiment source descriptions below identify that work explicitly. The [QFT validation report](https://github.com/FragileTech/fragile/blob/main/algorithmic-gas/QFT_VALIDATION.md) records derivations, independent checks, and remaining measurement gaps across all 66 entries.
:::

(sec-partvi-experiments-measured-outcomes)=
## What the measured comparisons establish

:::{div} feynman-prose
The full research run contains 544 trajectories and 1,632 audited archives, spanning 483,328 complete main updates and 78,381,056 main walker updates. The audit recomputes balance and curvature summaries from saved state. Exact kinetic telescoping has zero residual. Across 14 global configurations, standardized thermostat residuals range from $-2.7127$ to $1.8343$; the three local configurations give $0.5615$, $0.1905$, and $-0.2815$. These are comparisons with the executed conditional noise law, including its finite-step scale. The 36,864 independent replica updates also test conditional metric predictions: the global absolute standardized held-out residual has median $0.4831$ and 95th percentile $1.9131$. Uncertainty is computed within each configuration, because reused seed addresses can correlate runs across configurations.

A proposed two-coefficient spatial curvature/stress relation fares differently. In the local study its held-out RMSE is $51{,}363$, compared with $31{,}600$ for the constant predictor and $32{,}372$ after shuffling the training association. The global fit improves on the constant predictor by only $0.56\%$. This measured relation does not explain the sampled curvature variation. Its inputs are the averaged spatial Einstein tensor, centered kinetic second moment, and metric. Missing transport, clone contributions, volume normalization, or other state information must be investigated through the actual balances; an assumed constitutive equation would conceal the discrepancy.

VI-45 and VI-51 therefore answer distinct questions. The material metric experiment measures the metric's actual conditional evolution. The mechanical experiment measures copying, force work, noise, transport, and eligibility contributions. The supplied Einstein-tensor panel checks a stated tensor identity. The research fit tests whether a particular relation between measured spatial quantities predicts new data, and the errors above give its answer. Detailed per-configuration tables, archive checks, and reproduction commands are in the [physics verification report](https://github.com/FragileTech/fragile/blob/main/algorithmic-gas/VERIFICATION.md).

For spectral and compressed-channel experiments, inspect held-out complex errors and predictive losses directly. A computed CAR identity, a positive empirical covariance, or a fitted decay rate answers its own algebraic or statistical question. The evolving field is characterized by its observed covariance, phase, decay, and prediction error under the chosen algorithm. Those readouts retain their values when a compact spectral or constitutive description fits poorly.
:::

(sec-partvi-experiments-lattice)=
## Lattice QFT and recorded exterior constructions

:::{div} feynman-prose
The first seven workbenches start from finite objects: an oriented route, a pair of cloning scores, a Gram matrix, or a transition channel. Each can be displayed completely. Their algebraic identities prepare the reader to recognize the same structures in a recorded gas, while the density-kernel experiment shows how a specific continuum normalization emerges under refinement.
:::

(sec-partvi-experiment-01)=
### VI-01 — Oriented diagrams and transport

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-fg-lqft-wilson-loops`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. An algorithmic holonomy measurement requires a recorded transport connection; the color-projector triangle has its own observable definition.

Follow three oriented edges and multiply their SU(2) comparison matrices in traversal order. The curve compares the loop with its reversed traversal. A separate metric compares the trace before and after a basepoint frame conjugation. Conjugation preserves the normalized trace; reversal gives its complex conjugate. Change **Phase / angle** to change the links and **Amplitude** to change the frame rotation. This makes the cancellation visible even when the matrices do not commute. The independent check compares indexed matrix products, reversed products, and the conjugated loop product. The supplied unitary links are the experiment's inputs; color-projector triangle products are a separate observable.
:::

**Control values:** `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-02)=
### VI-02 — Cloning antisymmetry and exclusion

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-fermions`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

For every ordinary recorded clone gate, reconstruct the score from the source fitness, the pool-aligned donor fitness, and the configured regularizer. Divide by the actual saturation and clip to $[0,1]$ to obtain the predicted acceptance probability. Compare that value with the archived probability and with the realized gate. Historical donors use the fitness rescoring consumed by this update. Selected source references identify the exact immutable records.

The gate innovation is the accepted indicator minus its conditional probability. Its predicted conditional variance is $p(1-p)$, so accumulated innovations can be compared with their predictable variation without treating successive swarm states as independent samples. Deterministic revivals are counted separately. The reference fitness sweep displays the weighted score antisymmetry and the mutually exclusive positive directions; **Amplitude** controls that supplied pair. Recorded mode tests the configured gate itself, including its saturation and clipping.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-03)=
### VI-03 — Exterior algebra from recorded observables

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-lqft-recorded-fermionic-reconstruction`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Take one function of the whole swarm at each pre-clone frame: the eligible coordinate mean of positions, velocities, or their phase-space concatenation. **Recorded field** selects those functions. Center them with equal frame weights and diagonalize their empirical Gram matrix. The retained positive eigenspace supplies the actual modes used to build creation and annihilation matrices. Their anticommutator is compared with the projected measured Gram matrix, and the Fock dimension is $2$ raised to the retained rank.

**Fermionic modes** limits the retained modes; the discarded covariance norm reports the measured information excluded by that choice. Null directions do not supply extra modes. Exported epoch, step, generation, and contributing slot identities show exactly which complete records define the empirical law. Direct matrix multiplication checks the CAR identity, and an independent sum of squared replica determinants checks its exterior inner product. The empirical time series can be correlated; constructing its finite Gram quotient requires no independence assumption about the interacting walkers. Reference mode supplies a small Gram fixture whose own positive rank limits its occupation construction.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `modes` = 3; range 1–6; `amplitude` = 0.5; range 0–2; `fields` = `phase_space`; choices: phase_space, positions, velocities.

(sec-partvi-experiment-04)=
### VI-04 — Recorded transitions and CAR evolution

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-record-car-channel`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Measure eligible whole-swarm velocity means at two times separated by **Recorded lag**. An initial chronological block estimates the joint source/target law. Center and whiten the two marginal covariance matrices separately, then inspect the resulting transfer singular values. Separate whitening retains the observed change of marginal law during transient evolution. The covariance drift and a disjoint later block of cross moments show whether the measured transition description carries over to later data.

The measured one-particle contraction determines a vacuum-environment fermionic dilation. Explicit exterior-power Kraus matrices check its unitality, CAR action, and Choi positivity. The mode defect is $1-\eta_j^2$ in canonical singular coordinates. These residuals test the constructed channel, while the later cross-moment residual tests the recorded dynamics. A two-time measurement supplies a transition between its two marginal spaces; repeated powers require a separate test of temporal closure. **Recorded training fraction** selects the split, and **Fermionic channel modes** limits the constructed channel. Reference mode uses the supplied contractions $\eta_j=\exp[-r(1+0.2j)]$ and its explicitly defined repeated-power curve.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `modes` = 2; range 1–3; `rate` = 0.4; range 0.01–2; `steps` = 32; range 2–128; `lag` = 1; range 1–32; `train_fraction` = 0.6; range 0.25–0.8.

(sec-partvi-experiment-05)=
### VI-05 — Regional covariance and locality

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-record-locality-defect`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Take one observable in each of two regions. Their whitened cross-covariance is the number $\rho$ controlling the mixed CAR anticommutator. **Amplitude** changes $\rho$, with the numerical example bounded below one, and **Rate** changes the selected exponential spatial covariance profile. The operator calculation combines two explicit creation matrices and measures the anticommutator, while the covariance calculation supplies its predicted coefficient. The separation plot shows the chosen covariance model. It explains why the regional observable space and its covariance both matter: whitening an entire collection can mix the regions, changing the quantities whose locality is being measured.

In recorded mode, **Recorded spatial split** divides eligible pre-clone walkers by their first position coordinate. At each frame containing both regions, the calculation takes the regional mean of the first velocity component. Their paired time courses supply the empirical variances and cross-covariance; the resulting normalized coefficient enters a separate two-mode CAR matrix calculation. The operator residual checks that its mixed anticommutator has the measured coefficient. Move the split and inspect the regional population counts: a nearly empty region changes the observable's sampling quality. The covariance describes the retained paired frames, and its normalization uses their actual count. The reference amplitude and exponential spatial rate belong to the supplied covariance fixture.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2; `region_split` = 0; range -1–1.

(sec-partvi-experiment-06)=
### VI-06 — Products and replica wedges

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {prf:ref}`thm-lqft-product-obstruction`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Compare the ordinary product of two functions evaluated on the same full swarm record with their antisymmetrized product evaluated on two independent draws from the empirical record law. Recorded mode uses the centered whole-frame observables selected by **Recorded field** and **Measured Gram modes**. It computes the squared determinant directly over all pairs of empirical records and compares that sum with the separately calculated two-by-two Gram determinant. The factor $1/\sqrt{2!}$ fixes the exterior normalization.

The product law here samples complete recorded states. Walkers within one swarm keep their interactions and common cloning history. The same-frame product remains a distinct measured quantity, so a zero wedge can be read precisely as linear dependence in the selected Gram space. Reference mode uses a two-state example with $g=a f$; **Amplitude** varies $a$ and makes the vanishing exterior product visible term by term.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `modes` = 6; range 1–6; `fields` = `phase_space`; choices: phase_space, positions, velocities.

(sec-partvi-experiment-07)=
### VI-07 — Density-aware continuum operators

**Placement:** {doc}`2_fractal_set/03_lattice_qft` at {ref}`sec-scalar-fields`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The continuum comparison uses the supplied density and kernel normalization. Applying it to the gas requires measuring the executed donor kernel and its sampling density.

A kernel samples both the field and the density of points around it. Here $f(x)=\sin x$, $\rho(x)=1+a\cos x$, and the query is $x=0.35$. **Amplitude** sets $a=0.4\,\mathrm{Amplitude}$. The bandwidth curves compare unnormalized, row-normalized, and inverse-density-corrected Gaussian quadrature. The unnormalized limiting value is $(\rho f''+2\rho'f')/2$; row normalization divides by $\rho$, while inverse-density correction gives $f''/2$. These limits explain the drift generated by uneven sampling.

**Independent samples** sets the number of independently drawn periodic points with density $\rho/(2\pi)$. Their periodized Gaussian sum estimates the unnormalized operator, alongside its estimated plus and minus two standard errors. The same population supplies every bandwidth, so points on the bandwidth curve are correlated. Deterministic quadrature supplies an independent finite-bandwidth comparison, and analytic derivatives supply the limiting target. Increase the sample count to study sampling variation, then inspect how the quadrature approaches its target as bandwidth shrinks. These two comparisons separate sampling error from the kernel's finite-range bias.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `n` = 512; range 32–4096.

(sec-partvi-experiments-standard)=
## Direct fields and Standard Model coordinates

:::{div} feynman-prose
These experiments follow the observable map from force and velocity to color, from color to invariant coordinates, and from direct channels to predictions. Keep the individual descriptors long enough to inspect their masks and joint law. The later averages and effective actions then have an explicit source.
:::

(sec-partvi-experiment-08)=
### VI-08 — Recorded color states

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-direct-observables`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Each component of the viscous force receives the velocity phase $e^{i\kappa v^a}$. Normalizing the resulting complex vector produces the recorded color. **Phase / angle** sets $\kappa$ and **Threshold** sets the force-norm cutoff. The reference fixture includes small forces around that threshold; **Amplitude** changes its force scale. Recorded mode reads the actual B1 viscous force together with its matching input velocity. Compare the raw norm, the valid mask, and the zero-extended color: a subthreshold vector may have a nonzero raw norm while its masked observable is zero. This placement makes the validity convention explicit before the Gram, determinant, and triangle channels use it.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2; `threshold` = 1e-12; range 1e-15–1.

(sec-partvi-experiment-09)=
### VI-09 — SU(3) invariant coordinates

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-direct-orbit-isomorphism`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Read valid color vectors from the actual B1 viscous force and its matching input velocity. Select consecutive valid triples, retaining their slot mapping and masks. Compute complex pair inner products, the baryon determinant, the triangle product, and the rank-one projectors. A common non-diagonal SU(3) transformation should preserve those invariant coordinates. Independent ray phases preserve the triangle, while the baryon records the determinant phase of a common U(3) change.

Two independent contractions check the measured colors: dense projector multiplication gives the triangle trace, and the Gram determinant gives the baryon squared modulus. **Phase / angle** sets the color phase scale and **Recorded force threshold** selects force validity. **Recorded common SU(3) rotation** changes the common frame while holding the extracted colors fixed. **Amplitude** selects the reference fixture. Fewer than three valid colors leaves the triple readout unavailable, with the contributing mask visible. Reference mode uses supplied color vectors to inspect the same algebra. The color projectors and invariants are explicit algorithmic observables; constructing a transport connection is a separate operation.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2; `threshold` = 1e-12; range 1e-12–1; `rotation` = 0.7; range -3.14–3.14.

(sec-partvi-experiment-10)=
### VI-10 — Companion doublets

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`def-sm-direct-companion-doublet`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Follow the cloning companion actually selected for each walker, then follow that companion's own selected edge. The two complex directed-edge scalars form the doublet. **Recorded comparison length** sets the distance scale and **Recorded fitness phase scale** sets the fitness phase scale. Recorded distance and fitness values use the configured metric, source regularizer, and donor rescoring. A historical second hop resolves the immutable source record's own companion scalar; a source outside archive coverage is reported through its mask.

For independent count-one cloning selection, the configured categorical kernel supplies the exact conditional scalar law. Compute its variance both about its complex mean and through the pairwise half-sum formula, then compare selected scalar innovations with their conditional means. The observed doublets retain the actual joint directed map. For mutual or greedy dependent samplers, the doublets are still measurable, while a full conditional doublet-variance comparison requires that sampler's joint law. Reference mode enumerates eight assignments of a three-walker model; its independent samples test the stated finite law and its two-hop correlations.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2; `n` = 512; range 32–4096; `length` = 1; range 0.1–3; `phase_scale` = 1; range 0.1–3.

(sec-partvi-experiment-11)=
### VI-11 — Attribution holonomy

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`prop-sm-attribution-holonomy-defect`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The node frames and attribution factors are supplied here. Their law must be extracted from executed records before the curve predicts algorithmic holonomy.

Transport defined entirely by coherent node frames telescopes around a closed loop. An extra attribution rotation on an edge changes the product. **Phase / angle** and **Amplitude** change the node-frame inputs; **Rate** controls the attribution mismatch. The plot follows both products as their strength changes. Direct ordered multiplication checks the coherent loop against the identity and compares the attributed loop with the declared mismatch construction. This puts a numerical size on the coherence condition in the proposition: the geometric frames and the extra attribution factor are both explicit, so a nonzero holonomy has an identifiable source.
:::

**Control values:** `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2.

(sec-partvi-experiment-12)=
### VI-12 — Channel compression and memory

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-direct-channel-memory`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Project a four-state transition matrix onto a selected channel and repeatedly apply the projected one-step map. Compare that curve with the same channel evaluated after full transition powers. **Amplitude** changes the coupling to discarded coordinates; **Lag / horizon** extends the comparison. Their two-step difference is the product $BC$, equal to $b^2$ in this symmetric fixture. A third calculation retains the memory recurrence and recovers the full answer. At zero coupling the recurrence closes in the retained channel. The three curves let the reader see exactly which term carries information that leaves the selected channel and later returns.

Recorded mode fits a finite predictive partition to the **Recorded descriptor** time series. **Predictive states** sets the requested quantile partition, **Training fraction** selects the chronological training interval, and **Prediction lag** sets the longest tested horizon. Quantile edges, occupancy, and the one-step transition matrix are fitted using training observations only. A guard interval separates those data from the validation block. Duplicate quantiles reduce the effective number of bins, which the result reports. Missing frames and unseen training departure rows produce explicit availability diagnostics.

For each lag, the plots compare the fitted matrix-power predictor with a training-occupancy baseline through their held-out Brier scores and compare the predicted transition rows with empirical held-out frequencies. Their sample counts remain visible; the time series retains its temporal dependence. A separate training-matrix calculation compares a coarsened two-step map with two applications of the coarsened one-step map. Its algebraic residual measures memory introduced by the chosen grouping. **Amplitude**, the reference sample count, and the reference horizon affect the finite fixture described above.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `steps` = 32; range 2–128; `bins` = 4; range 2–8; `lag` = 2; range 1–8; `train_fraction` = 0.6; range 0.5–0.8; `descriptor` = `mean_speed_squared`; choices: mean_speed_squared, mean_position_x, mean_velocity_x.

(sec-partvi-experiment-13)=
### VI-13 — Predictive partition refinement

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`thm-sm-predictive-partition-convergence`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

A coarse descriptor merges states that can have different future expectations. Refine its partition and compare the prediction with the complete four-state conditional law. **Amplitude** changes the transition fixture and **Independent samples** changes the independent validation sample. The native calculation obtains predictors from exact conditional probabilities, then evaluates their loss using a separate random sample. The details retain the transition matrix, conditional predictor, and standard errors. Refinement reveals when a descriptor resolves a predictive distinction. Since the predictor is known exactly in this fixture, validation fluctuations can be separated from the error caused by grouping distinct states together.

Recorded mode estimates the descriptor's finite conditional law from an initial chronological block and tests it on a later block. **Recorded descriptor**, **Predictive states**, **Training fraction**, and **Prediction lag** define the observable, partition, and split. Quantile edges use training origins only; repeated edges collapse to fewer effective bins. The one-step training matrix predicts each held-out lag through its matrix powers. The Brier-score curves compare those predictions with a fixed training-occupancy baseline, while matrix errors and counts expose the occupied transitions. Increase the requested bin count and compare both predictive loss and occupancy. A finer description can resolve more distinctions while leaving fewer observations per transition. The guard interval prevents overlap of the fitted and validated transitions; exported counts describe the temporally correlated observations. **Independent samples** controls the separate exact-law reference validation.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `n` = 512; range 32–4096; `bins` = 4; range 2–8; `lag` = 2; range 1–8; `train_fraction` = 0.6; range 0.5–0.8; `descriptor` = `mean_speed_squared`; choices: mean_speed_squared, mean_position_x, mean_velocity_x.

(sec-partvi-experiment-14)=
### VI-14 — Complete updates and pair cancellation

**Placement:** {doc}`2_fractal_set/04_standard_model` at {prf:ref}`prop-sm-implemented-collision-increments`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

For every accepted recipient, reconstruct the literal copied velocity from the weighted immutable donor rows. Compare it directly with the recorded `literal_clone` stage. Sum incoming donor contributions and subtract the overwritten recipients to predict the total copying momentum change. Historical donors use their actual source populations. A missing source is a capability error because the write cannot then be independently reconstructed.

Follow the later transform, reconciliation, and boundary stages through their measured momentum increments and eligibility counts. General directed copying can change total momentum; the reciprocal disjoint-pair restitution configuration has its own conservation identity. This makes the mechanism visible: a conservation calculation must describe the writes the engine actually executes. Reference mode enumerates acceptance patterns of a balanced three-walker donor map, with **Amplitude** setting the acceptance probability. That reference cancellation can be compared with the measured directed-copying budget without replacing it.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-15)=
### VI-15 — Coupling statistics and CP

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-coupling-matching`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The mixture is an explicit color law. Estimating CP statistics of the gas requires the corresponding joint law of its recorded colors.

Complex conjugation reverses a CP-odd triangle component. Pair a color fixture with its conjugate and vary their mixture to inspect the cancellation. **Phase / angle** controls the color phases and **Amplitude** changes the mixture or invariant comparison used by the fixture. The plot compares the enumerated odd average with bias times the odd observable, vanishing at symmetric weighting. The calculation also displays the fundamental SU(3) Casimir $4/3$ in the stated normalization. These dimensionless group quantities are read directly from the selected algebraic construction. No mass or coupling calibration is needed to test the conjugation identity.
:::

**Control values:** `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-16)=
### VI-16 — Effective action and generating functions

**Placement:** {doc}`2_fractal_set/04_standard_model` at {ref}`sec-sm-wilson-loops`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Sixty-four length-two histories of a four-state chain can be listed completely. Sum their probabilities before and after introducing a source coupled to the chosen descriptor. **Amplitude** changes the transition law and **Phase / angle** changes the source. The partition function satisfies $Z(0)=1$; its derivative equals the weighted descriptor moment. The workbench calculates that moment by enumeration and compares it with a symmetric finite difference of the partition function. It also groups histories with the same descriptor before taking a logarithm. This shows concretely how marginalizing a history produces an effective action for the retained observable.

Recorded mode pairs primitive path-action accounting with an empirical generating function. Each retained step contributes the bounded descriptor equal to the hyperbolic tangent of the mean terminal position coordinate. The displayed empirical partition function averages its exponential source weight uniformly over retained steps. **Phase / angle** sets that source. Direct differentiation of the finite sum is checked against a symmetric difference quotient, and its value at zero is one. The empirical average describes the recorded step sample, including its temporal dependence. The separately reported execution likelihood supplies sampler, gate, revival, and innovation action components; it is not used to reweight this empirical partition function. This makes both the measured law and the differentiation rule inspectable.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `angle` = 0.7; range -3.14–3.14.

(sec-partvi-experiments-yang-mills)=
## Yang–Mills, response, and quantum reconstruction

:::{div} feynman-prose
A source changes a probability law, a frame change changes coordinates, and a conditional expectation predicts an update. The following workbenches put those operations in separate calculations and then connect them. Some predictions are cancellations, some are decay laws, and the labeled reflection example predicts a negative value. Each result is meaningful because the measured operation is specified.
:::

(sec-partvi-experiment-17)=
### VI-17 — Executed path-action accounting

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-first-principles`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

The finite reference transition combines a companion choice, an accepted gate, a Gaussian innovation, and survival. **Amplitude** changes the companion factor, **Rate** the Gaussian width, and **Phase / angle** its source mean. Multiply the conditional factors and compare their negative logarithm with the sum of the four action contributions. The Gaussian normalization contributes $\log\sigma$ alongside the squared standardized increment. Changing the width makes this determinant contribution visible.

Recorded mode evaluates the probabilities of the actual companion assignments, accepted and rejected clone gates, revival donors, and sampled innovations. The component plot and cumulative action show how their negative logarithms add over retained steps. Gaussian terms separate the quadratic form, normalization constant, and factor log determinant. The default independent uniform sampler is supported, as are uniform sampling without replacement, uniform Fisher–Yates matching, and supported Gaussian-weighted independent choices. Historical donor weights use the recorded source states. The exact supported-law and unavailable-component conventions appear in the native path-accounting section below.

The likelihood is conditional on the retained starting state and epoch anchors. Its discrete factors use counting measure on sampler outcomes; a nonsingular Gaussian raw-noise density uses Lebesgue measure on $B\xi$, while a singular map or uniform innovation retains its latent innovation-coordinate density. Deterministic copying and subsequent stage maps carry their own state updates. This choice describes the primitive execution trace, including sampled innovations that a later map may discard. A final-state density requires a separate pushforward calculation. The survival readout retains the executed transition factors with their unconditioned normalization.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2; `angle` = 0.7; range -3.14–3.14.

(sec-partvi-experiment-18)=
### VI-18 — Kinetic metric from one record

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-kinetic-metric-correspondence`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

For each eligible archived A1 row, read the actual $d\times r$ diffusion factor and the sampled raw innovation. The factor and the Gaussian or standardized uniform innovation law predict the second and fourth moments before thermostat integration scaling. An addressed source shift contributes its nonzero mean. Recorded mode compares these predictions with the executed components, retaining the factor rank and eligibility mask.

The fourth-moment prediction includes the innovation's fourth cumulant. It is zero for Gaussian innovations and $-6/5$ for standardized uniform draws. Thus two noise laws with the same covariance can be distinguished by this measurement. In a rank-one, three-dimensional uniform run with donor history two, 96 updates, and eight walkers, the measured second moment is $1.74313$ against $1.75$ predicted; the fourth is $10.30073$ against $10.2375$. These aggregates retain their trajectory dependence. VI-22 and VI-48 include the physical damping and timestep factors for the complete O-stage prediction.

Reference mode freezes a scalar Gaussian factor $s$, selected by **Amplitude**, and uses **Independent samples** to test $\mathbb E(s\xi)^2=s^2$ and $\mathbb E(s\xi)^4=3s^4$. Its independent sampling uncertainty belongs to that supplied scalar experiment.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `n` = 512; range 32–4096.

(sec-partvi-experiment-19)=
### VI-19 — Gaussian source response

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-native-source-response`.

:::{div} feynman-prose
**Measurement source:** Independent complete-checkpoint engine continuations, with a selectable finite reference.

In reference mode the observable is a threshold of a Gaussian innovation. Shift the innovation and reevaluate the threshold, then compare with likelihood reweighting and Gaussian quadrature. **Amplitude** moves the reference shift and **Independent samples** controls its draws. The Hermite readouts give derivatives at zero source. Recorded mode uses complete gas continuations from a frozen checkpoint: one addressed future Gaussian innovation is shifted, and all subsequent choices and updates run again. **Engine source shift**, **Engine replicas per independent group**, and **Engine continuation steps** set that experiment. Independent shifted and weighted groups expose their different variances while testing the same source-response identity.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `replicas` = 16; choices: 8, 16, 32, 64; `horizon` = 4; choices: 1, 2, 4, 8; `theta` = 0.25; range 0.01–1.5; `amplitude` = 0.5; range 0–2; `n` = 512; range 32–4096.

(sec-partvi-experiment-20)=
### VI-20 — Force sources and information

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-metric-force-sources`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The scalar source model illustrates score compression. The complete-engine Gaussian innovation intervention is measured in VI-19; an archive-derived metric force-source score remains a separate measurement.

A Gaussian mean shift of size $\theta$ costs path relative entropy $\theta^2/2$. Keeping only the sign of the sample retains less information. **Amplitude** sets $\theta$ and **Independent samples** controls the likelihood and score averages. The workbench compares measured path KL with its exact expression and computes the Bernoulli descriptor KL from the threshold probability. At zero source the path Fisher information is one and the sign descriptor retains $2/\pi$. The curves teach the information loss caused by an observable map. Analytic Gaussian probabilities and independently sampled likelihoods provide separate calculations of the comparison.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `n` = 512; range 32–4096.

(sec-partvi-experiment-21)=
### VI-21 — Geometry-fiber disintegration

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-native-geometry-fiber-action`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Arrange a joint law as a two-by-three table: two geometry states, each with three fiber states. **Amplitude** changes its energies and **Phase / angle** changes its source weighting. Sum rows to obtain the geometry law and divide within each row to obtain the conditional fiber law. The action at each entry then separates into minus the logarithm of its row mass and minus the logarithm of its conditional probability. The native calculation checks all normalizations and the maximum decomposition residual. The plot makes the contributions visible one entry at a time, giving disintegration a finite probability calculation the reader can reproduce by hand.

Recorded mode constructs a joint histogram of two bounded step descriptors: the hyperbolic tangent of mean terminal position and the accepted clone fraction. It reports their joint, marginal, and conditional probabilities from the same integer counts. Four bins per coordinate are used by the browser; native requests can select `bins` from two to sixteen. The histogram plots and normalization residuals check disintegration of this empirical law exactly. Empty geometry bins have zero marginal mass and an absent conditional law. These histograms describe retained, potentially dependent trajectory steps; the primitive likelihood accounting is exported separately. The source and energy controls apply to the finite reference table.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `angle` = 0.7; range -3.14–3.14.

(sec-partvi-experiment-22)=
### VI-22 — Stochastic Noether balance

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-noether`.

:::{div} feynman-prose
**Measurement source:** Independent complete-checkpoint engine continuations, with a selectable finite reference.

Freeze the complete gas state and run two independently seeded continuation groups. The selected observable's full change includes cloning, forces, noise, boundary handling, and eligibility. One group estimates its conditional drift; the other measures the same conditional increment independently. **Engine replicas per independent group** and **Engine continuation steps** set the sample size and horizon.

A second comparison has an analytic prediction. At each actual A1 state, use the executed O-stage damping, factor, innovation law, and source shifts to calculate conditional momentum and kinetic-energy moments. Subtract those predictions from the raw O-stage measurements and sum the centered increments along each complete continuation. Sum the corresponding predictable variances as well. Each continuation is one independent sampling unit, preserving the dynamics and dependence among its stages.

Across 384 continuations from three initial seeds with donor history two, viscosity, and horizon three, the reported momentum residual sum is $0.03094$ with standard error $0.11834$; the energy residual sum is $-0.49027$ with standard error $0.51571$. Realized martingale variance divided by its prediction is $0.97501$. The complete stage ledger explains how the conditional thermostat balance combines with the other operations. Reference mode separately uses an autoregressive chain with known drift; **Rate**, **Lag / horizon**, and **Independent samples** control that calculation.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `replicas` = 16; choices: 8, 16, 32, 64; `horizon` = 4; choices: 1, 2, 4, 8; `rate` = 0.4; range 0.01–2; `n` = 512; range 32–4096; `steps` = 32; range 2–128.

(sec-partvi-experiment-23)=
### VI-23 — Wilson variation and Ward identity

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`lem-ym-discrete-ward`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The links are supplied. A gas Ward-divergence measurement requires a recorded connection and its actual intervention law.

Differentiate the finite Wilson action $1-\operatorname{Re}\operatorname{Tr}(U_xU_y)/2$. **Phase / angle** varies one link and **Amplitude** sets the other. Direct matrix multiplication evaluates the action; the displayed symmetric finite-difference error approaches the analytic derivative $\sin(a)\cos(b)$. A separate simultaneous conjugation follows the gauge orbit, along which the closed-product action remains constant. The two operations distinguish a physical link variation from a change of frame. Their residuals are computed from explicit matrices and perturbations, making the Ward cancellation and ordinary action derivative independently inspectable.
:::

**Control values:** `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-24)=
### VI-24 — Continuum Yang-Mills consistency

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-continuum`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The connection is supplied. Relating its small-loop limit to the gas requires an extracted transport connection and a measured refinement regime.

Even a spatially constant connection can have curvature when its two matrix components fail to commute. **Amplitude** and **Rate** set two noncommuting SU(2) generators. Construct the ordered square loop at successively smaller side length $h$, subtract the identity, divide by $h^2$, and compare with the commutator. The displayed error decreases as the loop becomes small. The curvature reference is calculated directly from the generators, while the measured curve uses finite group elements and their ordered product. This gives the small-loop expansion a concrete matrix example with no spatial derivative term to obscure its non-Abelian contribution.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2.

(sec-partvi-experiment-25)=
### VI-25 — Units, discrete time, and gaps

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-constants`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The finite generator is supplied. Estimating an algorithmic spectral gap requires measured transitions or correlations of the complete gas.

For a symmetric two-state chain, the centered one-step eigenvalue is $e^{-2qh}$. **Rate** sets $q$, **Time step** sets $h$, and **Lag / horizon** sets the number of displayed steps. Repeated matrix evolution and the exponential decay curve agree when plotted against the same algorithmic time. The calculation recovers the generator rate through $-\log\lambda/h$ and compares it with $2q$.

A refinement plot applies the same conversion to the Euler transition, whose centered eigenvalue is $1-2qh$, at successively smaller timesteps. Its recovered rate approaches $2q$, while the exact sampled semigroup gives that rate at every tested timestep. This makes the distinction between a time-unit conversion and a discretization error visible. Keep the rate fixed while changing the timestep, and compare both curves against the shared generator target. The result records the time convention needed before attaching physical units.
:::

**Control values:** `rate` = 0.4; range 0.01–2; `steps` = 32; range 2–128; `dt` = 0.1; range 0.005–0.2.

(sec-partvi-experiment-26)=
### VI-26 — Fluctuation and collision budgets

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-complete-fluctuation-equations`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The shared swap gate defines this reference joint law. VI-14 and VI-51 measure the configured clone and mechanical updates.

A shared binary gate either leaves a pair unchanged or swaps its values. **Amplitude** controls the gate probability. The two increments have a negative covariance $-p(1-p)(y-x)^2$, and their sum is zero. The product increment contains both first-order terms and the quadratic cross term. The workbench enumerates the two outcomes and plots those contributions, then checks their complete sum against the direct product change. It explains why replacing a joint collision by independent gates changes a fluctuation equation: the same mean behavior can conceal a different cross-walker covariance.
:::

**Control values:** `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-27)=
### VI-27 — QSD windows and burn-in

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`thm-ym-qsd-window-relaxation`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The killed matrix and its QSD are known inputs. The gas requires its own conditional-law and relaxation measurements.

A positive two-state killed kernel has a dominant left eigenmeasure and a survival eigenvalue $\alpha$. **Amplitude** changes killing and **Lag / horizon** extends the evolution. Repeated subprobability evolution gives surviving mass; renormalizing that same vector gives the conditional state law. The plot compares its distance from the QSD with the stated finite-window bound containing $\alpha^{-2}$. The eigenmeasure is obtained independently from the two-by-two spectral formula. The reader can see the two normalizations at once: convergence of the conditional state and loss of total mass are different measurements of one killed transition.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `steps` = 32; range 2–128.

(sec-partvi-experiment-28)=
### VI-28 — Reflection matrices

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-native-labeled-color-reflection-sign`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The off-diagonal reflected moments are supplied. A corresponding gas reflection test requires those moments under its actual labeled-word law.

Choose the explicit reflected two-word matrix $H=\begin{pmatrix}1&u\\v&0\end{pmatrix}$ and the word coefficient vector $(-(u+v)/2,1)$. Its quadratic value is $-(u+v)^2/4$. **Amplitude** sets $u$ and **Rate** sets $v$. Direct matrix evaluation is plotted against the negative formula, with the Hermitian defect shown separately when $u\ne v$. This experiment belongs beside the chapter's labeled-word sign calculation: the negative value is its prediction. It also gives a precise test to perform before attempting a positive reflection quotient of a selected observable family.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2.

(sec-partvi-experiment-29)=
### VI-29 — Scalar triangles and outer plaquettes

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-native-scalar-face-evaluation`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The scalar links are supplied. Their native geometric and gauge source construction must be measured to identify algorithmic face observables.

Assign scalar edge phases from differences of four potentials. Their product around the outer loop telescopes to one, while the separately defined triangle readout can retain a defect. **Phase / angle** and **Amplitude** change the potentials. The workbench evaluates the oriented outer product and triangle expression independently and displays both. The result includes the effective phase scale and the actual complex products. This makes it possible to inspect which phase assignments cancel on the outer route and which terms belong to the triangle readout. A changing triangle curve therefore need not imply a changing outer scalar plaquette.
:::

**Control values:** `angle` = 0.7; range -3.14–3.14; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-30)=
### VI-30 — Translation and localization

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {prf:ref}`prop-ym-anchored-regulator-translation-test`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The anchored readout is explicit. A comparison of translated gas laws must additionally retain the configured boundaries, rewards, and donor rules.

Measure a fixed point configuration with a Gaussian window anchored at zero, then translate the points. **Amplitude** sets the translation and **Rate** sets the window scale. One curve keeps the regulator fixed; another translates its anchor along with the points. The anchored measurement generally changes, while the co-transformed measurement agrees with the initial value. The independent check evaluates both sums directly. This is a compact example of why the transformation rule for a localization function belongs in the observable's definition: the same translated state can produce different answers under two explicitly different measurement protocols.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2.

(sec-partvi-experiment-31)=
### VI-31 — Regional algebras and interventions

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-qft-axioms-verification`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The ring transition is supplied. Actual regional interventions must rerun the gas with its full dependency structure.

A perturbation starts at one node of a nine-node ring whose update couples nearest neighbors. **Amplitude** sets the hop probability and **Lag / horizon** extends the propagation. Repeated multiplication by the full transition matrix measures the response at a target four edges away. It stays zero before the update's dependency cone reaches that target. The result separately records the commuting multiplication-algebra statement. The ring example lets the reader trace the actual sequence of dependencies responsible for a response. It provides a finite update comparison for the chapter's discussion of regional algebras and interventions.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `steps` = 32; range 2–128.

(sec-partvi-experiment-32)=
### VI-32 — Connected algorithmic QFT

**Placement:** {doc}`2_fractal_set/05_yang_mills_noether` at {ref}`sec-ym-algorithmic-qft-synthesis`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Start from a stationary four-state law, center a readout, form its Gram variance, and follow its multitime correlation. **Amplitude** controls coupling to an omitted channel; **Lag / horizon** controls how long the comparison runs. The plot puts the full conditional correlation beside the compressed Markov prediction and a one-mode CAR multiplicative defect. The exported transition matrix and stationary-law residual expose each stage of the calculation. This workbench ties together the chapter's construction as one finite pipeline. The memory correction remains a measurable part of the pipeline whenever the selected one-channel description loses predictive information.

Recorded mode connects descriptor extraction, a training-only quantile partition, the fitted transition matrix, and later-block predictive checks. **Recorded descriptor**, **Predictive states**, **Training fraction**, and **Prediction lag** determine that pipeline. Its trace plot identifies training, guard, and validation observations; transition heatmaps retain the fitted matrix and its empirical comparison. Brier scores compare matrix-power predictions with the occupancy baseline at each lag. A weighted coarse partition of the fitted matrix gives a separate two-step memory residual. This single recorded pipeline lets the reader follow an observable from source frames through compression to a numerical prediction. The reference amplitude and horizon continue to select the completely specified four-state construction.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `steps` = 32; range 2–128; `bins` = 4; range 2–8; `lag` = 2; range 1–8; `train_fraction` = 0.6; range 0.5–0.8; `descriptor` = `mean_speed_squared`; choices: mean_speed_squared, mean_position_x, mean_velocity_x.

(sec-partvi-experiments-twistors)=
## Twistor constructions and spectral readouts

:::{div} feynman-prose
The twistor section starts with a two-by-two determinant identity and moves to the effective edge map on a recorded triplet. Once a readout is fixed, its temporal tracking and covariance define the spectral problem. The sequence makes that order of construction explicit.
:::

(sec-partvi-experiment-33)=
### VI-33 — Spinor and mass identities

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-twistor-spinor-conventions`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The bispinors and vectors are supplied. VI-34 resolves effective spinor fields from recorded algorithmic edges.

Build a Hermitian two-by-two bispinor from two spinors and read off its four-vector using the Pauli matrices. **Phase / angle** changes a complex phase and **Amplitude** changes the second spinor. The determinant of the bispinor equals $E^2-|\mathbf p|^2$ in the displayed $+---$ signature. A single outer product has determinant zero and supplies the null comparison. Independent determinant and component calculations check the mass identity. The workbench gives the spinor factorization a geometric interpretation: adding a second independent contribution can turn a null bispinor into a nonnull one.
:::

**Control values:** `amplitude` = 0.5; range 0–2; `angle` = 0.7; range -3.14–3.14.

(sec-partvi-experiment-34)=
### VI-34 — Recorded triplet twistor operators

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-fractal-set-twistorization`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

For each distance/clone triplet, form the edge four-vectors and their effective bispinors, select the normalized spinors, and compute the bracket-based scalar, pseudoscalar, and squared-modulus readouts. **Amplitude** controls the velocity contribution in the reference and recorded edge maps. Recorded mode uses pre-clone positions, velocities, companions, and the physical timestep of a three-dimensional gas. The plot retains one entry per valid triplet, and the exported details include the mask and readouts. Historical donors resolve their immutable source positions, velocities, and validity flags; the result reports source coverage explicitly. These details make the actual effective edge map reproducible before its temporal correlations are formed.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2.

(sec-partvi-experiment-35)=
### VI-35 — Source-frozen lag tracking

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {ref}`sec-twistor-vs-euclidean`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Choose a triplet at the source time and retain its source references through the lag. Current source slots advance to the later frame; historical source snapshots remain fixed. Compare its complex correlation with a construction that chooses the companions again at the sink. **Amplitude** sets the effective velocity contribution and **Lag / horizon** sets the maximum requested lag. Recorded mode gathers both readouts from the stored frames. The real and imaginary curves have separate valid source/sink counts, retained in the output. A current numerical slot can change generation through cloning; its lag readout follows that slot. A historical source retains its recorded generation and values. The comparison teaches how sink reselection changes an observable and why each lag must be normalized by its own available comparisons.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `amplitude` = 0.5; range 0–2; `steps` = 32; range 2–128.

(sec-partvi-experiment-36)=
### VI-36 — Spectral decay and twistor amplitudes

**Placement:** {doc}`2_fractal_set/08_twistor_formulation` at {prf:ref}`thm-effective-twistor-spectral-meaning`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Recorded mode starts from a precisely defined complex field. **Recorded spectral channel** selects phase-space coordinates $x^j+i v^j$ or the triplet scalar and Pauli readouts. **Recorded correlator** defaults to `frame_mean`: zero-extend every invalid local readout and average over the fixed population size $N$. Each frame's twistor fields use its actual current or immutable historical donor sources. This is the whole-state observable in {prf:ref}`thm-effective-twistor-spectral-meaning`. Its transient correlation follows the complete executed kernel, with the source and sink means belonging to their respective marginal laws.

The selected window estimates complex temporal correlations. **Recorded channels** selects up to three channels, **Maximum recorded lag** sets the lag window, and **Fit start lag** and **Fit end lag** define the fitting interval. Training frames alone set centering, the positive covariance subspace, and fitted decay/frequency parameters. A guard interval precedes the held-out block. The window's training-centered estimate and its held-out counterpart are reported with their counts; pooling a transient window does not impose an invariant law on the gas.

Choose `source_pairs` to inspect the separate source-frozen diagnostic. Current source slots advance through the lag and historical snapshots remain fixed. Each lag divides by its own valid-pair count. This observable differs from the fixed-$1/N$ frame mean: it retains selected source/sink local pairs rather than all cross-walker products of the frame sums. Comparing the two modes shows how tracking and normalization change the measured correlation.

Nonzero-lag matrices can be non-Hermitian. The calculation retains complex eigenvalue phase and magnitude, checks the characteristic-polynomial residual, and fits decay and frequency in algorithmic time. The held-out complex RMSE measures predictive error and can reveal a poor exponential fit. In the audited seed-516 run with donor history two, the frame Gram value is $0.8500623473$ by both direct calculation and the readout. Its fitted mode has decay rate $-1.78624$, frequency $-18.08976$, and held-out complex RMSE $0.39244$. The negative decay parameter describes growth over that fit window. These measured transient oscillations remain part of the field description. The result keeps `mass` null; positive real fitted decay is reported by its own name. A mass interpretation requires the positive transfer representation of {prf:ref}`cor-effective-twistor-positive-transfer` for these same algorithmic fields. Reference mode supplies two known exponential decays and a complex mixing matrix, allowing whitening and spectral extraction to be checked independently.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `aggregation` = `frame_mean`; choices: frame_mean, source_pairs; `amplitude` = 0.5; range 0–2; `rate` = 0.4; range 0.01–2; `steps` = 32; range 2–128; `readout` = `phase_space`; choices: phase_space, twistor; `channels` = 2; range 1–3; `max_lag` = 12; range 2–24; `fit_start` = 1; range 1–8; `fit_end` = 6; range 2–16.

(sec-partvi-experiments-gravity)=
## Curvature, transport, and congruences

:::{div} feynman-prose
A geometric measurement needs a connection, a metric, and a rule for following the object being measured. The curvature workbenches make these inputs visible through transport, area normalization, metric derivatives, and material volumes. Their reference geometries let numerical integration and differentiation be checked independently.
:::

(sec-partvi-experiment-37)=
### VI-37 — Transport in a changing geometry

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-tessellation-to-curvature`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The density and finite-volume drift are supplied. An algorithmic density equation must also retain the measured cloning, memory, and boundary contributions.

Transport an initially spatial vector along a spatial path in an expanding Lorentzian slab. Ambient transport develops a time component $-\sinh(Hx)$, while intrinsic transport on the selected flat spatial slice keeps that component zero. **Expansion** sets $H$, **Length** sets the path, and **Resolution** sets the RK4 subdivision. The plot compares numerical ambient transport with the exact hyperbolic function and the intrinsic reference. Its metric-norm residual checks preservation of $-V_t^2+V_x^2$. This placement establishes which connection a geometric measurement uses before interpreting a loop or curvature estimate.
:::

**Control values:** `expansion` = 0.4; range -1–1; `length` = 1; range 0.05–3; `resolution` = 32; range 4–128.

(sec-partvi-experiment-38)=
### VI-38 — Curvature per loop area

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-riemann-scutoid-dictionary`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The local geometric inputs are supplied. VI-39 applies the fitness geometry to the actual recorded conditional field.

Transport a tangent vector around coordinate rectangles on the unit sphere. The return angle divided by the enclosed spherical area should give curvature one. **Latitude** keeps the chart away from the poles and changes the rectangle location; **Resolution** controls integration along each edge. The workbench runs a sequence of loop widths, plots the recovered curvature, and separately shows numerical error. The measured return angle comes from ordered Levi-Civita transport, while the reference area is $w[\cos\theta-\cos(\theta+w)]$. This makes the area normalization explicit: a smaller loop gives a smaller angle without changing the curvature it measures.
:::

**Control values:** `latitude` = 0.8; range 0.3–1.2; `resolution` = 64; range 8–256.

(sec-partvi-experiment-39)=
### VI-39 — Conditional fitness curvature

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-algorithmic-curvature-computation`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Move one query while keeping its donor source fixed and differentiating the shared fitness statistics. A packed fourth-order jet supplies the Hessian metric and curvature in three dimensions. **Mode** selects conditional fitness or a polynomial fixture, **Query** moves the first coordinate, and **Epsilon** regularizes the metric. **Difference step** controls an independent numerical differentiation of the metric itself. The Ricci-component plot compares the jet result with that finite-difference reconstruction, and the scalar residual summarizes the discrepancy. The mixed-sign fixture uses the stated clipped policy; other modes use strict positivity. The exported jet and metric spectrum make the derivatives and branch choice inspectable.

Recorded mode evaluates the actual conditional fitness provider installed in the gas. The last completed step supplies the pre-clone population, selected donor context, objective, and standardizer parameters. Historical donor positions and rewards are taken from their immutable source pool together with the normalizer inputs consumed by the operator. **Query** replaces the first coordinate of the selected walker's chart point; its other coordinates stay fixed. The donor source remains frozen while the target query and its dependent fitness statistics are differentiated. **Epsilon** sets the analyzed metric regularizer. The provider's metric policy and threshold travel with the result. When the Hessian lies on a clipping branch boundary, the spectrum remains available and the curvature carries its branch diagnostic.

**Curvature calculation** selects host f64, CPU f64, CPU f32, or WebGPU f32. The selected batch backend evaluates the same exported jet with the same epsilon, policy, and threshold; its scalar curvature appears beside the host result. Backend availability is reported directly. Compare the tensor calculation with finite differences first, then compare arithmetic backends at that same field and query. **Mode** selects the reference geometry; recorded mode uses its archived provider context.
:::

**Control values:** `backend` = `host_f64`; choices: host_f64, cpu_f64, cpu_f32, webgpu_f32; `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `mode` = `conditional`; choices: conditional, coupled, separable, mixed; `epsilon` = 3; range 0.1–10; `query` = 0.2; range -0.7–0.7; `difference_step` = 0.002; range 0.0002–0.02.

(sec-partvi-experiment-40)=
### VI-40 — Connection identifiability

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-discrete-connection`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The transport measurements are generated from the stated reference connection. Identifying the gas connection requires sufficient recorded directional data.

Recover a symmetric connection coefficient triple from integrated transport measurements. In **identified** mode, several independent directions constrain the three coefficients. In **single_velocity** mode, repeating one direction leaves the design rank deficient. The calculation scales edge and time length together, solves the normal equations when identifiable, and plots coefficient error under refinement. The actual measured response comes from integrating transport; the reference coefficients are supplied separately. This example explains why more accurate data cannot resolve an unmeasured direction. Both modes display the computed eigenvalues of the scaled design Gram matrix. A zero eigenvalue exposes an unconstrained coefficient direction. In the deficient mode, coefficient estimates and their error curve are absent because the measurements do not identify the full triple; the Gram spectrum still shows exactly why the solve is unavailable.
:::

**Control values:** `mode` = `identified`; choices: identified, single_velocity.

(sec-partvi-experiment-41)=
### VI-41 — Raychaudhuri term balance

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-raychaudhuri`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The spacetime metrics and congruences are supplied. Applying Raychaudhuri to measured gas geometry requires the corresponding metric, congruence, and acceleration terms.

An anisotropic exponential Bianchi I slab has scale rates with a shared expansion and a controlled spread. **Dimension**, **Expansion**, and **Anisotropy** set those rates. Differentiate the log volume numerically to measure $\theta$, then display $-\theta^2/d$, minus the shear norm, vorticity, and minus the Ricci contraction. Their signed sum vanishes for this constant-expansion geodesic congruence. Increasing anisotropy grows shear even if the mean expansion stays fixed. The separate scale-factor and tensor calculations make the Raychaudhuri balance visible term by term, including the curvature contribution needed to balance the shear.

Choose **Congruence** `rotation` to inspect a stationary, rigidly rotating congruence in Minkowski spacetime. **Rotation speed / c** sets the local tangential speed and **Rotation probe radius** sets the query radius, hence the angular velocity. The four-velocity is differentiated numerically and projected to compute expansion, shear, and vorticity. Expansion and shear vanish, while the nonzero vorticity contraction cancels the acceleration divergence. The plot compares these contractions with the analytic rigid-rotation terms. Change the speed to increase both terms together, then set it to zero and recover their common zero. This supplies a direct test of the acceleration term in the full identity. **Expansion** and **Anisotropy** select the separate Bianchi geometry.
:::

**Control values:** `mode` = `bianchi`; choices: bianchi, rotation; `dimension` = 3; range 2–3; `expansion` = 0.4; range -1–1; `anisotropy` = 0.3; range 0–0.8; `rotation_speed` = 0.4; range 0–0.8; `radius` = 0.6; range 0.2–1.

(sec-partvi-experiment-42)=
### VI-42 — Material and reconstructed volumes

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-discrete-raychaudhuri`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The deterministic flow is supplied. An algorithmic material-volume measurement must retain clone reassignment, stochastic motion, and boundary flux.

Follow a small material interval under $\dot x=a x^2$ and measure the derivative of its log length. **Rate** sets $a$ and **Time** selects the pre-caustic observation time. Refine the initial width and compare that measurement with the pointwise divergence $2ax(t)$. The exported cell data also retain the expansion of a reconstructed Voronoi interval and its normalized flux difference from the material cell. The plot teaches the local limit through actual moving endpoints: finite cells sample a range of velocities, while the point prediction concerns one trajectory. Reconstructing a cell changes the flux bookkeeping and is explicitly recorded.
:::

**Control values:** `rate` = 0.4; range 0.05–0.8; `time` = 0.5; range 0–0.8.

(sec-partvi-experiment-43)=
### VI-43 — Focusing and caustics

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-focusing-theorem`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The Riccati coefficients are supplied. A focusing prediction for the gas requires measuring or deriving those complete terms.

Begin with negative expansion and integrate the Raychaudhuri Riccati equation with zero vorticity and nonnegative Ricci contraction. **Initial expansion**, **Ricci**, and **Dimension** control the focusing. The zero-Ricci comparison has a focusing time $d/|\theta_0|$; a positive Ricci term increases contraction. One plot compares the expansion curves and another integrates the material Jacobian. Stop interpreting the regular congruence at a vanishing Jacobian or singular expansion. The numerical integration and analytic comparison come from separate calculations, so the curves display the mechanism behind the finite focusing bound rather than just its endpoint.
:::

**Control values:** `dimension` = 3; range 2–3; `initial_expansion` = -1; range -3–-0.1; `ricci` = 0.2; range 0–2; `shear_squared` = 0; range 0–2.

(sec-partvi-experiment-44)=
### VI-44 — Curvature and topology

**Placement:** {doc}`3_fitness_manifold/03_curvature_gravity` at {ref}`sec-curvature-topology`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The triangulation is supplied. A topological measurement of the gas requires a specified complex built from its recorded points or interactions.

Triangulate a Euclidean disk by joining its boundary vertices to a center. **Vertices** changes the triangulation. The interior angle deficit vanishes while the boundary turning angles sum to $2\pi$, matching its Euler characteristic. A separate three-tetrahedra hinge fixture reports integrated scalar curvature $2\ell\delta$; **Scale** changes its edge length $\ell$. The workbench computes the disk angles and counts explicitly and compares them through Gauss–Bonnet. The hinge calculation shows why integrated three-dimensional curvature carries a length factor, while the disk's angle sum is dimensionless.

Choose **Surface topology** `sphere` for a closed triangulated surface. Starting from an octahedron, **Sphere subdivisions** subdivides its faces and projects the added vertices to the selected sphere. The native calculation measures every Euclidean triangle angle, sums the deficits around each vertex, and independently counts vertices, edges, and faces. The total deficit remains $4\pi=2\pi\chi$, with $\chi=2$, across refinements and scales. The vertex plot shows how this integrated curvature is distributed over the mesh. Every edge has two incident faces, and the boundary contribution is zero. **Vertices** selects the disk fixture, while **Scale** sets the geometric size in either surface mode.
:::

**Control values:** `surface` = `disk`; choices: disk, sphere; `vertices` = 8; range 3–32; `scale` = 1; range 0.1–4; `refinement` = 1; range 0–3.

(sec-partvi-experiments-fields)=
## Field equations and mechanical measurements

:::{div} feynman-prose
Energy, pressure, diffusion, and stress use different derivatives and normalizations. This group gives each one a controlled model before assembling a comparison. The recorded mechanical budget supplies an additional view of what the algorithm actually changes during one complete update.
:::

(sec-partvi-experiment-45)=
### VI-45 — Fluctuation energy and metric evolution

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-algorithmic-metric-evolution`.

:::{div} feynman-prose
**Measurement source:** Independent complete-checkpoint engine continuations, with a selectable finite reference.

Recorded mode freezes the complete checkpoint, including donor memory, and runs independent calibration and validation continuations. The default **Metric observable** is `material`: follow a selected walker slot and evaluate the programmed conditional fitness metric at its final position. The six packed components $(11,12,13,22,23,33)$ supply the conditional increment and its full covariance. Changing the donor window, timestep, viscosity, or boundary changes the executed law whose metric evolution is being measured.

Resolve each metric increment into field change at the initial probe, change from the literal clone's probe displacement, and subsequent motion to the final probe. Their means sum to the total, and their full cross-covariance matrix reconstructs the total covariance. At a one-step horizon this displays the successive clone and kinetic/boundary contributions directly. For longer horizons, the intermediate probe contribution also carries motion from earlier steps. Choose `fixed_probe` to measure the lagged conditional field at an unchanged chart coordinate instead.

**Engine replicas per independent group** sets the group size and **Engine continuation steps** sets the horizon. Compare the two independent mean increments with their estimated discrepancy covariance. Extinction and declared ineligible or spectral-unavailable metric outcomes use the stated zero extension, with their counts displayed. Unsupported provider or historical-source errors propagate. All replicas remain in the mean; the result consequently describes this precise extended field observable.

The conditional drift and covariance characterize the material metric under the full state. A closed equation using only the metric must predict those quantities while accounting for omitted state dependence. Reference mode separately varies a periodic density in an entropy-plus-pair energy, checks its quadratic remainder, compares a binomial law with its KL rate, and runs a finite-state metric-drift example. **Amplitude**, reference **Epsilon**, **Population**, and **Reference ensemble samples** select those supplied calculations.
:::

**Control values:** `readout` = `material`; choices: material, fixed_probe; `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `replicas` = 16; choices: 8, 16, 32; `horizon` = 1; choices: 1, 2, 4; `amplitude` = 0.4; range 0–0.8; `epsilon` = 0.15; range 0.03–0.4; `population` = 64; range 8–256; `reference_replicas` = 1024; range 128–8192.

(sec-partvi-experiment-46)=
### VI-46 — Pressure and deformation

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-elastic-pressure`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The pair energy is specified independently. A pressure description of the gas must be derived from the configured copying and kinetic response.

Dilate a transported density on a one-dimensional interval and differentiate its Gaussian pair energy with respect to the dilation. **Epsilon** sets the interaction range and **Mode** chooses a positive, signed, or attractive fixture. The analytic pressure integral and a symmetric energy finite difference should agree. The main pressure experiment is one-dimensional. **Dimension** belongs to the accompanying stiffness comparison, which contrasts fixed kernel amplitude, scaling as $\epsilon^{d+2}$, with unit kernel mass, scaling as $\epsilon^2$. These paired plots let the reader see how the sign of the energy and the kernel normalization enter two different constitutive calculations.
:::

**Control values:** `epsilon` = 0.3; range 0.05–0.6; `mode` = `positive`; choices: positive, signed, attractive; `dimension` = 3; range 1–3.

(sec-partvi-experiment-47)=
### VI-47 — Density-wave dispersion

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-linearized-dynamics`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The redistribution closure is supplied. Its dispersion must be compared with the actual donor, acceptance, memory, and noise response before it predicts the gas.

Evolve a sinusoidal density mode using a periodized Gaussian redistribution operator and a centered grid Laplacian. **Epsilon**, **Diffusion**, **Rate**, **Length**, **Wave number**, and **Resolution** set the physical range, diffusion, redistribution rate, box, Fourier mode, and spatial grid. The exact continuum rate is $Dk^2+\lambda(1-e^{-\epsilon^2k^2/2})$. Compare its exponential with the grid operator's time integration, then inspect the exact multiplier beside its fourth-order long-wave expansion. The truncated polynomial has a restricted wavelength regime that the exact multiplier makes visible. Spatial and time-discretization errors remain identifiable through the recorded grid rate and timestep.
:::

**Control values:** `epsilon` = 0.15; range 0.03–0.5; `diffusion` = 0.02; range 0–0.2; `rate` = 0.7; range 0–2; `length` = 1; range 0.5–5; `wave_number` = 1; range 1–6; `resolution` = 96; range 32–256.

(sec-partvi-experiment-48)=
### VI-48 — BAOAB diffusion and thermostat heating

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-algorithmic-balance-laws`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Recorded mode reads each executed O-stage velocity before boundary operations, together with its actual damping, integration scale, diffusion factor, innovation law, and addressed source shift. The browser installs the clipped fitness metric provider, so changing the swarm also changes the noise geometry. The prediction is calculated conditionally on the recorded input through $v_+=cv+sB(\xi+u)$. Its kinetic mean and variance retain the rectangular factor $B$, deterministic shift $u$, and the innovation's fourth moment.

Plot realized kinetic increments beside their conditional means and accumulate their centered residuals. The square root of accumulated predictable variation gives the fluctuation scale. Gaussian and standardized uniform innovations have fourth moments $3$ and $9/5$, respectively; their equal variance does not make energy fluctuations equal. **Engine innovation law** changes the actual draws and the corresponding moment prediction. Eligibility masks and raw-before-boundary conventions keep killing and boundary changes in their own stage budgets.

Reference mode freezes a scalar incoming velocity and factor. **Friction**, **Time step**, **Velocity**, **Noise**, **Mode**, and **Samples** select this independent sampler check. At positive friction it also compares free BAOAB long-time diffusion with the continuous OU value. The recorded state-dependent thermostat and the scalar free-motion calculation have their own predictions, so a diffusion reference is never substituted for the interacting update.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `friction` = 1; range 0–3; `time_step` = 0.2; range 0.01–1; `velocity` = 1; range -3–3; `noise` = 1; range 0.05–3; `mode` = `gaussian`; choices: gaussian, uniform; `samples` = 8192; range 128–32768.

(sec-partvi-experiment-49)=
### VI-49 — Gaussian mode thermodynamics

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-radiation-pressure`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The independent Gaussian mode state is supplied. Establishing this thermodynamics for the gas requires its actual modal law and relaxation measurements.

A fixed collection of real quadratic modes has variance $\Theta/k_j$. **Modes**, **Temperature**, **Volume**, and **Dimension** set the collection and its stiffness scaling; **Mobility** changes the recorded relaxation rates. Independent Gaussian draws test each variance. Separately, differentiate the mode partition function with respect to volume and compare the pressure with $\Theta M/(dV)$. Holding the set of modes and its reference measure fixed is part of that derivative. The variance plot and pressure calculation explain two consequences of the same quadratic energy, while the details show that changing mobility affects relaxation without changing the equilibrium variance.
:::

**Control values:** `modes` = 8; range 1–32; `temperature` = 1; range 0.1–3; `volume` = 2; range 0.3–5; `dimension` = 3; range 1–3; `mobility` = 0.7; range 0.1–3.

(sec-partvi-experiment-50)=
### VI-50 — Pressure crossover

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-pressure-regimes`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The two pressure scales are supplied. Their crossover becomes an algorithmic prediction only after those scales are measured in a common normalization.

The declared two-term pressure is $P=B-A\epsilon^{d+2}$. **Attraction** sets $A$, **Mode pressure** sets $B$, and **Dimension** sets the power. Its zero occurs at $(B/A)^{1/(d+2)}$. A numerical bracket-and-bisect calculation finds the same crossing, and the plot displays pressure against range divided by that crossing. This is a constitutive reference experiment: the coefficients are supplied through the controls. Placing it after the separate pressure calculations helps the reader understand exactly which competing terms are being combined and how each moves the crossover.
:::

**Control values:** `attraction` = 1; range 0.1–3; `mode_pressure` = 1; range 0.1–3; `dimension` = 3; range 1–3.

(sec-partvi-experiment-51)=
### VI-51 — Mechanical balances and gravitational stress

**Placement:** {doc}`3_fitness_manifold/04_field_equations` at {ref}`sec-stress-energy-tensor`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Measure the kinetic energy and momentum at every recorded clone, transform, kinetic, boundary, and eligibility stage. Their differences telescope to the directly measured complete change. For each raw force kick, calculate the work from the actual total acceleration, including viscosity: $\Delta E=v\cdot\Delta v+|\Delta v|^2/2$. Two separately recorded kicks give independent stage checks against the observed velocity changes. The thermostat prediction includes its addressed source shifts.

The weak local momentum readout multiplies momentum by a spatial test function. Its change splits into impulse, motion through that function, and eligibility terms. This exposes transport and killing contributions that a global second moment can conceal. The centered kinetic moment is also measured; volume normalization and the remaining mechanical terms must be specified when using it as a stress component. Literal-copying and restitution increments keep their own immutable-source bookkeeping.

Reference controls **Velocity** and **Restitution** inspect a two-walker collision. **Dimension**, **Energy density**, **Pressure**, and **Lambda** select the separate Einstein constitutive contraction. That tensor calculation checks the supplied equation's algebra. The recorded quantities test the algorithm's mechanics, and the held-out spatial constitutive comparison in the measured-outcomes section tests a proposed relation between those measurements. Its reported error remains visible when that relation predicts poorly.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `velocity` = 1; range -3–3; `restitution` = 0.5; range 0–1; `dimension` = 3; range 2–3; `energy_density` = 1; range -1–3; `pressure` = 0.2; range -2–2; `lambda` = 0.1; range -1–1.

(sec-partvi-experiments-holography)=
## Boundaries, entropy, and holographic calculations

:::{div} feynman-prose
A graph cut is a sum of crossing capacities, entropy is a functional of a stated law, and a surface variation changes a specified geometric input. These experiments begin with those finite quantities. Their continuum and thermodynamic comparisons keep the kernel normalization, probability support, and boundary constraints visible.
:::

(sec-partvi-experiment-52)=
### VI-52 — Genealogical separators and cuts

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ig-entropy`.

:::{div} feynman-prose
**Measurement source:** Recorded algorithm observable, with a selectable finite reference.

Read the actual selected directed interaction channels and their recorded weights. Count crossings of a fixed slot partition, retaining absent weight information as unavailable. These edges are the sparse, directed interactions the gas executed, so their cut statistics characterize the selected companion graph. Their counts can be compared directly with the recorded Fractal Set construction.

The ancestry panel collapses persistence within each epoch-slot-generation episode and follows accepted-clone parent edges. **Episode separator** selects roots or terminal episodes. Check that the selected episodes form an antichain and that each chosen terminal ancestry chain meets it exactly once. A still-living ancestor can make the terminal-episode set fail the antichain condition. Historical sources outside coverage give a partial-coverage result.

A separate auxiliary graph uses nonnegative Gaussian capacities on the first two coordinates of the terminal population. **Epsilon** sets its range, and opposite extreme points are pinned as terminals. Independently sum capacities across the computed cut to check max-flow/min-cut. **Population** selects the finite reference point set. This reconstructed complete symmetric graph has its own edge law; its geometric cut is displayed separately from selected algorithmic interactions and from episode ancestry.
:::

**Control values:** `source` = `recorded`; choices: reference, recorded; `walkers` = 32; choices: 16, 32, 64; `engine_memory` = 2; choices: 0, 2, 8; `engine_dt` = 0.04; choices: 0.01, 0.04, 0.1; `engine_friction` = 1; choices: 0, 0.5, 1, 2; `engine_viscosity` = 1; choices: 0, 0.5, 1, 2; `engine_boundary` = `unbounded`; choices: unbounded, absorbing_box, periodic_box; `engine_innovation` = `gaussian`; choices: gaussian, standardized_uniform; `population` = 24; range 8–64; `epsilon` = 0.25; range 0.05–0.7; `separator` = `roots`; choices: roots, terminal_episodes.

(sec-partvi-experiment-53)=
### VI-53 — Jump, modular, and path entropy

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-algorithmic-thermodynamic-response`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The quantum state and finite reverse path law are supplied. A gas entropy measurement requires its own normalized forward and reverse laws on the stated support.

A qubit perturbation changes entropy and modular energy. **Perturbation** sets the noncommuting state change, and direct eigenvalue entropy verifies $D(\omega\|\sigma)=\Delta\langle-\log\sigma\rangle-\Delta S$. A three-state cycle supplies a separate path-entropy experiment controlled by **Forward**, **Reverse**, and **Path length**. Enumerate its paths and compare path KL with the analytic cycle expression; check the integral fluctuation expectation on their common support. Setting reverse probability to zero exposes support mismatch explicitly. The plotted symmetric jump cost and its quadratic expansion connect the finite entropy calculation to the local fluctuation term.
:::

**Control values:** `perturbation` = 0.1; range -0.2–0.2; `forward` = 0.4; range 0.05–0.55; `reverse` = 0.2; range 0–0.4; `path_length` = 5; range 1–8.

(sec-partvi-experiment-54)=
### VI-54 — Gaussian cuts and perimeter

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-gamma-convergence`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The density and complete Gaussian graph are supplied. The selected interaction graph in VI-52 has to satisfy its own measured geometric scaling conditions.

Cut a unit torus into two half strips and integrate a periodized Gaussian across the partition. **Density contrast** sets $\rho(x)=1+c\cos(2\pi x)$, **Dimension** sets the transverse Gaussian factor, and **Resolution** sets quadrature accuracy. Divide the cut by $\epsilon^{d+1}$ and refine bandwidth. Its target is $(2\pi)^{(d-1)/2}[(1+c)^2+(1-c)^2]$, accounting for both boundary components and their density squared. The measured pair integral and the boundary formula are evaluated separately. The plot shows both the continuum trend and the loss of quadrature resolution when a kernel becomes too narrow for the chosen grid.
:::

**Control values:** `dimension` = 2; range 2–3; `density_contrast` = 0.3; range 0–0.7; `resolution` = 128; range 32–256.

(sec-partvi-experiment-55)=
### VI-55 — Finite-population area scaling

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-area-law`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. Independent populations define this sampling experiment. The gas requires a dependence-aware scaling comparison of its actual selected interactions.

Generate independent uniform torus populations and sum their crossing Gaussian pair capacities. **Population**, **Dimension**, and **Range factor** set $\epsilon=\ell N^{-1/d}$; **Replicas** sets the number of independent populations. The expectation is $N(N-1)$ times the one-sided pair integral. Compare the replica cut values with that quadrature prediction and inspect the normalized area estimate. The standard error uses one entire population as a sample, preserving dependence among pairs that share a walker. The experiment turns the finite-population prefactor into a measured quantity and gives the area-law normalization a direct statistical check.
:::

**Control values:** `population` = 64; range 16–256; `replicas` = 24; range 8–64; `dimension` = 2; range 2–3; `range_factor` = 0.7; range 0.3–1.2.

(sec-partvi-experiment-56)=
### VI-56 — Three first variations

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-first-law`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. Each variation has its stated state space. A first-law comparison for the gas requires a common intervention protocol and the resulting measured changes.

For a circle in a fixed radial density $\rho(r)=1+cr^2$, the weighted surface energy is $E(r)=2\pi\tau r\rho(r)^2$. **Radius**, **Density contrast**, and **Tension** set the fixture. Differentiate the energy numerically and compare its normal pressure with $-\tau[\rho^2/r+4cr\rho]$. The second term is the ambient density-gradient contribution. The details also contain a finite graph density variation at a fixed partition, checked by a separate difference quotient.

The state-variation panel perturbs a qubit with reference eigenvalues $(1/4,3/4)$ in a trace-preserving, noncommuting direction. Direct spectral entropy approaches its modular first variation, whose derivative is $\log3$. Its finite-difference residual is reported separately. These three calculations change shape, spatial density, and a density matrix respectively. Holding the other inputs fixed in each calculation makes the meaning of each derivative explicit.
:::

**Control values:** `radius` = 0.6; range 0.2–1.2; `density_contrast` = 0.3; range -0.3–0.7; `tension` = 1; range 0.1–3.

(sec-partvi-experiment-57)=
### VI-57 — Gibbs QSD candidates

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-qsd-thermal`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The killed two-state kernel is supplied. A Gibbs approximation of the gas QSD needs its own invariant-law residual and contraction control.

Normalize a Gibbs weight on two energy levels and test whether it is a left eigenmeasure of a killed kernel. **Temperature** changes the candidate and **Killing contrast** changes the state-dependent survival probabilities. Repeated normalized killed evolution supplies the QSD reference. The candidate residual and a verified contraction constant give a bound on its total-variation error, which can be compared with the error measured directly. The plot follows the candidate under normalized iteration. This teaches a concrete way to test a proposed thermal form: its name and normalization are inputs, while its eigenmeasure residual decides whether it matches this kernel.
:::

**Control values:** `temperature` = 1; range 0.2–3; `killing_contrast` = 0.1; range 0–0.3.

(sec-partvi-experiment-58)=
### VI-58 — Response and Fisher information

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-algorithmic-thermodynamic-response`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The controlled finite kernel is supplied. The gas stationary-response calculation needs its actual transition derivative and mixing law; VI-19 measures finite-horizon response directly.

Change one transition probability of a positive two-state Markov chain. **Transition a** and **Transition b** set the kernel, **Difference step** controls a stationary finite difference, and **Path length** enters the path Fisher calculation. The Poisson-equation response is compared with $b/(a+b)^2$ and with the independently recomputed invariant laws. A delayed-response curve shows how the contributions sum to the stationary derivative. The details also evaluate a static exponential tilt, whose derivative is a covariance, and keep stationary and transition Fisher information separate. Each response has an explicit perturbation law, making their different formulas straightforward to compare.
:::

**Control values:** `transition_a` = 0.3; range 0.1–0.8; `transition_b` = 0.4; range 0.1–0.8; `difference_step` = 0.001; range 0.0001–0.02; `path_length` = 8; range 1–64.

(sec-partvi-experiment-59)=
### VI-59 — Euclidean horizons and AdS

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ads-cft`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The cone and AdS metric are supplied. Neither a gas temperature nor its curvature radius is determined by this calculation.

A Euclidean near-horizon metric becomes an ordinary plane when its angular period satisfies $\kappa\beta=2\pi$. **Surface gravity** sets $\kappa$ and **Period ratio** moves the chosen period away from or toward that value. The circumference plot shows the resulting cone-angle defect. A separate constant-curvature AdS reference uses **Lambda** and **Dimension** to determine its radius and directly checks the Einstein contraction. The two calculations share the lesson that normalization matters: the period uses a specified time coordinate and the curvature uses a specified spacetime dimension. Their formulas and conventions are retained in the native result.
:::

**Control values:** `lambda` = -0.5; range -2–-0.1; `dimension` = 3; range 2–3; `surface_gravity` = 1; range 0.2–3; `period_ratio` = 1; range 0.5–1.5.

(sec-partvi-experiment-60)=
### VI-60 — Minimum cuts and limiting surfaces

**Placement:** {doc}`3_fitness_manifold/05_holography` at {ref}`sec-holography-ads-cft`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The local weighted network is supplied. Its minimizing surface describes that network; VI-52 measures the actual selected gas edges separately.

Put a density valley through a rectangular capacity grid and require a cut to separate the left and right boundaries. **Density contrast** deepens the valley and **Resolution** refines the grid. The minimum-cut interface should follow the low-cost region. Compare its network capacity with independently enumerated straight-interface costs and the continuum value $(1-c)^2$. This workbench uses a local finite-volume perimeter approximation; the Gaussian nonlocal approximation is measured in VI-54. The interface plot makes the minimization visible, while the distinct grid and continuum numbers show how the discrete geometry approaches its specified surface problem.
:::

**Control values:** `resolution` = 16; range 8–32; `density_contrast` = 0.4; range 0–0.8.

(sec-partvi-experiments-cosmology)=
## Conditional stationarity and cosmological references

:::{div} feynman-prose
The cosmology workbenches connect conditional laws, geometric scales, expansion, predictive closure, and units. Their inputs are deliberately explicit: a killed chain, a supplied congruence, a finite-bit law, or a chosen Hubble value. That makes each prediction directly reproducible and exposes which additional input enters the next calculation.
:::

(sec-partvi-experiment-61)=
### VI-61 — Conditional stationarity and survival

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-three-vacuum-energies`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The two-state QSD is supplied. Actual finite-horizon field increments are measured in VI-22 and VI-45, including probability lost through killing.

A two-state mixing chain is killed independently with a constant survival probability. **Initial probability** changes its starting law, **Survival** changes the mass lost each step, and **Energy shift** changes the energy reference. The centered observable relaxes under the conditioned law, while a second curve shows survival decaying. Adding a constant energy leaves the centered expectation unchanged. The invariant distribution is computed from the conservative kernel and used in a total-variation bound. This is a controlled example in which conditional stationarity and falling survival occur together, with all normalizations visible.
:::

**Control values:** `initial_probability` = 0.8; range 0–1; `energy_shift` = 0; range -3–3; `survival` = 0.85; range 0.5–1.

(sec-partvi-experiment-62)=
### VI-62 — Interaction range and curvature scale

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-uv-regime`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The circle locality calculation and AdS algebra have separate geometric inputs. Algorithmic curvature must be extracted from the programmed metric, as in VI-39.

Compare a Gaussian integral along a circle with its tangent-line approximation. **Lambda** and **Dimension** set a reference radius through the stated curvature-scale formula; **Range ratio** sets the largest interaction range relative to that radius. The measured relative correction is compared with the leading $\epsilon^2/(8R^2)$ term. The curved integral uses direct quadrature while the tangent expression is analytic. The selected circle supplies the local geometric measurement, and the radius assignment is a declared reference scale. The plot explains why a small interaction range relative to geometric curvature makes a local flat approximation accurate.
:::

**Control values:** `lambda` = -0.5; range -2–-0.1; `dimension` = 3; range 2–3; `range_ratio` = 0.1; range 0.02–0.4.

(sec-partvi-experiment-63)=
### VI-63 — Expansion histories

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-raychaudhuri-expansion`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The forcing bound and spacetime examples are supplied. The gas expansion law must retain its measured transport, stochastic, clone, and curvature terms.

Positive expansion can occur in de Sitter geometry or in Milne coordinates on flat spacetime. **Dimension** and **Expansion** set the de Sitter rate, and the plot compares its constant expansion with $d/t$ for Milne. A second calculation integrates a Riccati equation whose complete forcing is bounded below; **Forcing** sets that bound and the analytic comparison is a hyperbolic tangent. The two plots keep expansion and curvature separately observable. Their direct scale-factor and differential-equation calculations explain why the full Raychaudhuri terms, rather than the sign of expansion alone, determine the geometric interpretation.
:::

**Control values:** `dimension` = 3; range 2–3; `expansion` = 0.4; range 0.1–1; `forcing` = 0.3; range 0.05–1.

(sec-partvi-experiment-64)=
### VI-64 — Predictive information

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-closure-theory`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The reveal channel is supplied. Measuring donor-memory information requires the gas history and conditional future law, whose predictive effects VI-12 and VI-32 investigate.

Let the microscopic record sometimes reveal a future fair bit while the macro record retains only the present bit. **Reveal probability** controls that extra information and **Samples** controls the sampled joint table. Conditional entropy predicts discarded predictive information $p\log2$. The measured point is placed on the exact line, with the full counts available in the details. The macro process still has one causal state in this fixture. This example gives predictive information and the number of macro forecasting states separate numerical meanings, so that an information loss can be measured even when the macro-state count stays fixed.
:::

**Control values:** `reveal_probability` = 1; range 0–1; `samples` = 4096; range 128–16384.

(sec-partvi-experiment-65)=
### VI-65 — Dynamical closure

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-closure-theory`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. The generator and projection are supplied. A gas closure must pass the corresponding conditional-drift test with its complete-state dynamics.

Project a four-state continuous-time chain onto two blocks. With zero **Rate perturbation**, the generator is compatible with that projection. Increasing the perturbation makes two microscopic states in one block evolve differently. **Time** sets the integration horizon. The workbench evaluates the exact generator on the macro indicator, compares it with the proposed macro generator, and integrates the resulting Dynkin residual from two initial microstates. The residual curves stay within the displayed $\pm t\epsilon$ envelope obtained from the supremum generator defect. This is a direct operational test of approximate closure, with the full rate matrix available for inspection.
:::

**Control values:** `rate_perturbation` = 0.15; range 0–0.5; `time` = 2; range 0.1–5.

(sec-partvi-experiment-66)=
### VI-66 — Vacuum units and observables

**Placement:** {doc}`3_fitness_manifold/06_cosmology` at {ref}`sec-cosmological-observations`.

:::{div} feynman-prose
**Measurement source:** Supplied finite model or geometric input. This is a unit conversion for supplied cosmic parameters. Algorithmic field measurements need a separately specified calibration before acquiring these physical units.

Supply a Hubble value in kilometres per second per megaparsec and a vacuum density fraction. **Hubble** and **Omega** determine $\Lambda=3H^2\Omega_\Lambda/c^2$ in the declared flat Friedmann reference. The native calculation converts to inverse seconds, then obtains energy density, mass density, and pressure with their SI units. Reconstructing the input fraction from the resulting curvature checks the conversion independently. The pressure is minus the vacuum energy density; at zero density their ratio is unavailable. The curve displays how the supplied density fraction changes the inferred curvature, keeping the illustrative inputs separate from any observational parameter estimate.
:::

**Control values:** `hubble` = 70; range 30–100; `omega` = 0.7; range 0–1.

(sec-partvi-experiments-native)=
## Native calculations, archives, and reproducible comparisons

:::{div} feynman-prose
The browser and native runner call the same Rust calculation. A native run is therefore a way to increase a sample budget, inspect the complete result, or repeat a parameter sweep without changing the quantity being measured. Start by reproducing one browser request. Then change one parameter and retain both requests with their outputs. For an archive calculation, keep the archive as well: the seed alone does not describe an imported trajectory.

A numerical check is strongest when its two sides are calculated in different ways. For a matrix identity, use a direct contraction and an independent indexed expansion. For a continuum coefficient, compare quadrature with analytic derivatives. For a conditional expectation on the engine, use independent continuation groups with their uncertainty. For a finite law, enumerate all outcomes when its size permits. The workbench details identify the comparison that is actually available, so the result can be interpreted at the scale at which it is computed.
:::

### Request and result interfaces

The native interface accepts an `ExperimentRequest` with an integer `experiment` in `1..=66` and a scalar-valued `parameters` object. Rust `physics::partvi::analyze` evaluates a reference request. `analyze_archive` accepts a validated `RunArchive<f64>` and uses the archive for the readouts described above. `algorithmic_gas_benchmarks::qft_experiments::run` executes the independent engine continuation calculations for experiments 19, 22, and 45.

The WASM interface exposes the same experiment request and result structures. A result contains `experiment`, `title`, `model`, `metrics`, `plots`, `notes`, and `details`. Plot series retain their coordinate pairs and labels. Metric values are optional when the calculation reports unavailable or nonfinite quantities. A command-line bundle uses the schema `fragile-partvi-results-v1`, contains a `results` array, and records its command and run configuration where applicable.

### Run a reference experiment

From the repository's `algorithmic-gas` directory, pipe a request to the native binary. This example computes the complex covariance and whitened spectral fixture in VI-36:

```bash
printf '%s\n' '{"experiment":36,"parameters":{"amplitude":0.5,"rate":0.4,"steps":32,"seed":7}}' \
  | cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
      analyze - --output /tmp/partvi-spectrum.json
```

The file contains the known decay rates, covariance rank, whitening residual, plotted modes, and the complex input matrix. Import this result bundle into the lecture viewer to inspect the same native arrays.

### Execute and inspect recorded colors

The `run` command builds a gas, records its updates, and evaluates the request on the resulting archive. Its default profile uses 32 walkers in three dimensions with a viscous-force configuration. Use `--config` with a serialized `RunConfig` when a precise engine configuration is part of the comparison.

```bash
printf '%s\n' '{"experiment":8,"parameters":{"angle":0.7,"threshold":1e-12,"seed":7}}' \
  | cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
      run - --steps 32 --save-archive /tmp/partvi-color-archive.json \
      --output /tmp/partvi-colors.json
```

Reanalyze that exact archive with a different color threshold without rerunning its dynamics:

```bash
printf '%s\n' '{"experiment":8,"parameters":{"angle":0.7,"threshold":0.1,"seed":7}}' \
  | cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
      analyze - --archive /tmp/partvi-color-archive.json \
      --output /tmp/partvi-color-threshold.json
```

The force and velocity arrays stay tied to their recorded B1 input stage. The threshold and phase scale belong to the analysis request. Changing them changes the observable map applied to those arrays.

### Inspect recorded path factors and temporal predictions

The supplied request presets in `algorithmic-gas/examples/partvi/` select reproducible calculations. The action preset runs VI-17 on a recorded trajectory:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
  run examples/partvi/recorded-action.json --steps 32 \
  --save-archive /tmp/partvi-action-archive.json --output /tmp/partvi-action.json
```

VI-16 and VI-21 can analyze this same archive, retaining the executed components while adding their empirical generating function or disintegration. Primitive likelihoods are conditional on each recorded source state. Uniform independent choices support replacement and distinct draws; uniform matching sums the multiplicity of shuffles inducing the observed matching. Gaussian-weighted independent choices reconstruct the configured distance and kernel weights on the actual eligible source pool, including available historical sources. Single choices and replacement draws have an explicit ordered likelihood. Weighted multi-donor sampling without replacement needs the unrecorded selection order, and Gaussian-greedy matching needs its shuffle path. Those components report unavailable with their reason.

Each clone gate contributes its acceptance probability or its complement. Revival uses its actual current-frame eligible donor law; historical donors remain available to the configured distance calculation. Nonsingular Gaussian raw samples use their recorded covariance factor, with quadratic, constant, and log-determinant contributions displayed separately. Singular Gaussian maps retain their exact latent Gaussian density, and standardized uniform innovations retain their product density on the sampled latent coordinates. Each component states its carrier. Unknown custom providers without a supplied probability law and unrecorded external-environment transitions give typed unavailable components; the built-in conditional-fitness provider uses the configured categorical and gate laws. Available components remain plotted; a missing factor leaves the total likelihood unavailable. Recorded surviving transitions retain their executed law, with no inferred conditioning normalizer.

Use a longer trajectory for the chronological prediction and spectral presets:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
  run examples/partvi/recorded-memory.json --steps 96 \
  --save-archive /tmp/partvi-memory-archive.json --output /tmp/partvi-memory.json

cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
  run examples/partvi/recorded-spectrum.json --steps 96 \
  --output /tmp/partvi-recorded-spectrum.json
```

The predictor uses its training partition and a separated validation block. The spectrum fits decay and phase on its training block and reports the corresponding held-out error. For a direct comparison between readouts on one trajectory, save its archive and use `analyze --archive` with both requests. Changing the descriptor or fit window then changes the analysis of the same data.

### Run full source interventions

For experiment 19, the `run` command invokes the complete checkpoint-continuation calculation. The request below selects the bounded tagged-velocity readout, shifts the first coordinate of slot zero at the addressed future Gaussian O step, and runs four subsequent steps in each continuation:

```bash
printf '%s\n' '{"experiment":19,"parameters":{"seed":7,"replicas":32,"horizon":4,"warmup":2,"theta":0.25,"walker":0,"coordinate":0,"observable":"tagged_velocity"}}' \
  | cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
      run - --steps 2 --output /tmp/partvi-source-response.json
```

The result distinguishes independent likelihood-weighted and direct-shifted groups. Within the direct group, plus, minus, and baseline runs share their future seed for finite-difference comparisons. The finite-source reweighting identity is compared at the specified shift. Hermite-score estimates concern derivatives at zero; finite differences retain their finite-shift bias. Their separate standard errors and the importance effective sample size are recorded.

For experiment 22, use the same command with `"experiment":22` and select, for example, `"observable":"momentum"`. The runner estimates the full conditional increment in one group and checks it in another, and independently compares raw O-stage momentum and energy with their analytic conditional moments. Supported native readouts are `tagged_velocity`, `momentum`, `mean_position`, `kinetic_energy`, and `alive_fraction`. The tagged-velocity readout is the hyperbolic tangent of the selected slot component. Killed continuations have a zero terminal readout, with their survival counts retained.

### Estimate conditional metric evolution

The metric preset selects VI-45 and requests two groups of sixteen independent continuations. Its native default profile installs the conditional fitness metric provider with clipped Hessian policy, epsilon $0.1$, temperature one, and clipping threshold $10^{-8}$:

```bash
cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
  run examples/partvi/metric-replicas.json --output /tmp/partvi-metric-replicas.json
```

A supplied `--config` determines the executed provider, including its metric regularizer, temperature, standardizers, and donor law. Use that option to reproduce the browser metric profile described above. The request's `warmup` sets the baseline checkpoint, `horizon` sets the future interval, and `replicas` sets the size of each independent group. Native requests also accept `walker` and `probe_offset`; the latter shifts the first probe coordinate by a value in $[-1,1]$. The request selects `readout: "material"` by default or `"fixed_probe"`. The result includes baseline and terminal metric contexts, material or fixed probe positions, per-replica future seeds, survival and availability flags, component means, the full component cross-covariance, total increment covariance, and comparison uncertainties.

VI-19, VI-22, and VI-45 use ensemble results as their replay artifact. Their `run` commands reject `--save-archive` because each result contains several distinct continuations. The exported result retains the replica provenance and measured arrays. Their native `replicas` parameter selects the relevant calculation; the browser names VI-45's reference-only sample slider `reference_replicas` so it remains distinct from the engine replica selector.

### Sweep a parameter

A sweep accepts either explicit requests or a shared request with one parameter and a list of values. The following example increases the quadrature resolution for the Gaussian perimeter comparison:

```bash
printf '%s\n' '{"experiment":54,"parameters":{"dimension":2,"density_contrast":0.3,"seed":7},"parameter":"resolution","values":[32,64,128,256]}' \
  | cargo run --release -p algorithmic-gas-benchmarks --bin algorithmic-gas-qft -- \
      sweep - --output /tmp/partvi-perimeter-sweep.json
```

Sweep requests use reference or supplied-archive analysis. Full independent engine continuation groups for VI-19, VI-22, and VI-45 use `run`; passing those identifiers to reference `analyze` or `sweep` selects their explicitly finite reference models. Importing a multi-result bundle advances through its results in order.

The supplied `all-reference.json` enumerates all 66 reference requests. `density-refinement.json` and `gap-refinement.json` provide sample-budget and timestep sweeps; `recorded-color.json`, `recorded-memory.json`, and `recorded-spectrum.json` provide archive readouts; `gaussian-response.json`, `stochastic-balance.json`, and `metric-replicas.json` provide continuation requests. Their filenames describe which native command to select using the conventions above.

### Budgets and conventions

The command-line input limit is 128 MiB, and a sweep contains at most 4096 requests. The `run` step count is capped at 100,000, with recorded data subject to its memory budget. Native VI-19 and VI-22 continuation requests accept 2–128 replicas per group, 1–16 horizon steps, at most 16 warmup steps, and at most 256 walkers. VI-45 accepts 2–32 replicas per group, 1–4 horizon steps, 1–4 warmup steps, 2–64 walkers, and three spatial dimensions. The browser uses the smaller selectable budgets listed above. They require CPU f64 BAOAB with the named position and velocity fields. Source response requires Gaussian innovations and an initially empty innovation-shift schedule.

Individual experiments bound matrix size, population, quadrature resolution, or enumeration length. The visible controls stay within their documented ranges. Increasing a parameter beyond the computational model's supported range does not define an additional numerical experiment; use the effective values and model information in the result when comparing runs. A six-mode exterior fixture already has 64 occupation states, and exact finite-path enumeration grows with both the number of states and the path length.

Complex arrays use explicit real and imaginary components in serialized details. Temporal readouts use the stated physical or algorithmic timestep. Kernel calculations state whether their Gaussian has fixed amplitude, unit mass, periodic images, or a finite grid normalization. Entropy uses natural logarithms. SI quantities appear only in the explicit unit-conversion workbench. These conventions are inputs to a numerical comparison and should travel with exported data.

### What to compare when reviewing a result

1. **Identity calculations:** inspect the residual and the exact inputs. Reverse an orientation, change a common frame, approach a rank deficiency, or repeat an exterior vector. The identity should respond according to its stated algebraic condition.
2. **Sampling calculations:** retain the seed, independent sampling unit, sample count, and uncertainty. Compare independent populations as populations and independent continuation groups as groups; pair or coordinate counts alone do not supply their standard error.
3. **Refinement calculations:** hold the physical model fixed while refining the grid, timestep, difference step, or loop size. Keep discretization error and finite-bandwidth bias separate from random sampling variation.
4. **Record calculations:** check the stage, mask, source slot or generation convention, and donor coverage. A reconstructed value should be traceable to the exact fields and identities used by the measurement.
5. **Theoretical comparisons:** read the linked statement with the explicit finite law, geometry, or engine profile shown by the workbench. The formula index provides a route through the chapter; the measured residual and exported calculation identify the particular comparison performed.
