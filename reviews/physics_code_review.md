# Review of the simulation launched by `make physics`

Review date: 2026-09-05. Repository HEAD: `cad50b910a55d0d019a325aeb028ceb4e578dbfb`.

No simulation, analysis, or existing test code was changed. The deliverables are this report and a standalone diagnostic script, [physics_review_repro.py](physics_review_repro.py).

## Assessment

There are implementation defects that can materially change both the simulated dynamics and the reported spectrum. The most consequential are a doubled kinetic force increment, incorrect cloning history, replacement of pair correlators during covariance estimation, severely underestimated resampling errors, cancellation of the standard tensor observable, and failures in the claimed Dirac parity construction.

This report documents **27 findings**, with affected settings and evidence. It does not claim that every possible defect in the repository has been found, or that these defects explain a particular saved result: no specific problematic run was supplied. Some findings affect the default dashboard; others require optional settings. Modeling questions are separated from implementation findings at the end.

P1 means an incorrect trajectory, observable, inference, or data association that should be addressed before trusting affected results. P2 means a conditional defect or reproducibility/precision problem. “Reproduced” refers to execution against the current code; “source trace” means the conclusion follows from inspected control flow or algebra but was not exercised through a browser.

## What actually runs

The entry path is `Makefile:76` → `uv run fragile physics` → `src/fragile/__main__.py:60` → `physics/app/dashboard.py:create_app`.

The dashboard constructs `SimulationTab`, which configures `physics/fractal_gas/EuclideanGas`. Its iteration computes geometry rewards, fitness, a proposed clone, post-cloning tessellation, and kinetic integration. `RunHistory` then feeds several distinct analysis implementations:

| Dashboard functionality | Main implementation |
|---|---|
| Simulation | `physics/fractal_gas/`, `physics/geometry/` |
| Companion correlators | `physics/app/companion_correlators.py`, primarily `physics/new_channels/` |
| Electroweak correlators | `physics/electroweak/` |
| Bayesian mass extraction | `physics/mass_extraction/` |
| Strong-force AIC | `physics/aic/`, also imports fitting functions from `physics/new_channels/correlator_channels.py` |
| Shared/new operator pipeline | `physics/operators/`, also supplies helpers and result containers to the dashboard |

This distinction matters: many `tests/qft/` tests exercise the older `fragile.fractalai.qft` implementation. Passing those tests does not establish correctness of `make physics`. Likewise, finding a defect in `operators/pipeline.py` does not imply the dashboard's single-scale companion button calls that function.

The dashboard overrides defaults to 500 walkers, 750 iterations, float32, `clone_every=20`, zero initial positions and velocities, zero clone position jitter, restitution 1, one kinetic substep, `delta_t=0.002`, `nu=3`, `beta_curl=1`, and automatic thermostat temperature 0.33. These are particularly relevant to findings 1, 2, 6, 9, and 19.

## Findings affecting simulation and recorded history

### 1. P1 — Each kinetic iteration applies twice the intended B-step force and rotation duration

**Locations:** [kinetic_operator.py:468](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:468), [kinetic_operator.py:573](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:573), [kinetic_operator.py:596](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:596).

**Evidence: reproduced.** `apply()` calls `_apply_boris_kick()` before and after the A–O–A sequence. Each call already includes two force increments of `dt/2` and one Boris rotation constructed for duration `dt`. Thus each nominal BAOAB half kick actually advances the force subproblem for a full step.

For a constant force F, initial v=0, and suppressed random noise, ordinary BAOAB gives `v_final = (dt/2)(c1+1)F`. The implementation gives `dt(c1+1)F`. The diagnostic returned an actual/expected ratio of **1.999999865**. The constant-force mock isolates the integrator arithmetic; it is not a proposed physical replacement for viscosity.

**Effect:** viscosity and curl are stronger relative to drift and thermostat evolution than their configured coefficients indicate. This changes relaxation rates and potentially the invariant distribution. A global rescaling of fitted masses cannot generally repair a change in the balance among terms.

**Future acceptance check:** deterministic constant-force increment; graph-Laplacian velocity decay; a constant skew field rotation with known angular advance; convergence as `dt` decreases. Check total force/rotation duration over the whole iteration, not merely conservation of speed during each Boris rotation.

### 2. P1 — Skipped cloning steps still record clones, displacements, and copied fitness

**Locations:** [euclidean_gas.py:432](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:432), [history.py:737](/home/guillem/fragile/src/fragile/physics/fractal_gas/history.py:737), [electroweak_observables.py:87](/home/guillem/fragile/src/fragile/physics/electroweak/electroweak_observables.py:87).

**Evidence: reproduced.** The cloning operator executes on every iteration. `clone_every` determines whether its positions/velocities are used, but does not reset `will_clone`, `num_cloned`, `clone_delta_x`, `clone_delta_v`, or jitter. `fitness = other_cloned.get(...)` executes even when the proposed clone is discarded. The recorder saves these values without consulting `cloning_applied`; that flag is not a corresponding history field.

With `clone_every=20`, a three-step run performed no position cloning, yet recorded **[6, 6, 7] cloning events**. The maximum actual cloning displacement was **0**, while the maximum recorded displacement was **3.1020**. Stored fitness differed from fitness recomputed on the unmodified pre-clone state by **1.5992**.

**Effect:** the default dashboard has this inconsistency on 19 of every 20 iterations. Cloning diagnostics, lineage reconstruction, walker-type classification, and electroweak observables can describe transitions that never occurred. Even on an actual cloning iteration, the stored copied fitness must not be interpreted as the original pre-clone fitness alongside the original rewards and scores.

**Future acceptance check:** skipped iterations have false executed-clone masks and zero executed deltas; recorded fitness has an explicit stage and matches that stage; proposal diagnostics, if retained, have distinct names.

### 3. P1 — Disabling viscosity does not disable the force, and missing weights crash a nonempty graph

**Locations:** [kinetic_operator.py:308](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:308), [euclidean_gas.py:289](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:289).

**Evidence: reproduced.** `_compute_viscous_force()` never checks `use_viscous_coupling`. It unconditionally indexes `edge_weights[valid]`. The gas deliberately sets weights to `None` when viscosity is disabled. Consequently a nonempty graph produces `TypeError: 'NoneType' object is not subscriptable`. Supplying weights directly instead still produces a nonzero force with viscosity disabled.

The same crash occurs if the selected weighting mode was not included in `neighbor_weight_modes`. The promised on-the-fly weighting fallback is absent. A directly constructed kinetic operator defaults to inverse Riemannian distance, while the gas defaults to precomputing only Riemannian kernel-volume weights, so natural standalone composition can hit this immediately. The dashboard explicitly sets matching modes and avoids that particular default mismatch.

**Effect:** ablation runs cannot reliably disable this interaction; changes to weight settings can terminate simulations.

**Future acceptance check:** disabled viscosity on a nonempty graph gives zero force; each documented fallback either works or is rejected before running; compatible defaults compose successfully.

### 4. P2 — The integrator selector is unused

**Locations:** [kinetic_operator.py:88](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:88), [kinetic_operator.py:493](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:493).

**Evidence: reproduced.** `integrator` accepts `baoab` and `boris-baoab`, but runtime integration never branches on it. Rotation depends on `beta_curl` and field availability. With nonzero external curl and the same seed, both selector values produced **exactly identical** velocities.

**Effect:** users of this configuration parameter believe they are comparing different numerical methods when they are not. This is conditional; the current kinetic widget list does not expose every declared parameter.

**Future acceptance check:** ordinary BAOAB omits Boris rotation even with a configured curl field; Boris-BAOAB includes it with the correct duration.

### 5. P2 — The viscous kernel length setting never reaches the weights

**Locations:** [kinetic_operator.py:96](/home/guillem/fragile/src/fragile/physics/fractal_gas/kinetic_operator.py:96), [delaunai.py:280](/home/guillem/fragile/src/fragile/physics/geometry/delaunai.py:280), [weights.py:compute_edge_weights](/home/guillem/fragile/src/fragile/physics/geometry/weights.py).

**Evidence: reproduced.** `compute_delaunay_data()` invokes `compute_edge_weights()` without `length_scale`; the latter uses its default 1.0. The kinetic force uses supplied weights without recomputing them. Changing `viscous_length_scale` from **0.001 to 1000** left weights exactly unchanged.

**Effect:** length-scale sweeps and tuning through the visible control do not alter the simulated kernel. The dashboard's current value happens to be 1, so the defect is hidden until it is changed.

**Future acceptance check:** changing this parameter changes the relative weights for unequal edge lengths and changes the resulting force.

### 6. P1 — Exact position cloning silently removes walkers from the Delaunay interaction graph

**Locations:** [simulation.py:117](/home/guillem/fragile/src/fragile/physics/app/simulation.py:117), [simulation.py:138](/home/guillem/fragile/src/fragile/physics/app/simulation.py:138), [delaunai.py:68](/home/guillem/fragile/src/fragile/physics/geometry/delaunai.py:68), [hessian_estimation.py:581](/home/guillem/fragile/src/fragile/physics/geometry/hessian_estimation.py:581).

**Evidence: reproduced.** With `sigma_x=0`, cloning creates duplicate coordinates. Edge construction takes vertices only from SciPy Delaunay simplices, without reinserting omitted duplicate particles or representing their multiplicity. A 20-walker point cloud containing one exact duplicate returned a zero-degree walker. The graph did not cover all particles.

The omitted walker receives no viscous coupling. Its neighbor covariance is only the numerical regularizer, so its inverse metric is approximately `1e5 I`, unrelated to the geometry of its coincident peer. This feeds volume and reward-related fields. At the dashboard's completely coincident initialization, tessellation is empty for the first step; independent noise later separates walkers, but real cloning repeatedly reintroduces duplicates.

**Effect:** graph interactions and metric statistics depend on how duplicate vertices are selected by triangulation, rather than treating equal-position particles consistently. This is not just a performance issue or a startup warning.

**Future acceptance check:** graph/metric treatment of coincident particles must be explicitly defined and stable under index permutations. Do not silently change the fixed algorithm by adding jitter without deciding whether that modification is intended.

### 7. P1 — Color states combine pre-clone velocities with post-clone viscous forces

**Locations:** [color_states.py:43](/home/guillem/fragile/src/fragile/physics/qft_utils/color_states.py:43), [euclidean_gas.py:467](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:467), [history.py:726](/home/guillem/fragile/src/fragile/physics/fractal_gas/history.py:726).

**Evidence: source trace.** Color construction reads `v_before_clone[frame]` but multiplies its phase by `force_viscous[frame-1]`. Those offsets correctly address the history arrays, but they do not address the same dynamical state: the force is measured by `kinetic_op.apply()` on the post-clone state. Positions can have jumped, and non-unit restitution also changes velocities. With multiple kinetic substeps, `kinetic_info` is overwritten each time, so the recorded force belongs to the start of the **last** substep.

**Effect:** the complex color vector can combine momentum from one state with force from another. This affects essentially every downstream color-based channel, especially around real clone events and when increasing substeps. With one substep and no executed clone, these stages coincide; the finding is not an across-the-board off-by-one accusation.

**Future acceptance check:** explicitly identify the phase/force evaluation stage and record both operands at that stage; compare against independently evaluated forces on a step that actually changes positions and velocities.

### 8. P1 — Chunked recording truncates absolute step metadata to the last buffer

**Locations:** [euclidean_gas.py:728](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:728), [history.py:889](/home/guillem/fragile/src/fragile/physics/fractal_gas/history.py:889), [history.py:945](/home/guillem/fragile/src/fragile/physics/fractal_gas/history.py:945).

**Evidence: reproduced.** The recorder resets `recorded_idx` after flushing a chunk. `run()` nevertheless truncates `recorded_steps` using that local buffer index. `build()` concatenates all tensor chunks but accepts the already truncated labels.

A 12-step run with `chunk_size=10` produced **13 frames** and `recorded_steps=[0,1,2,3]`. Step lookup and clone-event code that indexes this list can fail or lose access to later states.

**Effect:** long runs using the memory-saving option have incomplete time metadata despite intact recorded arrays.

**Future acceptance check:** chunked and unchunked runs agree in every history field, including labels, across exact and partial flush boundaries and `record_every>1`.

### 9. P2 — Saved parameters omit the active thermostat and other trajectory-defining settings

**Location:** [euclidean_gas.py:591](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:591).

**Evidence: reproduced.** `_build_params()` saves manual `beta` but not `auto_thermostat`, `temperature`, or effective beta. It also omits `viscous_length_scale`, `integrator`, and `tessellation_timing`. A run at automatic temperature 0.33 saved `beta=1.0` without recording that the active inverse temperature was about 3.0303.

**Effect:** the metadata can identify a different thermostat from the one used, impeding reproducibility and comparisons. Raw trajectories remain available; the failure is in reconstructing and interpreting their generating configuration. Omitting currently ignored controls is less consequential than omitting the active temperature.

**Future acceptance check:** round-trip all active numerical parameters and distinguish manual from effective thermostat values; replay a small seeded run from saved configuration.

### 10. P2 — Recorded-time interpretation omits kinetic substeps and assumes a uniform final interval

**Locations:** [euclidean_gas.py:467](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:467), [euclidean_gas.py:739](/home/guillem/fragile/src/fragile/physics/fractal_gas/euclidean_gas.py:739), [new_channels/correlator_channels.py:983](/home/guillem/fragile/src/fragile/physics/new_channels/correlator_channels.py:983), [new_channels/multiscale_strong_force.py:2115](/home/guillem/fragile/src/fragile/physics/new_channels/multiscale_strong_force.py:2115).

**Evidence: reproduced for substeps; source trace for final intervals.** Each iteration executes `n_kinetic_steps` integrations of duration `delta_t`. Analysis commonly derives elapsed spacing as `history.delta_t * record_every`, without the substep count. The probe executed 0.04 units per iteration but stored a base `delta_t=0.01`, which these consumers treat as the whole iteration duration.

Separately, the run always records its final step, even if it is off the recording grid. A 12-step run recorded every 5 steps has labels `[0,5,10,12]`; selected pre-clone frames then have a short final interval. FFT processing treats adjacent samples as uniformly spaced.

**Effect:** masses expressed per kinetic time and time-axis diagnostics acquire scale errors; the irregular endpoint can distort lag estimates. If the intended unit is deliberately one Monte Carlo iteration, report it as such. The dashboard AIC-from-correlators path deliberately sets `dt=1`; that convention is not itself an error.

**Future acceptance check:** separate iteration units from integrated time, and reject/drop/resample an off-grid final frame before uniform-grid correlation.

## Findings affecting analysis state, observables, and fitting

### 11. P1 — Deferred history changes retain results and priors associated with the previous run

**Locations:** [simulation.py:243](/home/guillem/fragile/src/fragile/physics/app/simulation.py:243), [companion_correlators.py:1151](/home/guillem/fragile/src/fragile/physics/app/companion_correlators.py:1151), [electroweak_correlators.py:705](/home/guillem/fragile/src/fragile/physics/app/electroweak_correlators.py:705), [mass_extraction_tab.py:1163](/home/guillem/fragile/src/fragile/physics/app/mass_extraction_tab.py:1163), [strong_force_aic_tab.py:780](/home/guillem/fragile/src/fragile/physics/app/strong_force_aic_tab.py:780).

**Evidence: source trace.** Both normal history loading and simulation completion use deferred dashboard updates. Analysis callbacks return on `defer=True` before clearing result objects. `state['history']` changes while companion/electroweak correlators, masses, and AIC outputs still refer to the old run. Existing plots also remain.

Some fit buttons are correctly disabled when history changes; therefore this is **not** a claim that every old result can immediately be refit with one click. The retained data and plots are nevertheless not invalidated, and AIC prior seeding reads the retained AIC result through `on_aic_ready()`. Consumers have no run identity attached to results to reject this mismatch.

**Effect:** stale displays and potential cross-run prior/result association during iterative experimentation. A failed recomputation also leaves a previous result in shared state.

There is a concrete route back to fitting stale data: the dashboard's completion wrapper checks only whether `companion_correlator_output` (or its electroweak counterpart) is non-None. If computing run B fails, retained output from A still satisfies that test and re-enables the fit button. The wrapper does not check that computation succeeded for the current history.

**Future acceptance check:** computing on run A and then loading B invalidates A's derived data immediately, independently of whether expensive plotting is deferred. Derived objects should carry a history/configuration identity.

### 12. P1 — Covariance selection replaces the supplied pair-propagation observable

**Locations:** [data_preparation.py:198](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:198), [meson_phase_channels.py:257](/home/guillem/fragile/src/fragile/physics/new_channels/meson_phase_channels.py:257), [mass_extraction_adapter.py:90](/home/guillem/fragile/src/fragile/physics/new_channels/mass_extraction_adapter.py:90).

**Evidence: reproduced.** The companion meson correlator propagates a fixed source pair to the sink and averages pair products. Its exported operator series is only the per-frame spatial average. These are different statistics:

`mean_pairs[O_pair(t) O_same_pair(t+lag)]` is not `mean_pairs[O_pair(t)] mean_pairs[O_pair(t+lag)]`.

For bootstrap/jackknife, `_single_correlator_to_gvar()` discards the supplied values except for their length, FFT-correlates the averaged series, and returns the resample mean as the fitted data. A mutually paired pseudoscalar is a decisive example: opposite directed pair values cancel in the spatial average but not in their pair products. The probe had **C(0)=0.147267** and an identically zero averaged series; the correlator passed to the fit became **0**.

The conversion also silently uses `use_connected=True` regardless of the supplied observable. A raw constant scalar correlator equal to 9 became zero.

**Effect:** a setting described as covariance estimation changes what is being measured. Pion/meson and other source-pair channels can disappear or have entirely different decay. The default `uncorrelated` mode preserves the supplied central correlator and avoids this replacement, although its errors have their own problem in finding 24.

**Future acceptance check:** preserve central correlators and resample the same pair-level estimator, including its connected/raw convention, masks, source indices, and normalization. The averaged series alone is insufficient to reconstruct it.

### 13. P1 — Vector/tensor resampling averages components before correlation

**Location:** [data_preparation.py:184](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:184).

**Evidence: reproduced.** Repeated `series.mean(dim=-1)` reduces a vector series to one scalar. Its correlation includes cross-component products and a different normalization, rather than the intended sum of component autocorrelations. This defect also affects the shared operator pipeline where finding 12's source-pair distinction does not apply.

For the valid vector series `[a(t), -a(t)]`, the intended contracted correlator had **C(0)=2.266663**, but conversion returned **0**. Momentum tensor adaptation separately replaces cos/sin components with a nonlinear amplitude in `new_channels/mass_extraction_adapter.py:extract_tensor_momentum`; resampling that amplitude also cannot reproduce the supplied contracted correlator.

**Effect:** component cancellation, spurious cross terms, and changed decay scales under covariance selection.

**Future acceptance check:** resample vector/tensor data with their original component contraction; verify invariance under orthogonal component rotations and against a sum of independently computed component correlations.

### 14. P1 — Bootstrap and jackknife uncertainties use the covariance-of-the-mean formula for independent data

**Location:** [data_preparation.py:207](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:207).

**Evidence: reproduced.** Both methods send resampled estimates to `gvar.dataset.avg_data()` with its default semantics. That estimates uncertainty of a mean of independent observations; it does not directly give the sampling variance of an estimator from bootstrap or delete-block replicas. The [gvar documentation](https://gvar.readthedocs.io/en/latest/gvar_dataset.html) distinguishes error on the mean from the spread of a distribution.

For B bootstrap estimates, their spread estimates uncertainty of the original estimator; dividing it again by approximately `sqrt(B)` is wrong. With **100 bootstrap replicas**, the reported/reference standard-error ratio was **0.099499**. Increasing replica count therefore produces artificially tighter reported errors without new simulation data.

For K delete-block jackknife replicas `theta_k`, the covariance is `(K-1)/K * sum_k[(theta_k-theta_bar)(theta_k-theta_bar)^T]`. It is not the independent-sample covariance of the replica mean. With **10 blocks**, the reported/reference ratio was **0.105409**.

**Effect:** misleadingly narrow fit errors, inflated significance, and excessive weighting of data against priors. Numerical SVD regularization does not repair this overall normalization.

**Future acceptance check:** reproduce known bootstrap/jackknife covariance formulas and show stable uncertainty when the number of bootstrap replicas increases.

### 15. P1 — Cross-channel and cross-scale covariance is discarded

**Locations:** [data_preparation.py:121](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:121), [data_preparation.py:208](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:208), [models.py:43](/home/guillem/fragile/src/fragile/physics/mass_extraction/models.py:43).

**Evidence: reproduced.** Every key is converted to gvars in a separate call containing only that channel. The resulting objects are mutually independent even when the operators and resampling indices are identical. The fit then shares energies across variants as if these were independent information sources.

Two identical input channels in the diagnostic had returned correlation **0**, although their true correlation is 1.

**Effect:** duplicated or strongly related operators/scales can produce unjustified confidence, and mass-ratio errors lose common-mode cancellation. The multi-run conversion uses a joint dataset and does not have this exact single-run implementation defect.

**Future acceptance check:** joint resampling and one joint covariance construction; adding an identical channel must not create independent information.

### 16. P1 — Delete-block jackknife closes time gaps before computing lag products

**Location:** [data_preparation.py:69](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:69).

**Evidence: reproduced.** `series[mask]` physically removes a time block, concatenates the two remaining parts, and FFT-correlates the result. This treats observations on opposite sides of the missing block as adjacent. It changes the lag being estimated.

For `[0,1,2,3,4,5,6,7]`, deleting `[2,3]` introduces the false lag-one product `1*4`. The implementation's lag-one mean is **19.2**; averaging only retained pairs at their original lag gives **23**.

The bootstrap branch joins sampled blocks too; block bootstrap can be an approximation, but its boundary effects need a block length suited to the lags/correlation scale. A default block length 10 with fits extending to lag 80 is not evidence that those effects are negligible.

**Effect:** even correcting finding 14's covariance prefactor leaves an estimator with altered temporal relationships.

**Future acceptance check:** delete contribution blocks while retaining original source/sink time differences, or use a justified block-level estimator; establish block-length stability.

### 17. P1 — Multiscale Bayesian channels disappear during key matching

**Locations:** [pipeline.py:165](/home/guillem/fragile/src/fragile/physics/mass_extraction/pipeline.py:165), [data_preparation.py:125](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:125), [companion_correlators.py:1078](/home/guillem/fragile/src/fragile/physics/app/companion_correlators.py:1078).

**Evidence: reproduced.** Channel groups are detected before `[S,L]` correlators are expanded. A group requests `scalar`, while conversion creates `scalar_scale_0`, `scalar_scale_1`, etc. Filtering against available keys drops the original name and can leave no active groups. The dashboard constructs multiscale results with exactly this unexpanded format and also detects groups before extraction.

A two-scale exponential input returned `channels={}` even though data contained both scale keys. Mixed inputs can silently fit only their single-scale subset. `include_multiscale=True` does not repair the key mismatch.

**Effect:** missing multiscale fits and misleadingly partial analyses.

**Future acceptance check:** expand keys before group detection, including explicit group configurations; assert every intended scale contributes a model.

### 18. P1 — The shared multiscale correlator flattens time and component axes in the wrong order

**Location:** [operators/correlators.py:64](/home/guillem/fragile/src/fragile/physics/operators/correlators.py:64).

**Evidence: reproduced.** A `[S,T,C]` tensor is directly reshaped into `[S*C,T]`. Contiguous storage interleaves C inside each time sample, so each resulting “time series” mixes components and times. The intended layout requires moving the component axis before time before reshaping.

The multiscale calculation differed from stacking correctly computed single-scale vector results by **0.792228** in the probe.

**Effect:** incorrect vector/tensor temporal correlations in `operators/pipeline.py` multiscale mode. The dashboard's current companion multiscale button calls `new_channels.multiscale_strong_force`; this specific flattening defect should not be attributed to that button without tracing a shared call.

**Future acceptance check:** batched multiscale output exactly matches a loop over scales and components for nonconstant, distinguishable inputs.

### 19. P1 — The standard tensor channel cancels identically for mutual companion pairs

**Locations:** [tensor_operators.py:33](/home/guillem/fragile/src/fragile/physics/operators/tensor_operators.py:33), [tensor_operators.py:120](/home/guillem/fragile/src/fragile/physics/operators/tensor_operators.py:120), [companion_correlators.py:997](/home/guillem/fragile/src/fragile/physics/app/companion_correlators.py:997).

**Evidence: reproduced and algebraic.** The sigma matrices are Hermitian: off-diagonal entries are `+i` and `-i`. Therefore `Im(c_j† sigma c_i) = -Im(c_i† sigma c_j)`. The simulation generates reciprocal companion maps. The standard tensor path averages both directions before computing its FFT correlator, so each pair cancels.

Random valid complex colors with mutual pairs produced an operator series whose maximum absolute value was **0**. Unlike the pair-propagation pseudoscalar, this dashboard tensor implementation computes its correlator from the already cancelled series.

**Effect:** a tensor channel consisting of zeros or rounding residue even when local tensor bilinears are nonzero. A subsequent fit can be prior-driven or measure numerical noise.

**Future acceptance check:** choose and document a tensor estimator with the required exchange behavior; evaluate a nonzero known configuration using the actual mutual companion selection. Merely checking tensor shapes cannot catch the cancellation.

### 20. P1 — The color-to-Dirac map does not obey its stated parity transformation

**Location:** [dirac_spinors.py:267](/home/guillem/fragile/src/fragile/physics/new_channels/dirac_spinors.py:267).

**Evidence: reproduced and algebraic.** The code states that physical inversion sends color to `-conj(color)` and the derived spinor to `gamma0 @ psi`. It maps `Re(color)` through a Hopf section into the lower two components. A sign change of a 3-vector does not map its Hopf spinor to its negative: those represent the same direction, whereas opposite vectors represent antipodal directions.

The actual `color_to_dirac_spinor(-color.conj())` differed from `gamma0 @ color_to_dirac_spinor(color)` by up to **1.39970** in the probe. Upper components remain unchanged, so an arbitrary common phase does not generally repair this mismatch.

**Effect:** Clifford algebra checks on the gamma matrices do not establish that the constructed observables carry the claimed parity under inversion of simulation variables. Channel labels and comparisons premised on that construction are unsupported by the implementation.

**Why current tests miss it:** parity tests apply `gamma0` directly to arbitrary spinors, which tests bilinear matrix algebra rather than the color-to-spinor map.

**Future acceptance check:** invert the original color/simulation variables and propagate them through the entire observable construction.

### 21. P1 — The dashboard's Dirac pseudoscalar is the imaginary part of a scalar bilinear

**Location:** [dirac_spinors.py:447](/home/guillem/fragile/src/fragile/physics/new_channels/dirac_spinors.py:447).

**Evidence: reproduced.** `compute_dirac_operators_from_spinors()` loads `gamma5` but implements the pseudoscalar as `Im(psi_i† gamma0 psi_j)`, without `gamma5`. Under the standard Dirac transformation `psi -> gamma0 psi`, this expression is parity-even. The diagnostic used a single directed pair to avoid cancellation: its even-parity residual was **0**, while its required odd-parity residual was **1.99184**.

With the dashboard's symmetric pairing and uniform aggregation, its imaginary scalar pair values also cancel by Hermitian exchange. The separately implemented `operators/dirac_operators.py` does use gamma5; the dashboard's lazy Dirac computation calls the affected `new_channels` function.

**Effect:** the channel labeled Dirac pseudoscalar is not the documented pseudoscalar. Fixing only the color map in finding 20 would leave this error.

**Future acceptance check:** test the exact operator function used by the dashboard with nontrivial directed pairs, the documented Gamma insertion, and parity inversion. Avoid tests where the supposed pseudoscalar is identically zero.

### 22. P2 — AIC mass extraction accepts `dt` but returns a per-sample slope

**Locations:** [new_channels/correlator_channels.py:541](/home/guillem/fragile/src/fragile/physics/new_channels/correlator_channels.py:541), [aic/correlator_channels.py:585](/home/guillem/fragile/src/fragile/physics/aic/correlator_channels.py:585).

**Evidence: reproduced.** `extract_mass_aic()` never uses `dt`. Its convolution fits time indices 0,1,2,… and returns `-slope`. For `C[n]=exp(-2*0.1*n)` with `dt=0.1`, the correct mass in the supplied time unit is 2; the returned value was **0.199999988**. The neighboring linear and effective-mass functions do divide by `dt`.

**Effect:** disagreement between effective masses and fitted masses, wrong scale comparisons, and incorrectly interpreted mass bounds when `dt != 1`. The dashboard's explicit per-sample `dt=1` path avoids this numerical factor; history-based/multiscale callers with other spacing do not.

**Future acceptance check:** a synthetic exponential returns the same physical mass for multiple sampling intervals, with errors, per-window masses, and bound checks in consistent units.

### 23. P1 — The function documented as block-bootstrap errors actually resamples individual time points

**Locations:** [new_channels/correlator_channels.py:453](/home/guillem/fragile/src/fragile/physics/new_channels/correlator_channels.py:453), [aic/correlator_channels.py:497](/home/guillem/fragile/src/fragile/physics/aic/correlator_channels.py:497).

**Evidence: source trace.** Despite its block-bootstrap docstring, it draws an independent random index for every position in every replica using `torch.randint(0,T,(n_bootstrap,T))`. There is no block length or contiguous block draw. Serial structure is destroyed before estimating uncertainty in serial correlation.

**Effect:** errors are estimated from effectively shuffled trajectories, not from fluctuations of the observed time-dependent process. This differs from finding 16, which concerns the Bayesian block-resampling implementation. It applies when this bootstrap-error option is enabled; not every dashboard fit calls it.

**Future acceptance check:** an autocorrelated process retains its within-block temporal structure and uncertainty estimates are checked against independent-run variation.

### 24. P1 — Default fit uncertainties are fabricated relative errors; AIC mass error omits statistical fit uncertainty

**Locations:** [data_preparation.py:158](/home/guillem/fragile/src/fragile/physics/mass_extraction/data_preparation.py:158), [new_channels/correlator_channels.py:567](/home/guillem/fragile/src/fragile/physics/new_channels/correlator_channels.py:567), [new_channels/correlator_channels.py:390](/home/guillem/fragile/src/fragile/physics/new_channels/correlator_channels.py:390), [aic/mass_extraction_adapter.py:109](/home/guillem/fragile/src/fragile/physics/aic/mass_extraction_adapter.py:109).

**Evidence: source trace.** The default Bayesian covariance method sets `sigma(t)=0.1*abs(C(t))+1e-15`, independent of sample count, temporal correlation, or measured variability. Missing operator data silently falls back to the same rule even when a resampling method was requested. Near a zero crossing, this incorrectly implies exceptionally precise information.

The AIC path similarly uses constant log error 0.1. Even where `compute_channel_correlator()` calculates bootstrap errors, `extract_mass_aic()` receives no such errors. Its returned `mass_error` is only the weighted spread of fitted slopes across windows. No within-window statistical uncertainty is included. If all windows agree, that spread can be zero even when the supplied observations would have nonzero measurement errors.

**Effect:** masses can be fitted, but quoted uncertainty, chi-squared quality, tensions in sigma units, and window confidence are not calibrated statistical quantities. This is a scientific-output defect rather than an exception. Some assumptions are documented internally, but the resulting fields and displays treat them as fit uncertainties.

**Future acceptance check:** propagate measured covariance through fitting and resampling; distinguish window-selection spread from statistical error. A fallback with assumed errors should be explicit in the returned result and presentation, not silently masquerade as measured covariance.

### 25. P2 — Float64 correlators silently lose precision before connected subtraction

**Location:** [fft.py:37](/home/guillem/fragile/src/fragile/physics/qft_utils/fft.py:37).

**Evidence: reproduced.** `_fft_correlator_batched()` begins with `series.float()`. This downcasts float64 input before subtracting its mean. The float64 sequence `1e8 + [0,1,0,-1]` has connected C(0)=0.5, but the implementation returned **0**, because the deviations were rounded away first.

Other color/spinor routines also force intermediate float32, sometimes converting back to complex128 afterward. A larger output dtype does not restore lost input precision.

**Effect:** choosing float64 for a simulation does not provide an end-to-end float64 analysis. Weak fluctuations on large means and small residual signals can vanish. Normalized order-one inputs are less exposed; the large-offset probe demonstrates the failure mechanism, not a claim that every color channel has that offset.

**Future acceptance check:** preserve precision through centering and FFT; compare connected covariance against a direct float64 reference on weak-fluctuation signals.

### 26. P2 — Companion multiscale completion does not notify the mass-fitting tabs

**Locations:** [companion_correlators.py:1055](/home/guillem/fragile/src/fragile/physics/app/companion_correlators.py:1055), [dashboard.py:162](/home/guillem/fragile/src/fragile/physics/app/dashboard.py:162).

**Evidence: source trace.** The multiscale button stores a new `PipelineResult` but has no completion notification to `companion_mass_section.on_correlators_ready()` or `strong_force_aic_section.on_correlators_ready()`. The dashboard registers these notifications only on the single-scale run button. The section's returned interface does not expose a multiscale-completion callback.

**Effect:** running multiscale first after loading a history can leave downstream fit buttons disabled. If a single-scale result was fitted previously, multiscale replaces the shared result without refreshing fit selectors/settings. This UI wiring defect is independent of finding 17's key-expansion problem: fixing either alone leaves the other.

**Future acceptance check:** single-scale and multiscale completion both notify dependent tabs exactly once and refresh controls for the actual result shape and keys.

### 27. P2 — Single-scale companion and electroweak buttons compute their results twice

**Locations:** [companion_correlators.py:1051](/home/guillem/fragile/src/fragile/physics/app/companion_correlators.py:1051), [electroweak_correlators.py:640](/home/guillem/fragile/src/fragile/physics/app/electroweak_correlators.py:640), [dashboard.py:165](/home/guillem/fragile/src/fragile/physics/app/dashboard.py:165).

**Evidence: source trace.** Each section registers `run_button.on_click(on_run)`. The dashboard then registers another click callback that invokes that same `on_run` before enabling dependent tabs. One click therefore invokes the expensive calculation twice.

**Effect:** unnecessary computation and responsiveness problems. Deterministic channels should agree between calls, so this is not evidence of a mass bias on its own. Any stochastic sampling performed by a selected branch can produce a second result that overwrites the first, consuming additional random numbers and complicating reproducibility. Failures during either invocation also interact with the stale-result behavior in finding 11.

**Future acceptance check:** instrument compute invocation counts and verify one click performs exactly one computation followed by one completion notification.

## Modeling and interpretation questions requiring a separate decision

These are observations worth investigating, but I have not counted them as additional confirmed implementation bugs.

1. **The nominal 3D simulation uses 2D reward/neighbor geometry.** `_compute_tessellation()` sets `spatial_dims=d-1` whenever d≥3 and slices the first coordinates. The dashboard fixes d=3, while color and velocity observables use three components. Arbitrarily changing only the third position coordinate left graph edges and Ricci rewards exactly unchanged in the probe. This creates a preferred axis. It might be an intended space/time convention, but that convention needs to be reconciled with three spatial color components and three-dimensional kinetic/curl calculations. If d means three spatial coordinates, this is an additional major implementation error.

2. **The recorded diffusion tensor does not drive the kinetic noise.** Geometry computes `g^(-1/2)` and stores it in history, but the actual O step uses isotropic noise only. Several docstrings still describe adaptive force and anisotropic noise. An isotropic model can be intentional; a saved tensor alone is not evidence that anisotropic diffusion was simulated.

3. **Reward geometry is intentionally lagged in `after_cloning` mode.** The default caches post-clone geometry and reuses it after a kinetic evolution. This is explicitly documented, so I have not called the cache itself a bug. It is an approximation whose sensitivity should be measured, especially with many kinetic substeps or infrequent graph updates. It does not explain away the mixed stages in color construction.

4. **The Ricci quantity is explicitly a graph/conformal proxy.** It is not automatically the full scalar curvature of the anisotropic metric. Reward interpretation and continuum claims require additional validation; this review does not establish them.

5. **AIC weights compare windows containing different observations.** The code combines `chi2+4` over different starts and widths without a treatment of omitted data or likelihood normalization for changing observation sets. Interpret these as a window-scoring heuristic unless a statistical derivation establishes model-comparison semantics. The fixed-error and missing-error-propagation implementation issues are already finding 24.

6. **Algorithmic-time decay is not, by itself, a physical particle mass.** The reviewed paths predominantly measure relaxation along generated histories. Establishing a physical spectral interpretation, spin assignment, and continuum limit requires more than a good exponential fit or calibration against one mass. I have not used failure to prove that interpretation as a code bug.

## Verification and limitations

The full existing `tests/physics` suite was run:

```bash
UV_CACHE_DIR=/tmp/fragile-review-uv uv run --offline pytest -q tests/physics --disable-warnings --maxfail=12
```

Result: **1,761 passed, 2 failed, 14 warnings**, in 16.90 seconds.

The failures were:

- `tests/physics/aic/test_radial_channels.py::TestParityApplyProjection::test_projection_parity[pseudoscalar]`: the newer pseudoscalar projection differs from the legacy projection by up to about 5.95. This is a migration-parity test, not proof that the legacy formula is physically correct; a deliberately corrected definition could invalidate it.
- `tests/physics/mass_extraction/test_pdg_comparison.py::test_build_ratio_comparison_smoke`: the table now includes `Matches` and `Error (%)`, which the exact-column assertion does not expect. This looks like an outdated output-schema expectation, not a numerical mass bug.

The standalone probes were then run successfully:

```bash
MPLCONFIGDIR=/tmp/fragile-review-mpl UV_CACHE_DIR=/tmp/fragile-review-uv \
  uv run --offline python reviews/physics_review_repro.py
```

They exercise constant-force integration, configuration switches, real gas runs, duplicate-point tessellation, history metadata, pair correlators, covariance construction, multiscale handling, parity transformations, tensor cancellation, synthetic exponential fits, and precision. They print evidence rather than assert corrected behavior, so they are a diagnostic artifact rather than a proposed regression-test patch. Seeded values reported above correspond to this script; minor floating-point differences can occur across environments.

No full 750-step production run, browser interaction, CUDA run, or validation against a supplied saved history was performed. The review prioritized numerical generation and interpretation over exhaustive coverage of every plotting function, every gravity diagnostic, and all experimental branches. Short real runs and synthetic exact checks provide evidence for the listed mechanisms; they do not quantify their contribution to a particular experimental discrepancy.

## Suggested order for subsequent work — no fixes applied

First correct and verify the trajectory/history issues: the B-step duration, executed-versus-proposed cloning record, graph handling of duplicate particles, color evaluation stage, and chunked metadata. Any change to the actual dynamics requires regenerating affected simulation histories; correcting only a plotting or covariance function cannot repair an already wrong trajectory.

Next settle the observable definitions and test the full transformation from simulation variables: standard tensor cancellation, the Dirac parity map, and the pseudoscalar Gamma insertion. Preserve source-pair information so uncertainty estimation can operate on the same statistic that is plotted.

Then repair covariance normalization, cross-channel covariance, temporal resampling, multiscale key/layout handling, and timing. Finally reevaluate fit stability, statistical uncertainties, and calibrated mass ratios on fixed saved inputs and independent runs. Existing histories may still support reanalysis for defects limited to postprocessing; missing or inconsistent history information must be assessed before assuming that reanalysis is sufficient.
