# Physics review: implemented corrections

This records the implementation follow-up to [the original review](physics_code_review.md).
The original report and reproducer describe the pre-fix code; the reproducer is
not the acceptance test for the corrected APIs. Regression tests live in
`tests/physics/test_review_regressions.py` and the existing physics suites.

## Coverage of the 27 findings

| Review finding | Correction |
| --- | --- |
| 1 | Each B stage advances force and Boris rotation by half a kinetic step. |
| 2 | Skipped cloning records zero events and displacements; recorded pre-clone fitness is no longer overwritten or mutated by tensor cloning. |
| 3–5 | Viscosity respects its enable flag, missing graph weights are constructed, the integrator selection is honored, and the configured kernel length reaches tessellation weights. |
| 6 | Triangulation operates on unique sites and lifts edges to every coincident particle, including within-site edges. Rank-deficient swarms use their affine span without random jitter. |
| 7 | New histories pair the first kinetic-substep force with after-clone velocity when constructing color states. |
| 8 | Chunk flushing no longer truncates absolute recorded-step labels. |
| 9 | Thermostat settings, effective beta, integrator and stage/time conventions are serialized. |
| 10 | Iteration duration includes kinetic substeps. Uniform-lag analysis excludes an irregular final recording interval. Legacy substep metadata is interpreted by the recorded-time helper. |
| 11 | Deferred history changes clear analysis outputs, fitted results, selectors and priors instead of keeping stale results. |
| 12–16 | Origin-level lag sums and counts accompany pair correlators. Fitting retains the supplied central estimate, contracts vector components after products, applies the correct replica covariance factors, estimates joint channel/scale covariance, and never closes deleted time gaps. |
| 17–18 | Scale keys are expanded before grouping; multiscale component/time axes are ordered correctly. |
| 19 | Tensor correlations multiply fixed-pair bilinears before spatial averaging, including scale gates. The displayed RMS operator is a diagnostic, not the correlator's input. |
| 20 | The Dirac feature embedding is explicitly odd on the lower block, giving the documented end-to-end parity transformation. |
| 21 | Dirac pseudoscalar uses the gamma-five bilinear, with an imaginary projection that is parity odd and symmetric under pair exchange. |
| 22 | AIC masses, window masses, limits and errors consistently use the supplied time unit. |
| 23 | Bootstrap errors resample blocks of original lag contributions rather than independently shuffled time points. |
| 24 | Measured block-jackknife covariance is the default. AIC reports measured fit uncertainty plus window spread when supported; otherwise uncertainty is unavailable, not a fabricated 10%. |
| 25 | Shared and AIC FFT paths retain float64 through connected subtraction and correlation. |
| 26–27 | Both single-scale and multiscale completion notify downstream tabs. Each compute button has one registered computation callback. |

## Numerical compatibility and limitations

- Rerun simulations before comparing corrected spectra with previous calibrations:
  trajectories, graph connectivity and observables have changed. Old histories
  cannot recover forces from a substep/stage that was never recorded. They retain
  legacy color-stage interpretation; new histories explicitly identify the corrected convention.
- The odd Dirac embedding establishes the stated parity identity, not rotational
  equivariance or physical spin/charge-conjugation quantum numbers. It is a changed
  observable definition; old Dirac spectra are not numerically interchangeable.
- Pair correlators carry transient `correlator_statistics` tensor metadata.
  `stack_correlators` preserves it across scales. Arbitrary tensor arithmetic can
  discard it. Fitting validates that statistics reproduce the supplied correlator;
  a matching scalar/vector series can reconstruct statistics when appropriate.
  Missing or incompatible statistics now cause an explicit error rather than
  silently fitting a different observable.
- `assumed_relative` is an explicit opt-in covariance method for synthetic
  curves or exploratory fits. Its uncertainties are assumptions, not measurements,
  and it emits a warning. Inter-run covariance requires at least two aligned runs.
- Resampling uses contiguous blocks of time-origin lag contributions. Connected
  products retain the full-sample centering convention. Incomplete terminal blocks
  are omitted from replicas, not the reported central estimate. Unsupported
  trailing lags are excluded from fitting. Block-size sensitivity and adequate
  independent sampling still need to be checked for any scientific result.
- The separate modeling questions in the original review remain unchanged:
  projected tessellation dimensions, isotropic kinetic noise, lagged reward
  geometry, curvature interpretation, cross-window AIC comparability, and
  physical calibration require their own modeling decisions.

## Validation

Follow-up to the reported `scalar_twistor` covariance exception: all six twistor
triplet channels now retain raw and connected origin-level statistics through
the dashboard adapter. Fit preparation restricts covariance to selected expanded
keys, so an unselected channel with missing statistics cannot block a fit.
Restart the dashboard and recompute Companion Correlators after updating; an
already-computed in-memory output cannot acquire the missing triplet statistics.

Physics suite after the twistor follow-up: **1,790 passed**, with 21 warnings. Command:
`uv run --offline pytest -q tests/physics --disable-warnings --tb=short`.
`git diff --check` passes for the changed code, tests and changelog.

The physics test suite, dashboard construction, CLI help, and a 500-particle
25-step simulation smoke test are exercised. The smoke test uses the dashboard
kinetic/cloning settings and coincident zero-velocity initialization; it crosses
the first scheduled cloning step, retains finite positions and does not terminate
early. This is not a long-run convergence or Standard Model calibration claim.

Formatting is applied to changed Python files. The full configured lint check
still reports issues, including pre-existing legacy annotations and style rules;
no repository-wide lint-clean claim is made. Unrelated Volume 1 documentation
edits in the worktree were preserved.

## Second round (2026-09-05): end-to-end run of `make physics`

The corrected tree was exercised headlessly: build every dashboard section the
way `create_app` does, run the simulation with the dashboard defaults
(500 walkers, 750 steps), and click every compute button in order. Four
read-only reviews of the simulation core, the dashboard wiring, the correlator
layer and the fitting layer ran in parallel. Regression tests for this round
live in `tests/physics/test_review_round2.py`.

### What failed or misbehaved before this round

| Area | Defect | Correction |
| --- | --- | --- |
| Holographic tab | Geodesic matrices were computed for every recorded frame (all-pairs shortest paths over 750 frames of 500 walkers: over 40 minutes, hundreds of MB kept in state). | Only the transitions selected by warmup / stride / `max_frames` are cached (4 minutes on the default run). |
| Algorithm tab | Pairwise distance statistics materialised `[T, N, N]` float64 tensors (about 7 GB peak on the default run). | Chunked over time (3 s, small memory). |
| Mass extraction | One joint `corrfitter` fit over all 37 correlators (16 minutes; over 45 minutes after a multiscale run). | Groups share no parameters, so each physical group is fitted on its own covariance block (5 s). `MassExtractionConfig.joint_fit=True` restores the joint fit. |
| Electroweak masses | With `cloning_frames_only` the series keeps one frame per cloning step, but the lag unit was the recording interval (masses 20x too large); `max_lag=40` exceeded the 30 frames and the FFT zero-padded lags, so the fit saw zero-variance points and `lsqfit` raised "Residuals are not finite". Dirac spinor channels used every frame and could not share a covariance with the cloning-frame channels. | Lag duration is the recorded time step times the median frame spacing; lags are clamped to the frame count; correlators carry exact origin statistics; spinor channels use the same frame subset. Channels whose covariance is singular are excluded with a warning instead of aborting; too few points raise a message naming the group and the fix. |
| Companion multiscale | `extract_mass_aic` raised "Insufficient supported lags" for one sparse channel and aborted every channel. | Sparse statistics degrade to a point estimate flagged `uncertainty_method="unavailable"` with an `uncertainty_note`. |
| Direct multiscale | Correlators came from a float32 FFT that `ensure_statistics` could not always match, leaving every channel without a measured error, so the 30% error filter rejected all of them ("0 channels"). `tensor_companion` passed a norm series that cannot reproduce its correlator. | Exact origin statistics are computed for every (channel, scale) series; the tensor channel recovers statistics from its (cos, sin) operator components on the frames it used. |
| Multiscale scale selection | Recorded kernel-volume *weights* were used as edge *lengths* in the shortest-path graph; distances were about 1e-27, all scales collapsed to the clamp floor and every scale returned identical results. | Inverse-type weights are inverted; kernel-type modes use the recorded geodesic edge distance (Euclidean fallback). Scales on the default run now span the physical inter-walker distances. |
| AIC window fits | Sufficient statistics were accumulated in float32 with an expanded-form residual (chi2 lost all digits for |log C| above about 10, giving negative chi2 and R2 above 1); chi2 was normalised by the window-mean variance so noise windows weighed as much as signal windows (errors off by 2.5x to 20x). | Float64 weighted least squares per window; `AIC = chi2 + 2k + 2 N_cut` with the excluded-point penalty; points with relative error above 50% never enter a window; the reported error combines the propagated covariance with the window spread. On an AR(1) process with mass 0.3 the pull distribution now has width 1.0 (was 0.39). |
| Bayesian priors | `corrfitter.fastfit` was called with keyword arguments it does not accept; the `TypeError` was swallowed, so seeding never ran. The amplitude prior was a fixed `0.5(5)`, biasing masses low by 10% to 25% whenever C(0) was far from one. With `nexp=2` the reported ground state could be a zero-amplitude prior artefact. | `fastfit(G, tmin=...)`; the amplitude prior is scaled from the data (`PriorConfig.scale_amplitude_prior`); default `nexp=1`; the ground state is the lowest level whose source-sink amplitude is resolved from zero (`ChannelMassResult.ground_state_index`). |
| Strong-force AIC tab | "AIC from Correlators" fitted with `dt=1` while "Direct Multiscale" used the recorded time step (same table, masses differing by 1/dt = 500). Prior seeding looked results up by group name but they are keyed by correlator key, so it never fired; "Best Window" read keys the fitter does not emit. | Both buttons use the recorded time step; seeding maps correlator keys to groups and converts to lag units; best-window rows read `r2` and the new per-window `mass_error`. |
| Deferred history changes | Holographic and coupling outputs, tables and plots from the previous run stayed on screen (the dashboard always defers). | Outputs are cleared on every history change. |
| Companion tab | Multiscale ignored `ell0_method`; the scale slider stayed visible after a single-scale run; the status counted best results as "channels". | Fixed. |
| `history.bounds` | Glueball/tensor momentum projection dereferenced attributes `RunHistory` does not have. | Tolerant access. |
| Fitness bilinear channels | No origin statistics, so the `pseudoscalar_fitness_*` modes could never be fitted with measured errors. | Statistics recorded per lag. |
| Electroweak settings | `epsilon_d` and `lambda_alg` were exposed but ignored. | Honoured when set. |
| Spinor history entry | `compute_electroweak_spinor_from_history` paired the colour state of frame t with the fitness of frame t+1. | Aligned; `compute_color_states_batch` now rejects `start_idx=0`. |

### Simulation core

* Seeded runs with `beta_curl > 0` were not reproducible: `torch.linalg.lstsq`
  is non-deterministic. The curl estimate now uses a Tikhonov-regularised
  `solve` (ridge relative to the local scale of the normal matrix); identical
  seeds give bit-identical histories.
* With `neighbor_graph_update_every > 1` the first `update_every - 1` steps ran
  without any graph (no coupling, no recorded edges). The graph is built on the
  first step.
* `sigma_min = 0` (dashboard default) produced NaN z-scores whenever a channel
  was constant (the first frames from a coincident start). Constant channels now
  give zero z-scores.
* `compute_companion_batch` filled neighbour columns beyond the two companions
  with walker 0.
* The full-metric Ricci scalar used `(d-2)|grad u|^2` with a metric-contracted
  norm instead of `(d-2)/2 |grad u|^2` with the flat norm (factor 2 in the
  gradient term).
* `RunHistory.n_steps` recorded the step reached instead of the requested
  length, so early termination was reported as "N/N".
* `KineticOperator(...)` rejected `auto_thermostat`, `n_kinetic_steps` and
  `integrator`; uniform edge weights ignored the run dtype; `chunk_size=1`
  allocated empty buffers; Qhull failures were silent (now a warning).
* The emergent-curl Boris rotation saturates near pi per half step for most
  walkers during the first ~50 steps of a coincident start (the curl estimate
  scales like 1/spacing). This is a modelling question left to the user; the
  operator now records `boris_rotation_angle` in the kinetic info and warns
  once per run when the median angle exceeds 1 rad.

### Default run after this round (500 walkers, 750 steps, headless)

| Step | Before | After |
| --- | --- | --- |
| Algorithm diagnostics | 77 s, ~7 GB | 3 s |
| Holographic principle | > 40 min | 4 min |
| Companion correlators | 51 s | 46 s |
| Mass extraction (37 correlators) | 16 min | 5 s |
| Electroweak masses | error | fits (degenerate velocity channels skipped with a warning) |
| Companion multiscale | error on sparse channels | 25 s, 10 of 16 channels pass the best-scale filters |
| Direct multiscale | 0 channels | 137 s, 9 channels across 8 scales |

### Left as modelling questions or follow-ups

* Restitution 1 (dashboard default) leaves cloned velocities unchanged.
* Tab computations still run inside the Bokeh callback, so the "Computing..."
  status is not shown until they finish.
* The kernel-multiscale `vector`, `axial_vector` and `tensor` channels
  correlate the norm of the frame-averaged operator; the companion variants use
  the contracted pair products.
* Global `nexp` / `use_log_dE` / `use_fastfit_seeding` in the mass tabs only
  seed the per-channel widgets created when correlators arrive.
* The SU(2) phase normalisation differs between `electroweak_observables`
  (`S + eps`) and `operators/electroweak_operators` (`|S| + eps`).
