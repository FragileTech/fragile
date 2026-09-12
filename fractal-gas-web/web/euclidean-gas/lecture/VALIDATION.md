# Numerical validation of the Volume II experiments

The lecture tests compare observables with independently derived identities,
finite-sample uncertainty, and the assumptions of the displayed prediction.
Run all family, host, and engine-moment checks with:

```sh
npm --prefix fractal-gas-web run test:euclidean-lectures
```

The family tests exercise the failure cases found in the September 2026 review,
alongside the existing descriptor, control-endpoint, and replay checks. The
following checklist records what must remain aligned when changing a demo.

| Experiment | Required comparison or diagnostic |
| --- | --- |
| I-03 | Local statistics include the recipient by default, matching the chapter; the alternative convention is explicit. |
| I-05 | The printed BAOAB update and the engine use force–drift–OU–drift–force, a fresh final force, and the exact OU variance. |
| I-06 | Independent seeds expose finite-time well trapping and reward/force separation. |
| I-08 | Donors are conditioned on eligibility before sampling; singleton, revival, and extinction have explicit outcomes. The position mixture is compared using cumulative bin probabilities, including atoms. |
| I-10 | Prescribed metric geometry and measured constant-factor diffusion have distinct labels. |
| II-01 | The default high/low partition is nondegenerate; measured fitness gaps and overlap are visible. Fixed-center displacement and recentered variance are distinguished. |
| II-02 | Slope uncertainty accompanies the drift fit. Unsupported, negative, or extrapolated zero crossings are not reported as stationary floors. Pure Gaussian jumps have an exact constant-drift reference. |
| II-04 | The transport proxy is an upper bound; the exact centered distance need not decrease at each step. |
| II-05 | OU mean and variance comparisons use physical time and finite-sample uncertainty. |
| II-08 | Exact survival and conditional shape are separated; expected survivor counts determine a useful empirical horizon. |
| III-03 | Sparse pair histograms have an independence calibration. Final marginal variance, independent replicas, and active cloning are included in population comparisons. |
| III-04 | Paired discrete/continuous generator residuals expose timestep bias separately from sampling scatter. |
| III-06 | Spectral truncation error is separated from transient error and decreases with retained modes. |
| IV-01 | The modified entropy uses the chapter's admissible matrix and exponential envelope, including low-friction stress settings. |
| IV-02 | Roundoff-scale negative KL is clamped; invariant and survival-conditioned references have distinct meanings. |
| IV-04 | Both directions of the Gaussian density ratio are identified, and the absorbing estimator respects zero endpoint density. |
| IV-05–IV-06 | Derivative comparisons and normalized-weight identities retain numerical tolerances and distinguish external query sums from self-inclusive walker sums. |
| IV-07 | Taylor reconstruction reports its empirical useful neighborhood and error as order and window change. |
| IV-08 | The greedy companion law averages shuffled processing orders when comparing with the engine. |
| IV-09 | Metropolis transitions have their own clock; independent initializations expose slow well exchange. |
| IV-10 | Second-order covariance bias and fourth-order Gaussian KL bias are separate diagnostics; finite physical duration remains visible. |
| IV-11 | Covariance estimation has a sample-size-dependent error reference. |
| IV-12 | A cut-based Rayleigh bound and numerical resolution check prevent roundoff from masquerading as a resolved spectral gap. |
| IV-13 | Stationary sampling, stationary timestep bias, and a bound for the absolute-error transient follow the chapter's decomposition. Finite-time sampling bands are labeled separately. |
| IV-14 | Refinement holds physical duration fixed and uses exact moments to isolate second-order weak bias. |
| IV-15 | Feasibility allows equality; the finite-width sufficient formula and saturation denominator are labeled precisely. |
| IV-16 | Matched-seed survival comparisons actually cross the boundary, and population comparisons use independent replicas with standard errors. |

## Independent engine relaxation regression

`lecture-theory-regression.test.mjs` runs 64 independently seeded, 64-particle
WASM populations in the two-dimensional conservative harmonic model. Positions
start uniformly in `[-1, 1]`, velocities at zero, with timestep `0.04`, friction
`1`, and temperature `1`. It compares mean squared radius at seven times through
time `12` with a moment recurrence derived directly from the five scalar BAOAB
stages. It does not import the lecture's Gaussian moment helpers.

The standard error includes the fourth cumulant of the initial uniform law;
using a Gaussian variance formula at early times would give the wrong sampling
reference. Fixed independent seeds make the check reproducible. Its four-standard-
error threshold checks the ensemble mean, not every individual population.

## Browser and publication checks

After updating formulas or controls, regenerate the computed posters and chapter
manifest, then run the browser tests against the local preview server:

```sh
npm --prefix fractal-gas-web run build:euclidean-lectures
uv run --no-project python fractal-gas-web/tools/serve-control.py --port 8770
npm --prefix fractal-gas-web run test:euclidean-lectures-browser
```

The browser checks cover every demo's initialization, stepping, reset,
alternative controls, replay/export, and narrow-screen layout. Documentation
tests in `tests/docs/test_fragile_gas_demos.py` validate all 62 placements and
the published asset paths. `captions.json` supplies both the live captions and
the generated manifest, so formula changes must be reflected there before
regenerating assets.

## Repair verification, 12 September 2026

- 84 numerical and host tests pass: 26 foundations, 23 convergence, 30 entropy,
  four host/publication checks, and the independent 64-replica WASM regression.
- All 42 default experiments were rerun, with finite results through their
  declared horizons or 1,000 host steps for open-ended experiments.
- All 73 documentation placement tests pass. All 42 views pass browser
  initialization, stepping, reset, and alternative-control checks. Replay,
  exports, mobile layouts, extinction recovery, and unresolved-gap display also
  pass in Chromium.
- The repaired Keystone default has high/low fractions `0.15625` and `0.84375`.
  Its observed overlap `0.15625` exceeds the displayed lower bound `0.130244`;
  all 160 tested repetitions satisfy the reported hypotheses.
- The repaired active-cloning experiment reports a cloning fraction of about
  `0.2609`. Its selected pair discrepancy `0.37413` is compared with a permutation
  interval `[0.20007, 0.39625]`, rather than interpreted against zero.
- Independent review of 27 hypocoercive parameter corners found no increase or
  envelope violation over 350 steps. Shuffled-greedy probabilities agreed with
  a separate enumeration to `1.11e-16`.
- In the nearly disconnected graph case, the display reports unresolved gap
  with Rayleigh upper bound `3.23474e-43`. The absorbing KDE has exactly zero
  endpoint values and is normalized after reporting its retained mass.

The full default-run snapshots from this verification are local generated
artifacts under `outputs/lecture-theory-fixes/`; the committed regression tests
provide the reproducible checks independently of those artifacts.

## Part V validation (12 September 2026)

- Rust workspace: **106 tests passed**, including 17 geometry, 14 independent
  analysis, 10 tracking and 5 adaptive-provider checks. Full workspace Clippy
  passes with warnings denied. The WASM WebGPU feature configuration compiles;
  the adaptive teaching profile explicitly requires CPU/WASM f64.
- Lecture numerical/host checks: **108 passed** (84 existing and 24 Part V).
  Part V checks include all declared scientific-control endpoints, exact metric
  identities, covariance sampling uncertainty, the dimension reference, and
  batch-16 versus sixteen single-step archive/checkpoint parity.
- Documentation: **98 placement/extension checks passed**. Sphinx builds all
  chapter pages with 62 figures; the build reports 116 asset-copy/reference
  warnings. Twenty new posters are generated from native calculations.
- All **62 browser demos** initialize, advance, reset and change controls. The
  dedicated Part V browser test verifies shared recording across graph views,
  archive import/export, view-only camera/layer/selection controls, adaptive O
  noise, variable geometry, spacetime cells, physical causal orders, and a
  390-pixel viewport without horizontal overflow.
- Published embeds pass lazy loading, one-live-iframe behavior, hidden-view
  pause, Expert Mode, close/reopen, no-JavaScript posters and project-prefix URLs.

The detailed per-demo prediction and observed statistical comparisons are in
[the Part V experiment document](../../../../docs/source/project/volume2_partv_interactive_experiments.md).
The default curvature and narrow-bandwidth experiments expose large measured
uncertainty; independent quadrature supplies the separate finite-resolution
reference. The UI does not substitute that reference for the sampled result.

Local visual evidence is saved in `fractal-gas-web/outputs/partv-review/`
(ignored build output). Run the scene suite with
`node fractal-gas-web/tests/euclidean-gas/lecture-fractal-browser.mjs`.

## Part V independent audit (12 September 2026)

The follow-up audit compares the native calculations with independent exact
identities, direct BAOAB ensembles, planar time-slice integration, analytic
segment lengths, and compiled-WASM reference experiments. Confirmed fixes:

- Stable metric eigenvalues and near-axis Spin(2) encoding preserve small
  components instead of losing them to subtraction.
- Positive-volume affine score ties have one spacetime-cell owner. Moving-cell
  refinement is checked independently of total-volume closure.
- Off-grid geodesic source attachments use the same midpoint quadrature as
  ordinary graph edges. Physical time cuts clip partial faces and edges.
- Harmonic sample variance uses N−1 and has an exact transient reference with
  the initial uniform law's fourth-cumulant uncertainty.
- Manufactured-operator variance, MSE and SEM have independent quadrature
  predictions even when every observed support is empty. Curvature predictions
  identify the shrinking sampling cube and show finite-bandwidth bias separately.
- Dimension inversions retain their original replica indices. Count variances
  have uncertainty computed from Poisson/binomial fourth moments.
- IG/historical relations include available source-relative displacement data.
  Archive validation checks operator coverage and consistent event identities
  within and across recorded steps, including epoch anchors.

No contradiction with the audited chapter statements was established. The
zero-gradient manufactured field has leading variance 1/(N ε³), sharper than
the general bound. The curvature cube shrinks with ε, giving 1/(N ε⁴); the
fixed-density theorem describes a different sampling protocol.

Detailed counterexamples, corrections and numerical measurements are recorded
in Section 7 of the linked Part V experiment document.

Validation: 125 Rust workspace tests, 111 lecture numerical/host tests, and
98 documentation placement tests pass. Workspace Clippy passes with warnings
denied, and the CPU/WASM release builds successfully. New regressions include
an actual 32-replica engine check at both anisotropy endpoints, independent
512-ensemble harmonic uncertainty, exact polynomial variance integrals,
spacetime refinement, archive boundary continuity, and partial time cuts.

All 62 demos pass browser initialization, stepping, reset and control changes.
The dedicated Part V suite passes archive import/export, shared histories,
corrected harmonic/bandwidth plots, partial time cuts and a 390-pixel layout.
The standalone WASM contract suite passes. The incremental Sphinx build succeeds
with four DOI-network/tooltip warnings from unavailable external DOI metadata.
