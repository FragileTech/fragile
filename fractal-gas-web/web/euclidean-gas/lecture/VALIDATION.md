# Lecture experiment validation

The 128 registered experiments share Rust execution and analysis in native and
compiled form. Validation must check the provenance of the data, the numerical
measurement, the applicable prediction and the rendered interpretation. A
successful browser view alone establishes none of the scientific identities.

## Executed evidence

Each session resolves a registered request to complete run configurations and
records the updates used by its measurements. Evidence import validates those
configurations and archives before recomputing a result. Tests must reject
mismatched sources, missing required recording and a result bundle with no
underlying run evidence. Continuation experiments must preserve their complete
checkpoint and independent continuation addresses.

Relevant native suites include `lecture_registry`, `lecture_source_identity`,
`lecture_diagnostics`, `lecture_fractal_observables`, `lecture_field_geometry`,
`lecture_field_controls`, and `lecture_qft_evidence` in the benchmarks crate.
The early-part module also tests actual archived updates for all 42 IDs at seeds
0, 7 and 516. Test scope and execution outcome should be reported separately;
listing a test is not evidence that an unexecuted broad sweep passed.

## Numerical comparisons

| Family | Required interpretation and check |
|---|---|
| Operator stages and copying | Stage values come from the executed archive. Selected-donor acceptance probabilities are compared with the corresponding decisions; forced revival has its own accounting. |
| Fitness derivatives | Automatic derivatives and independent scalar differences use the same recorded donor realization, other measured rows and normalizers. Taylor coefficients are computed from that conditional field. |
| Thermostat moments | Predictions use the executed input, damping, factor, innovation law and eligible rows. Covariance and uncentered second moments remain distinct. |
| Harmonic component runs | Constant fitness disables copying. The actual BAOAB matrix and recorded initialization predict finite-step observables such as mean cos(v); they are not substituted for interacting dynamics. |
| Coordinate transport | Sorted matching is exact for equal-mass one-dimensional empirical marginals. The independent coupling cost is an upper bound for that same marginal problem. |
| Density and entropy | Histograms state the coordinate window, bins and retained mass. Discrete binned entropy is not relative entropy to a supplied invariant law. |
| Survival | Pre-clone eligibility, revival and final eligibility remain distinct. Conditional shape comes from actual survivors. Extinction is an explicit terminal outcome. |
| Independent-run comparisons | Run endpoints supply the sampling unit. Within-population variance, across-run variance and standard error are different quantities. |
| Timestep comparisons | Physical horizons match. Common seeds alone do not produce Brownian-bridge coupling; endpoint differences retain sampling fluctuations. |
| Fractal Set identities | Events preserve epoch, step, version, slot and generation. Scalar decoding, triangle incidence and displacement closure test distinct parts of reconstruction. |
| Cells and slabs | Original slot identities survive changing eligibility. Common-slot slab support is explicit. Partition closure and refinement accuracy are separate checks. |
| Empirical kernels | Row normalization preserves constants on the actual cloud. A spatial smoother does not by itself identify the geometric volume measure or a continuum wave operator. |
| Recorded dimension | Correlation slopes use Euclidean distances in the explicit `(x,y,c*t)` chart. Shared ancestry, finite support and saturation remain part of the data. |
| Executed curvature | Compare recorded scalar curvature with inverse-metric Ricci contraction on available evaluations. This is tensor consistency, not an independent curvature reconstruction. |
| QFT channels and geometry | Each contract names its actual recorded observable and physical interpretation. Mode fits, projected transitions, graph cuts and curvature closures require their own predictive checks. |

The Part V native integration suite runs all twenty analyzers on an executed
planar archive, checks finite plotted values and actual scene nodes, and verifies
that analysis leaves the archive bytes unchanged. Cell cases provide scene faces.
Focused tests exercise scalar finite-difference Hessians, serialized reanalysis,
common slab slots and absorbing-boundary interface labels.

The harmonic compiled regression independently derives the BAOAB transition from
its five stages and compares position moments across 64 separately seeded gas
runs. Its uncertainty calculation includes the initial uniform distribution’s
fourth cumulant. That check concerns its explicit non-cloning harmonic
configuration, not every lecture configuration.

The [Part V guide](../../../../docs/source/project/volume2_partv_interactive_experiments.md)
and [Part VI guide](../../../../docs/source/2_fractal_gas/partvi_experiments.md)
state the implemented observables. Detailed QFT comparisons belong in the
[numerical evidence report](../../../../algorithmic-gas/QFT_VALIDATION.md), with
its actual sample sizes and estimators.

## Control stress and failure outcomes

The native registry generates default requests at seeds 0, 7 and 516, numerical
endpoints, every categorical value, and pairwise combinations of controls. Run
bounded slices with `gas-lecture stress ID START COUNT`; inspect and retain each
result. Cases must exercise actual transitions rather than only configuration
serialization. An invalid request, unsupported readout or missing archive is a
reported error, never a substituted model.

Scientific outcomes need separate treatment. Extinction is an algorithmic
outcome. A singular or unavailable derivative has a coverage status. A spectral
fit can be inconclusive. A proposed constitutive relation can fail held-out
prediction. These results remain useful experiments when the display describes
them accurately; they must not be counted as confirmation of the proposed law.

## Compiled and browser checks

From the repository root:

```bash
npm --prefix fractal-gas-web run build:euclidean-gas
npm --prefix fractal-gas-web run build:euclidean-lectures
npm --prefix fractal-gas-web run test:euclidean-gas
npm --prefix fractal-gas-web run test:euclidean-lectures
npm --prefix fractal-gas-web run test:euclidean-lectures-browser
npm --prefix fractal-gas-web run test:euclidean-lecture-embeds-browser
```

Check CPU initialization, retained-module resets, bounded advancement, error
propagation, evidence reanalysis, checkpoint restoration and session disposal.
Verify that the rendered plots contain the Rust arrays with matching labels and
units. Browser tests also cover desktop/mobile controls, scene ownership and the
published deployment prefix.

Build the Theory book and inspect all 128 embeds, their section placements,
posters and Expert Mode lifecycle. Posters must be generated through completed
real sessions. Validation summaries must report completed checks and remaining
coverage explicitly; no fixed test total is a substitute for the
current test output.

## Executed validation record

The final native workspace run passed 293 tests, including the
browser-number JSON round trip, request/configuration rejection, fractional-count
rejection, exact displayed conditioning-state comparison and 96-step scene-window
checks. The additional VI-51 archive-only regression and its five-test suite passed, as did all 35 private field-reference tests. Workspace Clippy passed with warnings denied.

Adaptive VI-39 refinement passed the actual default 96-step regression, all 35
field-reference tests and all 15 core physics/geometry tests. The final compiled
QFT suite passed 67 tests. Chromium checks completed VI-19, VI-22 and VI-45 with
32 continuations per group, VI-39 with converged independent refinement, and
VI-51 with archive-only mechanical readouts.

The registry stress run passed all 258 Part V–VI defaults across seeds 0, 7 and
516, and all 1,455 individual categorical/range endpoints. The continuation-state
follow-up passed nine seeded defaults and 63 control endpoints. These counts do
not claim exhaustive execution of every generated pairwise combination.

Compiled checks passed the early-part suites and 64-replica harmonic comparison,
all 20 Fractal Set analyzers, all 66 QFT analyzers plus evidence rejection, the
four host checks and the 12 base engine checks. The browser sweep passed all
128 demos through initialization, advancement, reset and control changes, plus
replay, exports, mobile layout and project-prefix loading. Focused checks cover
96-step graph-window provenance and continuation replay. The assembled book
passed checks for all 128 figures and its iframe lifecycle.

At 96 steps and 32 walkers, the Fractal Set regression retains 9,248 graph nodes
and 208,896 edges for measurement. Its eight-transition scene contains 807 nodes
and 17,132 actual edges, including historical endpoints; the result is about
1.50 MB. Complete archives retain the full graph reconstruction inputs.

For default 96-step VI-19, VI-22 and VI-45 runs with 32 independent continuations
per group, exported native bundles occupy 43.15, 43.17 and 63.07 MiB respectively.
Comparing the compact conditioning
recordings with full audit recordings gave identical plots, metrics and complete
baseline archives. Conditioning checkpoints preserve the complete algorithm
state and donor history; metric experiments additionally retain the recorded
context needed to resolve historical companions.

The Theory build and portal assembly passed. External DOI tooltip lookups emitted
network-resolution warnings; local lecture assets and navigation passed the
assembled-book browser checks. Browser scientific execution uses the CPU/WASM
engine. Compiling the WebGPU profile does not constitute a GPU hardware run.
