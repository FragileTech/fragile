# Algorithmic physics verification

The calculations use the programmed Algorithmic Gas update: sampled companions,
fitness normalization, literal cloning, BAOAB, boundary handling, and the
configured clipped fitness metric. Metric derivatives follow the inputs consumed
by the noise provider. Full-update response includes changes in sampling probabilities.

## Numerical checks

| Check | Result |
|---|---|
| Native Rust workspace | 230 tests passed |
| Clippy, all targets and all features | Passed with warnings denied |
| Packed derivatives and curvature | Analytic references in 1D, 2D, 3D, 4D, and 8D; independent Christoffel and finite-difference checks passed |
| Clipped 3D curvature | Mixed-sign and rotated flat metrics passed in f32/f64; tensor symmetry is preserved during assembly |
| Fitness dependence | Coupled population gradient/Hessian agree with finite differences of the production fitness pipeline |
| Recorded dynamics | Recording preserves trajectories, reports, and replay; f32/f64, global/local normalization, revival, and boundary coverage checked |
| Research archive integrity | All 1,632 archives exactly reproduce the reported balances, curvature means, and availability counts; all 36,864 replica keys are distinct |
| Concurrent research runners | Two processes execute each shared job once; all 12 smoke archives reproduce their reports |
| Thermostat fluctuations | Gaussian and standardized-uniform mean/variance checked with 131,072 independent increments for each of four cases |
| Checkpoints and archives | Current schema round trips pass; unsupported schemas and missing field coverage are rejected |
| CPU curvature batch | 4,096 3D queries: maximum scalar/Ricci error 5.56e-17 in f64 and 2.99e-8 in f32 |
| CUDA and WebGPU | Native adapters compile; accelerator execution is unavailable in this environment |
| Theory documentation | Build passes; 112 DOI lookup warnings and four HoloViews MIME warnings |

The CPU batch measurements use the scalar Hessian contraction as the reference.
Separate tests compare that contraction with general metric curvature and analytic
geometries. The accelerator probes record their capability results in
`outputs/physics-validation/` at the repository root.

## Simulation results

Both studies are complete: 544 trajectories, 483,328 complete updates, and
78,381,056 walker updates, plus 36,864 independent replica updates. The archives
contain 236,544 curvature queries. All recorded kinetic budgets have zero
telescoping residual.

The global study's independent metric-increment comparison has median absolute
standardized residual 0.483 and 95th percentile 1.913. Its thermostat energy
residuals, standardized separately for each of the 14 configurations, range from
-2.713 to 1.834. Every global curvature query is available. In the 3D baseline,
the median final population-mean scalar curvature is 28,347.8 for 32 walkers,
64,795.9 for 128 walkers, and 77,480.7 for 512 walkers.

The tested two-coefficient spatial moment relation is
`mean(Ricci - scalar*g/2) = kappa*centered_kinetic_moment + lambda*mean(g)`.
Its coefficients are fitted on seeds 7–22 and evaluated on seeds 23–38 across
all configurations. Geometry and kinetic moments use the same available walkers
at the actual O-stage input.

| Study | Two-coefficient held-out RMSE | Constant baseline | Shuffled-training control |
|---|---:|---:|---:|
| Global normalization | 13,340,129.4 | 13,414,635.6 | 13,384,667.1 |
| Local normalization | 51,363.2 | 31,599.7 | 32,372.4 |

The global fit improves RMSE by 0.56% over the constant baseline; the local fit
has greater error than both controls. These measurements do not establish a
useful common two-coefficient relation across the tested configurations.

The complete configuration tables and figures are available in the
[global results](../outputs/physics-validation/physics-research.md) and
[local results](../outputs/physics-validation/physics-local-research.md).
The [global figure](../outputs/physics-validation/physics-research.png) and
[local figure](../outputs/physics-validation/physics-local-research.png) show
the individual seeds, curvature distributions, and thermostat residuals.
Machine-readable summaries are saved with the
[global archives](../outputs/physics-research/summary.json) and
[local archives](../outputs/physics-local-research/summary.json).

## Research protocol

The global study has 448 trajectories: 32 seeds for each of 14 configurations,
with 1,024 updates per trajectory. It covers 2D/3D populations of 32, 128, and
512 walkers, and controlled changes in metric regularization, temperature,
friction, fitness regularization, and time step. Six seed-7 baseline runs also
have 1,024 calibration and 1,024 validation replicas at three recorded updates,
for 36,864 independent replica updates.

The local study has 96 completed 3D trajectories: 32 seeds for each Gaussian
neighborhood width 0.3, 1, and 3; 32 walkers and 256 updates per trajectory.
Their standardized thermostat residuals are 0.562, 0.191, and -0.281. Every
recorded kinetic budget has zero telescoping residual. Three of the 9,216
curvature queries are marked unavailable at the clipping threshold.

Thermostat uncertainty is evaluated within each configuration across independent
seeds. Configurations share random addresses; their residuals are correlated.
Curvature is evaluated per query before averaging. All measurements, masks,
parameters, and replica keys are retained in the JSON reports and CBOR archives.

## Reproduce

From `algorithmic-gas/`:

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo run --release -p algorithmic-gas-benchmarks --example physics_research -- --research --output ../outputs/physics-research
cargo run --release -p algorithmic-gas-benchmarks --example physics_research -- --research --local --output ../outputs/physics-local-research
cargo run --release -p algorithmic-gas-benchmarks --example physics_analyze_research -- ../outputs/physics-research
cargo run --release -p algorithmic-gas-benchmarks --example physics_analyze_research -- ../outputs/physics-local-research
```

The research runner resumes completed jobs, locks each job through archive and
report writes, and supports process partitions with `--shards K --shard I`.
The analyzer checks every archive against its report before publishing a summary.
From the repository root, render completed
summaries as PNG, PDF, and Markdown tables:

```sh
uv run python algorithmic-gas/tools/plot_physics.py outputs/physics-research/summary.json outputs/physics-local-research/summary.json --output outputs/physics-validation
```
