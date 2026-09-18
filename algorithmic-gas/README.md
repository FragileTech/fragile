# Algorithmic Gas

An independent Rust implementation of the Volume 2 [architecture specification](../docs/source/2_fractal_gas/architecture/01_algorithmic_gas.md). It does not modify or wrap the Python Atari, robotics or Euclidean engines, or the C++ Optimization Lab.

CPU and WASM CPU support both `f32` and `f64`. Accelerator adapters use **host-orchestrated Burn batches**: objective/gradient evaluation, pair reductions, noise factors and kinetic arithmetic can execute on the selected device; sampling decisions, fitness statistics/maps, cloning and random-number generation run on the host. Transfers are explicit and counted. There is no automatic CPU fallback or precision reduction.

## Build and run

Requirements: Rust 1.95.0 (pinned in `rust-toolchain.toml`), and Node 22 for the browser build. Run Cargo commands from this directory.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --release -p algorithmic-gas-benchmarks --bin gas-benchmark -- --precision f64 --walkers 256 --steps 100
```

The runner accepts `--benchmark ID` (any catalog id; `--list-benchmarks` prints the catalog), `--instance N` for COCO BBOB, `--param KEY=VALUE` for benchmark parameters, `--dimensions`, `--seed`, `--backend`, and `--config FILE`. Arguments apply in order; place `--config` before overrides. Explicit file boundary settings are preserved. Output is JSON with resolved configuration, initialization and step timings, raw reward summaries, submitted reward-row count, synchronization count and transfer bytes. Initialization/compilation must be reported separately from steady-state performance.

```sh
cargo run --release -p algorithmic-gas-benchmarks --features wgpu --bin gas-benchmark -- --backend wgpu --precision f32
cargo run --release -p algorithmic-gas-benchmarks --features cuda --bin gas-benchmark -- --backend cuda --precision f64
```

These profiles require a compatible adapter/driver. Compilation alone does not verify GPU execution. The WGPU adapter rejects software CPU adapters. CUDA `f64` must pass the device's dtype and primitive-operation checks; it is not promised to be fast.

## Canonical Euclidean Gas and population diagnostics

`GasConfig::euclidean(dimensions, dt)` selects the fixed-step kernel analyzed in
Volume 2. Supply the quadratic benchmark and gradient for the canonical lecture
configuration, with `dt = 0.04`. The configuration retains weighted sampled
fitness, immediate weighted revival, frozen connected-component collisions with
one shared Haar rotation, BAOAB, final independent position diffusion, the smooth
radial velocity cap, and terminal absorption. Explicit library configurations
can select other donor, noise and boundary rules; their mathematical scope is
specified by those choices.

`mean_field::step_diagnostics` reports actual clone/revival counts, gate variance,
component sizes, the forest edge identity, shared-component probabilities, and
bounded empirical observables. `physics::field_evolution::collision_field_balance`
integrates the shared component rotation, including donor velocity changes and
cross-walker covariance. `mean_field_population::AtomicMeanField` independently
samples the marked rooted population law from an entering atomic measure; it
retains sampled fitness and incoming cloners. Exploration capacity errors are
returned without truncating or resampling oversized components.

Part III experiments expose these calculations with configuration and seed
provenance. Reproduce the population study with:

```sh
cargo run --release -p algorithmic-gas-benchmarks --example mean_field_validation -- 128 0 canonical
```

The other study names are `boundary_stress`, `shared_initial`, and
`shifted_initial`. An optional fourth argument selects comma-separated population
sizes. [Validation results](MEAN_FIELD_VALIDATION.md) separate independent-run
uncertainty, exact conditional balances and population comparisons; the
[engineering audit](MEAN_FIELD_AUDIT.md) records the repaired proof steps.

## Numerical algorithmic physics

`physics::jet::JetSpace` provides packed multivariate Taylor algebra through order
twelve, with a 1,024-coefficient admission limit checked before allocation. In 3D, order three uses 20 coefficients (value, three gradients, six Hessian
and ten third-derivative components). `FitnessJet::from_jet` exports ordinary
derivatives in lexicographic nondecreasing-axis order. Global conditional fitness
uses prefix/suffix Welford summaries: O(N) preparation and O(1) work in population
size per query. `pipeline_from_measurement_jets` retains all derivatives in local
normalization or population-dependent measurements, using an O(N) query fallback.

`physics::geometry::hessian_curvature` computes Ricci, scalar, and eigenframe
sectional curvatures directly from the Hessian and third derivatives. A full
Riemann tensor is optional. `fitness_curvature` preserves the configured spectral
clipping: mixed-sign clipping requires fourth derivatives and spectral divided
differences; a threshold query reports unavailable classical curvature. The
constant negative branch has metric epsilon times identity and zero curvature.
`metric_curvature` accepts general metric derivatives as an independent
Levi-Civita reference. Its derivative layout is `dg[k,i,j]`, `ddg[k,l,i,j]`.

`ExecutionContext::smooth_curvature_batch` performs whitening and Ricci/scalar/
sectional contractions on the selected Burn backend, with one readback and bounded
batch allocation. Small symmetric eigensolves remain host computations. CPU
supports f32/f64, WebGPU f32, and CUDA the device-supported requested precision.
Mixed-sign clipped queries require the general metric path; this batch method
returns an explicit capability error instead of falling back to CPU.

Set `RunConfig.physics_metric` to a `PhysicsMetricConfig` for general-dimensional
fitness-conditioned BAOAB diffusion. Metric factors are evaluated at the actual O
query, with selection-stage donors and eligibility frozen. Curvature recording is
observational and skips D3/D4 work when no archive is requested. Archived field
coverage distinguishes a missing curvature from a computed zero.

`physics::balances::analyze_step` computes stage energy, momentum, particle-count,
and kinetic-stress diagnostics. Pre-boundary snapshots separate the thermostat
from killing or periodic repair. Conditional energy mean and variance use the
executed diffusion factor, integrator scaling, source shifts, and innovation law.
The report isolates the O substep before boundary changes.
`Checkpoint::with_future_seed` creates a replica checkpoint with unchanged
physical/history/provider state and schedule, fresh future random addresses, and
a fresh recording boundary. Identical checkpoint replay remains deterministic.

`physics::evolution` checks disjoint calibration/validation replica keys;
`physics::closure` reports held-out constitutive errors against a constant
baseline. Empirical transition matrices, their fitted response diagnostics and the fitness
Hessian are separate numerical objects. Their measurement contracts state the
observable, normalization and applicable mathematical identity.

Run the reproducible smoke simulation or the full research preset:

```sh
cargo run --release -p algorithmic-gas-benchmarks --example physics_research -- --output ../outputs/physics-smoke
cargo run --release -p algorithmic-gas-benchmarks --example physics_research -- --research --output ../outputs/physics-research
cargo run --release -p algorithmic-gas-benchmarks --example physics_analyze_research -- ../outputs/physics-research
```

The research preset runs 32 seeds, d=2/3, N=32/128/512 and 1,024 updates, plus
eight one-factor comparisons at N=128,d=3. Seed-7 baseline checkpoints at updates
1, 512 and 1,024 have 1,024 replicas in each of two independent pools. JSON reports
and CBOR archives are written per job; repeat the command to resume completed
jobs. The manifest records exact coverage and limitations. A half-step comparison
keeps the clone operator fixed while changing the BAOAB time step.
Independent processes can partition the preset with `--shards K --shard I`,
where `I` runs from zero through `K-1`; job IDs and random keys are unchanged.
Adding `--local` runs the neighborhood-width study: 32 seeds, 32 walkers in 3D,
256 updates, and Gaussian widths 0.3, 1, and 3, for 96 trajectories. Use a separate
output directory for this study; the same analysis command summarizes it.
The [verification report](VERIFICATION.md) records numerical checks and study
results. `tools/plot_physics.py` renders completed summaries as scientific plots
and Markdown tables using the repository's Python environment.

Archives and checkpoints validate their supported schema and explicit field coverage.

## Tessellation geometry and the Einstein–Hilbert gas

`algorithmic_gas::tessellation` tessellates the swarm and estimates the geometry
the walkers sample. The stage is a pipeline of independently replaceable
components, each a small trait implemented by a serde config enum:

| Role | Trait / config | Provided |
|---|---|---|
| Sites | `Projection`, `TessellationDomain` | any subset of coordinates (for example all but Euclidean time); open space, clip box (mirror images) or periodic box (translated images) |
| Tessellator | `Tessellator` / `TessellatorKind` | sorted path (1D), Spade Delaunay (2D), native Bowyer–Watson with exact predicates and ghost tetrahedra (3D) |
| Degeneracy | `DegeneracyPolicy` | coincident walkers share a site and form a clique; a swarm in an affine subspace is triangulated inside it; nothing is ever perturbed |
| Voronoi dual | `VoronoiCells` | cell volumes, facet areas, vertices and boundary classes from circumcenters, `V_i = Σ_j A_ij |x_j − x_i| / 2d` |
| Metric | `MetricEstimator` / `MetricKind` | inverse neighbor covariance (emergent metric), Voronoi-cell covariance, neighbor finite-difference Hessian, an observation field, identity |
| Volume element | `VolumeElement` / `VolumeKind` | `sqrt(det g)`, Voronoi cell volume, their product, unit |
| Edge weights | `EdgeWeighting` / `WeightMode` | uniform, inverse Euclidean or Riemannian distance, inverse Voronoi or Riemannian volume, Euclidean and Riemannian kernels, kernel × volume, facet-area weights |
| Curvature | `CurvatureEstimator` / `CurvatureKind` | conformal graph Laplacian of `log det g`, local quadratic fit (scalar and Ricci tensor), Regge deficit angles (2D vertices, 3D edges), Voronoi volume/shape distortion, Raychaudhuri expansion |
| Reward | `RewardAllocation` / `RewardAllocationKind` | per-walker Einstein–Hilbert density `R_i · vol_i`, curvature only |

`GeometryPipelineConfig::evaluate` composes them and
`evaluate_with(&dyn Tessellator, …)` accepts a custom tessellator; the
`GasOperators::geometry` hook replaces the whole stage. Setting
`GasConfig.geometry` runs the stage inside the step. With the default
`GeometrySchedule::EveryStage` it tessellates whenever positions changed before
a reward evaluation (after cloning and after the kinetic update), so every
reward, the committed fields and `AlgorithmicGas::graph` describe their own
positions. `GeometrySchedule::PostClone { every, on_clone }` tessellates the
post-cloning population only, at most once per step: the other rewards read
the fields carried by the walkers and graph forces reuse the last graph.
Per-walker results are written to observation fields
(`geometry.curvature.<name>`, `geometry.volume_element`, `geometry.diffusion`),
so they are checkpointed with the population, copied from donor to clone, and
readable by reward sources (`GeometryReward`) and by `NoiseGeometry::Full` as a
diffusion factor. Being observation fields they are part of every recorded
population; `RecordingConfig.graph` adds the `GraphSnapshot` that drove each
step's graph forces to the `RunArchive`. The neighbor graph is kept as a
`GraphSnapshot` for graph forces: `QftExecutionConfig.graph_viscosity` is the viscous coupling
`F_i = ν Σ_j w_ij (v_j − v_i)` over tessellation neighbors, and
`QftExecutionConfig.curl` turns every B step into quarter kick, Cayley rotation
by the curl of that force, quarter kick (Boris BAOAB). `CloneDecision.every`
gates cloning to a period and `CollisionRotation::Identity` keeps the direction
of relative velocities in collisions.

Tessellation predicates are exact (`robust`, Spade); numerical geometry runs in
the run dtype. With the default `parallel` feature per-walker, per-edge and
per-cell stages run on a Rayon pool, and `par::map_tasks` runs independent
tessellations (replicas, time slices) concurrently. Every output element is a
pure function of immutable inputs and no floating-point reduction crosses
threads, so serial and parallel results are bit-identical; wasm32 and
`--no-default-features` builds compile the serial path only.

`GasConfig::einstein_hilbert(temperature, dt)` is the free gas whose reward
(`GeometryReward`) is `r_i = R_i · sqrt(det g_i)`: no potential force
(`ZeroPotential`), graph viscosity with Boris rotation, an Ornstein–Uhlenbeck
thermostat, and uniform random mutual pairings for both the diversity and the
cloning companion. Like `GasConfig::euclidean` it returns an ordinary
configuration whose fields are then edited directly. In the benchmark crate
`RunConfig.geometry_reward` selects a geometry reward in place of the objective
(the force comes from `RunConfig.potential`, or vanishes), equal initial bounds
start every walker at one point, and `RunConfig::einstein_hilbert()` is the
500-walker, three-dimensional run from the origin at rest:

```sh
cargo run --release -p algorithmic-gas-benchmarks --bin gas-benchmark -- --einstein-hilbert --steps 750
cargo run --release -p algorithmic-gas-benchmarks --bin gas-benchmark -- --einstein-hilbert --walkers 60 --steps 40 \
    --record run.json --record-graph --record-bytes 2000000000
uv run python tools/eh_archive_to_history.py run.json history.pt   # RunArchive -> RunHistory for the QFT analyzers
uv run python tools/export_tessellation_fixtures.py               # reference values for tests/tessellation_parity.rs
```

`--record` writes the engine's `RunArchive` (CBOR, or JSON for a `.json` path) for any
benchmark run; the summary gains a `geometry` block when the gas has a geometry stage.

Two estimator properties matter when choosing components. The quadratic-fit
ridge is absolute, so it must stay far below the squared neighbor spacing. A
Regge deficit angle is `O(h²)`, the same order as the relative error of an edge
length built from endpoint metrics; Regge curvature converges with exact
geodesic lengths (`regge::curvature` accepts any lengths) and is a biased
indicator with `ReggeLengths::Geodesic`.

## Project layout

| Path | Responsibility |
|---|---|
| `crates/algorithmic-gas` | Public data contracts, selection, fitness, cloning, kinetics, noise, tessellation geometry, private Burn adapters, transactional engine and checkpoints |
| `crates/benchmarks` | Objective catalog (classic tensor-graph objectives with analytic gradients, pure-Rust COCO BBOB suite), initialization, native runner and end-to-end tests |
| `crates/wasm` | Asynchronous browser bindings and precision-preserving binary checkpoints |
| `../fractal-gas-web/web/euclidean-gas` | Independent worker, controls, 2D/3D population views, objective surface, convergence plot and walker inspector |

Burn 0.21.0 is pinned and remains behind `ComputeBackend` / `ExecutionContext`. No public signature exposes a Burn tensor. `Real` is sealed to `f32` and `f64`; one precision applies to all numerical fields, rewards, random innovations and arithmetic within a run. Configuration values are converted once at their use sites; unrepresentable nonzero model parameters are rejected rather than replaced with dtype-dependent defaults. Indices use `u32`, with checked `i32` limits for the initial backend adapters.

## Public API

Build a benchmark run asynchronously:

```rust
use algorithmic_gas::Precision;
use algorithmic_gas_benchmarks::RunConfig;

async fn run() -> algorithmic_gas::Result<()> {
    let mut config = RunConfig::default();
    config.gas.precision = Precision::F64;
    let mut gas = config.build::<f64>().await?;
    let report = gas.step().await?;
    assert_eq!(report.step, 1);
    let saved = gas.checkpoint().to_bytes()?;
    gas.step().await?;
    gas.restore(algorithmic_gas::Checkpoint::from_bytes(&saved)?)?;
    Ok(())
}
```

For another domain, construct `Population<T>` and use `GasBuilder::new(population, reward_source)`. Supply `.config(...)`, `.gradient(...)`, `.domain(...)`, and `.operators(...)` as needed. The engine provides immutable population access; changes enter through a step, an explicit extraction barrier, validated `replace_population`, or checkpoint restore.

### Data, observations and rewards

`InputBatch<T>` contains named numerical tensors, named byte rows, a row count and an input version. It is an immutable external input context for one step, not an opaque simulator snapshot. Extractors also receive an immutable view of current algorithm state.

`ExtractionPipeline` runs an optional `DerivedFieldProvider` once, then independent `ObservationExtractor` and `RewardExtractor` instances on the same expanded inputs. A shared expensive function can therefore feed both outputs. `NamedObservationExtractor` maps input fields to numerical observation fields; `NamedRewardExtractor` selects a scalar field. Custom extractors can decode RAM, create image tensors, evaluate an action functional, or combine input data with positions/velocities.

- `ObservationBatch<T>` holds named `TensorBatch<T>` values with shape `[N, ...item_shape]`. Vectors, phase space and RGB images retain their declared shape. An operator explicitly selects a field; an image is not implicitly a physical position.
- `RewardBatch<T>` holds one **raw scalar** per walker, validity and provenance. Maximize/minimize changes fitness orientation, not the stored raw value. Rewards do not need to be a function of the chosen distance features.
- `StateStore` holds optional opaque byte snapshots and a codec identifier. Built-in selection and fitness never inspect their contents. Domain adapters interpret snapshots, restore/advance simulators, and refresh numerical fields. Custom hooks currently receive the population and must honor this separation; capability-restricted numerical views are not yet enforced by the type system.

Use `step_with_extraction(&input, &pipeline)` to admit new observations and rewards together, or `step_with_input(Some(&input))` when only the reward provider consumes external inputs. The configured `RewardSource` must be able to refresh the same objective after cloning and kinetics; extracted reward is not silently retained after a state change. Derived values have no implicit cross-stage cache. A version belongs to one population snapshot, not just one walker slot.

Custom domain methods operate on the uncommitted population. They must keep numerical observations consistent with snapshots, reject unsupported coordinate edits, and advance the population version when refreshing changes its observations. Irreversible external simulator side effects are outside transactional guarantees; use restore/advance/capture from supplied snapshots. The test suite includes a small opaque-state domain; no Atari/robotics integration is included.

### Independent donor roles

`GasConfig.distance_donors` and `GasConfig.cloning_donors` are separate `DonorModule` instances and have separate addressed random streams. Each specifies its own distance, kernel, joint sampling law, self rule, replacement policy, odd-population policy and retained history window.

| Choice | Implemented variants |
|---|---|
| Distance | Euclidean or squared Euclidean; scaled phase space; cosine dissimilarity with explicit zero-vector tolerance |
| Kernel | Uniform; Gaussian with distance/squared-distance convention; explicit exponential kernel for general dissimilarity |
| Sampling law | Independent directed draws; uniform Fisher–Yates mutual matching; sequential Gaussian greedy matching; permutation |
| Diversity reduction | Mean, weighted mean, minimum, maximum of scalar pair comparisons |

Distance companions are dense masked `[N,K]` pool indices, default `K=1`. Multiple mutual rounds produce multiple companions; each round is a matching, not one globally disjoint multi-edge graph. Cloning currently requires `K=1`. `ClonePlan` records a list of weighted sources per recipient so future recombination can extend the representation, but more than one contributing donor is explicitly rejected today.

Without replacement, `insufficient: "use_available"` is the default: remaining capacity is masked when fewer candidates survive. `"reject"` selects a strict error. Uniform replacement draws use direct ranks; uniform sampling without replacement uses a sparse partial shuffle. Mutual Fisher–Yates uses one precomputed slot lookup, so its single-round sampling work is linear.

Sources are resolved by frame, slot, generation and population version. They come from the current eligible population or retained past snapshots. **Ancestry never restricts default selection.** Historical cloning reevaluates source rewards/diversity under current global statistics, rather than reading a current slot's fitness. Initial historical cloning does not support local standardization or external per-slot input batches; those combinations fail explicitly pending source-aligned input/statistics adapters. Historical mutual matching is also unsupported; use independent sampling.

Gaussian independent sampling streams tiles without an `N × M` matrix. Uniform independent draws and CPU/WASM Fisher–Yates avoid quadratic comparisons. Greedy matching remains sequential; a GPU tensor backend does not make Fisher–Yates or greedy dependency chains parallel. Uniform independent sampling and uniform mutual matching are distinct joint laws.

### Fitness, boundaries and cloning

Fitness separates measurement, standardization, positive mapping and combination. Built-ins include global alive-only regularized population statistics, local kernel-weighted statistics, sample statistics, logistic and asymmetric maps, additive positivity floors, independent reward/diversity exponents, and a distance measurement floor. The distance floor applies **after** multi-companion reduction as `sqrt(reduced² + floor²)`.

Reducers convert squared-distance edge values to distances before mean, weighted mean, minimum or maximum. Cosine remains a dimensionless dissimilarity and uses scaled normalization to preserve small/large nonzero vectors. Local statistics default to `include_self: false`; an empty singleton neighborhood falls back to global statistics, recorded by `Statistics.global_fallback`. Set `include_self: true` explicitly for self-inclusive local statistics.

An invalid/nonfinite reward on an eligible row fails by default. Explicit `invalid_reward: "exclude"` instead marks that row ineligible before donor draws; its reward validity flag remains recorded. No implicit fixed-point iteration reevaluates a population-dependent reward after excluding rows.

Boundary policies are unbounded, absorbing box, periodic box, external termination and compositions. Invalid data, out-of-bounds, termination and truncation remain distinct flags. Terminated/truncated rows are excluded by default. Periodic distances must use the same domain as periodic repair. Repair invokes domain reconciliation. A singleton may choose itself, with zero separation before the floor. Extinction returns an explicit error; dead slots revive from available current alive donors.

All clone reads use a frozen snapshot and separate destination storage: chains, swaps and repeated donors are safe. Literal copying includes every observation field, raw reward and opaque snapshot. Optional position jitter and pairwise velocity restitution are explicit transforms. Restitution requires disjoint alive mutual pairs; overlapping directed donors are rejected, not treated as momentum-conserving. Revival uses a separate donor-copy rule.

### Kinetics and independent noise

`Noise` combines `InnovationLaw` with `NoiseGeometry`. Gaussian and standardized uniform innovations have unit covariance. Isotropic, diagonal, full and low-rank factors support constant, per-walker and observation-field values. For `eta = L xi`, conditional covariance is `L Lᵀ`; full/low-rank inputs are **factors**, not covariance matrices. Noise is independent across walkers and has no temporal scaling.

The integrator owns scaling:

- Direct jump: `x += amplitude * eta`.
- Brownian jump: `x += amplitude * sqrt(dt) * eta`.
- BAOAB: explicit potential gradients at both B stages; scalar friction `gamma`; O stage `v = exp(-gamma*dt) v + s eta`, with `s² = (1-exp(-2 gamma dt))/(2 gamma)` and `s²=dt` at zero friction. Here `L` is the diffusion factor in `dv = ... + L dW`, not the equilibrium covariance. For scalar thermal mass-one dynamics choose `L=sqrt(2 gamma T) I`.
- Environment: the domain adapter restores, advances and snapshots eligible walkers.

A non-Gaussian O-stage innovation changes the transition law; Gaussian thermostat/equilibrium claims do not transfer automatically. Boundaries are checked after clone transforms and each BAOAB substep. The second potential gradient is recomputed at post-A positions.

### Extension points

`GasOperators<T>` has batch-level hooks for boundary checks, companion sampling, distance reduction, fitness, historical fitness, clone decisions, clone transforms, noise and kinetics. Override only the relevant hooks; others retain the built-in implementation. Both donor roles call the same interface with different configurations and streams. These hooks are selected once per batch, not per particle.

`AlgorithmicDistance`, `InteractionKernel`, `CompanionSampler`, `NoiseSource`, `RewardSource`, `ObservationExtractor`, `RewardExtractor`, `DerivedFieldProvider` and `DomainAdapter` are reusable interfaces. `KineticOperator::advance_with_noise` also accepts a custom noise source directly. `GradientProvider`, `HessianProvider`, `DriftProvider` and `DiffusionProvider` define derivative/dynamics contracts; only gradients are consumed by a built-in derivative-dependent integrator today. Other providers are consumed through custom kinetic/noise hooks. Automatic differentiation is not required and no AD adapter is implemented yet.

Generate implementation examples from the public Rust API and its checked tests. Generate the implemented API reference with `cargo doc --workspace --no-deps`. Custom providers/operators must have stable, parameter-sensitive IDs and be stateless for replay, or place all persistent per-walker state in the supplied domain snapshots. Generic serialization of custom operator state is not implemented.

## Step, recording and replay

1. Refresh observations; validate/repair boundaries; evaluate raw reward.
2. Draw distance companions, reduce pair distances and compute pre-clone fitness.
3. Draw cloning companions and construct the immutable clone plan.
4. Copy simultaneously, apply transforms, reconcile domain state, classify/repair boundaries, then refresh raw reward.
5. Recheck boundaries; apply eligible kinetics with substep boundary checks.
6. Check final boundaries, refresh final reward, record and commit.

`StepReport` distinguishes pre-clone reward/fitness/companions from the resulting population and final reward. Its reward-evaluation count is submitted scalar rows, including masked rows and explicit extraction; it is not a count of arbitrary internal objective operations or simulator calls. Rendering a landscape does not consume this counter or random streams. `HistorySink` is a caller-driven recording interface; the engine retains its last report and the bounded donor archive.

Addressed RNG keys contain seed/run identity, step, operator role, walker, substep and draw counter. Current streams use RNG schema version 1. Checkpoints are versioned CBOR and preserve dtype, config, population, opaque snapshots, history, lineage and RNG addresses. They require matching execution configuration and provider IDs. Replay on the same supported execution is tested; bitwise agreement between CPU, CUDA and WebGPU is not promised. Compare cross-platform numerical tolerances and distributions instead.

Checkpoints preserve explicit pre-clone eligibility and local-statistics fallback masks. Restore validates the schema, nested report shapes, masks, source identities/generations, clone decisions, counters and population provenance before changing live state. Public tensor deserialization also enforces construction invariants. Population replacement assigns the new population version before evaluating rewards; no eligible walkers means remaining kinetic providers are skipped.

Cancellation is checked at transaction barriers and never exposes a partial population. Work already submitted to a device is not forcibly cancelled. Failed/cancelled calls can still incur counted execution transfers; checkpoints describe committed state, not reversible hardware activity. Browser pause stops at the next one-step batch boundary.

## Browser lab

From the repository root:

```sh
rustup target add --toolchain 1.95.0 wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114 --locked
make algorithmic-gas-web
make algorithmic-gas-lab
```

Open `http://127.0.0.1:8770/euclidean-gas/`. The lab starts paused with 256 walkers, seed 7, 2D Rastrigin and WASM CPU `f32`. It runs in a Web Worker with asynchronous initialization and bounded step batches. Controls expose independent donor laws, multiple distance companions, objective direction, metrics, fitness, boundaries, noise and kinetics. The inspector shows raw rewards, pre-clone fitness, donor identities and clone decisions. The population has three views: a 2D projection, a 3D spatial view of any three coordinates, and a 3D landscape whose height is the asinh-scaled objective over the sampled surface with contour lines. Hidden coordinates are fixed by slice sliders (from the best or the selected walker). Walkers are coloured by raw reward or pre-clone fitness; overlays show distance companions, clone donors and 30-step trails that break at cloning events. Lennard–Jones runs add an atom viewer for the selected walker.

The benchmark list is the engine's objective catalog (`benchmark_catalog()`): Sphere, Quadratic Well, Mexican Hat, Rastrigin, EggHolder, Styblinski–Tang, Rosenbrock, Easom, Holder Table, Lennard–Jones, Constant, Stochastic Gaussian, Mixture of Gaussians, the 24 COCO BBOB functions (dimensions 2, 3, 5, 10, 20, 40; instances 1–1000) and the unit lecture quadratic. Classic objectives are tensor graphs with analytic gradients and run on every backend. BBOB is a pure-Rust port of COCO 2.8.2 that reproduces the reference instances; it is evaluated on the host in `f64` for every backend, its gradient is a central difference costing `2d` evaluations, and the inspector reports both (`host_reward_evaluations`, `host_gradient_evaluations`). Differences from the C++ Optimization Lab: EggHolder and Holder Table gradients are analytic with a finite regularisation at cusps; Lennard–Jones clamps squared pair distances at `1e-4` instead of returning infinity; the lab's `quadratic` (α = 0.1) is `quadratic_well`, while `quadratic` remains the unit well of the lecture experiments; mixture components are seeded only. Stochastic Gaussian noise is addressed by seed, population version, stage and walker, so checkpoints replay it exactly. Reference values are regenerated with `python3 fractal-gas-web/tests/euclidean-gas/generate_objective_goldens.py` after `make optimization-native`.

Configuration/results export as JSON. Checkpoints export as binary `.agc` files or save explicitly in IndexedDB; restore recreates the selected precision/profile. IndexedDB quota/private-mode failures are reported. Plot traces restart at the restored frame and are not checkpoint state. WebGPU requires a secure context (localhost or HTTPS), an available adapter, supported operations and sufficient memory. `f64` selects WASM CPU explicitly; WebGPU never silently narrows it. The current WASM CPU build is single-threaded; a future shared-memory build needs browser isolation headers and a thread-pool adapter.

Both CPU and WebGPU bundles are generated under ignored `engine/` directories; third-party rendering files under ignored `vendor/`. `WASM_BINDGEN=/path/to/wasm-bindgen` overrides the pinned CLI location. `node fractal-gas-web/tools/build-euclidean-gas.mjs --cpu-only` builds only the CPU profile. CI builds both; deployment assembly publishes this lab separately from existing labs.

### Lecture sessions and recorded stages

All 128 Volume II demos use the Rust experiment registry and the same resumable
session as the native runner. Rust initializes populations, executes configured
interventions, records the gas and computes every scientific series. JavaScript
schedules bounded advances and renders the returned values. The worker reuses its
initialized WASM module between experiments.

`LectureExperiment.create(request)` creates a session; `advance(count)` executes
bounded batches; `snapshot()` reads completed measurements. `evidence()` exports
the request, resolved configurations and actual archives. `checkpoint()` and
`LectureExperiment.restore(bytes)` preserve every member of an ensemble and its
recording. `await lecture_analyze(evidence)` recomputes archive measurements and,
where required, the complete-checkpoint continuation protocol.

Recordings retain `pre_clone`, `literal_clone`, `post_transform`, `post_clone`,
BAOAB substages and boundary inputs. Tracing does not draw additional randomness.
Donor and transform sources retain frame, version, slot and generation; intermediate
viscous-force sources resolve to their actual recorded operator-stage events.
An all-ineligible initial population produces an explicit terminal extinction
outcome without fabricating a committed update.

`RunConfig.potential` optionally selects an independent analytic force potential, while
`RunConfig.reward_shift` translates the reward objective by a finite `d`-vector (empty means
zero). An explicit potential supplies force `−∇U` for either reward direction; omitting it
preserves the benchmark's existing objective-direction convention.
For example, `benchmark: "quadratic", potential: "quadratic", reward_shift: [1, 0]`
places the favorable reward at `(1, 0)` while the force remains `−x`. Provider identities
include these choices so checkpoint restoration verifies the experiment configuration.

## Verification and remaining work

```sh
# From this directory:
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo check -p algorithmic-gas-benchmarks --features wgpu --locked
cargo check -p algorithmic-gas-benchmarks --features cuda --locked
cargo check -p algorithmic-gas-wasm --target wasm32-unknown-unknown --features webgpu --locked

# From the repository root, after building WASM:
npm --prefix fractal-gas-web run test:euclidean-gas
make algorithmic-gas-browser-test  # sandbox views; needs `make algorithmic-gas-lab` running
```

Tessellation tests cover the empty-circumsphere property and Euler relation in 2D and 3D, cospherical lattices, degenerate swarms, Voronoi volume closure in clip and periodic boxes, facet reciprocity, flat and conformally flat curvature, Regge curvature of a sphere, serial/parallel bit equality, checkpoint resume of the Einstein–Hilbert gas, and agreement with the Python reference estimators. Contract tests cover shapes, scalar precision, matching laws, separate streams, multi-companion reduction, frozen cloning, historical identities/rescoring, boundary timing, degenerate fitness, cosine zeros, anisotropic covariance, thermostat scaling, extraction, custom hooks, opaque state, replay and failure paths. GPU feature checks verify compilation, not runtime on every vendor. Long-run statistical equivalence, GPU numerical conformance and full performance sweeps remain required before claiming accelerator parity or speedup.

`max_batch_elements` limits individual batches. `max_memory_bytes` (default 512 MiB) adds conservative admission reservations for populations, frozen pools, reports, retained history and inputs; each compute graph checks aggregate node/staging bytes against the remaining allowance. Checked arithmetic rejects overflow. These are engine-owned working-set guards, not an OS/browser/driver memory sandbox: custom-provider allocations and backend-internal workspaces still require deployment limits. See [SAFETY.md](SAFETY.md) for the enforced unsafe-code policy, fuzz/Miri commands and dependency trust boundaries.

Remaining architecture targets include device-resident population/fitness/cloning and reusable scratch buffers, device-only permutation, index-only hybrid shuffle transfers, local/historical input alignment, true multi-donor recombination, cached dependency graphs, capability-restricted numerical views, declarative units/component descriptors, persistent custom-operator serialization, Student/stateful/correlated noise, AD adapters, matrix friction and production simulator adapters. Device loss can terminate a worker; the last explicit checkpoint is the recovery boundary.

Experiment contracts record the configured update, observable and applicable hypotheses alongside the measurements.

### Durable Part V archives and computations

Call `start_recording(RecordingConfig::default())` before stepping. The archive
retains every committed microstep, typed boundary/BAOAB stages, actual clone
probabilities, source identities, force evaluations, innovations and factors.
`recording().graph()` separates CST, IG, IA, material ancestry and restitution
influence. `reconstruct(epoch, step, stage)` validates indexed scalar coverage;
`compare_orders(speed, dt, max_nodes)` compares CST closure with physical light
cones. Population replacement creates an epoch barrier. Recording is bounded
by both its own budget and the engine working-memory limit.

Checkpoints preserve the archive. Standalone `RunArchive::to_bytes/from_bytes`
uses CBOR with validation. JSON is useful for
finite lecture data; CBOR preserves nonfinite invalid observations.

`partv_geometry::analyze` supplies exact 2D conditional fitness derivatives,
spectral metric functions, constant-metric clipped Voronoi cells, Spade 2.15.1
Delaunay maintenance, refining variable-metric distances, and shared tetrahedral
spacetime partitions. The registered Part V session constructs geometry from recorded walkers, donor
relations and events; its transport and curvature calculations use that recorded
context.

The benchmark `RunConfig.physics_metric` option accepts
`{epsilon, temperature, policy, curvature, clipping_threshold}`. It evaluates the frozen conditional fitness
field at the actual O-stage position, keeping the force potential separate.
The provider supports general-dimensional BAOAB, smooth global or local
normalization, logistic maps, and one distance companion. Covered historical
distance donors resolve by immutable source identity. Historical cloning requires
the supported global standardization configuration.
Already-dead rows have unavailable field coverage; revived rows explicitly use
their donor's frozen field. Unsupported configurations return capability errors.


## Registered lecture experiments

The single catalog is `crates/algorithmic-gas/src/lecture_experiments.json`:
10 Part I, eight Part II, eight Part III, 16 Part IV, 20 Part V and 66 Part VI
experiments. `algorithmic_gas_benchmarks::lecture::LectureSession` is their shared
execution path. Each configured run supplies its observations; standalone model
simulations are not lecture data sources.

Part VI uses three execution patterns:

- Archive measurements compute fields, correlations, geometry, path carriers,
  graph quantities and mechanical ledgers from executed transitions.
- VI-15, VI-30 and VI-31 use complete paired runs for parity, translation and
  local intervention. Random addresses are shared within each pair; different
  pairs use independent seeds.
- VI-19, VI-22 and VI-45 execute independent continuations of complete checkpoints,
  retaining donor memory, provider context and the future schedule.

Use the common CLI from this directory:

```sh
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- catalog
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  VI-18 7 96 --output /tmp/gas-noise-evidence.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  analyze /tmp/gas-noise-evidence.json --output /tmp/gas-noise-analysis.json
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  stress IV-07 0 8 --output /tmp/gas-derivative-stress.json
```

For an explicit request, save this JSON as `/tmp/gas-request.json`:

```json
{
  "id": "VI-18",
  "seed": 7,
  "steps": 96,
  "parameters": {"engine_memory": 2}
}
```

```sh
cargo run --release -p algorithmic-gas-benchmarks --bin gas-lecture -- \
  run /tmp/gas-request.json --output /tmp/gas-evidence.json
```

`run` and `analyze` also accept `-` to read JSON from standard input. A run result
contains a rendered snapshot and its evidence; `analyze` accepts that complete
bundle or the evidence object. Required protocol durations can determine per-run
step budgets, including matched-physical-time refinement. The result records the
actual counts. `gas-physics` and `algorithmic-gas-qft` expose this same CLI.

`stress ID START COUNT` executes a reproducible slice of the registry's generated
control cases and reports every completed or failed case. The process fails if
any selected case fails. Scientific fits can instead return an explicit
inconclusive outcome with their measurements and diagnostics intact.

The browser route is `/euclidean-gas/lecture/?demo=VI-18`. Native and browser
results share the same Rust calculations, result schema and derivation contracts.
Imported evidence is validated and analyzed again. Analytical comparisons are
identified as predictions of the stated observable; a fitted physical hypothesis
retains its measured residual and interpretation limits.

[QFT_VALIDATION.md](QFT_VALIDATION.md) records the validated identities, measured
prediction failures and current test coverage. The
[Part VI contracts](crates/algorithmic-gas/src/physics/partvi_contracts.json) state
what each experiment measures. The [lecture guide](../docs/source/2_fractal_gas/partvi_experiments.md)
provides the chapter context.
