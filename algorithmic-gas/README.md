# Algorithmic Gas

An independent Rust implementation of the Volume 2 [architecture specification](../docs/source/2_fractal_gas/architecture/01_algorithmic_gas.md). It does not modify or wrap the Python Atari, robotics or Euclidean engines, or the C++ Optimization Lab.

This is an initial, executable implementation, not completion of every performance and extension target in the specification. CPU and WASM CPU support both `f32` and `f64`. Accelerator adapters use **host-orchestrated Burn batches**: objective/gradient evaluation, pair reductions, noise factors and kinetic arithmetic can execute on the selected device; sampling decisions, fitness statistics/maps, cloning and random-number generation currently run on the host. Transfers are explicit and counted. There is no automatic CPU fallback or precision reduction.

## Build and run

Requirements: Rust 1.95.0 (pinned in `rust-toolchain.toml`), and Node 22 for the browser build. Run Cargo commands from this directory.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --release -p algorithmic-gas-benchmarks -- --precision f64 --walkers 256 --steps 100
```

The runner accepts `--benchmark sphere|rastrigin|rosenbrock|styblinski_tang`, `--dimensions`, `--seed`, `--backend`, and `--config FILE`. Arguments apply in order; place `--config` before overrides. Explicit file boundary settings are preserved. Output is JSON with resolved configuration, initialization and step timings, raw reward summaries, submitted reward-row count, synchronization count and transfer bytes. Initialization/compilation must be reported separately from steady-state performance.

```sh
cargo run --release -p algorithmic-gas-benchmarks --features wgpu -- --backend wgpu --precision f32
cargo run --release -p algorithmic-gas-benchmarks --features cuda -- --backend cuda --precision f64
```

These profiles require a compatible adapter/driver. Compilation alone does not verify GPU execution. The WGPU adapter rejects software CPU adapters. CUDA `f64` must pass the device's dtype and primitive-operation checks; it is not promised to be fast.

## Project layout

| Path | Responsibility |
|---|---|
| `crates/algorithmic-gas` | Public data contracts, selection, fitness, cloning, kinetics, noise, private Burn adapters, transactional engine and checkpoints |
| `crates/benchmarks` | Analytic objectives/gradients, initialization, native runner and end-to-end tests |
| `crates/wasm` | Asynchronous browser bindings and precision-preserving binary checkpoints |
| `../fractal-gas-web/web/euclidean-gas` | Independent worker, controls, objective plot, convergence plot and walker inspector |

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
| Sampling law | Independent directed draws; uniform Fisher–Yates mutual matching; sequential Gaussian greedy matching; legacy permutation |
| Diversity reduction | Mean, weighted mean, minimum, maximum of scalar pair comparisons |

Distance companions are dense masked `[N,K]` pool indices, default `K=1`. Multiple mutual rounds produce multiple companions; each round is a matching, not one globally disjoint multi-edge graph. Cloning currently requires `K=1`. `ClonePlan` records a list of weighted sources per recipient so future recombination can extend the representation, but more than one contributing donor is explicitly rejected today.

Without replacement, `insufficient: "use_available"` is the default: remaining capacity is masked when fewer candidates survive. `"reject"` selects a strict error. Uniform replacement draws use direct ranks; uniform sampling without replacement uses a sparse partial shuffle. Mutual Fisher–Yates uses one precomputed slot lookup, so its single-round sampling work is linear.

Sources are resolved by frame, slot, generation and population version. They come from the current eligible population or retained past snapshots. **Ancestry never restricts default selection.** Historical cloning reevaluates source rewards/diversity under current global statistics, rather than reading a current slot's fitness. Initial historical cloning does not support local standardization or external per-slot input batches; those combinations fail explicitly pending source-aligned input/statistics adapters. Historical mutual matching is also unsupported; use independent sampling.

Gaussian independent sampling streams tiles without an `N × M` matrix. Uniform independent draws and CPU/WASM Fisher–Yates avoid quadratic comparisons. Greedy matching remains sequential; a GPU tensor backend does not make Fisher–Yates or greedy dependency chains parallel. Uniform independent sampling and uniform mutual matching are distinct joint laws.

### Fitness, boundaries and cloning

Fitness separates measurement, standardization, positive mapping and combination. Built-ins include global alive-only regularized population statistics, local kernel-weighted statistics, explicit legacy sample statistics, logistic and legacy asymmetric maps, additive positivity floors, independent reward/diversity exponents, and a distance measurement floor. The distance floor applies **after** multi-companion reduction as `sqrt(reduced² + floor²)`.

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

The spec's signatures are architecture sketches, not copy-paste declarations of this initial API. Generate the implemented API reference with `cargo doc --workspace --no-deps`. Custom providers/operators must have stable, parameter-sensitive IDs and be stateless for replay, or place all persistent per-walker state in the supplied domain snapshots. Generic serialization of custom operator state is not implemented.

## Step, recording and replay

1. Refresh observations; validate/repair boundaries; evaluate raw reward.
2. Draw distance companions, reduce pair distances and compute pre-clone fitness.
3. Draw cloning companions and construct the immutable clone plan.
4. Copy simultaneously, apply transforms, reconcile domain state, classify/repair boundaries, then refresh raw reward.
5. Recheck boundaries; apply eligible kinetics with substep boundary checks.
6. Check final boundaries, refresh final reward, record and commit.

`StepReport` distinguishes pre-clone reward/fitness/companions from the resulting population and final reward. Its reward-evaluation count is submitted scalar rows, including masked rows and explicit extraction; it is not a count of arbitrary internal objective operations or simulator calls. Rendering a landscape does not consume this counter or random streams. `HistorySink` is a caller-driven recording interface; the engine retains its last report and the bounded donor archive.

Addressed RNG keys contain seed/run identity, step, operator role, walker, substep and draw counter. Current streams use RNG schema version 1. Checkpoints are versioned CBOR and preserve dtype, config, population, opaque snapshots, history, lineage and RNG addresses. They require matching execution configuration and provider IDs. Replay on the same supported execution is tested; bitwise agreement between CPU, CUDA and WebGPU is not promised. Compare cross-platform numerical tolerances and distributions instead.

Checkpoint format **2** and built-in operator identity **v2** include corrected sampling/measurement semantics, explicit pre-clone eligibility, and local-statistics fallback masks. Version-1 checkpoints are rejected rather than silently migrated: start a new run from its configuration. Restore validates nested report shapes, masks, source identities/generations, clone decisions, counters and population provenance before changing live state. Public tensor deserialization also enforces construction invariants. Replacement assigns the new version before evaluating rewards; no eligible walkers means remaining kinetic providers are skipped.

Cancellation is checked at transaction barriers and never exposes a partial population. Work already submitted to a device is not forcibly cancelled. Failed/cancelled calls can still incur counted execution transfers; checkpoints describe committed state, not reversible hardware activity. Browser pause stops at the next one-step batch boundary.

## Browser lab

From the repository root:

```sh
rustup target add --toolchain 1.95.0 wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114 --locked
make algorithmic-gas-web
make algorithmic-gas-lab
```

Open `http://127.0.0.1:8770/euclidean-gas/`. The lab starts paused with 256 walkers, seed 7, 2D Rastrigin and WASM CPU `f32`. It runs in a Web Worker with asynchronous initialization and bounded step batches. Controls expose independent donor laws, multiple distance companions, objective direction, metrics, fitness, boundaries, noise and kinetics. The inspector shows raw rewards, pre-clone fitness, donor identities and clone decisions. A 2D projection supports higher-dimensional populations; 3D rendering is not implemented.

Configuration/results export as JSON. Checkpoints export as binary `.agc` files or save explicitly in IndexedDB; restore recreates the selected precision/profile. IndexedDB quota/private-mode failures are reported. Plot traces restart at the restored frame and are not checkpoint state. WebGPU requires a secure context (localhost or HTTPS), an available adapter, supported operations and sufficient memory. `f64` selects WASM CPU explicitly; WebGPU never silently narrows it. The current WASM CPU build is single-threaded; a future shared-memory build needs browser isolation headers and a thread-pool adapter.

Both CPU and WebGPU bundles are generated under ignored `engine/` directories; third-party rendering files under ignored `vendor/`. `WASM_BINDGEN=/path/to/wasm-bindgen` overrides the pinned CLI location. `node fractal-gas-web/tools/build-euclidean-gas.mjs --cpu-only` builds only the CPU profile. CI builds both; deployment assembly publishes this lab separately from existing labs.

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
```

Contract tests cover shapes, scalar precision, matching laws, separate streams, multi-companion reduction, frozen cloning, historical identities/rescoring, boundary timing, degenerate fitness, cosine zeros, anisotropic covariance, thermostat scaling, extraction, custom hooks, opaque state, replay and failure paths. GPU feature checks verify compilation, not runtime on every vendor. Long-run statistical equivalence, GPU numerical conformance and full performance sweeps remain required before claiming accelerator parity or speedup.

`max_batch_elements` limits individual batches. `max_memory_bytes` (default 512 MiB) adds conservative admission reservations for populations, frozen pools, reports, retained history and inputs; each compute graph checks aggregate node/staging bytes against the remaining allowance. Checked arithmetic rejects overflow. These are engine-owned working-set guards, not an OS/browser/driver memory sandbox: custom-provider allocations and backend-internal workspaces still require deployment limits. See [SAFETY.md](SAFETY.md) for the enforced unsafe-code policy, fuzz/Miri commands and dependency trust boundaries.

Remaining architecture targets include device-resident population/fitness/cloning and reusable scratch buffers, device-only permutation, index-only hybrid shuffle transfers, local/historical input alignment, true multi-donor recombination, cached dependency graphs, capability-restricted numerical views, declarative units/component descriptors, persistent custom-operator serialization, Student/stateful/correlated noise, AD adapters, matrix friction and production simulator adapters. Device loss can terminate a worker; the last explicit checkpoint is the recovery boundary.

No convergence or equilibrium result follows merely from choosing modules with familiar names. Apply the book's theory only after checking the exact configuration and hypotheses.
