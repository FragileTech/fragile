# Shared native Fractal library

Arcade, Control, Optimization, and LLM Lab link `fg_fractal_core`. Algorithm templates are specialized in their C++ adapters; no browser, scene, emulator, benchmark, or application JSON type is required by this directory. The old numerical include paths forward here for source compatibility. They contain no alternate implementation.

| Core | Responsibility | Consumers |
| --- | --- | --- |
| `wave.hpp` | Elite restoration, fitness, donor decisions, action generation, transitions, metadata, history, elite retention | `FractalGas` for Arcade and optimization; `PackedWave` for Lab |
| `planner.hpp` | Incremental FMC and Jump Wave, root selection, ancestry consensus, horizon extension, fallbacks | `ArcadePlanner` for Arcade and optimization; Lab `FmcPlanner` |
| `graph.hpp` | Graph's distinct leaf masks, frozen parents, cloning exclusions, sparse transitions, visits, growth | Arcade and optimization `FractalTree` |
| `euclidean.hpp` | Phase-space companions, fitness, cloning/revival, BAOAB, periodic geometry, evaluation accounting | Optimization's `EuclideanAdapter` |
| `population.hpp` | Contiguous metadata and typed actions, C++17 array views, row-copy primitives | Wave and backends |
| `distance.*` | Observation distance dispatch, validation, serial/threaded batch kernels | Wave and Graph in all adapters |
| `checkpoint.hpp`, `exploration_tree.*` | Backend-independent serialization primitives and recorded action ancestry | Wave and planners |

The comparison algorithms (CMA-ES, iCEM, MPPI, GAS 2017) and Python/Torch research algorithms keep their own implementations. Graph remains absent from Control Lab's menu. Euclidean dynamics require coordinate/velocity storage and are not applied to emulator snapshots.

## Observation distances

`fg::DistanceMetric` serializes through `parse_distance_metric` and `distance_metric_name`
as `"l2"` or `"cosine"`. Missing configuration fields mean L2. `row_distance(a, b, d, metric)`
compares a single pair; `companion_distances_into(observations, companions, n, d, pool, out, metric)`
compares flat `[n, d]` rows to their indexed companions and reuses `out`. Pass `nullptr`
for serial execution. Inputs must be finite and the output must not alias observations.
The original L2 arithmetic and `l2_norm_companions[_into]` wrappers are retained.

Cosine computes `clamp(1 - dot(a,b)/(norm(a)*norm(b)), 0, 2)`; two zero vectors have
distance 0 and exactly one zero vector has distance 1. Unknown metric names fail explicitly.
Add metrics and dispatch in `distance.*`; fitness normalization and lifecycles need no edits.
Wave's cloning operator and Graph configuration carry this choice; FMC and Jump Wave use
their underlying Wave setting. Physical Euclidean phase-space geometry remains separate.

## Token environment

`llm/environment.hpp` implements snapshot `BatchEnv` using an injected batch transport.
Snapshots contain immutable record IDs, token counts, cumulative log probability and stop
status; the browser archive owns prefixes, token bytes and embeddings. Duration is a requested
new-token count, and actions identify sampled continuations within the fixed run configuration.
The worker caches realized transitions for replay. Rewards are changes in total or mean log
likelihood, so they telescope across variable-sized chunks and cloning copies the whole state.

`BatchEnv::best_candidate` defaults to true for existing backends. Token states exclude empty
roots and unused slots. LLM Wave disables historical elite reinjection; completed and deepest
partial candidates remain available in its separate trace archive. `FG_LLM_ONLY` builds a
standalone Asyncify module: one native operation awaits concurrent worker requests. Errors
invalidate the engine; the last committed population remains available for inspection. This
transport contract can support future native planners without a second algorithm lifecycle.

## Backend contract

`Wave<State, Backend, ActionPolicy>` owns the current/output metadata and two elite banks. `State::states` is backend-owned storage: opaque snapshots for emulators and optimization's existing seed-action environment, or an aligned `StateBatch` for Lab. `Backend` supplies resizing, row copying, observation extraction, transition execution, optional visit rewards, and presentation-only pose/flags. `ActionPolicy` samples typed actions and durations from the immutable donor population.

`transition(input, Selection, actions, durations, output)` never modifies input. `Selection` contains a batch size and optional source/destination index views; an empty view means identity. Action and duration rows follow selection order. Outputs include step rewards, terminal/truncated/recoverable flags and actual executed durations. Graph gathers selected leaves into a reusable compact batch and commits only those destinations after all donors have been read. Lab passes the donor indices directly to `Physics::step`; it allocates no state object per walker. Existing worker pools, precision, SIMD kernels, and host scheduling remain in place.

Fitness companions and clone companions are separate draws. Wave fitness retains asymmetric distance/reward rescaling, cumulative versus step-reward selection, coefficients, and optional visit bonuses. All-dead recovery revives only deaths explicitly marked recoverable, excluding truncation. Recorded edges contain actual executed durations; planners filter zero-duration edges.

Elite selection sorts descending cumulative reward. Equal rewards prefer prior elites in their existing order, then current population index. Each elite carries its state, observations, both rewards, actions/root actions, durations, terminal metadata, fitness, and lineage. The second elite bank makes replacement safe even when donor indices form cycles. Non-cumulative fitness therefore sees the elite's own last step reward.

## Planning and host responsibilities

`Planner<Action>::begin`, `advance`, and `finish` share all search decisions. Each `advance` completes one whole Wave iteration before deciding whether to return a plan. `finish` selects from completed work without another iteration. A result contains typed action rows, actual path durations, the selected leaf, and an execution reason; the planner exposes search depth.

FMC has two named root selection policies: surviving discrete first-action voting for Arcade/optimization, and whole-population root-action averaging for Lab. Jump Wave uses surviving-lineage consensus, optional full-path execution, maximum-horizon extension, stable best-leaf ties, and all-dead or horizon fallback. Lab's JavaScript controller only starts/advances the native planner and formats the returned trajectory. Workers still own deadlines, cancellation between iterations, and live simulation timing. Optimization keeps its reproducible seed-action representation and freezes adaptive proposal geometry throughout a planning cycle; updates occur at complete search/population boundaries.

## Checkpoints and replay

Control Lab checkpoints use version **2**, with a native/WASM backend tag, scene fingerprint, configuration, physical state, shared Wave metadata and elites, RNG, action policy settings, recorded/pruned lineage, and incremental planner progress. Restore validates a temporary engine before replacing the live state. Version 1 checkpoints are rejected; there is no migration or legacy execution path. State snapshots keep their existing physical-state ABI.

Optimization exports identify the new engine as `fgopt-4`. Recorded visualization data can still be viewed, while an older engine identifier is not a promise of identical re-simulation. Historical seeded trajectories can change with the unified lifecycle, stable elite ties, complete elite metadata, and truncation bookkeeping. The new version supports deterministic continuation within the same backend/build. Native versus WASM physics comparisons retain their existing numerical tolerance.

## Validation

The existing mathematical fixture suites remain unchanged. New shared-core fixtures compare packed and opaque environments with injected independent draws, verify donor cycles/root inheritance, elite ties and metadata, sparse Graph execution/growth, recoverable deaths/truncation/actual duration, and early planner finish. Native allocation instrumentation covers the entire Lab Wave lifecycle with recording off at 16/256 walkers and 1/4 threads, in addition to the physics-only allocation check. Python and JavaScript integration tests exercise native planning and checkpoint continuation.

The reproducible performance harnesses are `tests/fractal_benchmark.cpp` and `tests/fractal-wasm-benchmark.mjs`. The native harness links against either source version; the WASM harness accepts either version's absolute `web` directory. Each workload uses eight warmup operations and nine repeats of forty operations. Recording growth is reported separately from reusable engine scratch; fixed working-memory estimates exclude recorder growth and backend kernel scratch. Measured results and remaining environment limitations are recorded in [the performance report](../../tests/fractal-performance.md).
