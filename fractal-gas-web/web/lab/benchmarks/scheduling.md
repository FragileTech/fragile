# Dynamic scheduling of physics futures

Control Lab workers claim small contiguous batches of worlds from an atomic
queue. The caller participates in the same queue. With 128 walkers and eight
threads, each claim contains two worlds. A worker that finishes inexpensive
worlds can immediately take more work instead of waiting for another worker's
collision-heavy partition. The batch barrier still completes before planner
selection and cloning proceed.

Each worker retains exclusive scratch storage. Each world retains its own
state, random stream, and result index; planner sampling and reductions keep
their existing ordering. Results accumulate locally before being written back
to reduce concurrent writes to neighboring result slots. Single-thread runs
use a direct loop. Other backends retain static scheduling because some
emulators depend on stable worker assignment.

## Reproduction

Preserve the pre-change `control.mjs`, `control.wasm`, `control-threaded.mjs`,
and `control-threaded.wasm` in a separate directory, rebuild both Control Lab
WASM targets, then run from the repository root:

```sh
node fractal-gas-web/tools/control-scheduling-benchmark.mjs /path/to/static-engine /tmp/scheduling.json
```

The paired benchmark alternates execution order and checks exact equality of
physics snapshots, rewards, selected leaves, and selected trajectories. Fixed
workloads step 128 worlds for 12 frames each, mixing initial stock harvesting
states with states settled for 180 frames. The settled worlds are deliberately
clustered to expose static partition imbalance. Lethal walls are disabled only
for these fixed workloads. Their timings are medians of five samples after
warm-up.

Complete stock flight-harvesting searches retain 128 walkers, horizon 64, and
12 action frames, including the existing Wave Jump horizon-extension behavior.
Seeds 7, 11, and 19 follow a full-search warm-up. Both workloads run with one and
eight threads. Exact state comparisons prevent trajectory differences from
being mistaken for scheduling gains.

## Measured results

On 2026-09-08, the completed eight-thread comparison produced the following
results. The static baseline already includes the collision optimizations.
The host was shared and timings varied substantially; these are observed
workload-dependent gains, not latency guarantees. The single-thread comparison
was interrupted during the search phase and is not included in this table.

| Workload | Static (ms) | Dynamic (ms) | Static / dynamic |
|---|---:|---:|---:|
| Fixed: 0 of 128 settled worlds | 26.86 | 42.86 | 0.63× |
| Fixed: 16 of 128 settled worlds | 173.68 | 62.42 | 2.78× |
| Fixed: 128 of 128 settled worlds | 300.48 | 263.76 | 1.14× |
| Planner: seed 7 | 20557.41 | 19756.37 | 1.04× |
| Planner: seed 11 | 5531.70 | 1924.48 | 2.87× |
| Planner: seed 19 | 11867.55 | 9784.98 | 1.21× |

The deliberately imbalanced fixed workload improved 2.78×. Median speedup
across the three complete searches was 1.21×; the individual search ratios
ranged from 1.04× to 2.87×. The light fixed workload regressed in this run,
so these measurements do not establish an across-the-board improvement.
Dynamic scheduling addresses uneven worker loads; it does not reduce the
collision work within each future.

All paired snapshots, step results, final planner states, selected rewards,
and selected trajectories matched exactly. Native tests also compare one and
eight threads across mixed frame counts, duplicate parent rows, inactive
worlds, and repeated planning generations, and exercise retained harvesting
cargo and persistent hooks. Native allocation checks remain passing.

Raw measurements are in [scheduling.json](scheduling.json). Set
`CONTROL_BENCH_THREADS=8` to reproduce only the completed configuration.
