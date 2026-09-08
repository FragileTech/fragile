# Fractal consolidation: validation and performance

Measured on 2026-09-08. The shared engines pass the correctness checks below. Median paired performance changes pass the 5% threshold in every measured native and WASM workload. Native absolute-time parity remains inconclusive under the host’s fluctuating background load: two Lab workloads have separate timing medians above that threshold despite faster median paired execution. These results cover the specified workloads on this machine, not every environment size or browser.

## Method

Baseline: `d91d23d0a522d28dc4650229f8bc80889b1d52b4` (`Optimize physics engine in lab`), captured before source changes and rebuilt from its saved source. Candidate: the shared Fractal implementation in this change. Both native baselines and both standalone WASM baselines were rebuilt; measurements do not rely on stale native archives.

Machine: 13th Gen Intel(R) Core(TM) i9-13900H, Linux x86_64. Native: GCC 15.2, C++17; libraries use Release `-O3 -DNDEBUG -ffp-contract=off`, and both harnesses use `-O3 -ffp-contract=off`. WASM: Emscripten 6.0.8, Release `-O3 -DNDEBUG -ffp-contract=off -fexceptions -msimd128`, single thread, 32 MiB initial memory with growth enabled; Node 22.22.1. Baseline and candidate use identical build flags. Native Lab also includes four-thread runs.

Each workload has eight warmup operations followed by forty timed operations, repeated nine times. Each CSV reports the median of those nine runs. Native measurements use seven adjacent baseline/candidate pairs per workload, alternating execution order; single-thread workloads are pinned to performance CPU 4, and four-thread workloads to CPUs 4, 6, 8, and 10. WASM measurements use three alternating complete-process pairs without affinity pinning. Build jobs and browser checks were stopped for the comparisons. Separate performance investigations improved scratch reuse in Wave, Graph, Euclidean Gas, and history pruning before these final measurements.

Native unrestricted-affinity runs exposed substantial timing variance; those measurements are retained in `performance/fractal-unification/native-unpinned/`. The host had a load average around 15 when inspected, and fixed-core timings still fluctuated. The final tables show both independently aggregated timing medians and the median of the within-pair percentage changes. The paired statistic compares adjacent runs under closer load conditions; it need not equal the ratio of the two separate medians. We expanded native sampling from three to seven pairs to investigate the apparent Lab regressions. All paired medians improve, but the raw-median increases are shown rather than discarded. A quiet-machine native run remains necessary to establish absolute-time parity without this qualification.

Native allocation counts/bytes instrument ordinary C++ `new`/`new[]`, including standard-library containers. They do not intercept every allocator (for example aligned state-bank allocation). Persistent banks and the harness measurement buffers are allocated before the timed iterations. RSS is the process high-water mark, including the harness and allocator retention (and earlier workloads for the WASM sequences); it is not an isolated core-memory measurement. WASM reports linear-memory reservation separately from Node process RSS.

Recording modes in workload names: `0` off, `1` pruned, `2` full. Native Wave uses 256 three-float opaque mock snapshots and four elites. Lab uses 256 packed one-body worlds, one frame per action, and four elites; `-4t` uses four execution slots. Graph starts at 256, permits 512, and steps only selected leaves. Arcade FMC/Jump measurements time incremental search/execution operations with 256 mock walkers and horizon 8/max 16. Lab cycle measurements complete a search at 128 walkers, horizon 8/max 16. Optimization uses a cheap eight-dimensional sphere objective with 128 walkers; Euclidean uses uniform companions, periodic bounds, and either enabled or disabled potential forces. WASM optimization measurements call the native step directly; Lab cycle measurements include the controller result adapter.

## Native results

Times are microseconds per operation; negative changes mean faster. A cycle is an entire search, while the other rows measure an iteration or incremental planner advance.

| Workload | Baseline median µs | Shared median µs | Raw median change | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| `lab-0` | 1,571.5 | 1,720.9 | +9.5% | -6.6% |
| `wave-0` | 406.3 | 253.2 | -37.7% | -39.3% |
| `lab-1` | 1,728.8 | 1,689.0 | -2.3% | -8.7% |
| `wave-1` | 697.5 | 495.8 | -28.9% | -28.9% |
| `lab-2` | 1,467.2 | 1,549.4 | +5.6% | -3.4% |
| `wave-2` | 834.9 | 465.6 | -44.2% | -36.8% |
| `lab-0-4t` | 1,571.4 | 1,427.3 | -9.2% | -7.6% |
| `lab-1-4t` | 2,069.2 | 2,040.5 | -1.4% | -17.8% |
| `lab-2-4t` | 1,879.5 | 1,626.2 | -13.5% | -6.7% |
| `graph` | 481.8 | 396.6 | -17.7% | -19.3% |
| `fmc` | 995.3 | 613.1 | -38.4% | -37.8% |
| `jump` | 370.0 | 280.0 | -24.3% | -37.0% |
| `lab-fmc-cycle` | 8,007.1 | 7,959.6 | -0.6% | -6.3% |
| `euclidean` | 1,216.7 | 1,109.8 | -8.8% | -6.1% |
| `euclidean-no-force` | 1,210.5 | 1,133.1 | -6.4% | -5.7% |

## WASM results

| Workload | Baseline median µs | Shared median µs | Raw median change | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| `lab-0` | 3,111.2 | 2,483.5 | -20.2% | -20.4% |
| `lab-1` | 3,693.0 | 2,794.8 | -24.3% | -22.6% |
| `lab-2` | 3,929.7 | 2,469.3 | -37.2% | -37.2% |
| `lab-fmc-cycle` | 13,799.2 | 10,993.2 | -20.3% | -20.3% |
| `lab-wave-jump-cycle` | 36,033.8 | 22,708.5 | -37.0% | -37.0% |
| `optimization-wave` | 3,271.9 | 2,541.6 | -22.3% | -22.3% |
| `optimization-graph` | 2,006.6 | 1,354.2 | -32.5% | -25.2% |
| `optimization-fmc` | 3,201.8 | 2,459.7 | -23.2% | -15.0% |
| `optimization-wave_jump` | 3,191.2 | 2,752.5 | -13.7% | -15.8% |
| `optimization-euclidean` | 2,264.1 | 2,128.2 | -6.0% | -6.0% |
| `optimization-euclidean-no-force` | 2,017.0 | 1,923.0 | -4.7% | -4.7% |

## Allocations, recording growth, and memory

| Native workload | Allocations/op before → after | Allocated bytes/op before → after |
| --- | ---: | ---: |
| `lab-0` | 23.00 → 1.00 | 15,792.0 → 72.0 |
| `lab-1` | 550.02 → 257.27 | 32,007.2 → 4,528.5 |
| `lab-2` | 279.30 → 257.30 | 63,483.4 → 47,763.4 |
| `wave-0` | 569.00 → 0.00 | 55,652.0 → 0.0 |
| `wave-1` | 1,095.15 → 257.20 | 95,242.8 → 29,988.3 |
| `wave-2` | 828.30 → 256.30 | 105,407.0 → 47,691.4 |
| `graph` | 238.47 → 3.40 | 28,825.7 → 40.8 |
| `lab-fmc-cycle` | 2,339.03 → 1,045.03 | 143,566.0 → 17,181.6 |
| `euclidean` | 1,235.65 → 768.02 | 86,955.0 → 49,152.8 |
| `euclidean-no-force` | 725.90 → 256.02 | 54,211.0 → 16,384.8 |

The dedicated whole-Wave allocation check reports **one ordinary allocation per complete Lab iteration**, unchanged between 16 and 256 walkers and between one and four threads. It includes non-cumulative fitness, four elites, action inheritance, and physics, after twelve warmup iterations. The physics-only bound also passes. See [allocation output](performance/fractal-unification/allocations.txt).

Recording-off Wave reuses population scratch; fractional counts in the other workloads reflect occasional storage growth. Recorded modes still allocate history nodes and grow the retained history. The pruned recorder reuses its pin/leaf scratch; node insertion and retained ancestry remain recording costs. Euclidean core temporaries are reused. In the sphere adapter, the remaining ~256 ordinary allocations without forces come from two temporary coordinate vectors per benchmark evaluation; enabling forces adds four gradient vectors per walker. Benchmark implementations remain outside the algorithm consolidation.

Median process high-water marks (taking the maximum across workloads in each pair) were 14,244 → 14,244 KiB for native and 101,812 → 96,880 KiB for WASM/Node. Both WASM modules remained at 33,554,432 bytes of linear memory. The packed engine now accounts for both elite banks and all population metadata in its fixed working-memory estimate; recorder growth and backend kernel scratch are separate.

| Standalone WASM artifact | Baseline bytes | Shared bytes | Change |
| --- | ---: | ---: | ---: |
| control | 452,808 | 470,982 | +4.0% |
| optimization | 1,133,603 | 1,109,252 | -2.1% |

## Correctness and integration

- Native Control: 39 cases plus the physics/whole-Wave allocation executable passed. Native optimization: 33 cases; shared swarm: 47 cases. The Arcade/NES/visit suite passed 69 cases. Existing mathematical fixtures were retained; expected trajectories were not regenerated to hide differences.
- New shared-core tests compare packed float-action and opaque integer-action environments with injected draws, assert distinct fitness/clone sampling, donor-cycle safety, root-action inheritance, complete elite metadata and stable ties, pruned elite ancestry, recoverable death/truncation/actual duration, Graph sparse stepping/growth, and early planner finish.
- JavaScript: 339 Lab tests and 22 optimization tests passed, including native/WASM streams, deterministic reset/replay, proposal freezing, evaluation budgets, and standard comparison algorithms. Python: all 21 control-binding tests passed, including new native Jump Wave results, mid-search/pruned/full checkpoint continuation, finished-plan stability, and rejection of old versions without mutation.
- Native and WASM build targets passed, including the Lab pthread module and full Arcade module. Core headers compile independently of application headers. Ruff and `git diff --check` passed.
- The complete Lab Jump Wave browser test passed with Chromium software rendering after correcting the default benchmark-target input validity. Its real worker assertions cover complete trajectory execution, real-time search freeze, pause, step, and checkpoints during both search and execution.
- Arcade browser worker assertions passed for NES Mario, Atari Breakout, Atari Montezuma, and Genesis Sonic, each with FMC, shared-prefix Jump, and full-path Jump. They verify reset replay, committed-state isolation during search, pause/resume, and cancellation of a long search. These are the existing worker assertions run independently of the UI-heavy first portion of the browser script.
- Optimization Chromium browser checks passed for algorithms, landscape/spatial views, high-dimensional slices, molecule views, save/load/replay, reset, and mobile layout.

## Remaining validation limits

- Native absolute-time parity is not certified: separate timing medians show +9.5% for recording-off Lab and +5.6% for full-recording Lab under high, variable background load. Seven fixed-core paired comparisons instead show median improvements of 6.6% and 3.4%, respectively. Both statistics and the unrestricted-affinity measurements are retained. Repeat on a quiet machine before treating native performance parity as unconditional.
- The complete Arcade UI script timed out trying to click Pause; its separate real-ROM worker assertions all passed. The optimization Firefox run did not finish in this environment and was stopped; Xvfb is unavailable here. Full Firefox and Arcade UI end-to-end validation remains for the supported browser CI environment.
- Performance gates cover native mock snapshots, packed Lab, and standalone optimization/Lab WASM. They do not provide an independent real-ROM Arcade WASM throughput/artifact-size baseline, a threaded-WASM timing baseline, or a sweep over large scenes and all objective dimensions. Native four-thread performance and native thread determinism/allocation bounds were checked. Recorded growth remains workload-dependent.
- Same-build checkpoint continuation is a Lab computation-checkpoint feature. Arcade and optimization keep their existing recorded playback/reset interfaces; no new computation-checkpoint menu/API was added to them. Historical seeds are not expected to reproduce the pre-refactor trajectories. Whole-site documentation and the unrelated Python research suites were not rebuilt/run for this native refactor.

## Reproduce and inspect

The native harness is [fractal_benchmark.cpp](fractal_benchmark.cpp); compile it with each version’s headers and libraries using the same flags documented above. Run each workload in adjacent baseline/candidate pairs, alternating order. These native results use seven pairs with `taskset -c 4` (or `taskset -c 4,6,8,10` for the `-4t` workloads); choose equivalent available cores on another machine. For the candidate, link `fg_control_core`, `fg_swarm_core`, `fg_fractal_core`, `fg_optimization_core`, `fg_coco`, and `fg_libcmaes` in a linker group plus `-pthread`. The baseline instead links `fg_numeric_core`. An optional first argument selects one workload, such as `lab-0-4t`.

The [WASM harness](fractal-wasm-benchmark.mjs) runs unchanged against either version’s `web` directory:

```sh
node fractal-gas-web/tests/fractal-wasm-benchmark.mjs /absolute/path/to/version/web
# Optional final argument selects one workload:
node fractal-gas-web/tests/fractal-wasm-benchmark.mjs /absolute/path/to/version/web lab-wave-jump-cycle
```

Raw paired CSVs (seven native pairs, three WASM pairs), the unrestricted-affinity investigation, per-run artifact sizes, aggregate and paired summaries, and build/artifact hashes are in [performance/fractal-unification](performance/fractal-unification/builds.json). The shared interfaces and dependency direction are described in [the core README](../src/fractal/README.md).
