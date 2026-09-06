# Native batch measurements

Measured 2026-09-06 on an Intel Core i9-13900H (20 logical CPUs), Linux x86-64,
GCC 15.2, Release build, without fast-math. See `native.csv` for the full matrix.
The machine was shared with browser tests and other processes; these are
throughput observations, not latency guarantees or isolated CPU comparisons.

For 1024 worlds with 256 controlled bodies per world:

| Threads | Worlds stepped / second | Body frames / second |
| --- | ---: | ---: |
| 1 | 1,183 | 302,719 |
| 2 | 2,318 | 593,378 |
| 4 | 3,633 | 930,132 |
| 8 | 5,712 | 1,462,270 |

Each row has 8,224 payload bytes and an 8,256-byte aligned stride. Simultaneous
gather measured 1.85–2.66 GB/s across those cases, counting payload bytes copied
(not read+write traffic). Checksummed serialization measured 104–164 MB/s.
Physics stepping includes fused gather, thrust, integration, broad phase and
four substeps; this sparse scene has no sustained collision load and is not
representative of dense cave contacts or full FMC search.

Allocation instrumentation measured one allocation per physics batch at both
16 and 256 worlds, with either one or four threads. This is the thread-pool
callback, independent of population size; the physics loop allocates no
per-world objects. Optional tree recording and fitness operators have separate
allocation costs.

Regenerate the matrix from the repository root:

```sh
make control-native
fractal-gas-web/build-control-native/control/fg_control_benchmark > native.csv
```
