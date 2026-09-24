# Arcade resource validation

## Reproducing the checks

Build `fractal_gas` and `retro_shim`, serve the project with `serve.py 8091`, and
provide the existing local test ROMs. The browser server must set COOP/COEP.
Run from `fractal-gas-web`:

```sh
npm run test:arcade-resources
npm run test:arcade-browser
npm run test:arcade-memory-browser
ARCADE_BROWSER=firefox npm run test:arcade-memory-browser
npm run test:arcade-growth-browser
npm run test:arcade-resources-browser
FG_ROM="$PWD/web/test-rom.nes" build/fg_tests
```

The full matrix uses 1,000 Sonic walkers, 20 emulator workers and an 8 GiB
combined WebAssembly budget. It runs Coords, RAM, Gray and RGB observations:
100 updates each for Wave and Graph, and three **completed** planning/execution
cycles each for FMC and Jump Wave. The short workload uses seed 7, dt 1–2,
two elites, horizon 4 and maximum horizon 8. These are memory and lifecycle
checks; they do not estimate long-horizon planning performance or guarantee
arbitrarily long retained history. Graph has a separate growth/recovery test,
and native tests verify shared-prefix compaction and retained ancestry.

`ARCADE_MODES`, `ARCADE_ALGORITHMS`, `ARCADE_REPORT`, `ARCADE_TEST_URL`, and
`ARCADE_BROWSER` configure the matrix runner. `ARCADE_TIMEOUT_MS` overrides the
15-minute per-case test watchdog for loaded desktops; it does not alter engine
worker failure timeouts. Observation IDs are RAM=0,
RGB=1, Gray=2, Coords=3; algorithm IDs are Wave=0, Graph=1, FMC=2, Jump Wave=3.
The memory runner checks imported memory ceilings, actual main-heap growth
past 2 GiB, read/write access and real NES pthread stepping above that address,
emulator memory growth and its ceiling, and repeated large/small Sonic resets.

## Small-run baseline (2026-09-24)

Sonic Wave, Coords, seed 7, dt 6–30, two elites; three trials of 30 updates,
timing the last 20. Baseline and candidate use identical worker counts.
Measurements were taken on an Intel Core i9-13900H Linux desktop with 64 GiB
RAM. Desktop activity and browser compilation affect timing.

| Walkers | Workers | Baseline median ms/update | Candidate median ms/update |
|---:|---:|---:|---:|
| 2 | 2 | 96.34 | 58.50 |
| 48 | 8 | 822.48 | 768.11 |

All six paired traces matched exactly for iteration, total frames, mean reward,
maximum reward and clone count. Neither size showed a repeatable regression
above 10%. The main heap stayed at 512 MiB for small runs; a two-walker Sonic
run creates two emulator instances even when 20 workers are selected.
Compact evidence is in `arcade-small-run-results.json`.

## Limits being measured

Allocated WebAssembly memory is the current main heap plus every emulator heap,
not process RSS. JavaScript, browser, graphics and compiler overhead are outside
the selected budget. Main memory is capped at 4 GiB; each emulator has its own
page-aligned share, capped at 2 GiB. The 8 GiB selection with 20 Sonic workers
assigns 4 GiB to the main engine and 214,695,936 bytes to each emulator.

The Graph cap budgets two snapshot/observation populations, elite buffers,
communication regions, runtime/transient headroom and 128 MiB of frozen history.
Oversized initial populations are rejected. Frozen history stops at its limit
without silently discarding ancestry. Current observations reserve their final
safe capacity near half the Graph cap to avoid a late contiguous reallocation;
small populations retain small allocations.

## Boundary, failure and lifecycle results

Passed in Chrome for Testing 145.0.7632.6 and Firefox 146.0.1:

- An imported 512 MiB main-memory ceiling rejected a 600 MiB allocation.
- A deliberately exhausted heap returned `std::bad_alloc` from threaded NES
  stepping without leaving the coordinator waiting at its barrier.
- Main memory grew beyond 2 GiB. A subsequent allocation above address 2 GiB
  was readable/writable, and two NES execution slots stepped successfully with
  their state allocations above that address.
- An emulator module grew from 64 MiB to 93.44 MiB and refused an allocation
  that would exceed its supplied 128 MiB maximum.
- Sonic repeatedly switched through 1,000, 2, 48 and 2 walkers, then reset.
  Both two-walker runs used two emulator instances and a 512 MiB main heap.

The Chromium resource UI test also verified saved selections, rapid consecutive
changes, and recovery after refusing both an impossible startup budget and an
oversized population. RGB Graph then grew from 1,000 to its calculated cap of
1,030 and completed 100 updates. Its combined WebAssembly peak was
5,294,391,296 bytes (4.93 GiB), including a 3.68 GiB main heap. This check caught
and fixed a reset buffer swap that had lost the pre-reserved observation capacity.

The native core suite passed 75 tests, including donor order/repetition,
immutable sources, pool allocation errors, bounded prefix compaction and history
limit refusal. The nine-test native Atari suite passed with both Breakout and
Montezuma ROMs (each ROM covers its applicable cases); all nine native Retro
checks passed, including Sonic. The resource/worker JavaScript suites passed all
ten tests, and the existing Arcade UI/planner browser suite passed. The combined
trajectory integration subsequently passed 79 native core tests.

Firefox required a longer cold-boot watchdog under concurrent desktop load:
boot has a bounded 120-second wait; normal emulator jobs retain 30 seconds.
The timeout reports the worker and operation. Throughput remains hardware-
and workload-dependent.

## Large-run matrix (2026-09-24)

The following Chromium measurements use the workload above. Other desktop tests
and builds ran concurrently, so timings are observations rather than performance
guarantees. Peak values count allocated WebAssembly pages.

| Observation | Algorithm | Updates | Completed cycles | Peak GiB | ms/update |
|---|---|---:|---:|---:|---:|
| RAM | Wave | 100 | 0 | 3.754 | 6527.25 |
| RAM | Graph | 100 | 0 | 3.762 | 1046.15 |
| RAM | FMC | 16 | 3 | 3.786 | 4203.89 |
| RAM | Jump Wave | 28 | 3 | 3.786 | 4699.51 |
| RGB | Wave | 100 | 0 | 4.871 | 4691.11 |
| RGB | Graph | 100 | 0 | 4.883 | 1875.00 |
| RGB | FMC | 16 | 3 | 4.903 | 4081.31 |
| RGB | Jump Wave | 28 | 3 | 4.903 | 4383.84 |
| Gray | Wave | 100 | 0 | 3.801 | 4902.12 |
| Gray | Graph | 100 | 0 | 3.800 | 1228.78 |
| Gray | FMC | 16 | 3 | 3.833 | 5485.69 |
| Gray | Jump Wave | 28 | 3 | 3.833 | 5469.37 |
| Coords | Wave | 100 | 0 | 3.280 | 4419.22 |
| Coords | Graph | 100 | 0 | 3.280 | 636.18 |
| Coords | FMC | 16 | 3 | 3.282 | 3119.47 |
| Coords | Jump Wave | 28 | 3 | 3.282 | 12778.81 |

All 16 cases passed. Raw measurements are in `arcade-large-run-results.json`.
The matrix process was interrupted after 15 cases; the final Coords/Jump Wave
case was rerun separately against the final combined trajectory/resource build.

## Final build and integration

The final Release WebAssembly build passed for `fractal_gas` and `retro_shim`.
Its output matched the isolated combined resource/trajectory build byte for byte.
An extra RGB Wave recording run hit the 15-minute test watchdog while other
checks and compilation were active. It was rerun with a 30-minute test allowance
after those jobs finished; the desktop remained under substantial external CPU
load. This watchdog is separate from emulator operation/failure timeouts.

The final combined RGB Wave run passed all 100 updates and acknowledged
clean disposal. Peak combined allocation was 5,230,362,624 bytes
(4.871 GiB); main allocation peaked at
3.621 GiB. Mean update time was
6394.25 ms on this loaded desktop. Evidence is in
`arcade-combined-rgb-results.json`.
