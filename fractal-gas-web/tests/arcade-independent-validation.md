# Independent Arcade playback validation

## Configuration and reproduction

Build `fractal_gas`, `retro_shim`, and `arcade_playback`, then serve the project
with COOP/COEP headers. These checks use local NES, Breakout, and Sonic test ROMs.
Run from `fractal-gas-web`:

```sh
npm run test:arcade-resources
npm run test:arcade-independent-browser
ARCADE_BROWSER=firefox npm run test:arcade-independent-browser
npm run test:arcade-trajectory-browser
ARCADE_BROWSER=firefox npm run test:arcade-trajectory-browser
npm run test:arcade-browser
ARCADE_TEST_URL=http://127.0.0.1:8096/web/ ARCADE_PLAYBACK=1 ARCADE_TIMEOUT_MS=1800000 npm run test:arcade-resources-browser
FG_ROM="$PWD/web/test-rom.nes" build/fg_tests
```

The independent/player browser tests default to port 8096; set `ARCADE_TEST_URL`
to use another server. The resource and existing planner suites retain their
8091 default. Browser versions: Chromium 145.0.7632.6 and Firefox 146.0.1.
Measurements were taken on an i9-13900H desktop with 64 GiB RAM during other
desktop activity; timings are workload observations, not throughput guarantees.

## Correctness and lifecycle checks

- The native suite passes 80 tests, including owned Graph recordings after
  source reset, frozen ancestry, terminal transitions, planner prefixes,
  bounded reconstruction, and recording-size refusal.
- Fifteen JavaScript tests cover resource partitioning, search scheduling,
  replacement/reset cleanup, stale messages and errors, failure isolation,
  and retry without sending search pause commands.
- Frame comparisons capture the chosen walker's frame at the exact export
  boundary using test-only instrumentation. Playback seeks to the end, rewinds,
  and seeks again while the original population continues changing.
- Paired runs compare the first 12 search-update traces at identical seeds and
  worker counts, including planner phases and committed frames.
- The browser limit case deliberately supplies an oversized recording, checks
  isolated refusal, then retries a valid capture and verifies unchanged search
  traces and exact replay frames.
- The UI suite checks independent play/pause, preserved recordings on search
  Start, explicit replacement, reset cleanup, and desktop/mobile layout.

Both Chromium and Firefox passed all 12 console/algorithm frame and search-trace
comparisons, all four UI modes, and the oversized-recording recovery case. The
existing Arcade planner browser suite also passed. All three Release WASM
targets built successfully.

## Memory accounting

The selected combined WebAssembly budget reserves 256 MiB for a lazy playback
module. Its initial allocation is 64 MiB. The remaining search partition retains
the 4 GiB main-engine ceiling. With 20 Sonic workers and an 8 GiB budget, each
search emulator has a 192 MiB maximum; playback does not consume a search slot.
Recording payloads are bounded to 32 MiB and export scratch is checked before
allocation. JavaScript recording storage and browser/graphics overhead remain
outside the WebAssembly budget.

The large-run matrix uses 1,000 Sonic walkers, 20 search workers, seed 7,
Coords/RAM/Gray/RGB, dt 1–2, two elites, and planner horizon 4 / maximum 8.
Each case captures a fixed path after its first update and repeatedly replays
and rewinds it in a separate emulator during search. Wave/Graph run 100 updates;
FMC/Jump Wave finish three complete planning/execution cycles. Reported peaks
include playback memory, and `playbackFrames` confirms concurrent player activity.

## Large-run results (2026-09-24)

All 16 cases passed. Coords/RAM/Gray ran sequentially; RGB ran in a separate
concurrent browser process. The duplicate RGB portion of the first process was
stopped after its 12 completed cases. Every case acknowledged search disposal
and terminated its playback worker. Raw measurements are in
`arcade-independent-results.json`.

| Observation | Algorithm | Updates | Completed cycles | Replay frames | Peak GiB | ms/update |
|---|---|---:|---:|---:|---:|---:|
| Coords | Wave | 100 | 0 | 2155 | 3.343 | 3543.79 |
| Coords | Graph | 100 | 0 | 283 | 3.343 | 458.76 |
| Coords | FMC | 16 | 3 | 279 | 3.344 | 3491.66 |
| Coords | Jump Wave | 28 | 3 | 626 | 3.344 | 4226.14 |
| RAM | Wave | 100 | 0 | 3005 | 3.817 | 5166.07 |
| RAM | Graph | 100 | 0 | 677 | 3.825 | 1166.18 |
| RAM | FMC | 16 | 3 | 333 | 3.848 | 4265.76 |
| RAM | Jump Wave | 28 | 3 | 702 | 3.848 | 4613.93 |
| Gray | Wave | 100 | 0 | 3059 | 3.864 | 5226.90 |
| Gray | Graph | 100 | 0 | 786 | 3.863 | 1535.21 |
| Gray | FMC | 16 | 3 | 177 | 3.895 | 1792.97 |
| Gray | Jump Wave | 28 | 3 | 228 | 3.895 | 1210.88 |
| RGB | Wave | 100 | 0 | 3987 | 4.934 | 6903.43 |
| RGB | Graph | 100 | 0 | 1623 | 4.945 | 2652.87 |
| RGB | FMC | 16 | 3 | 560 | 4.965 | 7494.63 |
| RGB | Jump Wave | 28 | 3 | 1106 | 4.965 | 8338.50 |

Maximum combined allocation: 5,331,222,528 bytes
(4.965 GiB), including the independent player.
These short-horizon tests do not guarantee unlimited retained history or
long-horizon throughput.
