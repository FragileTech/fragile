# fractal-gas-web

The [LLM Lab guide](../docs/source/project/llm_laboratory.md) covers parallel token decoding
with the shared Wave and Graph algorithms. Run `make llm-lab` from the repository root and
open `http://127.0.0.1:8080/llm/`. It builds a standalone Asyncify engine, uses OpenRouter
generation/log probabilities and embeddings, and keeps the API key in browser memory.
Defaults are DeepSeek V4 Flash, text-embedding-3-small, and cosine observation distance.
Other core consumers continue to default to L2. `make llm-test` runs offline native,
WebAssembly, and browser tests. With an explicitly authorized local key,
`npm --prefix fractal-gas-web run test:llm-live` tests actual continuation and cloning
against a running lab (`LLM_LAB_URL`, default `http://127.0.0.1:8876/llm/`).

The new [continuous-control laboratory](web/lab/README.md) runs a custom C++
physics engine in the browser and Python. It supports packed state batches,
joint continuous actions, Wave/FMC, extensible agent types, editable scenes,
complete world-motion replay and exploration history.
From the repository root, run `make control-native`, `make control-web`,
then `make control-lab` and open
`http://127.0.0.1:8080/lab/`.

`make control-web` reuses an active Emscripten SDK or installs version 6.0.8 in
`.cache/emsdk/6.0.8` automatically (internet required on first use). The build
activates the SDK without changing your shell startup files. To prepare it
separately, run `make control-setup`; for another installation, use
`EMSDK_DIR=/path/to/emsdk make control-web`. Both serial and threaded engines
are built. This follows the official [Emscripten SDK installation workflow](https://emscripten.org/docs/getting_started/downloads.html).

Run `make serve` to build and preview the documentation at
`http://127.0.0.1:8000/docs/`, or `make docs-serve` to preview an existing build.
The same server serves the built simulator at `/lab/`; `make control-lab` also
serves built documentation at `http://127.0.0.1:8080/docs/`.

Start with the [published laboratory user guide](https://fragiletech.github.io/fragile/docs/lab/)
for walkthroughs of the controls, scene editor, replay, experiments, and extension APIs.

A faithful C++ port of the fractal gas algorithm
(`src/fragile/fractalai/fractal_gas.py` + `videogames/cloning.py` +
`videogames/kinetic.py`) driving the [nes-py](https://github.com/FragileTech/nes-py)
C++ NES emulator core, compiled both natively and to WebAssembly. The web demo
runs a swarm of walkers playing Super Mario Bros entirely in the browser.

## Layout

- `src/` — the algorithm port: `tensor_ops` (asymmetric_rescale, l2_norm,
  random_alive_compas), `cloning` (FractalCloningOperator), `kinetic`
  (RandomActionOperator), `walker_state`, `fractal_gas` (the step loop +
  elite buffer), `nes_env` (per-thread emulator pool + Mario RAM reward),
  `thread_pool`, `atari_env` (ALE emulator pool) + `montezuma_logic`
  (Montezuma's Revenge RAM map, pyramid layout, coords tuple and shaped
  reward, header-only), `wasm_bindings` (embind surface).
- `third_party/nes-py/` — git submodule; its C++ core is compiled directly by
  our CMake (the Python/SCons parts are unused).
- `swarm_algorithm` (the interface both algorithms implement),
  `fractal_tree` + `visit_grid` (the "Graph" tree algorithm, see
  "Algorithms").
- `tests/` — dependency-free test harness; the key tests replay recorded
  runs of the Python reference implementations draw-for-draw (wave:
  `tests/fixtures/generate_fixtures.py`; graph:
  `tests/fixtures/generate_tree_fixtures.py` on the vendored cb9f3296 code).
- `native/main.cpp` — CLI to run the gas on a ROM and measure throughput.
- `web/` — the frontend (no dependencies): sidebar controls, canvas of the
  best walker, live plots (reward, virtual reward, clone %, alive, dt),
  mirroring the `make videogames` Panel dashboard. For Mario, a "Level map
  — swarm" panel draws the full level (`web/maps/mario-W-S.gif`, 1 map px =
  1 world px, from ian-albert.com) with every walker's RAM x/y overlaid as
  a dot (per-walker `walkerX/Y/World/Stage/Alive` arrays in the step stats)
  and the best walker highlighted — the whole gas spreading through the
  level at a glance. Sonic builds its map from walker tiles (fog of war)
  and Montezuma shows the temple pyramid with each room's picture captured
  as the swarm discovers it (see "Consoles").

## Algorithms

The demo runs two swarm algorithms behind one interface
(`src/swarm_algorithm.hpp`), selected by the sidebar's **Algorithm** toggle
(also `fg_cli --algo wave|graph`):

- **Wave** (default) — `fg::FractalGas`, the fractal gas: a fixed population
  of N walkers, every walker steps each iteration, low-fitness walkers clone
  onto high-fitness ones, elite buffer. Reference: `src/fragile/fractalai/`
  (`fractal_gas.py`, `videogames/cloning.py`, `videogames/kinetic.py`).
- **Graph** — `fg::FractalTree`, the tree variant used by the old Montezuma
  demo, ported line by line from git commit **cb9f3296**
  (`src/fragile/core.py` FractalTree, `videogames.py` MontezumaTree,
  `fractalai.py`, `actions.py`), vendored verbatim under
  `tests/fixtures/reference_cb9f3296/`. Every visited state stays as a node:
  per iteration, companions are drawn among the alive walkers, the virtual
  reward is `relativize(distance)^dist_coef * relativize(cum_reward, leaf
  mean/std)^reward_coef * other`, the clone decision is
  `(vr[compa] - vr) / vr > rand`, walkers chosen as clone sources are
  protected, dead walkers always clone, and only the **leaves** that clone
  copy their companion's state, record it as their parent and step the
  environment; interior nodes are frozen and the best walker never clones.
  The population grows so that at least `min_leafs` leaves exist (fresh
  nodes are dead children of the root until they clone), capped at
  `max_walkers`. `other` is the visit-count reward of `MontezumaTree`:
  a float32 grid per plane (Montezuma room, Mario world/stage, Sonic
  zone/act) counting the cells walkers stand in, `+1` once per distinct cell
  per batch, decayed by `erase_coef` and clipped to `[0, 1000]` every
  update, summed over 5x5 blocks and relativized with leaf statistics —
  active only in **Coords** observation mode on games with a map (generic
  Atari games and RAM/RGB/Gray modes use `other = 1`). The UI's Walkers
  input becomes "start walkers = min leaves", Elite is hidden, and Max
  walkers / Erase coef appear. Maps draw the graph: thin lines join each
  node to its parent, small squares are interior nodes, dots are leaves.
  The map panel's **visits** button (Graph + Coords) shows the visit-count
  reward like the old demo did: the level in greyscale with the 5x5-block
  visit sums (exported by `VisitGrid::export_blocks`, `getVisitBlocks` in
  the bindings) overlaid as a "fire" colormap at alpha 0.7, auto-ranged
  to the largest displayed block, never-visited blocks transparent, the
  walkers on top; `web/autotest-visits.html` checks the export. The
  pooling window is the **Visit pooling (px)** input (default 5, the
  reference's block size): counts are stored per pixel in 32x32 tiles and
  only summed over `B x B` blocks when the reward and the heatmap read
  them, so the size changes live mid-run without losing history (1 =
  per-pixel novelty, 160 = one cell per Montezuma room). The reference
  required grid sizes divisible by the block; edge blocks may be partial here.
  The **Visit reward** On/Off switch (same section, `visitReward` in the
  bindings, live) is the ablation control: Off sets the term to 1 so cloning
  is driven by distance and cumulative reward alone, while visits keep being
  counted, so the heatmap stays available and switching back on keeps the
  history (the reference's `count_visits=False` disabled both). The same
  switch, pooling size, erase coefficient and heatmap are available to the
  **Wave** as well (`FractalGasParams::visit_reward`, default OFF so the
  default Wave is the plain fractal gas): when on, the wave multiplies
  `relativize(-block_sum)` over all walkers into its virtual reward, the
  per-walker visit keys travelling with the walkers through cloning and
  elite injection (`WalkerState::infos`). The **Visit coef** slider in the
  Fitness section is the exponent of the term for both algorithms
  (`visit_coef`, live; 1 = the reference's plain product, 0 = term off).

  RNG draw order per iteration (replayable, `FractalTreeSampler`):
  companions for the distance term, companions for the clone term, the
  uniforms, then the actions and dt of the stepped walkers (actions first;
  dt is uniform in `[dt_min, dt_max]` inclusive, the reference's
  `UniformDtSampler(1, 5)` being `{1..4}`). `reset()` draws the start
  actions before the env reset, then the dt, and steps every walker once;
  walker 0 keeps the reset state as the root (dead, like the reference)
  but takes the stepped observation.

  Documented deviations from the reference: `total_steps` counts the
  walkers that really stepped (the reference also counts the preallocated
  `will_clone` flags of freshly added slots), `max_walkers` is a hard cap
  (the reference preallocates that many rows and would index past them),
  and a walker that gathered an empty state (only when every walker is
  dead, where the reference crashes on a `None` state) is dropped from the
  batch. Fidelity is checked by `tests/test_fractal_tree.cpp`, which
  replays draws recorded from the historical code by
  `tests/fixtures/generate_tree_fixtures.py` (a plain run and a
  visit-counting run) and requires identical parents, masks, growth,
  rewards, virtual rewards, visit rewards and final visit cells;
  `tests/test_visit_grid.cpp` checks the sparse visit grid bit-for-bit
  against a dense float32 transcription. `web/autotest-graph.html` is the
  browser smoke test.

**Env frames**: the "Env frames" stat is an exact count. Every env reports
how many frames it really emulated per walker step (`frames_stepped`; NES,
ALE and the Genesis shim stop a step early on death / life loss, the shim
returning the count through its shared-memory header), and both algorithms
accumulate it (`total_frames`), so the number only ever grows — the old
`total_steps x mean dt` estimate wobbled with each iteration's random dt.

## Fidelity notes

The algorithm is translated line-for-line, including: Bessel-corrected std in
`asymmetric_rescale`, with-replacement companion sampling when walkers are
dead, the second independent companion draw for the cloning decision, forced
cloning of dead walkers, gather-semantics cloning, and the elite top-k buffer.
Bitwise compatibility with torch's RNG is impossible, so equivalence is
proven by fixture tests that replay recorded Python RNG draws through the C++
implementation (`tests/fixtures/generate_fixtures.py`). One deliberate
deviation: elite top-k ties are broken by lower index (torch.topk tie order
is arbitrary).

The Mario reward deviates from gym-super-mario-bros in two documented,
env-side ways (the algorithm is untouched): grabbing the flagpole pays a
one-time +500 bonus outside the ±15 frame clip, and `flag_get` does not end
the episode — otherwise finishing a level would look like death to the gas
(done walkers are forced to clone away) and the swarm would avoid the pole.
Walkers play on through the castle walk into the next level.

Parallelism: all RNG draws happen on the caller thread; the `std::thread`
pool (wasm: pthreads) parallelizes emulator stepping (one emulator instance
per thread) and, for large observation batches, the L2 distance rows of the
fitness phase (RNG-free, so still deterministic per seed regardless of
thread count). The distance kernel uses four independent accumulators so
the compiler vectorizes it (`-msimd128` on wasm). Walkers travel as `dump_state()` blobs with the Mario reward
carry (`x_last`, `time_last`) appended, so cloning a walker is a byte copy.

## Build (native)

```bash
git submodule update --init fractal-gas-web/third_party/nes-py
cd fractal-gas-web
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fg_tests                       # unit + replay tests
FG_ROM=third_party/nes-py/nes_py/tests/games/super-mario-bros-1.nes \
  ./build/fg_tests                     # + NES smoke tests
./build/fg_cli --rom <smb.nes> --n 64 --iters 200 --seed 7 --cumulative --elite 2
```

To regenerate the fixtures after changing the Python reference (run from the
repo root): `uv run python fractal-gas-web/tests/fixtures/generate_fixtures.py`.

## Arcade FMC and Jump Wave

The arcade algorithm selector offers **Wave** (the default), **Graph**, **FMC**,
and **Jump Wave** for NES, Atari (including Montezuma), and Genesis. Wave and
Graph display the leading search walker. FMC and Jump Wave display one committed
game, while the plots and maps continue to show the planning population.

FMC searches ahead and votes on the inherited first discrete action of surviving
walkers. Action ties use the lowest action ID; the chosen duration comes from
the highest-reward supporting walker, with ties resolved by walker index. It
plays that action and searches again from the resulting emulator snapshot.

Jump Wave defaults to **Stop at first bifurcation**: play the exact ancestral
path shared by surviving walkers. If there is no executable shared prefix,
continue searching up to the maximum horizon, then play one action from the
best surviving path. Turn the checkbox off to play the full winning path.
Both modes select one first-edge fallback when every search walker dies, and
stop only when the committed game ends. Recoverable life loss discards any
remaining queued trajectory and replans from the post-life-loss snapshot.

**Search horizon** defaults to 32 iterations (1–4096). Jump Wave's **Maximum
search horizon** defaults to 0, meaning twice the normal horizon, capped at
4096. An explicit maximum must be at least the normal horizon. Population,
fitness, elite, visit, and frame-skip controls also apply to the new solvers.
Changing live settings discards the pending plan and replans from the committed
game. Visit history persists across replanning. Action ancestry uses pruned,
bounded storage without recording pixel observations.

**Pause** preserves search and trajectory progress. **Reset** and algorithm
changes discard it. Played score and played frames describe committed gameplay;
search depth, population rewards and Env frames describe planning. The played
score uses the emulator's display score, or cumulative reward when there is no
separate display score. Each worker turn performs one search iteration or one
trajectory edge, and replay uses actual emulated durations, including shortened
terminal edges.

The WASM algorithm IDs are 0=Wave, 1=Graph, 2=FMC, 3=Jump Wave. `FgParams` adds
`horizon`, `consensusPrefix`, and `maxHorizon`; worker callers can omit these and
receive the defaults above. Direct embind callers must supply all three fields.
Planner step messages add `phase`, `searchDepth`, `searchAdvanced`,
`committedScore`, `committedReward`, `playedFrames`, `gameDone`, and
`executionMode`, plus committed world/level/lives when available. The worker
emits `gameDone` for a completed committed game rather than stopping when the
planning population dies.

After building WASM and running `python3 serve.py 8091`, run
`npm run test:arcade-browser` from this directory. The test uses local Mario,
Breakout, Montezuma and Sonic ROM fixtures; set `ARCADE_TEST_URL` to use another
server. Native planner tests are included in `fg_tests`.

## Build (WebAssembly)

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html):

```bash
source <emsdk>/emsdk_env.sh
cd fractal-gas-web
emcmake cmake -B build-wasm
cmake --build build-wasm -j            # outputs web/fractal_gas.{js,wasm}
```

## Consoles

The web demo runs four game setups end-to-end in wasm, selectable in the
top bar:

- **NES / Super Mario Bros** (nes-py core, pthread-parallel) — the default.
- **Atari 2600** (ALE core, pthread-parallel; rewards/termination come from
  ALE's built-in per-game handlers). All ALE-supported games are bundled
  under `web/roms/atari/` (copied from `ale_py/roms`, gitignored) and picked
  from a dropdown; the default is Ms. Pac-Man.
- **Sonic The Hedgehog** (Sega Genesis via Genesis Plus GX). Reward =
  x-progress with a glitch guard plus delta-shaped bonuses (all env-side,
  see the kSonic* constants in src/retro_game_logic.hpp): +5000 one-time
  act-completion bonus (Mario flagpole-style, so the signpost beats any
  wandering and the tally screen isn't a reward valley), +500 per newly
  visited 64px map cell (a per-walker visited bitmask carried in the state
  blob — clones inherit it — cleared on act change; pays for the
  backtracking detours acts like Marble 1 require), signed ring deltas
  x3 (collection pays; a hit dumps all rings, an automatic damage penalty
  that makes shields/invincibility valuable), score deltas x0.5 (boss hits,
  badniks, monitors, end-of-act time+ring tally), and +1000 per gained
  life. Done on any death, showcase ranked by zone/act/x (RAM map from
  stable-retro's SonicTheHedgehog-Genesis-v0 integration). The sidebar's Zone/Act selects start the run in any level,
  Mario-style: the boot sequence pokes Sonic 1's v_zone/v_act bytes
  ($FFFE10/$FFFE11) every frame until gameplay is detected, winning the
  race against the START handler's GHZ1 write. The ROM is copyrighted and
  not committed (`web/sonic.rom`, gitignored). Airstriker remains in the
  engine as a freeware test game for the native suite but is not exposed in
  the UI.

  Genesis architecture notes: libretro cores are global-state singletons,
  so isolation comes from separate instantiations. NATIVE: one dlopen'd
  file-copy of the core .so per thread (private globals per copy),
  pthread-parallel. WASM: emscripten's dynamic linking faults under
  pthreads, so instead the core is statically linked into a small shim
  module (`web/retro_shim.{js,wasm}`, src/retro_shim.cpp) and each of N
  plain Web Workers instantiates its OWN copy — separate memories, true
  isolation, true parallelism. RetroFarmEnv (src/retro_farm_env.cpp)
  dispatches per-walker step jobs to the workers through shared memory +
  Atomics; worker.js pre-spawns the farm before fg.init (nested workers
  need a live JS event loop, which blocked C++ does not have). Games BOOT
  from power-on to gameplay (retro_boot_to_gameplay) instead of loading
  stable-retro's savestates — those are host-ABI-specific byte layouts
  (GPGX serializes whole structs containing pointers) and cannot be
  restored on wasm32; walker blobs are unaffected (written and read by the
  same build). The per-game math lives in src/retro_game_logic.hpp, shared
  by the native env and the shim.

wasm prerequisites beyond emsdk: `embuilder build zlib` once (ALE and the
Genesis core need the zlib port).


**Sonic fog-of-war map**: the "Level map — swarm" panel for Sonic is BUILT
by the swarm itself. Each walker step ships a 40x28 RGB tile (its 320x224
frame downsampled 8x) plus the camera position (Sonic 1 RAM v_screenposx/y
@ $FFF700/$FFF704); the UI stitches tiles into a per-level canvas at 1/8
scale, so the level image emerges as the gas explores and unexplored areas
stay dark. Same resizable viewport and dot overlay as the Mario map.

- **Montezuma's Revenge** (the Atari/ALE core with dedicated game logic,
  `src/montezuma_logic.hpp`; the C++ side is Atari console 1 with
  `game = 1`). The RAM map: room `ram[3]`, x `ram[42]`, y `ram[43]`
  (grows upward), level `ram[57]`, lives `ram[58]`, inventory bitmask
  `ram[65]`, death timer `ram[55]`. Its Coords tuple is
  `[global_x, global_y, room, x, y, level, inventory, lives]` where
  `global_x/y` place the walker on the 9x4 room grid of the temple
  (`cell * 160 + in-room pixel`), so walkers in adjacent rooms are a
  room-width apart in distance space instead of one unit. Reward = ALE's
  score delta plus a one-off **new-room bonus** (default +500) the first
  time a walker's lineage enters one of the 24 rooms — a per-walker
  bitmask in the state blob (`AtariCarry`, clones inherit it, reset on the
  next temple level), not paid during the death animation; both terms are
  live sliders. Death = ALE life loss (recoverable when lives remain, like
  every Atari game) or entering room 8, the level-1 dead end (a hard death,
  plangym's `death_room_8`). The **pyramid map** panel is BUILT by the swarm like
  Sonic's: the first time any walker stands in a `(level, room)` the worker
  renders that walker's frame (`renderWalkerFrame`), crops the 50-row HUD
  to the 160x160 room image and ships it; the UI pastes it on the room's
  cell (unexplored rooms show their number) and overlays every walker at
  its in-room position — the projection `(x + 3, 263 - y)` was calibrated
  against Panama Joe's face pixels. `web/autotest-montezuma.html` is the
  headless smoke test.

**ROM vault — one password for every game**: the ROMs are copyrighted, so
the deployment ships only password-encrypted blobs under `web/roms-enc/`
(committed): `mario.nes.enc`, `sonic.rom.enc`, `atari/<game>.bin.enc` for
the whole Atari picker (Montezuma included) and a `manifest.json`. Build
them from the plaintext ROMs next to the page (all gitignored:
`web/test-rom.nes`, `web/sonic.rom`, `web/roms/atari/*.bin`) with

```bash
FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs --check web/roms-enc/sonic.rom.enc  # reuse the current password?
FG_ROM_PASSWORD='...' node tools/encrypt-rom.mjs --all
```

Blob layout: `salt(16) | iv(12) | AES-256-GCM ciphertext+tag`, key =
PBKDF2-SHA256(password, salt, 250k); `--all` uses ONE salt for the whole
vault so the browser derives the key once. In the browser each ROM is
loaded plaintext-first (local dev needs no password), then from the
IndexedDB cache, then decrypted with the password, which is asked once and
remembered (memory + IndexedDB) so every other game and later visits unlock
silently; Sonic additionally accepts a one-time upload of your own ROM.

## Run the demo

```bash
python serve.py            # COOP/COEP headers (required for wasm threads)
# or, from the repo root:  make web   (WEB_PORT=nnnn to change the port)
# open http://localhost:8000/web/arcade.html
```

`web/autotest.html` is a headless smoke test of the wasm pipeline: copy a ROM
to `web/test-rom.nes`, start `serve.py`, and open the page (or run headless
Chrome against it) — it initializes the swarm, runs 10 iterations, and
reports `OK ...` on the page and as a beacon request in the server log.

**Observation modes** (sidebar toggles; also `fg_cli --obs ram|rgb|gray|coords`):
walker distances can be measured on the 2KB CPU RAM (default), the RGB
screen (184320 dims), the grayscale screen (61440 dims), or a fast Coords
tuple `[x, y_pixel, y_viewport, world, stage, time, h_vel, v_vel, sub_area,
power_up]` read straight from RAM. Changing the mode restarts the run (the
observation shape is part of the walker state). Sonic's Coords tuple is
game-specific (see src/retro_game_logic.hpp): `[x, y, x_vel, y_vel,
ground_speed, zone*3+act]`, rescaled to comparable magnitudes — position and
momentum only; score/rings/lives are deliberately excluded so the distance
metric measures state diversity, not reward.

**Start level**: the sidebar's World/Stage selects (CLI: `--world 1-8
--stage 1-4`) start the run in any SMB level, via the gym-super-mario-bros
RAM trick — the world/stage/area bytes are written while pressing START
during boot. The sub-area mapping matters: worlds 1, 2, 4 and 7 have an
extra intro sub-area, so stages >= 2 there use `area = stage + 1`.

The demo loads the ROM from `web/test-rom.nes` — it never leaves the
browser. The nes-py submodule ships test ROMs under
`third_party/nes-py/nes_py/tests/games/`. Any plain `http.server` will NOT
work: SharedArrayBuffer needs the cross-origin-isolation headers serve.py
sets.

The Lab’s **Vehicle count** control selects 1–128 vehicles in every environment, including racing tracks. Changing it restarts the scene paused and clears replay history. Added vehicles use the scene’s existing vehicle types and clear starting positions; cargo and obstacles remain in place. Each environment remembers its selected count for the session.

### LLM analysis

`/llm/` opens Generation. Analysis shares the run and selection and draws the
recorded tree (Wave) or actual population slots (Graph) on Canvas. Playback,
metric coloring, branch comparison and PNG export work offline. Imported
recordings open Analysis at their final step. The engine remains `fgllm-1`;
`fgllm` version 3 adds EOS completion identities, generated-token accounting,
and run stopping reasons to the version 2 decision diagnostics. Versions 1 and
2 remain readable; unavailable legacy diagnostics show as “Not recorded”.

An EOS response ends and archives its branch. Its walker slot can clone an
unfinished companion with that companion's complete prefix and ancestry. Runs
succeed after saving one EOS completion per initial walker, or stop at the
shared token budget (`walkers × sequence cap`), exhaustion of active branches,
or the iteration limit. Already-running responses are retained; queued requests
stop when the EOS target is reached. Run and Step stay disabled until Reset.

The default generation model is `qwen/qwen3.5-35b-a3b`, with Alibaba preferred
and revalidated for each run. Qwen requests to Alibaba mark the full final
assistant prefix with `partial: true`. DeepSeek remains selectable and receives
its `prefix: true` marker, but tested routes still restarted answers despite
receiving the complete prefix and marker. Cloning itself preserved the donor's
full sequence; the provider treated it as a completed turn.

Discovery now verifies three consecutive chunks against an exact prose passage
at temperature zero, including continuation from two inherited chunks. It never
continues a probe that ended with EOS. The checks and expected text are recorded
separately from generation; inherited prefixes, cached results, and probes do
not consume the Fractal generated-token allowance.

`make llm-web` also bundles the pinned `d3-hierarchy` 3.1.2 source and license
locally under `web/llm/vendor/`. Publish the whole `web/llm/` directory, including
`engine/` and `vendor/`. Use `npm run test:llm` and `npm run test:llm-browser`
for recording/diagnostic and browser coverage; `test:llm-live` is the opt-in
OpenRouter acceptance check.

### LLM benchmark generation and storage

The **Benchmark** tab, after Generation and Analysis in `/llm/`, runs the current Wave/Graph settings, independent
sampling matched by starting population or actual generated-token work (or both),
and one temperature-zero answer per trial. Every method uses the same continuation
chunks and pinned generation provider. Browser runs autosave to IndexedDB; export
`.fgllmbench` files to retain a portable copy. The tab uses the full workspace width;
Generation retains the model and algorithm settings.

Choose **Current Fractal run** to inspect the current or imported recording without
running baselines. Saved/imported benchmarks provide method, trial, termination and
attempt filters. Full-answer comparisons include nonempty EOS and sequence-capped
answers, with counts shown separately; partial-only runs have a labeled preview.
**Archived answers** includes discarded generated endpoints once each. **Retained
population** preserves final slot multiplicity, using only the Graph frontier.
Independently generated identical answers remain observations; exact-text duplicates
are reported separately.

Both populations have reward/length distributions, ECDFs, embedding distance and
nearest-neighbor distributions, shared PCA projections, recorded-work progress
curves, clone/concentration histories, and separate generation/embedding usage.
Select a plotted range or endpoint to filter the trace browser, then pin two traces
to compare complete text, bytes, probabilities, chunk boundaries and ancestry.
Charts download as SVG or PNG; numeric data exports as JSON. Each trial receives
equal weight. Paired bootstrap intervals resample trials, never cloned answers or
distance pairs. Above 50,000 pairs per group, distance sampling is deterministic;
sampled nearest-neighbor distances are upper bounds. Missing measurements remain
unavailable. Likelihood uses the provider’s generation settings and is not a
common-temperature rescore or an answer-quality grade.

The **Evaluation** tab follows Benchmark and contains grading, judge charts, the
trace browser and pinned comparisons. Chart selections in Benchmark open the
linked traces in Evaluation. Both tabs share the source, trial filters and reports.
Enter the shared OpenRouter session key and click **Evaluate with Gemini Flash**
to grade missing or failed answers. Counts, request limits and available cost
estimates are informational; no separate preview action is required.
The default `~google/gemini-flash-latest` alias resolves through the catalog to the
latest stable standard Flash release, then pins a compatible active provider.
Advanced settings allow a different judge without changing generation.
The method-blind judge uses four editable 0–4
criteria and an optional reference; only fully assessed grades receive an overall
0–100 score. Endpoint-level structured-output support is checked, the provider is
pinned, and every response is validated locally. Grading uses concurrency two,
temperature zero and a 1,024-token response cap. Pause, stop and explicit retry
preserve valid grades. Changing model, provider, rubric or reference creates a
separate session; judge costs are kept separate from generation.

Select **Pairwise ranking** in Evaluation to compare answers in both presentation
orders. Gemini Flash judges correctness, relevance, completeness, clarity, and an
explicit overall rubric. A regularized Davidson model provides Elo-scale ratings,
approximate posterior intervals, component-specific rank intervals, and predictions
for unjudged pairs. Unobserved answers and disconnected comparisons remain unavailable.
The default cap is 600 provider POST attempts including retries; no extra requests
are scheduled automatically. Plans allocate 50% of pairs to adaptive ranking, 40%
to random method audits, and 10% to held-out validation (80%/20% for Fractal-only
recordings). Large runs use a reproducible ranking subset, while method audits sample
the full saved populations and preserve clone weights without duplicating judge evidence.

Method audits report equal-trial preference shares, assessment coverage, missing-data
bounds, and simultaneous finite-source confidence bounds. These describe the frozen
source and judge protocol, not factual correctness. Across-trial bootstrap intervals
are separate and approximate. Training is frozen before held-out requests. Validation
shows calibration, log loss, Brier score, order disagreement, cycles and prior sensitivity.
An optional second-model or blinded human audit samples up to 20 completed pairs;
its judgments and costs remain separate. Pause, stop, explicit retries and linked
extensions retain earlier evidence. Version 2 `.fgllmcompare` reports include frozen
sources and incremental ranking journals; version 1 reports still import. Offline CLI
processing exports `ranking_*.jsonl` tables and recomputes ratings without credentials.
Browser/CLI floating-point results are checked within `1e-8`; the independent SciPy
optimizer reference is checked within `2e-5` in latent parameters.

Run `npm --prefix fractal-gas-web run test:llm-ranking-browser` for the mocked browser
workflow, and `uv run python fractal-gas-web/tests/llm/ranking-reference.py` for the
independent numerical check. Paid live judging remains opt-in.

Browser comparison reports retain the source snapshot, grading sessions and view
settings. Export `.fgllmcompare` for an offline, portable report; ordinary `.fgllm`
versions 1–3 and `.fgllmbench` remain supported. The CLI can process reports without
credentials or network access:

```sh
npm run benchmark:llm -- process --input report.fgllmcompare --output ../outputs/comparison
```

This adds metrics, comparison traces/progress/compute, grading sessions and grades
to the existing JSONL tables. Paid grading is a browser action in this version.

Build the LLM engine with `make llm-web` from the repository root. The same runner
also works directly in Node 22+, without a browser. From `fractal-gas-web/`:

```sh
npm run benchmark:llm -- run --config benchmark.json --output ../outputs/llm-benchmark
npm run benchmark:llm -- resume --output ../outputs/llm-benchmark --retry-incomplete
npm run benchmark:llm -- export --output ../outputs/llm-benchmark --file ../outputs/benchmark.fgllmbench
npm run benchmark:llm -- process --input ../outputs/benchmark.fgllmbench --output ../outputs/benchmark-tables
```

Set `OPENROUTER_API_KEY` in the environment for generation. Export and processing
work offline without a key or a built engine. A minimal `benchmark.json` is:

```json
{
  "config": {
    "prompt": "Explain why the sky is blue.",
    "algorithm": "wave",
    "walkers": 8,
    "chunk_tokens": 32,
    "sequence_tokens": 256
  },
  "comparison": "both",
  "repetitions": 3
}
```

Unspecified lab settings use the current defaults. `comparison` accepts `tokens`,
`population`, or `both`; the benchmark defaults to `tokens` and one trial. The
sequence cap applies to each independent answer. Only Fractal uses the lab's
iteration limit; its accepted newly generated tokens determine the token baseline
budget. The latter replenishes terminated independent trajectories and can end
with partial answers. Three empty rounds stop it as incomplete. Embeddings are
recorded for every nonempty prefix and their usage remains separate from generation.

The CLI writes `events.fgllmbench` incrementally and checkpoints `manifest.json`.
A writer lock prevents concurrent generation into the same directory. Resume
recovers the last complete journal line, preserves completed methods, and requires
`--retry-incomplete` to restart an interrupted method as a separate attempt.
Browser **Retry unfinished** has the same behavior. Remote samples are not made
reproducible by the recorded Fractal seed or by temperature zero.

The version 1 archive has a manifest header followed by ordered JSONL events.
Request starts, provider attempts/responses, accepted continuations, generation
boundaries, and method outcomes retain stable benchmark/run/request identities.
Uncommitted accepted responses remain in the archive. Processing emits JSONL tables
for runs, trajectories, nodes, tokens, requests, snapshots and clone decisions,
plus ordinary `.fgllm` files for Fractal attempts. Join tables using `benchmark_id`
and `run_id`, with `node_id` and chunk token indices for finer detail. Likelihood/NLL
and usage totals are direct calculations; unavailable provider measurements have
null totals and explicit missing counts. Export/processing refuse to overwrite
existing files. Archives include prompts and model output but exclude API keys.
If journaled responses push a failed attempt beyond the legacy `.fgllm` size
limit, its extracted recording explicitly omits request provenance; the complete
requests and responses remain in `.fgllmbench` and the processed request table.

Run `npm run test:llm` for runner/storage/CLI tests and
`npm run test:llm-benchmark-browser` for mocked browser persistence and recovery.
`npm run test:llm-comparison-browser` covers comparison sources, reports and grading.
Live provider checks remain opt-in.


### LLM scoring objectives

New runs default to **Beam-style length normalization** (`objective: "beam"`), which divides cumulative generated-token log probability by the token count raised to `beam_alpha` (default 0.6, range 0–2). **Negative mean Xent** (`objective: "mean"`) divides by the generated-token count; equivalently, it is beam-style scoring with α = 1. These objectives need no extra requests.

**Mean XED** (`objective: "xed"`) scores the same full answer twice with Together AI (`scoring_model: "Qwen/Qwen3.5-9B"`): once with the original user question and once with an empty question in the same non-thinking assistant scaffold. Their log-probability difference is divided by the scorer's answer-token count. `xed_direction` selects `"maximize"` (default) or `"minimize"`. Enter a separate Together key in Generation settings; CLI benchmarks read `TOGETHER_API_KEY`. Credentials are never exported. This measures question-conditioned likelihood contrast, not correctness.

Together scoring passed live checks with one ignored output token per request; zero-token requests are rejected by the tested route. The adapter verifies echoed probabilities against the pinned Qwen tokenizer, including split Unicode tokens, and stops on incompatible responses. XED alone lazily loads the locally bundled tokenizer (~12.8 MB data); its source revision and license are in `llm/tokenizer/`. `make llm-web` bundles `@huggingface/tokenizers` 0.2.0 alongside it. Cached answers reuse both scores without merging EOS identities.

The native engine receives cumulative utility explicitly, so cloning and answer ranking use the same score. Analysis defaults to **Selected objective**, with raw XED and its direction shown separately from maximizing utility. Version 3 recordings and benchmark exports retain both XED token-probability sequences, scorer identity, direction, and separate scoring usage; legacy total/mean imports keep their original interpretation. Scoring input/output tokens do not consume the generated-sequence budget.
