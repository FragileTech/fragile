# fractal-gas-web

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
- `tests/` — dependency-free test harness; the key test replays a recorded
  run of the Python reference implementation draw-for-draw.
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
  every Atari game). The **pyramid map** panel is BUILT by the swarm like
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
# open http://localhost:8000/web/
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
