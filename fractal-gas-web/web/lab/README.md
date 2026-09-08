# Fragile Tech · Control Laboratory

Read the [published laboratory user guide](https://fragiletech.github.io/fragile/docs/lab/) on GitHub Pages. The lab's **User guide ↗** link opens it in a new tab so your running session stays available.

The guide's source pages are also available in the repository:

- [Getting started](../../../docs/source/project/control_lab_getting_started.md): build, first run, keyboard driving, and troubleshooting.
- [Task tutorials](../../../docs/source/project/control_lab_tasks.md): illustrated walkthroughs for all six tasks, including every racing circuit.
- [Controls and planners](../../../docs/source/project/control_lab_controls.md): every planning setting, clock, view, and telemetry field.
- [Scene editor tutorial](../../../docs/source/project/control_lab_scenes.md): three illustrated workshops, every editing control, templates, and manual actions.
- [Scene JSON reference](../../../docs/source/project/control_lab_scene_reference.md): field meanings, units, defaults, limits, and downloadable examples.
- [Replay, recordings, and checkpoints](../../../docs/source/project/control_lab_replay.md): replay workflows, save formats, and device storage.
- [Experiments and diagnostics](../../../docs/source/project/control_lab_experiments.md): benchmarks, fork comparisons, and performance measurements.
- [Engine architecture and extensions](../../../docs/source/project/control_lab_architecture.md): state ownership, Python batches, and extension contracts.

These source links work in a repository viewer. In the rendered book, open
**Continuous-control laboratory** in the navigation to browse the guide.

A custom C++17 engine for Fractal Gas: complete mutable world states in compact
batches, parallel rollout stepping, joint continuous actions, and optional
exploration history. The browser and Python call the same C ABI. The physics
engine has no external dependencies beyond the C++ standard library. Three.js
is used only for rendering.

## Run the laboratory

The Lab starts paused with a first-visit task chooser. Its persistent toolbar keeps
Run, Step, Restart, and Save / Open beside the simulation. Setup, Controller,
Rewards, and View panels expose common controls before advanced settings.

Configuration fields edit a draft. **Apply and restart** commits scene, controller,
and recording-policy changes together; **Discard changes** restores active values.
Reward-only edits use **Apply to current run**, preserving the world and recording.
Before replacing a run, the Lab pauses the worker, saves on this device, prepares
the replacement, and commits it only after initialization succeeds. Storage errors
offer Retry, Export, Cancel, and an explicit Continue without saving.

**Inspect / Edit / Drive** selects the interaction mode. Drive starts paused;
**Start driving** advances fixed physics steps continuously, including after keys
are released. Pause, focus loss, a hidden tab, or a modal stop driving. Unsupported
keyboard channels remain accessible through the Drive actuator sliders.

The timeline separates **World motion** from **Planner decisions**. Opened runs are
read-only; **Create run from this frame** preserves the parent and starts a distinct
paused child with a fresh planner. Planner checkpoints remain the mechanism for
exact search continuation. Comparisons expose structured A/B settings, duplication,
a difference summary, and playback synchronized by simulation time.

The initial and reset camera view fits the playable collision boundary with a 5%
margin and centers that boundary. The fit is recomputed for the current viewport,
so the whole arena remains visible in overhead and flight side views. Scroll zoom
and agent focus still use their independent manual magnifications.

Run the workspace acceptance checks against a local server with:

```sh
cd fractal-gas-web
CONTROL_TEST_URL=http://127.0.0.1:8089/lab/ npm run test:lab-workspace
node tools/capture-workspace.mjs
```

On GitHub Pages the lab is published at `/fragile/lab/`, alongside the arcade
at `/fragile/`, the documentation portal at `/fragile/docs/`, the lectures at
`/fragile/docs/theory/`, and the lab guide at `/fragile/docs/lab/`. The Pages
workflow builds the lab as a separate artifact containing both engine variants,
renderer modules, scenes, and assets. It tests the project URL with an ordinary static server:
the service worker supplies isolation for threaded planning, with a serial
fallback when service workers are unavailable. Manual workflow runs on `main`
can also deploy the complete site.

Run that deployment smoke test after building the browser targets:

```sh
node fractal-gas-web/tests/control-pages-browser.mjs
CONTROL_BROWSER=firefox node fractal-gas-web/tests/control-pages-browser.mjs
```

From the repository root:

```sh
make control-native
# Activate an Emscripten SDK first; tested with 6.0.8.
make control-web
make control-lab
```

Open **http://127.0.0.1:8080/lab/**. The existing arcade links to this lab.
`CONTROL_PORT=8089 make control-lab` selects another port. Node 20 or newer and
CMake 3.16 or newer are needed for the browser build. No ROMs, emulator
submodules, accounts, or external asset servers are needed by the lab.

`make control-web` builds both single-thread and pthread WebAssembly modules,
copies the pinned renderer and GLB loader into `vendor/`, retains the Blender
collections, and exports the legacy procedural GLB assets.
Generated engine and vendor files are ignored by Git and rebuilt by CI. The
committed assets and scenes work offline once the application has loaded.

The supplied server sends COOP/COEP headers. On hosts without these headers,
the existing isolation service worker can supply isolation; otherwise the lab
uses its single-thread module. The HUD reports the actual thread count.
The **Worker threads** control accepts 1–64 (default 8) for the live planner,
including FMC, iCEM, and MPPI. The calling thread participates, so selecting 64
prewarms 63 additional pthread workers; smaller settings allocate smaller pools.
Native Python also accepts `ControlEngine(scene, worlds=256, threads=64)`.
Thread count does not change snapshot size or contents. Actual throughput depends
on available CPU cores and batch size. Benchmark and comparison experiments
currently use one thread independently of this live-planner setting.
See [Emscripten's pthread requirements](https://emscripten.org/docs/porting/pthreads.html).

## Work with batches in Python

```python
from pathlib import Path
import numpy as np
from fragile.fractalai.control import ControlEngine

scene = Path("fractal-gas-web/web/lab/scenarios/mining.json")
with ControlEngine(scene, worlds=256, threads=4, seed=7) as engine:
    # Every row is a whole world, including both ships and all cargo.
    actions = np.zeros((engine.worlds, engine.action_dim), dtype=np.float32)
    actions[:] = (engine.action_low + engine.action_high) / 2
    checkpoint = engine.get_states()
    engine.step_batch(actions, dt=6)
    results = engine.transition_results()  # [worlds, 4]
    # Columns: reward, frames advanced, terminal flag, collision count.
    engine.set_states(checkpoint)
    engine.gather_states(np.arange(256, dtype=np.int32) // 2)
    compact = engine.serialize_states()
    engine.deserialize_states(compact)
```

`get_states(out=buffer)` fills an existing contiguous uint8 array of shape
`[worlds, stride * 4]`. `get_states(copy=False)` borrows read-only native storage
without copying. A lease keeps the storage alive after engine destruction;
later writes detach any retained storage. Holding many leases consumes memory,
so release views after use. `BatchState.kinematics` views `[worlds, bodies, 6]`
with columns `x, y, vx, vy, angle, omega`; editing a copied batch and calling
`set_states` is supported. Metadata words in raw rows are integer bit patterns,
so do not interpret the entire row as floating-point kinematics.

`observations(out=buffer)` and `transition_results(out=buffer)` reuse caller
storage. Calls into the library release Python's GIL. A single engine is owned
by one caller at a time; independent Python callers should use separate engines.
Set `FRAGILE_CONTROL_LIBRARY` to load a custom native build. Binary wheels do
not bundle this experimental engine; the default build uses a source checkout.

`ControlEnv` adapts the engine to existing `RoboticFractalGas` and
`PlanningFractalGas` classes. It performs one native batch step but constructs
Python state objects for compatibility; use `ControlEngine` for the fastest
get/set and planning path. Keep `record_frames=False` in that adapter. Browser
rendering is separate from the Python environment.

## Native state and parallel execution

`src/control/scene.hpp`, `state.hpp`, `physics.hpp`, and `wave.hpp` expose the
C++ API. Link `fg_control_core`. `src/control/c_api.h` declares the shared ABI;
`web/lab/native.js` and `src/fragile/fractalai/control/engine.py` are thin bindings.

A compiled scene owns masses, hulls, boundaries, holes, gravity wells, zone
definitions, and a static edge grid. None of this geometry is copied during
cloning. Each 64-byte-aligned world row contains six float32 body arrays plus
only the mutable task data. All bodies in a world form one joint FMC meta-agent.
Parallelism is across worlds; a world is solved in canonical body/contact order.

Flight mode keeps this same 2D state and collision contract while interpreting
the source `x` axis as horizontal travel and `y` as altitude. Set
`environment.flight` to `true` to force it, `false` to disable it, or omit it to
auto-enable the mode when a controlled body has `flight_capable: true`. Flight
scenes apply `environment.downward_gravity` (default `9.81 m/s²`) to every body;
there is no hover assist, so rockets and drones must use their existing actuator
forces to stay aloft.
The sidebar's **Flight mode** checkbox sits beside the rock properties: `AUTO`
preserves capability-based detection, while clicking it forces `ON` or `OFF` and
reloads the scene paused.

For B bodies, C controlled bodies, T tether slots, P pickups and A optional extension fields:

| Row data | Bytes |
| --- | ---: |
| Tick, 64-bit environmental RNG, task counts and terminal flag | 32 |
| Position, velocity, angle and angular velocity | 24 B |
| Active/delivered flags | 4 B |
| Per-agent next-gate indices | 4 C |
| Tether target and rest length | 8 T |
| Pickup position and respawn timer | 12 P |
| Optional actuator/world extension float32 fields | 4 A |

The lossless serialized payload per world is `32 + 28B + 4C + 8T + 12P + 4A` bytes.
Alignment padding is omitted. A batch file has one additional 32-byte header:
magic `FGCS`, format version, scene fingerprint, world count, payload word count,
and a 64-bit FNV-1a payload checksum. Words are little-endian. A snapshot is
accepted only for the same scene fingerprint, layout, and batch size. All rows
are validated before state is committed. The checksum detects accidental
corruption; it is not cryptographic authentication.

The physics path copies each selected source row into the alternate bank and
steps there. Gather source rows always refer to the original population, even
when destinations exchange places or multiple walkers choose the same source.
State stepping allocates no per-world heap objects after initialization. The
allocation test observes one `std::function` allocation per batch at both 16
and 256 worlds. Contacts and other scratch are preallocated per worker. A dense
scene exceeding contact capacity returns an error without committing the output.

Physics snapshots include environmental RNG and future-affecting task state.
They do **not** checkpoint the planner's coordinator RNG, its in-progress
population, or UI state. Save an exploration archive for completed tree history;
restart a plan from a restored root using the same seed for repeatability. Use
`checkpoint()` / `restore_checkpoint()` for the complete in-progress planner,
including its population, elite bank, ancestry and RNG. These checkpoints require
the same engine build/backend; they are larger than world snapshots.

## Physics and scene configuration

`scenario-catalog.json` supplies the environment picker and benchmark suite.
The files under `scenarios/` are editable examples:

| Preset | Bodies controlled | Task |
| --- | ---: | --- |
| Asteroid harvesting | 1 | Hook polygon ore and deliver it to a base under local gravity |
| Ants & drops | 1–128 (default 5) | Choose any of the four vehicle types; seeded drop respawning |
| Tandem flight | 2 | Sequential checkpoint loop and formation penalty |
| Collaborative mining | 2 | Two elastic tethers sharing one liftable asteroid |
| Mining rocket / thinking graphs | 1 | Tethered search, risk and tree diagnostics |
| Racing / select track | 1 | Ordered checkpoints around a closed circuit, with lap progress and manual driving |

### Flight mining defaults

Asteroid harvesting and Collaborative mining start with upright rockets and
24 N of thrust each. A single rocket can lift the stock rock and hook with
turning headroom; wall contacts remain physical but do not end the run.
The presets reward target progress and catches, with unrestricted movement
reward (`distance_squared`) disabled so falling or racing around is not a goal.

Their `controller_defaults` select a 32-action horizon, 6 physics frames per
action, and 4 elites. Wave Jump keeps the standard 128 walkers and shared-prefix
execution. New tasks use these recommendations; switching scenes updates values
that still match the previous preset and preserves custom controller settings.
Imported runs and checkpoints retain their recorded settings.

Run `npm run test:lab-mining` from `fractal-gas-web/` to check ten simulated
seconds of both shipped presets in WASM at seed 7. The acceptance run requires
a new catch, sustained towing, a delivery, and no termination in each mode;
`CONTROL_MINING_SEEDS=7,19,42` checks additional seeds. This longer gameplay
check is separate from the quick preset and lift-budget unit tests.

### Ants & drops

Every environment and racing track offers **Vehicle type** (Rockets, Drones,
Karts or Harvesters) and **Vehicle count** beneath the environment selector.
Ants & drops defaults to 5 harvesters; counts from 1 to 128 are supported. These are vehicles in the live world, distinct from the planner's
population of candidate worlds. Each vehicle uses its archetype's physics and model.
Rockets have two action channels; drones, karts and harvesters have three.

Changing type replaces the whole fleet’s physics and visuals, preserving the world
setup, vehicle count, starting positions, rocks, tethers and environment settings.
Changing count preserves existing vehicles and places additional vehicles in clear
space. Both controls restart paused and clear run, replay and editor history.
Reset retains the current scene. Each environment remembers its type selection
within the tab session; Racing shares its selection across tracks. Existing preset
defaults remain until a type is explicitly selected. Explicit Flight mode settings
are preserved; otherwise rockets and drones automatically enable flight.
Exported scenes and recordings restore their actual bodies without regeneration.
Mixed or unrecognized fleets show **Mixed / custom** until a standard type is chosen.

There are 24 drop slots. Each collected drop returns at a seeded random playable
position after three **simulation seconds**, indefinitely—even after the entire
pool has been collected. Pausing also pauses the timer; slow planning can make three
simulation seconds take longer than three wall-clock seconds.

### Kart circuit library

Choose **Racing** in **Environment**, then use **Select track** to choose one of
the six circuits, ordered from Easy to Hard.
The outline, difficulty, checkpoint count, driving description, and historical
reference update with the selected scene. **Violet Circuit** retains the `racing`
ID and its original geometry. The Mite R uses the native kart actuator:
signed throttle, signed steering, and a brake channel. Enable **Keyboard control**
in the main sidebar, click the world, and use W/S to drive, A/D to steer, and
Space to brake. **Follow agent** and the mouse wheel give a closer kart view;
**2D / 3D** switches the circuit view. All registered planners can drive it.

Violet's eight-unit-wide circuit has a physical outer boundary and an infield hole.
Curbs, asphalt, lane paint, the start gantry and the documentation logo are visual
assets; native boundaries define collisions. Walls bounce and penalize contact.
Sixteen native proximity checkpoints must be reached in order. The final zone
is at the start/finish marking; a lap is `floor(checkpoints / 16)`. The zones are
not directional timing lines. Waiting at the finish or skipping checkpoints does
not complete a lap. The experiment's default goal is 16 checkpoints (one lap).
Lap counters, the highlighted target, and kart/world movement restore during replay.

The five historical layouts are Roots Oval (Easy), Fearless Circuit (Medium),
Sepang Kart, Original Obstacle Circuit, and Fearless Obstacle Field (Hard).
They use identical vehicle settings. Their uniformly scaled video traces preserve
visible proportions; their dimensions are simulation units, not surveyed metres.
See [circuit sources and reconstruction notes](circuits.md) for the inventory,
uncertainties, difficulty criteria, and regeneration commands. Each circuit has
its own ordered checkpoint count. Imported scenes and replay archives retain
their own boundary, holes, route, checkpoints, and optional `circuit` metadata.

Generate the library with
`uv run --script fractal-gas-web/tools/make-racing-scene.py`; append `--check`
to validate polygon geometry and verify the committed output without writing.
No new engine task, agent-specific planner code or mutable state fields are needed:
the kart world occupies 64 bytes, or 96 bytes including its snapshot header.

### Branding and environment extensions

The header and favicon use exact copies of `docs/logo.png` and `docs/favicon.png`.
`npm run build:lab` refreshes them in `branding/`; purple `#9059a4` comes from the
documentation mark. The interface uses a lighter purple for contrast on dark panels.

Add a preset ID and label to `scenario-catalog.json` and its scene JSON under
`scenarios/`. `scene.presentation` selects task labels and metric readouts through
`scene-presentation.js`; optional score divisors and progress cycles support laps
or other grouped objectives. `registerSceneMetric` adds custom readouts.

`visuals/environments/index.js` imports environment renderer registrations. A
`registerEnvironment(kind, factory)` factory receives the scene and returns
`{group, replacesGates?, update?}`. `update(state, info)` can derive decorations
from live or replayed state. Select the renderer with `scene.environment.kind`.
The default arena renderer and the circuit renderer share this interface. Circuit
road geometry follows `boundary` and `holes`; update its decorative `centerline`,
`start`, and `sponsor` metadata when redesigning the layout. Physics remains the
source of collision geometry, independently of rendering.

Actions have shape `[worlds, action_dim]`. Query `engine.channels`, `action_low`,
and `action_high`; dimension and bounds come from compiled actuators. The default
vector actuator has thrust `[0,1]` and torque `[-1,1]`; karts have signed throttle,
signed steering, and brake `[0,1]`. Holonomic drives have three signed channels;
thruster arrays have one bounded channel per independent thruster. Each built-in
actuator also accepts an optional `action_multipliers` object, keyed by channel
name, with values from `0` to `10`. The multiplier scales that channel's native
range and corresponding physical output; `0` disables it and omitted values are
`1`. The Lab exposes these values as per-agent-type sliders under Problem
properties. The catalog includes a three-thruster tug. Kart forces model lateral
grip, velocity-dependent steering and braking. Positive thrust points along local
+X. Position uses metres, time seconds, mass kilograms, force newtons, torque N·m
and angle radians. Physics advances at `dt=1/60` by default, with four fixed
substeps.
Actions can repeat for 0–4096 frames; planning uses 1–4096 frames per transition.

The engine implements circles and convex polygon rigid bodies, angular inertia,
exponential linear/angular drag, softened inverse-square attraction, contact
impulses, restitution/friction, conservative continuous collision checks,
positional correction, and implicit spring/damper tethers with break forces.
The outer boundary may be non-convex, and may have multiple non-convex holes.
Static raycasting uses the same compiled boundary geometry.

Important JSON fields and defaults:

| Object | Fields |
| --- | --- |
| Root | `version:1`, `name`, `task`, `size:[64,44]`, `boundary`, `holes`, `bodies`, `gravity`, `bases`, `gates`, `pickups`, `tethers`, optional `environment.flight` and `environment.downward_gravity` |
| `physics` | `dt:1/60`, `substeps:4`, `solver_iterations:8`, `lethal_walls:false`, `lethal_bodies:false` |
| Body | `position`, `velocity:[0,0]`, `angle:0`, `omega:0`, `radius:0.5`, optional convex local `vertices`, `mass:1`, optional `inertia`, `drag:0.15`, `angular_drag:2`, `thrust:12`, `torque:8`, `restitution:0.25`, `friction:0.3`, `controlled:false`, `cargo:false`, `flight_capable:false` |
| Gravity | `position`, `strength:10`, `softening:2`; negative strength repels |
| Base/gate/pickup | `position`, `radius:1` |
| Tether | `a` source body, `b:-1` for disconnected, `rest_length`, `stiffness:25`, `damping:6`, `break_force:500`, `hook_range:2`, `automatic:false` |
| `rewards` | `progress:1`, `collision:2`, `pickup:10`, `delivery:100`, `gate:30`, `formation:0.15` |
| Root task settings | `formation_distance:3`, `respawn_seconds:4` |

Bodies may have up to 32 convex vertices. The compiler validates geometry,
finite values, ranges, simple boundaries, hole separation and body centers.
Packed state is limited to 100,000 float32 words per world.
The working limits are 4096 bodies and 8192 worlds, also subject to a 512 MiB
estimated working-memory guard. These are limits, not interactive performance
targets. Snapshots never contain GPU assets or cached contact impulses.

Automatic tethers attach to the nearest active cargo within range, using the
current distance as the rest length. Tethers act between centers, not arbitrary
local attachment points. Cargo inside a base is marked delivered and detached.
Food respawns after its timer using the row's PRNG. Gates advance independently
for each controlled body. The tandem overlay draws a forward virtual anchor;
the reward penalizes deviation from the requested formation spacing.

This is a fixed-step custom solver, not an exact mechanics oracle. Conservative
advancement has iteration/bounce caps, and dense contact configurations can
exhaust scratch capacity. The solver is intended for these control tasks,
not general-purpose deformable-body or 3D collision simulation. Replays are
exact within the same build/backend and independent of thread count. Native
and WebAssembly floating-point library implementations may differ in their
last bits, so cross-backend continued trajectories require tolerance.

## Wave, FMC, clocks and recording

```python
with ControlEngine(scene, threads=4) as engine:
    action, metrics = engine.plan(walkers=128, horizon=16, frames=6,
                                  elites=2, recording=1, seed=7)
    engine.step_batch(action, dt=6)
    tree = engine.exploration_tree()
    graph = tree.to_networkx()
    tree.save("search.npz")
    engine.replay_node(int(tree.metadata[-1, 0]))
```

The new packed Wave calls the existing C++ fitness and cloning operators. The
phase order is elite injection, pre-clone fitness, a second companion draw for
cloning, simultaneous source selection, action sampling, fused gather/step,
and elite update. Planner randomness is sampled on the coordinator, so worker
scheduling does not change it. Continuous FMC selects the mean of the final
population's inherited first actions, matching the Python planner's selection
rule. Elite state, ancestry and first actions travel together. If every walker
dies, the native planner selects zero projected into each channel interval. Optional inertial sampling
perturbs inherited actions with Gaussian noise, then clamps action bounds.

`begin_plan` / `advance_plan` / `selected_action` expose incremental planning.
Calling `selected_action` finishes the current complete iteration; it can
therefore end a plan before the configured horizon. `wave_step` continues the
population for exploration. The browser's **Advance Wave population** control
shows walker zero's complete world and the whole sampled future cloud.

**Reproducible mode** waits for each complete plan, then advances the physical
world by the configured action duration. The simulation clock is independent
of rendering and wall-clock speed. **Real-time mode** advances fixed physical
ticks while a planner worker targets a future tick predicted under the committed
action. Results are accepted only when scene revision, target tick and the
complete predicted root snapshot match. A late or mismatched result is rejected;
the fallback action is zero projected into each channel interval. Rendering stays on the UI
thread. Overloaded real-time simulation limits catch-up work and can run slower
than wall time; it does not change the physics step size.

Recording modes are Off (0), Pruned (1), and Full (2). A tree stores one complete
root snapshot plus node IDs, parent IDs, actual frame durations, actions,
controlled-body positions, rewards, fitness, dead flags and tether flags.
Branch replay reconstructs full states from the root and transition sequence.
Pruning removes orphan leaves while protecting current walkers, elites and
their required ancestors, following the Python tree's parent-protection rule.
The existing arcade `FractalGas` also supports this optional recorder through
`FractalGasParams::recording`; its default behavior remains unchanged.

The browser keeps the latest 32 decisions by default. **Keep all decisions**
retains every decision up to a 64 MiB tree budget and stops with an export prompt at capacity.
**Export run** saves scene, settings, complete world motion and optional trees
in a version-2 `.fgclab` file (version-1 tree-only files still import);
**Open run** reloads it. The exploration slider selects a decision. Click a nearby branch
in the viewport or enter a node ID, then **Replay branch** to restore that future.
**Save state / Load state** use the smaller `.fgcs` binary files. Scene edits
invalidate snapshots from the old scene; undo restores the previous scene.
Configuration changes restart the experiment and clear its current history.

The HUD reports measured values:

| Metric | Meaning |
| --- | --- |
| Toy / experiment number | Selected preset |
| Dead ratio | Terminal fraction of the final Wave population |
| Selected-action risk | Terminal fraction of 16 separately sampled continuations: chosen action for F frames, then independent uniformly sampled actions for 2F frames |
| Evaporated | Nodes removed by the most recent pruning pass |
| Cloned | Fraction of walkers cloned in the most recent Wave iteration |
| AI budget used | Completed Wave iterations divided by the configured horizon |
| Missed deadlines | Physical action boundaries without an acceptable real-time plan |

The 16-sample risk statistic is a short empirical diagnostic under random
continuations, not a safety guarantee or a calibrated probability of failure
under future FMC decisions. Wave-only mode leaves it unevaluated. Green paths
have above-average recorded reward, red paths are terminal, purple paths carry
tethers, and blue paths are alternatives. The renderer caps drawing at roughly
50,000 segments without pruning or changing the native tree.

## Replay the executed world

**WORLD REPLAY** records the initial world and every executed physics frame,
including reproducible FMC steps, real-time control and manual actions. Recording
continues when **Tree** is Off. Each frame contains all body poses and velocities,
active/delivered flags, tether attachment and rest lengths, pickup positions and
respawn timers, task counters and the environment PRNG. Static geometry and
meshes remain in the shared scene.

Use the world slider to seek, **Play world** to animate at ¼×–4×, and **Back to
live** to return to the paused authoritative world. Seeking only changes the
presentation. **Continue here** restores the selected packed row in the native
engine and starts a labeled segment; run or step to plan from there. Snapshot
loads, search-branch replay and Wave selection also start labeled segments.
Those discontinuities are cuts, not interpolated motion. Search replay records
every edge's physical frames so that alternative trajectories can also be played.
The matching recorded decision tree is shown when available.

The stream stores `4 * (scene.words + action_dim + 1)` bytes per frame:
unpadded world bytes, float32 actions and a uint32 decision ID. The world tick is
already in the state header. At 60 Hz, the six-body harvesting preset uses
224 bytes/frame (about 13.1 KiB/s). Storage uses 256-frame typed-array chunks,
constant-time frame lookup and transfer of worker packets; renderer state views
are borrowed, and a padded copy is created only when restoring the native world.
There is no physics stepping or search reconstruction during ordinary playback.

The default in-memory recorder has a separate 64 MiB payload budget and stops recording/control at capacity
without evicting old frames. Export and reset for a longer experiment. Both budgets
exclude JS object/chunk overhead and temporary export/import buffers. Archives
use base64 inside JSON and CRC32 on the motion payload; native snapshot validation
checks the scene fingerprint when importing. The raw packed state and `.fgcs`
formats are unchanged. Planner population, internal planning RNG and wall-clock
scheduling are not restored by world replay.

## Agent types and extension points

An agent type is static scene data. `agent_types` defines reusable `physics`
and `visual` defaults; `extends` inherits another type, and each body names its
`agent_type`. The native C++ compiler resolves physics once, before allocating
world batches. Instance fields override defaults, and array fields replace the
inherited array. Unknown types, cycles and malformed definitions are errors.
All supplied presets include the editable catalog from `agent-catalog.json`.
Old scenes with explicit body fields continue to work. Set
`physics.flight_capable: true` on a controlled custom rocket or drone to opt it
into automatic flight detection; an explicit scene-level `environment.flight`
value always wins.

For example, add this definition alongside the preset types and place a body
`{"agent_type": "courier", "position": [12, 14]}`:

```json
{
  "courier": {
    "extends": "drone",
    "label": "Courier · light transport",
    "physics": {"mass": 0.5, "thrust": 9, "drag": 0.25},
    "visual": {
      "model": "kit",
      "color": "#ffce70",
      "parts": [
        {"shape": "box", "size": [1, 0.45, 0.2], "position": [0, 0, 0.3]},
        {"shape": "ring", "size": [0.4, 0.04], "position": [0, 0, 0.5], "emissive": true}
      ]
    }
  }
}
```

The native engine and Python binding accept the same JSON. Static types have no
per-row overhead. Optional custom mutable actuator or world-extension fields are
appended once to each packed row and automatically follow get/set, gather,
snapshots and planner checkpoints. Native actuators compile once through
`register_actuator(name, compiler, evaluator)` in `src/control/actuators.hpp`.
The compiler declares channel names and arbitrary finite low/high bounds, initial mutable state and optional
immutable parameters. Runtime evaluation uses a resolved function pointer and
must keep all future-affecting mutable data in the row. Add the plugin source to
`fg_control_core` and register it before compiling scenes.

`src/control/extensions.hpp` provides `register_world_extension(name, compiler)`
for custom task/reward/sensor logic. Scene `extensions` are compiled in order;
callbacks run after built-in mechanics each physical frame and may append
observations. Their initial state also lives in the packed auxiliary region.
Callbacks must be deterministic, thread-safe, finite-valued and allocation-free
on the hot path. Planar rigid-body dynamics remain the built-in physics domain.

Algorithms register with `registerController(id, definition)` in
`controllers/registry.js`; add their imports in `controllers/index.js` once.
`instantiateController(id, engine, settings, scene)` accepts an engine contract;
`createController` supplies the native allocation adapter. A controller exposes
`begin(root, seed)`, incremental `advance()` and `result()` with a bounded action.
Use opaque snapshots, batch state operations, channel descriptors and rewards;
controllers do not need to know body counts, geometry or row offsets. Optional
`checkpoint`/`restore` methods preserve variant-specific search state. FMC,
seeded random control, CEM, iCEM and MPPI are built-in examples. Optional
`parameters` descriptors (`label`, `default`, `min`, `max`, `step`) produce the
controller's numeric controls under **Planner settings**. Tree
pose dimensions are independent of action dimensions.

### Sampling baselines

Select **iCEM** or **MPPI** in the controller selector or in **Experiments**.
The same settings are accepted as experiment JSON overrides and by the CLI.
`search_iterations` defaults to 3 for both methods. They optimize normalized
controls in `[-1, 1]`, mapped to each channel's declared bounds, without using
body types or internal state layouts. World simulation uses native parallel
batches; sampling and optimizer updates run in the JavaScript planning worker.

- **iCEM** (`controllers/icem.js`) uses Gaussian noise with a `1/f^beta` spectrum,
  shifted means and elites between decisions, elite reuse, distribution momentum,
  decreasing populations, and an evaluated mean candidate in its last round.
  Defaults: `icem_beta=2`, `icem_sigma=0.5`, `icem_min_sigma=0.01`,
  `icem_elite_fraction=0.1`, `icem_keep_fraction=0.3`, `icem_alpha=0.1`,
  `icem_decay=1.25`. Retained elites and the mean occupy slots within `walkers`
  total capacity. Retained sequences are re-evaluated; inactive slots use zero
  simulation frames but still incur native row-copy overhead. The dependency-free
  noise generator pads to a power of two, crops to the horizon, and normalizes
  ensemble variance. These are explicit implementation choices relative to the
  [iCEM paper](https://proceedings.mlr.press/v155/pinneri21a.html).
- **MPPI** (`controllers/mppi.js`) uses a fixed diagonal Gaussian with
  `mppi_sigma=0.5` in normalized coordinates and `mppi_temperature=1` in summed
  reward units. Its softmax includes the Gaussian importance correction
  `u^T Sigma^-1 epsilon`. Latent Gaussian samples are retained for this correction;
  only physical actuation and the updated mean are clipped. The normalized prior
  is centered on the channel midpoint. It shifts the optimized plan between
  decisions. No smoothing filter or learned model is used. See the
  [information-theoretic MPPI paper](https://arxiv.org/abs/1707.02342).

`controllers/shooting.js` owns incremental batched rollouts, bounds mapping,
work accounting, and checkpoint validation; algorithms own sampling and updates.
Both checkpoints preserve partial rollouts, RNG and warm-start memory and can be
restored repeatedly without mutating the saved checkpoint. Creating a fresh
controller resets episode memory. In deadline mode a completed round's action
is returned, or the current mean if no round has finished. Both methods provide
future-state clouds and executed-world replay; they do not export search trees.

Visuals can select `rocket`, `kart`, `drone`, or `kit`; override `color` and `scale`
per type or body. Kits support box (XYZ lengths), sphere (radius), cylinder/cone
(radius, height), and ring (major radius, tube radius) parts, with position, XYZ
Euler rotation, color and emissive fields. For richer models, add a factory with
`registerAgentModel(name, factory)` in `visuals/registry.js`. A factory returns a
Three.js Group with any hierarchy. Tag parts with `userData.motion`: `thrust`,
`steer`, `wheel`, or `rotor`. Animation derives from recorded time, speed and
action, so repeated seeks reproduce the same appearance. Wheels are visual speed
indicators, not integrated wheel-angle state. Collision hulls remain explicit
physics definitions; visual scale and attachments do not change collisions.

| Module | Responsibility |
| --- | --- |
| `src/control/actuators.*`, `extensions.*` | Compiled dynamics/task/sensor plugins and mutable extension state |
| `controllers/`, `experiments.js` | Algorithm registration, incremental search and reproducible trials |
| `storage/`, `storage-panel.js` | Compressed browser storage, chunk loading and checkpoint/file controls |
| `experiment-panel.js`, `comparison-view.js` | Benchmark controls and synchronized world comparison |
| `entity-properties.js`, `physics-inspector.js` | Generic property/action controls and physics/performance diagnostics |
| `src/control/agent_types.*` | Native compile-time physics defaults and inheritance |
| `agent-types.js` | Resolve the same scene defaults for presentation |
| `visuals/primitives.js`, `visuals/vehicles.js` | Reusable geometry and original model kits |
| `visuals/registry.js` | Model factories, declarative parts, animation hooks |
| `visuals/body-layer.js` | Body transforms and instancing grouped by visual definition/color; supports nested animated parts |
| `visuals/lighting.js` | Procedural reflections and shared contact shadows |
| `renderer.js` | World, camera, environmental geometry and diagnostic overlays |
| `motion.js` | Packed world recording, chunked storage and authoritative capture |
| `playback.js` | Independent seek/playback clock; no DOM or native-engine dependency |
| `replay-panel.js` | Replay controls and presentation callbacks |
| `scene-editor.js` | Placement, selection, JSON editing and undo/redo |
| `manual-control.js` | Keyboard-to-action input adapter |
| `archive.js`, `binary.js` | Versioned recording import/export and binary encoding/checksums |
| `simulation-worker.js` | Authoritative clock and transitions; forwards captured motion |
| `planner-worker.js` | Parallel search and risk diagnostics |

## Editor and original assets

**Edit scene** pauses control. Place an agent using the **Agent type** selector, convex ore, gravity wells, food,
bases or gates; drag entities; join two bodies with a tether; or draw boundary
and hole polygons. **Apply entity** edits physical parameters. The full JSON
editor exposes every scene field. Undo/redo and JSON import/export are supported.
Shift-click selects several entities; drag moves them together. Duplicate
selection copies internal tethers, and numeric properties show physical units.
Save a selected body as an agent template; channel sliders support all compiled
actuators. Keyboard W/S, A/D, Q/E and Space control drive, steering, strafe and
braking on the selected controlled body, or the first agent. Mouse wheel zoom,
side/overhead switching for flight scenes, 2D/3D switching for standard scenes,
and **Follow agent** (selected body, or the first controlled agent), and
middle/right-drag panning are available independently of diagnostics. **Whole
arena** resets the camera.

The masthead's **Visual style** selector switches between **Futuristic** and
**Steampunk** vehicles, props, scenery materials, lighting and interface accents.
It preserves the running simulation, replay position, camera and selection, and
updates comparison viewports together. The successful choice is remembered on
this device. While models load the current view remains visible; failed loading
offers Retry.

**Animations** beside the style selector enables vehicle suspension, banking,
rotors, wheels, engine motion and world effects. Paused scenes keep gentle idle
motion. Turning it off freezes decorative motion and skips its update loops.
Static thrust and independent-jet cues, steering, and reverse/brake lamps still
follow current commands; physics, cargo levels, pickups and diagnostics continue
normally. The preference is remembered across Lab tabs and defaults off when your
system requests reduced motion. Crowds above 16 agents update secondary vehicle
motion at 30 Hz, and hidden tabs skip it.

**Action guides** defaults off and remembers your explicit choice across Lab
tabs. It shows signed command arrows and a numeric readout for the selected
controlled body, falling back to the first controlled body. Percentages express
commands relative to each channel's configured action limits, not measured forces.
Guides remain available with animations off.

The Workshop starts with neutral actuator sliders and its preview paused. Sliders
use the selected asset's catalog actuator; rocket variants expose vector
thrust/torque or individual thrusters. **Neutral** sets every channel to zero and
**Max** sets each to its upper bound. Static action cues update while animations
are off. **Play animation** advances decorative motion using those fixed commands;
it does not simulate physics or invent changing inputs. **Side** and **Top** reset
motion while preserving slider values. Values also survive style, detail and asset
changes within the workshop session. The master **Animations** switch controls
preview playback.

**Models** opens the [Vehicle Workshop](asset-gallery.html), where all eight
Blender-authored vehicles can be rotated beside their concept sheets and downloaded
as GLB models or editable `.blend` sources. Each collection also supplies recovery
docks, checkpoints and reactors. Small or crowded views use simplified geometry;
close views reveal the detailed models. Native scene hulls still define collisions.

The workshop's **Artwork and prompts** link opens the
[paired concept library](concepts/index.html): sixteen additional futuristic and
steampunk sheets for drops, towable rocks, gravity wells, docks, checkpoints,
arena and racing scenery, and capture/motion effects. Original PNG downloads and
complete prompts accompany the sheets. The Asset Workshop includes all 42 world
asset types in both styles, a Detailed/Simplified selector and a shared-envelope
overlay. Their structural designs differ while native collision geometry remains
identical. See [world collections](assets/README.md#world-collections) for Blender
sources, GLB packs, and optional custom-scene placements.

`visuals/assets.js` caches the authored models, and `visual-style.js` coordinates
style transitions across renderers. `visuals/vehicles.js` and `models.js` retain
the procedural factories for older consumers and initial loading.
`tools/build-control-assets.mjs` regenerates their root-level exports without
overwriting the Blender collections. No external images, fonts or asset services
are needed. See [`assets/README.md`](assets/README.md) for provenance, authoring,
animation tags, LOD thresholds and verification instructions.

## Reproducible experiments and durable replay

**Experiments** runs variants over explicit seeds and a frame budget in a separate
worker. Choose a success criterion and target; **All preset scenes** runs a
suite. JSON parameter overrides support comparing variants of the same algorithm.
Reports include raw scene/specification data, trial outcomes, success rate, contact
counts, completion time among successes, integrated squared action and measured
planning latency and actual simulated world-frames (also shown in the results
table). `simulatorFrames` counts completed native physics frames across all
planning worlds, including FMC, and excludes the authoritative world and risk
probes. In summaries it is the mean per episode. Identical walker/horizon
settings do not imply equal work: search rounds, population decay and early
termination matter. Planning time includes worker yields and depends on hardware
and scheduling; success and
trajectory comparisons use fixed simulation ticks. Registered evaluation metrics
can extend success criteria. Goal counters are absolute within each episode.

**Fork and compare** takes the selected world's complete packed state as the
common root. Both variants run independently; synchronized side-by-side replay
shows executed world movement. Export preserves their recordings and parameters.
The performance probe measures physics and get/set/gather throughput in a separate
batch. The live readout shows FPS, CPU render submission, draws and triangles.
**Physics inspector** overlays velocity, external forces, current nearby contact
normals and signed spring/damper tether force. It does not reconstruct past contact
impulses. Native tracked-buffer bytes exclude some scratch/tree/allocator overhead;
WASM memory includes reserved linear memory, not browser/GPU memory.

```sh
node fractal-gas-web/tools/control-benchmark.mjs \
  fractal-gas-web/web/lab/scenarios/harvest.json \
  fractal-gas-web/web/lab/benchmarks/smoke-spec.json /tmp/control-report.json
```

The first input can also be a JSON array of scenes. Python exposes native counters
through `profile(reset=True)` and first-world debug vectors through `inspect()`.

Enable **Store long runs on this device**, then reset. The recorder writes gzip
chunks (raw fallback when compression is unavailable) with CRC32 to IndexedDB,
keeps at most eight durable chunks resident for ordinary scrubbing, and loads
older chunks asynchronously. A bounded pending-write budget pauses recording if
storage cannot keep up. Full chunks flush immediately; incomplete chunks flush
every five seconds and on visibility changes. **Save recording** waits for writes;
**Saved runs** reopens or deletes recordings. Recent unflushed frames can be lost
on abrupt close. Browser storage is origin-specific and subject to quota/eviction.

Device-backed recordings keep decision trees on disk while retaining a bounded
recent tree window. **Export run** produces a `.fgcrec` binary file containing
compressed chunks and metadata; **Open run** imports it with shape/checksum checks.
Portable export holds compressed chunks, not expanded full motion, in memory.
Delivery, pickup, gate and terminal counters produce event markers; add named
markers and jump to them during replay. **Save planner checkpoint** writes `.fgcp`
at a complete iteration boundary. **Load checkpoint**, then Step, resumes its
search; a plain world replay continues by starting a new search. If the visible
world is from Wave mode, checkpoints preserve that active population and
**Advance Wave population** continues it after restore. Checkpoint save
aligns the planner root with the paused authoritative world. Future-target plans
from real-time mode may therefore restart at that world before checkpointing.

## Validation and performance

```sh
make control-test
npm --prefix fractal-gas-web run test:lab
# With the local server running and Chromium/Firefox installed for Playwright:
cd fractal-gas-web
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-help-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-rewards-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-replay-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-experiments-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-racing-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ npm run test:lab-vehicles
```

Native tests cover conservation/free flight, downward flight gravity, gravity
wells, elastic and breaking tethers, restitution, fast collisions, holes, task/RNG restoration, simultaneous
gather, thread-count determinism, exact clone/elite branch replay, input
validation and allocation scaling. Python tests cover leases, buffer reuse,
state identity, history round trips and compatibility with the existing gas.
WebAssembly and browser tests cover state/recording replay, all presets, the
editor, clocks, rendering layers and responsive layout. Additional tests cover
archetype/native agreement, mixed instanced crowds, model kits, every-frame
world restoration, recording chunk boundaries, corrupted archives, replay
speeds, continuation and motion-only export/import. Emulator ROM-dependent
tests require separately supplied ROMs.

`fg_control_benchmark` reports all combinations of 64/256/1024 worlds,
2/64/256 controlled bodies, and 1/2/4/8 threads. The benchmark steps one physical
frame with four substeps in a sparse contact scene. Results and measured machine
details are in [`benchmarks/README.md`](benchmarks/README.md); rerun on your
target hardware before choosing a real-time budget.

### Ants & Drops cargo and refineries

Cargo-enabled vehicles hold five drops by default. Pickups earn `rewards.pickup`; filling a tank
earns `cargo.full_reward` (defaults to the pickup reward). Full vehicles return
to the marked refinery and unload over two simulation seconds, earning
`rewards.delivery` proportionally across one full load. Leaving pauses discharge;
collection stays locked until empty. Partial loads cannot start unloading.

```json
{
  "cargo": { "capacity": 5, "unload_seconds": 2, "full_reward": 10 },
  "refineries": [{ "position": [12, 36], "radius": 6 }],
  "rewards": { "pickup": 10, "delivery": 100 }
}
```

`cargo` is optional; scenes without it preserve unlimited pickup mechanics.
Capacity is an integer from 1 to 10000; unloading time is 0.01–10000 seconds.
Cargo-enabled scenes require at least one refinery. Vehicles can unload together.
The refinery tool edits zones; cargo settings are edited through the scene JSON.

Vehicles show a segmented capacity meter and a resource pile that fills as they
collect. Nearby labels show held/capacity, Loading +N, Full or Unloading, plus
cumulative picked and delivered units. Selection takes priority when labels
would overlap; the viewport reuses at most 16 labels. The inspector also exposes
picked and delivered totals, including controlled vehicles with disabled actuators.

Observed gains in consecutive displayed states trigger a short intake-to-storage
transfer. Each piece travels once, then settles into the pile. Unloading carries
pieces from storage to the refinery receiving hopper. The animation follows
simulation time, so pausing holds its pose; seeks and sparse jumps clear transient
loading. Pickup ownership is not guessed from proximity. Counts always come from
native cargo state (picked = held + delivered), with no recording format change.
**Animations off** immediately shows the authoritative pile and meter, hides
transfer motion and keeps numeric readouts current. Empty harvesters have empty
hoppers in both styles and LODs.

Cargo uses three shared instanced draws, at most 106 triangles per visible vehicle,
and no new textures, lights or render passes. Hidden/inactive vehicles are culled;
secondary attachment updates reuse the crowd animation cadence. Labels add no GPU
draws. Scenes without cargo keep their existing unlimited-pickup mechanics and do
not display invented per-vehicle resource totals.

Place refinery machinery behind its apron: it extends north from `y + radius`
to approximately `y + 1.7 × radius`. The preset uses the north arena boundary
to keep vehicles outside that machinery. Custom scenes must supply appropriate
boundaries if refinery machinery lies inside the arena.

Native info field 15 is the optional cargo offset (zero when disabled). Four
float32 words per controlled body store load, return-phase latch, delivered
units, and full-cycle count. They follow actuator/extension state and are copied
by the existing snapshot, gather, checkpoint and recording mechanisms. Existing
info fields and metric indices retain their meanings; completed loads use the
delivery counter. Cargo observations add normalized load, return phase and the
normalized vector to the nearest refinery. Motion recordings accept the appended
layout and record full, unloading/resumed, and empty events. Legacy recordings
without cargo retain their original layout.

Mining environments provide **Rock size (×)** from 0.1 to 2 and **Rock weight (×)** from 0.01 to 10. Asteroid harvesting also provides **Rock count** from 1 to 20; collaborative mining keeps one shared rock. Apply rock settings restarts paused and clears the run. Size scales collision geometry and visuals while weight scales mass. Delivered rocks respawn in clear space, reusing their existing body slots. These settings are saved with exported scenes.
