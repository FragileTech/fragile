# Fragile Tech · Control Laboratory

The book includes a practical user guide:

- [Getting started](../../../docs/source/project/control_lab_getting_started.md): build, first run, keyboard driving, and troubleshooting.
- [Controls and planners](../../../docs/source/project/control_lab_controls.md): every planning setting, clock, view, and telemetry field.
- [Environments and scene editing](../../../docs/source/project/control_lab_scenes.md): presets, editing tools, templates, and manual actions.
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
copies the pinned renderer into `vendor/`, and exports the original GLB assets.
Generated engine and vendor files are ignored by Git and rebuilt by CI. The
committed assets and scenes work offline once the application has loaded.

The supplied server sends COOP/COEP headers. On hosts without these headers,
the existing isolation service worker can supply isolation; otherwise the lab
uses its single-thread module. The HUD reports the actual thread count.
The **Worker threads** control accepts 1–64 (default 4) for the live planner,
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
| Ants & drops | 48 | Joint 144-dimensional control and seeded food respawning |
| Tandem flight | 2 | Sequential checkpoint loop and formation penalty |
| Collaborative mining | 2 | Two elastic tethers carrying the same heavy asteroid |
| Mining rocket / thinking graphs | 1 | Tethered search, risk and tree diagnostics |
| Violet Circuit / kart racing | 1 | Ordered checkpoints around a closed circuit, with lap progress and manual driving |

### Violet Circuit

Choose **Violet Circuit · kart racing**. The Mite R uses the native kart actuator:
signed throttle, signed steering, and a brake channel. Enable **Keyboard control**
in the main sidebar, click the world, and use W/S to drive, A/D to steer, and
Space to brake. **Follow agent** and the mouse wheel give a closer kart view;
**2D / 3D** switches the circuit view. All registered planners can drive it.

The eight-metre-wide circuit has a physical outer boundary and an infield hole.
Curbs, asphalt, lane paint, the start gantry and the documentation logo are visual
assets; native boundaries define collisions. Walls bounce and penalize contact.
Sixteen native proximity checkpoints must be reached in order. The final zone
is at the start/finish marking; a lap is `floor(checkpoints / 16)`. The zones are
not directional timing lines. Waiting at the finish or skipping checkpoints does
not complete a lap. The experiment's default goal is 16 checkpoints (one lap).
Lap counters, the highlighted target, and kart/world movement restore during replay.

Generate the preset with `python3 fractal-gas-web/tools/make-racing-scene.py`.
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
thruster arrays have one bounded channel per independent thruster. The catalog
includes a three-thruster tug. Kart forces model lateral grip, velocity-dependent
steering and braking. Positive thrust points along local +X. Position uses metres,
time seconds, mass kilograms, force newtons, torque N·m and angle radians.
Physics advances at `dt=1/60` by default, with four fixed substeps.
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
| Root | `version:1`, `name`, `task`, `size:[64,44]`, `boundary`, `holes`, `bodies`, `gravity`, `bases`, `gates`, `pickups`, `tethers` |
| `physics` | `dt:1/60`, `substeps:4`, `solver_iterations:8`, `lethal_walls:false`, `lethal_bodies:false` |
| Body | `position`, `velocity:[0,0]`, `angle:0`, `omega:0`, `radius:0.5`, optional convex local `vertices`, `mass:1`, optional `inertia`, `drag:0.15`, `angular_drag:2`, `thrust:12`, `torque:8`, `restitution:0.25`, `friction:0.3`, `controlled:false`, `cargo:false` |
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
Old scenes with explicit body fields continue to work.

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
2D/3D switching and **Follow agent** (selected body, or the first controlled
agent), and middle/right-drag panning are available independently of diagnostics. **Whole arena** resets the camera.

`visuals/vehicles.js` authors distinct Kestrel rockets, Mite electric karts and
Wisp survey drones, with engine bells, fins, cockpit glass, wheel hubs, roll
cages, lights and rotors. `models.js` supplies veined ore, recovery docks, flux
gates and gravity reactors. The runtime
constructs these assets directly, adapting hull geometry to scene definitions.
`tools/build-control-assets.mjs` also exports them as reusable files in `assets/`.
No external images, fonts or asset services are needed. See
[`assets/README.md`](assets/README.md) for provenance.

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
# With the local server running and Chromium installed for Playwright:
cd fractal-gas-web
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-replay-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-experiments-browser.mjs
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ node tests/control-racing-browser.mjs
```

Native tests cover conservation/free flight, gravity, elastic and breaking
tethers, restitution, fast collisions, holes, task/RNG restoration, simultaneous
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
