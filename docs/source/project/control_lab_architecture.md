(sec-control-lab-architecture)=
# Engine architecture and extension guide

:::{div} feynman-prose
The useful question for this engine is: can we copy a possible world, try an action,
and restore exactly what we copied? Every piece of data that can change the answer to
that experiment must travel with the world. Walls and vehicle definitions are shared;
positions, velocities, task progress, and random-generator state are copied. This is
what lets Fractal Gas clone futures without rebuilding a simulator for every walker.

This page explains the implementation contracts. For operating the interface, read
{doc}`control_lab_getting_started`, {doc}`control_lab_controls`, and
{doc}`control_lab_scenes`. For recordings and comparisons, read {doc}`control_lab_replay`
and {doc}`control_lab_experiments`. {doc}`control_laboratory` contains additional native
extension examples and the original technical reference.
:::

(sec-lab-architecture-ownership)=
## Follow the world through its owners

:::{div} feynman-prose
The collision and motion engine is custom C++17. Browser builds compile it to
WebAssembly; Python calls the native shared library through `ctypes`. Three.js draws
the browser view. It does not integrate the bodies or determine their collisions.
Positions, velocities, and rotation evolve in two dimensions even when the camera
shows a three-dimensional kart, rocket, or environment.

The browser separates presentation, authoritative motion, and speculative planning.
The simulation worker is the sole writer of the live world. The planner receives a
snapshot and returns an action; it cannot mutate the simulation worker's world by
writing its own batch. This separation also allows replay to display an old frame
without silently changing the live experiment.
:::

```text
Scene JSON ──compile──> immutable geometry, types, channels, layout
                               │
Browser main thread            │          Python ControlEngine
  UI / editor / renderer       │            ctypes → C API
  replay presentation          │                  │
          │ commands           ▼                  ▼
          └──────> simulation-worker.js       native runtime
                     one live world             world batch
                          │ snapshot
                          ▼
                   planner-worker.js
                     controller plugin
                          │ bounded actions / complete states
                          ▼
                   native planning batch
                     1–64 execution slots
                          │ selected action
                          └─────────> simulation worker commits motion

Experiments → experiment-worker.js → separate serial runtimes → reports/replays
```

:::{div} feynman-prose
`main.js` connects the panels; `simulation-worker.js` handles live stepping, the clock,
and captured motion; `planner-worker.js` runs incremental decisions. The main thread
uses `renderer.js` for presentation and `replay-panel.js` for playback. Recording and
storage are separate modules (`motion.js`, `playback.js`, and `storage/`), so saving a
world does not require a particular visual model or controller.

A native runtime has one caller. Give simultaneous Python callers separate
`ControlEngine` instances; sharing one handle across calling threads is not supported.
`ctypes` releases the Python GIL during native calls, and each runtime may itself
parallelize independent worlds. Browser workers likewise own separate native runtimes.
:::

(sec-lab-architecture-state)=
## Compile once, copy complete state rows

:::{div} feynman-prose
`src/control/scene.cpp` compiles JSON into immutable body definitions, boundary edges,
zones, actuator programs, extension definitions, and a state layout. Agent inheritance
resolves at this point. Geometry and type catalogs are not duplicated into each world.
The compiled scene has a fingerprint used to check state compatibility.

Each world row stores the physical tick, environment random-generator state, task
counters, terminal status, body poses and velocities, active/delivered flags,
per-agent gate progress, tether attachments and rest lengths, pickup positions and
respawn timers, and any actuator or world-extension auxiliary state. Merely saving
positions would lose, for example, a food respawn timer and change the future after
restoration. Planner populations and the planner's own random generator belong to a
larger planner checkpoint, not to this physical-world row.

Within each row, body kinematics use a structure-of-arrays layout: all x coordinates,
then all y coordinates, then the two velocity components, angles, and angular
velocities. Worlds occupy contiguous rows. This is a per-world structure of arrays,
not one global array of x coordinates across every world. The allocation and row
stride are aligned to 64 bytes. Native state words contain both float32 values and
uint32 bit patterns; treating the entire row as an ordinary numeric feature vector
would corrupt counters and random state.
:::

:::{div} feynman-added
| Quantity | Size and purpose |
|---|---|
| `words` | Meaningful 32-bit words in one complete world; `4 * words` payload bytes. |
| `stride` | `words` rounded up to a multiple of 16 words; includes alignment padding. |
| Raw batch | `worlds * stride * 4` bytes for get/set buffers. |
| Serialized batch | `32 + worlds * words * 4` bytes; no row padding. |
| Violet Circuit world | 16 words = 64 payload bytes; a one-world snapshot is 96 bytes. |
:::

:::{div} feynman-prose
The 32-byte snapshot header identifies the format version, scene fingerprint, world
count, row width, and payload checksum. Serialization preserves word bits in
little-endian order. Deserialization checks all rows before committing any state.
The fingerprint covers scene JSON data; matching geometry alone is not sufficient
to assume compatibility after editing other scene fields.

`get` and `set` copy entire batches through reusable buffers. `gather` takes one source
index per destination row: `[1, 0, 0]` swaps the first two worlds and clones the old
world 0 into the third. Reads come from the original source bank, so this operation
has simultaneous semantics. Native stepping and gather use separate source and
destination banks; retaining Python leases may require allocating a new destination.
`broadcast` restores a one-world snapshot into every row of a planning batch. It is
useful at a search root, but includes snapshot validation and temporary storage; it
is not a zero-copy operation.
:::

(sec-lab-architecture-python)=
## Work with batches from Python

:::{div} feynman-prose
Import `ControlEngine` from `fragile.fractalai.control`. Construction accepts a scene
dictionary, JSON string, or file path; defaults are one world, one thread, and seed 7.
Choose `worlds` for batch capacity and `threads` for execution parallelism. Actions
have shape `[worlds, action_dim]`; derive `action_dim` and bounds from the engine,
not from a presumed number of channels per vehicle. `dt` in `step_batch` is an integer
number of physics frames, while `scene.physics.dt` is seconds per frame. A scalar
frame count broadcasts across worlds; an array sets durations individually.

Run this example from the repository root after building the native library as in
{doc}`control_lab_getting_started`. It gives four worlds different velocities,
clones them simultaneously, and checks that replaying the same actions restores the
same serialized result. One controlled body per world is enough to demonstrate the
batch interface.
:::

```python
import numpy as np
from fragile.fractalai.control import ControlEngine

scene = {
    "size": [64, 44],
    "bodies": [{"position": [20, 20], "controlled": True, "drag": 0}],
}
with ControlEngine(scene, worlds=4, threads=4, seed=7) as engine:
    state = engine.get_states()  # Owned, writable copy.
    state.kinematics[:, 0, 2] = [1, 2, 3, 4]  # vx, metres/second.
    engine.set_states(state)
    engine.gather_states([1, 0, 0, 3])
    np.testing.assert_array_equal(
        engine.get_states().kinematics[:, 0, 2], [2, 1, 1, 4]
    )

    root = engine.serialize_states()
    actions = engine.neutral_action()  # float32 [4, action_dim].
    engine.step_batch(actions, dt=[1, 2, 3, 4])
    expected = engine.serialize_states()
    transitions = engine.transition_results()
    engine.deserialize_states(root)
    engine.step_batch(actions, dt=[1, 2, 3, 4])
    assert engine.serialize_states() == expected
    np.testing.assert_array_equal(transitions[:, 1], [1, 2, 3, 4])

    buffer = np.empty((engine.worlds, engine.stride * 4), dtype=np.uint8)
    engine.get_states(out=buffer)  # Reuse this allocation on later reads.
    print(engine.descriptor())
```

:::{div} feynman-prose
`BatchState.data` is a uint8 array of shape `[worlds, stride * 4]` carrying scene
identity. Its `kinematics` view has shape `[worlds, bodies, 6]`, ordered x, y, vx, vy,
angle, omega. The default `get_states()` owns a writable copy. `get_states(copy=False)`
returns a read-only leased view that remains valid after later engine writes or even
after the engine closes. The native owner detaches storage before overwriting a leased
bank. Release views you no longer need; keeping many historical leases keeps their
buffers alive. Use `state.copy()` before editing a borrowed state.

Observations are a derived float32 array, not a restorable state. The built-in vector
contains seven normalized kinematic values per body, then per-controlled-body gate
counts, tether attachment/rest-length pairs, and appended extension observations.
Position uses the maximum scene dimension as scale, velocities use 20 m/s, and angular
velocity uses 10 rad/s; angle is represented by cosine and sine. These observations
omit information such as the environment RNG and pickup respawn timers.
:::

:::{div} feynman-added
| Python API | Result or use |
|---|---|
| `reset(seed=7)` | Reset every world and return copied state. |
| `get_states(copy=True, out=None)` / `set_states(state)` | Read/restore complete scene-compatible rows. |
| `gather_states(indices)` | Simultaneous row selection and cloning. |
| `serialize_states()` / `deserialize_states(bytes)` | Portable physical-state payload for the matching scene and batch size. |
| `broadcast_snapshot(bytes)` | Restore a one-world snapshot into every destination world. |
| `neutral_action()` | Zero projected into each channel's valid interval; may be nonzero. |
| `step_batch(actions, dt=1)` | Advance native physics; return metrics. Durations are integers from 0 to 4096 frames. |
| `transition_results(out=None)` | Float32 `[worlds, 4]`: reward, actual frames, terminal flag, collisions. |
| `observations(out=None)` | Float32 `[worlds, observation_dim]` derived observations. |
| `metrics()` | Last-transition aggregates plus first-world task counters and planner statistics. |
| `begin_plan(seed=7, **settings)` / `advance_plan()` | Start native FMC and advance incrementally; `True` means complete. |
| `selected_action()` / `plan(**settings)` | One joint action; `plan` runs FMC to completion and returns `(action, metrics)`. |
| `wave_step()` | Advance the active native Wave population. |
| `checkpoint()` / `restore_checkpoint(bytes)` | Save/restore native world and search computation state. |
| `exploration_tree()` / `replay_node(node_id)` | Export FMC ancestry or restore a recorded future. |
| `profile(reset=False)` | Timings, completed world-frames, transfer counts, and tracked buffers; optionally reset after reading. |
| `inspect(actions=None)` | First-world force/contact/tether vectors, shape `[n, 8]`, without stepping. |
| `raycast(origin, direction, distance=1000)` | Distance in metres to static boundary/hole edges, capped by the requested distance; excludes dynamic bodies. |
| `descriptor()` / `close()` | Discover dimensions/capabilities; release runtime ownership. |
:::

:::{div} feynman-prose
`ControlEnv` in `env.py` adapts the engine to the existing Python Fractal Gas interfaces,
including per-walker `ControlState` objects and observations. Use it to compose those
algorithms, with `record_frames=False` because RGB rendering belongs to the browser.
Use `ControlEngine` directly when avoiding per-walker Python objects matters. Python
`plan()` invokes native FMC; the JavaScript CEM, iCEM, and MPPI plugins are available
through browser and Node hosts, not through that Python method.
:::

(sec-lab-architecture-parallelism)=
## Parallelism and reproducibility

:::{div} feynman-prose
`src/thread_pool.*` provides fixed execution slots with static block partitioning.
Each slot has its own physics scratch storage, while compiled scene definitions are
shared. A setting of 64 means the caller plus up to 63 worker threads. Empty blocks
are allowed when the batch is smaller than the pool. Threads advance independent
worlds; they do not distribute one world's contact solve across 64 workers.

In the browser, **Worker threads** selects 1–64 slots for live planning, default 8.
The threaded WebAssembly module prewarms the requested pool. It needs cross-origin
isolation and the separate `control-threaded` build. The planner falls back to the
serial module if those requirements are unavailable. Its 16-world risk diagnostic
uses one slot. The authoritative simulation, current Experiments worker, and benchmark
CLI use the serial module. More threads can cost more than they save on small batches;
use measured planning latency and world-frame throughput to choose the setting.

Complete physical state and deterministic callbacks allow repeated transitions to
match when scene, backend, input actions, durations, and initial state match. Native
tests check thread-count determinism. This does not promise bitwise equality between
arbitrary compiler versions, CPU architectures, and WebAssembly implementations.
Planner checkpoints additionally identify the backend and preserve search RNG and
optimizer memory. Match the backend when resuming computation.

The reproducible clock waits for search; the real-time clock may commit a fallback
when a deadline is missed. Scheduling can therefore change real-time trajectories
even with the same seed. Matching a physical snapshot does not recreate past planner
warm starts or deadline timing. Use the appropriate world or planner record described
in {doc}`control_lab_replay`.
:::

(sec-lab-architecture-controllers)=
## Implement a controller against the engine contract

:::{div} feynman-prose
`web/lab/native.js` adapts the C API. Its `descriptor()` reports version, state layout,
channels, observation dimension, and capabilities. JavaScript actions are flattened
float32 batches of length `worlds * dim`. `states()` and `copyStates(out)` return raw
padded rows; `restoreRows(rows)` restores them. `snapshot()`/`restore(bytes)` handle
serialized batches, `broadcast(bytes)` duplicates a root, and `gather(indices)` clones
rows. `step(actions, frames)` and `results()` provide transitions. Keep rows opaque in
algorithm code; engine adapters and visualization layers own layout knowledge.

Register through `controllers/registry.js`. `instantiateController(id, engine,
settings, scene)` accepts a host-owned engine; `createController(module, scene,
settings, threads)` allocates the native adapter. Optional `worlds(settings)` selects
batch capacity, otherwise one world is allocated. The required methods are
`begin(root, seed)`, `advance()`, and `result()`. `advance()` returns whether the
decision is complete and should do bounded work so cancellation and deadlines can be
handled between calls. `result()` must remain usable when a deadline ends search.

A result needs one finite, bounded joint `action` of length `engine.dim`. Optional
`tree`, `cloud`, `metrics`, and `budgetUsed` populate diagnostics; the planner supplies
defaults when omitted. Action dimension is independent of the tree's pose dimension.
Optional `checkpoint()`/`restore(saved)` preserve algorithm memory, RNG, partial work,
and root; `dispose()` releases controller-owned resources. Native engine checkpoints
alone do not save a JavaScript optimizer's arrays or warm-start plan.
:::

### A complete small controller plugin

:::{div} feynman-prose
Save this as `web/lab/controllers/fixed-fraction.js` under `fractal-gas-web`, and add
`import "./fixed-fraction.js";` to `controllers/index.js`. All supplied JavaScript hosts
load that shared entry point. This diagnostic controller uses the same fraction of
every channel's interval, with no search. A fraction of 0.5 means channel midpoint,
which may produce thrust; it is not necessarily a zero-action controller.
:::

```javascript
import { registerController } from "./registry.js";

registerController("fixed_fraction", {
  label: "Fixed channel fraction",
  parameters: {
    fixed_fraction: {
      label: "Channel fraction", default: 0.5, min: 0, max: 1, step: 0.05,
    },
  },
  create: ({ engine, settings }) => {
    const fraction = settings.fixed_fraction ?? 0.5;
    if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1)
      throw new Error("Channel fraction must be in [0, 1]");
    const action = Float32Array.from(
      engine.channels, (c) => c.low + fraction * (c.high - c.low),
    );
    return {
      begin(root) { engine.restore(root); },
      advance() { return true; },
      result() { return { action: action.slice(), budgetUsed: 1 }; },
    };
  },
});
```

:::{div} feynman-prose
The parameter descriptor supplies numeric controls under **Planner settings**. Factory
validation still matters because benchmark specifications can bypass the UI. This
example has no checkpoint methods; a checkpoint request reports that limitation.
For search implementations, inspect `controllers/builtins.js` for FMC/CEM,
`controllers/shooting.js` for shared batched rollout machinery, and `icem.js`/`mppi.js`
for optimizer-specific sampling and updates. Keep custom imports explicit rather than
adding agent-specific branches to the controller host.
:::

(sec-lab-architecture-extensions)=
## Choose the extension point and validate it

:::{div} feynman-prose
Most new experiments need scene data or one plugin. Choose the component that owns
the change. Paths in the following table are relative to `fractal-gas-web` unless
otherwise indicated. Native plugins must be registered before compiling scenes.
:::

:::{div} feynman-added
| Extension | Definition and registration | Build or import requirement |
|---|---|---|
| Preset / reusable agent | Scene JSON `agent_types`, body `agent_type`, `extends`; `web/lab/scenario-catalog.json` | Save scene JSON and catalog entry; reload. See {doc}`control_lab_scenes`. |
| Declarative visual kit | `visual.model: "kit"`, `parts` with box/sphere/cylinder/cone/ring geometry | Scene data only; physical hull stays separate. |
| Agent model | `registerAgentModel(name, factory)` in `web/lab/visuals/registry.js`; returns a Three.js Group | Import registration before models are created. Tag animated parts with `userData.motion`. |
| Environment renderer | `registerEnvironment(id, factory)` in `web/lab/visuals/environments/registry.js` | Import in `visuals/environments/index.js`; choose `scene.environment.kind`. Factory returns `group`, optional `replacesGates`, and `update(state, info)`. |
| Actuator | `register_actuator(name, compiler, evaluator)` in `src/control/actuators.hpp` | Add source to `fg_control_core` in `control/CMakeLists.txt`, register, rebuild native and both WASM targets. |
| Task / reward / sensor | `register_world_extension(name, compiler)` in `src/control/extensions.hpp`; select in scene `extensions` | Same native registration and rebuild requirements. |
| Controller | `registerController(id, definition)` in `web/lab/controllers/registry.js` | Import plugin in `controllers/index.js`; no native rebuild for JavaScript-only logic. |
| Score readout | `registerSceneMetric(id, read)` in `web/lab/scene-presentation.js`; configure `scene.presentation` | Import registration into the presentation host; affects display. |
| Experiment success | `registerEvaluationMetric(name, evaluate)` in `web/lab/experiments.js`; configure `evaluation` or trial goal | Import registration in every experiment runner; evaluator receives metrics and episode statistics. |
:::

:::{div} feynman-prose
Actuator compilers declare named finite-bounded channels, immutable parameters, and
optional initial mutable state. Runtime evaluators receive their state slice through
a resolved function pointer. World extensions run after built-in mechanics each
physics frame, in scene order, and may append observations. All future-affecting
mutable data must live in the packed auxiliary region. Keep callbacks deterministic,
thread-safe, finite-valued, and free of per-step allocation. A hidden mutable global
would make clones depend on execution order. See the
{ref}`native extension contracts <sec-control-extension-contracts>` and the preceding
actuator example for callback signatures and state-offset access.

Visual animation tags are `thrust`, `steer`, `wheel`, and `rotor`. They derive appearance
from recorded time, speed, and actions; wheel rotation is a visual indicator, not an
additional simulated state. New visual assets do not change collision geometry.
Changing the underlying planar rigid-body model requires extending or replacing the
physics implementation, while keeping complete-state and batch contracts intact.

After a change, run checks appropriate to its boundary. A native plugin needs tests
that restore state, clone it, and continue with identical inputs, including its own
mutable fields; also test several thread counts. A controller needs action-bound,
seed, cancellation, and checkpoint tests when those capabilities are provided. A
visual or scene change needs compilation and browser inspection. Use these repository
commands from the project root; build instructions are in
{doc}`control_lab_getting_started`.
:::

```bash
make control-test
npm --prefix fractal-gas-web run test:lab
uv run pytest -q tests/fractalai/test_control_engine.py
# With the lab server running and Playwright Chromium installed:
CONTROL_TEST_URL=http://127.0.0.1:8080/lab/ \
  node fractal-gas-web/tests/control-browser.mjs
```

:::{div} feynman-prose
`profile()` reports completed world-frames and tracked native buffers; it does not
measure all browser, GPU, allocator, or temporary-export memory. Native buffer reuse
reduces transfer overhead, but increasing world count, bodies, horizon, or retained
history still increases work or storage. Measure on the target hardware rather than
assuming compact snapshots alone guarantee a particular control frequency.
:::
