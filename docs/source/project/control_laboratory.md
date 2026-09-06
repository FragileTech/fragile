(sec-control-laboratory)=
# Continuous-control laboratory

:::{div} feynman-prose
The laboratory runs custom C++ physics in Python and in the browser through WebAssembly.
Its central operation is copying a complete possible world, advancing the copy, and
comparing what happened. The browser adds a cyberpunk view, scene editing, and recorded
world playback and search paths. Its Fragile Tech logo, favicon, and purple accent match
the book. Physics remains two-dimensional when the camera shows
a three-dimensional view. See {ref}`the extension contracts <sec-control-extension-contracts>`
for where new actuation, sensors, rewards, and controllers belong.
:::

(sec-control-laboratory-user-guide)=
## User guide

:::{div} feynman-prose
Start by making one decision and watching what moves. Then learn which settings
change the search, save an interesting trajectory, and compare alternatives from
the same starting world. This is the independent Lab guide, so its navigation stays
focused on operating and extending the laboratory. The following pages are listed
beneath **Continuous-control laboratory** in the Lab guide navigation:

- {doc}`control_lab_getting_started`: build and open the lab, understand worlds and
  walkers, run the six presets, drive the kart, and choose up to 64 threads.
- {doc}`control_lab_controls`: use the controller settings, clocks, observation
  layers, camera, and measured diagnostics.
- {doc}`control_lab_scenes`: create and edit environments, agents, geometry,
  tethers, physical properties, and reusable types.
- {doc}`control_lab_replay`: record movement and thinking traces, restore a world,
  resume a planner checkpoint, and save recordings on the device or in files.
- {doc}`control_lab_experiments`: compare controllers over explicit seeds and
  budgets, fork a selected world, and read the resulting measurements.
- {doc}`control_lab_architecture`: follow the engine and controller interfaces,
  packed-state lifecycle, and extension contracts when implementing new features.

For a first session, read Getting started, then Controls, then Replay. Follow with
Scenes when you want a new task, Experiments when you want evidence about a
controller, and Architecture when you want to extend the implementation. The
reference material below supplies additional API examples and technical detail.

The Lab guide always shows its practical explanations and examples; it does not use
the Theory site's Full/Expert reading switch. From the repository root, `make docs`
builds the documentation portal together with the separate Theory and Lab sites.
`make serve` builds the same assembled documentation tree and opens access to its
portal at [http://localhost:8000/docs/](http://localhost:8000/docs/), where you can
choose either site. The server redirects `/` to `/docs/` and also serves the
laboratory at `/lab/` when its browser build is available. Use `make docs-serve` to
preview an existing documentation build without rebuilding, or
`DOCS_PORT=8001 make serve` to choose another port. The `make control-lab` server
also mounts locally built documentation at
[http://localhost:8080/docs/](http://localhost:8080/docs/).
:::

(sec-control-laboratory-build)=
## Build and open the laboratory

:::{div} feynman-prose
Run these commands from the repository root. The native build needs a C++17 compiler,
CMake, and the repository's Python environment. The browser build also needs Node.js
and npm. `make control-web` reuses an active Emscripten SDK when available; otherwise
it installs SDK 6.0.8 into `.cache/emsdk/6.0.8` in the repository. The first installation
needs internet access. The build activates the SDK in its own subprocess, so you do
not need to activate it in your shell. Set `EMSDK_DIR` to use a different SDK directory.
You can run `make control-setup` separately to prepare the SDK before building.
:::

```bash
make control-native
make control-web
make control-lab
```

:::{div} feynman-prose
Open [the local laboratory](http://127.0.0.1:8080/lab/). `make control-web` builds both
WebAssembly variants automatically: simulation uses the serial module, while planning
can use the threaded module. The supplied server sends the HTTP isolation headers needed for
shared WebAssembly memory. Without isolation, or if the threaded module cannot load,
the planner falls back to one thread. The interface reports the active backend.

**Worker threads** selects between 1 and 64 total simulation threads for the live
planner, with a default of 4. The calling thread participates, so a setting of 64 adds
up to 63 pthread workers. The browser prewarms only the selected worker pool. This
setting controls parallel rollout computation; the **Experiments** worker continues
to use the serial module. More threads do not guarantee faster decisions: the available
hardware and the amount of work in each batch determine whether parallelism pays off.

`npm run build:lab` also copies `docs/logo.png` and `docs/favicon.png` into
`fractal-gas-web/web/lab/branding/` through `tools/build-lab.mjs`. Update the shared
documentation source assets and rebuild to keep the two interfaces aligned.
:::

(sec-control-laboratory-worlds)=
## Worlds, bodies, and joint actions

:::{div} feynman-prose
Imagine duplicating a table with two spacecraft and an asteroid. Four copies of that
table are four **worlds**, each containing three **bodies**. Cloning a world copies the
asteroid, both spacecraft, their tethers, and their task progress together. It never
copies just one spacecraft out of its surroundings.

With `W` worlds, an action batch has shape `[W, action_dim]`. The dimension is the sum
of the controlled bodies' actuator channels. Read `engine.channels` for each channel's
body index, name, and bounds; `engine.action_low` and `engine.action_high` expose the
bounds as arrays. Channels appear in scene-body order. Position coordinates have their
own dimension: a tree records `2*C` position values for `C` controlled bodies, whatever
their action dimension. Controllers must keep action and pose dimensions separate.

Channel intervals may be positive-only, negative-only, or span zero. Python
`engine.neutral_action()` returns a `[W, action_dim]` batch with zero projected into
each interval; JavaScript `engine.neutralAction()` returns the flattened equivalent.
This is a valid default input, which can still produce force when zero is unavailable.

`dt` in `step_batch` is an integer number of physics frames;
`scene["physics"]["dt"]` gives seconds per frame. Native worker threads advance
independent worlds. Give concurrent Python callers separate `ControlEngine` instances.
The native Python engine also accepts 1–64 total simulation threads, for example
`ControlEngine(scene, worlds=256, threads=64)`. The calling thread takes part in this
work, just as it does in the browser planner.
:::

:::{div} feynman-added
| Built-in actuator `kind` | Channels, in order | Meaning |
|---|---|---|
| `vector` (default) | `thrust` `[0, 1]`, `torque` `[-1, 1]` | Forward thrust and turning torque. |
| `kart` | `throttle`, `steering` `[-1, 1]`; `brake` `[0, 1]` | Planar driving with lateral grip and a steering-dependent yaw target. |
| `holonomic` | `force_x`, `force_y`, `torque`, all `[-1, 1]` | Two body-local force components and turning torque. |
| `thrusters` | One channel per thruster | Force at a body-local position; reversible thrusters accept `[-1, 1]`, others `[0, 1]`. |
:::

### Define a reusable agent type

:::{div} feynman-prose
The scene's `agent_types` map collects physics defaults and visual choices. A type can
inherit from another through its `extends` field; each body's `agent_type` selects those
defaults, and fields on that body override them. Types resolve when the scene compiles,
so the registry itself adds no mutable state. The type's `physics.actuator` or the body's
`actuator` selects an actuator, for example `{"kind": "kart", "wheelbase": 1.2}`.

Paste this complete scene into the JSON editor to create a beacon courier. It inherits
the courier's physics, increases thrust in its type definition, and sets mass to `2`
on this particular body. Physics overlays are shallow: a replacement shape array
replaces the inherited array completely.
:::

```json
{
  "version": 1,
  "size": [32, 24],
  "agent_types": {
    "courier": {
      "physics": {
        "controlled": true, "mass": 1, "radius": 0.7, "thrust": 12, "torque": 3
      },
      "visual": {"model": "rocket"}
    },
    "beacon": {
      "extends": "courier",
      "physics": {"thrust": 18},
      "visual": {
        "model": "kit", "color": "#9f82ff",
        "parts": [
          {"shape": "box", "size": [1, 0.45, 0.22], "position": [0, 0, 0.3]},
          {"shape": "ring", "size": [0.5, 0.05], "position": [0, 0, 0.48],
           "emissive": true}
        ]
      }
    }
  },
  "bodies": [{"agent_type": "beacon", "position": [8, 12], "mass": 2}]
}
```

:::{div} feynman-prose
`visual.model` selects the original detailed `rocket`, `kart`, or `drone` models, or a
declarative `kit` such as this one. Kit parts support boxes, spheres, cylinders, cones,
and rings. A body can override `visual` fields too. Changing the visible mesh does not
change its collision shape: edit physics `radius` or `vertices` for that. The editor's
**Agent type** selector uses the types declared in the scene.
:::

### Add a compiled actuator

:::{div} feynman-prose
`fractal-gas-web/src/control/actuators.hpp` defines the extension contract. Register a compiler and an
evaluator before compiling scenes. The compiler validates configuration, declares
channel names and bounds, and can retain immutable data in `ActuatorDef::extension`.
Bounds must be finite with `low < high`. The native stepping path clamps actions to those bounds for
every actuator, including plugins. Put every mutable per-body actuator variable in
`initial_state`; its float words join the packed state and follow cloning, snapshots,
and replay. Actuators without such variables add no auxiliary state words.

For example, this drive integrates its input into one stored force value. The evaluator
receives the body's current motion, local action slice, substep duration, and its own
state slice; it returns world-space force and torque.
:::

```cpp
#include "control/actuators.hpp"

void install_accumulator_drive() {
  using namespace fg::control;
  register_actuator("accumulator_drive",
      [](const Json&) {
        ActuatorDef def;
        def.channels = {{"charge", -1.f, 1.f}};
        def.initial_state = {0.f};
        return def;
      },
      [](const BodyDef&, Vec2, float, float, const float* action,
         float dt, float* state) {
        state[0] += action[0] * dt;
        return ActuatorForce{{state[0], 0.f}, 0.f};
      });
}
```

:::{div} feynman-prose
Compile this source into `fg_control_core`, call the installer before scene compilation,
and select `"actuator": {"kind": "accumulator_drive"}` on a controlled body. Rebuild
each native and WebAssembly target that must understand it. These are compiled C++
extensions, not JSON-loaded executable plugins. Evaluators run concurrently across
worlds: keep immutable data shared, mutable data in the provided state slice, and
avoid allocations in the stepping path. The built-in collision and motion solver
remains two-dimensional; a new visual model does not supply new physics.
:::

(sec-control-extension-contracts)=
### Extend tasks, rewards, and observations

:::{div} feynman-prose
For a new reward rule, task counter, or sensor, use `register_world_extension(name,
compiler)` from `fractal-gas-web/src/control/extensions.hpp`. The compiler returns a `WorldExtension`
with immutable `parameters`, optional `initial_state`, and `step` and/or `observe`
callbacks. Select registered extensions in scene JSON with
`"extensions": [{"kind": "your_extension"}]`.

The frame callback receives the scene, extension, complete mutable row, joint action,
and `StepResult`; callbacks run in scene order after each physical frame. Its state
starts at `row[scene.layout.auxiliary + extension.state_offset]`. The observation
callback reads the row and writes the declared `observation_size` values into its
provided output slice. Those values append to the ordinary observation vector. Compile
the registration into each engine target before loading the scene. Keep callbacks
deterministic and safe across concurrent worlds, with all changing data in the packed
row. This contract adds task and sensor behavior without changing the integrator or
requiring controllers to understand the world's internal layout.
The native engine rejects nonfinite extension state.

Choose the extension boundary that owns the behavior you need:
:::

:::{div} feynman-added
| Change | Extension contract | What remains shared |
|---|---|---|
| Reusable agent defaults | Scene `agent_types`, inheritance, body overrides | Batch API and state ownership. |
| Appearance | `registerAgentModel` factory or declarative visual kit | Physical collision and action definitions. |
| Environment appearance | `registerEnvironment` factory selected by `scene.environment.kind` | Physical scene geometry, state, and controller interface. |
| Task readout | Scene `presentation` metadata and `registerSceneMetric` | Native task counters and episode success criteria. |
| Actuation and actuator memory | Compiled `register_actuator` compiler/evaluator | World stepping, cloning, and snapshots. |
| Reward, task state, or sensor | Compiled `register_world_extension` callbacks | Integrator and generic controller interface. |
| Action selection | JavaScript `registerController` factory | Engine descriptors and batched physics. |
| Episode success criterion | `registerEvaluationMetric` evaluator | Episode runner and report structure. |
| Different collision or motion solver | New compiled engine implementation | The existing built-in solver supplies 2D rigid-body physics only. |
:::

(sec-control-laboratory-states)=
## Save, restore, and clone a batch

:::{div} feynman-prose
A snapshot contains the mutable information needed to continue the world: body motion,
activity flags, tethers, task counters, pickup timers, environmental random state, and
any auxiliary actuator or world-extension state.
Compiled geometry and rendering assets are shared separately. Restoring requires the
same compiled scene identity; save the scene JSON alongside snapshots you keep.
The compiled payload is limited to 100,000 32-bit words per world, including auxiliary
state; this is separate from recording and planner working-memory budgets.

The example below advances four worlds, restores their starting states, and performs a
simultaneous gather. The gather reads every source from the batch before replacement,
so duplicating or permuting worlds does not overwrite a source prematurely.
:::

```python
from pathlib import Path

import numpy as np

from fragile.fractalai.control import ControlEngine

scene = Path("fractal-gas-web/web/lab/scenarios/harvest.json")
with ControlEngine(scene, worlds=4, threads=4, seed=7) as engine:
    initial = engine.get_states()
    snapshot = engine.serialize_states()
    actions = engine.neutral_action()  # [worlds, action_dim]
    for index, channel in enumerate(engine.channels):
        if channel["name"] in {"thrust", "throttle"}:
            actions[:, index] = 0.5 * (channel["low"] + channel["high"])
    engine.step_batch(actions, dt=6)

    engine.set_states(initial)
    engine.gather_states(np.array([2, 2, 0, 3], dtype=np.int32))
    engine.deserialize_states(snapshot)

    output = np.empty_like(initial.data)
    engine.get_states(out=output)
    borrowed = engine.get_states(copy=False)
    positions = borrowed.kinematics[:, :, :2]  # [worlds, bodies, 2]
```

:::{div} feynman-prose
`get_states()` returns an owned copy by default. Reuse `out` to avoid allocating the
Python output array. A borrowed view is read-only and retains its storage, even after
the engine closes; keeping such views forces later writes to detach that storage.
Release views when finished. The portable binary snapshot omits row-alignment padding
and checks its version, scene identity, and checksum. It saves the physical world,
not an in-progress planner's random generator or search population. Bitwise replay is
intended within the same engine build and backend; native and WebAssembly floating-point
results need not be bitwise identical.
:::

### Pause and resume a planner

:::{div} feynman-prose
A physical snapshot answers “where is the world?” A planner checkpoint also saves the
search population, elite bank, inherited actions, tree, iteration progress, and planner
random generator. In Python, use `checkpoint()` and `restore_checkpoint()` when an
unfinished search must continue from the same computational state:
:::

```python
with ControlEngine(scene, threads=4, seed=7) as engine:
    engine.begin_plan(walkers=64, horizon=8, frames=6, seed=7)
    engine.advance_plan()
    checkpoint = engine.checkpoint()
    engine.advance_plan()
    engine.restore_checkpoint(checkpoint)
    while not engine.advance_plan():
        pass
    action = engine.selected_action()
```

:::{div} feynman-prose
In the browser, **Save planner checkpoint** pauses between controller advances and exports
`.fgcp`, including scene and controller settings. For a decision-controller checkpoint,
**Load checkpoint**, then **Step**, continues the saved controller. Built-in FMC,
random, CEM, iCEM, and MPPI controllers include their respective search or random state.
iCEM and MPPI also preserve partly evaluated action sequences, rollout depth, and the
plan carried between decisions; iCEM includes its elite bank and sampling deviations.
If the current world no longer matches the planner's
root when saving, the worker begins a search rooted at the current world first.

When a local Wave population is active, saving preserves that population. After loading,
the interface reports that Wave was restored; **Advance Wave population** continues
the saved population. Ordinary **Step** instead starts the selected decision controller
from the restored physical world.

Planner checkpoints require the same engine build and backend. They do not reproduce
elapsed wall time, browser scheduling, or future real-time deadline outcomes. Timing
counters are not reproducible planner state. Use fixed seeds and the reproducible clock for
algorithm comparisons, and report timing measurements as measurements of that run.
:::

(sec-control-laboratory-planning)=
## Plan actions and replay their ancestry

:::{div} feynman-prose
The browser's **Controller** selector offers FMC, a seeded random baseline,
cross-entropy shooting (CEM), improved cross-entropy (iCEM), and model predictive path
integral control (MPPI). CEM samples joint action sequences, evaluates them in a
native world batch, and refits its sampling distribution to the highest-reward fraction.
Its budget depends on population, horizon, and search iterations; matching one slider
between controllers does not establish equal computation. Random, CEM, iCEM, and MPPI
return empty search trees while still producing world recordings and outcomes. iCEM
and MPPI expose their final rollout-state clouds; these are simulated possibilities,
whereas **WORLD REPLAY** records the actions and motion that actually occurred.

Wave advances a population of possible worlds using the existing C++ fitness and
cloning operators. FMC starts that population from the current physical world, runs a
configured search, selects an action, and starts another search after the action is
committed. The continuous selection averages the inherited first actions in the final
population. If no walkers are alive or no search iteration has run, it returns zero
projected into each channel interval.

Search-tree recording follows the world that was actually cloned. A child's parent is
its selected source's previous node, which may belong to another walker. The root
snapshot and the actions and frame counts along this ancestry reconstruct the branch.
:::

```python
with ControlEngine(scene, threads=4, seed=7) as engine:
    action, metrics = engine.plan(
        walkers=128, horizon=16, frames=6, seed=7, recording=1
    )
    history = engine.exploration_tree()
    history.save("exploration.npz")
    leaf = int(history.metadata[-1, 0])
    replayed = engine.replay_node(leaf)
    engine.deserialize_states(history.root_snapshot)
    engine.step_batch(action, dt=6)
```

:::{div} feynman-prose
Planning and branch replay use a single physical world; the planner creates its own
walker batch. Recording modes are `0` for off, `1` for pruned, and `2` for full. Pruning
removes abandoned leaves while retaining ancestry needed by active walkers and elites.
`history.to_networkx()` exports the tree for analysis. `ControlEnv` adapts the same
physics to existing Python gas interfaces, with per-walker Python objects; use
`ControlEngine` directly for packed batches and `record_frames=False` with the adapter.
:::

### Use iCEM and MPPI

:::{div} feynman-prose
Imagine planning a turn by trying many sequences of steering inputs. Independent
samples can twitch left and right at successive instants. iCEM gives nearby instants
correlated perturbations, so a sample can explore a sustained turn. It fits a new mean
and standard deviation to the elite sequences, retains some elites for another round,
and reduces the sampled population as the search proceeds. The implementation follows
the temporal correlation and memory approach of
[Pinneri et al.](https://proceedings.mlr.press/v155/pinneri21a.html).

After executing the first action, the controller shifts its mean plan and retained
elites forward by one action slot. It repeats the mean's final action to fill the new
tail and samples new elite tail actions. Standard deviations reset for each decision,
allowing exploration around the shifted plan. Retained elites are simulated again from
the current root. Within a decision, the final round also evaluates the current mean;
the action returned comes from the highest-reward sequence found across completed rounds.

MPPI perturbs a mean plan with Gaussian noise and updates it using weighted samples.
High-reward rollouts receive more weight, with temperature controlling how sharply
reward differences affect those weights. The weighting also includes the Gaussian
sampling correction involving the mean and perturbation. This implements the fixed
diagonal covariance version of information-theoretic control described by
[Williams et al.](https://arxiv.org/abs/1707.02342), without their smoothing filter or
learned components. MPPI executes the updated mean's first action and shifts the plan
for the next decision.
:::

:::{div} feynman-prose
Both controllers search in dimensionless coordinates where each channel spans
`[-1, 1]`, then map through that channel's own physical bounds. A normalized standard
deviation of `0.5` therefore means half the channel's half-range. Initial mean zero
maps to the channel midpoint, which need not be zero physical input. iCEM clips sampled
sequences to the normalized bounds. MPPI retains unbounded Gaussian samples for its
sampling correction and update, while clipping actions before simulation and clipping
the updated mean. Neither controller needs to know whether the channels drive a kart,
a rocket, or a newly registered actuator.

Select a controller and open **Planner settings** for its numeric parameters. The same
keys work in experiment variant JSON. Both use `walkers` as batch capacity, `horizon`
as the number of action slots, and `frames` as physics frames per slot. Their shared
`search_iterations` defaults to `3`. These are JavaScript controllers backed by batched
native physics and are available in the browser, experiment worker, and Node benchmark
runner. Python `ControlEngine.plan()` continues to run FMC.
:::

:::{div} feynman-added
| Parameter | Default | Effect |
|---|---:|---|
| `icem_elite_fraction` | `0.1` | Elite count is at least one, otherwise the floor of this fraction of `walkers`. |
| `icem_keep_fraction` | `0.3` | Fraction of the available elite bank reused and re-evaluated in the next round. |
| `icem_decay` | `1.25` | Divides the population target each round, down to twice the elite count, capped at batch capacity. |
| `icem_beta` | `2` | Exponent of the power-law noise spectrum; zero gives independent Gaussian noise. |
| `icem_alpha` | `0.1` | Weight of the previous mean and standard deviation when fitting the elite distribution. |
| `icem_sigma` | `0.5` | Initial normalized standard deviation, reset at each decision. |
| `icem_min_sigma` | `0.01` | Standard deviation floor; must not exceed `icem_sigma`. |
| `mppi_temperature` | `1` | Temperature in accumulated reward units; tune with the scene's reward scale. |
| `mppi_sigma` | `0.5` | Fixed normalized Gaussian standard deviation for exploration. |
:::

:::{div} feynman-prose
The iCEM noise generator uses a padded radix-2 FFT to construct a power-law spectrum
and crops the result to the horizon. Its variance is normalized across the Gaussian
ensemble, not separately for each sampled sequence. Reused elites and the final mean
occupy slots inside the `walkers` capacity; this is an adaptation of the paper's
additive sample counts. Inactive slots receive zero frame durations, saving physics
steps, although the batch interface still copies their state rows. These choices matter
when interpreting timing and comparing implementations.

Keep the controller instance between consecutive decisions to retain its plan. Create
a fresh controller for a new episode or an unrelated world state; `begin()` shifts the
existing plan when called again. Checkpoint restoration preserves that memory explicitly.
These implementations provide configurable baselines; their presence does not establish
that any controller has the best performance on the laboratory's tasks.
:::

### Register a controller

:::{div} feynman-prose
`instantiateController(id, engine, settings, scene)` constructs a controller against
an existing implementation of the engine contract; the host owns that engine.
`createController(...)` is the adapter that allocates a `NativeEngine` and supplies it
to the same factory. A controller supplies `begin(root, seed)`, `advance()` returning
whether the decision is complete, and `result()`. Only the joint `action` is required
in a result; it must contain the channel count's worth of finite, bounded values. The
browser planner supplies defaults for omitted `tree`, `cloud`, and `metrics`. Results
may also supply `budgetUsed`, a fraction of the configured search budget completed;
the built-ins use it for progress display, independently of elapsed wall time.
`advance()` should perform bounded work so
the worker can process cancellation and deadlines between calls. The native adapter
uses optional `worlds(settings)` to select batch size; `checkpoint()` and `restore()`
add checkpoint support, and
`dispose()` releases controller-owned resources.

An optional `parameters` map on the registration describes numeric controls by setting
key, with `label`, `default`, `min`, `max`, and `step` fields. **Planner settings** builds
its inputs from this metadata, so a new controller can expose its options without a
controller-specific UI branch. Validate settings inside the controller as well; the
experiment runner and other callers can supply settings directly.

For a minimal controller that returns the engine's valid default action, create
`fractal-gas-web/web/lab/controllers/coast.js` beside the registry:
:::

```javascript
import { registerController } from "./registry.js";

registerController("coast", {
  label: "Default action baseline",
  create: ({ engine }) => ({
    begin(root) {
      engine.restore(root);
    },
    advance() { return true; },
    result() {
      return { action: engine.neutralAction() };
    },
  }),
});
```

:::{div} feynman-prose
Add `import "./coast.js";` once to `fractal-gas-web/web/lab/controllers/index.js`.
Browser, worker, and CLI hosts load this shared plugin entry point, registering the
controller in each JavaScript context. This minimal controller has no checkpoint
methods. Use `engine.channels` and `engine.descriptor()` rather than assumptions about body geometry
or fixed action dimensions; native state rows remain opaque to the controller.
:::

### Compare controllers on the same task

:::{div} feynman-prose
Open **Experiments**, choose two variants, fixed seeds, an episode frame limit, and a
success metric and target. **Run benchmark** evaluates each variant/seed combination
until success, terminal state, or the frame limit. The report includes success rate,
contact count, completion time among successful episodes, accumulated reward, planning
time, simulator work, and squared channel input integrated over simulation time. That
input measure is an effort proxy; it is not mechanical energy and depends on channel scaling.

Enable **All preset scenes** to run the same variant/seed grid across the scene catalog.
The report includes individual trials, aggregate summaries, and `perScene` summaries.
The UI applies one success criterion to the entire suite; choose a criterion meaningful
across those scenes, or evaluate tasks separately with their own goals.

**Fork and compare** starts both variants from the currently displayed complete world,
including a selected replay frame, and uses the first listed seed. Scrub their separate
recordings together or choose **Play both**. A shorter completed branch holds its final
frame. **Export experiment** saves the report. This is a controlled comparison of
chosen initial conditions, not evidence of general performance from a single episode.

The same `runBenchmark({module, scene, spec})` runner is available headlessly. Save this
specification as `experiment.json` and run the command below from the repository root:
:::

```json
{
  "seeds": [7, 11, 19],
  "maxFrames": 600,
  "goal": {"metric": "deliveries", "target": 1},
  "variants": [
    {"algorithm": "icem", "walkers": 64, "horizon": 8, "frames": 6,
     "search_iterations": 3, "icem_beta": 2, "recording": 0},
    {"algorithm": "mppi", "walkers": 64, "horizon": 8, "frames": 6,
     "search_iterations": 3, "mppi_temperature": 1, "mppi_sigma": 0.5,
     "recording": 0}
  ]
}
```

```bash
node fractal-gas-web/tools/control-benchmark.mjs \
  fractal-gas-web/web/lab/scenarios/harvest.json experiment.json report.json
```

:::{div} feynman-prose
The CLI uses the serial WebAssembly build. Its first JSON file may contain one scene
or an array of scenes; the API equivalent is `runBenchmark({module, scenes, spec})`.
Omitting `spec.goal` uses each scene's `evaluation`, falling back to survival through
the episode limit. Default success metrics are `deliveries`, `pickups`, `gates`,
`survival`, and `reward`; match the criterion to the task. For a
custom metric, register `registerEvaluationMetric(name, evaluate)` from `experiments.js`;
the evaluator receives world metrics and episode statistics and returns the quantity
compared with the positive target. Import its registration in every runner that uses it.
The report's `simulatorFrames` counts completed world-frames in the planner engine,
using its native profile counter. It excludes unexecuted frames after termination;
one frame advanced in each of 64 worlds counts as 64 world-frames. Planning wall time
includes worker yields and therefore depends on hardware and scheduling, even when
trajectories and seeds are fixed. Equal population, horizon, and search-round settings
do not give equal simulator work: iCEM reduces its active population and controllers
can encounter terminal states at different times. Use these reported quantities when
comparing computation; this example does not enforce equal compute budgets.
:::

(sec-control-laboratory-world-replay)=
## Replay the executed world

:::{div} feynman-prose
The **WORLD REPLAY** timeline records the authoritative physical world after every
executed physics frame, independently of whether search-tree recording is off, pruned,
or full. Each frame contains its complete packed state, applied action, and decision
number. State rows omit alignment padding. Cargo motion, tether attachments, task
progress, food respawn timers, environmental random state, and auxiliary extension state
travel with the agents.
Static geometry and visual assets stay outside the frame stream.

Scrub the timeline or press **Play world**, choosing a playback speed from ¼× to 4×.
Playback displays the stored frames without advancing physics. **Back to live** returns
to the latest live world; **Continue here** restores the selected complete state into
the engine, ready for the next action. Snapshot restores, search-branch replays, Wave
selections, and continuations create labeled segment boundaries in the recording.
These cuts can jump between physical ticks; their labels distinguish such changes from
ordinary consecutive motion. **Follow agent** keeps the controlled body in view during
simulation and playback; toggle it back to **Whole arena** for the overview.

The event selector jumps to recorded changes and markers. Use **Add marker** to label
a frame with your own note.
:::

### Store and export longer runs

:::{div} feynman-prose
Enable **Store long runs on this device**, then reset. `StoredMotionRecording` writes
256-frame chunks to IndexedDB, using gzip when browser compression is available and
raw chunks otherwise. Stored chunks are checksummed. The playback cache normally keeps
eight chunks resident and loads older chunks on demand; evicting a cached chunk does
not remove the saved frames. Trees and requested planner checkpoints are stored as
separate records.

Partial chunks flush automatically every five seconds and when the page becomes hidden.
An abrupt close can still lose recent unflushed frames. **Save recording** waits for
queued writes, including the partial final chunk, before reporting completion. **Saved
runs** opens the local library. These records belong to this browser origin; export a
file for a separate copy. **Export run** writes a compressed `.fgcrec` archive for a
device-backed run, and **Open run** imports it. Storage quota or write-backlog errors
are reported; device storage does not promise unlimited capacity or persistence after
browser data is cleared.

Without device storage, **Export run** writes `.fgclab` version 2, containing scene,
world motion, and retained trees. Version 1 tree-only imports remain supported. This
in-memory mode has separate 64 MiB tree and motion payload budgets: trees normally keep
the latest 32 decisions; **Keep all decisions** stops at the tree budget; motion never
discards earlier frames and stops at its budget. Device-backed mode instead keeps a
bounded tree cache and retrieves saved trees when needed. Compressed file size, cached
chunks, and other working buffers are different memory quantities.

**Save state** and **Load state** use `.fgcs` for one physical state. **Save planner
checkpoint** uses `.fgcp` to resume computation. A motion archive supplies recorded
worlds; it need not contain a planner checkpoint for every frame.
:::

(sec-control-laboratory-clocks)=
## Choose a clock and read diagnostics

:::{div} feynman-prose
**Reproducible** mode waits for planning before advancing the physical world. **Real
time** mode keeps the physics clock moving and plans for a future commit tick. A result
is accepted only when its tick, scene revision, and predicted root snapshot match the
actual world. Missed deadlines use zero projected into each channel interval; stale
results cannot replace an action
already committed. Changing the clock therefore changes how much computation is
available before a decision.

Under FMC, the final Wave population's **dead ratio** describes the search population. The
**selected-action risk** instead evaluates 16 continuations of the selected action:
one configured action duration followed by a randomly sampled action held for twice
that duration. Its displayed horizon is therefore three action durations. This is an
empirical terminal fraction for that continuation policy and finite horizon. Sixteen
samples provide a coarse estimate, and zero observed deaths is no safety guarantee.
:::

### Inspect costs and forces

:::{div} feynman-prose
The performance readout separates browser frame rate, CPU render-submission time,
draw calls, triangle count, native world-frame throughput, and tracked native buffers.
Render-submission time is not GPU execution time. **Experiments → Measure batch
throughput** measures batch get/set/gather and stepping in a separate serial WebAssembly
worker. Python `engine.profile(reset=True)` returns accumulated timing, copied-byte,
world-frame, and planning counters, then resets the counters. Tracked native buffers
are selected engine allocations, not total process or browser memory; WebAssembly
linear-memory size is another distinct quantity.

For the standalone native batch benchmark after the native build, run:
:::

```bash
fractal-gas-web/build-control-native/control/fg_control_benchmark > native.csv
```

:::{div} feynman-prose
It varies world count, body count, and thread count, reporting stepping throughput,
gather bandwidth, and snapshot bandwidth without rendering. These measurements describe
that workload and machine; they do not establish a universal speedup.

Enable **Physics inspector** to inspect the selected body, or the first controlled body.
The overlay shows velocity, external force, nearby contact normals, and estimated tether
force. Python `engine.inspect(action)` returns first-world rows with columns
`[kind, body_a, body_b, x, y, vector_x, vector_y, magnitude]`; kinds `0`, `1`, and `2`
mean force, contact, and tether. Inspection does not advance physics, and auxiliary
actuator state is copied before evaluating diagnostic forces. Contact vectors describe
current proximity, not collision impulse history. Inspection itself costs computation;
disable it when measuring the stepping workload alone.
:::

(sec-control-laboratory-experiments)=
## Explore and edit the presets

:::{div} feynman-prose
Use **Step** for one decision from the selected controller, **Run experiment** for
repeated decisions, or
**Advance Wave population** to inspect successive population updates. Toggle rollout
paths, the future-state cloud, tethers, and collision geometry independently.
:::

:::{div} feynman-added
| Preset | Experiment |
|---|---|
| Ants & drops | Joint control of 1–128 harvesters or drones, defaulting to 48 harvesters. Each of the 24 pickups returns at a seeded random playable position three simulation seconds after collection, indefinitely. |
| Asteroid harvesting | Attach cargo and deliver it through an arena with obstacles and gravity. |
| Tandem flight | Coordinate two bodies through sequential gates with a formation reward. |
| Collaborative mining | Move a heavy shared load with two controlled thrusters and tethers. |
| Mining rocket · thinking graphs | Inspect search ancestry, cloning, and collision diagnostics. |
| Violet Circuit · kart racing | Drive one Mite R kart through 16 ordered checkpoints around a closed circuit. |
:::

### Drive Violet Circuit

:::{div} feynman-prose
Choose **Violet Circuit · kart racing** in the environment picker. The circuit is
eight metres wide, with a rounded outer boundary and an infield hole. Asphalt, curbs,
a chequered start line, and a Fragile Tech gantry mark the course; the active checkpoint
shows where to drive next. Select any registered controller and run decisions, or enable
**Keyboard control** and use **W/S** for drive, **A/D** for steering, and **Space** for braking.
**Follow agent** follows the kart, and the 2D/3D view control changes the camera view.

Think of the lap counter as a checklist. The kart must reach each of the 16 checkpoint
zones in order. Every 16 accepted checkpoints adds one completed lap, and the sequence
repeats. Passing the finish alone or skipping ahead earns no lap. These are proximity
zones, so the task does not check the direction of crossing a timing line. Wall contacts
bounce the kart and incur a penalty; they do not end the episode. The scene's default
experiment goal is `{"metric": "gates", "target": 16}`, meaning one completed lap.

The kart uses the existing three-channel `kart` actuator. Lap and checkpoint displays
come from the existing gate counter, so racing introduces no extra native state fields:
this scene uses a 64-byte state row and a 96-byte serialized snapshot including its
header. World replay preserves the kart's motion and checkpoint progress together.
Scrub to an earlier turn and choose **Continue here** to resume from that physical
state, with its corresponding place in the checkpoint sequence.
:::

### Add a scene and its environment renderer

:::{div} feynman-prose
The picker and the benchmark's preset suite read
`fractal-gas-web/web/lab/scenario-catalog.json`. Add a scene JSON file under `scenarios/`
and its `id` and `label` to that catalog to expose another environment. The racing
source is `scenarios/racing.json`; `fractal-gas-web/tools/make-racing-scene.py` generates
its geometry and checkpoints. A scene's `evaluation` sets its default success metric
and target. The experiment panel uses those defaults when first opened for that scene;
the benchmark API uses them when the specification omits `goal`.

Display choices live in `scene.presentation`: `task_label` names the task, `score`
selects a `metric`, `label`, and optional `divisor`, and `progress` selects a metric,
label, and cycle length. `scene-presentation.js` converts these into the readout. Racing
uses gate count divided by 16 for laps and a cycle of 16 for the next checkpoint.
Use `registerSceneMetric` there to expose another metric; displaying a metric does not
change the underlying reward or the experiment's success condition.

Environment renderers live in `fractal-gas-web/web/lab/visuals/environments/`. Register
a factory with `registerEnvironment(id, create)` and import its module in that directory's
`index.js`. The factory receives the scene and returns a `group`, optionally
`replacesGates` and an `update(state, info)` callback. Select it with
`scene.environment.kind`; scenes without a selection use `arena`. The `circuit` renderer
builds its road and walls from the same boundary and holes that physics uses. Its
centerline and start-line decoration use additional visual metadata; update those fields
when changing the track geometry. This keeps visual specialization out of the engine
and the planners.
:::

### Edit and replay a scene

:::{div} feynman-prose
**Edit scene** selects and moves entities, adds bodies and task objects, connects
tethers, and draws outer boundaries or holes. Shift-click selects several entities;
drag moves the selection. **Duplicate selection** also copies tethers whose endpoints
are both selected. Deleting selected bodies removes their attached tethers and remaps
remaining body indices. Use middle/right-drag or Alt-drag to pan, and scroll to zoom.

The numeric property panel edits the selected entity's physical values and nested
extension parameters; **Apply properties** applies them. Entity JSON and complete scene
JSON expose the full configuration. **Save selection as agent type** saves the selected
body as a reusable scene template. **Actuator channels** builds sliders from the native
channel metadata; set a joint action and choose **Apply action · 1 frame**. Keyboard
control addresses the selected agent, or the first agent, using the supported channel
names; sliders also handle custom channels.

Applying scene edits recompiles the scene and resets its state. Undo/redo restores scene
edits, while WORLD REPLAY restores simulated states. Export JSON to reuse the scene in
Python. Render meshes are independent assets: changing appearance does not change
collision geometry or snapshot size.

Use the decision slider and **Replay branch** to reconstruct a search-tree node. The
reconstructed frames appear in a labeled segment of **WORLD REPLAY**, where you can
inspect the resulting motion of every body before continuing the experiment.
:::
