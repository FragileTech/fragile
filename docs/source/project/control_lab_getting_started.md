(sec-control-lab-getting-started)=
# Getting started with the control laboratory

Use the book's **Full** reading mode for this practical guide. **Expert** mode
hides its explanatory instructions, tables, and figures.

:::{div} feynman-prose
Start with one world you can see. The controller makes copies of that world, tries
possible actions in the copies, and chooses an action for the world on screen.
The interesting part is watching how those imagined futures influence the motion
that actually happens. This guide gets you from a source checkout to that first
experiment, then puts you in the kart yourself.

Use {doc}`control_lab_controls` for every control and diagnostic,
{doc}`control_lab_scenes` for building worlds, and {doc}`control_lab_replay` for
saving and revisiting runs. {doc}`control_lab_experiments` covers comparisons;
{doc}`control_lab_architecture` explains the implementation and extension points.
The original {doc}`control_laboratory` remains a technical reference.
:::

(sec-lab-start-mental-model)=
## Know what is being copied

:::{div} feynman-prose
Imagine two rockets carrying one asteroid. A possible future must contain both
rockets, the asteroid, their connections, and the task progress. Copying only one
rocket would leave out the very things determining its next movement. A **world**
is that complete moving system. The static scene describes its boundaries,
geometry, and physical parameters once; those definitions are shared by copies.

A **body** is one physical object within a world. A **controlled body** has action
channels, such as throttle and steering. A **joint action** supplies the channels
for all controlled bodies at once. The laboratory therefore supports coordinated
control without pretending that neighboring agents are fixed scenery.

A **walker** is a planning candidate with its own complete world state. Setting
**Walkers** to 128 does not put 128 additional karts on the circuit: it gives the
planner 128 alternative worlds. These alternatives appear as paths or a
future-state cloud when the chosen controller provides them.

A **physics frame** advances the world by the scene's fixed time step. Every
supplied preset uses 1/60 second. **Action frames** says how many such frames one
selected action lasts. A **decision** is a controller choice committed to the
executed world. With **Action frames = 6**, an ordinary completed decision advances
0.10 seconds unless the world becomes terminal before all six frames complete.
The screen's rendering frames are a separate clock.

For FMC, **Horizon = 16** requests 16 Wave iterations. With six physics frames per
iteration, a surviving search trajectory can look 96 frames, or 1.6 simulated
seconds, ahead. Only the chosen first action is executed; the next decision starts
from the resulting world. Other controllers interpret their search budget through
their own rollout and optimization loops; see {doc}`control_lab_controls`.
:::

(sec-lab-start-build)=
## Build and open the browser

:::{div} feynman-prose
Run the following from the repository root. You need a C++17 compiler, CMake 3.16
or newer, Node.js 20 or newer with npm, the repository's `uv` Python environment,
and an activated Emscripten SDK providing `emcmake` for the browser build. The lab
has been built with Emscripten 6.0.8. No emulator ROMs or emulator submodules are
needed for this continuous-control application.

The first command builds the native library for Python and native checks. The
second builds both browser engine variants, installs the pinned JavaScript
packages, copies the docs branding, and generates the lab assets. The server
command stays running in its terminal.
:::

```bash
make control-native
make control-web
make control-lab
```

:::{div} feynman-prose
Open [http://127.0.0.1:8080/lab/](http://127.0.0.1:8080/lab/).
For a different port, replace the last command with:
:::

```bash
CONTROL_PORT=8089 make control-lab
```

:::{div} feynman-prose
Wait for **Step** and **Run experiment** to become available. The top-right backend
indicator reports the actual thread count, for example **4 THREADS / WEBASSEMBLY**.
The first page load automatically demonstrates one planned action; use **↺ Reset**
to inspect the initial state at tick zero.

The native library alone cannot supply the browser engine. Conversely, the
browser does not need the native shared library to run once `make control-web`
has produced its WebAssembly files. After changing C++ engine sources, rebuild
`make control-web`; after changing only lab assets or the docs logo, run
`npm --prefix fractal-gas-web run build:lab`. The server supplies local files
without requiring an external asset service.
:::

:::{figure} ../../_static/control_lab/circuit-overview.png
:alt: Fragile Tech laboratory showing Violet Circuit, controller settings on the left, a kart track in the center, and recording controls around the world view.
:class: feynman-added

The circuit view places planning controls beside the executed world. Paths and
clouds are possible futures; world replay follows the movement actually recorded.
:::

(sec-lab-start-first-run)=
## Make a reproducible first run

:::{div} feynman-prose
Use this small procedure to establish what a decision does before experimenting
with larger populations. Each configuration change reloads the current scene and
clears the current in-memory history. Set the configuration first, then collect
measurements or export a recording.

1. Choose **Violet Circuit · kart racing** under **Environment**.
2. Set **Controller** to **Fractal Monte Carlo**, **Clock** to **Reproducible · wait for planning**,
   **Walkers** to **128**, **Horizon** to **16**, **Action frames** to **6**, and
   **Seed** to **7**. These numeric values are the initial page defaults.
3. Open **Planner settings** and leave **Worker threads** at **4** initially.
   Keep **Keyboard control** unchecked so held keys cannot replace planned input.
4. Press **↺ Reset** and wait for the controls to become ready. The circuit starts
   with zero completed laps and its first checkpoint as the target.
5. Press **Step** once. The worker completes one search and commits one action.
   On this fresh circuit the world reaches **TICK 000006**, or **0.10 s**.
   The cloud and any recorded paths describe the search that selected the action.
6. Press **Run experiment** to repeat decisions. Press **Pause experiment** to
   stop continuous execution. **Step** also pauses continuous running before
   requesting its single decision.
7. Press **↺ Reset** to begin again with the same scene, settings, and seed.
   Use **Export run** before resetting if you want to preserve the current session.

A repeatable setup specifies the scene, seed, controller, and all its settings.
Reproducible mode lets a decision finish regardless of how slowly your machine
computes it. It is the useful starting point for comparing trajectories; it does
not promise that a particular planner setting will solve every scene. The
**Experiments** tool adds explicit trial seeds, budgets, and reports when you are
ready to make a measured comparison.
:::

(sec-lab-start-worlds)=
## Choose what to investigate

:::{div} feynman-prose
The six presets exercise different parts of the same engine. The environment
picker changes the scene; the controller picker changes how actions are chosen.
A preset's task description tells you what to pursue, while an experiment's
success criterion specifies exactly what counts as a successful trial.
:::

:::{div} feynman-added
| Environment | Controlled bodies | What to try and watch |
|---|---:|---|
| **Asteroid harvesting** | 1 | Hook ore and deliver it to a base; watch local gravity and tether motion. |
| **Ants & drops** | 48 | Collect food that respawns from seeded randomness; the joint action has 144 channels. |
| **Tandem flight** | 2 | Visit ordered checkpoints while maintaining a formation; observe the cost of poor coordination. |
| **Collaborative mining** | 2 | Move a shared heavy asteroid with two elastic tethers; both rockets affect the same load. |
| **Thinking graphs** | 1 | Inspect tethered rocket search, cloning, ancestry, and short continuation risk. |
| **Violet Circuit · kart racing** | 1 | Reach 16 checkpoints in order for one lap; compare manual and planned kart control. |
:::

:::{div} feynman-prose
The circuit is eight metres wide. Its walls and infield hole determine collisions;
paint, curbs, and the glowing gantry supply visual orientation. The highlighted
checkpoint is the next required proximity zone. A lap is counted after all 16
zones have been visited in sequence. These are proximity tests, not directional
finish-line crossings: waiting at the finish or skipping checkpoints does not
complete a lap. The circuit's default experiment target is **gates = 16**.
:::

(sec-lab-start-drive)=
## Drive the kart and inspect the view

:::{div} feynman-prose
Choose **Violet Circuit · kart racing**, reset, and enable **Keyboard control**.
Click the world to move focus away from the checkbox or a numeric field. Keyboard
input is ignored while an input, text area, or select element has focus.

Hold **W** for forward throttle; **S** requests reverse throttle. **A/D** steer,
and **Space** applies the brake. Try W on the first straight, then brief W+A input
to follow the left-hand bend toward the next highlighted checkpoint. Brake before
trying a tight turn. Q/E supply lateral input for actuators with that capability;
the kart has no strafe channel.

Manual input pauses the planner and advances two physics frames per keyboard
update, sampled at about 30 updates per second while a mapped key is held. Release
all mapped keys and manual stepping stops; the world does not continue coasting
on its own. Holding Space advances the world while braking. This makes manual
control useful for controlled interventions, but it is not a continuously running
keyboard game loop. Uncheck **Keyboard control** and press **Run experiment** or
**Step** to return control to the planner. Manual movements appear in world replay.

The keyboard acts on the selected body if it is controlled, otherwise on the first
controlled body when there is no body selection. Selecting an uncontrolled body
in the editor does not transfer its input to a different agent. For this
single-kart preset, a fresh scene starts with the kart as the default target.

Use **Follow agent** to center the camera on the selected body, or the first
controlled body if none is selected. Its button becomes **Whole arena**, which
returns to the arena view. Scroll over the world to zoom. Middle-button drag,
right-button drag, or Alt-drag pans and leaves follow mode. **2D / 3D** switches
between top and angled views; the underlying physics stays planar.

**Clean view** hides the diagnostic layers to make the vehicle and track easier
to inspect; **Show diagnostics** restores the previous layer selection. Camera
and display changes do not modify the physical world. Ordinary replay seeking
also changes the displayed world without advancing physics; restoring that frame
for a new experiment uses the separate **Continue here** action.
:::

(sec-lab-start-clock-threads)=
## Choose the clock and parallelism

:::{div} feynman-prose
In **Reproducible · wait for planning**, simulation time waits while the controller
searches. A demanding decision takes more wall-clock time, but the next action
still begins from its intended state. In **Real time · fixed simulation clock**,
the authoritative world advances while the planner works on a predicted future
root. A result is accepted only when its target tick, scene revision, and root
state match the world being controlled.

When a required real-time plan is unavailable at its boundary, the worker uses
the engine's neutral action and increments **Missed deadlines**. Neutral action
is zero clamped into each channel's valid interval, not necessarily a brake.
The scheduler limits catch-up after a long delay, so this mode is not a hard
real-time guarantee. **Step** always requests a completed single decision, even
when the selected running clock is real time.

**Planner settings → Worker threads** accepts **1–64**, default **4**, for live
planning. The caller participates: 64 total threads means up to 63 additional
pthread workers. Physics work is divided across independent candidate worlds;
increasing this number does not change how many bodies a world contains or the
size of its state. Python accepts the same range, for example
`ControlEngine(scene, worlds=256, threads=64)`; see {doc}`control_lab_architecture`.

Keep both browser builds. The authoritative simulation uses the serial module;
the planner can use the threaded one when shared-memory isolation is available.
The supplied server sets the required COOP/COEP headers. The existing isolation
service worker may provide isolation on another host; if isolation or the threaded
module is unavailable, the planner falls back to one thread. Read the backend
indicator to confirm what actually loaded. **Experiments** and comparison trials
currently use one thread independently of this live setting.
:::

(sec-lab-start-troubleshoot)=
## Resolve common startup and performance problems

:::{div} feynman-added
| Symptom | Check or next action |
|---|---|
| `emcmake` is missing | Activate your Emscripten SDK in the build terminal, then rerun `make control-web`. |
| The server reports its port is in use | Choose another port with `CONTROL_PORT=8089 make control-lab`, then open that port's `/lab/` URL. |
| The engine remains unavailable or a module request fails | Build with `make control-web`, serve with `make control-lab`, and reload. Open the HTTP URL rather than opening `index.html` as a local file. Read the status message and browser console for the failing resource. |
| The backend shows one thread after requesting more | Confirm the threaded build exists and use the supplied isolation-header server. Reload; the backend reports actual threads, not just the requested setting. |
| Thread settings or observation layers are missing | Widen the browser above 780 CSS pixels; the compact layout hides advanced sidebar details and layer checkboxes. |
| W/A/S/D does nothing | Enable Keyboard control, click the world, and check that no form field has focus. For the simplest check, reload the racing preset to clear editor selection. |
| Manual motion stops when keys are released | This is the current manual-step behavior. Hold a mapped key to advance frames, or return to the planner for continuous execution. |
| Decisions are slow or real-time deadlines are missed | Start with reproducible mode; reduce Walkers or Horizon and measure again. More threads can add overhead when the batch is small. |
| Rendering is slow while planning is acceptable | Try Clean view and a less crowded scene. Distinguish rendering FPS from measured planning time using the diagnostics in {doc}`control_lab_controls`. |
| Recording stops at its memory budget | Export before resetting, or configure device-backed recording for a longer run; see {doc}`control_lab_replay`. |
:::
