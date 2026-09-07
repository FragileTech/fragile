(sec-control-lab-controls)=
# Controls, planners, and diagnostics

:::{div} feynman-prose
The laboratory has one executed world and a collection of possible futures. **Step**
asks the controller to examine those futures, choose an action, and apply it to the
executed world. The bright vehicle shows what happened; the paths and cloud show
what the planner considered. Keeping those two roles separate makes the controls
much easier to understand.

Start with {doc}`control_lab_getting_started` for installation and a first run.
This page is the operating reference. {doc}`control_lab_scenes` explains the tasks
and editor, {doc}`control_lab_replay` explains recordings, and
{doc}`control_lab_experiments` explains controlled comparisons.
:::

(sec-lab-controls-running)=
## Run, step, reset, and inspect a Wave

:::{div} feynman-prose
Wait until the backend indicator reports **WEBASSEMBLY** and the run buttons become
available. The first page load automatically plans and commits one action, so the
initial view can already contain movement and a search tree. Later preset changes
and resets leave the new world paused.
:::

:::{div} feynman-added
| Control | Operation | What to expect |
|---|---|---|
| **Environment** | Load a preset from the scene catalog. | Rebuilds physics and clears the current in-memory history. |
| **Vehicle type** | Choose **Harvesters** or **Drones** in Ants & Drops. | Rebuilds the original preset with one type for all vehicles, clears run and editor history, and leaves the world paused. Defaults to **Harvesters**. |
| **Vehicle count** | Set the Ants & Drops vehicle count to a whole number from 1 to 128. | Defaults to 5. Applies the same rebuild as **Vehicle type**; invalid entries leave the scene intact. |
| **Run experiment** | Repeatedly plan and execute actions using the selected clock. | Button becomes **Pause experiment**. |
| **Pause experiment** | Stop further execution. | The displayed state remains available for inspection and export. |
| **Step** | Plan once, then execute **Action frames** physics frames. | Pauses continuous running; always waits for this plan, including with the real-time clock selected. |
| **↺** (Reset) | Rebuild the current scene with the current seed. | Resets task progress and recordings; preserves scene edits and selected settings. |
| **Advance Wave population** | Advance the native FMC population by one Wave iteration. | Displays population row zero as the world, with a labeled **Wave selection** recording cut. |
:::

:::{div} feynman-prose
The Ants & Drops vehicle controls appear beneath **Environment**. Their selections
persist when switching presets within the tab session, and **Reset** retains the
current scene. **Vehicle count** counts physical vehicles; **Walkers** counts
planner candidates, each representing a possible future for the whole group.

**Advance Wave population** is an inspection tool for the native Fractal Gas search.
Its first click initializes a population from the current world; later clicks
advance that same population. It runs native FMC even when another controller is
selected. It does not execute the usual mean first action in the original world.
The displayed row is one population member, not a declaration of the best future.
Wave runs in the serial simulation worker; **Worker threads** controls the separate
live planner. Selected-action risk is not evaluated in this mode.

Use **Reset** before switching from a Wave demonstration to an ordinary control
trial. Wave checkpoints can preserve its population; see
{doc}`control_lab_replay`. The authoritative world stops continuous running when
its terminal flag is set. A collision only ends an episode if that scene's physics
and task rules make it terminal.
:::

(sec-lab-controls-clock)=
## Simulation time and parallel execution

:::{div} feynman-prose
A physics frame has duration `scene.physics.dt` seconds, with `1/60` second used by
the browser when it is omitted. **Action frames** is an integer count of these
frames. With the defaults, one action lasts `6 / 60 = 0.1` simulated seconds and a
16-action horizon reaches nominally `1.6` seconds ahead. Terminal worlds can stop
earlier. Increasing the horizon spends more computation looking ahead; it does not
change the physics step size.

**Reproducible · wait for planning** holds the executed world still until the
requested search finishes. It then applies the action for the configured duration.
Simulation can run faster or slower than wall time. Use this mode to compare
decisions without making CPU scheduling part of the control problem.

**Real time · fixed simulation clock** advances the world on a scheduled physics
clock while the planner works separately. The planner starts from a prediction of
the next action boundary under the current action. A result is accepted only when
its scene revision, target tick, and complete root snapshot match the executed
world at that boundary. If no acceptable result arrives, the simulation uses the
neutral action and increments **MISSED DEADLINES**. Neutral means zero clamped to
each channel's valid bounds; it need not cancel velocity, gravity, or momentum.

The search budget is the action duration in milliseconds minus 20 ms, with a minimum
of 1 ms. The planner checks this budget between incremental advances; one advance
and the subsequent risk probes can overrun it. The simulation catches up at most
four physics steps per timer callback, then reschedules if still behind. Browser
throttling therefore prevents this mode from being a hard real-time guarantee.
:::

:::{div} feynman-added
| Session control | Default and range | Meaning |
|---|---|---|
| **Clock** (`clock`) | `reproducible`; alternative `realtime` | Worker scheduling policy. Changing it restarts the current scene. |
| **Worker threads** (`threads`) | 8; integers 1–64 | Total native simulation threads in the live planning engine, including its calling thread. |
:::

:::{div} feynman-prose
The browser prewarms the selected pthread pool: 64 total threads means up to 63
additional pthread workers. The authoritative world, prediction world, risk batch,
and **Experiments** worker remain serial. JavaScript sampling and optimizer updates
also run in their planning worker rather than across those native threads.
More threads help only when there is enough batch work and hardware capacity.
Small populations can spend more time coordinating than simulating.

Threaded planning requires the pthread build and cross-origin isolation. If either
is unavailable, the planner falls back to one thread. Read the masthead's actual
thread count after loading; the requested number alone does not establish that
parallelism is active. Build instructions are in {doc}`control_lab_getting_started`.
Clock and thread count are session execution controls, not members of the current
portable planner-settings object. Set them explicitly when reproducing a run.
:::

(sec-lab-controls-settings)=
## Shared settings and FMC controls

:::{div} feynman-prose
Changing a field in **Planner settings**, controller, seed, clock, thread count,
or **Tree** mode rebuilds the scene and clears its current in-memory history.
The coefficients in **Reward terms** instead use **Apply settings**, described
below. Export anything you want to keep first. Algorithm-specific values are remembered while switching
controllers within the current page session. The table gives browser defaults and
UI ranges; lower-level APIs can have different limits.
:::

:::{div} feynman-added
| UI label | Settings key | Default; UI range | Meaning and applicability |
|---|---|---|---|
| **Controller** | `algorithm` | `fmc` | `fmc`, `random`, `cem`, `icem`, or `mppi`. |
| **Walkers** | `walkers` | 128; integers 1–8192 | FMC population or shooting batch capacity. Random control does not use a rollout population. |
| **Horizon** | `horizon` | 32; integers 1–4096 | FMC iterations per decision; action depth per shooting round. Ignored by random action selection. |
| **Action frames** | `frames` | 12; integers 1–60 | Physics frames per candidate action and executed control action. Applies to all controllers. |
| **Seed** | `seed` | 7; integers 0–4294967295 | World reset seed and base planner seed; successive decisions derive seeds by adding the decision count modulo `2^32`. |
| **Diversity coefficient** (in **Reward terms**) | `distance_coef` | 1; 0–10, increment 0.1 | FMC exponent on rescaled observation distance in cloning fitness. |
| **Reward coefficient** (in **Reward terms**) | `reward_coef` | 1; 0–10, increment 0.1 | FMC exponent on rescaled reward signal in cloning fitness. Does not edit scene reward weights. |
| **Action noise** | `noise` | 0.2; 0–10, increment 0.05 | FMC Gaussian standard deviation in each channel's action units when perturbing inherited actions. |
| **Elites** | `elites` | 0; integer 0–Walkers | FMC elite-bank size. The UI initially caps at 128 and updates that cap when Walkers changes. |
| **Perturb inherited actions** | `inertial` | Checked (`true`) | FMC: after the first iteration, perturb the selected companion's inherited action; unchecked samples fresh uniform actions every iteration. |
:::

:::{div} feynman-prose
**Fractal Monte Carlo** repeatedly compares population members using observation
distance and accumulated reward, copies selected companion worlds, and advances
their actions. The first actions travel with their descendants. At the end, the
controller averages those inherited first actions over the final population;
cloning supplies the implicit weighting. If no population members survive, it
returns the neutral action. It does not simply choose the highest-reward leaf.

The two **coefficient** fields are exponents in a product of rescaled distance and reward
signals, not an additive meter of distance plus points. Setting an exponent to zero
removes that factor's variation from this product. **Elites** retains high-reward
states together with their actions and ancestry for reinsertion into later Wave
iterations. It is separate from a shooting controller's elite fraction.

The coefficients remain visible in **Reward terms** and the other FMC-specific
controls remain under **Planner settings** when another algorithm is selected,
but random, CEM, iCEM, and MPPI do not consume them. In
particular, changing **Action noise** does not change iCEM or MPPI exploration.
Their own noise fields are listed next. The algorithm-engine contract is described
in {doc}`control_lab_architecture` and {doc}`control_laboratory`.
:::

(sec-lab-controls-rewards)=
## Reward coefficients and term weights

:::{div} feynman-prose
**Reward terms** is expanded by default. **Diversity coefficient** and **Reward
coefficient** each have a synchronized slider and numeric input. They control how
FMC selects possible futures. The term weights beneath them control the reward
earned by the simulated world: movement, target progress, collisions, pickups,
deliveries, checkpoints, formation, and full loads. Those reward weights also
affect the futures evaluated by the shooting controllers.

Edits stay pending until you press **Apply settings**. This applies the coefficients
and term weights together, preserves the current physical world, discards previous
plans, and starts a new recording under the applied settings. An experiment that
was running resumes; a paused experiment stays paused. Export the previous
recording first if you want to keep it. Recordings and exports use applied settings,
so typing a pending value does not relabel an existing run.

**Distance travelled²** defaults to weight **1**; set it explicitly to **0** to
disable the movement bonus. At each physics frame, each controlled vehicle
contributes its squared displacement, `Δx² + Δy²`, in square metres. The reward is
the weight times the mean of these contributions across the vehicles. Cargo bodies
do not contribute, stationary vehicles contribute zero, and respawn teleportation
does not count as travel. These frame rewards are summed over an action or journey;
the total journey distance is not squared. Movement in any direction earns this
bonus, while **Target progress** separately rewards approaching the task target.

Term weights are saved with the scene; the FMC coefficients retain the planner
settings keys `distance_coef` and `reward_coef`. A scene that omits
`rewards.distance_squared` receives the default weight of 1, including older scene
files. An explicit zero remains disabled. See {doc}`control_lab_scene_reference`
for the scene fields.
:::

(sec-lab-controls-shooting)=
## Random, CEM, iCEM, and MPPI

:::{div} feynman-prose
**Seeded random baseline** samples one independent uniform value within each
channel's declared bounds. It performs no search before executing that joint
action. Its live risk diagnostic still simulates continuations, so selecting random
does not eliminate all diagnostic computation.

Shooting controllers sample whole action sequences, simulate them from the same
root, and use their summed rewards to revise a distribution. They execute only the
first action and plan again from the resulting world. **Search rounds** counts
distribution updates, while **Horizon** counts actions inside each sequence.
Equal walkers and horizon therefore need not mean equal work across algorithms.
:::

:::{div} feynman-added
| UI label | Settings key | Default; range | Applies to |
|---|---|---|---|
| **Search rounds** | `search_iterations` | 3; integers 1–128 | CEM, iCEM, MPPI. |
| **Elite fraction** | `icem_elite_fraction` | 0.1; 0.01–0.5 | iCEM: `max(1, floor(Walkers × fraction))` elites. |
| **Reuse elite fraction** | `icem_keep_fraction` | 0.3; 0–1 | iCEM: fraction of retained elites offered to the next batch. |
| **Population decay factor** | `icem_decay` | 1.25; 1–10 | iCEM: reduce active candidates across rounds, within population and elite constraints. |
| **Noise spectral exponent** | `icem_beta` | 2; 0–4 | iCEM: exponent of the `1/f^beta` noise spectrum; larger values emphasize slower temporal variation. |
| **Distribution momentum** | `icem_alpha` | 0.1; 0–0.99 | iCEM: fraction of the previous distribution retained in mean/std updates. |
| **Initial normalized noise** | `icem_sigma` | 0.5; 0.001–2 | iCEM: initial standard deviation in normalized action coordinates. |
| **Minimum normalized noise** | `icem_min_sigma` | 0.01; 0.0001–1 | iCEM: standard deviation floor; must not exceed `icem_sigma`. |
| **Temperature (reward units)** | `mppi_temperature` | 1; `0.000001`–`1000000` | MPPI: scale for reward differences in exponential weights. |
| **Normalized exploration noise** | `mppi_sigma` | 0.5; 0.001–2 | MPPI: fixed diagonal Gaussian standard deviation. |
:::

:::{div} feynman-prose
**Cross-entropy shooting** starts each decision with a channel-midpoint mean and
half-channel-range standard deviation. It samples independent clipped Gaussian
sequences, retains the top 15% (at least one), and refits their mean and standard
deviation, with a floor of 0.03 in channel units. It returns the first action of the
best sequence in the latest completed round. This basic CEM resets its distribution
at every decision.

**iCEM · improved cross-entropy** adds temporally correlated noise, a shifted
previous mean and elites, elite reuse, distribution momentum, and decreasing active
population across rounds. Retained sequences are evaluated again from the current
root. The final round also evaluates the mean. Reused elites and that mean occupy
slots inside **Walkers**, rather than adding extra worlds. The returned first action
comes from the best evaluated sequence across completed rounds of this decision.

**MPPI · path integral control** shifts its previous optimized sequence, samples a
fixed Gaussian around it, and updates the mean through exponentially weighted noise.
Its weights include the Gaussian importance correction as well as summed reward.
It returns the updated mean's first action. A smaller temperature sharpens the
reward contribution to these weights; it does not change the exploration standard
deviation. Changing the scene's reward scale can change an appropriate temperature.

iCEM and MPPI operate in normalized `[-1, 1]` coordinates and map back to each
channel's bounds. Thus noise 0.5 means half the channel's half-range, before clipping.
Zero normalized input is the channel midpoint, which can differ from neutral input.
In deadline mode, shooting uses a completed round's action when available, otherwise
its current mean. A partial batch's attractive future is not automatically the
action that gets executed.

These baselines produce future-state clouds and executed-world recordings but no
search trees. The random baseline's cloud is just its root state. Large shooting
settings are checked against a 128 MiB sample-array limit; that is one array's
limit, not a bound on total process memory. Reduce walkers or horizon if allocation
is rejected. Use {doc}`control_lab_experiments` to measure quality, simulated work,
and elapsed planning time on the same tasks and seeds.
:::

(sec-lab-controls-view)=
## Camera, manual control, and observation layers

:::{div} feynman-prose
Camera and layer controls change presentation without changing physics or resetting
the run. **2D / 3D** switches between overhead and angled views of the same planar
world. Scroll to zoom; middle/right-drag or Alt-drag pans. **Follow agent** follows
the selected body, or the first controlled body when none is selected. Its
**Whole arena** state returns to the arena view. Panning disengages following.

For manual driving, enable **Keyboard control**, then click the world so a text,
numeric, or selection input no longer has focus. Select a body in **Edit scene**
to control that body; otherwise input targets the first controlled body. Manual
input pauses autonomous running and records its actual motion. The keyboard sends
two physics frames about 30 times per wall-clock second while a recognized key is
held. Releasing all keys stops these manual steps; it does not coast the world on
an independent clock. Disable the checkbox before resuming autonomous control.
:::

:::{div} feynman-added
| Input | Channel mapping |
|---|---|
| **W / S** | Positive/negative `thrust`, `throttle`, or body-local `force_x`, clamped to bounds. A forward-only rocket cannot reverse its thrust channel. |
| **A / D** | Positive/negative `torque` or `steering`. |
| **Q / E** | Positive/negative body-local `force_y`. |
| **Space** | `brake = 1` where a brake channel exists. |
| **Actuator channels → Apply action · 1 frame** | Set named sliders in **Edit scene**, then commit one physics frame using the full joint action. |
:::

:::{div} feynman-prose
Channels outside the selected body receive neutral inputs during keyboard control.
The live keyboard adapter recognizes the channel names listed above. Independent
`thruster_N` channels currently require the **Actuator channels** sliders: the native
channel descriptors supplied to the adapter include names and bounds, but not the
thruster geometry needed to combine drive, strafe, and turning input. Unrecognized
custom channels also use sliders until a keyboard mapping is added. Sliders expose
every controlled body's channels, bounds, and current proposed value, with increments
of one two-hundredth of each channel range. Moving a slider alone does not advance
the world.
:::

:::{div} feynman-added
| Layer/control | Default | Visible meaning |
|---|---|---|
| **Rollout paths** | On | Recorded controlled-body paths, available for FMC. Green is at/above mean recorded reward; rose indicates terminal state; violet indicates a tethered path; blue shows other alternatives. Terminal color takes precedence over tether color. |
| **Future-state cloud** | On | Controlled-body positions in the planner's returned world batch, which may be at a partial horizon in deadline mode. |
| **Tethers & formation** | On | Current tether geometry linking bodies. |
| **Collision geometry** | Off | Native hull outlines, useful when a decorative model differs from its collider. |
| **Clean view / Show diagnostics** | Diagnostics visible | Temporarily hides these four layers, then restores their previous visibility. The separate Physics inspector remains independent. |
:::

:::{div} feynman-prose
The renderer samples large trees to draw roughly at most 50,000 path segments.
This drawing limit does not prune the native record. Hiding paths also does not
disable recording. For that, change **Tree** below the world view.
:::

(sec-lab-controls-diagnostics)=
## Read the telemetry and choose recording detail

:::{div} feynman-prose
Three measurements answer different questions: simulation time measures what the
world has done, planning progress measures search work, and risk probes examine
short random continuations of the chosen action. A cloud with many dead worlds
does not by itself measure the risk of the action ultimately selected.
:::

:::{div} feynman-added
| Readout | Exact interpretation |
|---|---|
| **SIMULATION / TICK** | Executed or replayed world tick times `physics.dt`; not wall-clock runtime. |
| **DEAD RATIO** | FMC final Wave terminal fraction. Shooting reports terminal rows divided by allocated Walkers; iCEM's inactive capacity can dilute this fraction. Random has no searched population, so its zero is not evidence of safe exploration. |
| **SELECTED-ACTION RISK** | Fraction terminal in 16 separate continuations: chosen action for `F` frames, then independently sampled uniform joint actions held for `2F` frames, where `F = Action frames`. Display includes sample count and total horizon. |
| **EVAPORATED / CLONED** | Nodes removed by the latest FMC pruning pass, and fraction cloned in the latest Wave iteration. Other controllers do not perform these operations. |
| **AI BUDGET USED** | FMC iterations / Horizon; shooting completed depth advances / (Horizon × Search rounds); random reports 100%. Capped at 100% in the HUD. This is search progress, not CPU utilization. |
| **ITERATIONS / MS** | FMC Wave iterations or shooting depth advances, with measured planner elapsed milliseconds. Live elapsed time includes worker yields and risk evaluation. |
| **SCORE** | Scene-defined task readout: for example deliveries, pickups, gates, or completed laps. See each task in {doc}`control_lab_scenes`. |
| **BYTES / WORLD** | Unpadded mutable world-state size. Excludes shared scene data, serialization header, planning arrays, and graphics. |
| Footer | Body count, joint action dimension, and missed real-time action deadlines. |
:::

:::{div} feynman-prose
The risk statistic is a short empirical test under random continuations, not a
calibrated failure probability under future planner decisions or a safety guarantee.
It has resolution `1/16 = 6.25%`. In Wave mode it is unevaluated. Historical
decisions display their recorded diagnostics when available, not a fresh test of
the currently visible replay frame.

**Physics inspector** shows velocity, external force, nearby contact normals, and
signed spring/damper tether force. Its text summarizes the selected or first
controlled body: velocity in m/s, angular velocity in rad/s, force in newtons,
nearby contact count, and peak absolute tether force. Cyan arrows use velocity
times 0.25 s; amber and magenta use force times 0.05 m/N. Red normals indicate
current nearby geometry, not reconstructed past collision impulses.

The live performance line reports FPS, CPU render submission time, draw calls,
triangles, authoritative-engine world frames per measured stepping second, and
tracked native buffer bytes. Submission time is not GPU completion time, and tracked
buffers exclude some scratch/tree/allocator overhead. In **Experiments**, control
effort means integrated squared channel input, not mechanical energy; planning
world-frame counts and throughput probes provide separate work measurements.
:::

:::{div} feynman-added
| Recording control | Operation |
|---|---|
| **Tree → Pruned** (`recording = 1`, default) | Remove orphan leaves while protecting current walkers, elites, and their ancestors. |
| **Tree → Full** (`recording = 2`) | Retain all FMC search nodes; useful for examining alternatives, with higher memory use. |
| **Tree → Off** (`recording = 0`) | Disable search-tree recording. Executed-world recording continues. |
| **Keep all decisions** | For memory-backed recordings, replace the usual latest-32-decisions window with retention up to the 64 MiB tree payload budget. Capacity stops control with an export prompt; it does not silently discard the oldest retained decision. |
:::

:::{div} feynman-prose
Without **Keep all decisions**, older trees leave the in-memory window when either
32 decisions or the tree budget is exceeded. A single oversized tree is still an
error. Device-backed recordings store trees separately and keep only a bounded
recent window resident. World motion has its own storage budget and remains
available even when a controller provides no tree. Follow
{doc}`control_lab_replay` for world playback, branch reconstruction, snapshots,
planner checkpoints, portable files, device storage, and event markers.
:::
