(sec-control-lab-controls)=
# Controls, planners, and diagnostics

:::{div} feynman-prose
The laboratory has one executed world and a collection of possible futures. **Step**
asks the controller to examine those futures and apply its decision to the executed
world: one action for FMC, or a selected trajectory for Wave Jump (one action if
all final walkers are dead). The
bright vehicle shows what happened; the paths and cloud show what the planner
considered. Keeping those two roles separate makes the controls
much easier to understand.

Start with {doc}`control_lab_getting_started` for installation and a first run.
This page is the operating reference. {doc}`control_lab_scenes` explains the tasks
and editor, {doc}`control_lab_replay` explains recordings, and
{doc}`control_lab_experiments` explains controlled comparisons.
:::

(sec-lab-controls-running)=
## Choose a task, run, and inspect a Wave

:::{div} feynman-prose
Wait until the backend indicator reports **WEBASSEMBLY** and the run buttons become
available. A fresh scene starts paused at tick zero; no action runs automatically. The first-visit
chooser introduces the six tasks; **Tasks** opens it again later. Choose a task,
press **Run**, then click a vehicle to inspect the result. The short introduction
can be dismissed, and the browser remembers that choice.

The toolbar keeps the environment, execution controls, and status above the world.
**Setup**, **Controller**, **Rewards**, and **View** organize the settings. Advanced
sections start collapsed and remember their expanded state. On narrow screens,
**Settings** and **Inspector** open drawers so the world remains the main surface.

Changing a physical or planner setting edits a draft. The pending-changes summary
shows the old and proposed values; **Apply and restart** commits them together in
one rebuild. The active world continues to use its existing configuration until
then. **Discard changes** restores that configuration. Nonempty runs are saved on
this device before replacement; a failed save keeps the original run available
and offers recovery choices. See {doc}`control_lab_replay` for storage and exports.

Flight is a three-way scene choice hiding behind one field. If
`environment.flight` is omitted (or null), the lab detects flight when at least one
controlled body is marked `flight_capable`; setting it to `true` forces flight on,
and setting it to `false` forces it off, even for a flight-capable body. In flight,
the environment applies a downward acceleration of **9.81 m/s²** by default (the
`environment.downward_gravity` value). A rocket or drone must therefore spend some
of its thrust countering gravity before that thrust can produce upward acceleration.
:::

:::{div} feynman-added
| Control | Operation | What to expect |
|---|---|---|
| **Environment** | Choose a preset from the scene catalog. | Stages the chosen scene; **Apply and restart** loads it paused after preserving the current run. |
| **Flight override** (`environment.flight`) | Omitted/null, `true`, or `false` | Omitted/null auto-detects flight from a controlled `flight_capable` body; `true` enables flight; `false` disables it. |
| **Downward gravity** (`environment.downward_gravity`) | 9.81 m/s²; scene value | Downward acceleration used in flight mode. A rocket or drone's propulsion must counter it to climb or hover. |
| **Vehicle type** | Choose **Rockets**, **Drones**, **Karts**, or **Harvesters** in any environment or racing track. | Stages physics and visuals for all controlled vehicles while preserving other world edits. **Apply and restart** commits the fleet change paused. Each preset keeps its default until you choose a type. |
| **Vehicle count** | Set the vehicle count to a whole number from 1 to 128 in any environment. | Starts from the preset's count. Stages a fleet change for **Apply and restart**; invalid entries leave the active scene intact. |
| **Problem properties** | Set each controlled agent type's action multiplier from `0×` to `10×`, independently for every compiled degree of freedom. | `1×` keeps the native range; `0×` disables a channel; larger values expand the action range and its built-in physical output. **Apply and restart** commits the draft values to the scene's agent type properties and rebuilds paused. |
| **Run** | Repeatedly plan and execute actions using the selected clock. | Button becomes **Pause**. |
| **Pause** | Stop further execution. | The displayed state remains available for inspection and export. Wave Jump preserves its remaining trajectory for resumption. |
| **Step action / Execute trajectory** | Plan once, then execute one action, or the selected trajectory for Wave Jump; if all final walkers are dead, execute only its first positive-duration action. | Pauses continuous running and waits for planning with either clock. A paused Wave Jump trajectory finishes its remaining actions. |
| **Restart** | Rebuild the active scene with its active seed. | Preserves the previous run, resets task progress, and starts paused. With pending edits the button reads **Apply and restart** and commits those edits together. |
| **Start driving / Step physics frame** | In Drive mode, start continuous physics or apply the current command for one frame. | Start driving becomes **Pause**; releasing keys returns to neutral input while motion continues. |
| **Advance Wave population** | Advance the native FMC population by one Wave iteration. | Displays population row zero as the world, with a labeled **Wave selection** recording cut. |
:::

:::{div} feynman-prose
The vehicle controls appear in **Setup** in all six environments.
The tab remembers your chosen type separately for each environment, with one
shared choice across racing tracks. Changing type keeps the vehicle count and
starting positions, world edits, rocks, tethers, rewards, and environment settings.
An explicit **Flight mode** choice remains in effect; automatic mode detects
flight for rockets and drones. **Restart** retains the current scene.

Imported scenes load unchanged. A mixed or unrecognized fleet displays the
disabled **Mixed / custom** placeholder; choosing a standard type replaces the
whole controlled fleet. **Vehicle count** counts physical vehicles; **Walkers**
counts planner candidates, each representing a possible future for the whole group.

Mining environments expose **Rock size** from **0.1×** to **2×** and **Rock weight**
from **0.01×** to **10×** in their rock settings. Weight can make a rock one
hundred times lighter or ten times heavier without changing its hull or rendered
size. They also expose **Hook stiffness (N/m)**.
The numeric input accepts values from **0** to **1,000,000** and stays synchronized
with a logarithmic slider. These settings update the draft; **Update draft** also
stages the displayed rock properties. Press **Apply and restart** to commit the
value to every tow hook and leave the new world paused. Pending edits do not change
the running physics. The presets retain their stiffness defaults: **35 N/m**
for collaborative mining and **25 N/m** for harvesting.

Beside the rock properties, the **Flight mode** control makes that scene choice
visible without editing JSON. **AUTO** preserves capability-based detection: flight
is enabled when a controlled body is marked `flight_capable`. From **AUTO**, the
first click forces the opposite of the current effective mode: **OFF** for an
auto-detected rocket/drone scene, or **ON** for an auto-planar scene. Subsequent
clicks toggle the forced **ON**/**OFF** state. Either forced choice enters the draft.
**Apply and restart** rebuilds the scene and leaves the new world paused. The control changes the mode
for the rebuilt scene, not the already-running physics.

A low stiffness makes a hook stretch like a rubber band. Raising it makes the
connection approximately fixed in length, but it remains a spring with finite
stiffness. It can still swing: resisting changes in length does not stop a rocket
moving around the rock. At **0**, the spring force disappears while radial damping
remains, resisting relative motion along the hook. Zero therefore does not detach
the hook or disable all of its forces.

**Advance Wave population** is an inspection tool for the native Fractal Gas search.
Its first click initializes a population from the current world; later clicks
advance that same population. It runs native FMC even when another controller is
selected. It does not execute the usual mean first action in the original world.
The displayed row is one population member, not a declaration of the best future.
Wave runs in the serial simulation worker; **Worker threads** controls the separate
live planner. Selected-action risk is not evaluated in this mode. This inspection
button is separate from the **Wave Jump** controller, which executes the chosen
branch through ordinary physics steps in the original world.

Use **Restart** before switching from a Wave demonstration to an ordinary control
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

**Real time · fixed simulation clock**, for controllers other than Wave Jump,
advances the world on a scheduled physics
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

Wave Jump waits for the complete search with either clock: the executed world stays
still while it plans. Real-time mode paces only the selected trajectory's execution.
Each action uses its recorded edge duration, which can be shorter than **Action
frames** when a sampled world terminates. Zero-duration edges are skipped. The
trajectory duration is the sum of these frame counts times `scene.physics.dt`.
:::

:::{div} feynman-added
| Session control | Default and range | Meaning |
|---|---|---|
| **Clock** (`clock`) | `reproducible`; alternative `realtime` | Worker scheduling policy. A change is staged until **Apply and restart**. |
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
The **Controller** panel holds the controller choice and common planner settings.
Controller, seed, clock, thread count, and **Tree** changes enter a shared draft.
Editing Walkers and Horizon, for example, produces two pending changes and one
rebuild when you press **Apply and restart**. Recordings and the running planner
continue to use the active values while you edit. Invalid values must be corrected
before Apply can replace the world; **Discard changes** restores the active values.

Reward-only changes use **Apply to current run**, described below. If the draft
also contains changes that require a restart, **Apply and restart** commits the
whole draft together. Presentation controls in **View** take effect immediately.
Algorithm-specific values are remembered while switching controllers within the
current page session. The table gives browser defaults and UI ranges; lower-level
APIs can have different limits.
:::

:::{div} feynman-added
| UI label | Settings key | Default; UI range | Meaning and applicability |
|---|---|---|---|
| **Controller** | `algorithm` | `wave-jump` | `fmc`, `wave-jump`, `random`, `cem`, `icem`, or `mppi`. |
| **Walkers** | `walkers` | 128; integers 1–8192 | FMC/Wave Jump population or shooting batch capacity. Random control does not use a rollout population. |
| **Horizon** | `horizon` | 64; integers 1–4096 | FMC/Wave Jump normal search depth; Wave Jump can extend it when shared-path execution is enabled. Action depth per shooting round. Ignored by random action selection. |
| **Stop at first bifurcation** | `consensus_prefix` | Checked (`true`) | Wave Jump only: execute the recorded ancestral path shared by every alive final walker, stopping before their branches diverge. |
| **Maximum search horizon** | `max_horizon` | 0 (automatic); integers 0–4096 | Wave Jump shared-path mode only. Zero means twice Horizon, capped at 4096. An explicit nonzero value must be at least Horizon. |
| **Action frames** | `frames` | 12; integers 1–60 | Physics frames per candidate action. Wave Jump executes each selected edge for its actual recorded duration; other controllers execute one action for this count. |
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

**Wave Jump** uses the same population search parameters and cloning procedure as FMC.
**Stop at first bifurcation** is checked by default, including when saved settings omit
this option. An explicitly saved unchecked setting remains unchecked. With
**Stop at first bifurcation** unchecked, after
the search, it selects the alive final walker with the highest accumulated path
reward; ties choose the lower walker index. Alive means nonterminal according to
the native physics. It follows that
walker's recorded parent links back to the root and executes the resulting action
sequence in forward order. This follows the ancestry through cloning, rather than
reading successive actions from one walker slot. Selection always uses accumulated
reward, independently of any reward-signal setting used by resampling.

Imagine the search finds a useful sequence of turns. FMC uses its population to
choose the next turn, then searches again. With the toggle unchecked, Wave Jump commits to the whole selected
sequence before it searches again. If every final walker is dead, it selects the
highest accumulated-reward final walker, with the same tie rule, but executes only
the first positive-duration action for its recorded frame count before replanning.
It therefore makes one planning decision per executed trajectory, including this
one-action fallback. The best final walker need not be the best node ever sampled.
Execution stops if the actual world terminates. An empty executable path stops
with a status message instead of starting repeated searches.

With **Stop at first bifurcation** checked, trace the ancestry of every alive final
walker back toward the root. Execute their shared initial chain and stop before
the first branch where these surviving futures disagree. Agreement means the same
recorded ancestors; two independently sampled edges with similar actions do not
count as agreement. Discarded branches and archived elites outside the current
population do not enter this comparison. A single survivor shares its entire
path with itself, so its full path is executable.

The search first reaches the normal **Horizon**. If the shared chain contains no
positive-duration action, it continues the same population search one iteration at
a time, stopping as soon as an executable shared prefix appears. **Maximum search
horizon** bounds this extension: zero chooses twice the normal horizon, capped at
4096; a nonzero value must lie between the normal horizon and 4096. If agreement
is still absent at that limit, execute just the best surviving path's first
positive-duration action for its recorded duration. If all walkers die, use the
highest-score single-action fallback immediately. Either fallback searches again
after that action unless the actual world terminates. The world stays still
throughout the search and any extension. **Step** completes one search and its
shared prefix or fallback, then pauses.

**Pause** retains the action index and remaining frames. A planner checkpoint
preserves these alongside the world and search state, so restoration can resume
either a search, including its extension, or a partly executed trajectory without
repeating completed work. Checkpoints retain the shared-path setting and effective
search limits; legacy in-flight planner checkpoints without a saved shared-path
setting restore with the toggle off to preserve their original execution. Changing the scene, applied
rewards, algorithm, or world state discards the queued trajectory.

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
FMC and Wave Jump select possible futures. The term weights beneath them control
the reward earned by the simulated world: movement, target progress, collisions, pickups,
deliveries, checkpoints, formation, and full loads. Those reward weights also
affect the futures evaluated by the shooting controllers.

For reward-only edits, press **Apply to current run**. This applies the coefficients
and term weights together. At a physics-frame boundary, the lab prepares replacement
native engines and a planner, discards old plans, and carries across the current
world state, tick, cargo, camera, selection, and decision count. A running
experiment resumes; a paused experiment remains paused. If preparation fails, the
existing world and settings are retained.

Think of the recording as a continuous film with markers, not a stack of unrelated
films. Applying a reward configuration adds a configuration and snapshot boundary;
it does not rewind the world or recalculate earlier rewards. Replay continuation
restores the historical reward configuration at each such boundary, so the old
frames keep the rewards they actually recorded. Typing a pending value therefore
changes neither the running physics nor the existing recording until **Apply to
current run** succeeds.

**Distance travelled²** defaults to weight **1**; set it explicitly to **0** to
disable the movement bonus. At each physics frame, each controlled vehicle
contributes its squared displacement, `Δx² + Δy²`, in square metres. The reward is
the weight times the mean of these contributions across the vehicles. Cargo bodies
do not contribute, stationary vehicles contribute zero, and respawn teleportation
does not count as travel. These frame rewards are summed over an action or journey;
the total journey distance is not squared. Movement in any direction earns this
bonus, while **Target progress** separately rewards approaching the task target.

**Hooked rock travel** defaults to **1 reward per metre**. Its slider runs from
**0–10** in increments of **0.1**; the numeric input accepts **0–1,000**. Set it
to **0** to disable this term, then press **Apply to current run** to apply the change
while preserving the physical world. At the start of each physics frame, the
engine identifies cargo rocks hooked to an active controlled vehicle. It sums
their distances travelled during that frame, `sqrt(Δx² + Δy²)`, and multiplies
the sum by this weight. Each distinct rock counts once, even when several hooks
hold it. Moving a rocket around a stationary rock earns none of this bonus, and
respawn teleportation does not count as travel. Rock motion in any direction
earns it, including circular motion; **Target progress** and **Delivery bonus**
provide the incentive to bring the rock to the refinery.

In mining, **Delivery bonus** pays for bringing a rock into the refinery;
**Target progress** rewards approaching a rock and, once attached, moving the
hauled rock toward the refinery. Progress now measures both ends of each physics
frame against the hauling target selected at that frame's start. Breaking a hook
therefore cannot earn a bonus simply by switching the distance being measured
from refinery distance to rocket-to-rock distance. This removes an incentive for
repeated attachment and breakage; it does not prevent physical swinging around a
rock, or the movement bonus when its weight is positive.

Term weights are saved with the scene; the FMC coefficients retain the planner
settings keys `distance_coef` and `reward_coef`. A scene that omits
`rewards.distance_squared` receives the default weight of 1, including older scene
files. An explicit zero remains disabled. See {doc}`control_lab_scene_reference`
for the scene fields.

The hooked-rock weight is saved as `rewards.hooked_rock_distance`, also with a
default of **1** when omitted and with an explicit **0** preserved.
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
the run. In an ordinary planar scene, **2D / 3D** switches between overhead and
angled views of the same world. In flight mode the control instead switches between
a side-on view (showing altitude against the horizontal direction) and an overhead
view (showing the physics plane from above). These are two projections of the same
state, not two different simulations; the side-on view does not turn a planar
flight model into full three-dimensional physics. Scroll to zoom, then left-drag to bring another part of the environment
into view without changing the zoom. In **Inspect**, a click selects a body;
select **Planner decisions** in the timeline to inspect exploration nodes instead.
In **Edit**, left-drag moves draft scene objects; use middle/right-drag or Alt-drag
to pan instead. These pan shortcuts also work outside Edit. Scene edits support
Undo/Redo and remain a draft until **Apply and restart**. Leaving a dirty editor
offers Apply and restart, Discard, or Cancel. **Follow agent** follows the selected
body, or the first controlled body when none is selected. Panning disengages following so the camera
stays where you put it. The follow button's **Whole arena** state and **Reset view**
return to the center of the collision boundary's bounding box with the largest
view that leaves a 5% margin around the playable arena. Think of this as placing
a snug rectangle around the actual fence, then widening it just enough to leave
breathing room; the rectangle is recalculated for the viewport and applies to
both overhead and flight side views.

**Inspect / Edit / Drive** describes how you interact with the world; **Live /
Replay** describes which world you are looking at. Opening a saved run gives
read-only replay. Use **Create run from this frame** before editing or driving
that historical state.

Enter **Drive** to pause autonomous execution and open the selected vehicle's
controls in the inspector. With no controlled vehicle selected, driving uses the
first controlled body. Selection is shared with camera follow and action guides.
Click the world if a text or numeric field has focus: typing into settings should
not steer a vehicle. The inspector shows the supported keys and pointer-operated
buttons, together with actuator sliders.

**Start driving** starts the worker's physics clock. It advances fixed
`scene.physics.dt` frames independently of keyboard repeat. Releasing the keys
returns the command to neutral; the vehicle can coast, fall, or continue moving
under its existing forces. Neutral input is not a brake. Each timer callback
advances at most five frames and discards excess catch-up backlog, so a slow tab
can run slower than wall time rather than suddenly jumping ahead. Actual executed
motion enters the recording.

**Pause** stops physics and clears held input. Losing focus, hiding the tab,
opening a modal, or leaving Drive also pauses and clears commands. Selecting
**Inspect** returns to paused controller operation, with stale plans discarded
before the next run. While paused, **Step physics frame** or **Apply action · 1
frame** advances exactly one frame using the proposed actuator command.
:::

:::{div} feynman-added
| Input | Channel mapping |
|---|---|
| **W / S** | Positive/negative `thrust`, `throttle`, or body-local `force_x`, clamped to bounds. A forward-only rocket cannot reverse its thrust channel; in flight, thrust must also supply the upward force needed to counter downward gravity. |
| **A / D** | Positive/negative `torque` or `steering`. |
| **Q / E** | Positive/negative body-local `force_y`. |
| **Space** | `brake = 1` where a brake channel exists. |
| **Drive controls → Apply action · 1 frame** | Set named sliders in the inspector, then commit one physics frame while paused. |
:::

:::{div} feynman-prose
Channels outside the selected body receive neutral inputs during keyboard control.
The live keyboard adapter recognizes the channel names listed above. Independent
`thruster_N` channels currently require the **Actuator channels** sliders: the native
channel descriptors supplied to the adapter include names and bounds, but not the
thruster geometry needed to combine drive, strafe, and turning input. Unrecognized
custom channels also use sliders until a keyboard mapping is added. Sliders expose
the selected controlled body's channels, bounds, and current proposed value, with
increments of one two-hundredth of each channel range. In paused Drive, moving a
slider changes the proposed command without advancing physics. During continuous
driving, the new command takes effect on subsequent physics frames.
:::

:::{div} feynman-added
| Layer/control | Default | Visible meaning |
|---|---|---|
| **Animations** (beside **Visual style**) | On; off when the system requests reduced motion and no explicit choice is saved | Enables cosmetic vehicle motion and effects across live views, replay, comparisons, and the workshop. Your explicit choice persists in this browser. Off freezes decorative motion and skips its updates. Static thrust and individual-jet cues, steering, reverse/brake lamps, and enabled action guides still follow current commands; native movement, cargo amounts, pickup visibility, tether connections, and diagnostics still update. Toggling does not reset the simulation or change recordings. |
| **Action guides** (beside **Animations**) | Off; explicit choice persists across Lab tabs | Shows signed command arrows and a numeric readout for the selected controlled body, falling back to the first controlled body. Percentages are relative to each channel’s configured action limits, not measured forces. Guides remain available with animations off. |
| **Pause** with animations on | Gentle idle motion continues | Stops simulation-dependent wheel motion and event progression while retaining small hover and engine motion. Turn **Animations** off for a still presentation of the paused world. |
| Workshop **Play animation / Pause animation** | Stopped initially | Advances or pauses decorative motion using the current actuator sliders; playback does not change commands or simulate physics. The global **Animations** switch must be on to play. Selecting **Side** or **Top** resets decorative motion while preserving slider values. |
| Workshop actuator sliders and **Neutral / Max** | Neutral commands | Shows the selected asset’s catalog actuator channels with their signed bounds. **Neutral** sets all channels to zero; **Max** sets each to its upper bound. Rocket variants expose vector thrust/torque or individual thrusters. Values persist across asset, style, and detail changes during the workshop session; static action cues update even with animations off. |
| **Rollout paths** | On | Recorded controlled-body paths, available for FMC and Wave Jump. Green is at/above mean recorded reward; rose indicates terminal state; violet indicates a tethered path; blue shows other alternatives. Terminal color takes precedence over tether color. |
| **Future-state cloud** | On | Controlled-body positions in the planner's returned world batch, which may be at a partial horizon in deadline mode. |
| **Tethers & formation** | On | Current tether geometry linking bodies. |
| **Collision geometry** | Off | Native hull outlines, useful when a decorative model differs from its collider. |
| **Clean view / Show diagnostics** | Diagnostics visible | Temporarily hides these four layers, then restores their previous visibility. The separate Physics inspector remains independent. |
:::

:::{div} feynman-prose
The renderer samples large trees to draw roughly at most 50,000 path segments.
This drawing limit does not prune the native record. Hiding paths also does not
disable recording. For that, change **Tree** in the **Planner decisions** timeline
and commit the change with **Apply and restart**.
:::

(sec-lab-controls-diagnostics)=
## Read the telemetry and choose recording detail

:::{div} feynman-prose
Task progress, simulation time, and planning progress remain near the world.
Open the inspector's **Diagnostics** for population, risk, memory, and performance
measurements. **Selected decision** shows recorded action, path reward, outcome,
and reward weights when available. Missing values read **Not recorded**; individual
reward contributions are not inferred from the weights or recomputed for old frames.
The timeline separates **World motion** from **Planner decisions**, whose cursors
count different things: physics frames and controller decisions.

For Wave Jump, the selected path reward describes the chosen final walker and
trajectory progress describes execution of the full path, shared prefix, or
one-action fallback. The status identifies shared-prefix execution or fallback
and shows search depth. These are separate from
search progress: completing the search starts the journey.

Three measurements answer different questions: simulation time measures what the
world has done, planning progress measures search work, and risk probes examine
short random continuations of the chosen action. A cloud with many dead worlds
does not by itself measure the risk of the action ultimately selected.
:::

:::{div} feynman-added
| Readout | Exact interpretation |
|---|---|
| **SIMULATION / TICK** | Executed or replayed world tick times `physics.dt`; not wall-clock runtime. |
| **DEAD RATIO** | FMC/Wave Jump final Wave terminal fraction. Shooting reports terminal rows divided by allocated Walkers; iCEM's inactive capacity can dilute this fraction. Random has no searched population, so its zero is not evidence of safe exploration. |
| **SELECTED-ACTION RISK** | Fraction terminal in 16 separate continuations: chosen action for `F` frames, then independently sampled uniform joint actions held for `2F` frames, where `F = Action frames`. Display includes sample count and total horizon. |
| **EVAPORATED / CLONED** | Nodes removed by the latest FMC pruning pass, and fraction cloned in the latest Wave iteration. Wave Jump shares these operations; shooting and random controllers do not. |
| **AI BUDGET USED** | FMC/Wave Jump iterations / Horizon; shooting completed depth advances / (Horizon × Search rounds); random reports 100%. Capped at 100% in the HUD. This is search progress, not CPU utilization. |
| **ITERATIONS / MS** | FMC/Wave Jump Wave iterations or shooting depth advances, with measured planner elapsed milliseconds. Live elapsed time includes worker yields and risk evaluation. |
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
| **Tree → Full** (`recording = 2`) | Retain all FMC/Wave Jump search nodes; useful for examining alternatives, with higher memory use. |
| **Tree → Off** (`recording = 0`) | Disable visible search-tree recording. Wave Jump still uses at least pruned recording internally to reconstruct its trajectory. Executed-world recording continues. |
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
