(sec-control-lab-task-harvest)=
# Asteroid harvesting: hook, tow, and deliver

:::{div} feynman-prose
Your rocket has a simple job: bring loose ore into the delivery base. The difficulty
is that steering the rocket does not directly steer its cargo. A tether transmits
force, the rock keeps its momentum, and a gravity well bends both trajectories.
Learn to recognize an approach, an attachment, and a delivery as three different
stages. Then the planner's decisions become much easier to interpret.

Start the application using {doc}`control_lab_getting_started`. This tutorial uses
the stock **Asteroid harvesting** environment. Find the other tasks in
{doc}`control_lab_tasks`, the complete controls in {doc}`control_lab_controls`, and
scene customization in {doc}`control_lab_scenes`.
:::

(sec-lab-harvest-landmarks)=
## Read the arena before moving

:::{div} feynman-prose
Choose **Asteroid harvesting** from **Environment**, wait for the backend to become
ready, and press **↺** (Reset). This removes any movement from the initial page
load. Use **2D / 3D** to compare the overhead map with the angled view. Scroll to
zoom; middle/right-drag or Alt-drag pans. These camera operations preserve physics.

The scene occupies a 64-by-44 world-coordinate rectangle, but its irregular outer
boundary encloses a smaller playable region. The rocket starts at `[16, 16]`.
The nearest rock starts at `[19, 19]`, and the base is centered at `[12, 11]` with
radius 3. These coordinates are useful when inspecting entities in the editor;
they are world units, not screen pixels. Screen directions change with the camera.

There are five cargo rocks, initially at `[19, 19]`, `[45, 31]`, `[47, 12]`,
`[16, 31]`, and `[41, 9]`. Their masses differ: 3, 5, 2, 2, and 4 respectively,
compared with the rocket's mass of 1. The central polygon is a hole in the arena,
not a shortcut. The gravity well at `[46, 22]` attracts bodies; it has strength 28
and softening 3. Softening moderates the field near its center, but does not remove
its pull.

Enable **Collision geometry** to see the physical boundaries. The preset has
lethal walls, including the hole boundary: a controlled rocket's wall collision
can end the episode. Body collisions are not configured as lethal in this scene,
but bumping a rock can still spoil an approach or throw the tow toward a wall.
:::

:::{figure} ../../_static/control_lab/tutorials/harvest-overview.png
:alt: Angled view of the reset asteroid harvesting arena showing the rocket, cargo rocks, base, central obstacle, and gravity well.
:class: feynman-added

The reset arena in the angled view. Locate the base and nearby cargo before
turning on continuous control; this image shows the starting layout, not a delivery.
:::

(sec-lab-harvest-first-run)=
## Make a reproducible first run

:::{div} feynman-prose
Use a modest, explicit planning configuration so you can repeat the sequence and
inspect each decision. This is an operating baseline, not a claim that the solver
will always deliver cargo with these settings.

1. Select **Fractal Monte Carlo** in **Controller**. Set **Clock** to
   **Reproducible · wait for planning**.
2. Set **Walkers** to **128**, **Horizon** to **16**, **Action frames** to **6**,
   and **Seed** to **7**. Under **Planner settings**, set **Worker threads** to
   **1**. This deliberately changes the UI default of 4 to a portable serial
   baseline. Leave **Distance weight** and **Reward weight** at 1,
   **Action noise** at 0.2, **Elites** at 0, and **Perturb inherited actions** checked.
3. Select **Pruned** under **Tree** and press **↺** after setting the fields.
   Settings changes rebuild the world and clear its recording, so finish setup
   before starting a run you intend to save.
4. Leave **Rollout paths**, **Future-state cloud**, and **Tethers & formation**
   enabled. Press **Step** once. The planner computes a decision and commits up
   to six physics frames. At the preset's `1/60` second timestep, an ordinary
   nonterminal step advances 0.1 simulated seconds.
5. Switch to the overhead view. Identify the solid rocket's new pose separately
   from the paths and cloud. Those overlays show candidate futures; a path passing
   through the base does not establish that the actual cargo reached it.
6. Press **Step** several more times, inspecting the gap to the nearest cargo.
   Once you recognize what changes after a decision, press **Run experiment**.
   Use **Pause experiment** to inspect an approach, attachment, or apparent stall.
7. Watch for a visible tether and then for an increase in the delivery counter.
   If the episode ends at a wall, pause and inspect the last frames before reset.
   A failed run still provides the information needed to diagnose the maneuver.

**Walkers** counts candidate worlds, each containing the rocket and all five rocks.
It does not add 128 rockets to the arena. The nominal 16-action lookahead spans
1.6 simulated seconds at six frames per action; it is not a promised arrival time.
:::

:::{figure} ../../_static/control_lab/tutorials/harvest-detail.png
:alt: Top-down asteroid harvesting view after one planner Step with candidate paths and diagnostic overlays.
:class: feynman-added

One decision viewed from above. Compare the executed rocket with the planning
overlays. This early inspection frame does not show a completed cargo delivery.
:::

(sec-lab-harvest-mechanics)=
## Recognize attachment and delivery

:::{div} feynman-prose
There is no hook key. The preset supplies one automatic tether whose source is the
rocket and whose target starts disconnected. After physics advances, a disconnected
automatic tether chooses the nearest active cargo with center-to-center distance
strictly below its **2.8** hook range. Merely seeing the edges of two models touch
is not the attachment test. The nearest starting cargo is about 4.24 units from
the rocket's center, so the initial scene is not already attached.

On attachment, the engine sets the tether's current rest length to the acquisition
distance, with a minimum of 0.1. The scene's configured `rest_length: 2.5` therefore
does not mean an automatically acquired cable instantly pulls both centers to
exactly 2.5 units apart. The tether behaves as a damped constraint and can break
under sufficiently large force; a disconnected automatic tether can acquire cargo
again when the range condition is satisfied.

Delivery occurs when an active cargo's **center** enters the base's radius-3 disk.
The rocket entering the base on its own does not count. A delivery increments the
counter, adds the delivery reward, deactivates that cargo, and detaches its tether.
These five rocks do not respawn in the stock harvesting scene. There is also no
special automatic victory termination after the first delivery; the experiment
runner's chosen success criterion is a separate stopping rule.

The reward signal includes progress shaping. Before attachment, it encourages
approaching available cargo; after attachment, its distance term measures the
attached cargo's distance to a base. Consequently positive accumulated reward can
appear before a delivery. Use the delivery counter to answer “did ore arrive?” and
reward to study what the controller was optimizing. A displayed task score, reward
total, and experiment success flag need not report the same quantity.
:::

(sec-lab-harvest-manual)=
## Try the maneuver with your own controls

:::{div} feynman-prose
Export any run you want to preserve before resetting. For a manual practice run,
press **↺**, enable **Keyboard control**, and click the world so a text or numeric
field no longer has focus. With this single controlled rocket, input targets it
automatically unless you have selected another body in the editor.

1. Tap **A** or **D** to apply positive or negative turning torque. Turning changes
   the direction of future thrust; it does not instantly rotate the velocity.
2. Use short **W** presses to thrust toward the nearby rock. Approach gradually,
   and watch **Tethers & formation** for attachment.
3. After attachment, orient toward the base and apply short thrust pulses. Keep
   watching the rock: towing past the base with the rocket while the cargo swings
   outside its disk is not a delivery.
4. To slow the rocket, turn against its motion and thrust. **S** cannot produce
   reverse thrust on this forward-only actuator. **Q/E** and **Space** have no
   strafe or brake channel to operate on this rocket.
5. For finer inspection, open **Edit scene**, expand **Actuator channels**, set the
   named thrust and torque sliders, and press **Apply action · 1 frame**. Moving
   a slider alone does not advance physics. A zero-input frame still allows
   momentum, gravity, and drag to act.

Manual input pauses autonomous execution. While keys are held, the keyboard adapter
requests two physics frames roughly 30 times per wall-clock second. Releasing all
keys stops these manual steps; it does not leave the simulation coasting on a
separate clock. Uncheck **Keyboard control** before returning to **Step** or
**Run experiment**. See {doc}`control_lab_scenes` for the full editor workflow.
:::

(sec-lab-harvest-experiments)=
## Change one variable and record the outcome

:::{div} feynman-prose
First compare two nearby planning budgets. Export the baseline run, change
**Horizon** from 16 to 32, reset, and repeat with seed 7 and the other fields fixed.
The nominal planning reach doubles; computation also grows, and successful delivery
is still something to measure. A second useful comparison changes **Action frames**
from 6 to 3: each executed action becomes finer, but the same 16-action horizon now
looks only 0.8 seconds ahead. Do not describe that as a pure improvement in control
precision without acknowledging the shorter prediction interval.

For a delivery benchmark, open **Experiments**, leave **All preset scenes** unchecked,
and explicitly choose **Cargo deliveries** in **Success metric** and **1** in
**Success target**. Set **Population** to 128, **Lookahead · actions** to 16,
**Action duration · frames** to 6, **Seeds** to `7,11,19`, and choose an explicit
**Episode limit · frames**, such as 1200. Select the two controller variants and
press **Run benchmark**. A 1200-frame limit gives each trial up to 20 simulated
seconds; it does not guarantee that either variant can complete a delivery.

The benchmark UI always sends its selected goal and may retain a prior task's
selection. The stock harvest JSON has no `evaluation` declaration: an API or CLI
benchmark that omits its goal falls back to survival through its frame limit.
Therefore set `{"metric":"deliveries","target":1}` explicitly in a programmatic
goal too. Surviving and delivering are different experiments. A terminal frame
cannot count as a successful delivery trial. See {doc}`control_lab_experiments`
for parameter overrides, timing, and report interpretation.

To keep an illustrative live run, press **Pause experiment**, then **Export run**.
Use **WORLD REPLAY** to seek back through actual motion and **Play world** to watch
it. **Back to live** restores the paused authoritative view. An in-memory export
contains the scene, settings, recorded motion, and retained exploration entries;
**Save state** alone is only a world snapshot. Follow {doc}`control_lab_replay`
for device-backed recording and planner checkpoints.
:::

(sec-lab-harvest-troubleshooting)=
## Diagnose what prevented delivery

:::{div} feynman-added
| Observation | What to check and try |
|---|---|
| No tether appears near a rock | Check center distance and that **Tethers & formation** is enabled. Acquisition requires an active cargo inside the hook range and a physics step. |
| The rocket reaches the base but deliveries stay at zero | Follow the attached rock. Its center must enter the base; the rocket's position is insufficient. |
| The rocket keeps moving after zero thrust | Momentum and gravity remain. In manual mode, advance zero-input frames to observe coasting, or turn and thrust against motion. |
| Motion stops and Run does not continue | Inspect terminal status and the last recorded wall contact. Reset starts another episode; camera changes cannot revive a terminal world. |
| The cloud reaches the base but actual motion does not | The cloud contains predictions. Inspect the solid bodies and **WORLD REPLAY** for executed motion. |
| Cargo vanishes near the base | Check the delivery counter. Delivered stock cargo is deactivated and does not respawn. |
| Keyboard input seems ineffective | Enable **Keyboard control**, click outside form fields, and check that the selected body is the controlled rocket. |
| A settings adjustment lost the trajectory | Settings rebuild the scene. Export the next run before changing fields; use a reopened recording to revisit an earlier trial. |
| Reward rises without experiment success | Check the selected metric. Progress reward can increase without a cargo delivery; success also requires a nonterminal world. |
:::
