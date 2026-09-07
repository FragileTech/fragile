(sec-control-lab-task-mining)=
# Collaborative mining: haul a heavy rock together

:::{div} feynman-prose
The two shipped rockets pull one rock toward a delivery base. They now fly
side-on in a downward gravitational field, so altitude is part of the job: an
engine must spend thrust not only on hauling but also on keeping its rocket from
falling.
The useful question is not simply whether both engines are firing: are their
forces helping the same journey? A rocket pulling across its partner's path can
stretch the connections and waste motion. Watch the rock, both tethers, the
altitude, and the destination together.

This tutorial starts with the supplied **Collaborative mining** environment. You
will inspect its connections, run a planner, compare one powered rocket with two,
and learn what happens when cargo is delivered. For installation, begin with
{doc}`control_lab_getting_started`; the other lessons are in
{doc}`control_lab_tasks`.
:::

(sec-lab-mining-layout)=
## Find the cargo, base, and hazards

:::{div} feynman-prose
Choose **Collaborative mining** under **Environment**, then press **↺** to reset.
The scene contains two controlled rockets and **one** passive cargo rock. The
rock has mass 24 kg and linear drag 0.8; each rocket has mass 1 kg and maximum
thrust 16 N. One rocket can move the load slowly. Cooperation provides additional
force; this is not a rule that makes solo transport mathematically impossible.

The rockets start at `[19, 10]` and `[19, 15]`, flanking the rock at `[22, 12.5]`.
The delivery base is the circular zone centered at `[12, 12]`, with radius 3 m.
The rock's center must enter that zone to register a delivery. Merely flying a
rocket into the base does not deliver its cargo.

The irregular outer boundary and central polygonal hole are lethal wall geometry.
Downward gravity pulls the rockets and cargo toward lower altitude, so a rocket
that stops thrusting will descend; propulsion is needed to hold altitude while
hauling. The gravity well at `[47, 22]` attracts bodies too; it is a force
source, not another base. The direct starting journey is toward the nearby base,
away from the central obstacle. Later pursuit of respawned cargo can bring the
team into more awkward configurations. After reset, the whole arena is already
visible and the camera button reads **Follow agent**. Pressing it follows a
rocket and changes the label to **Whole arena**; press that to return to the
overview. Use **Side / overhead** to switch between the side view and overhead
inspection.
:::

:::{figure} ../../_static/control_lab/tutorials/mining-overview.png
:alt: Side reset view of Collaborative mining showing two rockets, the heavy rock, delivery base, central obstacle, and gravity well.
:class: feynman-added

The reset scene has one cargo rock and two connections. Identify the delivery base before watching the planner's paths.
:::

(sec-lab-mining-first-run)=
## Run and inspect your first decisions

:::{div} feynman-prose
Use a reproducible starting configuration so that changes have a clear meaning.
Changing these settings resets the current world and recording, so set them before
collecting a trial.

1. Select **Controller → Fractal Monte Carlo**, **Walkers → 128**,
   **Horizon → 16**, **Action frames → 6**, and **Seed → 7**.
2. Select **Clock → Reproducible · wait for planning** and **Worker threads → 1**.
   The current UI defaults to four requested threads; explicitly choosing one
   removes that difference from this exercise. Wait for **WEBASSEMBLY** readiness.
3. Leave the other FMC settings at their defaults and press **↺**. The preset
   and subsequent reset are paused. Check the two rockets and single rock against
   the overview.
4. Press **Step** once. This asks the planner for one joint action and applies it
   for six physics frames, or 0.1 simulated seconds at this scene's 60 Hz rate.
   Look for a small change in positions and tether shape. One step is an
   inspection exercise, not a promised delivery.
5. Switch to overhead view. Compare the rock's position with its starting position
   and the base, and check whether the rockets are losing altitude. Distinguish
   the bright executed bodies from candidate paths:
   the paths describe futures considered during planning.
6. Press **Run experiment**, watch several decisions, then **Pause experiment**.
   Check whether the cargo has approached the base, whether both connections
   remain present, and whether the delivery counter increased. Record the actual
   outcome, including stalled or unsuccessful runs.
7. Use **WORLD REPLAY** to inspect movement frame by frame. Press **Back to live**
   before resuming. See {doc}`control_lab_controls` for the planner diagnostics
   and {doc}`control_lab_replay` for saving the observation.

Each walker represents a possible future for the whole three-body system. It is
not an extra physical rocket. Increasing **Walkers** gives the search more
candidates; it does not add towing force to the executed world.
:::

:::{figure} ../../_static/control_lab/tutorials/mining-detail.png
:alt: Overhead Collaborative mining view after one planned Step, showing the nearby rockets, cargo, and base.
:class: feynman-added

After one Step, inspect the cargo and both tethers against the base. This capture illustrates an early decision, not a completed delivery.
:::

(sec-lab-mining-tethers)=
## Understand attachment, strain, and respawning

:::{div} feynman-prose
Both tethers are already connected at reset. Their JSON supplies rocket endpoints
`a: 0` and `a: 1`, and cargo endpoint `b: 2`. Although they are marked
`automatic: true`, initial attachment comes from those explicit endpoints. The
initial rocket-to-rock distances are slightly greater than the 3.5 m hook range;
that does not invalidate an existing connection.

Each tether begins with rest length 3.5 m, stiffness 35, and damping 8. Think of a
spring with damping rather than a rigid bar: its current length can differ from
its rest length, and the resulting impulse affects both bodies. A connection can
break if the magnitude of its computed impulse divided by the physics substep
duration exceeds the configured force threshold. This
preset omits `break_force`, so the compiler default is 500 N. **Hook range is not
a breaking distance.**

Open the rock settings to change **Hook stiffness · N/m** for all tow hooks in
the scene. The allowed range is 0 to 1,000,000 N/m; the supplied mining preset
keeps its default of 35 N/m. A low positive value permits more stretching, like
a rubber band. A high value makes the connection approximately fixed in length;
it remains a spring, not an exact rigid constraint. At zero, the spring force
vanishes but radial damping remains. Press **Apply rock settings** to use the
new value: this restarts the scene paused, so compare trials from that restart.
The same panel's **Rock size** control spans 0.1× to 2×. Apply rock settings
after changing the size too.

Stiffness resists changes in separation. It does not directly resist motion
around the rock: a rocket can move tangentially while keeping nearly the same
tether length. Increasing stiffness therefore cannot, by itself, cure orbiting.

When an automatic tether is disconnected, it searches for active cargo with its
center strictly within 3.5 m of the rocket's center. Reattachment sets a new
runtime rest length to the current separation, with a minimum of 0.1 m. Therefore,
a reacquired connection need not have the original 3.5 m rest length. There is no
manual “grab” key: steer close enough and advance physics.

When the cargo center enters the base, the delivery counter increases and the
cargo's connections detach. Because this rock has `respawn: true`, it normally
respawns immediately at a randomly sampled collision-free position across the
entire playable map, outside delivery bases and with clearance from walls, holes, and active
bodies. It keeps its configured initial angle and has zero linear and angular
velocity. The same cargo body is reused; another rock is not added. Seeded replay
reproduces the respawn positions. The rockets remain where they are. Their
automatic tethers must find the respawned rock again, and can do so immediately
if they are already close enough. Otherwise, the next job is to travel to the
rock and reacquire it. Each delivery changes the starting geometry for the next
haul.

If none of 256 sampled positions is clear, the rock stays delivered and inactive
until the next frame's placement attempts. Waiting for space does not count as
another delivery.
:::

(sec-lab-mining-manual)=
## Compare one powered rocket with two

:::{div} feynman-prose
Here is a short experiment that isolates a useful question: how does the cargo's
motion change when a second engine joins the first? Use equal simulated duration,
not equal time spent clicking. This experiment measures initial displacement; the
rockets initially point away from the base, so it is not a delivery recipe.

1. Reset the unchanged preset. Press **Save state** and keep the downloaded
   snapshot. This stores the same physical starting point for both trials.
2. Open **Edit scene**, choose an overhead view, and expand **Actuator channels**.
   Do not move or edit any entities. Editing the scene would change the experiment
   and can invalidate the snapshot's scene fingerprint.
3. Set every channel to zero. Set body 0's **thrust** to `1`, leaving its **torque**
   at `0`. Keep both channels for body 1 at `0`.
4. Press **Apply action · 1 frame** 30 times. This advances 0.5 simulated seconds.
   Watch the rock's displacement and the asymmetry in the two tethers. Save an
   image or add an **Event note** such as “one engine, 30 frames” and **Add marker**.
5. Press **Load state** and choose the saved reset snapshot. Check that the
   original positions return. Explicitly set the sliders again: loading a world
   snapshot does not make slider positions part of the physical state.
6. Set **thrust** to `1` for both bodies and both **torque** channels to `0`.
   Press **Apply action · 1 frame** exactly 30 times. Compare the cargo's movement
   with the first trial, including direction and tether deformation.
7. Export the run if you want both segments. The snapshot restoration marks a
   discontinuity in the recording; it is not physical travel back to the start.

This compares one *powered* rocket with two in the same connected system. The
unpowered rocket still has mass, drag, gravity, and a tether; it has not been
removed. In the one-engine trial, the powered rocket also has to spend thrust
to maintain altitude, so some of its force is not available for hauling. Do not
infer a universal speed ratio from this brief transient. Turning, connection
geometry, drag, gravity, altitude control, and collisions all matter over a
longer journey.

For freehand practice, enable **Keyboard control**, choose **Select & move**, and
click a rocket without dragging it. Click away from form fields before using
**W** for forward thrust and **A/D** for turning. **S** clamps to zero on these
forward-only vector rockets; it does not reverse them. Keyboard control targets
the selected body, or the first controlled body when none is selected. Selecting
the passive rock will not steer a rocket. To command both rockets simultaneously,
use the full slider vector. Releasing all keys stops manual time advancement;
apply zero-input frames if you want to observe coasting.
:::

(sec-lab-mining-score)=
## Measure deliveries and preserve the evidence

:::{div} feynman-prose
Use cargo deliveries as the task outcome. Reward is a separate signal: the native
model adds a delivery bonus and also rewards changes in distance-based progress,
with collision penalties. For an attached rocket, progress follows the
cargo's distance to a base; for a disconnected rocket, it follows distance to
active cargo. Positive reward can therefore occur before any delivery.

Each physics frame measures progress toward the target selected at that frame's
start, using that same target before and after movement. Breaking a tether
therefore does not earn a bonus merely by switching from the distant base to a
nearby rock. The next frame can select a new target for the disconnected rocket.

**Hooked rock travel** adds a separate reward for moving the cargo itself. Open
the reward settings to adjust it from 0 to 1000 reward units per metre; the
default is 1. Its scene field is `rewards.hooked_rock_distance`. Press **Apply
settings** to activate a change while preserving the current world. Setting it
to zero and applying disables this term.

At the start of each physics frame, the model identifies active cargo rocks
hooked to an active controlled vehicle. It then adds their centre-to-centre
travel distances over that frame and multiplies the sum by the coefficient.
Each rock counts once, even when both rockets hook it. At the default setting,
moving one hooked rock's centre 0.2 m earns 0.2 reward units. Hook attachments
and detachments change eligibility on the next frame; a respawn contributes no
travel reward.

This measures linear distance, not squared distance. Flying around a stationary
rock earns nothing from this term, and spinning a rock without moving its centre
earns nothing either. Moving the rock in any direction does earn reward, however,
including taking it around a loop. Keep **Target progress** and **Delivery
bonus** active to give that motion a destination; cargo travel alone does not
distinguish hauling toward the base from hauling away.

For a controlled benchmark, open **Experiments** with this scene selected, leave
**All preset scenes** unchecked, and explicitly select **Success metric → Cargo
deliveries** and **Success target → 1**. Check these fields even if you previously
used a different task. Set **Population → 128**, **Lookahead · actions → 16**, and
**Action duration · frames → 6** to match the live baseline; the dialog has its
own defaults. Choose seeds and an episode limit before comparing controllers,
and report that limit alongside results. A trial that runs out of frames without
a delivery has not met this goal, even if its reward improved. The benchmark also
requires a nonterminal world at success. Details are in
{doc}`control_lab_experiments`.

Pause and **Export run** to preserve executed motion and retained planning
records. Use **Jump to event…** to inspect a delivery marker if one occurred.
Saving a snapshot preserves a restart point, while exporting the run preserves
the trajectory that explains the result. Keep the same scene and record your
clock and thread selection separately.
:::

(sec-lab-mining-troubleshooting)=
## Recover from common problems

:::{div} feynman-prose
If the rock barely moves, inspect the actual thrust direction, altitude, and
whether both rockets are helping. A heavy load responds gradually, and each
rocket must also counter downward gravity; increasing planner population does
not increase engine strength. If a connection disappears, inspect
for a delivery or force-induced break, then approach the active cargo within hook
range. Advancing physics is necessary for reacquisition.

If the rockets orbit rapidly, check **Movement reward** and press **Apply
settings** after setting it to zero. A positive value rewards displacement in
any direction, including circling without hauling cargo. Zero removes that
incentive; it does not remove existing tangential velocity or guarantee that a
finite search finds a useful pull. Inspect the cargo's progress toward the base
and compare hook stiffness from the same paused starting state. A tighter spring
can keep a rocket close to the rock while it continues to circle.

If delivery stays at zero, inspect the cargo's center rather than a rocket's
position or an attractive search path. If the rock suddenly returns to its spawn
point while the counter increases, that is the intended replenishment cycle.
If the world becomes terminal, inspect the last frames for boundary or hole
contact and reset before retrying. If keyboard input fails, check focus, selection,
and **Keyboard control** before changing the scene.

For a modified course, follow {doc}`control_lab_scenes` and consult
{doc}`control_lab_scene_reference` for tether, cargo, gravity, and reward fields.
Save the original scene and change one property at a time: otherwise you cannot
tell which change improved the haul.
:::
