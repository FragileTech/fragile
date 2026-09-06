(sec-control-lab-task-tandem)=
# Tandem flight: guide two rockets through ordered gates

:::{div} feynman-prose
Tandem flight asks one controller to steer two rockets around the same checkpoint
loop. Each rocket has its own engine, turning command, and checkpoint progress.
They share an objective that also favors a chosen separation. Imagine coordinating
two swimmers around buoys: both must visit the buoys in order, but one can get ahead.
There is no physical rope forcing the other to follow.

This tutorial starts with a short autonomous run, then separates the two controls
by hand and changes the desired formation distance. Begin with
{doc}`control_lab_getting_started` if the Lab is not running. The
{doc}`control_lab_tasks` index covers the other tasks.
:::

(sec-lab-tandem-arena)=
## Read the arena before moving

:::{div} feynman-prose
Select **Tandem flight** under **Environment**. The scene contains two vector
rockets, an irregular outer wall, one central hole, and six circular checkpoint
zones. Their starting centers are `[12, 12]` and `[12, 16]`: four world units apart.
Both initially point along the positive x direction.

The checkpoint centers, in required order, are `[23, 10]`, `[45, 10]`, `[53, 22]`,
`[44, 34]`, `[20, 34]`, and `[12, 21]`. Each zone has radius `3.5`. A rocket earns a
checkpoint when its center enters its own next zone. Merely touching a different
zone does not advance it. After checkpoint six, its next target becomes checkpoint
one again; the circuit can repeat.

Switch to **2D** for an overhead view. Locate the first zone relative to both
rockets and trace the route around the central obstacle. Enable **Collision
geometry** when you want to see the physical boundaries beneath the artwork.
Outer walls and hole boundaries are lethal in this preset. Contact can terminate
the whole world, even when the other rocket still has room to maneuver. Body-to-body
collisions are not configured as lethal here.
:::

:::{figure} ../../_static/control_lab/tutorials/tandem-overview.png
:alt: Angled view of the reset Tandem flight arena with two rockets, checkpoint zones, and a central obstacle.

The reset arena. Identify both rockets and the first checkpoint before starting;
the central hole constrains the route around the loop.
:::

:::{div} feynman-prose
The **Tethers & formation** layer draws a formation guide between the rockets.
It is a visual aid. This preset has no physical tether connecting the pair, and
hiding the guide does not change the task. A stretched guide therefore does not
mean a spring is applying a restoring force.
:::

(sec-lab-tandem-first-run)=
## Make a reproducible first run

:::{div} feynman-prose
Use a small, explicit baseline so that your next experiment has a clear reference.
The Lab initially requests four worker threads; change that setting to one for
this walkthrough. Changing controller settings rebuilds the world and clears the
current in-memory recording, so finish configuration before collecting motion.

1. Select **Tandem flight** and wait for **WEBASSEMBLY** and enabled run controls.
2. Set **Controller** to **Fractal Monte Carlo**, **Walkers** to `128`, **Horizon**
   to `16`, **Action frames** to `6`, and **Seed** to `7`.
3. Set **Clock** to **Reproducible · wait for planning** and **Worker threads** to
   `1`. Leave the remaining FMC settings at their defaults, including **Distance
   weight** `1`, **Reward weight** `1`, **Action noise** `0.2`, and **Elites** `0`.
   Keep **Perturb inherited actions** checked and **Tree** at **Pruned**.
4. Disable **Keyboard control**, then press **↺** to reset. This also removes any
   automatic first action from the initial page load. Confirm the pair is back at
   the start and **Gates crossed** is zero.
5. Press **Step** once. Wait for planning to finish. One decision supplies controls
   for both rockets, then advances six physics frames: `0.1` simulated seconds at
   this scene's `1/60` second frame duration. Motion can be small at this scale.
6. Press **Step** a few more times, watching both bodies rather than just the
   leading one. Use **Run experiment** for a longer observation, then **Pause
   experiment** before inspecting the result. Stop after a fixed number of Steps
   for a comparison, or at terminal failure if it occurs sooner.

The 128 walkers are candidate futures for the entire two-rocket world. They are
not 128 physical rockets. Each candidate action includes both rockets' thrust and
torque. The selected joint action can give the two bodies different commands.
Neither this population size nor this horizon guarantees a completed loop.
:::

:::{figure} ../../_static/control_lab/tutorials/tandem-detail.png
:alt: Overhead Tandem flight view after one planned Step, showing the two rockets and nearby checkpoint geometry.

After one Step from the baseline reset. This short interval helps locate the pair
and its surroundings; the image is not evidence of completed checkpoints.
:::

(sec-lab-tandem-counters)=
## Distinguish checkpoints, reward, and success

:::{div} feynman-prose
The large **Gates crossed** score sums checkpoint events from both rockets. If
one enters the first gate, the score increases by one. When the other enters its
first gate, it increases again. Each rocket retains its own next-gate index. A
score of six therefore does not establish that both have completed the six-gate
loop. Even twelve events alone do not prove that each contributed exactly six.

To inspect individual progress, replay slowly and follow each rocket through the
ordered zones, keeping two separate tallies. The main score is an aggregate,
not a two-agent progress table. The formation guide also does not certify that
both rockets currently target the same checkpoint.

Accumulated reward is a different quantity. The default checkpoint reward is
`30` per event, but reward also includes changes in the progress potential and
collision costs. Progress can earn reward before any checkpoint event. For this
task the potential combines distance to each rocket's next gate with a formation
term, then averages across the controlled bodies. The engine rewards its change
during motion, using `rewards.progress`, before processing checkpoint events.
Consequently, a changing reward readout is not a checkpoint counter.

An experiment's success criterion is separate again. In **Experiments**, explicitly
choose **Gates crossed** as the success metric and set the desired target; do not
assume the task selector supplies a suitable goal. A target of `2` is a useful
first checkpoint-event exercise. A target of `12` measures twelve aggregate events,
not synchronized completion. Live running does not automatically end at your
informal target; batch experiments stop according to their configured goal,
frame limit, or terminal state. A terminal frame cannot also count as success.
See {doc}`control_lab_experiments` for controlled comparisons.
:::

(sec-lab-tandem-manual)=
## Control one rocket, then control both

:::{div} feynman-prose
Manual input makes the joint-action idea concrete. First export any run you want
to keep, then reset. Open **Edit scene**, choose **Select & move**, and click the
second rocket without dragging it. Enable **Keyboard control** and click the world
so a form field no longer holds keyboard focus.

Hold **W** briefly to thrust the selected rocket forward. **A/D** apply opposite
turning torques. **S** gives zero thrust here: vector rockets have a forward-only
thrust channel. They have no keyboard brake, so **Space** does not stop them.
The other rocket receives neutral input, though neutral input does not erase
existing velocity. With no body selected, keyboard control targets the first
controlled body.

Release the keys. Manual frame requests stop; this is not an independent clock
that keeps the rockets coasting after release. To examine coasting, apply a
zero-input frame with the sliders. Disable **Keyboard control** before returning
to autonomous control.

Now reset and expand **Actuator channels** in the editor. There are thrust and
torque channels for each rocket. Set both thrust sliders to `0.5`, leave both
torques at `0`, and press **Apply action · 1 frame** several times. Each click
advances one physics frame, regardless of **Action frames**. The full slider
vector controls both bodies, independently of selection. A thrust command of
`0.5` requests half the configured maximum thrust, not a new maximum engine size.

Set the second rocket's thrust to zero and repeat a few clicks. Compare the pair's
relative motion. Return every slider to zero before another manual test. Opening
or selecting in the editor does not itself change the scene, but dragging a body
or applying an edit rebuilds the world. See {doc}`control_lab_controls` and
{doc}`control_lab_scenes` for the complete input and editing workflows.
:::

(sec-lab-tandem-formation)=
## Change only the desired separation

:::{div} feynman-prose
The preset sets `formation_distance` to `4`. With exactly two rockets, each is half
their separation from their common center. The formation part of the potential
therefore favors a separation of four world units. It does not demand a particular
heading or side-by-side orientation.

1. Reset the baseline, run exactly ten **Step** decisions unless terminal failure
   occurs first, and export the run. Note elapsed frames, checkpoint count, and
   whether the gap between the rockets grows or shrinks.
2. Reset, open **Edit scene → Edit complete scene JSON**, and change only
   `"formation_distance": 4` to `"formation_distance": 8`.
3. Press **Compile scene**. Keep the original rocket positions and all controller
   settings. The rebuilt pair still starts four units apart; you changed the
   preferred separation, not the placement.
4. Repeat the same ten-Step observation and export it separately. Compare equal
   simulated times in the two recordings. Restore `4` and compile when finished.

The coefficient `rewards.formation` defaults to `0.15`; this scene omits an explicit
`rewards` object and uses compiler defaults. That coefficient weights separation
error inside the potential. It is not a constant penalty charged every frame for
remaining separated incorrectly: an unchanged formation error makes no change to
that potential term. Moving closer to the preferred gap improves the term; moving
away worsens it. Gate approach and collisions also affect the decision, so doubling
the preferred gap need not make the observed gap double.

Do not change **Reward weight** to perform this experiment. That planner control
changes FMC fitness weighting, whereas `rewards.formation` changes the scene's
reward definition. The {doc}`control_lab_scene_reference` explains their underlying
scene fields and defaults.
:::

(sec-lab-tandem-recording)=
## Save observations and recover from trouble

:::{div} feynman-prose
Pause, use **WORLD REPLAY** to seek through actual movement, and select a slow
playback speed when checking gate entries. Add an **Event note** such as “second
rocket approaches gate 1”, then press **Add marker**. Use **Export run** before
changing settings or compiling another scene. **Open run** restores the recorded
motion; **Back to live** shows its paused endpoint. Read {doc}`control_lab_replay`
for snapshots, continuation, and long recordings.

If a gate does not count, check that it is that rocket's next gate and that the
rocket's center entered the zone. If the world stops, inspect the last frames for
contact with a lethal boundary. Reset to retry; more thrust is not a repair for
a terminal world. If keyboard input seems ignored, check keyboard focus, selected
body, and **Keyboard control**. If only one rocket responds, remember that keyboard
input targets one body; use the full set of actuator sliders to command both.

If the pair separates, first distinguish an objective from a constraint. There is
no missing tether to reattach. Inspect the trajectory and gate targets, restore
the baseline, and change one parameter at a time. Record an observed failure as
carefully as a success: it tells you which part of the coordination problem your
next experiment should examine.
:::
