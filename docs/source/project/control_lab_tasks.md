(sec-control-lab-tasks)=
# Task tutorials

:::{div} feynman-prose
Each Lab task gives you a different control problem to investigate. Start by
learning what counts as progress, then watch a few individual decisions before
running continuously. The tutorials below take you from the reset scene through
its controls, useful experiments, and recording a result you can inspect later.
Each has real application screenshots; the Racing page contains a separate
walkthrough for each of its six circuits.

If this is your first visit, complete {doc}`control_lab_getting_started` first.
Keep {doc}`control_lab_controls` nearby for shared settings and diagnostics.
:::

(sec-lab-tasks-choose)=
## Choose your first task

:::{div} feynman-added
| Environment and tutorial | What you will learn | Task progress to inspect |
|---|---|---|
| {doc}`Asteroid harvesting <control_lab_task_harvest>` | Approach cargo, acquire an automatic tether, and tow a moving rock into a base. | Cargo deliveries; the rocket reaching the base alone does not count. |
| {doc}`Ants & drops <control_lab_task_ants>` | Start with five harvesters, fill five-drop tanks, and return to the refinery to unload; try rockets, drones, karts and other fleet sizes. | Collected food, delivered units, and completed loads; full tanks must empty before collecting again. |
| {doc}`Tandem flight <control_lab_task_tandem>` | Guide two rockets through ordered checkpoints and investigate formation settings. | Gates crossed, together with each rocket's individual checkpoint progress. |
| {doc}`Collaborative mining <control_lab_task_mining>` | Inspect two tethers and coordinate transport of one heavy cargo. | Cargo deliveries and the rock's arrival at a base. |
| {doc}`Thinking graphs <control_lab_task_rocket>` | Record candidate futures, inspect ancestry, select a node, and replay an alternative branch. | Deliveries in the physical task; a saved and reconstructed decision for this tutorial. |
| {doc}`Racing <control_lab_task_racing>` | Drive with keyboard or planner, follow ordered checkpoints, and practise all six circuits. | Ordered checkpoint progress and completed laps. |
:::

:::{div} feynman-prose
For a first encounter with physical control, start with **Asteroid harvesting**:
one rocket makes it easier to follow the difference between steering and towing.
Both this solo preset and **Collaborative mining** start their rockets upright
($\pi/2$ radians), with **24 N** of thrust per rocket. Walls still cause physical
collisions, but wall contact does not kill the rockets (`lethal_walls = false`).
Their reward settings use **Progress = 1**, **Catch = 10**, and
**Distance squared = 0**, so moving toward cargo and catching it remain useful
without rewarding motion or falling for their own sake.

Choose **Racing** if you would rather begin with familiar driving controls. Its
circuit sections explain the different layouts instead of assuming that one
successful route transfers to every track.

Next try **Tandem flight** or **Collaborative mining** to see why controlling two
bodies changes the problem. **Ants & drops** starts with five harvesters sharing
a refinery. Each full five-drop tank takes two simulated seconds to unload there;
pickup slots become available again after three simulated seconds.
Once the difference between actual motion and predicted motion is clear,
**Thinking graphs** shows how to inspect the alternatives behind a decision.
:::

(sec-lab-tasks-baseline)=
## Use a shared starting configuration

:::{div} feynman-prose
The tutorials use an explicit comparison configuration below. For everyday play,
the **Asteroid harvesting** and **Collaborative mining** presets instead recommend
**Wave Jump**, **128 Walkers**, **Horizon 32**, **Action frames 6**, **Elites 4**,
and **Stop at first bifurcation** enabled. Switching presets updates settings
that still match the previous recommendations and preserves custom controller
settings; starting a fresh task resets them to its recommendations.

Select the task before applying the tutorial comparison settings below, then
finish configuration before collecting a recording you want to keep.

1. Choose **Fractal Monte Carlo** under **Controller** and
   **Reproducible · wait for planning** under **Clock**.
2. Set **Walkers** to **128**, **Horizon** to **16**, **Action frames** to **6**,
   and **Seed** to **7**.
3. Open **Planner settings** and set **Worker threads** to **1**. This portable
   serial baseline deliberately differs from the interface default of 4.
4. Follow the task page's **Tree** recording instructions, then press **↺**
   to reset. Press **Step** once and inspect the result before continuing.

Walkers are candidate worlds used by the planner. They are separate from the
vehicles in the physical scene. Increasing Walkers therefore does not add more
harvesters, rockets, or karts. The baseline is a starting budget, not a promise of
successful delivery or a completed lap.

Use **2D / 3D** to switch between overhead and angled views. **Follow agent**
centers the camera on the selected body, or the first controlled agent when
nothing is selected. While following, that button reads **Whole arena**; press
it to restore the arena view. Camera changes help you inspect motion without
changing the task's physics.
:::

(sec-lab-tasks-measure)=
## Decide what your result means

:::{div} feynman-prose
Keep three measurements separate. **Task progress** records events such as food
collection, gate crossings, or deliveries. **Reward** includes the signals the
controller optimizes, which can improve before a task event occurs. An
**experiment success criterion** specifies the metric and target a benchmark
must reach. A positive reward is therefore insufficient evidence of a delivery.

Check the success metric explicitly whenever you change tasks in
{doc}`control_lab_experiments`. A tandem aggregate count does not establish that
both rockets completed the route; inspect their individual progress too. A
rollout crossing a finish line describes a candidate future. Use the executed
world and its counters to establish what happened.

Pause and use {doc}`control_lab_replay` to inspect or export an interesting run
before changing settings. To change the arena itself, follow the exercises in
{doc}`control_lab_scenes`; use {doc}`control_lab_scene_reference` when you need
the meaning of an individual JSON option.
:::

(sec-lab-tasks-captures)=
## Refresh tutorial screenshots

:::{div} feynman-prose
Maintainers can regenerate the screenshots from the repository root. If native
code changed, rebuild with `make control-web` first. Start the local application
in one terminal:
:::

```bash
CONTROL_PORT=8097 make control-lab
```

:::{div} feynman-prose
Then capture and validate the documentation examples in another terminal:
:::

```bash
npm --prefix fractal-gas-web run capture:lab-docs
npm --prefix fractal-gas-web run test:lab-docs
```

:::{div} feynman-prose
Set `CONTROL_CAPTURE_GROUP=tasks` or `CONTROL_CAPTURE_GROUP=editor` to refresh
only that group. Set `CONTROL_SCREENSHOTS=/tmp/lab-captures` to write an inspection
copy elsewhere, or `CONTROL_TEST_URL` if the server uses another address.
The capture manifest records actual capture states alongside the images.
Review the images and captions together: a reset view, a single Step, or an
editor placement does not establish a completed task event.
:::
