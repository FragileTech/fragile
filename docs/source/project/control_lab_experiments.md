(sec-control-lab-experiments)=
# Experiments, comparisons, and performance

:::{div} feynman-prose
A convincing-looking trajectory is a useful observation, but a controller comparison
needs a repeatable question. Which world, which seeds, what counts as success, and
how much simulation work was allowed? The **Experiments** panel records those choices
alongside the outcomes so you can inspect the comparison rather than trust its label.

Use {doc}`control_lab_getting_started` to launch the lab,
{doc}`control_lab_controls` to understand planner parameters, and
{doc}`control_lab_tasks` to choose a task tutorial. This page covers repeated trials,
two futures from one world, and measurements of physics and state movement.
:::

(sec-lab-experiments-panel)=
## Configure a benchmark

:::{div} feynman-prose
Click **Experiments** above the viewport. Opening the dialog pauses live control.
**Run benchmark** resets an independent world for each seed and controller variant;
it does not start trials from the current vehicle position. The selected scene,
including edits, is used unless **All preset scenes** is checked. Trials run
sequentially in a separate worker, with complete planning at each decision.
The live **Clock** setting does not impose deadlines here.

The experiment worker loads the serial WebAssembly module. Setting the live
**Worker threads** field to 64 does not make these benchmarks use 64 threads.
Keep this distinction in mind when comparing their timing with the live planner.

Select **Wave Jump** (`wave-jump`) in either variant to compare trajectory commitment
with FMC. Both use the same FMC search parameters. Wave Jump selects the alive
walker with the highest accumulated reward in the final population and follows its
ancestral action sequence for the recorded duration of every positive-duration
edge before searching again. Alive means nonterminal according to the physics.
If every final walker is dead, it selects the highest-scoring final walker but
executes only the first positive-duration action for its recorded frames, then
searches again. Ties go to the lower walker index in either case. Selection uses
accumulated reward even when FMC resampling uses a different reward setting.

Imagine the search proposes three turns. FMC commits to one action and asks again;
Wave Jump commits to the selected sequence when its leaf is alive, or just its
first executable action when all leaves are dead. A Wave Jump decision therefore means
one search and its trajectory, which can contain several actions. Compare executed
frames and simulated planning work alongside decision counts: an equal number of
decisions does not imply an equal amount of movement.
:::

:::{div} feynman-added
| Field | Default and accepted values | Meaning |
|---|---|---|
| **Variant A** | Fractal Monte Carlo (`fmc`) | First controller. All registered controllers appear. |
| **Variant B** | Cross-entropy shooting (`cem`) | Second controller; may equal A for parameter comparisons. |
| **Seeds** | `7,11,19`; 1–64 comma-separated unsigned 32-bit integers | Reset seed and base planning seed for each episode. Duplicate seeds are permitted, but do not add independent trials. |
| **Episode limit · frames** | 240; integer 1–36000 | Maximum number of executed physics frames in each trial. |
| **Population** | 32; integer 1–8192 | `walkers` for both variants before overrides. |
| **Lookahead · actions** | 8; integer 1–4096 | `horizon` for both variants before overrides. |
| **Action duration · frames** | 4; integer 1–4096 | `frames` per proposed action. Unlike the live field, this accepts more than 60. |
| **Success metric** | Cargo deliveries (`deliveries`) initially | Counter or accumulated quantity used to stop successfully. |
| **Success target** | 1 initially | Finite positive threshold. The input displays a minimum of 0.01; episode validation accepts any finite value greater than zero. |
| **Variant A/B parameter overrides (JSON)** | `{}` | Final settings overrides for that variant. Use a JSON object. |
| **All preset scenes** | Unchecked | Fetch all catalog presets and apply this specification to each. Ignores edits to the current scene. |
:::

:::{div} feynman-prose
When opening the dialog for a different scene object, the panel adopts that scene's
`evaluation.metric` and `evaluation.target` if the metric exists in the selector.
All six Racing circuits supply `gates` with a target equal to their checkpoint
count; Violet Circuit supplies 16. The nonracing stock presets do not declare an
evaluation default, so the panel retains its previous explicit goal; on first use
that is one cargo delivery. Always check the goal after switching tasks. This UI
behavior differs from the `runEpisode` API: when no goal is supplied and the scene
has no evaluation, the API uses survival for the episode's frame limit.

Variant settings are assembled in this order: current live planner settings, then
the dialog's controller/population/lookahead/action-duration fields and
`recording: 0`, then the variant JSON overrides last. An override can therefore
change even `algorithm`, `walkers`, or `recording`. Live algorithm-specific fields
only exist for the currently selected live controller; absent options for another
variant use that algorithm's defaults. Set both JSON objects explicitly when the
comparison depends on those values.

Wave Jump retains at least a pruned tree internally, including with `recording: 0`,
because parent links are needed to recover the selected sequence. That internal
ancestry does not turn an ordinary benchmark into an exported motion recording.
:::

```json
{"search_iterations": 3, "mppi_sigma": 0.35, "mppi_temperature": 2}
```

:::{div} feynman-prose
For this example choose MPPI in the corresponding variant selector. To compare two
MPPI noise scales, select MPPI in both selectors and put a different `mppi_sigma`
in each override. An optional `label` string names a comparison pane and is retained
in the report; the summary table still displays the algorithm ID. See the complete
parameter reference in {doc}`control_lab_controls`.
:::

(sec-lab-experiments-goals)=
## Define success and interpret the stopping rule

:::{div} feynman-prose
An episode ends at the first successful physics frame, terminal state, or frame
limit. The runner checks after every frame, even when an action was planned to last
longer. A terminal frame cannot also count as success: the implementation requires
the world to be nonterminal when its goal threshold is met.
These checks also apply to every frame inside a Wave Jump trajectory: reaching the
goal, actual death, or frame limit stops execution without finishing the remaining
sequence, including during the all-dead fallback action.
:::

:::{div} feynman-added
| Success metric label / key | Quantity compared with target |
|---|---|
| **Cargo deliveries** / `deliveries` | World delivery counter. |
| **Food pickups** / `pickups` | World pickup counter. |
| **Gates crossed** / `gates` | World gate counter. Each Racing circuit declares a goal equal to its checkpoint count; Violet Circuit has 16. Checkpoints count zone entry, not directional timing-line detection. |
| **Frames survived** / `survival` | Frames executed within this trial; units are frames, not seconds. |
| **Accumulated reward** / `reward` | Sum of the world's per-frame rewards during this trial. |
:::

:::{div} feynman-prose
See {doc}`control_lab_task_racing` for all six circuit layouts, their checkpoint
counts, and a walkthrough of the one-lap experiment goal.

Delivery, pickup, and gate thresholds use absolute world counters. A fork whose root
already has 16 gates succeeds immediately under target 16, without completing
another lap. Raise the target to the desired total before comparing continuations.
Survival frames and accumulated trial reward start at zero for each fork. A root
that is already terminal finishes immediately as a failure.

For **All preset scenes**, the UI sends one explicit goal to every preset. Cargo
deliveries cannot meaningfully score a food task, and a gate target cannot score an
arena without gates. Choose a common criterion such as survival for a smoke test,
or run task-specific benchmarks separately. The API and command-line specification
can omit `goal`; then each scene's `evaluation` is used, falling back to survival
for `maxFrames` when a scene has no evaluation. The UI always sends a goal.

**Run benchmark** displays completed/total episode progress. **Cancel** requests
cooperative cancellation at a worker yield; it does not interrupt the middle of a
native operation. Partial trials are not returned as a new completed report.
Closing the dialog terminates its worker and disposes the comparison renderers.
Opening another job also replaces the previous worker. A previously completed
report can remain available for export, so confirm **Benchmark complete** before
treating an exported report as the result of a new job.
:::

(sec-lab-experiments-results)=
## Read and export the results

:::{div} feynman-prose
The table summarizes all episodes with identical complete settings. With an
all-preset suite, a row pools those episodes across scenes; it is not a per-scene
ranking. If A and B resolve to identical settings, they merge into one summary row.
The exported report retains individual trials and separate per-scene summaries.
:::

:::{div} feynman-added
| Table heading | Report summary field | Meaning |
|---|---|---|
| **Controller** | `settings.algorithm` | Algorithm ID; inspect `settings` for the actual variant. |
| **Episodes** | `episodes` | Number of completed trials in this settings group. |
| **Success** | `successRate` | Fraction of trials meeting the goal without terminal failure. |
| **Contacts** | `collisions` | Mean accumulated native collision/contact count per episode, not a count of distinct accidents. |
| **Completion · s** | `completionSeconds` | Mean simulated completion time among successful episodes only; blank when none succeeded. |
| **∫u²dt** | `controlEffort` | Mean per-episode sum of squared channel inputs integrated over simulated seconds. |
| **Planning · ms** | `planningMs` | Total measured planning milliseconds across grouped trials divided by their total decisions. |
| **Simulated world-frames** | `simulatorFrames` | Mean actual completed planning-world physics frames per episode. |
:::

:::{div} feynman-prose
Control effort uses the channel values actually sent to physics. Built-in channels
are normalized actuator commands, so this quantity is not force, fuel, work, or
mechanical energy. Custom channel bounds and different joint action dimensions
change its interpretation. Compare it alongside task success and motion, rather
than declaring the smaller number universally better.

Planning time includes worker yields and depends on hardware and scheduling.
It excludes the authoritative-world execution loop, and the benchmark does not run
the live selected-action risk probes. `simulatorFrames` counts completed native
physics frames across planning worlds, including FMC and Wave Jump. Terminal worlds
and iCEM's zero-duration inactive slots reduce the count. Copying inactive rows still costs
time. Equal population and horizon can therefore give different measured work,
especially when search-round counts differ.

Click **Export experiment** to download `control-experiment.json`. A benchmark
report has `version: 1`, `engine: "fractal-control-2"`, the first `scene`, all
`scenes`, the requested `spec`, raw `results`, grouped `summary`, and `perScene`.
Each raw trial includes seed, resolved settings, frame and decision counts, total
reward, collision count, effort, planning time, success/death flags, goal,
simulation/completion seconds, mean planning time, native profile counters,
simulated planning frames, and its scene name/index.

The report records outcomes and configuration; ordinary benchmark trials do not
contain motion recordings. Use **Fork and compare** to export actual trajectories.
The current dialog has no report-import button. Keep the JSON for analysis or use
its scene and specification as inputs to the command-line runner.
:::

(sec-lab-experiments-forks)=
## Fork the displayed world and compare motion

:::{div} feynman-prose
A fork asks a different question from a fresh episode: what would two controllers do
from this exact situation? Pause at a useful live state, or seek a recorded world
frame using {doc}`control_lab_replay`, then open **Experiments** and click
**Fork and compare**. The root combines the recording's compatible scene snapshot
with the complete currently displayed world row. All bodies, task counters, and
world random state move together into both independent branches.

Changing the exploration-decision slider alone selects a thinking trace, not a new
world root. Use world playback to select an executed state, or **Replay branch** to
restore a speculative state before forking. The two variants use only the first
entry in **Seeds** for their planning random streams. Restoring the world row
preserves the root's environment random state rather than resetting it to that seed.
**All preset scenes** does not affect a fork.

After both branches finish, the panes show their executed world movement. Drag the
**Comparison frame** slider to inspect the same frame index in both recordings.
**Play both** advances one index at about 60 wall-clock updates per second;
**Pause both** stops it. The shorter branch holds its last frame while the longer
one continues. Playback is frame-synchronized and has no speed selector; scenes
with `physics.dt` other than `1/60` will not play at their simulated real-time rate.

Pane titles show the variant label or algorithm and success, time limit, or
“collision death.” That last label denotes a terminal flag; custom task extensions
can terminate a world for reasons other than collision. The display shows no search
trees, and tethers remain visible.

**Export experiment** now exports a comparison report with `version`, `scene`,
`spec`, and `branches`. Each branch contains `stats` and an `archive` string holding
a complete `.fgclab` motion archive. There is no comparison-report import flow in
the UI. To inspect one branch with the ordinary replay tools, extract its archive
string to a `.fgclab` file and open that file through **Open run**. Closing the dialog
stops synchronized playback and releases its renderers.
:::

(sec-lab-experiments-performance)=
## Measure state movement and inspect physics

:::{div} feynman-prose
Open **Performance probe** in the experiment dialog and click **Measure batch
throughput**. This allocates an independent serial batch of 256 worlds for the
selected scene; it does not advance the live world. It takes one initial state copy,
then repeats get, set, and reverse-index gather 32 times, followed by 16 physics
frames with zero requested actions. Channels clamp those actions to their bounds.
The worker supports 1–1024 probe worlds internally, but the UI fixes the count at 256.

The output reports world frames/s, get/set/gather GB/s, and MiB of WebAssembly linear
memory. Copy rates use native counters, not the entire browser-to-worker round trip.
Set timing includes row validation. Byte counters count copied batch bytes once,
including row stride padding, not both read and write traffic. Gather reorders
source rows simultaneously into the destination batch. WASM memory includes reserved
linear memory and is not the same as resident browser memory or GPU allocation.
:::

:::{div} feynman-added
| Native `profile` index | Accumulated quantity |
|---|---|
| 0 / 1 | Batch-step milliseconds / actual completed world frames. FMC frame counts also enter index 1; FMC planning time is tracked at index 8. |
| 2 / 3 | Get-state milliseconds / copied bytes. |
| 4 / 5 | Set-state milliseconds / copied bytes. |
| 6 / 7 | Gather milliseconds / copied batch bytes. |
| 8 / 9 | Native FMC planning milliseconds / advance calls. |
| 10 | Tracked native buffers at the time of the query. |
| 11 | Serialized native state size for that engine batch. |
:::

:::{div} feynman-prose
The main view's performance line is a different measurement: FPS, CPU render
submission time, draws, triangles, authoritative-world stepping throughput, and
tracked native buffers. It is not the live planner's throughput. Tracked buffers
exclude some scratch arrays, search trees, and allocator overhead. The probe's
numbers are separate from the most recent benchmark/comparison report; **Export
experiment** does not add a probe result to that report.

Enable **Physics inspector** below the main recording controls. At roughly 250 ms
intervals, it examines the displayed row in a separate prediction engine. Cyan
arrows show velocity times 0.25 s; amber external force and magenta tether force use
0.05 m/N; red arrows show nearby contact normals. Arrows are capped at 8 world units
for readability. External force includes drag, gravity wells, and actuator force;
tethers are shown separately. Contacts describe current nearby geometry, not stored
collision impulses. The selected or first controlled body's text reports m/s,
rad/s, newtons, contact count, and maximum absolute tether force.

During world replay, positions and velocities come from the selected frame, but the
current UI supplies the last live action to the force inspector. Actuator-force
arrows are therefore not an exact historical force reconstruction. Seek while
paused for stable geometric inspection; use recorded actions when doing quantitative
offline force analysis. More on the packed state and extension APIs is in
{doc}`control_lab_architecture` and {doc}`control_laboratory`.
:::

(sec-lab-experiments-cli)=
## Run the same experiment without the interface

:::{div} feynman-prose
The Node.js runner uses the same serial WebAssembly engine and experiment code.
Build the browser module first as described in {doc}`control_lab_getting_started`.
Save this short specification as `/tmp/control-docs-spec.json`. It compares two
controllers over two seeds for twelve frames: a smoke test of the workflow, not a
useful racing performance claim.
:::

```json
{
  "seeds": [7, 11],
  "maxFrames": 12,
  "goal": {"metric": "survival", "target": 12},
  "variants": [
    {"algorithm": "fmc", "walkers": 8, "horizon": 2, "frames": 2, "recording": 0},
    {"algorithm": "wave-jump", "walkers": 8, "horizon": 2, "frames": 2, "recording": 0}
  ]
}
```

:::{div} feynman-prose
Run from the repository root. Progress reports `Episode 1/4` through `Episode 4/4`,
then the report path. The output has the benchmark schema described above; measured
timing will vary between runs.
:::

```bash
node fractal-gas-web/tools/control-benchmark.mjs \
  fractal-gas-web/web/lab/scenarios/racing.json \
  /tmp/control-docs-spec.json /tmp/control-docs-report.json
```

:::{div} feynman-prose
The first input may instead be a JSON array of 1–16 complete scene objects. The
specification accepts 1–8 variants, whereas the UI exposes two. Each variant must
provide integer `walkers`, `horizon`, and `frames` within the ranges listed above.
There are no inherited live UI settings in the CLI. Omit `goal` to use each scene's
evaluation or the survival fallback. The repository also supplies
`fractal-gas-web/web/lab/benchmarks/smoke-spec.json`, supplying a ready-made controller suite.
An omitted output argument writes `control-benchmark.json` in the current directory.

For a native C++ throughput matrix rather than controller episodes, build and run
the benchmark executable. It measures combinations of world count, body count,
and 1/2/4/8 threads; it does not automatically sweep all 64 supported thread counts.
:::

```bash
make control-native
fractal-gas-web/build-control-native/control/fg_control_benchmark \
  > /tmp/control-native-benchmark.csv
```

:::{div} feynman-prose
The benchmark's scene and measurement conditions are documented in
`fractal-gas-web/web/lab/benchmarks/README.md`. Use the machine you intend to run
on, and measure the actual contact density and state sizes that matter to your
experiment before selecting a live deadline or thread count.
:::
