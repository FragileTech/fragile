(sec-optimization-laboratory)=
# Optimization Lab

:::{div} feynman-prose
In Optimization Lab, each walker is one proposed answer to an optimization problem.
Its complete state contains every coordinate the objective needs. The view lets
you inspect where those answers move, which ones survive, and whether their
objective values improve. Changing the camera does not change the problem.

The application lives at `/optimization/`, alongside the control laboratory at
`/lab/`. It shares the C++ swarm machinery and browser tooling, but its walkers
explore benchmark functions directly. Start with the short experiment below,
then use the two views to separate spatial motion from objective improvement.
:::

(sec-optimization-start)=
## Build and run the first experiment

:::{div} feynman-prose
From the repository root, build the native engine, build its WebAssembly version,
and start the local server. The browser build uses the existing Emscripten SDK
setup and pinned JavaScript packages. The first setup may download dependencies;
console emulator builds and ROMs are not required.
:::

```bash
make optimization-native
make optimization-web
make optimization-lab
```

:::{div} feynman-prose
Open [http://127.0.0.1:8081/optimization/](http://127.0.0.1:8081/optimization/).
Use HTTP rather than opening `index.html` as a local file. The initial session is
paused: **Euclidean Gas**, **256 walkers**, **3D Rastrigin**, **seed 7**, and the
**spatial view**. **Objective** defaults to **Minimize**, and **Perturbation** to
**Gaussian (mean 0)** with **Standard deviation = 1**. One simulation worker owns
the engine. The evaluation budget defaults to **0**, meaning unlimited.

1. Wait for **C++ / WebAssembly**, then press **Step**. Inspect the objective values
   and the change in the cloud before advancing again.
2. Press **Run**, let the swarm advance, and press **Pause**. The convergence chart
   follows best and mean objective values. **Convergence axis** defaults to
   **Evaluations** and also offers **Iteration**; an inverse-hyperbolic-sine
   (`asinh`) scale compresses large value ranges vertically.
3. Select a walker to inspect its complete coordinates and available diagnostics.
4. Change the dimension or algorithm in the configuration panel, then press
   **Apply and reset** once. This starts a new session with the edited settings.
5. Use **Reset** to restart the active configuration and seed. Save a recording
   first if you want to keep the current history.

Settings are a draft until applied. Camera, projection, coloring, and other view
changes affect the display immediately and preserve the run. An iteration is an
engine operation, not a screen refresh: a slower renderer does not increase the
integration time step. For reproducibility, retain the engine version as well as
the configuration and seed. Python and C++ do not promise identical random streams
from the same seed.
:::

(sec-optimization-algorithms)=
## Choose how the swarm explores

:::{div} feynman-prose
Choose **FMC**, **Wave**, **Wave Jump**, **Graph**, or **Euclidean Gas** under
**Algorithm**. All five support **Objective → Minimize** and **Maximize**. The
selection score is the negative of the function value for minimization and the
function value itself for maximization. Surfaces, colors, and the inspector keep
the raw objective convention: maximizing does not flip the landscape upside down.
**Best** decreases during minimization and increases during maximization.

Internal fitness can also depend on diversity, so a walker's fitness and its
objective value answer different questions. Read the objective when comparing
solutions; use fitness and companion overlays to understand selection. **Fitness
(pre-step)** and **Pre-step fitness** label the selection-stage value retained in
a snapshot. It is not recomputed at the displayed post-step coordinates.

**Wave** and **Graph** use the existing C++ `FractalGas` and `FractalTree`
implementations in a continuous benchmark environment. A move earns the change
in selection score: objective decrease when minimizing, increase when maximizing.
Cumulative reward is anchored to a common baseline, so different path lengths do
not give a walker credit merely for taking more steps. Cloning carries the
corresponding state and reward history together. Graph also retains ancestry;
its population can grow beyond its starting set of walkers.

**FMC** and **Wave Jump** reuse the existing `ArcadePlanner` over a Wave search.
They keep one committed position and explore possible continuations from it.
**Step** advances one search iteration or executes one chosen action; it does not
complete an entire planning cycle. During search, the cloud can move while the
committed position stays fixed. The next search starts from the position reached
by the preceding execution.

Both planners default to **Search horizon = 32**. FMC chooses the most represented
first action in the surviving search ancestry and executes it before searching
again. Wave Jump defaults to **Execute shared ancestry prefix**: it executes the
common initial path when surviving walkers share one. Without a shared prefix,
it can extend the search to **Maximum horizon (0 = automatic)**; automatic means
twice the search horizon, capped at 4096. At that limit it falls back to one action
from the best branch. Disabling the shared-prefix option executes the selected
best branch's complete path. Each action along a chosen path still requires its
own execution step.

**Euclidean Gas** maintains positions and velocities. It selects companions,
computes fitness, applies cloning at the configured interval, and advances motion
with the core BAOAB integrator. Cloning includes position jitter and velocity
restitution; the kinetic step has independent noise in each coordinate and an
optional potential force. The force follows the selected objective direction.
Time step, friction, inverse temperature, cloning settings, and companion
strategies belong to the algorithm configuration. Adaptive fitness forces,
anisotropic diffusion, and viscous and curl extensions remain outside this app.

**Perturbation** selects the noise distribution. **Gaussian (mean 0)** and
**Uniform (mean 0)** both interpret **Standard deviation** as the per-coordinate
standard deviation, rather than the uniform distribution's half-width. For Wave,
Graph, FMC, and Wave Jump, this is a position displacement in coordinate units per
proposal step. Changing the domain bounds therefore does not silently rescale a
new run's perturbations. The old domain-fraction `proposal` setting is translated
only when loading legacy configurations without an explicit `perturbation_std`.

For Euclidean Gas, the perturbation is the random velocity input to BAOAB's
O stage, scaled by the existing thermal-noise coefficient. Gaussian noise with
standard deviation 1 retains the configured temperature convention. Increasing
the standard deviation increases the kick variance. Uniform noise with the same
standard deviation matches that variance but does not reproduce the Gaussian
Ornstein–Uhlenbeck transition law. The perturbation setting does not replace the
separate **Clone position jitter** control.

The bounded-domain mode marks out-of-bounds walkers as dead. Periodic boundaries
instead wrap coordinates across the box when enabled. Wrapping changes the
boundary rule; it does not make an arbitrary benchmark smooth across that seam.
Wave, Graph, and Euclidean Gas stop when every walker becomes invalid. FMC and
Wave Jump can use the shared planner's all-dead search fallback while their
committed position remains valid; an invalid committed position ends the run.
Reduce the perturbation standard deviation or kinetic step size, review the
bounds, and apply a reset to recover.
:::

(sec-optimization-benchmarks)=
## Choose a benchmark and its dimension

:::{div} feynman-prose
The catalog contains **37 benchmarks**: the 13 functions below and all 24 functions
in COCO's noiseless, single-objective BBOB suite. Each entry supplies dimension
restrictions, default bounds, parameters, gradient availability, and any known
minimum or reference value. The table below describes minimization experiments.
A catalog reference minimum remains a minimum when you choose **Maximize**; it
is not a maximization target.
Changing the number of displayed axes never reduces the dimension passed to the
objective. A ten-dimensional walker still has ten coordinates even when you see
only three of them.
:::

:::{div} feynman-added
| Benchmark | Dimension | What to inspect |
|---|---|---|
| Sphere | At least 1 | Radial concentration toward the origin. |
| Quadratic Well | At least 1 | How curvature changes motion and selection. |
| Mexican Hat | At least 1 | A degenerate set of minima when the tilt is zero; a tilt changes the preference. |
| Rastrigin | At least 1 | Competing basins in a regularly oscillating landscape. |
| EggHolder | Exactly 2 | Irregular basins and nonsmooth locations. |
| Styblinski–Tang | At least 1 | Competing wells with unequal depths. |
| Rosenbrock | At least 2 | Progress along a curved, narrow valley with adjacent-coordinate coupling. |
| Easom | Exactly 2 | Discovery of a narrow well in a broad, nearly flat domain. |
| Holder Table | Exactly 2 | Separated wells and nonsmooth locations. |
| Lennard–Jones | 3 times the atom count | Atomic configurations and pair interactions. |
| Constant | At least 1 | Exploration without spatial objective preference. |
| Stochastic Gaussian | At least 1 | Selection driven by sampled noise without a spatial optimum. |
| Mixture of Gaussians | At least 1 | Basins of a negative log mixture density. |
:::

:::{div} feynman-prose
For a Gaussian mixture, the realized component centers, deviations, and weights
are part of the saved configuration. Component centers are reference points, not
guaranteed minima of the combined objective. Keep those realized parameters when
repeating an experiment.

Stochastic Gaussian draws an objective sample during optimization. Its displayed
surface is the expected value, zero; recorded walker heights retain their sampled
values. Rotating the camera or resampling a slice consumes no simulation random
numbers. Potential forces are disabled for this benchmark because its noise is
not a differentiable spatial landscape.

Smooth functions in this original group use analytic gradients. EggHolder and
Holder Table use a central finite-difference fallback at nonsmooth locations;
this gives a numerical force, not a proof that a derivative exists there.
Singular or nonfinite evaluations,
such as coincident Lennard–Jones atoms, remain invalid. They do not become best
candidates, and the surface omits triangles with invalid samples.
:::

:::{div} feynman-prose
Choose an entry beginning **BBOB f** to use the official COCO problem. The IDs are
`bbob_1` through `bbob_24`; accepted dimensions are **2, 3, 5, 10, 20, and 40**.
The **COCO instance** selects one of **1–1000**, default **1**, independently of
the swarm's random seed. The default domain is **[-5, 5]** in each coordinate.

An instance includes the shifts, rotations, and other transformations specified
for its function. Its optimum value can differ from zero and from another
instance's optimum. Retain the function, dimension, instance, and resolved
reference minimum together. In BBOB minimization runs, **Gap to minimum** displays
the best observed value minus that instance's reference minimum. It is hidden
when maximizing. The suite's function groups are listed in the
[official BBOB overview](https://coco-platform.org/testsuites/bbob/overview.html).
:::

:::{div} feynman-added
| BBOB IDs | Functions in numerical order |
|---|---|
| `bbob_1`–`bbob_5` | Sphere; separable Ellipsoid; separable Rastrigin; Bueche–Rastrigin; Linear slope. |
| `bbob_6`–`bbob_9` | Attractive sector; Step ellipsoid; original Rosenbrock; rotated Rosenbrock. |
| `bbob_10`–`bbob_14` | Rotated Ellipsoid; Discus; Bent cigar; Sharp ridge; Different powers. |
| `bbob_15`–`bbob_19` | Rotated Rastrigin; Weierstrass; Schaffer F7, condition 10; Schaffer F7, condition 1000; Griewank–Rosenbrock. |
| `bbob_20`–`bbob_24` | Schwefel; Gallagher 101 peaks; Gallagher 21 peaks; Katsuura; Lunacek bi-Rastrigin. |
:::

:::{div} feynman-prose
Use these functions to isolate different difficulties. On the
[rotated Ellipsoid](https://coco-platform.org/testsuites/bbob/functions/f10.html),
the directions of steep and shallow change no longer align with the coordinate
axes. A displacement that moves far enough along the valley can overshoot across
it. Compare this with the separable Ellipsoid before attributing the problem only
to a small step size.

[Gallagher's peaks](https://coco-platform.org/testsuites/bbob/functions/f21.html)
have independently arranged locations and heights; finding one attractive basin
does not reveal a dependable route to the best one.
[Katsuura](https://coco-platform.org/testsuites/bbob/functions/f23.html) is rugged
and repetitive across fine scales, so a coarse surface can conceal structure the
optimizer still evaluates. [Lunacek bi-Rastrigin](https://coco-platform.org/testsuites/bbob/functions/f24.html)
combines two broad funnels with many smaller local optima. Watch whether the
swarm concentrates in one funnel before it has adequately explored the other.

For BBOB, **Potential force** starts disabled. Enabling it explicitly uses central
finite differences with coordinate step `1e-5 * max(1, abs(x[k]))`; one gradient
costs **2 × dimension** objective calls. These calls count against the evaluation
budget. A finite-difference force on a rugged or nonsmooth function is a numerical
choice, not an analytic gradient supplied by COCO. Surface sampling remains
separate from optimization and does not consume that budget.
:::

(sec-optimization-views)=
## Read the spatial and landscape views

:::{div} feynman-prose
Choose **View → Spatial** to place walkers using the **Horizontal**, **Depth**,
and **Vertical** coordinates. **Color** offers **Objective**, **Fitness
(pre-step)**, and **Uniform**. A two-dimensional benchmark occupies a plane.
Drag to orbit, right-drag to pan, scroll to zoom, and use **Reset camera** to
recover the initial view. Gold identifies the current best walker; white marks
your selection. For FMC and Wave Jump, a larger marker identifies the committed
position. It is teal unless the best or selection highlight overrides that color.
The other markers are search candidates. The snapshot appends the committed
position after the search rows, so a planner configured with 256 walkers displays
257 rows. **Role** in the inspector distinguishes **Search walker** from
**Committed position**.

**Mean** summarizes the valid rows currently displayed, including a planner's
committed row. **Best** retains the best valid objective query observed anywhere
in the run, including finite-difference probes and search candidates later
discarded. It can therefore be better than every visible walker. Gold marks the
best current row, rather than the historical query behind **Best**. A candidate
found during planning is not necessarily a position the planner has executed.
Inspect the committed row's **Objective** to track the executed route; its
pre-step fitness is not a search-selection diagnostic.

Open **Display and slices** for **Point size**, **Opacity**, and **Links**.
The link choices are **Distance companions**, **Clone companions**, and
**Ancestry**; relationships appear when the algorithm supplies them. **Trails
(up to 96 walkers)** shows recent recorded motion, or the selected walker's trail
when one is selected. Trails break at cloning events and when planners begin a
new search from their committed position. **Show objective slice** adds a sampled
plane to the spatial view without changing the run.

Choose **View → Landscape** to use the **Horizontal** and **Depth** coordinates
for the domain and an `asinh`-scaled objective for height. This monotone transform
compresses large objective differences; the displayed height is not the raw
objective value. The surface and walkers share the same transform. **Height
scale** adjusts the vertical magnification, and **Surface resolution** changes
the sample grid. The surface and contours are sampled from the same C++ evaluator
used by the optimizer. Read raw objective values in the inspector and metrics.

For a higher-dimensional function, you must also choose fixed values for the
coordinates absent from the surface. That surface is one slice. Each walker is
drawn at the transformed height of its full-dimensional objective, so it need not
lie on the displayed surface. For example, two walkers can overlap
in the selected coordinates and have different heights because their other
coordinates differ. Move the slice coordinates to investigate that difference;
do not interpret the gap as a rendering error.

Click a walker, or enter its **Walker index**, to inspect its complete coordinates,
objective, pre-step fitness, status, leaf flag, cloned flag, and parent index.
Velocity components appear for Euclidean Gas. For a valid Lennard–Jones walker,
**Atom configuration** turns its coordinate triples into atom positions. One
swarm point represents an entire molecule: the atoms in the inspector are
components of that candidate, rather than additional optimization walkers.
:::

(sec-optimization-recordings)=
## Save, inspect, and replay a run

:::{div} feynman-prose
**Record history** is off by default and takes effect on reset. With recording
off, only the current frame is retained; replay, trails, convergence history, and
exports are unavailable. Enable **Record history** before **Apply and reset**
to retain the run history. Imported recordings remain available for replay and export.

Use **Save recording** to export a versioned `.fgopt` file. It contains the active
configuration, realized benchmark parameters, engine version, metrics, and
recorded snapshot arrays. These snapshots preserve the full coordinate vectors,
so a replay can use a different projection without rerunning the optimization.

Dragging the timeline pauses computation and displays a recorded frame. **Play
replay** advances through recorded frames at the selected **Replay speed**; the
button becomes **Stop replay** while playback runs. **Latest** returns to the
newest frame and permits continuation of the still-live session. Scrubbing does
not restore an earlier random-generator or algorithm state.

**Load recording** validates the file format and array shapes before replacing
the displayed session. Loaded recordings provide exact visual replay of their
frames. The current engine identifies itself as `fgopt-3`; earlier `fgopt-1` and
`fgopt-2` files retain exact replay of their saved frames. Exporting an imported
recording preserves its original engine identifier and stored metrics, including
the earlier engine's evaluation-count and best-value semantics.

The **Reset** button becomes **Rerun settings**: use it to start a fresh run with
the saved settings, including realized mixture parameters. This rerun uses the
current engine, so an older engine's trajectory need not be regenerated.
**Run** and **Step** remain disabled for the imported history itself. A `.fgopt`
file is a frame recording, not a resumable engine checkpoint; a fresh rerun starts
at its initial state. Python `.pt` recordings are not imported by this application.

The recording budget is **64 MiB**. The Lab pauses before the next frame would
exceed it and retains the existing recording for export. Larger populations and
dimensions fill that budget sooner. Save the current history, then reset to begin
a new recording. If an imported file is rejected, keep the current session and
inspect the reported validation error rather than treating a partial file as a
usable run.
:::

(sec-optimization-budgets)=
## Compare results at an evaluation budget

:::{div} feynman-prose
A screen frame is not a unit of search effort. Neither is an iteration: one
iteration can ask the objective many times, especially when a finite-difference
force is enabled. An evaluation budget gives each run a limit on those queries.
IOHanalyzer's **Fixed-Budget Results** analyzes solution quality against function
evaluations; it is an analysis view of the continuous problems already in the
catalog. See the [IOHanalyzer GUI guide](https://iohprofiler.github.io/IOHanalyzer/GUI/).

Set **Evaluation budget (0 = unlimited)** (`max_evaluations`) before **Apply and
reset**. A positive budget must cover initialization. The counter includes every
optimization objective query, including initialization,
finite-difference force probes, and evaluations of proposals that are later
discarded or invalid. Invalid values cannot improve **Best**. Drawing surfaces,
changing slice coordinates, and reading the reference optimum do not increment
the optimization counter.

The engine checks the next complete step before starting it. If its evaluation
bound exceeds the remaining budget, the run pauses with its last frame and
history unchanged.
It does not shrink the population, shorten a planner search step, or partially
execute an integration step to fill the remainder. Graph uses a conservative
bound because its population can change, so it can stop with unused budget even
when the next step would actually have been cheaper. Report the final actual
**Evaluations** count alongside the requested cap.

For a first comparison, choose one BBOB function, dimension, and COCO instance;
keep those fixed while changing algorithm and swarm seed across separate runs.
Set the same evaluation cap, then choose **Convergence axis → Evaluations**.
The **Iteration** view remains useful for understanding the algorithm's phases.
Keep minimization selected for the ordinary BBOB task; maximization explores a
different objective direction even though it uses the same function evaluator.

Use **Export CSV** to save checkpoint results for IOHanalyzer. The export contains
`evaluations`, `best`, `function`, `algorithm`, `dimension`, and `run`, followed by
`instance`, `seed`, `objective`, `budget`, `engine`, `perturbation`, and
`perturbation_std`. Enable IOHanalyzer's **use custom csv format**, then map
the first six columns to evaluation counter, function values, function ID,
algorithm ID, problem dimension, and run ID respectively. Set its minimization/maximization
option to match the run. Keep the evaluation column mapped: these checkpoints
are not sequential single-query observations. See the
[custom CSV data format](https://iohprofiler.github.io/IOHanalyzer/data/).

Each CSV row describes a saved snapshot checkpoint; duplicate evaluation counts
and checkpoints without a finite best value are omitted. The actual evaluation
count and best value at that checkpoint are recorded; the export does not invent the
query at which an improvement occurred between snapshots. For fixed-budget
analysis, choose budgets supported by the recorded checkpoints and inspect how
the analyzer handles sparse data or runs that stop early. Do not interpret a
plotted interpolation as an additional measured result. For shifted BBOB
instances, compare raw values within the same instance or derive an objective
gap using that run's saved `reference_minimum` before aggregating instances.
When combining different parameter variants with the same algorithm and seed,
assign distinct algorithm or run IDs in the combined data before import so the
analyzer does not merge their checkpoints into one run.

CSV exports use IOHanalyzer's custom format. The app also retains `.fgopt` visual
recordings; it does not emit an official COCO observer/postprocessing archive.
Discrete pseudo-Boolean optimization (PBO) suites are not included in this
continuous-domain integration.
:::

:::{warning}
:class: feynman-added

Using the official COCO evaluator does not by itself reproduce an unmodified
official BBOB experimental protocol. The Lab stores walker coordinates as
float32; maximization, periodic wrapping, custom domain bounds, and optional
finite-difference potential forces also change the experimental procedure or
numerical assumptions. Identify these settings when comparing or reporting runs.
The custom CSV export is not an official COCO observer archive.
:::

(sec-optimization-engine)=
## Extend the engine and check changes

:::{div} feynman-prose
The C++ optimization engine owns benchmark evaluation and swarm state. The browser
worker calls its handle-based C API to create a session, step it, request a
snapshot, sample objectives, and destroy it. The main thread receives transferred
snapshot buffers and draws them with Three.js. It does not implement a second
version of the objective or advance the algorithm during rendering.

`fg_swarm_core` supplies the shared `FractalGas`, `FractalTree`, and
`ArcadePlanner` implementations. Optimization's Wave, Graph, FMC, and Wave Jump
adapters configure those classes; they do not reimplement the swarm or planning
algorithms. The config IDs are `wave`, `graph`, `fmc`, and `wave_jump`; the general
Euclidean Gas implementation uses `euclidean`.

BBOB evaluation compiles the pinned official **COCO 2.8.2** C source at revision
`e5d068f69e36f346c86cc2934413369abe36fc22` into both native and WebAssembly builds.
The app neither rewrites these benchmark formulas nor adds a replacement
optimizer for them. The active configuration records `coco_version`,
`coco_problem_id`, and `reference_minimum`. The COCO adapter owns separate
simulation and display problem objects. Both evaluate the same instance; only
the display object is used to read the reference optimum. Visualization cannot
alter the simulation object's counters or reveal an optimum to its search.

Start in `fractal-gas-web/src/optimization/`: the benchmark implementation supplies
the catalog, values, gradients, initialization, and boundary behavior; the engine
interface supplies settings, populations, and algorithm factories. A new benchmark
needs a catalog descriptor and evaluator, dimension and parameter validation,
gradient behavior, and reference tests. Use the same evaluator for visualization
sampling, and keep stochastic display sampling independent of the simulation
random stream.

A new optimizer implements the algorithm interface and calls
`register_algorithm(id, name, velocity, factory, parameters)`. The optional
`parameters` array describes its controls: each entry has `id`, `label`, `type`,
and `default`, with `min` and `max` for numeric limits or `options` for an enum.
Supported types are `number`, `integer`, `boolean`, and `enum`; enum options are
`[[id, label], ...]`. The frontend renders these descriptors automatically. The
factory reads custom settings from `Settings::json` and validates the values
needed by its algorithm.

For a new proposal distribution, implement
`Perturbation::sample(position, delta, dimensions, rng)` and register its factory
with `register_perturbation(id, name, factory, parameters)`. Write one displacement
vector into `delta` using the supplied random generator. The factory receives the
benchmark and configuration, allowing parameters or state-dependent proposals.
Its parameter descriptors use the same schema as algorithm controls and populate
the **Perturbation** UI automatically. The benchmark environment adds the sampled
vector to position; Euclidean Gas passes it through the BAOAB thermal coefficient
as velocity noise. Account for that difference when designing a distribution.

Return the shared population and metric fields, and expose optional velocity,
companion, or ancestry data only when they have defined meaning. The renderer and
recording format can then consume the result without requiring an
algorithm-specific copy of the application. Keep stored objective values raw;
use `Settings::score` and `Settings::better` when the algorithm needs the selected
optimization direction.

Route objective queries through the benchmark's optimization evaluation path so
evaluation counts and best-observed values include all attempted candidates.
Implement `next_evaluations_upper_bound()` for a new algorithm; the session uses
it to admit whole steps within `max_evaluations`. Account for initialization and
any finite-difference queries without changing the underlying optimizer's step
semantics. Keep display sampling on the separate evaluation path.

The C API is declared in `c_api.h`. Returned strings and snapshot pointers are
borrowed; copy data needed beyond the next mutation of that session. The snapshot
contains a header followed by full-dimensional rows and diagnostics. Treat its
version and documented layout as an interface contract when changing producers,
the worker decoder, or recording validation. Planner snapshots append their
committed-position row after the search walkers; consumers must preserve that
role when interpreting counts, diagnostics, and replay.

The benchmark environment uses 24-bit action IDs as perturbation seeds, exactly
representable in the shared planner's float action recorder. Given the same
engine, configuration, starting state, action ID, and proposal-step count, the
environment repeats the same perturbations and stochastic objective sample.
Thus executing a recorded search edge does not draw a different future. Planner
rewards are changes from their search root's score; the adapter adds that root
baseline back when exposing an absolute candidate score. This preserves the
objective ranking across searches without changing the planner's reward logic.

Optimization uses MT19937-64 with explicit portable bit mappings for uniform real
and integer draws, shared by the native and WebAssembly builds. This makes their
random draws comparable without depending on a platform's standard-library
distribution implementation. Retain the engine version when rerunning settings;
recorded-frame replay uses the saved arrays directly and does not require
regenerating those draws. Python uses its own random streams, so matching seeds
alone does not establish Python/C++ operator parity.

Two details matter when comparing Euclidean Gas with its Python reference. The
reference `random_pairing` operator pairs live rows and leaves dead rows pointing
to themselves. The C++ operator preserves that rule; the optimization host then
replaces a dead cloning donor with a live donor so cloning can revive the row.
This revival is a small host extension, not a change to the pairing operator.
Also, when **Clone every N iterations** skips a cloning update, the engine still
consumes the companion, decision, and proposed-clone random draws before
discarding that proposal, following the reference's draw order. Removing those
draws would change subsequent motion even on iterations without applied cloning.

For BAOAB operator comparisons, the Python `integrator="baoab"` reference now
applies one half-duration force kick in each B stage. The C++ core follows that
force-kick convention. The separate Boris path is unchanged by this correction.

Run the focused checks after changing the engine or browser contracts:
:::

```bash
make optimization-test
```

:::{div} feynman-prose
Relevant checks include benchmark values and gradients, both objective
directions, perturbation moments and replay, cloning and kinetic operators,
cumulative-score ranking, committed planner motion, deterministic resets,
recording round trips, and native/WebAssembly agreement. Browser verification
should exercise all five algorithms, both views, dimension changes,
Lennard–Jones inspection, loading all supported recording engine versions, and reset
while a worker request is outstanding. The **Step** and **Draw** timings report
simulation and rendering duration separately; either can explain a slow-looking
experiment.

For BBOB, check all 24 functions, supported dimensions, multiple instances,
reference values, native/WebAssembly agreement, and independence of display
sampling. Budget checks should cover initialization rejection, finite-difference
costs, the last admitted step, Graph's conservative bound, and best values from
discarded candidates. Verify CSV column mapping and sparse checkpoints alongside
`.fgopt` compatibility.

The browser CI runs Chromium and Firefox on `ubuntu-latest`. For Ubuntu 24.04,
reproduce its Xvfb and software Mesa setup, using `LIBGL_ALWAYS_SOFTWARE=1` and
`OPTIMIZATION_FIREFOX_HEADLESS=0`; this supplies the WebGL context needed by the
renderer. Startup checks report browser and WebGL errors immediately instead of
waiting for an unexplained readiness timeout. Reproduce that display setup when
investigating a Linux CI startup failure.

Geometry and fluid benchmarks, additional optimizers, automated experiment suites,
adaptive or viscous physics, and resumable checkpoints are outside this release.
For the related continuous-control engine and its separate world-state contracts,
see {doc}`control_lab_architecture`.
:::
