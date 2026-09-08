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
**spatial view**. One simulation worker owns the engine.

1. Wait for **C++ / WebAssembly**, then press **Step**. Inspect the objective values
   and the change in the cloud before advancing again.
2. Press **Run**, let the swarm advance, and press **Pause**. The convergence chart
   follows best and mean objective values over optimization iterations, using an
   inverse-hyperbolic-sine (`asinh`) scale to compress large value ranges.
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
All three algorithms minimize the selected objective. Their internal fitness can
also depend on diversity, so a walker's fitness and its objective value answer
different questions. Read the objective when comparing candidate solutions; use
fitness and companion overlays to understand selection. **Fitness (pre-step)**
and **Pre-step fitness** label the selection-stage value retained in a snapshot.
It is not a fitness recomputation at the displayed post-step coordinates.

**Wave** and **Graph** use the existing C++ swarm implementations with Gaussian
position proposals in a continuous benchmark environment. A move earns the
decrease in objective. Cumulative reward is anchored to a common baseline, so
different path lengths do not give a walker credit merely for taking more steps.
Cloning carries the corresponding state and reward history together. Graph can
also expose retained ancestry; its population need not have the same structure as
a fixed cloud of independent trajectories.

**Euclidean Gas** maintains positions and velocities. It selects companions,
computes fitness, applies cloning at the configured interval, and advances motion
with the core BAOAB integrator. Cloning includes position jitter and velocity
restitution; the kinetic step has isotropic noise and an optional potential force.
The time step, friction, noise, cloning settings, and companion strategies belong
to the **Algorithm parameters** configuration. This implementation does not
include adaptive fitness forces, anisotropic diffusion, or viscous and curl
extensions.

The bounded-domain mode marks out-of-bounds walkers as dead. Periodic boundaries
instead wrap coordinates across the box when enabled. Wrapping changes the
boundary rule; it does not make an arbitrary benchmark smooth across that seam.
If every walker becomes invalid, the session stops with an error. Reduce the
proposal or kinetic step size, review the bounds, and apply a reset to recover.
:::

(sec-optimization-benchmarks)=
## Choose a benchmark and its dimension

:::{div} feynman-prose
The catalog supplies each function's dimension restrictions, default bounds,
parameters, gradient availability, and any known minimum or reference value.
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

Smooth functions use analytic gradients. EggHolder and Holder Table use a central
finite-difference fallback at nonsmooth locations; this gives a numerical force,
not a proof that a derivative exists there. Singular or nonfinite evaluations,
such as coincident Lennard–Jones atoms, remain invalid. They do not become best
candidates, and the surface omits triangles with invalid samples.
:::

(sec-optimization-views)=
## Read the spatial and landscape views

:::{div} feynman-prose
Choose **View → Spatial** to place walkers using the **Horizontal**, **Depth**,
and **Vertical** coordinates. **Color** offers **Objective**, **Fitness
(pre-step)**, and **Uniform**. A two-dimensional benchmark occupies a plane.
Drag to orbit, right-drag to pan, scroll to zoom, and use **Reset camera** to
recover the initial view. Gold identifies the current best walker; white marks
your selection.

Open **Display and slices** for **Point size**, **Opacity**, and **Links**.
The link choices are **Distance companions**, **Clone companions**, and
**Ancestry**; relationships appear when the algorithm supplies them. **Trails
(up to 96 walkers)** shows recent recorded motion, or the selected walker's trail
when one is selected. Trails break at cloning events. **Show objective slice**
adds a sampled plane to the spatial view without changing the run.

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
frames. The **Reset** button becomes **Rerun settings**: use it to start a fresh
run with the saved settings, including realized mixture parameters. **Run** and
**Step** remain disabled for the imported history itself. A `.fgopt` file is a
frame recording, not a resumable engine checkpoint; a fresh rerun starts at its
initial state. Python `.pt` recordings are not imported by this application.

The recording budget is **64 MiB**. The Lab pauses before the next frame would
exceed it and retains the existing recording for export. Larger populations and
dimensions fill that budget sooner. Save the current history, then reset to begin
a new recording. If an imported file is rejected, keep the current session and
inspect the reported validation error rather than treating a partial file as a
usable run.
:::

(sec-optimization-engine)=
## Extend the engine and check changes

:::{div} feynman-prose
The C++ optimization engine owns benchmark evaluation and swarm state. The browser
worker calls its handle-based C API to create a session, step it, request a
snapshot, sample objectives, and destroy it. The main thread receives transferred
snapshot buffers and draws them with Three.js. It does not implement a second
version of the objective or advance the algorithm during rendering.

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

Return the shared population and metric fields, and expose optional velocity,
companion, or ancestry data only when they have defined meaning. The renderer and
recording format can then consume the result without requiring an
algorithm-specific copy of the application.

The C API is declared in `c_api.h`. Returned strings and snapshot pointers are
borrowed; copy data needed beyond the next mutation of that session. The snapshot
contains a header followed by full-dimensional rows and diagnostics. Treat its
version and documented layout as an interface contract when changing producers,
the worker decoder, or recording validation.

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
Relevant checks include benchmark values and gradients, cloning and kinetic
operators, Wave/Graph reward ranking, deterministic resets, recording round trips,
and native/WebAssembly agreement. Browser verification should exercise both views,
dimension changes, Lennard–Jones inspection, loading a recording, and reset while
a worker request is outstanding. The **Step** and **Draw** timings report
simulation and rendering duration separately; either can explain a slow-looking
experiment.

Geometry and fluid benchmarks, additional optimizers, experiment comparisons,
adaptive or viscous physics, and resumable checkpoints are outside this release.
For the related continuous-control engine and its separate world-state contracts,
see {doc}`control_lab_architecture`.
:::
