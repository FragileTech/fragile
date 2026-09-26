# Adaptive exploration architecture

The Optimization Lab registers `adaptive_fractal` for Wave, Graph, FMC,
Wave Jump, GAS and Euclidean Gas. This is an experimental optimization strategy,
not an implementation of CMA-ES and not an equilibrium-preserving sampler.

## Responsibilities

- `fractal/proposal.hpp`: optional frozen population, lineage, proposal family,
  branch and outcome contracts. Existing perturbations retain their draw paths.
- `adaptive.hpp/.cpp`: bounded local learning, covariance representations,
  proposal mixtures, geometry validation, and a shared trial evaluator. A
  movement kernel returns a complete candidate; the evaluator owns endpoint
  replication, paired selection and outcome feedback.
- `environment.cpp`: seed-action replay for Wave/Graph/planners, with inherited
  lineage in opaque states. Planner geometry and donor context stay frozen
  through search and committed execution.
- `gas.cpp`: Euclidean kernels reuse the shared selection, cloning and BAOAB
  integrator. Frozen kicks mirror stochastic alternatives; positions and
  velocities commit together. Direct proposals zero velocities on entry.
- `gas2017.cpp`: GAS uses the same evaluator and preserves its separate tabu
  memory and local-search implementation.
- `controller.hpp/.cpp`: population/scale scheduling and round evidence. It does
  not own an optimizer. `Session` constructs replacement adapters at safe
  boundaries, retaining the benchmark, global accounting and best coordinates.
- `basin.hpp/.cpp`: bounded heuristic basin evidence, compatible archive files,
  validation of imported reference values and restart placement. It never
  changes objective values or cloning fitness.

Live mutations use the existing serialized worker path. Manual population and
scale edits pin those parameters; Auto restores scheduling. Disabling automatic
restarts retains the current optimizer. A paused restart request waits for the
next step. Archive imports do not restore optimizer state or evaluation credit.
Geometry files are typed and validated before use; periodic changes retain
coordinates but clear geometry and avoidance confidence.

## Bounded scale rule (fgopt-7)

`adaptive_min_scale` and `adaptive_max_scale` set the statistical scale per
proposal draw (defaults 0.0001 and 1). An objective percentile in the frozen
valid population interpolates logarithmically between them. Ties use midranks;
an entirely tied or missing reference population uses the midpoint. A zero lower
bound uses linear interpolation and equal bounds fix the scale. In velocity
mode these are kick scales before BAOAB thermal integration, not position limits.

The evolution path affects covariance shape only; there is no accumulated scale
multiplier. Broad proposals cap their fourfold scale at the round maximum.
Difference proposals normalize the donor direction to unit coordinate RMS before
applying the selected scale, including subsequent kicks. Gaussian travel distance
is unbounded; these controls are not hard displacement limits.

Focused controller rounds narrow the effective upper endpoint within the user's
outer bounds. Manual bound edits pin automatic scale scheduling and restore the
full range. Live changes preserve geometry and use the existing safe boundaries.
Version-one geometry files still carry a neutral `scale: 1` field; old archived
scale values are validated but never restored. `adaptive_scale: false` remains
an API ablation using fixed `perturbation_std` clamped to the bounds; the UI uses
equal bounds for fixed scale. Legacy perturbations keep their original paths.

## Validation and scope

Native tests cover the adapters, deterministic replay, accounting, mode changes,
geometry validation, scheduling, pinning, archive compatibility and budget
resumption. WebAssembly tests compare native trajectories and recording round
trips. Chromium and Firefox exercise the live controls and archive workflow.
The historical path-based comparison is in `tests/optimization/reports/adaptive.md`.
It predates bounded scaling and must not be presented as results for fgopt-7.
The bounded revision has a separate reproducible comparison in
`tests/optimization/reports/bounded-scales.md`: stronger refinement on the smooth,
rotated and boundary cases at the tested range, but worse Rastrigin results.

Important experimental limits:

- The high-dimensional model compresses the signed update in the evidence
  subspace, retaining eight positive components and folding residual diagonal
  variance into the diagonal. This remains a low-rank approximation.
- Stochastic controller rounds use allowances rather than declaring stagnation
  from raw noise extremes. Basin representatives are independently re-evaluated
  with three samples before archive matching.
- Basin radius/matching and repeated-basin evaluation attribution are heuristics.
  A reported basin is not a certified local minimum.
- Recordings retain round/configuration/model summaries and archive state;
  format 4 references unchanged archive snapshots rather than repeating them.
  Internal trial provenance is bounded; reusable geometry is in archive files.
- GAS refinement results retain refined/candidate provenance and refinement
  evaluation costs in archive records. No new mandatory local optimizer is added.

The pilot and tests establish the checks described above; they do not establish
convergence guarantees or broad performance superiority.
