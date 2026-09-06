# Changelog

Unreleased
----------

* Add a six-page control-laboratory user guide covering setup, every control, scene editing, replay and storage, experiments, diagnostics, and engine extension contracts, with browser screenshots and worked examples.

* Support up to 64 control-engine execution threads in the browser and Python, prewarm the selected browser pool, and synchronize empty batch partitions safely.

* Align the control lab with the Fragile Tech documentation identity and add the Violet Circuit kart environment, ordered lap checkpoints, replayable lap progress, and extensible environment renderers and score readouts.

* Add iCEM and MPPI control-lab baselines with generic parameter controls, batched rollouts, warm starts, exact search checkpoints, and measured simulator-work comparisons.

* Add generic actuator, world-extension and controller registries; kart and independent-thruster dynamics; exact planner checkpoints; reproducible benchmark suites and fork comparisons; compressed persistent replay with events; multi-selection, templates and unit-bearing editor controls; and native/browser performance and physics inspection.

* Add reusable agent archetypes, distinct kart/rocket/drone models, declarative visual kits, mixed crowd instancing, and complete world-motion recording with playback, exact seeking and continuation in the control laboratory.

* Add a custom C++ continuous-control engine with parallel world batches, compact complete-state snapshots, native Python bindings, Wave/FMC planning, optional cloning-aware exploration trees, and a browser laboratory with five editable cyberpunk scenarios and original 3D assets.

* Make the full `make physics` workflow run end to end on the default configuration: per-group Bayesian fits, exact origin statistics for FFT correlators, float64 weighted AIC window fits with a signal-to-noise cut, working prior seeding and data-scaled amplitude priors, electroweak lag units and frame alignment, geodesic edge lengths for multiscale scale selection, deterministic curl estimation, and dashboard fixes (frame-restricted geodesic cache, chunked distance statistics, consistent AIC time units, stale-output clearing). See reviews/physics_fixes.md (second round).

* Retain twistor triplet lag statistics for measured mass-fit covariance and exclude unselected channels from covariance preparation.

* Correct physics integration, cloning records, duplicate-particle geometry, analysis timing, parity observables, multiscale fitting, measured covariance, and dashboard invalidation. See reviews/physics_fixes.md for numerical compatibility notes.

* Reorganize Fractal Gas as Volume 2, promote analytical proofs to main chapters, and remove the retired framework volume.
* Restrict mathematical downloads to published chapters and preserve migrated Fractal Gas URLs.

* Clarify the binding-field remark on beta-function sign dependence and confinement justification.

0.2.0 (2024-10-10)
------------------

* First release on PyPI.
