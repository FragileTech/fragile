# Changelog

Unreleased
----------

* Make the full `make physics` workflow run end to end on the default configuration: per-group Bayesian fits, exact origin statistics for FFT correlators, float64 weighted AIC window fits with a signal-to-noise cut, working prior seeding and data-scaled amplitude priors, electroweak lag units and frame alignment, geodesic edge lengths for multiscale scale selection, deterministic curl estimation, and dashboard fixes (frame-restricted geodesic cache, chunked distance statistics, consistent AIC time units, stale-output clearing). See reviews/physics_fixes.md (second round).

* Retain twistor triplet lag statistics for measured mass-fit covariance and exclude unselected channels from covariance preparation.

* Correct physics integration, cloning records, duplicate-particle geometry, analysis timing, parity observables, multiscale fitting, measured covariance, and dashboard invalidation. See reviews/physics_fixes.md for numerical compatibility notes.

* Reorganize Fractal Gas as Volume 2, promote analytical proofs to main chapters, and remove the retired framework volume.
* Restrict mathematical downloads to published chapters and preserve migrated Fractal Gas URLs.

* Clarify the binding-field remark on beta-function sign dependence and confinement justification.

0.2.0 (2024-10-10)
------------------

* First release on PyPI.
