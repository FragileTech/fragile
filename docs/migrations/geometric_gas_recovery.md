# Geometric Gas chapter recovery

Edited only `docs/source/2_fractal_gas/convergence_program/17_geometric_gas.md`.
2729 lines replaced by 1278 lines / 7007 words. All 82 original formal and section labels remain.

## Preserved and recovered analytic content

- Normalized local measurement pipeline: corrected raw symmetric kernel versus row normalization; fixed-finite-swarm nearest-candidate and global limits; derivative convention explicit. Uses repaired normalized-weight/variance/Z-score calculus in14_a.
- UEPH: complete spectral proof with the required positive margin epsilonSigma−LambdaMinus. Full swarm derivative bounds remain explicit inputs.
- Matrix regularity: complete resolvent proof of inverse-square-root derivative bound, avoiding false commuting matrix chain rule.
- Stratonovich conversion: exact full-field contraction; zero correction for position-dependent velocity noise. No false div_x D/2 drift.
- Alignment: complete degree-weighted momentum/energy identity and normalized-Laplacian similarity proof. Moving degrees and Euclidean norm costs retained. References full frozen OU covariance proof recovered in15 from4_ymmg.
- Quadratic moments: exact acceleration and diffusion-trace identities. Adaptive acceleration contributes zero directly to positional variance.
- Lyapunov: complete pointwise perturbation-transfer proof, plus direct conservative geometric kinetic quadratic Lyapunov proof with explicit constants, bounded adaptive-force cost and row-normalized alignment norm costs. Adds actual cloning-generator drift when that bound is established for the chosen functional.
- Harris: conservative invariant-law theorem with drift and actual minorization data; QSD normalization and killed-process criteria distinguished. Existing backbone component proof references and support label retained.
- LSI: full-gradient actual-law criterion using four complete15routes; geometric form and density comparisons proved explicitly. Velocity-only full-law LSI removed using spatial test. Full normalized entropy identity includes killing/boundary normalization; modified derivative closure uses15and10. No false LSI→velocity-generator spectral-gap inference.
- Commutator: exact missing mixed derivative restored; weighted Young bound retains velocity-weighted Fisher term. Constant-diffusion second-derivative dissipation remains available for absorption, not falsely bounded by first-order Fisher.
- Macroscopic transport: complete Poincare plus conditional second-moment proof retained; velocity-average Sobolev domain stated.
- Mean-field: correct normalized coefficient integral order, alignment and actual clone generator included; precise finite-time synchronous coupling and separate empirical sampling term; stationary marginal LSI passes with same constant. No dimension-free empiricalW2 rate or viscosity-as-density-perturbation claim.
- Concentration: complete Herbst proof and joint-LSI empirical variance consequence.
- Exponential tails: backward generator chain rule, actual exponential drift, QSD eigenvalue correction. Gives exponential moments/tail probabilities; no pointwise Gaussian density inference.
- Density derivatives: exact compact-interior logarithmic derivative bounds without false compact support from confinement.
- Variance-to-deviation support lemma retained with exact proof.
- L2 support corollary: requires actual generator form to dominate relevant Poincare form; kinetic/QSD qualification explicit.
- Holonomy: Ambrose–Singer retained; small-loop theorem corrected to shape-controlled family with curvature-squared remainder in addition to curvature-gradient remainder.
- Raychaudhuri: full geodesic derivation retained, dimensions and signature explicit.
- Reynolds: full moving-domain proof, time-dependent metric volume correction added.
- Voronoi face: exact Euclidean correction formula retained and curved estimate includes velocity amplitude and separation/shape control.
- Divergence remainder: direct Lipschitz bound avoids invented symmetry cancellation.
- Discrete Raychaudhuri: complete proof under differentiated volume consistency; material-cell case proved via Reynolds and variance identity. Voronoi reconstruction needs normalized flux defect and time derivative, which shape regularity alone does not establish.
- Metric/connection/curvature definitions retained with correct derivative regularity and Laplace–Beltrami drift distinction; no unsupported field-equation identification.

## Sources

Current chapter and support proofs read in full. Old baseline17 and archived geometric/LSI routes were searched; the same archived11/15 routes had already been audited in the15recovery and supplied no valid automatic arbitrary-QSD LSI closure. Reused complete10/15proofs and repaired14_a primitives. Weighted OU route is recovered from `docs/source/4_ymmg/01_convergence.tex` in15. Standard Harris result verified against the authors' paper https://www.hairer.org/papers/harris.pdf . Original geometric bibliography citations remain and resolve.

## Checks

- All 82 original labels retained; no removed labels or external reference remapping needed.
- 26 proof/section references and 13 document links resolve.
- Directive fences balanced; 174 display-math fences paired; single H1.
- `git diff --check` passes.
- No removed framework terminology, appendix proof-home prose, or review/status scaffolding.
- All three bibliography keys resolve.
- 100 random weighted-alignment identity checks: max absolute error 3.410605131648481e−13.
- 100 noncommuting positive-matrix derivative checks: max ratio to proved bound0.9384866942851092; max relative finite-difference error3.5197809017671038e−09.
