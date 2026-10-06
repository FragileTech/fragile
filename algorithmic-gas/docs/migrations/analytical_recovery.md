# Analytical proof recovery

This record is excluded from the published book and mathematical downloads.
The source inventory records the original files and Git provenance. Retaining
a label preserves cross-references; it does not mean that the old statement
has been left unchanged. The chapter statements specify the proved hypotheses.

## Equilibrium profiles

`convergence_program/07_discrete_qsd.md` retains all five original formal labels.
The nearest-neighbor calculation includes the missing gamma factor and separates
Poisson sampling from kernel companion selection. The power-law equilibrium is
proved for its stated local replicator equation. The absorbing interval
calculation proves the sine eigenprofile and distinguishes a source-driven
stationary profile. The OU thermalization proof retains its kinetic reference
law; a finite reset-rate calculation demonstrates why jumps cannot be omitted.
The decorated Gibbs result now bounds a candidate profile's error by its full
stationary residual using the recovered mean-field resolvent contraction.

## Mass, Hellinger, and transport

`convergence_program/11_hk_convergence.md` retains every original formal label.
The main proof uses the exact root-mass decomposition, Hellinger–entropy
inequality, and LSI transport inequality. It therefore obtains the quantitative
convergence implication without the previous global density-ratio argument.
Fixed-size cloning replaces walkers; the corrected alive-mass proof counts
revival and kinetic survival and retains its variance floor. Conditional
independence, finite-time conditioning, and survival denominators are explicit.

The density arguments recover kernel domination, local Gaussian lower bounds,
positive QSD densities, and fixed-time smoothing. They do not infer a positive
global density minimum on an infinite-volume space. The linearization proof
uses the normalized killed equation and signed mass increments. Bounded
perturbation and regularity-bootstrap statements retain their operator inputs.
Finite atomic empirical measures are distinguished from the probability laws
used in entropy estimates.

## Exchangeability and the mean-field limit

`convergence_program/12_qsd_exchangeability_theory.md` retains every original
formal label. The finite sampling-with/without-replacement proof replaces an
incorrect exact infinite-mixture representation for a finite swarm. Kernel
equivariance yields QSD exchangeability. The covariance identity preserves the
empirical-variance term. A complete entropy-inequality and Gaussian-auxiliary-
variable proof gives bounded-observable variance of order `1/N` from a uniform
bound on total relative entropy. Joint LSI passes to a marginal and then its
weak limit. The frozen kinetic Gaussian remains distinguished from the actual
moving-swarm conditional law.

## Quantitative errors

`convergence_program/13_quantitative_error_bounds.md` retains every original
formal label. The transport proof uses LSI of the reference measure in the
correct entropy direction, with a symmetrized coupling for marginal bounds.
The empirical Wasserstein estimate retains the independent-sampling term.
Fourth moments, ordered splitting expansions, finite-time telescoping, and
Poisson-equation stationary perturbations are proved with their relevant test
and operator domains. A separate normalized-kernel perturbation proof handles
QSDs. Local defects, survival denominators, and physical-time mixing factors
remain in the final error estimate.

## Formal spot checks

Five hundred random finite probability pairs verified the mass/shape identity
and its entropy inequalities to floating-point precision (maximum identity
error `5.56e-16`). A noncommuting matrix calculation verified the BAOAB quadratic
coefficient. These checks supplement the written proofs; they are not proofs
of the analytical hypotheses for an arbitrary algorithm configuration.

The independent KL and hypocoercivity recovery record is in `kl_recovery.md`;
its explicit old-label correspondence is in `kl_label_map.json`.

## Downstream geometry and recorded observables

The formal statements in `3_fitness_manifold/03_curvature_gravity.md` now
carry the small-loop shape and curvature-squared bounds proved in the
Geometric Gas chapter. The discrete-connection error is proved by telescoping
edge transports. Curvature uses two distinct oriented tangent directions;
contracting both antisymmetric slots against one vector would give zero.
The Lorentzian comparison transport uses face maps and oriented boosts.
The cell-expansion theorem retains differentiated volume consistency and the
normalized Voronoi flux defect. Its relaxation corollary states the required
positive damping bound. The AdS metric calculation in `05_holography.md`
constructs and verifies the stated constant-curvature metric; identifying
that metric with a sampled geometry requires a separate estimate.

The implementation-exact propositions in `2_fractal_set/09_qft_calibration.md`
now retain the zero same-frame delta-to-right channels proved by the
Standard Model chapter's role partition. The implemented scalar phase is
distinguished from an SU(2) matrix transport. The runtime algorithms and
recorded-data APIs are unchanged.


## Foundational transition estimates

The composite independent-output displacement estimate now retains the
`N^(alpha_B-1)` normalization, every fractional power including
`V^(alpha_B/2)`, and the stochastic offsets. A separate common-measure coupling
bounds Wasserstein distance by total variation and either output diameter or
fourth moments. The Gaussian realization has a complete finite-history mixture
proof of local total-variation continuity; equality at identical random seeds
is no longer used as a substitute for a quantitative bound. Normalized-density
convergence proves the Feller integration step without an invalid Lebesgue
majorant. Malformed foundational directives were repaired and their labels
checked against the rendered HTML.

## Field equations and pressure

The finite-partition independent-sampling rate is proved from the multinomial
formula. Interacting concentration instead cites the recovered total-entropy
and full-law LSI results. The homogeneous Gaussian closure retains its exact
Fourier multiplier, with a positive box gap and the correct whole-space
low-frequency behavior. The integrated OU covariance gives the diffusion
coefficient directly. Gaussian pair-energy dilation and finite Gaussian mode
partition functions give pressure derivatives, while the jump quadratic form
gives a distinct stiffness. The prescribed two-term pressure has the corrected
crossover sign. The Ricci contraction retains spacetime dimension and the
cosmological term and explicitly uses its Einstein constitutive equation;
conservation and the Raychaudhuri identity alone do not imply that equation.


## Expansion and macroscopic closure

The cosmology chapter retains every original label while proving the actual
QSD centered-observable identity, quantitative excess-observable bounds,
Raychaudhuri comparison, explicit AdS and de Sitter metrics, and the Milne
zero-curvature expansion example. The information-closure implication is
proved; an overlapping-bit stationary process disproves its old converse.
Exact Markov lumpability and a quantitative Dynkin residual provide the
appropriate coarse-graining statements. Vacuum energy, mass density, pressure,
and curvature units are kept distinct. These classical results replace
unsupported physical identifications without changing the swarm algorithm.

## Additional validation

Two thousand randomized checks verify the corrected composite-power bound
and Gaussian Fourier multiplier bounds. The rendered HTML audit verifies
that labelled formal content has a target and is visible in Expert Mode.


## Complete coefficient regularity and geometric applications

The regularity chapters preserve the normalized mean/variance and composition
proofs and now handle the actual sequential greedy history probabilities.
The induction controls the sum of all coordinate derivatives independently
of the population size. It distinguishes smooth cutoffs from analytic
coefficient bounds and fixed-query derivatives from moving-walker derivatives.
The all-orders and composition calculations passed 1,154 numerical checks.

The geometry chapters identify the implemented spectral clipping and the
noise-scale normalization, prove volume-weight and quadrature estimates, and
separate a smooth Hessian metric from its clipped runtime counterpart. The
Hessian curvature formula cancels fourth derivatives explicitly. Holonomy,
connection reconstruction, Raychaudhuri, and discrete Gauss–Bonnet statements
retain their actual geometric and sampling conditions.

The holography chapter proves the correctly normalized Gaussian perimeter
limit and graph-cut variance estimate. The recovered joint entropy bounds
transfer the area limit to dependent sampling. Boundary-cell correspondence,
finite-matrix entropy identities, Gibbs residuals, static response, and explicit
constant-curvature metrics supply the other classical results. Forty-five
numerical or symbolic checks cover the geometric and entropy calculations.
