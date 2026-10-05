# Regional Gaussian and weighted-tail inputs

The reusable Rust API is `convergence_structural_tails`. Full proofs are in
Chapter 6, labels `lem-convergence-gaussian-regional-tail` and
`lem-convergence-moment-tail-transfer`. Its bounds retain unbounded Gaussian
noise and use the actual landscape gradient in the BAOAB center.

| API | Analytic inputs | Returned bound | Scope |
|---|---|---|---|
| `gaussian_region_budget` | Proved mean coordinate intervals; target analysis box; covariance eigenvalue bounds `sigma_min²`, `sigma_max²` | Landing lower bound and escape upper bound, with dimension and log density bound | One-particle conditional transition; correlated Gaussian coordinates allowed |
| `gaussian_tail_budget` | Dimension, center-norm upper bound, covariance operator-norm upper bound, cutoff, cost order | Gaussian mass and polynomial-cost tails, with log bounds | Chi-square exponential tilt plus Hölder for fractional cost order |
| `moment_tail_budget` | `moment_order=p`, proved `moment_bound=M`, `cutoff=R`, `cost_order=r<p` | Mass `min(1,M/R^p)` and cost `M/R^(p-r)` | Applies to the specified normalized law; finite sampled moments do not establish the input hypothesis |
| `coupled_tail_cost_upper` | Both marginal p-moments, cutoff, cost growth `c0+c1(norm(x)^r+norm(y)^r)` | Tail remainder on either marginal leaving the cutoff | Any coupling; cross terms bounded by Hölder, without independence |
| `baoab_position_center` | Actual position, velocity and landscape gradient, step size, friction | `x+bv-(hb/2)grad(V)(x)`, `b=h(1+exp(-gamma*h))/2` | Native positional center before the final force kick and velocity cap |
| `nonquadratic_gaussian_example` | Dimension | Explicit core/slow regional envelopes and conditional one-step tail budgets | Cosine perturbation of a confining quadratic, without modifying the algorithm |

The nonquadratic example is
`V(x)=norm(x)^2/2 + 0.2 sum_j cos(2*x_j)`. On all of space its Hessian lies
between `0.2 I` and `1.8 I`, its force perturbation norm is at most
`0.4 sqrt(d)`, and `V(x)>=norm(x)^2/2-0.2*d`. Its slow box is centered at
zero and its core box at `pi/2` in each coordinate, both with coordinate
radius `0.15`; their Hessian intervals are `[0.2,0.2357308]` and
`[1.7642692,1.8]`. Velocity coordinates lie within `[-0.1,0.1]` for these
regional hypotheses. The mean enclosures are uniform over those phase
regions, and landing is evaluated in the declared larger target boxes of
radius `0.3`. Conditional Gaussian fourth moments are bounded using the
largest admitted mean norm. They are one-step bounds, not uniform-time
stationary moment certificates.

For the parent's regional rate compiler, a landing probability cannot be
used as a proved coupled error-transfer coefficient by itself. The caller
must prove its region-weighted discrepancy inequality and supply moment
envelopes for the actual laws. In a permutation-invariant matched-pair law,
moments remain normalized averages and incur no factor of swarm size.
Survival-conditioned alive marginals use their own survival/alive-count
normalization; that moment input is distinct from an N-normalized alive
observable. Cutoffs and regional boxes are analysis choices, not support
restrictions on the Gaussian or on the unbounded landscape.

Exact real formulas are evaluated in f64. Log bounds and positive upper
floors prevent rare tails from becoming false zero guarantees; unsupported
overflow ranges return configuration errors. These evaluations are not
arbitrary-precision interval arithmetic. Regression tests compare weighted
Gaussian tails with independent quadrature and check dimension, correlated
couplings, population normalization and nonquadratic force centers.
