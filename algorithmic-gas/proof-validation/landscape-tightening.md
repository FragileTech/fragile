# Dimension and structural landscape rates

The [completion catalog](estimate-completion.md) adds the final source-expression
checks, seven parameter sweeps, global selected-source tail closure and the
shorter Rastrigin certificate. The coordinate-variance proof reduces the global
segment threshold to `sqrt(d)` with lower constant at least `1207`, independent
of population and dimension. A new 144-case native dataset and 3,537 independent
review checks pass; its saved native gradients are distinct from the original
long-segment experiment.

The Rust extension evaluates landscape profiles on declared well cores,
transition zones, slow-mixing zones and exterior shells. Every transport and
moment error is normalized by population size. Dimension enters the force,
Gaussian and tail constants explicitly.

| Input or estimate | Implemented bound or rate | Analytic requirement | Validation |
|---|---|---|---|
| Regional landscape axioms | Force amplitude, modulus, excess modulus, radial and pairwise restoring defects, curvature interval and reward growth | Analytic bounds on each declared region | `convergence_tightening_chapter04` and its retained-stage review |
| Regional discrepancy transfer | `rho = max_i sum_j w_j A[j][i] / w_i`; rate `-log(rho)/(m h)` | Complete conditional error-transfer bounds, including interfaces and copying | Rust closure/slow-zone regressions; empirical transition matrices are gated |
| Regional weight tightening | Positive left-vector search followed by directed column verification; unweighted prefactor `max(w)/min(w)` retained | Same complete regional inequalities; uniformly controlled weights for population uniformity | Independent characteristic-polynomial regression recovers contraction hidden by unweighted column sums |
| Nonquadratic native kinetics | Harmonic sector rate plus exact two-kick residual-force defect | Shared-noise isotropic BAOAB with fixed radial cap; both residual bounds | Native Rastrigin stage/force checks and independent matrix certificate |
| Within-well Rastrigin rate | With midpoint curvature, certified squared-error rates at least 0.920621 at half-width 0.01 and 0.538123 at half-width 0.03 per physical time | Both actual kick-query pairs remain in the same declared well box | Independent 60-decimal interval endpoint and nonlinear-modulus certificate |
| Existing root-core phase bound | Exact KUR.1 mean-map multiplier, accepted jitter variance, gate refresh, positional coefficient and noise floor | Declared root-core source geometry and actual jitter, collision, viscosity and diffusion parameters | Rust bridge to the existing Python landscape formulas, independent numerical fixtures and retained-state eligibility checks |
| Refined root-core bound | For the published d3, jitter-0.1 reference: multiplier 0.798942041 to 0.796643187, physical multiplier rate 5.611672 to 5.683709, one-step floor 0.206133580 to 0.201967666 | Exact accepted Gaussian variance and 32 analytic root-enclosure refinements; complete KUR source premises | 20 directed interval fixtures and 40 independent full Gaussian quadratures; both viscous normalizations |
| Wider Rastrigin regions | The local contraction test fails at half-widths 0.05 and 0.15 | Their slow-zone/interface contributions need a complete regional certificate | Failed closure retained, never presented as a positive rate |
| Gaussian region communication | Landing lower bound, escape upper bound and weighted chi-square tails | Certified actual mean intervals and covariance spectral bounds | Correlated-noise and dimension regressions; actual nonquadratic force centers |
| Unbounded polynomial reward tails | Tail mass `M_p/R^p`; cost tail `M_p/R^(p-r)`; arbitrary-coupling tail remainder | A proved population-normalized moment with `p>r` | Moment/Holder and cutoff regressions; sampled moments remain diagnostics |
| Complete weighted mixing register | `q_w = 1-epsilon_2/2+2 B_w L_rem+L_rem^2` | Actual kernel/profile/component hypotheses and `q_w<1` | Independent primitive reconstruction and feedback/underflow gate tests |
| Complete bounded-reward register | `q_2 = 1-epsilon_2+2 L_R+L_R^2` | Bounded configured reward and all source hypotheses | Exact profile and component gates; dimension-specific log minorization |
| Dimension-tightened box barrier | `d C(A) min(1,A)^(d-1)` | Final independent positional Gaussian and integrable coordinate barrier | Fresh d4/d8 full native archives and retained d1/d2 datasets |

The within-well rate is a conditional nonlinear certificate. The global
bounded-force perturbation gives decay to an explicit remainder. A complete
stationary-law rate requires closure of the corresponding regional or weighted
mixing inequalities. At the usual Rastrigin timestep 0.04, the specific global
finite-force-center hypothesis of Chapter 6a Sections 15–16 fails; regional
bounds cannot be substituted for that supremum. The implementation exposes this
failed hypothesis while retaining the valid regional estimates.

The tiny-perturbation active-cloning stress case is retained separately: its
within-pair coupling error increases from 0.000008 to 0.043682 while every
conditional refined kinetic bound passes. This separates preparation feedback
and coupling floors from a kinetic rate. It does not identify the distance
between the full laws. The complete experiment matrix measures finite-separation
well, slow-zone and tail starts, with exact empirical whole-law endpoint
transport evaluated separately from the paired coupling.

## Completed native validation

The new datasets contain **177,408 native updates, 224,436 native comparisons
with zero failures, and 24,165 lossless artifacts** (3,682,246,111 compressed
bytes). Every artifact was checksum verified and deeply decoded. These counts
exclude analyses that reuse stored trajectories. The separate Chapter 4 review
passes 2,002 comparisons without advancing an engine.

| Completed experiment | Native updates | Comparisons | Result |
|---|---:|---:|---|
| Nonquadratic Rastrigin kinetics, d=1/2/4, N=4/64, well/saddle/tail starts | 73,728 | 110,592 | All pass |
| Same landscape with positive fitness, cloning, jitter and Haar collisions | 73,728 | 110,592 | All pass |
| Dimension/curvature sector probes, d=1/2/4/8 | 20,480 | 380 | All pass |
| Fresh d=4/8 barrier and law probes | 8,192 | 2,672 | All pass |
| Actual root-core parameters, d=1/2/3/4/8, N=4/64, jitter=0/0.1, count/row viscosity | 1,280 | 200 | All pass |

The complete local kinetic audit covers all 147,456 Rastrigin native updates.
There are **22,598** eligible walker-step checks at well half-width 0.01 and
**307,262** at half-width 0.03, with no violation of the respective certified
multipliers. Another **319** complete swarm steps satisfy the half-width 0.03
residence premises and pass. Eligibility checks use both actual force queries,
live masks, final states, and shared innovation identities. These pathwise
checks do not condition an expectation on Gaussian residence or replace an
interface/escape bound. Their 4,608 derived ledgers are separately verified.

The endpoint review minimizes the population-normalized phase-space cost over
both particle permutations and the 32-replica empirical law coupling. The
initial cost is `0.08*d`, independent of N. The following are final/initial
squared-cost ratios after 64 steps (physical time 2.56):

| Starting region | Isolated nonlinear kinetics | Actual selected/cloning gas |
|---|---:|---:|
| Well | 1.58e-12 to 1.38e-8 | 0.108410 to 0.919677 |
| Slow saddle zone | 24.7447 to 24.7457 | 16.7914 to 25.1011 |
| Tail | 1.86e-12 to 2.70e-9 | 0.116154 to 0.964527 |

All 12 well/tail cases in each dataset decrease at the whole-law endpoint;
all six saddle cases grow. These finite empirical-law minima describe the
stored ensemble. They do not certify a population-law convergence rate. The
full-time paired-coupling plots retain their separate meaning and replica
uncertainty; active cloning has a larger coupling floor at N=4.

The 40 fresh root-core groups use all entering frozen velocities equal to zero,
including inputs to any revived recipient's collision. This permits the proved
state-dependent velocity remainder without changing the configured cap. The
report retains both configured-cap and state-dependent floors, full Gaussian
position/jitter tails, exact conditional moments, accepted clone plans, measured
residuals and replica standard errors. Formula validation includes 20 independent
fixtures, 40 full Gaussian quadratures, and directed interval certificates.
Physical-input guards reject unsupported restitution and invalid primitive
parameters; the original and tightened formulas remain separately available.

The complete regional rate is computed from declared discrepancy-transfer
inequalities, including slow-zone amplification, passage into contracting
cores, preparation feedback, and exterior error. Its weighted multiplier is
`rho`; its physical rate is `-log(rho)/(m*h)` and its remainder is
`B/(1-rho)`. The unweighted conversion retains `max(w)/min(w)` and
`B/[min(w)*(1-rho)]`. Uniformity in N requires uniform transfer constants,
defects and weight ratios. The implemented positive-weight optimization can
tighten this bound, but sampled occupation frequencies alone do not establish
those inequalities. A complete global certificate for the standard Rastrigin
configuration remains an analytic obligation.

## Proofs and reusable APIs

- [Regional rate, tail remainder and full nonlinear kinetic proofs](../../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md#sec-slc-regional-error-rates)
- [Chapter 4 regional profiles and sharper reward/moment bounds](chapter04-tightening.md)
- [Chapter 6 Gaussian and arbitrary-coupling tail APIs](chapter06-structural-tails.md)
- [Regional composition and force-rate Rust API](../crates/benchmarks/src/convergence_structural_rates.rs)
- [Complete primitive mixing Rust API](../crates/benchmarks/src/convergence_structural_mixing.rs)
- [Gaussian and polynomial-tail Rust API](../crates/benchmarks/src/convergence_structural_tails.rs)
- [Independent directed Rastrigin interval certificate](../outputs/convergence/structural-landscape-tightening-20261004/independent-regional-interval-reviewed.json)
- [Existing landscape parameterization and refined constants](../crates/benchmarks/src/convergence_landscape_phase.rs)
- [Directed root-core rate and noise-floor certificates](../outputs/convergence/structural-landscape-tightening-20261004/independent-existing-phase-interval-reviewed.json)
- [Native dataset catalog and verification links](../outputs/convergence/structural-landscape-tightening-20261004/dataset-catalog.json)
- [Detailed Chapter 5 constants and all 40 regional experiment groups](../outputs/convergence/dimension-tightening-20261004/chapter05-tightened-results.md)
- [Complete local nonlinear kinetic audit](../outputs/convergence/structural-landscape-tightening-20261004/local-kinetic-audit/complete-20261004/report.json)
- [Exact empirical-law endpoints with active cloning](../outputs/convergence/structural-landscape-tightening-20261004/exact-law-selected/endpoint-table.md)
- [Time-resolved active-cloning coupling plots](../outputs/convergence/structural-landscape-tightening-20261004/paired-error-selected-final/results.md)
- [Previous complete Chapters 4–6 experiment catalog](chapters04-06-full-results.md)
