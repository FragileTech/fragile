# Convergence estimates and native simulation completion

This catalog links the individual source-expression comparisons, the native
experiments and the statement ledgers for convergence Chapters 1–6. Numerical
agreement concerns each recorded estimate under its own hypotheses. A formula
test, a conditional Gaussian estimate, a finite empirical-law distance and a
stationary-law theorem have separate evidence.

Every physical swarm error uses probability normalization. Transport minimizes
over particle permutations; stored array rows only address simulation data.
Alive empirical laws omit dead coordinates and use their own live cardinality.
Forced revival is evaluated at the copied live companion and its complete
collision/jitter preparation, rather than treating a dead coordinate as a live
convergence error.

The final strict Chapters 1–3 run passes **all 3,146 required quantitative
expressions**, with **1,955,776 comparisons**, **261,585 hypothesis comparisons**
and **zero failures or phase errors**. It retains 86,296 source-bound evidence
records. [Compact result and checksum](../outputs/convergence/estimate-completion-20261004/chapters01-03-final-summary.json),
[full numerical report](../outputs/convergence/estimate-completion-20261004/chapters01-03-final.json)
and [executed source snapshots](../outputs/convergence/estimate-completion-20261004/chapters01-03-final-provenance.json)
are saved separately.

The [final machine-readable completion index](../outputs/convergence/estimate-completion-20261004/completion-index-v2.json)
records the current artifact checksums, chapter totals and verified archive
catalogs. Focused Rust regressions, Clippy with warnings denied and the final
Python Ruff checks pass.

The detailed tables contain every required expression, its source formula,
observed/bound ranges, exact evidence indices and recorded scope:
[Chapter 1: 1,365 estimates](../outputs/convergence/estimate-completion-20261004/chapters01-03-tables/chapter01-required-estimates.md),
[Chapter 2: 403 estimates](../outputs/convergence/estimate-completion-20261004/chapters01-03-tables/chapter02-required-estimates.md),
[Chapter 3: 1,378 estimates](../outputs/convergence/estimate-completion-20261004/chapters01-03-tables/chapter03-required-estimates.md).

## Chapter evidence

| Chapter | Estimates and constants evaluated | Detailed evidence |
|---|---|---|
| 1: Fragile Gas Framework | Companion-support decompositions; native raw-distance continuity and variance; centering, normalization and regularization constants; squash/map derivatives; cloning gate probabilities; branch expectations; Gaussian and Hermite estimates; population-normalized finite-support bounds | Final strict formula report and [complete source inventory](chapter01_inventory.json). The remaining-expression manifest contains 328 individually bound expressions, including support changes, singleton and empty branches. |
| 2: Euclidean Gas | Force/reward/noise hypotheses; kinetic matrices and Gaussian moments; displacement, Lipschitz and curvature estimates; dimension-dependent landscape constants; the global nonquadratic gradient certificate and its tighter coordinate-variance certificate | Final strict formula report and [complete source inventory](chapter02_inventory.json); two native gradient datasets described below. |
| 3: Cloning | Actual donor laws and acceptance; replacement/jitter/collision conditional moments; signed drift and variance; Keystone constants and finite-population corrections; centered empirical errors and source-pressure bounds | Final strict formula report and [Chapter 3 evidence](../outputs/convergence/chapter03-complete.json). Native proposal experiments retain 102 cases, 208,896 proposals and 1,062 comparisons. |
| 4: Wasserstein contraction | Optimal barycenter/centered transport; alive normalization; complete pressure integration; regional geometry; Gaussian provider Jacobians; tagged survival; finite-QSD density, eigenfunction, minorization and rate constants | [Statement table with measured operands](chapter04-statement-results.md) and [completion scope](chapter04-estimate-completion.md): 35 statements, 447,172 numerical comparisons, zero failures. Twenty-eight statements have numerical subestimate evidence; seven retain their analytic law/closure obligations. |
| 5: Kinetic contraction | Native transient position moments, force families, variance/barycenter identities, coupled kinetic matrices, cap sectors, Gaussian moment/density constants and full dimension dependence | [Chapter 5 table and results](../outputs/convergence/chapter05-completion-20261004/results.md): 2,760 source-scoped comparisons, zero failures, over 11,520 retained native left stages. |
| 6: Convergence | Drift iteration, actual native BAOAB raw/centered displacement budgets, moment/tail transfer, dimension barriers, survival hazards, conditional final-Gaussian LSI, sensitivity/optimization algebra, complete selected-source moment closure and sharper root-moment bounds | [Final Chapter 6 estimate table](../outputs/convergence/chapter06-completion-20261004/estimates-table-v5/chapter06-estimates-table.md) and [current statement ledger](../outputs/convergence/chapter06-completion-20261004/statement-ledger-v2/statement-ledger.md). The current source inventories 54 statements and 510 expressions; each entry distinguishes the actual law and the needed hypotheses. |

The Chapter 1 supplement exhausts the declared finite companion-support
fixtures and uses exact conditional probability sums where available. It does
not convert finite enumeration into a proof of every possible landscape axiom.
The initially saved `chapter01-current-v1.json` exposed an evaluator error:
the identical-law lemma used two status-masked observables rather than the same
fixed observable. The corrected evaluator retains both observables and a
regression that rejects the incorrect substitution. The failed report is kept.

## New native experiments and saved data

| Dataset | Configurations | New native engine updates | Comparisons | Result |
|---|---:|---:|---:|---|
| [Seven parameter profiles](../outputs/convergence/estimate-completion-20261004/parameters/results.md) | 126 cases; d=1,2,4; N=4,64; well, saddle and tail starts; 16 independent paired seeds | 64,512 | 96,768 | Zero bound violations |
| Exact empirical-law endpoints for those profiles | Inner particle assignment followed by outer 16-replica assignment | 0 | 504 | Zero comparison failures |
| [Global nonquadratic gradient certificate](../outputs/convergence/estimate-completion-20261004/nonquadratic-gradient-axiom/report.json) | 144 segments; d=1,2,4,8; varying lengths, centers and directions | 0 | 576 | Zero violations; 1,179,792 actual native gradient evaluations |
| [Tighter short-segment certificate](../outputs/convergence/estimate-completion-20261004/nonquadratic-short-gradient-axiom/report.json) | 144 new segments starting at length sqrt(d), including well, saddle and exterior centers | 0 | 576 | Zero violations; 1,179,792 additional actual native gradient evaluations |
| [Exact native selected-source integration](../outputs/convergence/chapter06-completion-20261004/weak-exact-source-integration/report.json) | Actual saved weak-selection fitness, every nonself donor and its native gate/kernel | 0 | 331,776 | Zero violations; 9,313,146 positive acceptance contributions |
| [Sharper selected-source moment closure](../outputs/convergence/chapter06-completion-20261004/weak-native-moment-tightened/report.json) | Complete p=4,8 bounds in well, saddle and tail cases | 0 | 4,608 | Zero violations over 9,216 previously counted native updates |
| [Exact native displacement budgets](../outputs/convergence/chapter06-completion-20261004/native-displacement-v2/report.json) | Raw/centered conditional BAOAB increments, positional variance and separately source-bound terminal velocity caps | 0 | 80,840 | Zero discrepancies over 11,520 prepared stages and 200 independent seed groups |
| [Original killed-chain moment reanalysis](../outputs/convergence/chapter06-completion-20261004/original-killed-moments/report.json) | 396,339 retained updates and 792,678 conditional p4/p8 rows; each law uses its own seed denominator | 0 | 3,194 | Zero discrepancies; complete original archives retained |
| [Native barrier, Gaussian and tail subestimates](../outputs/convergence/chapter06-completion-20261004/native-tail-subestimates-v1/report.json) | Exact source-scoped finite-row and conditional Gaussian estimates | 0 | 6,760 | Zero discrepancies; law-specific hypotheses retained |

The parameter profiles vary timestep, friction, cap, positional diffusion,
jitter, restitution, reward/diversity exponents and Gaussian donor bandwidth.
All native prepared stages, innovations, force queries, configurations,
checkpoints and raw outputs are saved. The seven profiles contain 8,204 lossless
archives, 1,361,886,211 compressed bytes; every archive was checksum verified
and deeply decoded. Each gradient dataset has 146 verified lossless archives.
Derived reviews reuse trajectories and never add to the engine-update count.

The [independent short-gradient review](../outputs/convergence/estimate-completion-20261004/short-native-independent-review/proof-and-results.md)
passes 3,537 comparisons against the actual native arrays, independently
reconstructs their endpoint integrals and verifies every archive checksum.

The earlier completed [Chapters 4–6 experiment matrix](chapters04-06-full-results.md)
contains 832,435 native updates. The separate [landscape matrix](landscape-tightening.md)
contains 177,408 updates. Their original provenance and archives are retained.

The final [Chapter 6 verified catalog](../outputs/convergence/chapter06-completion-20261004/verified-catalog-v4/catalog.json)
contains 12 current evidence reports and 305 deeply verified derived artifacts,
138,323,687 compressed bytes. Its retained/native subestimate reports pass
437,382 comparisons with zero failures. Superseded displacement and weak-moment
reports are preserved and excluded from this current total.

## Tighter constants and rates

| Estimate | Result | Conditions |
|---|---|---|
| Global Rastrigin segment nondeception | L_grad=sqrt(d), kappa_grad>=1207; real certificate 1207.362552744874… | Complete coordinate-variance proof on the unbounded landscape; independent of N and d in the lower constant |
| Within-well squared kinetic error | Physical rate at least 0.9206218693 for half-width .01 and 0.5381239957 for .03 | Both actual force queries belong to the same declared well box; cap-sector proof and directed interval calculation |
| Root-core variance tightening | Reference multiplier .798942041 to .796643187; floor .206133580 to .201967666 | Actual root-core source premises, full Gaussian jitter, viscosity and positional noise retained |
| Complete weak-selection p4 moment | Young multiplier .997229264093; root-moment multiplier .998521285199 | Positive exponents 1e-5, Gaussian bandwidth 3, h=.04, gamma=1, cap=2, jitter=.1, positional diffusion=.1, restitution=.5, viscosity=0 |
| Complete weak-selection p8 moment | Young multiplier .994117846919; root-moment multiplier .998476325811 | Same native profile; coefficients independent of N |
| d=1 p4 asymptotic moment floor | Original 5.5320655e9; refined energy/jitter 4.8854469e8; root bound 3.2556245e8 | Worst-case analytic bounds, rather than fitted stationary moments |
| d=1 p8 asymptotic moment floor | Original 3.5383016e20; refined 1.0303735e19; root bound 9.9385384e17 | Dimension-dependent complete force/noise constants and actual saturation amplitude retained |
| Unbounded reward tails | Mass M_p/R^p and degree-r cost M_p/R^(p-r) | Proved normalized moment with p>r; no compact-support replacement |
| Complete regional rate | rho=max_i(sum_j w_j A[j,i]/w_i); physical rate -log(rho)/(m*h) | All source, slow-zone and interface transfer inequalities and rho<1; unweighted prefactor max(w)/min(w) retained |

For the root bound, u=(E M_p)^(1/p) satisfies u_next<=r*u+C.
Consequently E M_p(n)<=(r^n*u0+C*(1-r^n)/(1-r))^p. With a positive floor,
r^p is not a claimed linear decay coefficient for the unrooted moment excess.
The [independent sharp-tail review](../outputs/convergence/estimate-completion-20261004/sharp-tail-independent-review/proof-and-results.md)
reconstructs the native coefficients and 18,432 retained root-bound predictions
at 80-digit precision. All 19,080 comparisons pass; separate regressions check
accepted Gaussian jitter, normalized collision energy and rejection of the
incorrect unrooted excess rate.
The complete proofs are in [Chapter 2](../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md)
and [Chapter 6a Section 20](../../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md#sec-slc-global-selected-moments).

## Error decrease and scope

The 64-step selected Rastrigin experiment shows decreasing exact finite
empirical-law error in all 12 well/tail cases: final/initial squared-cost ratios
are .108410–.919677 in wells and .116154–.964527 in tails. All six saddle starts
grow, as expected from the local expansive region. The isolated kinetic
experiment decreases in the same 12 regions by much larger factors.

The new 16-step weak-selection profile decreases in all 12 well/tail cases:
ratios .004700–.016772 in wells and .009293–.053472 in tails. Every parameter
profile retains its full time series, uncertainty, endpoint assignments and
growth cases in the [parameter comparison catalog](../outputs/convergence/estimate-completion-20261004/parameters/comparison-catalog.json).
Strong-feedback profiles can have a transient error increase while their
complete bound with an explicit remainder still holds.

The original killed-chain ensembles also show decreasing completed moments
at time 128 for N=4,16,64 and d=1,2: final/initial p4 ratios are
.00372983–.00794452 and p8 ratios are .0000718307–.000431766. These are
unconditioned normalized moments with each law's own seed denominator.
Absorbed N=2 ensembles are shown separately; their zero output is not survivor
mixing or a distance between laws.

The standard positive-fitness Rastrigin configuration does not yet have a
proved population-uniform global law-mixing rate. Source-pressure, provider
feedback and slow-zone flux must close for that claim. The finite-QSD primitive
bound is positive only on logarithmic scales and is too pessimistic to predict
visible relaxation over 128 steps. Conditional Gaussian concentration does not
identify a joint QSD/invariant-law LSI. These obligations stay attached to the
corresponding statement entries; successful numerical component checks do not
remove them.
