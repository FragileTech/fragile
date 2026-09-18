# Fixed-step mean-field validation

All trajectories use the Rust Euclidean Gas engine at h=0.04. Each population-size cohort contains 128 independent seeds and observations after 1, 4, 16 and 64 updates. Seeds reused across population sizes are paired; no aggregate standard error treats those cohorts as independent.

## Exact conditional balances

Entries are cumulative martingale residuals divided by the square root of their summed predictable variance. Each number is calculated within one independently seeded population-size cohort.

| Study | N | Clone gates | Shared-Haar stress | Final Gaussian position | Revivals |
|---|---:|---:|---:|---:|---:|
| canonical (seed 0) | 16 | 0.832 | -1.213 | 1.615 | 0 |
| canonical (seed 0) | 32 | 0.321 | 2.780 | 0.272 | 0 |
| canonical (seed 0) | 64 | 1.582 | 0.707 | -0.616 | 0 |
| canonical (seed 0) | 128 | -1.254 | 0.285 | 0.625 | 0 |
| canonical (seed 0) | 256 | 1.459 | -0.270 | 0.468 | 0 |
| canonical (seed 30000) | 16 | -0.077 | 0.703 | -0.267 | 0 |
| canonical (seed 30000) | 64 | -1.108 | 1.377 | 0.659 | 0 |
| canonical (seed 30000) | 256 | -0.178 | 0.510 | -0.165 | 0 |
| boundary_stress (seed 10000) | 16 | -0.706 | -0.840 | -0.517 | 11175 |
| boundary_stress (seed 10000) | 64 | -1.057 | -0.537 | 0.129 | 44586 |
| boundary_stress (seed 10000) | 256 | -1.400 | 0.216 | 0.751 | 177878 |
| shared_initial (seed 20000) | 16 | -1.396 | 0.440 | -0.700 | 0 |
| shared_initial (seed 20000) | 64 | 0.337 | 1.995 | -1.046 | 0 |
| shared_initial (seed 20000) | 256 | 0.458 | 0.804 | -2.216 | 0 |
| shifted_initial (seed 40000) | 16 | -0.418 | -0.297 | 1.349 | 0 |
| shifted_initial (seed 40000) | 64 | 0.759 | -0.208 | -0.381 | 0 |
| shifted_initial (seed 40000) | 256 | 1.348 | 0.923 | 0.385 | 0 |

## Population-size comparisons

The table reports the actual full-slot mean of cos(x₀) after 64 updates. Standard errors come from variation across independent runs. Decreasing variance is evidence about these observables at these horizons. The conditional one-step variance theorem gives an N⁻¹ bound at a frozen input; obtaining a uniform-in-time bound additionally requires the collective contraction estimate.

| Study | N | Mean | Standard error | Across-run variance | N × variance |
|---|---:|---:|---:|---:|---:|
| canonical (seed 0) | 16 | 0.98065025 | 0.00067757 | 5.8764236e-05 | 0.00094022778 |
| canonical (seed 0) | 32 | 0.98104738 | 0.00054933 | 3.8625299e-05 | 0.0012360096 |
| canonical (seed 0) | 64 | 0.97955469 | 0.00048503 | 3.0112573e-05 | 0.0019272047 |
| canonical (seed 0) | 128 | 0.97969001 | 0.00030993 | 1.2295591e-05 | 0.0015738356 |
| canonical (seed 0) | 256 | 0.98004745 | 0.00022684 | 6.5862358e-06 | 0.0016860764 |
| canonical (seed 30000) | 16 | 0.98160011 | 0.00081937 | 8.5934622e-05 | 0.001374954 |
| canonical (seed 30000) | 64 | 0.97968453 | 0.00038071 | 1.8552074e-05 | 0.0011873327 |
| canonical (seed 30000) | 256 | 0.97983837 | 0.00020017 | 5.1287952e-06 | 0.0013129716 |
| boundary_stress (seed 10000) | 16 | 0.95552013 | 0.00141480 | 0.00025621069 | 0.0040993711 |
| boundary_stress (seed 10000) | 64 | 0.95655403 | 0.00072742 | 6.773077e-05 | 0.0043347693 |
| boundary_stress (seed 10000) | 256 | 0.95666334 | 0.00032280 | 1.3337296e-05 | 0.0034143479 |
| shared_initial (seed 20000) | 16 | 0.97918919 | 0.00090079 | 0.00010386247 | 0.0016617996 |
| shared_initial (seed 20000) | 64 | 0.97950384 | 0.00040695 | 2.1198327e-05 | 0.0013566929 |
| shared_initial (seed 20000) | 256 | 0.97951879 | 0.00023170 | 6.8714109e-06 | 0.0017590812 |
| shifted_initial (seed 40000) | 16 | 0.98115310 | 0.00092225 | 0.00010886943 | 0.0017419109 |
| shifted_initial (seed 40000) | 64 | 0.98019811 | 0.00044845 | 2.5742193e-05 | 0.0016475004 |
| shifted_initial (seed 40000) | 256 | 0.98019814 | 0.00019245 | 4.7409367e-06 | 0.0012136798 |

## Shared-initial-law control

The x₀ coordinate initially shares one random draw across every row. This is an exchangeable mixture rather than deterministic initial chaos. The empirical sin(x₀) variance should remain macroscopic at update zero; the following table verifies that the diagnostics retain it.

| N | Initial variance of empirical sin(x₀) | Variance after 64 updates |
|---|---:|---:|
| 16 | 0.296818 | 0.00437802 |
| 64 | 0.296818 | 0.00108783 |
| 256 | 0.296818 | 0.000261577 |

## Independent operator integration

The four-slot canonical configuration x=(0,0,0,0.1), v=0 has initial positional variance 0.001875. Independent enumeration of all eight distinct measurement-distance patterns, donor/gate integration and Gaussian jitter predicts post-clone variance 0.00297011744615149. Across 4,096 disjoint actual Rust seeds the measured mean is 0.00293403846878196 with standard error 0.00004049322189395: a residual of −0.891 standard errors. Thus the asserted universal negative positional cloning drift is false, and the corrected positive drift agrees with the engine.

Separate tests check the full BAOAB joint covariance, including its h=2 quadratic-force velocity resonance, frozen-source copying, 120 five-slot strict-fitness forests, transported permutation innovations, and shared-rotation momentum/energy identities in dimensions 1–3. Interactive Part III experiments also integrate an independent rooted population law from the entering atomic measure, retaining sampled fitness and incoming cloners. Its Monte Carlo error is reported separately from the finite-population trajectory.

## Keystone cluster and kinetic verification

The machine-readable record is `validation/keystone-repair.json`; its two
Rust test files contain the exact configurations, initial states and seed
addresses. The measured positive centered error of an unchanged walker is
consistent with a negative collective drift. For the canonical three-slot
cloud `(-0.5,0,0.5)`, conditioned on both endpoints measuring the center:

| Quantity | Analytic prediction | Seeds 120000–124095, mean ± SE | Seeds 130000–134095, mean ± SE |
|---|---:|---:|---:|
| Center's squared centered position | 0.012118593 | 0.011824722 ± 0.000485642 | 0.012702111 ± 0.000491833 |
| Collective positional variance drift | −0.058740013 | −0.055254539 ± 0.001837922 | −0.060385315 ± 0.001850193 |

Each batch executes 4,096 full Rust steps, with 1,082 and 1,091 occurrences
of the specified measurement event. Fitness is calculated independently from
the sampled companion and compared with the retained engine fitness on every
run. The analytic predictions integrate donor/gate choices and Gaussian jitter.

Another 640 full Rust steps use exactly one alive donor at N=16,32,64,128,256,
with 128 independent seeds per cohort. All other slots revive and the entire
population participates in one collision component. Every measured barycenter
variance lies within 1.93 Gaussian variance standard errors of the exact
`j²(N−1)/N²` prediction. The maximum error in full-slot velocity-mean
conservation is `1.91e−17`. This test includes retained dead positions outside
the terminal box and retained dead velocities; alive-only input momentum is
not substituted for full-slot momentum.

Sixty-three exact calls to the production component collision operator cover
dimensions 1–3, restitution 0, 0.5 and 1, and population sizes through 1,024.
They include 1,023-to-1 component loads. Reassigning a revived recipient while
coupling rotations by the unchanged alive backbones verifies the exact
component-average cancellation and the bound `(6+8 alpha)V`, with no factor
depending on component size. These operator checks supply deterministic
orthogonal matrices and make no claim about Haar sampling statistics.

The kinetic test uses actual Rust BAOAB execution for quadratic and Rastrigin
forces, dimensions 1–3, and OU amplitudes B=16,64,256. The 18 configurations
provide 36,864 paired row comparisons, including 705 coupling failures and
1,304 matched terminal exits. Matched positions, terminal marks and the
nonlinear final-force cancellation agree with the exact coupling identities.
All measured mean costs lie below their analytic bounds. For Rastrigin in
dimension three:

| B | Mean marked cost ± independent-run SE | Proved bound |
|---|---:|---:|
| 16 | 0.143320 ± 0.004435 | 0.463533 |
| 64 | 0.049318 ± 0.002478 | 0.115883 |
| 256 | 0.013519 ± 0.001399 | 0.028971 |

The independent script `tools/check_keystone_flux.py` checks the two signed
cluster-flow inequalities on 10,000 finite weighted configurations, eight
deterministic cases and singleton persistence. It retains actual row
normalizers and clipped acceptance. Maximum floating-point lower-bound
violation is `1.11e−16`; maximum normalization error is `2.22e−16`. Its report
is `validation/keystone-flux-identities.json`. These are algebraic checks,
separate from both engine trajectories and the analytic proofs.

All five new Rust tests and strict targeted Clippy checks pass. The theory
documentation builds successfully with the eight existing unresolved
downstream references listed by the broader build. Volume 2 proof downloads
were refreshed from the publication sources. The stationary concentration
theorem is not established by these finite-step validations.

## Interpretation

### Complete measurement averaging

`validation/keystone-measurement-average.json` records 13,568 additional full
Rust engine updates. The source is
`crates/benchmarks/tests/keystone_measurement_average.rs`. Every trajectory
uses the complete canonical quadratic configuration, with a recorded cloning
stage used to measure the contribution being compared.

For three slots at `(-a,0,a)`, independent integration sums all measurement,
donor and gate outcomes and integrates Gaussian jitter through its second and
fourth moments. It also verifies the decomposition of total variance into
conditional variance and variance of the conditional mean. No measurement
event is selected or discarded.

| a | Exact averaged variance drift | First batch mean | Second batch mean | Exact SE per batch |
|---|---:|---:|---:|---:|
| 0.1 | 0.000620054 | 0.000651036 | 0.000592976 | 0.000077129 |
| 0.5 | −0.050087501 | −0.051380478 | −0.052431336 | 0.001603112 |
| 1.9 | −1.295651128 | −1.284566930 | −1.299731612 | 0.018790232 |

Each batch contains 2,048 independent runs. The six seed ranges are disjoint
and recorded in the JSON. All six deviations are below 1.47 exact standard
errors. The small-radius expansion and larger-radius contraction both match
the same complete measurement-averaged calculation.

A second family places half the population at each of `-0.5` and `+0.5`.
Independent binomial integration of measurement counts preserves those two
geometric clusters and yields the exact conditional row-moment average.
There are 128 runs in each of two disjoint batches for every population size:

| N | Exact averaged variance drift | First batch mean ± SE | Second batch mean ± SE |
|---|---:|---:|---:|
| 16 | −0.007425945 | −0.006051685 ± 0.001646503 | −0.006567542 ± 0.001461683 |
| 32 | −0.002362432 | −0.001992388 ± 0.000968896 | −0.004434673 ± 0.001022799 |
| 64 | 0.000093525 | 0.000063747 ± 0.000678613 | −0.000668191 ± 0.000599511 |
| 128 | 0.001301557 | 0.002002064 ± 0.000490950 | 0.001880303 ± 0.000441020 |
| 256 | 0.001900490 | 0.001696181 ± 0.000301404 | 0.002118555 ± 0.000308009 |

All deviations are below 2.03 independent-run standard errors. None of these
1,280 runs has an all-equal retained fitness vector. Thus positive cloning-stage
internal-variance drift is possible with active selection. The analytic lower
bounds at N=128 and N=256 are positive. This does not contradict an affine
drift bound with a noise offset, or establish growth of coupled structural
error: the latter has its own cross-swarm covariance term. The tests verify
these distinct quantities without identifying one with another.

The structural-pressure integration separately verifies the proved coefficient
`chi = 0.30251915319478023`, with zero offset, for balanced paired clouds at
N=4,10,16,32,64,128,256 and radii 0.5,0.75,1,1.9. This uses the actual
optimal paired structural discrepancy `(a-b)^2`, not either cloud's internal
variance. The analytic theorem covers every even N>=4 and radii in `[0.5,2)`.

All four tests in the new target and strict targeted Clippy pass. The theory
documentation build passes; its broader pass reports ten occurrences of
eight existing unresolved downstream labels. The final incremental build
after the notation edits passes without additional warnings. Volume 2 proof
exports contain the new formal statements and complete proofs. No production
engine or browser implementation changed in this batch.

### Proof-repair checks

The geometric error clusters remain the organizing strategy for collective contraction. The collision-component moment bounds are auxiliary estimates with constants independent of N. They control innovation replacement and one-step fluctuations; they do not replace the signed cluster flux estimate needed for stationary closure.

The conditional experiment in `crates/benchmarks/examples/mean_field_conditional.rs` uses the actual Rust engine with N=16,32,64,128,256 and frozen inputs entering updates 1,4,16,64. It covers the canonical configuration and a boundary-stress configuration. Each input has 128 independent one-step executions in each of two disjoint seed batches, 92000–92127 and 94000–94127. The reports `validation/mean-field-conditional.json` and `validation/mean-field-conditional-heldout.json` contain configurations, source states, seeds, means, variances, standard errors, component distributions, revival and extinction counts. Together they record 10,240 probes and 107,520 revivals, with no extinction. Seeds are reused across input cohorts, which are not pooled as independent observations.

At update 64, the held-out conditional variance of the empirical cos(x₀) observable decreases from 1.7895×10⁻⁵ at N=16 to 2.805×10⁻⁶ at N=256 in the canonical configuration, and from 6.6531×10⁻⁵ to 1.4388×10⁻⁵ in the boundary-stress configuration. Intermediate values need not be monotone: the theorem bounds variance and the frozen inputs differ across population sizes. These measurements do not estimate stationary concentration or an exact asymptotic variance constant.

Independent verification includes:

- Fourteen finite-kernel entropy and survival identity checks, with maximum algebraic residual 4.45×10⁻¹⁶ (`validation/mean-field-proof-identities.json`). These check identities, not gas trajectories.
- Three Rust tests of the exact BAOAB matrix identity, synchronous capped-kinetic coupling in dimensions 1–3, and the full-engine kinetic memory observable with revival (`crates/benchmarks/tests/kinetic_proof_repair.rs`).
- Strict workspace Rust linting and five compiled-engine test-file checks, all passing.
- Eight Part III browser workbenches passing rendering, reset/replay and rooted-reference checks.
- A successful documentation build. The broader rebuild reported eight unresolved references in downstream documents, which are outside this repair's editing scope.
- The workspace test log records 337 passing tests in 49 completed suites and no test failures. The command session returned status 143, so the recorded successes are not reported as a clean full-workspace command exit.

Python compilation checks pass. Ruff could not run because it is absent from the environment and the package index was unreachable. Production engine code was not changed in this proof-repair batch; compiled checks used the existing CPU/WebGPU assets.

### Scope of the conclusions

Finite-horizon population consistency and finite-N QSD uniqueness are proved for the same canonical transition. The latter uses the actual full-kernel Gaussian smoothing and verified minorization, without a positional cloning-contraction premise. Stationary uniqueness of the nonlinear population map is a separate result; these fixed-horizon runs and the finite-N theorem do not establish it. Shifted-initial-law comparisons report actual relaxation rather than label 64-update outputs as exact stationary samples.


## Complete coverage and certified structural expansion

Chapter 03 equations (3.CC1)--(3.CC14) close the uncovered-cluster calculation
with the actual measurement law and an explicit N^-2 self-exclusion term.
Three independent mathematical reviews checked the fixed-cover argument,
normalization, positive finite-score increment and partial-alive interface.
No minimum fraction of valid clusters is assumed.

The signed claim has a concrete obstruction. The exact canonical four-walker
inputs A=(-1.5,-1.5,-1.5,1), B=.01*A have zero velocities and nonconstant
retained fitness in every measurement outcome. Their entering structural
error is 1.1485546875. The exact finite-kernel interval calculation proves
cloning-stage expected structural error above 1.27648593291590 under every
coupling. Continuing through the full canonical B=1 update, including the
cap and terminal survival normalization, gives a certified structural
increment greater than .09091348; the theorem uses the safe bound .08.

This certificate is deterministic arithmetic, not a statistical significance
test. `validation/certify_keystone_structural.py` evaluates all 81 individual
measurement vectors per input using directed rational intervals, exact
integer square-root bounds and exponential series with proved remainder.
`validation/keystone-structural-intervals.json` records the rational endpoints.
The identities retain sampled global fitness, both donor-flow directions,
component collisions, jitter, kinetic noise, and the actual survival law.

A separate Rust test integrates the same finite kernel by binomial mark
counts in the unchanged two geometric groups, then compares it with complete
Rust-engine runs. The source is
`crates/benchmarks/tests/keystone_collective_expansion.rs`. The independent
batches contain 128 runs each at N=4,16,32,64,128,256 and scales 1,.01:
3,072 updates in 24 cohorts, all with unequal retained fitness. Cloning
variance predictions and measured means are:

| N, scale | Exact post-cloning variance | First cohort mean ± SE | Second cohort mean ± SE |
|---|---:|---:|---:|
| 4, 1 | 1.319544640 | 1.306963735 ± .027318142 | 1.297008346 ± .029220461 |
| 4, .01 | .000357118 | .000308931 ± .000074385 | .000402221 ± .000124325 |
| 256, 1 | 1.487380807 | 1.484536372 ± .002731628 | 1.491500165 ± .002620339 |
| 256, .01 | .000383313 | .000391055 ± .000014792 | .000364865 ± .000013884 |

Every cloning comparison is within 1.84 independent-run standard errors.
The test also checks the exact complete BAOAB preboundary variance prediction;
terminal alive variances are separately read from actual engine eligibility.
The artifact `validation/keystone-collective-expansion.json` records all
cohorts, configurations, disjoint seed ranges, predictions and uncertainties.
These are one-update identities and counterexamples. The established
1,4,16,64-update conditional studies are unaffected and were not counted as
new runs. The second execution of the same seed batches after adding the
kinetic assertion is a reproducibility check, not additional independent
samples.

Two Rust tests and strict targeted Clippy pass. The directed certificate
reproduces exactly and Python compilation passes. Ruff is unavailable in the
current environment; no Python lint success is claimed. Production engine
code, browser experiments and CPU/WebGPU assets did not change in this batch.
The completion report keeps the original full strong-diffusion dynamical
closure open; neither the disproved cloning-only sign nor this B=1
counterexample settles that different statement.

The final complete BAOAB preboundary comparisons all lie within 2.21
independent-run standard errors. The documentation build passes with eight
existing downstream-reference warnings and no new warning in Chapter 03.
Proof exports were refreshed from the final source, preserving complete formal
proofs. The exported volume contains 2,472 formal blocks with proofs and
1,682 blocks without proofs, from 43 published documents.


## Relation of component-growth checks to the recovered proof

The original combined proof permits cloning expansion of inter-swarm distance
and includes finite drift offsets. The structural-growth certificates above
are therefore operator diagnostics, not counterexamples to that combined
claim or completion requirements for the TV/entropy route. No equilibrium law
was used in those measurements, so they do not measure movement away from
equilibrium. The proof repair uses their exact moment accounting where needed
while keeping the original combined functional and stage order.


### Verification of the combined-input correction

The four `keystone_cluster_flux` tests and the
`keystone_kinetic_coupling` test pass, as does strict targeted Clippy. The
quadratic/Rastrigin kinetic checks preserve the actual BAOAB/cap stages;
no production simulation code changed. The reviewed new moment and weighted
composition estimates are in Chapter 03 equations (3.W1)--(3.W3) and
(3.AC1)--(3.AC9).

The final documentation build passes with ten occurrences of the existing
downstream missing-reference warnings and no new warning in Chapter 03.
Proof exports contain 2,476 formal blocks with proofs and 1,685 without
proofs, from 43 published documents. Scoped whitespace and new-label checks
pass. These checks certify the local operator-input and composition repair,
not completion of every remaining analytic application in the volume.
