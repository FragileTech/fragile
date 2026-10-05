# Native Rust experiments for Chapters 4–6

The full experiment matrix is retained in
`outputs/convergence/chapters04-06-experiments/full-20261004/`.
The generated [summary](../outputs/convergence/chapters04-06-experiments/full-20261004/summary/summary.md)
contains final counts, observed endpoint rates, source-expression coverage and plots.
The [data catalog](../outputs/convergence/chapters04-06-experiments/full-20261004/dataset-catalog.json)
binds all four verified datasets and their derived results by SHA256.
They retain 832,435 native updates in 55,623 indexed artifacts (10.02 GiB
compressed). All 63,535 primary numerical comparisons pass. Independent audits
and reanalyses reuse those samples and are recorded separately.

Every distance uses an empirical measure or a probability coupling of unlabelled
states. Particle costs are normalized by N. Permutation and population-duplication
regressions protect this convention. Revival discards the overwritten dead
position; the killed-chain state retains dead velocities when the native
collision operator still reads them.

| Chapter | Experiment | Quantities compared with theory | Scope |
|---|---|---|---|
| 4 | Four landscapes, N=4,16,64, d=1,2; 48 paired cases | Donor floors, acceptance probabilities, optimal transport, centered error, clone-jitter variance, survivor fractions, reset envelopes and conditional contraction | Every comparison records its hypotheses; constant reward supplies a control without a restoring force |
| 4 | Two dense viscous references, N=200,d=3 | Count/row normalization, stage force identities, κ_F, β_F and admissible timesteps | Complete native B1/B2 stages are checked independently |
| 5 | Six capped unit-quadratic cases, 256 independent noise replicates, 128 steps | η, dissipative trace, determinant, δ, conservative decay rate and mean optimal Q error at each step | Actual fixed-cap operator; three fixed perturbation directions use stratified uncertainty |
| 5 | Shared-Brownian timestep refinement | Exact SDE covariance, weak moment bias, squared strong error, and a nonconvex cosine-force reference certificate | Uncapped extension with its explicitly stated generator |
| 5 | Exact Gaussian cubature, h=.04,.02,.01 | Analytic weak bias and squared strong error to roundoff, second-order ratios, population-independent C_weak | 96 weighted nodes integrate all quadratic observables exactly; zero sampling uncertainty |
| 5 | Paired near-boundary updates | Terminal-status disagreement under shared innovations | Includes every native stage producing the event |
| 6 | Independent selected-cloning killed-chain ensembles | Own survival probabilities, normalized whole-swarm and uniform-alive law transport, covariance, safe counts and alive barrier moments | Each surviving law has its own denominator; zero-survivor laws remain undefined |
| 6 | Small-box hazard experiment | Gaussian per-particle death probabilities, Poisson-binomial absorption, native k=0 versus externally stopped k<2 survival | Native and externally stopped processes remain separately identified |

Detailed tables and independent audits:

- [Chapter 4 constants and bounds](chapter04-full-results.md).
- [Chapter 5 complete experiment results](../outputs/convergence/chapters04-06-experiments/full-20261004/chapter05-full-results.md), with
  [native experiments and the explicit weak-error coefficient](chapter05-native-experiments.md).
- [Chapter 6 constants, bounds and exact empirical-law results](../outputs/convergence/chapters04-06-experiments/full-20261004/derived-chapter06/full-results-table.md), with
  [review instructions](../outputs/convergence/chapters04-06-experiments/full-20261004/derived-chapter06/README.md).
- [Exact timestep refinement plot](../outputs/convergence/chapters04-06-experiments/full-20261004/summary/native-exact-timestep-refinement.png).

The source repair uses a positive quadratic metric and an explicit macroforce
closure for the continuous-time location estimate. The dependent transport
statement carries those conditions. Discretization transfer requires an
observable error certificate for a generator-consistent family; the fixed radial
cap uses its own discrete estimate. The proof now contains the complete
unit-quadratic derivation of C_weak=12.299258189966439 rather than an unspecified
prefactor. Run-time source snapshots remain immutable, so later proof corrections
are distinguishable from the source used when an experiment was launched.

Raw state and stage arrays, proposals, evaluated forces, Gaussian innovations,
Brownian bridge inputs, independent-replicate designs, native configurations,
seeds, source snapshots and executable hashes are saved in lossless gzip CBOR or
JSON. SHA256 indices and append/fsync journals retain each completed chunk.
Compressed checkpoints reproduce the next native transition exactly. The
`gas-proof-experiments reanalyze` command has also recomputed 2,658,048 conditional
cloning checks from the Chapter 4 archives with zero failures and zero new engine
steps.

The expression ledger in the generated summary records the exact coverage.
These finite experiments do not establish the global QSD eigenvalue, a joint-law
LSI constant, global sensitivity, or population-uniform minorization. A positive
finite-N minorization certificate is not a population-uniform mixing rate.
Passing specialized numerical estimates must not be presented as complete
validation of every theorem in Chapters 4–6.
