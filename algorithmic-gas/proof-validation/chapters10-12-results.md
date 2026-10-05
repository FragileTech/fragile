# Chapters 10–12: constants, bounds and measured convergence

The Rust validation now covers Chapters 10–12. The inventories retain **61 formal
items and 369 individual mathematical expressions**. There are **208 finite
numerical bindings** and **161 separately recorded definitions, hypotheses or
limiting assertions**. Every finite binding cites its own source formula and
operands. An analytic hypothesis is not counted as a numerical pass.

The experiment root is
[`chapters10-12-completion-20261005`](../outputs/convergence/chapters10-12-completion-20261005).
Its [completion index](../outputs/convergence/chapters10-12-completion-20261005/completion-index.json)
records source hashes, current ledgers, checks and integrity evidence. Earlier
ledger versions remain available; the tables below identify the final versions.

| Chapter | Full expression table | Finite bindings | Other dispositions | Successful numerical comparisons |
|---|---|---:|---:|---:|
| 10: KL hypocoercivity | [181 expressions / 22 formal items](../outputs/convergence/chapters10-12-completion-20261005/chapter10/ledger-v4/table.md) | 75 | 106 | 2,484,344 |
| 11: Hellinger, transport and alive mass | [134 expressions / 28 formal items](../outputs/convergence/chapters10-12-completion-20261005/chapter11/ledger-v1/results.md) | 96 | 38 | 13,948,528 |
| 12: QSD exchangeability | [54 expressions / 11 formal items](../outputs/convergence/chapters10-12-completion-20261005/chapter12/final-ledger-v1/expressions-table.md) | 37 | 17 | 44,569 |

These counts combine the final native/reference operands and their supplemental
checks. They exclude superseded ledgers and repeated extraction of retained runs.
Comparisons are numerical assertions, not counts of independent experiments.

## Constants and estimates

| Chapter | Constant or bound | What is measured and compared | Evidence scope |
|---|---|---|---|
| 10 | Temperature `θ=D/γ=σᵥ²/(2γ)`; force Hessian bound `M` | Exact units, Gibbs normalization, Gaussian covariance and Rastrigin derivatives | Conservative continuous kinetic reference; native configurations use `D=0.5` and `θ=0.5/γ` |
| 10 | LSI `Cₓ≤maxⱼ (θ/κⱼ)exp(bⱼ/θ)`, `C₀=max(Cₓ,θ)` | Separable coordinate curvature and perturbation oscillation, followed by tensorization across coordinates and walkers | New complete corollary/proof in Chapter 10; no artificial exponential dimension or population factor |
| 10 | `L_M=2M+γ+2`, `η=D/[2(1+2M+L_M²)]`, `r=η/(C/2+3η)` | Positive gradient matrix eigenvalues `η,3η`, dissipation and predicted exponential envelope | 81 Gaussian density trajectories and 78 nonquadratic positive density cases |
| 10 | `H`, `Iₓ`, `Iᵥ`, mixed `Iₓᵥ`, modified entropy `Φ` | All four evolution identities, full positive gradient form, mixed second derivatives and finite quadrature refinement | Gaussian evolution and nonquadratic Gibbs-relative generator calculations |
| 10 | Selection/Fisher coefficients, jump decay, survival centering, Doob normalization | Exact selected-source terms, full positive-matrix selection Fisher bound, six common-refresh density time curves, specified killed kernels | Each operator retains its stated reference law; native cloning is not substituted for a refresh operator |
| 10 | Forcing floors, discrete defects and normalized variance `1/N` | Exact recurrences, 24,576 complete independent Gaussian reference populations | Errors are averages/probabilities; all raw populations are retained |
| 11 | Hellinger affinity, mass–shape identity and root-mass constant `1/(4m₀)` | Exact finite measures and recorded native alive submeasures, including zero-mass conventions | Probability normalization; normalized shape undefined at zero mass |
| 11 | Canonical HK reaction coefficient `1/4`, KL/Hellinger transfer, T2 coefficient `2C` | Reaction-path cost, finite transport paths, Gaussian reference transport and continuous-density entropy transfer | Finite-path and identified reference-law checks; no atomic-to-continuous KL estimate |
| 11 | Alive conditional mean `N⁻¹Σpᵢ`, variance `N⁻²Σpᵢ(1−pᵢ)≤1/(4N)` | Actual native prepared-state Gaussian survival probabilities, terminal masks and Poisson-binomial law | Whole kinetic conditioning and post-B2 conditioning are recorded separately |
| 11 | Hoeffding tails, stopped innovation/bracket identities and extinction | 306 calibration groups, 918 simultaneous distribution-free checks, actual killing times | All checks pass; conditional variances include native geometry, dimension and population |
| 11 | Mass recurrence `R`, forcing floor, continuous birth/death floor, conditional-survival factors | Exact recurrence operands and specified finite/reference laws; native mass histories stored separately | A native contraction premise is not inferred from fitting a curve |
| 11 | Shape transport, entropy-to-distance decay and tail budgets | 108 finite-grid law curves; 54 fresh unquantized projected pooled transport curves; 78 nonquadratic reference cases | Whole-trajectory uncertainty; explicit Gaussian envelope tails, with no position-barrier cost on the velocity tail |
| 12 | Physical permutation invariance | Complete transported companion graphs, clone gates, component rotations, jitter, BAOAB, cap and killing | 720 fresh native replays; maximum physical coordinate residual `7.494×10⁻¹⁵` |
| 12 | Exact collision `1−(N)ₖ/Nᵏ` and bound `k(k−1)/(2N)` | 737,280 independent sampling index lists on physical empirical clouds | Sampling `k=1,2,4,8`; exact finite empirical-orbit law |
| 12 | Covariance diagonal term, entropy variance budget `4B²(H_N+log(2)/2)/N` | Exact joint bounded-density tilts and normalized observable covariances | Total entropy budget remains independent of population; diagonal contribution is retained |
| 12 | Product LSI/T2 constants, Fisher factor `4`, OU covariance | Exact Gaussian laws and nonquadratic reference constant transfer | Conservative references; finite killed QSD symmetry also has a specified exact kernel |
| 12 | QSD uniqueness, uniform native LSI and weak/infinite limits | Individual hypotheses and dependent statements audited and retained | Analytic requirements; finite runs cannot establish their universal quantifiers |

## Experiment matrix and saved data

The new native experiment contains **54 configurations and 2,592 independently
seeded trajectories**: dimensions **1,2,4**, populations **8,32,128**, and six
profiles: quadratic confinement, Rastrigin wells, saddle preparation, tails,
quadratic absorbing/revival and Rastrigin absorbing/revival. Each side has 24
independent seeds, with different initial laws, and a requested horizon of 32
steps at `h=0.04`.

There are **82,642 complete trajectory updates plus 720 complete permutation
replays: 83,362 fresh native updates**. Eighteen trajectories reach the cemetery;
their stopping transitions and zero-mass continuation remain in the data.
Chapter 11 also reuses 30,528 earlier updates, explicitly distinguished from
new experiments. Complete states, configurations, seeds, source graphs,
component-shared randomness, forces, masks and innovations are saved as lossless
JSON/CBOR archives. Storage addresses do not label physical walkers.

## Does the measured error decrease?

All **81 Gaussian continuous-reference trajectories** decrease in entropy. Their
measured finite-horizon rates span **0.1473–3.6341**. The proved modified-entropy
envelope and instantaneous nonquadratic dissipation checks pass. The reference
Rastrigin parameters match the native force: `κ=2`, amplitude `10`, frequency
`2π`, `M=2+40π²`. The sufficient global rates are approximately
`1.3374×10⁻²³` at `γ=1,θ=0.5` and `1.1335×10⁻⁴⁰` at `γ=2,θ=0.25`.
These conservative barrier-based lower bounds are much slower than local
measured decay; they are independent of `d` and `N` in the separable model.

| Fresh native diagnostic | Endpoints decreased | Interpretation |
|---|---:|---|
| Alive submeasure grid Hellinger squared | 51/54 | Explicit finite probability pushforward |
| Alive normalized shape grid Hellinger squared | 51/54 | Separate shape from surviving mass |
| Grid `D²=H²+W₂²` | 48/54 | Combined mass/shape diagnostic |
| Grid projected shape transport squared | 45/54 | Quantized law contrast |
| Unquantized pooled projected transport squared | 45/54 | Exact monotone transport of pooled alive coordinates |
| Mean independent-pair empirical projected transport squared | 30/54 | Includes independent finite-population sampling error |

The following exact pooled measurements use `d=2,N=32` and physical time
`0–1.28`. Rates fit the stored time curve after time zero; they are descriptive
slopes with a sampling floor, not newly proved contraction constants.

| Profile | Initial `W₂²` | Final `W₂²` | Fitted decay rate |
|---|---:|---:|---:|
| Quadratic | 0.0144 | 0.00034769 | 3.719 |
| Rastrigin well | 0.0144 | 0.00009276 | 1.450 |
| Rastrigin tail | 0.0144 | 0.00005382 | 2.078 |
| Quadratic absorbing/revival | 0.0025361 | 0.00023315 | 0.497 |
| Rastrigin absorbing/revival | 0.0025361 | 0.00019031 | 0.810 |
| Rastrigin saddle | 0.0144 | 0.59238546 | −1.359 |

The saddle preparation can put the two initial ensembles into different wells.
Its error grows on this horizon. A global asymptotic convergence estimate with
an initial prefactor and an extremely slow barrier rate does not require this
pairwise distance to decrease at every short time. The measured result therefore
remains visible. Particle sampling error also explains why independent-cloud
transport can plateau while ensemble-law error decreases.

[All exact pooled curves and bootstrap operands](../outputs/convergence/chapters10-12-completion-20261005/chapter11/pooled-transport-v1/report.json)
include point estimates for all 54 configurations and 200 whole-trajectory
bootstrap replicates for the six representative profiles. The finite-grid
curves use 1,000 whole-trajectory bootstrap replicates. Intervals are pointwise.

![Native Hellinger decay](../outputs/convergence/chapters10-12-completion-20261005/figures-v1/native-grid_H_squared.png)

## Verification

The focused Rust integration tests, Python analysis regressions, release builds
and Clippy checks pass. The independent archive audit verifies complete decoding,
gzip integrity and compressed/decoded hashes. Exact source/executable snapshots
make the stored runs reusable after code edits. See the
[check log index](../outputs/convergence/chapters10-12-completion-20261005/checks/report.json)
and the final integrity report linked in the completion index.
