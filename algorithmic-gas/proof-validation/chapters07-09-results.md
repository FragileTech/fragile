# Chapters 7–9: estimates and empirical validation

The native Rust implementation is extended with source-specific checks for the
next three convergence chapters. Every expression retains its exact formula,
chapter line, hypotheses and law. The probability normalization is `1/N`;
swarm addresses serve replay and recording. No walker labels enter a theorem or
observable. Death and mandatory companion revival are replayed as actual native
events.

The source inventory contains **105 formal statements and 1,021 expressions**:
9/54 in Chapter 7, 31/142 in Chapter 8, and 65/825 in Chapter 9. An expression
inventory includes definitions, assumptions and limiting assertions as well as
finite estimates; their dispositions are individually visible in the linked
tables.

All paths below are relative to `algorithmic-gas`. The retained experiment root is
[`outputs/convergence/chapters07-09-completion-20261004`](../outputs/convergence/chapters07-09-completion-20261004).

| Chapter | Constants and estimates | Measurements and comparisons | Detailed evidence |
|---|---|---|---|
| 7: Discrete QSD | Alive/dead balance, conditional survival, conservative generator residual | Specified equilibrium laws and exact recorded native alive/measurement/kinetic stages | [All 9 statements and 54 expressions](../outputs/convergence/chapters07-09-completion-20261004/chapter07/ledger-v1/results.md) |
| 7 | Gaussian-weighted distance integral; PPP ball volume, Gamma mean and intensity exponent `−1/d` | Actual native measurement kernel versus nearest distance; 122,880 independent Poisson clouds, with every point retained | [Native operands](../outputs/convergence/chapters07-09-completion-20261004/chapter07/native-laws-v1/report.json), [PPP experiments](../outputs/convergence/chapters07-09-completion-20261004/chapter07/poisson-clouds-v1/report.json) |
| 7 | Cloning-equilibrium normalization `Z`, inverse temperature, iso-fitness constant | Exact Gaussian equilibrium references; analytic divergence of an inadmissible normalization remains explicit | [Expression ledger](../outputs/convergence/chapters07-09-completion-20261004/chapter07/ledger-v1/statement-ledger.json) |
| 7 | Scalar absorbing decay `D₀π²/L²`, sine profile and forced-source parabola | 27,648 Rust PDE integration steps; explicit spatial/time discretization allowance | [PDE data](../outputs/convergence/chapters07-09-completion-20261004/chapter07/diffusion-reference-v1/raw.json), [discretization derivation](chapter07-diffusion-reference.md) |
| 7 | OU temperature, covariance, `exp(−γt)` W₂ multiplier; reset decay `2γ+r` | Exact OU/reset reference laws; 2,193,408 native O-stage coordinates and complete selected/capped moment decomposition | [Native and reference checks](../outputs/convergence/chapters07-09-completion-20261004/chapter07/native-laws-v1/report.json) |
| 7 | Decorated Gibbs `q_C,K,a,C`, residual and L¹/observable bounds | Independently specified finite two-state law and resolvent; native stationary applicability keeps its hypotheses | [Complete statement ledger](../outputs/convergence/chapters07-09-completion-20261004/chapter07/ledger-v1/statement-ledger.json) |
| 8: Mean field | Positive kernel floors, feature diameter, self-exclusion, sampled measurement law and nonlinear fitness | Exact native marked distributions, global normalizers and frozen fitness arrays | [All 31 statements and 142 expressions](../outputs/convergence/chapters07-09-completion-20261004/chapter08/final-ledger-v3/expressions-table.md) |
| 8 | Clone gates, incoming/component moments, momentum, restitution energy and shared Haar covariance | Actual accepted forests, mandatory revival and complete component rotations; no independent per-row collision surrogate | [Native suite](../outputs/convergence/chapters07-09-completion-20261004/chapter08/full-v1/report.json), [supplement ledger](../outputs/convergence/chapters07-09-completion-20261004/chapter08/final-ledger-v3/report.json) |
| 8 | B1/A1/O/A2/B2 stages, cap, terminal alive/dead probabilities and Chernoff exception budget | Complete physical operands, Gaussian draws, force kicks, actual terminal classification and survival probabilities | [Native suite](../outputs/convergence/chapters07-09-completion-20261004/chapter08/full-v1/report.json) |
| 8 | Dimension-dependent `A_q,B_q`, Gaussian norm moment `2^(q/2)Γ((d+q)/2)/Γ(d/2)` | Explicit N-independent normalized moment recurrence, force envelopes, actual component cap and Gaussian terms | [Formal derivation](../../docs/source/2_fractal_gas/convergence_program/08_mean_field.md), [bound checks](../outputs/convergence/chapters07-09-completion-20261004/chapter08/final-ledger-v3/report.json) |
| 8 | Complete rooted population update and observable law | 23,040 independent untruncated rooted draws versus 2,880 independent complete native updates; sampled fitness marks and shared collision randomness preserved | [Complete operands](../outputs/convergence/chapters07-09-completion-20261004/chapter08/full-v1/report.json) |
| 8 | Viscosity kicks and finite self-exclusion; repeated cap and independent-reference transfer | Both actual native force stages, 32 cap applications and exact normalized coupling fixtures | [Viscosity evidence](../outputs/convergence/chapters07-09-completion-20261004/chapter08/viscosity-v2/report.json), [combined ledger](../outputs/convergence/chapters07-09-completion-20261004/chapter08/final-ledger-v3/report.json) |
| 9: Propagation of chaos | `C,D_D,L_q,H_s,L₀,F_*,F*,L_a,B,A_T,L_T,a,N₀`, component moments `M₁,M₂,M₃,M_p` | Actual parameter-derived bounds; high-order component moments evaluated in log space; correct native operator hypotheses checked | [All 65 statements and 825 expressions](../outputs/convergence/chapters07-09-completion-20261004/chapter09/estimates-table-v2/estimates.md), [constants and native comparisons](../outputs/convergence/chapters07-09-completion-20261004/chapter09/bridge-full-v1/report.json) |
| 9 | Influence `A_D`; conditional variance `A_φ/N`; bias `2‖φ‖∞B*/√N`; MSE with reference sampling allowance | Independent complete native seed vectors and independent rooted vectors; rigorous bound comparisons separated from fitted exponents | [Bound comparisons](../outputs/convergence/chapters07-09-completion-20261004/chapter09/bridge-full-v1/report.json), [measured rates](../outputs/convergence/chapters07-09-completion-20261004/chapter09/population-scaling-v1/rates.md) |
| 9 | One-companion replacement `E[D_r²]≤A_D` and fixed-exceptional conditional bound `9Q²M₂(2C)` | Exact native fitness/plan replay; actual globally recomputed fitness after one draw replacement; 30,720 exact row-conditioned graph preparations in 60 fixed strata | [Controlled replacement operands](../outputs/convergence/chapters07-09-completion-20261004/chapter09/native-measurement-replacement-full-v1/report.json) |
| 9 | Original active-selection influence | 2,880 original-parameter measurement replacements; 879 changed acceptance cases and 1,182 changed gates, with original donor/gate randomness held fixed | [Original active operands](../outputs/convergence/chapters07-09-completion-20261004/chapter09/native-original-measurement-replacement-full-v1/report.json) |
| 9 | Killed-kernel eigenvalue, survival reweighting, minorization, drift, Wasserstein transfer, empirical-law stationary defect and reward moment transfer | Exact independently supplied finite killed kernels and eigenpairs; reward growth and uniform-integrability conditions retained; unknown native QSD is an analytic application | [Complete Chapter 9 source ledger](../outputs/convergence/chapters07-09-completion-20261004/chapter09/algebra-final-v2/report.json) |
| 9 | Ordered-star covariance and one-step Gaussian special-case rates | Complete supplied zero-selection special case; native shared-Haar law retained separately | [Complete Chapter 9 source table](../outputs/convergence/chapters07-09-completion-20261004/chapter09/estimates-table-v2/estimates.md) |
| 9 | Distance-law Lipschitz constants; global `L_F,L_β,L_step,L_pos` | Actual normalized kernel perturbations and tagged source/jitter laws; full population-map variation retains its complete rooted-law proof | [Exact source bindings and operands](../outputs/convergence/chapters07-09-completion-20261004/chapter09/distance-variation-review-v3/results.md) |
| 9 | Quadratic kinetic memory identity and Fourier conditional mean | 1,152 retained complete native frames; inverse cap, both physical positions and actual B1/O/B2 terms checked independently | [451,638 comparisons](../outputs/convergence/chapters07-09-completion-20261004/chapter09/independent-native-memory-v1/report.json) |
| 9 | Resonance `h=2`: uncapped velocity `−V`, capped velocity `C(−V)` | 1,152 fresh complete native updates across dimensions, populations, prepared velocities, OU amplitudes and position diffusion | [107,520 comparisons](../outputs/convergence/chapters07-09-completion-20261004/chapter09/native-resonance-v1/report.json) |

The core native matrix has 45 configurations: dimensions **1/2/4**, populations
**8/32/128**, and quadratic unbounded, Rastrigin well/saddle/tail and quadratic
absorbing/revival inputs. Each configuration uses **64 independent complete native
updates** and **512 independent rooted draws**. The rooted sampler fails on an
insufficient capacity; it never changes the law through conditional truncation.
Separate resonance experiments add 1,152 complete native updates. Thus these
experiments add **4,032 fresh complete native updates**, distinct from reference
draws, retained updates, primitive counterfactuals and PDE steps.

The Chapter 7 ledger contains **4,131,993 comparisons, zero failures**. Chapter 8
contains **4,904,999 comparisons, zero failures**. Cross-chapter variance/bias
references are aliases and are not counted again in Chapter 8.

Chapter 9 contains **37,764 distinct source assertions** after excluding native
identity aliases, with zero failures and no unmatched finite-estimate row.
**237 expressions** have individual numerical bindings: 234 finite comparisons,
two finite inequalities whose limit clauses remain analytic, and one full-law
certificate explicitly restricted to the complete zero-gate kernel. The remaining
380 definitions/proof intermediates, 184 hypotheses/domain contracts and 24
global/limit obligations have explicit dispositions. Native primitive identity
counts of **451,638** for memory and **107,520** for resonance are reported
separately; their aggregate source rechecks are aliases.

The [population scaling analysis](../outputs/convergence/chapters07-09-completion-20261004/chapter09/population-scaling-v1/report.json)
retains 2,000 bootstrap replicates per configuration, resampling complete observable
vectors rather than walker rows. All **92 measurable variance** and **93 measurable
MSE** endpoint comparisons decrease from N=8 to N=128. Zero-variance observables
and a sparse survival-variance interval remain explicitly undefined. Reference
sampling variance is retained; measured MSE is against an estimated rooted mean.
The bootstrap intervals are pointwise descriptive uncertainty, not simultaneous
universal rate certificates.

![Native population variance](../outputs/convergence/chapters07-09-completion-20261004/chapter09/population-scaling-v1/native-population-variance.png)

Measured PPP density slopes at dimensions 1/2/3/4/8 are
**−1.002902/−0.496919/−0.332988/−0.250156/−0.124864**, compared with the
predictions **−1, −1/2, −1/3, −1/4, −1/8**. These are actual finite Poisson clouds,
with empty clouds, censoring and tail bias retained.

The [finite-horizon analysis](../outputs/convergence/chapters07-09-completion-20261004/chapter09/horizon-variance-table-v1/table.md)
reuses **27,648 native updates in 1,728 independently seeded trajectories**.
At step 16, sine-observable variance decreases from N=4 to N=64 in **18/18**
weak-selection, **14/18** friction-2 and **16/18** cap/jitter/diffusion cases.
Those parameter-dependent measurements do not supply a uniform-in-time theorem.
Paired sides remain separate when computing independent-seed uncertainty.

Chapter 8 now includes a complete explicit dimension-dependent moment derivation.
Chapter 9 admits continuous rewards of at most quadratic growth and derives
reward-square uniform integrability from the stated higher-moment envelope;
Rastrigin is covered by its bounded smooth perturbation. Unbounded confinement
and moment conditions are retained rather than replaced by compact support.

Global QSD concentration, attraction, fixed-point uniqueness, LSI and infinite-law
limits keep their explicit analytic hypotheses. OU, Gibbs, scalar PDE and
finite-kernel references validate their own formulas and discretizations. They
are not measurements of a stationary law of the complete selected gas. Large
global constants remain visible in log space; finite agreement does not make
those bounds sharp.

The replacement experiment uses **`D_upper`**, the conservative union of
affected native collision components. It includes rows with unchanged individual
gates when their shared component can change; it does not measure final physical
coordinate differences. Conditioning fixes both measurement arrays and all
exceptional outcomes and rejects each remaining row only on acceptance
disagreement. It never rejects a large component. Frozen accepted-edge intensity
diagnostics are retained beside global constants, with their narrower input scope.
For the original active parameters, mean `D_upper²/N²` decreases from
**0.104720 to 0.008054 to 0.000547** at N=8/32/128. These are complete
preparation-replicate measurements, not a temporal convergence rate or a sharp
estimate of the global `A_D` constant.

All complete native stages, innovations, checkpoints, configurations, rooted
components, reference operands and resampling addresses are saved in lossless
archives with compressed and decoded checksums. Reports preserve source and
input-index hashes. Failed development probes are retained separately from final
datasets.

Focused validation passed **20 Rust tests** and Clippy for the new chapter
modules and experiment binaries; the Python analysis helpers pass Ruff. The
three Chapter 9 Python regression tests also pass. The
independent native-statistics recomputation passes **1,260 checks**, separately
from theory comparison counts.

The final artifact audit independently read **7,144 indexed compressed
artifacts** across 14 final datasets: **850,846,834 compressed bytes** and
**5,601,055,988 decoded bytes**, with no integrity failure or materially missing
compressed checksum. Every indexed native CBOR record was completely parsed;
JSON operands, gzip CRC, recorded decoded hashes/lengths and source snapshots
were checked separately from mathematical comparisons.

Reproduction commands and runner names are in [README.md](README.md#chapters-79-equilibrium-models-marked-laws-and-population-rates).
The [machine-readable completion index](../outputs/convergence/chapters07-09-completion-20261004/completion-index.json)
binds all three final ledgers to the current exact sources and records final
dataset/report hashes without double counting reused native updates.
