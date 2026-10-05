# Chapter 4: dimension and structural landscape certificates

The implementation is `convergence_tightening_chapter04.rs`; `gas-tighten-chapter04`
re-analyzes the immutable Chapter 4 native dataset and writes a separate lossless
certificate archive. It evaluates every retained copying frame and every dense
reference force stage. It performs zero new native updates. Local certificates do
not receive coverage credit for an untested global source theorem.

The main certificates are normalized by probability mass. Their constants are
bounded independently of the number of walkers. Dimensions enter through actual
coordinate transport, covariance traces, Gaussian variance, and structural force
amplitudes. Dead storage is excluded from source measures; revival output remains
in the complete population.

| Certificate | Previous quantity | Refined quantity | Required hypotheses |
|---|---|---|---|
| Live-support positional reset | Pair bound `D² + 2(1−1/N)dσ²` | Sum of `rD²/[2(r+1)] + (1−1/N)dσ² p̄` for each swarm | Genuine live-support diameter; certified affine-rank upper bound; independent centered jitter. The implementation uses an exact coordinate-span upper bound, with no numerical-rank inference. |
| Normalized occupation reset | Support diameter | Actual donor-column occupation variance plus Gaussian contribution | Complete realized fitness and stochastic live-donor law. This upper bound permits dependent copying. |
| Exact conditional variance | Generic reset ceiling | Occupation variance minus copy barycenter covariance plus Gaussian contribution | Independent recipient copying after conditioning on complete fitness; native independent donor law. |
| Global reward moment | Global Hessian bound times coupling moments | Quadratic moment term plus `|a| Σ min(2,ω√C_b)` | Admissible probability coupling and finite normalized second moments; unrestricted support. Undisplaced coordinates pay zero periodic charge. |
| Genuine-box reward modulus | Maximum-coordinate triangle inequality | Exact scalar stationary-point maximum, multiplied by `√d` | The declared box belongs to the actual kernel; includes all analytic critical points. |
| Count viscous position derivative | Maximum-row coefficient `4νV_c e^(−1/2)/ρ` | Population Hilbert coefficient `2νV_c e^(−1/2)/ρ` | The two coefficients use their declared norms. Frozen bounded B1 velocities; full positions may be unbounded. |
| Conditional row derivative | Global position-radius ceiling | `ν s_v(s_x+r_i)/ρ²` from normalized donor covariance | Actual excluded-self Gaussian row law; positive represented denominator; derivative at a retained state. A uniform application requires a propagated moment/operator certificate. |
| Propagated position/velocity moments | Coordinate suprema or Gaussian cutoff | Normalized input `H₂`, bounded periodic-force amplitude, and full Gaussian covariance | Original BAOAB assembly; convex first viscous kick; bounded collision velocities; independent full Gaussian innovations. An unbounded-source law needs confinement/Safe Harbor input `H₂`. |
| Population row continuity register | Crude Gaussian Young bound | Optimized Young exponential bound and direct normalized second moments | The harmonic source-box population laws of 18a's RWM register. This does not verify the separate small-viscosity population-attraction endpoints. |
| Regional landscape response | Single global convexity/Lipschitz constant | Chapter 6a force, modulus, radial-defect and pairwise-defect profiles | Convex declared analysis regions; analytic cosine interval bounds; exterior/tail charges remain separate. |

For the canonical saved quadratic inputs, one live support has diameter two and
rank one; the other support is collapsed. The geometric reset term therefore
decreases from four to one. For the singleton revival case it becomes zero.
Isotropic noise still contributes its full ambient dimension.

For canonical Rastrigin inputs in either saved dimension, the actual mean reward
difference is `20.75`. The global curvature moment bound is approximately
`210.43`; the separated bound is `21.0606601718`. Only the first coordinate moves
under the collapsed-right probability coupling, so the refined bound pays no
extra periodic charge for the second coordinate.

The row event-radius derivative remains an **auxiliary finite-population**
comparison: `16νV_cR/ρ²` decreases to `(3√3/2)νV_cR/ρ²` on the same source proof
event. The original `J(log N)` dependence is retained. It is excluded from the
primary population-uniform comparison table. At the reference, the resulting
first-drift margin is below `0.012`; the count Hilbert margin is below `0.001`.
Neither margin is a fitted QSD law rate.

## Regional API for a general rate compiler

```rust
regional_landscape_profile(
    potential: &PotentialProfile,
    lower: &[f64],
    upper: &[f64],
    scale: f64,
    restoring_k: f64,
    comparison_lipschitz: f64,
) -> Result<RegionalLandscapeProfile>
```

The returned analytic fields are:

| Field | Meaning |
|---|---|
| `dimension` | Explicit ambient dimension |
| `harmonic_curvature` | `κ` in `F=−κx+e(x)` |
| `bounded_perturbation_amplitude` | Global `M_d=|a|ω√d` |
| `perturbation_lipschitz`, `regional_perturbation_lipschitz` | Global/regional periodic-force derivative upper bounds |
| `curvature_lower`, `curvature_upper` | Analytic Hessian enclosure on the convex box |
| `force_sup_upper`, `force_modulus_upper` | Certificates for Chapter 6a's `M_A`, `ω_A(r)` |
| `perturbation_modulus_upper` | `min(L_{e,A}r,2M_d)` |
| `excess_modulus_upper` | Certificate for `D_A(L,r)` |
| `radial_defect_upper`, `pairwise_defect_upper` | Certificates for the distinct `b_A(k)`, `J_A(k,r)` profiles |
| `global_radial_defect_upper` | Unbounded-space envelope `M_d²/[4(κ−k)]` when `k<κ`; absent when this method supplies no finite certificate |
| `reward_quadratic_growth`, `reward_constant_growth` | `|R(x)|≤κ|x|²/2+2|a|d` |
| `certified_restoring_region` | Whether the analytic curvature lower bound is positive |

For the native Rastrigin benchmark, `κ=2`, `M_d=20π√d`,
`L_e=40π²`, and `|R|≤|x|²+20d`. The central interval
`[−1/8,1/8]` has positive curvature lower bound `2+20√2π²`; the barrier interval
`[3/8,5/8]` has negative curvature throughout. These are the integer/half-integer
regions used by `landscape_phase.py`'s existing regional machinery. The API accepts
arbitrary declared boxes, including translated wells and barriers. A regional
profile alone does not certify residence, communication, selection pressure or
global mixing. Those quantities are separate inputs to the rate compiler.

Every sampled force/moment diagnostic is stored separately from its analytic
envelope. Empirical expectation comparisons use independent fixed-state restores
and paired residual standard errors; dense reference checks inspect each raw
stage independently. The final report preserves the source report hash and links
each derivative or occupation certificate to its original native archive.

## Completed retained-data checks

The separate `tightening-20261004/chapter04/report.json` contains 48 cases,
12,288 copying frames, 2,002 comparisons with zero failures, and 674,050
individual exact numerical predicates. It performs zero new native updates.
The six focused Rust regressions pass. The adjacent
`chapter04-derived-verification.json` independently verifies all 417 registered
derived artifacts against their compressed SHA256 and byte counts, gzip CRC,
and complete JSON decoding (99,470,964 decoded bytes).

| Retained case / quantity | Previous bound | Refined bound | Measured evidence |
|---|---:|---:|---|
| Quadratic canonical, `N=4,d=1` positional reset | 4.015 | 1.003881365 dimension / 0.366553042 occupation | Every retained fitness/donor frame; Gaussian expectation check on fixed-state independent restores |
| Sphere canonical, `N=4,d=1` positional reset | 4.015 | 1.003913048 / 0.340983003 | Same conditional source/support checks |
| Rastrigin canonical, `N=4,d=1` positional reset | 4.015 | 1.003941631 / 0.340152260 | Same conditional source/support checks |
| Constant canonical, `N=4,d=1` positional reset | 4.015 | 1.003881810 / 0.953481422 | Complete output includes revived rows |
| Quadratic selection, `N=4,d=1` positional reset | 4.06 | 1.018235944 / 0.018235944 | Analytic donor occupation, not fitted support radius |
| Singleton revival positional reset | 0.015 | 0.01125 / 0.01125 | The original geometric term was already zero; the improvement is the finite-population centered jitter factor |
| Canonical Rastrigin reward error, both saved dimensions | 210.426586 | 21.060660 separated coordinate moments | Actual error 20.75; displaced coordinate only |
| Dense count first-drift margin, `N=200,d=3` | 0.001564539 | 0.000982269 Hilbert certificate | Same declared normalized Hilbert norm for the refined bound; old maximum-row norm remains separately named |
| Dense row first-drift margin, `N=200,d=3` (auxiliary event) | 0.070452071 | 0.011775039 | Same finite-population event radius; excluded from the primary population-uniform table |
| Harmonic population row-continuity register `log M` | 33.039721 | 18.901401 | Analytic full-Gaussian expectation, source-box hypothesis |
| Harmonic population row-continuity register `H` | 33.039721 | 18.056853 | Direct normalized second moments |
| Harmonic population row-continuity register `log K` | 71.621201 | 43.085755 | `γ=0.05` and marked exponent `7.8125e−5` unchanged; no unverified row-attraction endpoints inferred |

The largest dense native force reconstruction residuals are `8.33e−17` (count)
and `4.44e−16` (row); the largest analytic-Jacobian versus central-difference
residuals are `1.78e−10` and `3.93e−11`. The largest ratios of actual derivative
action to its normalized covariance certificate are `0.380275` and `0.101995`.
Each reference stage is checked separately; standard errors cannot hide a bad
timestep. These are local operator validations, not measurements of a QSD rate.

The separate helper `audit_local_structural_kinetics.py` uses the independent
Chapter 6a interval certificate to check native Rastrigin well residence at both
actual force queries. It records all candidates before filtering, verifies actual
common OU and position noise, and evaluates each eligible pathwise `G` residual
with the midpoint-curvature metric. Costs for a full swarm are divided by `N`.
Selected-cloning inputs begin at the actual prepared B1 stage. The helper does
not turn post-residence conditioning into an expectation estimate or a complete
selected-update convergence claim.

For a full local audit, build the optional structural CBOR skipper and run the
helper against both completed native datasets. The skipper validates discarded
subtrees; required numerical fields still use the existing Python decoder.

```bash
cc -O3 -Wall -Wextra -Werror -shared -fPIC proof-validation/skip_native_cbor.c -o /tmp/fragile-skip-native-cbor.so
python3 proof-validation/audit_local_structural_kinetics.py \
  outputs/convergence/structural-landscape-tightening-20261004/independent-regional-interval-final.json \
  outputs/convergence/structural-landscape-tightening-20261004/local-kinetic-audit/new-complete-audit \
  outputs/convergence/structural-landscape-tightening-20261004/native-rastrigin-full \
  outputs/convergence/structural-landscape-tightening-20261004/native-selected-rastrigin-full \
  --skip-library /tmp/fragile-skip-native-cbor.so
```

The system compiler is optional; omitting `--skip-library` uses the original
package-free subtree traversal. Seven focused local-audit regressions pass,
including death exclusion, a single numerical violation, second-query well
membership, all-step decoding, malformed/deep CBOR rejection, and cached-ledger
predicate recomputation. A representative retained `N=64,d=2` archive produced
identical decoded fields with both traversals. The helper and skipper sources,
binary hashes, independent interval certificate, native archive hashes, and
every pre-filter membership are retained in the derived results. A recording
snapshot can be extended to the complete native indexes with
`complete_local_structural_kinetics.py`; only newly committed chunks are decoded,
and every cached predicate and artifact hash is checked again.

## Complete conditional native Rastrigin rate audit

The final result is
`structural-landscape-tightening-20261004/local-kinetic-audit/complete-20261004/report.json`.
It covers both completed native datasets: 4,608 paired archive chunks, 73,728
paired kinetic steps (147,456 native engine updates), and 5,013,504 walker/radius
candidates before filtering. The matrix uses `N=4,64`, `d=1,2,4`, 32 independent
addressed-seed replicates, 64 steps, and well/barrier/tail starts. Every candidate
has both actual query memberships retained, along with B1/B2/terminal live masks,
actual force identity and common OU/position noise checks. Full-swarm costs use
the normalized mean of the coupling-row costs.

| Well half width | Certified one-step `ρ` | Qualifying pure / selected walker checks | Largest observed pure / selected cost ratio | Fully resident pure / selected swarm steps | Violations |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.963844914129 | 20,533 / 2,065 | 0.961396033 / 0.960965491 | 0 / 0 | 0 |
| 0.03 | 0.978705048854 | 231,801 / 75,461 | 0.965366711 / 0.964125526 | 300 / 19 | 0 |

All 329,860 eligible individual residuals are nonpositive without using the
explicit numerical tolerance; no standard-error allowance is applied. The
319 fully resident swarm-step checks at width 0.03 also pass. There are no fully
resident swarm steps at width 0.01, so only the individual pathwise specialization
is empirically instantiated there. These constants use the independent interval
certificate's midpoint curvature, `α=1−ω_mid h²/4`, `β=0.025`, `δ=0.038`,
`h=0.04`, and the actual terminal soft cap. The constants are independent of `N`
and, for this separable well specialization, of `d`.

The complete per-dimension/population applicability table is in
`complete-20261004-case-applicability.json`. Every full-swarm resident event
occurs at `N=4`; no `N=64` swarm stays entirely in the required box at both
queries. Individual eligible pairs at `N=64` are nevertheless checked in all
three dimensions at width 0.03. The selected dimension-four runs have no
eligible individual checks at width 0.01, which is recorded as zero coverage
for that specialization rather than a successful empty comparison.

The final extension reuses 4,381 previously verified ledgers and decodes only
227 newly committed pure-source chunks. It recomputes every cached eligibility,
individual residual and normalized swarm predicate, rechecks their artifact and
native source hashes, and verifies complete archive/step coverage against both
final native indexes. The adjacent `complete-20261004-verification.json` records
the separate verification of every final ledger.

These are conditional pathwise kinetic estimates on actual resident force
queries. Selected-cloning inputs start after the actual native preparation.
The audit does not infer post-residence expectation contraction, full selected
update contraction, communication between wells, or a global stationary/QSD rate.
