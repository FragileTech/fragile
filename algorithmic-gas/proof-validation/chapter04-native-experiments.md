# Chapter 4 native experiments

`gas-proof-experiments` executes the actual Rust cloning and BAOAB providers and
retains their complete records. The Chapter 4 module evaluates normalized
empirical probability measures: permuting a swarm's storage order never changes
the physical error. Independent stochastic outputs are observations under a
declared coupling; their distance is not identified with the optimal distance
between transition laws.

Run the full chapter into a fresh directory:

```sh
cargo run --release --offline -p algorithmic-gas-benchmarks --bin gas-proof-experiments -- run proof-validation/chapter04-native-full.json --chapter 4 --output outputs/convergence/chapter04-new-dataset --strict
```

`chapter04-native-smoke.json` selects five small proposal profiles and both dense
reference kernels. The full matrix includes N = 4, 16, 64; d = 1, 2; Quadratic,
Sphere, Rastrigin, and Constant landscapes. Additional Quadratic profiles vary
selection parameters, revive a half-dead swarm, revive a single survivor, and
permute the intrinsic swarm representation. There are 48 two-swarm proposal
cases, each with 256 fresh native draws from a fixed entering empirical state.
Both count-normalized and row-normalized dense-viscous kernels run for 128 steps
at N = 200, d = 3. The full configuration executes 24,832 native updates.

Each proposal draw restores its entering state through the native population
replacement API. The native step address advances, so randomness is fresh.
The two engines use distinct recorded seeds, except for the exact permutation
probe, which uses the same seed and intrinsic canonical representation.
Reference trajectories advance continuously. Archived anchors distinguish
repeated proposals from continuous trajectories.

| Quantity | Experimental measurement and prediction | Applicability |
|---|---|---|
| Full transport decomposition | Solve both full and centered optimal q transport problems, with q(dx,dv) = \|dx\|² + \|dv\|² + 0.5 dx·dv; compare their difference with the q cost between barycenters. | Exact identity, checked on every proposal. |
| Centered positional bound | Solve optimal centered positional transport and compare with the sum of both output positional variances. | Exact geometric inequality, checked on every proposal. |
| Positional reset C_reset | Compare mean output variance proxy and mean centered positional transport with D_x² + 2(1 − 1/N)d σ_clone². D_x uses eligible incoming donors. | Chapter 4 quotes require all-alive entering states. Partial-input revival cases use the Chapter 3 eligible-donor reset certificate. |
| Conditional moment | Integrate the retained sampled fitness, donor probabilities, clone acceptance probabilities and jitter covariance; compare the exact conditional proxy expectation with the realized proxy. | Mean identity over independent native primitive draws. No averaging over a substituted fitness vector. |
| H, U, T and target fractions | H is selected from entering positional geometry before fitness sampling; U uses fitness at or below its realized arithmetic mean; T = H ∩ U. Compare measured fractions with their derived bounds. | Only positive recorded arithmetic fitness gaps admit the target certificate. Excluded outcomes remain archived. |
| s_*², R_*, B_acc, p_u | Derive the variance floor, configured bounded-fitness range, acceptance denominator and donor-overlap floor; compare the actual conditional target-row pressure with p_u. | Exact conditional inequalities, checked individually. |
| c_H, a_x, b_x, c_err, g_err | Use recorded target geometric energy, a fixed a_x = 1/2, b_x = 0, and explicit geometric admission; retain missed-target and complement error terms. | Generated matched-velocity input family; constants are conditional certificates, not universal empirical fits. |
| Q, χ, g_max | Integrate both swarms' actual conditional row probabilities against entering centered positional error; check Q ≥ χ V_struct − g_max. | Exact admitted-target inequality, checked individually. Positive-gap and positive-lower-bound counts are separate. |
| Revival and permutation | Check all N proposal outputs are alive; compare empirical output transport with zero for the exact permutation probe. | Mandatory revival and storage-order invariance. Dead storage positions are excluded from entering physical measures. |
| Dense viscosity | Reconstruct each B1/B2 Gaussian force from saved positions and velocities, using the actual eligible-count or off-diagonal row-mass normalization; compare with the native evaluated force. | Both N = 200 kernels. Count normalization additionally checks zero summed viscous force. |
| Harmonic force | Compare the actual native potential-force field with −x at B1/B2. | N = 200 harmonic references. |
| h, t, c, q², s, σ_J, V_c | Derive the BAOAB primitive quantities from the exact retained configuration. | Harmonic dense-viscous reference configuration. |
| Survival floor, r_*, J, C_x | Evaluate a rigorous Gaussian ball/interval lower bound in log space, then the induced jitter quantile and force derivative majorant. | A conservative analytic primitive certificate; no sampled survival minimum is substituted for a probability lower bound. |
| κ_F, β_F | Compare κ_F with 0.9936 / 0.9876 for count / row normalization; compare β_F with 0.002 / 0.071. | Native N = 200, d = 3, h = 0.04, γ = 1, ν = 0.3, ρ = 1. |

Expectation comparisons retain the mean residual, its standard error, and a
six-standard-error discrepancy test. Conditional and geometric inequalities
instead inspect every retained sample and report the worst numerical witness;
their violations cannot be hidden by ensemble averaging. This is finite
empirical evidence under explicit inputs, not a proof of every global theorem.

Raw native archives contain all input/output population fields, liveness,
sampled fitness, measurement and cloning donor choices, clone plans, random
primitives, stages and the component graph. Companion JSON archives retain the
three optimal transport plans, conditional moments, target certificates and
dense-force ledgers. Every stream chunk also saves a resumable checkpoint.
The shared store compresses CBOR/JSON and journals SHA256 digests after each
chunk. Dense reference chunks contain at most eight steps.

```sh
target/release/gas-proof-experiments verify DATASET --deep
target/release/gas-proof-experiments reanalyze DATASET --output EMPTY_DERIVED_DIRECTORY
```

Reanalysis verifies and recomputes conditional cloning quantities without
advancing the native engine. Source snapshots and inventories determine
whole-expression coverage. Missing global QSD/eigenfunction, minorization and
coupling hypotheses remain explicit; a decreasing pair error is not used to
certify their spectral rates. Chapter 6's separate survivor-law experiment is
responsible for its own distributional comparisons.

The initial compact strict run (16 samples, two reference steps) passed 63
comparisons and retained 109 archives covering 164 native updates. Deep decoding
and checksum verification passed. The source-expression applicability gate was
subsequently tightened for partial-input centered-reset provenance without
changing any measured numerical quantity; the complete source inventory remains
explicitly incomplete.
