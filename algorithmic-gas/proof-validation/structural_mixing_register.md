# Structural landscape rate register

The Rust API in `crates/benchmarks/src/convergence_structural_mixing.rs` evaluates the actual primitive register in Chapter 6a Sections 15–16. It takes the declared kinetic parameters, configured selection channels, proved landscape profiles and complete kernel applicability conditions. Population size is absent from the rate parameters. A finite empirical approximation still has its separately proved particle error.

| Quantity | Source | Computation and meaning | Required condition |
|---|---|---|---|
| `H_c` | `def-slcc-regime`, `def-slcw-regime` | Global supremum of the actual completed center `|x+ηF(x)|`, with `η=(h/2)²(1+exp(-γh))` | A proved finite global supremum; a regional sample maximum is insufficient |
| `L_F` | Same definitions | Global force Lipschitz envelope, including nonconvex perturbations | Actual configured force, not an auxiliary replacement |
| `ε₂` | `lem-slcc-base-mixing` | Two-update reference-kernel minorization, from `H_c,L_F,d`, the OU/position noises, cap and declared radii | Positive noises, finite centers; both small-set probability and density factors included |
| `a_*,c_*` | `lem-slcc-selection-perturbation` | Actual clipped gate and accepted rooted-component bounds | `2c_*<1`; copying, full component collision and jitter retained |
| `L_R` | Same lemma | Signed full selection perturbation, including fitness normalizer and component exploration sensitivity | Bounded configured raw reward; finite comparison-feature diameter |
| `q₂` | `thm-slcc-active-contraction` | `1−ε₂+2L_R+L_R²`, giving TV envelope `q₂^floor(n/2)` | Conservative, all-alive, current-frame, nonviscous kernel; `q₂<1` |
| `M₄,β_w,B_w` | `def-slcw-regime`, `lem-slcw-weighted-kernel` | Actual output fourth moment, `β_w=ε₂/(2M₄)`, `B_w=1+ε₂/2` | Completed center valid even after selected copy/jitter; capped/collision-prepared velocities |
| `B_r,B_{r²},C_var` | `lem-slcw-normalization` | Raw reward mean and variance sensitivity in weighted TV | Proved `|R|≤C_r(1+|x|²)` and finite fourth moment |
| `L_rem` | `lem-slcw-selection-perturbation` | Weighted signed selection perturbation using root-averaged reward cost | Actual global normalization and component laws; `2c_*<1` |
| `q_w` | `thm-slcw-active-contraction` | `1−ε₂/2+2B_wL_rem+L_rem²`, giving weighted TV envelope `B_w q_w^floor((n−1)/2)` | Raw reward growth profile and `q_w<1`; no clipping imposed |
| Physical squared Wasserstein | `thm-slcw-alive-uniform-law` | `C_G sqrt(min(1,B_w q_w^floor((n−1)/2)+ε_N))` | The separate proved finite-particle error `ε_N` and moment-to-transport coefficient `C_G`; the time rate is independent of `N` |

The API records failed kernel, center, component and feedback conditions. It does not infer a positive rate from measured decreases. It preserves tiny probabilities and normalization constants logarithmically; an underflowed coefficient or positive exponent is not replaced by a made-up representable number. These formula evaluations are distinguished from the directed interval certificates in the Chapter 5 harmonic reference module.

For actual standard Rastrigin,
`F_j=−2x_j−20π sin(2πx_j)` and
`x+ηF(x)=(1−2η)x−20πη sin(2πx)`.
At `h=.04,γ=1`, the global center supremum is infinite, so the Sections 15–16 sufficient global certificate fails. This does not invalidate regional force/curvature, signed selection, tail or regeneration estimates; those must be combined through their own discrepancy operator and escape charges.

The source's `h=1,γ=0` Rastrigin regime has exactly `η=.5`, `H_c=10π√d`, and `L_F=2+40π²`. Its explicit global minorization is extraordinarily small. The CLI computes the full logarithmic register and positive-exponent interval for `d=1,2,4,8`, alongside an explicitly specified `p=.01` failed feedback gate, the actual `.04` center rejection and closed/failed bounded-center controls. No native trajectories or accepted edges are claimed for exponents that cannot be represented in binary64.

Run `cargo run --offline -p algorithmic-gas-benchmarks --bin gas-structural-mixing -- OUTPUT_JSON [PRIMITIVES_JSON]`. A custom JSON contains `kinetic`, `selection`, `profiles`, and `hypotheses`. The output retains exact formal source statements and their SHA256. `gas-kinetic-dimension sector OMEGA` provides an interval-certified harmonic reference at a declared well curvature, with the metric and both endpoint LMIs. Applying it to a nonquadratic well additionally requires the actual force remainder at both kick queries and the probability-weighted observable cost of leaving the well.
