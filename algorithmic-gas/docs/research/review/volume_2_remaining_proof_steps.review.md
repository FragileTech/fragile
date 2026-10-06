# Mathematical Review: Volume 2 exact-algorithm closure chain

## Metadata

- Reviewed roadmap: [volume_2_remaining_proof_steps.md](/home/guillem/fragiletech/fragile/docs/research/volume_2_remaining_proof_steps.md).
- Review date: 2026-10-02.
- Branch: `main`.
- Reviewer: Codex integration review, with parallel algorithm/readout, coupled-analysis and physical/spectral reviewers.
- User constraint: retain the actual algorithms, original hypotheses and parameters. Derive missing properties; do not introduce a convenient Hamiltonian, law, geometry, cutoff, noise, projection, vacuum or variant in their place.
- Framework anchors: the complete component tuples and parameter registers in Chapter 04 and structural landscape convergence; the executed native/Python call paths; the real-coordinate kernel scope in Chapter 18; the original selection consistency inputs retained by Chapter 19; lossless record transfer; the actual conditional-expectation/CAR operator; and the existing physical representation obstructions.
- This audit writes review artifacts only. It does not alter any proof, algorithm, parameter, source prose, branch, or user-owned work.

## Executive summary

**The requested airtight characterization is not established.** The new coupled real-coordinate estimates and finite spectral algebra survive the reviewed local checks, but that does not close the unchanged native physical chain. There is also a concrete false universal runtime claim in the older Einstein–Hilbert section.

- Critical: no newly identified false main theorem in the checked new finite-gas or finite-matrix proof blocks. This is not a global correctness certificate.
- Major: a concrete Einstein–Hilbert totality failure; substantial uncompleted physical operator/state identifications described below.
- Moderate: the asserted complete Latent/Python correspondence, finite-budget recording scope, and spectral mode/time-unit provenance.
- Minor: several readout/naming/constructor-comment scope issues, detailed in the algorithm report.
- Primary themes: identical names or common records do not prove identical operators, laws, states, time units, local algebras, or limits. Conditional examples remain tools, not algorithm characterizations.

## Dynamics-first rule and actual regime certificates

The user's clarification governs every recommendation in this report. The required order is

\[
\text{fixed configured transition }P_{\theta,U}
\;\longrightarrow\;
\text{explicit estimates of its actual stages}
\;\longrightarrow\;
\text{landscape/parameter regions where those estimates close}.
\]

A sufficient hypothesis from a general theorem is a conclusion to establish from these estimates, not an input to postulate. Failure of a sufficient bound does not imply failure of the algorithm. It identifies where that certificate stops. No recommendation here authorizes adjusting the algorithm or its configured parameters to make the certificate pass.

For each application the missing deliverable is a derived register containing: the exact variant and all configured controls; the landscape and its computed derivative/growth/tail profiles; the intermediate-stage bounds; the resulting sufficient parameter inequalities; their values at the existing configured instance; and any genuinely unclosed remainder. A profile can be infinite on an admitted landscape. It cannot be silently replaced by an assumed finite bound.

There are already dynamics-first examples in the reviewed coupled proofs:

| Actual calculation | Derived certificate / region | Existing-instance evaluation and limit |
|---|---|---|
| Count force has symmetric Gaussian weights bounded by one, divided by the actual population count | `0 ≤ L_x ≤ I`; actual kick is `I−(hν/2)L_x`; its stochastic/convex bound holds when `0 ≤ hν/2 ≤ 1` | At `h=0.04`, `ν=0.3`, the coefficient is `0.006`; no velocity-noise truncation is used. |
| Row force has actual nonself degrees and symmetric raw weights | Degree-weighted normalized Laplacian has spectrum in `[0,2]`; row kick is convex for `0 ≤ hν/2 ≤ 1`; an unweighted comparison retains the actual degree ratio | No global Gaussian degree floor is assumed on unbounded sites. Chapter 19 derives local target mass from its moment budget and retains exact `1/N` self subtraction. |
| Exact coupled B2 map under the existing quadratic landscape `F(x)=−λx` | Count coercivity margin `κ_count=1−λh²/4−hν/2`; row margin `κ_row=1−λh²/4−hν`; positivity is a derived sufficient region for the stated finite-population proof, with all remaining native component/noise/domain controls retained | For the unchanged reference `λ=1`, `h=0.04`, `ν=0.3`, margins are `0.9936` and `0.9876`; the configured reference normalization is count. Evaluating the existing row option does not silently switch the run. |
| Exact count-normalized force, both B kicks, both A drifts, OU, final position noise and original cap | `‖F_visc‖_(p,N) ≤ 2ν‖v‖_(p,N)` and the stage recursion below give explicit moment budgets | For quadratic `U`, `L_F=λ`, `f_0=0` are calculated from the landscape. For another landscape they must be computed or replaced by a proved applicable growth estimate, not assumed. |
| Actual unmasked cloning score | `K−Kᵀ=a uᵀ−u aᵀ`, rank at most two and the exact spectral formula in I-005 | Equal/near-equal fitness produces a genuine zero/small-gap region. That region must remain in the characterization, not be excluded to force a desired theorem. |

For example, with `a=h/2`, actual OU coefficient `c_h=e^(−γh)`, actual OU standard deviation `q`, actual final-position standard deviation `s`, and
\(g_{d,p}=(2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2))^{1/p}\), Chapter 19 derives

\[
\begin{aligned}
X_0&=B_D+\sigma_Jg_{d,p},&V_0&=(1+2|\alpha_{\rm col}|)V,\\
V_1&=(1+h\nu)V_0+aL_FX_0+af_0,&X_1&=X_0+aV_1,\\
V_2&=c_hV_1+qg_{d,p},&X_2&=X_1+aV_2,\\
V_3&=(1+h\nu)V_2+aL_FX_2+af_0,&X_+&=X_2+sg_{d,p},\\
V_+&=\min\{V,V_3\}.&&
\end{aligned}
\]

Here `B_D` is the radius bound of the algorithm's existing alive donor domain, not a new restriction of the output support. Gaussian outputs remain unbounded. The actual amplitudes are

\[
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,\end{cases}
\qquad s^2=\sigma_x^2h.
\]

The register still retains donor widths, selection/fitness floors and maps, gates, collision law, normalization, masks, geometry/recording and precision parameters even when a particular uniform moment bound does not depend on all of them. The recurrence estimates this count-normalized kinetic stage; it is not a full selection, stationary, row-normalized, numerical-error or geometry-feedback certificate.

For the remaining joint-law, continuum and physical identifications, the required work is likewise to estimate the existing law/descriptor dynamics and derive the admissible landscape/parameter regime. An assertion such as “assume joint LSI,” “assume the native action is Yang–Mills,” or “assume reflection positivity” cannot be entered as a discharge.

## Error and obligation log

The **Error** rows identify an actual source/contract problem. The **Open target** rows identify missing discharges or an obstruction to a proposed identification; their underlying conditional finite results may be correct.

| ID | Status | Location | Severity / type | Finding |
|---|---|---|---|---|
| I-001 | Error | Gas variants, `prop-variant-eh-identities`, Item 6 | Major / algorithm mismatch | Finite valid inputs need not yield successful native geometry; an existing budget branch gives an explicit counterexample. |
| I-002 | Error / contract ambiguity | Gas variants, `def-variant-latent`, implementation sentence | Moderate / algorithm mismatch | The inspected Python core does not establish the entire chart-metric latent component tuple. |
| I-003 | Error if interpreted as an unconditional API guarantee | Passive instrument versus native archive | Moderate / scope restriction | Recording can stop execution through its existing step/memory limits despite leaving common successful updates unchanged. |
| I-004 | Open target; ledger ambiguity | Lattice QFT spectral register; roadmap rows 55–56 and Stage 8 | Major / proof gap | The finite edge Hamiltonian is not identified with the actual recorded time evolution or physical transfer generator. |
| I-005 | Open target; exact obstruction | Explicit unmasked cloning-score matrix | Major / scope restriction | Its antisymmetrization has rank at most two; generic invertibility, unique ground and uniform record-wise gap do not follow. |
| I-006 | Open target | Filled ground, parity/gauge restriction, physical-time representation | Major / scope restriction | A different state or trivial gauge action cannot discharge native vacuum/correlation/local-Yang–Mills identification by itself. |
| I-007 | Open target, correctly conditional in sources | Chapters 15–17 | Note / hypothesis provenance | Actual joint-law LSI, complete entropy derivative and native correlated geometry/action comparisons remain unproved for the required coupled instance. |
| I-008 | Scope boundary, mostly explicitly retained | Chapters 18–19 and recorded readouts | Note / scope restriction | Real Gaussian versus finite-bit execution; completed-state versus tagged-history limits; passive geometry versus consumed geometry. |

## Detailed findings

### [I-001] A finite-input native geometry counterexample exists

The claim at [04_gas_variants.md:1291](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/1_the_algorithm/04_gas_variants.md:1291) says geometry is defined for every finite configuration and every finite innovation yields finite population output. The duplicate/rank convention does not prove this for the runtime API.

The algorithm reviewer traced this existing configuration without modifying the algorithm:

1. Eleven finite `f32` slots, all positions/velocities zero; the original `GasConfig::einstein_hilbert(0.33, 0.002)`, default geometry reward/memory and zero potential; existing `max_batch_elements=99`.
2. The largest prepared diffusion field has `11*3*3=99` elements, so the batch check passes.
3. Initial geometry receives `max_edges=99/2=49`.
4. Default duplicate lifting creates `11*10/2=55` clique edges and returns the native budget error because `55>49`.

This is a source-validated counterexample, not a claimed executed regression test. Failure is not physical killing: neither may be silently replaced by the other. The valid statement is about finite formulas/common successful execution with the native success/error outcome retained, not unconditional runtime totality or unlimited `M=N always` execution.

**Fix direction:** narrow that claim and its dependent uses; retain the existing failure, budgets and precision. Do not invent a fallback, enlarge budgets or modify the gas. No new assumption is needed to correct the overstatement.

### [I-002–I-003] The implementation and instrument must be named literally

The Python core uses Cartesian drift, Euclidean cap and its configured graph-viscous/adaptive options. Its frozen overlapping donor-star collision is not the native connected-component Haar collision. Those facts do not establish the full chart-metric Latent tuple declared near [04_gas_variants.md:1421](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/1_the_algorithm/04_gas_variants.md:1421). The new canonical viscous proofs are not proofs for that general Python or consumed-geometry update.

The formal passive instrument satisfies a marginal-kernel identity by construction. The standalone geometry helper is read-only. However, the bounded native archive checks `max_steps` before an update and reserves extra influence-record memory. An otherwise executable update can be rejected under those limits. Preserve the common-success qualification and the actual error outcomes; seed/checkpoint equality does not prove unlimited execution or a bit-level QSD transfer.

**Fix direction:** publish the component-by-component implementation correspondence and distinguish the ideal instrument, generic helper, bounded archive and consumed geometry. Do not retrofit a metric, cap, collision or force to make a correspondence true.

### [I-004] The existing Hamiltonian is valid, but its native physical identity is missing

The existing Hamiltonian is retained: `H=dΓ(i(K−Kᵀ))`. Its finite occupation-spectrum, parity and ground-space proofs are valid for their specified matrix. No new Hamiltonian or cutoff is required for that calculation.

But [03_lattice_qft.md:253](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md:253) calls the score coupling auxiliary, and [03_lattice_qft.md:825](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/2_fractal_set/03_lattice_qft.md:825) explicitly requires one-particle operator equality before identifying it with recorded evolution. The new spectral register still declares mode selection, basis, coefficient recipe and clock conversion as readout data rather than deriving that equality.

The actual recorded one-particle transition is the conditional-expectation contraction `C_t`. Its CAR channel has the already computed defect

\[
\mathcal Q_t(a(f)a^\dagger(g))-
\mathcal Q_t(a(f))\mathcal Q_t(a^\dagger(g))
=\langle f,(I-C_t^*C_t)g\rangle I.
\]

Where this is nonzero, the same CAR evolution cannot be Hamiltonian conjugation, which is multiplicative. A lossless record change transports that defect; it does not remove it. This does not exclude a separately proved reconstruction from the algorithm. It prevents pretending that the literal existing objects have already been identified.

The cloning score is dimensionless; multiplying its eigenvalues by an action unit is not an energy calibration until the existing score-to-time conversion is derived. The theorem retains this conditional; Stage 8's unconditional physical-energy wording needs it restored.

**Fix direction:** keep the finite theorem as a record-derived spectral result; derive the concrete native modes, their actual law inner product, coefficient transformation and time relation. Never assume the desired intertwining or replace the executed evolution to obtain it.

### [I-005] The displayed score has an explicit low-rank obstruction

For the chapter's unmasked score set `u_i=V_i+ε>0`, `a_i=1/u_i`. Direct substitution gives

\[
K_{ij}=u_j/u_i-1,\qquad
K-K^{\mathsf T}=a u^{\mathsf T}-u a^{\mathsf T}.
\]

Therefore rank is at most two. For nonconstant `u`, its two nonzero one-particle energies are

\[
\pm\sqrt{\left(\sum_i u_i^2\right)
                  \left(\sum_i u_i^{-2}\right)-m^2}.
\]

There are `m−2` zero modes for `m≥3`, giving full Fock ground dimension `2^(m−2)`. For equal fitness the Hamiltonian is zero; near equal fitness the nonzero magnitude approaches zero. In parity sectors, zero modes can still give degeneracy, although the `m=3` parity sectors each have a one-dimensional ground. No general parity-sector uniqueness conclusion is being negated beyond its actual formula.

Read-only numerical checks reproduced rank zero/equal-fitness and rank two/nonconstant examples. The factorization, not the numerical zero threshold, is the proof. A masked or weighted recorded edge recipe may behave differently; that exact original recipe must be exhibited, not substituted to eliminate zero modes.

**Fix direction:** apply the generic finite theorem to the actual recipe and retain all zero modes. Do not assume invertibility, exclude equal fitness, discard modes or add coefficients to force a target gap.

### [I-006] State, gauge and physical time are additional exact-object tests

Filling negative modes yields a different state from the empty exterior vacuum used in recorded CAR correlation identities. A scalar energy shift preserves conjugation dynamics, not those vacuum expectations. Occupation of a negative mode is zero in the empty vacuum and one in the filled ground. Parity and the specified regional algebra also remain part of the state identification.

Likewise, `u_g=I` correctly verifies commutation on invariant scalar modes. It does not derive the algorithm's local non-Abelian connection, action or physical Gauss-sector representation. Trivial gauge action on physical invariant observables is not itself an error; counting it as a local Yang–Mills discharge would be.

The existing [pullback-spectrum lemma](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/2_fractal_set/05_yang_mills_noether.md:6585) gives another precise obstruction: literal same-law pullback translations and their stated even exterior lift have symmetric energy-momentum spectrum. Forward-cone spectrum plus a unique invariant vacuum forces the trivial one-dimensional case. Merely shifting a different Hamiltonian does not identify those time implementers. The roadmap correctly recognizes this reconstruction obligation; it must remain open.

### [I-007–I-008] Keep the conditional and already verified scopes visible

| Proof block | What the reviewed proof establishes | What it does not establish |
|---|---|---|
| Chapter 18 | Exact coupled kicks, finite-population QSD/entropy, uniform marginal tails for the named real-Gaussian quadratic/capped/terminal-box kernel | Bit-level density/full support, joint LSI, uniform stationary mixing or a geometry-feedback variant |
| Chapter 19 | Under the stated initial empirical/measurement/moment conditions: fixed-step finite-horizon completed-state population limit; both Gaussian normalizations, exact self-exclusion and locally derived row degree | Arbitrary-initial-array closure, stationary attraction, tagged multi-time path convergence, shrinking-scale graph/action consistency; initialization conditions must be verified from the actual initializer |
| Passive record transfer | Same marginal kernel, covered state/readout laws, correctly defined incoming-record QSD lift | A state LSI for every incoming-noise function, or stationarity of a growing clock/history |
| Chapter 15 | Full proofs of stated LSI/entropy implications and exact marked-step identities | Verification of all structural law/derivative premises for the coupled marked QSD |
| Chapter 16 | Conditional local-estimator consistency, including correlated samples with the displayed covariance budget; sufficient kernel and independent-evaluation action examples | Actual shrinking-family episode covariance, native operator/action comparison, or independent native evaluation samples |
| Chapter 17 | Stated continuous coefficient/moment/entropy implications and supplied-metric geometric identities | Identity with the fixed-step Geometric/Latent update or native Einstein evolution |

The coupled reviewer checked regeneration of selection inputs at each fixed horizon and the exact native component order. The cited compact positive-operator theorem was also checked against its original statement: positivity, compactness, strong positivity and positive spectral radius are supplied by the local proof, not assumed from its name.

## Scope restrictions, coverage and validation

- Complete source-order reviews: the roadmap, Gas variants, Chapters 15–19 and Lattice QFT. The Yang–Mills review covers the relevant physical/gauge/spectral chain with its own coverage register. Dependencies in Fractal Set, Standard Model, structural landscape convergence, mean field, field equations and native/Python components were checked in the regions enumerated in the linked reports.
- **Not claimed:** a literal line-by-line audit of every other Volume 2 chapter, all optional implementations, or all external references. This review cannot honestly certify “every proof in the volume is airtight.” The established blockers already prevent the requested global closure.
- Tests and builds check implementation/examples/rendering, not mathematical validity. The preceding implementation turn's Python and compatibility Rust test results are historical evidence, not proof certificates. Required Rust 1.95 verification remains separate from the prior successful Rust 1.93 `--ignore-rust-version` compatibility run.
- No new analytical assumption, external theorem, source-proof fix, cutoff, projection, dynamics, state or physical reconstruction was introduced by this audit.

## Detailed review files

- [Algorithm and readout audit](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/1_the_algorithm/review/04_gas_variants.review.md).
- [Coupled finite-population proofs](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/review/18_coupled_gas_discharge.review.md).
- [Coupled population-limit proofs](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/review/19_color_geometry_mean_field.review.md).
- [KL/law provenance](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/review/15_kl_convergence.review.md).
- [Continuum provenance](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/review/16_continuum_discharge.review.md).
- [Geometric-gas provenance](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/review/17_geometric_gas.review.md).
- [Existing Hamiltonian/operator/state audit](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/2_fractal_set/review/03_lattice_qft.review.md).
- [Yang–Mills physical/gauge audit](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/2_fractal_set/review/05_yang_mills_noether.review.md).

## Proposed edits

1. Correct the concrete runtime/implementation overstatements without changing the implementation.
2. Narrow the discharge ledger to the exact record-derived finite calculations and named mathematical kernel actually proved.
3. For each remaining obligation, estimate the exact configured dynamics first and derive the supported landscape/parameter region with complete constants; do not postulate the hypothesis or retune the run.
4. Retain missing same-operator, same-state, same-clock, same-law and same-record limit proofs as unresolved obligations wherever those estimates do not close.

These are recommendations, not source edits performed under this review request.

## Open questions

1. What exact existing mode/edge recipe and physical-time construction connects the recorded dynamics to its intended Hamiltonian, state and local algebra?
2. Which joint stationary estimates can be derived for the unchanged required coupled/geometry-feedback instance, rather than its passive or frozen comparison?
3. Can the native same-record graph, metric, field, action and first variations converge jointly with all constants derived from the existing parameter register?
