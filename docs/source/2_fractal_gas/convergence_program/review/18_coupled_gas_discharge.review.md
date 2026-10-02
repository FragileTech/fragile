# Mathematical Review: 18_coupled_gas_discharge.md

## Metadata

- Reviewed file: [18_coupled_gas_discharge.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/18_coupled_gas_discharge.md).
- Review date: 2026-10-02.
- Reviewer: Codex integration reviewer, with independent spectral/population review.
- Scope: Linear review of the complete formal chapter, including the added population-independent marginal QSD tail certificate.
- Framework anchors:
  - `def-slc-parameter-register`, `def-eg-baoab-canonical`, and `def-variant-viscous-euclidean`: unchanged marked component-collision kernel and exact B1/A1/O/A2/B2/noise/cap/boundary order.
  - `thm-chaos-canonical-finite-n-qsd`: effective-coordinate compactification preserves retained physical dead positions.
  - `lem-kl-bounded-reweighting` and `thm-hypocoercive-canonical-discrete-entropy`: bounded Doob tilting and entropy comparison, with complete survival conditioning.
  - `thm-variant-b2-color-nondegeneracy` and `thm-variant-passive-record-transfer`: same-stage color and passive observation-law transfer.
  - The compact positive-operator permit already used in Chapter 09; its [cited Theorem 1.1](https://arxiv.org/pdf/1606.04377) requires a compact positive operator, positive spectral radius, and a solid total cone, with strong positivity for a strictly positive principal eigenfunction.

## Executive summary

- Critical: 0 active.
- Major: 0 active.
- Moderate: 0 active.
- Minor: 0 active.
- Notes: 8 scope/quantitative records: N-001–N-004 and A18-001–A18-004. No note is counted as a discharged downstream theorem.
- Primary themes: Coupled output rows must not be factorized; a density lower bound needs bounded preimages and an upper Jacobian, not global invertibility; marginal QSD tails do not provide joint LSI or stationary chaos.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| N-001 | Nonlinear smoothing/minorization | Note | Scope clarification | Proper surjectivity and a null critical set are sufficient; global nonsingularity and two-sided density bounds are not concluded. |
| N-002 | Doob and QSD entropy formulas | Note | Definition clarification | These are distinct laws related by bounded reweighting; killed output is normalized once through the full iterate. |
| N-003 | Color and numerical-kernel scope | Note | Scope clarification | B2 force validity is separate from B1 consensus and terminal force recomputation; continuous density applies to the book's real-Gaussian kernel, not bit-level finite states. |
| N-004 | Mixing-rate parameter register | Note | Quantitative remainder | Eigenfunction extrema are exact spectral functions of the full kernel, not primitive-parameter closed bounds. |
| A18-001 | `rem-cgd-real-kernel-scope`; native `random.rs` | Note | Scope restriction | Exact Gaussian/Haar analytical law is not the addressed finite-bit execution law; their quantitative comparison remains unproved. |
| A18-002 | `thm-cgd-readout-law`; native recording/bookkeeping | Note | Scope restriction | Deterministic state decoration does not turn an incoming record or an accumulating history/clock into a stationary state function. |
| A18-003 | Reference QSD and tail corollaries | Note | Parameter scope | Verified quadratic reference certificates do not discharge other landscapes, feedback geometry, or all native configuration options. |
| A18-004 | `cor-cgd-reference-uniform-tails` | Note | Computational scope | The positive survival certificate is extremely conservative and underflows if evaluated directly in ordinary floating point. |

## Detailed findings and verified proofs

### [N-001] Alignment, smoothing and a common density

1. The count Laplacian quadratic form is bounded by the complete unit-weight graph form divided by population size. This proves `0 <= L <= I`. Convexity of the kick follows in its stated step regime. Row normalization instead preserves the degree measure, not ordinary summed momentum.
2. The intermediate second kick is the actual map of the OU velocities with positions `x1 + t z`. Its norm lower bound has the declared count or row margin. The affine homotopy cannot have a boundary zero on a sufficiently large ball, so the existing degree argument supplies a preimage of every target.
3. At consensus, derivatives of the kernel weights multiply zero velocity differences. The remaining derivative has a convergent Neumann inverse under the computed margin. Its analytic determinant is therefore nontrivial, and the critical source set is null.
4. The joint map has the final independent position Gaussian in addition to the OU Gaussian. Its determinant separates into the positive position-noise factor and the B2 determinant. Countably many noncritical charts establish absolute continuity; surjectivity and positive Gaussian density establish full support.
5. For the fixed target box and cap ball, coercivity bounds all preimages. The displayed local derivative bounds give an upper determinant bound. A regular target has at least one bounded preimage; its Gaussian density divided by that determinant gives the common lower density. Mixing over finitely many patterns and all jitters/rotations preserves the bound by Fubini.
6. TV continuity follows by the proved local-chart pushforward lemma. The effective input space is compact and the output has full support. Thus compactness and strong positivity satisfy every hypothesis of the already cited positive-operator permit.

- Required new assumptions: None for the existing reference conclusion. Its force, noises, terminal box, cap, widths and floors are the original configured data; the margin is numerically and algebraically verified. General parameter formulas are applicability certificates, not statements that every configuration passes.
- Validation: Both margins are tested independently in the numerical diagnostics. Referenced theorem labels resolve within the published TOC.

### [N-002] QSD versus the invariant Doob law

- The eigenfunction-weighted transition is Markov by the eigen-equation. Its common part has the stated weight, because the eigenfunction is normalized by its maximum.
- The common-part decomposition contracts TV and relative entropy by its complement. Reweighting at the start and end reconstructs the exact normalized killed iterate.
- The two tilts each have cost at most the reciprocal eigenfunction minimum. This gives the displayed entropy prefactor without substituting an LSI or factoring coupled velocities.
- The killed eigenmeasure and conservative Doob invariant measure remain distinct objects throughout the chapter.
- Required new assumptions: None beyond the already proved kernel eigenproblem and finite relative entropy for a finite entropy bound.

### Population-independent retained-position tails

- Location: `thm-cgd-uniform-marginal-qsd-tails` and its reference corollary.
- Verification:
  1. The existing positive QSD margin implies the convex-kick regime for both normalizations. Every post-collision velocity is bounded by the original collision/cap envelope.
  2. Substituting the actual first kick into the exact position formula isolates a Gaussian marginal containing the tagged clone jitter and the two position-affecting kinetic innovations. The remaining alignment contribution is bounded pathwise even when correlated with that Gaussian. No independence of those two terms is claimed.
  3. Squaring the triangle inequality gives the explicit Gaussian exponential-moment bound. Its denominator is positive at the stated analysis choice; the reference evaluates every primitive coefficient.
  4. For the survival lower bound, only the tagged latent jitter is restricted to a proof event. Other rows' jitters remain arbitrary because the alignment contribution is bounded by convexity. The final tagged position is Gaussian conditional on the complete preparation; density-times-ball-volume gives a positive survival floor independent of population size.
  5. The QSD eigenmeasure identity divides the unconditional output envelope by the survival eigenvalue. The proved uniform survival floor bounds that divisor. Monotone convergence, exponential Markov inequality and the displayed Gaussian tail integral give all retained-row moments and tails.
  6. The output alive fraction is zero at extinction, so its killed expectation agrees with its full-update expectation. Averaging the tagged lower bounds proves an expected alive-fraction bound, not a probability concentration estimate.
- Required new assumptions: None. Analysis radii and exponential-test parameter restrict calculations, not the algorithm or Gaussian support.
- Validation: The reference formulas for the drift factor and both integrated variances were independently substituted from its complete register. The certificate is conservative and can be astronomically small; rounded floating-point evaluation must not be mistaken for its analytic positivity.

### [N-003] Color, geometry and finite precision

- The terminal physical QSD has an absolutely continuous stratum law by the kernel eigenidentity. A nontrivial analytic terminal-force component therefore vanishes on a null set.
- Native B2 color uses the actual uncapped OU velocities and their corresponding A2 positions. Its direct analytic proof is separately cited. Neither assertion is used to declare a resting B1 force nonzero.
- Passive state decoration is a measurable conjugacy and preserves laws, TV and entropy exactly. The recorded last-step instrument retains its own source/terminal distinction.
- The chapter explicitly inherits the book's real-coordinate Gaussian analytical kernel. The native precision, rounding and underflow branches are not erased, and a continuous-density theorem is not asserted for a finite bit-state pseudorandom implementation.

### [N-004] Explicit primitive constants versus exact spectral data

- The minorization, extinction, marginal survival and tail constants are displayed functions of all consumed primitive parameters and analysis choices. Independence of unused reward/donor values is proved by uniformity over their realized patterns.
- The eigenfunction minimum and its weighted minorizing mass are exact quantities of the identified full-kernel eigenproblem. No unproved primitive closed expression is supplied for them.
- A fully primitive mixing prefactor/rate remains a quantitative extension. Replacing these quantities by unnamed constants or assuming a uniform lower bound would exceed the proof.

## Scope restrictions and clarifications

- The QSD proof concerns the existing terminal-box quadratic/capped gas; it does not establish an unbounded conservative invariant law.
- Uniform marginal QSD tightness is now proved for both force normalizations. Joint concentration, stationary chaos and a full continuous/discrete entropy inequality are not consequences of it.
- The initial law for the finite-QSD attraction statement lies in the actual nonextinct capped state space.
- The status obstruction rules out a continuous-gradient-only LSI on a law with multiple positive-mass status strata; the discrete form cannot be omitted.
- Geometry readout consistency, non-Abelian curvature and a physical gauge mass gap remain separate target identifications.

## Proposed edits

- No active mathematical correction remains in the reviewed proof blocks.
- Keep the spectral-data remainder and numerical-kernel scope visible in downstream summaries.

## Open questions

- Derive primitive bounds for the eigenfunction ratio rather than assume them.
- Prove population-uniform stationary dependence/entropy control for the unchanged required gas.
- Establish same-record continuum geometry and the native physical-sector/operator identifications without changing the kernel.

## Adversarial re-audit — 2026-10-02

### Coverage and method

This pass read every formal definition, estimate, proof and scope proposition in the 1,212-line current Chapter 18, then checked the consumed source arguments and native transition stages. It does **not** claim an exhaustive review of the entire book. In particular, the 18,901-line Chapter 06a was checked for its complete parameter/profile register, cap and collision envelope, and the specific nonlinear-density/Brouwer arguments relevant here; its many other landscape theorems were not re-certified. Chapter 09 was checked for the finite-QSD compactification proof, canonical component-collision regime, component bounds and the used innovation-variance/survival-conditioning distinctions. Chapter 15 was checked for the bounded-reweighting proof, Doob/QSD identities and status-entropy obstruction, not for all of its continuous-time estimates.

Native evidence was read in `variants/euclidean.rs`, `variants/viscous_euclidean.rs`, `kinetic.rs` (`recorded_force` and BAOAB), `cloning.rs` (decision, frozen literal copy, accepted components and shared component rotations), `fitness.rs` (global regularized statistics), `donor.rs` (current no-self/no-history defaults), `noise.rs`, `random.rs`, and the corresponding `engine.rs` complete-step ordering. Reference initialization was checked in `benchmarks/src/lib.rs`. No proof or core implementation was changed by this pass.

### Reconstructed transition and boundary cases

- Frozen measurement statistics use the eligible rows; sampled diversity is not replaced by its conditional mean. Cloning donors and gates are drawn from that frozen input. Every dead row is revived from a current eligible donor. Literal copies read the frozen pool, not earlier destination writes.
- Accepted connected components are built from the simultaneous graph. One Haar rotation acts on every component's **old** velocities, including retained dead velocities. An unaccepted donor may still be in a component and have its velocity changed. This is the canonical component rule, not ordered donor-star collision.
- In the reference `EndOfStep` schedule, no absorbing-box classification occurs between revival/jitter and the B stages. Thus all rows are eligible at both B1 and B2, and the count denominator is exactly N. B2 recomputes its force at the actual A2 positions and uncapped OU velocities. The final position noise, radial cap and terminal classification retain their native order.
- N=1 has zero viscous force; the row proof does not divide by a zero nonself degree. For N≥2 in the real-coordinate model, every nonself Gaussian weight is strictly positive. There is no mathematical zero-degree case, although there is no global lower degree or physical-coordinate Lipschitz constant after Gaussian motion. Native raw-exponential underflow is the distinct A18-001 case.
- The count contraction uses a symmetric complete-graph comparison. The row contraction uses its degree measure; it does not conserve ordinary momentum. None of the row moment conclusions here imports an unproved global degree ratio.

### Nonlinear smoothing, compactification and eigenproblem checks

1. The actual nonlinear map is `T(z)=(1−λt²)z−tλx1+tFvisc(x1+tz,z)`. The coercive margin uses the Euclidean product norm for count normalization and maximum row norm for row normalization. Both margins are calculated from the original reference parameters: 0.9936 and 0.9876. They are not added assumptions about that instance.
2. The homotopy boundary estimate is uniform in its interpolation parameter. At consensus, derivatives of position-dependent weights multiply zero velocity differences. The remaining derivative is invertible by the displayed margin. Therefore its analytic determinant is nontrivial. Critical **source** points are null. Critical values are then null as well: the C1 map is Lipschitz on each compact ball and sends a null source set to a null set in the same dimension. This avoids importing an additional regular-value hypothesis.
3. A regular target has a compact discrete preimage, hence finitely many inverse branches. Coercivity bounds every branch relevant to the target, and the displayed derivative bound is an **upper** determinant bound. Positive source Gaussian density supplies the required lower output density. No global injectivity, everywhere-invertible derivative or global upper density is inferred.
4. The TV argument is local in noncritical inverse charts, with discarded Gaussian mass made arbitrarily small. For fixed patterns and jitters/rotations, convergence of effective inputs gives C1 convergence on compact sets. Integration of these probability laws is justified by bounded convergence. Coupled output rows are never factorized.
5. The effective compactification retains all physical information on actual states. Only dead donor features are compactified; those raw coordinates are replaced by frozen alive donor positions before viscosity evaluates them. Artificial feature-boundary, velocity-cap-boundary and status-boundary points receive no output mass under the absolutely continuous kernel, so the eigenmeasure lift does not give them physical mass.

The existing cited positive-operator theorem was checked against its primary PDF, not its name: [Lei Zhang, Theorem 1.1, PDF page 2](https://arxiv.org/pdf/1606.04377). It requires a Banach space, total cone, compact positive bounded linear operator and positive spectral radius; its strong conclusion additionally requires a nonempty cone interior and strong positivity. Here X=C(K_N) with the supremum norm; the nonnegative cone is total and solid; Q is bounded and positive; TV continuity on compact K_N gives uniform equicontinuity and Arzelà–Ascoli compactness; full support gives Qf>0 everywhere for every nonzero f≥0, and compactness gives a positive minimum; finally Q^n1≥ε_N^n gives r(Q)≥ε_N>0. These conditions are proved in the chapter. The generalized Theorems 1.2/1.3 are not being substituted for the cited result.

### QSD, entropy and uniform tails

- The left eigenmeasure of Q is the survival-conditioned QSD. The invariant law of the conservative Doob kernel is its eigenfunction-weighted law; they are not identified. The full normalized iterate is reconstructed by weighting at the start and unweighting after the entire Markov iterate.
- The independent common part contracts relative entropy, not a continuous-gradient Dirichlet form. The two bounded reweightings each cost at most 1/m_N. Chapter 15's accept/reject proof verifies this comparison. No conditional intermediate-step renormalization or joint LSI is used.
- In the tail proof, `W_x vcol` can depend on and be correlated with clone jitter. The argument bounds it pathwise by V_c before bounding the Gaussian residual; it does not assert independence. The survival calculation conditions on all jitter but restricts only the tagged jitter event. Other rows can have arbitrarily large jitters without invalidating the convex-kick envelope.
- The killed eigenmeasure identity applies first to bounded truncations. Extinction removes nonnegative mass, the eigenvalue is bounded below by the tagged survival floor, and monotone convergence gives retained dead-row exponential moments. Expected alive fraction is bounded below; concentration of the alive fraction is not proved.

### Detailed scope records

#### [A18-001] Real-coordinate versus finite-bit kernel

The native `RandomStream` is determined by seed, update, stream, row, substep and counter. Its uniforms use 52 or 23 bits, its Box–Muller outputs have finite representable support, and its QR rotation construction has finite-precision retry/error branches. With a fixed seed this is not a sequence of independent continuous Gaussian/Haar random variables. The book's inherited analytical kernel supplies those random variables; the chapter explicitly restricts its density/full-support/QSD results to that kernel. The statements are sound in that scope, but no numerical-kernel perturbation bound has been derived. Consequently these proofs cannot be counted as density or unique-QSD theorems for a bit-level native execution or its full seed/clock state. Required new assumptions for the stated analytic theorem: none. Required further work for numerical transfer: an actual comparison/error theorem, not an assertion that discretization preserves continuous density.

#### [A18-002] Physical state versus passive record/history

`(id,Γ)` gives an exact measurable bijection only when Γ is the specified deterministic function of the current physical state. A B1/B2 source record, previous-cell-volume-dependent readout or accumulated `RunHistory`/FractalSet is generally not determined by the terminal coordinates alone. Chapter 04's separate incoming-record instrument is the appropriate law for the former. A monotonically advancing update/version clock or ever-growing archive cannot acquire a stationary law from this physical-state QSD. The chapter's consistent deterministic-decoration theorem is not a theorem about the entire engine checkpoint/archive. No new premise is needed for the stated decoration theorem; broader record transfer must keep the existing instrument or explicitly retained history state.

#### [A18-003] Parameter applicability, not universal variant discharge

The positive QSD conclusion is verified for the quadratic/capped/terminal-box reference and its declared row-normalization bit. General formulas keep their algebraic margin and positive-noise requirements. They do not certify a divergent force profile, unbounded conservative domain, adaptive covariance, geometric/Boris force, history donor pool or geometry feedback. N-independent **marginal** tails do not discharge stationary chaos, joint concentration, uniform entropy/LSI, metric consistency or a physical gauge generator. The remaining spectral eigenfunction ratio is exact kernel data rather than a primitive closed bound. These limits are already displayed and must remain present in the completion ledger.

#### [A18-004] Mathematical positivity versus floating evaluation

Direct substitution of the reference into the survival certificate gives approximately log η=−26787.56 for the displayed proof choices. Its exact expression is positive but `exp(log η)` underflows in ordinary double precision. Validation of a numerical certificate must use logarithmic quantities and must not test this primitive expression by requiring its floating value to be strictly positive. The proof itself uses no rounded lower bound and has no such defect.

### Adversarial disposition

No active Critical, Major, Moderate or Minor defect was found in the current stated Chapter 18 results. This is a bounded-scope mathematical audit, not confirmation that the whole Volume II chain is complete. The numerical-law comparison, primitive eigenfunction bounds and population-uniform stationary/geometric/operator obligations remain unproved. All historical entries above are preserved; no additional landscape premise or algorithmic modification is proposed as a remedy.
