# Completion checklist for the Euclidean Gas proof

This checklist covers the recovered mean-field, propagation-of-chaos,
stationary, entropy and quantitative convergence claims. It is an engineering
acceptance document; it does not add hypotheses to lecture theorems or change
the algorithm. The geometric cluster strategy remains the proof strategy.

The completion standard is a closed, independently reviewed implication chain:
every retained conclusion follows from the specified transition and the
theorem's actual hypotheses, with every application verified. No checklist
guarantees that human error is impossible. Machine checking the completed
argument can provide additional assurance.

## Fixed proof strategy

The argument to repair is the one in commit
`b416d3a98112b360e6ae7f6655fb47497916cfcb`. Chapter 03 combines the geometric
cloning estimates with kinetics using
`V_total = V_W + c_V*(V_Var,x + lambda_v*V_Var,v) + c_B*W_b`.
Its cloning contribution to `V_W` is bounded expansion; additive offsets
remain in every affine drift estimate. Chapter 06's principal total-variation
argument instead uses its stated internal-variance, velocity-barycenter and
boundary functional. Its proof does not require pairwise Wasserstein
contraction. The entropy route must likewise retain its own original reference
law and functional.

A diagnostic counterexample for one-step structural contraction is not a new
completion gate. Do not require zero offsets, negativity of every component,
or a negative-drift region for each bounded component separately. Repair the
actual input needed by the combined theorem, then check that theorem's rates,
offsets, survival normalization and mixing hypotheses. Preserve the geometric
cluster strategy and the full algorithm throughout.

## Proven inputs to preserve

These ingredients have implementations and proofs in the chapters; their
stated domains still matter. They require final integration review, not a
replacement proof merely because the stationary argument is unfinished.

- The fixed-step marked population map and the complete stage schedule.
- Permutation equivariance, canonical one-step consistency and fixed-horizon
  propagation of chaos, with their stated initialization and moment conditions.
- Conditional bounded-observable bias and variance estimates in their stated
  alive-mass regime.
- Exact centered displacement, signed donor-flux identities and cluster
  fitness-gap bounds, retaining sampled fitness and both flow directions.
- Canonical cloning barycenter variance of order `1/N` for every nonempty
  alive pool, and exact full-slot cloning momentum conservation.
- Component-average cancellation for changed revival donors when alive
  backbones agree.
- Bounded marked-transport smoothing for actual BAOAB/capping under its stated
  global force and timestep bounds, verified for quadratic and Rastrigin forces.
- Canonical fixed-point existence, the stated finite-population QSD results,
  stationary tightness/survival inputs and invariant-mixture identification.
- Exact survival-normalized entropy identities and finite-N entropy decay.

Their existence does not automatically verify the wider original regime or
the stationary application. The gates below separate those questions.

## 1. Freeze the theorem and transition ledger

**Status: final audit and scope reconciliation.**

- [ ] List every retained main theorem, its original hypotheses, exact
  conclusion, rate, state space and dependency labels.
- [ ] Identify whether its object is a finite killed chain, its QSD, a
  conservative invariant law, the discrete nonlinear map, or a continuous
  limit. Record each change of object as a proved implication.
- [ ] Match the transition to weighted measurement and donor sampling,
  retained fitness, frozen gates, order-one acceptance, scheduled revival,
  frozen copying/jitter, shared component rotations, BAOAB, position noise,
  capping and terminal boundaries. Include retained dead coordinates.
- [ ] Distinguish full-slot and alive-only quantities, and the singleton,
  empty-alive-pool and extinction cases.
- [ ] Certify the comparison used by the cluster geometry against the actual
  sampling metric. The canonical implementation compares separately squashed
  features; physical-distance arguments need a proved comparison wherever
  they are used. Preserve the cluster construction rather than substituting
  another sampling process.

**Pass criterion:** every main theorem names the same specified process as
its implementation; every prerequisite has an identified proof, and no
conditional application is marked complete before its conditions are proved.

## 2. Discharge the Keystone target estimates

**Status: geometric coverage and measurement probabilities proved; structural
interfaces explicit.**

Chapter 03 now proves (3.CC1)--(3.CC14) for the stated bounded alive-domain,
active-diversity regime. Every geometric cluster contributes, with no assumed
coverage fraction. The actual reward, powered rescaling and sampled-fitness
normalization determine the positive constants. Every population size has an
explicit N^-2 self-exclusion correction; the stronger linear bound has a
derived population threshold. Constants are uniform in N, though potentially
very conservative.

- [x] Derive actual target-wise averaged cloning probabilities and the
  error-weighted capture bound, including small and invalid geometric clusters.
- [x] Average favorable measurement events under the actual independent
  companion laws while retaining shared empirical fitness normalization.
- [x] Keep velocity discrepancy, alive normalization and unmatched-label
  contributions explicit when converting positional pressure to structural error.
- [x] Verify reward/rescaling constants for the actual smooth reward and
  active positive diversity exponent, with canonical parameters as a specialization.

**Pass criterion for coverage:** no unknown uncovered-error mass or assumed
favorable cluster fraction remains. This does not assert that positive pressure
has the dissipative sign required by the next gate.

## 3. Assemble the original collective drift with its offsets

**Status: exact cloning balances and the original bounded-expansion inputs
are available; retain the original combined-functional application.**

The exact measurement-averaged donor flux and shared-component energy balance
feed the same weighted Lyapunov functional used in the recovered argument.
Cloning may increase `V_W` and the spread of a concentrated cloud. The original
proof explicitly permits this through its finite additive offsets. The N4
structural-expansion calculation and its near-maximum variant test stronger
cloning-only claims; they are not obstructions to the original affine
combination and are not prerequisites for completing it.

- [x] Average the actual retained measurement law jointly with all signed
  geometric and fitness quantities.
- [x] Retain both donor-flow directions, moving barycenters, within-cluster
  remainders, revival and shared-component velocity contributions.
- [x] Derive the bounded inter-swarm expansion needed by the original proof
  from frozen eligible sources and full-slot component energy conservation.
- [x] Supply the affine positional reset bound with its actual Gaussian
  offset, and keep the original velocity weight in the combined functional.
- [ ] Complete the remaining kinetic/boundary application estimates for the
  original combined functional and original parameter regime, preserving
  the actual stage order and survival law.

**Pass criterion:** establish the original complete affine drift with verified
constants and its original mixing/entropy dependencies. An additive offset
covering a bounded component is admissible; the full Lyapunov argument and its
mixing theorem determine convergence. No stronger one-step structural
contraction requirement is introduced.

## 4. Control changes of accepted alive backbones

**Status: open long-time coupling estimate.**

- [ ] Couple the actual changes in alive measurement, donor and gate
  innovations, including ties and clipped acceptance.
- [ ] Handle merging and splitting alive backbones, their incoming cloners,
  altered component means and shared rotations with the correct marginal laws.
- [ ] Preserve component-average and geometric-cluster cancellations when
  summing output errors. Use the proved revival-leaf bound for leaf changes.
- [ ] Bound the relevant conditional first/second moments of the physical
  output error in the norm needed by the kinetic argument.

**Pass criterion:** an N-uniform quantitative interface from entering errors
to the actual post-collision errors needed by the long-time proof. Existing
component exploration estimates already serve one-step consistency; their
constants cannot simply be called contraction coefficients.

## 5. Close alive-law and survival normalization

**Status: open interface estimate; exact identities and survival inputs exist.**

- [ ] Keep normalization of the eligible donor law separate from conditioning
  the whole swarm on nonextinction.
- [ ] Control both numerator and denominator perturbations, and the
  low-alive-count events outside the main estimate, using derived probabilities
  and tails rather than an imposed alive-fraction floor.
- [ ] Carry every dependence on noise, timestep, cap, jitter, selection and
  boundary parameters. The improving kinetic factor `1/B` is not sufficient
  if another constant grows with `B`.
- [ ] Apply the exact QSD identity `nu_N Q_N = alpha_N nu_N`, retaining its
  survival tilt and normalization in variance and entropy calculations.

**Pass criterion:** the combined normalized estimates have explicit finite
constants and the required N-uniform margin throughout a verified nonempty
parameter regime.

## 6. Assemble the complete discrete dynamical closure

**Status: open central synthesis.**

- [ ] Specify one complete metric or functional and an invariant class for
  the actual full map. Connect the physical errors entering kinetic smoothing
  to the marked-state errors it produces.
- [ ] Combine the cluster and kinetic estimates in the actual stage order.
  Use a proved block of updates if one step is insufficient.
- [ ] Establish the required attraction or mixing through a complete route:
  contraction in an appropriate metric, or collective drift plus a verified
  local coupling/entropy mechanism. Global strict squared-distance contraction
  is not mandatory.
- [ ] Demonstrate that the constants close for the original claimed regime,
  including its strong-diffusion statement. Give an explicit parameter bound
  or a rigorous existence argument for the regime; do not just state that
  diffusion is sufficiently large.

**Pass criterion:** a theorem with its application fully discharged, not
an implication whose contraction or attraction premise remains assumed.

## 7. Complete stationary concentration and identification

**Status: open central conclusion; limit-identification machinery exists.**

- [ ] Complete at least one sufficient route: global attraction of the actual
  population map, or direct concentration of its finite-population QSD
  empirical laws together with uniqueness/identification of the limiting law.
- [ ] For the variance route, close the inherited term involving
  `Var_nu_N[F_h(L_N) phi]` in the exact QSD variance budget. Fresh one-step
  noise of order `1/N` is already controlled; inherited randomness is separate.
- [ ] Cover a convergence-determining family of observables, including status
  information. Barycenter concentration alone is not that criterion.
- [ ] Show every stationary subsequential empirical limit is the same
  deterministic fixed point. Exclude invariant mixtures and periodic-law
  mixtures through the chosen proof, not through fixed-point uniqueness alone.
- [ ] Derive stationary fixed-marginal chaos using the proved exchangeability
  and empirical-to-marginal argument.

**Pass criterion:** prove `Lambda_N -> delta_mu_*` and
`nu_N^(ell) -> mu_*^(tensor ell)` for every fixed `ell`, with every use of
concentration or attraction established earlier in the dependency chain.

## 8. Recover the claimed entropy and functional-inequality results

**Status: open quantitative closure; exact identities exist.**

- [ ] Name the reference law at every step: actual QSD, survivor-tilted law,
  nonlinear population law or conditioned product law.
- [ ] Derive and bound the signed interaction and survival contribution
  against the backward entropy loss in the full-step entropy identity.
  Establish the centering and cancellations responsible for population scaling.
- [ ] Track total entropy `H_N` separately from `H_N/N`. Product-relative
  `H_N=o(N)` can support qualitative concentration; the advertised `1/N`
  observable rate through this route needs an appropriate bounded total-entropy
  estimate. QSD-relative decay supplies neither estimate automatically.
- [ ] Where a joint LSI/Poincare or hypocoercive inequality remains a claimed
  theorem, prove it for the actual joint marked law and compatible form,
  including changes of status and shared-rotation correlations.
- [ ] At the QSD, control the remaining quadratic coefficient
  `nu_N(g^2) - Var_nu_N(A_N g)` with the required N-uniform margin. The proved
  first-order cancellation does not establish this coercivity, and local
  quadratic coercivity must still be connected to the claimed global entropy
  inequality. Continuous inequalities within fixed-status strata alone do
  not control entropy between statuses.
- [ ] Derive kernel regularity and derivative bounds on the physical regions
  actually visited; include boundary and survival terms. Complete the
  square-root-term cancellation or its valid combined estimate.
  Use direct discrete or justified weak/piecewise estimates where clipped
  acceptance prevents the classical differentiation being invoked.
- [ ] Track both the decay rate and its prefactor. Finite-N whole-swarm
  minorization and Doob reweighting do not currently provide N-uniform
  constants; derive the required uniformity through the collective mechanism.

**Pass criterion:** every retained entropy/LSI/hypocoercive rate follows from
proved estimates with the stated N-dependence. An alternative route can close
stationary chaos, but does not by itself prove an independently advertised LSI.

## 9. Verify the full original analytic regime

**Status: open where current verification is only configuration-specific.**

- [ ] Check the original potential, reward, regularization and geometry class,
  not only the quadratic demonstration.
- [ ] For unbounded physical space, derive the required uniform-in-time and
  stationary moments/tails from the actual confinement. Fixed-horizon moment
  propagation and bounded algorithmic features are not substitutes.
- [ ] Verify completeness, self-mapping, positivity and normalization for
  the stationary construction in its selected topology.
- [ ] Derive any force bound needed beyond the valid domain from the actual
  configured force, or prove the required localized estimate with actual tail
  control. Do not extend or alter the force to fit the proof.
- [ ] If a retained statement is false, supply its explicit contradiction
  and prove the corrected behavior for the same algorithm. This applies, for
  example, to the unrestricted empirical `W2^2 <= C/N` rate in dimension
  greater than two, which conflicts with the established quantization bound.

**Pass criterion:** the general theorem and each named example satisfy their
actual hypotheses; no false or vacuous applicability claim survives.

## 10. Assemble quantitative chaos and approximation rates

**Status: assembly after the analytic closure, plus final rate audit.**

- [ ] Preserve the proved one-step conditional bias/variance and two-root
  consistency estimates, with their metrics, test classes and moment conditions.
- [ ] Iterate them using the stability estimate actually proved. Distinguish
  fixed horizons, arbitrary long times and stationarity.
- [ ] Keep conditional bias, conditional variance, finite-population error,
  stationary bias, mixing error and numerical integration error separate.
- [ ] Audit diagonal covariance terms, fixed-marginal sampling factors, total
  versus per-particle normalization and observable versus empirical-Wasserstein
  rates. Prove the tails required for each unbounded observable.

**Pass criterion:** each advertised exponent and prefactor has an explicit
derivation for its stated quantity, uniformity regime and initialization.

## 11. Identify any retained continuous-time or timestep claim

**Status: separate gate for such claims; not a prerequisite for a purely
fixed-step theorem.**

- [ ] Derive it from repeated applications of the same `F_h`, retaining
  order-one acceptance, immediate revival, component collisions and repeated
  capping at their declared schedules.
- [ ] Prove tightness, identify the limit and control the accumulated errors
  over the growing number of updates required by the limit.
- [ ] Establish the order of the complete composition. Second-order BAOAB
  alone does not identify the order of cloning plus kinetics plus boundaries.

**Pass criterion:** a continuous equation or timestep rate is called an
algorithmic theorem only after its convergence from the actual transition
has been proved. No Poissonization or changed acceptance scaling is used to
complete this gate.

## 12. Independent review, reproducible verification and sign-off

**Status: final gate; substantial operator tests already exist.**

- [ ] A reviewer independently reconstructs each central estimate and checks
  every cited theorem's hypotheses. Record specific equations and review
  outcomes; approval of prose is not mathematical review.
- [ ] Trace each main theorem backward through an acyclic dependency ledger.
  In particular, stationary chaos must not establish the concentration or
  functional inequality that is then used to prove stationary chaos.
- [ ] Complete exact enumeration/operator checks for the affected formulas,
  including shared donors, backbone merges/splits, giant revival components,
  frozen copies, momentum/energy, extinction and transported innovations.
- [ ] Revalidate affected predictions with actual Rust runs on
  `N=16,32,64,128,256`, updates `1,4,16,64`, 128 independent seeds, and a
  disjoint validation batch. Use nonzero selection, killing/revival and
  controls that expose common randomness. Report uncertainty within each
  independent-run cohort. Do not rerun unaffected studies merely for volume.
- [ ] Run relevant Rust/lint/compiled-engine/browser checks when their code
  changes, rebuild CPU/WebGPU assets after production changes, and build the
  documentation and proof exports. Resolve proof-chain reference errors
  without adding assumptions throughout downstream applications.
- [ ] Preserve complete formal proofs in Expert Mode and use one Feynman
  educator per affected chapter after mathematical review.
- [ ] Sign off every retained main theorem only when all its required gates
  are closed. Keep scope-specific failures or contradictions explicit.

**Pass criterion:** the proof, implementation, parameter regime, dependency
ledger and verification artifacts agree, with no assumed central estimate.
Formalizing the central chain in a proof assistant is an additional route
to machine-checked assurance; it does not replace the algorithm-identification
or implementation checks.

## Work order

Complete gates 1–5 together, since the metric and actual regime determine the
constants needed by the cluster argument. Then close gates 6–7. Develop gate
8 in parallel where it contributes to that closure, and finish every retained
entropy/functional-inequality claim. Gate 9 must be checked throughout rather
than deferred to a change of assumptions at the end. Assemble gate 10, finish
gate 11 only for retained limit claims, and perform gate 12 against the final
mathematics. Historical comparisons remain in `MEAN_FIELD_AUDIT.md`.
