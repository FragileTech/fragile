# Mathematical Review: 16_continuum_discharge.md

## Metadata

- Reviewed file: [16_continuum_discharge.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/16_continuum_discharge.md).
- Review date: 2026-10-02.
- Reviewer: Codex, adversarial exact-algorithm review.
- Scope: Complete chapter and its hypothesis/algorithm provenance; no new native continuum theorem inferred.
- Framework anchors: `def-continuum-hypothesis-status`, `lem-continuum-a2-smooth-fields`, `lem-continuum-a3-qsd-sampling`, `lem-continuum-a5-kernel`, `lem-continuum-a4-mixing`, `cor-continuum-consistency-conditional`.

## Executive summary

- Critical: 0 newly identified local errors.
- Major: 0 newly identified local errors.
- Moderate: 0 newly identified local errors.
- Minor: 0 newly identified local errors.
- Notes: 3 unclosed native identifications.
- Primary theme: The chapter explicitly supplies conditional consistency and examples, not an unconditional continuum characterization of native episodes.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| N-001 | A1–A4 and their sufficient conditions | Note | Proof gap in a native application | Smooth geometry, exact sampling and shrinking-observable dependence estimates are not derived for the coupled record. |
| N-002 | Kernel construction and local bias | Note | Definition mismatch if substituted | The constructed tangent-space kernel is not automatically the existing proper-time/episode kernel. |
| N-003 | Conditional operator/action consistency | Note | Algorithm mismatch if substituted | Independent evaluation samples and an ideal estimator cannot replace the actual same-record action. |

## Detailed findings

### [N-001] All geometry, law and dependence inputs must come from the same record

- Location: lines 21–53, smooth-field lemma, sampling lemma and covariance lemma.
- The hypotheses include a specified continuum geometry, temporal/spatial regularity, positive denominators, the exact selected sampling density and a covariance bound for bandwidth-dependent summands. Fresh Gaussian kinetic noise alone does not prove all those statements.
- QSD sampling, Doob-stationary sampling and whole-episode survivor selection give different laws. Exchangeability or finite-horizon chaos does not establish the needed stationary sampling identity. An exact Gibbs density is an identification to prove, not a consequence of naming a potential.
- Downstream impact: A fixed-test consistency theorem cannot be used as a shrinking-neighborhood variance estimate; a one-time metric bound is not a differentiated spacetime metric limit.
- Framework-first fix direction: Push forward the exact marked path law through the existing estimator; derive its density/importance factors, local geometric error, derivative bounds and bandwidth-dependent covariance budget. Retain tails from proved confinement rather than compactifying the gas.
- Required new assumptions: None authorized. The chapter's displayed assumptions stay conditional until derived for the chosen run family.

### [N-002] A consistent auxiliary kernel does not characterize the configured kernel

- Location: `lem-continuum-a5-kernel`, `lem-continuum-local-bias` and `rem-continuum-downstream-conditions`.
- The tangent-space differentiated bump is a mathematical sufficient example. The finite proper-time ansatz has a separate moment/solvability calculation on its actual domain. Neither permits replacing the implemented kernel or imposing a new dynamical cutoff.
- Fix guidance: Evaluate the existing kernel's moments, density correction, neighborhood and coordinate errors with its existing parameters. If a correspondence with the ideal estimator is not proved at its singular normalization, keep the limit open.
- Validation: Verify the actual signed moments and bounds on the actual domain. An analytic observation bandwidth schedule must not be silently equated with an algorithmic interaction radius or with an independently sampled population.

### [N-003] Same-record operator/action comparison is still unproved

- Location: lines 736–814.
- The corollary expressly requires `tilde L_N f - hat L_N f -> 0` in probability. Its action example uses new independent evaluation samples, independent of the estimator samples, and labels that construction a sufficient mathematical example.
- The native graph and action use the same interacting records. Their joint weighted error, quadrature, metric and causal-neighborhood comparison have not been supplied by the example.
- Fix guidance: Prove the native comparison and joint integration estimate using the existing same-record observations. Do not introduce the independent sample as an algorithm stage or claim that its proof discharges the native action.
- Required new assumptions: None. Keep this as an open derivation, rather than assuming decorrelation.
- Validation: Control errors after multiplication by the estimator normalization; retain particle/time correlations and prove first-variation convergence separately when required downstream.

## Scope restrictions and clarifications

- No contradiction was found in the conditional local estimator calculation itself.
- Its sufficient constructions are legitimately labeled examples, but are not completed proofs for the algorithm requested by the user.
- Source-parameter smoothness, spatial smoothness, metric convergence and physical-time reconstruction are different statements.

## Proposed edits

- No proof or algorithm source was edited. Downstream closure registers must retain all A1–A6 obligations and the native comparison error.

## Open questions

- Can the actual correlated graph/geometry/action satisfy one common scaling with all error constants derived from existing parameters?
- Which native kernel/domain moment calculation realizes the stated limiting operator without changing the algorithm?
