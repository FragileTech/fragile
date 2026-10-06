# Mathematical Review: 17_geometric_gas.md

## Metadata

- Reviewed file: [17_geometric_gas.md](/home/guillem/fragiletech/fragile/docs/source/2_fractal_gas/convergence_program/17_geometric_gas.md).
- Review date: 2026-10-02.
- Reviewer: Codex, adversarial exact-algorithm review.
- Scope: Complete chapter, with coefficient, law, operator and geometric-identity provenance checked. External classical results are not re-proved or newly imported here.
- Framework anchors: `def-gg-sde`, `axiom-gg-ueph`, `lem-gg-viscous-dissipative`, `thm-gg-foster-lyapunov-drift`, `thm-gg-lsi-main`, `prop-gg-entropy-fisher-gap`, `appx-discrete-raychaudhuri`.

## Executive summary

- Critical: 0 newly identified local errors.
- Major: 0 newly identified local errors.
- Moderate: 0 newly identified local errors.
- Minor: 0 newly identified local errors.
- Notes: 3 native application boundaries.
- Primary theme: Coefficient regularity, continuous diffusion estimates and geometry identities do not establish the complete discrete geometric gas characterization without their remaining same-operator inputs.

## Error log

| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| N-001 | `def-gg-sde`, drift/recurrence results | Note | Algorithm mismatch if transferred | A continuous jump-SDE is not the fixed-step sampled algorithm. |
| N-002 | `thm-gg-lsi-main`, `prop-gg-entropy-fisher-gap` | Note | Proof gap in a native application | The actual joint-law LSI and complete derivative premise remain to be derived. |
| N-003 | Holonomy and differentiated cell expansion | Note | Scope restriction | A supplied smooth metric/congruence is not the reconstructed native geometry or an Einstein equation. |

## Detailed findings

### [N-001] Preserve the named operator and weighted alignment norm

- The chapter explicitly distinguishes its continuous jump realization from the actual discrete update. Its perturbation theorem correctly requires an actual kernel difference for a discrete transfer.
- Positive Hessian shift, bounded derivatives, a degree ratio, bounded adaptive force, jump nonexplosion and minorization are displayed inputs; not all have been established on the unbounded native state space. A finite moment bound cannot replace a global degree-ratio bound for a decaying Gaussian kernel.
- Verified local calculations retain degree-weighted alignment dissipation, the moving-degree caveat, velocity-only diffusion traces, and the full Stratonovich correction. They do not falsely infer conservation of unweighted momentum or position diffusion from velocity noise.
- Fix guidance: Use the exact configured discrete covariance, force, cloning map, masks and ordering. Prove the corresponding discrete difference, or derive an identified limit before applying a continuous generator calculation. Do not change metric regularization or viscosity normalization to satisfy a sufficient estimate.
- Required new assumptions: None authorized; unverified displayed inputs stay unresolved.

### [N-002] Geometric form comparison does not identify the joint law

- Location: source lines 719–785.
- The static theorem presupposes an identified invariant law/QSD satisfying one of Chapter 15's structural LSI criteria. The dynamical proposition presupposes the complete normalized derivative with positive margin.
- Bounded matrix coefficients, velocity ellipticity, Gaussian marginal tails and the frozen conditional OU law are not proofs of these premises. The chapter explicitly retains weighted Fisher and discrete-status obligations.
- Fix direction: Derive the actual marked law's inequality and the complete numerical or identified continuous entropy balance, retaining killing, cloning and coefficient derivatives. If no estimate closes, do not count the conditional proposition as a native discharge.
- Validation: Match law, state space, gradient/form, clock and operator; propagate every remaining condition to concentration and stationary-limit applications.

### [N-003] Geometric identities are not emergent geometry or field equations

- Holonomy and Raychaudhuri concern the supplied smooth metric and geodesic congruence. The discrete cell result needs differentiated volume consistency, including normalized Voronoi flux and its time derivative. Shape regularity alone is explicitly insufficient.
- Fix direction: Derive the native reconstruction's flux/derivative bounds and metric limit from the same records. Do not replace Voronoi cells by material cells, a Riemannian metric by a Lorentzian one, or an identity by an Einstein evolution law without the corresponding identification proof.
- Required new assumptions: None. These conditions remain derivations to complete.

## Scope restrictions and clarifications

- No local computational contradiction was found in the reviewed coefficient, moment, commutator and geometric implication proofs.
- This does not certify the conditions for the discrete Latent, Geometric or Einstein–Hilbert implementations; the passive viscous readout is another distinct object.

## Proposed edits

- No proof/core source was edited. Maintain the same-operator and hypothesis distinctions in downstream summaries.

## Open questions

- Which exact configured geometric variant has all coefficient/moment inputs derived on its actual reachable state space?
- Can the complete joint-law and native cell/action limits be obtained without introducing any new assumption or substituting a different evolution?
