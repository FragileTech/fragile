# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/06_cosmology.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/06_cosmology.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-holographic-boundary-vacuum: Lambda_holo from IG pressure at horizons.
  - def-bulk-qsd-vacuum: Lambda_bulk at QSD equilibrium.
  - def-effective-exploration-vacuum: Lambda_eff from non-equilibrium exploration currents.

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: bulk Lambda=0 proof assumes Einstein equations as exact; Friedmann-derived Lambda_eff formula is dimensionally inconsistent; exploration-current coupling is ad hoc; observational Lambda values are external to the framework.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | thm-vanishing-bulk-vacuum (proof Steps 2-4) | Major | Proof gap / external dependency | Uses Einstein equations as exact despite only a structural correspondence in Volume 3. |
| E-002 | prop-geometric-distinction (Lambda_eff formula from Friedmann) | Major | Parameter inconsistency | Formula does not match the d+1 dimensional Friedmann equations stated elsewhere. |
| E-003 | def-effective-exploration-vacuum (source term J^mu) | Moderate | Algorithm mismatch | The J^mu u^nu coupling is introduced without derivation from LFG dynamics. |
| E-004 | prop-observed-lambda and observational discussion | Moderate | External dependency | Observational value of Lambda_obs and cosmology data are outside the framework. |

## Detailed findings

### [E-001] Bulk Lambda=0 proof assumes exact Einstein equations
- Location: thm-vanishing-bulk-vacuum (proof Steps 2-4)
- Severity: Major
- Type: Proof gap / external dependency
- Claim (paraphrase): At QSD, Lambda_bulk = 0 follows from Einstein equations with vanishing source.
- Why this is an error in the framework: In Volume 3, the field equations are presented as a structural correspondence (not a proved dynamical law). The proof implicitly assumes exact Einstein equations and stress-energy conservation as axioms, which exceed the stated framework.
- Impact on downstream results: The theorem's status should be downgraded to a conditional statement: "if the structural correspondence is upgraded to an exact field equation, then Lambda_bulk = 0." As written, it is over-asserted.
- Fix guidance (step-by-step):
  1. Rephrase the theorem as conditional on the exact field equation assumption.
  2. State the required hypotheses explicitly (closed system, no currents, exact conservation).
  3. If an exact field equation is intended, add a Volume 3 permit that elevates the correspondence to an axiom.
- Required new assumptions/permits (if any): A permit elevating the structural correspondence to an exact field equation.
- Framework-first proof sketch for the fix: Under the assumed field equation and J^mu = 0, show that the stationary homogeneous solution implies Lambda_bulk = 0; otherwise treat it as a derived condition rather than a theorem.
- Validation plan: Ensure later claims about Lambda_eff clearly distinguish conditional vs proven statements.

### [E-002] Friedmann-based Lambda_eff formula is dimensionally inconsistent
- Location: prop-geometric-distinction (Lambda_eff formula)
- Severity: Major
- Type: Parameter inconsistency
- Claim (paraphrase): Lambda_eff = d(d-1)/2 H^2 - (d-1) a¨/a - 8 pi G_N/(d-1) rho_matter.
- Why this is an error in the framework: The formula does not match the standard d+1 dimensional Friedmann equations (even at the level of coefficients and signs). If d is spatial dimension, the relation between H^2, a¨/a, rho, and Lambda has different prefactors; the given expression mixes acceleration and energy-density terms without the correct normalization.
- Impact on downstream results: The "geometric distinction" section may mislead readers about how Lambda_eff is inferred from expansion dynamics.
- Fix guidance (step-by-step):
  1. Derive the correct d+1 dimensional Friedmann equations within the chosen convention and solve for Lambda explicitly.
  2. Replace the formula with the correct coefficient structure, or mark it as schematic.
  3. Ensure consistency with the Raychaudhuri conventions used earlier.
- Required new assumptions/permits (if any): A permit defining the FRW reduction in the LFG framework or an explicit statement that the formula is schematic.
- Framework-first proof sketch for the fix: Starting from Einstein equations in d+1 dimensions with FRW metric, derive H^2 and a¨/a equations, then solve for Lambda.
- Validation plan: Check that the revised formula reproduces the 4D (d=3) case and matches the sign conventions in sec-raychaudhuri.

### [E-003] Exploration-current coupling lacks derivation
- Location: def-effective-exploration-vacuum (field equation with J^mu)
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Proof gap / omission)
- Claim (paraphrase): Non-equilibrium exploration currents enter the field equation as kappa (J_mu u_nu + J_nu u_mu).
- Why this is an error in the framework: The coupling is introduced ad hoc without a derivation from the LFG stochastic dynamics or a conservation principle. It is unclear why this is the correct tensorial form, or how J^mu is computed from the algorithm.
- Impact on downstream results: The sign and magnitude of Lambda_eff are not anchored to the actual dynamics; the mechanism is qualitative rather than quantitative.
- Fix guidance (step-by-step):
  1. Define J^mu operationally from the swarm dynamics (e.g., from a balance law for density or energy).
  2. Derive the tensor structure by enforcing covariance and conservation or by variational principles.
  3. If the form is heuristic, label it as an ansatz and isolate downstream claims as conjectural.
- Required new assumptions/permits (if any): A permit for the non-equilibrium stress-energy correction term.
- Framework-first proof sketch for the fix: Start from a coarse-grained continuity equation for rho and energy, identify source terms, and build the minimal symmetric rank-2 tensor compatible with u^mu and J^mu.
- Validation plan: Compare the derived tensor against simulations or simpler mean-field models.

### [E-004] Observational Lambda values are external to the framework
- Location: prop-observed-lambda and observational discussion
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): Lambda_obs ~ 1.1e-52 m^-2 inferred from supernovae, CMB, and large-scale structure.
- Why this is an error in the framework: These observational values and datasets are external empirical inputs. They should be clearly labeled as outside-framework data rather than implied consequences of the LFG theory.
- Impact on downstream results: The interpretation of Lambda_eff as "observed" becomes conflated with framework predictions.
- Fix guidance (step-by-step):
  1. Explicitly label the observational values as external benchmarks.
  2. Separate empirical inputs from framework-derived quantities in the narrative.
- Required new assumptions/permits (if any): None if labeled as external inputs.
- Framework-first proof sketch for the fix: Not applicable; this is a labeling/attribution correction.
- Validation plan: Ensure no statement implies the value is derived from the framework unless a derivation is provided.

## Scope restrictions and clarifications
- All cosmological inferences are conditional on the structural correspondence being elevated to an exact field equation.

## Proposed edits (optional)
- Recast thm-vanishing-bulk-vacuum as a conditional statement.
- Correct the Friedmann-based Lambda_eff formula or mark it as schematic.
- Introduce a formal definition of J^mu tied to the algorithm.

## Open questions
- Is the intent to treat the field equations as axioms (exact) or as structural analogies only?
- Should observational Lambda values live in a separate "external data" note to avoid mixing provenance?