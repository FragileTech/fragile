# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/04_field_equations.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/04_field_equations.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-ig-free-energy: IG free energy functional and correlation kernel.
  - thm-elastic-pressure: elastic pressure formula derived from the boost perturbation.
  - thm-dispersion-relation: linearized McKean-Vlasov dispersion relation.

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 2
- Minor: 0
- Notes: 0
- Primary themes: large-deviation and Chapman-Enskog steps rely on external results; the Einstein-tensor contraction uses a d=3 formula while the chapter treats general d; pressure derivation needs explicit boundary/strain hypotheses.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | lem-ig-rate-function | Major | External dependency | Sanov/Gartner-Ellis large-deviation claim is imported without a framework permit.
| E-002 | thm-structural-correspondence ("R_{mu nu} u^mu u^nu = 4 pi G_eff (rho + 3P)") | Major | Parameter inconsistency | Formula is specific to d=3 but the chapter uses general d.
| E-003 | thm-elastic-pressure proof (boost perturbation) | Moderate | Scope restriction | Surface-tension extraction assumes fixed-strain boundary conditions and omits bulk terms without stating the subtraction.
| E-004 | thm-einstein-relation | Moderate | External dependency | Chapman-Enskog/Einstein relation is cited as standard, not derived within the framework.

## Detailed findings

### [E-001] IG rate function uses external large-deviation theorems
- Location: lem-ig-rate-function ("Sanov's theorem" and "Gartner-Ellis")
- Severity: Major
- Type: External dependency (secondary: Proof gap / omission)
- Claim (paraphrase): The empirical density large-deviation rate function is relative entropy with respect to the QSD.
- Why this is an error in the framework: The proof invokes Sanov's theorem and Gartner-Ellis, which are external results not certified by Volume 3 permits. The framework requires internal permits or explicit assumptions when importing classical large-deviation machinery.
- Impact on downstream results: The IG free energy functional and the subsequent pressure derivation depend on this rate function; the logical base is unlicensed within the framework.
- Fix guidance (step-by-step):
  1. Add a Volume 3 permit for the empirical-measure LDP under the IG cloning process, or explicitly mark the lemma as an external permit.
  2. State the precise assumptions needed (independence/exchangeability, QSD uniqueness, exponential tightness).
  3. If only a formal motivation is intended, reclassify the lemma as a heuristic rather than a theorem.
- Required new assumptions/permits (if any): A framework LDP permit for the IG cloning process.
- Framework-first proof sketch for the fix: Use exchangeability + propagation of chaos to reduce to iid sampling from the QSD, then apply a framework-licensed Sanov-type statement to the empirical measure.
- Validation plan: Check that later theorems (elastic pressure, linearization) cite the revised status of the rate function.

### [E-002] Einstein-tensor contraction uses a d=3 formula in general d
- Location: thm-structural-correspondence (precise statement)
- Severity: Major
- Type: Parameter inconsistency
- Claim (paraphrase): R_{mu nu} u^mu u^nu = 4 pi G_eff (rho_eff + 3 P_eff) in d+1 dimensions.
- Why this is an error in the framework: The coefficient "3" is specific to d=3 spatial dimensions. For general d, the contraction yields R_{mu nu} u^mu u^nu = 4 pi G_eff (rho_eff + d P_eff) under the stated perfect-fluid form. The chapter uses general d elsewhere.
- Impact on downstream results: The field-equation correspondence and Lambda_eff interpretation are dimensionally inconsistent when d != 3.
- Fix guidance (step-by-step):
  1. Replace "3" with "d" and add a line explaining the dependence on spatial dimension.
  2. If the intent is d=3 only, state this restriction explicitly in the theorem header.
  3. Check all later uses of this contraction for consistency with the revised coefficient.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Contract Einstein's equation with u^mu u^nu, use T_{mu nu} = (rho + P) u_mu u_nu + P g_{mu nu} and T = -rho + d P.
- Validation plan: Verify that the corrected formula reproduces the standard 4D (d=3) case.

### [E-003] Elastic pressure extraction needs explicit boundary/strain hypotheses
- Location: thm-elastic-pressure proof (boost perturbation, Steps 1-3)
- Severity: Moderate
- Type: Scope restriction (secondary: Proof gap / omission)
- Claim (paraphrase): The L^{-2} term in Delta F yields surface-tension pressure, with bulk terms treated as volume-independent.
- Why this is an error in the framework: The derivation assumes a fixed-strain (fixed kappa) perturbation and effectively subtracts bulk contributions without stating the boundary conditions or the thermodynamic ensemble. The omission of explicit subtraction or fixed-boundary conditions makes the pressure extraction ambiguous.
- Impact on downstream results: The elastic pressure formula is plausible but not fully specified; the interpretation as surface tension requires an explicit ensemble choice.
- Fix guidance (step-by-step):
  1. State the thermodynamic ensemble (fixed strain vs fixed volume) and boundary conditions.
  2. Explicitly subtract the bulk free-energy contribution before differentiating to define surface tension.
  3. Add a short remark clarifying which terms are absorbed into a bulk modulus and which define the pressure.
- Required new assumptions/permits (if any): A framework definition of surface tension for the IG functional under the chosen ensemble.
- Framework-first proof sketch for the fix: Decompose Delta F into bulk (O(V)) and boundary (O(L^{d-2})) parts under the boost; define pressure as the derivative of the boundary part with respect to area or L.
- Validation plan: Re-derive Pi_elastic for a simple 1D or 2D box and check the scaling with L.

### [E-004] Einstein relation relies on external Chapman-Enskog theory
- Location: thm-einstein-relation
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): D_eff = v_T^2 / gamma is obtained from a standard Chapman-Enskog expansion.
- Why this is an error in the framework: The Chapman-Enskog expansion and fluctuation-dissipation relation are external results. The framework requires a permit or a derivation if used as a theorem.
- Impact on downstream results: The numerical value of D_eff used in the dispersion relation and pressure regime analysis is not internally justified.
- Fix guidance (step-by-step):
  1. Add a Volume 3 permit for the kinetic reduction or mark the relation as an external assumption.
  2. State the scaling regime explicitly (high-friction, time-scale separation).
- Required new assumptions/permits (if any): A permit for the high-friction kinetic limit and Einstein relation.
- Framework-first proof sketch for the fix: Use a Hilbert expansion of the kinetic equation and identify the leading-order Maxwellian plus first-order flux closure.
- Validation plan: Verify the relation by comparing with the Ornstein-Uhlenbeck marginal in a simplified model.

## Scope restrictions and clarifications
- The linearized QSD analysis assumes spatially uniform fitness and a translation-invariant cloning kernel; these conditions should be repeated near thm-dispersion-relation and thm-qsd-stability.

## Proposed edits (optional)
- Replace the 3 in the Einstein-tensor contraction with d (or restrict the theorem to d=3).
- Add an explicit ensemble statement in the elastic-pressure derivation.
- Mark the large-deviation and Chapman-Enskog steps as external permits if not derived internally.

## Open questions
- Is there a Volume 3 permit for the IG cloning large-deviation principle, or should lem-ig-rate-function be labeled heuristic?
- Should D_eff be treated as a parameter rather than derived, unless a kinetic-theory appendix is added?