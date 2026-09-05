# Mathematical Review: docs/source/3_fractal_gas/3_fitness_manifold/05_holography.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/3_fitness_manifold/05_holography.md
- Review date: 2026-01-29
- Reviewer: Codex (GPT-5)
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - def-ig-entanglement-entropy: min-cut definition of IG entropy.
  - def-nonlocal-perimeter: nonlocal perimeter functional for IG correlations.
  - thm-informational-area-law: claimed proportionality between IG entropy and CST boundary area.

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 1
- Minor: 0
- Notes: 0
- Primary themes: Gamma-convergence limit is mis-scaled; holographic pressure equality conflicts with earlier definitions; QSD-as-Gibbs and Unruh/Hawking claims rely on external results; RT/minimal-surface identification needs explicit geometric assumptions.

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | thm-gamma-convergence | Major | Invalid inference | Kernel scaling is missing; the claimed Gamma-limit constant c0 = C0 is incorrect. |
| E-002 | thm-holographic-pressure (and its claim of equality with Pi_elastic) | Major | Parameter inconsistency | a0 is independent of epsilon_c but the pressure matching requires epsilon_c dependence. |
| E-003 | thm-qsd-gibbs, prop-unruh-hawking-connection | Major | External dependency | Gibbs/Unruh claims are asserted without framework permits. |
| E-004 | thm-ryu-takayanagi and Informational Area Law interpretation | Moderate | Scope restriction | Minimal-surface identification from antichains is assumed without geometric hypotheses. |

## Detailed findings

### [E-001] Gamma-convergence statement is mis-scaled
- Location: thm-gamma-convergence (Gamma-convergence to local perimeter)
- Severity: Major
- Type: Invalid inference (secondary: Computational error)
- Claim (paraphrase): As epsilon -> 0, P_epsilon Gamma-converges to c0 * int_{partial A} rho^2 dSigma with c0 = C0.
- Why this is an error in the framework: The kernel K_epsilon is not scaled by epsilon in amplitude. With K_epsilon(z,z') = C0 exp(-|z-z'|^2/(2 epsilon_c^2)), the double integral scales like epsilon_c^2 and tends to 0 as epsilon_c -> 0 unless a 1/epsilon scaling is introduced. The constant cannot be C0 without an explicit rescaling; the limit is zero as written.
- Impact on downstream results: The area-law derivation based on Gamma-convergence is invalid in the stated scaling; the proportionality constant alpha is not justified.
- Fix guidance (step-by-step):
  1. Introduce the standard nonlocal-perimeter scaling: K_epsilon = (1/epsilon^{d+1}) K((x-y)/epsilon), or define a rescaled functional P_epsilon / epsilon^2 with a finite limit.
  2. Recompute the constant c0 using the scaled kernel integral.
  3. Update the statement and proof sketch to match the corrected scaling.
- Required new assumptions/permits (if any): A framework lemma specifying the scaled-kernel Gamma-convergence limit on Riemannian manifolds.
- Framework-first proof sketch for the fix: With the scaled kernel, use tubular coordinates and the standard limit of the nonlocal perimeter to the local perimeter; the constant c0 equals the first moment of K in the normal direction.
- Validation plan: Check the limit in Euclidean space for a half-space and verify c0 numerically.

### [E-002] Holographic pressure matching conflicts with a0 definition
- Location: thm-holographic-pressure (and its "agreement with elastic pressure" claim)
- Severity: Major
- Type: Parameter inconsistency
- Claim (paraphrase): Pi_IG(L) = -C0 rho_0^2 (2 pi)^{d/2} epsilon_c^{d+2} / (8 d L^2) and equals Pi_elastic from thm-elastic-pressure.
- Why this is an error in the framework: The CST area quantum a0 is defined as (Vol/N)^{(d-1)/d} and is independent of epsilon_c, but the matching formula requires epsilon_c^{d+2} scaling. Without redefining alpha or a0 to carry epsilon_c dependence, the equality Pi_IG = Pi_elastic does not follow. The factor 1/(8 d) also conflicts with the 1/4 in thm-elastic-pressure unless additional constants are inserted.
- Impact on downstream results: The central consistency check between holographic and elastic pressure is unsupported; the AdS-radius formula derived from it is unstable.
- Fix guidance (step-by-step):
  1. Make explicit which constant (alpha or a0) carries epsilon_c dependence and calibrate it to match Pi_elastic.
  2. Re-derive the pressure from the entropy variation with the corrected constants and show the factor match.
  3. If the match is heuristic, mark it as such and avoid using it as a theorem.
- Required new assumptions/permits (if any): A framework calibration lemma linking alpha, a0, and the IG kernel parameters.
- Framework-first proof sketch for the fix: Compute dS/dL using the CST area definition and compare with the IG pressure from the jump Hamiltonian; solve for alpha in terms of kernel parameters and record it as a calibration.
- Validation plan: Verify that the calibrated constants also preserve the area-law normalization in thm-informational-area-law.

### [E-003] QSD Gibbs state and Unruh/Hawking temperature are external claims
- Location: thm-qsd-gibbs, prop-unruh-hawking-connection
- Severity: Major
- Type: External dependency
- Claim (paraphrase): The QSD has Gibbs form with effective Hamiltonian including H_jump, and T_eff equals surface gravity at horizons.
- Why this is an error in the framework: These statements rely on external equilibrium statistical mechanics and horizon thermodynamics (Unruh/Hawking). The framework does not provide permits establishing detailed balance or KMS conditions for the IG process.
- Impact on downstream results: The thermodynamic interpretation of T_eff and the holographic temperature claims are not internally justified.
- Fix guidance (step-by-step):
  1. Downgrade the statements to conjectures or physically motivated remarks.
  2. If intended as theorems, add permits establishing detailed balance or a KMS-like condition for the QSD.
  3. State explicit assumptions (stationarity, horizon generator symmetry, small epsilon_c) in the theorem headers.
- Required new assumptions/permits (if any): A framework equilibrium permit for the QSD and a horizon-thermodynamics permit.
- Framework-first proof sketch for the fix: Prove that the generator is self-adjoint in a weighted measure, giving a Gibbs form; then derive the local temperature from the generator's stationary measure near a Killing horizon.
- Validation plan: Check the Gibbs form in the linearized regime where the IG operator is symmetric.

### [E-004] Minimal-surface interpretation needs geometric hypotheses
- Location: thm-ryu-takayanagi and the Informational Area Law discussion
- Severity: Moderate
- Type: Scope restriction
- Claim (paraphrase): The separating antichain with minimal cardinality equals the minimal-area surface in the bulk.
- Why this is an error in the framework: The identification of antichains with minimal surfaces requires a convergence theorem from CST antichains to smooth hypersurfaces in the emergent metric, plus a control of discretization error. These hypotheses are not stated.
- Impact on downstream results: The RT interpretation is at best heuristic unless the CST-to-geometry limit is proved.
- Fix guidance (step-by-step):
  1. State the required Safe Harbor and sampling assumptions (quasi-uniformity, bounded curvature).
  2. Add a lemma linking antichain cardinality to Riemannian area up to O(epsilon_N) error.
  3. Restrict the RT statement to the limit N -> infinity under those assumptions.
- Required new assumptions/permits (if any): A framework lemma on antichain-to-area convergence.
- Framework-first proof sketch for the fix: Use the CST sprinkling density to approximate area by counting antichain elements in a tubular neighborhood; control the error by curvature bounds.
- Validation plan: Compare the antichain count against analytic areas in a simple curved metric (e.g., sphere) in simulation.

## Scope restrictions and clarifications
- The area-law and RT statements require explicit CST-to-geometry convergence assumptions; otherwise they should be labeled as conjectural.

## Proposed edits (optional)
- Rescale the nonlocal perimeter functional or its kernel to obtain a nontrivial Gamma-limit.
- Calibrate alpha and/or a0 explicitly so that Pi_IG matches Pi_elastic, or downgrade the match to a heuristic check.
- Mark QSD Gibbs and Unruh/Hawking connections as conjectures unless internal permits are added.

## Open questions
- Do we want to introduce an explicit scaling of K_epsilon to make the Gamma-limit nontrivial, or treat the area law as a heuristic proportionality?
- Is there an existing CST-to-area convergence lemma in Volume 3 that can be referenced for RT-style statements?