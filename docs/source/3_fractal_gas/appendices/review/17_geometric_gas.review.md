# Mathematical Review: docs/source/3_fractal_gas/appendices/17_geometric_gas.md

## Metadata
- Reviewed file: docs/source/3_fractal_gas/appendices/17_geometric_gas.md
- Review date: 2026-01-27
- Reviewer: Codex
- Scope: Full document
- Framework anchors (definitions/axioms/permits):
  - Axiom {prf:ref}`axiom-gg-confining-potential` and backbone Foster-Lyapunov result in {doc}`/3_fractal_gas/appendices/06_convergence`
  - Axiom {prf:ref}`axiom-gg-ueph` and Theorem {prf:ref}`thm-gg-ueph-construction` (uniform ellipticity of D_reg)
  - Definition {prf:ref}`def-gg-sde` (Geometric Gas SDE) and {prf:ref}`def-gg-fitness-potential`
  - C^3 regularity of V_fit in {doc}`/3_fractal_gas/appendices/14_b_geometric_gas_cinf_regularity_full`
  - Definition {prf:ref}`def-gg-hypocoercive-fisher` (state-dependent Fisher information)

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 4
- Minor: 3
- Notes: 0
- Primary themes: ellipticity constant mismatches; commutator control gaps; unproved QSD structure; missing internal citations for QSD/LSI/mean-field claims; kernel limit scope

## Error log
| ID | Location | Severity | Type | Short description |
|---|---|---|---|---|
| E-001 | Sec. 9.2 (def-gg-hypocoercive-fisher) + Sec. 9.3/9.6 | Major | Parameter inconsistency | Ellipticity comparison uses c_min^2/c_max^2 though c_min/c_max already bound D_reg. |
| E-002 | Sec. 6.4 (lem-gg-diffusion-perturbation) | Moderate | Computational error | Diffusion intensity bound mixes c_max^2 vs c_max and sign; dimensionally inconsistent. |
| E-003 | Sec. 9.4 (lem-gg-commutator-error) | Major | Proof gap / omission | Uses ||∇_v^2 f|| controlled by Fisher information without proof or assumptions. |
| E-004 | Sec. 9.3 (lem-gg-velocity-fisher-dissipation) | Major | Conceptual | Assumes QSD has Maxwellian v-marginal; not derived for state-dependent diffusion + cloning. |
| E-005 | Sec. 2.2-2.4 (localization kernel + limits) | Moderate | Scope restriction | Global limit requires compact X; hyper-local limit stated as nearest-neighbor though weights collapse to self. |
| E-006 | Sec. 5.2 (cor-gg-well-posedness) | Moderate | External dependency | Well-posedness invoked via Stroock-Varadhan without internal permit. |
| E-007 | Sec. 8.3 (thm-gg-geometric-ergodicity) | Minor | Citation / reference error | Missing internal citation to QSD existence/ergodicity proof in the framework. |
| E-008 | Sec. 9.6 (thm-gg-lsi-main) | Minor | Citation / reference error | Missing internal LSI references for backbone proof and perturbation stability. |
| E-009 | Sec. 10.2-10.3 (mean-field LSI + chaos) | Minor | Citation / reference error | Missing internal references for mean-field LSI and propagation-of-chaos results. |
| E-010 | Sec. 11.1 (cor-gg-kl-convergence) | Moderate | Computational error | KL decay inequality has wrong sign; implies growth. |

## Detailed findings

### [E-001] Ellipticity constants squared incorrectly
- Location: Sec. 9.2 (def-gg-hypocoercive-fisher) “Uniform Ellipticity Comparison”; Sec. 9.3 proof steps; Sec. 9.5/9.6 constants.
- Severity: Major
- Type: Parameter inconsistency (secondary: Definition mismatch)
- Claim (paraphrase): From UEPH, c_min^2 I_v <= I_hypo^Σ <= c_max^2 I_v, yielding alpha_hypo = gamma c_min^2 - d L_Sigma and C_LSI = (c_max^2/c_min^2)/alpha_hypo.
- Why this is an error in the framework: UEPH defines c_min, c_max as bounds on D_reg = Sigma_reg^2, not on Sigma_reg. For I_hypo^Σ = ∫ <∇_v sqrt(f), D_reg ∇_v sqrt(f)>, the comparison is c_min I_v <= I_hypo^Σ <= c_max I_v (no squares). Squaring propagates into the microscopic coercivity, commutator gap, and LSI constants, overstating coercivity and distorting thresholds.
- Impact on downstream results: All stated LSI constants, friction thresholds gamma_min, and any bound using c_min^2/c_max^2 are numerically incorrect; some positivity conditions may be false if corrected.
- Fix guidance (step-by-step):
  1. Decide whether c_min/c_max denote bounds for D_reg or for Sigma_reg (and state it explicitly in Axiom UEPH).
  2. If c_min/c_max bound D_reg (as written), replace every c_min^2/c_max^2 in Fisher comparisons, alpha_hypo, gamma_min, and C_LSI with c_min/c_max respectively.
  3. Recompute dependent thresholds (epsilon_F^*, gamma_min) and update statements in Sec. 9.3-9.7 and Appendix B.
- Required new assumptions/permits (if any): None beyond clarifying the definition of c_min/c_max.
- Framework-first proof sketch for the fix: Use eigenvalues of D_reg to bound <∇_v sqrt(f), D_reg ∇_v sqrt(f)> between c_min and c_max times ||∇_v sqrt(f)||^2, then propagate linearly into alpha_hypo and C_LSI.
- Validation plan: Track all occurrences of c_min^2/c_max^2 in Sec. 9 and Appendix B; verify they align with the UEPH definition of D_reg bounds.

### [E-002] Diffusion perturbation bounds use inconsistent constants
- Location: Sec. 6.4 (lem-gg-diffusion-perturbation), “Noise Intensity Change” and definition of C_diff,0.
- Severity: Moderate
- Type: Computational error (secondary: Dimensional mismatch)
- Claim (paraphrase): tr(Sigma_reg^2)/d is bounded by c_min^2 and c_max^2, and C_diff,0 = d * |c_max^2 - sigma^2|.
- Why this is an error in the framework: Sigma_reg^2 = D_reg, whose eigenvalues are bounded by c_min and c_max (not squared). The bound and C_diff,0 should use c_min/c_max, and the displayed inequality mixes c_max and sigma^2 with a mismatch of units (sigma^2 vs c_max^2).
- Impact on downstream results: The bias term in the Foster-Lyapunov drift and the threshold epsilon_F^*(rho) are numerically wrong; stability margins can be overstated.
- Fix guidance (step-by-step):
  1. Replace c_min^2/c_max^2 with c_min/c_max in the tr(Sigma_reg^2) bound.
  2. Re-derive the norm inequality for the diffusion perturbation to fix the sigma^2 comparison (use |c_max - sigma^2| or |c_max - sigma^2| with consistent units).
  3. Update C_diff,0 and any downstream formulas using it (Sec. 7.2/7.3, cor-gg-joint-thresholds).
- Required new assumptions/permits (if any): None if using existing UEPH bounds.
- Framework-first proof sketch for the fix: Use eigenvalue bounds on D_reg to bound tr(D_reg)/d between c_min and c_max, then estimate the difference with sigma^2 by direct comparison of diffusion coefficients.
- Validation plan: Check all appearances of C_diff,0 and confirm they use the corrected constants.

### [E-003] Commutator control uses second-derivative bound without support
- Location: Sec. 9.4 (lem-gg-commutator-error), Step 4: “||∇_v^2 f||^2 <= C I_hypo^Σ(f)”.
- Severity: Major
- Type: Proof gap / omission
- Claim (paraphrase): Second velocity derivatives of f are controlled by Fisher information, enabling commutator bounds.
- Why this is an error in the framework: Fisher information controls first derivatives of sqrt(f), not second derivatives of f. There is no provided permit or lemma linking H^2 regularity to I_hypo^Σ in this framework, so the commutator estimate is unsupported.
- Impact on downstream results: The hypocoercive gap and LSI proof rely on this commutator bound; without it the macroscopic transport control is unproven.
- Fix guidance (step-by-step):
  1. Add a framework lemma that upgrades I_hypo^Σ control to H^2 control for the class of solutions considered (e.g., via a hypocoercive Lyapunov functional on derivatives).
  2. Alternatively, restrict the statement to smooth densities with bounded H^2 and include explicit dependence on ||∇_v^2 f|| in the inequality.
  3. Recompute C_comm and the gap condition using the new bound and check N-uniformity.
- Required new assumptions/permits (if any): A regularity permit for solutions (e.g., hypoelliptic smoothing or Sobolev control under the generator).
- Framework-first proof sketch for the fix: Differentiate the Kolmogorov equation in v, derive an energy estimate for ||∇_v^2 sqrt(f)||, and close it with the friction term plus ellipticity bounds.
- Validation plan: Provide a standalone appendix lemma for the second-derivative estimate and cite it in Sec. 9.4.

### [E-004] QSD Maxwellian assumption is unproved
- Location: Sec. 9.3 (lem-gg-velocity-fisher-dissipation), Step 3.
- Severity: Major
- Type: Conceptual (secondary: Proof gap / omission)
- Claim (paraphrase): The QSD has form pi_N ∝ exp(-V_pot(x) - ||v||^2/(2T)), so v_i · ∇_{v_i} log pi_N = -|v_i|^2/T.
- Why this is an error in the framework: With state-dependent diffusion and cloning, the invariant/QSD density need not factor into a Gaussian in v with constant temperature. No internal derivation shows this structure or even conditional Gaussianity in v.
- Impact on downstream results: The microscopic coercivity estimate and the stated friction dissipation bound are not justified, undermining the LSI proof.
- Fix guidance (step-by-step):
  1. Derive the invariant density (or conditional v-marginal) from the adjoint generator under the stated dynamics, or state a new axiom that the v-marginal is Gaussian with temperature T_eff.
  2. If a full density is unavailable, reformulate the lemma using integration by parts with respect to Lebesgue measure and a coercivity estimate that does not require explicit pi_N.
  3. Propagate the corrected coercivity constant into Sec. 9.5-9.6.
- Required new assumptions/permits (if any): A permit describing the QSD structure in v, or a framework lemma bounding entropy production by friction without explicit pi_N.
- Framework-first proof sketch for the fix: Use the generator adjoint to write the entropy production identity, then bound the friction term using a v-Poincare/LSI inequality conditional on x, derived from the Ornstein-Uhlenbeck part if its invariant measure is shown to be Gaussian.
- Validation plan: Provide an explicit calculation for the v-marginal under fixed x and confirm the dependence of T_eff on Sigma_reg and gamma.

### [E-005] Localization limits require compactness and clarify hyper-local regime
- Location: Sec. 2.2-2.4 (def-gg-localization-kernel, prop-gg-rho-limits).
- Severity: Moderate
- Type: Scope restriction (secondary: Miswording)
- Claim (paraphrase): As rho -> infinity, K_rho -> 1/|X| and weights become 1/k; as rho -> 0, moments become nearest-neighbor evaluations.
- Why this is an error in the framework: The global limit to 1/|X| assumes compact X or a confining envelope; on unbounded domains |X| is infinite and the limit does not exist. In the discrete sum, rho -> 0 yields w_ii -> 1 (self-weight), not a nearest-neighbor evaluation unless self-weight is excluded by definition.
- Impact on downstream results: The continuity claims between local and global regimes may fail on unbounded domains or with self-weighting, affecting the interpretation of rho limits.
- Fix guidance (step-by-step):
  1. Add an explicit assumption: X compact or a confining envelope that normalizes K_rho for large rho.
  2. Clarify the hyper-local limit: either define w_ii = 0 and renormalize to nearest neighbors, or state the limit as self-evaluation.
  3. Update Prop. 2.4 proof to reflect the corrected limit and any required domain assumptions.
- Required new assumptions/permits (if any): A compactness or confining-envelope permit from the backbone framework.
- Framework-first proof sketch for the fix: On compact X, use dominated convergence to show K_rho becomes uniform; for the discrete hyper-local regime, compute the asymptotic of the Gaussian weight ratio explicitly.
- Validation plan: Add a remark clarifying the discrete vs. continuous limit and verify consistency with backbone statistics in {doc}`/3_fractal_gas/appendices/03_cloning`.

### [E-006] Well-posedness corollary relies on external SDE theory
- Location: Sec. 5.2 (cor-gg-well-posedness).
- Severity: Moderate
- Type: External dependency
- Claim (paraphrase): Unique strong solution follows from Stroock-Varadhan theory.
- Why this is an error in the framework: Volume 3 guidance requires internal permits; no internal lemma is cited that establishes strong existence for the stated state-dependent diffusion with cloning.
- Impact on downstream results: Well-posedness is a prerequisite for ergodicity and LSI claims; without an internal permit the logic chain is incomplete.
- Fix guidance (step-by-step):
  1. Add an internal permit/lemma in the appendices that proves strong existence under the stated Lipschitz and growth conditions within the Fractal Gas framework.
  2. If external theorems must be used, mark this corollary explicitly as relying on external theory and move it to a “references_do_not_cite” appendix.
  3. Ensure the permit accounts for the hybrid diffusion+cloning dynamics, or state that the corollary applies between cloning events.
- Required new assumptions/permits (if any): A “SDE well-posedness” permit aligned with the framework’s axioms.
- Framework-first proof sketch for the fix: Combine local Lipschitz + linear growth (from C^3 regularity and confining potential) with a piecewise construction between cloning events; then show the jump mechanism preserves existence globally.
- Validation plan: Provide a short verification checklist for the drift and diffusion regularity constants used.

### [E-007] Missing internal QSD citation in geometric ergodicity proof
- Location: Sec. 8.3 (thm-gg-geometric-ergodicity), proof.
- Severity: Minor
- Type: Citation / reference error
- Claim (paraphrase): Foster-Lyapunov + minorization implies QSD existence and exponential TV convergence.
- Why this is an error in the framework: The framework already proves QSD existence/TV convergence for the Euclidean backbone in {doc}`/3_fractal_gas/appendices/06_convergence` (Theorem {prf:ref}`thm-main-convergence`). The geometric proof follows the same drift+minorization template with perturbed constants, but the internal anchor is not cited.
- Impact on downstream results: Reduces traceability to the internal QSD proof; not a correctness issue.
- Fix guidance (step-by-step):
  1. Cite {doc}`/3_fractal_gas/appendices/06_convergence`, Theorem {prf:ref}`thm-main-convergence`, as the internal QSD existence/ergodicity anchor.
  2. State explicitly that the geometric perturbation bounds in Section {ref}`sec-gg-perturbation-analysis` supply the modified constants.
  3. Optionally cite {doc}`/3_fractal_gas/appendices/12_qsd_exchangeability_theory` for the mean-field/QSD linkage.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Refer to the backbone proof with the updated constants; no new argument required.
- Validation plan: Ensure the cross-references render and the proof text makes the adaptation explicit.

### [E-008] Missing internal LSI citations for backbone and perturbation stability
- Location: Sec. 9.6 (thm-gg-lsi-main), Step 3.
- Severity: Minor
- Type: Citation / reference error
- Claim (paraphrase): Entropy-Fisher inequality implies LSI via Bakry-Emery/Villani spectral theory.
- Why this is an error in the framework: Internal LSI proofs already exist (Euclidean Gas LSI in {doc}`/3_fractal_gas/appendices/15_kl_convergence`, Theorem {prf:ref}`thm-kl-convergence-euclidean`; hypocoercive route in {doc}`/3_fractal_gas/appendices/10_kl_hypocoercive`, Theorem {prf:ref}`thm-unconditional-lsi-explicit`) and the geometric perturbation stability is captured by {prf:ref}`thm-lsi-perturbation` and {prf:ref}`cor-adaptive-lsi` in {doc}`/3_fractal_gas/appendices/15_kl_convergence`. The section should point to those internal anchors.
- Impact on downstream results: Low; proof is traceable but not cited.
- Fix guidance (step-by-step):
  1. Add citations to {doc}`/3_fractal_gas/appendices/15_kl_convergence` and {doc}`/3_fractal_gas/appendices/10_kl_hypocoercive` in Step 3.
  2. Reference the perturbation stability results ({prf:ref}`thm-lsi-perturbation`, {prf:ref}`cor-adaptive-lsi`) as the geometric extension mechanism.
  3. Keep external citations as optional context, but foreground the internal proofs.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Cite the internal LSI theorems and the perturbation stability result; no new derivation required.
- Validation plan: Confirm the added references build and cross-link in Jupyter Book.

### [E-009] Missing internal references for mean-field LSI and propagation of chaos
- Location: Sec. 10.2-10.3 (thm-gg-mean-field-lsi, prop-gg-propagation-chaos).
- Severity: Minor
- Type: Citation / reference error
- Claim (paraphrase): Mean-field LSI and propagation-of-chaos statements are presented without internal anchors.
- Why this is an error in the framework: The propagation-of-chaos limit is already established internally as Theorem {prf:ref}`thm-propagation-chaos-qsd` in {doc}`/3_fractal_gas/appendices/12_qsd_exchangeability_theory` (full proof in {doc}`/3_fractal_gas/appendices/09_propagation_chaos`), and the mean-field LSI bound is recorded in Corollary {prf:ref}`cor-mean-field-lsi` in the same appendix. These should be cited directly.
- Impact on downstream results: Low; missing references reduce auditability.
- Fix guidance (step-by-step):
  1. Cite {prf:ref}`thm-propagation-chaos-qsd` with {doc}`/3_fractal_gas/appendices/12_qsd_exchangeability_theory` in Sec. 10.3, and point to {doc}`/3_fractal_gas/appendices/09_propagation_chaos` for the full proof.
  2. Cite {prf:ref}`cor-mean-field-lsi` in Sec. 10.2 for the mean-field LSI limit.
  3. Keep external references (Cattiaux-Guillin/Sznitman) as context, but foreground the internal results.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Refer to the internal propagation-of-chaos and mean-field LSI corollary; no new derivation required.
- Validation plan: Ensure all new references resolve in the built docs.

### [E-010] KL convergence inequality sign is wrong
- Location: Sec. 11.1 (cor-gg-kl-convergence), proof.
- Severity: Moderate
- Type: Computational error (secondary: Miswording)
- Claim (paraphrase): d/dt D_KL >= (1/C_LSI) D_KL, then integrate to get decay.
- Why this is an error in the framework: The sign is reversed. LSI gives d/dt D_KL <= -(1/C_LSI) D_KL. The stated inequality implies growth, contradicting convergence.
- Impact on downstream results: The displayed differential inequality is incorrect; the corollary’s rate statement is unsupported as written.
- Fix guidance (step-by-step):
  1. Replace the inequality with d/dt D_KL <= -(1/C_LSI) D_KL.
  2. Integrate to obtain D_KL(t) <= D_KL(0) e^{-t/C_LSI}.
  3. Align the rate with kappa_QSD only after the corrected inequality.
- Required new assumptions/permits (if any): None.
- Framework-first proof sketch for the fix: Use the standard LSI entropy dissipation inequality with the corrected sign.
- Validation plan: Verify consistency with the entropy production sign convention used earlier in Sec. 9.

## Scope restrictions and clarifications
- Appendix C explicitly uses classical differential geometry results (Ambrose-Singer, Raychaudhuri). External citations are allowed, but the appendix should clearly cite the literature where these results are invoked.
- Several “standard” statements (Gaussian kernel limits, Stroock-Varadhan, Meyn-Tweedie, Villani/Bakry-Emery, Sznitman) should be backed either by internal theorems or explicit external citations.

## Proposed edits (optional)
- Add a short “Parameter conventions” paragraph near Sec. 5.1 defining whether c_min/c_max bound D_reg or Sigma_reg and use it consistently.
- Add a remark in Sec. 2.4 specifying compactness or confining envelope assumptions for the rho -> infinity limit.

## Open questions
- Which internal reference should be the primary citation for LSI in this appendix: {doc}`/3_fractal_gas/appendices/15_kl_convergence` or {doc}`/3_fractal_gas/appendices/10_kl_hypocoercive`?
- For propagation of chaos, should citations point first to {doc}`/3_fractal_gas/appendices/12_qsd_exchangeability_theory` (summary) or directly to {doc}`/3_fractal_gas/appendices/09_propagation_chaos` (full proof)?
