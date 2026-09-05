# Mathematical Review: docs/source/1_agent/05_geometry/02_wfr_geometry.md

## Metadata
- Reviewed file: docs/source/1_agent/05_geometry/02_wfr_geometry.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (908 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/05_geometry/01_metric_law.md:102-104, 185-188, 219-242` (`thm-capacity-constrained-metric-law`, `def-extended-risk-tensor`, density defined w.r.t. $d\mu_G$)
  - `docs/source/1_agent/05_geometry/04_equations_motion.md:220-242` (`def-bulk-drift-continuous-flow`, $\beta_{\text{curl}}$)
  - `docs/source/1_agent/06_fields/02_reward_field.md:196-238, 324` (Hodge decomposition $A := \delta\Psi + \eta$, $\mathcal F = dA$, gauge invariance of $\mathcal F$, discount rate $\lambda$)
  - `docs/source/1_agent/06_fields/01_boundary_interface.md:526-552` (downstream use of `def-the-wfr-action` and the $\lambda$ threshold)
  - `docs/source/1_agent/01_foundations/02_control_loop.md:708-714` (units of $G$: $\mathrm{nat}\,[z]^{-2}$)
  - `docs/source/1_agent/03_architecture/01_compute_tiers.md:1983-2036, 2354-2530` (`def-the-rescaling-operator-renormalization`, `def-total-reconstruction`, Section 7.13 atlas / overlaps)
  - `docs/source/1_agent/04_control/02_belief_dynamics.md:268-290, 384` (GKSL generator, classical limit, Node 23)
  - `docs/source/1_agent/02_sieve/01_diagnostics.md:121` (Sieve node registry, Node 23 = NEPCheck)
  - `docs/source/1_agent/10_appendices/03_wfr_tensor.md:14-19, 32, 59-96, 121-131` (Appendix C, WFR stress-energy tensor)
  - `docs/source/1_agent/10_appendices/04_faq.md:485` (Node 23 WFRCheck reference)

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 9
- Minor: 10
- Notes: 2
- Primary themes:
  1. The vector-potential extension of the WFR action is asserted to be gauge invariant and dimensionally consistent when it is neither, and the chapter gives four mutually inconsistent statements of "the WFR action"; the object called $d^2_{\mathrm{WFR}}$ is not a well-defined squared distance.
  2. The length scale $\lambda$ receives three incompatible characterisations (chart-overlap radius, injectivity radius, $\sqrt{\mathrm{tr}\,G^{-1}/n}$), the default value has the wrong units, and the stated transport/reaction crossover "at $\lambda$" is off by a constant factor between $2\sqrt2$ and $\pi$ under the chapter's own action.
  3. The GKSL / master-equation section misstates standard facts: the commutator does not vanish for diagonal states unless $H$ is diagonal, the dissipator does not preserve diagonality in general, and a reversible master equation is a discrete-Wasserstein (Maas) gradient flow, not a Fisher-Rao / "WFR reaction" gradient flow.
  4. The WFR stress-energy tensor is presented as the source term of the capacity-constrained metric law, contradicting the explicit Risk Tensor definition in `01_metric_law.md`, and its pressure term depends on an unstated variational convention.
  5. The scale-dependent teleportation cost $\lambda^{(\ell)} \propto \sigma^{(\ell)}$ has the wrong monotonicity given the upstream definition of $\sigma^{(\ell)}$.
  6. Registry and notation issues: Node 23 is already NEPCheck in the Sieve; $\lambda$ is also the discount rate in the cross-referenced section; a non-existent "Axiom D" is cited.

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | `def-the-wfr-action`, Remark (Gauge Invariance), lines 105-137 | Major | Invalid inference | Framework | this chapter | Vector-potential WFR action is not gauge invariant; $d(d\chi)=0$ is a non sequitur; functional is not a squared distance |
| E-002 | `def-the-wfr-action` lines 111, 127 vs `rem-units` line 193 | Moderate | Dimensional mismatch | Framework | this chapter | $\langle\mathbf A,v\rangle$ term differs from $\|v\|_G^2$ by a factor $1/\text{time}$ under the chapter's own units |
| E-003 | Lines 111, 174, 577, 646 (and Appendix C line 14) | Minor | Definition mismatch | Framework | this chapter | Four mutually inconsistent statements of "the WFR action" ($\mathbf A$ term, factor $\tfrac12$, horizon, measure vs density) |
| E-004 | `def-the-wfr-action`, Conservative Limit, line 129 | Minor | Scope restriction | External | upstream `06_fields/02_reward_field.md` | Gauge $\mathbf A=0$ when $\mathcal F=0$ fails when $H^1(\mathcal Z)\neq0$ (harmonic part survives) |
| E-005 | `def-the-wfr-action`, Non-Conservative Case, line 131 | Moderate | Proof gap / omission | Framework | this chapter | Euler-Lagrange equations of the action cannot "yield" the Lorentz-Langevin SDE |
| E-006 | Physics Isomorphism box, lines 165-188 | Minor | Conceptual | External | this chapter | "Unique metric" claim false; domain stated as $\mathcal P(\mathcal Z)$ instead of $\mathcal M^+(\mathcal Z)$ |
| E-007 | Transport vs. Reaction item 2 (line 218); `prop-limiting-regimes` lines 303-310 | Minor | Conceptual | External | this chapter | Pure-reaction limit is Hellinger on $\mathcal M^+(\mathcal Z)$, not Fisher-Rao on $\Delta^{|\mathcal K|}$; cone lift written incorrectly |
| E-008 | Transport vs. Reaction item 3, lines 242-248; lines 193, 200 | Moderate | Computational error | Framework | this chapter | Transport/reaction crossover is not at $\|z_A-z_B\|_G=\lambda$; pure-strategy crossover $2\sqrt2\lambda$, pure reaction only for $d\ge\pi\lambda$ |
| E-009 | Line 248 vs `def-canonical-length-scale` lines 258-279 | Moderate | Definition mismatch | Framework | this chapter | Three incompatible characterisations of $\lambda$; "overlap radius" has no upstream definition |
| E-010 | `def-canonical-length-scale`, Default value, lines 269-275 | Moderate | Dimensional mismatch | Framework | this chapter | $\lambda_{\text{default}}=\sqrt{\mathrm{tr}(G^{-1})/n}$ has units $[z]/\sqrt{[G]}$, not $G$-length |
| E-011 | `def-canonical-length-scale`, Cross-reference, line 277 | Minor | Notation conflict | Framework | this chapter | $\lambda$ is simultaneously the WFR length and the discount rate of the cross-referenced section |
| E-012 | Connection to RL #26, lines 332-360 | Note | Conceptual | Framework | this chapter | Distributional RL is not exhibited as a limit of any equation in the chapter |
| E-013 | `thm-classical-master-equation-wfr` lines 373-384; line 395; table row 2 line 404; lines 366, 371 | Moderate | Invalid inference | Framework | this chapter (also upstream `04_control/02_belief_dynamics.md:290`) | Master equation is a discrete-Wasserstein gradient flow, not a WFR/reaction gradient flow; reversibility not hypothesised |
| E-014 | `cor-gksl-classical-limit` lines 386-397; table row 1 line 403 | Moderate | Invalid inference | Framework | this chapter (also upstream `04_control/02_belief_dynamics.md:290`) | Commutator does not vanish for diagonal states unless $H$ is diagonal; dissipator does not preserve diagonality |
| E-015 | Correspondence Table row 3, line 405 | Minor | Computational error | Framework | this chapter | Balanced-reaction condition misstated ($\int r\,d\mu=0$ instead of $\int\rho r\,d\mu=0$) |
| E-016 | Line 418; "Why This Connection Matters" lines 425-439 | Minor | Citation / reference error | External | this chapter | Continuous WFR gradient-flow theorem misattributed to Chizat et al. (2018); "trace preservation (optional)" contradicts CPTP definition |
| E-017 | `def-wfr-world-model`, lines 450-536 | Minor | Algorithm mismatch | Framework | this chapter | Text attributes $(v,r)$ to the policy; code is an action-conditioned world model with no goal input |
| E-018 | `def-scale-dependent-teleportation-cost`, lines 582-605 | Moderate | Definition mismatch | Framework | this chapter | $\lambda^{(\ell)}\propto\sigma^{(\ell)}$ has the wrong monotonicity given the upstream definition of $\sigma^{(\ell)}$ |
| E-019 | Line 638; `thm-wfr-stress-energy-tensor-variational-form` lines 640-683; Appendix C section C.4 | Major | Definition mismatch | Framework | this chapter (and `10_appendices/03_wfr_tensor.md:121-131`) | WFR stress-energy tensor conflicts with the metric law's explicit Risk Tensor (different definition, different units) |
| E-020 | `thm-wfr-stress-energy-tensor-variational-form`, lines 658-681; Appendix C section C.2 | Moderate | Conceptual | Framework | this chapter | Pressure term depends on holding the $G$-dependent density fixed, which does not preserve belief mass |
| E-021 | "Consistency with existing losses" table, lines 732-741 | Minor | Citation / reference error | Framework | this chapter | "Axiom D" does not exist; $r<0$ is mass destruction, not entropy production; stray `:::` at line 741 |
| E-022 | Sasaki vs WFR table line 771; summary item 3 line 901 | Note | Scope restriction | External | this chapter | Convexity holds only in $(\rho,\rho v,\rho r)$ with fixed endpoints and fixed metric |
| E-023 | Node 23: WFRCheck, lines 865-880 | Minor | Notation conflict | Framework | this chapter | Node number 23 is already assigned to NEPCheck in the Sieve |

## Detailed findings

### [E-001] The vector-potential WFR action is not gauge invariant (and is not a squared distance)
(was F-001)
- Location: The WFR Metric (Benamou-Brenier Formulation), `def-the-wfr-action`, Remark (Gauge Invariance), lines 105-137 (esp. 111, 133)
- Severity: Major
- Type: Invalid inference (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 133): "The action is invariant under gauge transformations $\mathbf{A} \to \mathbf{A} + d\chi$ for any scalar $\chi$, since $d(d\chi) = 0$."
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:211-227` defines $\mathcal{F} := d\mathcal{R} = dA$ and states gauge invariance of $\mathcal{F}$ only ("$\mathcal{F}$ is gauge-invariant: if $\mathcal{R}\to\mathcal{R}+d\chi$ then $\mathcal{F}\to\mathcal{F}$"). Nothing upstream claims the action is invariant.
- Why this is an error: $d(d\chi)=0$ shows that the curvature $\mathcal F$ is gauge invariant, not the action, which contains $\mathbf A$ itself. The only $\mathbf A$-dependent term of the action at line 111 is $-2\int_0^1\!\int\langle\mathbf A,v\rangle\,d\rho_s\,ds$. Under $\mathbf A\to\mathbf A+d\chi$ with $\chi$ independent of $s$ and no-flux boundary, using the chapter's own constraint $\partial_s\rho+\nabla\cdot(\rho v)=\rho r$:
  $\Delta\mathcal{E} = -2\int_0^1\!\!\int \rho\,v\cdot\nabla\chi = 2\int_0^1\!\!\int \chi\,\nabla\cdot(\rho v) = 2\int_0^1\!\!\int \chi\,(\rho r-\partial_s\rho) = 2\int_0^1\!\!\int \chi\,\rho\,r\,ds - 2\int \chi\,(\rho_1-\rho_0)$.
  The second term depends only on endpoints (so in the balanced case $r\equiv0$ minimisers are unchanged, though the action value is not invariant). The first term depends on the path whenever $r\neq0$, which is the entire point of the unbalanced setting. The verifier checked this numerically with a single Dirac of mass $m(s)$ at $x(s)$, $\chi=x^2$, endpoints $\rho_0=\delta_0$, $\rho_1=2\delta_1$: the path "move then grow" has $\Delta\mathcal E=-2.000$, the path "grow then move" has $\Delta\mathcal E=-4.000$, both matching the formula; in the balanced case both test paths give $-2.0=-2[\chi]_0^1$. Hence the infimum $d^2_{\mathrm{WFR}}(\rho_0,\rho_1)$ as defined depends on the gauge choice, and the stated reason is a non sequitur. Two further consequences: the $\mathbf A$ term is odd under path reversal $v\to-v$, so the infimum is not symmetric in $(\rho_0,\rho_1)$; and by completing the square, $\|v\|_G^2-2\langle\mathbf A,v\rangle=\|v-\mathbf A^\sharp\|_G^2-\|\mathbf A\|_G^2$, so the "squared distance" can be negative (e.g. $v=\mathbf A^\sharp$ gives integrand $-\|\mathbf A\|^2$). The Coulomb condition at line 133 is a gauge fixing, which is only needed because the action is not invariant.
- Impact on downstream results: `06_fields/01_boundary_interface.md:526ff, 546` and `05_geometry/04_equations_motion.md` cite `def-the-wfr-action` as the WFR equation/distance; Appendix C (`10_appendices/03_wfr_tensor.md:14-19`) silently drops the $\mathbf A$ term when "recalling" the same definition.
- Fix guidance (step-by-step):
  1. Either (a) drop the $\mathbf A$ term from the definition of $d^2_{\mathrm{WFR}}$ and keep it in a separately named driven action $\mathcal S_{\mathbf A}$ whose minimisers are gauge-independent only when $r\equiv0$; or (b) keep it, state that $\mathcal S_{\mathbf A}$ changes by $2\int\!\!\int\chi\rho r - 2\int\chi(\rho_1-\rho_0)$ under a gauge transformation, fix the Coulomb gauge as part of the definition, and stop calling the resulting quantity a squared distance.
  2. Replace "since $d(d\chi)=0$" with the correct statement that only $\mathcal F$ is gauge invariant.
  3. Make Appendix C and `06_fields/01_boundary_interface.md` recall whichever object is chosen.
- Required new assumptions/permits (if any): if option (b), the Coulomb gauge condition becomes part of the definition.
- Validation plan: Recompute $\Delta\mathcal E$ for a one-Dirac path with $r\neq0$ under two orderings (move-then-grow vs grow-then-move) and confirm the values differ; confirm the revised definition is symmetric and non-negative.

### [E-002] Units of the vector-potential term are inconsistent with the chapter's own unit remark
(was F-002)
- Location: `def-the-wfr-action` (lines 111, 127) vs `rem-units` (line 193)
- Severity: Moderate
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 111, 127, 193): "$\|v_s(z)\|_G^2 + \lambda^2|r_s(z)|^2 - 2\langle\mathbf{A}(z), v_s(z)\rangle$"; "*Units:* $[\mathbf{A}] = \mathrm{nat}/[\text{length}]$"; "$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$."
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:199` "*Units:* ... $[\eta]=\mathrm{nat}/[\text{length}]$" (so $[\mathbf A]=\mathrm{nat}/\text{length}$ is inherited faithfully); `docs/source/1_agent/01_foundations/02_control_loop.md:714` "Units: $[(G_V)_{ij}]=\mathrm{nat}\,[z]^{-2}$".
- Why this is an error: With the stated units, $\|v\|_G^2 = G_{ij}v^iv^j$ has units $\mathrm{nat}/\text{time}^2$, whereas $\langle\mathbf A,v\rangle$ has units $\mathrm{nat}/\text{time}$, and $\lambda^2r^2$ has units $\text{length}^2/\text{time}^2$. No choice of whether $s$ carries time units reconciles all three: if $s$ is dimensionless, $[\|v\|_G^2]=[\langle\mathbf A,v\rangle]=\mathrm{nat}$ but $[\lambda^2r^2]=\text{length}^2$; if $s$ has time units, the $\mathbf A$ term differs from the kinetic term by $1/\text{time}$.
- Impact on downstream results: Any attempt to set the coupling of the Lorentz term in `def-bulk-drift-continuous-flow` from this action will be off by an undetermined rate constant.
- Fix guidance (step-by-step):
  1. Insert a coupling with units of $1/\text{time}$ in front of the $\mathbf A$ term (e.g. $-2\,\beta_{\mathrm{curl}}\tau^{-1}\langle\mathbf A,v\rangle$, matching the dimensionless $\beta_{\mathrm{curl}}$ of `04_equations_motion.md:242`), or declare $s$ dimensionless and adjust `rem-units` to $[v]=\text{length}$, $[r]=1$, $[\lambda]=\text{length}$.
  2. State $[\lambda]=\sqrt{\mathrm{nat}}\cdot\text{length}$ or declare nat dimensionless for the purposes of $G$-norms, so that $\lambda^2r^2$ matches $\|v\|_G^2$.
- Required new assumptions/permits (if any): none.
- Validation plan: Tabulate the units of each of the three integrand terms after the edit and confirm they agree.

### [E-003] Inconsistent statements of "the WFR action" within the chapter
(was V-001)
- Location: lines 111, 174, 577, 646; `docs/source/1_agent/10_appendices/03_wfr_tensor.md:14`
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 111 defines $\mathcal E=\int_0^1\!\int(\|v\|_G^2+\lambda^2r^2-2\langle\mathbf A,v\rangle)\,d\rho_s\,ds$; line 174 writes $d^2_{\text{WFR}}=\inf\int_0^1\!\int(\|v\|_G^2+\lambda^2r^2)\rho\,d\mu_G\,dt$; line 577 "Recall the WFR action: $\mathcal E=\int(\|v\|_G^2+\lambda^2|r|^2)d\rho$"; line 646 "Let the WFR action be $\mathcal S_{\text{WFR}}=\tfrac12\int_0^T\!\int\rho(\|v\|_G^2+\lambda^2r^2)d\mu_G\,ds$".
- Upstream anchor: `docs/source/1_agent/10_appendices/03_wfr_tensor.md:14` recalls the line-646 form as `def-the-wfr-action`.
- Why this is an error: The four statements differ in the presence of the $\mathbf A$ term, the normalisation (factor $\tfrac12$), the horizon ($1$ vs $T$), the time variable ($s$ vs $t$), and whether $\rho$ is a measure or a density w.r.t. $d\mu_G$. The factor $\tfrac12$ changes $T_{ij}$ and $P$ in `thm-wfr-stress-energy-tensor-variational-form` by a factor 2 relative to what the definition at line 111 would give, and the missing $\mathbf A$ term means the stress-energy theorem is not about the object defined as `def-the-wfr-action`.
- Impact on downstream results: E-001, E-019, E-020 and Appendix C all depend on which action is meant.
- Fix guidance (step-by-step):
  1. Give the driven action (with $\mathbf A$) and the undriven action distinct names and labels.
  2. Fix one normalisation and one horizon and use it in all four places.
  3. Make Appendix C recall the correct definition.
- Required new assumptions/permits (if any): none.
- Validation plan: grep the chapter and Appendix C for every displayed action and confirm each carries the agreed name and normalisation.

### [E-004] "Choose the gauge $\mathbf A=0$ when $\mathcal F=0$" fails on non-simply-connected latent spaces
(was F-004)
- Location: `def-the-wfr-action`, Conservative Limit, line 129
- Severity: Minor
- Type: Scope restriction
- Criterion: External
- Origin: upstream `docs/source/1_agent/06_fields/02_reward_field.md` (inherited), restated here
- Claim (verbatim, line 129): "When $\mathcal{F} = 0$ ..., we can choose the gauge $\mathbf{A} = 0$".
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:196` "Define the non-exact component $A := \delta\Psi + \eta$"; `:231-238` "$\mathcal F = 0$ ... Equivalently, $\mathcal R = d\Phi$ ... (the solenoidal and harmonic components vanish)."
- Why this is an error: Since $d\eta=0$ by definition of the harmonic part, $\mathcal F = d\delta\Psi = 0$ forces only $\delta\Psi = 0$ (as $\langle d\delta\Psi,\Psi\rangle = \|\delta\Psi\|^2$) and leaves $\eta$ free. On a latent space with $H^1(\mathcal Z)\neq0$, which `03_architecture/01_compute_tiers.md:2388-2400` explicitly contemplates (atlas glued into a cylinder), $\eta$ is closed but not exact, $\mathbf A = \eta \neq 0$ cannot be removed by any gauge transformation, and the holonomy $\oint\mathbf A$ is physical. The upstream "equivalently" is already wrong in that case; the chapter inherits it.
- Impact on downstream results: Conservative-limit statements in `04_equations_motion.md` and `06_fields` implicitly assume $H^1=0$.
- Fix guidance (step-by-step):
  1. Add the hypothesis "$H^1_{\mathrm{dR}}(\mathcal Z)=0$ (or $\eta = 0$)" to the Conservative Limit, or write "$\mathbf A$ is harmonic; if $\mathcal Z$ is simply connected we may take $\mathbf A = 0$."
  2. Correct the upstream equivalence in `06_fields/02_reward_field.md:231-238` accordingly.
- Required new assumptions/permits (if any): $H^1_{\mathrm{dR}}(\mathcal Z)=0$ where $\mathbf A=0$ is used.
- Validation plan: Check the Conservative Limit on the cylinder example of Section 7.13 with a constant angular 1-form $\eta$.

### [E-005] Euler-Lagrange equations of the WFR action do not "yield" the second-order Lorentz-Langevin SDE
(was F-003)
- Location: `def-the-wfr-action`, Non-Conservative Case, line 131
- Severity: Moderate
- Type: Proof gap / omission (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 131): "The Euler-Lagrange equations of this action yield the Lorentz-Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`)."
- Upstream anchor: `docs/source/1_agent/05_geometry/04_equations_motion.md:220-236`: "$dp_k = [-\partial_k\Phi_{\text{eff}} - \gamma p_k + \beta_{\text{curl}}\mathcal F_{kj}G^{j\ell}p_\ell - \Gamma^m_{k\ell}G^{\ell j}p_jp_m + u_{\pi,k}]ds + \sqrt{2\gamma T_c}(G^{1/2})_{kj}dW^j_s$".
- Why this is an error: The action of line 111 contains no potential $\Phi_{\mathrm{eff}}$, no friction $\gamma$, no control field $u_\pi$, no temperature $T_c$ and no noise. Its Euler-Lagrange equations (Lagrangian $\|v\|_G^2-2\langle\mathbf A,v\rangle$ with the continuity constraint) give at most a geodesic equation with a Lorentz term proportional to $\mathcal F\cdot v$ and a pressure-like term from the reaction cost. Four of the six force terms of the cited SDE and its stochastic character cannot be obtained from a deterministic variational principle without further hypotheses. No such derivation exists in the chapter, Appendix C, or the cited definition.
- Impact on downstream results: Readers of `04_equations_motion.md` are told the SDE is grounded in the WFR action; it is not.
- Fix guidance (step-by-step):
  1. Replace the sentence by: "The Euler-Lagrange equations of this action produce the velocity-dependent Lorentz term $\beta_{\mathrm{curl}}\mathcal F_{kj}G^{j\ell}p_\ell$ of the geodesic Langevin equation; the potential, friction, control and noise terms are added separately (see Definition `def-bulk-drift-continuous-flow`)."
  2. Optionally supply the one-line Euler-Lagrange computation.
- Required new assumptions/permits (if any): none.
- Validation plan: Derive the Euler-Lagrange equation for a single particle with Lagrangian $\|\dot x\|_G^2 - 2\langle\mathbf A,\dot x\rangle$ and confirm only the geodesic and Lorentz terms appear.

### [E-006] Uniqueness claim for WFR is false; domain stated as $\mathcal P(\mathcal Z)$ instead of $\mathcal M^+(\mathcal Z)$
(was F-005)
- Location: Physics Isomorphism: Wasserstein-Fisher-Rao Geometry (lines 165-188, esp. 169, 171)
- Severity: Minor
- Type: Conceptual (secondary: Notation conflict)
- Criterion: External
- Origin: this chapter
- Claim (verbatim, lines 169, 171): "It is the unique metric allowing both mass transport and creation/annihilation"; "The belief density $\rho$ evolves under the WFR metric on $\mathcal{P}(\mathcal{Z})$".
- Upstream anchor: not applicable; lines 40 and 336 of the same chapter say $\rho_s\in\mathcal M^+(\mathcal Z)$.
- Why this is an error: (i) Unbalanced-transport metrics form a large family (partial optimal transport, Piccoli-Rossi generalised Wasserstein, bounded-Lipschitz norms, the one-parameter WFR/HK family indexed by $\lambda$, and $p\neq2$ variants); Liero-Mielke-Savaré and Chizat et al. characterise HK as a distinguished (cone / Benamou-Brenier) metric, not the unique metric with mass creation. The chapter's own free parameter $\lambda$ already makes uniqueness false on its own terms. (ii) With $r\neq0$ total mass changes along paths, so the geometry lives on $\mathcal M^+(\mathcal Z)$, not on the probability simplex; the box contradicts lines 40 and 336. The formula at line 174 also drops the $\mathbf A$ term (see E-003).
- Impact on downstream results: The uniqueness rhetoric is repeated in the Sasaki-vs-WFR prose (line 782).
- Fix guidance (step-by-step):
  1. Replace with "It is the canonical (cone-space / Benamou-Brenier) metric combining $W_2$ transport with Hellinger/Fisher-Rao reaction".
  2. Replace $\mathcal P(\mathcal Z)$ by $\mathcal M^+(\mathcal Z)$.
- Required new assumptions/permits (if any): none.
- Validation plan: Text check against lines 40 and 336.

### [E-007] Pure reaction limit gives Hellinger on $\mathcal M^+(\mathcal Z)$, not Fisher-Rao on $\Delta^{|\mathcal K|}$; cone lift written incorrectly
(was F-009)
- Location: Transport vs. Reaction Components item 2 (line 218); `prop-limiting-regimes` item 2 and proof sketch (lines 303-310)
- Severity: Minor
- Type: Conceptual (secondary: Proof gap / omission)
- Criterion: External
- Origin: this chapter
- Claim (verbatim, lines 218, 310): "In the limit $v \to 0$, the dynamics reduce to the Fisher-Rao metric on the probability simplex $\Delta^{|\mathcal{K}|}$"; "The cone-space representation of WFR (lifting $\rho$ to $(\sqrt{\rho}, \sqrt{\rho}\cdot z)$) shows ...".
- Upstream anchor: not applicable; defined in this chapter.
- Why this is an error: With $v\equiv0$ the action (111) is $\lambda^2\int\!\!\int r^2\,d\rho\,ds$, whose induced distance is the scaled Hellinger distance $4\lambda^2\|\sqrt{\rho_0}-\sqrt{\rho_1}\|^2_{L^2(\mu)}$ on $\mathcal M^+(\mathcal Z)$, pointwise in $z$ (mass is created or destroyed in place). It is not a metric on the $|\mathcal K|$-simplex: the $z_n$ dependence is not integrated out, total mass is not preserved along pure-reaction geodesics, and the Fisher-Rao geodesic distance on $\Delta^{|\mathcal K|}$ (spherical arc length) differs from the Hellinger chord. What is true is that the Fisher-Rao tensor is the restriction of the Hellinger tensor to constant total mass. The lift "$(\sqrt\rho,\sqrt\rho\cdot z)$" is not meaningful on a manifold ($z$ cannot be scaled); the cone over $\mathcal Z$ is $(\mathcal Z\times\mathbb R_{\ge0})/(\mathcal Z\times\{0\})$ with radial coordinate $2\lambda\sqrt\rho$ and cone metric $dR^2 + R^2\,d_G^2/(4\lambda^2)$ (as used in the E-008 recomputation).
- Impact on downstream results: The "Fisher-Rao on the simplex" identification feeds the GKSL section (E-013).
- Fix guidance (step-by-step):
  1. Replace "Fisher-Rao metric on $\Delta^{|\mathcal K|}$" by "Hellinger (spherical Fisher-Rao) geometry on $\mathcal M^+(\mathcal Z)$, whose restriction to unit total mass has the Fisher-Rao tensor".
  2. Write the cone lift as $(z, 2\lambda\sqrt{\rho})$.
- Required new assumptions/permits (if any): none.
- Validation plan: Compute the $v\equiv0$ geodesic between two Diracs at the same point with masses $m_0, m_1$ and confirm the cost $4\lambda^2(\sqrt{m_1}-\sqrt{m_0})^2$.

### [E-008] The transport/reaction crossover is not at $\|z_A-z_B\|_G = \lambda$ under the chapter's own action
(was F-006)
- Location: Transport vs. Reaction Components, item 3 (lines 242-248); also `rem-units` (line 193) and prose (line 200)
- Severity: Moderate
- Type: Computational error (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 244-246): "If $\|z_A - z_B\|_G < \lambda$: Transport is preferred (continuous regime). If $\|z_A - z_B\|_G > \lambda$: Reaction is preferred (discrete chart transition)".
- Upstream anchor: not applicable; the action is defined at line 111.
- Why this is an error: Per-particle reduction of (111) without $\mathbf A$: a Dirac of mass $m$ at $x$ has action $m\dot x^2+\lambda^2\dot m^2/m$; with $u=\sqrt m$, $\lambda^2\dot m^2/m=4\lambda^2\dot u^2$, so annihilating or creating unit mass over $s\in[0,1]$ costs $4\lambda^2$ each, total $8\lambda^2$ for teleportation (the Hellinger value with disjoint supports). Pure transport over geodesic distance $d$ costs $d^2$. Pure-strategy crossover: $d^2=8\lambda^2$, i.e. $d=2\sqrt2\,\lambda\approx2.83\lambda$, not $\lambda$. Cone reduction: with $R=2\lambda u$ the action is $\dot R^2+R^2(\dot x/2\lambda)^2$, a cone over $\mathcal Z$ with angle $x/(2\lambda)$; the HK distance between unit Diracs is $8\lambda^2\bigl(1-\cos\min(d/2\lambda,\pi/2)\bigr)$ (the truncation because independent in-place annihilation/creation, cost $R_0^2+R_1^2$, beats the cone geodesic once $\cos\le0$). Table (transport / teleport / HK): $d=\lambda$: 1.000 / 8.000 / 0.979; $d=2\sqrt2\lambda$: 8.000 / 8.000 / 6.752; $d=\pi\lambda$: 9.870 / 8.000 / 8.000. A direct numerical minimisation over a three-component ansatz (moving particle with variable mass plus in-place annihilation/creation at both ends, 60 time steps) reproduces the closed form to three figures (0.980, 6.780, 8.001) and shows the optimum at $d=\lambda$ already modulates mass along the way. So the bullets are wrong both as a threshold ($\lambda$ vs $2\sqrt2\lambda$ or $\pi\lambda$) and as a dichotomy (there is no pure-transport regime for $d>0$).
- Impact on downstream results: `06_fields/01_boundary_interface.md:551-552` repeats "$d_{\mathrm{WFR}}<\lambda$: transport dominates / $>\lambda$: reaction dominates" (there additionally comparing a distance between measures with a length in $\mathcal Z$).
- Fix guidance (step-by-step):
  1. Replace the bullets with: "$\lambda$ sets the crossover scale: under the action (111) pure reaction beats pure transport for $d>2\sqrt2\lambda$, and the WFR geodesic between point masses is pure reaction iff $d\ge\pi\lambda$; for $d\ll\lambda$ the geodesic is transport-dominated with reaction corrections of relative order $d^2/\lambda^2$."
  2. Correct the repeated threshold in `06_fields/01_boundary_interface.md:551-552`.
- Required new assumptions/permits (if any): none.
- Validation plan: Reproduce the three-column table above numerically for the chosen normalisation of the action.

### [E-009] Three incompatible characterisations of $\lambda$; the "overlap radius" has no upstream definition
(was F-007)
- Location: item 3 "Operational interpretation" (line 248) vs `def-canonical-length-scale` (lines 258-279, esp. 261-264)
- Severity: Moderate
- Type: Definition mismatch (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 248, 261-264): "$\lambda$ is exactly the **radius of the chart overlap region** ({ref}`Section 7.13 <sec-factorized-jump-operators-efficient-chart-transitions>`)"; "The canonical choice for $\lambda$ is the **geodesic injectivity radius** $\lambda := \min_{z\in\mathcal Z}\mathrm{inj}_G(z)$".
- Upstream anchor: `docs/source/1_agent/03_architecture/01_compute_tiers.md:2354-2530` (Section 7.13) never defines an overlap radius; overlaps are defined by a router-weight threshold at `:2499`: "A point $x$ is in the overlap $U_i\cap U_j$ if both router weights exceed a threshold". A grep for "overlap" in that file returns only atlas prose, the overlap-consistency loss and this threshold.
- Why this is an error: (i) "Exactly" points to a quantity that does not exist upstream. (ii) Even granting an overlap radius, it is an atlas/router property, whereas the injectivity radius is intrinsic to $(\mathcal Z,G)$ and independent of chart layout; at most one of the two sentences can define $\lambda$. (iii) $\min_z \mathrm{inj}_G(z)$ over $\mathcal Z=\mathcal K\times\mathcal Z_n\times\mathcal Z_{\text{tex}}$ is ill-posed: the discrete factor $\mathcal K$ has no exponential map, and on bounded charts the injectivity radius tends to $0$ at the boundary.
- Impact on downstream results: Every "teleportation length" statement (lines 358, 899) and the boundary-interface usage inherit an undefined constant.
- Fix guidance (step-by-step):
  1. Pick one definition (recommended: per-fibre injectivity radius of $(\mathcal Z_n, G_n(\cdot;K))$ on the interior of each chart, infimum over charts).
  2. Delete "exactly the radius of the chart overlap region" or downgrade it to a heuristic remark.
  3. Drop the pointer to Section 7.13 unless a radius is defined there.
- Required new assumptions/permits (if any): positivity of the chosen injectivity radius (compact interior or explicit lower bound).
- Validation plan: Confirm the chosen definition is finite and positive on the chapter's own examples and that the section pointer resolves to a definition.

### [E-010] The default $\lambda_{\text{default}}=\sqrt{\mathrm{tr}(G^{-1})/n}$ mixes coordinate lengths with $G$-lengths
(was F-008)
- Location: `def-canonical-length-scale`, Default value (lines 269-275)
- Severity: Moderate
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 272-275): "$\lambda_{\text{default}} = \sqrt{\frac{\text{tr}(G^{-1})}{n}} \approx \text{mean characteristic length of } \mathcal{Z}$. This corresponds to the RMS geodesic step size in an isotropic metric."
- Upstream anchor: `docs/source/1_agent/01_foundations/02_control_loop.md:714` "Units: $[(G_V)_{ij}]=\mathrm{nat}\,[z]^{-2}$".
- Why this is an error: In the action $\lambda^2 r^2$ is added to $\|v\|_G^2$, and $\lambda$ is compared to $\|z_A-z_B\|_G$ (line 245), so $\lambda$ must be a $G$-length ($[\lambda]=\sqrt{[G]}\,[z]=\sqrt{\mathrm{nat}}$). But $\sqrt{\mathrm{tr}(G^{-1})/n}$ has units $[z]/\sqrt{[G]}=[z]/\sqrt{\mathrm{nat}}$. Isotropic check $G=gI$: $\lambda_{\text{default}}=1/\sqrt g$ and $\|\Delta z\|_G=\sqrt g|\Delta z|$, so the criterion reads $|\Delta z|<1/g$, which scales incorrectly (a genuine $G$-length threshold would give $|\Delta z|<c/\sqrt g$). The RMS geodesic length of a unit coordinate step is $\sqrt{\mathrm{tr}\,G/n}$, the reciprocal expression. In addition $G$ is a field on $\mathcal Z$, so "$\mathrm{tr}(G^{-1})$" needs a point or an average to be defined.
- Impact on downstream results: Any implementation copying this default sets $\lambda$ with the wrong scaling in $G$ (off by a factor $g$ in the isotropic case).
- Fix guidance (step-by-step):
  1. Set $\lambda_{\text{default}} := \sqrt{\mathrm{tr}(\bar G)/n}\cdot \delta z$ for a chosen coordinate step $\delta z$ (explicitly a $G$-length), or define $\lambda$ as a dimensionless multiple of the $G$-length unit and say so.
  2. Specify the averaging over $z$ that defines $\bar G$.
- Required new assumptions/permits (if any): none.
- Validation plan: Check the isotropic case $G=gI$ gives a threshold $|\Delta z|\propto 1/\sqrt g$.

### [E-011] Symbol $\lambda$ is simultaneously the WFR length scale and the temporal discount rate in the cross-referenced section
(was F-019)
- Location: `def-canonical-length-scale`, Cross-reference (line 277)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 277): "The screening length $\ell_{\text{screen}}=1/\kappa$ from Section 24.2 plays an analogous role for temporal horizons; $\lambda$ plays the corresponding role for spatial horizons".
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:324` "Let the temporal discount rate be $\lambda := -\ln\gamma/\Delta t$ and define the **spatial screening mass** $\kappa:=\lambda/c_{\text{info}}$".
- Why this is an error: The sentence invokes the very section in which $\lambda$ denotes a rate ($1/\text{time}$) while using $\lambda$ for a length; the analogy equates $\ell_{\text{screen}}=c_{\text{info}}/\lambda_{\text{rate}}$ with $\lambda_{\text{length}}$ under one symbol. Also, $\ell_{\text{screen}}$ is itself a spatial screening length upstream, so "temporal horizons" mislabels it.
- Impact on downstream results: none mathematical.
- Fix guidance (step-by-step):
  1. Rename one of the two symbols (e.g. $\lambda_{\mathrm{WFR}}$ or $\ell_{\mathrm{WFR}}$).
  2. Correct "temporal" to "spatial (discount-induced)".
- Required new assumptions/permits (if any): none.
- Validation plan: grep both chapters for the chosen symbol.

### [E-012] Connection to RL #26 does not exhibit distributional RL as a limit of anything defined
(was F-021)
- Location: Connection to RL #26 (lines 332-360, esp. 344-353)
- Severity: Note
- Type: Conceptual
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Restrict to value distributions at single states (no spatial transport). Use Euclidean metric ($G\to I$). ... This recovers **Distributional RL**".
- Upstream anchor: not applicable.
- Why this is an error: The stated degenerate limit keeps only the reaction term, i.e. Hellinger geometry on return distributions; the distributional Bellman operator $Z\stackrel{D}{=}R+\gamma Z'$ is neither a gradient flow nor a geodesic in that geometry, and C51 / QR-DQN use KL projections and $W_1$ quantile regression respectively, neither of which is Hellinger. Unlike Connection #1 (`02_control_loop.md:733ff`), no equation of this chapter reduces to the displayed one.
- Impact on downstream results: none.
- Fix guidance (step-by-step):
  1. Either give the actual reduction (e.g. Wasserstein-projected distributional Bellman as a $W_2$ step) or label the box as an analogy.
- Required new assumptions/permits (if any): none.
- Validation plan: text check.

### [E-013] The master equation is a discrete-Wasserstein (Maas) gradient flow, not a Fisher-Rao / "reaction" or WFR gradient flow; hypotheses not stated
(was F-011)
- Location: `thm-classical-master-equation-wfr` (lines 373-384, esp. 382), `cor-gksl-classical-limit` last sentence (line 395), Correspondence Table row 2 (line 404), prose (lines 366, 371)
- Severity: Moderate
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (also asserted upstream at `docs/source/1_agent/04_control/02_belief_dynamics.md:290`)
- Claim (verbatim): "By Theorem ..., this evolution is a gradient flow in WFR geometry." (line 395); table: dissipator $\leftrightarrow$ "Reaction rate $r$ (jump-induced mass redistribution)" (line 404); theorem: "is the gradient flow of the relative entropy ... with respect to a discrete Wasserstein-type metric, where $\pi$ is the stationary distribution satisfying detailed balance" (line 382).
- Upstream anchor: `docs/source/1_agent/04_control/02_belief_dynamics.md:290` "This classical master equation is **rigorously equivalent** to a gradient flow in the Wasserstein-Fisher-Rao metric ... reaction corresponds to jump-induced mass redistribution".
- Why this is an error: (i) The theorem body is the Maas/Mielke result: the reversible master equation is the gradient flow of $H(p\|\pi)$ for a transport metric on the graph (mass flows along edges with logarithmic-mean mobility; total mass is conserved along every path). In the chapter's own dictionary (lines 82, 218) jumps between charts are reaction (pointwise creation/annihilation). The Hellinger/Fisher-Rao gradient flow of $H(p\|\pi)$ is the nonlinear replicator equation $\dot p_j = -p_j\bigl(\log(p_j/\pi_j) - \sum_k p_k\log(p_k/\pi_k)\bigr)$, not the linear master equation. So the corollary's conclusion and the table's identification of the dissipator with $r$ do not follow from the theorem and are false for the WFR structure at line 111. (ii) The theorem states detailed balance and existence of $\pi$ as facts rather than hypotheses; the Maas result requires irreducibility and reversibility ($W_{jk}\pi_k = W_{kj}\pi_j$). Non-reversible master equations are not gradient flows of $H(p\|\pi)$ in any metric of this type. The theorem itself is an external dependency correctly cited; the fault is the inference from theorem to corollary within the chapter.
- Impact on downstream results: The claimed "rigorous" WFR-GKSL identity is used rhetorically in lines 415-423 and in the Sasaki-vs-WFR table ("Gradient flow of entropy (rigorous)").
- Fix guidance (step-by-step):
  1. Rewrite the theorem with hypotheses "irreducible, reversible w.r.t. $\pi$" and rename it "Classical master equation as discrete-Wasserstein gradient flow".
  2. In the corollary and table, say that chart jumps in the GKSL classical limit correspond to graph transport on $\mathcal K$ (Maas metric), a mass-conserving structure distinct from the reaction term of `def-the-wfr-action`.
  3. Delete "gradient flow in WFR geometry" or supply a proof.
  4. Correct the parallel claim at `04_control/02_belief_dynamics.md:290`.
- Required new assumptions/permits (if any): irreducibility and detailed balance of the rate matrix.
- Validation plan: Compute the Hellinger gradient flow of $H(p\|\pi)$ on a two-state chain and confirm it is nonlinear in $p$, unlike the master equation.

### [E-014] Corollary: the commutator does not vanish for diagonal states, and the dissipator does not preserve diagonality, without extra hypotheses
(was F-010)
- Location: `cor-gksl-classical-limit` (lines 386-397, esp. 389-395) and Correspondence Table row 1 (line 403)
- Severity: Moderate
- Type: Invalid inference (secondary: Scope restriction)
- Criterion: Framework
- Origin: this chapter (the same sentence appears upstream at `docs/source/1_agent/04_control/02_belief_dynamics.md:290`; both are wrong)
- Claim (verbatim, lines 389-395): "When the GKSL density matrix is diagonal, $\varrho=\mathrm{diag}(p_1,\ldots,p_K)$, the GKSL equation reduces to a classical master equation with rates $W_{jk}=\sum_\ell\gamma_\ell|\langle j|L_\ell|k\rangle|^2$. The commutator term $-i[H,\varrho]$ vanishes identically for diagonal states."
- Upstream anchor: `docs/source/1_agent/04_control/02_belief_dynamics.md:268-283` (GKSL generator with general Hermitian $H$ and arbitrary operators $L_j$).
- Why this is an error: For $\varrho=\mathrm{diag}(p)$, $[H,\varrho]_{jk} = H_{jk}p_k - p_jH_{jk} = H_{jk}(p_k-p_j)$, which is zero for all $j\neq k$ only if $H$ is diagonal in the same basis or $p$ is uniform. Likewise $(L\varrho L^\dagger)_{jj'} = \sum_k L_{jk}p_k\overline{L_{j'k}}$ has off-diagonal entries in general, so a diagonal state does not remain diagonal and the dynamics does not close on $p$. The stated Pauli master equation holds under the standard secular/classical hypotheses: $[H,\varrho]=0$ (e.g. $H$ diagonal) and jump operators mapping basis states to basis states (e.g. $L_\ell\propto|j\rangle\langle k|$), or an additional dephasing that kills coherences. None is stated. The table row "Vanishes (no off-diagonal elements to rotate)" repeats the error.
- Impact on downstream results: The corollary is the only bridge between Section 12.5 (GKSL beliefs) and this chapter; `04_control/02_belief_dynamics.md:290` cites it back.
- Fix guidance (step-by-step):
  1. Add hypotheses "$H$ diagonal in the $\{|k\rangle\}$ basis (or $[H,\varrho]=0$) and $L_\ell=|j_\ell\rangle\langle k_\ell|$ (transition operators)"; then the reduction holds with $W_{jk}=\sum_{\ell: (j_\ell,k_\ell)=(j,k)}\gamma_\ell$.
  2. Correct the table row to "vanishes when $H$ is diagonal in the belief basis".
  3. Correct the same sentence upstream at `04_control/02_belief_dynamics.md:290`.
- Required new assumptions/permits (if any): $H$ diagonal in the belief basis; transition-type jump operators (or a secular approximation).
- Validation plan: Evaluate $[H,\mathrm{diag}(p)]$ for a $2\times2$ Hermitian $H$ with $H_{12}\neq0$ and $p_1\neq p_2$; confirm it is nonzero.

### [E-015] Balanced-reaction condition misstated
(was F-012)
- Location: Correspondence Table (Classical Limit), row 3 (line 405)
- Severity: Minor
- Type: Computational error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 405): "Probability conservation $\sum_k\dot p_k=0$ | Balanced reaction ($\int r\,d\mu = 0$ globally)".
- Upstream anchor: not applicable; continuity equation defined at line 117 of this chapter.
- Why this is an error: Integrating $\partial_s\rho+\nabla\cdot(\rho v)=\rho r$ over $\mathcal Z$ with no-flux boundary gives $\frac{d}{ds}\int\rho\,d\mu = \int\rho\,r\,d\mu$. Mass conservation is therefore $\int\rho\,r\,d\mu = 0$; $\int r\,d\mu=0$ is neither necessary nor sufficient.
- Impact on downstream results: The code at lines 806-836 and the WFRCheck semantics ("Transport-Reaction balance") should enforce or monitor $\sum_k \rho_k r_k = 0$, not $\sum_k r_k = 0$.
- Fix guidance (step-by-step):
  1. Replace with "$\int\rho\,r\,d\mu = 0$".
  2. Check the code at lines 806-836 uses the mass-weighted sum.
- Required new assumptions/permits (if any): none.
- Validation plan: Integrate the continuity equation with $r$ constant and non-zero and confirm $\int r\,d\mu \neq 0$ while mass changes.

### [E-016] Misattribution of the continuous WFR gradient-flow theorem; "trace preservation (optional)" for GKSL
(was F-013)
- Location: prose (line 418) and "Why This Connection Matters" (lines 425-439, esp. 432)
- Severity: Minor
- Type: Citation / reference error (secondary: Conceptual)
- Criterion: External
- Origin: this chapter
- Claim (verbatim, lines 418, 432): "Maas (2011) and Mielke (2011) proved it for discrete state spaces; Chizat et al. (2018) extended it to continuous spaces with the full WFR metric."; "Trace preservation (optional): If you want total probability conserved, you can enforce it. If you want to allow mass creation/destruction, you can do that too."
- Upstream anchor: `docs/source/1_agent/04_control/02_belief_dynamics.md:268-272` defines the GKSL generator as "completely-positive trace-preserving (CPTP) evolution".
- Why this is an error: Chizat-Peyré-Schmitzer-Vialard (2018) construct and analyse the unbalanced/WFR transport distance; they do not prove that a reaction-diffusion master equation is a WFR gradient flow of relative entropy (that line of results is Kondratyev-Monsaingeon-Vorotnikov 2016 and Gallouët-Monsaingeon 2017). Presenting the wrong paper as the source of a theorem is a substantive misattribution in a paragraph whose stated purpose is "to be precise about what is rigorous". Separately, trace preservation is part of the definition of GKSL form (CPTP upstream); a non-trace-preserving generator is not GKSL, so listing it as optional contradicts the anchor.
- Impact on downstream results: none mathematical; misleads readers checking the literature.
- Fix guidance (step-by-step):
  1. Cite Kondratyev-Monsaingeon-Vorotnikov (2016) and Gallouët-Monsaingeon (2017) for the continuous case.
  2. Delete "(optional)" and the following sentence, or state that mass non-conservation requires leaving the GKSL class (e.g. adding a non-Hermitian term).
- Required new assumptions/permits (if any): none.
- Validation plan: Check `references.bib` entries for the cited works.

### [E-017] Definition text says the policy outputs $(v,r)$ while the code is an action-conditioned world model
(was F-020)
- Location: `def-wfr-world-model` (lines 450-536, esp. 453 and 494-499)
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 453): "The policy outputs a generalized velocity field $(v,r)$ to minimize the WFR path length to the target distribution (goal)." followed by `class WFRWorldModel(nn.Module)` whose `forward(z_t, mass_t, action_t)` consumes an action.
- Upstream anchor: not applicable; defined in this chapter.
- Why this is an error: A world model maps (state, action) to next state; a policy maps state to action. The formal sentence attributes $(v,r)$ to the policy and adds a goal-directed objective that the code neither receives (no goal input) nor optimises (it integrates one Euler step). The definition and its implementation describe different objects.
- Impact on downstream results: `04_faq.md:485` cites this definition as what Node 23 verifies.
- Fix guidance (step-by-step):
  1. Replace with "The world model outputs a generalized velocity field $(v,r)$ conditioned on the action; the policy selects actions to minimise the WFR path length to the goal distribution."
- Required new assumptions/permits (if any): none.
- Validation plan: Compare the signature of `forward` with the definition text.

### [E-018] $\lambda^{(\ell)}\propto\sigma^{(\ell)}$ has the wrong monotonicity given the upstream definition of $\sigma^{(\ell)}$
(was F-014)
- Location: `def-scale-dependent-teleportation-cost` (lines 582-605, esp. 586-593)
- Severity: Moderate
- Type: Definition mismatch (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$\lambda^{(\ell)}\propto\sigma^{(\ell)}$ (jump cost scales with residual variance) where $\sigma^{(\ell)}$ is the scale factor from Definition `def-the-rescaling-operator-renormalization`. Layer 0 (Bulk / IR): High $\lambda^{(0)}$ ... Layer $L$ (Texture / UV): Low $\lambda^{(L)}$."
- Upstream anchor: `docs/source/1_agent/03_architecture/01_compute_tiers.md:2018-2028` "$x^{(\ell+1)} = \dfrac{z_{\text{tex}}^{(\ell)}}{\sigma^{(\ell)}+\epsilon},\quad \sigma^{(\ell)}=\sqrt{\mathrm{Var}(z_{\text{tex}}^{(\ell)})+\epsilon}$ ... renormalize the residual to unit variance"; `:2030-2036` "Define $\Pi^{(\ell)}:=\prod_{j=0}^{\ell-1}\sigma^{(j)}$ ... $\hat x = \sum_\ell \Pi^{(\ell)}\hat x^{(\ell)} + \Pi^{(L)}x^{(L)}$".
- Why this is an error: Each block's input $x^{(\ell)}$ ($\ell\ge1$) is renormalised to unit variance, so $\sigma^{(\ell)}$ is the fraction of a unit-variance input left unexplained by block $\ell$, a number in $(0,1]$ that is not monotone in $\ell$. Per the upstream RG narrative (`:1983`, `:2170`: "Block $L$ contains irreducible noise"), the last blocks explain little, so $\sigma^{(\ell)}\to1$ at the UV end, while $\sigma^{(0)}$ is in raw observation units and incomparable with the rest. Hence $\lambda^{(\ell)}\propto\sigma^{(\ell)}$ makes $\lambda$ largest at the texture layers, the opposite of the stated interpretation. The quantity carrying the absolute scale of layer $\ell$ (and decreasing with depth) is the cumulative factor $\Pi^{(\ell)}$ from `def-total-reconstruction`.
- Impact on downstream results: The $\Lambda^{(\ell)}\sim1/(\lambda^{(\ell)})^2$ correspondence and the bulk/boundary interpretation (lines 598-603, 615-627) inherit the reversed ordering.
- Fix guidance (step-by-step):
  1. Set $\lambda^{(\ell)}\propto\Pi^{(\ell)}=\prod_{j<\ell}\sigma^{(j)}$ and note that $\lambda^{(0)}\propto\Pi^{(0)}=1$ is then the largest scale.
  2. Note that a per-layer $\Lambda^{(\ell)}$ conflicts with "$\Lambda$ and $\kappa$ are constants" in `thm-capacity-constrained-metric-law` (`01_metric_law.md:188`) unless the law is applied per layer.
- Required new assumptions/permits (if any): none.
- Validation plan: Instantiate a three-block hierarchy with $\sigma^{(0)}=10$, $\sigma^{(1)}=0.5$, $\sigma^{(2)}=0.9$ and check which of $\sigma^{(\ell)}$ or $\Pi^{(\ell)}$ decreases with $\ell$.

### [E-019] WFR stress-energy tensor conflicts with the metric law's explicit Risk Tensor (different definition, different units)
(was F-015)
- Location: Connection to Einstein Equations (line 638) and `thm-wfr-stress-energy-tensor-variational-form` (lines 640-683, esp. 669-671); Appendix C section C.4
- Severity: Major
- Type: Definition mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter (and `docs/source/1_agent/10_appendices/03_wfr_tensor.md:121-131`)
- Claim (verbatim, lines 638, 669-671): "The WFR dynamics provide the **stress-energy tensor** $T_{ij}$ that drives curvature in Theorem {prf:ref}`thm-capacity-constrained-metric-law`." ... "$T_{ij}=\rho\,v_iv_j+P\,G_{ij}$".
- Upstream anchor: `docs/source/1_agent/05_geometry/01_metric_law.md:185-188` "$R_{ij}-\tfrac12 R\,G_{ij}+\Lambda G_{ij}=\kappa\,T_{ij}$, where ... $T_{ij}$ is the **total Risk Tensor** induced by the reward field"; `:219-242` `def-extended-risk-tensor`: "$T_{ij}=T^{\text{gradient}}_{ij}+T^{\text{Maxwell}}_{ij}$, $T^{\text{gradient}}_{ij}=\partial_i\Phi\,\partial_j\Phi-\tfrac12G_{ij}\|\nabla\Phi\|_G^2$, ... *Units:* $[T_{ij}]=\mathrm{nat}^2/[z]^2$." Appendix C `10_appendices/03_wfr_tensor.md:124-130` substitutes the WFR tensor into the same equation "with the WFR stress-energy acting as the risk tensor".
- Why this is an error: The same symbol $T_{ij}$ on the right-hand side of the same headline field equation receives two different definitions: upstream it is a functional of the reward 1-form $(\Phi,\mathcal F)$ with units $\mathrm{nat}^2/[z]^2$; here it is a functional of the belief flow $(\rho,v,r)$ with units $[\rho][v]^2$ (a density times a squared velocity). Neither chapter says the two are to be added, nor that one replaces the other, nor how a single $\kappa$ could convert both to curvature units. Appendix C silently overrides `def-extended-risk-tensor`. As it stands the metric law has an ambiguous source term.
- Impact on downstream results: `thm-capacity-constrained-metric-law` is a headline result; every downstream use of "the Risk Tensor" (Maxwell stress in `06_fields`, $\mathcal L_{\text{EFE}}$ at `01_metric_law.md:297`) depends on which $T_{ij}$ is meant.
- Fix guidance (step-by-step):
  1. State explicitly that the total source is $T_{ij}=T^{\text{risk}}_{ij}+\kappa_{\mathrm{WFR}}\,T^{\text{WFR}}_{ij}$ with a separate coupling of the right units, and add this to `def-extended-risk-tensor`; or
  2. Demote the WFR tensor to an analogy and remove "provide the stress-energy tensor that drives curvature in Theorem ...", and correct Appendix C section C.4 accordingly.
- Required new assumptions/permits (if any): a new coupling constant $\kappa_{\mathrm{WFR}}$ with units converting $[\rho][v]^2$ to $\mathrm{nat}^2/[z]^2$, if option 1.
- Validation plan: Unit-check both sides of the metric law with the chosen source term.

### [E-020] The pressure term depends on holding the metric-dependent density fixed, which does not preserve belief mass
(was F-016)
- Location: `thm-wfr-stress-energy-tensor-variational-form`, definition of $T_{ij}$ "holding $\rho,v,r$ fixed" and proof sketch (lines 658-681, esp. 661-675); Appendix C section C.2 (`10_appendices/03_wfr_tensor.md:32, 59-96`)
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$T_{ij}:=-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal L_{\mathrm{WFR}})}{\delta G^{ij}}$ (holding $\rho,v,r$ fixed). Then $T_{ij}=\rho v_iv_j+P\,G_{ij}$, $P=\tfrac12\rho(\|v\|_G^2+\lambda^2r^2)$, ... with reaction contributing an additive pressure term."
- Upstream anchor: `docs/source/1_agent/05_geometry/01_metric_law.md:102-104` "$\rho(z,s)$ ... **defined with respect to the Riemannian volume measure** $d\mu_G=\sqrt{|G|}\,dz^n$"; this chapter line 40 "a *measure* (belief state) $\rho_s\in\mathcal M^+(\mathcal Z)$" and line 111 integrates against $d\rho_s(z)$.
- Why this is an error: The algebra of the stated convention is correct (re-derived by both reviewer and verifier): $\delta\|v\|_G^2=-v_iv_j\delta G^{ij}$ and $\delta\sqrt{|G|}=-\tfrac12\sqrt{|G|}G_{ij}\delta G^{ij}$ give $T_{ij}=\rho v_iv_j+\mathcal L_{\mathrm{WFR}}G_{ij}$. But the convention is not the physically meaningful one for this framework. The belief is a measure; the density $\rho$ w.r.t. $d\mu_G$ is $G$-dependent. Holding the density fixed while varying $G$ changes the belief measure and its total mass ($\delta\int\rho\,d\mu_G=-\tfrac12\int\rho G_{ij}\delta G^{ij}d\mu_G\neq0$), i.e. it varies the metric together with the agent's belief. Holding the belief measure $\tilde\rho:=\rho\sqrt{|G|}$ fixed instead gives $\delta(\tfrac12\tilde\rho(G_{ij}v^iv^j+\lambda^2r^2))=-\tfrac12\tilde\rho\,v_iv_j\delta G^{ij}$, hence $T_{ij}=\rho v_iv_j$ with no pressure and no reaction contribution at all. Holding the momentum covector $p_i=G_{ij}v^j$ fixed instead of $v^i$ flips the sign of the kinetic term. The headline conclusion ("reaction contributes an additive pressure", implications at lines 674, 690, 720-722, the "cosmological constant"/"capacity stress" narrative) is therefore an artefact of an unstated choice of what is held fixed.
- Impact on downstream results: lines 695-722 (Physics Isomorphism box and implications) and Appendix C.
- Fix guidance (step-by-step):
  1. State the convention explicitly as a hypothesis ("$\rho$ is held fixed as a density w.r.t. $d\mu_G$ and $v^i$ as a contravariant field").
  2. Justify it physically, or switch to the measure-fixed convention and drop the pressure claims.
  3. Note that the result is convention dependent.
- Required new assumptions/permits (if any): the chosen variational convention, stated as a hypothesis of the theorem.
- Validation plan: Recompute $T_{ij}$ under the density-fixed, measure-fixed, and momentum-fixed conventions and confirm the three differ as described.

### [E-021] "Axiom D" does not exist in Volume 1; $r<0$ is mass destruction, not entropy production; stray closing fence
(was F-017)
- Location: "Consistency with existing losses" table (lines 732-741)
- Severity: Minor
- Type: Citation / reference error (secondary: Typo)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 738): "| Dissipation (Axiom D) | $r<0$ (entropy production) | Compatible |" followed by a bare `:::` on line 741.
- Upstream anchor: `grep -rn "Axiom D" docs/source/1_agent` returns only line 738; no axiom named "D" or "Dissipation" is defined anywhere in Volume 1.
- Why this is an error: The row cites a non-existent axiom. Mathematically, $r<0$ is local annihilation of belief mass; entropy production in the chapter's own framework is governed by the gradient-flow structure of Section 20.5 ($\tfrac{d}{ds}H(p\|\pi)\le0$), a property of the transport (Maas) part, not of the sign of $r$ (mass annihilation can decrease or increase entropy). Directive structure check: the theorem closes at 683, the four-colon admonition at 717, the feynman-prose at 730; the `:::` on line 741 closes nothing and will render as literal text or break the following block.
- Impact on downstream results: none mathematical.
- Fix guidance (step-by-step):
  1. Point to the actual dissipation/entropy result, or delete the row.
  2. Replace "$r<0$ (entropy production)" with a correct statement.
  3. Delete the stray `:::` at line 741.
- Required new assumptions/permits (if any): none.
- Validation plan: Build the chapter and inspect the rendered table.

### [E-022] "Convex optimisation" claim holds only in the balanced/undriven momentum variables with fixed endpoints and metric
(was F-022)
- Location: Sasaki vs WFR table row "Optimization: Convex (generalized geodesics)" (line 771; also 778); summary item 3 (line 901)
- Severity: Note
- Type: Scope restriction
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "Convex (generalized geodesics)" / "Finding optimal belief trajectories is a convex problem".
- Upstream anchor: not applicable.
- Why this is an error: The Benamou-Brenier problem is convex in the variables $(\rho, m=\rho v, \mu=\rho r)$ with fixed endpoints; it is not convex in $(\rho,v,r)$ as written, and once the goal distribution, the learned state-dependent metric $G(z)$, or the world-model parameters are themselves optimised, convexity is lost.
- Impact on downstream results: none.
- Fix guidance (step-by-step):
  1. Add the scope qualifier "in momentum variables $(\rho,\rho v,\rho r)$ with fixed endpoints and fixed metric".
- Required new assumptions/permits (if any): none.
- Validation plan: text check.

### [E-023] Node number 23 is already assigned to NEPCheck in the Sieve
(was F-018; severity adjusted from Moderate to Minor by the verifier)
- Location: Node 23: WFRCheck (lines 865-880)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "| **23** | **WFRCheck** | **World Model** | **Dynamics Consistency** | ... | $O(BK)$ |"
- Upstream anchor: `docs/source/1_agent/02_sieve/01_diagnostics.md:121` "| **23** | **NEPCheck ($\mathrm{NEP}$)** | **Belief / Boundary** | **Update vs Evidence** | ... | $O(B|\mathcal K|)$ ✓ |"; `docs/source/1_agent/04_control/02_belief_dynamics.md:384` "### Update vs Evidence Check (Node 23) and Metric Speed Limit (Node 24)"; `10_appendices/04_faq.md:485` repeats "Node 23 (WFRCheck)".
- Why this is an error: The Sieve's node table is the registry the chapter claims to follow ("Following the diagnostic node convention (Section 3.1)"). Two different diagnostics with different proxies and components share index 23; any downstream table, trigger map or code keyed on node number is ambiguous. WFRCheck appears nowhere in the Sieve's own node list. This is a registry conflict that does not alter any mathematical statement, hence Minor.
- Impact on downstream results: FAQ (`04_faq.md:485`), any implementation indexing nodes.
- Fix guidance (step-by-step):
  1. Assign WFRCheck an unused node number and add it to the master table in `02_sieve/01_diagnostics.md`, or document it as a sub-check of an existing node.
  2. Update `04_faq.md:485`.
- Required new assumptions/permits (if any): none.
- Validation plan: grep for "Node 23" across `docs/source/1_agent` and confirm a single assignment.

## Scope restrictions and clarifications
- The mechanical cross-reference pre-pass reports no dangling `{prf:ref}` targets and no duplicate labels in this file; all findings are semantic.
- E-001, E-002, E-003, E-019 and E-020 interact: the choice of which action is "the" WFR action (with or without $\mathbf A$, with or without the factor $\tfrac12$) determines the correct form of the gauge remark, the unit table, and the stress-energy tensor. They should be resolved together.
- E-004 is inherited from `06_fields/02_reward_field.md`; the chapter is consistent with its source, but the source is wrong when $H^1(\mathcal Z)\neq0$.
- E-013 and E-014 have parallel statements at `04_control/02_belief_dynamics.md:290`; fixing one chapter without the other leaves the inconsistency in place.
- The numerical checks for E-001 and E-008 (verifier script `wfr_verify.py`) use the per-particle reduction of the action without the $\mathbf A$ term and the normalisation of line 111; if the normalisation is changed (E-003), the constants $2\sqrt2\lambda$ and $\pi\lambda$ rescale accordingly.

## Open questions
- Should the driven (vector-potential) action be retained at all in the geometry chapter, or moved entirely to `04_equations_motion.md` where the Lorentz term is introduced with its own coupling $\beta_{\text{curl}}$?
- Is the intended source term of the capacity-constrained metric law the reward Risk Tensor, the WFR stress-energy tensor, or their sum? The answer determines the fix for E-019 and whether Appendix C needs to be rewritten.
- Which convention for the metric variation (density fixed vs measure fixed) does the framework intend? The physical narrative around pressure and the cosmological constant depends on it (E-020).
- Is $\lambda$ meant to be an intrinsic property of $(\mathcal Z, G)$, an atlas property, or a tunable hyperparameter? The three current characterisations (E-009, E-010) point in different directions.

## Rejected candidate findings
- None. All 22 stage-1 findings were confirmed or adjusted by the verifier; four were adjusted (F-001, F-010, F-011: criterion External to Framework; F-018: severity Moderate to Minor).
