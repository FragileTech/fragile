# Mathematical Review: docs/source/1_agent/01_foundations/02_control_loop.md

## Metadata
- Reviewed file: docs/source/1_agent/01_foundations/02_control_loop.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1523 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/01_foundations/01_definitions.md` (reward as 1-form and Hodge split, lines 250-262; units and dimensional conventions, lines 365-392)
  - `docs/source/1_agent/06_fields/02_reward_field.md` (`thm-hodge-decomposition`, lines 180-200; `thm-the-hjb-helmholtz-correspondence`, lines 325-361)
  - `docs/source/1_agent/06_fields/01_boundary_interface.md` (`ax-motor-texture-firewall`, lines 403-409)
  - `docs/source/1_agent/02_sieve/01_diagnostics.md` (node table, lines 95-116; scaling coefficients, lines 188-205)
  - `docs/source/1_agent/02_sieve/03_failures_interventions.md` (failure-mode table, lines 55-66)
  - `docs/source/1_agent/03_architecture/01_compute_tiers.md` (lines 1360-1372) and `02_disentangled_vae.md` (lines 55-66) for the TopoEncoder split
  - `docs/source/1_agent/05_geometry/02_wfr_geometry.md` (`def-the-wfr-action`, lines 100-120)
  - `docs/source/1_agent/07_cognition/01_supervised_topo.md` (lines 200-236) and `07_cognition/06_causality.md` (lines 300-312) for the latent equations of motion
  - `docs/source/1_agent/10_appendices/02_parameters.md` (lines 40-50)

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 8
- Minor: 12
- Notes: 1
- Primary themes:
  1. Sign and symbol conventions. The chapter works in a cost convention but reuses the reward-side Hodge symbols $\Phi$, $\Psi$, $\eta$, $A$ of `thm-hodge-decomposition` for their negatives, so the bare statement "$V=\Phi$" contradicts the upstream definition and the chapter's own $\rho_c=-\rho_r$ (E-001). The natural-gradient step changes sign between displays (E-009).
  2. The chapter violates its own anti-mixing rules. $\dot V$ is defined through a metric-raised vector (E-005), a "geodesic distance" is given by a constant quadratic form (E-015), and four dimensionless subsystem diagnostics are presented as a metric on $\mathcal Z$ (E-010).
  3. Heuristic derivations presented as formal results. The action functional's value term is a boundary term that cannot steer the Euler–Lagrange flow (E-008); the HJB display is the undiscounted fixed-policy identity yet is forward-linked to a screened Poisson equation (E-014); the closure-defect/CMI equivalence holds in one direction only (E-016); the variance–curvature lemma has no proof and conflicts with an earlier claim (E-019).
  4. Unit bookkeeping. Nats are tracked for $V$ but dropped for $\log\pi$ (E-011), $\beta_{\rm cpl}=1/T_c$ is ill-typed (E-018), the transport field lacks a rate constant (E-020), and the WFR one-step proxy is off by a factor $\Delta s$ (E-013).
  5. Reference hygiene: a dead section pointer and two unrendered `{ref}` roles (E-002), an axiom cited as a theorem (E-004), and Sieve node/mode names that do not match the Sieve chapter (E-021).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Anatomy table and covariant gradient, 141-147; HJB Correspondence, 1015-1017; Helmholtz forward reference, 1069 | Moderate | Notation conflict | Framework | this chapter | Reward-side Hodge symbols $\Phi$, $A=\delta\Psi+\eta$ reused for their cost-side negatives; "$V=\Phi$" contradicts the upstream definition of $\Phi$ |
| E-002 | Latent factorisation, 242; Metatheorems, 555; Closure Defect remark, 1123 | Minor | Citation / reference error | Framework | this chapter | "Sec. 29.13" has no target; two `{ref}` roles lack backticks |
| E-003 | Residual Split, 342-360 | Minor | Algorithm mismatch | Framework | this chapter | Chart-centre term $c_{\rm bar}$ of the implemented TopoEncoder omitted |
| E-004 | Design Goal remark, 391 | Minor | Citation / reference error | Framework | this chapter | `ax-motor-texture-firewall` is an axiom, cited as a theorem and credited with a duality it does not state |
| E-005 | Connection to RL #18, 598; implementation paragraph, 765-772 | Moderate | Definition mismatch | Framework | this chapter | $\dot V$ defined via metric contraction $\nabla_AV^\top\dot z$ and a second $M^{-1}$; contradicts Rule #2 and is not $dV(\dot z)$ |
| E-006 | Bridge table, 584; Connection to RL #18, 595-598 | Minor | Scope restriction | Framework | this chapter | $\dot V\le-\lambda V$ is not invariant under the constant shift allowed for $V$; normalisation not stated |
| E-007 | Connection to RL #18, 601; Regularity assumption 4, 1147 | Note | Scope restriction | Framework | this chapter | "enforces $\|\nabla_AV\|>\epsilon$" reads as a global bound incompatible with an interior equilibrium; upstream node is a residual |
| E-008 | Geometry-Regularized Objective, 635-641; Connection to RL #2, 646-671 | Moderate | Conceptual | Framework | this chapter | $\int\dot V\,dt$ is a boundary term; the functional yields pure geodesics and no descent dynamics |
| E-009 | Upgrade table, 729; Connection to RL #1, 740, 751; metric table, 1051 | Minor | Typo | Framework | this chapter | Natural-gradient step written with and without the minus sign under a cost convention |
| E-010 | Practical Diagonal Sensitivity Metric, 809-823 | Moderate | Dimensional mismatch | Framework | this chapter | $\mathrm{diag}(\alpha,\beta_\pi,\gamma_{\rm wm},\delta)$ is not a $(0,2)$-tensor on $\mathcal Z$ |
| E-011 | Definition Complete Latent Space Metric, 843 | Minor | Dimensional mismatch | Framework | this chapter | Units of $G_\pi$ and $\lambda_G$ do not follow the tracked-nat convention |
| E-012 | Dimensional Verification, 852 | Minor | Proof gap / omission | Framework | this chapter | $G\succ0$ stated as a result; it is Assumption 2 |
| E-013 | Connection to RL #3, quadratic proxy, 964-972 | Moderate | Computational error | Framework | this chapter | One-step WFR proxy needs $(\Delta s)^2$, not $\Delta s$ |
| E-014 | HJB Correspondence, 1008-1017 and 1061-1069 | Moderate | Scope restriction | Framework | this chapter | Undiscounted, fixed-action identity called "the HJB equation" and linked to a discounted, diffusive Helmholtz form |
| E-015 | Metric table, 1052 | Minor | Definition mismatch | Framework | this chapter | Geodesic distance given by a constant quadratic form, against Rule #3 |
| E-016 | Closure Defect, Computational Meaning, 1130 | Moderate | Invalid inference | Framework | this chapter | $\delta_{\rm CE}>0$ is not equivalent to $I(K_{t+1};Z_t\mid K_t,K^{\rm act}_t)>0$ |
| E-017 | Trinity table, 179; Definition Local Conditioning Scale, 1278-1281 | Minor | Notation conflict | Framework | this chapter | $\Theta$ is both the parameter manifold and a scalar field on $\mathcal Z$ |
| E-018 | Definition Local Conditioning Scale, 1284-1288; summary table, 1425-1426 | Minor | Dimensional mismatch | Framework | this chapter | $\beta_{\rm cpl}$ (nat/[z]$^2$) set equal to $1/T_c$ (dimensionless) |
| E-019 | Lemma Variance–Curvature Correspondence, 1292-1307; Connection to RL #10, 520 | Moderate | Proof gap / omission | Framework | this chapter | No proof, mismatched shapes, contradicts $\Sigma_\pi\propto G^{-1}$ at line 520, ill-typed defect functional |
| E-020 | Definition Transport Field, 1343-1349 | Minor | Dimensional mismatch | Framework | this chapter | $[G^{ij}\partial_jV]=[z]$, not $[z]$/time |
| E-021 | Corollary Boundary filter interpretation, 1481-1489 | Minor | Citation / reference error | Framework | this chapter | Wrong Sieve node names; Mode B.E misassigned |

## Detailed findings

### [E-001] Reward-side Hodge symbols reused for their cost-side negatives; "$V=\Phi$" contradicts the upstream definition (was F-001)
- Location: Anatomy table and covariant gradient paragraph, lines 141-147; HJB Correspondence, lines 1015-1017; Helmholtz forward reference, line 1069.
- Severity: Moderate
- Type: Notation conflict (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 102: "This chapter uses a cost convention (lower is better)." Line 141: "(identify $V=\Phi$ for the exact component)". Lines 146-147: "$A := \delta\Psi + \eta$ is the non-conservative component of the cost 1-form $\mathcal{C} = dV + A$ (reward convention: $\mathcal{R}=-\mathcal{C}=d(-V)+(-A)$; conservative case: $A=0$)". Line 1016: "The exact component of $\mathcal{C}$ defines a scalar potential $\Phi$ that the critic tracks (so $V=\Phi$ up to a constant)". Line 1069: "$-\Delta_G V + \kappa^2 V = \rho_c$ with $\rho_c := -\rho_r$".
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:186-196` (`thm-hodge-decomposition`): "$\mathcal{R} = d\Phi + \delta \Psi + \eta$ ... We identify $\Phi$ with the critic value $V$ (the exact component), so $d\Phi = dV$ ... Define the non-exact component $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $\mathcal{F} = dA$." `:163`: "$d\Phi$ ... points from low value to high value." `:329-333`: the Bellman condition $V(z)=\mathbb E[r+\gamma V(z')]$ yields $-\Delta_G V+\kappa^2V=\rho_r$. `01_foundations/01_definitions.md:259-262` states the same reward-side decomposition.
- Why this is an error: The two chapters use different sign conventions. Upstream is in reward convention: $\Phi$ is the reward potential (higher is better), $V:=\Phi$, and $A=\delta\Psi+\eta$ is the non-exact part of $\mathcal R$. This chapter is in cost convention, and line 147 Hodge-decomposes the cost 1-form with the same letters: $\mathcal C=dV+A$, hence $\mathcal R=-\mathcal C=d(-V)+(-A)$. Read on its own terms line 147 is self-consistent, and the map to upstream is $V_{\rm cost}=-\Phi_{\mathcal R}$, $A_{\mathcal C}=-A_{\mathcal R}$; the forward reference at line 1069 is exactly the upstream Helmholtz equation under $V\mapsto-V$ ($-\Delta_GV+\kappa^2V=-\rho_r=\rho_c$), which confirms that the chapter intends $V=-\Phi_{\mathcal R}$. The error is that the chapter never says so. It reuses the symbols fixed by the labelled theorem (including the defining formula $A:=\delta\Psi+\eta$) for their negatives, and at lines 141 and 1016 writes "$V=\Phi$", which against the upstream definition of $\Phi$ asserts that the cost-to-go equals the reward potential, in contradiction with lines 102, 147 and 1069. Line 1016 does locally redefine $\Phi$ as the exact potential of $\mathcal C$, so the sentence is internally rescued, but the book then carries two objects named $\Phi$ (and two named $A$, $\mathcal F=dA$) of opposite sign. A first-stage claim that the two definitions "force $A=0$" was checked and does not hold: it equates the chapter's $A$ with the upstream $A$, which the chapter's parenthetical explicitly negates.
- Impact on downstream results: No result in this chapter depends on the sign, but the ambiguity propagates to every downstream use of $\Phi$, $A$, $\mathcal F$ and $\rho$ (07_cognition equations of motion with $\beta_{\rm curl}\mathcal F$, 06_fields sources, 10_appendices/06_losses.md:424 "$V=\Phi$"). A reader cannot determine from this chapter which sign of the curl term or source density applies.
- Fix guidance:
  1. Keep $\Phi$, $\Psi$, $\eta$, $A$ as the Hodge data of $\mathcal R$ (as upstream) and state once, at line 147: "$V:=-\Phi$ (cost-to-go), so $\mathcal C=-\mathcal R=dV-A$ with $A:=\delta\Psi+\eta$ the non-exact part of $\mathcal R$ and $\mathcal F=dA$."
  2. Replace "$V=\Phi$" by "$V=-\Phi$" at line 141 and line 1016, or, if the chapter prefers cost-side symbols, rename them ($\Phi_{\mathcal C}:=-\Phi$, $A_{\mathcal C}:=-A$) and state the map.
  3. Keep $d_AV:=dV-A$ only if the relative sign matches the chosen $A$; note explicitly that $\nabla_AV$ is the gauge-covariant gradient used in the dynamics $\dot z=-G^{-1}\nabla_AV$, not the raised cost 1-form.
- Required new assumptions/permits: none.
- Validation plan: after the edit, check that line 1069 ($\rho_c=-\rho_r$), `07_cognition/01_supervised_topo.md:207-209` and `06_causality.md:307` all read consistently under the single stated map; grep `V=\Phi`, `V = \Phi` across `docs/source/1_agent` and reconcile `10_appendices/06_losses.md:424`.

### [E-002] Stale numeric section pointer and malformed `{ref}` directives (was F-002)
- Location: line 242; line 555; line 1123 (also bare "Section 2.6/2.7", "Sections 3-6 and 15" at 103, 633, 664, 1331, 1408, 1429).
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 242: "gauge fiber in Sec.\ 29.13"; line 555: "(see Meta-Theorem: Levin-Search in {ref}sec-appendix-a-full-derivations)"; line 1123: "(see Meta-Theorem: Micro-Macro Consistency in {ref}sec-appendix-a-full-derivations)".
- Upstream anchor: `grep -rn "gauge fib" docs/source/1_agent` returns only this line; the label `sec-appendix-a-full-derivations` exists and is used correctly at `05_geometry/01_metric_law.md:177`.
- Why this is an error: no section 29.13 exists in the current book; the two `{ref}` roles lack backticks, so MyST renders them as literal text and the xref scanner does not see them.
- Impact on downstream results: navigation only.
- Fix guidance: replace "Sec. 29.13" by a labelled reference to the section that treats $z_n$ as a gauge fibre (or delete the parenthetical); wrap the two appendix references as ``{ref}`sec-appendix-a-full-derivations` ``.
- Required new assumptions/permits: none.
- Validation plan: rebuild and confirm the roles resolve; rerun the xref pre-pass.

### [E-003] TopoEncoder geometric latent omits the chart-centre term (was F-003)
- Location: The Residual Split, lines 342-360 (also line 325).
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 325: "$K_{\text{code},i}(x) := \arg\min_c \|v(x)-e_{i,c}\|_2^2$"; lines 355-360: "$z_{\text{geo}} = z_q^{\text{st}} + z_n$, $z_q^{\text{st}} := v(x) + \operatorname{sg}[z_q(x)-v(x)]$, so reconstruction uses the discrete macro code plus structured nuisance."
- Upstream anchor: `docs/source/1_agent/03_architecture/01_compute_tiers.md:1366-1369`: "`c_bar = router_weights @ chart_centers`, `v_local = v - c_bar`, then per-chart VQ ... `z_geo = c_bar + z_q_st + z_n`"; `02_disentangled_vae.md:58-62`: "$z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n$ where $c_{\mathrm{bar}}$ is the chart center mixture".
- Why this is an error: both texts claim to describe the shutter in `topoencoder.py` (line 310); the architecture chapter quantises the chart-local value $v-c_{\rm bar}$ and adds $c_{\rm bar}$ back, this chapter quantises $v$ directly and omits $c_{\rm bar}$. The closing sentence also omits that the decoder receives $z_{\rm tex}$ (line 461).
- Impact on downstream results: descriptive only; the shutter loss (lines 463-471) is unaffected.
- Fix guidance: write $z_{\rm geo}=c_{\rm bar}+z_q^{\rm st}+z_n$ with $K_{\text{code},i}(x):=\arg\min_c\|v(x)-c_{\rm bar}-e_{i,c}\|^2$, or label this block a simplified variant pointing to `03_architecture/02_disentangled_vae.md`; amend the closing sentence to say texture enters as a separate residual at the decoder.
- Required new assumptions/permits: none.
- Validation plan: compare against `PrimitiveAttentiveAtlasEncoder.forward`.

### [E-004] Axiom cited as a theorem and credited with a symplectic duality (was F-004)
- Location: Design Goal remark, line 391.
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The motor texture ... is dual to visual texture via the symplectic form (Theorem {prf:ref}`ax-motor-texture-firewall`)."
- Upstream anchor: `docs/source/1_agent/06_fields/01_boundary_interface.md:403-409`: ":::{prf:axiom} Motor Texture Firewall ... $\partial_{z_{\text{tex,motor}}}\dot z=0, \; \partial_{z_{\text{tex,motor}}}u_\pi=0$."
- Why this is an error: the target is an assumption stating a decoupling condition, not a proved duality.
- Impact on downstream results: overstates the epistemic status of the motor/visual duality.
- Fix guidance: "(Axiom {prf:ref}`ax-motor-texture-firewall`; the symplectic duality is stated in {ref}`sec-motor-texture-the-action-residual`)".
- Required new assumptions/permits: none.
- Validation plan: check the rendered cross-reference label.

### [E-005] $\dot V$ defined with a metric contraction, contradicting Rule #2 and the meaning of $\dot V$ (was F-005)
- Location: Connection to RL #18, line 598; implementation paragraph, lines 765-772.
- Severity: Moderate
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 598: "$\dot{V}(z) := \nabla_A V(z)^\top \dot{z} \le -\lambda_{\text{Lyap}} V(z)$"; line 765: "The covariant derivative uses a diagonal inverse metric $M^{-1}(z)$ to scale $\dot V$"; line 769: "$\dot{V}_M = \nabla_A V(z)^\top M^{-1}(z) \frac{\Delta z}{\Delta t}$".
- Upstream anchor: chapter's own line 145: "$\nabla_A V := G^{-1}(dV - A)$"; line 1032: "$\mathcal{L}_f V = dV(f)$ ... NO metric $G$ appears"; line 1058: "The Lie derivative $\mathcal{L}_f V = dV(f)$ is a pairing, not an inner product".
- Why this is an error: with $\nabla_AV$ a vector, $\nabla_AV^\top\dot z=(dV-A)^\top G^{-1}\dot z$ is a Euclidean contraction of two tangent vectors, coordinate dependent, and equals the pairing $d_AV(\dot z)$ only when $G=I$. Line 769 inserts a further $M^{-1}$, so the quantity is doubly raised. Neither expression is $\frac{d}{dt}V(z(t))=dV(\dot z)$; even the pairing $d_AV(\dot z)$ differs from it by the circulation work $A(\dot z)$, so the display does not certify decrease of $V$.
- Impact on downstream results: Node 7 Lyapunov residuals in `02_sieve` and the covariant Lyapunov signal in 07_cognition inherit whichever reading is intended.
- Fix guidance:
  1. Define $\dot V:=dV(\dot z)=\partial_iV\,\dot z^i$ (metric free).
  2. If a gauge-covariant rate is wanted, name it $\dot V_A:=d_AV(\dot z)=\dot V-A(\dot z)$ and say it differs from $\dot V$ by circulation.
  3. Remove $M^{-1}$ from line 769, or rename the quantity as the preconditioned descent rate $-\|d_AV\|^2_{M^{-1}}$ obtained by substituting $\dot z=-M^{-1}d_AV$.
- Required new assumptions/permits: none.
- Validation plan: check that for $G\neq I$ the redefined $\dot V$ is invariant under a linear change of latent coordinates.

### [E-006] Exponential Lyapunov condition is not invariant under the constant shift allowed for $V$ (added by verifier)
- Location: Bridge table, line 584; Connection to RL #18, lines 595-598; cf. line 1016.
- Severity: Minor
- Type: Scope restriction (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 584: "Ensure stability: $\dot{V}(z) \le -\lambda_{\text{Lyap}} V(z)$"; line 598 (same inequality); line 1016: "(so $V=\Phi$ up to a constant)".
- Why this is an error: under $V\mapsto V+c$ the left side $\dot V$ (and $\nabla_AV$) is unchanged while the right side shifts by $-\lambda c$. The condition therefore holds or fails depending on an arbitrary constant and cannot hold near a minimum of $V$ if $V<0$ there. The standard control-Lyapunov setting requires $V\ge0$ with $V=0$ on the target set, and no such normalisation appears in the Regularity Conditions (lines 1143-1148).
- Impact on downstream results: every use of $\dot V\le-\lambda V$ (Node 7 residuals; `10_appendices/02_parameters.md:42`).
- Fix guidance: state the normalisation "$V\ge0$, $V=0$ on the goal/absorbing set (this fixes the constant in line 1016)", or use the shift-invariant form $\dot V\le-\lambda\,(V-V_{\min})$.
- Required new assumptions/permits: the normalisation above (an explicit convention, not a new hypothesis).
- Validation plan: confirm downstream residuals use the same normalised $V$.

### [E-007] "Enforces $\|\nabla_AV\|>\epsilon$" reads as a global bound (was F-006)
- Location: line 601; Regularity assumption 4, line 1147.
- Severity: Note
- Type: Scope restriction
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The Sieve (Node 7: StiffnessCheck) enforces $\|\nabla_A V\| > \epsilon$".
- Upstream anchor: `docs/source/1_agent/02_sieve/01_diagnostics.md:100`: Node 7 residual "$\max(0, \epsilon - \Vert \nabla_A V \Vert)$".
- Why this is a concern: a Lyapunov function has $dV=0$ at the equilibrium it certifies, and on a compact $\mathcal Z$ a $C^2$ function has an interior critical point unless the minimum sits on $\partial\mathcal Z$; a pointwise lower bound everywhere is impossible. Upstream states a soft batch residual, so nothing is broken; the wording is what suggests a global constraint.
- Impact on downstream results: none if read as a monitor.
- Fix guidance: "monitors that the batch-averaged $\|\nabla_AV\|$ stays above $\epsilon$ away from the goal set".
- Required new assumptions/permits: none.
- Validation plan: wording check.

### [E-008] The value term in the action functional is a boundary term (was F-007)
- Location: A Geometry-Regularized Objective, lines 635-641; Connection to RL #2, lines 646-671; comparison table 674-682.
- Severity: Moderate
- Type: Conceptual (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 638: "$\mathcal{S} = \int \left( \frac{1}{2} \lVert\dot{z}\rVert^2_{G} - \frac{d V}{d t} \right) dt$" with the second term labelled "Value improvement"; line 653: "The Euler-Lagrange equations yield updates along geodesics"; lines 660-664: "Set $G = I$ ... this recovers Euclidean SGD"; line 671: "updates correspond to stationary paths of the action functional on the value landscape".
- Upstream anchor: the book's latent dynamics carry a force term, `07_cognition/06_causality.md:307`: "$\ddot{z}^m + \Gamma^m_{ij}\dot{z}^i\dot{z}^j = -G^{mk}(\nabla_A V)_k - \dots$"; `07_cognition/01_supervised_topo.md:207`: "$\dot z = \mathcal{M}_{\rm curl}(-G^{-1}(z)\nabla_A V_y(z))$".
- Why this is an error: $\int_{t_0}^{t_1}\frac{dV}{dt}dt=V(z(t_1))-V(z(t_0))$ depends only on the endpoints, so for fixed endpoints the Euler–Lagrange equations of $\mathcal S$ are those of $\int\tfrac12\|\dot z\|_G^2dt$ alone: $\ddot z^k+\Gamma^k_{ij}\dot z^i\dot z^j=0$, with no force from $V$. The chapter's "yields geodesics" is true precisely because the value term does nothing, so the labels "value improvement" and "stationary paths on the value landscape" are empty, and the functional cannot be the variational origin of the descent dynamics used downstream. For $G=I$ the EL equation is $\ddot z=0$, not $\dot z=-\nabla V$, so the "SGD limit" is not a limit of this functional.
- Impact on downstream results: Connection #2 and the comparison table rest on this functional; nothing formal downstream cites it.
- Fix guidance: replace $-\frac{dV}{dt}$ by $-V(z)$ (Lagrangian $\tfrac12\|\dot z\|_G^2-V$ gives $\ddot z^k+\Gamma^k_{ij}\dot z^i\dot z^j=-G^{kj}\partial_jV$, the second-order form of the book's dynamics), or keep $-\dot V$ and say explicitly that it is a boundary term affecting only free-endpoint transversality, deleting the "value improvement" gloss and the SGD claim.
- Required new assumptions/permits: none.
- Validation plan: derive the EL equation of the corrected Lagrangian and compare with `06_causality.md:307` in the overdamped limit.

### [E-009] Sign of the natural-gradient step inconsistent within the chapter (was F-008)
- Location: line 729; line 740; line 751; line 1051.
- Severity: Minor
- Type: Typo (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 729: "$z \leftarrow z - \eta G^{-1}(z)\nabla_z \mathcal{L}$"; line 740: "$\delta z = G^{-1}(z) \nabla_z \mathcal{L}$"; line 1051: "$\delta z = G^{-1} \nabla_z \mathcal{L}$"; line 751: "$\delta\theta = \nabla_\theta J(\theta)$ (REINFORCE)".
- Why this is an error: under the declared cost convention a loss is minimised, so the step is $-G^{-1}\nabla\mathcal L$ (lines 729 and 1346); lines 740 and 1051 describe ascent on a cost, and line 751 ascent on a return, so the displays are not specialisations of one another.
- Impact on downstream results: none beyond confusion.
- Fix guidance: write $\delta z=-G^{-1}(z)\nabla_z\mathcal L$ at 740 and 1051; write the REINFORCE limit as $\delta\theta=-\nabla_\theta\mathcal L$ or $+\nabla_\theta J$ with $J=-\mathcal L$ stated.
- Required new assumptions/permits: none.
- Validation plan: local reading.

### [E-010] $\operatorname{diag}(\alpha,\beta_\pi,\gamma_{\rm wm},\delta)$ is not a metric on $\mathcal Z$ (was F-009)
- Location: A Practical Diagonal Sensitivity Metric, lines 809-823.
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "We construct a diagonal state-space sensitivity metric using the scaling coefficients from {ref}`sec-scaling-exponents-characterizing-the-agent`: $G = \text{diag}(\alpha, \beta_{\pi}, \gamma_{\text{wm}}, \delta)$".
- Upstream anchor: `docs/source/1_agent/02_sieve/01_diagnostics.md:192-203`: "four scaling coefficients. These are diagnostic summaries of state-space behavior, not optimizer statistics ... $\alpha$ | dimensionless ... $\beta_\pi$ | dimensionless ... $\gamma$ | dimensionless ... $\delta$ | dimensionless"; the genuine per-coordinate diagonals are at `:196-199` ("`state_fisher`: $G_{ii}=\mathbb E[(\partial\log\pi/\partial z_i)^2]+\mathrm{Hess}_z(V)_{ii}$").
- Why this is an error: $G$ is declared a $(0,2)$-tensor on the $d$-dimensional latent manifold with units $\mathrm{nat}[z]^{-2}$ (lines 852, 1424). The four coefficients are one dimensionless scalar per subsystem, not per coordinate; their diagonal is not indexed by $\mathcal Z$, has the wrong units, and does not transform as a tensor. This is the category error the chapter's Anti-Mixing rules forbid. The closing sentence (line 823) describes them as subsystem summaries, which is correct and contradicts the display.
- Impact on downstream results: used as a preconditioner it would rescale arbitrary blocks of $z$ by subsystem health numbers; the correct per-coordinate diagonals appear at lines 1173-1182.
- Fix guidance: delete the display or relabel: "the four scaling coefficients are scalar summaries (e.g. $\alpha\approx\overline{\mathrm{tr}\,G_V}$) used by the update scheduler, not entries of $G$"; point to Practical Approximations A.
- Required new assumptions/permits: none.
- Validation plan: confirm no code path builds $G$ from these four scalars.

### [E-011] Units of the state-space Fisher term and $\lambda_G$ (was F-010)
- Location: Definition Complete Latent Space Metric, line 843; cf. line 714 and line 1320.
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Units: the Fisher term has units $[z]^{-2}$; therefore $\lambda_G$ carries the same units as $V$ (here $\mathrm{nat}$) so both addends match."
- Upstream anchor: `docs/source/1_agent/01_foundations/01_definitions.md:372-374`: entropies and log-likelihood quantities are "in nats (dimensionless but tracked)", $[V]=\mathrm{nat}$; chapter line 1320: "$H(\pi(\cdot\mid z)) := -\mathbb{E}[\log\pi(a\mid z)]$ ... (in nats)"; chapter line 714: "$c_V$ carries units $\mathrm{nat}^{-1}$" (so $\partial_iV\partial_jV$ is tracked as $\mathrm{nat}^2[z]^{-2}$).
- Why this is an error: if $-\mathbb E[\log\pi]$ is in nats, then $\mathbb E[\partial_i\log\pi\,\partial_j\log\pi]$ has units $\mathrm{nat}^2[z]^{-2}$, and matching $[G_V]=\mathrm{nat}[z]^{-2}$ requires $[\lambda_G]=\mathrm{nat}^{-1}$. The chapter tracks nats for $V$ (line 714) but drops them for $\log\pi$ in the same definition.
- Impact on downstream results: `10_appendices/02_parameters.md` units for $\lambda_G$; the $\beta_{\rm cpl}$ chain (E-018).
- Fix guidance: decide once whether $\log$-probabilities carry the tracked nat; if yes write "$[G_\pi]=\mathrm{nat}^2[z]^{-2}$, hence $[\lambda_G]=\mathrm{nat}^{-1}$"; if no, remove "(in nats)" from line 1320 and adjust the dimensionless argument for $T_c$.
- Required new assumptions/permits: none.
- Validation plan: units audit of every occurrence of $\lambda_G$ and $c_V$.

### [E-012] Positive definiteness of $G$ stated as a result (was F-011)
- Location: Dimensional Verification, line 852; Definition State-Space Sensitivity Metric, lines 705-714.
- Severity: Minor
- Type: Proof gap / omission
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 852: "Result: $G$ is a positive-definite $(0,2)$-tensor that defines the Riemannian structure on $\mathcal{Z}$"; line 1145: "Positive Definiteness: $G(z)\succ0$ for all $z$" (Assumption 2).
- Why this is an error: the Hessian of a nonconvex $V$ is indefinite; the Gauss–Newton proxy $c_V\nabla V\nabla V^\top$ is rank one; $G_\pi$ is singular in coordinates the policy ignores (lines 937, 1231). Positivity is an assumption enforced operationally, not a consequence of the definition.
- Impact on downstream results: every $G^{-1}$, $\sqrt{|G|}$, Levi-Civita connection and $\Theta$ (line 1281) presuppose $G\succ0$.
- Fix guidance: change "Result:" to "Requirement (Assumption 2):" and note that positivity is enforced by the damped inverse $(\widehat G+\epsilon_t\mathbf 1)^{-1}$ of Practical Approximations D, or by replacing $\mathrm{Hess}\,V$ with its absolute-eigenvalue version plus $\epsilon I$.
- Required new assumptions/permits: none beyond Assumption 2.
- Validation plan: local reading.

### [E-013] One-step WFR quadratic proxy off by a factor $\Delta s$ (was F-012)
- Location: Connection to RL #3, quadratic proxy, lines 964-972.
- Severity: Moderate
- Type: Computational error (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "for a small update over $\Delta s$ with predicted transport/reaction $(v, r_{\text{WFR}})$ ... $d_{\mathrm{WFR}}^2(p_{\text{new}}, p_{\text{old}}) \approx \Delta s \int_{\mathcal{Z}} \left(\|v\|_G^2 + \lambda_{\text{WFR}}^2 |r_{\text{WFR}}|^2\right) p_{\text{old}}\, d\mu_G$".
- Upstream anchor: `docs/source/1_agent/05_geometry/02_wfr_geometry.md:106-116`: "The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0,\rho_1)$ is the infimum of ... $\int_0^1\int_{\mathcal Z}\big(\|v_s(z)\|_G^2+\lambda^2|r_s(z)|^2-2\langle\mathbf A,v_s\rangle\big)\,d\rho_s\,ds$" with $v_s$ a velocity field and $r_s$ a reaction rate.
- Why this is an error: upstream parametrises paths on $s\in[0,1]$. A step of duration $\Delta s$ at constant rates $(v,r)$ rescaled to the unit interval has $v_\sigma=\Delta s\,v$, $r_\sigma=\Delta s\,r$, so $d^2\approx(\Delta s)^2\int(\|v\|^2_G+\lambda^2r^2)p_{\rm old}\,d\mu_G$. Check in one Euclidean dimension with $v=0.7$, $\Delta s=0.3$: true $d^2=(v\Delta s)^2=0.0441$; the chapter's $\Delta s\,v^2=0.147$; $(\Delta s)^2v^2=0.0441$. Dimensionally the chapter's expression is a rate (nat s$^{-1}$), not a squared distance. The formula is correct only if $(v,r)$ are per-step displacements, in which case the prefactor should be absent.
- Impact on downstream results: a trust-region radius implemented from this proxy would scale wrongly with the solver step.
- Fix guidance: replace $\Delta s$ by $(\Delta s)^2$ (keeping $v,r$ as rates), or drop the prefactor and define $v,r$ as one-step displacement and log-mass-change fields; add a units line.
- Required new assumptions/permits: none.
- Validation plan: verify against the Benamou–Brenier form $d^2=T\int_0^T\int\|v\|^2\rho$ for a constant-velocity path.

### [E-014] HJB stated undiscounted and without minimisation or diffusion, yet linked to a screened Poisson equation (was F-013)
- Location: HJB Correspondence, lines 1008-1017 and 1061-1069.
- Severity: Moderate
- Type: Scope restriction (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 1009: "We replace the heuristic Bellman equation with the rigorous Hamilton-Jacobi-Bellman (HJB) Equation:"; line 1011: "$\mathcal{L}_f V + \mathfrak{D}(z, a) = -\mathcal{C}(f)$"; line 1069: "the Bellman/HJB equation becomes the Screened Poisson (Helmholtz) Equation ... $-\Delta_G V + \kappa^2 V = \rho_c$ ... where $\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln\gamma / \Delta t$".
- Upstream anchor: `docs/source/1_agent/06_fields/02_reward_field.md:325-361`: "Let the temporal discount rate be $\lambda := -\ln\gamma/\Delta t$ ... for a diffusion process $dz = b\,dt + \sigma\,dW$ with $\sigma\sigma^T = 2T_cG^{-1}$ ... $\kappa V = r + \nabla_A V\cdot b + T_c\Delta_G V$."
- Why this is an error: the displayed equation has no killing term $\lambda V$, no $\min_a$ (or soft-min), and no second-order $T_c\Delta_GV$ term; it is the undiscounted policy-evaluation identity along a fixed action. The screening mass in line 1069 comes from the omitted $\lambda V$ term and the Laplacian from the omitted diffusion, so the "correspondence" as stated cannot produce the equation it forward-references.
- Impact on downstream results: the identification of $V$ as a Green's function (line 1069) and the HJB-interface coupling (line 1501) rely on the discounted form.
- Fix guidance: state the discounted, regularised HJB in cost form, $\lambda V=\min_a[\mathcal C_if^i+\mathfrak D+\mathcal L_{f(\cdot,a)}V]\,(+\,T_c\Delta_GV)$ with $\lambda=-\ln\gamma/\Delta t$, and remark that the undiscounted fixed-policy case reduces to line 1011.
- Required new assumptions/permits: the discount and diffusion model already assumed upstream.
- Validation plan: check that the stationary, $b=0$ specialisation reproduces `thm-the-hjb-helmholtz-correspondence` with $\rho_c=-\rho_r$.

### [E-015] Geodesic distance given by a constant-metric quadratic form (was F-014)
- Location: Where the Metric Appears table, line 1052.
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "| Geodesic Distance | $d_G(z_1, z_2)^2 = (z_1-z_2)^T G (z_1-z_2)$ | YES |".
- Upstream anchor: chapter's own Rule #3, line 1256: "use metric arc-length: $\int ds\sqrt{\dot z^TG\dot z}$".
- Why this is an error: for position-dependent $G$ the geodesic distance is $\inf_\gamma\int_0^1\sqrt{\dot\gamma^TG(\gamma)\dot\gamma}\,dt$; the quadratic form is only its infinitesimal expansion and does not say where $G$ is evaluated.
- Impact on downstream results: misleading for the trust-region discussion.
- Fix guidance: rename the row "Local (infinitesimal) distance" with $ds^2=dz^TG(z)dz$, or give the arc-length infimum.
- Required new assumptions/permits: none.
- Validation plan: local reading.

### [E-016] $\delta_{\rm CE}>0$ is not equivalent to positive conditional mutual information (was F-015)
- Location: Closure Defect, Computational Meaning, line 1130; Definition Closure Defect, lines 1103-1121.
- Severity: Moderate
- Type: Invalid inference
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "If $\delta_{\text{CE}} > 0$ (or equivalently $I(K_{t+1};Z_t\mid K_t,K^{\text{act}}_t)>0$), then the learned macro predictor is not sufficient".
- Upstream anchor: chapter lines 1116-1118: $\delta_{\rm CE}:=\mathbb E_{z,a}[D_{\rm KL}(P_\Pi(\cdot\mid z,a)\,\|\,\bar P(\cdot\mid\Pi(z),a))]$ with $\bar P$ the learned kernel.
- Why this is an error: with $P^\star(\cdot\mid k,a):=\mathbb E[P_\Pi(\cdot\mid Z,a)\mid\Pi(Z)=k]$ the chain rule gives exactly $\delta_{\rm CE}=I(K_{t+1};Z_t\mid K_t,a)+\mathbb E_{k,a}[D_{\rm KL}(P^\star(\cdot\mid k,a)\,\|\,\bar P(\cdot\mid k,a))]$. So $I>0\Rightarrow\delta_{\rm CE}>0$, but $\delta_{\rm CE}>0$ with $I=0$ occurs whenever $\bar P\neq P^\star$. Numerical check on a random six-microstate, three-macrostate example: $\delta_{\rm CE}=0.2864=0.0870+0.1994$; for an enclosure-correct process with the same misfit $\bar P$: $\delta_{\rm CE}=0.1994$, $I=0$. The sentence conflates a property of the process (closure) with a property of the model (fit).
- Impact on downstream results: `11_implementation/01_encoder.md:63,800` and `02_world_model.md:1017,1172` use $\delta_{\rm CE}$ as the closure diagnostic; a large value may be model misfit, which calls for world-model training rather than a representation penalty (line 1222).
- Fix guidance: replace "or equivalently" by "in particular, since $\delta_{\rm CE}=I(K_{t+1};Z_t\mid K_t,K^{\rm act}_t)+\mathbb E[D_{\rm KL}(P^\star\|\bar P)]\ge I(\cdot)$"; state that $I=0$ is the enclosure condition and $\bar P=P^\star$ the fit condition, and that $\delta_{\rm CE}$ tests their conjunction.
- Required new assumptions/permits: none.
- Validation plan: the decomposition above is an identity; add it as a remark after the definition.

### [E-017] Symbol $\Theta$ reused for the parameter manifold and the conditioning scale (was F-016)
- Location: Trinity table, lines 170 and 179; Definition Local Conditioning Scale, lines 1278-1281; summary table, line 1520.
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 179: "| Parameter/Model | $\Theta$ | $\theta\in\mathbb R^P$ | $\mathcal F(\theta)$ |"; line 1281: "$\Theta: \mathcal{Z}\to\mathbb{R}^+$ ... $\Theta(z) := \frac{1}{d}\operatorname{Tr}(G^{-1}(z))$".
- Upstream anchor: `docs/source/1_agent/10_appendices/02_parameters.md:46` inherits the overload.
- Why this is an error: the chapter that forbids confusing parameter space with state space (Rule #1) denotes the parameter manifold and a scalar field on the state manifold by the same letter.
- Impact on downstream results: notation only.
- Fix guidance: rename the conditioning scale (e.g. $\vartheta(z)$ or $\ell_G^2(z)$) here and in `02_parameters.md:46`.
- Required new assumptions/permits: none.
- Validation plan: grep for `\Theta(` across the volume.

### [E-018] $\beta_{\rm cpl}=1/T_c$ equates nat/[z]$^2$ with a dimensionless quantity (was F-017)
- Location: Definition Local Conditioning Scale, lines 1284-1288; Lemma sketch, line 1303; summary table, lines 1425-1426.
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$\beta_{\text{cpl}}(z)=[\Theta(z)]^{-1}$ ... in an isothermal approximation ... set $\beta_{\text{cpl}} = 1/T_c$. Units: ... $[\Theta]=[z]^2/\mathrm{nat}$ and $[\beta_{\text{cpl}}]=\mathrm{nat}/[z]^2$ (dimensionless when $z$ is normalized)"; line 1312: "(dimensionless) trade-off coefficient $T_c\ge0$".
- Upstream anchor: `docs/source/1_agent/01_foundations/01_definitions.md:383-387`: "$T_c$ ... is dimensionless ... $\beta_{\text{cpl}}$ ... carries units $\mathrm{nat}/[z]^2$"; `10_appendices/02_parameters.md:46-47` gives $\Theta$ "dimensionless (when $z$ normalized)" and $\beta_{\rm cpl}$ "$\mathrm{nat}/[z]^2$", non-reciprocal units for a reciprocal pair.
- Why this is an error: normalising $z$ removes $[z]$ but not the tracked nat, so $\beta_{\rm cpl}=1/T_c$ is ill-typed.
- Impact on downstream results: `01_definitions.md:387`, `02_parameters.md:46-47` and the summary table inherit the mismatch.
- Fix guidance: introduce a reference scale $\ell_0$ with $[\ell_0^2]=[z]^2/\mathrm{nat}$ and define $\Theta(z):=\frac1d\mathrm{Tr}(G^{-1})/\ell_0^2$, making $\beta_{\rm cpl}$ dimensionless; update `02_parameters.md:46-47` to reciprocal units.
- Required new assumptions/permits: the reference scale $\ell_0$ (a convention).
- Validation plan: units audit of the Lemma and summary table.

### [E-019] Lemma Variance–Curvature Correspondence is unproved, ill-typed, and contradicts line 520 (was F-018)
- Location: Lemma, lines 1292-1307; Connection to RL #10, line 520; summary table, line 1520.
- Severity: Moderate
- Type: Proof gap / omission (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 1298: "$\Sigma_\pi(z)\propto\beta_{\text{cpl}}(z)^{-1}\cdot G^{-1}(z)$"; lines 1301-1305: "stationary distributions over latent states often take an exponential form $p(z)\propto\exp(-V(z)/T_c)$ ... Matching this form with a geometry-aware update implies that policy covariance scales inversely with the sensitivity metric. Deviations can be measured by a consistency defect $\mathcal D_{\beta_{\rm cpl}}:=\|\nabla\log p+\beta_{\rm cpl}\nabla_AV\|_G^2$"; line 520: "policy covariance $\Sigma_\pi(z)\propto G^{-1}(z)$".
- Upstream anchor: no other occurrence of $\Sigma_\pi$ in `docs/source/1_agent`; the lemma is not used downstream.
- Why this is an error: (i) the sketch reasons about a state law $p(z)$ and concludes about the action covariance $\Sigma_\pi(z)=\mathrm{Cov}_{a\sim\pi(\cdot\mid z)}(a)$ with no relation between $\pi$ and $p$ invoked; (ii) $\Sigma_\pi$ is $\dim\mathcal A\times\dim\mathcal A$ while $G^{-1}$ is a $(2,0)$-tensor on $T_z\mathcal Z$, so "$\propto$" is undefined unless $\mathcal A\cong T_z\mathcal Z$ is declared; (iii) line 520 gives $\Sigma_\pi\propto G^{-1}$, the lemma gives $\Sigma_\pi\propto\Theta(z)G^{-1}=\frac1d\mathrm{Tr}(G^{-1})G^{-1}$; for $G=gI$ these are $g^{-1}I$ versus $g^{-2}I$ and agree only when $\beta_{\rm cpl}$ is constant, which the pointwise "$\beta_{\rm cpl}(z)$" does not assume; (iv) by line 145 $\nabla_AV$ is a vector while $\nabla\log p$ is a covector, so the defect mixes index positions (and units, cf. E-018); the consistent form is $\|d\log p+\beta_{\rm cpl}d_AV\|^2_{G^{-1}}$.
- Impact on downstream results: summary row "Variance-curvature coupling via $\Theta(z)$" (line 1520) and Connection #10's claim; nothing formal downstream cites the lemma.
- Fix guidance: demote to a design principle, or make it a lemma with hypotheses: a Gaussian policy over latent displacements ($\mathcal A\cong T_z\mathcal Z$) whose stationary law is $e^{-V/T_c}d\mu_G$; the derivable conclusion is $\Sigma_{\rm step}\propto T_cG^{-1}$ (cf. `07_cognition/01_supervised_topo.md:233`, $dz=\dots+\sqrt{2T_c}G^{-1/2}dW$). Use one scaling at both line 520 and here; rewrite the defect with matched index positions.
- Required new assumptions/permits: the identification $\mathcal A\cong T_z\mathcal Z$ and a stationary-law hypothesis.
- Validation plan: check the isotropic case $G=gI$ gives one scaling at both locations.

### [E-020] Units of the transport field do not follow from its definition (was F-019)
- Location: Definition Transport Field, lines 1343-1349; continuity equation, line 1359.
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$v^i(z) := -G^{ij}(z)\frac{\partial V}{\partial z^j}$ ... Units: if computation time is measured in solver units, then $[v]=[z]/\mathrm{solver\ time}$".
- Upstream anchor: line 1424: "$G$ ... $\mathrm{nat}\,[z]^{-2}$"; $[V]=\mathrm{nat}$.
- Why this is an error: $[G^{ij}\partial_jV]=([z]^2/\mathrm{nat})(\mathrm{nat}/[z])=[z]$, not $[z]$/time; a mobility constant $\mu$ with $[\mu]=\mathrm{time}^{-1}$ is missing, and the continuity equation then mixes $[p]/\mathrm{time}$ with $[p]$.
- Impact on downstream results: local balance statements only.
- Fix guidance: write $v^i:=-\mu\,G^{ij}\partial_jV$ with $[\mu]=\mathrm{solver\ time}^{-1}$, or state $\mu=1$ in solver units.
- Required new assumptions/permits: none.
- Validation plan: units check of lines 1359 and 1370.

### [E-021] Sieve node names and Mode B.E do not match the Sieve chapter (was F-020)
- Location: Corollary Boundary filter interpretation, lines 1481-1489.
- Severity: Minor
- Type: Citation / reference error (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Sieve Nodes 13-16 (Boundary/Overload/Starve/Align) ... Mode B.E (Injection): Occurs when interface inflow exceeds the effective capacity of the manifold (Levin capacity) ... Mode B.D (Starvation): ... interface inflow is too weak".
- Upstream anchor: `docs/source/1_agent/02_sieve/01_diagnostics.md:111-114`: "13 BoundaryCheck ... 14 InputSaturationCheck ($\mathrm{Bound}_B$) ... 15 SNRCheck ($\mathrm{Bound}_\Sigma$) ... 16 AlignCheck"; `02_sieve/03_failures_interventions.md:61-63`: "B.E | Sensitivity Expl. | Critic | Fragility ... B.D | Resource Depletion | Boundary/Shutter | Starvation ... B.C | Control Deficit | Policy | Overwhelmed".
- Why this is an error: Nodes 14 and 15 are InputSaturationCheck and SNRCheck; there is no "Starve" node. Mode B.E is critic fragility, not over-inflow; the over-coupling regime is labelled upstream as BarrierScat/symbol dispersion (`04_control/03_coupling_window.md:133`).
- Impact on downstream results: following this corollary would route the intervention table (`03_failures_interventions.md:124`, SurgBE) to the wrong subsystem.
- Fix guidance: "Sieve Nodes 13-16 (BoundaryCheck / InputSaturationCheck / SNRCheck / AlignCheck)"; replace "Mode B.E (Injection)" by the upstream over-coupling label or by Node 14; keep B.D for starvation.
- Required new assumptions/permits: none.
- Validation plan: cross-check against the two upstream tables.

## Scope restrictions and clarifications
- The chapter's cost convention is coherent on its own; the difficulty in E-001 is purely the reuse of upstream symbols without stating the sign map $V_{\rm cost}=-\Phi_{\mathcal R}$, $A_{\mathcal C}=-A_{\mathcal R}$.
- The metric $G$ is used throughout as if $G\succ0$; this is Assumption 2 (line 1145) and is enforced operationally by damping, not derived (E-012).
- Several unit findings (E-011, E-018, E-020) follow from the book's stated rule that nats are "dimensionless but tracked"; if the author prefers to treat $\log$-probabilities as untracked, the same rule must be applied consistently to $V$ and $H$.
- The HJB display (line 1011) is valid as the undiscounted policy-evaluation identity for a fixed action; the claims that go beyond that (E-014) require the discounted, diffusive form.

## Open questions
- Upstream `06_fields/02_reward_field.md` identifies $V$ with the reward potential $\Phi$ (Bellman $V=\mathbb E[r+\gamma V']$), while `07_cognition` descends $V$ via $\dot z=-G^{-1}\nabla_AV$ with $A$ from the reward 1-form. Which sign of $V$ the book intends globally should be fixed in one place; this chapter's fix for E-001 should follow that decision.
- Is $\nabla_AV:=G^{-1}(dV-A)$ meant as a gauge-covariant derivative (minimal coupling, $d-A$) or as the raised total 1-form? The two readings differ in the relative sign of $A$ and the chapter does not say which.
- Which normalisation of $V$ (E-006) do the Node 7 residuals in `02_sieve` assume?

## Rejected candidate findings
- None. All twenty first-stage findings were confirmed or adjusted; within F-001 the sub-claim that the definitions "force $A=0$" and the claim of an inverted relative sign in $\nabla_AV$ were rejected (see E-001), and the finding was downgraded from Major to Moderate.
