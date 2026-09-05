# Mathematical Review: docs/source/1_agent/05_geometry/01_metric_law.md

## Metadata
- Reviewed file: docs/source/1_agent/05_geometry/01_metric_law.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (404 lines)
- Framework anchors (definitions/axioms/permits):
  - `10_appendices/01_derivations.md` A.1.1-A.3.2 (`def-a-boundary-capacity-form`, `def-a-boundary-capacity-constraint-functional`, `def-a-risk-lagrangian-density`, `def-a-capacity-constrained-curvature-functional`, `thm-a-capacity-consistency-identity-proof-of-theorem`)
  - `01_foundations/01_definitions.md` (`def-bounded-rationality-controller`, `def-boundary-markov-blanket`)
  - `01_foundations/02_control_loop.md` (`def-source-residual`, generalized conservation of belief)
  - `04_control/03_coupling_window.md` (`def-grounding-rate`, `thm-information-stability-window-operational`)
  - `06_fields/02_reward_field.md` (Hodge decomposition, `def-value-curl`, `def-conservative-reward-field`, screened-Poisson value equation)
  - `06_fields/03_info_bound.md` (`def-saturation-limit`, `lem-metric-divergence-at-saturation`, node table)
  - `07_cognition/01_supervised_topo.md` (`node-40`), `08_multiagent/04_dnn_blocks.md` (node table)
  - `10_appendices/02_parameters.md` (Appendix B units table), `10_appendices/06_losses.md` (Appendix F)
  - `03_architecture/02_disentangled_vae.md` (heading "Curvature as Conditioning")

All symbolic claims below (risk-tensor sign, divergence identities, entropy scaling) were recomputed with sympy; the script is recorded in the verification log.

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 5
- Minor: 6
- Notes: 0
- Primary themes:
  1. The bulk "information" is defined as a differential entropy of the belief density. It is not non-negative, it is not a mutual information, and the data-processing inequality does not bound it; the central inequality $I_{\text{bulk}}\le C_\partial$ is therefore not the DPI it is claimed to be, and the diagnostic ratio $\nu_{\text{cap}}$ can be negative. Appendix A uses the same symbol $\rho_I$ for a different object.
  2. The derivation the main theorem cites (Appendix A) never uses the capacity constraint: the saturation functional is defined and dropped, the boundary penalty has zero variation under clamping, and $\Lambda$ is a free constant. What is derived is the Einstein-Hilbert equation with a cosmological constant and a minimally coupled scalar. The explicit risk tensor in Appendix A is the negative of its own definition, the chapter's version drops the on-site potential, and the Maxwell stress is not derived anywhere and has the wrong physical dimension relative to the gradient stress.
  3. With the chapter's gradient stress, the contracted Bianchi identity forces $(\Delta_G\Phi)\,\partial_j\Phi=0$, which is inconsistent with the screened-Poisson equation the book imposes on $\Phi=V$; the metric law is overdetermined unless $V$ is put on shell for the same action.
  4. The "geometric reflow" remedy has the wrong sign: enlarging Riemannian volume increases the differential entropy that the chapter calls $I_{\text{bulk}}$.
  5. Bookkeeping: a dangling section reference, a stale note about a term that no longer exists, a hard-coded "Definition 18.1.2a", the boundary bandwidth cited to the bulk-volume label, two names for one regulariser with a pointer to an appendix that does not define it, the same proxy $\mathbb{E}[I(X_t;K_t)]$ used for both sides of the inequality, and a Node 40 collision with PurityCheck.

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Intro, line 40 | Minor | Citation / reference error | Framework | this chapter | `sec-differential-geometry-view-curvature-as-conditioning` is not defined anywhere |
| E-002 | Definitions, lines 61-116; diagnostic, lines 327-345 | Major | Conceptual (Definition mismatch) | External (secondary Framework) | this chapter | $I_{\text{bulk}}$ is a differential entropy: not $\ge0$, not DPI-bounded; $\rho_I$ conflicts with Appendix A |
| E-003 | Note, lines 100-103 | Minor | Definition mismatch (Miswording) | Framework | this chapter | Note explains a $\tfrac12\rho\log\det G$ term absent from the definition |
| E-004 | Boundary capacity, lines 128-140; theorem line 182 | Moderate | Conceptual (Scope restriction) | Framework | this chapter (shared with `10_appendices/01_derivations.md`) | $\partial\mathcal{Z}$ is never constructed; the book's boundary is an interface tuple |
| E-005 | Remark, lines 142-148; lines 54, 71, 372 | Moderate | Definition mismatch (Dimensional mismatch) | Framework | this chapter and upstream `10_appendices/01_derivations.md:48` | Same proxy for $C_\partial$ and $I_{\text{bulk}}$; nat vs nat/step; $\log|\mathcal{K}|$ bound misapplied |
| E-006 | Theorem, lines 177-194 | Major | Proof gap / omission (Miswording) | Framework | upstream `10_appendices/01_derivations.md` (re-asserted here) | Capacity constraint plays no role in the cited derivation; $\sigma$ hypothesis unused |
| E-007 | Theorem line 185; gradient stress, lines 230-235 | Moderate | Computational error (Definition mismatch) | Framework | upstream `10_appendices/01_derivations.md:150,168` (sign); this chapter ($U$ dropped) | Appendix A's explicit $T_{ij}$ is minus its own definition; chapter omits $-G_{ij}U(V)$ |
| E-008 | Theorem line 185 with lines 230-241 | Major | Proof gap / omission (Parameter inconsistency) | Framework | this chapter (shared with `10_appendices/01_derivations.md`) | Bianchi identity needs $(\Delta_G\Phi)\partial_j\Phi=0$, inconsistent with $(-\Delta_G+\kappa^2)V=\rho_r$ |
| E-009 | Theorem, line 190; diagnostic, line 336 | Minor | Citation / reference error | Framework | this chapter | "Definition 18.1.2a" dead number; bandwidth cited to the bulk-volume label |
| E-010 | Lines 192 and 297 | Minor | Notation conflict (Citation / reference error) | Framework | this chapter | $\mathcal{L}_{\text{cap-metric}}$ vs $\mathcal{L}_{\text{EFE}}$; Appendix B has no loss list; Frobenius norm not invariant |
| E-011 | Prose, line 200 | Minor | Miswording | External | this chapter | Parallel transport does not change volumes; Ricci controls geodesic-ball volume |
| E-012 | Extended risk tensor, lines 219-248 | Moderate | Dimensional mismatch (Proof gap / omission) | Framework | this chapter | Maxwell stress has units $\mathrm{nat}^2/[z]^4$, gradient stress $\mathrm{nat}^2/[z]^2$; Maxwell term not derived |
| E-013 | Node table line 325; line 389 | Minor | Notation conflict | Framework | this chapter and upstream `07_cognition/01_supervised_topo.md:1149`, `08_multiagent/04_dnn_blocks.md:3163` | Node 40 is PurityCheck elsewhere; `#node-40` resolves there |
| E-014 | Diagnostic line 343; tip lines 347-360; line 211 | Moderate | Computational error (Conceptual) | Framework | this chapter | Increasing $\lvert G\rvert$ raises $h[\rho]$; reflow prescription has the wrong sign |

## Detailed findings

### [E-001] Dangling cross-reference at line 40 (was F-001)
- Location: intro paragraph after the Researcher Bridge, line 40
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 40): "{ref}`Section 9.10 <sec-differential-geometry-view-curvature-as-conditioning>` used a "gravity" analogy to motivate curvature as a regulator."
- Upstream anchor: no `(sec-differential-geometry-view-curvature-as-conditioning)=` label exists in `docs/`. The matching heading is `03_architecture/02_disentangled_vae.md:365` "## Differential-Geometry View (No Physics): Curvature as Conditioning", which has no label; `myst.yml` / `_config.yml` do not enable automatic heading slugs.
- Why this is an error: the reference cannot resolve. The same target is used at `intro_agent.md:430` and `10_appendices/04_faq.md:43`.
- Impact on downstream results: none mathematically; broken link.
- Fix guidance: add `(sec-differential-geometry-view-curvature-as-conditioning)=` immediately above the heading at `02_disentangled_vae.md:365` (fixes all three uses), or repoint the sentence.
- Required new assumptions/permits: none.
- Validation plan: rebuild; confirm the xref pre-pass no longer lists the target.

### [E-002] $I_{\text{bulk}}$ is a differential entropy, not a DPI-bounded information; $\rho_I\ge0$ is false (was F-002)
- Location: Definition "Information density and bulk information volume" (lines 85-98), Definition "Bulk Information Volume" (lines 105-116), Definition "DPI / boundary-capacity constraint" (lines 61-73), diagnostic (lines 327-345)
- Severity: Major
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: External (information theory), secondary Framework (mismatch with Appendix A)
- Origin: this chapter
- Claim (verbatim): line 88 "The **information density** $\rho_I(z,s)\ge 0$ is defined as the local Shannon entropy density: $\rho_I(z,s) := -\rho(z,s) \log \rho(z,s)$"; line 96 "Integrating $\rho_I$ over $\mathcal{Z}$ ... yields the coordinate-invariant differential entropy $h[\rho]$"; line 56 "This is the data-processing inequality"; line 341 "$\nu_{\text{cap}} > 1$: **Violation** of the DPI constraint".
- Upstream anchor: `10_appendices/01_derivations.md:48` "$\rho_I(G,V)$ is an *information density* (nats per unit $d\mu_G$) compatible with the agent's representation scheme (Definition 17.1.2). This $\rho_I$ is distinct from the belief density $p$".
- Why this is an error: (i) $-\rho\log\rho<0$ wherever $\rho>1$; a belief concentrated on a region of Riemannian volume below 1 has $\rho>1$ there, and $h[\rho]\to-\infty$ in the Dirac limit. So $\rho_I\ge0$ is false, $I_{\text{bulk}}$ can be negative, and $\nu_{\text{cap}}=I_{\text{bulk}}/C_\partial$ can be negative, which makes the "$\nu_{\text{cap}}\ll1$" reading (line 339) meaningless. (ii) The data-processing inequality bounds mutual information along a Markov chain ($I(\text{world};Z)\le I(\text{world};X)$); it says nothing about the entropy of the marginal $\rho(z)$. A maximally ignorant belief (uniform on a large region) has the largest $h$ while carrying no grounded information, the opposite of what lines 52-58 describe. (iii) Appendix A, which claims to prove the theorem, defines $\rho_I$ as a function of $(G,V)$ distinct from the belief density; the chapter builds it from the belief density (line 88) and then repeats "conceptually distinct from the probability-mass balance" (line 114). The two definitions of the same symbol are incompatible.
- Impact on downstream results: Node 40 / $\nu_{\text{cap}}$, the reflow prescription (E-014), Appendix A.1.2, `06_fields/03_info_bound.md:190-193` (saturation $I_{\text{bulk}}=C_\partial$), `07_cognition/03_memory_retrieval.md:1021` ($G_{rr}\propto(1-\tilde I_{\text{bulk}}/C_\partial)^{-1}$), Connection to RL #25.
- Fix guidance:
  1. Redefine $I_{\text{bulk}}$ as a non-negative, coordinate-invariant, channel-bounded quantity: either the mutual information between boundary history and $Z$, or the relative entropy $D_{\mathrm{KL}}(\rho\,\|\,\rho_{\text{ref}})$ against an ungrounded reference belief (prior). Both are $\ge0$ and the first is genuinely DPI-bounded.
  2. If the entropy density is retained for other purposes, delete "$\ge0$", remove the DPI language at lines 56, 58, 341, 372, 388, and stop calling $I_{\text{bulk}}$ "grounded structure".
  3. Reconcile with Appendix A.1.2: one definition of $\rho_I$, stated once, and referenced from both places.
- Required new assumptions/permits: a declared reference belief $\rho_{\text{ref}}$ (KL option) or a declared boundary-history variable (MI option).
- Validation plan: check $I_{\text{bulk}}\ge0$ on a Dirac-like and a uniform belief; confirm the DPI derivation of $I_{\text{bulk}}\le C_\partial$ goes through for the chosen definition.

### [E-003] Stale note about a "second term" that the definition does not contain (was F-003)
- Location: note after the information-density definition, lines 100-103
- Severity: Minor
- Type: Definition mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 102): "The second term $\frac{1}{2}\rho \log\det G$ might seem like a technicality, but it's doing important work."
- Upstream anchor: n/a; the definition immediately above (lines 91-96) has one term and states "no explicit metric correction term is needed".
- Why this is an error: the note describes and motivates a term that is not present and asserts the opposite of the remark it follows; an implementer cannot tell which version is intended.
- Impact on downstream results: implementation of $I_{\text{bulk}}$.
- Fix guidance: delete the note, or rewrite it to explain why defining $\rho$ relative to $d\mu_G$ absorbs the $\log\det G$ correction.
- Required new assumptions/permits: none.
- Validation plan: reread lines 85-116 for a single consistent definition.

### [E-004] Boundary interface identified with an unconstructed topological boundary of $\mathcal{Z}$ (was F-004)
- Location: Definition "Boundary capacity: area law at finite resolution", lines 128-140; theorem hypotheses, line 182
- Severity: Moderate
- Type: Conceptual (secondary: Scope restriction)
- Criterion: Framework
- Origin: this chapter (the same gap is in `10_appendices/01_derivations.md:23-33, 86`)
- Claim (verbatim, lines 131-136): "Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$. ... $C_{\partial}(\partial\mathcal{Z}) := \frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G$".
- Upstream anchor: `01_foundations/01_definitions.md:117-129` "The boundary variables at time $t$ are the interface tuple $B_t := (x_t, r_t, d_t, \iota_t, a_t)$"; `01_definitions.md:98-104` "$\mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}}$"; `06_fields/03_info_bound.md:209-212` "The Poincare metric ... with the horizon at $|z|=1$. In computation we truncate at $|z|=1-\varepsilon$ ...; this is a numerical cutoff".
- Why this is an error: the book's boundary is a set of interface random variables, not a hypersurface in $\mathcal{Z}$. The latent space in use (discrete $\mathcal{K}$ times the open Poincare disk times $\mathcal{Z}_{\text{tex}}$) has no geometric boundary; the ideal boundary is at infinite $G$-distance, and the $1-\varepsilon$ truncation is declared numerical. So $\oint_{\partial\mathcal{Z}}dA_G$ is undefined, zero, or cutoff-dependent, and nothing connects it to sensor bandwidth. The clamped-boundary hypothesis of Appendix A.2 inherits the same conflation.
- Impact on downstream results: theorem hypotheses; Appendix A.1.1-A.1.2; `03_info_bound.md:287` (Area$(\partial\mathcal{Z})=\Omega_{D-1}r_h^{D-1}$).
- Fix guidance:
  1. Either construct $\partial\mathcal{Z}$ explicitly (e.g. the cutoff sphere $|z|=1-\varepsilon$ with $\varepsilon$ tied to $\ell$) and state that $C_\partial$ is cutoff-dependent,
  2. or define $C_\partial$ directly as the interface channel capacity ($\sup_{p(x)}I(X;K)$ per step times a window) and present the area law as a separate modelling assumption with its own hypotheses.
- Required new assumptions/permits: an explicit hypersurface-interface identification, or an explicit area-law postulate.
- Validation plan: verify that $C_\partial$ is finite and well-defined on the Poincare-disk latent space used in `05_geometry/03_holographic_gen.md`.

### [E-005] Same proxy $\mathbb{E}[I(X_t;K_t)]$ for both sides of the inequality; stock vs rate units (was F-005)
- Location: Remark (discrete macro specialization), lines 142-148; lines 54, 71, 372
- Severity: Moderate
- Type: Definition mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter and upstream `10_appendices/01_derivations.md:48`
- Claim (verbatim, line 145): "$C_{\partial}\ \approx\ \mathbb{E}[I(X_t;K_t)]\ \le\ \log|\mathcal{K}|$"; line 71 "Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$"; line 54 "capacity $C_\partial$ (measured in nats per unit time, say)"; line 372 "$I_{\text{bulk}} \le C_\partial \le \log|\mathcal{K}|$".
- Upstream anchor: `10_appendices/01_derivations.md:48` "the most conservative computable proxy is a global one, $I_{\text{bulk}}\approx \mathbb{E}[I(X;K)]$ (Node 13)"; `04_control/03_coupling_window.md:85-91` "$G_t:=I(X_t;K_t)$ ... $\lambda_{\text{in}} := \mathbb{E}[G_t]$. Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$"; `10_appendices/02_parameters.md:61-62` list $C_\partial$ and $I_{\text{bulk}}$ in nat.
- Why this is an error: (i) the chapter uses $\mathbb{E}[I(X_t;K_t)]$ as the proxy for the capacity and Appendix A uses it as the proxy for the bulk information; with both, $\nu_{\text{cap}}\equiv1$ and the inequality is vacuous. (ii) The quantity is a per-step rate ($\lambda_{\text{in}}$, nat/step) while $C_\partial$ is declared a stock (nat) at line 71 and in Appendix B, and a rate at line 54. (iii) $\log|\mathcal{K}|$ bounds the discrete proxy, not the area-law $C_\partial$.
- Impact on downstream results: Node 40 implementation; consistency with Node 13 and the coupling-window theorem.
- Fix guidance:
  1. Decide whether $I_{\text{bulk}}$ and $C_\partial$ are stocks or rates; make lines 54, 71 and Appendix B agree (a window length $W$ converts nat/step to nat).
  2. Use distinct proxies: $C_\partial\approx\sup_{p(x)}I(X;K)$ or $\log|\mathcal{K}|$ per step; $I_{\text{bulk}}\approx$ realised $I(X;K)$ or $H(K)$.
  3. Restrict the $\le\log|\mathcal{K}|$ bound (line 372) to the discrete proxy.
- Required new assumptions/permits: a declared averaging window if stocks are kept.
- Validation plan: compute both proxies on a toy shutter and check $\nu_{\text{cap}}$ is not identically 1.

### [E-006] The capacity constraint plays no role in the derivation the theorem cites (was F-006)
- Location: Theorem "Capacity-constrained metric law", lines 177-194
- Severity: Major
- Type: Proof gap / omission (secondary: Miswording)
- Criterion: Framework
- Origin: upstream `10_appendices/01_derivations.md` A.1-A.3, re-asserted by this chapter with an additional unused hypothesis
- Claim (verbatim, line 182): "under the soundness condition that bulk structure is boundary-grounded (no internal source term $\sigma$ ...), stationarity of a capacity-constrained curvature functional implies $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}$"; line 190 "Curvature is the geometric mechanism that prevents the internal information volume ... from exceeding the boundary's information bandwidth".
- Upstream anchor: `01_derivations.md:69-73` "$\mathcal{S}[G,V] := \int_{\mathcal{Z}}\left(R(G)-2\Lambda + 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G - 2\kappa\oint_{\partial\mathcal{Z}}\omega_{\partial}$"; `:174` "Under the clamped boundary condition $\delta G\vert_{\partial\mathcal{Z}}=0$, its first variation vanishes"; `:66` "let $\Lambda\in\mathbb{R}$ be a constant"; `:176` "The remaining constant $\Lambda$ ... plays the role of the bulk Lagrange multiplier"; `:40-47` define $\mathcal{C}[G,V]=I_{\text{bulk}}-C_\partial$, which never appears in $\mathcal{S}$.
- Why this is an error: the only capacity-carrying objects in Appendix A are the boundary term, whose variation is zero by hypothesis, and $\Lambda$, which is a free constant (no constraint equation $\mathcal{C}=0$ is imposed, no variation in the multiplier, no relation between $\Lambda$ and $\eta_\ell$, $\ell$ or $C_\partial$). The saturation functional of A.1.2 is never used. The Euler-Lagrange equation obtained is that of $\int(R-2\Lambda)d\mu_G+2\kappa\int\mathcal{L}_{\text{risk}}\,d\mu_G$: the Einstein-Hilbert action with cosmological constant and a minimally coupled scalar. Nothing in it depends on $C_\partial$, $I_{\text{bulk}}$ or $\eta_\ell$, so the theorem does not establish the operational reading at line 190, and the "no internal source term $\sigma$" hypothesis is never invoked in the proof.
- Impact on downstream results: every use of the theorem as a capacity statement: `06_fields/03_info_bound.md:199-224, 287-297`, `07_cognition/04_ontology.md:219, 470-478`, `07_cognition/03_memory_retrieval.md:1021`, `05_geometry/03_holographic_gen.md:699`, Connection to RL #25.
- Fix guidance:
  1. Minimal option: restate the theorem as "stationarity of the curvature-plus-risk functional with clamped boundary yields $R_{ij}-\tfrac12RG_{ij}+\Lambda G_{ij}=\kappa T_{ij}$ with free constants $\Lambda,\kappa$", drop the capacity reading and the unused $\sigma$ hypothesis, and move the capacity interpretation to a clearly labelled conjecture.
  2. Full option: add $\lambda\,\mathcal{C}[G,V]$ to $\mathcal{S}$, vary with respect to $G$ and $\lambda$, derive the extra term $\lambda\,\delta(\rho_I\,d\mu_G)/\delta G^{ij}$ and the relation fixing $\lambda$ (hence $\Lambda$) in terms of $C_\partial$. This requires a $G$-differentiable $\rho_I$ (see E-002).
- Required new assumptions/permits: for the full option, a $G$-differentiable information density and a well-defined $C_\partial$ (E-004).
- Validation plan: check that the resulting equation changes when $C_\partial$ changes; if it does not, the constraint is still not active.

### [E-007] Sign error in the Appendix A risk tensor; the chapter's gradient stress inherits it and drops $U(V)$ (was F-007)
- Location: Theorem, line 185; Definition "Extended Risk Tensor", item 1, lines 230-235
- Severity: Moderate
- Type: Computational error (secondary: Definition mismatch)
- Criterion: Framework
- Origin: upstream `10_appendices/01_derivations.md:150, 168` (sign); this chapter (omission of $U$)
- Claim (verbatim, line 233): "$T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - \frac{1}{2}G_{ij} \|\nabla\Phi\|_G^2$".
- Upstream anchor: `01_derivations.md:150` "$T_{ij} := -\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\text{risk}})}{\delta G^{ij}}$"; `:57` "$\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V)$"; `:168` "$T_{ij} = \nabla_i V\,\nabla_j V - G_{ij}\left(\frac12\,G^{ab}\nabla_a V\nabla_b V + U(V)\right)$".
- Why this is an error: with $\partial\mathcal{L}/\partial G^{ij}=\tfrac12\nabla_iV\nabla_jV$ and $\delta\sqrt{|G|}=-\tfrac12\sqrt{|G|}G_{ij}\delta G^{ij}$, the definition gives $T_{ij}=-\nabla_iV\nabla_jV+G_{ij}\mathcal{L}_{\text{risk}}$, the exact negative of the displayed tensor (verified symbolically for a generic 2D metric with $U$ and $\nabla V$ symbolic: $T_{\text{def}}+T_{\text{stated}}=0$ componentwise). The origin is the Riemannian sign of the kinetic term ($+\tfrac12(\nabla V)^2$ here versus $-\tfrac12(\partial\phi)^2$ in Lorentzian field theory). The remainder of Appendix A is consistent with the definition, so stationarity actually yields $R_{ij}-\tfrac12RG_{ij}+\Lambda G_{ij}=-\kappa\,T^{\text{stated}}_{ij}$. Since $\kappa$ is free this is repairable by $\kappa\to-\kappa$, but the sign of $\kappa$ is used downstream (`03_info_bound.md:207`, $\mu(r)>0$ for a horizon; `:289`, $\kappa=8\pi\ell_L^{D-1}$). Separately, the chapter's $T^{\text{gradient}}$ omits the $-G_{ij}U(V)$ term of the derived tensor without stating $U\equiv0$.
- Impact on downstream results: sign of the curvature response (inflate vs deflate, Cliff Walk example lines 262-272); `lem-metric-divergence-at-saturation`.
- Fix guidance:
  1. In Appendix A, define $T_{ij}:=+\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\mathcal{L}_{\text{risk}})}{\delta G^{ij}}$ (or flip the sign of $\mathcal{L}_{\text{risk}}$ in the action) and state the intended sign of $\kappa$.
  2. In this chapter, include $-G_{ij}U(\Phi)$ in $T^{\text{gradient}}$ or state $U\equiv0$ as a hypothesis.
  3. Also align the boundary-penalty coefficient ($2\kappa$ at A.1.4 versus $\kappa$ at A.2.4).
- Required new assumptions/permits: none.
- Validation plan: re-run the symbolic check; confirm the derived $T_{ij}$ equals the displayed one.

### [E-008] The risk tensor as written violates the contracted Bianchi identity given the book's value equation (was F-009)
- Location: Theorem, line 185, with the Extended Risk Tensor, lines 230-241
- Severity: Major
- Type: Proof gap / omission (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter (shared with `10_appendices/01_derivations.md`, which holds $V$ fixed and never imposes its Euler-Lagrange equation)
- Claim (verbatim, line 185): "$R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}$" with the gradient stress of line 233.
- Upstream anchor: `06_fields/02_reward_field.md:196-197` "We identify $\Phi$ with the critic value $V$ (the exact component), so $d\Phi = dV$"; `:401` "The value function satisfies $(-\Delta_G + \kappa^2)V = \rho_r$"; `:211-221` $\mathcal{F}=dA=d\delta\Psi$.
- Why this is an error: the left side is identically divergence-free (contracted Bianchi identity plus $\nabla^iG_{ij}=0$), so any solution needs $\nabla^iT_{ij}=0$. Recomputed for the conformal metric $G=e^{2\omega}\delta$: $\nabla^iT^{\text{gradient}}_{ij}=(\Delta_G\Phi)\,\partial_j\Phi$ exactly (the Hessian terms cancel by symmetry). Hence the metric law imposes $(\Delta_G\Phi)\partial_j\Phi=0$, i.e. $\Phi$ harmonic wherever $\nabla\Phi\ne0$. Upstream, the same $\Phi=V$ satisfies $\Delta_GV=\kappa_{\text{scr}}^2V-\rho_r$ in the same metric, which is generically nonzero. The pair {metric law, screened-Poisson equation} is therefore overdetermined and inconsistent wherever $\nabla V\ne0$ and $\kappa_{\text{scr}}^2V\ne\rho_r$. Keeping $U$ (as Appendix A does) helps only if $V$ is on shell for the same action, $\Delta_GV=U'(V)$, which then gives $\nabla^iT_{ij}=(\Delta_GV-U'(V))\partial_jV=0$; neither the chapter nor Appendix A imposes this. For the Maxwell part, $\nabla^iT^{\text{Maxwell}}_{ij}=(\nabla^i\mathcal{F}_{ik})\mathcal{F}_j{}^k$ (recomputed in flat 3D), so the source-free Maxwell equation $\nabla^i\mathcal{F}_{ik}=0$ is also required and is not implied by $\mathcal{F}=d\delta\Psi$.
- Impact on downstream results: existence of solutions to the metric law; the regulariser $\mathcal{L}_{\text{EFE}}$ cannot vanish on regions with $\Delta_GV\ne0$; `lem-metric-divergence-at-saturation` survives only because its $\sigma_{\max}$ is constant.
- Fix guidance:
  1. Retain $U(V)$ in $T_{ij}$ and add the hypothesis that $V$ is stationary for the same action ($\Delta_GV=U'(V)$).
  2. Choose $U$ compatible with the screened-Poisson equation, e.g. $U(V;z)=\tfrac12\kappa_{\text{scr}}^2V^2-\rho_r(z)V$, and declare the explicit $z$-dependence through $\rho_r$.
  3. State that the metric law is consistent only on shell for $V$ (and for $A$, with $\nabla^i\mathcal{F}_{ik}=0$, in the non-conservative case).
- Required new assumptions/permits: on-shell hypotheses for $V$ and $A$; a position-dependent on-site potential in Appendix A.1.3.
- Validation plan: verify $\nabla^iT_{ij}=0$ symbolically with the chosen $U$ and the screened-Poisson equation substituted.

### [E-009] Dead definition number and wrong label in the operational reading (was F-012)
- Location: Theorem, line 190; diagnostic, line 336
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 190): "prevents the internal information volume (Definition 18.1.2a) from exceeding the boundary's information bandwidth (Definition {prf:ref}`def-a-bulk-information-volume`)"; line 336 "per Definition 18.1.2a".
- Upstream anchor: `def-a-bulk-information-volume` is the bulk volume (line 106); the boundary capacity is `def-boundary-capacity-area-law-at-finite-resolution` (line 129). "Definition 18.1.2" occurs only at these two lines in `docs/source/1_agent`.
- Why this is an error: the bandwidth is cited to the bulk-volume definition, and the hard-coded number does not resolve.
- Impact on downstream results: none mathematically.
- Fix guidance: cite `def-a-bulk-information-volume` for the bulk volume and `def-boundary-capacity-area-law-at-finite-resolution` for the bandwidth; remove "18.1.2a".
- Required new assumptions/permits: none.
- Validation plan: rebuild and check the links.

### [E-010] Two names for the regulariser and a pointer to an appendix that does not define it (was F-013)
- Location: Theorem "Implementation hook", line 192; Physics Isomorphism "Loss Function", line 297
- Severity: Minor
- Type: Notation conflict (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter (the same Appendix B pointer is at `10_appendices/01_derivations.md:218`)
- Claim (verbatim, line 192): "defines a capacity-consistency regularizer $\mathcal{L}_{\text{cap-metric}}$; see Appendix B for the consolidated list of loss definitions"; line 297 "$\mathcal{L}_{\text{EFE}} := \|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\|_F^2$".
- Upstream anchor: Appendix B (`10_appendices/02_parameters.md`) is a units table with no loss list; the loss reference is Appendix F (`10_appendices/06_losses.md:1-2`); neither contains "cap-metric" or "EFE".
- Why this is an error: one object, two names, and a dead pointer. Also, the plain Frobenius norm of a covariant 2-tensor is not coordinate invariant; the invariant residual is $G^{ik}G^{jl}X_{ij}X_{kl}$ integrated against $d\mu_G$.
- Impact on downstream results: implementation of the regulariser.
- Fix guidance: pick one name, add the entry to Appendix F, cite Appendix F from both places, and write the norm with $G$-contractions.
- Required new assumptions/permits: none.
- Validation plan: grep for both names after the edit.

### [E-011] Ricci curvature attributed to parallel transport (was F-014)
- Location: prose after the theorem, line 200
- Severity: Minor
- Type: Miswording
- Criterion: External
- Origin: this chapter
- Claim (verbatim, line 200): "$R_{ij}$: the Ricci tensor. This measures how volumes change when you parallel transport them around. Positive Ricci curvature means volumes shrink; negative means they expand."
- Upstream anchor: n/a (standard Riemannian geometry).
- Why this is an error: parallel transport is an isometry of tangent spaces and preserves volumes. Ricci curvature governs the second-order deviation of geodesic-ball volume, $\mathrm{vol}\,B_r(p)=\omega_nr^n\big(1-\tfrac{R}{6(n+2)}r^2+O(r^4)\big)$, equivalently the convergence of a geodesic spray. The sign conclusion is right; the mechanism named is wrong, and the sentence is used to justify the inflate/deflate reading at line 211.
- Impact on downstream results: intuition only.
- Fix guidance: replace with "measures how the volume of small geodesic balls deviates from Euclidean: positive Ricci curvature means geodesics converge and volumes are smaller than flat; negative means they diverge".
- Required new assumptions/permits: none.
- Validation plan: none needed.

### [E-012] Maxwell stress not derived and dimensionally inconsistent with the gradient stress (was F-008)
- Location: Definition "Extended Risk Tensor with Maxwell Stress", lines 219-248
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 239-242): "$T_{ij}^{\text{Maxwell}} = \mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}$ ... *Units:* $[T_{ij}] = \mathrm{nat}^2/[z]^2$."
- Upstream anchor: `06_fields/02_reward_field.md:198` "$[\Phi] = \mathrm{nat}$"; `:221` "$[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$"; `10_appendices/02_parameters.md:65` "$\Lambda$ ... $[z]^{-2}$" (so $G_{ij}$ is dimensionless). Appendix A's Lagrangian (`01_derivations.md:57`) contains no $\mathcal{F}$ term, and no Maxwell stress derivation exists elsewhere in Volume 1.
- Why this is an error: $[\partial_i\Phi\partial_j\Phi]=\mathrm{nat}^2/[z]^2$ but $[\mathcal{F}_{ik}\mathcal{F}_{jl}G^{kl}]=\mathrm{nat}^2/[z]^4$; the two summands of $T_{ij}$ differ by $[z]^{-2}$, so the sum at line 225 and the stated units are inconsistent. In electromagnetism this does not arise because the gauge potential and scalar have the same dimension; here the book fixes $[\mathcal{R}]=[A]=\mathrm{nat}/[\text{length}]$, so $[\mathcal{F}]=[\partial\Phi]/[\text{length}]$. The Maxwell term is asserted, not derived from the cited action.
- Impact on downstream results: `06_fields/02_reward_field.md:807` ("Both the scalar potential $\Phi$ and the Value Curl $\mathcal{F}$ contribute to risk, and therefore modify the metric").
- Fix guidance:
  1. Introduce a coefficient $\alpha_{\mathcal{F}}$ with $[\alpha_{\mathcal{F}}]=[z]^2$ (e.g. $\ell^2$) in front of the Maxwell stress.
  2. Add $-\tfrac14\alpha_{\mathcal{F}}\mathcal{F}_{ab}\mathcal{F}^{ab}$ to $\mathcal{L}_{\text{risk}}$ in Appendix A.1.3 and derive $T^{\text{Maxwell}}$ by the same variation (this also fixes its sign convention relative to E-007).
- Required new assumptions/permits: the new coefficient $\alpha_{\mathcal{F}}$ and its entry in Appendix B.
- Validation plan: dimensional audit of every term in $T_{ij}$; symbolic check that the variation reproduces the displayed Maxwell stress.

### [E-013] Node 40 name collision (was F-011)
- Location: diagnostic table, line 325; Connection to RL #25, line 389
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter and upstream `07_cognition/01_supervised_topo.md:1149-1157`, `08_multiagent/04_dnn_blocks.md:3163`
- Claim (verbatim, line 325): "| 40 | CapacitySaturationCheck | Bulk-boundary information ratio | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon$ |".
- Upstream anchor: `07_cognition/01_supervised_topo.md:1149-1150` "(node-40)= **Node 40: PurityCheck (CapacitySaturationCheck)**" with proxy $H(Y\mid K)$; `08_multiagent/04_dnn_blocks.md:3163` "| 40 | PurityCheck | $SU(N_f)_C$ confinement |"; `08_multiagent/03_parameter_sieve.md:595`, `10_appendices/07_architecture.md:1219` (PurityCheck); `06_fields/03_info_bound.md:683` links CapacitySaturationCheck to `#node-40`.
- Why this is an error: Node 40 names two unrelated diagnostics, and the only `node-40` label resolves to PurityCheck. This chapter defines no label of its own.
- Impact on downstream results: node registry, sieve tables, tooling keyed by node number.
- Fix guidance: assign a distinct number or name to one of the two, add a label for the capacity diagnostic in this chapter, and repoint `03_info_bound.md:683`.
- Required new assumptions/permits: none.
- Validation plan: grep "Node 40" across the volume after the edit.

### [E-014] Geometric reflow prescription has the wrong sign (was F-010)
- Location: diagnostic cross-reference, line 343; tip "What Geometric Reflow Looks Like", lines 347-360; prose line 211
- Severity: Moderate
- Type: Computational error (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 343): "the metric $G$ must increase $|G|$ (expand volume) to bring $I_{\text{bulk}}$ back within bounds."; line 352 "larger volumes mean the same probability distribution is 'spread out' over more space, carrying less information density."; line 211 "The curvature 'inflates' the important regions, giving them more effective volume ... The total stays within budget."
- Upstream anchor: n/a; $I_{\text{bulk}}=h[\rho]=-\int\rho\log\rho\,d\mu_G$ is defined in this chapter (lines 85-116).
- Why this is an error: fix the belief measure and rescale $G\to c^2G$ with $c>1$; then $d\mu_G\to c^nd\mu_G$, $\rho\to c^{-n}\rho$, and $h[\rho]\to h[\rho]+n\log c$ (recomputed). The uniform density on a region of volume $V$ has $h=\log V$. Differential entropy grows with volume, so increasing $|G|$ raises $I_{\text{bulk}}$ as defined and the remedy moves the wrong way. A uniform rescaling can still lower the ratio $\nu_{\text{cap}}$ because $C_\partial\propto$ Area grows like $c^{n-1}$, but that is a boundary effect, not the "less information density" mechanism described, and it does not apply to local inflation of "important regions", which raises $I_{\text{bulk}}$ without changing $C_\partial$.
- Impact on downstream results: remediation logic for Node 40; the intuition paragraph at line 211.
- Fix guidance: once $I_{\text{bulk}}$ is redefined (E-002), "reflow" becomes coarsening (reducing KL to the prior or the realised mutual information), obtained by lowering resolution, not by inflating volume; rewrite lines 211, 343, 350-352 accordingly. If the entropy definition is kept, reverse the sign of the prescription.
- Required new assumptions/permits: none beyond E-002.
- Validation plan: compute $I_{\text{bulk}}$ before and after a local metric inflation on a toy belief and confirm the direction.

## Scope restrictions and clarifications
- The Euler-Lagrange identity itself (Einstein tensor plus $\Lambda G_{ij}$ equals a coupling times the metric variation of the risk Lagrangian) is correct as a statement about stationarity of $\int(R-2\Lambda+2\kappa\mathcal{L}_{\text{risk}})d\mu_G$ with clamped boundary, up to the sign convention of E-007. What is not established is any dependence on boundary capacity.
- The saturation lemma in `06_fields/03_info_bound.md` uses a constant $\sigma_{\max}$, so it is not affected by the Bianchi obstruction of E-008; it is affected by the sign convention of E-007.
- All findings labelled External rest on standard information theory or Riemannian geometry (non-negativity of KL versus differential entropy, DPI for mutual information, volume comparison for Ricci curvature); no external theorem is imported to establish a framework result.

## Open questions
- Which object is $I_{\text{bulk}}$ meant to be: a mutual information with the boundary history, a KL divergence to an ungrounded prior, or an entropy? The answer determines the fixes for E-002, E-005, E-006 and E-014 jointly.
- Is $\partial\mathcal{Z}$ intended to be the Poincare-disk cutoff sphere $|z|=1-\varepsilon$? If so, the area law should say so and $C_\partial$ should be declared cutoff dependent.
- Appendix A.1.4 writes the boundary penalty as $-2\kappa\oint\omega_\partial$ while A.2.4 quotes $-\kappa\oint\omega_\partial$; harmless because the variation vanishes, but the coefficients should agree.
- Should the theorem be restated so that $V$ (and $A$) are varied alongside $G$, making the on-shell conditions of E-008 part of the result rather than an added hypothesis?

## Rejected candidate findings
None. All fourteen stage-1 findings were confirmed; F-009 was kept with sharpened wording (overdetermination with the upstream value equation rather than non-existence of solutions) and a refined type and origin.
