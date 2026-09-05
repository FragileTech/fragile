# Mathematical Review: docs/source/1_agent/06_fields/03_info_bound.md

## Metadata
- Reviewed file: docs/source/1_agent/06_fields/03_info_bound.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (700 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/05_geometry/01_metric_law.md`: `def-information-density-and-bulk-information-volume` (lines 86-112, $\rho_I := -\rho\log\rho$, $I_{\text{bulk}} = \int\rho_I\,d\mu_G$); boundary capacity $C_\partial = \frac{1}{\eta_\ell}\oint dA_G$ with $[\eta_\ell] = [dA_G]/\mathrm{nat}$ (136-147); `thm-capacity-constrained-metric-law` $R_{ij} - \tfrac12 R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$ with $T_{ij}$ the total Risk Tensor (180-190); `def-extended-risk-tensor` with $[T_{ij}] = \mathrm{nat}^2/[z]^2$ (220-242); `sec-diagnostic-node-capacity-saturation` (Node 40, 310-330)
  - `docs/source/1_agent/10_appendices/01_derivations.md`: `lem-a-divergence-to-boundary-conversion` (186-196); Appendix A.6: Levin Length as implementation resolution (756), microstate counting Thm A.6.0g/h (743-840), Lemma A.6.1 bulk-to-boundary identity (871-898), Proposition A.6.2 saturation metric (913-946), $n=2$ remark (959-966), Thm A.6.6 assembly and Fisher normalization (1017-1049)
  - `docs/source/1_agent/05_geometry/04_equations_motion.md`: `def-bulk-drift-continuous-flow` (220-226), `def-effective-potential` (418), summary table "Full Geodesic SDE" row (1289), overdamped limit (1291)
  - `docs/source/1_agent/07_cognition/01_supervised_topo.md:1149-1150` (`node-40` anchor, PurityCheck); `docs/source/1_agent/08_multiagent/01_gauge_theory.md:3683-3751` (Nodes 57-60)
  - Symbolic checks: `scratchpad/review/check_arealaw.py` (sympy; Ricci tensor of the spherically symmetric ansatz for $n = 3,4,5$, extrinsic curvature of $r = \text{const}$, Step 1-3 product, truncated Poincaré boundary length)

## Executive summary
- Critical: 2
- Major: 6
- Moderate: 4
- Minor: 3
- Notes: 0
- Primary themes:
  1. The field-theoretic derivation of the Area Law does not hold. The "saturation metric" $A(r)$ of `lem-metric-divergence-at-saturation` does not solve the Metric Law with uniform stress (verified symbolically for $n = 3,4,5$; the actual solution is a constant-curvature space with effective constant $\Lambda - \kappa\sigma$ and no mass term), the bulk-to-boundary identity $\int R\,d\mu_G = 2\oint\mathrm{Tr}K\,dA_G$ of Step 1 is false in every dimension, $\mathrm{Tr}K$ vanishes (rather than tending to $(D-1)/r_h$) where $G_{rr}$ diverges, and Steps 1-3 as written do not multiply out to the boxed formula (off by $\Omega_{D-1}r_h$, and not dimensionless). These steps are inherited verbatim from Appendix A.6 of `10_appendices/01_derivations.md`, but the chapter restates each as its own lemma or proof step.
  2. The capacity quantities are mutually inconsistent. $\ell_L := \sqrt{\eta_\ell}$ has length units only for $D = 3$; the Saturation Limit ($I_{\text{bulk}} = C_\partial$) and Causal Stasis ($I_{\text{bulk}} \to I_{\max}$) use different thresholds; the $D = 2$ formula $\mathrm{Area}/(4\ell_L)$ contradicts the appendix cited as its full proof ($\mathrm{Area}/(4\ell_L^2)$); and the Metric Law is sourced by the reward Risk Tensor, not by information density, so no framework statement links $I_{\text{bulk}} \to I_{\max}$ to $G_{rr} \to \infty$.
  3. Causal Stasis: the displayed equation of motion matches neither the cited second-order definition nor the overdamped limit, and the proof establishes only $v^r \to 0$ on the horizon sphere, not $\|v\|_G \to 0$.
  4. Diagnostic and registry plumbing: an undefined "Metabolic Pruning Criterion", a computational proxy that does not estimate $I_{\text{bulk}}/I_{\max}$, a notation table that contradicts the chapter's own symbol usage, and a Node-40 anchor that resolves to PurityCheck.
- Mechanical cross-reference pre-pass: `xref_report.md` lists no dangling references or duplicate labels for this file.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Physics Isomorphism box (60); Remark (124); `thm-causal-information-bound` Corollary (269-275); Node 56 special case (410); Key Results 3 (474) | Major | Parameter inconsistency (secondary: Citation / reference error) | Framework | this chapter vs upstream `10_appendices/01_derivations.md:1043-1049` | $D = 2$ formula $\mathrm{Area}/(4\ell_L)$ contradicts the cited full proof's $\mathrm{Area}/(4\ell_L^2)$; the only derived coefficient ($1/4$ per 2-dimensional area) contradicts $\nu_3 = 1$ |
| E-002 | The Levin Length, `def-levin-length` (149-164); Saturation Limit (185-196); Summary table (462-464); Key Results 2 (472) | Major | Dimensional mismatch (secondary: Definition mismatch) | Framework | this chapter | $\ell_L := \sqrt{\eta_\ell}$ has units $[z]^{(D-1)/2}$; $C_\partial$ and $I_{\max}$ become different thresholds |
| E-003 | Saturation Limit (185-196); `lem-metric-divergence-at-saturation` hypothesis (201, 207, 214); Corollary proof (366) | Major | Conceptual (secondary: Proof gap / omission) | Framework | this chapter (same identification upstream A.6.1-A.6.2) | Risk-Tensor source of the Metric Law is renamed "information mass"; no framework statement couples $G$ to $I_{\text{bulk}}$; units of $\sigma_{\max}$ and $\rho_I$ differ |
| E-004 | `lem-metric-divergence-at-saturation` (198-224) | Critical | Computational error (secondary: External dependency) | External (contradicts the book's own Metric Law) | upstream `10_appendices/01_derivations.md:913-939` | $A(r)$ does not solve the Metric Law; sign of $\Lambda_{\text{eff}}$ reversed; no mass term exists; horizon moves outward with $\sigma$ |
| E-005 | Lemma remark, $n = 2$ case (209-212); Corollary (269-275); Step 2 (287) | Moderate | Scope restriction (secondary: Proof gap / omission) | External | this chapter (same at upstream `:959-964`) | Einstein tensor vanishes identically in 2D; Poincaré boundary has infinite (cutoff-dependent) length |
| E-006 | `thm-causal-information-bound` proof sketch, Step 1 (279-285) | Critical | Invalid inference (secondary: External dependency) | External | upstream `10_appendices/01_derivations.md:871-898` (Lemma A.6.1) | $\int R\,d\mu_G = 2\oint\mathrm{Tr}K\,dA_G$ is false; $\rho_I$ never linked to $R$ |
| E-007 | Proof sketch, Step 2 (287) | Major | Computational error | External | upstream `10_appendices/01_derivations.md:1024` | $\mathrm{Tr}K = (D-1)\sqrt f/r \to 0$ where $G_{rr}$ diverges, not $(D-1)/r_h$ |
| E-008 | Proof sketch, Steps 1-3 and "Combining these steps" (279-291); Physics box (57) | Major | Computational error (secondary: Parameter inconsistency) | Framework | upstream `10_appendices/01_derivations.md:1017-1041`; $\kappa = 8\pi\ell_L^{D-1}$ is this chapter's | Product of Steps 1-3 differs from the boxed bound by $\Omega_{D-1}r_h$ and is not dimensionless; $\kappa$ given three incompatible values |
| E-009 | `thm-causal-stasis` statement and proof (324-350); Summary table (466) | Major | Proof gap / omission (secondary: Invalid inference) | Framework | this chapter | Proof yields only $v^r \to 0$ at $r = r_h$; angular block of $G^{-1}$ is untouched by $A(r)$ |
| E-010 | `thm-causal-stasis` proof (333-344) | Moderate | Citation / reference error (secondary: Proof gap / omission) | Framework | this chapter (equation form exists in `04_equations_motion.md:1289`) | Displayed SDE is neither the cited $(z,p)$ definition nor the overdamped limit; boundedness of $p_r$ not argued |
| E-011 | `cor-saturation-velocity-tradeoff` (357-369) | Moderate | Proof gap / omission (secondary: Invalid inference) | Framework | this chapter | $\eta = \mu/\mu_{\max}$ asserted, not derived; under the correct solution $G^{rr}$ grows with $\sigma$ |
| E-012 | Remediation list, item 2 (428) | Minor | Citation / reference error | Framework | this chapter | "Def: Metabolic Pruning Criterion" does not exist |
| E-013 | Computational Proxy (432-438) | Moderate | Algorithm mismatch (secondary: Dimensional mismatch) | Framework | this chapter | Proxy numerator overcounts by $\lvert\mathcal{K}\rvert$; denominator logarithmic in area; example evaluates to 4.38 |
| E-014 | Core Symbols table (492-565) | Minor | Definition mismatch (secondary: Notation conflict) | Framework | this chapter | $\rho_I$ row includes a term the formal definition excludes; $\kappa$ and $\eta$ overloaded |
| E-015 | Diagnostic Node Registry (637-700), rows 40 and 56-61; line 682 | Minor | Citation / reference error | Framework | this chapter (duplicate node number upstream) | Node-40 anchor resolves to PurityCheck; Nodes 57-60 missing from a "Complete" registry; prose line splits the table |

## Detailed findings

### [E-001] The $D = 2$ corollary contradicts the cited full proof, and $\nu_3 = 1$ contradicts the only derived coefficient (was F-005)
- Location: Physics Isomorphism box (line 60); Remark (124); Corollary inside `thm-causal-information-bound` (269-275); Node 56 special case (410); Key Results item 3 (474)
- Severity: Major
- Type: Parameter inconsistency (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter, inconsistent with upstream `docs/source/1_agent/10_appendices/01_derivations.md:1043-1049`, which is itself internally inconsistent
- Claim (verbatim): line 60: "For $D=2$: $I_{\max} = \text{Area}/(4\ell_L)$"; lines 272-275: "$I_{\max} = \nu_2 \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L}$ ... $\ell_L^{D-1} = \ell_L$ for the 1-dimensional boundary"; line 277: "full derivation in {ref}`sec-appendix-a-area-law`".
- Upstream anchor: `10_appendices/01_derivations.md:796`: "The channel capacity of a 2-dimensional boundary $\partial\mathcal{Z}$ with Riemannian area $A$ is: $C_\partial = \frac{A}{4\ell_L^2}$ nats"; `:834` (Thm A.6.0h): "$I_{\max} = \ln\Omega = \frac{A}{4\ell_L^2}$"; `:1049`: "**Special case ($n=2$, Poincare disk):** ... $I_{\max} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2}$ uses $\ell_L^2$ (rather than $\ell_L^{n-1} = \ell_L$) because the Poincare disk metric normalization $G(0) = 4I$ maps coordinate cells to Riemannian areas."
- Why this is an error: The chapter's $D = 2$ formula has $\ell_L^1$; the appendix it cites as the full proof states $\ell_L^2$ three times and explicitly says the $\ell_L^{n-1}$ rule is not used for $n = 2$. Dimensionally the chapter is right for a 1-dimensional boundary ($[\text{Area}] = [z]$), which means the appendix's microstate derivation (Thm A.6.0g/h, `:743-807`), which tiles a 2-dimensional boundary with cells of area $4\ell_L^2$, is a $D = 3$ statement. For $D = 3$ the chapter's table (line 116) gives $\nu_3 = 1$, i.e. $I_{\max} = \text{Area}/\ell_L^2$, not $\text{Area}/(4\ell_L^2)$. So the only coefficient actually derived anywhere ($1/4$ per 2-dimensional boundary area) contradicts $\nu_3 = 1$, and the chapter's $\nu_2 = 1/4$ is matched to a computation done for the wrong dimension. `07_cognition/07_metabolic_transducer.md:598` uses $\text{Area}/(4\ell_L^2)$, so downstream chapters already disagree with this one. No inherited-consistency defence applies because the source itself disagrees with the chapter.
- Impact on downstream results: Numerical values of $I_{\max}$ and $\eta_{\text{Sch}}$ in every dimension; Node 56 thresholds; the "$\nu_2 = 1/4$ = Bekenstein-Hawking" claim; `08_multiagent/03_parameter_sieve.md:266-290`.
- Fix guidance:
  1. Decide the bulk dimension of the microstate argument in Appendix A.6.0 (as written it is $D = 3$).
  2. Reconcile: either set $\nu_3 = 1/4$ (and revise `def-holographic-coefficient` and its table accordingly) or redo the counting for a 1-dimensional boundary.
  3. Make the chapter and Appendix A.6.5 / Thm A.6.0h state the same $D = 2$ formula; correct `07_metabolic_transducer.md:598` to match.
- Required new assumptions/permits: none; this is a consistency repair.
- Validation plan: Grep all occurrences of `4\ell_L`, `4\ell_L^2`, `\nu_2`, `\nu_3` across `docs/source/1_agent` and check that each agrees with the single chosen convention; check units of every stated $I_{\max}$ formula against $[\text{Area}] = [z]^{D-1}$.

### [E-002] The Levin Length $\ell_L := \sqrt{\eta_\ell}$ is dimensionally inconsistent with the bound and with $C_\partial$ (was F-001)
- Location: The Levin Length, Definition `def-levin-length` (lines 149-164); Saturation Limit (185-196); Summary table (462-464); Key Results item 2 (472)
- Severity: Major
- Type: Dimensional mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 152-160: "Let $\eta_\ell$ be the boundary area-per-nat at resolution $\ell$ ... $\ell_L := \sqrt{\eta_\ell}$. Units: $[\ell_L] = [z]$ ... A cell of area $\ell_L^2$ in the latent manifold corresponds to one nat"; line 472: "One nat of information occupies $(D-1)$-dimensional volume $\ell_L^{D-1}$".
- Upstream anchor: `05_geometry/01_metric_law.md:136-140`: "$C_{\partial}(\partial\mathcal{Z}) := \frac{1}{\eta_\ell}\oint_{\partial\mathcal{Z}} dA_G$ ... where $\eta_\ell$ is the effective boundary area-per-nat ... Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$", with $dA_G$ the induced $(n-1)$-form (line 132). `10_appendices/01_derivations.md:756`: "The Levin Length $\ell_L$ is the fundamental coordinate resolution of the computational manifold, determined by implementation constraints".
- Why this is an error: $dA_G$ is a $(D-1)$-form, so $[\eta_\ell] = [z]^{D-1}/\mathrm{nat}$ and $[\sqrt{\eta_\ell}] = [z]^{(D-1)/2}$. This equals $[z]$ only for $D = 3$; for $D = 2$ (the chapter's headline case) $\ell_L$ would have units $[z]^{1/2}$. Consequently: (a) with the chapter's own definitions $C_\partial = \mathrm{Area}/\eta_\ell = \mathrm{Area}/\ell_L^2$, whereas $I_{\max} = \nu_D\,\mathrm{Area}/\ell_L^{D-1}$ (line 260); these coincide only for $D = 3$, and for $D = 2$ they are $\mathrm{Area}/\ell_L^2$ versus $\mathrm{Area}/(4\ell_L)$. (b) Definition `def-saturation-limit` (line 191, "$I_{\text{bulk}} = C_\partial$") and `thm-causal-stasis` (line 327, "As $I_{\text{bulk}} \to I_{\max}$") therefore use different thresholds, yet the theorem's proof invokes "saturation" (the $C_\partial$ notion) to conclude about $I_{\max}$; Node 40 ($\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$) and Node 56 ($\eta_{\text{Sch}} = I_{\text{bulk}}/I_{\max}$) are presented as complementary (line 418) but measure different ratios. (c) The interpretation sentence at line 160 ("cell of area $\ell_L^2$ ... one nat", a $D = 2$ bulk cell) contradicts Key Result 2 at line 472 ("$(D-1)$-dimensional volume $\ell_L^{D-1}$", a boundary cell) and, for $D = 2$, the Corollary's one nat per $4\ell_L$ of boundary length. (d) The appendix cited as the full proof defines $\ell_L$ as a primitive implementation resolution, not as $\sqrt{\eta_\ell}$; the definition is made in this chapter.
- Impact on downstream results: Every use of $I_{\max}$ and $\eta_{\text{Sch}}$ (Node 56; `08_multiagent/03_parameter_sieve.md:266-290`, `09_economics/01_pomw.md:444-455`, `07_cognition/03_memory_retrieval.md:1016`, `07_cognition/08_intersubjective_metric.md:259,712`). `03_memory_retrieval.md:1016` already writes "$C_\partial = \nu_D\,\mathrm{Area}/\ell_L^{D-1}$", silently using the identification this chapter does not justify.
- Fix guidance:
  1. Define the Levin Length so that $\ell_L^{D-1}$ is the boundary $(D-1)$-volume per nat, e.g. $\ell_L := (\nu_D\,\eta_\ell)^{1/(D-1)}$, so that $I_{\max} \equiv C_\partial$ by construction; or drop the identification and state the relation between $C_\partial$ and $I_{\max}$ explicitly.
  2. Make `def-saturation-limit` and `thm-causal-stasis` use the same threshold.
  3. Rewrite the interpretation sentence to "a boundary cell of $(D-1)$-volume $\ell_L^{D-1}$ carries one nat".
- Required new assumptions/permits: none if option 1 is taken (it is a definitional choice); if $C_\partial \ne I_{\max}$ is retained, an explicit statement of their relation is needed.
- Validation plan: Unit check of every formula containing $\ell_L$, $\eta_\ell$, $C_\partial$, $I_{\max}$ in the chapter and in the downstream files listed above; confirm Node 40 and Node 56 ratios are either identical or explicitly distinguished.

### [E-003] The saturation regime conflates the Risk-Tensor source of the Metric Law with information density (was F-002)
- Location: Saturation Limit (lines 185-196); Lemma `lem-metric-divergence-at-saturation` hypothesis "uniform stress $T_{ij} = \sigma_{\max} G_{ij}$" (201, 207, 214); Corollary proof (366)
- Severity: Major
- Type: Conceptual (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (the same identification is made upstream in `10_appendices/01_derivations.md:872-895`, Appendix A.6.1-A.6.2, but the Definition, Lemma and Corollary that rely on it are this chapter's)
- Claim (verbatim): lines 201-207: "At saturation with uniform stress $T_{ij} = \sigma_{\max} G_{ij}$ ... $\mu(r) := \frac{\kappa}{n-2}\int_0^r \sigma_{\max} r'^{n-1}dr'$ is the integrated **information mass**"; line 214: "Substitute the uniform density into the Metric Law"; line 366: "$\eta := I_{\text{bulk}}/I_{\max} = \mu/\mu_{\max}$".
- Upstream anchor: `05_geometry/01_metric_law.md:188`: "$T_{ij}$ is the **total Risk Tensor** induced by the reward field"; `:220-242` (`def-extended-risk-tensor`): $T_{ij} = \partial_i\Phi\,\partial_j\Phi - \tfrac12 G_{ij}\|\nabla\Phi\|_G^2 + (\text{Maxwell term})$, "*Units:* $[T_{ij}] = \mathrm{nat}^2/[z]^2$"; `:86-91`: "$\rho_I(z,s) := -\rho(z,s)\log\rho(z,s)$".
- Why this is an error: In the framework the source of the Metric Law is the reward-gradient/curl stress, independent of the belief entropy density $\rho_I$. Nothing in `thm-capacity-constrained-metric-law` or its appendix proof makes $G$ respond to $I_{\text{bulk}}$; the "operational reading" at `01_metric_law.md:190` is prose, not a derived relation. The chapter nevertheless (i) sets $T_{ij} = \sigma_{\max}G_{ij}$ and calls $\int\sigma_{\max}$ an "information mass", (ii) calls $\sigma_{\max}$ a "density", and (iii) equates $\mu/\mu_{\max}$ with $I_{\text{bulk}}/I_{\max}$. Units also fail: with $[G_{ij}] = [z]^{-2}$ (chapter table, line 494), $[\sigma_{\max}] = [T_{ij}]/[G_{ij}] = \mathrm{nat}^2$, whereas $[\rho_I] = \mathrm{nat}/[z]^D$ (`01_metric_law.md:92`). The central causal chain "$I_{\text{bulk}} \to I_{\max} \Rightarrow G_{rr} \to \infty$" therefore has no supporting statement in the framework; it is asserted by renaming.
- Impact on downstream results: `thm-causal-stasis`, `cor-saturation-velocity-tradeoff`, the Node 56 interpretation thresholds, and all Causal Stasis claims in `10_appendices/04_faq.md:556` and `08_multiagent/03_parameter_sieve.md:372`.
- Fix guidance:
  1. Either add a lemma or permit relating $T_{ij}$ (or a new information-stress term) to $\rho_I$ under the capacity constraint, e.g. a Lagrange-multiplier term from the DPI constraint entering the action of Appendix A.2, with consistent units; or
  2. restate Causal Stasis as conditional on such a coupling ("if the risk stress grows with bulk information such that ...").
  3. At minimum rename "information mass" and state the identification $\eta = \mu/\mu_{\max}$ as an assumption.
- Required new assumptions/permits: a coupling statement between the Risk Tensor (or an added stress term) and $\rho_I$ / $I_{\text{bulk}}$, with $[\kappa\,T_{ij}] = [z]^{-2}$.
- Validation plan: Check that the new coupling has consistent units end to end; confirm that each use of "saturation" in the chapter refers to a quantity that actually enters the Metric Law.

### [E-004] The saturation metric $A(r)$ does not solve the Metric Law; the sign of $\Lambda_{\text{eff}}$ is reversed; no mass term exists (was F-003)
- Location: Lemma `lem-metric-divergence-at-saturation` (lines 198-224)
- Severity: Critical
- Type: Computational error (secondary: External dependency)
- Criterion: External (classical Riemannian computation); it also contradicts the framework's own field equation
- Origin: upstream `docs/source/1_agent/10_appendices/01_derivations.md:913-939` (Proposition A.6.2), inherited verbatim and restated as this chapter's lemma
- Claim (verbatim): lines 204-207: "$A(r) = \left(1 - \frac{2\mu(r)}{(n-2)r^{n-2}} - \frac{\Lambda_{\text{eff}} r^2}{n(n-1)}\right)^{-1}$, where $\mu(r) := \frac{\kappa}{n-2}\int_0^r \sigma_{\max} r'^{n-1}dr'$ ... and $\Lambda_{\text{eff}} = \Lambda + \kappa\sigma_{\max}$".
- Upstream anchor: `05_geometry/01_metric_law.md:180-188`: "$R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\,T_{ij}$"; `10_appendices/01_derivations.md:925-928`: identical formula with "$\Lambda_{\text{eff}} = \Lambda + \kappa\sigma_{\max}$".
- Why this is an error: With $T_{ij} = \sigma G_{ij}$ the Metric Law reads $R_{ij} - \tfrac12 R\,G_{ij} = -(\Lambda - \kappa\sigma)G_{ij}$, a pure Einstein-space equation with effective constant $\Lambda - \kappa\sigma$ (the source moves to the left with a minus sign), not $\Lambda + \kappa\sigma$. The Ricci tensor of the ansatz $ds^2 = dr^2/f(r) + r^2 d\Omega_{n-1}^2$ was computed symbolically (sympy, $n = 3,4,5$) and the combination $R_{ij} - \tfrac12 R G_{ij} + \Lambda G_{ij} - \kappa\sigma G_{ij}$ formed. The mixed-index $rr$ component is $\frac{(n-1)(n-2)}{2}\frac{f-1}{r^2} + (\Lambda - \kappa\sigma)$ (e.g. $n = 3$: $(\Lambda - \kappa\sigma)r^2 + f - 1 = 0$), whose unique regular solution is
  $f(r) = 1 - \frac{2(\Lambda-\kappa\sigma)}{(n-1)(n-2)}\,r^2$,
  which also satisfies the angular component identically (residual $0$ in all three dimensions). Substituting the lemma's $A(r)^{-1}$ with $\mu(r) = \kappa\sigma r^n/(n(n-2))$ leaves constant non-zero residuals in both components: $(5\Lambda - 11\kappa\sigma)/6$ for $n = 3$, $(6\Lambda - 13\kappa\sigma)/8$ for $n = 4$, $(21\Lambda - 47\kappa\sigma)/30$ for $n = 5$. A pure mass term $f = 1 - 2M/r^{n-2}$ is not a solution either ($n = 3$: residuals $-2M/r^3$ and $+M/r^3$ in the $rr$ and angular components): the Schwarzschild-Tangherlini form solves the Lorentzian vacuum equations in $n+1$ dimensions, whereas a Riemannian $n$-manifold satisfying an Einstein-space equation with the spherically symmetric ansatz has no mass parameter in any $n$ (for $n = 3$ it is a space form). Consequences: (i) a root $f(r_h) = 0$ exists only when $\Lambda > \kappa\sigma$, with $r_h^2 = (n-1)(n-2)/(2(\Lambda-\kappa\sigma))$, and is the equator of a round sphere (a coordinate singularity); (ii) increasing $\sigma$ pushes $r_h$ outward, the opposite of the claimed mechanism; (iii) the "integrated information mass" $\mu(r)$ plays no role. Additional upstream evidence: the reduced ODE displayed at `01_derivations.md:919` does not equal the $rr$ component of the Metric Law (difference $-2\Lambda + A'/(rA^2) + 2/r^2 - 2/(r^2A)$ for $n = 3$), so the appendix's starting equation is already wrong.
- Impact on downstream results: Everything built on $r_h$ and $A(r)$: Step 2 of `thm-causal-information-bound`, `thm-causal-stasis`, `cor-saturation-velocity-tradeoff`, Node 56 semantics, the "Schwarzschild horizon / saturation horizon" row of `pi-einstein-equations` in `01_metric_law.md`, and Appendix A.6.2-A.6.3. The Lemma is the sole support of `thm-causal-stasis` and of Step 2.
- Fix guidance:
  1. Replace the lemma by the correct statement: under $T_{ij} = \sigma G_{ij}$ the Metric Law forces a constant-curvature geometry with $f = 1 - 2(\Lambda-\kappa\sigma)r^2/((n-1)(n-2))$.
  2. If a Schwarzschild-type horizon driven by a localized source is wanted, introduce a non-uniform $T_{ij}$ (e.g. a radial perfect-fluid form with $T_{rr} \ne T_{\theta\theta}$) and derive the solution afresh.
  3. Re-prove or drop the saturation-implies-divergence claim on the basis of the corrected solution; propagate to Appendix A.6.2-A.6.3.
- Required new assumptions/permits: a specified, non-uniform stress profile if a horizon is to be retained.
- Validation plan: Re-run `check_arealaw.py` (or equivalent) on the replacement metric for $n = 3,4,5$ and confirm both components of the Metric Law vanish identically; check the sign and monotonicity of $r_h(\sigma)$.

### [E-005] The $D = 2$ case is ill-posed: the Metric Law is vacuous in 2D and the Poincaré boundary has infinite length (was F-004)
- Location: Remark ($n = 2$ case) in the Lemma (lines 209-212); Corollary (Poincaré disk, $D = 2$) in `thm-causal-information-bound` (269-275); Step 2 reference to $r_h$ for $D = 2$ (287)
- Severity: Moderate
- Type: Scope restriction (secondary: Proof gap / omission)
- Criterion: External
- Origin: this chapter (same assertion at upstream `10_appendices/01_derivations.md:959-964`)
- Claim (verbatim): lines 209-211: "For $n=2$ ... The Poincare metric $G_{ij} = 4\delta_{ij}/(1-|z|^2)^2$ is the correctly regularized saturation geometry, with the horizon at $|z|=1$"; lines 269-272: "For the 2-dimensional Poincare disk, the formula reduces to ... $\frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L}$", with line 265: "$\text{Area}(\partial\mathcal{Z}) = \oint_{\partial\mathcal{Z}} dA_G$ ... in the induced metric".
- Upstream anchor: `05_geometry/01_metric_law.md:180-188` (Metric Law); `10_appendices/01_derivations.md:959`: "For $n = 2$ (the Poincare disk case), the formula simplifies. The Poincare metric already encodes the horizon at $|z| = 1$".
- Why this is an error: (i) In two dimensions $R_{ij} = \tfrac12 R\,G_{ij}$, so $R_{ij} - \tfrac12 R\,G_{ij} \equiv 0$ and the Metric Law with $T_{ij} = \sigma G_{ij}$ reduces to $(\Lambda - \kappa\sigma)G_{ij} = 0$, which either has no solution or is satisfied by every metric; it does not single out the Poincaré metric, so "the correctly regularized saturation geometry" is unsupported. (ii) The Lemma is stated for $n \ge 3$, yet line 287 uses "the horizon radius $r_h$ from Lemma ..." for general $D$, including the $D = 2$ corollary. (iii) In the Poincaré metric the boundary $|z| = 1$ is at infinite distance and has infinite induced length; at the truncation $|z| = 1 - \varepsilon$ the length is $4\pi(1-\varepsilon)/(\varepsilon(2-\varepsilon)) = 2\pi/\varepsilon - \pi + O(\varepsilon)$ (sympy). So "$\text{Area}(\partial\mathcal{Z})$ in the induced metric" is infinite (bound vacuous) or cutoff-dependent, and the remark's "$\varepsilon$ tied to Levin length" makes $I_{\max}$ depend on $\ell_L$ twice (through $\varepsilon$ and through $4\ell_L$) without saying how.
- Impact on downstream results: The headline "$I_{\max} = \text{Area}/(4\ell_L)$" and every Bekenstein-Hawking analogy in the chapter; the $\eta_{\text{Sch}}$ special case (line 410); `07_cognition/07_metabolic_transducer.md:598`.
- Fix guidance:
  1. State `thm-causal-information-bound` for $D \ge 3$, where the Lemma applies, or supply a separate 2D argument that does not use the Metric Law.
  2. For the Poincaré disk, specify the cutoff $\varepsilon$ and write $I_{\max}(\varepsilon)$ explicitly, making the dependence on $\ell_L$ single-valued.
- Required new assumptions/permits: an explicit cutoff prescription for the $D = 2$ case.
- Validation plan: Confirm that no statement invoking $r_h$ or the Lemma is applied at $D = 2$; check that $I_{\max}(\varepsilon)$ is finite and has units of nats.

### [E-006] Step 1 (Holographic Reduction) rests on a false identity $\int R\,d\mu_G = 2\oint\mathrm{Tr}K\,dA_G$ and on an unrelated source (was F-006)
- Location: `thm-causal-information-bound` proof sketch, Step 1 (lines 279-285)
- Severity: Critical
- Type: Invalid inference (secondary: External dependency)
- Criterion: External
- Origin: upstream `docs/source/1_agent/10_appendices/01_derivations.md:871-898` (Lemma A.6.1), inherited verbatim
- Claim (verbatim): lines 279-282: "The bulk-to-boundary conversion relies on the Einstein tensor divergence identity (valid in arbitrary dimension): integrating the scalar curvature over a compact manifold with boundary yields a boundary term involving the extrinsic curvature ... $I_{\text{bulk}} = \int_{\mathcal{Z}}\rho_I\,d\mu_G = \frac{1}{\kappa}\oint_{\partial\mathcal{Z}}\text{Tr}(K)\,dA_G$".
- Upstream anchor: `10_appendices/01_derivations.md:893`: "$\int_{\mathcal{Z}} R\,d\mu_G = 2\oint_{\partial\mathcal{Z}}\text{Tr}(K)\,dA_G$", justified by "Integrating the Einstein tensor identity over $\mathcal{Z}$ and applying Lemma `lem-a-divergence-to-boundary-conversion`"; that lemma (`:186-196`) is the Riemannian divergence theorem $\int \operatorname{div}_G \mathbf{j} = \oint\langle\mathbf{j},\mathbf{n}\rangle$ for a vector field.
- Why this is an error: (i) The scalar curvature is not a divergence for $n \ge 3$; no identity expresses $\int_{\mathcal{Z}} R\,d\mu_G$ as a boundary integral of $\mathrm{Tr}K$. Counterexample: the closed upper hemisphere of the unit $S^3$ has $R = 6$ and volume $\pi^2$, so $\int R = 6\pi^2$, while its boundary (the equatorial $S^2$) is totally geodesic with $\mathrm{Tr}K = 0$. For $n = 2$ the Gauss-Bonnet theorem gives $\int R\,dA = 4\pi\chi - 2\oint k_g\,ds$, a topological term plus a boundary term of the opposite sign (flat disk: $\int R = 0$ while $2\oint k_g = 4\pi$). (ii) Even granting such an identity, the Metric Law relates $R$ to the trace of the Risk Tensor (E-003), not to $\rho_I$; the step "$\int\rho_I\,d\mu_G = \ldots$" needs $\rho_I \propto R$, which is nowhere derived. (iii) Units: $\oint\mathrm{Tr}K\,dA_G \sim [z]^{D-2}$, so $\kappa \sim [z]^{D-2}/\mathrm{nat}$, incompatible with Step 3's $\kappa = 8\pi\ell_L^{D-1}$ and with `01_metric_law.md:188,242` ($\kappa T_{ij}$ must have units $[z]^{-2}$ with $[T_{ij}] = \mathrm{nat}^2/[z]^2$, i.e. $\kappa \sim \mathrm{nat}^{-2}$). This is the first step of the chapter's proof of its main theorem and the only route to $\nu_D$ for general $D$; the microstate count of Appendix A.6.0h states at `:862` that it does not invoke the Metric Law and only produces a 2-dimensional-boundary coefficient.
- Impact on downstream results: The field-theoretic derivation of `thm-causal-information-bound` collapses; only the independent, $D = 3$-flavoured microstate counting of Appendix A.6.0 remains, and it gives a different coefficient (E-001). `intro_agent.md:376` ("Derived rigorously from the Capacity-Constrained Metric Law via generalized Gauss-Bonnet identity") inherits the error.
- Fix guidance:
  1. Remove Step 1, or replace it with a genuine permit: in $D = 2$ use Gauss-Bonnet with the topological term made explicit and the correct sign.
  2. In $D \ge 3$ no such reduction exists, so derive the bound from the DPI / channel-capacity argument alone and make that argument dimensionally consistent.
  3. Update `intro_agent.md:376` and Appendix A.6.1 to match.
- Required new assumptions/permits: a correct bulk-to-boundary statement (Gauss-Bonnet for $D = 2$) or an explicit acknowledgment that the bound is not derived from the Metric Law.
- Validation plan: Test any proposed identity on the hemisphere of $S^3$ and the flat disk; unit-check $\kappa$ against `01_metric_law.md:188,242`.

### [E-007] Step 2: $\mathrm{Tr}K \to (D-1)/r_h$ is wrong at a $G_{rr}$-divergence; $\mathrm{Tr}K \to 0$ there (was F-007)
- Location: Proof sketch, Step 2 (line 287)
- Severity: Major
- Type: Computational error
- Criterion: External
- Origin: upstream `docs/source/1_agent/10_appendices/01_derivations.md:1024`, inherited verbatim
- Claim (verbatim): line 287: "At the saturation limit, the extrinsic curvature approaches $\text{Tr}(K) \to (D-1)/r_h$ where $r_h$ is the horizon radius from Lemma ...".
- Upstream anchor: `10_appendices/01_derivations.md:1024`: "**At saturation:** The extrinsic curvature $\text{Tr}(K) = (n-1)/r_h$ for an $(n-1)$-sphere boundary."
- Why this is an error: For $ds^2 = A(r)dr^2 + r^2 d\Omega_{D-1}^2$ the unit normal to $r = \text{const}$ is $n = A^{-1/2}\partial_r$ and $\mathrm{Tr}K = (D-1)/(r\sqrt{A(r)})$; sympy (computing $\tfrac12 g^{ab} n^r\partial_r g_{ab}$ with $n^r = \sqrt f$) gives $(n-1)\sqrt f/r$, which equals the flat value $(n-1)/r$ only when $f = 1$. At the claimed horizon $A(r_h) \to \infty$, so $\mathrm{Tr}K \to 0$: the horizon sphere is a minimal (totally geodesic) surface, as for the throat of the Schwarzschild time-symmetric slice. Inserting the correct value into Step 1 gives $I_{\text{bulk}} \to 0$ at saturation, the opposite of the intended conclusion. Step 2 also assumes $\partial\mathcal{Z}$ is a round sphere, whereas the theorem is stated for arbitrary $\partial\mathcal{Z}$ with $\text{Area} = \oint dA_G$; the sphere assumption is a hidden hypothesis.
- Impact on downstream results: The value of $\nu_D$ (its factor $(D-1)$ comes from this step), hence $I_{\max}$ and $\eta_{\text{Sch}}$.
- Fix guidance:
  1. Use $\mathrm{Tr}K = (D-1)/(r\sqrt A)$ and re-derive; if the bound is to be evaluated at the horizon the Step-1 identity cannot be used.
  2. Restrict the theorem to spherical boundaries or prove the general case.
- Required new assumptions/permits: spherical boundary hypothesis if retained.
- Validation plan: Recompute $\mathrm{Tr}K$ for the replacement metric symbolically; check the limit as $A \to \infty$.

### [E-008] Steps 1-3 do not assemble to the boxed formula (missing factor $\Omega_{D-1}r_h$), and $\kappa$ is assigned three incompatible values (was F-008)
- Location: Proof sketch, Steps 1-3 and "Combining these steps yields the general bound" (lines 279-291); Physics box (57)
- Severity: Major (verifier adjusted from Critical: the finding is correct, but it is a local computational error in the final assembly of a derivation already invalidated at Steps 1 and 2 by E-004 and E-006; it invalidates no result those two do not already invalidate, and its fix is local)
- Type: Computational error (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: upstream `docs/source/1_agent/10_appendices/01_derivations.md:1017-1041` (Thm A.6.6), inherited; the value $\kappa = 8\pi\ell_L^{D-1}$ is introduced in this chapter (the appendix says $8\pi\ell_L^2$)
- Claim (verbatim): line 282: "$I_{\text{bulk}} = \frac{1}{\kappa}\oint\text{Tr}(K)\,dA_G$"; line 287: "$\text{Tr}(K) \to (D-1)/r_h$ ... $\text{Area}(\partial\mathcal{Z}) = \Omega_{D-1}r_h^{D-1}$"; line 289: "The coupling constant $\kappa = 8\pi\ell_L^{D-1}$ is fixed by consistency with the Fisher Information Metric"; line 291: "Combining these steps yields the general bound", i.e. $I_{\max} = \nu_D\,\text{Area}/\ell_L^{D-1}$.
- Upstream anchor: `10_appendices/01_derivations.md:1028`: "**Fisher normalization:** $\kappa = 8\pi\ell_L^2$"; `:1033`: "$I_{\max} = \frac{1}{8\pi\ell_L^{n-1}}\cdot\frac{n-1}{r_h}\cdot\Omega_{n-1}r_h^{n-1} = \frac{(n-1)\Omega_{n-1}}{8\pi}\cdot\frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{n-1}}$"; `05_geometry/01_metric_law.md:188`: "$\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units".
- Why this is an error: Multiplying the three steps exactly as stated (sympy-checked):
  $I = \frac{1}{8\pi\ell_L^{D-1}}\cdot\frac{D-1}{r_h}\cdot\Omega_{D-1}r_h^{D-1} = \frac{(D-1)\Omega_{D-1}}{8\pi}\cdot\frac{r_h^{D-2}}{\ell_L^{D-1}} = \nu_D\,\frac{\text{Area}(\partial\mathcal{Z})}{\Omega_{D-1}\,r_h\,\ell_L^{D-1}}$,
  which differs from the boxed $\nu_D\,\text{Area}/\ell_L^{D-1}$ by exactly the factor $\Omega_{D-1}r_h$; the appendix's second equality at `:1033` replaces $r_h^{D-2}$ by $\Omega_{D-1}r_h^{D-1}$. The assembled quantity is not dimensionless: $r_h^{D-2}/\ell_L^{D-1} \sim [z]^{-1}$, not nats. Moreover $\kappa$ is given as $8\pi\ell_L^{D-1}$ here, $8\pi\ell_L^2$ in the appendix, and must be $\sim\mathrm{nat}^{-2}$ by the Metric Law's own unit statement (`01_metric_law.md:188,242`); no "consistency with the Fisher Information Metric" argument is given anywhere (Appendix A.6.4 only restates the $4\ell_L^2$ cell area). The $1/(8\pi)$ in $\nu_D$ is therefore an imported GR normalization, not a derived quantity. The table values of $\nu_D$ and the peak at $D = 9$ ($\nu_9 \approx 9.45$) are arithmetically correct given the definition (checked numerically), but the definition itself is unsupported.
- Impact on downstream results: The definition of $\nu_D$ (`def-holographic-coefficient`) and its table; all downstream uses of $\nu_D$ listed under E-002.
- Fix guidance:
  1. Either drop the field-theoretic proof entirely and present $\nu_D$ as a definition / normalization convention justified by a correct counting argument;
  2. or redo Steps 1-3 with correct identities (E-004, E-006, E-007) and a single, dimensionally consistent $\kappa$.
  3. Reconcile the value of $\kappa$ between this chapter, Appendix A.6.6 and `01_metric_law.md:188,242`.
- Required new assumptions/permits: an explicit normalization convention for $\nu_D$ if option 1 is taken.
- Validation plan: Multiply the final steps symbolically and confirm the result equals the boxed formula and has units of nats; grep `\kappa =` across the book for a single value.

### [E-009] Causal Stasis proves only $v^r \to 0$ on the horizon sphere, not $\|v\|_G \to 0$ (was F-010)
- Location: `thm-causal-stasis` statement and proof (lines 324-350); Summary table row "Causal Stasis" (466)
- Severity: Major
- Type: Proof gap / omission (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 327-331: "As $I_{\text{bulk}} \to I_{\max}$: $\|v\|_G \to 0$"; proof, lines 345-350: "Lemma ... implies $G_{rr} \to \infty$, hence $G^{rr} \to 0$. The radial component of velocity: $v^r = -G^{rr}\partial_r\Phi_{\text{eff}} \to 0$. $\blacksquare$".
- Upstream anchor: not applicable (statement and proof are this chapter's); `lem-metric-divergence-at-saturation` gives divergence only at $r = r_h$ (lines 216-222); `def-effective-potential` at `04_equations_motion.md:418` includes a critic term $V_{\text{critic}}(z,K)$.
- Why this is an error: (i) For the ansatz $G^{-1} = \mathrm{diag}(1/A, r^{-2}g_S^{-1})$, so $\|v\|_G^2 = G_{rr}(v^r)^2 + r^2 g_S(v^\Omega, v^\Omega)$ with $v^\Omega = -G^{\Omega\Omega'}\partial_{\Omega'}\Phi_{\text{eff}}$; only the $rr$ block depends on $A(r)$, so the angular contribution $r^{-2}\|\nabla_S\Phi_{\text{eff}}\|^2$ does not vanish unless $\Phi_{\text{eff}}$ is spherically symmetric, a hypothesis not stated and false for the actual $\Phi_{\text{eff}}$. The radial part does vanish: $G_{rr}(v^r)^2 = G^{rr}(\partial_r\Phi_{\text{eff}})^2 \to 0$. (ii) Even the radial statement holds only at $r = r_h$ (the boundary), not for beliefs in the interior $r < r_h$, so "the agent's internal dynamics freeze" is not what is proved. (iii) The hypothesis "$I_{\text{bulk}} \to I_{\max}$" is never connected to "the boundary of $\mathcal{Z}$ approaches $r_h$" (E-003); the proof simply says "as the information density approaches saturation".
- Impact on downstream results: `cor-saturation-velocity-tradeoff`, Node 56 thresholds ("Update velocity degraded"), FAQ `10_appendices/04_faq.md:556`.
- Fix guidance:
  1. Restate: "Under the hypotheses of Lemma ..., the radial update velocity at the horizon satisfies $v^r \to 0$".
  2. Add spherical symmetry of $\Phi_{\text{eff}}$ as a hypothesis if the full norm is wanted.
  3. Add a lemma linking $I_{\text{bulk}}/I_{\max}$ to the position of $\partial\mathcal{Z}$ relative to $r_h$.
- Required new assumptions/permits: spherical symmetry of $\Phi_{\text{eff}}$ (for the full-norm statement); a link between saturation ratio and boundary position.
- Validation plan: Write out $\|v\|_G^2$ componentwise for the corrected metric and verify which terms vanish in the limit and where.

### [E-010] Causal Stasis proof cites an equation of motion that matches neither the cited definition nor the overdamped limit (was F-009)
- Location: `thm-causal-stasis` proof (lines 333-344)
- Severity: Moderate
- Type: Citation / reference error (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (the equation form exists upstream in the summary table `04_equations_motion.md:1289`, so the mislabelling is partly inherited)
- Claim (verbatim): lines 333-342: "From the Equation of Motion (Definition {prf:ref}`def-bulk-drift-continuous-flow`) ...: $dz^k = \left(-G^{kj}\partial_j\Phi_{\text{eff}} + u_\pi^k - \Gamma^k_{ij}\dot z^i\dot z^j\right)ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j_s$. The drift velocity scales as: $v^k \propto G^{kj}\partial_j\Phi_{\text{eff}}$".
- Upstream anchor: `05_geometry/04_equations_motion.md:220-226` (`def-bulk-drift-continuous-flow`, "Second-Order Geodesic Langevin Equation"): "$dz^k = G^{kj}(z)\,p_j\,ds$; $dp_k = [-\partial_k\Phi_{\text{eff}} - \gamma p_k + \beta_{\text{curl}}\mathcal{F}_{kj}G^{j\ell}p_\ell - \Gamma^m_{k\ell}G^{\ell j}p_jp_m + u_{\pi,k}]ds + \sqrt{2\gamma T_c}(G^{1/2})_{kj}dW^j_s$"; summary table row "Full Geodesic SDE" at `:1289`; overdamped form at `:1291`: "$dz = \mathcal{M}_{\text{curl}}(-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi)\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$".
- Why this is an error: The cited definition is the second-order $(z,p)$ system, so the citation is wrong as stated. The displayed equation is the "Full Geodesic SDE" row of the summary table minus the curl term (that row itself mixes a first-order left side with a $\Gamma(\dot z,\dot z)$ term); it is a first-order SDE in $z$ containing a second-order geodesic term (an acceleration added to a velocity) and a noise coefficient $\sqrt{2T_c}$ without $\gamma$. It is neither the cited definition nor the book's overdamped limit (which has no $\Gamma$ term and includes $\mathcal{M}_{\text{curl}}$). The substantive gap: in the cited definition $v^k = G^{kj}p_j$, so $v^r \to 0$ additionally requires $p_r$ to stay bounded as $G^{rr} \to 0$, which is not argued (the force $-\partial_r\Phi_{\text{eff}}$ acts on $p_r$ with no $G^{rr}$ suppression, and the noise coefficient $\sqrt{2\gamma T_c}(G^{1/2})_{rr}$ grows).
- Impact on downstream results: The proof of `thm-causal-stasis` and thus `cor-saturation-velocity-tradeoff`.
- Fix guidance:
  1. Cite and use the overdamped equation (`04_equations_motion.md:1291`, `thm-overdamped-limit`), where $v = -G^{-1}\nabla\Phi_{\text{eff}}$ holds directly; or
  2. work with the second-order system and add a boundedness hypothesis on $p$ (e.g. via the friction term).
- Required new assumptions/permits: overdamped regime, or bounded momentum.
- Validation plan: Check that the displayed equation is character-for-character one of the book's stated forms and that the velocity relation used follows from it.

### [E-011] Saturation-velocity scaling $(1-\eta)^{1/2}$: identification $\eta = \mu/\mu_{\max}$ unproven and direction reversed under the corrected solution (was F-011)
- Location: `cor-saturation-velocity-tradeoff` (lines 357-369)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 366: "Under uniform saturation, the information mass $\mu(r)$ grows with radius. At the horizon, $\mu(r_h) = \mu_{\max}$. The saturation ratio $\eta := I_{\text{bulk}}/I_{\max} = \mu/\mu_{\max}$ ... Near the horizon, $G^{rr} \sim (1-\mu/\mu_{\max}) = (1-\eta)$. Since velocity scales as $v^r \propto G^{rr}$, we have $\|v\| \sim (G^{rr})^{1/2} \sim (1-\eta)^{1/2}$".
- Upstream anchor: `lem-metric-divergence-at-saturation` (lines 204-207) for $A(r)$ and $\mu(r)$, defined in this chapter (see E-004).
- Why this is an error: (i) "$\mu(r)$ grows with radius" varies $r$ at fixed $\sigma_{\max}$, whereas "$\eta$ = fraction of capacity used" varies the information content at a fixed boundary; the proof switches between the two without justification, and $\mu/\mu_{\max} = I_{\text{bulk}}/I_{\max}$ holds only under the unproved identification of E-003. (ii) With the lemma's $A(r)$ taken at face value, $G^{rr}(r_h) = 1 - \Lambda_{\text{eff}}r_h^2/(n(n-1)) - 2\mu/((n-2)r_h^{n-2})$ is linear in $\mu$ at fixed $r_h$, but $\Lambda_{\text{eff}}$ also contains $\kappa\sigma_{\max}$, so "$\mu_{\max}$" is not a fixed constant; linearity survives only because both terms are linear in $\sigma_{\max}$. (iii) Under the actual solution of the Metric Law (E-004), $G^{rr} = 1 - 2(\Lambda-\kappa\sigma)r^2/((n-1)(n-2))$ increases with $\sigma$ at fixed $r$, so the velocity would grow, not vanish, with information content. The arithmetic of the interpretation ($\sqrt{0.1} = 0.316$, $\sqrt{0.01} = 0.10$) is correct given the scaling.
- Impact on downstream results: Node 56 warning thresholds (lines 415-416); FAQ.
- Fix guidance:
  1. Resolve E-003 and E-004 first.
  2. At minimum state $\eta = \mu/\mu_{\max}$ as a modelling assumption and specify what is held fixed.
- Required new assumptions/permits: the identification $\eta = \mu/\mu_{\max}$ as an explicit assumption.
- Validation plan: Differentiate $G^{rr}$ with respect to the information content under the corrected metric and check the sign.

### [E-012] "Def: Metabolic Pruning Criterion" does not exist (was F-012)
- Location: Remediation list, item 2 (line 428)
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 428: "**Chart Pruning**: Remove low-utility charts (Definition **Def: Metabolic Pruning Criterion**)."
- Upstream anchor: `grep -rn -i "metabolic pruning" docs/source/1_agent` returns only `03_info_bound.md:428`; no `prf:definition` with that title or a corresponding label exists (the codebook-liveness / symbol-utility material is in `07_cognition/04_ontology.md`, e.g. `node-codebook-liveness-check` at `:989`).
- Why this is an error: A remediation step points to a definition that is not in the book, and the pointer is plain bold text rather than a resolvable reference.
- Impact on downstream results: None mathematical; the remediation is unactionable as written.
- Fix guidance:
  1. Point to the actual criterion (e.g. Symbol Utility $U_k$ / `node-codebook-liveness-check` in `04_ontology.md`) with a `{prf:ref}`, or add the missing definition.
- Required new assumptions/permits: none.
- Validation plan: Build the docs and confirm the reference resolves.

### [E-013] Computational proxy for $\eta_{\text{Sch}}$ is not a proxy for $I_{\text{bulk}}/I_{\max}$ (was F-013)
- Location: "Computational Proxy" (lines 432-438)
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 435-438: "$\hat\eta_{\text{Sch}} = \frac{|\mathcal{K}|\cdot\bar H(z_n\mid K)}{\log|\mathcal{K}| + d_n\cdot\log(A_{\text{eff}}/\ell_L^2)}$, where $|\mathcal{K}|$ is the number of active charts, $\bar H(z_n\mid K)$ is the average conditional entropy of nuisance coordinates, $d_n = \dim(z_n)$, and $A_{\text{eff}}$ is the effective boundary area."
- Upstream anchor: `def-capacity-horizon-diagnostic` (line 402): $\eta_{\text{Sch}} = I_{\text{bulk}}/(\nu_D\,\text{Area}/\ell_L^{D-1})$; `05_geometry/01_metric_law.md:106-112` ($I_{\text{bulk}} = \int\rho_I\,d\mu_G$, an entropy in nats).
- Why this is an error: (i) Numerator: $H(K, z_n) = H(K) + H(z_n\mid K) \le \log|\mathcal{K}| + \bar H(z_n\mid K)$; multiplying the average conditional entropy by $|\mathcal{K}|$ overcounts by a factor of about $|\mathcal{K}|$. (ii) Denominator: $I_{\max}$ is linear in the boundary area ($\nu_D\,A/\ell_L^{D-1}$), whereas the proxy is logarithmic in $A_{\text{eff}}/\ell_L^2$ and uses the exponent 2 regardless of $D$. The ratio therefore does not approximate $\eta_{\text{Sch}}$ in any regime and can exceed 1 by a wide margin for moderate $|\mathcal{K}|$: with $|\mathcal{K}| = 512$, $\bar H = 1$ nat, $d_n = 8$, $A_{\text{eff}}/\ell_L^2 = 10^6$ the numerator is 512 and the denominator $\ln 512 + 8\ln 10^6 \approx 116.7$, giving $\hat\eta \approx 4.38$ (recomputed by the verifier). No derivation or justification is given.
- Impact on downstream results: Any implementation of Node 56 using this proxy; the thresholds 0.5 / 0.9 / 0.99 become meaningless.
- Fix guidance:
  1. Use $\hat I_{\text{bulk}} = \hat H(K) + \sum_k P(k)\hat H(z_n\mid K = k)$.
  2. Use $\hat I_{\max} = \nu_D\,\hat A_{\text{eff}}/\ell_L^{D-1}$, or the operational $C_\partial \approx \mathbb{E}[I(X_t;K_t)]$ of `01_metric_law.md:143-147`.
  3. State which of $C_\partial$ / $I_{\max}$ is being estimated (E-002).
- Required new assumptions/permits: none.
- Validation plan: Evaluate the revised proxy on synthetic cases with known $I_{\text{bulk}}$ and area and confirm it lies in $[0,1]$ and tracks the exact ratio.

### [E-014] Unified notation table misstates $\rho_I$ and overloads $\kappa$ and $\eta$ against this chapter's own usage (was F-014)
- Location: Core Symbols table (lines 492-565)
- Severity: Minor
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 496: "$\rho_I(z,t)$ | Information density | $-\rho\log\rho + \frac12\rho\log\det G$"; line 500: "$\kappa$ | Screening mass | $-\ln\gamma/(c_{\text{info}}\Delta t)$ | $[z]^{-1}$"; line 508: "$\eta$ | Harmonic Flux", while the chapter also uses $\eta := I_{\text{bulk}}/I_{\max}$ (360), $\eta_\ell$ (152), and $\eta_t$ inside $\Lambda_t$ (523).
- Upstream anchor: `05_geometry/01_metric_law.md:86-95`: "$\rho_I(z,s) := -\rho(z,s)\log\rho(z,s)$ ... By defining $\rho$ as a density with respect to $d\mu_G$ ... no explicit metric correction term is needed." (The $\tfrac12\rho\log\det G$ term appears only in a `feynman-added` note at `:100-103`, not in the formal definition.)
- Why this is an error: The table gives a formula for $\rho_I$ that the formal upstream definition explicitly rejects; within this chapter $\kappa$ denotes the Metric-Law coupling (lines 207, 282, 289) while the table defines it as the screening mass with different units; and $\eta$ is used for four different objects. These are the symbols a reader of this chapter is most likely to look up.
- Impact on downstream results: Reader confusion only; no formula in the chapter uses the extra $\log\det G$ term.
- Fix guidance:
  1. Align the $\rho_I$ row with `def-information-density-and-bulk-information-volume`.
  2. Add a row or footnote distinguishing $\kappa_{\text{coupling}}$ from $\kappa_{\text{screen}}$ (or rename one).
  3. Rename the corollary's $\eta$ to $\eta_{\text{Sch}}$ (it is the same quantity).
- Required new assumptions/permits: none.
- Validation plan: Grep each table symbol against its uses in the chapter and confirm one meaning per symbol.

### [E-015] Diagnostic Node Registry: Node-40 anchor resolves to PurityCheck; registry labelled "Complete" omits Nodes 57-60; table broken at line 682 (was F-015)
- Location: Diagnostic Node Registry (lines 637-700), rows 40 and 56-61; stray paragraph at line 682
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter (registry); upstream for the duplicate node number (`07_cognition/01_supervised_topo.md:1149-1150` vs `05_geometry/01_metric_law.md:310-330`)
- Claim (verbatim): line 683: "| 40 | [CapacitySaturationCheck](#node-40) | 18.3 | $I_{\text{bulk}}/C_\partial$ |"; line 638: "## Diagnostic Node Registry (Complete)"; line 682: "Here $v := \dot z$ and $\mathcal{M}_\gamma^{-1} = \ldots$" inserted between rows 39 and 40.
- Upstream anchor: `07_cognition/01_supervised_topo.md:1149-1150`: "(node-40)= **Node 40: PurityCheck (CapacitySaturationCheck)**" with proxy $H(Y\mid K)$; `05_geometry/01_metric_law.md:310-330` defines CapacitySaturationCheck (Node 40, $\nu_{\text{cap}}$) under `sec-diagnostic-node-capacity-saturation` with no `node-40` anchor; Nodes 57-60 are anchored at `08_multiagent/01_gauge_theory.md:3683,3703,3723,3751`.
- Why this is an error: The registry row for Node 40 (formula $I_{\text{bulk}}/C_\partial$, section 18.3) links to an anchor whose content is a different diagnostic (class purity, $H(Y\mid K)$); the number 40 is used for two different nodes in the book. The registry claims completeness yet skips 57-60 and jumps from 56 to 61. The prose line at 682 splits the markdown table, so rows 40-61 render as a header-less second table or as plain text.
- Impact on downstream results: Navigation and consistency of Sieve node numbering; the `def-capacity-horizon-diagnostic` cross-reference (line 418) to Node 40 is fine because it uses the section label.
- Fix guidance:
  1. Link row 40 to `sec-diagnostic-node-capacity-saturation`.
  2. Resolve the duplicate numbering upstream.
  3. Add rows 57-60.
  4. Move the "Here $v := \dot z$ ..." note below the table.
- Required new assumptions/permits: none.
- Validation plan: Build the docs and check the rendered registry table and the Node 40 link target.

## Scope restrictions and clarifications
- The findings against the Area Law derivation (E-004, E-006, E-007, E-008) concern material inherited verbatim from Appendix A.6 of `10_appendices/01_derivations.md`; the appendix should be corrected in the same pass, since this chapter presents the steps as its own lemma and proof sketch.
- The table of $\nu_D$ values and the peak at $D = 9$ ($\nu_9 \approx 9.45$) are arithmetically correct given `def-holographic-coefficient`; the objection is to the derivation of the formula, not to the arithmetic.
- The arithmetic in the interpretation of `cor-saturation-velocity-tradeoff` ($\sqrt{0.1} = 0.316$, $\sqrt{0.01} = 0.10$) is correct given the claimed scaling.
- The mechanical cross-reference pre-pass (`xref_report.md`) found no dangling `{prf:ref}` / `{ref}` targets or duplicate labels for this file; E-012 and E-015 concern plain-text pointers and a raw `#node-40` anchor that the pre-pass does not cover.
- The microstate counting of Appendix A.6.0g/h is independent of the Metric Law (per `:862`) and is not itself reviewed here beyond its dimensional mismatch with this chapter's $\nu_2$, $\nu_3$ (E-001).

## Open questions
- Which convention should fix the Levin Length and the holographic coefficient: $\ell_L^{D-1}$ as boundary volume per nat (making $I_{\max} \equiv C_\partial$), or the appendix's $4\ell_L^2$ cell on a 2-dimensional boundary (which is a $D = 3$ statement)? E-001 and E-002 cannot be closed independently of this choice.
- Is a Schwarzschild-type saturation horizon still wanted? If so, a non-uniform stress profile and a genuine coupling between the Risk Tensor and $\rho_I$ (E-003, E-004) are prerequisites; otherwise Causal Stasis and Node 56 need a different mechanism.
- Counting note: the verification log's Summary line reports "Major: 5", but its per-finding verdict table assigns final severity Major to six findings (F-001, F-002, F-005, F-007, F-008, F-010). This review follows the per-finding verdicts, giving Major: 6 and a total of 15 kept findings.

## Rejected candidate findings
- None. All fifteen stage-1 findings were confirmed or adjusted by the verifier; no candidate was rejected.
