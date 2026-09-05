# Mathematical Review: docs/source/1_agent/05_geometry/03_holographic_gen.md

## Metadata
- Reviewed file: docs/source/1_agent/05_geometry/03_holographic_gen.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (793 lines)
- Framework anchors (definitions/axioms/permits):
  - 05_geometry/04_equations_motion.md: `def-bulk-drift-continuous-flow` (lines 218-240), `def-effective-potential` (lines 400-430, 423, 427), Pure Control row (line 459), $\nabla_G U$ (line 477), $T_c$ units (line 515), `thm-overdamped-limit` (lines 912-930), `cor-recovery-of-holographic-flow` (lines 938-956), Node 25 (line 1310)
  - 01_foundations/01_definitions.md: state space $\mathcal Z=\mathcal K\times\mathcal Z_n\times\mathcal Z_{\text{tex}}$ (line 103), action space (lines 115-127, 238-242)
  - 01_foundations/02_control_loop.md: value function as cost-to-go (lines 139-143, 581-585, 1061-1065)
  - 06_fields/01_boundary_interface.md: `def-motor-texture-distribution` (lines 392-398), `ax-motor-texture-firewall` (lines 403-410), context-conditioned velocity $v_c$ (line 692)
  - 07_cognition/01_supervised_topo.md: $\Phi_{\text{eff}}=-\log p(y\mid z)$ (lines 52, 64)
  - 05_geometry/01_metric_law.md: `thm-capacity-constrained-metric-law` (lines 178-192), $I_{\text{bulk}}$ (lines 65-72, 109-113), Node 40 (line 325)
  - 02_sieve/02_limits_barriers.md: BarrierEpi (line 63); 02_sieve/01_diagnostics.md: Node 25 HoloGenCheck (line 123); 06_fields/03_info_bound.md:666
  - 08_multiagent/01_gauge_theory.md: `thm-higgs-mechanism` (lines 2071, 2091-2098), stabiliser at origin (line 1376), cross-link (line 1321)
  - 03_architecture/01_compute_tiers.md: `cor-the-hyperbolic-embedding` (line 1852)
  - 10_appendices/01_derivations.md:1177 and 08_multiagent/03_parameter_sieve.md:116 (Levin length)
  - Mechanical xref pre-pass: no dangling references or duplicate labels for this file; all 17 `{prf:ref}`/`{ref}` targets located upstream.

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 8
- Minor: 10
- Notes: 2
- Primary themes:
  1. The chapter's central "policy picks the direction at the origin" argument rests on the false claim $F_{\text{entropy}}(0)=0$. The chapter's own formula gives $|F_{\text{entropy}}|\to\tfrac12$ at the origin, and the later theorem explicitly says the origin is not a fixed point (E-005).
  2. The Angular Symmetry Breaking theorem is computationally wrong: the radial SDE omits the Itô (Bessel) drift and the radial component of $u_\pi$ (the "traceless Hessian" justification is reversed), the ratio $\eta$ is neither dimensionless nor the drift/diffusion ratio it claims to be, and the freeze-out conclusion $\eta\to\infty$ becomes $\eta\to0$ once the chapter's own definition $u_\pi=G^{-1}(\cdot)\propto(1-r^2)^2$ is substituted (E-011, E-012). The "phase transition / pitchfork" language and the Physics box restating the theorem with different formulas compound this (E-014, E-015).
  3. Sign conventions collide with the anchors: $V$ is a cost-to-go that controlled flow must decrease, yet the RL control field is $+G^{-1}\nabla V$ "toward high value"; the context-conditioned remark writes $u_\pi=+G^{-1}\nabla\Phi_{\text{eff}}$ while 06_fields writes the potential term as $-G^{-1}\nabla\Phi_{\text{eff}}$ separately from $u_\pi$ (E-008, E-010).
  4. Several formal-looking cross-claims are unsupported or mis-cited: the motor-texture "duality" $\Sigma_{\text{motor}}=\omega\Sigma_{\text{visual}}\omega^{-1}$, the BarrierEpi semantics, an "information stopping criterion" attributed to the metric law theorem, and a placeholder reference in `conn-rl-24` (E-017, E-018, E-021, E-022).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Intro (40); Def. Entropic Force (116-129); Def. Hyperbolic Information Potential remark (179); summary (758) | Moderate | Conceptual | Framework | this chapter | $S=2\operatorname{artanh} r$ is minimal at the origin, yet Def. 21.1.5 and the summary call the origin "maximum entropy" |
| E-002 | Def. Manifold Boundary and Interior (62-79) | Minor | Dimensional mismatch | External | this chapter | $\{z\in\mathbb C^n:\lvert z\rvert=1\}$ is $(2n-1)$-dimensional, not $(n-1)$; three ambient spaces used interchangeably |
| E-003 | Def. Hyperbolic Volume Growth (87-98); used at 377, 518 | Minor | Scope restriction | External | this chapter | $4\pi\sinh^2(r/2)$ is the $\mathbb H^2$ area; the chapter otherwise works in $D$ dimensions |
| E-004 | Prop. Riemannian Gradient of $U$, remark (208) | Note | Notation conflict | Framework | upstream 03_architecture/01_compute_tiers.md:1852 | Depth-to-radius map differs by a factor 2 from `cor-the-hyperbolic-embedding`; this chapter's normalisation is the correct one |
| E-005 | Prop. SO(D) Symmetry at Origin item 2 (240); box (247-255); Def. Control Field at Origin (280-291); vs. lines 122, 337 | Major | Computational error | Framework | this chapter | Prefactor $\tfrac{1-\lvert z\rvert^2}{2}\to\tfrac12$ at $0$, so $F_{\text{entropy}}(0)=0$ is false; line 337 says the opposite |
| E-006 | Prop. SO(D) Symmetry at Origin cross-reference (243); Physics box (402) | Minor | Notation conflict | Framework | this chapter | $\mu^2$ sign convention opposite to `thm-higgs-mechanism`; latent $SO(D)$ identified with the gauge group without argument |
| E-007 | Def. The Control Field (257-268); summary table (777) | Minor | Dimensional mismatch | Framework | this chapter | $G^{-1}$ applied to $\mathbb E[a]$ with $a\in\mathcal A$ (discrete component); $u_\pi$ a vector here, a covector $u_{\pi,k}$ upstream |
| E-008 | Def. The Control Field, remark (270) | Moderate | Definition mismatch | Framework | this chapter | $u_\pi=+G^{-1}\nabla\Phi_{\text{eff}}$ climbs $-\log p(y\mid z)$ and cancels $v_c$ in 06_fields:692 |
| E-009 | Thm. Unified Control Interpretation (299-312) | Minor | Miswording | Framework | this chapter | Three mutually exclusive settings called "equivalent"; the policy gradient theorem does not produce $\nabla_z V$ |
| E-010 | Thm. Unified Control Interpretation RL row (306); Algorithm 21.2.6 (465-469); `conn-rl-24` (504); box (317) | Moderate | Parameter inconsistency (sign) | Framework | this chapter | $u_\pi=+G^{-1}\nabla V_{\text{critic}}$ climbs a cost-to-go and double-counts the $V$ term of $\Phi_{\text{eff}}$ |
| E-011 | Thm. Angular Symmetry Breaking, radial SDE (331-337) and proof (356-366) | Major | Computational error | Framework | this chapter | Radial SDE misses $u_\pi^r$ and the Bessel drift $\tfrac{T_c(1-r^2)^2}{4r}$; $\operatorname{tr}\operatorname{Hess} r=1/r$, $\operatorname{tr}\operatorname{Hess}\theta=0$ (line 366 is reversed) |
| E-012 | Thm. Angular Symmetry Breaking, Phase Transition (347-354), proof (368-370); prose (383-385) | Major | Invalid inference | Framework | this chapter | $\eta=r^2\cdot a^2/\sigma_\theta^2$ (units $\tau^{-2}$), not the drift/diffusion ratio; with Def. 21.2.2, $\eta\to0$, not $\infty$ |
| E-013 | Thm. Angular Symmetry Breaking, proof (356-362) | Minor | Proof gap / omission | Framework | this chapter | `thm-overdamped-limit` has $\mathcal M_\gamma$, $\Phi_{\text{gen}}$ and no $u_\pi$; $\alpha=1$, $\gamma_{\text{risk}}=0$, $\mathcal F=0$ unstated |
| E-014 | Thm. Angular Symmetry Breaking (347-354, 368); prose (385); box (422); summary (764) | Moderate | Miswording | Framework | this chapter | Constant-drift diffusion on $S^1$ has a smooth von Mises law in $T_c$; no bifurcation or critical point is derived |
| E-015 | Physics Isomorphism: Spontaneous Symmetry Breaking (398-423) | Moderate | Computational error | Framework | this chapter | Box noise $\sqrt{T_c(1-r^2)}/r$ and $\eta$ disagree numerically with the theorem ($1.73$ vs $1.06$; $0.33$ vs $0.89$) |
| E-016 | Line 492 | Note | Typo | External | this chapter | Stray `:::` closes nothing (the `::::` at 399 closes at 423) |
| E-017 | `conn-rl-24` (494-522) | Minor | Citation / reference error | Framework | this chapter | Unresolved placeholder "Theorem **Thm: Hyperbolic Volume Growth**" (a definition); diffusion SDE labelled "Standard RL" |
| E-018 | Remark (Motor Extension) (539) | Moderate | Citation / reference error | Framework | this chapter | `ax-motor-texture-firewall` is an axiom with no conjugation formula; $\omega$ undefined anywhere in Volume 1 |
| E-019 | Axiom Bulk-Boundary Decoupling item 1 (544-551); Def. 21.1.1 (65) | Minor | Notation conflict | Framework | this chapter | $\mathcal Z$ redefined as $\mathcal K\times\mathcal Z_n$; 01_definitions.md:103 includes $\mathcal Z_{\text{tex}}$; line 65 uses $\mathcal Z$ for the disk |
| E-020 | Box "Why Conformal Scaling is the Right Choice" (632-642) | Minor | Conceptual | External | this chapter | Max-ent under a first-power distance constraint is Laplace-type, not Gaussian; $z_{\text{tex}}$ is not a tangent vector at $z$ |
| E-021 | Prop. Epistemic Barrier (671-676) | Moderate | Definition mismatch | Framework | this chapter | BarrierEpi is information overload (02_limits_barriers:63); proposition has no proof and a different referent |
| E-022 | Def. Stopping Criterion (690-701) | Moderate | Citation / reference error | Framework | this chapter | Cited theorem is the curvature field equation; $I_{\text{bulk}}$ is a functional of $\Omega$, not of $z$ |
| E-023 | Node 25 block (782-791); TLDR (10) | Minor | Notation conflict | Framework | this chapter | Node named HoloGenCheck in 02_sieve, 04, 06; $\tau_{\max}$ undefined in Def. 21.3.4 |

## Detailed findings

### [E-001] Which end is "high entropy"? The sign of "entropy" flips between §21.0, Def. 21.1.3 and Def. 21.1.5 (was F-001)
- Location: intro paragraph (line 40); Def. The Entropic Force (lines 116-129); Def. Hyperbolic Information Potential, Remark (line 179); summary prose (line 758); also lines 6, 27, 186
- Severity: Moderate
- Type: Conceptual (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 40: "radial expansion of the latent state from the low-entropy origin ($z=0$) toward the high-entropy boundary ($|z| \to 1$)"; line 119: "entropic volume term $S(r) = 2\operatorname{artanh}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force"; line 179: "At origin ($z=0$): $U = 0$ (maximum potential, maximum entropy). At boundary ($|z| \to 1$): $U \to -\infty$ (minimum potential, fully specified)"; line 758: "Starting from the origin (maximum entropy, all possibilities), the agent flows outward toward the boundary (minimum entropy, specific commitment)".
- Upstream anchor: n/a (defined in this chapter).
- Why this is an error: Def. 21.1.3 defines $S(r)=2\operatorname{artanh}(r)$, which is $0$ at the origin and $+\infty$ at the boundary, and derives the outward force as $\nabla_G S$ "to maximize entropy". Def. 21.1.5 then sets $U=-S$ and its Remark calls the origin ($U=0$, i.e. $S=0$) the point of "maximum entropy" and the boundary "fully specified". Both statements concern the same scalar $S=-U$ and contradict each other: $S$ is minimal, not maximal, at the origin. Line 40 agrees with Def. 21.1.3 (boundary = high entropy); lines 6, 27, 179, 186 and 758 say the opposite. Two different quantities are being conflated: (i) the log-volume / microstate count, which grows outward, and (ii) the residual uncertainty of the generated output, which shrinks outward. The drift derivation is only about (i).
- Impact on downstream results: The interpretation of $U$ as "information content" feeds `def-effective-potential` (04_equations_motion.md:427) and the intuitive explanations of the whole chapter; readers cannot tell whether "maximizing entropy" means moving inward or outward.
- Fix guidance:
  1. Keep $S(r)=2\operatorname{artanh}(r)$ as the (log-)volume entropy and state explicitly that $S$ is minimal at the origin and increases outward.
  2. Rewrite the Remark of Def. 21.1.5 as "At origin $U=0$ (maximum potential, minimum committed information); at boundary $U\to-\infty$ (fully specified)".
  3. Replace "maximum entropy" at lines 6, 27, 179, 186, 758 with "maximum output uncertainty / minimal commitment", or introduce a second named quantity (e.g. conditional output entropy $H(x\mid z)$) if that is what is meant.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for "entropy" and check every occurrence against the single stated monotonicity of $S$.

### [E-002] Boundary dimension and ambient space in Def. 21.1.1 do not match (was F-002)
- Location: Def. Manifold Boundary and Interior (lines 62-79)
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 65-74: "Let $\mathcal{Z}$ be the latent manifold with Poincare disk model. The **boundary** is the $(n-1)$-dimensional limit set: $\partial\mathcal{Z} := \{z \in \mathbb{C}^n : |z| = 1\}$."
- Upstream anchor: n/a (defined in this chapter; line 60 sets $\mathbb{D}=\{z\in\mathbb{C}:|z|<1\}$, i.e. $n=1$).
- Why this is an error: The unit sphere in $\mathbb{C}^n\cong\mathbb{R}^{2n}$ is $(2n-1)$-dimensional, not $(n-1)$-dimensional. For the Poincaré disk (line 60, $\mathbb{C}^1$) the ideal boundary is the circle $S^1$, which is 1-dimensional, whereas the formula gives $n-1=0$. Later the chapter treats the space as real $D$-dimensional ($SO(D)$ at lines 228-241, `z: [B, D]` and `r_sq = (z**2).sum` in the code), where the boundary is $S^{D-1}$. The three descriptions ($\mathbb{C}$, $\mathbb{C}^n$, $\mathbb{R}^D$) are used interchangeably.
- Impact on downstream results: Cosmetic for the 2D computations; matters for any statement quantified over $D$ (E-003, E-012).
- Fix guidance:
  1. Write $\mathcal{Z}\cong\mathbb{B}^D=\{z\in\mathbb{R}^D:|z|<1\}$ with $\partial\mathcal{Z}=S^{D-1}$ ($(D-1)$-dimensional).
  2. Use $D=2$ ($\mathbb{D}\subset\mathbb{C}$) only where the explicit polar computations require it, and say so.
- Required new assumptions/permits: none.
- Validation plan: check that every later use of $D$, $SO(D)$ and the code's `D` is consistent with the single ambient space chosen.

### [E-003] Volume-growth formula is 2-dimensional but the chapter works in $D$ dimensions (was F-003)
- Location: Def. Hyperbolic Volume Growth (lines 87-98); used at lines 377, 518
- Severity: Minor
- Type: Scope restriction
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "With metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially: $\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2(r/2) \approx \pi e^r$."
- Upstream anchor: n/a (defined in this chapter).
- Why this is an error: $4\pi\sinh^2(r/2)=2\pi(\cosh r-1)$ is the area of a geodesic disk in $\mathbb{H}^2$ (curvature $-1$). Recomputed by both reviewer and verifier: $4\pi\sinh^2(0.5)=3.41$, $4\pi\sinh^2(1)=17.36$, $\pi e^{10}=6.92\times10^4$, ratio $3987$, so the numbers in the box (lines 109-113) are right for $D=2$. In $\mathbb{H}^D$ the volume is $\mathrm{Vol}(S^{D-1})\int_0^r\sinh^{D-1}(s)\,ds\sim e^{(D-1)r}$. The metric is written in general index form and the chapter otherwise uses $SO(D)$ symmetry and $D$-dimensional tensors, so the definition is stated more generally than it holds.
- Impact on downstream results: The asymptotic entropy $S(\rho)\approx\log\mathrm{Vol}\approx(D-1)\rho$ would change the entropic drift by a factor $(D-1)$ in $D>2$, which changes $r(\tau)=\tanh(\tau/2)$ (Prop. 21.1.4) and `cor-recovery-of-holographic-flow` (04_equations_motion.md:942-952) unless $D=2$ is fixed.
- Fix guidance:
  1. Either state "$D=2$ (Poincaré disk)" as a standing hypothesis for §21.1-21.2,
  2. or give the general $D$ formula and carry $(D-1)$ through $S$, $F_{\text{entropy}}$ and Prop. 21.1.4.
- Required new assumptions/permits: standing hypothesis $D=2$ if option 1 is chosen.
- Validation plan: recompute $\nabla_G S$ and $r(\tau)$ with the chosen $D$ and confirm they match Prop. 21.1.4 and 04:951.

### [E-004] Depth-to-radius map differs by a factor 2 from the hyperbolic-hierarchy section (was F-005)
- Location: Prop. Riemannian Gradient of $U$, Remark (line 208)
- Severity: Note
- Type: Notation conflict
- Criterion: Framework
- Origin: upstream 03_architecture/01_compute_tiers.md:1852
- Claim (verbatim): line 208: "The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$."
- Upstream anchor: 03_architecture/01_compute_tiers.md:1852, `cor-the-hyperbolic-embedding`: "in the Poincare ball model, depth maps to $\tanh^{-1}(r)$ where $r \in [0,1)$ is the radial coordinate" (metric $ds^2=(dx^2+dy^2)/y^2$, curvature $-1$).
- Why this is an error: With the metric $G=4\delta_{ij}/(1-|z|^2)^2$ (curvature $-1$) used throughout this chapter, $d(0,z)=2\operatorname{artanh}|z|$, so this chapter is correct. The upstream corollary quotes $\operatorname{artanh}(r)$ for the same curvature-$(-1)$ model, which is the distance for $G=\delta_{ij}/(1-|z|^2)^2$ (curvature $-4$). The verifier placed the origin of the discrepancy upstream. Upstream only claims that depth "correlates with" distance, so nothing breaks; the two sections nevertheless quote different normalisations for the same map.
- Impact on downstream results: None beyond reader confusion.
- Fix guidance:
  1. Add "(up to the normalisation constant; Corollary `cor-the-hyperbolic-embedding` uses $\operatorname{artanh}(r)$)" here, or
  2. align the upstream corollary with $2\operatorname{artanh}(r)$.
- Required new assumptions/permits: none.
- Validation plan: one-line consistency check between the two files.

### [E-005] $F_{\text{entropy}}(0)=0$ is false; the chapter contradicts itself about the origin (was F-004)
- Location: Prop. SO(D) Symmetry at Origin, item 2 (line 240); box "Why Does the Entropic Force Vanish" (lines 247-255); Def. Control Field at Origin (lines 280-291); versus Def. The Entropic Force (line 122) and Thm. Angular Symmetry Breaking (line 337)
- Severity: Major
- Type: Computational error (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 240: "The entropic force vanishes: $F_{\text{entropy}}(0) = 0$"; line 250: "At $z = 0$, this gives exactly zero (the unit vector $\hat{z}$ is undefined, but the prefactor vanishes anyway)"; line 289: "Since $F_{\text{entropy}}(0) = 0$ (isotropic), the initial trajectory is determined **entirely** by $u_\pi(0)$"; line 337: "with drift $\frac{1-r^2}{2} > 0$ for all $r \in [0,1)$. The origin is not a fixed point; the drift pushes trajectories outward".
- Upstream anchor: 04_equations_motion.md:477: "$\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}$"; 04_equations_motion.md:951: "the radial coordinate ... satisfies $\dot{r} = \frac{1-r^2}{2}$". (The upstream display at 04:949 writes "$\dot{z} = \frac{(1-|z|^2)}{2}\,z$" without the hat, inconsistent with its own proof at line 951; that is an upstream typo and not what this chapter uses.)
- Why this is an error: With the chapter's own formula $F_{\text{entropy}}(z)=\frac{1-|z|^2}{2}\,\frac{z}{|z|}$ (lines 122, 199, 205), the prefactor at $z=0$ is $\frac{1-0}{2}=\tfrac12\neq0$. The magnitude $|F_{\text{entropy}}(z)|\to\tfrac12$ as $z\to0$ and the direction is undefined, because $U(z)=-2\operatorname{artanh}|z|$ has a cone point at the origin ($|\nabla U|=2/(1-|z|^2)\to2$, not $0$). So item 2 of Prop. 21.2.1, the box at 247-255 ("the prefactor vanishes anyway") and the premise of Def. 21.2.3 are all wrong as stated. The chapter itself says the opposite at line 337 ("origin is not a fixed point", radial drift $\tfrac12$ at $r=0$) and in the Prop. 21.1.4 table (line 165: "the particle starts fast"); upstream 04:951 gives $\dot r(0)=\tfrac12$. The conclusion "initial trajectory is determined entirely by $u_\pi(0)$" can be rescued only as a statement about the direction (the entropic field selects no direction at $0$), not the magnitude, and "$F_{\text{total}}=F_{\text{entropy}}+u_\pi(0)$" at line 286 is ill-defined at $z=0$.
- Impact on downstream results: The whole "policy as symmetry-breaking kick" narrative (lines 289-297, 433-488, 494-522, 764), `intro_agent.md` (lines 38, 68 "Policy as symmetry-breaking kick (pitchfork bifurcation)", 370) and 08_multiagent/01_gauge_theory.md:1321 rely on Prop. 21.2.1.
- Fix guidance:
  1. Replace item 2 by "$F_{\text{entropy}}$ is radial with $|F_{\text{entropy}}(z)|\to\tfrac12$ as $z\to0$; its direction $\hat z$ is undefined at $z=0$, so the entropic field selects no direction there."
  2. Restate Def. 21.2.3: for $z(0)=0$ the direction $\lim_{\tau\downarrow0}\hat z(\tau)$ is fixed by $u_\pi(0)$ (or by noise when $u_\pi(0)=0$), while the radial speed is $\tfrac12+u_\pi^r$.
  3. Delete "the prefactor vanishes anyway" at line 250.
  4. Alternatively, smooth $U$ near $0$ (e.g. $U=-2\operatorname{artanh}(|z|)\,\chi(|z|)$, or use $-\log(1-|z|^2)$) so that $F(0)=0$ holds and $0$ is an unstable fixed point; then Prop. 21.1.4 and line 337 must change accordingly.
- Required new assumptions/permits: none for options 1-3; a modified potential for option 4.
- Validation plan: evaluate $|F_{\text{entropy}}|$ at $|z|=10^{-3}$ from the chapter's formula (should be $\approx0.5$) and confirm that Prop. 21.2.1, Def. 21.2.3, Prop. 21.1.4 and line 337 all agree on the behaviour at the origin.

### [E-006] Higgs-potential sign convention contradicts the cited theorem, and the latent $SO(D)$ is not the gauge group it breaks (was F-006)
- Location: Prop. SO(D) Symmetry at Origin, cross-reference (line 243); Physics Isomorphism box (line 402)
- Severity: Minor
- Type: Notation conflict (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 243: "In multi-agent settings, this symmetry is spontaneously broken via the Higgs mechanism (Theorem `thm-higgs-mechanism`), yielding massive gauge bosons"; line 402: "the Mexican hat potential $V(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$: for $\mu^2 > 0$, the $U(1)$-symmetric origin becomes unstable".
- Upstream anchor: 08_multiagent/01_gauge_theory.md:2071, 2091-2094: $V(\Phi)=\mu^2|\Phi|^2+\lambda|\Phi|^4$; "When the Higgs mass parameter satisfies $\mu^2 < 0$, the potential $V(\Phi)$ has a non-trivial minimum, and the gauge symmetry is **spontaneously broken**. ... $v = \sqrt{-\mu^2/\lambda}$"; 08_multiagent/01_gauge_theory.md:1376: "At origin ($K = 0$, Semantic Vacuum): $G_0 = SO(D)$, $H_0 = \{e\}$".
- Why this is an error: The same symbol $\mu^2$ carries opposite sign conventions in the two places ($V=-\mu^2|\phi|^2+\dots$ with $\mu^2>0$ here; $\mu^2<0$ with $v=\sqrt{-\mu^2/\lambda}$ upstream). Each is self-consistent, but a reader following the cross-reference meets a contradictory criterion. Moreover, `thm-higgs-mechanism` breaks the gauge group of the multi-agent theory (giving $m_A=gv/2$), whereas Prop. 21.2.1 concerns the rotation group of the nuisance space at the origin; line 243 identifies the two without argument (08:1321 makes the same cross-link in the reverse direction, so the overstatement is mutual). The stabiliser claim $H_0=\{e\}$ is inherited from 08:1376 and is not charged to this chapter.
- Impact on downstream results: None mathematical; misleading cross-link.
- Fix guidance:
  1. Use the upstream convention ($V=\mu^2|\phi|^2+\lambda|\phi|^4$, $\mu^2<0$) or add "(sign convention differs from `thm-higgs-mechanism`)".
  2. Reword line 243 to "is analogous to" rather than "is the special case ... broken via".
- Required new assumptions/permits: none.
- Validation plan: cross-read line 402 against 08:2071-2094 after the edit.

### [E-007] Control field $u_\pi=G^{-1}\mathbb{E}_{a\sim\pi}[a]$ has the wrong type: $a$ lives in the action space, not in $T^*_z\mathcal{Z}$ (was F-007)
- Location: Def. The Control Field (lines 257-268); summary table (line 777)
- Severity: Minor
- Type: Dimensional mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 263: "The Policy $\pi_\theta(a|z)$ outputs a **control field** $u_\pi(z)$ on the tangent bundle $T\mathbb{D}$: $u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]$ ... Units: $[u_\pi] = [z]/\tau$."
- Upstream anchor: 01_foundations/01_definitions.md:240: "actions decompose into structured components: $a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}})$ where $K^{\text{act}}_t$ is the discrete motor macro"; 01_definitions.md:117-125: "$a_t\in\mathcal{A}$ is action (control signal sent outward)"; 04_equations_motion.md:222-236: "$u_{\pi,k}$ is the **control field** from the policy (Definition `def-the-control-field`)" entering $dp_k$ as a force with "$[p]=\text{length}/\tau$".
- Why this is an error: $G^{-1}(z)$ maps $T^*_z\mathcal{Z}\to T_z\mathcal{Z}$; applying it to $\mathbb{E}[a]$ requires $\mathbb{E}[a]$ to be a covector at $z$. But $a\in\mathcal{A}$ has a discrete component $K^{\text{act}}$ (no expectation is defined) and continuous motor components living in $\mathbb{R}^{d_{\text{motor}}}$, not in $T^*_z\mathcal{Z}$; no map $\mathcal{A}\to T^*\mathcal{Z}$ exists in the framework. In addition, the upstream Langevin equation uses $u_{\pi,k}$ (index down) as a covector force in the momentum equation, while this definition makes $u_\pi$ a velocity vector (index up, $G^{-1}$ already applied); the same symbol has different index placement (see also E-013).
- Impact on downstream results: Every formula that substitutes $u_\pi$ (Thm. 21.2.4 table, Thm. 21.2.5, code, `conn-rl-24`).
- Fix guidance:
  1. Define an explicit action-to-covector map $\iota_z:\mathcal{A}\to T^*_z\mathcal{Z}$ (e.g. the decoder Jacobian pull-back, or restrict to the "latent action = target displacement" case $a\in T^*_z\mathcal{Z}$) and set $u_\pi:=G^{-1}\iota_z(\mathbb{E}_\pi[a])$.
  2. State once whether $u_\pi$ is a force (covector, as in 04) or a velocity (vector, as here) and use $G^{-1}u_\pi$ consistently in the overdamped equations.
- Required new assumptions/permits: the map $\iota_z$ (a new definition).
- Validation plan: type-check every occurrence of $u_\pi$ in this chapter and in 04_equations_motion.md against the declared index placement.

### [E-008] Context-conditioned control field has the wrong sign / double-counts the effective potential (was F-008)
- Location: Def. The Control Field, Remark (Context-Conditioning) (line 270)
- Severity: Moderate
- Type: Definition mismatch (secondary: Computational error, sign)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 270: "The control field becomes $u_\pi(z,c) = G^{-1}(z) \cdot \nabla_z \Phi_{\text{eff}}(z,K,c)$ where the `def-effective-potential` depends on task context."
- Upstream anchor: 06_fields/01_boundary_interface.md:692: "$v_c(z) = -G^{-1}(z) \nabla_z \Phi_{\text{eff}}(z, K, c) + u_\pi(z, c)$ is the context-conditioned velocity"; 07_cognition/01_supervised_topo.md:52, 64: "The effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ tells us how 'expensive' it is to be at position $z$"; 04_equations_motion.md:222-224: "$dp_k = [-\partial_k \Phi_{\text{eff}} - \gamma p_k + \dots + u_{\pi,k}]ds$".
- Why this is an error: Upstream, the potential force is $-G^{-1}\nabla\Phi_{\text{eff}}$ and $u_\pi$ is an additional, independent term. Setting $u_\pi=+G^{-1}\nabla\Phi_{\text{eff}}$ (i) points uphill on $\Phi_{\text{eff}}=-\log p(y|z)$, i.e. away from the requested class or prompt, and (ii) substituted into 06_fields:692 gives $v_c\equiv0$: the control cancels the potential flow and generation stops. Either the sign is wrong (should be $-G^{-1}\nabla\Phi_{\text{eff}}$ with the potential term removed) or $u_\pi$ should be defined as a separate term as upstream does.
- Impact on downstream results: 06_fields/01_boundary_interface.md:692 and the classification-as-context section of 07_cognition/01 cite this definition.
- Fix guidance:
  1. Replace with "$u_\pi(z,c) = -G^{-1}(z)\nabla_z\Phi_c(z)$ where $\Phi_c$ is the context-dependent part of the potential (e.g. $-\log p(c\mid z)$)".
  2. Say explicitly that in the coupled velocity $v_c$ of 06_fields it replaces, rather than adds to, the corresponding potential term.
- Required new assumptions/permits: none.
- Validation plan: substitute the corrected $u_\pi$ into 06_fields:692 and check that $v_c$ is non-zero and descends $\Phi_{\text{eff}}$.

### [E-009] "Three equivalent interpretations" are three different cases; the proof invokes the policy-gradient theorem for a quantity it does not produce (was F-010)
- Location: Thm. Unified Control Interpretation (lines 299-312)
- Severity: Minor
- Type: Miswording (secondary: Invalid inference / External dependency)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The control field $u_\pi$ admits three equivalent interpretations ... *Proof.* In all cases, $u_\pi$ is a tangent vector at $z$. The RL case follows from the policy gradient theorem {cite}`sutton1999policy`; the generation case follows from treating the prompt as a target direction; the unconditional case reduces to pure Langevin dynamics".
- Upstream anchor: n/a (external theorem). Sutton et al. (1999) give $\nabla_\theta J(\theta)=\mathbb{E}[\nabla_\theta\log\pi_\theta(a|s)\,Q^\pi(s,a)]$, a gradient with respect to policy parameters.
- Why this is an error: The three rows assign different values to $u_\pi$ ($G^{-1}\nabla V$, $G^{-1}\text{embed}$, $0$); they are mutually exclusive settings, not "equivalent interpretations" of one object, so nothing is being proved. The policy gradient theorem concerns $\nabla_\theta J$ and says nothing about $\nabla_z V$ being the expected action $\mathbb{E}_\pi[a]$, which is what would be needed to derive the RL row from Def. 21.2.2. The verifier reclassified the criterion as Framework: the defect is that a `prf:theorem` inside the framework has no derivable content; the external theorem is merely invoked in vain.
- Impact on downstream results: None beyond the status of the statement (it should be a definition or table, not a theorem).
- Fix guidance:
  1. Demote to a definition ("Three modes of the control field"), or
  2. if a theorem is wanted, state the hypothesis that makes the rows coincide with Def. 21.2.2 (e.g. a Boltzmann policy $\pi(a|z)\propto\exp(-\langle a,\nabla V\rangle/T_c)$ with $a\in T^*_z\mathcal{Z}$ gives $\mathbb{E}[a]\propto-\nabla V$) and prove that.
- Required new assumptions/permits: the Boltzmann-policy hypothesis for option 2.
- Validation plan: confirm that the proof text derives each row from Def. 21.2.2 under the stated hypothesis.

### [E-010] RL control field $u_\pi=+G^{-1}\nabla_z V_{\text{critic}}$ climbs a cost-to-go (was F-009)
- Location: Thm. Unified Control Interpretation, table row "RL" (line 306); Algorithm 21.2.6 (lines 465-469); `conn-rl-24` (line 504); box at line 317
- Severity: Moderate (verifier adjusted from Major)
- Type: Parameter inconsistency (sign) (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 306: "**RL** | $u_\pi = G^{-1} \nabla_z V_{\text{critic}}$ | Points toward high-value regions"; lines 466-469: "# RL: u_pi = G^{-1} * grad V (points toward high value) ... u_pi = G_inv * grad_V"; line 504: "$u_\pi = G^{-1} \nabla_z V$ is the policy control field".
- Upstream anchor: 01_foundations/02_control_loop.md:141: "**Value Function:** assigns a scalar cost-to-go/value to points in $Z$, representing risk/undesirability (identify $V=\Phi$ for the exact component)"; 02_control_loop.md:583: "Minimize expected cost-to-go $V(z)$"; 02_control_loop.md:1063: "The critic $V$ is a value/cost-to-go function that should decrease along controlled trajectories"; 04_equations_motion.md:459: "**Pure Control** | $\alpha = 0$ | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)"; 04_equations_motion.md:423: "$\Phi_{\text{eff}}(z, K) = \alpha U(z) + (1 - \alpha) V_{\text{critic}}(z, K) + \dots$".
- Why this is an error: In the book's convention $V$ is a cost-to-go ("risk/undesirability", "should decrease along controlled trajectories") and the controlled flow is $-\nabla_G V$. The chapter's RL mode pushes along $+G^{-1}\nabla V$, i.e. toward higher cost-to-go. Moreover, $V_{\text{critic}}$ already enters the equations of motion through $\Phi_{\text{eff}}$ with the force $-(1-\alpha)G^{-1}\nabla V$; adding $u_\pi=+G^{-1}\nabla V$ gives a net $+\alpha G^{-1}\nabla V$, which vanishes at $\alpha=0$ (pure control), so the value then has no effect at all (double-counting confirmed by the verifier using the chapter's overdamped SDE at line 359 generalised to $\Phi_{\text{eff}}$). The verifier downgraded to Moderate because 04:406 describes $V$ as "how good a state is in terms of expected future reward" alongside "roll down this potential", so the upstream prose is itself ambiguous, and the fix is a single sign (or a one-line declaration $V_{\text{reward}}=-V$) plus one sentence on how $u_\pi$ relates to the $V$ term already in $\Phi_{\text{eff}}$.
- Impact on downstream results: Algorithm 21.2.6, `conn-rl-24`, the RL half of the "unification" claim, and any implementation copying `u_pi = G_inv * grad_V`.
- Fix guidance:
  1. Write $u_\pi=-G^{-1}\nabla_z V_{\text{critic}}$ ("descends the cost-to-go") and change the code to `u_pi = -G_inv * grad_V`; or
  2. if a reward-valued $V$ is intended, say so explicitly ($V_{\text{reward}}=-V$).
  3. In either case reconcile with the $-(1-\alpha)\nabla V$ term of `def-effective-potential` so the value is not counted twice with opposite signs.
- Required new assumptions/permits: none.
- Validation plan: write the total $V$-dependent force for general $\alpha$ after the fix and check it is $-G^{-1}\nabla V$ times a non-negative coefficient for all $\alpha\in[0,1]$.

### [E-011] Radial SDE in Thm. 21.2.5 omits the Itô (Bessel) drift and the radial policy component; the "traceless Hessian" justification is reversed (was F-011)
- Location: Thm. Angular Symmetry Breaking, radial dynamics (lines 331-337) and proof (lines 356-366)
- Severity: Major
- Type: Computational error (secondary: Proof gap / omission)
- Criterion: Framework (verifier corrected from External)
- Origin: this chapter
- Claim (verbatim): line 334: "$dr = \frac{1-r^2}{2}\,d\tau + \frac{1-r^2}{2}\sqrt{2T_c}\,dW_r$"; line 366: "The Itô corrections vanish for the radial component (since $\partial^2 r/\partial z^i\partial z^j$ is traceless) and contribute a drift correction for angular that cancels with geometric terms."
- Upstream anchor: 04_equations_motion.md:920-924 (`thm-overdamped-limit`): "$dz^k = [\mathcal{M}_\gamma(z)]^{k}{}_{j}(-G^{j\ell}(z)\partial_\ell\Phi_{\text{gen}}(z)) ds + \sqrt{2T_c}(G^{-1/2}(z))^{kj}dW^j_s$"; the chapter's own Cartesian SDE (line 359) with $G^{-1/2}=\frac{1-r^2}{2}I$ and $\Sigma=2T_cG^{-1}=\frac{T_c(1-r^2)^2}{2}I$.
- Why this is an error: The verifier redid the Itô computation symbolically (sympy 1.14) from the chapter's own Cartesian SDE $dz=[\tfrac{1-r^2}{2}\hat z+u]\,d\tau+\tfrac{1-r^2}{2}\sqrt{2T_c}\,dW$, with $u=u_r\hat r+u_\theta\hat\theta$, obtaining
  $$dr=\Big[\frac{1-r^2}{2}+u_\pi^{r}+\frac{T_c(1-r^2)^2}{4r}\Big]d\tau+\frac{1-r^2}{2}\sqrt{2T_c}\,dW_r,\qquad d\theta=\frac{u_\pi^\theta}{r}\,d\tau+\frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta,$$
  with $\operatorname{tr}\operatorname{Hess}r=1/r$ and $\operatorname{tr}\operatorname{Hess}\theta=0$ (the reviewer's finite-difference check at $(0.3,0.4)$ agrees: $2.000=1/r$ and $-9\times10^{-8}$). So line 366 is exactly backwards: the Itô correction vanishes for $\theta$ and is non-zero for $r$. The stated radial SDE omits both the Bessel drift $\frac{T_c(1-r^2)^2}{4r}$ (numerically $0.28125$ at $r=0.5$, $T_c=1$) and the radial control component $u_\pi^r$; nothing in the theorem sets the latter to zero (in RL mode $u_\pi=\pm G^{-1}\nabla V$ generally has one). In $D$ dimensions the Bessel term carries $(D-1)$. Consequently "drift $\frac{1-r^2}{2}>0$ for all $r$" and "monotonic expansion" are unproven: with $u_\pi^r<-\tfrac12$ near the origin the drift is inward. The angular SDE (line 342) and the radial noise coefficient are correct. The error is a computation inside the framework's own SDE, hence Framework.
- Impact on downstream results: The radial part of Thm. 21.2.5, its use in `conn-rl-24`, in 04_equations_motion.md:1310 (Node 25 "boundary reached") and intro_agent.md:370, and any reasoning about time-to-boundary with $T_c>0$.
- Fix guidance:
  1. Replace the radial SDE at line 334 by the expression above.
  2. Replace line 366 by "the Itô correction vanishes for $\theta$ ($\operatorname{tr}\operatorname{Hess}\theta=0$) and equals $\frac{T_c(1-r^2)^2}{4r}$ for $r$".
  3. Either add the hypothesis $u_\pi^r\ge0$ (or $u_\pi\perp\hat r$) or keep $u_\pi^r$ in the statement and weaken "monotonic expansion" accordingly.
  4. Restrict to $D=2$ or carry $(D-1)$.
- Required new assumptions/permits: a hypothesis on $u_\pi^r$ if monotonic expansion is to be kept.
- Validation plan: re-run the Itô computation (sympy or finite differences) on the corrected statement; simulate the Cartesian SDE and compare the empirical radial drift with the stated one at several $r$.

### [E-012] The symmetry-breaking ratio $\eta(r)$ is not dimensionless, is not the drift/diffusion ratio it claims to be, and the freeze-out conclusion is reversed under the chapter's own definition of $u_\pi$ (was F-012)
- Location: Thm. Angular Symmetry Breaking, Phase Transition (lines 347-354), proof paragraphs "Critical temperature" and "Direction freeze-out" (lines 368-370); prose (lines 383-385)
- Severity: Major
- Type: Invalid inference (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 350: "$\eta(r) := \frac{|u_\pi^\theta|^2}{T_c} \cdot \frac{2r^2}{(1-r^2)^2}$"; line 368: "The symmetry-breaking ratio $\eta(r)$ compares the squared angular drift to the angular diffusion coefficient"; line 370: "As $r$ increases toward the boundary, $\eta(r) \to \infty$ (the denominator $(1-r^2)^2 \to 0$), causing the angular distribution to concentrate."
- Upstream anchor: this chapter, Def. The Control Field (line 263): "$u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]$" with $G^{-1}=\frac{(1-|z|^2)^2}{4}I$; line 277: "Near the boundary, $G^{-1}$ goes to zero, which means the policy's influence weakens"; units: line 268 "$[u_\pi]=[z]/\tau$", 04_equations_motion.md:515 "$T_c$ ... Units: nat".
- Why this is an error: (a) From the (correct) angular SDE at line 342 the drift is $a=u_\pi^\theta/r$ and $\sigma_\theta^2=\frac{T_c(1-r^2)^2}{2r^2}$. Then $a^2/\sigma_\theta^2=\frac{2|u_\pi^\theta|^2}{T_c(1-r^2)^2}$, and the chapter's $\eta=r^2\cdot a^2/\sigma_\theta^2$ (numerically $0.889$ vs $3.556$ at $r=0.5$, $u=T_c=1$); the stray factor $r^2$ means $\eta$ is not the ratio described at line 368. With $[u_\pi]=[z]/\tau$ and $[T_c]=\mathrm{nat}$, $\eta$ has units $\tau^{-2}$, so "dimensionless ratio" is false. The genuinely dimensionless drift-to-diffusion number for a diffusion on $S^1$ is the Péclet / von Mises concentration $\kappa=a/\sigma_\theta^2=\frac{2r\,u_\pi^\theta}{T_c(1-r^2)^2}$ (up to a factor 2), which is linear, not quadratic, in $u_\pi^\theta$. (b) Line 370 treats $u_\pi^\theta$ as independent of $r$, but Def. 21.2.2 makes $u_\pi=G^{-1}\mathbb E[a]=\frac{(1-r^2)^2}{4}\mathbb E[a]$ (line 277 says so in words). Substituting, the chapter's $\eta\propto r^2(1-r^2)^2\to0$ (numerically $0.0176$, $0.00366$, $4.9\times10^{-5}$ at $r=0.5,0.9,0.99$ with $|\mathbb E a|=T_c=1$), and the Péclet number becomes $\frac{r\,\bar a^\theta}{2T_c}\to\frac{\bar a^\theta}{2T_c}$: finite and monotone, with no divergence. Either way, "$\eta\to\infty$", "the noise weakens relative to the policy" (line 383) and the "Direction freeze-out" step are not derivable from the chapter's definitions. What is true is that both angular drift and angular noise become integrable along $r(\tau)=\tanh(\tau/2)$ (both $\propto\operatorname{sech}^4(\tau/2)$ up to powers), so the direction converges, but not because the policy dominates.
- Impact on downstream results: The phase-transition / critical-temperature story (lines 385-396, 417, 764), `intro_agent.md:68`, and the "Temperature and Generation Quality" box.
- Fix guidance:
  1. Define the comparison from the SDE actually derived, using the dimensionless Péclet number $\kappa(r)=\frac{2r\,u_\pi^\theta}{T_c(1-r^2)^2}$ (verifier's refinement of the reviewer's $a^2/\sigma_\theta^2$).
  2. Substitute $u_\pi^\theta=\frac{(1-r^2)^2}{4}\bar a^\theta$ to get $\kappa=\frac{r\,\bar a^\theta}{2T_c}$ and state its finite limit.
  3. Replace "freeze-out because $\eta\to\infty$" by an argument based on finite total angular variance $\int_0^\infty\sigma_\theta^2(r(\tau))\,d\tau<\infty$ and finite total angular drift along the radial solution.
- Required new assumptions/permits: none.
- Validation plan: numerically integrate $\sigma_\theta^2(r(\tau))$ and $a(r(\tau))$ along $r(\tau)=\tanh(\tau/2)$ to confirm finiteness; simulate the angular SDE and check that the terminal angular spread depends on $T_c$ smoothly and without divergence of any policy-to-noise ratio.

### [E-013] Missing hypotheses ($\alpha=1$, $\gamma_{\text{risk}}=0$, $\mathcal F=0$) and $u_\pi$ enters the overdamped equation with the wrong index type (was F-013)
- Location: Thm. Angular Symmetry Breaking, proof (lines 356-362)
- Severity: Minor
- Type: Proof gap / omission (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Starting from the second-order geodesic Langevin equation (Definition `def-bulk-drift-continuous-flow`) ... we take the overdamped limit (Theorem `thm-overdamped-limit`). The overdamped position SDE in Cartesian coordinates is: $dz^k = -G^{kj}\partial_j U\, d\tau + u_\pi^k\, d\tau + \sqrt{2T_c}(G^{-1/2})^{kj}\,dW^j_\tau$".
- Upstream anchor: 04_equations_motion.md:222-224: "$dp_k = [-\partial_k \Phi_{\text{eff}} - \gamma p_k + \beta_{\text{curl}}\mathcal{F}_{kj}G^{j\ell}p_\ell - \Gamma^m_{k\ell}G^{\ell j}p_jp_m + u_{\pi,k}]ds + \dots$" (control is a covector force $u_{\pi,k}$); 04:423: "$\Phi_{\text{eff}} = \alpha U + (1-\alpha)V_{\text{critic}} + \gamma_{risk}\Psi_{\text{risk}}$"; 04:920-924 (`thm-overdamped-limit`): drift "$[\mathcal{M}_\gamma]^{k}{}_{j}(-G^{j\ell}\partial_\ell\Phi_{\text{gen}})$" with no $u_\pi$ term; 04:942 (`cor-recovery-of-holographic-flow`) lists the hypotheses for the control-free case.
- Why this is an error: The cited overdamped theorem (i) has drift $-\mathcal M_\gamma G^{-1}\nabla\Phi_{\text{gen}}$, not $-G^{-1}\nabla U$: obtaining $U$ alone needs $\alpha=1$, $\gamma_{\text{risk}}=0$ and $\mathcal F=0$ ($\mathcal M_\gamma=I$), none of which is stated; (ii) contains no control term at all, and in the second-order equation the control is the covector $u_{\pi,k}$, whose overdamped image is $G^{kj}u_{\pi,j}$ (with friction absorbed into the time unit), not $u_\pi^k$. The chapter's own Def. 21.2.2 makes $u_\pi$ a vector with $G^{-1}$ already applied, so the same symbol denotes a covector upstream and a vector here.
- Impact on downstream results: Any attempt to reproduce Thm. 21.2.5 from Chapter 22.
- Fix guidance:
  1. Add the hypotheses "$\alpha=1$, $\gamma_{\text{risk}}=0$, $\mathcal F=0$ (conservative), $\gamma$ absorbed into $\tau$".
  2. State "where $u_\pi^k=G^{kj}u_{\pi,j}$ is the control velocity (Def. `def-the-control-field`)".
- Required new assumptions/permits: the three listed hypotheses, stated explicitly.
- Validation plan: derive line 359 from 04:922 under the stated hypotheses and confirm the control term appears with the correct index placement.

### [E-014] "Phase transition" / "pitchfork bifurcation" language is not supported by the dynamics derived (was F-014)
- Location: Thm. Angular Symmetry Breaking, "Phase Transition" (lines 347-354, 368); prose (line 385); Physics Isomorphism box (line 422); summary prose (line 764)
- Severity: Moderate
- Type: Miswording (secondary: Proof gap / omission)
- Criterion: Framework (verifier corrected from External)
- Origin: this chapter
- Claim (verbatim): line 347: "The $SO(D)$ rotational symmetry undergoes spontaneous breaking controlled by the dimensionless ratio"; line 368: "At characteristic radius $r_*$, setting $\eta(r_*) = 1$ defines the critical temperature"; line 764: "The policy breaks this symmetry via a phase transition (pitchfork bifurcation). Below a critical temperature, the agent commits to a direction and flows deterministically. Above it, the agent dithers randomly."
- Upstream anchor: n/a; the only formal content is Thm. 21.2.5 in this chapter. `intro_agent.md:337` uses "pitchfork" for chart fission (Section 30), a different mechanism, so nothing upstream supplies it.
- Why this is an error: The angular equation $d\theta=\frac{u_\pi^\theta}{r}d\tau+\sigma_\theta(r)\,dW_\theta$ is a constant-drift diffusion on $S^1$ with $a,\sigma_\theta$ depending on $r$ only; at fixed $r$ its stationary law is a von Mises-type density with concentration $\propto a/\sigma_\theta^2$ that is analytic in $T_c$. There is no bifurcation of a deterministic fixed point (no cubic nonlinearity, no parameter at which an equilibrium changes stability) and no non-analyticity in $T_c$; a finite-dimensional SDE has no phase transition. "$\eta(r_*)=1$" is a crossover convention, not a critical point, and $r_*$ and $T_c^*$ are never defined in the theorem statement. Calling this a "pitchfork bifurcation" introduces a claim nothing in the chapter derives. The complaint concerns what the framework's own SDE does or does not exhibit, hence Framework.
- Impact on downstream results: "Temperature and Generation Quality" box (lines 388-396), `intro_agent.md:68`.
- Fix guidance:
  1. Replace "phase transition / spontaneous breaking / pitchfork bifurcation / critical temperature" with "crossover between noise-dominated and policy-dominated angular dynamics at the scale where the Péclet number $\kappa(r)$ (E-012) is of order one".
  2. If a genuine bifurcation is intended, add a potential with a symmetric unstable point (e.g. a Mexican-hat angular potential) and prove the bifurcation.
- Required new assumptions/permits: an angular potential, if option 2 is chosen.
- Validation plan: check that every remaining use of "critical temperature" or "bifurcation" in the chapter points at a defined object and a proved statement.

### [E-015] Physics Isomorphism box restates the theorem with different, mutually inconsistent formulas (was F-015)
- Location: Physics Isomorphism: Spontaneous Symmetry Breaking (lines 398-423)
- Severity: Moderate
- Type: Computational error (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 407: "$d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{\sqrt{T_c(1-r^2)}}{r}\,dW_\theta$"; line 410: "$\eta(r) = |u_\pi^\theta|^2 r^2 / [T_c(1-r^2)]$ exceeds unity"; line 417: "Critical temperature $T_c^*$ | $T_c^* \approx |u_\pi^\theta|^2 r_*^2$".
- Upstream anchor: this chapter, Thm. 21.2.5, line 342: "$d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta$"; line 350: "$\eta(r) := \frac{|u_\pi^\theta|^2}{T_c} \cdot \frac{2r^2}{(1-r^2)^2}$".
- Why this is an error: The box's noise coefficient $\sqrt{T_c(1-r^2)}/r$ differs from the theorem's $\frac{(1-r^2)\sqrt{2T_c}}{2r}=\sqrt{T_c/2}\,\frac{1-r^2}{r}$ (at $r=0.5$, $T_c=1$: $1.732$ vs $1.061$; different power of $(1-r^2)$). The box's $\eta=\frac{|u|^2r^2}{T_c(1-r^2)}$ differs from the theorem's $\frac{2|u|^2r^2}{T_c(1-r^2)^2}$ (same point: $0.333$ vs $0.889$). Even with the box's own $\eta$, $\eta(r_*)=1$ gives $T_c^*=|u|^2r_*^2/(1-r_*^2)$, not $|u|^2r_*^2$. Three formulas for the same objects, none agreeing.
- Impact on downstream results: Readers of the box get a different "critical temperature" from readers of the theorem.
- Fix guidance:
  1. Make the box quote the theorem's SDE and dimensionless ratio verbatim (after fixing E-012).
  2. Derive $T_c^*$ (or the crossover scale, per E-014) from that single definition.
- Required new assumptions/permits: none.
- Validation plan: evaluate both versions at $r=0.5$, $T_c=1$, $u=1$ and confirm they coincide.

### [E-016] Stray closing fence after Algorithm 21.2.6 (was F-016)
- Location: line 492
- Severity: Note
- Type: Typo
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 492: ":::"
- Upstream anchor: n/a.
- Why this is an error: The admonition opened with `::::` at line 399 is closed at line 423; lines 425-431 (`:::{div}`) pair; lines 433-490 are bare text and a fenced code block; the `:::` at line 492 has no opener. MyST will render it as a literal ":::" paragraph or swallow following content depending on the parser.
- Impact on downstream results: Rendering only.
- Fix guidance:
  1. Delete line 492, or
  2. wrap lines 433-490 in an admonition that this fence closes.
- Required new assumptions/permits: none.
- Validation plan: build the chapter and confirm no literal ":::" appears.

### [E-017] Broken and mislabelled pointers in Connection to RL #24 (was F-017)
- Location: `conn-rl-24` (lines 494-522)
- Severity: Minor
- Type: Citation / reference error (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 518: "natural hierarchy (Theorem **Thm: Hyperbolic Volume Growth**)"; lines 509-515: "**The Special Case (Standard RL):** $dz_t = s_\theta(z_t, t)\, dt + \sigma(t)\, dW_t, \quad z_T \sim \mathcal{N}(0, I)$ This recovers **Diffusion Models** {cite}`ho2020ddpm` and **Diffusion Policies**".
- Upstream anchor: this chapter, lines 87-88: ":::{prf:definition} Hyperbolic Volume Growth :label: def-hyperbolic-volume-growth" (a definition, not a theorem; no `{prf:ref}` is used).
- Why this is an error: "Theorem **Thm: Hyperbolic Volume Growth**" is an unresolved placeholder pointing at a definition. The displayed SDE is the reverse-time diffusion-model sampler, which the box itself says is a diffusion model, yet it is headed "Standard RL"; the box also says the degenerate limit "reverses the direction (boundary to origin)" although in the Euclidean limit $G\to I$ there is no boundary. These are labelling and meaning errors in a formal-looking box.
- Impact on downstream results: None.
- Fix guidance:
  1. Use "Definition {prf:ref}`def-hyperbolic-volume-growth`".
  2. Retitle the special case "Standard diffusion model / diffusion policy".
  3. Describe the limit as "Euclidean, time-reversed (noise $\to$ data)".
- Required new assumptions/permits: none.
- Validation plan: build and confirm the reference resolves.

### [E-018] Motor-texture "duality" $\Sigma_{\text{motor}}=\omega\,\Sigma_{\text{visual}}\,\omega^{-1}$ is attributed to a theorem that does not exist and is not stated anywhere upstream (was F-018)
- Location: Remark (Motor Extension) (line 539)
- Severity: Moderate
- Type: Citation / reference error (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Theorem {prf:ref}`ax-motor-texture-firewall` establishes the duality: $\Sigma_{\text{motor}} = \omega \cdot \Sigma_{\text{visual}} \cdot \omega^{-1}$."
- Upstream anchor: 06_fields/01_boundary_interface.md:403-410: ":::{prf:axiom} Motor Texture Firewall :label: ax-motor-texture-firewall  Motor texture is decoupled from the Bulk dynamics: $\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.$ The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$."; 06_fields/01_boundary_interface.md:392-398 (`def-motor-texture-distribution`): "$\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I$". A grep of `docs/source/1_agent` for "omega \cdot \Sigma" / "Sigma_{motor} = \omega" finds only this line.
- Why this is an error: The referenced object is an axiom, not a theorem, and contains no conjugation formula; $\omega$ is undefined in both chapters and anywhere in Volume 1. Given the upstream definitions, $\Sigma_{\text{motor}}$ and $\Sigma_{\text{visual}}$ are scalar multiples of identities on spaces of generally different dimension (06:396 uses $I_{d_{\text{motor,tex}}}$), so $\omega\Sigma\omega^{-1}$ is ill-typed unless the dimensions agree, in which case it reduces to $\sigma_{\text{motor}}^2=\sigma_{\text{tex}}^2$.
- Impact on downstream results: None (no one uses the formula), but it is a false attribution in a formal remark.
- Fix guidance:
  1. Delete the "duality" sentence, or
  2. replace by "Axiom `ax-motor-texture-firewall` imposes the same partition condition on motor texture; Definition `def-motor-texture-distribution` uses the same conformal scaling $\sigma^2G^{-1}$."
- Required new assumptions/permits: none.
- Validation plan: grep Volume 1 for $\omega$ conjugation formulas after the edit; none should remain.

### [E-019] $\mathcal{Z}$ is redefined as $\mathcal{K}\times\mathcal{Z}_n$, conflicting with the foundational definition and with $\mathcal Z=\mathbb D$ earlier in the chapter (was F-019)
- Location: Axiom Bulk-Boundary Decoupling, item 1 (lines 544-551); Def. 21.1.1 (line 65)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**: 1. **Interior (Planning Domain):** The trajectory $z(\tau)$ evolves strictly on the manifold $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n$. It contains no texture component."
- Upstream anchor: 01_foundations/01_definitions.md:103: "$Z_t := (K_t, z_{n,t}, z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}}$".
- Why this is an error: Upstream $\mathcal{Z}$ includes the texture factor; here the same symbol denotes the texture-free bulk, and at line 65 it denotes the Poincaré disk. The partition condition is therefore stated on a different space from the one the state $Z$ lives in, and "$\dot z=f(z,u_\pi)$ (no $z_{\text{tex}}$ dependence)" is trivially true on $\mathcal K\times\mathcal Z_n$ rather than a constraint on dynamics on $\mathcal Z$.
- Impact on downstream results: `ax-bulk-boundary-decoupling` is cited from at least nine other files (06_fields/01:419-420, 04:1062, 07_cognition/04:44, 08_multiagent/02:1136, 09_economics:459, `intro_agent.md:337`, ...), so the domain on which it is stated should match the foundational one.
- Fix guidance:
  1. Write "$\mathcal{Z}_{\text{bulk}} := \mathcal{K}\times\mathcal{Z}_n\subset\mathcal Z$".
  2. State the partition condition as: the projection $\Pi_{\text{bulk}}$ of the dynamics on $\mathcal Z$ satisfies $\partial_{z_{\text{tex}}}\Pi_{\text{bulk}}\dot Z=0$.
- Required new assumptions/permits: none (new notation $\mathcal Z_{\text{bulk}}$ only).
- Validation plan: check that each citing file reads the axiom on the intended space.

### [E-020] Maximum-entropy justification for $\Sigma\propto G^{-1}$ is misstated (was F-020)
- Location: box "Why Conformal Scaling is the Right Choice" (lines 632-642, feynman-added)
- Severity: Minor
- Type: Conceptual
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 639: "a Gaussian $\mathcal{N}(0, G^{-1})$ is the maximum-entropy distribution subject to a constraint on expected hyperbolic distance from the mean. It's the 'least assuming' distribution you can pick at each point. So conformal scaling isn't just convenient---it's the unique geometrically natural choice."
- Upstream anchor: n/a (standard fact); this chapter Def. 21.3.2 (line 598): "$\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z)$", with $z_{\text{tex}}$ sampled in a separate `texture_dim` space (Algorithm 21.3.7, lines 736-737).
- Why this is an error: The Gaussian is the maximum-entropy law under a constraint on the second moment $\mathbb{E}[v^{\top}G(z)v]$ on the tangent space at $z$; a constraint on expected (first-power) distance gives an exponential/Laplace-type law, not a Gaussian. Moreover $z_{\text{tex}}$ is not a tangent vector at $z$, so "isotropic in the hyperbolic metric" does not apply to it; only the scalar conformal factor is borrowed. The uniqueness claim is therefore unsupported.
- Impact on downstream results: None formal (box is hidden in expert mode).
- Fix guidance:
  1. Write "subject to a constraint on the expected squared $G(z)$-norm" and drop "unique".
  2. Note that the texture space merely inherits the conformal scalar $(1-|z|^2)^2/4$.
- Required new assumptions/permits: none.
- Validation plan: none beyond rereading the box.

### [E-021] Prop. Epistemic Barrier has no proof and mis-describes BarrierEpi (was F-021)
- Location: Prop. Epistemic Barrier (lines 671-676)
- Severity: Moderate
- Type: Definition mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The partition condition enforces **BarrierEpi** (Epistemic Limit): The agent does not waste capacity predicting the noise---it only predicts the *statistics* of the noise ($\Sigma$)."
- Upstream anchor: 02_sieve/02_limits_barriers.md:63: "| **BarrierEpi** | Epistemic | **VQ-VAE/WM** | **Information Overload** | Environment ... complexity exceeds $\log\lvert\mathcal{K}\rvert$ and/or WM class; closure breaks. | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{Sync}_{K-W}}$ (Distortion + Closure) |".
- Why this is an error: Upstream BarrierEpi is the information-overload failure mode (environment complexity exceeding the macro alphabet / world-model class), penalised by reconstruction plus closure losses. The proposition attributes to it a different content ("not predicting noise, only its statistics") and asserts that the partition condition "enforces" it, with no argument. The partition condition (texture does not enter the dynamics) neither bounds environment complexity nor controls $\mathcal L_{\text{recon}}+\mathcal L_{\text{Sync}}$; if anything it is relevant to the texture firewall. A `prf:proposition` with no proof and a mismatched referent is a formal error.
- Impact on downstream results: The Sieve mapping in 02_sieve.
- Fix guidance:
  1. Downgrade to a remark: "The partition condition keeps texture prediction out of the bulk, so the capacity that would be spent modelling $z_{\text{tex}}$ is not charged against $\log|\mathcal K|$; this supports (does not enforce) staying below BarrierEpi."
  2. Or prove a concrete statement, e.g. an inequality between the bulk information volume with and without texture.
- Required new assumptions/permits: none for option 1.
- Validation plan: cross-read against 02_limits_barriers.md:63 after the edit.

### [E-022] Stopping criterion is declared "equivalent" to an information criterion that the cited theorem does not contain (was F-022)
- Location: Def. Stopping Criterion (lines 690-701)
- Severity: Moderate
- Type: Citation / reference error (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "This is equivalent to the information stopping criterion $I_{\text{bulk}}(z) \ge C_\partial$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). In practice, choose $R_{\text{cutoff}} = 1 - \varepsilon$ with $\varepsilon$ tied to Levin length/resolution."
- Upstream anchor: 05_geometry/01_metric_law.md:180-190 (`thm-capacity-constrained-metric-law`): "stationarity of a capacity-constrained curvature functional implies $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}$"; 01_metric_law.md:67-71: "$I_{\text{bulk}} \le C_{\partial}$ ... Units: $[I_{\text{bulk}}]=[C_{\partial}]=\mathrm{nat}$"; 01_metric_law.md:111: "$I_{\text{bulk}}(\Omega) := \int_{\Omega} \rho_I(z,s)\, d\mu_G$"; 01_metric_law.md:325: "| 40 | CapacitySaturationCheck | ... | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon$ |". "Levin length" is defined only in 10_appendices/01_derivations.md:1177 and 08_multiagent/03_parameter_sieve.md:116, neither referenced here.
- Why this is an error: The cited theorem is the curvature field equation and contains no stopping criterion. Upstream $I_{\text{bulk}}$ is an integral over a region $\Omega$ of the belief density, not a function $I_{\text{bulk}}(z)$ of the particle position, and the upstream relation is a constraint $I_{\text{bulk}}\le C_\partial$ whose violation is a Node 40 failure, not a termination rule. No relation between $|z|$ and $I_{\text{bulk}}$ is given anywhere in Volume 1, so "equivalent" is unproven. The only pointwise information measure in this chapter is $-U(z)=2\operatorname{artanh}|z|$; if that is meant, the equivalence $|z|\ge R\iff 2\operatorname{artanh}|z|\ge C$ is trivial with $R=\tanh(C/2)$ and should be stated as such. "Levin length" is used without definition or reference.
- Impact on downstream results: Node 25 (lines 782-791), 04_equations_motion.md:1310, 06_fields/03_info_bound.md:666.
- Fix guidance:
  1. Replace by "Equivalently, in terms of the information depth $-U(z)=2\operatorname{artanh}|z|$ (Def. `def-hyperbolic-information-potential`), stop when $-U(z)\ge C_{\text{stop}}$ with $R_{\text{cutoff}}=\tanh(C_{\text{stop}}/2)$; choosing $C_{\text{stop}}\le C_\partial$ keeps the generated state within the boundary budget of Section 18 (Node 40)."
  2. Add a reference for the Levin length (Appendix A).
- Required new assumptions/permits: none.
- Validation plan: confirm the cited label after the edit resolves to a definition that contains the quantity used.

### [E-023] Node 25 naming and remedy refer to undefined objects (was F-023)
- Location: Node 25 block (lines 782-791); TLDR line 10
- Severity: Minor
- Type: Notation conflict (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 783-787: "**Node 25: RadialGenCheck (HoloGenCheck)** ... | **25** | **RadialGenCheck** | ..."; line 791: "Remedy: Increase $\tau_{\text{max}}$ or decrease $R_{\text{cutoff}}$."
- Upstream anchor: 02_sieve/01_diagnostics.md:123: "| **25** | **HoloGenCheck** | **Generator** | **Generation Validity** | Did flow reach boundary? | $\mathbb{I}(\lvert z_{\text{final}}\rvert \ge R_{\text{cutoff}})$ | $O(B)$ |"; 04_equations_motion.md:1310: "**Node 25 (HoloGenCheck)**"; 06_fields/03_info_bound.md:666: "| 25 | HoloGenCheck |".
- Why this is an error: The node is called HoloGenCheck everywhere else in the book; this chapter's table uses RadialGenCheck as the primary name, so the Sieve registry and this definition disagree. The remedy "increase $\tau_{\text{max}}$" refers to a horizon never defined in this chapter: the stopping rule (Def. 21.3.4, line 696) is $\tau_{\text{stop}}=\inf\{\tau:|z|\ge R_{\text{cutoff}}\}$ with no maximum time, so the check can only fail if some undeclared cap on $\tau$ exists. "Decrease $R_{\text{cutoff}}$" trades away exactly the specificity whose lack triggers the check, which should be said.
- Impact on downstream results: Sieve tables in 02_sieve/01_diagnostics.md and 06_fields/03_info_bound.md.
- Fix guidance:
  1. Name the node HoloGenCheck (alias RadialGenCheck).
  2. Add $\tau_{\text{max}}$ to Def. 21.3.4: $\tau_{\text{stop}}:=\min\{\tau_{\text{max}},\inf\{\tau:|z|\ge R_{\text{cutoff}}\}\}$.
  3. Phrase the remedy as "increase $\tau_{\text{max}}$ (preferred) or lower $R_{\text{cutoff}}$ at the cost of specificity".
- Required new assumptions/permits: the horizon $\tau_{\text{max}}$ as a declared parameter.
- Validation plan: check the three Sieve tables agree on the node name and failure condition.

## Scope restrictions and clarifications
- All explicit computations in §21.1-21.2 (volume growth, $r(\tau)=\tanh(\tau/2)$, polar SDEs) are valid only for $D=2$; the chapter should either fix $D=2$ or carry $(D-1)$ factors (E-003, E-011).
- Thm. 21.2.5 as derived requires $\alpha=1$, $\gamma_{\text{risk}}=0$, $\mathcal F=0$ and friction absorbed into the time unit (E-013); with these hypotheses stated, the angular SDE (line 342) and the radial noise coefficient (line 334) are correct.
- The verifier recomputed and confirmed the following formulas: $\nabla_G S=\tfrac{(1-r^2)^2}{4}\cdot\tfrac{2}{1-r^2}\hat z=\tfrac{1-r^2}{2}\hat z$ (line 122); $\tfrac{d}{d\tau}\tanh(\tau/2)=\tfrac{1-\tanh^2(\tau/2)}{2}$ (line 148); $\Sigma(0.5)=\sigma^2\cdot9/64$ (line 625); the numerical values in the volume-growth box (lines 109-113).
- The stabiliser claim $H_0=\{e\}$ at the origin is inherited from 08_multiagent/01_gauge_theory.md:1376 (and holds only for $D=2$); it is not charged to this chapter.
- The upstream typo $\dot z=\tfrac{1-|z|^2}{2}z$ (without the hat) at 04_equations_motion.md:949 is noted but is an upstream issue.

## Open questions
- Is "entropy" in the chapter's narrative intended to mean the log-volume $S(r)$ (grows outward) or the residual output uncertainty (shrinks outward)? The answer determines the wording fix for E-001 and the interpretation of $U$ in `def-effective-potential`.
- Is $u_\pi$ meant to be a covector force (as in the second-order Langevin equation of Chapter 22) or a velocity vector (as in Def. 21.2.2)? A single convention is needed before E-007, E-011 and E-013 can be fixed consistently.
- Is $V_{\text{critic}}$ a cost-to-go (02_control_loop) or a reward-valued function (04:406 prose)? The choice fixes the sign in E-010 and the reconciliation with $\Phi_{\text{eff}}$.
- Should the direction-persistence result of Thm. 21.2.5 be reformulated as an integrability statement (finite total angular variance along $r(\tau)$), abandoning the "phase transition" framing (E-012, E-014)?
- Should the chapter smooth $U$ near the origin so that $F_{\text{entropy}}(0)=0$ genuinely holds (E-005, option 4), or keep the cone singularity and weaken Prop. 21.2.1 to a directional statement?

## Rejected candidate findings
None. All 23 stage-1 findings were confirmed or adjusted by the verifier; no finding was rejected and none was added.
