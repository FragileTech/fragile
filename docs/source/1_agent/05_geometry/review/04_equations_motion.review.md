# Mathematical Review: docs/source/1_agent/05_geometry/04_equations_motion.md

## Metadata
- Reviewed file: docs/source/1_agent/05_geometry/04_equations_motion.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1364 lines)
- Framework anchors (definitions/axioms/permits):
  - docs/source/1_agent/05_geometry/03_holographic_gen.md: `def-hyperbolic-information-potential` (169-174), `def-the-control-field` (258-268), the radial flow proof (148), the Cartesian overdamped SDE (359)
  - docs/source/1_agent/05_geometry/02_wfr_geometry.md: WFR action with vector potential and unbalanced continuity equation (112-126)
  - docs/source/1_agent/06_fields/02_reward_field.md: `def-value-curl` (211-219), `def-conservative-reward-field` (231), `thm-ness-existence` (708)
  - docs/source/1_agent/10_appendices/01_derivations.md: Appendix A.4 overdamped limit (329-405), Appendix A.5 (410-425)
  - docs/source/1_agent/10_appendices/02_parameters.md: entry for $\Phi_{\text{gen}}$ (76)
  - docs/source/1_agent/01_foundations/02_control_loop.md: `sec-levi-civita-connection-and-parallel-transport` (859)
  - docs/source/1_agent/08_multiagent/04_dnn_blocks.md: references at lines 33, 2843, 3435
  - Recomputations: sympy/numpy scripts (Poincaré ball, $d=2,3$; Hamilton's equations, Fokker-Planck residuals, BAOAB A-step trace, Cayley rotation)

## Executive summary
- Critical: 0
- Major: 6
- Moderate: 7
- Minor: 6
- Notes: 1
- Primary themes:
  1. The central second-order SDE (`def-bulk-drift-continuous-flow`) has the geodesic/metric-gradient term in the momentum equation with the wrong sign relative to the Hamiltonian it claims to derive from and relative to the second-order form used later in the chapter.
  2. The "extended Onsager-Machlup action" is not the Onsager-Machlup functional of the diffusion it is attached to; its Euler-Lagrange equation accelerates uphill in $\Phi_{\text{eff}}$, and the cited derivation does not exist.
  3. The Boris-BAOAB definition and the reference implementation do not integrate the stated SDE: the written B-step applies $2h\nabla\Phi$ per step, the A-step never transports the momentum (so the geodesic term is dropped and kinetic energy is not conserved for free motion), the code adds a spurious Christoffel correction before an exact exponential map, reuses a stale gradient, and its jump branch is unreachable.
  4. The overdamped limit is internally inconsistent (noise without $\sqrt{\gamma}$, mobility applied to the drift only, two mobilities differing by $\gamma$, a proof pointer whose algebra does not give the stated equation), and the resulting Ito SDE lacks the $-T_cG^{ij}\Gamma^k_{ij}$ drift, so its stationary-law corollary holds only in $d=2$.
  5. The control field $u_\pi$ is used with three incompatible types; several diagnostic proxies and the adaptive-temperature section are dimensionally or logically inconsistent.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Stochastic Action Principle, 99-119, 135-156, 183-196 | Major | Conceptual (Computational error) | External | this chapter | Stated functional is not the OM functional of the diffusion; its minimiser rolls uphill; no derivation in A.4 |
| E-002 | def-extended-onsager-machlup-action 115; def-bulk-drift-continuous-flow 241 | Minor | Dimensional mismatch | Framework | this chapter | Units of action and momentum inconsistent with $[G]=[z]^{-2}$ and $dz=G^{-1}p\,ds$ |
| E-003 | def-bulk-drift-continuous-flow 224-230, 248, 252-257 | Major | Computational error (Invalid inference) | Framework | this chapter | Geodesic term in $dp_k$ has the sign opposite to $-\partial_kH$ |
| E-004 | def-bulk-drift-continuous-flow 227, 237; code 744-746; table 1291; Node 26, 1317 | Moderate | Definition mismatch (Dimensional mismatch) | Framework | this chapter | $u_\pi$ is a velocity upstream, a covector force here, an acceleration in Node 26 |
| E-005 | def-bulk-drift-continuous-flow 252-258 | Minor | Scope restriction | Framework | this chapter | Hamiltonian and Boltzmann claims need $\mathcal{F}=0$ and gradient control |
| E-006 | prop-explicit-christoffel-symbols-for-poincare-disk 324 | Minor | Miswording | External | this chapter | "Accelerates" should be "decelerates" for outward coordinate motion |
| E-007 | def-mass-evolution-jump-process 347; prop-jump-intensity-from-value-discontinuity 364-384 | Minor | Notation conflict (Miswording) | Framework | this chapter | $\lambda_{\text{jump}}(z)$ is target-dependent; proposition has no proof content |
| E-008 | def-baoab-splitting 556-575 vs code 743-747, 777-781 | Major | Algorithm mismatch (Computational error) | External | this chapter | B-step as written applies $2h\nabla\Phi$ per step; code applies $h$ and omits Boris rotation |
| E-009 | def-baoab-splitting 560-564, 579 | Moderate | Scope restriction (Dimensional mismatch) | External | this chapter | Boris rotation written with 3-D cross products; preserved norm misidentified |
| E-010 | def-baoab-splitting 567, 571; "Why Splitting Works" 588-593; code 748-757, 769-775 | Major | Algorithm mismatch | Framework | this chapter | A-step never transports $p$; geodesic term not integrated; spurious Christoffel correction in code |
| E-011 | Algorithm 22.4.2, 777-803; Algorithm 22.6.2, 1091-1110 | Moderate | Algorithm mismatch | Framework | this chapter | Stale gradient, unreachable jump branch, hard-coded target/$\eta$, unused kick |
| E-012 | thm-overdamped-limit 897-931, 936-938; table 1291; Node 27, 1330 | Major | Computational error (Invalid inference) | Framework | this chapter (proof upstream 10_appendices/01_derivations.md:329-405) | Overdamped limit inconsistent with the Definition and with its cited proof |
| E-013 | thm-overdamped-limit 921-929; cor-fokker-planck-duality 965-986; Fokker-Planck box 1007-1013; Detailed Balance box 871 | Major | Scope restriction (Proof gap / omission) | Framework | this chapter | Ito SDE lacks $-T_cG^{ij}\Gamma^k_{ij}$ drift; stationary law holds only for $d=2$; FP proof mixes reference measures |
| E-014 | thm-overdamped-limit 922; cor-recovery-of-holographic-flow 951; cor-fokker-planck-duality 971 | Minor | Notation conflict | Framework | this chapter | $\Phi_{\text{gen}}$ undefined here and inconsistent with $\Phi_{\text{eff}}$ |
| E-015 | cor-recovery-of-holographic-flow 940-955; prop-mode-interpretation 458 | Moderate | Computational error (Proof gap / omission) | Framework | this chapter | Displayed vector field off by $1/\lvert z\rvert$; hypotheses $\gamma_{risk}=0$, $u_\pi=0$, time unit missing |
| E-016 | def-einstein-relation-on-manifolds 1168-1179; prop-automatic-phase-transitions 1180-1192; Algorithm 22.7.4, 1213-1233; cor-deterministic-boundary 1247-1258 | Moderate | Conceptual (Computational error) | Framework | this chapter | Einstein relation does not fix $T_c(z)$; algorithm contradicts docstring; position-dependent $T_c$ breaks Boltzmann claims |
| E-017 | Summary table 1290 | Minor | Notation conflict (Computational error) | Framework | this chapter | "Full Geodesic SDE" row is not the Definition's SDE |
| E-018 | Node 27, 1326-1335 | Moderate | Dimensional mismatch (Definition mismatch) | Framework | this chapter | Proxy reduces to $1/\lVert v\rVert$ for $\beta=0$; does not test friction vs inertia |
| E-019 | Node 28, 1339-1345 | Moderate | Definition mismatch | Framework | this chapter | Proxy is neither detailed balance nor mass balance |
| E-020 | (external) 08_multiagent/04_dnn_blocks.md 33, 2843, 3435 | Note | Citation / reference error | Framework | upstream 08_multiagent/04_dnn_blocks.md | Three references target labels that do not exist |

## Detailed findings

### [E-001] The "extended Onsager-Machlup action" is not the OM functional of the stated diffusion (was F-001)
- Location: The Stochastic Action Principle, lines 99-119 (`def-extended-onsager-machlup-action`), 135-156 (Physics Isomorphism box), 183-196 (`prop-most-probable-path`)
- Severity: Major
- Type: Conceptual (secondary: Computational error)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 105: "$S_{\mathrm{OM}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds$"; lines 186-192: "For the controlled diffusion $dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s$, where $\sigma \sigma^T = G^{-1}$, the most probable path connecting $z(0) = z_0$ and $z(T) = z_1$ minimizes the Onsager-Machlup action $S_{\mathrm{OM}}[z]$"; line 194: "This follows from the Girsanov theorem and the Cameron-Martin formula ... See ... Appendix A.4"; line 155: "Path probability $e^{-S_{\text{OM}}/T_c}$".
- Upstream anchor: none (the functional is defined here). Appendix A.4 (10_appendices/01_derivations.md:329-405) contains only the singular-perturbation argument for the overdamped limit; there is no Girsanov or Cameron-Martin step anywhere in it.
- Why this is an error: For $dz = b\,ds + \sqrt{2T_c}\,\sigma\,dW$ with $\sigma\sigma^T=G^{-1}$ the Onsager-Machlup Lagrangian is $\frac{1}{4T_c}\|\dot z - b\|_G^2 + \frac12\nabla_G\!\cdot b$ (plus a curvature term). For a gradient drift $b=-G^{-1}\nabla\Phi$ this expands to $\frac{1}{4T_c}\|\dot z\|_G^2 + \frac{1}{2T_c}\dot z\cdot\nabla\Phi + \frac{1}{4T_c}\|\nabla\Phi\|^2_{G^{-1}} - \frac12\Delta_G\Phi$; the cross term is a total derivative and drops out for fixed endpoints. The potential therefore enters as $\|\nabla\Phi\|^2$, never as $\Phi$ itself. The stated functional differs on every count: (i) $\Phi_{\text{eff}}$ appears linearly; the Euler-Lagrange equation of $\frac12\|\dot z\|^2+\Phi$ is $\ddot z = +\nabla\Phi$ (verified symbolically), so the "most probable path" of the stated action accelerates toward higher potential, contradicting the force $-\partial_k\Phi_{\text{eff}}$ of line 227; (ii) the normalisation is $\frac12\|\dot z\|^2$ with weight $e^{-S/T_c}$, i.e. $\frac{1}{2T_c}$ instead of $\frac{1}{4T_c}$; (iii) the $\frac12\nabla\cdot b$ term is absent; (iv) the entropy term $T_cH_\pi$ has no counterpart, since the drift $b$ of line 189 (never specified) does not depend on the policy entropy; (v) the sign of the $\frac{1}{12}R$ term depends on the tube convention, which is not stated. The proof pointer is empty.
- Impact on downstream results: `prop-most-probable-path` is unsupported; the summary row (line 1289) and the closing statement "they all emerge from a single variational principle: minimize the Onsager-Machlup action" (line 1278) are not established; the Physics Isomorphism box (135-156) mislabels the correspondence.
- Fix guidance:
  1. Either rename the functional (e.g. "free-energy action") and drop the claim that its minimiser is the most probable path of the diffusion, or
  2. replace it by the genuine OM Lagrangian $L = \frac{1}{4T_c}\|\dot z - b\|_G^2 + \frac12\nabla_G\!\cdot b + c\,R$ with $b = -G^{-1}\nabla\Phi_{\text{eff}} + u_\pi$ and an explicitly stated tube convention fixing $c$;
  3. move the entropy term into the definition of $\Phi_{\text{eff}}$ or the MaxEnt objective;
  4. point the proof to an actual derivation (write it in Appendix A, or cite the external result as an external dependency).
- Required new assumptions/permits: a stated path-tube convention for the curvature term; specification of $b$.
- Validation plan: check that the Euler-Lagrange equation of the corrected functional, in the $T_c\to0$ limit, reproduces the deterministic part of line 227 (with the sign of E-003 fixed); check the flat-space case against the standard OM functional.

### [E-002] Units of the action and of the momentum are inconsistent with the equations (was F-002)
- Location: line 115; line 241; line 75
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 115: "Units: $[S_{\mathrm{OM}}] = \mathrm{nat}$."; line 241: "*Units:* $[z] = \text{length}$, $[p] = \text{length}/\tau$, $[\gamma] = 1/\tau$, $[\Phi_{\text{eff}}] = \mathrm{nat}$, $[T_c] = \mathrm{nat}$."; line 75: "Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric)."
- Upstream anchor: none (declared here).
- Why this is an error: With $[G_{ij}]=[z]^{-2}$ the kinetic term $\frac12G_{ij}\dot z^i\dot z^j$ has units $\tau^{-2}$ while $\Phi_{\text{eff}}$ has units nat; they cannot be added, and after $\int ds$ the action would have units nat$\cdot\tau$. From $dz^k = G^{kj}p_j\,ds$ (line 226), $[z]/\tau = [z]^2[p]$, hence $[p] = 1/([z]\tau)$, not length$/\tau$ (that is the unit of the velocity $G^{-1}p$). With the declared $[p]$, $\frac12G^{ij}p_ip_j$ would have units $[z]^4/\tau^2$.
- Impact on downstream results: the "Units" column of the summary table (1287-1294) and the fluctuation-dissipation statements (1174-1177).
- Fix guidance: (1) either declare once that $\tau$ is dimensionless (natural units, $T_c$ sets the scale) and remove per-symbol unit lines, or (2) correct to $[p]=1/([z]\tau)$ and make the kinetic term dimensionally nat by an explicit time scale.
- Required new assumptions/permits: none.
- Validation plan: dimensional check of every term in lines 105, 227, 255.

### [E-003] Sign error in the geodesic term of the momentum equation (was F-003)
- Location: `def-bulk-drift-continuous-flow`, lines 224-230, 248, 252-257
- Severity: Major
- Type: Computational error (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 227: "$dp_k = \left[ -\partial_k \Phi_{\text{eff}} - \gamma\, p_k + \beta_{\text{curl}}\, \mathcal{F}_{kj}\, G^{j\ell}\, p_\ell - \Gamma^m_{k\ell}\, G^{\ell j}\, p_j\, p_m + u_{\pi,k} \right] ds + \sqrt{2\gamma T_c}\, (G^{1/2})_{kj}\, dW^j_s$"; lines 252-255: "The deterministic part ($T_c = 0$, $\gamma = 0$) derives from the Hamiltonian: $H(z, p) = \frac{1}{2} G^{ij}(z)\, p_i\, p_j + \Phi_{\text{eff}}(z)$."
- Upstream anchor: internal consistency with line 916 ("$m\,\ddot{z}^k + \gamma\,\dot{z}^k - \ldots + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \ldots$") and line 319.
- Why this is an error: Hamilton's equation gives $\dot p_k = -\partial_kH = -\partial_k\Phi - \frac12\partial_kG^{ij}p_ip_j = -\partial_k\Phi + \frac12\partial_kG_{ab}v^av^b$ with $v=G^{-1}p$. The chapter's term is $-\Gamma^m_{k\ell}v^\ell p_m = -\Gamma_{mk\ell}v^mv^\ell = -\frac12(\partial_kG_{m\ell}+\partial_\ell G_{mk}-\partial_mG_{k\ell})v^mv^\ell = -\frac12\partial_kG_{m\ell}v^mv^\ell$, exactly the negative of the Hamiltonian term. Symbolic verification on the Poincaré ball ($d=3$): the difference between $-\partial_kH$ and the negative of the chapter's term is identically zero, while the difference from the chapter's term itself is $z_k(1-|z|^2)|p|^2$. Numerically at $z=(0.3,-0.2,0.4)$, $p=(0.7,0.1,-0.5)$: $-\nabla_zH = (0.0799,-0.0533,0.1065)$; chapter's term $=(-0.0799,0.0533,-0.1065)$. The correct Hamiltonian equations reproduce $\ddot z^k = -\Gamma^k_{ij}\dot z^i\dot z^j$ (residual zero), which is the form used at lines 319 and 916; the chapter's first-order and second-order statements of "the" SDE therefore disagree. The intuition "$-\Gamma(v,v)$" (lines 248, 285, 301) is correct for $\ddot z$, not for $\dot p$.
- Impact on downstream results: the Hamiltonian structure and Boltzmann statements (252-258), `prop-baoab-preserves-boltzmann`, `thm-overdamped-limit` (silently uses the correct form), Node 26, and any implementation following line 227.
- Fix guidance:
  1. Replace $-\Gamma^m_{k\ell}G^{\ell j}p_jp_m$ by $+\Gamma^m_{k\ell}G^{\ell j}p_jp_m$, equivalently $-\frac12\partial_kG^{ij}p_ip_j$.
  2. Reword item 4 (line 248): "metric-gradient force $-\frac12\partial_kG^{ij}p_ip_j$, which yields $\ddot z = -\Gamma(\dot z,\dot z)$ in second-order form."
- Required new assumptions/permits: none.
- Validation plan: recompute $\partial_t(G^{-1}p)$ from the corrected system and confirm it equals $-\Gamma(v,v) + \ldots$; check on a radial Poincaré geodesic that $|p|$ increases toward the boundary.

### [E-004] The control field $u_\pi$ is used with three incompatible types (was F-004)
- Location: lines 227, 237; code 744-746; table 1291; Node 26, line 1317
- Severity: Moderate
- Type: Definition mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 227: "$\ldots + u_{\pi,k} \right] ds$"; line 237: "$u_{\pi,k}$ is the **control field** from the policy (Definition def-the-control-field)"; line 745: "total_force = grad_Phi - u_pi"; line 1291: "$dz = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi\right)\,ds + \ldots$"; line 1317: "$\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi_{\text{eff}} - u_\pi - \ldots\rVert_G$".
- Upstream anchor: 03_holographic_gen.md:262-268: "$u_\pi(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi_\theta}[a]$ ... on the tangent bundle $T\mathbb{D}$ ... Units: $[u_\pi] = [z]/\tau$"; 03_holographic_gen.md:359: "$dz^k = -G^{kj}\partial_j U\, d\tau + u_\pi^k\, d\tau + \ldots$".
- Why this is an error: Upstream $u_\pi$ is a contravariant velocity added to $dz$. Here it is added to $dp_k$ as a covector force (units $[p]/\tau$), reappears as a velocity in the table and as an acceleration in Node 26. The code subtracts the vector from the covector `grad_Phi` component-wise, meaningful only for $G\propto I$. No conversion relates the three usages.
- Impact on downstream results: overdamped row of the table, Node 26, `run_agent_loop`, consistency of Sections 21 and 22.
- Fix guidance: keep the upstream velocity definition and write the momentum-space force as $\gamma G_{kj}u_\pi^j$ (so the overdamped limit returns $+u_\pi$ in $dz$), or define a separate covector "control force" $f_{\pi,k}$; make table, Node 26 and code use one object.
- Required new assumptions/permits: none.
- Validation plan: dimensional check of each occurrence; verify the overdamped limit reproduces 03_holographic_gen.md:359.

### [E-005] Hamiltonian and Boltzmann claims stated without the hypotheses they need (was F-005)
- Location: lines 252-258
- Severity: Minor
- Type: Scope restriction
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The deterministic part ($T_c = 0$, $\gamma = 0$) derives from the Hamiltonian ... The friction and noise terms implement an **Ornstein-Uhlenbeck thermostat** that samples the Boltzmann distribution $\rho(z,p) \propto \exp(-H(z,p)/T_c)$."
- Upstream anchor: 02_wfr_geometry.md:112-126 (WFR action with vector potential $\mathbf{A}$, $d\mathbf{A}=\mathcal{F}$).
- Why this is an error: With $T_c=\gamma=0$ the right-hand side still contains the Lorentz term and $u_\pi$; the former is Hamiltonian only with minimal coupling $\frac12G^{ij}(p_i-\beta A_i)(p_j-\beta A_j)$, the latter is not Hamiltonian for a generic policy. The Boltzmann statement requires $\mathcal{F}=0$ and gradient control; the chapter itself calls the $\mathcal{F}\ne0$ case a NESS (lines 262, 882).
- Impact on downstream results: `prop-baoab-preserves-boltzmann` inherits the same unstated hypotheses.
- Fix guidance: add "with $\beta_{\text{curl}}=0$ and $u_\pi = -G^{-1}\nabla\Phi_\pi$ (or absorbed into $\Phi_{\text{eff}}$)" to both sentences, or state the minimal-coupling Hamiltonian.
- Required new assumptions/permits: none beyond the stated restriction.
- Validation plan: verify $\dot z = \partial H/\partial p$, $\dot p = -\partial H/\partial z$ term by term against line 227.

### [E-006] Geometric interpretation of the Christoffel contraction has the wrong sign (was F-006)
- Location: `prop-explicit-christoffel-symbols-for-poincare-disk`, line 324
- Severity: Minor
- Type: Miswording
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "The first term $(z \cdot \dot{z})\dot{z}$ accelerates motion radially when moving outward; the second term $|\dot{z}|^2 z$ provides centripetal correction."
- Upstream anchor: none. The formula of lines 313-319 was verified symbolically and is correct.
- Why this is an error: The geodesic equation is $\ddot z = -\Gamma(\dot z,\dot z)$. For $z\cdot\dot z>0$ the first term contributes $\ddot z = -\frac{4(z\cdot\dot z)}{1-|z|^2}\dot z$, antiparallel to $\dot z$: coordinate speed decreases (radial geodesic $\dot r = c(1-r^2)/2$, $\ddot r<0$). This contradicts line 180 ("slower and slower").
- Impact on downstream results: none mathematical; misleads implementers debugging Node 26.
- Fix guidance: "decelerates coordinate motion when moving outward (keeping the hyperbolic speed constant)".
- Required new assumptions/permits: none.
- Validation plan: none beyond the radial geodesic check.

### [E-007] $\lambda_{\text{jump}}$ is target-dependent but written as a function of $z$ alone; the "Proposition" has no proof content (was F-019)
- Location: `def-mass-evolution-jump-process` line 347; `prop-jump-intensity-from-value-discontinuity` lines 364-384
- Severity: Minor
- Type: Notation conflict (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 347: "$N_s$ is a Poisson process with intensity $\lambda_{\text{jump}}(z)$"; lines 367-370: "The jump intensity $\lambda_{\text{jump}}(z)$ is determined by the value difference across chart boundaries: $\lambda_{\text{jump}}(z) = \lambda_0 \cdot \exp\left(\beta_{\text{ent}} \cdot \left( V_{\text{target}}(L(z)) - V_{\text{source}}(z) - c_{\text{transport}} \right) \right)$"; line 1292 and 1341 use "$\lambda_{K\to j}$".
- Upstream anchor: none.
- Why this is an error: The right-hand side depends on the (source, target) pair, so the object is a rate matrix $\lambda_{K\to j}(z)$; the Poisson process of line 347 needs the total intensity $\sum_j\lambda_{K\to j}$ and a target-selection law, neither stated. "Is determined by" is asserted, not derived; the code (line 794) hard-codes the target.
- Impact on downstream results: Node 28, jump step of Algorithm 22.4.2.
- Fix guidance: recast as a definition of $\lambda_{K\to j}(z)$; define $\lambda_{\text{jump}} = \sum_j\lambda_{K\to j}$ and target sampling $\propto\lambda_{K\to j}$.
- Required new assumptions/permits: none.
- Validation plan: consistency of lines 347, 370, 1292, 1341 and the code.

### [E-008] The Boris-BAOAB definition applies the gradient kick twice per B-step ($2h$ per step) (was F-007)
- Location: `def-baoab-splitting` lines 556-575 vs Algorithm 22.4.2 lines 743-747, 777-781
- Severity: Major
- Type: Algorithm mismatch (secondary: Computational error)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 558-565: "1. **B** (half kick + Boris rotation): Half-kick from gradient: $p^- \leftarrow p - \frac{h}{2}\nabla\Phi(z)$ ... Half-kick from gradient: $p \leftarrow p^+ - \frac{h}{2}\nabla\Phi(z)$"; line 573: "5. **B** ... Same as step 1".
- Upstream anchor: none.
- Why this is an error: Each B substep as written applies $h\nabla\Phi$; two B substeps per step give $2h\nabla\Phi$, so the scheme integrates $\dot p = -2\nabla\Phi+\ldots$ with an $O(1)$ error. In Boris-BAOAB the total kick per B substep is $h/2$ (two quarter-kicks around a rotation by angle $\propto h/2$). The code (746, 780) applies $h/2$ per B step, contradicting the definition, and contains no Boris rotation at all.
- Impact on downstream results: `prop-baoab-preserves-boltzmann` (813-822) and the $O(h^2)$ claims (264, 593, 820, 856) are false for the scheme as defined.
- Fix guidance: in steps 1 and 5 use quarter-kicks $p\leftarrow p-\frac h4\nabla\Phi$ before and after the rotation (or a single $\frac h2$ kick when $\mathcal{F}=0$); add the rotation to the code or state that the code covers $\mathcal{F}=0$ only.
- Required new assumptions/permits: none.
- Validation plan: harmonic-potential test in flat space: the corrected scheme must reproduce the $O(h^2)$ invariant-measure error of BAOAB.

### [E-009] Boris rotation written with 3-D cross products; not defined for general $d$ (was F-008)
- Location: `def-baoab-splitting` lines 560-564, 579
- Severity: Moderate
- Type: Scope restriction (secondary: Dimensional mismatch)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 561-564: "$t \leftarrow \frac{h}{2}\beta_{\text{curl}} G^{-1}\mathcal{F}$ (rotation vector); $p' \leftarrow p^- + p^- \times t$; $s \leftarrow \frac{2t}{1 + |t|^2}$; $p^+ \leftarrow p^- + p' \times s$"; line 579: "It rotates the momentum around the local Value Curl axis, preserving the norm $|p|$".
- Upstream anchor: 06_fields/02_reward_field.md:216-219: "$\mathcal{F} := d\mathcal{R}$ ... In coordinates: $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$" (a 2-form on the $d$-dimensional latent space).
- Why this is an error: $G^{-1}\mathcal{F}$ is a $d\times d$ matrix, not a vector; cross products, $|t|$ and a "rotation axis" exist only for $d=3$. The latent space is $d$-dimensional throughout (code shape `[B, d]`). The quantity preserved by the Lorentz force $\beta\mathcal{F}G^{-1}p$ is $\frac12p^TG^{-1}p$ (since $v^T\mathcal{F}v=0$), not the Euclidean $|p|$; a numerical Cayley step with random antisymmetric $\mathcal{F}$ and SPD $G$ in $d=4$ preserved $p^TG^{-1}p$ to $10^{-16}$ while $|p|^2$ changed from 0.490 to 0.505.
- Impact on downstream results: any implementation for $d\ne3$; the "preserves norm" remark.
- Fix guidance: replace the cross-product recipe by the Cayley transform $p^+ = (I-\frac h2\beta\mathcal{F}G^{-1})^{-1}(I+\frac h2\beta\mathcal{F}G^{-1})p^-$ and state that it preserves $\|p\|_{G^{-1}}$.
- Required new assumptions/permits: none.
- Validation plan: check $\|p^+\|_{G^{-1}} = \|p^-\|_{G^{-1}}$ numerically for random $\mathcal{F}$, $G$, $d$.

### [E-010] The A-step never updates the momentum; the geodesic term of the SDE is not integrated, and the code adds a spurious Christoffel correction (was F-009)
- Location: `def-baoab-splitting` steps 2 and 4 (lines 567, 571); "Why Splitting Works" (588-593); code lines 748-757, 769-775
- Severity: Major
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 567: "2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$"; lines 590-593: "**A-step** (drift): $z \to \exp_z(\frac{h}{2}v)$ is following a geodesic ... None of these is approximate---each sub-step is exact"; lines 754-757: "geodesic_corr = christoffel_contraction(z, velocity); velocity_corrected = velocity - (h / 4) * geodesic_corr  # half of half-step; z = poincare_exp_map(z, (h / 2) * velocity_corrected)".
- Upstream anchor: the SDE of line 227 (this chapter), whose momentum equation contains the metric-gradient term.
- Why this is an error: The exact flow of $\frac12G^{ij}p_ip_j$ over time $h/2$ is the geodesic flow on $T^*\mathcal{Z}$: it moves $z$ along $\exp_z$ and transports $p$ to $G(z')\dot\gamma(h/2)$. The definition and the code leave the components of $p$ unchanged, so the $\Gamma$ (equivalently $\partial_kG^{ij}p_ip_j$) term of the momentum equation is never applied anywhere in the scheme. The code additionally modifies the initial velocity by $-\frac h4\Gamma(v,v)$ before feeding it to `poincare_exp_map`, which already follows the exact geodesic. Numerical trace (numpy reimplementation of the chapter's functions; free motion on the Poincaré ball, $z=(0.3,-0.2,0.4)$, $p=(0.7,0.1,-0.5)$, 40 half-steps of $h=0.05$): kinetic energy $H=\frac12G^{ij}p_ip_j$ changes by $-2.71\%$ with the code's A-step and by $-2.65\%$ with the definition's A-step, while the exact cotangent geodesic flow conserves it to $3.5\times10^{-15}$; endpoint error against $\exp_z(v)$ is 0.0084 (code), 0.0087 (definition), $1.7\times10^{-7}$ (exact flow). The Christoffel "correction" does not remove the error.
- Impact on downstream results: the deterministic part of the integrator is not symplectic or reversible, so `prop-baoab-preserves-boltzmann` fails for it; Node 26 would flag the integrator's own trajectories; any chapter citing `def-baoab-splitting` as the geodesic integrator.
- Fix guidance:
  1. In steps 2 and 4 update both variables: $(z,p)\leftarrow\big(\exp_z(\tfrac h2v),\ G(z')P_{z\to z'}(v)\big)$ with $P$ the parallel transport along the geodesic (closed form on the Poincaré ball via gyrovector parallel transport).
  2. Delete the `velocity_corrected` lines and use `velocity` directly in `poincare_exp_map`.
  3. Implement the momentum transport in the code.
- Required new assumptions/permits: none.
- Validation plan: free-motion test (no forces, $T_c=0$): $H$ must be conserved to round-off; endpoint must match $\exp_z(v)$.

### [E-011] Reference implementation deviates from the stated algorithm in several places (was F-010)
- Location: Algorithm 22.4.2 lines 777-781, 783-803; Algorithm 22.6.2 lines 1091-1110
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 777-780: "# ===== B-step: half kick ===== # Recompute gradient at new position (for accuracy) # In practice, often reuse grad_Phi for efficiency; p = p - (h / 2) * total_force"; line 792: "if jumps.any() and value_fn is not None:"; line 794: "K_target = (K + 1) % 4  # Example: cycle through 4 charts"; line 802: "eta = 1.1"; line 1093: "u_pi = policy.symmetry_breaking_kick(z, mode='generation')"; line 1103: "u_pi = policy.control_field(state.z, state.K)"; lines 1106-1110 call `geodesic_baoab_step` without `value_fn`.
- Upstream anchor: none.
- Why this is an error: (i) The second B-kick must use $\nabla\Phi$ at the new position; reusing the stale `total_force` destroys the symmetric structure invoked at line 818 and reduces the invariant-measure accuracy to first order. (ii) `run_agent_loop` never passes `value_fn`, so the condition at line 792 is always false and no jump ever occurs, contradicting Phase 3 "BAOAB + Jumps" (line 1058). (iii) Target chart and $\eta$ are hard-coded rather than derived from lines 342-370; `jump_rate_fn(z, K)` cannot express the target dependence. (iv) The kick computed at line 1093 is overwritten at line 1103 before use, so Phase 2 is a no-op; moreover $\nabla U$ is undefined at $z=0$, where `compute_effective_potential_gradient` is first called.
- Impact on downstream results: the Lifecycle table (1054-1060) and the claim that the algorithms implement Definitions 22.2.1-22.2.2.
- Fix guidance: accept a gradient callable and recompute at the updated `z` before the final kick; pass `value_fn` or drop the condition; sample the target from $\lambda_{K\to j}$; apply the kick during the first step (or add it to the control field) and guard the origin.
- Required new assumptions/permits: none.
- Validation plan: unit test that jumps occur with nonzero rate; time-reversibility test of the deterministic part.

### [E-012] The overdamped limit is internally inconsistent and not what Appendix A.4 proves (was F-011)
- Location: `thm-overdamped-limit` lines 910-931; prose 897-901, 936-938; table 1291; Node 27 line 1330
- Severity: Major
- Type: Computational error (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter (cited proof upstream: docs/source/1_agent/10_appendices/01_derivations.md:329-405)
- Claim (verbatim): line 916: "$m\,\ddot{z}^k + \gamma\,\dot{z}^k - \beta_{\text{curl}} G^{km}\mathcal{F}_{mj}\dot{z}^j + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j$"; line 919: "In the limit $\gamma \to \infty$ with $m$ fixed"; line 922: "$dz^k = \left[\mathcal{M}_\gamma(z)\right]^{k}{}_{j}\left(-G^{j\ell}(z)\,\partial_\ell\Phi_{\text{gen}}(z)\right) ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s$"; line 899: "$\mathcal{M}_\gamma := (\gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$"; line 937: "$\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$".
- Upstream anchor: this chapter line 227 (noise on $p$ is $\sqrt{2\gamma T_c}(G^{1/2})_{kj}dW^j$); 01_derivations.md:336: "$m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,(G^{-1/2})^{kj}\,\xi^j$" (no Lorentz term); 01_derivations.md:359-369: "$\frac{dz_0^k}{d\tilde{s}} = -\frac{1}{\gamma}\,G^{kj}(z_0)\,\partial_j\Phi(z_0) + \sqrt{\frac{2T_c}{\gamma}}\,(G^{-1/2})^{kj}\,\tilde{\xi}^j$. Returning to original computation time $s = \gamma\tilde{s}$ and using $dz_0/ds = (1/\gamma)\,dz_0/d\tilde{s}$: $dz_0^k = -G^{kj}(z_0)\,\partial_j\Phi(z_0)\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s$."
- Why this is an error: (i) Converting line 227 to second order gives noise $\sqrt{2\gamma T_c}G^{-1/2}\xi$ (fluctuation-dissipation), not $\sqrt{2T_c}G^{-1/2}\xi$; the theorem also introduces an $m$ absent from the Definition and drops $u_\pi$. (ii) Setting $m\ddot z\approx0$ in line 916 gives $\dot z = \mathcal{M}_\gamma(-G^{-1}\nabla\Phi + \sqrt{2T_c}G^{-1/2}\xi)$: the mobility must multiply the noise too. (iii) $\mathcal{M}_\gamma=O(1/\gamma)$, so as $\gamma\to\infty$ the drift vanishes while the noise stays $O(1)$; line 922 is a $\gamma$-dependent expression, not a limit. The consistent statement is $dz = \frac1\gamma(I-\frac\beta\gamma G^{-1}\mathcal{F})^{-1}(-G^{-1}\nabla\Phi)\,ds + \sqrt{2T_c/\gamma}\,G^{-1/2}dW$, which after $s\to\gamma s$ has unit drift and noise $\sqrt{2T_c}$, with a curl correction of order $\beta/\gamma$. (iv) $\mathcal{M}_\gamma$ (899, 922, 1330) and $\mathcal{M}_{\text{curl}}$ (937, 1291; also 01_derivations.md:422) differ by the factor $\gamma$. (v) Appendix A.4 has no Lorentz term, so the curl-corrected limit is not proven there; its own algebra is inconsistent: from $dz_0/d\tilde s = -\frac1\gamma G^{-1}\nabla\Phi$ and $dz_0/ds = \frac1\gamma dz_0/d\tilde s$ one gets $-\gamma^{-2}G^{-1}\nabla\Phi$, not the unit coefficient of line 369; and the appendix's conclusion (unit coefficient, no $\mathcal{M}_\gamma$) is not the theorem's equation.
- Impact on downstream results: `cor-recovery-of-holographic-flow`, `cor-fokker-planck-duality`, Node 27, the overdamped table row, Appendix A.5 (01_derivations.md:418-425), and 03_holographic_gen.md:356-362, which cites this theorem.
- Fix guidance:
  1. State the theorem with FDT-consistent noise $\sqrt{2\gamma T_c}G^{-1/2}\xi$ and keep $u_\pi$.
  2. Take $m\to0$ or rescale time by $\gamma$; give $dz = \mathcal{M}(-G^{-1}\nabla\Phi_{\text{eff}} + \gamma^{-1}\text{control})\,ds + \sqrt{2T_c}\,\mathcal{M}^{1/2}$-type noise with a single dimensionless $\mathcal{M} = (I-\frac{\beta_{\text{curl}}}{\gamma}G^{-1}\mathcal{F})^{-1}$.
  3. Use one symbol for the mobility throughout the chapter and the appendices.
  4. Rewrite Appendix A.4 to include the Lorentz term and correct the time-rescaling algebra.
- Required new assumptions/permits: a stated scaling of $\beta_{\text{curl}}$ with $\gamma$ if the curl correction is to survive the limit.
- Validation plan: check that the corrected limit reproduces 03_holographic_gen.md:359 for $\mathcal{F}=0$; simulate the second-order SDE at large $\gamma$ and compare with the first-order equation.

### [E-013] Overdamped SDE lacks the Ito curvature drift; the Boltzmann stationary density holds only for $d=2$; the Fokker-Planck proof mixes reference measures (was F-014)
- Location: `thm-overdamped-limit` lines 921-929; `cor-fokker-planck-duality` lines 965-986; Fokker-Planck box 1007-1013; Detailed Balance box line 871
- Severity: Major
- Type: Scope restriction (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (the same Ito form is inherited by 03_holographic_gen.md:359, where it is applied to the disk)
- Claim (verbatim): line 922: "$dz^k = \left[\mathcal{M}_\gamma\right]^{k}{}_{j}\left(-G^{j\ell}\,\partial_\ell\Phi_{\text{gen}}\right) ds + \sqrt{2T_c}\,\left(G^{-1/2}\right)^{kj}\,dW^j_s$"; lines 926-928: "The geodesic term $\Gamma(\dot{z},\dot{z}) \sim O(|\dot{z}|^2) = O(\gamma^{-2})$ is negligible"; line 971: "$p_*(z) \propto \exp\left(-\frac{\Phi_{\text{gen}}(z)}{T_c}\right)\,\sqrt{|G(z)|}$"; lines 976-982: "The Fokker-Planck equation for the overdamped dynamics is: $\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi + T_c\,\partial_j p \right) \right)$. Setting $\partial_s p = 0$ and using detailed balance gives $p \propto e^{-\Phi/T_c} \sqrt{|G|}$."
- Upstream anchor: line 168 (this chapter) uses the $d$-dimensional ball "$I_d$"; the code uses shape `[B, d]`.
- Why this is an error: Read as an Ito SDE, line 922 (with $\mathcal{F}=0$) has generator $\mathcal{L} = -G^{ij}\partial_i\Phi\,\partial_j + T_cG^{ij}\partial_i\partial_j$. The Laplace-Beltrami operator is $\Delta_G = G^{ij}(\partial_i\partial_j - \Gamma^k_{ij}\partial_k)$, so the SDE whose stationary law is $e^{-\Phi/T_c}\,d\mathrm{Vol}_G$ is $dz^k = (-G^{kj}\partial_j\Phi - T_cG^{ij}\Gamma^k_{ij})\,ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j$. The $O(\gamma^{-2})$ argument for dropping $\Gamma(\dot z,\dot z)$ fails for the noise-driven part of $\dot z$, whose quadratic variation is $O(1)$; that is exactly what produces the $T_cG^{ij}\Gamma^k_{ij}$ drift. Symbolic verification on the Poincaré ball: $G^{ij}\Gamma^k_{ij} = -(d-2)\frac{1-|z|^2}{2}z^k$; in $d=3$ the Fokker-Planck residual of line 922 against $e^{-\Phi/T_c}\sqrt{|G|}$ is $\frac{T_c}{2}(3+|z|^2) + \frac12(|z|^2-1)\,z\cdot\nabla\Phi \neq 0$ and becomes identically zero once the drift $-T_cG^{ij}\Gamma^k_{ij}$ is added; in $d=2$ the contraction vanishes and the residual is zero. Hence the corollary is correct for the SDE as written only in $d=2$; for $d\ne2$ the missing term is an outward radial drift of the same form as the entropic drift. Independently, the proof writes the FP equation with covariant divergence (density w.r.t. $d\mathrm{Vol}_G$, stationary solution $\propto e^{-\Phi/T_c}$) and reports the Lebesgue density with the extra $\sqrt{|G|}$, using one symbol $p$ for both. Finally, the corollary is stated for "the overdamped SDE", which per the theorem includes $\mathcal{M}_\gamma$ with curl; for $\mathcal{F}\ne0$ the steady state is a NESS (line 882), so the hypothesis $\mathcal{F}=0$ is missing. The criterion is Framework because the chapter's own SDE and its own Fokker-Planck equation disagree; no external result is needed to see it.
- Impact on downstream results: Detailed-balance box (871), Fokker-Planck box (1013), the Stein-discrepancy loss (1024), Appendix A.5 (classification as relaxation), and any $d>2$ implementation of the overdamped generator (the phase-space density $e^{-H/T_c}$ of the corrected second-order system does marginalise to $e^{-\Phi/T_c}\sqrt{|G|}$, so the target density is right and only the first-order Ito drift is missing).
- Fix guidance:
  1. State the overdamped SDE in Ito form with the drift $-T_cG^{ij}\Gamma^k_{ij}$, or declare it a Stratonovich / manifold-Brownian-motion equation and give the Ito form for implementation.
  2. Add the hypothesis $\mathcal{F}=0$ to `cor-fokker-planck-duality`.
  3. Write the FP equation and its solution with respect to one reference measure.
- Required new assumptions/permits: none.
- Validation plan: symbolic FP residual check in $d=2,3$ (as above); Monte Carlo histogram of the corrected SDE on the ball in $d=3$ against $e^{-\Phi/T_c}\sqrt{|G|}$.

### [E-014] $\Phi_{\text{gen}}$ is undefined in this chapter and conflicts with $\Phi_{\text{eff}}$ (was F-012)
- Location: line 922; line 951; line 971; vs Fokker-Planck box 1010-1013
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 922: "$-G^{j\ell}(z)\,\partial_\ell\Phi_{\text{gen}}(z)$"; line 971: "$p_*(z) \propto \exp\left(-\frac{\Phi_{\text{gen}}(z)}{T_c}\right)\,\sqrt{|G(z)|}$"; line 1013: "with stationary distribution $p_*(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)\sqrt{|G(z)|}$".
- Upstream anchor: 10_appendices/02_parameters.md:76: "$\Phi_{\text{gen}}$ | generative potential $\alpha U + (1-\alpha)V_{\text{critic}}$ (Section 22.3)".
- Why this is an error: $\Phi_{\text{gen}}$ is never defined in Section 22.3, which defines $\Phi_{\text{eff}} = \alpha U + (1-\alpha)V + \gamma_{risk}\Psi_{risk}$ (line 423). The parameter table's definition omits the risk term, so the theorem and the corollary refer to a different potential from the Definition and the Fokker-Planck box. The theorem also starts with $\Phi$ (916) and ends with $\Phi_{\text{gen}}$ (922).
- Impact on downstream results: parameter table entry; `cor-recovery-of-holographic-flow`.
- Fix guidance: use $\Phi_{\text{eff}}$ throughout, or define $\Phi_{\text{gen}} := \Phi_{\text{eff}}|_{\gamma_{risk}=0}$ explicitly and state where it applies.
- Required new assumptions/permits: none.
- Validation plan: grep for $\Phi_{\text{gen}}$ across the volume after the edit.

### [E-015] Corollary "Recovery of Holographic Flow": wrong vector field, missing hypotheses, does not follow from the theorem as stated (was F-013)
- Location: `cor-recovery-of-holographic-flow` lines 940-955; `prop-mode-interpretation` line 458
- Severity: Moderate
- Type: Computational error (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 943-949: "Setting $\alpha = 1$ (pure generation), $T_c \to 0$ (deterministic limit), and $\mathcal{F} = 0$ (conservative case) in the overdamped equation recovers ... $\dot{z} = -G^{-1}(z)\,\nabla U(z)$. For the Poincare disk, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,z$, which integrates to $|z(\tau)| = \tanh(\tau/2)$."; line 951: "Direct substitution of $\Phi_{\text{gen}} = U$ into the overdamped equation."; line 953: "the **optimal control trajectory**".
- Upstream anchor: 03_holographic_gen.md:171-174: "$U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|)$"; 03_holographic_gen.md:148: "The overdamped equation $\dot{r} = (1-r^2)/2$ ... integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$".
- Why this is an error: (i) $-G^{-1}\nabla U = \frac{(1-|z|^2)^2}{4}\cdot\frac{2}{1-|z|^2}\hat z = \frac{1-|z|^2}{2}\hat z = \frac{1-|z|^2}{2|z|}z$ (consistent with the chapter's own line 477), not $\frac{1-|z|^2}{2}z$. Symbolically the difference is $\frac{(|z|-1)(|z|^2-1)}{2|z|}z\ne0$; numerically at $z=(0.3,-0.2,0.4)$: $(0.198,-0.132,0.264)$ vs displayed $(0.107,-0.071,0.142)$. The displayed field gives $\dot r = r(1-r^2)/2$, with $r=0$ a fixed point and no $\tanh(\tau/2)$ solution; the proof silently switches to the correct $\dot r=(1-r^2)/2$. (ii) $\alpha=1$ does not remove $\gamma_{risk}\Psi_{risk}$ from $\Phi_{\text{eff}}$ (line 423); the hypothesis $\gamma_{risk}=0$ is needed here and in `prop-mode-interpretation` (line 458). (iii) With the theorem's equation and $\mathcal{F}=0$, $\mathcal{M}_\gamma=\gamma^{-1}I$, so substitution gives $\dot z = -\gamma^{-1}G^{-1}\nabla U$; the unit coefficient needs $\gamma=1$ or the time rescaling the theorem does not perform. The remark's "optimal control trajectory" is not established anywhere in the section.
- Impact on downstream results: the consistency claim with Section 21 (lines 953, 957-963).
- Fix guidance: write $\dot z = \frac{1-|z|^2}{2}\frac{z}{|z|}$; add hypotheses $\gamma_{risk}=0$, $u_\pi=0$ and time measured in units of $1/\gamma$; drop or justify "optimal control trajectory".
- Required new assumptions/permits: none.
- Validation plan: integrate the corrected field from $r_0>0$ and compare with $\tanh(\tau/2+\operatorname{artanh}r_0)$.

### [E-016] Adaptive thermodynamics: the Einstein relation does not determine $T_c(z)$; the algorithm contradicts its docstring; position-dependent $T_c$ breaks the Boltzmann claims (was F-015)
- Location: `def-einstein-relation-on-manifolds` 1168-1179; `prop-automatic-phase-transitions` 1180-1192; Algorithm 22.7.4, 1213-1233; `cor-deterministic-boundary` 1247-1258
- Severity: Moderate
- Type: Conceptual (secondary: Computational error)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 1174-1177: "$\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)}$ ... This ensures the correct equilibrium distribution."; line 1183: "With adaptive temperature $T_c(z)$ satisfying the Einstein relation"; lines 1224-1226: "T_c(z) = base_T * (1 - |z|^2)^2 / 4. This maintains constant effective noise: sigma^2 * G = 2 * gamma * T_c"; lines 1253-1256: "$T_c(z) \to 0, \qquad \text{noise} \to 0.$ The agent becomes deterministic at the boundary, ensuring reproducible outputs."
- Upstream anchor: this chapter line 227 (velocity-noise covariance $2\gamma T_cG^{-1}$) and line 1059 (Boundary phase samples $z_{tex}\sim\mathcal{N}(0,\Sigma(z))$).
- Why this is an error: (i) The relation links $\sigma,\gamma,T_c$ and does not select a function $T_c(z)$; the table's coordinate-noise profile already holds at constant $T_c$ from $\sigma^2\propto G^{-1}$. (ii) The algorithm sets $T_c\propto G^{-1}$; then $\sigma^2G = 2\gamma\,\text{base\_T}\,(1-|z|^2)^2/4$ is not constant, contradicting the docstring; in intrinsic units the FDT noise at constant $T_c$ is already constant and the algorithm makes it vanish instead. (iii) With position-dependent temperature the stationary density is no longer $e^{-\Phi_{\text{eff}}/T_c}\sqrt{|G|}$ or $e^{-H/T_c}$, so `prop-baoab-preserves-boltzmann` and `cor-fokker-planck-duality` do not cover Algorithm 22.7.4, yet line 1177 claims the correct equilibrium. (iv) "Reproducible outputs" contradicts Phase 4, where the texture is sampled stochastically at the boundary.
- Impact on downstream results: the "automatic phase transition" narrative; any implementation combining `adaptive_temperature` with the thermostat while expecting Boltzmann statistics.
- Fix guidance: present $T_c(z)$ as an explicit modelling choice with its stationary law stated; make docstring and formula agree; add "$T_c$ constant" as a hypothesis of `prop-baoab-preserves-boltzmann`; reword the corollary to "the bulk position becomes deterministic; output randomness is confined to the texture sample".
- Required new assumptions/permits: none.
- Validation plan: compute $\sigma^2G$ from the algorithm and confirm whether it is constant; compare simulated stationary histograms with and without adaptive $T_c$.

### [E-017] Summary-table "Full Geodesic SDE" row is not the Definition's SDE (was F-016)
- Location: Summary table, line 1290
- Severity: Minor
- Type: Notation conflict (secondary: Computational error)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Full Geodesic SDE | $dz = (-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi + \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z} - \Gamma(\dot{z},\dot{z}))\,ds + \sqrt{2T_c}\,G^{-1/2}\,dW_s$ | Second-order".
- Upstream anchor: this chapter lines 224-228.
- Why this is an error: The row mixes $dz$ on the left with $\dot z$-dependent terms on the right (it should be $\ddot z=\ldots$), omits $-\gamma\dot z$, has noise $\sqrt{2T_c}$ instead of $\sqrt{2\gamma T_c}$, and adds $u_\pi$ with velocity units to acceleration terms.
- Impact on downstream results: readers using the table as the reference form.
- Fix guidance: "$\ddot z = -\Gamma(\dot z,\dot z) - \gamma\dot z - G^{-1}\nabla\Phi_{\text{eff}} + \beta_{\text{curl}}G^{-1}\mathcal{F}\dot z + \gamma u_\pi + \sqrt{2\gamma T_c}\,G^{-1/2}\,\xi$" with the $u_\pi$ convention of E-004.
- Required new assumptions/permits: none.
- Validation plan: term-by-term comparison with the corrected line 227.

### [E-018] Node 27 proxy is dimensionally inconsistent and does not measure friction vs inertia (was F-017)
- Location: Node 27, lines 1326-1335
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 1328: "Is friction dominating inertia? | $\gamma / \lVert \mathcal{M}_\gamma^{-1}\,v\rVert$"; line 1330: "Here $v := \dot{z}$ and $\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F}$."
- Upstream anchor: this chapter line 899.
- Why this is an error: $\mathcal{M}_\gamma^{-1}v = \gamma v - \beta G^{-1}\mathcal{F}v$ has units $[z]/\tau^2$; the ratio has units $\tau/[z]$ and for $\beta=0$ equals $1/\|v\|$, independent of $\gamma$, so it cannot test the regime criterion of line 905 ($1/\gamma\ll$ position timescale).
- Impact on downstream results: monitoring built on Node 27.
- Fix guidance: proxy $\gamma\lVert\dot z\rVert_G/\lVert\ddot z\rVert_G$ (or $\gamma\tau_{\text{pos}}$ with $\tau_{\text{pos}} = \|\dot z\|/\|\ddot z\|$), "High = overdamped".
- Required new assumptions/permits: none.
- Validation plan: dimensional check; confirm the proxy grows with $\gamma$ at fixed trajectory.

### [E-019] Node 28 proxy is neither detailed balance nor mass balance (was F-018)
- Location: Node 28, lines 1339-1345
- Severity: Moderate
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 1341: "Are jump rates consistent with WFR? | $\lvert\sum_j \lambda_{K\to j} - \sum_i \lambda_{i\to K}\rvert$"; line 1344: "High JumpConsistencyCheck: Jump rates violate detailed balance; may cause mass accumulation/depletion."
- Upstream anchor: 05_geometry/02_wfr_geometry.md:116-122: "$\partial_s \rho + \nabla \cdot (\rho v) = \rho r$ ... $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass)"; this chapter line 370.
- Why this is an error: Detailed balance is $\rho_K\lambda_{K\to j} = \rho_j\lambda_{j\to K}$ pairwise; stationarity of chart mass is $\rho_K\sum_j\lambda_{K\to j} = \sum_i\rho_i\lambda_{i\to K}$. The proxy compares unweighted rates: it can vanish while mass accumulates and be nonzero at a balanced stationary state. With line 370, $\lambda_{K\to j}/\lambda_{j\to K} = e^{2\beta_{\text{ent}}(V_j-V_K)}$ (symmetric transport cost), so the proxy is generically nonzero even when detailed balance holds with $\rho_j/\rho_K = e^{2\beta_{\text{ent}}(V_j-V_K)}$.
- Impact on downstream results: monitoring of the jump process; Detailed Balance box (859-883).
- Fix guidance: proxy $\sum_j\lvert\rho_K\lambda_{K\to j}-\rho_j\lambda_{j\to K}\rvert$ (detailed balance) or $\lvert\rho_K\sum_j\lambda_{K\to j}-\sum_i\rho_i\lambda_{i\to K}\rvert$ (mass balance), with $\rho$ the chart occupancy.
- Required new assumptions/permits: access to chart occupancies $\rho_K$.
- Validation plan: evaluate both proxies on a two-chart example with the chapter's intensities.

### [E-020] Dangling references into this chapter from 04_dnn_blocks.md (was F-020)
- Location: (external) docs/source/1_agent/08_multiagent/04_dnn_blocks.md lines 33, 2843, 3435
- Severity: Note
- Type: Citation / reference error
- Criterion: Framework
- Origin: upstream docs/source/1_agent/08_multiagent/04_dnn_blocks.md
- Claim (verbatim): line 2843: "Section {ref}`sec-geodesic-integrator`"; lines 33 and 3435: "{ref}`sec-equations-of-motion-langevin-sdes-on-information-manifolds`".
- Upstream anchor: this chapter's labels are `sec-the-equations-of-motion-geodesic-jump-diffusion` (line 1) and `sec-the-geodesic-baoab-integrator` (line 532); a grep over docs/source/1_agent finds no definition of the referenced labels.
- Why this is an error: broken cross-references; not an error of this chapter, recorded because the fix may be placed here (alias labels) or upstream.
- Impact on downstream results: broken links in Chapter 08.
- Fix guidance: retarget the three references to the labels above.
- Required new assumptions/permits: none.
- Validation plan: rebuild the book and check for unresolved references.

## Scope restrictions and clarifications
- The Christoffel symbols of `prop-explicit-christoffel-symbols-for-poincare-disk` (lines 313-319), the exponential map `poincare_exp_map`, the O-step coefficients ($p\sim\mathcal{N}(0,T_cG)$), and the Phase 1 initialisation ($G(0)=4I$) were verified and are correct.
- The phase-space density $e^{-H/T_c}$ of the (sign-corrected) second-order system marginalises to $e^{-\Phi_{\text{eff}}/T_c}\sqrt{|G|}$, so the target density of `cor-fokker-planck-duality` is the right one; the defect (E-013) is confined to the first-order Ito equation and its proof.
- `cor-fokker-planck-duality` and `prop-baoab-preserves-boltzmann` hold, after the fixes, only for $\mathcal{F}=0$, gradient control, and constant $T_c$.
- The overdamped-limit statements need a declared time unit ($1/\gamma$) and a declared scaling of $\beta_{\text{curl}}$ with $\gamma$.

## Open questions
- Which object is the canonical control input: the tangent-vector field $u_\pi$ of Section 21.2, or a covector force? The answer determines the fixes for E-004, E-012, E-015 and E-017.
- Is the $\frac{T_c}{12}R$ term intended in the Riemannian-distance tube convention (which gives $-\frac{1}{12}R$ in the standard references) or in some other convention? The chapter should state it.
- Should $\lambda_{K\to j}$ be a symmetric-cost Metropolis-type rate with an explicit target-selection law, so that Node 28 can test a well-defined balance condition?

## Rejected candidate findings
None. All twenty stage-1 findings were confirmed on independent recomputation; one (F-014) had its criterion changed from External to Framework.
