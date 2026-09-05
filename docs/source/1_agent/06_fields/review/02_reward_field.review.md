# Mathematical Review: docs/source/1_agent/06_fields/02_reward_field.md

## Metadata
- Reviewed file: docs/source/1_agent/06_fields/02_reward_field.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1411 lines)
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/02_control_loop.md` (HJB equation, Lie derivative as metric-free pairing, unit conventions, lines 1005-1050)
  - `05_geometry/01_metric_law.md` (thm-capacity-constrained-metric-law, coupling constant $\kappa$, line 203)
  - `05_geometry/02_wfr_geometry.md` (def-the-wfr-action, unbalanced continuity equation, rem-units, lines 104-134, 188-195)
  - `05_geometry/04_equations_motion.md` (prop-mass-scaling-near-boundary, def-bulk-drift-continuous-flow, def-effective-potential, def-cognitive-temperature, lines 160-172, 220-262, 418-446, 515-525)
  - `10_appendices/01_derivations.md` (Appendix A, checked for the cited Helmholtz derivation)
  - `10_appendices/05_proofs.md` (E.8 varentropy proof, lines 818-882; E.12 HJB-Klein-Gordon derivation, lines 1060-1200)
  - `10_appendices/06_losses.md` (def-f-critic-td, line 789)
  - Downstream consumers checked: `08_multiagent/03_parameter_sieve.md`, `08_multiagent/01_gauge_theory.md`, `10_appendices/04_faq.md`, `06_fields/01_boundary_interface.md`, `06_fields/03_info_bound.md`, `intro_agent.md`, `07_cognition/07_metabolic_transducer.md`

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 10
- Minor: 6
- Notes: 0
- Primary themes:
  1. The screening-mass identification $\kappa=\lambda/c_{\text{info}}$ (natural units $\kappa=-\ln\gamma$) is not what the chapter's own Bellman expansion produces. The expansion gives a screening mass squared linear in the discount rate ($\kappa^2=\lambda/T_c$), so $\kappa\sim\sqrt{-\ln\gamma}$ and the screening length for $\gamma=0.99$ is about 10, not 100. Appendix E.12 contains the same silent jump, and `06_losses.md:789` already uses the linear convention. Many later chapters quote the chapter's numbers.
  2. The sign convention of the scalar potential is internally contradictory: $\Phi=E-T_cS$ is a free energy to be minimised, yet the canonical ensemble is $e^{+\Phi/T_c}$, the density is said to concentrate in high-$\Phi$ regions, the gradient force is $-\nabla\Phi$ "toward value peaks", and the closing SDE uses $-\partial V$ with a reward-like $V$. The upstream effective potential shares the defect.
  3. The "Boltzmann distribution emerges from WFR" corollary does not hold with the stated reaction term: $e^{\Phi/T_c}$ kills the transport flux but the reaction term $\rho(\Phi-\bar\Phi)/s_r$ is not zero pointwise. The reaction-rate theorem it rests on has no proof.
  4. Hodge-theoretic statements (harmonic sector is "topological", conservative iff curl-free, boundary pullback as source density) are stated for closed manifolds while the chapter's geometry is a truncated hyperbolic ball with boundary; the Green's-function decay rate and the "$\gamma\to1\Rightarrow\ell\to\infty$" row fail on hyperbolic space.
  5. Implementation and restatement drift: the summary SDE drops friction and the Lorentz term and changes the noise amplitude relative to its cited definition; the Hessian code computes a Euclidean Frobenius norm of the coordinate Hessian; the Node 36 and Node 61 proxies do not measure what they claim.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | The Reward 1-Form / Reward Flux, 71-78, 103-106 | Minor | Dimensional mismatch | Framework | this chapter | $r_t$ is a rate at line 74 and a per-step nat at line 106; `_G` subscript on a 1-form/vector pairing |
| E-002 | Reward Flux / Green's Function Interpretation / Three BCs, 108-111, 464-470, 1287 | Moderate | Conceptual | External | this chapter | Pullback $\iota^*d\Phi$ is Dirichlet-type tangential data, not a Neumann flux or source density |
| E-003 | Terminal Boundary, Conformal Coupling, WFR theorem, 122, 340, 546, 672-689, 807 | Minor | Notation conflict | Framework | this chapter | $\kappa$ (metric-law coupling vs screening mass) and $r$ (reward vs WFR reaction rate) overloaded |
| E-004 | Hodge Decomposition, 179-208 | Moderate | Scope restriction | External | this chapter | Harmonic sector is infinite-dimensional on a manifold with boundary; "topological" reading needs the Neumann/Dirichlet piece |
| E-005 | Conservative Reward Field / Cycle Detection / Connection #31, 233-276 | Moderate | Invalid inference | Framework | this chapter | "Conservative iff curl-free" conflates closed and exact; Stokes identity written without $\gamma=\partial\Sigma$ |
| E-006 | HJB-Helmholtz Correspondence and dependents, 309-312, 323-363, 375-393, 401-444, 521-548 | Major | Invalid inference | Framework | this chapter | Derivation yields $\kappa^2=\lambda/T_c$ (linear in $-\ln\gamma$); chapter asserts $\kappa=\lambda/c_{\text{info}}$; all numbers off by $\sqrt{\cdot}$ |
| E-007 | HJB-Helmholtz proof sketch, 344-361 | Moderate | Proof gap / omission | Framework | this chapter | Source $r=\mathcal{R}_ib^i$ vanishes when $b=0$; $\nabla_A$ not produced by the expansion |
| E-008 | HJB-Helmholtz proof sketch, 355-360 | Minor | Typo | Framework | this chapter | Prose and nested `$` inside the `$$` block; definition of $\nabla_A$ does not render |
| E-009 | HJB-Helmholtz proof sketch, 361 | Minor | Citation / reference error | Framework | this chapter | "Details in Appendix A" points to a section with no such derivation |
| E-010 | Green's Function isomorphism / decay / Table 24.2.5 / Connection #30 / Node 36, 484, 509-520, 541, 560-566, 626-636 | Moderate | Scope restriction | External | this chapter | Flat-space Bessel asymptotics stated for "bounded curvature"; on $\mathbb{H}^d$ the rate is $\ge d-1$ and never vanishes |
| E-011 | Thermodynamic Interpretation, EM isomorphism, RL-as-Electrodynamics, 141, 163, 589-640, 692-703, 1264 | Major | Definition mismatch | Framework | this chapter and upstream `05_geometry/04_equations_motion.md` | Free-energy axiom vs reward-convention ensemble and forces; SDE drives to value minima |
| E-012 | WFR Consistency: Value Creates Mass, 669-691 | Moderate | Proof gap / omission | Framework | this chapter | Proof restates the claim; WFR action has no $V$; $\lambda$ collision; $[s_r]$ wrong |
| E-013 | Conservative Equilibrium Distribution, 692-705 | Major | Invalid inference | Framework | this chapter | $e^{\Phi/T_c}$ is not stationary under the unbalanced continuity equation with $r\propto\Phi-\bar\Phi$ |
| E-014 | Varentropy-Stability Relation, 757-791 | Minor | Dimensional mismatch | Framework | this chapter (upstream typo at `05_proofs.md:823`) | Adiabatic condition lacks a rate; E.8 proves only the identity |
| E-015 | Conformal Laplacian Transformation, 880-903 | Moderate | Invalid inference | Framework | this chapter | $\tilde\kappa^2=\Omega^{-2}\kappa^2$ is a rewriting of the same equation; no self-focusing follows |
| E-016 | HolographicCritic implementation / Conformal Coupling definition, 809-828, 1092-1141 | Moderate | Algorithm mismatch | Framework | this chapter | Euclidean Frobenius norm of coordinate Hessian instead of $G$-operator norm of covariant Hessian; `self.kappa` unused |
| E-017 | RL as Electrodynamics on a Curved Manifold, 1248-1279 | Moderate | Definition mismatch | Framework | this chapter (upstream sign issue at `04_equations_motion.md:227`) | SDE restated without friction/Lorentz term and with noise $\sqrt{2T_c}$; cited definition differs |
| E-018 | Node 36 / Table 24.8.1, 1326-1335, 1404 | Moderate | Conceptual | Framework | this chapter | Proxy $\lvert V(z)-V(z')\rvert e^{\kappa d_G}$ grows with separation; cannot be $\approx1$ |
| E-019 | Node 61, 1371-1398 | Minor | Algorithm mismatch | Framework | this chapter | Signed circulation equated with norm of curl; code sums rewards, not TD errors; tests exactness not curl |

## Detailed findings

### [E-001] Reward-rate units and pairing notation (was F-001)
- Location: The Reward 1-Form (lines 71-78), The Reward Flux (lines 103-106)
- Severity: Minor
- Type: Dimensional mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim: Line 74: "$r_t = \langle \mathcal{R}(z), v \rangle_G = \mathcal{R}_i(z) \dot{z}^i$"; line 78: "$[\mathcal{R}] = \mathrm{nat}/[\text{length}]$"; line 106: "$[J_r] = \mathrm{nat}/[\text{length}]$, $[r_t] = \mathrm{nat}$."
- Upstream anchor: `01_foundations/02_control_loop.md:1030-1035`: "$\mathcal{L}_f V = dV(f) = \partial_i V \cdot f^i = \nabla V \cdot f$ ... This is the natural pairing between the 1-form $dV$ and the vector field $f$---NO metric $G$ appears." Line 1043: "$[\nabla V \cdot f] = \frac{[V]}{[z]} \cdot \frac{[z]}{[t]} = \mathrm{nat}\,\mathrm{step}^{-1}$".
- Why this is an error: $\mathcal{R}_i\dot z^i$ with the stated units is nat/time (a rate), which is what "reward rate" and the anchor say; line 106 assigns nat to the same symbol. Line 350 uses $r\,\Delta t$ (rate) while line 329 uses $r$ as a per-step nat. The `_G` subscript puts a metric on a pairing that has none.
- Impact on downstream results: Units of $\rho_r$ (line 363) and of the Helmholtz source inherit the ambiguity (see E-006, E-007).
- Fix guidance: (1) Write $r_t=\mathcal{R}(v)=\mathcal{R}_i\dot z^i$ without the subscript. (2) State $[r]=\mathrm{nat}/\text{time}$ for the rate and $r_t:=r\,\Delta t$ (nat) for discrete samples. (3) Correct line 106 accordingly.
- Required new assumptions/permits: none.
- Validation plan: check that every occurrence of $r$ in lines 329, 350, 356 carries the same units after the edit.

### [E-002] Pullback boundary data is Dirichlet-type, not source data (was F-002)
- Location: The Reward Flux (lines 108-111), Green's Function Interpretation (lines 464-470), The Three Boundary Conditions (line 1287)
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: External
- Origin: this chapter
- Claim: Lines 109-111: "In the conservative case ($\mathcal{R}=d\Phi$), boundary reward reduces to Dirichlet/Neumann data for $\Phi$ (equivalently a boundary source density $\sigma_r$)." Lines 467-468: "$V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z')$".
- Upstream anchor: not applicable; defined in this chapter.
- Why this is an error: Pullback commutes with $d$, so $\iota^*\mathcal{R}=d(\Phi|_{\partial\Omega})$. This fixes only tangential derivatives of the boundary trace, i.e. Dirichlet data up to one additive constant per boundary component. It carries no normal-derivative information. A single-layer potential $\int_{\partial\Omega}G_\kappa\sigma_r$ is characterised by a jump in the normal derivative $[\partial_nV]=-\sigma_r$, which is Neumann-type data. Tangential 1-form, Dirichlet trace and surface charge are three different objects, and prescribing more than one over-determines a second-order elliptic problem.
- Impact on downstream results: Corollary cor-the-three-boundary-conditions item 3 ("Inject charge $\sigma_r$ at boundary"), the Holographic Dictionary row "Reward: Source (Poisson)", and `08_multiagent/01_gauge_theory.md` (thm-hjb-klein-gordon imports $\rho_r$ "derived from boundary reward flux").
- Fix guidance: (1) Choose one: (a) $\iota^*\mathcal{R}=J_r$ prescribes $\Phi|_{\partial\Omega}$ up to a constant (Dirichlet), drop "source density"; or (b) define $\sigma_r$ independently as a Neumann flux $\partial_n\Phi|_{\partial\Omega}=\sigma_r$ and state that the pullback is not used. (2) Update line 1287 and prop-green-s-function-interpretation to the chosen data type.
- Required new assumptions/permits: if (b), an explicit definition of $\sigma_r$ from environment data.
- Validation plan: verify that the Green's representation used at 459 matches the chosen boundary datum (Dirichlet: double-layer/Green's second identity; Neumann: single-layer).

### [E-003] Symbol collisions: $\kappa$ and $r$ (was F-019)
- Location: lines 122, 340, 546, 672-689, 807, 1267
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim: Line 807: "From Theorem thm-capacity-constrained-metric-law, the curvature is driven by the Risk Tensor" alongside line 340: "$\kappa^2$ is the 'mass' of the scalar field". Line 122: "include a killing rate $\kappa_{\text{term}}(z) \ge 0$ or a reaction term $r<0$" alongside reward $r_t$ (line 74) and "$r(z) = \frac{1}{s_r}(V(z) - \bar{V})$" (line 675).
- Upstream anchor: `05_geometry/01_metric_law.md:203`: "$\kappa$: a coupling constant. It controls how strongly the risk tensor sources curvature." `05_geometry/02_wfr_geometry.md:122`: "$r_s(z) \in \mathbb{R}$ is the reaction rate".
- Why this is an error: The chapter invokes the metric-law theorem (coupling $\kappa$) in the sections where $\kappa$ is the screening mass, and uses $r$ for the reward and for the WFR reaction rate inside one theorem block (669-689) and across the Bellman derivation (329, 350). Neither overload is flagged.
- Impact on downstream results: readability of cross-references to 05_geometry; line 122 is ambiguous.
- Fix guidance: (1) Write $\kappa_{\text{EH}}$ (or $8\pi\ell_L^2$ per Appendix A) when citing the metric law. (2) Use $r_{\text{WFR}}$ or $\varrho$ for the reaction rate in this chapter.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for `\kappa` and `r(z)` after the edit.

### [E-004] Harmonic sector is not "topological" on the chapter's own geometry (was F-003)
- Location: Hodge Decomposition of the Reward Field (lines 179-208, esp. 182-183, 192, 200-206)
- Severity: Moderate
- Type: Scope restriction (secondary: External dependency)
- Criterion: External
- Origin: this chapter
- Claim: Lines 182-183: "On a compact latent Riemannian manifold $(\mathcal{Z}, G)$ with boundary (or on a complete manifold with suitable decay and boundary conditions)"; line 192: "$\eta \in \mathcal{H}^1(\mathcal{Z})$ (Harmonic Flux): Topological cycles from manifold holes. Satisfies $d\eta = 0$ and $\delta\eta = 0$."
- Upstream anchor: `05_geometry/04_equations_motion.md:166-168`: "For the Poincare disk, the mass tensor scales as $\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d$"; this chapter line 128: "we truncate the hyperbolic disk at $|z| = 1-\varepsilon$"; line 861: "Default hyperbolic bulk geometry".
- Why this is an error: On a compact manifold with boundary the space $\{d\eta=0,\ \delta\eta=0\}$ is infinite-dimensional. The Hodge-Morrey-Friedrichs decomposition splits it further, and only the Neumann (or Dirichlet) harmonic fields are isomorphic to cohomology and hence "topological". The proof sketch names absolute/relative boundary conditions (line 201) but the theorem statement does not impose them, and the Green's operator formula $\Phi=\delta G\mathcal{R}$, $\Psi=dG\mathcal{R}$ is the closed-manifold formula. For the truncated ball, $H^1=0$, so a hole-induced $\eta$ never exists; any non-zero harmonic field is boundary-driven. (The verifier notes that the reviewer's additional remark about infinite-dimensional $L^2$ harmonic 1-forms on the complete hyperbolic space holds for $d=2$ only; the load-bearing point is the boundary case.)
- Impact on downstream results: def-conservative-reward-field (E-005), Node 61 "Topology" interpretation, the Researcher Bridge (line 50).
- Fix guidance: (1) State the decomposition as Hodge-Morrey-Friedrichs with an explicit boundary condition, e.g. $\mathcal{R}=d\Phi+\delta\Psi_N+\eta$ with $\Phi|_{\partial}$ fixed, $\Psi$ satisfying the normal condition, $\eta\in\mathcal{H}^1_N$. (2) Restrict "topological" to $\mathcal{H}^1_N\cong H^1(\mathcal{Z};\mathbb{R})$. (3) For the truncated ball note $H^1=0$, so $\eta=0$ under the stated condition.
- Required new assumptions/permits: an explicit boundary condition for the decomposition.
- Validation plan: check that line 50 ("harmonic component is fixed to zero by boundary conditions") and line 192 agree after the edit.

### [E-005] "Conservative iff curl-free" conflates closed and exact (was F-004)
- Location: Conservative Reward Field (lines 233-246), Value Cycle Detection (255-260), Connection #31 (273-276)
- Severity: Moderate
- Type: Invalid inference (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: Lines 233-239: "conservative if and only if $\mathcal{F} = d\mathcal{R} = 0$ ... Equivalently, $\mathcal{R} = d\Phi$ for some scalar potential $\Phi$ (the solenoidal and harmonic components vanish)." Line 244: "$\oint_\gamma \mathcal{R} = \int_\Sigma d\mathcal{R} = \int_\Sigma \mathcal{F} = 0$."
- Upstream anchor: contradicts this chapter's line 50: "scalar value as the special case where $\mathcal{R}$ is exact (curl vanishes and the harmonic component is fixed to zero by boundary conditions)".
- Why this is an error: $d\mathcal{R}=0$ removes the coexact part but $\eta$ is closed by definition (line 192), so $\mathcal{F}=0$ gives $\mathcal{R}=d\Phi+\eta$, not $d\Phi$. The identity $\oint_\gamma\mathcal{R}=\int_\Sigma\mathcal{F}$ requires $\gamma=\partial\Sigma$; for a non-bounding loop $\oint_\gamma\eta\neq0$ with $\mathcal{F}\equiv0$, so "$\mathcal{F}=0\Rightarrow$ zero circulation around any closed loop" (241) and the converse diagnostic (258) are both invalid in general.
- Impact on downstream results: Node 61, Connection #31, the Researcher Bridge; upstream def-bulk-drift-continuous-flow cites def-conservative-reward-field for its "Conservative Limit".
- Fix guidance: (1) Define conservative as $\mathcal{R}$ exact. (2) State the equivalence as: exact iff $\mathcal{F}=0$ and $\oint_\gamma\mathcal{R}=0$ for all closed loops (iff $\mathcal{F}=0$ when $H^1(\mathcal{Z})=0$ and the harmonic boundary condition of E-004 is imposed). (3) Write the Stokes identity with the hypothesis $\gamma=\partial\Sigma$.
- Required new assumptions/permits: the boundary condition from E-004.
- Validation plan: confirm line 50, 239 and 241 make the same statement.

### [E-006] Screening mass: the derivation yields $\kappa^2\propto-\ln\gamma$, not $\kappa=-\ln\gamma$ (was F-005)
- Location: feynman prose (309-312), The HJB-Helmholtz Correspondence (323-363), rem-helmholtz-dimensions (375-393), Yukawa isomorphism (401-406), Connection #4 (432-444), cor-discount-as-screening-length and Table 24.2.5 (521-548)
- Severity: Major
- Type: Invalid inference (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter (the same gap is reproduced in `10_appendices/05_proofs.md` E.12; `10_appendices/06_losses.md:789` uses the other convention)
- Claim: Line 326: "Let the temporal discount rate be $\lambda := -\ln\gamma / \Delta t$ and define the spatial screening mass $\kappa := \lambda / c_{\text{info}}$"; line 335: "$-\Delta_G V(z) + \kappa^2 V(z) = \rho_r(z)$"; line 350: "$V(z) = r \Delta t + \gamma \mathbb{E}[V(z')] \approx r \Delta t + (1 - \kappa \Delta t)\left(V + \nabla_A V \cdot b \Delta t + T_c \Delta_G V \Delta t\right)$"; line 356: "$\kappa V = r + \nabla_A V \cdot b + T_c \Delta_G V$"; line 361: "For the stationary case ($b = 0$) and absorbing the temperature into the source term, this yields the Helmholtz equation $-\Delta_G V + \kappa^2 V = \rho_r$." Line 542: "$\gamma = 0.99$ | $\kappa \approx 0.01$ | $\ell \approx 100$".
- Upstream anchor: `10_appendices/05_proofs.md:1118-1126`: "For stationary states ($\partial_t V = 0$) with zero drift ($b = 0$): $-T_c \Delta_G V + \kappa_t V = r$. This is the Helmholtz equation ... Note that $\kappa_t$ here has temporal units." Line 1145: "$\kappa := \kappa_t / c_{\text{info}}$"; line 1163: "$\ldots = r - \kappa^2 V + \Delta_G V$" with no connecting algebra. `10_appendices/06_losses.md:789`: "$\kappa^2 = -\ln \gamma$ – screening mass from discount factor". `10_appendices/01_derivations.md` (the appendix cited at line 361) contains no Helmholtz or Bellman derivation.
- Why this is an error: Redoing the expansion at line 350 with $\gamma=e^{-\lambda\Delta t}\approx1-\lambda\Delta t$: $V=r\Delta t+V+\nabla V\cdot b\,\Delta t+T_c\Delta_GV\,\Delta t-\lambda V\Delta t+O(\Delta t^2)$. Cancelling $V$ and dividing by $\Delta t$: $\lambda V=r+\nabla V\cdot b+T_c\Delta_GV$. With $b=0$: $-\Delta_GV+(\lambda/T_c)V=r/T_c$. The coefficient of $V$ is $\lambda/T_c$, the first power of the discount rate divided by the diffusion constant, so the PDE's own screening mass is $\kappa_{\text{PDE}}=\sqrt{\lambda/T_c}$. The identification $\kappa^2=\lambda^2/c_{\text{info}}^2$ is asserted at line 326 and never connected to this coefficient; "absorbing the temperature into the source term" rescales the right-hand side and cannot change the power of $\lambda$. (Line 350 also writes $(1-\kappa\Delta t)$ with the spatial $\kappa$ where the temporal $\lambda$ belongs; $\kappa\Delta t$ has units time/length.) The lattice limit at lines 436-442 confirms the linear scaling independently: $(I-\gamma P)=(1-\gamma)I+\gamma(I-P)=(1-\gamma)I+\gamma L_{\text{graph}}$, so the discrete mass term is $(1-\gamma)/\gamma\approx-\ln\gamma$ against a unit-coefficient Laplacian. Numerically: $\gamma=0.99$: $-\ln\gamma=0.01005$, chapter $\ell=99.5$, derived $\kappa=0.1003$, $\ell=9.97$ (lattice $\sqrt{0.01/0.99}=0.1005$); $\gamma=0.9$: chapter $\ell=9.49$, derived $\ell=3.08$. The factor $\sqrt{\cdot}$ is the diffusive scaling: 100 steps of a random walk cover about 10 lattice spacings. The conversion $\kappa=\lambda/c_{\text{info}}$ is a ballistic conversion appropriate to a first-order transport or wave operator, not to the second-order diffusion expansion actually performed. Dimensionally, with upstream $[T_c]=\mathrm{nat}$ (`04_equations_motion.md:242`), $[T_c\Delta_GV]=\mathrm{nat}^2/\text{length}^2$ while $[\lambda V]=\mathrm{nat}/\text{time}$, so line 356 is itself inconsistent unless $T_c$ carries a hidden mobility $\text{length}^2/(\text{time}\cdot\mathrm{nat})$.
- Impact on downstream results: cor-discount-as-screening-length and Table 24.2.5; feynman prose line 312; `CriticConfig.screening_mass` (973-975); Node 36's $e^{\kappa d_G}$; `08_multiagent/03_parameter_sieve.md:96-103, 860, 883, 922-935` ("$\ell_\gamma\approx100\,\ell_0$"); `08_multiagent/01_gauge_theory.md:588`; `10_appendices/04_faq.md:352` (Prediction 3, falsification test at rate $\kappa=-\ln\gamma$); `06_fields/01_boundary_interface.md:717`; `06_fields/03_info_bound.md:500`; `intro_agent.md:330-331, 367`; `07_cognition/07_metabolic_transducer.md:733`. `06_losses.md:789` disagrees with this chapter.
- Fix guidance: Either (a) keep the diffusive derivation: set $\kappa^2:=\lambda/D$ with $D=T_c\mu$ the dimensionful latent diffusion constant, $\ell=\sqrt{D/\lambda}=\sqrt{D\Delta t/(-\ln\gamma)}$, replace $(1-\kappa\Delta t)$ by $(1-\lambda\Delta t)$ at line 350, and recompute every table and number ($\gamma=0.99$, $D\Delta t=1$: $\ell\approx10$); or (b) derive the Helmholtz equation from a first-order transport (speed $c_{\text{info}}$) or Klein-Gordon static limit, in which case $\kappa=\lambda/c_{\text{info}}$ is legitimate but the proof sketch at 344-361 and E.12 Steps 5-7 must be replaced. In both cases reconcile `06_losses.md:789` and the downstream list above.
- Required new assumptions/permits: for (a), a definition of the latent diffusion constant $D$ with units $\text{length}^2/\text{time}$; for (b), a first-order or hyperbolic value-propagation postulate.
- Validation plan: after the fix, the coefficient of $V$ obtained by expanding line 350 must equal the $\kappa^2$ used in line 335; check the $\gamma=0.99$ row against the lattice value $\sqrt{(1-\gamma)/\gamma}$ (option a) or against the stated $c_{\text{info}}$ (option b); rerun the grep list for downstream numbers.

### [E-007] Source term vanishes in the "stationary" limit; $\nabla_A$ unmotivated (was F-006)
- Location: HJB-Helmholtz proof sketch (lines 344-361)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: Line 347: "with instantaneous reward rate $r := \mathcal{R}_i(z) b^i(z,a)$"; line 361: "For the stationary case ($b = 0$) ... this yields the Helmholtz equation $-\Delta_G V + \kappa^2 V = \rho_r$."
- Upstream anchor: this chapter line 87: "A stationary agent ($\dot{z} = 0$) receives zero instantaneous reward."
- Why this is an error: With $r=\mathcal{R}_ib^i$, setting $b=0$ forces $r=0$, so the derived equation is homogeneous and its bounded solution with homogeneous data is $V=0$. The non-zero $\rho_r$ is inserted by hand. Either the reward needs a velocity-independent scalar density (contradicting def-reward-1-form and lines 61-63, 87) or the drift must be kept, giving an advection-diffusion equation. Separately, a second-order Taylor expansion of $V(z+dz)$ produces $\nabla V\cdot b$, not $\nabla_AV\cdot b$; the replacement at line 357 is asserted without derivation, and with $V=\Phi$, $r+\nabla_AV\cdot b=(\nabla\Phi+A)\cdot b+(\nabla\Phi-A)\cdot b=2\nabla\Phi\cdot b$, which double-counts the gradient and cancels $A$.
- Impact on downstream results: prop-green-s-function-interpretation, HolographicCritic, Nodes 35-36, thm-hjb-klein-gordon.
- Fix guidance: (1) Introduce a scalar reward density $\rho_r(z)$ explicitly as bulk/boundary data distinct from the 1-form rate. (2) State the Bellman condition as $V=\rho_r\Delta t+\gamma\mathbb{E}V(z')$. (3) Drop $\nabla_A$ from the expansion, or derive it from a minimal-coupling argument with explicit bookkeeping.
- Required new assumptions/permits: a definition of $\rho_r$ as part of the environment data.
- Validation plan: verify the $b=0$ limit of the corrected expansion is non-trivial and matches line 335.

### [E-008] Prose inside the display-math block of the key equation (was F-008)
- Location: HJB-Helmholtz proof sketch (lines 355-360)
- Severity: Minor
- Type: Typo
- Criterion: Framework
- Origin: this chapter
- Claim: Lines 355-360: "$$ \kappa V = r + \nabla_A V \cdot b + T_c \Delta_G V. Here $\nabla_A V := \nabla V - A$ with $A := \delta\Psi + \eta$ the non-conservative component of $\mathcal{R}$ (conservative case: $A=0$). $$"
- Upstream anchor: not applicable.
- Why this is an error: The sentence and nested `$` delimiters sit between the `$$` fences, so the only definition of $\nabla_A$ in the chapter is emitted as invalid LaTeX and does not render.
- Impact on downstream results: readability of the central theorem.
- Fix guidance: close the `$$` after "$T_c \Delta_G V.$" and move the sentence to prose.
- Required new assumptions/permits: none.
- Validation plan: build the page and check the rendered block.

### [E-009] "Details in Appendix A" points to a section without the derivation (was F-007)
- Location: HJB-Helmholtz proof sketch (line 361)
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim: Line 361: "Details in {ref}`sec-appendix-a-full-derivations`."
- Upstream anchor: `10_appendices/01_derivations.md`: sections A.1-A.6 (metric law variation, pitchfork, overdamped limit, classification as relaxation, area law); no occurrence of "Helmholtz", "screen" or "Bellman". The only derivation is `10_appendices/05_proofs.md:1060-1195` (E.12).
- Why this is an error: the label resolves, but the promised content is absent; the reader is sent to the wrong appendix.
- Impact on downstream results: none beyond E-006/E-007.
- Fix guidance: point to E.12 (or add the derivation to Appendix A).
- Required new assumptions/permits: none.
- Validation plan: follow the link.

### [E-010] Green's-function decay and "$\gamma\to1$ gives infinite range" fail on hyperbolic space (was F-009)
- Location: Green's Function isomorphism (line 484), prop-green-s-function-decay (509-520), Table 24.2.5 row "$\gamma\to1$" (541), Connection #30 (560-566), Node 36 (626-636)
- Severity: Moderate
- Type: Scope restriction (secondary: External dependency)
- Criterion: External
- Origin: this chapter
- Claim: Lines 512-515: "On a manifold with bounded curvature, the Green's function decays exponentially: $G_\kappa(z,z') \sim \frac{1}{d_G(z,z')^{(d-1)/2}} \exp(-\kappa\, d_G(z,z'))$"; line 541: "$\gamma \to 1$ | $\kappa \to 0$ | $\ell \to \infty$ | Infinite horizon (massless field)".
- Upstream anchor: `05_geometry/04_equations_motion.md:166-168` (Poincaré metric $\frac{4}{(1-|z|^2)^2}I_d$, curvature $-1$); this chapter lines 128, 861.
- Why this is an error: The stated asymptotics is the flat-space Bessel-$K$ form. On $\mathbb{H}^d$ the radial Green's equation $f''+(d-1)\coth r\,f'=\kappa^2f$ has $f\sim e^{-sr}$ with $s^2-(d-1)s-\kappa^2=0$, so $s=\frac{d-1}{2}+\sqrt{\frac{(d-1)^2}{4}+\kappa^2}$. Check: $d=3$, $\kappa=0$ gives $G_0=e^{-r}/(4\pi\sinh r)\sim e^{-2r}$, $s=2$. Consequences: for $d=2$, $\gamma=0.99$, $\kappa=0.01$: $s=1.0001$, so the screening length is about one curvature radius, not 100; $\gamma\to1$ gives $\ell\to1/(d-1)$, not $\infty$; Node 36's $e^{\kappa d_G}$ rescaling does not yield a constant even for a converged critic. "Bounded curvature" does not suffice; negative curvature adds exponential volume growth to the decay.
- Impact on downstream results: cor-discount-as-screening-length, Connection #30, Node 36, `10_appendices/04_faq.md:352`, `08_multiagent/03_parameter_sieve.md:922-935`.
- Fix guidance: (1) Restrict the proposition to (asymptotically) flat $G$, or state the hyperbolic rate $s(\kappa)$. (2) Replace the "$\gamma\to1\Rightarrow\ell\to\infty$" row by the curvature-limited horizon. (3) Rescale Node 36 by $e^{s\,d_G}$ or by an empirically fitted rate.
- Required new assumptions/permits: a stated curvature regime for the decay proposition.
- Validation plan: numerically solve the radial Green's equation on the Poincaré disk for a few $\kappa$ and compare the fitted rate to $s(\kappa)$.

### [E-011] Sign of the scalar potential: free-energy axiom vs reward-convention ensemble and forces (was F-010)
- Location: Thermodynamic Interpretation (lines 589-640, 692-703), Electromagnetism isomorphism (141), Hodge prose (163), RL-as-Electrodynamics SDE (1264)
- Severity: Major
- Type: Definition mismatch (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter and upstream `05_geometry/04_equations_motion.md` (def-effective-potential)
- Claim: Lines 612-617: "$\Phi(z) = E(z) - T_c S(z)$, where $E(z)$ is the task risk/cost ... $S(z)$ is the exploration entropy"; line 622: "the scalar potential $\Phi$ is the complete value function $V(z)$"; lines 633-638: "$P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V(z)}{T_c}\right)$ ... Throughout this document we use the Reward convention"; lines 698-703: "$\rho_\infty(z) \propto \exp(\Phi(z)/T_c)$ ... concentrates in high-$\Phi$ regions"; line 141: "Gradient force: $-\nabla\Phi$ (climb toward value peaks)"; line 1264: "$dp_k = -\frac{\partial V}{\partial q^k} ds \ldots$".
- Upstream anchor: `04_equations_motion.md:227, 258`: "$dp_k = [-\partial_k \Phi_{\text{eff}} - \gamma\, p_k + \ldots]ds$ ... samples the Boltzmann distribution $\rho(z,p) \propto \exp(-H(z,p)/T_c)$"; `:421-446`: "$\Phi_{\text{eff}} = \alpha U + (1-\alpha) V_{\text{critic}} + \gamma_{risk}\Psi_{\text{risk}}$ ... The value term pulls it toward high-reward regions ... The agent rolls downhill in this combined landscape."
- Why this is an error: With $E$ a cost and $-T_cS$ rewarding entropy, $\Phi=E-T_cS$ is a free energy to be minimised; the equilibrium weight is $e^{-\Phi/T_c}$ and the force $-\nabla\Phi$ is correct. But the chapter simultaneously identifies $\Phi=V$ in the reward convention (194, 622, 638), uses $e^{+V/T_c}$ and $e^{+\Phi/T_c}$ (633, 651, 698) with density concentrating in high-$\Phi$ regions, and says the gradient part points from low to high value (163, 170). Under that reading, maximising $\Phi=E-T_cS$ maximises cost and penalises entropy, the opposite of MaxEnt RL which the same section claims to recover (599, 666). The SDE force $-\partial V/\partial q$ with reward-like $V$ drives the agent to value minima, contradicting the stationary law $e^{+V/T_c}$. Upstream $\Phi_{\text{eff}}$ has the same defect ($+V_{\text{critic}}$ inside an energy with force $-\nabla\Phi_{\text{eff}}$ while claiming to pull toward high reward). Terminology aside, "Gibbs free energy $F=E-TS$" at line 592 is the Helmholtz free energy; not load-bearing.
- Impact on downstream results: def-canonical-ensemble, cor-equilibrium-distribution, the reaction sign in thm-wfr-consistency-value-creates-mass, Node 37 (KL to Boltzmann), Node 39 (corr$(m,V)>0.5$), thm-rl-as-electrodynamics-on-a-curved-manifold, and the upstream Langevin thermostat statement.
- Fix guidance: (1) Fix one convention and apply it everywhere: reward $V$ (maximise), free energy $\mathcal{F}:=-V=E-T_cS$ (minimise), Boltzmann $\rho\propto e^{-\mathcal{F}/T_c}=e^{+V/T_c}$, force $+\nabla V=-\nabla\mathcal{F}$. (2) Change line 141 to $+\nabla\Phi$ or redefine $\Phi:=-V$. (3) Change the SDE at 1264 to $+\partial_kV$ (and upstream $\Phi_{\text{eff}}$ to $-(1-\alpha)V_{\text{critic}}$). (4) Rewrite the axiom as $-\Phi=E-T_cS$ or rename $\Phi$.
- Required new assumptions/permits: none; a convention choice.
- Validation plan: for a one-dimensional quadratic $V$, simulate the chosen SDE and compare the empirical stationary density to the chosen Boltzmann law.

### [E-012] "Value creates mass" theorem has no proof and overloads $\lambda$ (was F-011)
- Location: WFR Consistency: Value Creates Mass (lines 669-691)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim: Line 675: "$r(z) = \frac{1}{s_r}(V(z) - \bar{V})$"; line 689: "The WFR optimal reaction rate minimizes $\int\lambda^2 r^2\,d\rho$ subject to the constraint that the endpoint marginals match. The solution is $r \propto (V - \bar{V})$, where $V$ appears because it determines the target stationary distribution."
- Upstream anchor: `05_geometry/02_wfr_geometry.md:106-132`: "$\mathcal{E}[\rho,v,r] = \int_0^1\int_{\mathcal{Z}}(\|v_s\|_G^2 + \lambda^2|r_s|^2 - 2\langle\mathbf{A},v_s\rangle)\,d\rho_s\,ds$ subject to $\partial_s\rho + \nabla\cdot(\rho v) = \rho r$ ... $\lambda > 0$ is the length-scale parameter"; `:191-193`: "$[r] = 1/\text{time}$, and $[\lambda] = \text{length}$". No statement relating $r$ to $V$ exists in that chapter.
- Why this is an error: Minimising the WFR action between two fixed marginals gives the WFR geodesic, whose reaction rate is the dual Hamilton-Jacobi potential ($r=\varphi/\lambda^2$); $V$ enters only if the target marginal is chosen as a function of $V$, and the theorem never specifies which target nor why the result is affine in $V$ with the mean subtracted. The proof restates the conclusion. Secondary: $\lambda$ (WFR length scale) collides with $\lambda:=-\ln\gamma/\Delta t$ (lines 326, 530); with $[r]=1/\text{time}$ and $[V]=\mathrm{nat}$, $s_r$ has units $\mathrm{nat}\cdot\text{time}$, not "time scale".
- Impact on downstream results: cor-equilibrium-distribution (E-013), `compute_wfr_reaction_rate`, Node 39.
- Fix guidance: (1) Demote to a modelling definition ("we choose $r:=(V-\bar V)/s_r$") with mass conservation $\int r\rho=0$ as justification, or supply a variational derivation with an explicit target law. (2) Rename the WFR length scale in this chapter (e.g. $\lambda_{\mathrm{WFR}}$). (3) Fix the units of $s_r$.
- Required new assumptions/permits: if kept as a theorem, an explicit target marginal.
- Validation plan: check that the definition satisfies $\int\rho r\,d\mu_G=0$ for normalised $\rho$.

### [E-013] $e^{\Phi/T_c}$ is not stationary under WFR with the stated reaction rate (was F-012)
- Location: Conservative Equilibrium Distribution (lines 692-705)
- Severity: Major
- Type: Invalid inference
- Criterion: Framework
- Origin: this chapter
- Claim: Lines 695-703: "At equilibrium ($\partial_s\rho = 0$), the WFR dynamics with reaction rate $r(z) \propto (\Phi(z) - \bar{\Phi})$ converge to the Boltzmann distribution: $\rho_\infty(z) \propto \exp(\Phi(z)/T_c)$, which is exactly the canonical ensemble."
- Upstream anchor: `05_geometry/02_wfr_geometry.md:114-117`: "subject to the Unbalanced Continuity Equation: $\partial_s\rho + \nabla\cdot(\rho v) = \rho r$".
- Why this is an error: The chapter's own current (lines 713, 736) is $J=\rho v-D\nabla\rho$ with $J_{\text{gradient}}=-D\rho\nabla\ln\rho+\rho\nabla\Phi$, i.e. $v=\nabla\Phi$. For $\rho\propto e^{\Phi/T_c}$, $\nabla\ln\rho=\nabla\Phi/T_c$ and $J=\rho\nabla\Phi(1-D/T_c)=0$ when $D=T_c$; then $\partial_s\rho=\rho\,r=\rho(\Phi-\bar\Phi)/s_r$, which vanishes only where $\Phi=\bar\Phi$. Total mass is conserved but the shape is not stationary: the reaction term keeps pumping mass into high-$\Phi$ regions. The true stationary state of $\partial_s\rho=\nabla\cdot(T_c\nabla\rho-\rho\nabla\Phi)+\rho(\Phi-\bar\Phi)/s_r$ is the principal eigenfunction of a Schrödinger-type operator, sharper than the Gibbs measure and dependent on $s_r$; the Gibbs measure is recovered only as $s_r\to\infty$. The remark "zero probability current" is correct for the transport part but ignores the reaction flux.
- Impact on downstream results: Node 37, Node 39, the Thermodynamic-RL dictionary row "Boltzmann distribution | Conservative solution", the prose at 599/666 that MaxEnt "emerges" from the dynamics.
- Fix guidance: Either (a) set $r\equiv0$ in the conservative-equilibrium corollary (pure Langevin transport, then $e^{\Phi/T_c}$ is stationary subject to E-011's sign) and treat the reaction term as a transient selection mechanism; or (b) state the actual stationary condition $T_c\Delta_G\rho-\nabla\cdot(\rho\nabla\Phi)+\rho(\Phi-\bar\Phi)/s_r=0$ and characterise $\rho_\infty$ as its ground state, with $e^{\Phi/T_c}$ as the $s_r\to\infty$ limit.
- Required new assumptions/permits: for (b), existence of the ground state (compact domain suffices).
- Validation plan: one-dimensional numerical integration of the unbalanced Fokker-Planck equation with quadratic $\Phi$; compare the long-time profile to $e^{\Phi/T_c}$ for finite and large $s_r$.

### [E-014] Adiabatic annealing condition lacks a rate; appendix proves only the identity (was F-013)
- Location: Varentropy-Stability Relation (lines 757-791)
- Severity: Minor
- Type: Dimensional mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (upstream typo at `05_proofs.md:823`)
- Claim: Line 786: "$|\dot{T}_c| \ll \frac{T_c}{\sqrt{V_H(z)}}$"; line 789: "*Proof:* See Appendix E.8."
- Upstream anchor: `10_appendices/05_proofs.md:823`: "Statement: $V_H(z) = T_c^2\frac{\partial H(\pi)}{\partial T_c}$"; line 878: "Final Result: $V_H(z) = T_c\frac{\partial H(\pi)}{\partial T_c} = \beta_{\text{ent}}^2\mathrm{Var}(Q) = C_v$."
- Why this is an error: $[\dot T_c]=\mathrm{nat}/\text{time}$ while $[T_c/\sqrt{V_H}]=1$; the right-hand side is missing a relaxation rate. Item 3 is not proved anywhere: E.8 proves only the identity $V_H=T_c\,\partial_{T_c}H$. Numerical check with a random 6-action $Q$ at $T_c=0.7$: $V_H=0.7456$, $\beta^2\mathrm{Var}(Q)=0.7456$, $T_c\partial_{T_c}H=0.7456$, $T_c^2\partial_{T_c}H=0.5219$, so the chapter's line 776 is correct and the E.8 header carries a stray $T_c^2$.
- Impact on downstream results: Governor chapter (07_cognition) if it uses the adiabatic bound.
- Fix guidance: (1) Insert the relaxation time: $|\dot T_c|\,\tau_{\text{relax}}\ll T_c/\sqrt{V_H}$. (2) Prove item 3 (linear response: $\Delta H\approx\partial_{T_c}H\,\dot T_c\tau$ small against $\sqrt{V_H}$) or mark it as heuristic. (3) Fix the header at 05_proofs.md:823.
- Required new assumptions/permits: a definition of $\tau_{\text{relax}}$.
- Validation plan: dimensional check of the corrected inequality.

### [E-015] Conformal rescaling of $\kappa$ is a coordinate identity (was F-014)
- Location: Conformal Laplacian Transformation (lines 880-903)
- Severity: Moderate
- Type: Invalid inference
- Criterion: Framework
- Origin: this chapter
- Claim: Lines 892-899: "$-\Delta_{\tilde G}V + \tilde\kappa^2 V = \tilde\rho_r$, with effective screening mass $\tilde\kappa^2 = \Omega^{-2}\kappa^2$. ... In high-curvature regions ($\Omega$ large), the effective screening mass decreases, making the field more 'massless' and allowing longer-range correlations. This is the self-focusing effect."
- Upstream anchor: not applicable; the transformation formula at line 886, $\Delta_{\tilde G}f=\Omega^{-2}(\Delta_Gf+(d-2)\Omega^{-1}G^{ij}\partial_i\Omega\,\partial_jf)$, was recomputed and is correct.
- Why this is an error: Multiplying $-\Delta_GV+\kappa^2V=\rho_r$ by $\Omega^{-2}$ and using line 886 gives $-\Delta_{\tilde G}V+\Omega^{-2}\kappa^2V=\Omega^{-2}\rho_r-(d-2)\Omega^{-3}\langle\nabla\Omega,\nabla V\rangle_G$. So "$\tilde\kappa^2=\Omega^{-2}\kappa^2$" is the same equation rewritten in the rescaled metric (with the first-order term absorbed into $\tilde\rho_r$); the solution $V$ is unchanged. Distances rescale as $\tilde d=\Omega d$ locally, so $\tilde\kappa\tilde d=\kappa d$: the decay per unit of the new distance is identical. A genuine back-reaction would pose the PDE in $\tilde G$ with the original $\kappa$, which shortens the range in $G$-distance; the proposition asserts the opposite.
- Impact on downstream results: Connection #27 "Self-consistency" bullet, Node 38 interpretation.
- Fix guidance: (1) State which metric the Helmholtz equation is posed in. (2) If $\tilde G$ with fixed $\kappa$: drop the $\tilde\kappa$ rescaling and revise the interpretation (range in $G$-distance shrinks by $\Omega$). (3) If merely re-expressed: delete the self-focusing interpretation and keep the $(d-2)$ term explicit.
- Required new assumptions/permits: none.
- Validation plan: solve the 1D screened Poisson equation with a localised conformal factor under both readings and compare decay lengths.

### [E-016] Hessian code computes a Euclidean Frobenius norm, not the $G$-operator norm (was F-015)
- Location: def-value-metric-conformal-coupling (809-828), HolographicCritic implementation (1092-1141)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: Line 822: "$\Omega(z) = 1 + \alpha_{\text{conf}}\cdot\|\nabla^2_G V(z)\|_{\text{op}}$"; lines 1121-1123: "# Operator norm = largest singular value / # For efficiency, use Frobenius norm as upper bound / hess_norm = torch.linalg.matrix_norm(H, ord='fro')" where `H` is built from `torch.autograd.grad(grad_V[:, i].sum(), z, ...)`.
- Upstream anchor: `05_geometry/04_equations_motion.md:166-168` (Poincaré metric).
- Why this is an error: `H` is the coordinate Hessian $\partial_i\partial_jV$, not $\nabla^2_GV=\partial_i\partial_jV-\Gamma^k_{ij}\partial_kV$, and the norm is the Euclidean Frobenius norm, not $\|G^{-1}\nabla^2_GV\|_{\text{op}}$. On the Poincaré metric the two differ by $(1-|z|^2)^2/4$ plus Christoffel terms diverging like $1/(1-|z|^2)$ near the rim, so `conformal_factor` is off by orders of magnitude away from the origin and is not a monotone proxy; the stated units $[\alpha_{\text{conf}}]=\text{length}^2/\mathrm{nat}$ only balance with the $G$-norm. Secondary: `self.kappa` (line 990) is never used; `compute_helmholtz_loss` evaluates TD error plus a gradient penalty and never the Helmholtz residual, so Node 35's proxy is not implemented; Node 35's cost "$O(B\cdot D)$" understates a Laplace-Beltrami evaluation ($O(BD^2)$ at least).
- Impact on downstream results: `conformally_scaled_metric`, Node 38 (Var $\Omega$ range $0.1$-$2.0$ is meaningless in the wrong norm).
- Fix guidance: (1) Compute `H_cov = H - einsum(Gamma, grad_V)`, `A = G_inv @ H_cov`, and `torch.linalg.matrix_norm(A, ord=2)` (or power iteration). (2) Use `self.kappa` in a residual term or remove it and rename the loss. (3) Correct Node 35's cost.
- Required new assumptions/permits: access to Christoffel symbols of the metric (available upstream, prop-explicit-christoffel-symbols-for-poincare-disk).
- Validation plan: for $V(z)=|z|^2$ on the Poincaré disk, compare the code's $\Omega$ against the closed-form $G$-operator norm at several radii.

### [E-017] Geodesic SDE restated without friction/Lorentz term and with the wrong noise amplitude (was F-016)
- Location: RL as Electrodynamics on a Curved Manifold (lines 1248-1279)
- Severity: Moderate
- Type: Definition mismatch (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter (upstream sign issue at `05_geometry/04_equations_motion.md:227`)
- Claim: Lines 1260-1264: "moving according to the geodesic SDE (Definition def-bulk-drift-continuous-flow): $dq^k = G^{kj}p_j\,ds$, $dp_k = -\frac{\partial V}{\partial q^k}ds - \frac12\frac{\partial G^{ij}}{\partial q^k}p_ip_j\,ds + u_{\pi,k}\,ds + \sqrt{2T_c}\,(G^{1/2})_{kj}dW^j_s$"; line 1277: "Temperature $T_c$ | Thermal Bath (fluctuation-dissipation source)".
- Upstream anchor: `04_equations_motion.md:225-231`: "$dp_k = [-\partial_k\Phi_{\text{eff}} - \gamma\,p_k + \beta_{\text{curl}}\,\mathcal{F}_{kj}G^{j\ell}p_\ell - \Gamma^m_{k\ell}G^{\ell j}p_jp_m + u_{\pi,k}]\,ds + \sqrt{2\gamma T_c}\,(G^{1/2})_{kj}\,dW^j_s$"; line 245: "Thermal noise: $\sqrt{2\gamma T_c}\,G^{1/2}dW$ — fluctuation-dissipation balanced noise".
- Why this is an error: The chapter drops the friction $-\gamma p_k$ but keeps noise, and changes the amplitude from $\sqrt{2\gamma T_c}$ to $\sqrt{2T_c}$. Without friction there is no fluctuation-dissipation balance and no stationary Boltzmann state (momentum variance grows linearly in $s$), contradicting the table row and the canonical-ensemble sections. A theorem titled "RL as Electrodynamics" also omits the Lorentz term $\beta_{\text{curl}}\mathcal{F}G^{-1}p$ that the upstream definition and this chapter's own EM isomorphism (line 142) make the defining ingredient, and replaces $\Phi_{\text{eff}}$ by $V$ (see E-011). Upstream note: with $\dot q^a=G^{ab}p_b$, $\Gamma^m_{k\ell}G^{\ell j}p_jp_m=\tfrac12\partial_kG_{ab}\dot q^a\dot q^b$, so the upstream geodesic term is $-\tfrac12\partial_kG_{ab}\dot q^a\dot q^b$, whereas the Hamiltonian force $-\tfrac12\partial_kG^{ij}p_ip_j=+\tfrac12\partial_kG_{ab}\dot q^a\dot q^b$ (using $\partial_kG^{ij}=-G^{ia}\partial_kG_{ab}G^{bj}$), which agrees with the Lagrangian $\dot p_k=\partial L/\partial q^k$. The chapter's form is right and the upstream sign is wrong, so the claimed equivalence fails on a second, independent count.
- Impact on downstream results: cor-the-three-boundary-conditions, the Holographic Dictionary "Dynamics" row, BAOAB cross-reference.
- Fix guidance: (1) Quote the upstream SDE verbatim (friction, Lorentz term, $\sqrt{2\gamma T_c}$, $\Phi_{\text{eff}}$), or state explicitly the simplifications (conservative, undamped) and drop the thermal-bath row. (2) Flag the geodesic-term sign to the 04_equations_motion review.
- Required new assumptions/permits: none.
- Validation plan: check that the chapter's SDE and the upstream one agree term by term after the edit.

### [E-018] Node 36 proxy grows with separation instead of measuring decay (was F-017)
- Location: Node 36 (lines 1326-1335), Table 24.8.1 (line 1404)
- Severity: Moderate
- Type: Conceptual (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: Line 1331: "Proxy: $\mathbb{E}[\lVert V(z) - V(z')\rVert\cdot e^{\kappa d_G(z,z')}]$"; line 1404: "Healthy Range: $\approx 1.0$ (constant after scaling)".
- Upstream anchor: prop-green-s-function-decay (lines 512-518, this chapter).
- Why this is an error: The proposition bounds the kernel $G_\kappa(z,z')$, the response at $z$ to a unit source at $z'$, which decays. $|V(z)-V(z')|$ is a structure function: zero at coincidence, increasing with separation, saturating at the value range; multiplied by $e^{\kappa d_G}$ it diverges exponentially for every $V$, converged or not. The statistic cannot be near a constant and does not test screening. (With E-010 the exponent is also wrong on the hyperbolic bulk.) The verifier raised the severity from Minor to Moderate because the diagnostic is incorrect as stated, not merely imprecise.
- Impact on downstream results: Table 24.8.1, sieve integration.
- Fix guidance: (1) Use the empirical response of $V$ to localised reward perturbations, or (2) fit $\log\mathrm{Cov}(V(z),V(z'))$ against $d_G$ and compare the slope to $-s(\kappa)$.
- Required new assumptions/permits: none.
- Validation plan: evaluate the corrected statistic on a synthetic critic that solves the PDE exactly; it should return the expected constant.

### [E-019] Node 61 proxy/code do not compute a curl and do not use TD errors (was F-018)
- Location: Node 61: ValueCurlCheck (lines 1371-1398)
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: Line 1376: "Proxy: $\oint_\gamma\delta_{\text{TD}} \approx \int\|\nabla\times\mathcal{R}\|$"; lines 1392-1397: "Estimate Value Curl via loop integral of TD-errors ... loop_integral = rewards.sum().item(); return abs(loop_integral)".
- Upstream anchor: def-value-curl (lines 213-219, this chapter); see E-005.
- Why this is an error: (i) The proxy equates a signed circulation with the integral of a norm; by Stokes the right-hand side is the signed flux $\int_\Sigma\mathcal{F}$, and a loop can enclose cancelling curl with zero circulation. (ii) The docstring says TD errors but the code sums raw rewards; $\sum_t\delta_{\text{TD},t}=\sum_tr_t+\sum_t(\gamma V_{t+1}-V_t)$ equals $\sum r_t$ only for $\gamma=1$ on an exactly closed loop; otherwise the difference is $(\gamma-1)\sum_tV_{t+1}$, of order $T(1-\gamma)\bar V$. (iii) By E-005 a non-zero circulation may come from the harmonic component with $\mathcal{F}\equiv0$, so the statistic tests exactness, not curl. The reward sum is nevertheless a legitimate circulation estimator, so the code is usable under a corrected label.
- Impact on downstream results: Table 24.8.1; the "Productive vs Pathological curl" guidance.
- Fix guidance: (1) Return the signed `rewards.sum()` (or a $\gamma^t$-weighted TD sum) and rename to "circulation check". (2) If curl proper is wanted, estimate $\mathcal{F}_{ij}$ from small coordinate plaquettes.
- Required new assumptions/permits: none.
- Validation plan: run on a synthetic exact field ($\mathcal{R}=d\Phi$) and on a rotational field; the first must return zero up to closure error.

## Scope restrictions and clarifications
- The chapter's default bulk geometry is the truncated hyperbolic ball (line 128, 861). Several statements (Hodge theorem, conservative-iff-curl-free, Green's-function asymptotics, "$\gamma\to1\Rightarrow\ell\to\infty$") are valid on closed or flat manifolds and need explicit restriction or restatement for a compact manifold with boundary and for negative curvature (E-004, E-005, E-010).
- The Helmholtz derivation is diffusive (second-order in space, first-order in time). Any spatial screening mass it produces scales as $\sqrt{-\ln\gamma}$. The ballistic conversion $\kappa=\lambda/c_{\text{info}}$ belongs to a first-order or hyperbolic (Klein-Gordon) model that the chapter does not derive (E-006).
- The sign convention for $\Phi$ versus $V$ must be settled jointly with `05_geometry/04_equations_motion.md` (def-effective-potential, def-bulk-drift-continuous-flow) (E-011, E-017).
- The natural-unit statement "$\Delta t=1$, $c_{\text{info}}=1$" (line 382) does not by itself fix the relation between the curvature radius of the latent geometry and the unit of length; the screening-length numbers in Table 24.2.5 are meaningful only once that relation is stated (E-010).

## Open questions
- Which of the two screening conventions is intended for the book: $\kappa^2=-\ln\gamma$ (`06_losses.md:789`, and the chapter's own derivation) or $\kappa=-\ln\gamma$ (this chapter's statements and all downstream numbers)? The choice determines whether the $\ell\approx100$ figures survive.
- Is the reward meant to have a velocity-independent density component ($\rho_r$) in addition to the 1-form, or is $\rho_r$ purely boundary data? The Helmholtz source term requires one or the other (E-007).
- Is the Helmholtz equation posed in $G$ or in the conformally modified $\tilde G$? The self-focusing claim depends on the answer (E-015).
- Is the WFR reaction term intended to act at equilibrium, or only as a transient selection mechanism? The Boltzmann corollary holds only in the second reading (E-013).

## Rejected candidate findings
None. All nineteen stage-1 findings were confirmed; four were adjusted (F-003: item (ii) restricted to $d=2$; F-014: criterion changed to Framework; F-017: severity raised to Moderate; F-019: location line 107 corrected to 807).
