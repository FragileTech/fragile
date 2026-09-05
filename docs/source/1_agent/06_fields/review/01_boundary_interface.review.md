# Mathematical Review: docs/source/1_agent/06_fields/01_boundary_interface.md

## Metadata
- Reviewed file: docs/source/1_agent/06_fields/01_boundary_interface.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/05_geometry/03_holographic_gen.md:68` (ideal boundary $\partial\mathcal{Z} := \{z\in\mathbb{C}^n : |z|=1\}$), `:239` ($G(0)=4I$), `:263` (control field $u_\pi = G^{-1}\mathbb{E}[a]$), `:270`, `:306-308` (dreaming: $u_\pi=0$, thermal fluctuation), `:552-559` (visual texture firewall)
  - `docs/source/1_agent/05_geometry/01_metric_law.md:131` ($(n-1)$-dimensional area form on $\partial\mathcal{Z}$)
  - `docs/source/1_agent/05_geometry/02_wfr_geometry.md:114-117` (unbalanced continuity equation)
  - `docs/source/1_agent/05_geometry/04_equations_motion.md:68, :75` (metric $\mathbf{M}=G$, units), `:217` (boundary-condition cross-reference), `:225-229` (Langevin SDE, noise $\sqrt{2\gamma T_c}$), `:238` (units of $z,p$), `:264` (BAOAB preserves Boltzmann distribution), `:406, :423, :443, :459` (`def-effective-potential` and value semantics), `:507-515` (cognitive temperature $T_c$), `:553, :571-576, :596` (BAOAB splitting)
  - `docs/source/1_agent/04_control/03_coupling_window.md:82-90` (`def-grounding-rate`: $G_t := I(X_t;K_t)$, $\lambda_{\text{in}} := \mathbb{E}[G_t]$)
  - `docs/source/1_agent/03_architecture/02_disentangled_vae.md:120-129` (per-chart nearest-code VQ)
  - `docs/source/1_agent/06_fields/02_reward_field.md:620, :675` (units of $T_c S$; reaction rate $r(z)$)
  - `docs/source/1_agent/10_appendices/05_proofs.md:825` (Boltzmann policy $\exp(Q/T_c)$), `10_appendices/04_faq.md:466`

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 12
- Minor: 7
- Notes: 1
- Primary themes:
  1. The symplectic-boundary formalism is ill-typed. The boundary $\partial\mathcal{Z}$ is upstream the odd-dimensional sphere $|z|=1$, yet it is declared a symplectic manifold $\cong T^*\mathcal{Q}$; the Dirichlet/Neumann "boundary conditions" are neither conditions on $\partial\mathcal{Z}$ nor consistent with the WFR flux; and the Legendre-duality theorem is applied to an atlas on $\mathcal{Q}$ that carries no velocity coordinates.
  2. The thermodynamic-cycle section states unproved "theorems" and a Carnot bound violated by a simple in-framework example, and its isentropic Hamiltonian "dreaming" contradicts the book's thermal (Langevin) definition of dreaming and the chapter's own reflective-WFR dreaming.
  3. The Context Space policy $\pi(a|z,c)\propto\exp(-\Phi_{\text{eff}}/T_c)$ has no $a$-dependence, and in the RL row carries the wrong sign relative to the book's Boltzmann policy $\exp(Q/T_c)$ (partly inherited from upstream `def-effective-potential`), so that the context-conditioned RL velocity vanishes identically.
  4. The reference implementation does not implement the stated definitions (no VQ lookup; policy output discarded; texture slice unsafe).
  5. The grounding rate is redefined in conflict with its upstream definition; the waking/dreaming modes are keyed to $u_\pi$ rather than to the boundary condition the text claims defines them; several stale hard-coded reference numbers.
- Mechanical pre-pass: all 34 distinct `{ref}`/`{prf:ref}` targets used in this file resolve to exactly one label in `docs/source/1_agent`; no dangling references or duplicate labels.

Line numbers below refer to the file as read during verification.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Def. Symplectic Boundary Manifold, 68-75 | Moderate | Conceptual | Framework | this chapter | $\partial\mathcal{Z}$ (upstream: odd-dimensional sphere) declared a symplectic manifold $\cong T^*\mathcal{Q}$; $T^*\mathcal{M}$ undefined |
| E-002 | Def. Symplectic Boundary Manifold, 78 | Minor | Dimensional mismatch | Framework | this chapter | $[\omega]=[q][p]=\mathrm{nat}$ does not follow from either unit convention |
| E-003 | Def. Dirichlet BC (Sensors), 95-101; Def. Waking, 558-560 | Moderate | Conceptual | Framework | this chapter | "Dirichlet datum" is a Dirac mass at an interior point; Waking treats it as a relaxation target, not a clamp |
| E-004 | Def. Neumann BC (Motors), 119-130; 555, 569 | Moderate | Conceptual | Framework | this chapter | WFR boundary flux is $\rho v\cdot\mathbf{n}$, not $\nabla\rho\cdot\mathbf{n}$; three incompatible unit systems equated |
| E-005 | Hamiltonian BC admonition and Prop. Symplectic Duality, 146-175 | Moderate | Invalid inference | External | this chapter | Phase-space rotation does not map PDE Dirichlet to Neumann; proof sketch invokes a different map |
| E-006 | Thm. Atlas Duality via Legendre Transform, 207-286 | Major | Invalid inference | Framework | this chapter | $\mathcal{L}$ applied to an atlas without velocity coordinates; forces $|\mathcal{K}_{\text{vis}}|=|\mathcal{K}_{\text{act}}|$; sign of $\omega$ flips; uniqueness false |
| E-007 | Def. Neumann BC 125 vs Axiom Motor Texture Firewall 409 | Moderate | Definition mismatch | Framework | this chapter | Motor boundary datum depends on $z_{\text{tex,motor}}$, so the bulk solution does; firewall forbids this |
| E-008 | Def. Cycle Phases, Remark 453; Prop. Carnot 508 | Minor | Conceptual | External | this chapter | Called a Stirling cycle but has an isentropic leg and a Carnot bound |
| E-009 | Thms. Perception as Compression / Action as Expansion, 462-485 | Moderate | Proof gap / omission | Framework | this chapter | Labelled theorems without proof; three different "action" variables; bulk $\Delta S>0$ attributed to interface texture |
| E-010 | Def. Dreaming as Unitary Evolution, 494-505; Def. Dreaming: Reflective Boundary, 569-574 | Moderate | Conceptual | Framework | this chapter | Isentropic Hamiltonian dreaming contradicts upstream thermal dreaming and the chapter's own WFR dreaming |
| E-011 | Prop. Carnot Efficiency Bound, 514-519 | Moderate | Invalid inference | Framework | this chapter | Undefined temperatures, no proof; counterexample gives $\eta\approx34.4$ |
| E-012 | Waking/Dreaming definitions 541-566; Thm. WFR Mode Switching 583-586; Prop. Grounding Rate 599-601 | Minor | Definition mismatch | Framework | this chapter | Modes keyed to $u_\pi$, not to the boundary condition the text says defines them; "pure actuation" regime has no mode |
| E-013 | Prop. Grounding Rate via Boundary Flux, 592-601 | Moderate | Definition mismatch | Framework | this chapter | Signed net flux redefines $G_t$, upstream a nonnegative mutual information |
| E-014 | Lines 592, 753-1014 (code comments), 1095, 1115 | Minor | Citation / reference error | Framework | this chapter | Stale hard-coded reference numbers ("Definition 16.1.1", "Definition 23.x.y") |
| E-015 | Def. Context Space 640; Thm. Universal Context Structure item 3; docstring 923 | Moderate | Definition mismatch | Framework | this chapter | $\Phi_{\text{eff}}(z,K,c)$ has no $a$-dependence; the RL softmax is uniform |
| E-016 | Def. Context Instantiation Functor RL row 653; 640; 692; 712; summary table | Major | Parameter inconsistency | Framework | upstream `05_geometry/04_equations_motion.md` (`def-effective-potential`) and this chapter | With $\Phi_{\text{eff}}=V_{\text{critic}}$ the policy is $\exp(-V/T_c)$ and $v_c\equiv0$ in the RL row |
| E-017 | Thm. Universal Context Structure, 665-679 | Minor | Dimensional mismatch | Framework | this chapter | $G^{-1}(0)\in\mathbb{R}^{d_z\times d_z}$ applied to $e_c\in\mathbb{R}^{d_c}$; proof does not address the statement |
| E-018 | Cor. Prompt = Action = Label, 699-702; 649 | Note | Proof gap / omission | Framework | this chapter | "Isomorphic" with no structure specified; functor into a manifold |
| E-019 | Implementation, DualAtlasEncoder.forward 840-843 | Moderate | Algorithm mismatch | Framework | this chapter | `torch.zeros(1, dtype=long)` always selects code 0; no quantisation |
| E-020 | Implementation, HolographicInterface.forward_actuation/forward 1015-1072 | Moderate | Algorithm mismatch | Framework | this chapter | Policy `probs` unused; intention is the context embedding; state is nuisance only; texture slice unsafe |
| E-021 | Connection to RL #7, 1089-1112 | Minor | Conceptual | External | this chapter | BAOAB (with OU thermostat) claimed symplectic; upstream claims only Boltzmann preservation |
| E-022 | Diagnostic Nodes 30, 31, 33, 1145-1178 | Minor | Dimensional mismatch | Framework | this chapter | $\omega$ of two scalars undefined; Node 31 uses $\mathcal{L}$ backwards on codebook vectors; Node 33 contradicts Carnot |

## Detailed findings

### [E-001] The boundary $\partial\mathcal{Z}$ cannot be the symplectic manifold $T^*\mathcal{Q}$ (was F-001)
- Location: The Symplectic Interface, Definition Symplectic Boundary Manifold (lines 68-75)
- Severity: Moderate
- Type: Conceptual (secondary: Dimensional mismatch, Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 68-75): "The agent's interface is a symplectic manifold $(\partial\mathcal{Z}, \omega)$ with canonical coordinates $(q, p) \in T^*\mathcal{M}$ where $q \in \mathcal{Q}$ is the position bundle ... $p \in T^*_q\mathcal{Q}$ ... $\omega = \sum_{i=1}^n dq^i \wedge dp_i$."
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:68`: "$\partial\mathcal{Z} := \{z \in \mathbb{C}^n : |z| = 1\}$"; `docs/source/1_agent/05_geometry/01_metric_law.md:131`: "Let $dA_G$ be the induced $(n-1)$-dimensional area form on $\partial\mathcal{Z}$."
- Why this is an error: Upstream, $\partial\mathcal{Z}$ is the ideal boundary of the latent ball, the sphere $S^{2n-1}\subset\mathbb{R}^{2n}$, which has odd real dimension for every $n$ (no choice of $d$ is needed for the parity argument). A symplectic manifold is even-dimensional, and $T^*\mathcal{Q}$ has dimension $2\dim\mathcal{Q}$, so no symplectic form exists on the upstream $\partial\mathcal{Z}$. Independently of parity, the same symbol is used later in this chapter as the hypersurface on which the normal-derivative condition is imposed (line 119) and over which area integrals $\oint_{\partial\mathcal{Z}_{\text{sense}}} j\,dA$ are taken (line 595); these require $\partial\mathcal{Z}$ to be the geometric boundary of $\mathcal{Z}$, not a cotangent bundle. The symbol $\mathcal{M}$ in "$T^*\mathcal{M}$" is never introduced (its only occurrence in the chapter is line 68); presumably $T^*\mathcal{Q}$ is meant.
- Impact on downstream results: Every later statement "on $\partial\mathcal{Z}$" (Dirichlet/Neumann conditions, grounding-rate flux integrals, Node 30) inherits the ambiguity; the Symplectic Bridge (Node 48) cites this $\omega$.
- Fix guidance:
  1. Keep $\partial\mathcal{Z}$ as the geometric boundary, per upstream.
  2. Introduce a distinct interface phase space $\mathcal{P} := T^*\mathcal{Q}$ (or $T^*(\partial\mathcal{Z})$, which is symplectic for any $\partial\mathcal{Z}$) carrying $\omega$.
  3. Replace $T^*\mathcal{M}$ by $T^*\mathcal{Q}$ and state the relation between $\mathcal{Q}$ and $\partial\mathcal{Z}$ explicitly (e.g. $\mathcal{Q}\cong\partial\mathcal{Z}$ as the space of boundary sensory configurations).
- Required new assumptions/permits: none.
- Validation plan: Check that every occurrence of $\partial\mathcal{Z}$ in the chapter refers to the hypersurface and every occurrence of $\omega$ refers to $\mathcal{P}$; confirm $\dim\mathcal{P}$ is even.

### [E-002] Units of $\omega$ do not follow from the book's unit assignments (was F-002)
- Location: Definition Symplectic Boundary Manifold (line 78)
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 78): "Units: $[\omega] = [q][p] = \mathrm{nat}$."
- Upstream anchor: `docs/source/1_agent/05_geometry/04_equations_motion.md:238`: "*Units:* $[z] = \text{length}$, $[p] = \text{length}/\tau$, $[\gamma] = 1/\tau$, $[\Phi_{\text{eff}}] = \mathrm{nat}$, $[T_c] = \mathrm{nat}$."; `04_equations_motion.md:75`: "Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric)."
- Why this is an error: With the chapter's own identification $p = G(q)\dot q$ (line 269) and the upstream units $[G]=[z]^{-2}$, $[q]=[z]$, one gets $[p]=[z]^{-1}/\tau$ and $[q][p]=1/\tau$. With the upstream SDE convention $[p]=\text{length}/\tau$ one gets $[q][p]=\text{length}^2/\tau$. Neither is nat. The claim is asserted, not derived, and is inconsistent with both conventions.
- Impact on downstream results: The prose at line 87 ("$\omega$ measures information content") builds on this unit claim; Node 30 uses $\omega$ as a scalar proxy.
- Fix guidance:
  1. Either define $p$ as an information momentum with $[p]=\mathrm{nat}/[q]$ and derive $[\omega]$ from that convention (this also requires changing $p=G\dot q$), or
  2. Drop the unit claim and the interpretation that depends on it.
- Required new assumptions/permits: a stated unit convention for $p$ if option 1 is taken.
- Validation plan: Recompute $[q][p]$ from the chosen convention and confirm it matches the stated $[\omega]$.

### [E-003] The "Dirichlet" sensor condition is not a boundary condition and contradicts the chapter's own tracking dynamics (was F-003)
- Location: Definition Dirichlet Boundary Condition, Sensors (lines 95-101); Definition Waking (lines 558-560)
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 95-101): "$\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t))$ ... $q_{\text{obs}}(t) = E_\phi(x_t)$ ... This clamps the *configuration* of the belief state." Lines 558-560: "The internal belief $\rho_{\text{bulk}}$ evolves to minimize the WFR Geodesic Distance to $\rho_{\partial}$: Small Error ($d_{\text{WFR}} < \lambda$) ... Large Error ($d_{\text{WFR}} > \lambda$)."
- Upstream anchor: `docs/source/1_agent/05_geometry/02_wfr_geometry.md:114-117`: "subject to the Unbalanced Continuity Equation $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$"; `03_holographic_gen.md:68` ($\partial\mathcal{Z}=\{|z|=1\}$).
- Why this is an error: A Dirichlet condition for the continuity equation prescribes the trace of $\rho$ on the boundary hypersurface $\partial\mathcal{Z}$. The displayed object is a full density on the domain, a Dirac mass at $q_{\text{obs}}=E_\phi(x_t)$, and by line 326 ($E_\phi:\mathcal{Q}\to\mathcal{Z}$) this is a point of the open ball, not of $\partial\mathcal{Z}$. It is a target distribution in the bulk, not a boundary value. The Waking definition treats it exactly so: $\rho_{\text{bulk}}$ "evolves to minimize the WFR distance to $\rho_\partial$", which is incompatible with $\rho$ being clamped equal to $\delta$ (if clamped, $d_{\text{WFR}}=0$ identically and the small/large-error dichotomy at $\lambda$ is empty). A Dirac belief also contradicts the belief state being a posterior with nonzero uncertainty, which is the premise of the WFR transport/reaction dichotomy.
- Impact on downstream results: Theorem WFR Mode Switching, Proposition Grounding Rate, and the consumer `04_equations_motion.md:217` ("Dirichlet on sensors (clamping observed position)") rely on this reading.
- Fix guidance:
  1. Reformulate perception as a source/relaxation term, e.g. $\partial_s\rho + \nabla\cdot(\rho v) = \rho r + \kappa_{\text{obs}}(\rho_{\text{obs}}-\rho)$ with $\rho_{\text{obs}}$ the encoder posterior; or state a genuine Dirichlet datum on $\partial\mathcal{Z}$ (the boundary trace of $\rho$).
  2. Remove "clamps" or restrict it to the limit $\kappa_{\text{obs}}\to\infty$.
  3. Update the Waking definition and `04_equations_motion.md:217` to the chosen formulation.
- Required new assumptions/permits: a relaxation rate $\kappa_{\text{obs}}$ if option 1 is taken.
- Validation plan: Check that $d_{\text{WFR}}(\rho_{\text{bulk}},\rho_{\text{obs}})$ is nontrivial under the new formulation so the $\lambda$ dichotomy is meaningful.

### [E-004] The "Neumann" motor condition uses the wrong flux for the WFR equation and equates a density gradient with a decoder output (was F-004)
- Location: Definition Neumann Boundary Condition, Motors (lines 119-130); Definition Waking (line 555); Definition Dreaming (line 569)
- Severity: Moderate
- Type: Conceptual (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 119-130): "$\nabla \rho \cdot \mathbf{n} \big|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}(p, t)$ ... $j_{\text{motor}} = D_A(u_\pi) = \text{Decoder}(z, u_\pi, z_{\text{tex,motor}})$ ... Units: $[j_{\text{motor}}] = \mathrm{nat}/\text{step}$."
- Upstream anchor: `docs/source/1_agent/05_geometry/02_wfr_geometry.md:117`: "$\partial_s \rho + \nabla \cdot (\rho v) = \rho r$"; `05_geometry/04_equations_motion.md:225-229` (drift/diffusion with noise $\sqrt{2\gamma T_c}$); this chapter line 541 (the WFR equation governs $\rho$).
- Why this is an error: (i) For the WFR continuity equation the mass flux through the boundary is $\rho\,v\cdot\mathbf{n}$ (plus $-T_c G^{-1}\nabla\rho\cdot\mathbf{n}$ if the Fokker-Planck form of the SDE is used). A pure $\nabla\rho\cdot\mathbf{n}$ condition is a Neumann condition for a diffusion equation, not a flux condition for the transport equation the chapter says governs $\rho$; likewise the reflective condition $\nabla\rho\cdot\mathbf{n}=0$ (line 569) does not make the WFR system closed, whereas $\rho v\cdot\mathbf{n}=0$ does. (ii) Dimensionally, $\nabla\rho\cdot\mathbf{n}$ has units $[\rho]/[z]$, $j_{\text{motor}}$ is assigned nat/step, and it is simultaneously set equal to the Action Decoder output, which per Definition Action Atlas (line 232) is raw actuation $a_{\text{raw}}$ "(torques, voltages)". Three incompatible unit systems are equated. (iii) The notation $\nabla_n\rho\cdot\mathbf{n}$ (lines 139, 555, 569) is redundant.
- Impact on downstream results: Theorem WFR Mode Switching table; Proposition Grounding Rate; the cross-reference at `04_equations_motion.md:217`.
- Fix guidance:
  1. State the motor condition as a prescribed outward WFR flux $(\rho v)\cdot\mathbf{n}|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}$ (adding $-T_cG^{-1}\nabla\rho\cdot\mathbf{n}$ if diffusion is included).
  2. Define $j_{\text{motor}}$ as an information flux functional of the action (e.g. $I(A_t;K_t)/\Delta t$ per unit boundary area) and decouple it from the raw decoder output.
  3. Use the same corrected flux in the reflective condition and replace $\nabla_n\rho\cdot\mathbf{n}$ by $\nabla\rho\cdot\mathbf{n}$ or the flux expression.
- Required new assumptions/permits: none.
- Validation plan: Verify mass balance $\frac{d}{ds}\int\rho = \int\rho r - \oint(\rho v)\cdot\mathbf{n}\,dA$ closes with the stated conditions, and check units of both sides of the boundary condition.

### [E-005] Symplectic Duality Principle: a phase-space rotation does not interchange Dirichlet and Neumann conditions; the proof sketch invokes a different map (was F-005)
- Location: Physics Isomorphism: Hamiltonian Boundary Conditions and Proposition Symplectic Duality Principle (lines 146-175)
- Severity: Moderate
- Type: Invalid inference (secondary: Notation conflict)
- Criterion: External
- Origin: this chapter
- Claim (lines 168-175): "Under the canonical transformation $(q, p) \mapsto (p, -q)$: Dirichlet conditions become Neumann conditions ... *Proof sketch.* The symplectic form $\omega$ is invariant under canonical transformations. The Legendre transform $\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}$ maps velocity to momentum, exchanging position-fixing (Dirichlet) for flux-fixing (Neumann)." Line 146 asserts "Boundary conditions fix either $q$ (Dirichlet) or $\partial_n q \propto p$ (Neumann)".
- Upstream anchor: not applicable; the claim is this chapter's own. Judged against standard mechanics and PDE facts.
- Why this is an error: (i) Dirichlet and Neumann conditions are conditions on a field over a spatial domain (value versus normal derivative on $\partial\Omega$). The map $(q,p)\mapsto(p,-q)$ is a symplectomorphism of phase space; it sends the constraint $q=q_0$ to $p'=q_0$, a value constraint on the new momentum, not a normal-derivative constraint. Nothing identifies $p$ with $\partial_n q$; for a mechanical system $p=\partial L/\partial\dot q$ is a time derivative, not a spatial normal derivative. (ii) The proof sketch replaces the stated transformation by the Legendre transform, which is a map $T\mathcal{Q}\to T^*\mathcal{Q}$, not a canonical transformation of $T^*\mathcal{Q}$, and does not act on boundary conditions of a PDE. (iii) The correspondence tables are mutually inconsistent about what "momentum" is: motor flux $p\in T^*_q\mathcal{Q}$ (line 70), "Policy gradient $\nabla_z V$" (line 159), "Value gradient $\nabla_A V$" (line 311); and "Hamiltonian $H\leftrightarrow\Phi_{\text{eff}}$" (lines 162, 309) versus $H_{\text{internal}}=\tfrac12\|p\|^2_{G^{-1}}+V_{\text{critic}}$ (line 500), where $\Phi_{\text{eff}}$ can at most be the potential term.
- Impact on downstream results: Theorem Atlas Duality's Remark items 1-2 cite this proposition for "preserves the symplectic structure" and "interchanges Dirichlet and Neumann"; the "sensing = acting" narrative rests on it.
- Fix guidance:
  1. Either demote the proposition to a heuristic remark, or
  2. State a precise duality on the interface phase space $T^*\mathcal{Q}$: sensing prescribes the base coordinate $q$, acting prescribes the fibre coordinate $p$, and $(q,p)\mapsto(p,-q)$ swaps the two prescriptions. Remove the identification with PDE Dirichlet/Neumann conditions unless a field-theoretic model with $p\propto\partial_n q$ derived from a boundary Lagrangian is supplied.
  3. Consolidate the correspondence tables into one (momentum $=p$; Hamiltonian $=\tfrac12\|p\|^2_{G^{-1}}+\Phi_{\text{eff}}$).
- Required new assumptions/permits: a boundary Lagrangian if option 2 with PDE language is retained.
- Validation plan: Check that every use of "Dirichlet"/"Neumann" in the chapter refers to the same objects the proposition transforms.

### [E-006] Theorem Atlas Duality via Legendre Transform is ill-typed and its conclusion contradicts the atlas definitions (was F-006)
- Location: The Dual Atlas Architecture, Definitions Visual/Action Atlas (lines 207-233) and Theorem Atlas Duality (lines 248-286)
- Severity: Major
- Type: Invalid inference (secondary: Definition mismatch, Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (line 258): "$\psi_\beta \circ \mathcal{L} \circ \phi_\alpha^{-1} = \nabla_{\dot{q}} L(q, \dot{q})$"; line 273: "Let $(U_\alpha, \phi_\alpha)$ be a chart in $\mathcal{A}_{\text{vis}}$ with coordinates $(q^\alpha, \dot{q}^\alpha)$"; line 276: "$(q^\alpha, \dot{q}^\alpha) \mapsto (q^\alpha, G_{\alpha\beta}(q)\dot{q}^\beta)$"; line 271: "$\omega_{T^*\mathcal{Q}} = dp \wedge dq$"; line 281: "the unique smooth map".
- Upstream anchor: $\mathcal{A}_{\text{vis}}$ and $\mathcal{A}_{\text{act}}$ are defined in this chapter: lines 207-209 ("chart atlas on the sensory manifold $\mathcal{Q}$ ... $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$"), lines 226-229 ("chart atlas on the motor manifold $T^*\mathcal{Q}$ ... $\beta \in \mathcal{K}_{\text{act}}$"). For $G$: `docs/source/1_agent/05_geometry/04_equations_motion.md:68` "$\mathbf{M}(z) := G(z)$" (a metric on $\mathcal{Z}$, not on $\mathcal{Q}$).
- Why this is an error: (i) $\mathcal{L}$ is defined on $T\mathcal{Q}$, but $\mathcal{A}_{\text{vis}}$ is an atlas on $\mathcal{Q}$: its charts carry no velocity coordinate, so $\mathcal{L}\circ\phi_\alpha^{-1}$ is undefined; Step 3 silently upgrades $\phi_\alpha$ to a chart on $T\mathcal{Q}$, changing the definition mid-proof. (ii) The displayed identity is a type error: the left side (if lifted) maps $(q,\dot q)\mapsto(q,p)$, the right side is the single covector $\partial L/\partial\dot q$. (iii) In Step 3 the symbols $\alpha,\beta$ are used both as chart indices and as tensor indices of $G_{\alpha\beta}$. (iv) Step 3 constructs exactly one action chart per visual chart, so $\mathcal{A}_{\text{act}}=\mathcal{L}(\mathcal{A}_{\text{vis}})$ forces a bijection $\mathcal{K}_{\text{vis}}\leftrightarrow\mathcal{K}_{\text{act}}$; but the definitions give independent index sets with unrelated semantics ("Objects, Scenes, Viewpoints" versus "Gaits, Grasps, Tool Affordances"), and the implementation uses `num_visual_charts: int = 8`, `num_action_charts: int = 4` (lines 773-774), which the theorem forbids. (v) Line 286 declares $G$ in $L=\tfrac12\|\dot q\|_G^2-V$ to be the metric of `thm-capacity-constrained-metric-law`, which lives on $\mathcal{Z}$; no pull-back through $E_\phi$ is specified. (vi) Line 271 writes $\omega_{T^*\mathcal{Q}}=dp\wedge dq$ whereas line 75 has $\omega=\sum dq^i\wedge dp_i = -dp\wedge dq$. (vii) The Remark's uniqueness claim is false: composing $\mathcal{L}$ with any symplectomorphism of $T^*\mathcal{Q}$ gives another smooth map preserving the symplectic structure.
- Impact on downstream results: Node 31 (DualAtlasConsistencyCheck) and its "Legendre alignment loss" remedy are defined from this theorem; the Legendre Physics Isomorphism (lines 293-313) and the surrounding prose present it as established.
- Fix guidance:
  1. Restate as a definition/construction: given a Lagrangian $L$ on $T\mathcal{Q}$ with metric $G_{\mathcal{Q}} := E_\phi^*G$, define the induced action atlas $\mathcal{L}_*(\mathcal{A}^{T}_{\text{vis}})$ on $T^*\mathcal{Q}$ from the tangent lift of the visual atlas.
  2. State as a design constraint (with a loss) that $\mathcal{A}_{\text{act}}$ refines or is compatible with $\mathcal{L}_*(\mathcal{A}^T_{\text{vis}})$, allowing $|\mathcal{K}_{\text{act}}|\neq|\mathcal{K}_{\text{vis}}|$.
  3. Correct the transition-function formula to $\psi_\beta\circ\mathcal{L}\circ(T\phi_\alpha)^{-1}(q,\dot q)=(q,\,G_{\mathcal{Q}}(q)\dot q)$ with distinct tensor indices; fix the sign of $\omega$ to match line 75; delete "unique".
- Required new assumptions/permits: a pull-back metric $G_{\mathcal{Q}}=E_\phi^*G$ on $\mathcal{Q}$ (requires $E_\phi$ to be an immersion where used).
- Validation plan: Type-check every map in the statement (domain/codomain); confirm the code's chart counts are admissible under the restated constraint; confirm the sign of $\omega$ is consistent across lines 75 and 271.

### [E-007] Motor Texture Firewall contradicts the motor boundary condition, which feeds texture into the bulk density (was F-007)
- Location: Definition Neumann BC (line 125) versus Axiom Motor Texture Firewall (lines 403-419)
- Severity: Moderate
- Type: Definition mismatch (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter
- Claim (line 125): "$j_{\text{motor}} = D_A(u_\pi) = \text{Decoder}(z, u_\pi, z_{\text{tex,motor}})$"; line 409: "$\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0$."
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:559`: "$\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0$"; `:552` (texture exists only at the interface).
- Why this is an error: The chapter says the WFR equation governs the bulk belief density $\rho$ with the motor datum $j_{\text{motor}}$ (lines 119, 555). Since $j_{\text{motor}}$ depends on $z_{\text{tex,motor}}$, the bulk solution $\rho$, hence its mean flow $\dot z$ and everything the policy reads, is a functional of motor texture, violating the firewall axiom. The upstream visual firewall avoids this because texture is sampled only at $z_{\text{final}}$, after the bulk trajectory stops; the motor construction here injects texture into a boundary datum that drives the bulk PDE.
- Impact on downstream results: Node 32 (MotorTextureCheck) measures a leak the chapter's own definitions guarantee.
- Fix guidance:
  1. Define $j_{\text{motor}}$ as a function of $(z,u_\pi,z_{n,\text{motor}})$ only (texture-free).
  2. Sample $z_{\text{tex,motor}}$ downstream of the flux: $a_{\text{raw}} = D_A(z,u_\pi,z_{n,\text{motor}}) + \text{texture}$, with the boundary datum computed from the texture-free part.
- Required new assumptions/permits: none.
- Validation plan: Confirm $\partial_{z_{\text{tex,motor}}} j_{\text{motor}} = 0$ symbolically from the revised definition; Node 32 should then be a genuine test rather than a guaranteed failure.

### [E-008] Cycle is called a Stirling cycle but contains an isentropic leg and is bounded by a Carnot efficiency (was F-008)
- Location: Definition Cycle Phases, Remark (line 453) with table row line 450; Proposition Carnot (line 508)
- Severity: Minor
- Type: Conceptual
- Criterion: External
- Origin: this chapter
- Claim (line 453): "This cycle is structurally analogous to a Stirling cycle in thermodynamics." Line 450, Phase II: "$\Delta S = 0$ (isentropic)".
- Upstream anchor: not applicable; defined in this chapter.
- Why this is an error: A Stirling cycle consists of two isothermal and two isochoric (regenerative) legs and has no isentropic leg. The described cycle (compression, isentropic evolution, expansion) is Carnot/Otto-like, and the chapter itself then invokes a Carnot bound. The analogy is internally inconsistent.
- Impact on downstream results: None mathematical; the analogy motivates Proposition Carnot.
- Fix guidance:
  1. Replace "Stirling" by "Carnot-like (isentropic dreaming leg)" or drop the specific cycle name.
- Required new assumptions/permits: none.
- Validation plan: Read-through of the cycle section for consistency with the chosen analogy.

### [E-009] "Perception as Compression" and "Action as Expansion" are labelled theorems but contain no proof, and their entropy claims are inconsistent with the chapter (was F-009)
- Location: Theorems Perception as Compression and Action as Expansion (lines 462-485); Definition Cycle Phases table (line 451)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 462, 478, 485): "$W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0$"; "$W_{\text{expand}} = T_c \cdot I(K^{\text{act}}_t; K_t) \geq 0$"; "*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs."
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:552` (texture "exists only at the interface"); this chapter line 409 (Motor Texture Firewall).
- Why this is an error: (i) Neither statement is proved or derived from a framework result; the identification of a "work" with $T_c\times$ mutual information is a Landauer-type postulate and should be an axiom or definition, not a theorem (only the trivial $\ge0$ follows from $I\ge0$). (ii) The table (line 451) says action injects $I(A;K)$, the theorem uses $I(K^{\text{act}};K)$, and Node 34 (line 1189) uses $I(K^{\text{act}};c)$: three different quantities. (iii) The claim that the bulk entropy increases during action (line 450) is justified by injecting stochastic texture into motor outputs, but by the firewall (line 409) and the upstream partition axiom texture lives only at the interface and does not enter the bulk; the entropy of the emitted action is not the entropy of the belief state. The stage-1 unit objection ($T_c\cdot I$ in nat$^2$) is withdrawn: upstream `04_equations_motion.md:515` declares $T_c$ "nat (dimensionless in natural units where $k_B = 1$)" and `02_reward_field.md:620` writes $[T_c S]=\mathrm{nat}$, so the book's convention treats $T_c\cdot(\text{nat})$ as nat.
- Impact on downstream results: Proposition Carnot and Node 33 (ThermoCycleCheck) use $W_{\text{compress}}$, $W_{\text{expand}}$.
- Fix guidance:
  1. Relabel the two results as definitions (or one axiom, "Information-work correspondence").
  2. Use a single action variable consistently across the table, the theorem and Node 34.
  3. Either drop the $\Delta S_{\text{bulk}}>0$ claim or define $S$ as the entropy of the joint (bulk + interface) state.
- Required new assumptions/permits: the information-work correspondence as an explicit axiom.
- Validation plan: Confirm that every occurrence of $W_{\text{compress}}$, $W_{\text{expand}}$ and the cycle table uses the same mutual informations.

### [E-010] "Dreaming as Unitary/Isentropic Evolution" contradicts the book's definition of dreaming as thermal Langevin dynamics and this chapter's own reflective-WFR dreaming (was F-010)
- Location: Definition Dreaming as Unitary Evolution (lines 494-505); Definition Dreaming: Reflective Boundary (lines 569-574)
- Severity: Moderate
- Type: Conceptual (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 494-505): "$\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0$ ... (BAOAB integrator with $\gamma \to 0$). *Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange."
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:308`: "**Unconditional (Dreaming)** | $u_\pi = 0$ | Pure thermal fluctuation selects direction"; `:321`: "The policy contributes nothing; only thermal noise picks a direction."; `04_equations_motion.md:225`: noise term "$\sqrt{2\gamma T_c}\,(G^{1/2})_{kj}\,dW^j_s$".
- Why this is an error: (i) Upstream, dreaming is $u_\pi=0$ with dynamics driven by thermal noise at temperature $T_c$, a dissipative, entropy-producing Langevin process, not an isentropic Hamiltonian flow. Taking $\gamma\to0$ also kills the noise $\sqrt{2\gamma T_c}$, so "BAOAB with $\gamma\to0$" is incompatible with "pure thermal fluctuation selects direction". (ii) Within this chapter, lines 569-574 define dreaming as the WFR flow with reflective condition, reaction term and "Dynamics are driven purely by the internal potential $V_{\text{critic}}(z)$", an unbalanced gradient-type flow that is neither Hamiltonian nor isentropic. (iii) $\rho$ in the Liouville equation is a phase-space density on $T^*\mathcal{Z}$ while $\rho$ in the WFR definitions is a density on $\mathcal{Z}$; the same symbol denotes both. (iv) "Internal planning proceeds" is at odds with $u_\pi=0$.
- Impact on downstream results: Definition Cycle Phases ($\Delta S=0$ row); Proposition Carnot; the Waking/Dreaming narrative cited by `04_equations_motion.md:217`.
- Fix guidance:
  1. Choose one model of dreaming. If it is the upstream thermal mode, replace "isentropic/unitary" with "closed (no boundary flux) but thermal: $\Delta S_{\text{bulk}}\ge0$ with production $\propto\gamma T_c$" and drop the Liouville equation.
  2. If an isentropic leg is desired, define it as the $T_c\to0$, $\gamma\to0$ limit and say so explicitly.
  3. Distinguish the phase-space density $\varrho(z,p)$ from $\rho(z)$ notationally.
- Required new assumptions/permits: none beyond stating which limit is taken.
- Validation plan: Check the cycle table's entropy row against the chosen dreaming model; verify that `03_holographic_gen.md:308` and this chapter agree.

### [E-011] Carnot Efficiency Bound is unproved, uses undefined temperatures, and is violated by a simple in-framework example (was F-011)
- Location: Proposition Carnot Efficiency Bound (lines 508-519)
- Severity: Moderate
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (line 514): "$\eta = \frac{I(K^{\text{act}}_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}}$, where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries."
- Upstream anchor: $T_{\text{sensor}}$, $T_{\text{motor}}$ appear nowhere else in `docs/source/1_agent` (grep; only lines 514-519 of this chapter); only $T_c$ is defined (`05_geometry/04_equations_motion.md:507`).
- Why this is an error: (i) No proof is given and no framework result relates $I(K^{\text{act}};K)/I(X;K)$ to boundary temperatures; the two temperatures are never defined. (ii) The implied $\eta\le1$ (for $T_{\text{motor}}\ge0$) is false in general: the data-processing inequality for $X\to K\to K^{\text{act}}$ bounds $I(X;K^{\text{act}})$, not $I(K;K^{\text{act}})$. Counterexample (recomputed by the verifier): $X\sim\text{Bern}(1/2)$, $K$ = $X$ flipped with probability $0.4$, $K^{\text{act}}:=K$ (a legitimate deterministic policy). Then $I(X;K)=1-H_2(0.4)=0.02905$ bit and $I(K^{\text{act}};K)=H(K)=1$ bit, so $\eta=34.42$. The bound fails for every $T_{\text{motor}}\ge0$. (iii) The interpretation that $\eta=1$ requires $T_{\text{motor}}=0$ or $T_{\text{sensor}}\to\infty$ is likewise unsupported.
- Impact on downstream results: Node 33 remedy ("check Carnot efficiency bound"); the surrounding prose (lines 439-441) presents the bound as a law.
- Fix guidance:
  1. Either prove a genuine bound with the correct chain, e.g. $I(X;K^{\text{act}})\le\min\{I(X;K),\,I(K;K^{\text{act}})\}$ (data processing), and define $\eta:=I(X;K^{\text{act}})/I(X;K)\le1$; or
  2. Define the boundary temperatures (e.g. via noise variances $\sigma^2_{\text{motor}}$, $\sigma^2_{\text{sense}}$ of Gaussian channels) and derive the temperature form as a capacity ratio; otherwise
  3. Demote the statement to a conjecture.
- Required new assumptions/permits: definitions of $T_{\text{sensor}}$ and $T_{\text{motor}}$ if option 2 is taken.
- Validation plan: Test the restated bound on the binary counterexample above and on a Gaussian channel pair.

### [E-012] Waking and dreaming are defined by the policy switch, not by the boundary condition; a third regime appears without a mode (was V-001)
- Location: Waking/Dreaming definitions (lines 541-566); Theorem WFR Mode Switching (lines 583-586); Proposition Grounding Rate (lines 599-601)
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 541): "The distinction between Waking and Dreaming is rigorously defined by the **boundary condition** on $\rho$." Line 546: "During waking ($u_\pi \neq 0$) ..."; line 566: "During dreaming ($u_\pi = 0$), the sensory stream is cut." Lines 599-601 list three regimes: waking, dreaming, and "pure actuation (net information outflow to motors)".
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:308` (dreaming is $u_\pi=0$).
- Why this is an error: The definitions key the modes to $u_\pi$ (consistent with upstream) while the text claims the boundary condition is the defining datum. The two criteria disagree on the pure-actuation regime (sensors reflective, motors clamped): it is neither row of the two-mode theorem at lines 583-586, and having $u_\pi\neq0$ it would be classified as "waking" by line 546, although its sensory boundary is closed.
- Impact on downstream results: Theorem WFR Mode Switching (incomplete case split); Proposition Grounding Rate (sign discussion refers to a regime the mode table lacks).
- Fix guidance:
  1. Define modes by the pair (sensor BC, motor BC).
  2. Add a third row "Pure actuation" to the mode table, or drop that regime from the proposition.
  3. State $u_\pi=0$ as a consequence of the dreaming boundary condition rather than as its definition.
- Required new assumptions/permits: none.
- Validation plan: Confirm that every regime named in the grounding-rate proposition appears in the mode table and is classified identically by the boundary-condition criterion and by $u_\pi$.

### [E-013] Grounding rate is redefined in conflict with its upstream definition (was F-012)
- Location: Proposition Grounding Rate via Boundary Flux (lines 592-601)
- Severity: Moderate
- Type: Definition mismatch (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 592-601): "The grounding rate (cf. Definition 16.1.1) is: $G_t = \oint_{\partial\mathcal{Z}_{\text{sense}}} j_{\text{obs}} \cdot dA - \oint_{\partial\mathcal{Z}_{\text{motor}}} j_{\text{motor}} \cdot dA$ ... Positive during waking ... Zero during dreaming ... Negative during pure actuation".
- Upstream anchor: `docs/source/1_agent/04_control/03_coupling_window.md:82-90`: "Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step: $\lambda_{\text{in}} := \mathbb{E}[G_t]$. Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$."
- Why this is an error: Upstream $G_t := I(X_t;K_t)\ge0$ is a sensory-only quantity and the grounding rate is $\lambda_{\text{in}}=\mathbb{E}[G_t]$. This chapter redefines $G_t$ as a signed net sensor-minus-motor flux that is "negative during pure actuation", impossible for a mutual information, and calls it "the grounding rate" (not $\lambda_{\text{in}}$). It is presented as a proposition but is an incompatible new definition with no proof relating it to $I(X_t;K_t)$. "Definition 16.1.1" is a stale hard-coded number; the label is `def-grounding-rate`.
- Impact on downstream results: The coupling-window theorem (`04_control/03_coupling_window.md`) and Node 13 use the upstream sign-definite $G_t$; applying this chapter's signed version to those results gives wrong conclusions.
- Fix guidance:
  1. Rename the quantity (e.g. net boundary information flux $\Phi^{\text{net}}_t$).
  2. State $\mathbb{E}[\oint j_{\text{obs}}\,dA] = \lambda_{\text{in}}$ as the link to the upstream definition, with a proof or an explicit modelling assumption.
  3. Replace "Definition 16.1.1" by `{prf:ref}\`def-grounding-rate\``.
- Required new assumptions/permits: the modelling assumption relating $\oint j_{\text{obs}}\,dA$ to $I(X_t;K_t)$.
- Validation plan: Confirm the renamed quantity is never substituted for $G_t$ or $\lambda_{\text{in}}$ in cross-references to the coupling-window results.

### [E-014] Stale hard-coded cross-reference numbers (was F-021)
- Location: line 592 ("Definition 16.1.1"); lines 753, 760, 783, 864, 884, 965, 971, 977, 997, 1010, 1014 ("Definition 23.x.y" in code comments and docstrings); lines 1095, 1115 ("Definition 23.2", "Definition 23.1")
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (line 592): "The grounding rate (cf. Definition 16.1.1)"; line 884: "Definition 23.6.4: Context-conditioned policy for unified task handling."
- Upstream anchor: `docs/source/1_agent/04_control/03_coupling_window.md:83` (`:label: def-grounding-rate`).
- Why this is an error: The book uses MyST labels; these numeric references are not generated and do not match the current numbering even internally. By order of appearance, the fourth numbered item of the context section is Definition Context-Conditioned WFR (line 682), not a "context-conditioned policy", so "Definition 23.6.4" points at nothing; "Definition 16.1.1" refers to numbering that does not exist in the source. The stage-1 item "Algorithm 22.4.2 (BAOAB)" (line 1083) is withdrawn: it matches the hard-coded title "**Algorithm 22.4.2 (Full Geodesic BAOAB with Jump Step).**" at `05_geometry/04_equations_motion.md:596`, so it is merely hard-coded, not stale.
- Impact on downstream results: Reader navigation only.
- Fix guidance:
  1. Replace numeric references in prose with `{prf:ref}` labels (`def-grounding-rate`, `def-baoab-splitting`, `def-context-space`, etc.).
  2. Drop the numbers from code comments and docstrings, or replace them with label names.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for `Definition [0-9]` and `Algorithm [0-9]` after the edit.

### [E-015] The Context Space motor distribution is a softmax of an $a$-independent quantity (was F-013)
- Location: Definition Context Space (line 640); Definition Context Instantiation Functor, RL row (line 653); Theorem Universal Context Structure item 3; code docstring (line 923) versus code (line 932)
- Severity: Moderate
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 640): "$\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \Phi_{\text{eff}}(z, K, c)\right)$"; RL row (line 653): "$\Phi_{\text{eff}} = V_{\text{critic}}(z, K)$".
- Upstream anchor: `docs/source/1_agent/05_geometry/04_equations_motion.md:423`: "$\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z)$" (a function of $(z,K)$ only); `:512`: "$\pi(a|z) \propto \exp(Q(z,a)/T_c)$".
- Why this is an error: $\Phi_{\text{eff}}(z,K,c)$ does not depend on the action $a$, so normalising $\exp(-\Phi_{\text{eff}}/T_c)$ over $a$ yields the uniform distribution for every $z,c$; the definition carries no information about which action to take. The book's policy is a softmax over an action-dependent $Q(z,a)$. The classification and LLM rows survive only because their "potential" $-\log p(y|z)$ depends on the output $y$; the RL row does not. The `ContextConditionedPolicy` code (line 932) uses an action-dependent logit vector, contradicting the formula in its own docstring (line 923).
- Impact on downstream results: Theorem Universal Context Structure item 3, Corollary Prompt = Action = Label, the ContextConditionedPolicy code.
- Fix guidance:
  1. Define an action-dependent potential: $\Phi_{\text{eff}}(z,K,c;a) := -Q_c(z,a)$ (RL), $-\log p(y|z,c)$ (classification), $-\log p(\text{tok}|z,c)$ (LLM).
  2. Write $\pi(a|z,c)\propto\exp(-\Phi_{\text{eff}}(z,K,c;a)/T_c)$ and align the docstring with the code.
- Required new assumptions/permits: none.
- Validation plan: Check that the RL row's normalised distribution is non-uniform for a non-constant $Q$.

### [E-016] Sign error: with $\Phi_{\text{eff}}=V_{\text{critic}}$ the RL policy $\exp(-\Phi_{\text{eff}}/T_c)$ prefers low value and the context-conditioned velocity vanishes (was F-014)
- Location: Definition Context Instantiation Functor, RL row (line 653); Definition Context Space (line 640); Definition Context-Conditioned WFR (line 692); forward reference (line 712); Summary table
- Severity: Major
- Type: Parameter inconsistency (secondary: Conceptual)
- Criterion: Framework
- Origin: upstream `docs/source/1_agent/05_geometry/04_equations_motion.md` (`def-effective-potential`, sign of $V_{\text{critic}}$) and this chapter, which inherits the sign and makes it explicit
- Claim (line 653): "| **RL** | Action space $\mathcal{A}$ | Motor command (torques) | $V_{\text{critic}}(z, K)$ |"; line 640: "$\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \Phi_{\text{eff}}(z, K, c)\right)$"; line 692: "$v_c(z) = -G^{-1}(z) \nabla_z \Phi_{\text{eff}}(z, K, c) + u_\pi(z, c)$".
- Upstream anchor: `04_equations_motion.md:423` ("$\Phi_{\text{eff}} = \alpha U + (1-\alpha)V_{\text{critic}} + \gamma_{risk}\Psi_{\text{risk}}$"), `:227` (force "$-\partial_k\Phi_{\text{eff}}$"), `:406` ("$V_{\text{critic}}$ ... tells you how good a state is in terms of expected future reward"), `:443` ("The value term pulls it toward high-reward regions"), `:459` ("Pure Control | $\alpha=0$ | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)"), `:512` ("$\pi(a|z) \propto \exp(Q(z,a)/T_c)$"); `03_holographic_gen.md:306` ("$u_\pi = G^{-1}\nabla_z V_{\text{critic}}$ | Points toward high-value regions"), `:270` ("$u_\pi(z,c) = G^{-1}(z)\cdot\nabla_z\Phi_{\text{eff}}(z,K,c)$"); `02_reward_field.md:675` ("$r(z) = \frac{1}{s_r}(V(z) - \bar V)$"); `10_appendices/05_proofs.md:825` ("$\pi(a|z) = \frac1Z\exp(Q(z,a)/T_c)$").
- Why this is an error: Throughout the book $V_{\text{critic}}$ is a value (higher is better): the policy is $\exp(+Q/T_c)$, the control field points up $\nabla V$, and mass grows where $V$ is high. The upstream $\Phi_{\text{eff}}$ already carries $+V_{\text{critic}}$ inside a potential the dynamics descend while every verbal and policy statement treats $V$ as a value to ascend, so the upstream source is itself inconsistent (the finding is not a false positive against a consistent upstream). This chapter makes the problem explicit and worse: (a) the RL row sets $\Phi_{\text{eff}}=V_{\text{critic}}$, so the policy (line 640) is $\exp(-V_{\text{critic}}/T_c)$, the reciprocal of the book's Boltzmann policy, and the agent prefers low value, opposite to the classification/LLM rows where $-\log p$ is a genuine cost; (b) line 692 defines $v_c = -G^{-1}\nabla_z\Phi_{\text{eff}} + u_\pi(z,c)$, and with the RL row and the cited `thm-unified-control-interpretation` ($u_\pi=G^{-1}\nabla V_{\text{critic}}$, cited at line 712) one obtains $v_c = -G^{-1}\nabla V + G^{-1}\nabla V \equiv 0$: the context-conditioned RL velocity vanishes identically. The chapter also strips the $(1-\alpha)$, $U$, $\Psi_{\text{risk}}$ terms, so the sign error is the entire content of the RL row.
- Impact on downstream results: Definition Context-Conditioned WFR ($v_c$), Theorem Universal Context Structure, Corollary Prompt = Action = Label, the Summary table, the "Value Creates Mass" forward reference (which needs mass to accumulate in high-value regions), and the code comment "logits (negative effective potential)".
- Fix guidance:
  1. In this chapter write the RL potential as $-V_{\text{critic}}(z,K)$ (or $-Q_c(z,a)$, cf. E-015) so that $\exp(-\Phi_{\text{eff}}/T_c)\propto\exp(Q/T_c)$ and $-G^{-1}\nabla\Phi_{\text{eff}} = +G^{-1}\nabla V$ agree with `thm-unified-control-interpretation`.
  2. Re-derive $v_c$ and confirm it is non-degenerate in the RL row.
  3. Flag upstream `def-effective-potential` for the coordinated sign fix ($-(1-\alpha)V_{\text{critic}}$), or redefine $V_{\text{critic}}$ there as a cost-to-go; update `03_holographic_gen.md:270` accordingly.
- Required new assumptions/permits: none; a coordinated edit of `def-effective-potential` is required.
- Validation plan: Symbolically check that the RL-row $v_c$ is nonzero and points along $+\nabla V$; check the sign of the policy exponent against `05_proofs.md:825`.

### [E-017] Universal Context Structure: dimension mismatch in the symmetry-breaking kick and a proof that does not address the statement (was F-015)
- Location: Theorem Universal Context Structure (lines 665-679)
- Severity: Minor
- Type: Dimensional mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 665-672): "**Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ ... $u_\pi(0) = G^{-1}(0) \cdot e_c = \frac{1}{4} e_c$ (at the Poincare disk origin where $G(0) = 4I$)". Line 679: "*Proof.* The holographic expansion ... is invariant to the interpretation of the control field $u_\pi$."
- Upstream anchor: `docs/source/1_agent/05_geometry/03_holographic_gen.md:239` ("$G(0) = 4I$"), `:263` ($u_\pi\in T\mathbb{D}$, dimension $d_z$), `:307` ("$u_\pi = G^{-1}\cdot\text{embed}(\text{prompt})$").
- Why this is an error: $G^{-1}(0)$ is a $d_z\times d_z$ matrix and $u_\pi(0)\in T_0\mathcal{Z}\cong\mathbb{R}^{d_z}$, but $e_c\in\mathbb{R}^{d_c}$ with $d_c$ independent (`context_dim: int = 64` versus `latent_dim: int = 32`, lines 771, 776); the code inserts a learned `context_encoder: context_dim -> latent_dim` (lines 899-903) that the theorem omits. The product is undefined unless $d_c=d_z$. The proof argues only that bulk dynamics ignore the semantic interpretation of $u_\pi$; it does not establish items 1-3, which are design stipulations rather than consequences. The value $\tfrac14 e_c$ is itself consistent with upstream $G(0)=4I$.
- Impact on downstream results: Node 34 remedy ("check symmetry-breaking kick").
- Fix guidance:
  1. Insert an explicit embedding $\iota_c:\mathbb{R}^{d_c}\to T_0\mathcal{Z}$ and write $u_\pi(0)=G^{-1}(0)\,\iota_c(e_c)$.
  2. Relabel the theorem as a definition (Universal Context Interface), or state and prove a concrete invariance claim.
- Required new assumptions/permits: the map $\iota_c$ (learned, as in the code).
- Validation plan: Dimension check of $u_\pi(0)$ against $T_0\mathcal{Z}$ for $d_c\neq d_z$.

### [E-018] Corollary "Prompt = Action = Label" asserts an isomorphism without defining the category or proving it (was F-016)
- Location: Corollary Prompt = Action = Label (lines 699-702); Definition Context Instantiation Functor (line 649)
- Severity: Note
- Type: Proof gap / omission
- Criterion: Framework
- Origin: this chapter
- Claim (lines 699-702): "The following are isomorphic as boundary conditions on $\partial\mathcal{Z}$: RL Action $\cong$ Classification Label $\cong$ LLM Prompt."
- Upstream anchor: not applicable; defined in this chapter.
- Why this is an error: No structure on "boundary conditions" is specified for which "isomorphic" is meaningful; the three spaces differ in cardinality and topology (continuous torque space, finite label set, discrete sequences), so no isomorphism of sets or manifolds exists. What holds is that each induces a context vector $e_c$ and hence the same downstream dynamics (cf. E-017). The "functor $\mathcal{I}:\mathbf{Task}\to\mathcal{C}$" (line 649) is likewise undefined, since $\mathcal{C}$ is a manifold, not a category.
- Impact on downstream results: Transfer-learning remark only.
- Fix guidance:
  1. Replace the corollary with "each factors through the common interface $c\mapsto e_c\mapsto u_\pi(0)$".
  2. Drop the functor language or define $\mathbf{Task}$ and $\mathcal{C}$ as categories.
- Required new assumptions/permits: none.
- Validation plan: Read-through for remaining uses of "isomorphic" and "functor" in the context section.

### [E-019] Code: the "VQ from selected chart" never quantises; it always returns code index 0 (was F-017)
- Location: Implementation, `DualAtlasEncoder.forward` (lines 840-843)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (lines 840-843): `# VQ from selected chart / z_macro = torch.stack([ self.codebooks[c.item()](torch.zeros(1, dtype=torch.long, device=x.device)).squeeze() for c in chart_idx ])`
- Upstream anchor: The module claims (line 783) to implement Definitions Visual/Action Atlas of this chapter (lines 205-233), whose codebooks are "Discrete macro codes", and to extend the AttentiveAtlasEncoder, whose per-chart VQ is a nearest-code lookup (`docs/source/1_agent/03_architecture/02_disentangled_vae.md:120-129`).
- Why this is an error: `torch.zeros(1, dtype=long)` is the index tensor `[0]`, so entry 0 of the chosen chart's codebook is returned for every input; `codes_per_chart` and the features `h` play no role, no distance-based quantisation occurs, and `z_macro` carries only the chart identity. The returned triple is therefore not the $(K, z_n, z_{\text{tex}})$ decomposition the docstring claims. (The per-sample Python loop is also unbatched; the `.squeeze()` concern only bites for `latent_dim == 1` and is not load-bearing.)
- Impact on downstream results: Any reader treating Algorithm 23.7.1 as the reference implementation of the Dual Atlas.
- Fix guidance:
  1. Compute distances between a projected feature (e.g. `self.nuisance_head(h)` or a dedicated `pre_vq` projection) and `self.codebooks[c].weight`.
  2. Take `argmin`, gather the code, and apply a straight-through estimator, as in the AttentiveAtlasEncoder.
  3. Vectorise over the batch.
- Required new assumptions/permits: none.
- Validation plan: Unit test that distinct inputs in the same chart can produce distinct code indices.

### [E-020] Code: forward pass discards the policy output and uses the context embedding as the intention; state is nuisance only; texture slice can break (was F-018)
- Location: Implementation, `HolographicInterface.forward_actuation` / `forward` (lines 1015-1072)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 1058 `z = vis_out['z_nuisance']  # Use nuisance as state`; line 1061 `policy_out = self.policy(z, context, self.config.T_c)`; line 1072 `u_intent = policy_out['context_embedding']`; lines 1015-1024 `sample_motor_texture(z, self.config.latent_dim, ...)` then `a_raw = a_base + z_tex_motor[:, :self.config.action_dim]`.
- Upstream anchor: This chapter, Definition Action Atlas (line 231): "*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy ...)"; Definition Holographic Shutter ($D_A: T_z\mathcal{Z}\times\mathcal{Z}\to T^*\mathcal{Q}$); Axiom Motor Texture Firewall (line 412): policy operates on $(K, z_n, A, z_{n,\text{motor}})$.
- Why this is an error: (i) `ContextConditionedPolicy` computes `logits`/`probs` over `action_dim` (lines 932-935), but `forward` never consumes them: the intention fed to the decoder is `context_embedding`, a function of the context alone, independent of $z$ and of the policy network, so the actuation path does not implement "Intention from Policy" and the $T_c$-softmax is dead code. (ii) The bulk state passed to the policy and decoder is `z_nuisance` only; $K$ (`z_macro`, `chart_idx`) is dropped, contradicting the firewall statement that the policy acts on $(K,z_n,\dots)$. (iii) `sample_motor_texture` returns `[B, latent_dim]` and is sliced to `action_dim`; with `action_dim > latent_dim` the addition to `a_base` (`[B, action_dim]`) fails to broadcast; with `action_dim < latent_dim` (defaults 8 < 32) texture dimensions are silently dropped. Motor texture is also added in raw action units although its variance is set by the latent conformal factor.
- Impact on downstream results: Anyone using this module as the reference for the belief-evolution cycle (Phase III) gets an actuation that ignores the policy.
- Fix guidance:
  1. Produce the intention from `policy_out` (e.g. expected action $\mathbb{E}_{a\sim\pi}[a]$ mapped through a `latent_dim` head, matching `def-the-control-field` $u_\pi=G^{-1}\mathbb{E}[a]$).
  2. Feed `torch.cat([z_macro, z_nuisance])` (or the full $z$) as the state.
  3. Sample texture with `d_motor_tex = action_dim`, or project it to action space.
- Required new assumptions/permits: none.
- Validation plan: Assert that changing the policy weights changes `a_raw`; shape tests for `action_dim` both larger and smaller than `latent_dim`.

### [E-021] BAOAB is claimed to preserve the symplectic form; a Langevin (thermostatted) integrator does not (was F-019)
- Location: Connection to RL #7 (lines 1089-1112)
- Severity: Minor
- Type: Conceptual
- Criterion: External
- Origin: this chapter (same overstatement in `docs/source/1_agent/10_appendices/04_faq.md:466`)
- Claim (lines 1089-1092): "The BAOAB algorithm ... preserves the symplectic structure: $\hat{z}_{t+1} = \Phi_{\text{BAOAB}}(z_t)$ with $\omega(\Phi_* X, \Phi_* Y) = \omega(X, Y)$."
- Upstream anchor: `docs/source/1_agent/05_geometry/04_equations_motion.md:571-576`: O-step "$p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$ ... $c_1 = e^{-\gamma h}$"; `:264`: the scheme "preserves the Boltzmann distribution to $O(h^2)$" (no symplecticity claim).
- Why this is an error: The O-step is an Ornstein-Uhlenbeck thermostat, dissipative and stochastic; for $c_1<1$ it contracts momentum and adds noise, so the composite map is not a symplectomorphism and does not preserve phase-space volume. The "Liouville's theorem" bullet (line 1112) therefore does not apply to the algorithm actually used; only the deterministic B-A-B sub-steps ($\gamma=0$) are symplectic (and the geodesic/Lorentz corrections as implemented are not guaranteed to be). Upstream correctly claims only Boltzmann preservation. The map is also written on $z$ alone whereas $\omega$ lives on $(z,p)$.
- Impact on downstream results: Rhetorical (Dreamer comparison); the FAQ repeats the claim.
- Fix guidance:
  1. Replace by "BAOAB is a splitting of a symplectic (BAB) part and an exact OU thermostat; it preserves the Gibbs measure to $O(h^2)$ and, in the $\gamma\to0$ limit, is symplectic".
  2. Write the map on $(z,p)$.
  3. Apply the same correction to `04_faq.md:466`.
- Required new assumptions/permits: none.
- Validation plan: Numerically check $\det J$ of one BAOAB step for $\gamma>0$ (expected $e^{-\gamma h\,d}$, not 1).

### [E-022] Diagnostic proxies for Nodes 30, 31, 33 are ill-typed or contradict the chapter's own results (was F-020)
- Location: Summary Tables and Diagnostic Nodes, Nodes 30, 31, 33 (lines 1145-1178)
- Severity: Minor
- Type: Dimensional mismatch (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim: Node 30 proxy (line 1145) "$\lVert\omega(j_{\text{sense}}, j_{\text{motor}})\rVert$"; Node 31 proxy (line 1156) "$\lVert e_\alpha^{\text{vis}} - \mathcal{L}(e_\beta^{\text{act}})\rVert^2$"; Node 33 (line 1178) "Is perception/action thermodynamically balanced? $\lvert W_{\text{compress}} - W_{\text{expand}}\rvert$".
- Upstream anchor: not applicable; all objects are defined in this chapter (lines 75, 125, 130, 252, 462, 478, 514, 519).
- Why this is an error: (i) $\omega$ is a 2-form taking two tangent vectors of the interface phase space, but $j_{\text{sense}}$, $j_{\text{motor}}$ are scalar flux densities (line 130), so $\omega(j_{\text{sense}},j_{\text{motor}})$ is undefined. (ii) Theorem Atlas Duality (line 252) says $\mathcal{A}_{\text{act}}=\mathcal{L}(\mathcal{A}_{\text{vis}})$ (visual to action), so the aligned pair is $(\mathcal{L}(e^{\text{vis}}_\alpha), e^{\text{act}}_\beta)$, not $(e^{\text{vis}}_\alpha,\mathcal{L}(e^{\text{act}}_\beta))$; moreover $\mathcal{L}:T\mathcal{Q}\to T^*\mathcal{Q}$ is not defined on codebook vectors in $\mathbb{R}^{d_m}$, and the pairing $(\alpha,\beta)$ is unspecified. (iii) Proposition Carnot (line 519, "Real systems operate at $\eta<1$") mandates $W_{\text{expand}}<W_{\text{compress}}$, so "balance" $W_{\text{compress}}=W_{\text{expand}}$ is forbidden by the chapter's own bound; a diagnostic that fires on their difference conflicts with the remedy that cites the bound.
- Impact on downstream results: Sieve nodes 30, 31, 33 are not computable as specified.
- Fix guidance:
  1. Node 30: use a well-typed residual, e.g. the symplectic-area defect of paired boundary updates $|\omega(\delta q_{\text{obs}},\delta p_{\text{motor}})-\omega_0|$, or the Jacobian test $\|J^\top\Omega J-\Omega\|$.
  2. Node 31: $\lVert \mathcal{L}_{\theta}(e^{\text{vis}}_\alpha) - e^{\text{act}}_{\beta(\alpha)}\rVert^2$ with a learned or defined $\mathcal{L}_\theta:\mathbb{R}^{d_m}\to\mathbb{R}^{d_m}$ and an explicit pairing $\beta(\alpha)$.
  3. Node 33: monitor $\eta$ against a target band, or $|W_{\text{compress}}-W_{\text{expand}}-W_{\text{dissipated}}|$.
- Required new assumptions/permits: the map $\mathcal{L}_\theta$ and pairing $\beta(\alpha)$ for Node 31.
- Validation plan: Type-check each proxy's arguments against the definitions; confirm Node 33's pass condition is compatible with the (corrected) efficiency bound.

## Scope restrictions and clarifications
- Line numbers refer to the chapter as read during verification; the stage-1 reviewer's line citations were offset by 2-4 lines in several places and have been replaced.
- Findings against upstream `def-effective-potential` (E-016) record that the upstream definition is itself internally inconsistent (potential descended, value described as ascended); the upstream sign fix is flagged, not made, in this review.
- The stage-1 unit objection to $T_c\cdot I$ (part of F-009) was withdrawn because the book declares $T_c$ dimensionless in natural units; the remaining items of E-009 stand.
- The stage-1 item on "Algorithm 22.4.2" (part of F-021) was withdrawn because it matches a hard-coded title upstream; the remaining stale numbers in E-014 stand.
- External-criterion findings (E-005, E-008, E-021) are judged against standard mechanics, thermodynamics and PDE facts rather than against framework definitions.
- No Rejected findings: all 21 stage-1 findings were confirmed or adjusted, and one finding was added by the verifier.

## Open questions
- Which object is meant to carry $\omega$: $T^*\mathcal{Q}$, $T^*(\partial\mathcal{Z})$, or an interface phase space to be introduced (E-001)? The answer determines the fix for E-002, E-005 and Node 30 (E-022).
- Is dreaming intended to be the upstream thermal Langevin mode or an isentropic Hamiltonian limit (E-010)? This choice fixes the entropy row of the cycle table and the fate of Proposition Carnot (E-011).
- Should $V_{\text{critic}}$ be redefined upstream as a cost-to-go, or should `def-effective-potential` carry $-(1-\alpha)V_{\text{critic}}$ (E-016)? Either resolves this chapter's RL row, but the choice propagates to `03_holographic_gen.md:270, :306` and `02_reward_field.md:675`.
- Is $|\mathcal{K}_{\text{vis}}|=|\mathcal{K}_{\text{act}}|$ intended (E-006)? The code's 8 versus 4 suggests not, in which case the Legendre relation must be a compatibility constraint rather than an equality of atlases.

## Rejected candidate findings
- None. All stage-1 findings were confirmed or adjusted by the verifier.
