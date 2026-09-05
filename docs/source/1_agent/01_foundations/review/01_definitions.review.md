# Mathematical Review: docs/source/1_agent/01_foundations/01_definitions.md

## Metadata
- Reviewed file: docs/source/1_agent/01_foundations/01_definitions.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (522 lines)
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/02_control_loop.md`: symbol table (lines 25-36), sign convention (102-103), critic and covariant value gradient (141-147), two-level macro register and Attentive Atlas codebooks (247-253, 318-334), capacity bound and $I(X;K)=H(K)$ (480-487), entropy-regularised objective (497-503), Lyapunov constraint (583-598), cost 1-form and Helmholtz forward reference (1013-1018, 1063-1069), `def-local-conditioning-scale` (1278-1288)
  - `06_fields/02_reward_field.md`: reward 1-form and its units (75-88), `def-terminal-boundary` (115-125), Hodge decomposition and identification $\Phi=V$ (185-197), `thm-the-hjb-helmholtz-correspondence` (323-358), reward-convention statement (638)
  - `02_sieve/01_diagnostics.md`: node table (100-125), threshold units (131), projective value head and Lyapunov losses (516-553), Hodge-style alignment (256, 596-604)
  - `05_geometry/04_equations_motion.md`: value curl $\mathcal F_{ij}=\partial_i\mathcal R_j-\partial_j\mathcal R_i$ (234), `def-cognitive-temperature` (506-518), overdamped limit and `cor-recovery-of-holographic-flow` (895-951), summary table (1285-1295)
  - `05_geometry/03_holographic_gen.md`: `prop-isotropic-radial-expansion` (138-149)
  - `10_appendices/02_parameters.md`: parameter units table (40-48)
  - `08_multiagent/02_standard_model.md`: utility gauge (110-118)

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 3
- Minor: 8
- Notes: 1
- Primary themes:
  1. Sign and type drift around the reward 1-form and the critic. The chapter labels $V$ a cost-to-go and calls the reward pairing a "cost rate", yet Hodge-decomposes the reward 1-form as $\mathcal R=dV+A$. The two upstream anchors disagree on exactly this point: `02_control_loop.md` uses $\mathcal C=dV+A$, $\mathcal R=-dV-A$; `06_fields/02_reward_field.md` uses $\mathcal R=dV+A$. The conventions are related by the joint flip $(V,A,\mathcal F)\mapsto(-V,-A,-\mathcal F)$, so the shared symbols $A$, $\mathcal F$ and $\nabla_A V$ denote sign-opposite objects in the two chapters. This root chapter does not fix a convention.
  2. Symmetry group stated more generally than the objective allows. The positive-affine objective gauge is not a symmetry of the entropy-regularised objective with fixed $T_c$ or of episodic tasks with fixed terminal data; the symbol-permutation group is stated for a flat codebook although the register is two-level.
  3. A boundary flag identified with a latent subset. The exogenous termination signal $d_t$ is equated with hitting $\Gamma_{\text{term}}\subset\mathcal Z$ without the measurability hypothesis that makes the identification exact (the gap is shared with the upstream definition).
  4. Small index, domain, and units inconsistencies in the units and chronology sections (reward index and integral bounds, memory-time domain, $T_c$ versus $\beta_{\text{cpl}}$, scale time versus computation time, stale section numbers).
- The mechanical cross-reference pre-pass lists no dangling references or duplicate labels for this file.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Symbols table; Def. Boundary / Markov Blanket (28, 127) | Minor | Miswording | Framework | this chapter | "scalar in the conservative case" contradicts $r_t\in\mathbb R$ and $r_t=\langle\mathcal R,\dot z\rangle$ |
| E-002 | Def. Boundary / Markov Blanket; item 5 Termination (128, 272-276) | Moderate | Scope restriction (secondary: Definition mismatch) | Framework | upstream `06_fields/02_reward_field.md:118` (restated here) | Exogenous flag $d_t$ identified with a latent subset $\Gamma_{\text{term}}\subset\mathcal Z$ without a measurability hypothesis |
| E-003 | Item 2 Observation, boundary gate nodes (232) | Note | Conceptual | Framework | this chapter (phrase); statistic upstream `02_sieve/01_diagnostics.md:114` | $I(X;K)>0$ equals $H(K)>0$ for the deterministic shutter; "the only well-typed sense" overstates it |
| E-004 | Item 4 Reward; Discrete vs continuous reward (252-254, 380-382) | Moderate | Notation conflict (secondary: Dimensional mismatch, sign) | Framework | this chapter | $\mathcal R$ called a "cost rate" and given units $\mathrm{nat\,s^{-1}}$; upstream fixes $[\mathcal R]=\mathrm{nat}/[\text{length}]$ and $c_t=-r_t$ |
| E-005 | Item 4 Reward, Mechanism and Boundary interpretation (255-263); Units (373) | Major | Definition mismatch (secondary: Parameter inconsistency, sign convention) | Framework | upstream conflict `02_control_loop.md:147` vs `06_fields/02_reward_field.md:194-196`, inherited by this chapter | $V$ labelled cost-to-go while $dV$ is the exact part of the reward 1-form; the two anchors assign opposite signs to $V$, $A$, $\mathcal F$ |
| E-006 | Def. Agent symmetry group (308) | Minor | Notation conflict | Framework | this chapter | Symbol $a$ reused for the affine scale and for the action |
| E-007 | Def. Agent symmetry group (305-311, 331, 334, 343) | Moderate | Scope restriction (secondary: Parameter inconsistency) | Framework | this chapter | Positive-affine gauge is not a symmetry of the fixed-$T_c$ entropy-regularised objective nor of episodic tasks with fixed $V_{\text{term}}$ |
| E-008 | Def. Agent symmetry group; Principle of covariance (313, 342) | Minor | Scope restriction | Framework | this chapter | $S_{\lvert\mathcal K\rvert}$ stated for a flat codebook; the two-level register admits only $S_{N_v}\wr S_{N_c}$ |
| E-009 | Discrete vs continuous reward (382); cf. 148 | Minor | Dimensional mismatch (secondary: Notation conflict, index) | Framework | this chapter | Integral bounds mix a step index with seconds; the reward accrued over the step starting at $t$ is $r_{t+1}$ |
| E-010 | Regularization / precision coefficients (385-387) | Minor | Parameter inconsistency (units) | Framework | upstream `02_control_loop.md:1286`, `04_equations_motion.md:515` | $T_c$ dimensionless here, nat in `def-cognitive-temperature`, and $1/\beta_{\text{cpl}}$ with $[\beta_{\text{cpl}}]=\mathrm{nat}/[z]^2$ upstream |
| E-011 | Chronology table; Memory Time (415, 496) | Minor | Notation conflict (domain) | Framework | this chapter | Memory-time domain $\{t'\in\mathbb Z:t'<t\}$ vs $\{0,\dots,t-1\}$ |
| E-012 | Computation Time; Scale Time (445-453, 478-482) | Minor | Notation conflict (secondary: Conceptual) | Framework | upstream `05_geometry/04_equations_motion.md:944-951` | $\tau$ and $s$ declared orthogonal, but the cited $\tau$ law is the overdamped $s$-flow |
| E-013 | Scale Time (478) | Minor | Citation / reference error | Framework | this chapter | "Sections 21, 7.12" do not resolve |

## Detailed findings

### [E-001] "Scalar in the conservative case" qualifier contradicts $r_t\in\mathbb R$ (was F-001)
- Location: Symbols (Quick) table, line 28; Definition Boundary / Markov Blanket, line 127
- Severity: Minor
- Type: Miswording
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 127: "$r_t\in\mathbb{R}$ is the boundary reward sample (evaluation of the reward 1-form/flux; scalar in the conservative case)"; line 28: "Reward sample (boundary flux; scalar in conservative case)".
- Upstream anchor: `06_fields/02_reward_field.md:83`: "$R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t)\,dt$". Same chapter, line 253: "$r_t=\langle\mathcal{R},\dot{z}\rangle$".
- Why this is an error: the type declaration $r_t\in\mathbb R$ is unconditional, and the sample is the pairing of a 1-form with the latent velocity, a scalar whether or not $\mathcal R$ is exact. Exactness governs path-independence of the cumulative integral $\int_\gamma\mathcal R$, not the type of the integrand. The qualifier suggests a non-scalar reward channel in the non-conservative case, which the framework never defines.
- Impact on downstream results: none mathematically; invites a reader to look for a vector-valued reward channel.
- Fix guidance:
  1. Line 127: "$r_t\in\mathbb{R}$ is the boundary reward sample (the pairing $\langle\mathcal{R},\dot z\rangle$ of the reward 1-form along the step; its cumulative integral is path-independent only in the conservative case)".
  2. Line 28: "Reward sample (pairing of the boundary reward flux with the step)".
- Required new assumptions/permits: none.
- Validation plan: check that no other sentence in the volume conditions the scalar type of $r_t$ on exactness.

### [E-002] Exogenous termination flag $d_t$ identified with a latent subset $\Gamma_{\text{term}}\subset\mathcal Z$ (was F-002)
- Location: Definition Boundary / Markov Blanket, line 128; item 5 Termination, lines 272-276
- Severity: Moderate
- Type: Scope restriction (secondary: Definition mismatch)
- Criterion: Framework
- Origin: upstream `06_fields/02_reward_field.md:118-121` (the same identification is made there; this chapter restates it as a formal identity)
- Claim (verbatim): line 128: "$d_t\in\{0,1\}$ is termination (absorbing event / task boundary; corresponds to $\Gamma_{\text{term}}$ and $\tau_{\text{term}}$ in Definition def-terminal-boundary)"; lines 274-276: "Formally this is a terminal subset $\Gamma_{\text{term}}$ with stopping time $\tau_{\text{term}}$ and Dirichlet data $V|_{\Gamma_{\text{term}}}=V_{\text{term}}$".
- Upstream anchor: `06_fields/02_reward_field.md:118-121`: "Let $\Gamma_{\text{term}} \subset \mathcal{Z}$ denote the terminal subset representing end/death flags. Define the stopping time $\tau_{\text{term}} := \inf\{t \ge 0 : z_t \in \Gamma_{\text{term}}\}$ and kill the process upon hitting $\Gamma_{\text{term}}$. For the conservative value PDE, impose a Dirichlet condition $V|_{\Gamma_{\text{term}}} = V_{\text{term}}$ ... In WFR form, include a killing rate $\kappa_{\text{term}}(z) \ge 0$".
- Why this is an error: $d_t$ is an incoming boundary signal drawn from $P_\partial(x_{t+1},r_{t+1},d_{t+1},\iota_{t+1}\mid B_{\le t})$ (line 148), while $z_t\in\mathcal Z$ is "the agent's own construction" (line 113). Nothing in the definitions forces $\{t: d_t=1\}=\{t: z_t\in\Gamma_{\text{term}}\}$. Under partial observability and a fixed encoder, the same latent $z$ can be visited once with $d=0$ and once with $d=1$, so "the terminal subset" is not a well-defined subset of $\mathcal Z$, the hitting time $\tau_{\text{term}}$ differs from the first $t$ with $d_t=1$, and the Dirichlet condition has no well-posed domain. The identification holds exactly only when termination is $\sigma(Z_t)$-measurable, a hypothesis stated neither here nor upstream.
- Impact on downstream results: the Dirichlet data used in `thm-the-hjb-helmholtz-correspondence` and the termination handling of the value recursion presuppose $\Gamma_{\text{term}}\subset\mathcal Z$; the killing-rate form already offered upstream is the well-typed object under partial observability.
- Fix guidance:
  1. At line 274, replace "Formally this is a terminal subset" by "Modelled on the latent side as a terminal subset $\Gamma_{\text{term}}$ ... under the hypothesis that termination is $\sigma(Z_t)$-measurable".
  2. Either define $\Gamma_{\text{term}}$ as a model-side set (e.g. the level set $\{z:\bar P(d=1\mid z)\ge1-\epsilon\}$ of a learned termination head) and note the Dirichlet condition holds for that set, or state that in the general case the killing rate $\kappa_{\text{term}}(z):=\bar P(d=1\mid z)/\Delta t$ replaces the hard Dirichlet form.
  3. Flag `06_fields/02_reward_field.md:118` for the same hypothesis.
- Required new assumptions/permits: "termination is $\sigma(Z_t)$-measurable" wherever the hard Dirichlet form is used.
- Validation plan: verify that every use of $V|_{\Gamma_{\text{term}}}$ in 06_fields and 10_appendices either carries the hypothesis or uses the killing-rate form.

### [E-003] Node 13 "$I(X;K)>0$" is not a test of external signal under the deterministic quantiser (was F-003)
- Location: item 2 Observation, boundary gate nodes, line 232
- Severity: Note
- Type: Conceptual
- Criterion: Framework
- Origin: this chapter (phrase "the only well-typed sense"); statistic upstream `02_sieve/01_diagnostics.md:114`
- Claim (verbatim): "Node 13 (BoundaryCheck): the channel is open in the only well-typed sense: $I(X;K)>0$ (symbolic mutual information)."
- Upstream anchor: `02_sieve/01_diagnostics.md:114`: "13 | BoundaryCheck | VQ-VAE | Input Informativeness | External signal present at boundary? | $I(X;K)$ (Symbolic MI $>0$)"; `01_foundations/02_control_loop.md:487`: "For deterministic nearest-neighbor quantization, $H(K\mid X)=0$ and hence $I(X;K)=H(K)$".
- Why this is an error: with the deterministic VQ shutter, $K=K(X)$ and $I(X;K)=H(K)$ identically, so $I(X;K)>0$ holds iff more than one code is used, which Node 11 ($1-H(K)/\log\lvert\mathcal K\rvert$) already measures. Codes driven by sensor noise pass the check. Calling this "the only well-typed sense" of an open channel overstates what the statistic certifies.
- Impact on downstream results: diagnostic redundancy only; no theorem depends on it.
- Fix guidance:
  1. Drop "the only well-typed sense" and describe Node 13 as a non-collapse sanity condition, or
  2. Define Node 13 with a predictive quantity that is not a function of $H(K)$ alone (e.g. $I(K_t;X_{t+1}\mid K_{t-1})$).
- Required new assumptions/permits: none.
- Validation plan: none needed beyond wording.

### [E-004] $\mathcal R$ called a "cost rate" and assigned the units of its pairing (was F-004)
- Location: item 4 Reward, lines 252-254; Discrete vs continuous reward, lines 380-382
- Severity: Moderate
- Type: Notation conflict (secondary: Dimensional mismatch, sign)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 253: "In continuous time it appears as an instantaneous **cost rate** $r_t=\langle\mathcal{R},\dot{z}\rangle$"; line 381: "Per-step reward $r_t$ (or cost $c_t=-r_t$) has units $\mathrm{nat}$"; line 382: "A continuous-time cost rate $\mathcal{R}$ has units $\mathrm{nat\,s^{-1}}$ and links to discrete time by $r_t \approx \int_{t}^{t+\Delta t}\mathcal{R}(u)\,du$."
- Upstream anchor: `06_fields/02_reward_field.md:77`: "*Units:* $[\mathcal{R}] = \mathrm{nat}/[\text{length}]$"; `:83`: "$R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t) \, dt$"; `01_foundations/02_control_loop.md:35`: "$\mathcal{C}$ | Instantaneous cost 1-form (negative of the reward 1-form)".
- Why this is an error: (i) Sign. $\int_\gamma\mathcal R$ is the cumulative reward upstream, the chapter itself sets $c_t=-r_t$, and 02_control_loop defines the cost 1-form as $-\mathcal R$; naming $\langle\mathcal R,\dot z\rangle$ a cost rate flips the sign of the named quantity. (ii) Type and units. $\mathcal R$ is a 1-form with component units $\mathrm{nat}/[z]$ (stated upstream); the scalar in $\mathrm{nat\,s^{-1}}$ is the pairing $\langle\mathcal R,\dot z\rangle$. Line 382 assigns the 1-form the units of its pairing and integrates it against $du$ as a scalar function of time, while lines 253 and 258 use the same symbol for the 1-form.
- Impact on downstream results: this section is the units anchor cited by `02_sieve/01_diagnostics.md:131`; applying $[\mathcal R]=\mathrm{nat\,s^{-1}}$ to the 1-form gives wrong units for $A$, $\mathcal F=dA$ and $\beta_{\text{curl}}G^{-1}\mathcal F$.
- Fix guidance:
  1. Line 253: "instantaneous **reward rate** $\rho_t=\langle\mathcal{R},\dot{z}\rangle$ (cost rate $-\rho_t$)".
  2. Line 382: "The reward 1-form $\mathcal{R}$ has component units $\mathrm{nat}/[z]$; its pairing with the latent velocity, $\rho(u):=\langle\mathcal{R},\dot z(u)\rangle$, is a reward rate in $\mathrm{nat\,s^{-1}}$ and links to discrete time by $r_{t+1}\approx\int_{t\Delta t}^{(t+1)\Delta t}\rho(u)\,du$" (index and bounds as in E-009).
- Required new assumptions/permits: none.
- Validation plan: units check $[\mathcal R_i][\dot z^i]=(\mathrm{nat}/[z])([z]/\mathrm s)=\mathrm{nat\,s^{-1}}$; confirm the diagnostics chapter's threshold-unit sentence still reads correctly.

### [E-005] $V$ is a cost-to-go while $dV$ is the exact part of the reward 1-form; the two upstream anchors use opposite sign conventions (was F-005)
- Location: item 4 Reward, Mechanism and Boundary interpretation, lines 255-263; Units and Dimensional Conventions, line 373
- Severity: Major
- Type: Definition mismatch (secondary: Parameter inconsistency, sign convention)
- Criterion: Framework
- Origin: upstream conflict between `01_foundations/02_control_loop.md` and `06_fields/02_reward_field.md`, inherited by this chapter, which is the notation root and does not fix a convention
- Claim (verbatim): line 255: "the critic's $V$ is the internal value/cost-to-go for the **exact (conservative) component**"; lines 259-262: "By Hodge decomposition, $\mathcal{R} = d\Phi + \delta \Psi + \eta$; only the exact part $d\Phi$ (identified with $dV$, the critic's exact component) yields a scalar charge density ... Define the non-exact component $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $dA = \mathcal{F}$"; line 373: "Costs / values / losses (including $V$ and negative rewards) are measured in **nats**".
- Upstream anchors:
  - `01_foundations/02_control_loop.md:102-103`: "This chapter uses a cost convention (lower is better). Reward-form statements are recovered by negating costs; in Section 2.7 we set $\mathcal{C} = -\mathcal{R}$." `:146-147`: "$A := \delta\Psi + \eta$ is the non-conservative component of the cost 1-form $\mathcal{C} = dV + A$ (reward convention: $\mathcal{R}=-\mathcal{C}=d(-V)+(-A)$; conservative case: $A=0$)". `:583-598`: minimise cost-to-go $V$; $\dot V(z):=\nabla_A V(z)^\top\dot z\le-\lambda_{\text{Lyap}}V(z)$. `:1069`: "$-\Delta_G V + \kappa^2 V = \rho_c$ with $\rho_c := -\rho_r$".
  - `06_fields/02_reward_field.md:194-196`: "We identify $\Phi$ with the critic value $V$ (the exact component), so $d\Phi = dV$ ... $\mathcal{R} = d\Phi + A$ and $\mathcal{F} = dA$." `:357`: "$\nabla_A V := \nabla V - A$ with $A := \delta\Psi + \eta$ the non-conservative component of $\mathcal{R}$". `:350`: "$V(z) = r \Delta t + \gamma \mathbb{E}[V(z')]$". `:638`: "Throughout this document we use the **Reward convention** unless otherwise noted."
  - `05_geometry/04_equations_motion.md:234`: "$\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor".
- Why this is an error: in 02_control_loop, $\mathcal R=-dV-A$ with $V$ a cost-to-go; in 06_fields, $\mathcal R=+dV+A$ with $V$ reward-valued. Writing $V_R=-V_C$ and $A_R=-A_C$, the two systems map onto each other ($\mathcal R=-\mathcal C=-(dV_C+A_C)=dV_R+A_R$, $\nabla_A V_R=-\nabla_A V_C$, $\mathcal F_R=-\mathcal F_C$), so each chapter is self-consistent, but the shared symbols $V$, $A$, $\mathcal F$ and $\nabla_A V$ denote sign-opposite objects in the two chapters. This chapter presents both halves as one: line 255 labels $V$ a cost-to-go (cost convention) while lines 259-262 make $dV$ the exact part of the reward 1-form (reward convention). If $V$ is a cost-to-go, the exact part of $\mathcal R$ is $-dV$ and its non-conservative part is $-A$ in the control-loop sense. Line 373 is a units statement and is sign-insensitive; it does not by itself create a contradiction, but its grouping of $V$ with "costs" and "negative rewards" reinforces the cost reading. The equations of motion take $\mathcal F$ from $\mathcal R$ directly (`04_equations_motion.md:234`), i.e. the reward-convention $\mathcal F_R$, inside a drift $-G^{-1}\nabla\Phi_{\text{eff}}$ that descends a potential built from $V_{\text{critic}}$ (`:471`).
- Impact on downstream results: every place where the sign of $A$, $\mathcal F=dA$ or $\nabla_A V$ matters. A reader taking $A$ from 02_control_loop implements the Lorentz term $\beta_{\text{curl}}G^{-1}\mathcal F\dot z$ (this chapter line 448; `04_equations_motion.md:227, 937`) with the opposite sign from one taking it from 06_fields. The Hodge-alignment defect $1-\cos(\Delta z,-G^{-1}\nabla_A V)$ (`02_sieve/01_diagnostics.md:256`), the stiffness node $\lVert\nabla_A V\rVert$ (`:100, 550`) and the Lyapunov constraints (`:534-537`, which need the cost convention) all depend on which $\nabla_A V$ is meant; $\nabla_A V$ appears in 17 files of the volume.
- Fix guidance:
  1. Choose one convention in this chapter and state it once, next to the definition of $r_t$. The minimal choice consistent with the Lyapunov usage and the units section is the cost convention: $V$ is cost-to-go, $\mathcal C:=-\mathcal R$ is the cost 1-form, Hodge-decompose $\mathcal C=dV+A$ with $A:=\delta\Psi+\eta$ the non-conservative part of $\mathcal C$, and write line 259 as "$\mathcal{C}=dV+A$, equivalently $\mathcal{R}=-dV-A$; $\mathcal F:=dA$".
  2. Change line 255 to "cost-to-go" (drop "value/") or, if the reward convention is chosen instead, to "value" and rewrite the Lyapunov statements in 02_control_loop accordingly.
  3. Flag `06_fields/02_reward_field.md:194-196, 357` and `05_geometry/04_equations_motion.md:234` for the corresponding sign change ($\Phi=-V$, or decompose $\mathcal C$ and define $\mathcal F$ from $\mathcal C$), so that $\nabla_A V=G^{-1}(dV-A)$ and $\mathcal F$ are unambiguous across the volume.
- Required new assumptions/permits: none; this is a convention choice, but it must be applied uniformly.
- Validation plan: with the chosen convention, check that (a) $\dot V\le-\lambda V$ and (b) the policy descending $\nabla_A V$ are both "good" directions, (c) the Boltzmann weight in `06_fields/02_reward_field.md:636-638` carries the matching sign, and (d) $\mathcal F$ in the Boris integrator (`04_equations_motion.md:551-560`) is computed from the same 1-form as in the definition.

### [E-006] Symbol $a$ reused for the affine scale and for the action (was F-006)
- Location: Definition Agent symmetry group; operational, line 308
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$G_{\text{obj}} := \{(a,b): a>0,\ r\mapsto ar+b\}$".
- Upstream anchor: n/a; $a_t\in\mathcal A$ is the action at lines 27, 130, 240 of this chapter.
- Why this is an error: the same letter denotes the action throughout the chapter and the gauge scale inside a definition invoked next to policy statements (line 343); `02_sieve/01_diagnostics.md:538` inherits the overload ("targets still change under $r\mapsto ar+b$").
- Impact on downstream results: none mathematical.
- Fix guidance: write $G_{\text{obj}}:=\{(\alpha,\beta_0):\alpha>0,\ r\mapsto\alpha r+\beta_0\}$ (or $(c_1,c_0)$) here and in the diagnostics chapter.
- Required new assumptions/permits: none.
- Validation plan: grep for `ar+b` in the volume.

### [E-007] The positive-affine objective gauge is not a symmetry of the fixed-$T_c$ entropy-regularised objective or of episodic tasks with fixed terminal data (was F-008)
- Location: Definition Agent symmetry group; operational, lines 305-311, 331, 334, 343
- Severity: Moderate
- Type: Scope restriction (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 305-308: "$G_{\text{obj}}$ be an **objective/feedback gauge** acting on scalar feedback signals ... A common choice is the positive affine group $G_{\text{obj}} := \{(a,b): a>0,\ r\mapsto ar+b\}$"; line 331: "transformations of the scalar feedback scale/offset (and any potentials) that should not qualitatively change the policy update direction"; line 334: "The internal maps of the agent should be invariant/equivariant under $\mathcal{G}_{\mathbb{A}}$".
- Upstream anchor: `01_foundations/02_control_loop.md:499-501`: "$\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \big( V(z) - T_c\, H(\pi(\cdot|z)) \big) d\mu_G$ ... $T_c$ is the entropy weight"; `05_geometry/04_equations_motion.md:512`: "$\pi(a|z) \propto \exp(Q(z,a)/T_c)$"; `06_fields/02_reward_field.md:120-121`: "Dirichlet condition $V|_{\Gamma_{\text{term}}} = V_{\text{term}}$ (often $0$ or a terminal payoff)"; `02_sieve/01_diagnostics.md:538`: "This does **not** make the entire RL pipeline invariant to arbitrary reward rescaling by itself (targets still change under $r\mapsto ar+b$)".
- Why this is an error: under $r\mapsto ar$ every return, hence $V$ and $Q$, scales by $a$; with $T_c$ fixed the Boltzmann optimum becomes $\exp(aQ/T_c)$, the policy at temperature $T_c/a$, which differs from the original unless $T_c$ co-scales. Under $r\mapsto r+b$ with an absorbing boundary, continuation values shift by $b\sum_{k<H}\gamma^k$ over the random remaining horizon $H$ while $V_{\text{term}}$ does not shift, so the relative value of terminating versus continuing changes and the optimal policy changes (the survival-bonus effect). The group as stated is a symmetry only for $b=0$ (or infinite horizon without absorbing states) and with $T_c$ co-scaled. The chapter hedges ("candidate", "should not qualitatively change"), but line 334 turns it into an invariance/equivariance requirement without recording these hypotheses.
- Impact on downstream results: the projective value head (`02_sieve/01_diagnostics.md:516-538`) is motivated by this gauge, and the diagnostics chapter already concedes non-invariance; the utility gauge in `08_multiagent/02_standard_model.md:117-118` needs the co-transformation of $T_c$ and $V_{\text{term}}$ made explicit if it quotients by $G_{\text{obj}}$.
- Fix guidance:
  1. Define the objective gauge as acting jointly, $(r,T_c,V_{\text{term}})\mapsto(ar+b,\ aT_c,\ aV_{\text{term}}+b\,\Delta_\gamma)$ with $\Delta_\gamma$ the horizon factor ($1/(1-\gamma)$ for infinite horizon), or
  2. Restrict to the scale subgroup $\{a>0\}$ with $T_c$ co-scaled and state that offsets $b$ are a symmetry only in the absence of absorbing boundaries.
  3. In line 334, replace "invariant/equivariant under $\mathcal G_{\mathbb A}$" by "equivariant under $\mathcal G_{\mathbb A}$ with the co-transformations above".
- Required new assumptions/permits: co-scaling of $T_c$; horizon/terminal co-shift for offsets.
- Validation plan: verify on a two-state episodic example that the argmax of $Q$ and the softmax policy are unchanged under the joint transformation and changed under the naive one.

### [E-008] $S_{\lvert\mathcal K\rvert}$ is stated for a flat codebook but the macro register is two-level (was F-007)
- Location: Definition Agent symmetry group; operational, line 313; Principle of covariance, line 342
- Severity: Minor
- Type: Scope restriction
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "$S_{|\mathcal{K}|}$ be the **symbol-permutation symmetry** of the discrete macro register: relabeling code indices is unobservable if downstream components depend only on embeddings $\{e_k\}$."
- Upstream anchor: `01_foundations/02_control_loop.md:247-253`: "For the Attentive Atlas shutter, the macro register is a two-level tuple: $K_t = (K_{\text{chart}}, K_{\text{code}})$, with $K_{\text{chart}}\in\{1,\dots,N_c\}$ and $K_{\text{code}}\in\{1,\dots,N_v\}$"; `:322-334`: "Each chart has its own codebook $\{e_{i,c}\}_{c=1}^{N_v}$ ... $K_{\text{code},i}(x) := \arg\min_c \|v(x)-e_{i,c}\|_2^2$ ... $z_q(x) := \sum_i w_i(x)\, e_{i,K_{\text{code},i}(x)}$".
- Why this is an error: for the two-level register $\lvert\mathcal K\rvert=N_cN_v$, and a general element of $S_{N_cN_v}$ moves a code from one chart to another; this changes the set over which the receiving chart's argmin ranges and hence changes $z_q(x)$ for some inputs, so it is observable. The unobservable relabellings are chart permutations combined with within-chart code permutations, the wreath product $S_{N_v}\wr S_{N_c}$. The statement holds for a flat codebook only.
- Impact on downstream results: permutation-invariant diagnostics (line 342, Nodes 18/19) are unaffected in practice; any later claim of invariance "under $S_{\lvert\mathcal K\rvert}$" for the atlas shutter is false as stated.
- Fix guidance: write "$S_{\mathcal{K}}$ denotes the relabelling group of the macro register: $S_{|\mathcal{K}|}$ for a flat codebook, and $S_{N_v}\wr S_{N_c}$ (chart permutations and within-chart code permutations) for the two-level register of the Attentive Atlas", and use $S_{\mathcal K}$ in the product at line 325 and at line 342.
- Required new assumptions/permits: none.
- Validation plan: check that the routing softmax and the per-chart argmin commute with every element of $S_{N_v}\wr S_{N_c}$ (they do, since both act chart-wise).

### [E-009] Reward index and integral bounds inconsistent with the boundary-law convention (was F-009)
- Location: Discrete vs continuous reward, line 382; cf. Definition Environment as Generative Process, line 148, and Base units, line 371
- Severity: Minor
- Type: Dimensional mismatch (secondary: Notation conflict, index convention)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 382: "$r_t \approx \int_{t}^{t+\Delta t}\mathcal{R}(u)\,du$"; line 148: "$P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_{\le t})$"; line 371: "Interaction time is measured in **environment steps** ($t \in \mathbb{Z}_{\ge 0}$). If a physical clock is needed, introduce $\Delta t$ with $[\Delta t]=\mathrm{s}$."
- Upstream anchor: n/a; defined in this chapter.
- Why this is an error: (i) the lower bound $t$ is a dimensionless step index and the upper bound $t+\Delta t$ adds seconds to it; the physical window of step $t$ is $[t\Delta t,(t+1)\Delta t]$. (ii) By line 148 the reward produced in response to $B_t$ (which contains $a_t$) is $r_{t+1}$, so the reward accrued over the step starting at $t$ is $r_{t+1}$, not $r_t$ (equivalently, $r_t$ accrues over $[(t-1)\Delta t,t\Delta t]$). The memory tuple $(z_{t'},a_{t'},r_{t'})$ at line 500 is not itself an error: it groups the same indices as $B_{t'}$ (line 122), where $r_{t'}$ is the reward that arrived together with $x_{t'}$.
- Impact on downstream results: off-by-one risk in any TD target written from these definitions; `06_fields/02_reward_field.md:329, 350` write $V(z)=\mathbb E[r+\gamma V(z')]$ without index, so the ambiguity is not resolved upstream.
- Fix guidance: line 382: "$r_{t+1}\approx\int_{t\Delta t}^{(t+1)\Delta t}\langle\mathcal{R},\dot z(u)\rangle\,du$", and state once that $r_t$ in $B_t$ is the reward received for $a_{t-1}$.
- Required new assumptions/permits: none.
- Validation plan: check the Bellman recursion in 06_fields against the stated index once it is fixed.

### [E-010] Units of $T_c$ incompatible with the upstream identification $\beta_{\text{cpl}}=1/T_c$ (was F-010)
- Location: Regularization / precision coefficients, lines 385-387
- Severity: Minor
- Type: Parameter inconsistency (units)
- Criterion: Framework
- Origin: upstream `01_foundations/02_control_loop.md:1284-1286` and `05_geometry/04_equations_motion.md:515` (this chapter's two statements are individually correct and match the parameter table)
- Claim (verbatim): line 385: "this coefficient is dimensionless and simply sets relative weight in the objective"; line 387: "The metric coupling coefficient $\beta_{\text{cpl}}$ (Definition def-local-conditioning-scale) carries units $\mathrm{nat}/[z]^2$".
- Upstream anchor: `01_foundations/02_control_loop.md:1284-1286`: "in an isothermal approximation where $\beta_{\text{cpl}}$ is constant, set $\beta_{\text{cpl}} = 1/T_c$. Units: ... $[\beta_{\text{cpl}}]=\mathrm{nat}/[z]^2$ (dimensionless when $z$ is normalized)"; `05_geometry/04_equations_motion.md:515`: "*Units:* nat (dimensionless in natural units where $k_B = 1$)"; `10_appendices/02_parameters.md:44`: "$T_c$ | cognitive temperature / entropy-regularization coefficient | dimensionless".
- Why this is an error: three unit assignments coexist: dimensionless (this chapter, parameter table), nat (`def-cognitive-temperature`), and $[z]^2/\mathrm{nat}$ via $T_c=1/\beta_{\text{cpl}}$ (`def-local-conditioning-scale`). Since $V$ and $H$ are both in nats, $V-T_cH$ forces $T_c$ dimensionless, as this chapter says; the identification $\beta_{\text{cpl}}=1/T_c$ is then valid only under the "z normalised" caveat, which is not carried into the SDE noise $\sqrt{2T_c}\,G^{-1/2}dW_s$ (`04_equations_motion.md:1291`): with $[G]=\mathrm{nat}/[z]^2$, $[G^{-1/2}]=[z]/\mathrm{nat}^{1/2}$ and $[dW_s]=[s]^{1/2}$, the increment $dz$ carries $[z]$ only if $[T_c]=\mathrm{nat}/[s]$.
- Impact on downstream results: unit bookkeeping for the noise amplitude and for $\beta_{\text{cpl}}$-based checks; no theorem breaks, but the units anchor in this chapter cannot be used to check them.
- Fix guidance: add one sentence at line 385: "$T_c$ is dimensionless (nats per nat); the identification $\beta_{\text{cpl}}=1/T_c$ holds only for normalised latents ($[z]$ chosen so that $[G]=\mathrm{nat}$) and dimensionless computation time", and flag `04_equations_motion.md:515` and `02_control_loop.md:1286` to adopt the same statement.
- Required new assumptions/permits: none.
- Validation plan: redo the units of each row of the summary table at `04_equations_motion.md:1288-1294` under the stated normalisation.

### [E-011] Memory-time domain stated inconsistently (was F-011)
- Location: Chronology table, line 415; Memory Time, line 496
- Severity: Minor
- Type: Notation conflict (domain)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): table: "$\{t' \in \mathbb{Z} : t' < t\}$"; text: "$t'$ ranges over $\{0, 1, \ldots, t-1\}$".
- Upstream anchor: n/a; interaction time is $t\in\mathbb Z_{\ge0}$ (lines 371, 412).
- Why this is an error: with $t\in\mathbb Z_{\ge0}$ there are no stored states at negative indices, so the table's domain includes indices that cannot exist; the two statements of the same domain disagree.
- Impact on downstream results: none beyond holographic-screen indexing in 07_cognition.
- Fix guidance: table entry "$\{0,1,\dots,t-1\}$" (or $\{t'\in\mathbb Z_{\ge0}: t'<t\}$).
- Required new assumptions/permits: none.
- Validation plan: none needed.

### [E-012] Scale time $\tau$ and computation time $s$ declared distinct, but the cited holographic law is the overdamped $s$-flow (was F-012)
- Location: Computation Time, lines 445-453; Scale Time, lines 478-482; line 408
- Severity: Minor
- Type: Notation conflict (secondary: Conceptual)
- Criterion: Framework
- Origin: upstream `05_geometry/04_equations_motion.md:944-951` and `05_geometry/03_holographic_gen.md:148` (this chapter cites the law and asserts the distinction)
- Claim (verbatim): line 408: "We distinguish four temporal dimensions. They are orthogonal (or nested) and must not be conflated."; line 448: "$\frac{dz}{ds} = \mathcal{M}_{\text{curl}}(-G^{-1}\nabla \Phi_{\text{eff}} + \dots)$"; line 479: "**Dynamics:** $dr/d\tau = \tfrac{1}{2}\operatorname{sech}^2(\tau/2)$ (the holographic law)."
- Upstream anchor: `05_geometry/04_equations_motion.md:944-951`: "Setting $\alpha = 1$ (pure generation), $T_c \to 0$ ... in the overdamped equation recovers the holographic gradient flow ... For the Poincare disk, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,z$, which integrates to $|z(\tau)| = \tanh(\tau/2)$"; `:1291`: "Overdamped | $dz = \mathcal{M}_{\text{curl}}(-G^{-1}\nabla\Phi_{\text{eff}} + u_\pi)\,ds + \dots$"; `03_holographic_gen.md:148`: "The overdamped equation $\dot{r} = (1-r^2)/2$ ... integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$".
- Why this is an error: the law is arithmetically correct ($r=\tanh(\tau/2)\Rightarrow dr/d\tau=\tfrac12\operatorname{sech}^2(\tau/2)=\tfrac12(1-r^2)$), but upstream derives it as the solution of the overdamped equation whose independent variable is $s$, writing the solution parameter as $\tau$. So in the generative regime the book identifies scale time with computation time along a generation pass, while this chapter states the same ODE once in $s$ (line 448) and once in $\tau$ (line 479) and calls the two dimensions orthogonal. The identification $\tau=s$ (generation) is stated nowhere.
- Impact on downstream results: the category-error warning (lines 508-522) is weakened; anything in 05_geometry that integrates "in $\tau$" using the $s$-equation is silently using $\tau=s$.
- Fix guidance: add after the table: "In the generative regime the scale coordinate is traversed by the computation-time flow, $\tau=s$ along a generation pass (Corollary cor-recovery-of-holographic-flow); $\tau$ is an independent coordinate only when indexing a fixed hierarchy (layers $\tau_\ell$)."
- Required new assumptions/permits: none.
- Validation plan: check that 03_holographic_gen and 04_equations_motion use $\tau$ only where the generation pass is meant.

### [E-013] Stale hard-coded section numbers (was F-013)
- Location: Scale Time, line 478
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "This is the radial coordinate in the Poincare disk (Sections 21, 7.12)."
- Upstream anchor: the content lives at `05_geometry/03_holographic_gen.md:138-149` (`prop-isotropic-radial-expansion`) and `05_geometry/04_equations_motion.md:941-951` (`cor-recovery-of-holographic-flow`); no "Section 21" or "7.12" numbering exists in the current files.
- Why this is an error: the numbers refer to a superseded monolithic numbering and resolve to nothing; every other pointer in this chapter uses `{ref}` labels.
- Impact on downstream results: none mathematical.
- Fix guidance: replace with `{prf:ref}` links to `prop-isotropic-radial-expansion` and `cor-recovery-of-holographic-flow`.
- Required new assumptions/permits: none.
- Validation plan: build the docs and confirm the links resolve.

## Scope restrictions and clarifications
- The Hodge decomposition at line 259 and the objects $A$, $\mathcal F$, $\nabla_A V$ are valid under either sign convention; the defect is that no single convention is fixed and the two anchor chapters differ (E-005). Any downstream statement in which $A$ or $\mathcal F$ appears with a definite sign should be read against the convention of the chapter it comes from until the root convention is fixed.
- The terminal-boundary identification (E-002) is exact only when termination is $\sigma(Z_t)$-measurable; in the general partially observed case the killing-rate form is the well-typed object.
- The objective gauge (E-007) is a symmetry of the entropy-regularised objective only under joint transformation of $(r,T_c,V_{\text{term}})$.
- The symbol-permutation symmetry (E-008) is $S_{N_v}\wr S_{N_c}$ for the two-level register.
- The identification $\beta_{\text{cpl}}=1/T_c$ (E-010) requires normalised latents and dimensionless computation time.

## Open questions
- Which sign convention does the author intend as canonical for the volume? The Lyapunov constraints and the units section point to the cost convention; 06_fields and the equations of motion use the reward convention for $\mathcal R$ and $\mathcal F$.
- In the cost convention, $\nabla_A V=G^{-1}(dV-A)$ with $A$ the non-conservative part of $\mathcal C=dV+A$ is not the metric dual of $\mathcal C$ (which would be $G^{-1}(dV+A)$). Is $d_A=d-A$ meant as a gauge-covariant derivative with $A$ a connection, rather than as the raising of the full cost 1-form? This should be stated where $\nabla_A V$ is first defined (02_control_loop), not in this chapter.
- Should the memory tuple at line 500 be written as a transition tuple $(z_{t'},a_{t'},r_{t'+1})$ once the reward index at line 382 is fixed, or is the $B_{t'}$ grouping intended throughout 07_cognition?

## Rejected candidate findings
None. All thirteen stage-1 findings were retained; F-002, F-003, F-005 and F-009 were adjusted (origin, type, or supporting evidence) as recorded in the detailed entries.
