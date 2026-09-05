# Mathematical Review: docs/source/1_agent/02_sieve/01_diagnostics.md

## Metadata
- Reviewed file: docs/source/1_agent/02_sieve/01_diagnostics.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1046 lines)
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/01_definitions.md:370-398` (units: $[V]=\mathrm{nat}$, $c_t=-r_t$, KL in nats, stabilisers inherit units)
  - `01_foundations/02_control_loop.md:102, 141, 583, 1011-1065` (cost convention; $V$ is cost-to-go that decreases along controlled trajectories; Lie derivative $\mathcal L_fV=\nabla V\cdot f$)
  - `01_foundations/02_control_loop.md:695-850` (`def-state-space-sensitivity-metric`, `def-complete-latent-space-metric`, units of $G$, $\lambda_G$; diagonal metric $\mathrm{diag}(\alpha,\beta_\pi,\gamma_{\text{wm}},\delta)$)
  - `02_sieve/02_limits_barriers.md:59` (BarrierGap)
  - `02_sieve/04_approximations.md:421`, `10_appendices/06_losses.md:709` (InfoNCE with positive in the denominator)
  - `03_architecture/01_compute_tiers.md:137, 393-520` (core-tier stiffness loss; `metric_mode`)
  - `04_control/03_coupling_window.md:121-150` (`thm-information-stability-window-operational`)
  - `05_geometry/04_equations_motion.md:220-240, 905-925, 1310-1335` (`def-bulk-drift-continuous-flow`, `thm-overdamped-limit`, Nodes 26-27)
  - `06_fields/03_info_bound.md:634-693` (diagnostic node registry)
  - `07_cognition/02_governor.md:58-68, 234-250, 588` (subsumption of Methods A/B/C)

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 8
- Minor: 12
- Notes: 0
- Primary themes:
  1. Sign and type errors in the policy-regulation losses. The natural-gradient loss $\mathcal L_{\text{nat}}$ rewards the policy for increasing the cost-to-go $V$ (the book's convention is cost, lower is better), which makes it adversarial to the Lyapunov critic loss and the Hodge alignment term stated a few lines away; the "geodesic Zeno" term applies a state-space metric to a difference of action distributions.
  2. Dimensional problems. Node 27's ratio has units $\tau/\text{length}$; the `obs_var` diagonal metric has the units of $G^{-1}$; the scaling-coefficient hierarchy compares a KL in nats with a curvature; the eikonal constant 1 carries hidden units.
  3. Node table versus the equations of motion. Node 26 omits friction (and here also the control force), so an exact trajectory of the book's own Langevin dynamics registers a residual of order $\gamma\|\dot z\|$.
  4. Misstated standard results. InfoNCE is written with a negatives-only denominator and the CPC bound is then applied to it; the "PID" update integrates the proportional term; dual ascent is called memoryless.
  5. Counting and naming inconsistencies: 29 versus 32 table rows, 59 versus 60 nodes, duplicate symbol $\mathrm{SC}_{\partial c}$, a non-existent "CollapseCheck", one registry formula inverted, and four unrelated meanings of $\gamma$.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | TLDR / section title / closing admonition, lines 26, 31, 81, 1027 | Minor | Computational error | Framework | this chapter | "29 checks" but the table has 32 rows (27 numbered + 5 lettered) |
| E-002 | Researcher Bridge / Connection #8 / figure, lines 38, 48, 68, 77 | Minor | Parameter inconsistency | Framework | this chapter (+ upstream registry) | "60 nodes" and "59 constraints" for the same set; neither matches the book |
| E-003 | Stability table, lines 98, 103 | Minor | Notation conflict | Framework | this chapter | Nodes 5 and 7c share the identifier $\mathrm{SC}_{\partial c}$ |
| E-004 | Stability table, Node 26, line 124 | Moderate | Definition mismatch (Algorithm mismatch) | Framework | upstream `05_geometry/04_equations_motion.md:1317` + this chapter | Residual omits friction; chapter row also drops $u_\pi$ and $\Phi_{\text{eff}}$ |
| E-005 | Stability table, Node 27, lines 125-127 | Moderate | Dimensional mismatch | Framework | upstream `05_geometry/04_equations_motion.md:1328` + this chapter | Ratio has units $\tau/\text{length}$; "friction >> 1" ill-posed; no inertial quantity |
| E-006 | Geometric properties table, line 138 | Minor | Definition mismatch | Framework | this chapter | $\lambda_{\min}(G)>\epsilon$ is not the Node 7 condition $\|\nabla_AV\|>\epsilon$ |
| E-007 | Scaling coefficients, lines 193-197 | Moderate | Dimensional mismatch (Conceptual) | Framework | this chapter | `obs_var` has units of $G^{-1}$; `grad_rms` of $\sqrt G$; `state_fisher` omits $\lambda_G$ |
| E-008 | Scaling coefficients and gating code, lines 199-215, 1006-1019 | Moderate | Dimensional mismatch (Proof gap) | Framework | this chapter + upstream `02_control_loop.md:812-820` | Hierarchy compares undefined quantities; code compares KL (nat) with curvature |
| E-009 | Contrastive anchoring / Connection #29, lines 314, 392-403 | Moderate | Computational error (External dependency) | External | this chapter | InfoNCE denominator excludes the positive; CPC bound misapplied |
| E-010 | Connection #29, line 412 | Minor | Citation / reference error | Framework | this chapter | "Node 6 (CollapseCheck)" names a non-existent node |
| E-011 | Eikonal regulariser, lines 539-545 | Minor | Dimensional mismatch | Framework | this chapter | Constant 1 has hidden units nat/[z]; norm is coordinate-dependent |
| E-012 | Lyapunov stiffness, line 550 | Minor | Typo (Notation conflict) | Framework | this chapter | $\|\nabla_AV\|^2_{\text{reg}}$ undefined; literal reading contradicts the hinge |
| E-013 | Safety budget, lines 554-560 vs 94 | Minor | Definition mismatch (Miswording) | Framework | this chapter | Hinge vs squared hinge; fixed $\lambda_{\text{safety}}$ called "hard Lagrangian" |
| E-014 | Policy regulation table and Node 10 loss, lines 581-594 | Major | Invalid inference (Dimensional mismatch) | Framework | this chapter | $\mathcal L_{\text{nat}}$ has the wrong sign under the cost convention; free index $i$ |
| E-015 | Geodesic stiffness, lines 609-615 | Moderate | Dimensional mismatch (Definition mismatch) | Framework | this chapter | $\|\pi_t-\pi_{t-1}\|_G^2$ applies a $T_z\mathcal Z$ form to a measure on $\mathcal A$ |
| E-016 | Cross-network synchronisation, line 660 | Minor | Notation conflict (sign convention) | Framework | this chapter | TD target uses reward $r$ while $V$ is a cost elsewhere in the chapter |
| E-017 | Information-stability window and code-usage setpoints, lines 282, 709-719, 866, 966-980 | Moderate | Parameter inconsistency (Definition mismatch) | Framework | this chapter + upstream `04_control/03_coupling_window.md:124` | Same symbol $H(K)$ pushed to saturation and bounded away from it; $\epsilon$ range too loose |
| E-018 | Method B, lines 868-884 | Moderate | Computational error (External dependency) | External | this chapter | "PID" recursion integrates the proportional term and has no derivative action |
| E-019 | Neural Unification admonition, line 1041 | Minor | Conceptual | External (+ Framework) | this chapter + upstream `07_cognition/02_governor.md:63,246` | Dual ascent called "memoryless ($H=0$)"; it is an integrator and the Governor says $H=1$ |
| E-020 | Registry row for Node 11 vs line 108 | Minor | Citation / reference error | Framework | upstream `06_fields/03_info_bound.md:647` | Registry drops the "$1-$" of the capacity gap, inverting the defect's sense |
| E-021 | Chapter-wide; first at lines 125, 183, 201, 352, 534, 660 | Minor | Notation conflict | Framework | this chapter | $\gamma$ has four meanings, $\alpha$ two (with different units) |

## Detailed findings

### [E-001] Stated count "29" does not match the stability table (was F-001)
- Location: lines 26, 31, 81, 1027; table lines 92-125
- Severity: Minor
- Type: Computational error (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim: line 31 "Stability and data-quality are monitored via 29 distinct checks (Gate Nodes)."; line 81 "## The 29 Stability Checks"
- Upstream anchor: not applicable (defined here); the registry `06_fields/03_info_bound.md:637-663` lists nodes 1-27 for this section without lettered sub-nodes.
- Why this is an error: the table has 27 numbered rows plus 7a, 7b, 7c, 7d, 12a, i.e. 32 rows, each presented as a distinct check. Neither 27 nor 32 equals 29.
- Impact on downstream results: cosmetic, but repeated in three places and ambiguous for any later text that says "the 29 checks of Section 3".
- Fix guidance: (1) decide whether lettered sub-nodes count; (2) write "32 checks (27 numbered nodes plus five sub-nodes)" or renumber; (3) update lines 26, 31, 81, 1027.
- Required new assumptions/permits: none.
- Validation plan: `grep -n "29" 01_diagnostics.md` and recount rows after the edit.

### [E-002] "60 nodes" versus "59 constraints", and neither matches the book (was F-002)
- Location: lines 38, 48, 68, 77
- Severity: Minor
- Type: Parameter inconsistency
- Criterion: Framework
- Origin: this chapter (59 vs 60); upstream `06_fields/03_info_bound.md:634-693` for the global count
- Claim: line 48 "PASS if all 60 diagnostics pass"; line 68 "Typed diagnostics: 59 constraints with semantic identity"
- Upstream anchor: the "Diagnostic Node Registry (Complete)" lists 57 rows (1-56 and 61); explicit `(node-NN)=` labels in the volume run from node-25 to node-73 (54 and 55 carry `prf` labels).
- Why this is an error: two totals are given within twenty lines for the same index set, and the Sieve PASS condition at line 48 is defined by quantifying over that set. The book actually defines nodes 1-73 plus lettered sub-nodes, and its "complete" registry has 57 rows, so no numeral currently in the text is derivable.
- Impact on downstream results: every "all N diagnostics pass" statement (Sieve definition, PoUW audit in `09_economics`) inherits an ill-defined index set.
- Fix guidance: (1) replace the numerals at lines 38, 48, 68, 77 with a reference to the registry ("all registered diagnostics, {ref}`sec-diagnostic-node-registry`"); (2) complete the registry upstream to nodes 57-60, 62-73.
- Required new assumptions/permits: none.
- Validation plan: grep for hard-coded node totals across the volume after the edit.

### [E-003] Two nodes share the identifier $\mathrm{SC}_{\partial c}$ (was F-003)
- Location: lines 98, 103
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim: line 98 "**5** | **ParamCheck ($\mathrm{SC}_{\partial c}$)**"; line 103 "**7c** | **CheckSC ($\mathrm{SC}_{\partial c}$)**"
- Upstream anchor: not applicable; a grep across the volume finds the symbol only on these two lines.
- Why this is an error: the parenthesised symbol is the node's formal identifier (compare $\mathrm{Rec}_N$, $C_\mu$, $\mathrm{SC}_\lambda$). Node 5 (world-model stationarity) and Node 7c (critic new-mode variance) are unrelated checks with the same identifier.
- Impact on downstream results: references by symbol cannot be resolved.
- Fix guidance: give Node 7c a distinct identifier, e.g. $\mathrm{SC}_{\sigma^2}$ or $\mathrm{SC}_{\text{mode}}$.
- Required new assumptions/permits: none.
- Validation plan: grep for the new symbol; confirm no other node uses it.

### [E-004] Node 26 residual omits friction and, in this chapter, the control force (was F-004)
- Location: Stability table, Node 26, line 124
- Severity: Moderate
- Type: Definition mismatch (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: upstream `05_geometry/04_equations_motion.md:1317` (friction omitted there too); this chapter additionally drops $u_\pi$ and writes $\Phi$ for $\Phi_{\text{eff}}$
- Claim: line 124 "$\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi - \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}\rVert_G$"
- Upstream anchor: `05_geometry/04_equations_motion.md:916` "$m\,\ddot{z}^k + \gamma\,\dot{z}^k - \beta_{\text{curl}} G^{km}\mathcal{F}_{mj}\dot{z}^j + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2T_c}\,(G^{-1/2})^{kj}\,\xi^j$"; momentum equation `…:227` contains "$-\gamma\,p_k$" and "$+u_{\pi,k}$"; upstream Node 26 `…:1317` "$\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + G^{-1}\nabla\Phi_{\text{eff}} - u_\pi - \beta_{\text{curl}} G^{-1}\mathcal{F}\dot{z}\rVert_G$".
- Why this is an error: with $m=1$, rearranging the EOM gives $\ddot z+\Gamma(\dot z,\dot z)+G^{-1}\nabla\Phi-\beta_{\text{curl}}G^{-1}\mathcal F\dot z = -\gamma\dot z+\sqrt{2T_c}G^{-1/2}\xi$ identically along a trajectory of the book's own dynamics. The check's residual is therefore $\approx\gamma\|\dot z\|_G$ on a perfectly consistent trajectory, and it is largest precisely in the overdamped regime that Node 27 promotes. The check tests against a frictionless, uncontrolled geodesic that the framework does not predict. The chapter row also disagrees with the upstream row (no $u_\pi$, $\Phi$ instead of $\Phi_{\text{eff}}$), and the upstream "$-u_\pi$" is a covector inside a vector equation (should be $-G^{-1}u_\pi$).
- Impact on downstream results: the Node 26 trigger ("reduce time step; verify Christoffel computation", `…:1321-1322`) would fire on healthy runs; the registry row `06_fields/03_info_bound.md:663` repeats the upstream form.
- Fix guidance: (1) define the residual as the full EOM defect $r:=\ddot z+\gamma\dot z+\Gamma(\dot z,\dot z)+G^{-1}\nabla\Phi_{\text{eff}}-G^{-1}u_\pi-\beta_{\text{curl}}G^{-1}\mathcal F\dot z$; (2) compare $\|r\|_G$ to the thermal floor $\sim\sqrt{2\gamma T_c}$ per unit time, or state explicitly that the check applies only in the $\gamma\to0$, $u_\pi=0$ regime; (3) make the chapter row identical to the upstream row and fix the index placement of $u_\pi$ upstream.
- Required new assumptions/permits: none beyond the existing EOM.
- Validation plan: simulate the BAOAB integrator on a quadratic $\Phi$ and confirm the corrected residual is at the noise floor while the current one scales with $\gamma$.

### [E-005] Node 27 ratio is not dimensionless and does not measure inertia (was F-005)
- Location: Stability table, Node 27, lines 125-127
- Severity: Moderate
- Type: Dimensional mismatch
- Criterion: Framework (units declared in the book)
- Origin: upstream `05_geometry/04_equations_motion.md:1328-1330`; this chapter inherits it and adds the wording "Is friction >> 1 satisfied?"
- Claim: line 125 "Is friction >> 1 satisfied? | $\gamma / \lVert \mathcal{M}_\gamma^{-1}\,v\rVert$"; line 127 "$v := \dot{z}$ and $\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F}$"
- Upstream anchor: units at `…:236` "$[z]=\text{length}$, $[p]=\text{length}/\tau$, $[\gamma]=1/\tau$"; overdamped relation `…:920` $dz = \mathcal M_\gamma(-G^{-1}\nabla\Phi)\,ds+\dots$
- Why this is an error: $\mathcal M_\gamma^{-1}v=\gamma v-\beta_{\text{curl}}G^{-1}\mathcal Fv$ has the units of $\gamma v$, i.e. $\text{length}/\tau^2$, so $[\gamma/\|\mathcal M_\gamma^{-1}v\|]=(1/\tau)/(\text{length}/\tau^2)=\tau/\text{length}$. A quantity with these units cannot be compared with 1; the pass/fail decision depends on the length unit. The expression contains no inertial quantity ($\ddot z$ or $m$). In the exact overdamped limit $\mathcal M_\gamma^{-1}\dot z=-G^{-1}\nabla\Phi$, so the "check" returns $\gamma/\|G^{-1}\nabla\Phi\|$, which diverges near critical points of $\Phi$ regardless of regime.
- Impact on downstream results: the integrator-selection rule at `05_geometry/04_equations_motion.md:1333` and the registry row `06_fields/03_info_bound.md:668`. (No other chapter carries a Node 27 row.)
- Fix guidance: (1) replace with a dimensionless inertia-to-friction ratio such as $\|\ddot z\|_G/(\gamma\|\dot z\|_G)$, or the relative slaving defect $\|\dot z-\mathcal M_\gamma(-G^{-1}\nabla\Phi_{\text{eff}}+G^{-1}u_\pi)\|_G/\|\dot z\|_G$; (2) state the pass threshold as "$\ll1$"; (3) apply the same edit upstream and in the registry.
- Required new assumptions/permits: none.
- Validation plan: dimensional check of the new expression; confirm it tends to 0 as $\gamma\to\infty$ with $m$ fixed.

### [E-006] Node 7's "formal property" is a different condition from the Node 7 check (was F-006)
- Location: line 138 vs line 100
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 138 "**7 (Stiffness)** | $G \in T^*_2(\mathcal{Z})$ | Spectral Gap | Is $\lambda_{\min}(G) > \epsilon$? (No flat directions)"
- Upstream anchor: `02_sieve/02_limits_barriers.md:59` "BarrierGap | Spectral Gap | Critic | … $\max(0, \epsilon - \Vert \nabla_A V \Vert)$ (Stiffness)"; `01_foundations/02_control_loop.md:703-711` $(G_V)_{ij}=\partial^2V/\partial z_i\partial z_j$.
- Why this is an error: $\|\nabla_AV\|>\epsilon$ (first derivative) and $\lambda_{\min}(G)>\epsilon$ (second derivative plus Fisher) are logically independent: a linear $V$ satisfies the first and not the second; a strict minimum satisfies the second and not the first. The two $\epsilon$'s also have different units. The claim at line 142 that "each node corresponds to a verifiable geometric property" is not established for Node 7.
- Impact on downstream results: local; the Governor (`07_cognition/02_governor.md:588`) already lists $\|\nabla_AV\|$ and $\lambda_{\max}(G)$ as separate signals.
- Fix guidance: either rewrite the row as "Gradient lower bound: is $\|\nabla_AV\|_{G^{-1}}>\epsilon$?", or add a separate conditioning node for $\lambda_{\min}(G)$ and state which one BarrierGap enforces.
- Required new assumptions/permits: none.
- Validation plan: check the row against line 100 and `02_limits_barriers.md:59`.

### [E-007] Diagonal metric approximations with the wrong units (was F-007)
- Location: lines 193-197
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim: line 196 "`grad_rms`: $G_{ii} = \mathbb{E}[(\partial V / \partial z_i)^2]^{1/2}$"; line 197 "`obs_var`: $G_{ii} = \text{Var}(z_i)$"; line 195 "`state_fisher`: $G_{ii} = \mathbb{E}[(\partial \log \pi / \partial z_i)^2] + \text{Hess}_z(V)_{ii}$"
- Upstream anchor: `01_foundations/02_control_loop.md:711` "Units: $[(G_V)_{ij}]=\mathrm{nat}\,[z]^{-2}$ … in the proxy form, $c_V$ carries units $\mathrm{nat}^{-1}$"; `…:843` "the Fisher term has units $[z]^{-2}$; therefore $\lambda_G$ carries the same units as $V$".
- Why this is an error: $G$ is a $(0,2)$ sensitivity tensor with units $\text{nat}[z]^{-2}$. $\mathrm{Var}(z_i)$ has units $[z]^{+2}$, the units of $G^{-1}$; as a metric it inverts the intended behaviour everywhere $G$ is used (natural-gradient steps smallest where the state varies most; QSL distance and Zeno weight largest along wide directions). `grad_rms` takes a square root of the Gauss-Newton proxy and has units nat/[z], not commensurable with the Hessian term or with $\lambda_GG_\pi$. `state_fisher` omits the $\lambda_G$ (units nat) that upstream requires for the two addends to match.
- Impact on downstream results: the chapter's own uses of $G$ (lines 600, 612; Nodes 24, 26). `03_architecture/01_compute_tiers.md:393-520` exposes only `metric_mode="state_fisher"`, which inherits the missing $\lambda_G$; `obs_var` and `grad_rms` appear nowhere else in the volume.
- Fix guidance: (1) `obs_var`: $G_{ii}=1/(\mathrm{Var}(z_i)+\epsilon)$; (2) `grad_rms`: drop the square root, $G_{ii}=c_V\,\mathbb E[(\partial_iV)^2]$; (3) `state_fisher`: $G_{ii}=\lambda_G\,\mathbb E[(\partial_i\log\pi)^2]+\mathrm{Hess}_z(V)_{ii}$ with PSD projection; (4) mirror (3) in `_compute_state_fisher`.
- Required new assumptions/permits: none.
- Validation plan: unit check of each mode against `def-complete-latent-space-metric`; a rescaling test $z\mapsto cz$ should give $G\mapsto c^{-2}G$.

### [E-008] The scaling-coefficient hierarchy compares undefined and incommensurable quantities (was F-008)
- Location: lines 199-215; gating code lines 1006-1019
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter (hierarchy and code) + upstream `01_foundations/02_control_loop.md:812-820` (symbols introduced with glosses only)
- Claim: line 210 "$\delta \ll \gamma \ll \alpha,\qquad \beta_{\pi} \le \alpha$"; table lines 201-204 mark all four "dimensionless"; code line 1007 "`alpha = ema(alpha)  # critic signal / curvature proxy`", line 1008 "`beta_pi = ema(beta_pi_kl)  # mean KL(π_t || π_{t-1}) per update`", line 1018 "`elif beta_pi > min(beta_pi_max, alpha)`".
- Upstream anchor: `02_control_loop.md:814-820` "$G = \text{diag}(\alpha, \beta_{\pi}, \gamma_{\text{wm}}, \delta)$ … $\alpha$: critic curvature scale … $\beta_\pi$: policy stochasticity / exploration scale …"; units of curvature `…:711`; `01_definitions.md:372` "$[D_{\mathrm{KL}}]=\mathrm{nat}$".
- Why this is an error: neither chapter defines $\alpha,\beta_\pi,\gamma,\delta$ as measurable quantities. The only operational content is the code, where $\beta_\pi$ is a KL in nats and $\alpha$ a curvature proxy (nat/[z]$^2$); the inequality $\beta_\pi>\alpha$ is not invariant under rescaling $z$, so the rule "skip the policy update" has no coordinate-free meaning. No power law is stated, so "exponent" has no content, and "dimensionless" is asserted, not derived.
- Impact on downstream results: BarrierTypeII (`02_limits_barriers.md`, Section 4.1) is said to encode this requirement; the gating scheduler and the Governor's diagnostic stream depend on these numbers being comparable.
- Fix guidance: (1) define each coefficient as a ratio of like quantities, e.g. $\beta_\pi:=\overline{D_{\mathrm{KL}}(\pi_t\|\pi_{t-1})}/\epsilon_{\mathrm{KL}}$, $\alpha:=\|\nabla_AV\|_{G^{-1}}/\sigma_{\mathrm{TD}}$, $\gamma_{\text{wm}}:=\|S_t-S_{t-1}\|/\|S_t\|$, $\delta:=1-(\text{codebook overlap})$; or (2) state the hierarchy as learning-rate-normalised update timescales $\tau_\delta\gg\tau_\gamma\gg\tau_\alpha$ (the standard two-time-scale condition); (3) rename "exponent" to "coefficient" unless a scaling law is given; (4) apply the same definitions upstream.
- Required new assumptions/permits: none; the definitions are choices.
- Validation plan: verify each defined coefficient is invariant under $z\mapsto cz$ and under reward rescaling where intended.

### [E-009] InfoNCE written with a negatives-only denominator; CPC bound misapplied (was F-009)
- Location: lines 314 and 392-403
- Severity: Moderate
- Type: Computational error (secondary: External dependency)
- Criterion: External
- Origin: this chapter (the book's other statements of InfoNCE are correct)
- Claim: line 314 "$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum \exp(\text{sim}(z_t, z_{neg}))}$"; line 392 "$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k})/\tau)}{\sum_j \exp(\text{sim}(z_t, z^-_j)/\tau)}$"; line 403 "$I(z_t; z_{t+k}) \ge \log N - \mathcal{L}_{\text{CPC}}$"
- Upstream anchor: `02_sieve/04_approximations.md:421` "$\sum_{j=1}^{B} \exp(\text{sim}(z_t, z_j))$" (positive included); `10_appendices/06_losses.md:709` likewise.
- Why this is an error: write $p=\exp(\mathrm{sim}^+/\tau)$ and $S=\sum_j\exp(\mathrm{sim}^-_j/\tau)$. The standard loss is $\mathcal L=\log(1+S/p)$; the chapter's is $\mathcal L'=\log(S/p)$, and $\mathcal L=\log(1+e^{\mathcal L'})>\mathcal L'$ always. The CPC bound is derived for $\mathcal L$ (categorical cross-entropy over $N$ candidates including the positive); substituting $\mathcal L'$ yields a strictly stronger inequality that is false. For unbounded similarities $\mathcal L'\to-\infty$ (the loss has no minimiser); for cosine similarity $\mathcal L'\ge\log(N-1)-2/\tau$, so the claimed right-hand side can reach $\log N-\log(N-1)+2/\tau$, about 20 nat at $\tau=0.1$, exceeding $H(z_t)$ for any modest discrete latent. Line 376 ("$B^2$ pairs") shows the intended object is the standard one.
- Impact on downstream results: Node 6's regulariser, Connection #29, and the anti-collapse claim at line 412.
- Fix guidance: $\mathcal L_{\text{InfoNCE}}=-\log\frac{\exp(\mathrm{sim}(z_t,z_{t+k})/\tau)}{\exp(\mathrm{sim}(z_t,z_{t+k})/\tau)+\sum_j\exp(\mathrm{sim}(z_t,z^-_j)/\tau)}$ at both lines, with $N=1+\#\text{negatives}$ in the bound.
- Required new assumptions/permits: none.
- Validation plan: check that the corrected loss is bounded below by 0 and equals $\log N$ at chance.

### [E-010] "Node 6 (CollapseCheck)" names a non-existent node (was F-010)
- Location: line 412 vs line 99
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim: line 412 "**Audit-friendly**: Node 6 (CollapseCheck) monitors whether contrastive loss is preventing collapse"
- Upstream anchor: registry `06_fields/03_info_bound.md:642` "| 6 | GeomCheck | 3.5 | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE)"
- Why this is an error: no node named CollapseCheck exists in the chapter or the volume; Node 6 is GeomCheck. The collapse monitors are Node 3, Node 11 and Node 55.
- Impact on downstream results: none beyond the mislabel.
- Fix guidance: write "Node 6 (GeomCheck)"; if collapse is meant, cite Node 11 or Node 55.
- Required new assumptions/permits: none.
- Validation plan: grep for "CollapseCheck".

### [E-011] Eikonal regulariser compares $\|\nabla_zV\|$ with a hidden dimensionful constant (was F-018)
- Location: lines 539-545
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 543 "$\mathcal{L}_{\text{Eikonal}} = (\lVert\nabla_z V\rVert - 1)^2$"
- Upstream anchor: `01_definitions.md:396` "Numerical stabilizers like $\epsilon$ always inherit the units of the quantity they are added to."; `02_control_loop.md:846` "$\nabla_z V$ is a 1-form (covector)".
- Why this is an error: the "1" must carry units nat/[z]; rescaling the latent coordinates changes the loss and its minimiser, contrary to the coordinate-invariant regulation programme of the "Diagonal metric law" row (line 258). For a covector the framework's norm is $\|\cdot\|_{G^{-1}}$, and "distance-like" in the book's geometry means $\|\nabla V\|_{G^{-1}}=\text{const}$.
- Impact on downstream results: BarrierGap remediation.
- Fix guidance: $\mathcal L_{\text{Eikonal}}=(\|\nabla_zV\|_{G^{-1}}-\kappa)^2$ with an explicit scale $\kappa$ (nat per $G$-length), or normalise $V$ by $V_{\text{scale}}$ first.
- Required new assumptions/permits: none.
- Validation plan: rescaling test $z\mapsto cz$ leaves the corrected loss unchanged.

### [E-012] Undefined term $\lVert\nabla_A V(z)\rVert^2_{\text{reg}}$ (was F-020)
- Location: line 550
- Severity: Minor
- Type: Typo (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim: line 550 "$\mathcal{L}_{\text{Stiff}} = \max(0, \epsilon - \lVert\nabla_A V(z)\rVert)^2 + \lVert\nabla_A V(z)\rVert^2_{\text{reg}}$"
- Upstream anchor: `03_architecture/01_compute_tiers.md:137` "$\lambda_{\text{stiff}} \max(0, \epsilon - \Vert \nabla_A V \Vert)^2$" (no second term).
- Why this is an error: the subscript "reg" is defined nowhere; read literally the term drives $\|\nabla_AV\|\to0$ everywhere, the flatness the hinge is meant to prevent. The stated intent ("bounded, to prevent explosion") is an upper hinge.
- Impact on downstream results: local.
- Fix guidance: $\mathcal L_{\text{Stiff}}=\max(0,\epsilon-\|\nabla_AV\|)^2+\lambda_{\text{cap}}\max(0,\|\nabla_AV\|-\kappa_{\max})^2$, or drop the second term to match the core tier.
- Required new assumptions/permits: a cap $\kappa_{\max}$ if the upper hinge is kept.
- Validation plan: confirm the loss vanishes on $\epsilon\le\|\nabla_AV\|\le\kappa_{\max}$.

### [E-013] Safety-budget loss disagrees with the Node 1 row and is misdescribed as "hard" (was F-014)
- Location: lines 554-560 vs line 94
- Severity: Minor
- Type: Definition mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim: line 557 "$\mathcal{L}_{\text{Risk}} = \lambda_{\text{safety}} \cdot \mathbb{E}[\max(0, V(z) - V_{\text{max}})]$"; line 560 "Hard Lagrangian enforcement of the risk budget."; line 94 "$\max(0, V(z) - V_{\text{max}})^2$ (Cost Bound)"
- Upstream anchor: registry `06_fields/03_info_bound.md:637` uses the squared hinge.
- Why this is an error: two defect functionals (hinge, nat; squared hinge, nat$^2$) are given for one node, with different boundary gradients. By the chapter's own taxonomy (line 784 "static loss weights are a failure mode"; line 923 budgets go to Method A), a fixed $\lambda_{\text{safety}}$ is a soft penalty, so "hard Lagrangian" is a misdescription unless $\lambda_{\text{safety}}$ is the Method-A dual variable. The same hinge/squared-hinge split occurs for Node 7 (line 100 unsquared; line 550 and `01_compute_tiers.md:137` squared).
- Impact on downstream results: calibration row "Risk/cost budget" (line 969) and Method A examples.
- Fix guidance: use one functional per node (the hinge as constraint metric $\mathcal C_1$ with tolerance 0 is recommended) and write "enforced by the dual multiplier $\lambda_1$ of Method A"; align Node 7 the same way.
- Required new assumptions/permits: none.
- Validation plan: grep for both forms of each node's functional across chapter and registry.

### [E-014] $\mathcal{L}_{\text{nat}}$ has the wrong sign under the book's cost convention; the normalisation has a free index (was F-011)
- Location: lines 581-594
- Severity: Major
- Type: Invalid inference (sign error) (secondary: Dimensional mismatch, free index)
- Criterion: Framework
- Origin: this chapter (the formula appears nowhere else in Volume 1)
- Claim: line 590 "$\mathcal{L}_{\text{nat}} = -\mathbb{E}_{z, a \sim \pi} \left[ \frac{\nabla_z V(z) \cdot f(z, a)}{\sqrt{G_{ii}(z)}} \right]$"; line 582 "*What it maximizes:* Value-decrease rate normalized by $G$"; line 593 "Maximize the alignment between the value gradient $\nabla_A V$ and the realized dynamics $f(z,a)$"
- Upstream anchor (convention settled from the sources): `01_foundations/02_control_loop.md:102` "This chapter uses a cost convention (lower is better)"; `…:141` "assigns a scalar cost-to-go/value to points in $Z$, representing risk/undesirability"; `…:583` "Minimize expected cost-to-go $V(z)$"; `…:1063` "The critic $V$ is a value/cost-to-go function that should decrease along controlled trajectories."; `…:1050` "$\mathcal{L}_f V = dV(f) = \nabla V \cdot f$"; `01_definitions.md:373, 381` "$[V]=\mathrm{nat}$ … cost $c_t=-r_t$". The chapter itself uses the same convention: line 94 (budget $V\le V_{\max}$), line 491 ("value must decrease along trajectories"), line 534 (Lyapunov $\max(0,\dot V+\alpha V)^2$), line 575 ("choose actions that reduce expected cost"), line 600 (descent direction $-G^{-1}\nabla_zV$).
- Why this is an error: $\nabla_zV\cdot f(z,a)=\mathcal L_fV=\dot V$ is the rate of increase of $V$ along the dynamics. Minimising $\mathcal L_{\text{nat}}=-\mathbb E[\dot V/\sqrt{G}]$ maximises $\dot V$. Under the cost convention the policy is thereby rewarded for climbing the cost landscape, while the critic loss ten lines earlier is penalised whenever the trajectory does not descend it and the Hodge term aligns $\Delta z$ with $-G^{-1}\nabla V$. The row label "value-decrease rate" is the negation of what the formula rewards. Independently, $\sum_i\partial_iV\,f^i/\sqrt{G_{ii}}$ divides a scalar by a quantity with a dangling index $i$ and is not a well-formed scalar.
- Impact on downstream results: this is the primary policy objective of Section D and a component of the joint objective $\mathcal L_{\text{Fragile}}$ (line 760), which becomes internally adversarial. No other chapter reuses the formula, so the damage is contained to this chapter's policy-regulation scheme.
- Fix guidance: (1) $\mathcal L_{\text{nat}}:=+\mathbb E_{z,a\sim\pi}\big[\langle\nabla_zV(z),f(z,a)\rangle/\|f(z,a)\|_{G(z)}\big]$ (minimise the normalised ascent rate), or equivalently $\mathbb E[\nabla_zV^\top G^{-1}f]$, with the diagonal form $\sum_i\partial_iV\,f^i/G_{ii}$ if a diagonal metric is used; (2) change the row label to "normalised cost-descent rate"; (3) use $\nabla_zV$ or $\nabla_AV$ consistently in the Mechanism bullet.
- Required new assumptions/permits: none.
- Validation plan: on a quadratic $V$ with linear $f$, confirm that gradient descent on the corrected loss moves the policy toward $\dot V<0$ and that the Lyapunov loss decreases jointly.

### [E-015] "Geodesic Zeno" term applies the state-space metric to a difference of action distributions (was F-012)
- Location: lines 609-615
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: line 612 "$\mathcal{L}_{\text{Zeno}} = \lVert\pi_t - \pi_{t-1}\rVert^2_{G}$"; line 615 "Penalizes high-frequency switching, weighted by geometry."
- Upstream anchor: `01_foundations/02_control_loop.md:837-841` "$G_{ij}(z) = (G_V)_{ij}(z) + \lambda_G (G_\pi)_{ij}(z)$, $(G_\pi)_{ij}(z) := \mathbb{E}_{a\sim\pi}[\partial_{z_i}\log\pi(a|z)\,\partial_{z_j}\log\pi(a|z)]$" (a bilinear form on $T_z\mathcal Z$).
- Why this is an error: $\pi_t-\pi_{t-1}$ is a signed measure on $\mathcal A$; $G(z)$ acts on tangent vectors of $\mathcal Z$. The quadratic form is undefined unless $\mathcal A$ is identified with $T_z\mathcal Z$, which the book does not do. The prose describes a scalar state-dependent weight, not a $G$-norm.
- Impact on downstream results: Node 2's geometric variant and the "Geometry-aware (Natural)" policy column.
- Fix guidance: use the policy's own information metric $D_{\mathrm{KL}}(\pi(\cdot|z_t)\|\pi(\cdot|z_{t-1}))$ (already the "Euclidean fallback", but in fact the natural choice), optionally weighted by a scalar such as $\operatorname{tr}G(z_t)$; or, if actions are embedded in $\mathcal Z$ via the motor atlas, write $\|D_A(\pi_t)-D_A(\pi_{t-1})\|_G^2$ explicitly.
- Required new assumptions/permits: an explicit embedding $D_A:\mathcal A\to\mathcal Z$ if the $G$-norm is kept.
- Validation plan: type-check the expression: both arguments of the form must be elements of $T_z\mathcal Z$.

### [E-016] TD synchronisation target uses reward $r$ while $V$ is a cost in the chapter (was F-013)
- Location: line 660
- Severity: Minor
- Type: Notation conflict (sign convention)
- Criterion: Framework
- Origin: this chapter
- Claim: line 660 "$\mathcal{L}_{\text{Sync}_{V-\pi}} = \lVert V(z) - (r + \gamma V(z')) \rVert^2 \quad (\text{TD-Error})$"
- Upstream anchor: `01_definitions.md:381` "Per-step reward $r_t$ (or cost $c_t=-r_t$) has units nat"; `02_control_loop.md:102, 1063` (cost convention; $V$ decreases along controlled trajectories).
- Why this is an error: a critic trained to $V=r+\gamma V'$ is high where reward is high; the same chapter reads $V$ as cost (Node 1 budget $V\le V_{\max}$, line 94; $\mathcal L_{\text{Risk}}$, line 557; "log-risk" mapping, line 983; BarrierCap $V\to\infty$ on unsafe states). The two readings cannot both hold for one function.
- Impact on downstream results: the joint objective (line 760) mixes conventions; the Advantage-gap statement (line 663) inherits the ambiguity.
- Fix guidance: $\mathcal L_{\text{Sync}_{V-\pi}}=\|V(z)-(c+\gamma V(z'))\|^2$ with $c=-r$, plus a one-line convention statement at the top of Section C.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for "$r +$" and "$-r$" targets after the edit.

### [E-017] Information-stability window: loose threshold range and a conflated entropy symbol (was F-015)
- Location: lines 709-719; line 866; line 282; lines 966, 972-980
- Severity: Moderate
- Type: Parameter inconsistency (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter (conflation); upstream `04_control/03_coupling_window.md:121-129` for the $\epsilon$ range
- Claim: line 709 "With thresholds $0<\epsilon<\log|\mathcal{K}|$"; lines 712-719 "$\mathcal{L}_{\text{window}} := \mathrm{ReLU}(\epsilon - I(X_t;K_t))^2 + \mathrm{ReLU}(H(K_t)-(\log|\mathcal{K}|-\epsilon))^2$ … $H(K)$ must not saturate."; line 866 "**Code usage:** keep $H(K)$ away from collapse and away from saturation."; line 282 "$\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K)\,\Vert\,\mathrm{Unif}(\mathcal{K}))$ (anti-collapse)"; line 977 "$\log|\mathcal{K}|-H(K)\ \le\ -\log(1-\rho_{\text{dead}})$ … $\approx 0.051$ nats".
- Upstream anchor: `04_control/03_coupling_window.md:124-128` "constants $0<\epsilon<\log|\mathcal{K}|$ such that … $\epsilon \le I(X_t;K_t)$ and $H(K_t)\le \log|\mathcal{K}|-\epsilon$"; gloss `…:146` "your belief can't be spread uniformly over all macro-states".
- Why this is an error: (i) if $H(K_t)$ is the marginal entropy of the macro symbol (the reading under which $I(X_t;K_t)\le H(K_t)$ holds), then $\epsilon\le I\le H\le\log|\mathcal K|-\epsilon$ forces $\epsilon\le\frac12\log|\mathcal K|$; for $\epsilon$ in $(\frac12\log|\mathcal K|,\log|\mathcal K|)$, admitted by both the theorem and this chapter, the feasible set is empty, $\mathcal L_{\text{window}}$ cannot vanish, and by line 942 the dual multiplier diverges. (ii) Whichever reading is intended, the chapter uses one symbol $H(K)$ for the quantity that line 282 pushes toward $\log|\mathcal K|$, that line 977 requires to be within $0.051$ nat of $\log|\mathcal K|$, and that lines 719 and 866 require to stay $\epsilon$ below $\log|\mathcal K|$. These are jointly satisfiable only for $\epsilon\le0.051$ nat, which makes the grounding bound $I\ge\epsilon$ vacuous. The per-step belief entropy $H(q(K\mid x_t))$ (Node 3) and the usage entropy $H(\bar p(K))$ (Node 11) are different random variables and must not share a symbol or a setpoint.
- Impact on downstream results: Method A/B setpoints for code usage; the calibration table (lines 960-969); every place that cites "$H(K)$ must not saturate".
- Fix guidance: (1) state the window with $0<\epsilon\le\frac12\log|\mathcal K|$, or with two thresholds $\epsilon_I+\epsilon_H\le\log|\mathcal K|$, here and upstream; (2) write the per-step entropy as $H(q(K\mid x_t))$ and the usage entropy as $H(\bar p(K))$; (3) make line 866 read "keep $H(\bar p(K))$ near $\log|\mathcal K|$ (usage) and $H(q(K\mid x_t))$ low (compactness)".
- Required new assumptions/permits: none.
- Validation plan: check that the corrected constraint set is non-empty for every admissible $\epsilon$ (e.g. a deterministic encoder with uniform usage satisfies all of them).

### [E-018] The "discrete PID update" is not a PID controller (was F-016)
- Location: lines 868-884
- Severity: Moderate
- Type: Computational error (secondary: External dependency)
- Criterion: External
- Origin: this chapter (the Governor restates Method B in positional form)
- Claim: lines 871-882 "$\lambda_{t+1} = \Pi_{[\lambda_{\min},\lambda_{\max}]}\Big(\lambda_t + K_p e_t + K_i \sum_{t' \le t} e_{t'} + K_d(e_t-e_{t-1})\Big)$"
- Upstream anchor: `07_cognition/02_governor.md:64` "**3.5.B (PID):** $\lambda_{t+1} = K_p e_t + K_i \sum e + K_d \Delta e$"; `…:242` "PID (3.5.B) | Linear filter with fixed $(K_p,K_i,K_d)$, $H\ge2$"; the cited PID-Lagrangian method (Stooke et al. 2020) is positional.
- Why this is an error: ignoring the clip and unrolling, $\lambda_{t+1}=\lambda_0+K_p\sum_{s\le t}e_s+K_i\sum_{s\le t}\sum_{s'\le s}e_{s'}+K_d(e_t-e_{-1})$. The "$K_p$" gain integrates, "$K_i$" double-integrates, and "$K_d$" acts proportionally; there is no derivative action. This is neither the positional nor the incremental PID form. The double integrator makes the multiplier dynamics marginally stable at best and produces the oscillations that line 886 proposes to damp with $K_d$, which in this form damps nothing.
- Impact on downstream results: all setpoint regulators (lines 863-866, 924) and the Governor's claim to subsume Method B as a "linear filter, $H\ge2$".
- Fix guidance: either the positional form $\lambda_{t+1}=\Pi(K_pe_t+K_i\sum_{t'\le t}e_{t'}+K_d(e_t-e_{t-1}))$ (matches the citation and the Governor), or the incremental form $\lambda_{t+1}=\Pi(\lambda_t+K_p(e_t-e_{t-1})+K_ie_t+K_d(e_t-2e_{t-1}+e_{t-2}))$; note that with $K_p=K_d=0$ the incremental form is exactly Method A with $\eta_\lambda=K_i$.
- Required new assumptions/permits: none.
- Validation plan: simulate a first-order plant and confirm the corrected controller has standard P/I/D step responses.

### [E-019] Primal-dual ascent described as "memoryless ($H=0$)" (was F-017)
- Location: line 1041
- Severity: Minor
- Type: Conceptual
- Criterion: External (and Framework: contradicts the Governor's table)
- Origin: this chapter ($H=0$); upstream `07_cognition/02_governor.md:63, 246` ("memoryless")
- Claim: line 1041 "**Primal–Dual (Method A)** = affine policy, memoryless ($H=0$)"
- Upstream anchor: `07_cognition/02_governor.md:241` "Primal-Dual (3.5.A) | $\pi_{\mathfrak{G}}(s_t) = \lambda_{t-1} + \eta_\lambda s_t$ (affine, $H=1$)"; `…:246` "The Primal-Dual update is a memoryless affine map."; this chapter line 832.
- Why this is an error: the dual update accumulates the entire violation history, $\lambda_t=\Pi(\lambda_0+\eta_\lambda\sum_{s<t}(\mathcal C_s-\epsilon))$; it is the integral controller on which the cited PID-Lagrangian paper is built. "Memoryless" would mean $\lambda_t=g(\mathcal C_t)$, which is Method C. "$H=0$" here contradicts "$H=1$" upstream.
- Impact on downstream results: the Governor's subsumption proposition (`…:236-248`) is stated in terms of these window lengths.
- Fix guidance: "Primal–Dual (Method A) = integral (I) controller: affine recursion in $\lambda_{t-1}$ with unit pole; PID (Method B) adds P and D terms." Align the Governor's "memoryless" wording.
- Required new assumptions/permits: none.
- Validation plan: consistency check of the three window lengths across both chapters.

### [E-020] Registry formula for Node 11 inverts the capacity gap (added by verifier, V-001)
- Location: line 108 vs registry `06_fields/03_info_bound.md:647`
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: upstream `06_fields/03_info_bound.md:647`
- Claim: line 108 "**11** | **ComplexCheck ($\mathrm{Rep}_K$)** | … | $1 - H(K)/\log\lvert\mathcal{K}\rvert$ (Capacity Gap)"
- Upstream anchor: `06_fields/03_info_bound.md:647` "| 11 | ComplexCheck | 3.5 | $H(K)/\log\lvert\mathcal{K}\rvert$"
- Why this is an error: the chapter's defect is the gap (large means collapse); the registry's quantity is normalised usage (large means healthy). A reader following the registry thresholds the wrong tail.
- Impact on downstream results: any automated use of the registry formulas.
- Fix guidance: restore "$1-$" in the registry row, or state in both places that the monitored quantity is the gap.
- Required new assumptions/permits: none.
- Validation plan: grep both files for the Node 11 formula after the edit.

### [E-021] Symbol overloading: $\gamma$ has four meanings, $\alpha$ two with different units (was F-019)
- Location: $\gamma$: lines 125 (friction), 185/203 (WM volatility), 352 (VICReg target std), 660 (discount); $\alpha$: lines 183/201 (dimensionless coefficient) vs 534 (Lyapunov rate, step$^{-1}$ per line 968); also $\lambda$ (lines 54, 367, 760/816) and $J$ (lines 466-468 vs 478)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter (upstream uses $\gamma_{\text{wm}}$, `02_control_loop.md:817`)
- Claim: line 210 "$\delta \ll \gamma \ll \alpha$" and line 534 "$\max(0, \dot{V}(z) + \alpha V(z))^2$"
- Upstream anchor: `01_foundations/02_control_loop.md:817` "$\gamma_{\text{wm}}$: world-model volatility / non-stationarity scale."
- Why this is an error: "$\gamma\ll\alpha$" read with the Lyapunov $\alpha$ is a different (and dimensionally inconsistent) statement; the gating code's `gamma` is unrelated to the friction $\gamma$ of Node 27. This changes logical meaning and is not merely stylistic.
- Impact on downstream results: local readability; the upstream parameter table already distinguishes $\gamma_{\text{wm}}$.
- Fix guidance: use $\gamma_{\text{wm}}$ as upstream, keep bare $\gamma$ for the discount, $\sigma_\star$ for the VICReg target, $\alpha_{\text{Lyap}}$ for the decay rate, and $\iota_t$ (or $j_t$) for the jump index.
- Required new assumptions/permits: none.
- Validation plan: grep for bare $\gamma$ and $\alpha$ after the edit.

## Scope restrictions and clarifications
- The $V$ convention used to judge E-014 and E-016 was taken from the foundations chapters (cost-to-go, lower is better; reward form via $\mathcal C=-\mathcal R$) and from this chapter's own Node 1, Lyapunov and Hodge statements, not assumed.
- E-004 and E-005 are inherited from `05_geometry/04_equations_motion.md`; this chapter's rows should be brought back into agreement with the upstream rows once those are corrected.
- E-017(i) depends on reading $H(K_t)$ as the marginal entropy of the macro symbol; under the per-step-posterior reading the feasibility bound does not apply, but the symbol conflation in (ii) does.
- Several findings (E-007, E-008) concern quantities that no other chapter defines operationally; the fixes are definitional choices rather than derivations.

## Open questions
- Should lettered sub-nodes (7a-7d, 12a) be first-class members of the diagnostic index set used by the Sieve PASS condition?
- Is $\mathcal L_{\text{nat}}$ meant to replace or complement the advantage-based policy gradient? The table presents them as alternatives, but the joint objective (line 760) does not say which is $\mathcal L_{\text{Task}}$.
- Is the Node 26 check intended for the overdamped regime at all, or only as a numerical-integration diagnostic in the inertial regime?

## Rejected candidate findings
None. All twenty stage-1 findings were confirmed or kept with adjustments (F-005, F-007: impact claims corrected; F-008: origin widened; F-009, F-013, F-015: supporting arguments corrected or made conditional).
