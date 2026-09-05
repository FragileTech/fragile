# Mathematical Review: docs/source/1_agent/03_architecture/01_compute_tiers.md

## Metadata
- Reviewed file: docs/source/1_agent/03_architecture/01_compute_tiers.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (2801 lines); Torch checks re-run with `.venv/bin/python` (torch 2.7.1); code-vs-implementation claims checked against `src/fragile/core/layers/atlas.py`
- Framework anchors (definitions/axioms/permits):
  - docs/source/1_agent/01_foundations/01_definitions.md:313 (symbol-permutation symmetry $S_{|\mathcal K|}$); latent split $Z_t=(K_t,z_n,z_{\text{tex}})$ with texture reconstruction-only
  - docs/source/1_agent/01_foundations/02_control_loop.md:684-690 (sensitivity metric $G_{ij}$), :908-933 ($G_{zz}$, $G_{\pi,ij}(z)=\mathbb E_{a\sim\pi}[\partial_i\log\pi\,\partial_j\log\pi]$, coordinate-invariance rationale at :924)
  - docs/source/1_agent/02_sieve/01_diagnostics.md:100 (StiffnessCheck $\max(0,\epsilon-\Vert\nabla_A V\Vert)$), :138 (stiffness row, $G\in T^*_2(\mathcal Z)$, $\lambda_{\min}(G)>\epsilon$)
  - docs/source/1_agent/07_cognition/02_governor.md:613, :638 (the only other mentions of `metalearning.md`)
  - src/fragile/core/layers/atlas.py:19-93 (Poincare helpers), :99-110 (encoder return tuple), :420-441 (`TopologicalDecoder.forward`), :984-1099 and :1333-1395 (`SpectralLinear`/`NormGatedGELU` primitives), :1179-1275 (`PrimitiveAttentiveAtlasEncoder.forward`), :1397-1408 (`PrimitiveTopologicalDecoder.forward`); src/fragile/core/layers/topology.py:443 (`compute_orthogonality_loss`)
  - Internal definitions of this chapter: Macro-State Tree (lines 1829-1841), Peeling Step (line 2013), Rescaling (line 2023), Total Reconstruction (line 2035), Factorized Jump Operator (line 2446), `FactorizedJumpOperator.__init__` (lines 2551-2556), `compute_jump_consistency_loss` (lines 2649-2655)
  - External references: {cite}`lee2012smooth` (charts as homeomorphisms onto open subsets of $\mathbb R^n$), {cite}`mehta2014exact` (RG and deep learning)

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 11
- Minor: 12
- Notes: 0
- Primary themes:
  1. The Tier-4 "Riemannian" control code has a systematic sign error (the actor loss maximises the increase of a cost/Lyapunov value that the critic is simultaneously forced to decrease), its state-space Fisher estimator returns identically zero because it differentiates the log-density of an `rsample()`d action, routines A and B have no gradient path to the policy parameters, and one metric computed on observations is applied to tensors in latent and policy space.
  2. Several geometric statements labelled as rigorous are wrong against standard mathematics: the minimum-chart table (torus, Klein bottle, reason for the sphere), the norm-preservation proposition for rectangular semi-orthogonal matrices, the exponent sign in the hyperbolic latent metric, the Poincare-ball depth formula, and the claim that GELU is 1-Lipschitz.
  3. The Dynamical Isometry theorem is vacuous as stated (the reconstruction it differentiates is an algebraic identity, so $J=I$ for any weights) and its proof sketch supplies no lower bound on singular values; the stacked TopoEncoder code has an input/output dimension mismatch for blocks $\ell\ge1$, calls a method that does not exist, and its scale-decay loss pushes opposite to its stated purpose.
  4. Code/API mismatches between the chapter and itself (constructor keyword, tuple return, undefined `compute_state_space_fisher`) and between the chapter's "current implementation" blocks and `src/fragile/core/layers/atlas.py`; a reference to a non-existent `metalearning.md`.
- Mechanical pre-pass: `xref_report.md` lists no dangling references or duplicate labels for this file.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Tier 1: Core Fragile Agent, lines 137, 143-177 | Moderate | Algorithm mismatch | Framework | this chapter (symbol inherited from 02_sieve/01_diagnostics.md:100) | Stiffness penalty differentiates a tensor not in the graph; formula uses $\nabla_A$, code uses `states` |
| E-002 | Tier 2, lines 209, 270-278 | Minor | Algorithm mismatch | Framework | this chapter | Oscillation term: norm in formula, mean-squared in code |
| E-003 | Tier 2 docstring and Tier 4, lines 224-226, 360, 407-409, 590 | Minor | Citation / reference error | Framework | this chapter | `compute_state_space_fisher` referenced but defined nowhere |
| E-004 | Tier 4 routines A and B, lines 334-382, 387-434 | Moderate | Algorithm mismatch | Framework | this chapter | "Policy loss" in A and B has no gradient path to the actor |
| E-005 | Tier 4 routines A-D, lines 341-381, 419-433, 474-509, 612-643 | Major | Algorithm mismatch | Framework | this chapter | Actor loss maximises the increase of a cost/Lyapunov value |
| E-006 | Tier 4, lines 348-361, 375-378, 505-507, 680-688 | Moderate | Conceptual | Framework | this chapter | $\langle\nabla V,v\rangle_G$ implemented as $G^{-1}$-weighted covector-vector sum; G-norm written with $G^{-1}$ |
| E-007 | `GeometryAwareLearner._compute_state_fisher`, lines 520-539 | Major | Computational error | Framework | this chapter | Fisher estimator identically zero with `rsample()` |
| E-008 | `RiemannianFragileAgent.train_step`, `_geodesic_dist`, lines 586-591, 630-643, 680-688 | Moderate | Dimensional mismatch | Framework | this chapter | Metric on `batch.obs` applied to latent- and policy-space tensors |
| E-009 | Defect Functional Costs, line 714 | Minor | Citation / reference error | Framework | this chapter | Reference to non-existent `metalearning.md` |
| E-010 | Manifold Atlas Theory, lines 758-768 | Moderate | Computational error | External | this chapter | Minimum-chart table wrong for torus and Klein bottle; wrong reason for sphere |
| E-011 | Why Orthogonality? table and Proposition "Gradient Preservation via Orthogonality", lines 839-845, 2071-2083 | Moderate | Invalid inference | External | this chapter | Norm preservation fails for rectangular semi-orthogonal $W$ |
| E-012 | Definition "Attentive Routing Law", lines 1275-1302 | Minor | Miswording | Framework | this chapter | "Permutation invariant" describes equivariance |
| E-013 | Architecture Specification and Topological Decoder Module, lines 1331-1371, 1687-1777 | Minor | Algorithm mismatch | Framework | this chapter | "Current implementation" differs from atlas.py (Euclidean vs Poincare-ball pipeline) |
| E-014 | Corollary "The Hyperbolic Embedding", line 1852 | Minor | Computational error | External | this chapter | Poincare-ball depth is $2\tanh^{-1}(r)$, not $\tanh^{-1}(r)$ |
| E-015 | Proposition "Texture as the Ideal Boundary", lines 1886-1903 | Moderate | Proof gap / omission | Framework | this chapter | Proof needs an infinite-depth tree and objects never defined |
| E-016 | Definition "The Latent Metric Tensor", lines 1909-1928 | Moderate | Computational error | External | this chapter | Exponent sign $e^{-2\rho}$ contradicts the upper-half-space model |
| E-017 | Proposition "Forward Activation Stability", lines 2089-2110, 2270-2271 | Minor | Proof gap / omission | Framework | this chapter | Unit variance at $\ell=0$ not by construction; $\sigma<1$ assumed; $\sigma$ not detached |
| E-018 | Mechanism 3: Spectral Normalization, line 2120 | Minor | Computational error | External | this chapter | GELU is not 1-Lipschitz ($\sup\mathrm{GELU}'\approx1.129$) |
| E-019 | Theorem "Dynamical Isometry without Skip Connections", lines 2029-2038, 2137-2149 | Major | Invalid inference | Framework | this chapter | Theorem vacuous ($J=I$ identically); sketch gives no lower bound |
| E-020 | Rigorous Interpretation: RG Flow, lines 1983-1986, 2150-2176 | Moderate | Conceptual | External | this chapter | Wilsonian-RG identification reversed |
| E-021 | Stacked TopoEncoder Module, lines 2210-2230, 2248-2288 | Moderate | Dimensional mismatch | Framework | this chapter | Blocks $\ell\ge1$ have wrong input/output dimension |
| E-022 | `StackedTopoEncoder.orthogonality_loss`, lines 2318-2328 vs 2062-2070 | Minor | Algorithm mismatch | Framework | this chapter | `orthogonality_loss()` does not exist on the primitives |
| E-023 | Training Losses for Scale Separation, lines 2344-2350 | Moderate | Conceptual | Framework | this chapter | Scale-decay loss pushes opposite to its stated purpose |
| E-024 | Factorized Jump Operators, line 2368 vs 2478 | Minor | Miswording | Framework | this chapter | Prose says factorisation "automatically ensures consistency"; formal text says it does not |
| E-025 | Computational Cost Analysis, lines 2521-2531 | Minor | Computational error | External | this chapter | "Forward (all pairs)" is $O(K^2 r d_n)$, not $O(K r d_n)$ |
| E-026 | Integration with PrimitiveAttentiveAtlasEncoder, lines 2747-2787 | Minor | Algorithm mismatch | Framework | this chapter | Constructor called with unknown keyword; tuple stored as scalar loss |

## Detailed findings

### [E-001] Tier-1 stiffness penalty differentiates a tensor that is not in the graph, and with respect to a variable that differs from the formula (was F-001)
- Location: Tier 1: Core Fragile Agent (Minimal), lines 137, 143-177
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter (the `∇_A` symbol is inherited from docs/source/1_agent/02_sieve/01_diagnostics.md:100)
- Claim (lines 143-177): "states.requires_grad_(True); v = critic_values if critic_values.requires_grad else critic_values.detach(); grad_v = torch.autograd.grad(v.sum(), states, create_graph=True, retain_graph=True)[0]", implementing (line 137) "$\lambda_{\text{stiff}} \max(0, \epsilon - \Vert \nabla_A V \Vert)^2$".
- Upstream anchor: docs/source/1_agent/02_sieve/01_diagnostics.md:100 "StiffnessCheck ... $\max(0, \epsilon - \Vert \nabla_A V \Vert)$ (Gain > $\epsilon$)"; :138 "7 (Stiffness) | $G \in T^*_2(\mathcal{Z})$ | Spectral Gap | Is $\lambda_{\min}(G) > \epsilon$?".
- Why this is an error: `critic_values` is passed in already computed; setting `states.requires_grad_(True)` afterwards does not insert `states` into the graph that produced `critic_values`. The verifier reproduced the failure with torch 2.7.1: `torch.autograd.grad(v.sum(), states, ...)` raises a `RuntimeError` ("One of the differentiated Tensors appears to not have been used in the graph" when the critic has parameters; "does not require grad" in the `detach()` branch). Independently, the displayed formula penalises $\Vert\nabla_A V\Vert$ while the code differentiates with respect to `states`; these are different diagnostics. The `∇_A` notation is copied verbatim from the diagnostics table, whose row 138 places stiffness in state space, so the symbol ambiguity is an upstream inheritance; the autograd defect is local and real.
- Impact on downstream results: The Tier-1 loss is the base of Tiers 2-4 ($\mathcal L^{\text{std}}$, $\mathcal L^{\text{full}}$ add to it); the stiffness term is the one that covers Mode T.D (Freeze) in the coverage claim of line 140.
- Fix guidance:
  1. Recompute the critic inside the function on `states.requires_grad_(True)` (or pass the critic module in and call it there).
  2. Make the formula and code agree on the differentiation variable, following diagnostics.md:138 (state $z$): write $\max(0,\epsilon-\Vert\nabla_z V\Vert)^2$.
  3. Remove the `detach()` branch, which guarantees failure.
- Required new assumptions/permits: none.
- Validation plan: Run the function on a small critic MLP with a batch of states; confirm no exception and a non-zero gradient norm. Check that the symbol used matches the diagnostics table after the upstream ambiguity is resolved.

### [E-002] Oscillation term: norm in the formula, squared mean in the code (was F-002)
- Location: Tier 2, lines 209, 270-278
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 209): "$\lambda_{\text{osc}} \Vert z_t - z_{t-2} \Vert$"; (line 278): "return (z_t - z_t_minus_2).pow(2).mean()".
- Upstream anchor: n/a (defined in this chapter).
- Why this is an error: The displayed loss is the unsquared norm; the implementation is $\tfrac1Z\Vert z_t-z_{t-2}\Vert^2$. They have different units and different gradient behaviour near zero (the norm has a non-vanishing subgradient at $z_t=z_{t-2}$, the squared form does not), so $\lambda_{\text{osc}}$ is not transferable between them.
- Impact on downstream results: Local.
- Fix guidance:
  1. Write $\lambda_{\text{osc}}\Vert z_t-z_{t-2}\Vert^2$ in the formula, or
  2. use `.norm(dim=-1).mean()` in the code.
- Required new assumptions/permits: none.
- Validation plan: Confirm formula and code agree on a unit test with a known difference vector.

### [E-003] `compute_state_space_fisher` is referenced as an existing function but defined nowhere (was F-003)
- Location: Tier 2 docstring and Tier 4, lines 224-226, 360, 407-409, 590
- Severity: Minor
- Type: Citation / reference error (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 224-226): "compute G via compute_state_space_fisher() in state space ({ref}`sec-the-metric-hierarchy-fixing-the-category-error`)".
- Upstream anchor: docs/source/1_agent/01_foundations/02_control_loop.md:908-933 defines $G_{zz}$ and $G_{\pi,ij}(z)$ mathematically but contains no such function; `grep -rn "def compute_state_space_fisher" docs/source src` returns nothing.
- Why this is an error: Three of the four Tier-4 routines depend on this call (with an `include_value_hessian` keyword whose semantics are never given). The only concrete estimator in the chapter is `_compute_state_fisher` (lines 520-539), which is broken (E-007).
- Impact on downstream results: Tier-4 pseudocode is not executable as documented.
- Fix guidance:
  1. Define the function in the chapter (or the losses appendix), including the meaning of `include_value_hessian`, or
  2. point to a concrete module path.
- Required new assumptions/permits: none.
- Validation plan: `grep` for the definition after the edit; ensure every call site resolves.

### [E-004] Routines A and B have no gradient path to the policy parameters (was V-001)
- Location: Tier 4 routines A and B, lines 334-382, 387-434
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: Routine A "connect[s] Policy and Value"; routine B produces a "Policy Loss" (line 419, `loss_policy`). In both, `state_velocity = next_state - state` (line 373) and `dynamics = next_states - states` (line 422).
- Upstream anchor: n/a (defined in this chapter); contrast routines C (line 503) and D (line 631).
- Why this is an error: The velocities are differences of input data tensors, `grad_v` is a function of the critic only, and `policy_action` (line 337) is never used. The only object that can depend on the policy is `fisher_diag` (via $\partial\log\pi/\partial z$), which is not detached; so the "policy loss" either has zero gradient with respect to the actor or trains the actor to change its state sensitivity rather than its action. Because `create_graph=True` (lines 369, 416), the gradient of `loss_policy` flows into the critic parameters instead, steering the critic's gradient field against the observed dynamics. Routines C and D correctly route the velocity through `world_model(s, actor(s))`.
- Impact on downstream results: Routines A and B, presented as the natural-gradient and Lyapunov realisations of Tier 4, do not train the policy as described.
- Fix guidance:
  1. Compute the velocity through the world model and the current policy, as in C/D.
  2. Detach the metric (`fisher_diag.detach()`) so the policy loss does not act through $G$.
  3. Remove or use `policy_action`.
- Required new assumptions/permits: a differentiable world model (already assumed in C and D).
- Validation plan: Check that `torch.autograd.grad(loss_policy, actor.parameters())` is non-zero and that `grad(loss_policy, critic.parameters())` is zero after the fix.

### [E-005] Sign error: the "value-decrease" actor loss maximises the increase of a cost-to-go / Lyapunov function (was F-004)
- Location: Tier 4 routines A-D, lines 341-381, 419-433, 474-509, 612-643
- Severity: Major
- Type: Algorithm mismatch (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 378-381): "# 5. The Loss: maximize value decrease (make V decrease fast) / return -natural_decrease.mean()" with "natural_decrease = (grad_v * state_velocity * metric_inv).sum(dim=-1)"; routine C (lines 476-509): "cost = -r ... target_v = cost + gamma*V(s_next) ... actor_loss = -value_change_geo.mean()"; routines B/D (lines 430, 620): critic trained with "violation = relu(v_dot + target_decay * V)" / "lyap_loss = relu(V_dot + zeta*V)^2".
- Upstream anchor: n/a (the Lyapunov convention $\dot V\le-\alpha V$ is stated in this chapter at lines 400, 427, 618).
- Why this is an error: In every routine $V$ is a cost (C converts reward to cost explicitly; B and D impose $\dot V\le-\alpha V$, which is meaningful only for a positive cost-like function). `value_change_geo` $=\sum_i\partial_iV\,\dot z^i/G_{ii}$ is a metric-weighted estimate of $\dot V$. Minimising `-value_change_geo` maximises $\dot V$, driving the state uphill in cost, in direct opposition to the constraint the critic is simultaneously trained to satisfy. The docstring's Euclidean reference "L = -log_prob * advantage" is a reward-maximising objective and does not rescue the sign once $V$ is a cost. The "wait state" branch in D (line 644) therefore pauses a policy update that, when active, pushes in the wrong direction.
- Impact on downstream results: All Tier-4 training code (A, B, C, D) and any reader implementing "Algorithm 3"; Tier 4 is presented as the covariant realisation of the Lyapunov control of the Sieve.
- Fix guidance:
  1. Use `actor_loss = +value_change_geo.mean()` (minimise the metric-weighted $\dot V$) in A, B, C and `L_nat = +torch.mean((grad_V*velocity)*G_inv)` in D; or
  2. if $V$ is meant as a reward value, drop `cost = -r` and flip the Lyapunov constraint to $\dot V\ge\alpha(V^*-V)$ consistently in B and D.
- Required new assumptions/permits: none.
- Validation plan: On a 1-D quadratic cost with a linear world model, confirm the actor update decreases $V$ along trajectories and that the critic's Lyapunov violation decreases rather than being fought by the actor.

### [E-006] "$\langle\nabla V,v\rangle_G$" is implemented as a $G^{-1}$-weighted covector-vector sum; the "G-norm" is defined with $G^{-1}$ (was F-005)
- Location: Tier 4, lines 348-361, 375-378, 505-507, 680-688
- Severity: Moderate
- Type: Conceptual (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 348-361, 375-378): "RIEMANNIAN (This function): L = -<grad_V, velocity>_G # Inner product under sensitivity metric", implemented as "(grad_v * state_velocity * metric_inv).sum(dim=-1)"; (line 685): "RIEMANNIAN: ||π_new - π_old||²_G = (π_new - π_old)ᵀ G⁻¹ (π_new - π_old)".
- Upstream anchor: docs/source/1_agent/01_foundations/02_control_loop.md:921 "$G_{zz} = \lambda_G\,\mathbb{E}[(\nabla_z \log \pi)^2] + \text{Hess}_z(V)$"; :924 (coordinate-invariance rationale); :932 "$G_{\pi,ij}(z) = \mathbb{E}_{a\sim\pi}[\partial_i \log\pi\,\partial_j\log\pi]$".
- Why this is an error: $dV$ is a covector and $\dot z$ a vector; their pairing $\dot V=\partial_iV\,\dot z^i$ needs no metric. The Riemannian inner product of the Riemannian gradient $\operatorname{grad}_GV=G^{-1}dV$ with $\dot z$ is $G_{ij}(G^{ik}\partial_kV)\dot z^j=\partial_jV\,\dot z^j$, again metric-free. The implemented $\sum_i\partial_iV\,\dot z^i/G_{ii}$ is neither: it pairs two lower indices through $G^{ii}$ and is not coordinate-invariant, contradicting the invariance rationale at control_loop.md:924. Likewise $\Vert d\Vert_G^2$ means $d^\top Gd$, not $d^\top G^{-1}d$. The heuristic "large $G$ shrinks the step" is legitimately obtained by preconditioning the update with $G^{-1}$ (natural gradient), not by reweighting the objective.
- Impact on downstream results: The stated geometry-aware (coordinate-invariant) property of Tier 4 does not hold for the implemented objective.
- Fix guidance:
  1. Use the plain directional derivative $\partial_iV\,\dot z^i$ as the objective.
  2. Apply $G^{-1}$ as a preconditioner on the policy-parameter (or action) update.
  3. Define the trust region as $d^\top Gd$ with $G$ on the correct space (see E-008).
- Required new assumptions/permits: none.
- Validation plan: Check invariance of the objective under a linear reparametrisation $z\mapsto Az$ (with $G\mapsto A^{-\top}GA^{-1}$) numerically.

### [E-007] State-space Fisher estimator is identically zero because it differentiates the log-density of an `rsample()`d action (was F-006)
- Location: `GeometryAwareLearner._compute_state_fisher`, lines 520-539
- Severity: Major
- Type: Computational error
- Criterion: Framework (adjusted from External by the verifier)
- Origin: this chapter
- Claim (lines 520-539): "action = dist.rsample(); log_prob = dist.log_prob(action).sum(dim=-1); grad_z = torch.autograd.grad(log_prob.sum(), state_grad, ...)[0]; fisher_diag = grad_z.pow(2).mean(dim=0); return fisher_diag + 1e-6", documented as "$G_{ii}=\mathbb E[(\partial\log\pi/\partial z_i)^2]$".
- Upstream anchor: docs/source/1_agent/01_foundations/02_control_loop.md:932 "$G_{\pi,ij}(z) = \mathbb{E}_{a \sim \pi}[\partial_{z_i}\log\pi(a|z)\,\partial_{z_j}\log\pi(a|z)]$" (score evaluated at a fixed sample $a$).
- Why this is an error: With `rsample`, $a=\mu(z)+\sigma\varepsilon$, so $\log\pi(a|z)=-\varepsilon^2/2-\log\sigma-\tfrac12\log2\pi$ is independent of $z$ and its gradient vanishes exactly. The verifier re-ran the check with torch 2.7.1 (Gaussian policy, fixed std 0.5): `max|∂ log π(a|z)/∂z| = 0.0`, `fisher_diag = [0,0,0,0,0]`; with `dist.sample()` or `dist.rsample().detach()` the diagonal is $O(0.1$-$0.3)$. Hence $G\equiv10^{-6}$ and $G^{-1}=10^{6}$ everywhere: the Riemannian weighting collapses to a huge constant carrying no state information. The estimator fails to compute the framework's own quantity, so the criterion is Framework.
- Impact on downstream results: Every routine consuming `_compute_state_fisher`/`compute_state_space_fisher` (E-003): natural-gradient loss, Lyapunov loss, Algorithm 3 trust region and geodesic Zeno term.
- Fix guidance:
  1. Replace `dist.rsample()` by `dist.sample()` (or `dist.rsample().detach()`) before evaluating `log_prob`.
  2. Average over several samples per state for a usable estimate.
- Required new assumptions/permits: none.
- Validation plan: Unit test on a Gaussian policy with state-dependent mean: the diagonal must be strictly positive and vary with $z$.

### [E-008] Metric computed on observations / latent states is applied to tensors in other spaces (was F-007)
- Location: `RiemannianFragileAgent.train_step` and `_geodesic_dist`, lines 586-591, 630-643, 680-688
- Severity: Moderate
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (lines 590-591): "fisher_diag = compute_state_space_fisher(self, batch.obs); G_inv = 1.0/(fisher_diag + 1e-8)"; (line 635): "L_nat = -torch.mean((grad_V * velocity) * G_inv)" with `grad_V`, `velocity` in `z_macro` space; (line 688): "diff = policy_new - policy_old; return (diff * diff * G_inv).sum(dim=-1).mean()".
- Upstream anchor: docs/source/1_agent/01_foundations/02_control_loop.md:921 "$G_{zz}$ ... Lives On: Latent states"; 01_definitions.md defines $Z_t=(K_t,z_n,z_{\text{tex}})$ as the latent, distinct from the observation $x$.
- Why this is an error: `G_inv` has the dimension of `batch.obs` (observation space), yet at line 635 it multiplies `grad_V * velocity`, which live in the macro-latent space (`z_macro` $=e_{K_t}$, line 596), and at line 688 it multiplies `policy_new - policy_old`, which lives in policy-output space. Elementwise products fail unless $\dim x=\dim z=\dim a$, and even then unrelated coordinates are paired. The docstring's own definition ($\partial\log\pi/\partial z$) requires $G$ on $z$; a policy-space trust region needs a metric on the policy output (e.g. Fisher-Rao in action space), not $G_{zz}$.
- Impact on downstream results: Algorithm 3 (the "complete specification" of Tier 4).
- Fix guidance:
  1. Compute `fisher_diag` from `z_macro` (line 596) rather than `batch.obs`.
  2. For `_geodesic_dist`, use a KL/Fisher-Rao divergence between `policy_new` and `policy_old` (as in the Tier-1 Zeno term).
- Required new assumptions/permits: none.
- Validation plan: Run `train_step` with $\dim x\ne\dim z\ne\dim a$ and confirm no shape errors; check the trust-region term equals the KL between successive policies on a test batch.

### [E-009] Reference to a non-existent document `metalearning.md` (was F-008)
- Location: Defect Functional Costs, line 714
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter (the same phantom file is cited at docs/source/1_agent/07_cognition/02_governor.md:613, :638)
- Claim (line 714): "## Defect Functional Costs (from metalearning.md)".
- Upstream anchor: `find docs -name "metalearning*"` returns nothing; 02_governor.md:638 itself calls it an "unpublished document".
- Why this is an error: The defect functionals $K_C,K_D,K_{SC},K_{Cap},K_{LS},K_{TB}$ and the expected-defect recommendation $\mathcal R_A(\theta)$ (lines 718-727) are imported from a source that does not exist in the book, so none of the formulas can be checked.
- Impact on downstream results: Local.
- Fix guidance:
  1. Point to the actual location of the defect functionals (e.g. {ref}`sec-defect-functionals-implementing-regulation`), or
  2. delete the attribution and state the definitions in place.
- Required new assumptions/permits: none.
- Validation plan: Build the docs and confirm the reference resolves.

### [E-010] Minimum-chart table is wrong for the torus and the Klein bottle; wrong reason for the sphere (was F-009)
- Location: Manifold Atlas Theory: Why Single Charts Fail, lines 758-768
- Severity: Moderate
- Type: Computational error (secondary: Conceptual)
- Criterion: External
- Origin: this chapter
- Claim (lines 758-768): "| **Torus $T^2$** | 4 | Non-trivial first homology |", "| **Klein Bottle** | ∞ | Non-orientable |", "Sphere $S^2$ | 2 | No global flat coordinates (Hairy Ball Theorem)"; prose (line 766): "A torus needs 4."
- Upstream anchor: {cite}`lee2012smooth` (a chart is a homeomorphism onto an open subset of $\mathbb R^n$).
- Why this is an error: (i) $T^2=S^1\times S^1$: cover the first circle by two open arcs $I_1,I_2$; $I_k\times S^1$ is an open annulus, homeomorphic to $\{1<|x|<2\}\subset\mathbb R^2$; two charts suffice, and compactness plus invariance of domain gives the lower bound 2. First homology is irrelevant. (ii) The Klein bottle is compact, hence admits a finite atlas; as an $S^1$-bundle over $S^1$, trivial over each of two arcs, it is covered by two annular charts. "∞" is false and non-orientability has no bearing on chart count. (iii) $S^2$ does need 2 charts, but because no compact space is homeomorphic to an open subset of $\mathbb R^2$; the Hairy Ball Theorem concerns tangent vector fields and does not imply the chart bound. The table is wrong under its own cited reference.
- Impact on downstream results: The table motivates the whole Tier-5/6 atlas design; the qualitative conclusion (many manifolds need more than one chart) survives, but the numbers and reasons are wrong and repeated in prose.
- Fix guidance:
  1. Torus: 2 (compactness); Klein bottle: 2 (compactness); sphere: 2 (compactness / invariance of domain).
  2. Delete "Hairy Ball", "non-trivial first homology", "non-orientable" as reasons; fix line 766.
- Required new assumptions/permits: none.
- Validation plan: Cross-check the corrected entries against Lee, Introduction to Smooth Manifolds, Ch. 1.

### [E-011] Norm-preservation claims fail for rectangular semi-orthogonal $W$; the step "$\Vert W\Vert_2=1\Rightarrow$ gradient norms preserved" is invalid (was F-010)
- Location: Why Orthogonality? table, lines 839-845; Proposition "Gradient Preservation via Orthogonality", lines 2071-2083
- Severity: Moderate
- Type: Invalid inference (secondary: Scope restriction)
- Criterion: External
- Origin: this chapter
- Claim (lines 2071-2083): "Let $W$ be a weight matrix satisfying $W^TW=I$ (semi-orthogonality). Then: 1. All singular values of $W$ equal 1. 2. ... $\Vert\nabla_x\mathcal L\Vert=\Vert\nabla_y\mathcal L\Vert$ ... *Proof.* ... The Jacobian $\partial y/\partial x=W$ has $\Vert W\Vert_2=1$. By the chain rule, gradient norms are preserved."; table (lines 843-845): "Distance preservation $\Vert Wx\Vert=\Vert x\Vert$ ... Inverse stability $W^{-1}=W^T$ ... Information loss: None".
- Upstream anchor: n/a (defined in this chapter); the layers it is applied to are rectangular: `OrthogonalLinear(input_dim, 128)`, `OrthogonalLinear(128, latent_dim)` (lines 1117-1121).
- Why this is an error: For tall $W\in\mathbb R^{m\times n}$, $m>n$, with $W^TW=I_n$, forward norms are preserved but $W^T$ is an orthogonal projection, so $\Vert W^Tg\Vert\le\Vert g\Vert$ with equality only for $g\in\operatorname{range}W$ (verifier's recomputation with a $6\times3$ Q-factor: $\Vert Wx\Vert/\Vert x\Vert=1.000$, $\Vert W^Tg\Vert/\Vert g\Vert=0.754$). For wide $W$, $W^TW=I_n$ is impossible; the code's regulariser (lines 827-832) enforces $WW^T=I_m$, under which the backward norm is preserved but $\Vert Wx\Vert<\Vert x\Vert$ generically ($0.667$ in the $3\times6$ check) and information is lost (rank $m<n$). Statements 2-3 and the three table rows hold only for square $W$; the proof infers equality from an upper bound.
- Impact on downstream results: Mechanism 1 of the Dynamical Isometry argument (lines 2062-2083) and the "Isometry Triangle" row (line 2133); see E-019.
- Fix guidance:
  1. Restrict the proposition to square $W$, or
  2. state the one-sided results: tall semi-orthogonal $W$ preserves forward norms and is backward non-expansive; wide semi-orthogonal $W$ preserves backward norms and is forward non-expansive.
  3. Amend the table rows "Distance preservation", "$W^{-1}=W^T$", "Information loss: None" accordingly.
- Required new assumptions/permits: none.
- Validation plan: Numerical check on random Q-factors of both shapes.

### [E-012] Attentive routing is called "permutation invariant" but the stated property is equivariance (was F-011)
- Location: Definition "Attentive Routing Law", lines 1294-1302 (also prose 1275-1277, 1284)
- Severity: Minor
- Type: Miswording (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (line 1301): "This mechanism is **permutation invariant**: shuffling the memory order of the chart tokens merely shuffles the output indices without changing the underlying topology or geometry."
- Upstream anchor: docs/source/1_agent/01_foundations/01_definitions.md:313 "$S_{|\mathcal{K}|}$ be the **symbol-permutation symmetry** of the discrete macro register: relabeling code indices is unobservable if downstream components depend only on embeddings $\{e_k\}$"; this chapter's own table at line 1397: "Permutation-equivariant chart tokens".
- Why this is an error: Output indices permuting with the input permutation is equivariance, $w_{\sigma(i)}(x;\sigma C)=w_i(x;C)$, not invariance. The definition, the prose and the table disagree, and the definition asserts the stronger property, which does not hold for $w(x)\in\Delta^{N_c-1}$.
- Impact on downstream results: Local (the property needed downstream, per definitions.md:313, is equivariance).
- Fix guidance:
  1. Replace "permutation invariant" by "permutation equivariant (invariant up to relabelling)" at lines 1275-1277, 1284, 1301.
- Required new assumptions/permits: none.
- Validation plan: Text check for consistency with line 1397.

### [E-013] "Current implementation" description differs from `src/fragile/core/layers/atlas.py` (was F-012)
- Location: Architecture Specification and Topological Decoder Module, lines 1331-1371, 1362-1370, 1687-1777
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (lines 1365-1370): "Core steps (mirroring `PrimitiveAttentiveAtlasEncoder.forward`): ... 2. `v = val_proj(features)` ... 3. `c_bar = router_weights @ chart_centers`, `v_local = v - c_bar`"; decoder (lines 1746-1753): "z_geo = torch.tanh(z_geo)" with `forward(self, z_geo, z_tex=None, chart_index=None)`.
- Upstream anchor: src/fragile/core/layers/atlas.py:1179-1275 `PrimitiveAttentiveAtlasEncoder.forward(self, x, hard_routing=False, hard_routing_tau=1.0)`: `v = _project_to_ball(self.val_proj(features))`, `c_bar = _poincare_weighted_mean(chart_centers, router_weights)`, `v_local = _project_to_ball(mobius_add(-c_bar, v))`, `z_geo = _project_to_ball(mobius_add(c_bar, z_local))`; atlas.py:1397-1408 `PrimitiveTopologicalDecoder.forward(self, z_geo, z_tex=None, chart_index=None, router_weights=None, hard_routing=False, hard_routing_tau=1.0)` starting with `z_geo = _project_to_ball(z_geo)`; atlas.py:420-441 `TopologicalDecoder.forward` with `z_geo = torch.tanh(z_geo)`.
- Why this is an error: The production code works in the Poincare ball (values, centres and decoder input projected to the ball; Mobius operations; Poincare helpers at atlas.py:19-93), so the Euclidean `v_local = v - c_bar` and `tanh(z_geo)` are not what the code does, and the decoder signature differs. The verifier found that the chapter's decoder snippet matches the older `TopologicalDecoder.forward`, not the `PrimitiveTopologicalDecoder` the header names. The ten-item forward-return tuple does match atlas.py:99-110. The section header claims "(current implementation)".
- Impact on downstream results: The Encoder/TopoEncoder mermaid diagrams (lines 1467-1626, nodes `c_bar = sum(w_enc * c_k)`, `v_local = v - c_bar`, `tanh(z_geo)`) and the Stacked TopoEncoder module (E-021, E-022) inherit the stale description.
- Fix guidance:
  1. Label these blocks as a simplified Euclidean reference model, or
  2. update them to the ball-projected pipeline and current signatures, and name the decoder class actually shown.
- Required new assumptions/permits: none.
- Validation plan: Diff the chapter's step list against `PrimitiveAttentiveAtlasEncoder.forward` and `PrimitiveTopologicalDecoder.forward` after the edit.

### [E-014] Poincare-ball depth formula is off by a factor of 2 (was F-013)
- Location: Corollary "The Hyperbolic Embedding", line 1852
- Severity: Minor
- Type: Computational error
- Criterion: External
- Origin: this chapter
- Claim (line 1852): "tree depth $\ell$ maps to $\log(1/y)$; equivalently, in the Poincare ball model, depth maps to $\tanh^{-1}(r)$ where $r\in[0,1)$ is the radial coordinate."
- Upstream anchor: n/a (metric $ds^2=(dx^2+dy^2)/y^2$, curvature $-1$, fixed at the same line).
- Why this is an error: With curvature $-1$, the upper-half-space distance from $(x_0,1)$ to $(x_0,y)$ is $|\log y|$ (correct), but the ball distance from the origin to radius $r$ is $2\tanh^{-1}(r)$ (at $r=0.5$: $1.0986$ vs $\tanh^{-1}(0.5)=0.5493$). "Equivalently" fails by a factor 2; the models are isometric so the same depth must map to $2\tanh^{-1}(r)$.
- Impact on downstream results: Definition "The Latent Metric Tensor" uses $\rho$ = depth; an implementation using Poincare embeddings would place levels at half the intended radius.
- Fix guidance:
  1. Replace $\tanh^{-1}(r)$ by $2\tanh^{-1}(r)$.
- Required new assumptions/permits: none.
- Validation plan: Check against the standard ball distance formula $d(0,r)=2\tanh^{-1}(r)$.

### [E-015] "Texture as the Ideal Boundary" is not a proposition about the defined structure: it needs an infinite-depth tree the definitions do not provide (was F-014)
- Location: Proposition "Texture as the Ideal Boundary", lines 1886-1903
- Severity: Moderate
- Type: Proof gap / omission (secondary: Scope restriction)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 1889-1903): "Let $M$ be the Riemannian manifold constructed above ... 1. Consider a sequence of refining codes $(K^{(n)}_{\text{chart}},K^{(n)}_{\text{code}})$ representing a path $\gamma$ in the tree $\mathcal T$ extending to infinite depth. ... 6. Taking the limit $\epsilon\to0$, $z_{\text{tex}}$ maps to the **limit set** $\Lambda\subset\partial_\infty\mathbb H^n$. ... $\square$"; "**Operational Implication:** This formalizes why $z_{\text{tex}}$ must be excluded from dynamics ($S_t$) and control ($\pi_\theta$)."
- Upstream anchor: Definition "The Macro-State Tree" (this chapter, lines 1829-1841): levels 0, 1 (charts), 2 (codes) only. The exclusion of texture from closure is a definition upstream (docs/source/1_agent/01_foundations/01_definitions.md, texture reconstruction-only), restated at line 557 of this chapter.
- Why this is an error: The tree $\mathcal T$ has depth 2, so no path "extends to infinite depth"; the refinement assumed in step 1 is never constructed (the stacked TopoEncoder of lines 1961-2350 gives at most depth $L$ and is introduced later). "The Riemannian manifold constructed above" refers to nothing constructed. The cutoff surface $\Sigma_\epsilon$, the limit set $\Lambda$ and the map $z_{\text{tex}}\mapsto\Lambda$ are never defined; $z_{\text{tex}}$ is by step 4 (line 1896) a finite vector $\Delta_{\text{total}}-z_n\in\mathbb R^D$, not a point of $\partial_\infty\mathbb H^n$. The $\square$ closes a heuristic. The "operational implication" restates an upstream modelling choice rather than deriving it.
- Impact on downstream results: Summary "The Manifold Construction" item 4 and "This geometric picture justifies the Sieve architecture" (lines 1951-1957) cite this as established.
- Fix guidance:
  1. Downgrade to a remark/interpretation, or
  2. make it a genuine statement about the stacked encoder ($L$ levels) with an explicit limit object and hypotheses, and define $\Sigma_\epsilon$, $\Lambda$ and the map.
  3. Rephrase the operational implication as consistency with the upstream definition, not a consequence.
- Required new assumptions/permits: if kept as a proposition, an explicit infinite-refinement hypothesis.
- Validation plan: Confirm every object in the statement is defined earlier in the chapter or upstream.

### [E-016] Sign of the exponent in the latent metric tensor contradicts the stated upper-half-space model (was F-015)
- Location: Definition "The Latent Metric Tensor", lines 1909-1928
- Severity: Moderate
- Type: Computational error (secondary: Notation conflict)
- Criterion: External
- Origin: this chapter
- Claim (line 1915): "Working in the upper half-space model where depth $\rho\in[0,\infty)$ corresponds to $y=e^{-\rho}$, the metric ... takes the form $ds^2=d\rho^2+d\sigma_{\mathcal K}^2+e^{-2\rho}\Vert dz_n\Vert^2$ ... The factor $e^{-2\rho}$ indicates that as resolution increases (deeper in the tree), the effective magnitude of nuisance variations shrinks exponentially".
- Upstream anchor: Corollary at line 1852 (this chapter): "$\mathbb H^n=\{(x,y):y>0\}$ with metric $ds^2=(dx^2+dy^2)/y^2$". The $G$ invoked at line 1907 is the sensitivity metric of docs/source/1_agent/01_foundations/02_control_loop.md:684-690.
- Why this is an error: Substituting $y=e^{-\rho}$ gives $dy=-e^{-\rho}d\rho$, hence $(dx^2+dy^2)/y^2=d\rho^2+e^{+2\rho}\Vert dx\Vert^2$ (verified symbolically). The displayed $e^{-2\rho}$ corresponds to moving toward $y\to\infty$, the opposite ideal point from the boundary $\rho\to\infty$ stated in the same definition. The verbal claim (a unit metric displacement is an $e^{-\rho}$ coordinate displacement) is consistent with $e^{+2\rho}$, not $e^{-2\rho}$. Secondarily, this hyperbolic $ds^2$ is a different object from the sensitivity metric $G$ of control_loop.md:684 that the paragraph says it "implies a specific structure for".
- Impact on downstream results: "Rigorous Interpretation of $z_n$" (horosphere coordinates) and the Robot example; no downstream formula consumes the exponent numerically.
- Fix guidance:
  1. Write $ds^2=d\rho^2+d\sigma_{\mathcal K}^2+e^{2\rho}\Vert dz_n\Vert^2$ with $z_n$ the horospherical coordinate.
  2. Rephrase bullet 4 as "a unit metric displacement in $z_n$ is an $e^{-\rho}$ coordinate displacement".
  3. State explicitly that this is a geometric model distinct from the sensitivity metric $G$.
- Required new assumptions/permits: none.
- Validation plan: Re-derive the warped-product form from the half-space metric.

### [E-017] Forward-activation proposition: unit variance at $\ell=0$ is not "by construction", $\sigma<1$ is assumed, and the Jacobian ignores $\partial\sigma/\partial z_{\text{tex}}$ (was F-016)
- Location: Proposition "Forward Activation Stability", lines 2089-2110; forward code, lines 2270-2271
- Severity: Minor
- Type: Proof gap / omission
- Criterion: Framework
- Origin: this chapter
- Claim (lines 2093-2103): "1. $\mathrm{Var}(x^{(\ell)})=1$ for all $\ell$ (by construction). ... $\partial x^{(\ell)}/\partial z_{\text{tex}}^{(\ell-1)}=1/\sigma^{(\ell-1)}$ ... Since each block successfully explains part of the signal, the residual standard deviation $\sigma^{(\ell)}<1$".
- Upstream anchor: Definition "The Rescaling Operator / Renormalization" (this chapter, lines 2017-2028): "$x^{(\ell+1)}=z_{\text{tex}}^{(\ell)}/(\sigma^{(\ell)}+\epsilon)$, $\sigma^{(\ell)}=\sqrt{\mathrm{Var}(z_{\text{tex}}^{(\ell)})+\epsilon}$".
- Why this is an error: (i) $x^{(0)}:=x$ (line 1991) is the raw observation and is never rescaled, so item 1 holds only for $\ell\ge1$ or under an unstated input-standardisation hypothesis. (ii) `sigma = torch.sqrt(residual.var() + eps)` (line 2270) is not detached, so the exact Jacobian is $\sigma^{-1}(I-\text{rank-one correction})$, not the scalar $1/\sigma$. (iii) "$\sigma^{(\ell)}<1$" is a hypothesis about training success presented as a consequence.
- Impact on downstream results: Feeds the Isometry Triangle table and Theorem (E-019).
- Fix guidance:
  1. Add hypotheses "inputs standardised" and "each block reduces variance ($\sigma^{(\ell)}<1$)".
  2. Either detach $\sigma$ in the forward pass or state the exact Jacobian.
- Required new assumptions/permits: input standardisation; variance-reduction hypothesis.
- Validation plan: Compare autograd Jacobian of the rescaling step with $1/\sigma$ on a small batch.

### [E-018] GELU is not 1-Lipschitz (was F-017)
- Location: Mechanism 3: Spectral Normalization, line 2120
- Severity: Minor
- Type: Computational error
- Criterion: External
- Origin: this chapter
- Claim (line 2120): "Combined with 1-Lipschitz activations (e.g., GELU), this bounds the network Lipschitz constant by the product of per-layer spectral norms."
- Upstream anchor: n/a (standard fact); the network uses `nn.GELU()` (lines 1105-1120) and `NormGatedGELU` in the primitives.
- Why this is an error: $\mathrm{GELU}'(x)=\Phi(x)+x\varphi(x)$ attains $\sup_x\mathrm{GELU}'(x)\approx1.1289$ at $x\approx1.414$ (recomputed by the verifier), so GELU is $1.129$-Lipschitz. With $L$ GELU layers the bound must carry a factor $1.129^L$; "product of per-layer spectral norms" alone is not a valid Lipschitz bound. (ReLU, tanh and sigmoid/2 are 1-Lipschitz.)
- Impact on downstream results: The "$\Vert W_{SN}\Vert_2=1\Rightarrow$ 1-Lipschitz layer" claim and the spectral constant of Theorem E-019.
- Fix guidance:
  1. Replace "1-Lipschitz activations (e.g., GELU)" by "$L_\phi$-Lipschitz activations ($L_\phi\approx1.13$ for GELU, $1$ for ReLU)".
  2. Include $L_\phi^L$ in the bound.
- Required new assumptions/permits: none.
- Validation plan: Numerically maximise $\mathrm{GELU}'$.

### [E-019] Theorem "Dynamical Isometry without Skip Connections" is vacuous as stated and its proof sketch cannot give a lower singular-value bound (was F-018)
- Location: Combined Effect / Theorem, lines 2137-2149; Definition "Total Reconstruction", lines 2029-2038
- Severity: Major
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 2137-2149): "achieves approximate dynamical isometry: the singular values of the input-output Jacobian $J=\partial\hat x/\partial x$ satisfy $\sigma_i(J)\in[1/\kappa,\kappa]$ for a condition number $\kappa=O(K^L\cdot\prod_\ell(1+\epsilon_{\text{orth}}))$. *Proof sketch.* Each layer contributes a factor with singular values in $[1-\epsilon,1+\epsilon]$ (orthogonality) or $[0,K]$ (spectral norm). ... The product of $L$ such factors yields the stated bound."
- Upstream anchor: Definitions in this chapter: "The Peeling Step" (line 2013) $z^{(\ell)}_{\text{tex}}=x^{(\ell)}-\hat x^{(\ell)}$; "Rescaling" (line 2023) $x^{(\ell+1)}=z^{(\ell)}_{\text{tex}}/(\sigma^{(\ell)}+\epsilon)$; "Total Reconstruction" (line 2035) $\hat x=\sum_{\ell=0}^{L-1}\Pi^{(\ell)}\hat x^{(\ell)}+\Pi^{(L)}x^{(L)}$.
- Why this is an error: (i) Vacuity: from the two definitions, $x^{(\ell)}=\hat x^{(\ell)}+(\sigma^{(\ell)}+\epsilon)\,x^{(\ell+1)}$ identically, so unrolling gives $x=\sum_\ell\Pi'^{(\ell)}\hat x^{(\ell)}+\Pi'^{(L)}x^{(L)}$ with $\Pi'$ built from $\sigma+\epsilon$, which is the Total Reconstruction formula up to the $\epsilon$ bookkeeping. Hence $\hat x\equiv x$ for any weights and $J=\partial\hat x/\partial x=I$ exactly; the conclusion holds with $\kappa=1$ independently of hypotheses 1-3 and says nothing about gradient flow through the encoders. The meaningful object is the Jacobian of the encoding path ($\partial x^{(L)}/\partial x$ or $\partial\hat x^{(\ell)}/\partial x$). (ii) Even for that object the sketch is invalid: factors "in $[0,K]$" give no lower bound, and the pipeline contains piecewise-constant maps (VQ nearest-code lookup), contractive maps (softmax router, `tanh`, GELU for $x\ll0$) and a rank-deficient wide layer `OrthogonalLinear(128, latent_dim)` (E-011); $\sigma_i\ge1/\kappa$ cannot follow. (iii) With $\Vert W^TW-I\Vert_F<\epsilon$ the singular values lie in $[\sqrt{1-\epsilon},\sqrt{1+\epsilon}]$ ($0.949$, $1.049$ at $\epsilon=0.1$), not $[1-\epsilon,1+\epsilon]$.
- Impact on downstream results: This theorem is the section's justification ("Dynamical Isometry: Why Gradients Do Not Vanish", and the Researcher Bridge on RG vs ResNets) for removing skip connections.
- Fix guidance:
  1. Restate the theorem for the encoder-path Jacobian restricted to the continuous pathway (treat VQ via its straight-through surrogate).
  2. Assume square/isometric layers or state one-sided bounds (E-011); include $L_\phi$ (E-018).
  3. Derive an explicit upper bound $\sigma_{\max}\le(KL_\phi)^L\prod(1+\epsilon)^{1/2}$ and drop, or separately justify, the lower bound.
- Required new assumptions/permits: square or isometric layers; a stated surrogate for the VQ step.
- Validation plan: Compute $\partial\hat x/\partial x$ by autograd on the stacked module and confirm it equals $I$; then compute the encoder-path Jacobian's singular values on random inputs and compare to the restated bound.

### [E-020] The Wilsonian-RG identification is reversed: the flow acts on the discarded fluctuations and the "fixed point" is placed in the UV (was F-019)
- Location: Rigorous Interpretation: Renormalization Group (RG) Flow, lines 2150-2176; also rb-renormalization-resnets, lines 1983-1986
- Severity: Moderate
- Type: Conceptual
- Criterion: External
- Origin: this chapter
- Claim (lines 2155-2170): "This architecture is a direct algorithmic implementation of Kadanoff's block-spin transformation or Wilsonian RG flow"; table rows "**Integrating Out** | Subtracting the mean field: $z^{(\ell)}_{\text{tex}}=x^{(\ell)}-\hat x^{(\ell)}$", "**Fixed Point** | The texture distribution $p(x^{(L)})$ at the deepest layer"; "Block 0 (IR / Infrared) ... Block $L-1$ (UV / Ultraviolet): Captures the finest irreducible noise."
- Upstream anchor: external physics, {cite}`mehta2014exact`.
- Why this is an error: In Kadanoff/Wilson RG the short-wavelength (UV) fluctuations are integrated out and the flow acts on effective Hamiltonians for the retained coarse (IR) variables, with its fixed point in the IR. Here the retained coarse variables $(K^{(\ell)},z^{(\ell)}_n)$ are frozen as outputs and the propagated object, whose distribution is called the fixed point, is the residual $z_{\text{tex}}$, exactly what RG discards; the deepest block is labelled UV. Subtracting the mean field and passing the fluctuation is the complement of "integrating out". The construction is a coarse-to-fine multiresolution / matching-pursuit decomposition. As a metaphor this is harmless; as a "Rigorous Interpretation" and "direct algorithmic implementation" it is inverted.
- Impact on downstream results: The interpretation of $\mathcal L_{\text{scale-decay}}$ ("the RG flow moves toward a fixed point", E-023) and the RG-vs-ResNet bridge; no computation depends on it.
- Fix guidance:
  1. Relabel as a coarse-to-fine multiresolution (analysis-synthesis) decomposition, or
  2. map the flow correctly (effective theory for $(K^{(\ell)},z^{(\ell)}_n)$; fixed point = distribution of the coarsest retained variables) and drop "direct implementation".
- Required new assumptions/permits: none.
- Validation plan: Check the table against the standard RG account in the cited reference.

### [E-021] StackedTopoEncoder: blocks $\ell\ge1$ have the wrong input/output dimension (was F-020)
- Location: Implementation: Stacked TopoEncoder Module, lines 2210-2230, 2248-2288
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 2212-2213, 2226-2227): "PrimitiveAttentiveAtlasEncoder(input_dim=input_dim if i == 0 else latent_dim, ...)" and "PrimitiveTopologicalDecoder(..., output_dim=input_dim if i == 0 else latent_dim)"; loop (lines 2264-2286): "x_hat, _ = self.decoders[ell](z_geo, z_tex, K_chart); residual = x_current - x_hat; ... x_current = x_next".
- Upstream anchor: Definition "The Peeling Step" (line 2013) and "Rescaling" (line 2023): $x^{(\ell+1)}$ lives in the same space as $x^{(\ell)}$ and $\hat x^{(\ell)}$ for every $\ell$.
- Why this is an error: Block 0 decodes to `output_dim=input_dim`, so `residual` (hence `x_current` for block 1) has dimension `input_dim`, but block 1's encoder is built with `input_dim=latent_dim`; the call fails unless `input_dim == latent_dim`. Symmetrically, decoders $\ell\ge1$ output `latent_dim`-vectors that are subtracted from `input_dim`-vectors, and `reconstruct` (lines 2290-2316) sums tensors of different sizes. The code contradicts the definitions it claims to implement.
- Impact on downstream results: The whole stacked architecture and its `reconstruct` method; the setting of Theorem E-019.
- Fix guidance:
  1. Use `input_dim=input_dim` and `output_dim=input_dim` for all blocks (all residuals live in observation space), or
  2. insert an explicit projection and update the definitions accordingly.
- Required new assumptions/permits: none.
- Validation plan: Instantiate with `input_dim != latent_dim` and run `forward` and `reconstruct`.

### [E-022] `orthogonality_loss()` does not exist on the primitives; the primitives use `SpectralLinear`, not `OrthogonalLinear` (was F-021)
- Location: `StackedTopoEncoder.orthogonality_loss`, lines 2318-2328, vs Mechanism 1, lines 2062-2070
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 2324): "for encoder in self.encoders: total = total + encoder.orthogonality_loss()"; (lines 2062-2070): "The **OrthogonalLinear** layers enforce approximate isometry via the loss $\mathcal L_{\text{orth}}=\sum_\ell\Vert W_\ell^TW_\ell-I\Vert_F^2$".
- Upstream anchor: `grep -rn "orthogonality_loss\|class OrthogonalLinear\|def orth" src/fragile/core/layers/` finds only the free function `compute_orthogonality_loss` in topology.py:443; the primitives are built from `SpectralLinear` and `NormGatedGELU` (atlas.py:984-1099, 1333-1395; this chapter lines 1364, 1683-1684, 1712-1744).
- Why this is an error: The method call raises `AttributeError`, and Mechanism 1 (orthogonality regularisation) is absent from the `Primitive*` implementation the stacked module instantiates; only Mechanism 3 (spectral normalisation) is present. The "Isometry Triangle" describes a combination not realised by the module shown.
- Impact on downstream results: Hypothesis 1 of Theorem E-019 is not satisfied by the implementation offered as its realisation.
- Fix guidance:
  1. Add an orthogonality defect over `SpectralLinear` weights to the primitives (e.g. via `compute_orthogonality_loss`), or
  2. drop Mechanism 1 from the stacked module's description.
- Required new assumptions/permits: none.
- Validation plan: Call `orthogonality_loss()` on an instantiated module.

### [E-023] Scale-decay loss pushes in the opposite direction of its stated purpose (was F-022)
- Location: Training Losses for Scale Separation, lines 2344-2350
- Severity: Moderate
- Type: Conceptual (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (lines 2344-2350): "$\mathcal L_{\text{scale-decay}}=\sum_{\ell=0}^{L-2}\max(0,\sigma^{(\ell+1)}-\sigma^{(\ell)})^2$ This ensures that deeper blocks explain progressively less variance—the RG flow moves toward a fixed point."
- Upstream anchor: Definition "Rescaling" (line 2023): $\sigma^{(\ell)}$ is the standard deviation of block $\ell$'s residual, whose input $x^{(\ell)}$ has unit variance for $\ell\ge1$ (Proposition, line 2093).
- Why this is an error: For $\ell\ge1$ every block receives a unit-variance input, so $\sigma^{(\ell)2}$ is the fraction of its input variance left unexplained; block $\ell$ explains $1-\sigma^{(\ell)2}$ of its normalised input. Forcing $\sigma$ non-increasing with depth makes deeper blocks explain a larger fraction than shallower ones, the opposite of "progressively less". The verifier strengthened the argument: the absolute variance explained by block $\ell$ is about $\Pi^{(\ell)2}(1-\sigma^{(\ell)2})$, and the ratio between consecutive blocks is $\sigma^{(\ell)2}(1-\sigma^{(\ell+1)2})/(1-\sigma^{(\ell)2})$. Under the chapter's constraint this can exceed 1 ($\sigma$: $0.9\to0.1$ gives $4.22$), so the loss does not even secure absolute decay; under the reversed constraint ($\sigma$ non-decreasing) the ratio is bounded by $\sigma^{(\ell)2}<1$, which does. At the stated fixed point (irreducible noise) $\sigma\to1$, which the loss penalises.
- Impact on downstream results: Training recipe for the stacked encoder; interpretation in E-020.
- Fix guidance:
  1. Penalise $\max(0,\sigma^{(\ell)}-\sigma^{(\ell+1)})^2$ (monotone increase toward the noise floor) and change the text to "deeper blocks explain progressively less of their input", or
  2. drop the term and rely on $\Pi^{(\ell)}$.
  3. If kept, state the ratio bound $\Pi^{(\ell+1)2}(1-\sigma^{(\ell+1)2})/[\Pi^{(\ell)2}(1-\sigma^{(\ell)2})]\le\sigma^{(\ell)2}$ as the guarantee.
- Required new assumptions/permits: none.
- Validation plan: Numerically evaluate the explained-variance ratio for monotone $\sigma$ sequences of both orientations.

### [E-024] Prose claims factorisation "automatically ensures consistency"; the formal text says it does not (was F-023)
- Location: Factorized Jump Operators, feynman-prose line 2368 vs Overlap Consistency Loss line 2478
- Severity: Minor
- Type: Miswording
- Criterion: Framework
- Origin: this chapter
- Claim (line 2368): "This reduces the parameter count from $O(K^2)$ to $O(K)$, and it automatically ensures consistency because everything goes through the same intermediate representation." vs (line 2478): "The factorized parameterization does not automatically enforce transitivity. We add a **cycle consistency loss**".
- Upstream anchor: Definition "Factorized Jump Operator" (line 2446): $L_{i\to j}(z)=A_j(B_iz+c_i)+d_j$.
- Why this is an error: $L_{j\to k}\circ L_{i\to j}(z)=A_k(B_jA_j(B_iz+c_i)+B_jd_j+c_j)+d_k$ equals $L_{i\to k}(z)$ only if $B_jA_j=I_r$ and $B_jd_j+c_j=0$, which the parameterisation does not impose. The prose asserts the opposite of the formal section.
- Impact on downstream results: Reader confusion only; the formal treatment is correct.
- Fix guidance:
  1. Replace "automatically ensures consistency" by "makes consistency cheap to enforce (a single shared space; see the overlap consistency loss)".
- Required new assumptions/permits: none.
- Validation plan: Text check.

### [E-025] Cost table: "Forward (all pairs)" for the factorised operator is $O(K^2rd_n)$, not $O(Krd_n)$ (was F-024)
- Location: Computational Cost Analysis, lines 2521-2531
- Severity: Minor
- Type: Computational error
- Criterion: External
- Origin: this chapter
- Claim (line 2525): "| **Forward (all pairs)** | $O(K^2 d_n^2)$ | $O(K r d_n)$ | Batch lift + project |".
- Upstream anchor: n/a; lines 2529-2531 describe "all transitions from chart $i$": "Lift: $h=B_iz+c_i$ ... Project to all targets: $\{A_jh+d_j\}_{j=1}^K$ — shape $[B,K,d_n]$".
- Why this is an error: All $K$ lifts cost $Krd_n$ and all $K^2$ ordered projections cost $K^2rd_n$; total $O(K^2rd_n)$ (a factor $d_n/r$ saving over naive, not a factor $K$). The $O(Krd_n)$ figure is the cost of all targets from a single source, which is what the "Batch Efficiency" paragraph describes. The chapter's parameter arithmetic was recomputed and is correct ($64\times(2\cdot8\cdot16+8+16)=17{,}920$; naive $64\times63\times256=1{,}032{,}192$; ratio $57.6$).
- Impact on downstream results: Local.
- Fix guidance:
  1. Change the row to $O(K^2rd_n)$, or rename it "all targets from one source".
- Required new assumptions/permits: none.
- Validation plan: Count FLOPs for a batched implementation.

### [E-026] Integration snippet calls the constructor with a keyword it does not accept and stores a tuple as a scalar loss (was F-025)
- Location: Integration with PrimitiveAttentiveAtlasEncoder, lines 2747-2787
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (line 2755): "self.jump_op = FactorizedJumpOperator(num_charts=self.encoder.num_charts, latent_dim=self.encoder.latent_dim, global_rank=global_rank)"; (line 2786): "jump_loss = compute_jump_consistency_loss(z_n_all_charts, router_weights, self.jump_op) ... 'jump_consistency': jump_loss".
- Upstream anchor: this chapter: `FactorizedJumpOperator.__init__(self, num_charts, nuisance_dim, global_rank=None)` (lines 2551-2556); `compute_jump_consistency_loss(...) -> Tuple[torch.Tensor, dict]` returning `loss, info` (lines 2649-2655, 2738).
- Why this is an error: `latent_dim` is not a constructor parameter (`TypeError`); the consistency function returns `(loss, info)`, so the dict entry is a tuple, not a tensor that can be weighted by $\lambda_{\text{jump}}$ per the training schedule table.
- Impact on downstream results: Local.
- Fix guidance:
  1. `FactorizedJumpOperator(num_charts=..., nuisance_dim=self.encoder.latent_dim, global_rank=...)`.
  2. `jump_loss, jump_info = compute_jump_consistency_loss(...)`.
- Required new assumptions/permits: none.
- Validation plan: Execute the snippet against the chapter's own class definitions.

## Scope restrictions and clarifications
- Findings E-005, E-006 and E-007 are stated under the cost convention the Tier-4 routines themselves adopt (`cost = -r`, $\dot V\le-\alpha V$). If the authors intend $V$ as a reward value, the fixes reverse: keep the actor sign and flip the critic's Lyapunov constraint.
- E-001: the `∇_A` symbol in the stiffness formula is inherited from 02_sieve/01_diagnostics.md:100, whose own table (row 138) places stiffness in state space. Resolving the notation is an upstream task; only the autograd/graph defect is charged to this chapter.
- E-013 is a labelling issue: the Euclidean pipeline described is coherent in itself but is not the "(current implementation)" in atlas.py, which works in the Poincare ball. The chapter's ten-item return tuple does match the code.
- E-010, E-011, E-014, E-016, E-018, E-020 and E-025 are checked against standard external mathematics and physics (differential topology, linear algebra, hyperbolic geometry, Lipschitz analysis, RG, FLOP counting), not against the framework's own definitions.
- The chapter's parameter-count arithmetic for the factorised jump operator (lines 2455-2457) and the memory rows of the cost table were recomputed and are correct.
- Mechanical pre-pass: no dangling references or duplicate labels for this file.

## Open questions
- What are the intended semantics of `include_value_hessian` in `compute_state_space_fisher` (E-003), and should the Hessian term of $G_{zz}$ (control_loop.md:921) be included in the Tier-4 metric at all given the sign issues in E-005?
- Should the Tier-4 trust region (`_geodesic_dist`) be defined on policy outputs (Fisher-Rao / KL) or on latent states? The chapter currently mixes both (E-006, E-008).
- Is the "Texture as the Ideal Boundary" statement (E-015) meant to be a theorem about the stacked encoder in the limit $L\to\infty$? If so, the refinement structure and the limit object need to be defined before the proposition.
- Is the intended isometry claim (E-019) about the encoder path $\partial x^{(L)}/\partial x$ or about the per-block reconstruction maps $\partial\hat x^{(\ell)}/\partial x$? The fix differs.
- Should the stacked module's blocks all operate in observation space (E-021), or is a per-level projection intended? The definitions at lines 2013 and 2023 support the former.

## Rejected candidate findings
- None. All 25 stage-1 findings were confirmed or adjusted by the verifier; one finding (V-001, now E-004) was added.
