# Mathematical Review: docs/source/1_agent/02_sieve/04_approximations.md

## Metadata
- Reviewed file: docs/source/1_agent/02_sieve/04_approximations.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (561 lines); all numerical claims re-run with torch 2.7.1
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/02_sieve/01_diagnostics.md:95-112` (Sieve node table: nodes 6, 7a, 8, 9) and `:758-768` (adaptive multipliers)
  - `docs/source/1_agent/02_sieve/02_limits_barriers.md:55-68` (barrier table: BarrierVac, BarrierOmin, BarrierFreq, BarrierBode) and `:195-202` (BarrierBode regulariser)
  - `docs/source/1_agent/03_architecture/01_compute_tiers.md:55-85, 304` (tier tables, routing to this chapter)
  - `docs/source/1_agent/01_foundations/02_control_loop.md:100-104` (cost convention), `:139-143` (critic and covariant gradient), `:595-600` (Lyapunov decrease), `:1220-1226` (probing reference to this chapter)
  - `docs/source/1_agent/10_appendices/06_losses.md:705-720` (def-f-infonce), `:542-546` (metric-induced gradient norm)
  - `docs/source/1_agent/intro_agent.md:256, 514`

## Executive summary
- Critical: 0
- Major: 3
- Moderate: 5
- Minor: 4
- Notes: 2
- Primary themes:
  1. Three of the five surrogates do not measure the property they are said to preserve. The Jacobian-probe variance is a scale-dependent function of singular values, not eigenvalue spread (E-004); the TameCheck code differentiates the sum of outputs and returns exactly zero on non-tame maps (E-006); the TopoCheck chord alignment is not a necessary condition for reachability and penalises correct critics whenever the reachable set is non-convex (E-008).
  2. Sufficiency is asserted where the chapter itself denies it ("yields a path", "Ensures goal reachability", "Bounds Hessian norm"; E-007, E-009).
  3. The BarrierBode section replaces a different barrier (loop gain, upstream BarrierFreq) than the waterbed property assigned to BarrierBode upstream, and the tier table promises BarrierFreq and BarrierVac replacements that are absent (E-001, E-014).
  4. Cost and speedup accounting is internally inconsistent (E-005), and the in-batch InfoNCE branch includes the anchor among its own negatives (E-013).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | BarrierBode -> Temporal Gain Margin, 40-76, 552 | Moderate | Conceptual (Definition mismatch) | Framework | this chapter | Surrogate targets loop gain (BarrierFreq), not the waterbed property assigned to BarrierBode; the Bode integral does not detect instability |
| E-002 | BarrierBode code, 64 vs 100-114 | Note | Algorithm mismatch | Framework | this chapter | Displayed loss sums over lags; code averages over lags, time and batch; guard disables the check for T <= K |
| E-003 | BifurcateCheck, 137 vs 144-148 | Moderate | Conceptual (Definition mismatch) | Framework (secondary External) | upstream 02_sieve/01_diagnostics.md | det(J) does not test the unit-circle criterion the chapter states |
| E-004 | BifurcateCheck -> Jacobian Probing, 150-224, 553 | Major | Conceptual (Invalid inference) | External | this chapter | Var_v of the probe norm measures overall gain and singular-value fourth moment, not eigenvalue spread; blind to bifurcations |
| E-005 | BifurcateCheck/TameCheck cost remarks, 139, 148, 205, 258; table 552-554 | Minor | Computational error (Parameter inconsistency) | External | this chapter | Costs and speedups are mutually inconsistent and do not match the code |
| E-006 | TameCheck -> Lipschitz Gradient Proxy, 246-320, 554 | Major | Algorithm mismatch (Invalid inference) | External | this chapter | Code differentiates the sum of outputs; component Hessians cancel; loss identically zero on non-tame maps |
| E-007 | TameCheck prose, 256, 263 | Moderate | Invalid inference | External | this chapter | A bounded sampled ratio is a lower bound on the Hessian norm; the chapter asserts the converse |
| E-008 | TopoCheck -> Value Gradient Alignment, 342-404, 555 | Major | Conceptual (Invalid inference) | External | this chapter | Chord alignment is not necessary for reachability; correct critics are penalised in non-convex reachable sets |
| E-009 | TopoCheck, 359, 555 | Moderate | Invalid inference (Miswording) | External | this chapter | "Yields a path" and "Ensures goal reachability" assert sufficiency denied at line 354 |
| E-010 | TopoCheck formula and code, 345, 391-399 | Minor | Definition mismatch | Framework | this chapter | Flat chord and plain gradient used where the book uses the G-geodesic and the covariant gradient; flat limit not stated |
| E-011 | GeomCheck "Original", 418-423 | Note | Definition mismatch | Framework | this chapter | The "original" InfoNCE uses cosine similarity, not the geodesic kernel of def-f-infonce; equivalence only in the flat normalised limit |
| E-012 | GeomCheck prose, 434-436 | Minor | Conceptual | Framework (secondary External) | this chapter | Number of negatives sets the log(K+1) ceiling of the MI bound that node 6 (Cap_H) certifies |
| E-013 | GeomCheck code, 500-511 | Moderate | Algorithm mismatch | External | this chapter | In-batch negatives include the anchor itself for K of the B rows; comment says "excluding self" |
| E-014 | whole chapter; summary 543, 550-556 | Minor | Citation / reference error | Framework | upstream 03_architecture/01_compute_tiers.md | Tier table routes BarrierFreq and BarrierVac here; the chapter has no replacement for either |

## Detailed findings

### [E-001] BarrierBode surrogate addresses loop gain, not the waterbed property (was F-001)
- Location: BarrierBode -> Temporal Gain Margin, lines 40-76; summary table line 552
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 46: "Why care about this for neural policies? It detects instability. If your controller oscillates wildly, if errors amplify instead of shrink, the Bode integral catches it." Line 552: "| BarrierBode (FFT) | Temporal Gain | ~100x | Detects oscillatory instability |".
- Upstream anchor: `02_limits_barriers.md:65`: "| **BarrierBode** | Bode Sensitivity | **Policy** | **Waterbed Effect** | Suppressing error in one domain increases it in another. | $\int_{0}^{\infty} \log |S(j\omega)| d\omega = \text{const.}$ | FFT ✗ |"; `:199-202`: regulariser $\mathcal{L}_{\text{Bode}} = \lVert \mathcal{F}(e_t) \cdot W(\omega) \rVert^2$, "Explicitly decide *where* to be blind." The loop-instability barrier is `:64`: "| **BarrierFreq** | Frequency | **World Model** | **Loop Instability** | Positive feedback causes oscillation amplification. | $\Vert J_{WM} \Vert < 1$ (Jacobian Spectral Norm) | $O(Z^2)$ ✗ |".
- Why this is an error: The Bode sensitivity integral is an identity that every internally stable closed loop satisfies; it constrains how sensitivity is distributed across frequencies and does not signal instability. Upstream, BarrierBode is that distribution trade-off and its regulariser is a frequency-weighted allocation. The temporal gain ratio $\|e_{t+k}\|/\|e_t\|$ measures error amplification along trajectories, which is the loop-gain contract of BarrierFreq. Nothing in $\mathcal{L}_{\text{gain}}$ responds to a reallocation of sensitivity between bands (a decaying error with a different spectrum has the same ratio profile), so the "Preserved Property" entry does not hold for the barrier named. Additionally, a sustained constant-amplitude oscillation has ratio about 1 < G_max = 2 and is not flagged, so even "detects oscillatory instability" covers only growing oscillations.
- Impact on downstream results: `01_compute_tiers.md:81` routes BarrierBode, BarrierFreq and BarrierVac to this chapter. A reader implementing "BarrierBode" from here obtains a BarrierFreq monitor and no waterbed control (see E-014).
- Fix guidance:
  1. Retitle the section "BarrierFreq (loop gain) -> Temporal Gain Margin" and adjust the docstring at lines 86-89; the surrogate is a sound cheap proxy for $\|J_{WM}\| < 1$ along trajectories.
  2. Delete the sentence at line 46 or replace it with a statement about the waterbed trade-off.
  3. For BarrierBode, either keep the upstream frequency-weighted $\mathcal{L}_{\text{Bode}}$ on short windows, or state that the waterbed constraint is a design choice fixed by $W(\omega)$ rather than something a scalar probe detects.
  4. Update the row at line 552.
- Required new assumptions/permits: none.
- Validation plan: check that every barrier named in `01_compute_tiers.md:81` has a matching section heading here; confirm the text no longer claims the Bode integral detects instability.

### [E-002] Displayed gain loss and code differ in normalisation; guard disables short windows (was F-002)
- Location: line 64 vs lines 100-114
- Severity: Note
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 64: "$\mathcal{L}_{\text{gain}} = \sum_{k=1}^{K} \max(0, \frac{\Vert e_{t+k} \Vert}{\Vert e_t \Vert + \epsilon} - G_{\max})^2$"; lines 101-102: "if T <= K: return torch.tensor(0.0, ...)"; lines 112-114: "total_violation = total_violation + violation.mean() ... return total_violation / K".
- Why this is an error: The displayed loss is a per-$t$ sum over $K$ lags; the code returns $\frac{1}{K}\sum_k \mathbb{E}_{b,t}[\cdot]$, so per-term normalisations differ by $K$ and $B(T-k)$ (for $T = 1024$: 1023 to 1019). The early return at line 101 returns zero for $T = K$ although the loop at line 105 already handles $k < T$. Not a correctness issue given adaptive multipliers (`01_diagnostics.md:763`).
- Impact on downstream results: calibration of $\lambda_{\text{gain}}$ only.
- Fix guidance: write the formula as $\frac{1}{K}\sum_k \mathbb{E}_{b,t}[\cdot]$ or change the code to sum; change the guard to `T < 2`.
- Required new assumptions/permits: none.
- Validation plan: unit test with $T = K$ returns a non-zero loss on an amplifying sequence.

### [E-003] The "original" det(J) does not test the bifurcation criterion the chapter states (was F-004)
- Location: BifurcateCheck, line 137 vs lines 144-148
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework (secondary External)
- Origin: upstream `docs/source/1_agent/02_sieve/01_diagnostics.md`
- Claim (verbatim): line 137: "When an eigenvalue crosses the unit circle (discrete time) or imaginary axis (continuous time), you have a bifurcation." Line 145: "$\det(J_{S_t}) \quad \text{where } J_{S_t} = \frac{\partial S_t(z)}{\partial z}$".
- Upstream anchor: `01_diagnostics.md:101`: "| **7a** | **BifurcateCheck ($\mathrm{LS}_{\partial^2 V}$)** | **World Model** | **Instability Check** | Bifurcation point? | $\det(J_{S_t})$ (Jacobian Determinant) | $O(Z^3)$ ✗ |".
- Why this is an error: $S_t$ is a discrete-time map (`world_model: S_t(z, a) -> z_next`, line 185). $\det J = \prod_i \lambda_i$ vanishes when some $\lambda_i = 0$, an infinitely contracting direction, which is unrelated to $|\lambda| = 1$. Example: $J = \mathrm{diag}(1, 0.5)$ sits on a fold with $\det J = 0.5$; $J = \mathrm{diag}(0, 0.5)$ has $\det J = 0$ with no bifurcation. The chapter states the right criterion and then labels an unrelated scalar as the exact test; the surrogate of E-004 approximates neither.
- Impact on downstream results: any implementation of node 7a via $\det J$ (or $\log|\det J|$) does not detect bifurcations.
- Fix guidance:
  1. Upstream (`01_diagnostics.md:101`): replace the regulariser by the spectral radius $\rho(J_{S_t})$ with penalty $\mathrm{ReLU}(\rho - 1 + \delta)^2$, or by $\min_i \big| |\lambda_i| - 1 \big|$ as a distance to bifurcation.
  2. In this chapter, state the original as $\rho(J_{S_t})$ and use the power-iteration surrogate of E-004.
- Required new assumptions/permits: none.
- Validation plan: the new original and surrogate must both flag $J = \mathrm{diag}(-1.01, 0.5, \dots)$ and pass $J = \mathrm{diag}(0.99, 0.5, \dots)$.

### [E-004] Variance of the probe norm measures gain and singular-value spread, not eigenvalue spread (was F-003)
- Location: BifurcateCheck -> Stochastic Jacobian Probing, lines 150-224; summary line 553
- Severity: Major
- Type: Conceptual (secondary: Invalid inference)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 153: "$\mathcal{L}_{\text{bifurcate}} = \text{Var}_v\left[\Vert J_{S_t} v \Vert^2\right] \quad \text{where } v \sim \mathcal{N}(0, I)$"; line 158: "If eigenvalues are similar, the Jacobian stretches all directions roughly equally. If some are huge and others tiny, different random directions get stretched by wildly different amounts."; line 164: "This is the Hutchinson trace estimator"; line 167: "High variance in the Jacobian-vector product norm indicates instability (eigenvalue spread)."; line 553: "Detects eigenvalue spread".
- Upstream anchor: `01_diagnostics.md:101` (node 7a, quoted in E-003); the chapter's own criterion at line 137.
- Why this is an error: For $v \sim \mathcal{N}(0, I)$, $\|Jv\|^2 = v^\top J^\top J v$ has $\mathbb{E} = \sum_i \sigma_i^2$ and $\operatorname{Var} = 2\sum_i \sigma_i^4$ in terms of singular values. It is not scale-free and it does not see eigenvalues. Recomputation ($Z = 256$, $2\times10^5$ probes, float64, statistic as in the code: variance of the un-squared norm with threshold 1.0): $J = I$: 0.500; $1.2I$: 0.719; $1.5I$: 1.129 (exceeds the threshold with zero spread); $2I$: 2.003. Squared-norm variances 512, 1062, 2605, 8214 agree with $2Zc^4$. Non-normal $J = I + 50\,e_0 e_1^\top$ (all eigenvalues exactly 1): variance 749 (squared: $1.25\times10^7$). Diagonal $J$ with leading entry $0.99$ versus $-1.01$ (a flip bifurcation): 0.1306 vs 0.1308, indistinguishable. The scale-free quantity $\operatorname{Var}/\mathbb{E}^2 = 2\sum\sigma^4/(\sum\sigma^2)^2 \in [2/Z, 2]$ (recomputed 0.0078 = 2/Z for all isotropic cases) is not what is computed. "Hutchinson trace estimator" refers to $\mathbb{E}_v[v^\top A v] = \operatorname{tr} A$, i.e. to the mean $\mathbb{E}\|Jv\|^2 = \|J\|_F^2$ (recomputed 256.0 for $J = I$), not to the variance across probes. Code notes: `grad_outputs=v` computes $J^\top v$, not $Jv$ (same distribution for square $J$, notational only), and "efficient: O(Z)" is wrong (one backward pass is $O(P_{WM})$).
- Impact on downstream results: BifurcateCheck is in the Advanced tier (`01_compute_tiers.md:61`); `02_control_loop.md:1224` cites this section for "eigenvalue spread proxies" of the metric $G$. As written the loss penalises overall gain and rewards shrinking the world-model Jacobian, which is a contraction/BarrierFreq property, not a bifurcation detector.
- Fix guidance:
  1. Replace the statistic by a few steps of power iteration on $J$ (alternating JVP/VJP) to estimate the spectral radius or largest singular value, and penalise $\mathrm{ReLU}(\hat\rho - 1 + \delta)^2$, matching line 137.
  2. If a spread measure is wanted, use $\operatorname{Var}_v\|Jv\|^2 / (\mathbb{E}_v\|Jv\|^2)^2$ and say it measures singular-value concentration.
  3. Remove the Hutchinson attribution or attach it to the mean; fix the cost comment; update line 553.
- Required new assumptions/permits: none.
- Validation plan: re-run the four test matrices above; the new loss must be invariant under $J \to cJ$ for the spread variant, or must flag only $|\lambda| \ge 1 - \delta$ for the radius variant.

### [E-005] Cost accounting is internally inconsistent (was F-005)
- Location: lines 139, 148, 205, 258; summary table lines 552-554
- Severity: Minor
- Type: Computational error (secondary: Parameter inconsistency)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 139: "the full Jacobian costs $O(Z^2)$ to form and $O(Z^3)$ for eigenvalues"; line 148: "Computing the full Jacobian is $O(Z^3)$"; line 205: "# Vector-Jacobian product (VJP) via autodiff (efficient: O(Z))"; line 258: "The whole operation is $O(Z)$ instead of $O(Z^2)$"; lines 552-554: "~100x", "~$Z^2/K$", "~$ZP$".
- Upstream anchor: `01_diagnostics.md:101` "$O(Z^3)$ ✗" (7a), `:106` "$O(Z^2 P_{WM})$ ✗" (9); `02_limits_barriers.md:65` "FFT ✗".
- Why this is an error: (i) Lines 139 and 148 disagree on what costs $O(Z^3)$. (ii) Forming $J_{S_t}$ by reverse mode takes $Z$ backward passes, $O(Z P_{WM})$; the $O(Z^3)$ part is the decomposition. (iii) One VJP or gradient through a network with $P_{WM}$ parameters is $O(P_{WM})$, never $O(Z)$; the probe surrogate is $O(K P_{WM})$ and the TameCheck surrogate $O(P_{WM})$, so speedups are about $Z^3/(K P_{WM})$ and $Z^2$, not $Z^2/K$ and $ZP$. (iv) An FFT of a length-$T$ window is $O(T\log T)$ against $O(KT)$ for the gain loss; $T = 1024$, $K = 5$ gives $10240/5120 = 2$, not 100 (the upstream objection is stationarity, not FLOPs). (v) $256^3 = 16{,}777{,}216$ equals the MACs of one $256\times256$ dense layer on a batch of 256, so "Not practical" (line 139) is not carried by the number.
- Impact on downstream results: the table is advertised as an engineering index (line 13) and informs tier assignment in `01_compute_tiers.md`.
- Fix guidance: state costs in forward/backward passes (original: $Z$ backward passes plus $O(Z^3)$; probe surrogate: $K$ backward passes; TameCheck surrogate: 2 backward passes), give speedups $\approx Z/K$ and $\approx Z^2$, and replace "~100x" by "removes the stationary-window requirement; FLOPs comparable".
- Required new assumptions/permits: none.
- Validation plan: recompute each table cell from the stated costs.

### [E-006] TameCheck code differentiates the sum of outputs and does not bound the Hessian of $S_t$ (was F-006)
- Location: TameCheck -> Lipschitz Gradient Proxy, lines 246-320; summary line 554
- Severity: Major
- Type: Algorithm mismatch (secondary: Invalid inference)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 249: "$\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}$"; lines 276-278: "we estimate the Lipschitz constant of the gradient via finite differences. This bounds the Hessian spectral norm (tameness)."; lines 302-305: "# Sum over output dims to get [B, Z] gradient / grad1 = torch.autograd.grad(z1_next.sum(), z1, ...)"; line 554: "Bounds Hessian norm".
- Upstream anchor: `01_diagnostics.md:106`: "| **9** | **TameCheck ($\mathrm{TB}_O$)** | **World Model** | **Interpretability Check** | Dynamics Lipschitz-bounded? | $\Vert \nabla^2 S_t \Vert$ (Hessian Norm / Smoothness) | $O(Z^2 P_{WM})$ ✗ |"; signature "world_model: S_t(z, a) -> z_next" (lines 185, 281).
- Why this is an error: $S_t$ is vector-valued, so $\nabla^2 S_t$ is a rank-3 tensor of $Z$ component Hessians. The code differentiates $f(z) = \sum_i S_t^i(z)$, whose Hessian is $\sum_i \nabla^2 S_t^i$, and components can cancel. Recomputation with the chapter's `compute_tame_loss` verbatim on $S_t(z) = (z_1^2, -z_1^2)$ (each component Hessian has spectral norm 2): `lipschitz_estimate = [0.0, 0.0, 0.0, 0.0]`, loss 0.0; scaling both components by 100 still gives all zeros. A random output-direction probe $g = \nabla_z\langle u, S_t(z)\rangle$ on the same map gives non-zero estimates (2.53, 1.29, 0.17, 1.64 for four random $u$). Hence line 263 ("Bounded gradient Lipschitz constant implies bounded Hessian") holds for the scalar $f$ but does not transfer to $S_t$, and "Bounds Hessian norm" is false for the code shown.
- Impact on downstream results: node 9 is in the Advanced tier (`01_compute_tiers.md:61`); BarrierOmin (`02_limits_barriers.md:61`) relies on the same smoothness notion.
- Fix guidance:
  1. Draw one or more output directions $u \sim \mathcal{N}(0, I_Z)$ and compute $g_i = \nabla_z\langle u, S_t(z_i)\rangle = J(z_i)^\top u$ at $z_1, z_2$; use $\|g_1 - g_2\|/\|z_1 - z_2\|$, whose supremum over unit $u$ and $\delta$ is the Hessian-tensor norm. Equivalent alternative: finite difference of JVPs, $\|J(z_2)\delta - J(z_1)\delta\|/\|\delta\|^2$.
  2. Write the displayed formula with $J_{S_t}$ instead of $\nabla_z S_t$; fix the docstring and line 554.
- Required new assumptions/permits: none.
- Validation plan: the corrected loss must be non-zero on $(z_1^2, -z_1^2)$ and scale linearly with $c$ on $(c z_1^2, -c z_1^2)$.

### [E-007] A bounded sampled ratio does not imply a bounded Hessian (was F-007)
- Location: TameCheck prose, lines 256, 263
- Severity: Moderate
- Type: Invalid inference
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 256: "Measure how much the gradient changed relative to how much the input changed. Bounded ratio means bounded Hessian---that is literally what the Hessian measures."; line 263: "Bounded gradient Lipschitz constant implies bounded Hessian (by definition)."
- Why this is an error: For $C^2$ scalar $f$, $\|\nabla f(z_1) - \nabla f(z_2)\| = \|\nabla^2 f(\xi)\,\delta\| \le \|\nabla^2 f(\xi)\|\,\|\delta\|$, so a single finite-difference ratio along a random $\delta$ is a lower bound on the local Hessian norm. A bounded Lipschitz constant (a supremum over all pairs) bounds the Hessian; a bounded sample ratio does not. Recomputation at $Z = 256$ with a Hessian having one stiff direction of norm 1: the mean of $\|H\hat\delta\|$ over $10^4$ random unit $\hat\delta$ is 0.0498 ($\sqrt{2/(\pi Z)} = 0.0499$), a twenty-fold underestimate. The correct statement is "small estimate is necessary for tameness"; the prose asserts the converse.
- Impact on downstream results: the "Bounds Hessian norm" cell at line 554 and line 545 ("Enough gradient samples to bound Lipschitz constants") rest on this claim.
- Fix guidance: replace both sentences with "the sampled ratio is a lower bound on $\|\nabla^2 S_t\|$ near $z$; a large value certifies non-tameness, a small value is only evidence"; optionally add a few power-iteration steps on the Hessian-vector product to tighten the bound from below.
- Required new assumptions/permits: none.
- Validation plan: compare the estimate with the exact Hessian norm on a small quadratic model.

### [E-008] Value-gradient alignment is not necessary for reachability and penalises correct critics (was F-008)
- Location: TopoCheck -> Value Gradient Alignment, lines 342-404; summary line 555
- Severity: Major
- Type: Conceptual (secondary: Invalid inference)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 345: "$\mathcal{L}_{\text{topo}} = \mathrm{ReLU}\left(\left\langle \nabla_z V(z), \frac{z_{\text{goal}} - z}{\| z_{\text{goal}} - z \|} \right\rangle\right)$"; line 354: "This is necessary but not sufficient for reachability. If you cannot start moving the right direction, you certainly cannot arrive."; line 372: "This is a necessary condition for gradient-based reachability."; line 331: "If the latent space has holes or barriers, the value function might look smooth and encouraging while the goal is actually unreachable."
- Upstream anchor: `01_diagnostics.md:105`: "| **8** | **TopoCheck ($\mathrm{TB}_\pi$)** | **Policy** | **Sector Reachability** | Goal reachable? | $T_{\text{reach}}(z_{\text{goal}})$ (Reachability Map) | $O(HBZ)$ ✗ |". Cost convention: `02_control_loop.md:102`: "This chapter uses a cost convention (lower is better)."
- Why this is an error: "The right direction" is the direction of the reachable path, not the Euclidean chord. Counterexample (recomputed): agent at $z = (0, 0)$ inside a cup with walls at $x = \pm 1$ for $y \in [-1, 2]$ and a floor at $y = -1$; goal at $(0, -3)$, reachable by exiting at $y = 2$. The correct cost-to-go decreases in $+y$, so $\nabla V \propto (0, -1)$, $\hat d_{\text{goal}} = (0, -1)$, and $\langle \nabla V, \hat d_{\text{goal}}\rangle = \|\nabla V\| > 0$: the correct critic receives the maximal penalty, and a training step on $\mathcal{L}_{\text{topo}}$ pushes $V$ toward pointing into the floor. So the test is not necessary; it presupposes a star-shaped reachable set in the chosen coordinates, and it fails precisely in the "holes or barriers" regime it is introduced to catch. The loss also never consults $S_t$ or $\pi$, although node 8 is a Policy-interface check of dynamical reachability.
- Impact on downstream results: TopoCheck is in the Advanced tier (`01_compute_tiers.md:61`); `intro_agent.md:514` sends readers here. A critic trained with this penalty is biased in maze-like latent topologies.
- Fix guidance:
  1. Either restrict: "necessary only if the reachable set from $z$ to $z_{\text{goal}}$ is star-shaped with respect to $z_{\text{goal}}$" and use it as a monitor, never as a training loss;
  2. or replace the chord by a dynamically admissible direction: penalise $\mathrm{ReLU}\big(V(S_t(z, a)) - V(z) + \lambda_{\text{Lyap}} V(z)\big)$ (the Lyapunov decrease of `02_control_loop.md:598`), or use $k$-step model rollouts with $k \ll H$ as a truncated $T_{\text{reach}}$.
  3. Update line 372 and the row at line 555.
- Required new assumptions/permits: for option 1, an explicit star-shapedness assumption on the reachable set.
- Validation plan: the corrected check must give zero loss for the correct critic in the cup example.

### [E-009] Sufficiency is asserted where the section denies it (was F-009)
- Location: TopoCheck, line 359; summary line 555
- Severity: Moderate
- Type: Invalid inference (secondary: Miswording)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 359: "When $-\nabla_z V(z)$ aligns with the goal direction, gradient descent on $V$ yields a path to $z_{\text{goal}}$."; line 555: "TopoCheck ($O(HBZ)$) | Value Alignment | ~$H$ | Ensures goal reachability".
- Why this is an error: Pointwise alignment at $z$ says nothing about the gradient flow beyond $z$. Recomputed one-dimensional counterexample: $V(z) = (z^2 - 1)^2$, $z = 0.2$, $z_{\text{goal}} = 3$; $V'(0.2) = -0.768 < 0$, so $\mathcal{L}_{\text{topo}} = 0$, yet gradient descent from $0.2$ converges to $z = 1.000$ and never reaches 3. Line 354 already states "necessary but not sufficient ... Obstacles might still block the path"; line 359 and "Ensures" contradict it.
- Impact on downstream results: readers of the index table will read node 8 as satisfied when the loss is zero.
- Fix guidance: replace line 359 with "Negative alignment is a first-order local consistency check on $V$; it does not by itself guarantee a descent path to $z_{\text{goal}}$" and change the table cell to "First-order value-direction consistency (under the restriction of E-008)".
- Required new assumptions/permits: none.
- Validation plan: textual; confirm lines 354, 359 and 555 agree.

### [E-010] Alignment uses the flat chord and plain gradient without stating the flat-metric limit (was F-010)
- Location: TopoCheck, line 345 and lines 391-399
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 345 (formula above); code: "to_goal = goal_states - states" and "alignment = (grad_v * to_goal_normalized).sum(dim=-1)".
- Upstream anchor: `02_control_loop.md:141`: "Defines the gradient signal $\nabla_A V$."; `06_losses.md:544`: "The covariant gradient norm $\|\nabla_A V\|_G$ measures the gauge-invariant rate of change along geodesics."; `01_diagnostics.md:110` (node 12a) uses $d_G$ for latent distances.
- Why this is an error: Elsewhere the Sieve measures latent directions and distances with $G$ and the covariant gradient $\nabla_A V$. Under a non-trivial $G$ the descent direction is $-G^{-1}\nabla_A V$ and the direction to the goal is the initial velocity of the $G$-geodesic, not the chord. The chapter never states that the code is the $G = I$, $A = 0$ special case. Clarification of the stage-1 argument: the flat alignment $\langle \nabla V, z_{\text{goal}} - z\rangle$ is invariant under affine reparametrisations ($z' = Az$ gives $\nabla V^\top A^{-1} A (z_{\text{goal}} - z)$), so coordinate dependence arises only under nonlinear reparametrisations or a non-flat metric; the mismatch with the framework's metric objects stands, the linear sign-flip claim does not.
- Impact on downstream results: minor if read as the flat limit; otherwise inconsistent with the metric-aware losses of Appendix F.
- Fix guidance: write the alignment as $\langle \nabla_A V, \dot\gamma_{z\to z_{\text{goal}}}(0)\rangle_G$ (first-order: $\nabla_A V^\top G^{-1}(z_{\text{goal}} - z)$) and add "flat limit $G = I$, $A = 0$ shown in the code".
- Required new assumptions/permits: none.
- Validation plan: textual.

### [E-011] The "original" InfoNCE is not the framework's definition (added by verifier, V-001)
- Location: GeomCheck, lines 418-423 and code lines 491-511
- Severity: Note
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 421: "$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_t, z_{t+k}))}{\sum_{j=1}^{B} \exp(\text{sim}(z_t, z_j))}$" with `sim` unspecified; code: cosine similarity of projected, L2-normalised embeddings divided by `tau`.
- Upstream anchor: `06_losses.md:705-712` (def-f-infonce): "$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(-d_G(z_t, z_{t+k})^2 / \tau\right)}{\sum_{j} \exp\left(-d_G(z_t, z_j)^2 / \tau\right)}$ ... $d_G(z, z')$ – geodesic distance under metric $G$".
- Why this is an error: For unit-normalised embeddings and $G = I$, $-\|z - z'\|^2/\tau = 2\cos(z, z')/\tau - 2/\tau$ and the constant cancels in the softmax, so the chapter's form agrees with F.9.2 only in the flat, normalised limit with the temperature rescaled ($\tau_{\cos} = \tau_G/2$). The chapter presents its cosine form as "the" original without saying so, in the same way as E-010.
- Impact on downstream results: none numerically; the "Preserves slow features" row (line 556) is asserted for a different similarity than the one node 6 is defined with.
- Fix guidance: make `sim` explicit in the formula ($\cos/\tau$) and add one sentence stating the flat-metric, normalised-embedding equivalence with def-f-infonce.
- Required new assumptions/permits: none.
- Validation plan: textual.

### [E-012] The number of negatives sets the certifiable mutual-information ceiling (was F-012)
- Location: GeomCheck prose, lines 434-436
- Severity: Minor
- Type: Conceptual
- Criterion: Framework (secondary External)
- Origin: this chapter
- Claim (verbatim): line 434: "If the encoder distinguishes the true positive from 128 random negatives, it has learned something useful. Whether it could beat 1024 does not matter."
- Upstream anchor: `01_diagnostics.md:99`: "| **6** | **GeomCheck ($\mathrm{Cap}_H$)** | **VQ-VAE / WM** | **Blind Spot Check** | Unobservable states negligible? | $\mathcal{L}_{\text{contrastive}}$ (InfoNCE) | $O(B^2Z)$ ⚡ |"; `06_losses.md:718`: "maximizing mutual information between present and future representations."
- Why this is an error: InfoNCE with $K$ negatives certifies at most $I(z_t; z_{t+k}) \ge \log(K+1) - \mathcal{L}$. Recomputed: $\log 129 = 4.86$ nats versus $\log 1024 = 6.93$ nats for the full batch. Node 6 is a capacity diagnostic, so the number of negatives is exactly what sets the certifiable bound; "does not matter" is wrong for this node's role even if it is fine for representation quality in general.
- Impact on downstream results: any threshold on $\mathcal{L}_{\text{InfoNCE}}$ used to trigger GeomCheck must be expressed relative to $\log(K+1)$.
- Fix guidance: replace the sentence with "Sampling $K$ negatives caps the certifiable mutual information at $\log(K+1)$; choose $K$ (or a memory bank of size $M \gg B$) so that $\log(K+1)$ exceeds the capacity budget being checked", and present the memory bank as required whenever the bound matters.
- Required new assumptions/permits: none.
- Validation plan: textual; check thresholds in any downstream configuration.

### [E-013] In-batch negatives include the anchor itself (was F-011)
- Location: GeomCheck code, lines 500-511
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 501-505: "# Use other batch elements as negatives (in-batch) / K = min(self.n_negatives, B - 1) / # Shuffle and take first K (excluding self) / perm = torch.randperm(B, ...) / negatives = anchor[perm[:K]]"; line 511: "neg_sim = torch.mm(anchor, negatives.T) / self.tau".
- Upstream anchor: `06_losses.md:709-712`: "$z_j$ – negative samples (other timesteps or other sequences)".
- Why this is an error: The same $K$ rows are used as negatives for every anchor, so each of the $K$ anchors whose index lies in `perm[:K]` has its own embedding among its negatives with cosine similarity 1 (logit $1/\tau = 10$). Recomputed ($B = 8$, $K = 4$): `perm[:K] = [0, 4, 5, 1]` and rows 0, 1, 4, 5 each contain a self-similarity of exactly 1.0. Nothing excludes self, contrary to the comment. For the affected fraction $K/B$ of rows (12.5% at the defaults $K = 128$, $B = 1024$) the positive's softmax weight is capped at $1/2$, so the loss floor for a perfectly separated positive is $\log 2 = 0.693$ nats (recomputed 0.6932) instead of 0, and the self term contributes a gradient unrelated to the positive. A secondary point: this branch draws negatives from the anchors $z_t$ rather than from the $z_j$ of the displayed original at line 421, which is a legitimate variant but should be stated.
- Impact on downstream results: node 6 loss and the "Preserves slow features" claim (line 556) depend on a correct contrastive objective; the bias is silent.
- Fix guidance: sample per-row indices that never equal the row index, e.g. `idx = (torch.arange(B)[:, None] + torch.randint(1, B, (B, K))) % B`, or compute similarities against a random column subset and mask the diagonal with `-inf` before `logsumexp`; fix the comment.
- Required new assumptions/permits: none.
- Validation plan: unit test asserting no negative logit equals `1/tau` for identical anchor and positive inputs; loss of a perfectly separated batch must approach 0.

### [E-014] Tier table promises BarrierFreq and BarrierVac replacements that are absent (was F-013)
- Location: whole chapter (sections at lines 40, 129, 227, 323, 407; summary lines 543, 550-556)
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: upstream `docs/source/1_agent/03_architecture/01_compute_tiers.md`
- Claim (verbatim): line 543: "Five expensive theoretical tests, five cheap surrogates".
- Upstream anchor: `01_compute_tiers.md:81`: "| **Infeasible** | BarrierBode, BarrierFreq, BarrierVac | See {ref}`sec-infeasible-implementation-replacements` | Need replacements |"; `02_limits_barriers.md:64` (BarrierFreq, "$\Vert J_{WM} \Vert < 1$"), `:57` (BarrierVac, "$\Vert \nabla^2 V(z) \Vert$ ... $O(BZ^2)$ ✗").
- Why this is an error: the reference resolves but the promised content is missing: there is no BarrierFreq or BarrierVac replacement here. The temporal-gain surrogate (E-001) is a natural BarrierFreq proxy and the repaired TameCheck probe (E-006) applied to $V$ is a natural BarrierVac proxy, but neither mapping is stated.
- Impact on downstream results: implementers following the tier table find no guidance for two of the three Infeasible barriers.
- Fix guidance: add two short subsections (BarrierFreq -> temporal gain margin or power-iteration spectral norm of $J_{WM}$; BarrierVac -> Hessian-vector-product probe of $V$) and extend the summary table, or change `01_compute_tiers.md:81` to route only BarrierBode here and list the other two inline.
- Required new assumptions/permits: none.
- Validation plan: every barrier named at `01_compute_tiers.md:81` has a matching section here.

## Scope restrictions and clarifications
- The chapter's own closing note (line 560) says the surrogates are not equivalent to the originals. That disclaimer covers edge cases; it does not cover surrogates that measure a different quantity in the generic case (E-004, E-006, E-008) or stated implications in the wrong direction (E-007, E-009).
- The temporal gain margin, the corrected TameCheck probe, and sampled InfoNCE are sound tools for the properties they actually measure; the findings concern the labels and claims attached to them, not their usefulness.
- All code checks were run on toy maps in float32/float64 with torch 2.7.1; the conclusions do not depend on numerical tolerance.

## Open questions
- Should BifurcateCheck (node 7a) target the spectral radius of $J_{S_t}$ at fixed points of the world model, or along sampled trajectories? The power-iteration fix works for either, but the upstream table should say which.
- For TopoCheck, is the intended object a property of the critic alone (value consistency) or of the closed loop (dynamical reachability)? The upstream node is assigned to the Policy interface, which suggests the latter and favours the Lyapunov-decrease replacement.
- What capacity budget does GeomCheck certify, so that the required $\log(K+1)$ can be fixed?

## Rejected candidate findings
- None. All thirteen stage-1 findings were confirmed; three were adjusted (E-003: criterion retagged Framework; E-010: the linear-reparametrisation sign-flip argument was wrong and has been replaced; E-013: the loss floor is $\log 2$, not $1/\tau - \text{pos\_sim}$).
