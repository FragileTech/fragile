# Mathematical Review: docs/source/1_agent/03_architecture/02_disentangled_vae.md

## Metadata
- Reviewed file: docs/source/1_agent/03_architecture/02_disentangled_vae.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (390 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/01_foundations/02_control_loop.md` — `def-causal-enclosure-condition` (lines 1084-1096), residual split and $z_{\text{geo}}$ (lines 340-352), latent manifold table (line 178), information bottleneck bound (line 565)
  - `docs/source/1_agent/10_appendices/06_losses.md` — F.1.3 `def-f-closure-loss` (lines 66-82), F.3.4 `def-f-closure-ratio` (lines 368-386)
  - `docs/source/1_agent/11_implementation/01_encoder.md` — lines 10, 48, 60, 166-168, 318, 353, 459, 733-735, 798-801
  - `docs/source/1_agent/10_appendices/07_architecture.md` — router/encoder/decoder diagrams (lines 217-320)
  - `src/fragile/core/layers/atlas.py` — `AttentiveAtlasEncoder` (94-360), `TopologicalDecoder` (364-470), `CovariantChartRouter` (664-866), `PrimitiveAttentiveAtlasEncoder` (868-1290), `PrimitiveTopologicalDecoder` (1291-1464)
  - `src/experiments/topoencoder_2d.py` — lines 140, 1070, 1617-1618, 1753, 1784; `src/fragile/core/losses.py` lines 131-134

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 6
- Minor: 2
- Notes: 1
- Primary themes:
  1. The chapter attributes its encoder, router and decoder diagrams to the `Primitive*` classes and `CovariantChartRouter` in `atlas.py`, but the diagrams describe the earlier flat-space classes (Euclidean sums, dot-product routing, `tanh` clamp) and a skew/Cayley router that the code has removed in favour of Poincare-distance scoring with conformal-factor transport.
  2. The routing-entropy term is described with the opposite sign to the cited training script, and the chapter contradicts itself about what the term does.
  3. The "closure ratio" defined here is a single-time routing-sharpness statistic whose stated identity is false in general; the label is cited by Appendix F.3.4 for a different transition-prediction ratio with the opposite orientation.
  4. The causal-enclosure definition weakens the framework anchor (nuisance independence made optional) and adds an ill-posed "sharply concentrated" clause.
  5. Cost estimates omit the dominant terms of the components drawn, and the symbol $K$ is used for both key dimension and codebook size.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Core Concept, `def-three-channel-latent` (54-64) vs encoder diagram (134-140) | Note | Dimensional mismatch | Framework | this chapter | Vector sums force $d_n=d_{\mathrm{tex}}=D$; definition presents them as free |
| E-002 | Core Concept, `def-causal-enclosure` (67-90) | Moderate | Definition mismatch | Framework | this chapter | Weakens the anchor's joint CMI condition; adds an ill-posed "sharply concentrated" clause |
| E-003 | Architecture, encoder and decoder diagrams (95-144, 184-210), TLDR (11-12) | Moderate | Algorithm mismatch | Framework | this chapter | Euclidean diagrams attributed to the hyperbolic `Primitive*` classes |
| E-004 | Covariant Chart Router diagram (146-180), TLDR (11) | Moderate | Algorithm mismatch | Framework | this chapter | Skew/Cayley, dot-product router drawn; implementation is Poincare-distance scoring with conformal-factor transport |
| E-005 | Loss Function, `def-total-disentangled-loss` (254-267) vs 373-374 | Moderate | Algorithm mismatch | Framework | this chapter | Routing-entropy term described with the wrong sign; self-contradiction |
| E-006 | Runtime Diagnostics (295-312) | Moderate | Definition mismatch | Framework | this chapter | "Closure ratio" measures routing sharpness, not closure; conflicts with Appendix F.3.4 citing the same label |
| E-007 | `def-closure-ratio` (306-308) | Moderate | Computational error | External | this chapter | $1-H(K\mid X)/\log N_c = I(X;K)/\log N_c$ holds only for uniform chart usage |
| E-008 | Computational Costs (349-353) | Minor | Computational error | External | this chapter | Router and soft-equivariant costs omit dominant factors; $K$ overloaded |
| E-009 | Differential-Geometry View (365-369) | Minor | Conceptual | External | this chapter | Temperature tracks the conformal factor on a constant-curvature ball, not "high-curvature regions" |

## Detailed findings

### [E-001] Residual dimensions are forced to equal the latent width (was F-009)
- Location: `def-three-channel-latent`, lines 54-64; encoder diagram lines 134-140
- Severity: Note
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 54: "$z_{n,t} \in \mathbb{R}^{d_n}$"; line 55: "$z_{\mathrm{tex},t} \in \mathbb{R}^{d_{\mathrm{tex}}}$"; line 60: "$z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n$"; line 137: "z_tex = delta_blended - z_n" with `delta_blended` of shape `[B, D]`.
- Upstream anchor: `01_foundations/02_control_loop.md:178` introduces generic $d_n$, $d_{\mathrm{tex}}$; `:349` has $z_{\mathrm{tex}} = \Delta_{\text{total}} - z_n$ with the same implicit constraint.
- Why this is an error: $c_{\mathrm{bar}}, z_{q,\mathrm{st}} \in \mathbb{R}^D$, so the sum defining $z_{\mathrm{geo}}$ requires $d_n = D$; $\Delta \in \mathbb{R}^D$ requires $d_{\mathrm{tex}} = D$. The free parameters in the definition are not free in this architecture. Not a correctness defect.
- Impact on downstream results: None.
- Fix guidance: State $d_n = d_{\mathrm{tex}} = D$ in the definition, or insert explicit embeddings $E_n:\mathbb{R}^{d_n}\to\mathbb{R}^D$, $E_{\mathrm{tex}}$ in the two formulas.
- Required new assumptions/permits: None.
- Validation plan: Check that every downstream use of $d_n$, $d_{\mathrm{tex}}$ is compatible with the stated equality.

### [E-002] Causal-enclosure definition weakens and mis-states the anchor condition (was F-001)
- Location: `def-causal-enclosure`, lines 67-90
- Severity: Moderate
- Type: Definition mismatch (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 70-86: "The macro symbol must satisfy the causal enclosure property: $P(K_{t+1} \mid K_t, a_t)$ is sharply concentrated and texture independence: $I(K_{t+1}; Z_{\mathrm{tex},t} \mid K_t, a_t) = 0.$ Optionally, in the strongest form, nuisance independence also holds: $I(K_{t+1}; Z_{n,t} \mid K_t, a_t) = 0.$"
- Upstream anchor: `01_foundations/02_control_loop.md:1084-1096`, `def-causal-enclosure-condition`: "The macro-model requirement is the conditional independence $K_{t+1} \perp\!\!\!\perp (z_{n,t}, z_{\mathrm{tex},t}) \big| (K_t,K^{\text{act}}_t)$, equivalently ... $I(K_{t+1};z_{n,t},z_{\mathrm{tex},t}\mid K_t,K^{\text{act}}_t)=0.$"
- Why this is an error: (a) The anchor is a single joint conditional independence from both residual channels. This chapter makes nuisance independence optional and labels the anchor's condition "the strongest form", so `def-causal-enclosure` and `def-causal-enclosure-condition` are two different predicates with one name. Both labels are cited in `11_implementation/01_encoder.md` (lines 10, 48, 60, 459, 799 for the former; 353, 801 for the latter) as though they were one condition. By the chain rule, $I(K_{t+1};Z_n,Z_{\mathrm{tex}}\mid C)=I(K_{t+1};Z_{\mathrm{tex}}\mid C)+I(K_{t+1};Z_n\mid Z_{\mathrm{tex}},C)$; the second term is conditioned on $Z_{\mathrm{tex}}$, so even both marginal CMIs vanishing does not imply the anchor's joint condition. (b) "$P(K_{t+1}\mid K_t,a_t)$ is sharply concentrated" has no threshold or entropy bound, is absent from the anchor, and conflates sufficiency of $K_t$ with determinism of the macro transition; a stochastic environment can satisfy Markov sufficiency with a diffuse kernel.
- Impact on downstream results: Implementation chapters citing `def-causal-enclosure` for the enclosure probe and GRL loss inherit an ambiguous target; the determinism clause invites diagnostics that reward sharpness rather than sufficiency (see E-006).
- Fix guidance:
  1. Replace the body of `def-causal-enclosure` with the anchor's joint condition $I(K_{t+1};z_{n,t},z_{\mathrm{tex},t}\mid K_t,a_t)=0$, or make the definition a `{prf:ref}` to `def-causal-enclosure-condition`.
  2. Delete the "sharply concentrated" clause.
  3. If a texture-only variant is wanted, give it a distinct name (for example "texture no-leak") and state that it is weaker than causal enclosure.
- Required new assumptions/permits: None.
- Validation plan: Grep for `def-causal-enclosure` across `docs/source/1_agent` and confirm each citing sentence is true under the joint condition.

### [E-003] Encoder and decoder diagrams are Euclidean but the named `Primitive*` classes are hyperbolic (was F-002)
- Location: Architecture section, lines 95-98; encoder diagram lines 100-144; decoder diagram lines 184-210; TLDR lines 11-12
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter (the same diagrams appear in `10_appendices/07_architecture.md:250-320` and the "dot-product fallback" phrase in `03_architecture/00_architecture_at_a_glance.md`)
- Claim (verbatim): lines 95-98: "TopoEncoderPrimitives couples a PrimitiveAttentiveAtlasEncoder with a PrimitiveTopologicalDecoder. ... Implementation lives in `src/fragile/core/layers/atlas.py`." Diagram nodes: line 115 "c_bar = sum(w_enc * c_k)"; 117 "v_local = v - c_bar"; 120 "diff = v_local - codebook"; 123 "dist = ||diff'||^2"; 136 "delta_blended = v_local - z_q_blended (detach)"; 139 "z_q_st = v_local + (z_q_blended - v_local).detach"; 140 "z_geo = c_bar + z_q_st + z_n"; decoder 188 "tanh(z_geo)"; 189 "CovariantChartRouter or latent_router"; TLDR 11-12 "with a dot-product fallback when disabled".
- Upstream anchor: `src/fragile/core/layers/atlas.py:1197-1275` (`PrimitiveAttentiveAtlasEncoder.forward`): `v = _project_to_ball(self.val_proj(features))`; fallback `scores = _poincare_hyperbolic_score(v, chart_centers, ...)` (1211); `c_bar = _poincare_weighted_mean(chart_centers, router_weights)`; `v_local = _project_to_ball(mobius_add(-c_bar, v))`; `diff = mobius_add(-codebook_exp, v_exp); diff_tan = log_map_zero(diff)`; `delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))`; `z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))`; `z_geo = _project_to_ball(mobius_add(c_bar, z_local))`. `atlas.py:1408` (`PrimitiveTopologicalDecoder.forward`): `z_geo = _project_to_ball(z_geo)`; non-covariant fallback at 1427 is `_poincare_hyperbolic_score`. The diagram formulas match the flat classes verbatim: `atlas.py:297` (`torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)`), `:343-348` (`z_tex = delta_blended - z_n`, `z_q_st = v_local + (z_q_blended - v_local).detach()`, `z_geo = c_bar + z_q_st + z_n`), `:441` (`torch.tanh(z_geo)` in `TopologicalDecoder`). The training script cited by the chapter instantiates `TopoEncoderPrimitives` (`src/experiments/topoencoder_2d.py:1070`).
- Why this is an error: In the named classes every "+" and "-" is a Mobius operation on the Poincare ball, $c_{\mathrm{bar}}$ is a hyperbolic barycenter rather than $\sum_k w_k c_k$, codebook distances are taken in the tangent space at the origin, the decoder clamps by `_project_to_ball` rather than `tanh`, and the non-covariant fallback is hyperbolic-distance scoring, not a dot product. Mobius addition is neither commutative nor associative, so the Euclidean identities in the diagram do not hold in the model whose training results the chapter's checklist and diagnostics refer to. The definition `def-three-channel-latent` itself agrees with the framework anchor (`02_control_loop.md:340-352`); the error is the attribution of these diagrams to the `Primitive*` classes.
- Impact on downstream results: `10_appendices/07_architecture.md` (same diagrams), `00_architecture_at_a_glance.md` ("dot-product fallback"), and `11_implementation/01_encoder.md:318,798`, which maps `def-three-channel-latent` to `PrimitiveAttentiveAtlasEncoder.forward`.
- Fix guidance:
  1. Either state that the diagrams show the flat-space `AttentiveAtlasEncoder` / `TopologicalDecoder` and that `TopoEncoderPrimitives` replaces each vector sum by its Poincare-ball counterpart (`mobius_add`, `log_map_zero` / `exp_map_zero`, `_poincare_weighted_mean`, `_project_to_ball`, hyperbolic-distance fallback), or redraw the diagrams with the hyperbolic operations.
  2. Replace "tanh(z_geo)" by "_project_to_ball(z_geo)" and "latent_router" by "_poincare_hyperbolic_score" in the decoder diagram if it is meant to depict the `Primitive*` decoder.
  3. Update the TLDR's "dot-product fallback" to "hyperbolic-distance fallback".
- Required new assumptions/permits: None.
- Validation plan: Walk `PrimitiveAttentiveAtlasEncoder.forward` and `PrimitiveTopologicalDecoder.forward` line by line against the redrawn diagrams; propagate to `07_architecture.md` and `00_architecture_at_a_glance.md`.

### [E-004] Covariant Chart Router diagram describes a Cayley/dot-product router that the implementation does not contain (was F-003; severity adjusted from Major to Moderate)
- Location: Covariant Chart Router diagram, lines 146-180; TLDR line 11
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter (duplicated in `10_appendices/07_architecture.md:217-224`; contradicted by `11_implementation/01_encoder.md:166-168`)
- Claim (verbatim): lines 159-172: "Z --> Transport[transport_proj(z) -> skew [B, K, K] (if use_transport)]; Transport --> Cayley[Cayley: U(z) = (I+0.5S)^-1 (I-0.5S)]; ... Keys[keys = U(z) * base_queries [B, N_c, K]]; Keys --> Scores[scores = sum(keys * q) [B, N_c]]; ... W[w = softmax(scores/tau)]"; line 151: "CovariantChartRouter (shared by encoder + decoder)"; TLDR line 11: "Wilson-line transport".
- Upstream anchor: `src/fragile/core/layers/atlas.py:664-668`: "class CovariantChartRouter ... Uses O(n) Poincare ball parallel transport instead of O(n^3) Cayley transform."; `:716`: "# Note: transport_proj removed - using O(n) hyperbolic transport instead"; `:734-770` (`_transport_queries`): "P_{0->z}(v) = v / lambda(z) ... return queries_expanded / lambda_z.unsqueeze(1)"; `:779-812` (`_hyperbolic_score`): `dist = acosh(1 + 2|z-c|^2/((1-|z|^2)(1-|c|^2)))`, `return -dist / tau`; `:853-861` (`forward`): "scores = self._hyperbolic_score(z, centers)" then, only "if self.q_feat_proj is not None and features is not None", "q = self.q_z_proj(z); q += self.q_feat_proj(features); q += self._gamma_term(z); keys = self._transport_queries(...); feature_scores = (keys * q).sum(-1); tau = self._temperature(z); scores = scores + 0.1 * feature_scores / tau". Decoder call `:1420-1425` passes no `features`. `11_implementation/01_encoder.md:166-168`: "giving an $O(n)$ alternative to the $O(n^3)$ Cayley transform."
- Why this is an error: In the cited class there is no `transport_proj`, no skew matrix and no Cayley transform; transport is a scalar rescaling by $1/\lambda(z)$. The primary logit is the negative Poincare distance to the chart centres divided by $\tau(z)$, which the diagram omits entirely. The query/key inner product that the diagram presents as the score is an additive correction with a hard-coded weight $0.1$, active only when `features` are supplied (encoder path); in the decoder path the router is purely distance-based, so the Christoffel term and transport never act there, contrary to "shared by encoder + decoder" with one score. The book already states elsewhere that the Cayley design was replaced.
- Impact on downstream results: The Computational Costs section (E-008) and the Differential-Geometry paragraph (line 368) reason about this router; `07_architecture.md:217-247` repeats the diagram; `08_multiagent/05_architecture.md` defers to this section for the representation stack. No mathematical result in the book is derived from the diagram, which is why the severity is Moderate rather than Major.
- Fix guidance:
  1. Redraw the router as `scores = -d_Poincare(z, c_k) / tau(z)` with `tau(z) = sqrt(K) (1 - ||z||^2)/2`.
  2. Add an optional, encoder-only correction node: `+ 0.1 * <P_{0->z}(base_queries), q_z(z) + q_feat(f) + gamma(z)> / tau(z)` with `P_{0->z}(v) = v / lambda(z)`, `lambda(z) = 2/(1 - ||z||^2)`.
  3. Remove the `transport_proj` / skew / Cayley nodes, or move them to a clearly labelled remark on the previous design; replace "Wilson-line transport" in the TLDR by "conformal-factor transport".
  4. If the Cayley router is meant to be canonical, change the code and `11_implementation/01_encoder.md:166-168` instead and say so here.
- Required new assumptions/permits: None.
- Validation plan: Compare the redrawn diagram with `CovariantChartRouter.forward` and with the prose in `11_implementation/01_encoder.md:160-175`; update `07_architecture.md` in step.

### [E-005] Routing-entropy term described with the wrong sign and contradicted later in the chapter (was F-004)
- Location: `def-total-disentangled-loss`, lines 254-267; The Entropy-Regularized Objective Functional, lines 373-374
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter (Appendix F.1.3 `10_appendices/06_losses.md:66-82` inherits the sign and cites this section as its source)
- Claim (verbatim): line 267: "$\mathcal{L}_{\text{entropy}}$ is routing entropy (encourages sharp routing)."; lines 373-374: "In the representation stack, entropy regularizes chart usage and prevents dead charts."
- Upstream anchor: `src/experiments/topoencoder_2d.py:140`: "entropy_weight: float = 0.1  # Encourage high routing entropy (anti-collapse)"; `:1617-1618`: "entropy_value = compute_routing_entropy(enc_w); entropy_loss = math.log(config.num_charts) - entropy_value"; `:1753`, `:1784`: "+ config.entropy_weight * entropy_loss"; `src/fragile/core/losses.py:131-134`: "entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=1); return entropy.mean()". Appendix `06_losses.md:70-79`: "$\mathcal{L}_{\text{entropy}} = -\frac1B\sum_b\sum_k w_{bk}\log(w_{bk}+\epsilon)$ ... Purpose: Penalizes diffuse routing."
- Why this is an error: With $\lambda_{\text{ent}}>0$ the training script minimises $\log N_c - H(w)$, i.e. it raises routing entropy (anti-collapse), the opposite of "encourages sharp routing". The chapter contradicts itself: line 267 says the term sharpens routing, line 374 says it prevents dead charts, which requires entropy to be pushed up. Appendix F.1.3 writes $+H$ with "penalizes diffuse routing", agreeing with line 267 and disagreeing with the implementation and line 374. In addition, the per-sample entropy $\frac1B\sum_b H(w_b)$ pushed upward does not by itself prevent dead charts (all samples can share the same diffuse distribution over a subset of charts); usage balance is the role of the batch-level diversity term.
- Impact on downstream results: Appendix F.1.3, Training Checklist item 1 ("monitor routing entropy"), and the closure-ratio diagnostic (E-006/E-007): with the implemented sign, training pushes a soft-weight version of $\rho_{\text{close}}$ toward 0, which the chapter labels as bad.
- Fix guidance:
  1. Write $\mathcal{L}_{\text{entropy}} = \log N_c - \frac1B\sum_b H(w_b)$ and describe it as an entropy-raising anti-collapse regulariser (or keep the appendix form and state $\lambda_{\text{ent}}<0$).
  2. Reconcile line 267 with lines 373-374 and with Appendix F.1.3 (same formula, same sign, same stated purpose).
  3. Attribute dead-chart prevention to the diversity/usage term rather than to per-sample routing entropy.
- Required new assumptions/permits: None.
- Validation plan: Confirm the sign against `topoencoder_2d.py:1617-1618` and that the appendix entry and this definition read identically.

### [E-006] The "closure ratio" measures routing determinism, not causal closure, and conflicts with Appendix F.3.4 (was F-006)
- Location: Runtime Diagnostics: The Closure Ratio, lines 295-312
- Severity: Moderate
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 298-311: "Closure is monitored through routing entropy, mutual information, and chart usage. A convenient normalized metric is the closure ratio: ... Let $K$ be the chart assignment and $N_c$ the number of charts. Define $\rho_{\text{close}} = 1 - \frac{H(K\mid X)}{\log N_c}$ ... Values near 1 indicate sharp, informative routing; values near 0 indicate diffuse routing."
- Upstream anchor: `10_appendices/06_losses.md:368-386`, F.3.4 `def-f-closure-ratio`: "$\text{Closure Ratio} = \frac{H(K_{t+1}\mid K_t,a_t)}{H(K_{t+1})}$ ... $\ll 1$ Strong predictive law learned ... Source: Section 3.2, Definition `def-closure-ratio`"; `11_implementation/01_encoder.md:735`: "The closure ratio (`def-closure-ratio`) measures how well the transition model predicts the next macro state"; `01_foundations/02_control_loop.md:1096`: "$I(K_{t+1};z_{n,t},z_{\mathrm{tex},t}\mid K_t,K^{\text{act}}_t)=0$".
- Why this is an error: Closure in this book is a property of the macro transition ($K_{t+1}$ given $K_t, a_t$). The quantity defined here involves only the single-time encoder map $X\mapsto K$ and says nothing about $K_{t+1}$ or about residual leakage; it is a routing-sharpness index. Appendix F.3.4 and `01_encoder.md:735` cite the same label for a transition-prediction ratio with the opposite orientation (small is good) and a different domain, so one label names two incompatible quantities. Moreover, with the implemented hard assignment `K_chart = torch.argmax(router_weights, dim=1)` (`atlas.py:1221`, `:864`) the encoder is deterministic, so $H(K\mid X)=0$ and $\rho_{\text{close}}\equiv 1$ regardless of model quality; the diagnostic is informative only if $K$ is reinterpreted as the soft weight vector, which the definition does not say.
- Impact on downstream results: `06_losses.md` F.3.4, `11_implementation/01_encoder.md:733-735`, and the chapter's diagnostics list (lines 314-319).
- Fix guidance:
  1. Rename the quantity defined here (for example "routing sharpness" or "normalised routing information") and state explicitly that $H(K\mid X)$ is the entropy of the soft router weights $w(X)$.
  2. Define the closure ratio proper as in Appendix F.3.4 (transition-entropy ratio), citing `def-causal-enclosure-condition`, and make this chapter and the appendix agree on name, formula and orientation.
- Required new assumptions/permits: None.
- Validation plan: Grep for `def-closure-ratio` and check each citing sentence matches the chosen definition.

### [E-007] The identity $1 - H(K\mid X)/\log N_c = I(X;K)/\log N_c$ requires a uniform chart marginal (was F-005)
- Location: `def-closure-ratio`, lines 306-308
- Severity: Moderate
- Type: Computational error (secondary: Invalid inference)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 306-308: "$\rho_{\text{close}} = 1 - \frac{H(K \mid X)}{\log N_c} \;=\; \frac{I(X;K)}{\log N_c}.$"
- Upstream anchor: not applicable (defined here). Related bound at `01_foundations/02_control_loop.md:565`: "$I(X; K) \le H(K) \le \log|\mathcal{K}|$".
- Why this is an error: $I(X;K) = H(K) - H(K\mid X)$, so $\frac{I(X;K)}{\log N_c} - \Big(1 - \frac{H(K\mid X)}{\log N_c}\Big) = \frac{H(K) - \log N_c}{\log N_c} \le 0$, with equality iff $H(K)=\log N_c$ (uniform chart usage), which is neither assumed nor generally true. Counterexample: $N_c=3$, routing deterministic given $X$ but only two charts used with equal frequency: $H(K\mid X)=0$ gives the first form $=1$; $I(X;K)=H(K)=\log 2$ gives the second form $\log 2/\log 3 = 0.631$.
- Impact on downstream results: `10_appendices/06_losses.md:386` and `11_implementation/01_encoder.md:735` cite `def-closure-ratio`; monitoring code implementing one side while the text promises the other reports different numbers.
- Fix guidance:
  1. Pick one quantity. For normalised routing determinism keep $1 - H(K\mid X)/\log N_c$ and drop the second equality.
  2. For normalised mutual information write $\rho = I(X;K)/\log N_c = (H(K)-H(K\mid X))/\log N_c$ and note that it equals the first form only when chart usage is uniform.
- Required new assumptions/permits: None (or the explicit assumption $H(K)=\log N_c$ if the equality is retained).
- Validation plan: Evaluate both expressions on a logged routing-weight batch with non-uniform usage and confirm they differ by $(\log N_c - H(K))/\log N_c$.

### [E-008] Computational-cost claims omit the dominant terms; symbol $K$ overloaded (was F-007)
- Location: Computational Costs, lines 349-353
- Severity: Minor
- Type: Computational error (secondary: Notation conflict)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 351-353: "Routing: $O(B N_c K)$ for key comparisons (per batch, per chart, per key dim). Codebook distances: $O(B N_c K D)$ for per-chart VQ. Soft-equivariant metric: per-chart SoftEquivariantLayer adds $O(B N_c D)$ plus hidden-size overhead."
- Upstream anchor: chapter lines 159-160 and 177 (skew `[B, K, K]`, Cayley solve, `gamma = einsum(z_i z_j, W_q_gamma[k,i,j])`); `atlas.py:720-723` (`z_outer = z.unsqueeze(2) * z.unsqueeze(1)  # [B, D, D]`; `einsum("bij,kij->bk", z_outer, self.q_gamma)`); `atlas.py:1136-1160` (`_apply_soft_equiv_metric` reshapes `diff[:, chart_idx]` of shape `[B, K_code, D]` to `[B*K_code, D]` and passes it through `layer`).
- Why this is an error: (a) The router as drawn includes a linear map to a $K\times K$ skew matrix ($O(BDK^2)$), a $K\times K$ solve ($O(BK^3)$) and, for full tensorisation, the Christoffel einsum ($O(BKD^2)$); none is $O(BN_cK)$. For the implemented router the per-sample work is $O(N_cD + KD^2)$ (distances plus the quadratic term), again not $O(N_cK)$. (b) The soft-equivariant metric is evaluated on every (chart, code) pair with a layer of hidden width $H$, so it costs $O(BN_cK_{\text{code}}DH)$; the stated $O(BN_cD)$ drops a factor $K_{\text{code}}H$. (c) $K$ denotes the router key dimension at lines 152-172 and the codebook size at lines 120-123 and 352; the cost lines mix both meanings.
- Impact on downstream results: None mathematical; affects sizing decisions.
- Fix guidance: Use $K_{\text{key}}$ and $K_{\text{code}}$; state routing cost as $O(B(N_cD + K_{\text{key}}D^2))$ for full tensorisation (or $O(B(N_cD + RD))$ for rank $R$); state the soft-equivariant cost as $O(BN_cK_{\text{code}}DH)$.
- Required new assumptions/permits: None.
- Validation plan: Count FLOPs from tensor shapes in `CovariantChartRouter.forward` and `_apply_soft_equiv_metric`.

### [E-009] "High-curvature regions" is not what the temperature tracks (was F-008)
- Location: Differential-Geometry View, lines 365-369; also Literature Connections line 346
- Severity: Minor
- Type: Conceptual (secondary: Miswording)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 368-369: "The metric-aware temperature in routing behaves like local conditioning, sharpening attention in high-curvature regions."
- Upstream anchor: chapter line 169: "tau(z) = sqrt(K) * (1 - ||z||^2)/2"; `atlas.py:728-732`: "Poincare ball conformal factor lambda(z) = 2 / (1 - |z|^2)"; `atlas.py:772-777`: "Poincare conformal factor scales attention temperature by radius. ... tau = math.sqrt(self.key_dim) * denom / 2.0".
- Why this is an error: $\tau(z)=\sqrt{K}/\lambda(z)$ depends on the conformal factor, i.e. on $\|z\|$, and shrinks toward the boundary. The Poincare ball has constant sectional curvature $-1$; there are no "high-curvature regions" to detect. The correct statement, which line 346 gives ("routing temperature encodes conformal metric information"), is that attention sharpens where the metric scale factor is large, i.e. near the boundary.
- Impact on downstream results: Prose only; the section title "Curvature as Conditioning" carries the same misattribution.
- Fix guidance: Replace "high-curvature regions" by "regions where the conformal factor $\lambda(z)=2/(1-\|z\|^2)$ is large (near the boundary of the ball), where hyperbolic distances are stretched"; retitle or qualify the section.
- Required new assumptions/permits: None.
- Validation plan: Read the revised paragraph against `_temperature` and `_conformal_factor` in `atlas.py`.

## Scope restrictions and clarifications
- `def-three-channel-latent` (Euclidean $z_{\mathrm{geo}}$ formula) is consistent with the book's framework anchor in `02_control_loop.md:340-352`; the findings concern the attribution of the flat-space formulas to the hyperbolic `Primitive*` classes, not the definition itself.
- The severity of E-004 was set to Moderate rather than Major because no lemma or proposition in the book is derived from the router diagram and a correct description already exists in `11_implementation/01_encoder.md`.
- The verifier read `src/fragile/core/layers/atlas.py`, `src/experiments/topoencoder_2d.py` and `src/fragile/core/losses.py` directly; all code quotations above were checked at the cited lines.

## Open questions
- Is the flat-space `AttentiveAtlasEncoder` / `TopologicalDecoder` still a supported configuration, or should the chapter document only `TopoEncoderPrimitives`? The answer determines whether the diagrams should be relabelled or redrawn.
- Should the canonical router be the implemented Poincare-distance router, or is the Cayley design intended to return? The chapter, `07_architecture.md`, `01_encoder.md` and the code must agree.
- Which quantity should carry the name "closure ratio": the transition-entropy ratio of Appendix F.3.4 or a routing-sharpness index? Only one can keep the label `def-closure-ratio`.

## Rejected candidate findings
None. All nine stage-1 findings were confirmed; F-003 was kept with severity adjusted from Major to Moderate.
