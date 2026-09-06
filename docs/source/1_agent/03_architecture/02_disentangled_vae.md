(sec-the-disentangled-variational-architecture-hierarchical-latent-separation)=
(sec-topoencoder-architecture)=
# TopoEncoder Architecture: Attentive Atlas with Typed Latents

## TLDR

- The representation stack is an Attentive Atlas (TopoEncoderPrimitives) with chart routing and
  per-chart codebooks.
- The discrete macro state is charted: $K_t = (K_{\mathrm{chart}}, K_{\mathrm{code}})$, with
  continuous nuisance $z_n$ and texture $z_{\mathrm{tex}}$; geometry uses $z_{\mathrm{geo}}$.
- Routing uses CovariantChartRouter (conformal-factor transport + metric-aware temperature) with a
  hyperbolic-distance fallback when disabled.
- Decoding mixes chart projectors with router weights and adds a separate texture residual path.
- Training uses reconstruction + VQ + routing/consistency, plus tiered regularizers and optional
  jump and supervised topology losses.

## Roadmap

1. Typed latents and causal enclosure.
2. Encoder and chart routing.
3. Per-chart VQ and latent decomposition.
4. Decoder block.
5. Training wiring and loss tiers.
6. Diagnostics and extensions.

## Training Checklist (Practical)

1. Start with stable chart routing: monitor routing entropy and chart usage early.
2. Keep chart centers separated (chart center separation loss) before enabling expensive losses.
3. If `vision_preproc` is enabled, validate the input shape and `vision_*` settings.
4. If `soft_equiv_metric` is enabled, watch `soft_equiv_l1` and `soft_equiv_log_ratio` for drift.
5. Add jump consistency and supervised topology only after the atlas stabilizes.

(sec-the-core-concept-split-brain-architecture)=
## The Core Concept: Typed Latents and Causal Enclosure

The TopoEncoder keeps the split-brain idea but implements it as an atlas. The macro state is a
chart assignment plus a per-chart code index. The residual channels are still typed and explicitly
kept out of macro closure.

:::{prf:definition} The Three-Channel Latent Decomposition
:label: def-three-channel-latent

The internal state at time $t$ decomposes as:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})
$$

where:

1. $K_t = (K_{\mathrm{chart}}, K_{\mathrm{code}})$ is the discrete macro state. $K_{\mathrm{chart}}$
   selects an atlas chart, and $K_{\mathrm{code}}$ selects a local code within that chart.
2. $z_{n,t} \in \mathbb{R}^{d_n}$ is the structured nuisance (pose, basis, gauge residual).
3. $z_{\mathrm{tex},t} \in \mathbb{R}^{d_{\mathrm{tex}}}$ is reconstruction-only texture.

The implementation diagrams below use a common width $d_n=d_{\mathrm{tex}}=D$. If the nuisance or
texture widths differ, insert learned embeddings into the $D$-dimensional decoder input.

The geometry latent used by the decoder is

$$
z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n
$$

where $c_{\mathrm{bar}}$ is the chart center mixture and $z_{q,\mathrm{st}}$ is the straight-through
quantized code.
:::

:::{prf:definition} The Golden Rule of Causal Enclosure
:label: def-causal-enclosure

The macro symbol must satisfy the causal enclosure property:

$$
I\!\left(K_{t+1};(z_{n,t},z_{\mathrm{tex},t})\mid K_t,a_t\right)=0.
$$

This is the joint enclosure condition: once the current macro state and action are given, neither
residual channel carries additional predictive information about the next macro symbol. Predictive
concentration of $P(K_{t+1}\mid K_t,a_t)$ is a separate model-quality diagnostic.
:::

(sec-architecture-the-disentangled-vq-vae-rnn)=
## Architecture: TopoEncoder (Attentive Atlas)

TopoEncoderPrimitives couples a PrimitiveAttentiveAtlasEncoder with a PrimitiveTopologicalDecoder.
The encoder builds chart weights, per-chart VQ assignments, and the typed latents. The decoder uses
chart routing to mix per-chart projectors and produce the reconstruction. Implementation lives in
`src/fragile/core/layers/atlas.py`.

The encoder and decoder diagrams below are **flat reference schematics**. They show tensor shapes
and data flow with ordinary sums and differences so that the wiring is easy to read. Production
routing uses the metric-aware primitives in `atlas.py`: Poincare distances and barycenters,
Möbius transport/addition, logarithm and exponential maps, and projection back to the ball. The
flat expressions are placeholders for those operations, not a claim that the production router is
Euclidean.

### Encoder Block

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph ENC["PrimitiveAttentiveAtlasEncoder"]
        X["Input x [B, D_in]"] --> FE["Feature extractor\nSpectralLinear + NormGatedGELU\n(or CovariantRetina)"]
        FE --> F["features [B, H]"]
        F --> Vproj["val_proj -> v [B, D]"]
        ChartCenters["chart_centers c_k [N_c, D]"] --> RouterEnc["Chart router\nCovariantChartRouter or hyperbolic-distance"]
        F --> RouterEnc
        Vproj --> RouterEnc
        RouterEnc --> Wenc["w_enc [B, N_c]"]
        RouterEnc --> Kchart["K_chart [B]"]

        Wenc --> Cbar["reference c_bar = sum(w_enc * c_k) [B, D]\nproduction: Poincare barycenter"]
        ChartCenters --> Cbar
        Vproj --> Vlocal["reference v_local = v - c_bar [B, D]\nproduction: Mobius subtraction"]
        Cbar --> Vlocal

        Codebook["Codebook (deltas) [N_c, K, D]"] --> Diff["diff = v_local - codebook [B, N_c, K, D]"]
        Vlocal --> Diff
        Diff --> SoftEq["SoftEquivariantLayer per chart\n(optional)"]
        SoftEq --> Dist["dist = ||diff'||^2 [B, N_c, K]"]
        Diff -.-> Dist
        Dist --> Indices["indices per chart [B, N_c]"]
        Indices --> ZqAll["z_q_all [B, N_c, D]\n(+ soft-ST if soft_equiv_soft_assign)"]
        ZqAll --> ZqBlend["z_q_blended = sum(w_enc * z_q_all)"]

        ZqAll --> VQLoss["vq_loss = codebook + 0.25 * commitment"]
        Vlocal --> VQLoss

        ZqAll --> DeltaAll["delta_all = v_local - z_q_all (detach)"]
        DeltaAll --> Struct["structure_filter\nIsotropicBlock + SpectralLinear"]
        Struct --> ZnAll["z_n_all_charts [B, N_c, D]"]
        ZnAll --> Zn["z_n = sum(w_enc * z_n_all_charts) [B, D]"]
        ZqBlend --> DeltaBlend["delta_blended = v_local - z_q_blended (detach)"]
        DeltaBlend --> Ztex["z_tex = delta_blended - z_n"]

        ZqBlend --> ZqSt["z_q_st = v_local + (z_q_blended - v_local).detach"]
        ZqSt --> Zgeo["reference z_geo = c_bar + z_q_st + z_n\nproduction: Mobius addition + projection"]
        Zn --> Zgeo
        Cbar --> Zgeo
    end
```

### Covariant Chart Router

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph ROUTER["CovariantChartRouter (shared by encoder + decoder)"]
        Z["z [B, D]"] --> Dist["d_Poincare(z,c_k) [B, N_c]"]
        ChartTokens["chart centers c_k [N_c, D]"] --> Dist
        Dist --> Scores["scores = -d_Poincare(z,c_k) / tau(z)"]
        Z --> Tau["tau(z) = sqrt(K) * (1 - ||z||^2)/2\nclamp denom + tau_min"]
        Scores --> Scale["scores / tau"]
        Tau --> Scale
        Scale --> W["w = softmax(scores/tau) [B, N_c]"]
        W --> Kchart["K_chart [B]"]

        F["features [B, H]\n(encoder only)"] --> Corr["optional encoder correction\n0.1 <P_0->z(base_queries), q_z(z)+q_feat(f)+gamma(z)> / tau(z)"]
        Z --> Corr
        Corr --> Scale
    end
```

### Decoder Block

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph DEC["PrimitiveTopologicalDecoder (production metric routing)"]
        Zgeo["z_geo [B, D]"] --> BallG["_project_to_ball(z_geo)"]
        BallG --> RouterDec["Chart router\n_poincare_hyperbolic_score"]
        RouterDec --> Wdec["w_dec [B, N_c]"]
        ChartIdx["chart_index (optional)"] --> OneHot["one-hot -> w_hard"]
        OneHot --> Wdec

        BallG --> ChartProj["chart_projectors: SpectralLinear x N_c"]
        ChartProj --> Gate["NormGatedGELU on h_stack"]
        Gate --> Mix["h_global = sum(w_dec * h_stack)"]
        Wdec --> Mix

        Mix --> Renderer["renderer: SpectralLinear + NormGatedGELU x2 + SpectralLinear"]
        Mix --> Skip["render_skip: SpectralLinear"]
        Renderer --> AddSkip["x_hat_base = renderer + skip"]
        Skip --> AddSkip

        Ztex["z_tex [B, D]"] --> TexRes["tex_residual: SpectralLinear"]
        TexRes --> AddTex["x_hat = x_hat_base + tex_residual_scale * tex_residual"]
        AddSkip --> AddTex
        AddTex --> Xhat["x_hat [B, D_out]"]
    end
```

### Training Wiring (Optional Losses)

```{mermaid}
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    X["batch_X"] --> Enc["TopoEncoderPrimitives.encoder"]
    Enc --> Dec["TopoEncoderPrimitives.decoder"]
    Dec --> ReconLoss["recon_loss"]
    Enc --> VQLoss["vq_loss"]

    ReconLoss --> ReconTerm["recon_term\n(optional learned precision)"]
    VQLoss --> VQTerm["vq_term\n(optional learned precision)"]

    Enc --> Sup["SupervisedTopologyLoss (optional)"]
    Sup --> SupTerm["sup_term\n(optional learned precision)"]

    Enc --> Jump["FactorizedJumpOperator (optional)"]
    Jump --> LossA["atlas loss\n(recon + vq + regs + jump + sup)"]

    ReconTerm --> LossA
    VQTerm --> LossA
    SupTerm --> LossA
```

Implementation toggles wired in `src/experiments/topoencoder_2d.py`:
- `covariant_attn` and friends control routing tensorization, transport, and temperature.
- `vision_preproc` swaps the SpectralLinear + NormGatedGELU stack for CovariantRetina in the encoder.
- `soft_equiv_metric`, `soft_equiv_soft_assign`, and `soft_equiv_temperature` control the per-chart
  metric and soft straight-through assignments.

(sec-loss-function-enforcing-macro-micro-separation)=
## Loss Function: Enforcing Macro/Micro Separation

The TopoEncoder is trained with a compound objective. The core terms enforce reconstruction,
quantization, and routing alignment; the tiered regularizers stabilize geometry and codebook
health. Optional supervised topology and jump losses are added when enabled.

:::{prf:definition} The Total TopoEncoder Loss
:label: def-total-disentangled-loss

The compound loss is:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \lambda_{\text{ent}}\,\mathcal{L}_{\text{entropy}} +
\lambda_{\text{cons}}\,\mathcal{L}_{\text{consistency}} +
\sum_{i \in \text{tiers}} \lambda_i \mathcal{L}_i +
\lambda_{\text{jump}}\,\mathcal{L}_{\text{jump}} +
\lambda_{\text{sup}}\,\mathcal{L}_{\text{sup}}.
$$

Where:

- $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$ (MSE reconstruction).
- $\mathcal{L}_{\text{vq}}$ is the codebook + commitment loss.
- $\mathcal{L}_{\text{entropy}}=\log N_c-\frac1B\sum_bH(w_b)$ is an entropy-raising anti-collapse
  regularizer; chart usage/diversity terms prevent dead charts.
- $\mathcal{L}_{\text{consistency}}$ aligns encoder and decoder routing.
- Tiered losses include variance, diversity, separation, codebook centering, chart center
  separation, residual scale, window, disentangle, orthogonality, code entropy, per-chart code
  entropy, KL prior, orbit, and VICReg invariance.
- $\mathcal{L}_{\text{jump}}$ enforces chart transition consistency when the jump operator is
  enabled.
- $\mathcal{L}_{\text{sup}}$ applies supervised topology when labels are available.

Learned precisions can reweight reconstruction, VQ, and supervised terms when enabled.
:::

For exact definitions of each tiered loss, see {ref}`sec-appendix-f-loss-terms-reference`.

(sec-the-complete-training-loop)=
## The Complete Training Loop

The training loop in `src/experiments/topoencoder_2d.py` follows a fixed sequence:

1. Encode the batch to obtain $K_{\mathrm{chart}}$, routing weights, $z_n$, $z_{\mathrm{tex}}$,
   $z_{\mathrm{geo}}$, and VQ loss.
2. Decode in dreaming mode (no hard chart index) to obtain the reconstruction and decoder routing.
3. Compute reconstruction, VQ, routing entropy, and encoder-decoder consistency.
4. Add tiered regularizers and optional jump and supervised losses.
5. Apply learned precision reweighting if enabled, then backprop, clip gradients, and step.

Classifier readouts (if enabled) are trained on detached latents with their own optimizer.

(sec-runtime-diagnostics-the-closure-ratio)=
## Runtime Diagnostics: Routing Sharpness

Routing sharpness is monitored through the conditional entropy of the soft router weights. It is
distinct from the causal closure ratio, which compares transition-model cross-entropy with a
marginal baseline ({prf:ref}`def-f-closure-ratio`).

:::{prf:definition} Routing Sharpness
:label: def-routing-sharpness

Let $K$ be the chart assignment and $N_c$ the number of charts. Define

$$
\rho_{\text{route}} = 1 - \frac{H(K \mid X)}{\log N_c}.
$$

Values near 1 indicate deterministic soft routing and values near 0 indicate diffuse routing. The
quantity equals $I(X;K)/\log N_c$ only when the marginal chart usage is uniform.
:::

Additional diagnostics used in the TopoEncoder benchmark include:
- `window_loss` / $I(X;K)$ (stable learning window)
- routing entropy and perplexity
- per-chart code entropy
- chart center separation
- jump consistency loss (when enabled)

(sec-advanced-hierarchical-multi-scale-latents)=
## Advanced: Hierarchical Multi-Scale Latents

Atlas models extend naturally to multiple scales by stacking charted codebooks.

:::{prf:definition} Hierarchical Latent Stack
:label: def-hierarchical-latent

A multi-scale atlas uses a hierarchy of discrete chart codes:

$$
Z_t = (K_t^{(0)}, K_t^{(1)}, \ldots, K_t^{(L)}, z_{n,t}, z_{\mathrm{tex},t})
$$

where each level $\ell$ has its own chart set and codebook, and higher levels capture coarser
structure.
:::

In practice this can be implemented by stacking TopoEncoder blocks or by sharing a base encoder
with multiple chart routers and codebooks.

(sec-literature-connections)=
## Literature Connections (Mapping + Differences)

- Atlas models and mixture-of-experts: chart routing implements a learned partition of unity.
- VQ-VAE: per-chart VQ keeps discrete macro structure but with charted codebooks.
- Geometric deep learning: routing temperature encodes conformal metric information.
- Structured residuals: explicit nuisance and texture channels mirror disentanglement objectives.

## Computational Costs

Let $K_{\mathrm{key}}$ be the router key width, $K_{\mathrm{code}}$ the number of codes per
chart, and $H$ the hidden width of a soft-equivariant block. With a dense tensorized router, the
leading costs are:

- Routing: $O\!\left(B(N_cD+K_{\mathrm{key}}D^2)\right)$, plus the cost of the selected metric
  transport.
- Codebook distances: $O(BN_cK_{\mathrm{code}}D)$ for per-chart VQ.
- Soft-equivariant metric: $O(BN_cK_{\mathrm{code}}DH)$ for a block with hidden width $H$.

## Control Theory Translation: Dictionary

| Control concept | TopoEncoder component |
| --- | --- |
| Discrete state | $K_{\mathrm{chart}}, K_{\mathrm{code}}$ |
| Local coordinates | $z_n$ |
| Emission residual | $z_{\mathrm{tex}}$ |
| Transition map | Jump operator |
| Partition of unity | Router weights $w$ |

(sec-differential-geometry-view-curvature-as-conditioning)=
## Differential-Geometry View (No Physics): Curvature as Conditioning

Charts are local coordinate systems; chart centers define anchor points, and routing weights define
smooth transitions between charts. The metric-aware temperature in routing behaves like local
conditioning, sharpening attention where the conformal factor
$\lambda(z)=2/(1-\lVert z\rVert^2)$ is large near the ball boundary, where hyperbolic distances are
stretched.

(sec-the-entropy-regularized-objective-functional)=
## The Entropy-Regularized Objective Functional

Routing entropy and policy entropy play parallel roles: both penalize collapse and stabilize
exploration. In the representation stack, entropy regularizes chart usage and prevents dead charts.

## Atlas-Manifold Dictionary: From Topology to Neural Networks

### Core Correspondences

| Geometry | Network |
| --- | --- |
| Chart | Router weight + chart center |
| Atlas overlap | Soft routing across charts |
| Transition map | Jump operator |
| Partition of unity | Softmax router weights |

### When to Use Atlas Architecture

Use charted latents when the representation requires multiple local linearizations or when a single
codebook collapses under diverse modes.
