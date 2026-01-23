# Single Agent Architecture as Field Theory: Technical TLDR

Guillem Duran Ballester, Jan 2026

## 0. Architecture

The following diagrams illustrate the current implementation architecture of the TopoEncoder system, showing how observations are encoded into the split-latent space $(K, z_n, z_{tex})$ and decoded back to reconstructions. These diagrams mirror the code in `src/fragile/core/layers/atlas.py` and `src/experiments/topoencoder_2d.py`.

### 0.1 CovariantChartRouter

The chart router is shared by both encoder and decoder. It performs metric-aware, covariant chart assignment using:
- Geodesic query terms (linear + quadratic Christoffel correction)
- Wilson-line transport via Cayley transform for gauge invariance
- Position-dependent temperature scaled by local metric (conformal factor)

```mermaid
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph ROUTER["CovariantChartRouter (shared by encoder + decoder)"]
        Z["z [B, D]"] -- "z [B, D]" --> Qz["q_z_proj(z) [B, K]"]
        F["features [B, H]\n(encoder only)"] -- "features [B, H]" --> Qfeat["q_feat_proj(features) [B, K]"]
        Z -- "z [B, D]" --> Gamma["Christoffel term (z ⊗ z)\n-> gamma [B, K]"]
        Qz -- "q_z [B, K]" --> Qsum["q = q_z + gamma (+ q_feat) [B, K]"]
        Qfeat -- "q_feat [B, K]" --> Qsum
        Gamma -- "gamma [B, K]" --> Qsum

        Z -- "z [B, D]" --> Transport["transport_proj(z) -> skew [B, K, K]\n(if use_transport)"]
        Transport -- "skew [B, K, K]" --> Cayley["Cayley: U(z) = (I+0.5S)^-1 (I-0.5S)"]
        ChartTokens["chart_tokens c_k [N_c, D or K]\n(encoder: chart_centers)"] -- "c_k [N_c, D]" --> KeyProj["chart_key_proj [N_c, K]"]
        ChartTokens -.->|if K| KeyMerge
        ChartQ["chart_queries [N_c, K]\n(decoder default)"] -- "chart_queries [N_c, K]" --> KeyMerge["base_queries [N_c, K]"]
        KeyProj -- "projected [N_c, K]" --> KeyMerge
        KeyMerge -- "base_queries [N_c, K]" --> Keys["keys = U(z) * base_queries [B, N_c, K]\n(or base_queries if transport disabled)"]
        Cayley -- "U(z) [B, K, K]" --> Keys

        Keys -- "keys [B, N_c, K]" --> Scores["scores = sum(keys * q) [B, N_c]"]
        Z -- "z [B, D]" --> Tau["tau(z) = sqrt(K) * (1 - ||z||^2)/2\nclamp denom + tau_min"]
        Scores -- "scores [B, N_c]" --> Scale["scores / tau"]
        Tau -- "tau [B]" --> Scale
        Scale -- "scores/tau [B, N_c]" --> W["w = softmax(scores/tau) [B, N_c]"]
        W -- "argmax [B]" --> Kchart["K_chart [B]"]
    end

    subgraph TENS["Christoffel tensorization options"]
        Full["full: gamma = einsum(z_i z_j, W_q_gamma[k,i,j])"]
        Sum["sum: low-rank (U_k x V_k) with rank R"]
    end

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#e5e7eb;
    classDef feat fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#e5e7eb;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#e5e7eb;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#e5e7eb;
    classDef util fill:#262626,stroke:#a3a3a3,stroke-width:1px,color:#e5e7eb;

    class Z,Kchart geom;
    class F,Qfeat feat;
    class Qz,Gamma,Qsum,Transport,Cayley,ChartTokens,KeyProj,ChartQ,KeyMerge,Keys,Scores,Tau,Scale,W router;
    class Full,Sum util;
```

### 0.2 Full TopoEncoder (Encoder + Decoder)

Complete end-to-end architecture showing:
- **Encoder**: Feature extraction → Chart routing → VQ per chart → Split into $(z_{geo}, z_n, z_{tex})$
- **Decoder**: Chart-weighted reconstruction → Geometric base + Texture residual

```mermaid
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph TOP["TopoEncoderPrimitives (current code)"]
        subgraph ENC["PrimitiveAttentiveAtlasEncoder"]
            X["Input x [B, D_in]"] -- "x [B, D_in]" --> FE["Feature extractor\nMLP: SpectralLinear -> NormGatedGELU x2\nor CovariantRetina (vision_preproc)"]
            FE -- "features [B, H]" --> F["features [B, H]"]
            F -- "features [B, H]" --> Vproj["val_proj: SpectralLinear\nv [B, D]"]
            ChartCenters["chart_centers c_k [N_c, D]"] -- "c_k [N_c, D]" --> RouterEnc["Chart router\nCovariantChartRouter (covariant_attn)\nelse dot-product w/ chart_centers"]
            F -- "features [B, H]" --> RouterEnc
            Vproj -- "z = v [B, D]" --> RouterEnc
            RouterEnc -- "w_enc [B, N_c]" --> Wenc["w_enc [B, N_c]"]
            RouterEnc -- "K_chart [B]" --> Kchart["K_chart [B]"]

            Wenc -- "w_enc [B, N_c]" --> Cbar["c_bar = sum(w_enc * c_k) [B, D]"]
            ChartCenters -- "c_k [N_c, D]" --> Cbar
            Vproj -- "v [B, D]" --> Vlocal["v_local = v - c_bar [B, D]"]
            Cbar -- "c_bar [B, D]" --> Vlocal

            Codebook["Codebook (deltas) [N_c, K, D]"] -- "codebook [N_c, K, D]" --> Diff["diff = v_local - codebook [B, N_c, K, D]"]
            Vlocal -- "v_local [B, D]" --> Diff
            Diff -- "diff [B, N_c, K, D]" --> SoftEq["SoftEquivariantLayer per chart\n(optional when soft_equiv_metric)"]
            SoftEq -- "diff' [B, N_c, K, D]" --> Dist["dist = ||diff'||^2 [B, N_c, K]"]
            Diff -.-> Dist
            Dist -- "dist [B, N_c, K]" --> Indices["indices per chart [B, N_c]"]
            Indices -- "indices [B, N_c]" --> ZqAll["z_q_all [B, N_c, D]\n(gather; + soft-ST if soft_equiv_soft_assign)"]
            ZqAll -- "z_q_all [B, N_c, D]" --> ZqBlend["z_q_blended = sum(w_enc * z_q_all)"]

            Indices -- "indices [B, N_c]" --> Kcode["K_code (from K_chart)"]
            Kchart -- "K_chart [B]" --> Kcode

            ZqAll -- "z_q_all [B, N_c, D]" --> VQLoss["vq_loss = codebook + 0.25 * commitment"]
            Vlocal -- "v_local [B, D]" --> VQLoss

            ZqAll -- "z_q_all [B, N_c, D]" --> DeltaAll["delta_all = v_local - z_q_all (detach)"]
            DeltaAll -- "delta_all [B, N_c, D]" --> Struct["structure_filter\nIsotropicBlock + SpectralLinear"]
            Struct -- "z_n_all_charts [B, N_c, D]" --> ZnAll["z_n_all_charts [B, N_c, D]"]
            ZnAll -- "z_n_all_charts [B, N_c, D]" --> Zn["z_n = sum(w_enc * z_n_all_charts) [B, D]"]
            ZqBlend -- "z_q_blended [B, D]" --> DeltaBlend["delta_blended = v_local - z_q_blended (detach)"]
            DeltaBlend -- "delta_blended [B, D]" --> Ztex["z_tex = delta_blended - z_n"]

            ZqBlend -- "z_q_blended [B, D]" --> ZqSt["z_q_st = v_local + (z_q_blended - v_local).detach"]
            ZqSt -- "z_q_st [B, D]" --> Zgeo["z_geo = c_bar + z_q_st + z_n"]
            Zn -- "z_n [B, D]" --> Zgeo
            Cbar -- "c_bar [B, D]" --> Zgeo

            ZnAll -- "z_n_all_charts [B, N_c, D]" --> Jump["FactorizedJumpOperator (optional)"]
        end

        subgraph DEC["PrimitiveTopologicalDecoder"]
            Zgeo -- "z_geo [B, D]" --> TanhG["tanh(z_geo)"]
            TanhG -- "tanh(z_geo) [B, D]" --> RouterDec["Chart router\nCovariantChartRouter (covariant_attn)\nelse latent_router + softmax"]
            RouterDec -- "w_dec [B, N_c]" --> Wdec["w_dec [B, N_c]"]
            ChartIdx["chart_index (optional)"] -- "K_chart [B]" --> OneHot["one-hot -> w_hard"]
            OneHot -- "w_dec_hard [B, N_c]" --> Wdec

            TanhG -- "tanh(z_geo) [B, D]" --> ChartProj["chart_projectors: SpectralLinear x N_c"]
            ChartProj -- "h_i [B, N_c, H]" --> Gate["NormGatedGELU on h_stack"]
            Gate -- "h_stack [B, N_c, H]" --> Mix["h_global = sum(w_dec * h_stack)"]
            Wdec -- "w_dec [B, N_c]" --> Mix

            Mix -- "h_global [B, H]" --> Renderer["renderer: SpectralLinear + NormGatedGELU x2 + SpectralLinear"]
            Mix -- "h_global [B, H]" --> Skip["render_skip: SpectralLinear"]
            Renderer -- "h_render [B, D_out]" --> AddSkip["x_hat_base = renderer + skip"]
            Skip -- "h_skip [B, D_out]" --> AddSkip

            Ztex -- "z_tex [B, D]" --> TanhT["tanh(z_tex)"]
            TanhT -- "tanh(z_tex) [B, D]" --> TexRes["tex_residual: SpectralLinear"]
            TexRes -- "tex_resid [B, D_out]" --> AddTex["x_hat = x_hat_base + tex_residual_scale * tex_residual"]
            AddSkip -- "x_hat_base [B, D_out]" --> AddTex
            AddTex -- "x_hat [B, D_out]" --> Xhat["x_hat [B, D_out]"]
        end
    end

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#e5e7eb;
    classDef feat fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#e5e7eb;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#e5e7eb;
    classDef vq fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#e5e7eb;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#e5e7eb;
    classDef residual fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#e5e7eb;
    classDef decoder fill:#1f2b3b,stroke:#60a5fa,stroke-width:1px,color:#e5e7eb;
    classDef util fill:#262626,stroke:#a3a3a3,stroke-width:1px,color:#e5e7eb;

    class X,Xhat,ChartIdx,Kchart io;
    class FE,F,Vproj,Struct feat;
    class RouterEnc,RouterDec,Wenc,Wdec,OneHot router;
    classs Codebook,Diff,SoftEq,Dist,Indices,ZqAll,ZqBlend,VQLoss,Kcode vq;
    class ChartCenters,Cbar,Vlocal,Zgeo,ZqSt,ZnAll,Zn geom;
    class DeltaAll,DeltaBlend,Ztex,TanhT,TexRes residual;
    class TanhG,ChartProj,Gate,Mix,Renderer,Skip,AddSkip,AddTex decoder;
    class Jump util;
```

### 0.3 Decoder Detail (Inverse Atlas)

Focused view of the decoder showing how the geometric latent $z_{geo}$ and texture residual $z_{tex}$ are independently processed and combined:
- Geometric path: Chart projectors → Chart-weighted mixing → Renderer
- Texture path: Independent residual network
- Final output: Base reconstruction + scaled texture residual

```mermaid
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    subgraph DEC["PrimitiveTopologicalDecoder (current code)"]
        Zgeo["z_geo = c_bar + z_q_st + z_n [B, D]"] -- "z_geo [B, D]" --> TanhG["tanh(z_geo)"]
        TanhG -- "tanh(z_geo) [B, D]" --> RouterDec["Chart router\nCovariantChartRouter (covariant_attn)\nelse latent_router + softmax"]
        RouterDec -- "w_dec [B, N_c]" --> Wdec["w_dec [B, N_c]"]
        ChartIdx["chart_index (optional)"] -- "K_chart [B]" --> OneHot["one-hot -> w_hard"]
        OneHot -- "w_dec_hard [B, N_c]" --> Wdec

        TanhG -- "tanh(z_geo) [B, D]" --> ChartProj["chart_projectors: SpectralLinear x N_c"]
        ChartProj -- "h_i [B, N_c, H]" --> Gate["NormGatedGELU on h_stack"]
        Gate -- "h_stack [B, N_c, H]" --> Mix["h_global = sum(w_dec * h_stack)"]
        Wdec -- "w_dec [B, N_c]" --> Mix

        Mix -- "h_global [B, H]" --> Renderer["renderer: SpectralLinear + NormGatedGELU x2 + SpectralLinear"]
        Mix -- "h_global [B, H]" --> Skip["render_skip: SpectralLinear"]
        Renderer -- "h_render [B, D_out]" --> AddSkip["x_hat_base = renderer + skip"]
        Skip -- "h_skip [B, D_out]" --> AddSkip

        Ztex["z_tex [B, D]"] -- "z_tex [B, D]" --> TanhT["tanh(z_tex)"]
        TanhT -- "tanh(z_tex) [B, D]" --> TexRes["tex_residual: SpectralLinear"]
        TexRes -- "tex_resid [B, D_out]" --> AddTex["x_hat = x_hat_base + tex_residual_scale * tex_residual"]
        AddSkip -- "x_hat_base [B, D_out]" --> AddTex
        AddTex -- "x_hat [B, D_out]" --> Xhat["x_hat [B, D_out]"]
    end

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#e5e7eb;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#e5e7eb;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#e5e7eb;
    classDef residual fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#e5e7eb;
    classDef decoder fill:#1f2b3b,stroke:#60a5fa,stroke-width:1px,color:#e5e7eb;

    class Zgeo geom;
    class RouterDec,Wdec,OneHot router;
    class ChartIdx,Xhat io;
    class TanhG,ChartProj,Gate,Mix,Renderer,Skip,AddSkip,AddTex decoder;
    class Ztex,TanhT,TexRes residual;
```

### 0.4 Experiment Wiring (Training Losses)

Optional training components available in the experiment configuration:
- Learned precisions for automatic loss balancing (reconstruction, VQ, supervised)
- SupervisedTopologyLoss for semantic topology learning
- FactorizedJumpOperator for chart transition consistency
- InvariantChartClassifier (detached readout head with separate optimizer)

```mermaid
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    X["batch_X"] --> Enc["TopoEncoderPrimitives.encoder"]
    Enc -- "z_geo [B, D]" --> Dec["TopoEncoderPrimitives.decoder"]
    Enc -- "z_tex [B, D]" --> Dec
    Dec -- "recon_a [B, D_in]" --> ReconLoss["recon_loss = mse(recon_a, batch_X)"]
    X --> ReconLoss

    Enc -- "vq_loss" --> VQLoss["vq_loss (codebook + commitment)"]

    ReconLoss --> ReconTerm["recon_term\n(optional learned precision)"]
    VQLoss --> VQTerm["vq_term\n(optional learned precision)"]

    Enc -- "enc_w [B, N_c]" --> Sup["SupervisedTopologyLoss (optional)"]
    Enc -- "z_geo [B, D]" --> Sup
    Y["batch_labels [B]"] --> Sup
    Sup -- "sup_total + components" --> SupTerm["sup_term\n(optional learned precision)"]

    Enc -- "z_n_all_charts [B, N_c, D]" --> Jump["FactorizedJumpOperator (optional)"]
    Enc -- "enc_w [B, N_c]" --> Jump
    Jump -- "jump_loss (schedule weight)" --> LossA["atlas loss\n(recon + vq + regs + jump + sup)"]

    ReconTerm --> LossA
    VQTerm --> LossA
    SupTerm -- "sup_weight * sup_term" --> LossA

    Enc -- "enc_w (detach)" --> Cls["InvariantChartClassifier (optional)"]
    Enc -- "z_geo (detach)" --> Cls
    Y --> CE["cross_entropy"]
    Cls -- "logits [B, C]" --> CE
    CE --> OptCls["opt_classifier.step()"]
```

## 1. Latent Space Decomposition

**Split-Latent Structure:**

- Total latent: $Z = (K, z_n, z_{\text{tex}})$ where each component has distinct geometric/physical role
- $K \in \{1,\ldots,N_c\}$: Discrete macro state (VQ codebook index) - chart assignment on topological atlas
- $z_n \in \mathbb{R}^{D_n}$: Continuous nuisance latent - local coordinates within chart (gauge-invariant position)
- $z_{\text{tex}} \in \mathbb{R}^{D_t}$: Texture residual - holographic boundary degrees of freedom
- Decomposition satisfies $z_e = z_q + z_n + z_{\text{tex}}$ where $z_e$ is raw encoder output, $z_q$ is VQ-quantized macro

**Encoder Architecture (AttentiveAtlasEncoder):**

- Feature extraction: Conv layers (64→128→256 channels) → hidden_dim projection
- Cross-attention routing: $w_k = \text{softmax}(\langle K_k, Q(x) \rangle / \sqrt{D})$ where $K_k$ are learnable chart queries
- Query projection: $Q(x) = \text{key\_proj}(\text{features}(x))$ with LayerNorm stabilization
- Chart assignment: $K = \arg\max_k w_k(x)$
- VQ per chart: $N_c$ independent codebooks, each with $K_c$ codes, vectorized quantization
- Nuisance extraction: Structure filter $z_n = f_{\text{struct}}(z_e - z_q)$ removes VQ-residual structure
- Texture: Holographic residual $z_{\text{tex}} = (z_e - z_q) - z_n$ orthogonal to both macro and nuisance.
- **Texture Firewall:** Dynamics depend on $z_{\text{macro}}$ and $z_n$, screening out only $z_{\text{tex}}$.

**Training Objectives for Coarse-Graining:**
To enforce this split-latent structure and ensure valid coarse-graining, we minimize:
$$ \mathcal{L}_{\text{latent}} = \mathcal{L}_{\text{VQ}} + \mathcal{L}_{\text{closure}} + \mathcal{L}_{\text{slowness}} + \mathcal{L}_{\text{disentangle}} $$

1.  **VQ Loss:** $\|z_e - z_q\|^2 + \beta \|z_q - \operatorname{sg}[z_e]\|^2$. Stabilizes the discrete macro state $K$.
2.  **Causal Enclosure:** $-\log p(K_{t+1} | K_t, a_t)$. Ensures macro dynamics are self-contained (predictable without micro details).
3.  **Slowness (Anti-Churn):** $\|e_{K_t} - e_{K_{t-1}}\|_G^2$. Penalizes rapid flickering of the macro state to ensure temporal stability.
4.  **Disentanglement:**
    *   *Nuisance KL:* $D_{\mathrm{KL}}(q(z_n|x) \| \mathcal{N}(0,I))$. Minimal sufficient nuisance.
    *   *Texture KL:* $D_{\mathrm{KL}}(q(z_{\text{tex}}|x) \| \mathcal{N}(0,I))$. Texture should contain no macro info.

**Geometric Regularization (Quality Control):**
To ensure the latent space is well-conditioned (high-fidelity, isometric charts), we add:
$$ \mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{VICReg}} + \mathcal{L}_{\text{ortho}} $$

5.  **VICReg (Self-Supervision):** Prevents collapse without negative pairs.

    *   *Invariance:* $\|z - z'\|_G^2$. Robustness to view augmentation.
    *   *Variance:* $\max(0, \gamma - \sqrt{\text{Var}(z)})$. Forces code utilization.
    *   *Covariance:* $C(z) \approx I$. Decorrelates latent dimensions (whitening).
6.  **Orthogonality (Chart Isometry):** $\|W^T W - I\|_F^2$ on encoder weights. Ensures the mapping from observation to latent space is locally isometric (preserves distances), crucial for meaningful geodesic calculations.

## 2. The Reward Field & Hodge Decomposition

**Physical Context:** We model the agent as a particle with **Position and Momentum** performing a **Geodesic Random Walk** on the latent manifold. Because utility is harvested via trajectory traversal, the Reward is naturally defined as a differential **1-form** coupled to velocity, rather than a static scalar field.

**Constraint:** Reward is not a scalar $r(z)$, but a **1-form** $\mathcal{R}$ that depends on direction ($\mathcal{R}_i \dot{z}^i$). This requires a field-theoretic treatment of value.

**Hodge Decomposition:** The reward 1-form uniquely decomposes into three orthogonal components:
$$ \mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta\Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}} $$

-   **Gradient ($\Phi$):** Optimizable scalar potential (Conservative Value).
-   **Solenoidal ($\Psi$):** Vector potential generating the **Value Curl** (Magnetic Field) $\mathcal{F} = d\mathcal{R} = d\delta\Psi$.
    -   *Physics:* Just as a magnetic field $B$ exerts a **Lorentz Force** $v \times B$, the Value Curl $\mathcal{F}$ exerts a velocity-dependent force $\mathcal{F}_{ij}\dot{z}^j$ that steers the agent to **orbit** value cycles rather than converge.
-   **Harmonic ($\eta$):** Topological cycles (manifold holes).

**The Screened Poisson Equation (Conservative Case):**
When the **Value Curl** vanishes ($\mathcal{F} = 0$), the scalar value function $V(z)$ satisfies the **Helmholtz Equation** on the manifold:
$$ (-\Delta_G + \kappa^2)V(z) = \rho_r(z) $$
where $\Delta_G$ is the Laplace-Beltrami operator, $\kappa$ is the screening mass, and $\rho_r$ is the reward source density.

**The Composite Navigation Potential (Runtime):**
The agent navigates a **Composite Potential** $\Phi_{\text{eff}}$ constructed at runtime by summing learned and intrinsic signals:
$$ \Phi_{\text{eff}}(z) = \underbrace{V(z)}_{\text{Learned}} + \underbrace{U(z)}_{\text{Intrinsic}} + \underbrace{\mathcal{B}_{\text{safety}}(z)}_{\text{Fixed}} $$

1.  **Control ($V$):** **Learned** scalar value. Drives the agent towards high-reward regions.
    *   *Loss:* Helmholtz Residual (see below).
2.  **Generation ($U$):** **Intrinsic** entropy potential (e.g., Hyperbolic expansion $-\log \text{vol}(z)$). Drives exploration away from the origin.
    *   *Loss:* None (Fixed geometric prior).
3.  **Safety ($\mathcal{B}_{\text{safety}}$):** **Fixed** safety barrier. Hard constraints (e.g., capacity limits) modeled as high-energy walls.
    *   *Loss:* None (Fixed constraint).

**Neural Hodge Decomposition (Implementation):**
We approximate the Hodge components using a **Multi-Head Network** sharing a common feature backbone:

1.  **Scalar Head ($\Phi$):** Outputs scalar $V(z)$.
    *   *Loss:* Helmholtz Residual on the symmetric part of the reward.
    *   $\mathcal{L}_{\Phi} = \|V(z) - (r_{\text{sym}} + \gamma V(z'))\|^2$.
2.  **Solenoidal Head ($\Psi$):** Outputs vector potential $A(z) \in \mathbb{R}^D$ (where $\mathcal{F} = dA$).
    *   *Loss:* Residual reconstruction on the antisymmetric part.
    *   $\mathcal{L}_{\Psi} = \| \langle \mathcal{R}, v \rangle - (\langle \nabla\Phi, v \rangle + \langle \nabla \times A, v \rangle) \|^2$. The curl absorbs the non-integrable reward residual.
3.  **Harmonic Head ($\eta$):** A set of **learnable constant 1-forms** $\eta_k$ per chart $k$.
    *   *Mechanism:* Captures global topological currents (net flux through manifold holes) that are locally constant.
    *   *Loss:* Projected residual after removing Gradient and Curl components.

**Value Function Objectives:**
To define the value field $V(z)$ as the solution to the Screened Poisson Equation, we minimize:
$$ \mathcal{L}_{\text{critic}} = \underbrace{\|V(z) - (r + \gamma V(z'))\|^2}_{\text{Helmholtz Residual (TD)}} + \lambda_{\text{geo}} \underbrace{\|\nabla_G V\|^2}_{\text{Smoothness}} $$

1.  **Helmholtz Residual:** Enforces the PDE source term (approx. Bellman error).
2.  **Geometric Regularization:** Enforces field smoothness with respect to the manifold metric ($\|\nabla_G V\|^2 = G^{ij}\partial_i V \partial_j V$).

## 3. The Policy Network (Latent Action)

The Policy acts as an **External Force Field** $u_\pi(z)$ pushing the agent through the latent manifold. It operates in a **Latent Action Space** (the Tangent Bundle $T\mathcal{Z}$), decoupling low-level motor commands from high-level intent.

**A. The Policy Model ($\pi_\phi$):**

-   **Role:** Symmetry-breaking control field. Converts potential energy into kinetic motion.
-   **Input:** Latent State $z_t \in \mathcal{Z}$ (Position).
-   **Output:** Latent Force/Action $u_t \in T_{z_t}\mathcal{Z}$ (Tangent Vector).
    -   *Note:* This latent force is subsequently decoded into boundary motor torques action $a_t$ by the Motor/Action Decoder (Neumann Condition).
-   **Latent Action Space:** The Tangent Bundle $T\mathcal{Z}$. Actions are vectors "pushing" the state along geodesics.

**B. Training Losses (Tier 1):**

1.  **Task Loss ($\mathcal{L}_{\text{task}}$):** Standard Policy Gradient / Reinforce objective to maximize expected returns.
2.  **Entropy Bonus (Ergodicity):** $\mathcal{L}_{\text{ent}} = -H(\pi)$. Penalizes low entropy distributions to prevent premature mode collapse and ensure thermodynamic equilibrium.
3.  **Zeno Penalty (Temporal Smoothness):** $\mathcal{L}_{\text{zeno}} = D_{\mathrm{KL}}(\pi_t \| \pi_{t-1})$. Penalizes infinite-frequency oscillations (Zeno behavior) to ensure physically realizable trajectories.

## 4. The World Model (Covariant Integrator)

We define the World Model not as a generic RNN, but as a **Neural Integrator** that approximates the **Lorentz-Langevin SDE**:

$$ dz^k = \underbrace{\left( -G^{kj}\partial_j \Phi + u_\pi^k \right) ds}_{\text{Gradient + Policy}} + \underbrace{\beta G^{km} \mathcal{F}_{mj} \dot{z}^j ds}_{\text{Lorentz Force}} - \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j ds}_{\text{Geodesic Drift}} + \underbrace{\sqrt{2T_c} (G^{-1/2})^{jk} dW^j}_{\text{Thermal Noise}} $$

This equation unifies:

1.  **Gradient Descent** (on the Value Landscape)
2.  **Magnetic Steering** (from Value Curl)
3.  **Geodesic Motion** (on the curved Manifold)
4.  **Stochastic Exploration** (Langevin Dynamics)

The integration step is modeled as a **Covariant Cross-Attention** layer (Multi-Head Transformer).

**A. Architecture: Covariant Cross-Attention**

-   **Mechanism:** Attention heads act as **Wilson Lines** (Parallel Transport operators), comparing queries and keys only after transporting them to a common reference frame (Gauge Invariance).
-   **Metric-Temperature:** The softmax temperature is position-dependent: $\tau(z) \propto 1/\lambda(z)$. High-curvature regions (large metric $\lambda$) force low temperature (sharp attention), while flat regions allow high temperature (broad exploration).
-   **Geodesic Correction:** Christoffel symbols are encoded via linear and quadratic terms in the Query projection.

**B. Inputs & Outputs (Integration Step):**

-   **Input:**
    -   Current State $z_t$ (Query position).
    -   Action $u_t$ (Latent Force/Momentum).
    -   Memory Context (Keys/Values from past trajectory).
-   **Output:**
    -   Next State $z_{t+1}$ (Integrated position after Kick-Drift-Kick).

**C. Training Losses:**

1.  **Geodesic Distance Loss:** $\mathcal{L}_{\text{geo}} \approx (z_{\text{pred}} - z_{\text{true}})^T G(z_t) (z_{\text{pred}} - z_{\text{true}})$. Minimizes local Riemannian distance. Using the diagonal approximation (Section 5), this becomes a **Weighted MSE**: $\sum_i G_{ii} (z_{\text{pred}}^{(i)} - z_{\text{true}}^{(i)})^2$. High-risk dimensions (large $G_{ii}$) are penalized more heavily.
2.  **Thermodynamic Consistency:** $\mathcal{L}_{\text{NLL}} = -\log p(z_{t+1} | z_t, u_t)$. Ensures the model captures the stochastic thermal noise term ($\sqrt{2T_c} dW$) correctly.

**D. Structural Inductive Bias (Why it acts as an Integrator):**
We do not simply train a generic MLP to output $z_{t+1}$. Instead, we bake the **Boris-BAOAB** integration scheme directly into the attention mechanism, ensuring the model cannot violate the symplectic structure:

1.  **Metric as Temperature:** The attention temperature $\tau(z) \propto \sqrt{d_k} / \lambda(z)$ forces the update step size to scale inversely with curvature (conformal factor). High-curvature regions (large metric) automatically induce small, cautious steps (sharp attention).
2.  **Geodesic Query Terms:** We explicitly feed geometric terms ($z, z \otimes z$) into the Query projection $Q(z)$. This forces the attention scores to learn the **Christoffel Symbols** $\Gamma^k_{ij}$ needed to correct for manifold curvature, rather than making up arbitrary dynamics.
3.  **Operator Splitting:** We use **multiple attention heads** to implement the split operators of the BAOAB scheme:
    *   *Kick Head (B):* Updates momentum using force (Gradient + Curl).
    *   *Drift Head (A):* Updates position using momentum (Geodesic flow).
    *   *Thermostat Head (O):* Applies friction and noise (OU process).
    *   *Result:* The network is forced to learn a decomposable, reversible integrator rather than a "black box" transition.

## 5. Efficient Metric Computation (Adam-Style)

Computing the full Riemannian metric $G_{ij}$ ($D \times D$ tensor) is expensive ($O(D^3)$ inversion). We use the same engineering tricks as the **Adam Optimizer** to approximate it efficiently ($O(D)$).

**A. The "Adam" Isomorphism:**

-   **Adam Optimizer:** Maintains a diagonal approximation of the Hessian (via squared gradients) to precondition updates.
    -   $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ (Second Moment Estimate).
    -   Preconditioner: $P = \text{diag}(1/\sqrt{v_t})$.
-   **Fragile Agent:** Maintains a diagonal approximation of the Metric Tensor (via Risk Tensor) to curve the space.
    -   $\text{Metric}_t = \beta \text{Metric}_{t-1} + (1-\beta) \text{Risk}_t$ (Metric Evolution).
    -   Geometry: $G_{ij} \approx \text{diag}(\text{Metric}_t)$.

**B. Implementation Details:**

1.  **Diagonal Approximation:** We assume $G_{ij}$ is diagonal (independent curvature per dimension). This reduces storage from $O(D^2)$ to $O(D)$ and inversion to element-wise division.
2.  **Risk as Squared Gradient:** The Risk Tensor $T_{ij}$ is dominated by the gradient of the potential: $T_{ij} \approx \partial_i \Phi \partial_j \Phi$. Its diagonal is simply $(\nabla \Phi)^2$.
3.  **Low-Rank Updates (EMA):** We do not solve the Einstein Field Equations at every step. Instead, we update the metric using an **Exponential Moving Average (EMA)** of the Risk Tensor.
    -   *Update Rule:* `metric_diag.lerp_(risk_diag, 1 - momentum)`
    -   *Interpretation:* The geometry "flows" slowly towards the high-risk regions, smoothing out transient noise just like Adam smooths gradient variance.

**Result:** We get Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) benefits for the cost of standard SGD+Momentum.
