# TopoEncoder Architecture Diagrams (Current Implementation)

This document mirrors the current implementation in:
- `src/fragile/core/layers/atlas.py` (PrimitiveAttentiveAtlasEncoder + PrimitiveTopologicalDecoder)
- `src/experiments/topoencoder_2d.py` (TopoEncoderPrimitives wiring and config)

The diagrams include the covariant chart routing upgrades:
1) Wilson-line transport (Cayley transform of a learned skew matrix)
2) Metric-aware temperature `tau(z)` from conformal factor
3) Geodesic query terms (linear + quadratic in `z`) with **two tensorization options**

Implementation toggles wired in `src/experiments/topoencoder_2d.py`:
- `covariant_attn`: CovariantChartRouter vs dot-product (encoder) / latent_router (decoder).
- `vision_preproc`: CovariantRetina replaces the MLP feature extractor.
- `soft_equiv_metric`: per-chart SoftEquivariantLayer metric on codebook distances (+ optional L1 regularizer).

---

## CovariantChartRouter (Standalone)

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

        Z -- "z [B, D]" --> Transport["transport_proj(z) -> skew [B, K, K]"]
        Transport -- "skew [B, K, K]" --> Cayley["Cayley: U(z) = (I+0.5S)^-1 (I-0.5S)"]
        ChartTokens["chart_tokens c_k [N_c, D or K]\n(encoder uses chart_centers)"] -- "c_k [N_c, D]" --> KeyProj["chart_key_proj [N_c, K]"]
        ChartTokens -.->|if K| KeyMerge
        ChartQ["chart_queries [N_c, K]\n(fallback)"] -- "chart_queries [N_c, K]" --> KeyMerge["base_queries [N_c, K]"]
        KeyProj -- "projected [N_c, K]" --> KeyMerge
        KeyMerge -- "base_queries [N_c, K]" --> Keys["keys = U(z) * base_queries [B, N_c, K]"]
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

---

## Full TopoEncoder (Encoder + Decoder)

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
            Indices -- "indices [B, N_c]" --> ZqAll["z_q_all [B, N_c, D] (gather)"]
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
    class Codebook,Diff,SoftEq,Dist,Indices,ZqAll,ZqBlend,VQLoss,Kcode vq;
    class ChartCenters,Cbar,Vlocal,Zgeo,ZqSt,ZnAll,Zn geom;
    class DeltaAll,DeltaBlend,Ztex,TanhT,TexRes residual;
    class TanhG,ChartProj,Gate,Mix,Renderer,Skip,AddSkip,AddTex decoder;
    class Jump util;
```

---

## Decoder Detail (Inverse Atlas, Router External)

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

---

## Experiment Wiring (Supervised + Classifier Readout)

Supervised topology loss and the invariant classifier readout are optional in
`src/experiments/topoencoder_2d.py`. The classifier head is detached from atlas
gradients and trained with its own optimizer.

```mermaid
%%{init: {"themeVariables": {"background":"#0b111b","edgeLabelBackground":"#111827","textColor":"#e5e7eb","lineColor":"#9ca3af","primaryColor":"#1f2937","primaryTextColor":"#e5e7eb","clusterBkg":"#0f172a","clusterBorder":"#334155"}}}%%
flowchart TD
    X["batch_X"] --> Enc["TopoEncoderPrimitives.encoder"]
    Enc -- "enc_w [B, N_c]" --> Sup["SupervisedTopologyLoss (optional)"]
    Enc -- "z_geo [B, D]" --> Sup
    Y["batch_labels [B]"] --> Sup
    Sup -- "sup_total + components" --> SupTerm["sup_term\n(optional learned precision)"]
    SupTerm -- "sup_weight * sup_term" --> LossA["atlas loss\n(recon + vq + regs + sup)"]

    Enc -- "enc_w (detach)" --> Cls["InvariantChartClassifier (optional)"]
    Enc -- "z_geo (detach)" --> Cls
    Y --> CE["cross_entropy"]
    Cls -- "logits [B, C]" --> CE
    CE --> OptCls["opt_classifier.step()"]
```

Metric-only logging:
- `sup_acc` is computed from `enc_w @ p_y_given_k` and does not backpropagate.
