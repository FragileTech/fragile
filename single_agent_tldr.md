# Single Agent Architecture as Hyperbolic Field Theory (Implementation TL;DR)

Guillem Duran Ballester, Jan 2026

## Scope and implementation anchors

This document specifies the implemented single-agent architecture as a hyperbolic field theory over the Poincaré ball latent space.

Primary implementation files:
- `src/fragile/learning/core/layers/atlas.py`
- `src/fragile/learning/vla/covariant_world_model.py`
- `src/fragile/learning/vla/losses.py`
- `src/fragile/learning/vla/train_joint.py`

---

## 0. Field-theory objects and module correspondence

- **State field**: latent trajectory `z_t` on the Poincaré ball.
- **Chart field**: atlas routing weights and chart assignment.
- **Potential field**: conservative force and scalar potential from `CovariantPotentialNet`.
- **Hodge field**: solenoidal + harmonic decomposition channels.
- **Jump field**: Poisson-like sparse event process.
- **Observable field**: decoder reconstruction from geometric + texture channels.

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart LR
    subgraph FIELDS["Field-Theory Objects"]
        direction TB
        SF["State field z_t<br/>(Poincaré ball trajectory)"]
        CF["Chart field w(z)<br/>(atlas routing weights)"]
        PF["Potential field Phi_eff<br/>(conservative + intrinsic + risk)"]
        HF["Hodge field<br/>(solenoidal + harmonic channels)"]
        JF["Jump field<br/>(Poisson sparse event process)"]
        OF["Observable field<br/>(decoder reconstruction)"]
    end

    subgraph MODULES["Implementation Modules"]
        direction TB
        ENC["PrimitiveAttentiveAtlasEncoder<br/>(atlas.py)"]
        ROUTE["CovariantChartRouter<br/>(atlas.py)"]
        POT["CovariantPotentialNet<br/>(covariant_world_model.py)"]
        HODGE["CovariantValueCurl +<br/>HodgeDecomposer<br/>(covariant_world_model.py)"]
        JUMP["FactorizedJumpOperator +<br/>CovariantJumpRate<br/>(topology.py)"]
        DEC["PrimitiveTopologicalDecoder<br/>(atlas.py)"]
    end

    SF --> ENC
    CF --> ROUTE
    PF --> POT
    HF --> HODGE
    JF --> JUMP
    OF --> DEC

    classDef field fill:#1a1a2e,stroke:#e94560,stroke-width:1px,color:#ffffff;
    classDef module fill:#16213e,stroke:#0f3460,stroke-width:1px,color:#ffffff;

    class SF,CF,PF,HF,JF,OF field;
    class ENC,ROUTE,POT,HODGE,JUMP,DEC module;
```

---

## 1. End-to-end implementation schematic

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph DATA["VLA Sequence Batch"]
        direction TB
        X0["frame t0 [B, D_in]"]
        XH["frames t1...tH [B, H, D_in]"]
        A["actions a0...a(H-1) [B, H, A]"]
    end

    subgraph ATLAS["Hyperbolic Atlas Stack"]
        direction TB

        subgraph ENCBLOCK["Encoder + Router"]
            direction TB
            ENC["PrimitiveAttentiveAtlasEncoder"]
            ROUTER["CovariantChartRouter<br/>(shared encoder/decoder)"]
        end

        subgraph LATENT["Encoder Outputs"]
            direction TB
            ZGEO["z_geo [B, D] (ball)"]
            ZN["z_n [B, D] (tangent)"]
            ZTEX["z_tex [B, D] (tangent)"]
            WENC["w_enc [B, N_c]"]
            VQLOSS["vq_loss, indices, K_code"]
            ZNALL["z_n_all_charts [B, N_c, D]"]
        end

        subgraph DECBLOCK["Decoder"]
            direction TB
            DEC["PrimitiveTopologicalDecoder"]
            RECON["recon [B, D_out]"]
        end
    end

    subgraph WM["GeometricWorldModel"]
        direction TB
        ROLLOUT["BAOAB rollout in<br/>Poincare ball latent space"]

        subgraph WMOUT["World Model Outputs"]
            direction TB
            ZPRED["z_trajectory [B, H, D]"]
            CHLOG["chart_logits [B, H, N_c]"]
            MOM["momenta [B, H, D]"]
            PHI["Phi_eff [B, H, 1]"]
            JRATES["jump_rates [B, H, 1]"]
            HODGER["hodge ratios + harmonic forces"]
        end
    end

    subgraph LOSS["Loss Assembly"]
        direction TB
        BL["base_loss<br/>(recon+vq+entropy+consistency<br/>+diversity+cb_spread+cb_center<br/>+chart_collapse+code_collapse+window)"]
        ZNR["zn_reg_loss<br/>(uniformity+radial_cal+jump)"]
        DL["dyn_loss<br/>(geodesic+chart_trans+momentum<br/>+energy+jump_dyn+screened_poisson+hodge)"]
        TOT["total = a*base + b*zn_reg + c*dyn"]
    end

    X0 --> ENC
    XH --> ENC
    ENC --> ROUTER
    ROUTER --> ENC
    ENC --> LATENT

    ZGEO --> DEC
    ZTEX --> DEC
    WENC --> DEC
    DEC --> RECON

    ZGEO -->|"z0"| ROLLOUT
    WENC -->|"rw0"| ROLLOUT
    A --> ROLLOUT
    ROLLOUT --> WMOUT

    RECON --> BL
    X0 --> BL
    VQLOSS --> BL
    WENC --> BL

    ZNALL --> ZNR
    ZN --> ZNR

    XH -->|"target latents via encoder"| DL
    ZPRED --> DL
    CHLOG --> DL
    MOM --> DL
    PHI --> DL
    JRATES --> DL
    HODGER --> DL

    BL --> TOT
    ZNR --> TOT
    DL --> TOT

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef encoder fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef decoder fill:#1f2b3b,stroke:#60a5fa,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef residual fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;
    classDef dynamics fill:#16213e,stroke:#0f3460,stroke-width:1px,color:#ffffff;
    classDef loss fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;

    class X0,XH,A io;
    class ENC,ROUTER,WENC,VQLOSS,ZNALL encoder;
    class DEC,RECON decoder;
    class ZGEO,ZN geom;
    class ZTEX residual;
    class ROLLOUT,ZPRED,CHLOG,MOM,PHI,JRATES,HODGER dynamics;
    class BL,ZNR,DL,TOT loss;
```

---

## 2. Atlas implementation

### 2.1 CovariantChartRouter

The chart router is shared by both encoder and decoder. It performs hyperbolic chart assignment using:
- Poincare-ball distance scoring with conformal temperature
- O(n) parallel transport via conformal factor scaling (no Cayley transform)
- Optional feature-aware correction with Christoffel-style quadratic terms

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph ROUTER["CovariantChartRouter (Hyperbolic)"]
        direction TB

        subgraph INPUT["Inputs"]
            direction TB
            Z["z [B, D]<br/>(Poincare ball)"]
            F["features [B, H]<br/>(optional)"]
            ChartTokens["chart_centers c_k [N_c, D]"]
        end

        subgraph HYPER["Hyperbolic Distance"]
            direction TB
            Dist["d_P(z, c_k)"]
            Tau["tau(z)=sqrt(K)*(1-||z||^2)/2"]
            Sdist["s_dist = -dist / tau"]
        end

        subgraph CORR["Covariant Correction (optional)"]
            direction TB
            Qz["q_z_proj(z) [B, K]"]
            Qfeat["q_feat_proj(features) [B, K]"]
            Gamma["gamma(z*z) [B, K]"]
            Qsum["q = q_z + q_feat + gamma [B, K]"]
            Lambda["lambda(z)=2/(1-||z||^2)"]
            Keys["keys = base_queries / lambda(z)"]
            Sfeat["s_feat = sum(keys * q) / tau"]
        end

        subgraph SCORING["Scoring"]
            direction TB
            Scores["scores = s_dist + 0.1*s_feat"]
            W["w = softmax / hard-route [B, N_c]"]
            Kchart["K_chart = argmax(w)"]
        end
    end

    Z --> Dist
    ChartTokens --> Dist
    Z --> Tau
    Dist --> Sdist
    Tau --> Sdist

    Z --> Qz
    F --> Qfeat
    Z --> Gamma
    Qz --> Qsum
    Qfeat --> Qsum
    Gamma --> Qsum
    Z --> Lambda
    ChartTokens --> Keys
    Lambda --> Keys
    Qsum --> Sfeat
    Keys --> Sfeat
    Tau --> Sfeat

    Sdist --> Scores
    Sfeat --> Scores
    Scores --> W --> Kchart

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef feat fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;

    class Z,Kchart geom;
    class F,Qfeat feat;
    class Dist,Tau,Sdist,Qz,Gamma,Qsum,Lambda,Keys,Sfeat,Scores,W,ChartTokens router;
```

### 2.2 PrimitiveAttentiveAtlasEncoder

The encoder performs feature extraction, hyperbolic routing, hyperbolic VQ per chart, and splits the latent into $(z_{geo}, z_n, z_{tex})$ (with $z_{geo}$ on the Poincare ball and $z_n, z_{tex}$ in the tangent space):

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph ENC["PrimitiveAttentiveAtlasEncoder"]
        direction TB

        subgraph S1["Stage 1: Feature Extraction"]
            direction TB
            X["Input x [B, D_in]"]
            FE["Feature extractor<br/>MLP or CovariantRetina"]
            F["features [B, H]"]
            Vproj["val_proj -> v_raw [B, D]"]
            Vball["project_to_ball(v_raw) -> v"]
        end

        subgraph S2["Stage 2: Hyperbolic Chart Routing"]
            direction TB
            ChartCenters["chart_centers c_k (ball)"]
            RouterEnc["CovariantChartRouter (hyperbolic)"]
            Wenc["w_enc [B, N_c]"]
            Kchart["K_chart [B]"]
        end

        subgraph S3["Stage 3: Hyperbolic Local Coordinates"]
            direction TB
            Cbar["c_bar = hyp_barycenter(w,c_k)"]
            Vlocal["v_local = (-c_bar) + v (Mobius)"]
        end

        subgraph S4["Stage 4: Hyperbolic VQ"]
            direction TB
            Codebook["codebook [N_c, K, D] (ball)"]
            Diff["delta = (-codebook) + v_local (Mobius)"]
            Log["log_map_zero(delta) -> delta_tan"]
            Dist["dist = ||delta_tan||^2 (soft-equiv metric optional)"]
            Indices["indices per chart"]
            ZqAll["z_q_all [B, N_c, D]"]
            ZqBlend["z_q_blended = hyp_barycenter(w,z_q_all)"]
        end

        subgraph S5["Stage 5: Nuisance + Texture (tangent)"]
            direction TB
            DeltaAll["d = log_map_zero((-z_q_all) + v_local)"]
            Struct["structure_filter (tangent)"]
            ZnAllTan["z_n_all_tan [B, N_c, D]"]
            ZnTan["z_n_tan = sum(w*z_n_all_tan)"]
            DeltaBlend["d_blended = log_map_zero((-z_q_blended) + v_local)"]
            Ztex["z_tex = d_blended - z_n_tan"]
        end

        subgraph S6["Stage 6: Geometric Assembly (ball)"]
            direction TB
            ZqSt["z_q_st = v_local + exp_map(d_to_code)"]
            Zlocal["z_local = z_q_st + exp_map(z_n_tan)"]
            Zgeo["z_geo = c_bar + z_local (project_to_ball)"]
        end
    end

    X --> FE --> F --> Vproj --> Vball

    F --> RouterEnc
    Vball --> RouterEnc
    ChartCenters --> RouterEnc
    RouterEnc --> Wenc
    RouterEnc --> Kchart

    Wenc --> Cbar
    ChartCenters --> Cbar
    Vball --> Vlocal
    Cbar --> Vlocal

    Vlocal --> Diff
    Codebook --> Diff
    Diff --> Log --> Dist --> Indices --> ZqAll
    Wenc --> ZqBlend
    ZqAll --> ZqBlend

    ZqAll --> DeltaAll
    Vlocal --> DeltaAll
    DeltaAll --> Struct --> ZnAllTan
    Wenc --> ZnTan
    ZnAllTan --> ZnTan
    ZqBlend --> DeltaBlend
    Vlocal --> DeltaBlend
    DeltaBlend --> Ztex
    ZnTan --> Ztex

    Vlocal --> ZqSt
    ZqBlend --> ZqSt
    ZqSt --> Zlocal
    ZnTan --> Zlocal
    Cbar --> Zgeo
    Zlocal --> Zgeo

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef feat fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef vq fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef residual fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;

    class X,Kchart io;
    class FE,F,Vproj,Struct feat;
    class RouterEnc,Wenc router;
    class Codebook,Diff,Log,Dist,Indices,ZqAll,ZqBlend vq;
    class ChartCenters,Cbar,Vball,Vlocal,Zgeo,ZqSt,Zlocal geom;
    class DeltaAll,DeltaBlend,ZnAllTan,ZnTan,Ztex residual;
```

### 2.3 PrimitiveTopologicalDecoder

The decoder performs chart-weighted reconstruction from the hyperbolic geometric latent $z_{geo}$ and adds the texture residual $z_{tex}$:
- Geometric path: Hyperbolic routing -> SpectralLinear chart projectors -> NormGatedGELU -> Renderer
- Texture path: Tanh + SpectralLinear residual with learned scale
- Final output: Base reconstruction + scaled texture residual

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph DEC["PrimitiveTopologicalDecoder"]
        direction TB

        subgraph INPUTS["Inputs"]
            direction TB
            Zgeo["z_geo [B, D] (ball)"]
            Ztex["z_tex [B, D] (tangent)"]
        end

        subgraph GEO["Geometric Path"]
            direction TB
            Clamp["project_to_ball(z_geo)"]
            RouterDec["CovariantChartRouter (hyperbolic)"]
            Wdec["w_dec [B, N_c]"]
            ChartProj["chart_projectors x N_c<br/>(SpectralLinear)"]
            Gate["NormGatedGELU"]
            Mix["h_global = sum(w*h_stack)"]
            Renderer["renderer (SpectralLinear + NormGatedGELU)"]
            Skip["render_skip"]
            AddSkip["x_base = render + skip"]
        end

        subgraph TEX["Texture Path"]
            direction TB
            TanhT["tanh(z_tex)"]
            TexRes["tex_residual (SpectralLinear)"]
            Scale["alpha = tex_residual_scale"]
        end

        subgraph OUTPUT["Output"]
            direction TB
            AddTex["x_hat = x_base + alpha*tex_res"]
            Xhat["x_hat [B, D_out]"]
        end
    end

    Zgeo --> Clamp
    Clamp --> RouterDec --> Wdec
    Clamp --> ChartProj --> Gate --> Mix
    Wdec --> Mix
    Mix --> Renderer --> AddSkip
    Mix --> Skip --> AddSkip

    Ztex --> TanhT --> TexRes
    TexRes --> Scale

    AddSkip --> AddTex
    Scale --> AddTex
    AddTex --> Xhat

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef router fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef residual fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;
    classDef decoder fill:#1f2b3b,stroke:#60a5fa,stroke-width:1px,color:#ffffff;

    class Zgeo geom;
    class RouterDec,Wdec router;
    class Xhat io;
    class Clamp,ChartProj,Gate,Mix,Renderer,Skip,AddSkip,AddTex decoder;
    class Ztex,TanhT,TexRes,Scale residual;
```

---

## 3. World model implementation (hyperbolic field dynamics)

### 3.1 GeometricWorldModel internals

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph GWMODEL["GeometricWorldModel"]
        direction TB

        subgraph WIN["Inputs"]
            direction TB
            Z0["z_0 [B, D]<br/>(Poincare ball)"]
            ACT["actions [B, H, A]"]
            RW0["router_weights_0 [B, K]"]
        end

        subgraph TOK["Tokenizers"]
            direction TB
            TOKA["ActionTokenizer<br/>action -> [B, A, d_model]<br/>+ position tokens at z"]
            TOKC["ChartTokenizer<br/>rw -> [B, K, d_model]<br/>+ chart center tokens"]
        end

        subgraph FORCE["Force Computation"]
            direction TB
            POT["CovariantPotentialNet<br/>F = a*dU_dz + (1-a)*f_critic + g*f_risk<br/>Phi_eff = a*U + (1-a)*V_critic + g*Psi_risk"]
            CTRL["CovariantControlField<br/>u_pi(z, action, rw) -> [B, D]"]
            CURL["CovariantValueCurl<br/>F_mat [B, D, D] (antisymmetric)<br/>upper-triangle reconstruction"]
        end

        subgraph METRIC["Geometry"]
            direction TB
            CM["ConformalMetric<br/>lambda(z) = 2/(1-||z||^2)"]
            CHRIS["christoffel_contraction<br/>Gamma_ij^k v^i v^j [B, D]"]
            RISK["compute_risk_tensor<br/>T = f*f + T_maxwell [B, D, D]"]
        end

        subgraph INT["BAOAB Integrator (per timestep)"]
            direction TB
            B1["B1: half kick<br/>p -= (dt/2)*(force - u_pi)"]
            BORIS["Boris rotation<br/>norm-preserving via F_mat"]
            A1["A1: geodesic drift<br/>z = exp_map(z, (dt/2)*v_corr)"]
            OU["O: OU thermostat<br/>p = c1*p + c2*lambda*xi"]
            A2["A2: geodesic drift<br/>z = exp_map(z, (dt/2)*v_corr)"]
            B2["B2: half kick<br/>p -= (dt/2)*(force2 - u_pi2)"]
        end

        subgraph JUMP["Jump Process"]
            direction TB
            JRATE["CovariantJumpRate<br/>rate [B, 1] (softplus)"]
            JOP["FactorizedJumpOperator<br/>z_tgt = c_tgt + R((-c_src) + z)<br/>(Mobius chart transition)"]
            CPRED["CovariantChartTarget<br/>chart_logits [B, K]<br/>(CovariantAttention cross-attn)"]
        end

        subgraph DIAG["Diagnostics"]
            direction TB
            HODGE["HodgeDecomposer<br/>f_cons / f_sol / f_harmonic ratios"]
            MINIT["CovariantMomentumInit<br/>p_0 = lambda^2 * net(z_0)"]
        end

        subgraph WOUT["Outputs"]
            direction TB
            ZT["z_trajectory [B, H, D]"]
            MT["momenta [B, H, D]"]
            CL["chart_logits [B, H, K]"]
            PE["Phi_eff [B, H, 1]"]
            JR["jump_rates [B, H, 1]"]
            JM["jump_masks [B, H]"]
            HR["hodge_*_ratio [B, H]"]
            HFO["hodge_harmonic_forces [B, H, D]"]
        end
    end

    Z0 --> MINIT
    MINIT --> B1
    ACT --> TOKA
    RW0 --> TOKC

    TOKA --> CTRL
    TOKC --> POT
    Z0 --> POT
    Z0 --> CURL
    ACT --> CURL

    POT --> B1
    CTRL --> B1
    CURL --> BORIS
    B1 --> BORIS
    CM --> A1
    CHRIS --> A1
    BORIS --> A1
    A1 --> OU
    CM --> OU
    OU --> A2
    CM --> A2
    CHRIS --> A2
    A2 --> B2
    POT --> B2
    CTRL --> B2
    RISK --> CM

    B2 --> JRATE
    B2 --> JOP
    B2 --> CPRED
    ACT --> CPRED
    RW0 --> CPRED
    JRATE --> JOP

    POT --> HODGE
    BORIS --> HODGE

    B2 --> ZT
    B2 --> MT
    CPRED --> CL
    POT --> PE
    JRATE --> JR
    JOP --> JM
    HODGE --> HR
    HODGE --> HFO

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef force fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef integrator fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;
    classDef jump fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;
    classDef diag fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;

    class Z0,ACT,RW0,ZT,MT,CL,PE,JR,JM,HR,HFO io;
    class TOKA,TOKC,POT,CTRL,CURL force;
    class CM,CHRIS,RISK geom;
    class B1,BORIS,A1,OU,A2,B2 integrator;
    class JRATE,JOP,CPRED jump;
    class HODGE,MINIT diag;
```

### 3.2 BAOAB step decomposition as implemented pattern

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph BAOAB["Boris-BAOAB Integration Step"]
        direction TB

        subgraph B1["B1: First Half Kick"]
            direction TB
            F1["force, _ = potential_net(z, rw) [B, D]"]
            U1["u_pi = control_net(z, action, rw) [B, D]"]
            KICK1["kick = force - u_pi"]
            SQUASH1["psi_F(kick) force squash (if CFL)"]
            PM["p_minus = p - (dt/2)*kick"]
        end

        subgraph BR["Boris Rotation (norm-preserving)"]
            direction TB
            FMAT["F = curl_net(z, action) [B, D, D]<br/>(antisymmetric)"]
            TSCALE["T = (h/2)*beta_curl*lambda_inv_sq*F"]
            TVEC["t_vec = T @ p_minus"]
            PPRIME["p_prime = p_minus + t_vec"]
            SFAC["s = 2/(1 + ||T||^2_F)"]
            SVEC["s_vec = s*T @ p_prime"]
            PPLUS["p_plus = p_minus + s_vec"]
        end

        subgraph A1["A1: First Geodesic Drift"]
            direction TB
            CF1["lambda = conformal_factor(z) [B, 1]"]
            LINV1["lambda_inv_sq = 1/(lambda^2 + eps)"]
            VEL1["v = lambda_inv_sq * p [B, D]"]
            GEO1["geo_corr = christoffel_contraction(z, v)"]
            VCORR1["v_corr = v - (dt/4)*geo_corr"]
            EXP1["z = poincare_exp_map(z, (dt/2)*v_corr)"]
            PROJ1["z = project_to_ball(z)"]
        end

        subgraph OUST["O: Ornstein-Uhlenbeck Thermostat"]
            direction TB
            CF2["lambda = conformal_factor(z) [B, 1]"]
            CFCAP["(optional) lambda_cap = lambda_max*tanh(lambda/lambda_max)"]
            XI["xi = randn_like(p) [B, D]"]
            C1["c1 = exp(-gamma*dt)"]
            C2["c2 = sqrt(max(0, (1-c1^2)*T_c))"]
            POU["p = c1*p + c2*lambda*xi"]
        end

        subgraph A2S["A2: Second Geodesic Drift"]
            direction TB
            CF3["lambda = conformal_factor(z)"]
            VEL2["v = lambda_inv_sq * p"]
            GEO2["geo_corr2 = christoffel_contraction(z, v)"]
            EXP2["z = poincare_exp_map(z, (dt/2)*v_corr2)"]
            PROJ2["z = project_to_ball(z)"]
        end

        subgraph B2S["B2: Second Half Kick"]
            direction TB
            F2["force2, Phi_eff = potential_net(z, rw)"]
            U2["u_pi2 = control_net(z, action, rw)"]
            KICK2["kick2 = force2 - u_pi2"]
            PFINAL["p = p - (dt/2)*kick2"]
        end

        subgraph HDIAG["Hodge Diagnostic"]
            direction TB
            HCONS["f_conservative = force (from potential_net)"]
            HSOL["f_solenoidal = (p_plus - p_minus) / dt"]
            HTOT["f_total = kick + f_solenoidal"]
            HARM["f_harmonic = f_total - f_cons - f_sol"]
            HRAT["conservative / solenoidal / harmonic ratios"]
        end
    end

    F1 --> KICK1
    U1 --> KICK1
    KICK1 --> SQUASH1 --> PM

    PM --> TVEC
    FMAT --> TSCALE --> TVEC
    TVEC --> PPRIME --> SFAC
    PPRIME --> SVEC
    SFAC --> SVEC
    SVEC --> PPLUS

    PPLUS --> VEL1
    CF1 --> LINV1 --> VEL1
    VEL1 --> GEO1 --> VCORR1
    VEL1 --> VCORR1
    VCORR1 --> EXP1 --> PROJ1

    PROJ1 --> CF2
    CF2 --> CFCAP
    XI --> POU
    C1 --> POU
    C2 --> POU
    CFCAP --> POU

    POU --> VEL2
    CF3 --> VEL2
    VEL2 --> GEO2 --> EXP2 --> PROJ2

    PROJ2 --> F2
    F2 --> KICK2
    U2 --> KICK2
    KICK2 --> PFINAL

    F1 --> HCONS
    PM --> HSOL
    PPLUS --> HSOL
    KICK1 --> HTOT
    HSOL --> HTOT
    HCONS --> HARM
    HSOL --> HARM
    HTOT --> HARM
    HARM --> HRAT

    classDef force fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef integrator fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef diag fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;

    class F1,U1,KICK1,SQUASH1,PM,F2,U2,KICK2,PFINAL force;
    class FMAT,TSCALE,TVEC,PPRIME,SFAC,SVEC,PPLUS integrator;
    class CF1,LINV1,VEL1,GEO1,VCORR1,EXP1,PROJ1 geom;
    class CF2,CFCAP,XI,C1,C2,POU geom;
    class CF3,VEL2,GEO2,EXP2,PROJ2 geom;
    class HCONS,HSOL,HTOT,HARM,HRAT diag;
```

### 3.3 Screened Poisson implementation path

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph SP["Screened Poisson Loss (PDE Residual)"]
        direction TB

        subgraph SAMPLE["Trajectory Sampling"]
            direction TB
            ZTR["z_trajectory [B, H, D]"]
            SUB["subsample max_samples=64"]
            ZS["z_samples [B*S, D]"]
        end

        subgraph VEVAL["Value Evaluation"]
            direction TB
            VFUNC["V(z) via potential_net.v_critic_attn [B*S, 1]"]
        end

        subgraph HLAP["Hyperbolic Laplacian (Hutchinson)"]
            direction TB
            PROBE["v ~ Rademacher(+/-1) [B*S, D]"]
            FD["finite-diff: V(z +/- eps*v)"]
            HESS["Hv = (V_plus - 2V + V_minus)/eps^2"]
            TRACE["tr(H) = sum(Hv) (Hutchinson estimate)"]
            CONF["lambda(z) = 2/(1-||z||^2)"]
            GRAD["grad_V via finite diff"]
            LB["Delta_G V = lambda^-2 * [tr(H) + (D-2)*lambda*(z . grad_V)]"]
        end

        subgraph PDE["PDE Residual"]
            direction TB
            KAPPA["kappa^2 (screening mass)"]
            RHO["rho_r (reward source density)"]
            RES["residual = (-Delta_G + kappa^2)*V - rho_r"]
            LSP["L_SP = mean(residual^2)"]
        end
    end

    ZTR --> SUB --> ZS --> VFUNC

    ZS --> PROBE --> FD
    VFUNC --> FD
    FD --> HESS --> TRACE
    ZS --> CONF
    FD --> GRAD
    CONF --> LB
    TRACE --> LB
    GRAD --> LB

    VFUNC --> RES
    KAPPA --> RES
    LB --> RES
    RHO --> RES
    RES --> LSP

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef compute fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef geom fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef pde fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;

    class ZTR,ZS io;
    class VFUNC compute;
    class PROBE,FD,HESS,TRACE,CONF,GRAD,LB geom;
    class KAPPA,RHO,RES,LSP pde;
    class SUB io;
```

Advantages of this implementation choice:
- Enforces PDE-style structure on the potential field, not only pointwise regression.
- Couples value smoothness and geometry through hyperbolic Laplace-Beltrami.
- Integrates directly into `compute_phase2_loss` with explicit weight control.

---

## 4. Full loss architecture

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph LOSSARCH["Full Loss Architecture"]
        direction TB

        subgraph FORWARD["Forward Pass"]
            direction TB
            X["batch features [B, D_in]"]
            Enc["Encoder (hyperbolic)"]
            EncOut["z_geo, z_tex, w_enc,<br/>vq_loss, indices, z_n_all_charts"]
            Dec["Decoder (hyperbolic)"]
            Recon["recon [B, D_out]"]
        end

        subgraph ENCLOSS["Encoder-Side Losses"]
            direction TB

            subgraph BASE["base_loss"]
                direction TB
                ReconLoss["MSE(recon, x)"]
                VQLoss["VQ loss (tangent)"]
                Entropy["routing entropy"]
                Consistency["encoder/decoder consistency"]
                Diversity["diversity(w)"]
                CbSpread["codebook_spread"]
                CbCenter["codebook_center"]
                ChartCollapse["chart_collapse"]
                CodeCollapse["code_collapse"]
                Window["window_loss"]
            end

            subgraph ZNREG["zn_reg_loss"]
                direction TB
                Uniformity["uniformity(z_n)"]
                RadCal["radial_calibration(z_n)"]
                JumpCons["jump consistency<br/>(FactorizedJumpOperator)"]
            end
        end

        subgraph DYNLOSS["Dynamics Losses (compute_phase2_loss)"]
            direction TB
            Geodesic["geodesic d_P(z_pred, z_target)"]
            ChartTrans["chart_transition CE(logits, targets)"]
            MomReg["momentum_reg (1/2)*p^T*G^-1*p"]
            EnergyCons["energy_conservation Var(H)"]
            JumpDyn["jump_dynamics L1(rates)"]
            ScreenedP["screened_poisson PDE residual"]
            HodgeCons["hodge ||f_harmonic||^2"]
        end

        subgraph AGG["Loss Assembly"]
            direction TB
            BL["base_loss (weighted sum)"]
            ZL["zn_reg_loss (weighted sum)"]
            DL["dyn_loss (weighted sum)"]
            TOT["total = a_enc*base + a_zn*zn_reg + a_dyn*dyn"]
        end
    end

    X --> Enc --> EncOut --> Dec --> Recon

    Recon --> ReconLoss
    X --> ReconLoss

    EncOut --> VQLoss
    EncOut --> Entropy
    EncOut --> Consistency
    EncOut --> Diversity
    EncOut --> CbSpread
    EncOut --> CbCenter
    EncOut --> ChartCollapse
    EncOut --> CodeCollapse
    EncOut --> Window

    EncOut --> Uniformity
    EncOut --> RadCal
    EncOut --> JumpCons

    ReconLoss --> BL
    VQLoss --> BL
    Entropy --> BL
    Consistency --> BL
    Diversity --> BL
    CbSpread --> BL
    CbCenter --> BL
    ChartCollapse --> BL
    CodeCollapse --> BL
    Window --> BL

    Uniformity --> ZL
    RadCal --> ZL
    JumpCons --> ZL

    Geodesic --> DL
    ChartTrans --> DL
    MomReg --> DL
    EnergyCons --> DL
    JumpDyn --> DL
    ScreenedP --> DL
    HodgeCons --> DL

    BL --> TOT
    ZL --> TOT
    DL --> TOT

    classDef io fill:#0b1320,stroke:#93c5fd,stroke-width:1px,color:#ffffff;
    classDef encoder fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef decoder fill:#1f2b3b,stroke:#60a5fa,stroke-width:1px,color:#ffffff;
    classDef loss fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef vq fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;
    classDef dynamics fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef opt fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;

    class X io;
    class Enc,EncOut encoder;
    class Dec,Recon decoder;
    class ReconLoss,VQLoss,Entropy,Consistency,Diversity,CbSpread,CbCenter,ChartCollapse,CodeCollapse,Window,BL loss;
    class Uniformity,RadCal,JumpCons,ZL vq;
    class Geodesic,ChartTrans,MomReg,EnergyCons,JumpDyn,ScreenedP,HodgeCons,DL dynamics;
    class TOT opt;
```

Field-theory view of losses:
- `base_loss`: stabilizes atlas chart/state representation.
- `zn_reg_loss`: regularizes nuisance subfield statistics.
- `dyn_loss`: enforces geometric trajectory laws and force decomposition consistency.

---

## 5. Phase-3-only training implementation

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph P3TRAIN["Phase-3 Joint Training"]
        direction TB

        subgraph FWD["Forward Assembly"]
            direction TB
            E0["Frame t0 encoder pass<br/>base_loss + zn_reg_loss"]
            ER["Frames t1..tH batched encode<br/>target latent sequence"]
            WMR["World-model rollout<br/>z0 + actions -> z_pred"]
            DLoss["compute_phase2_loss<br/>-> dyn_loss"]
            TOTAL["total = a_enc*base + a_zn*zn_reg + a_dyn*dyn"]
        end

        subgraph GS["Three-Pass Gradient Surgery"]
            direction TB

            subgraph P1["Pass 1: a_enc * L_base.backward(retain_graph)"]
                direction TB
                P1G["Gradients -> feature_extractor, val_proj,<br/>cov_router, chart_centers, codebook"]
                P1Z["Zero: structure_filter.grad<br/>(blocks reconstruction -> z_n path)"]
            end

            subgraph P2["Pass 2: a_zn * L_zn_reg.backward(retain_graph)"]
                direction TB
                P2G["Gradients -> structure_filter<br/>(uniformity + radial calibration)"]
                P2S["Save: protected_params.grad snapshot<br/>{feat_ext, val_proj, cov_router,<br/>chart_centers, soft_equiv_layers}"]
            end

            subgraph P3["Pass 3: a_dyn * L_dyn.backward()"]
                direction TB
                P3G["Gradients accumulate on ALL params<br/>(encoder + world model)"]
                P3R["Restore: protected_params.grad = saved<br/>(overwrites dynamics on protected set)"]
            end

            subgraph STEP["optimizer.step() Result"]
                direction TB
                R1["feature_ext, val_proj, cov_router, chart_centers:<br/>updated by L_base + L_zn_reg only"]
                R2["codebook (VQ centers):<br/>updated by L_base + L_zn_reg + L_dyn"]
                R3["structure_filter (z_n):<br/>updated by L_zn_reg + L_dyn (NOT L_base)"]
                R4["world_model:<br/>updated by L_dyn only"]
            end
        end
    end

    E0 --> TOTAL
    ER --> DLoss
    WMR --> DLoss
    DLoss --> TOTAL

    TOTAL --> P1G
    P1G --> P1Z
    P1Z --> P2G
    P2G --> P2S
    P2S --> P3G
    P3G --> P3R
    P3R --> R1
    P3R --> R2
    P3R --> R3
    P3R --> R4

    classDef forward fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef pass1 fill:#2b1f1f,stroke:#f59e0b,stroke-width:1px,color:#ffffff;
    classDef pass2 fill:#1f2f2a,stroke:#34d399,stroke-width:1px,color:#ffffff;
    classDef pass3 fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;
    classDef result fill:#3b1f2b,stroke:#f472b6,stroke-width:1px,color:#ffffff;

    class E0,ER,WMR,DLoss,TOTAL forward;
    class P1G,P1Z pass1;
    class P2G,P2S pass2;
    class P3G,P3R pass3;
    class R1,R2,R3,R4 result;
```

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart LR
    subgraph PG1["Optimizer Group 1: Atlas"]
        direction TB
        ENCP["encoder + decoder +<br/>FactorizedJumpOperator params"]
        LR1["lr = lr_joint_encoder (1e-4)<br/>10x lower than Phase 1"]
    end

    subgraph PG2["Optimizer Group 2: World Model"]
        direction TB
        WMP["GeometricWorldModel<br/>all sub-modules"]
        LR2["lr = lr_joint_wm (1e-3)<br/>same as Phase 2"]
    end

    ENCP --> LR1
    WMP --> LR2

    classDef atlas fill:#111827,stroke:#22d3ee,stroke-width:1px,color:#ffffff;
    classDef wm fill:#1f2937,stroke:#a78bfa,stroke-width:1px,color:#ffffff;

    class ENCP,LR1 atlas;
    class WMP,LR2 wm;
```

---

## 6. Theoretical advantages over the state of the art

```mermaid
%%{init: {"themeVariables": {"background":"#1e293b","edgeLabelBackground":"#334155","textColor":"#ffffff","lineColor":"#cbd5e1","primaryColor":"#334155","primaryTextColor":"#ffffff","secondaryTextColor":"#ffffff","tertiaryTextColor":"#ffffff","titleColor":"#ffffff","nodeTextColor":"#ffffff","clusterBkg":"#334155","clusterBorder":"#64748b","fontSize":"18px"},"flowchart":{"nodeSpacing":60,"rankSpacing":70,"useMaxWidth":true}}}%%
flowchart TB
    subgraph FEATURES["Fragile-Agent Architecture"]
        direction TB
        HYP["Hyperbolic atlas latent<br/>(Poincare ball + charted atlas)"]
        SPL["Split latent with texture firewall<br/>z_geo + z_n + z_tex"]
        COV["Covariant routing + attention<br/>(gauge-invariant parallel transport)"]
        PHYS["Boris-BAOAB symplectic integrator<br/>+ Hodge force decomposition"]
        SPS["Screened-Poisson PDE critic<br/>(Helmholtz on Poincare ball)"]
        GSR["Three-pass gradient surgery<br/>(targeted parameter firewalls)"]
        DIAG["Built-in physics diagnostics<br/>(Hodge ratios + energy + jump)"]
    end

    subgraph SOTA["State-of-the-Art Comparison"]
        direction TB
        S1["Dreamer / IRIS / Genie / RT-2:<br/>flat Euclidean latent, polynomial volume"]
        S2["Standard VAE / VQ-VAE:<br/>monolithic z, entangled factors"]
        S3["Multi-head attention:<br/>Euclidean dot product, no transport"]
        S4["RSSM / GRU / Transformer WM:<br/>no phase space, no conservation laws"]
        S5["TD-learning / SAC critic:<br/>pointwise regression, no PDE structure"]
        S6["Frozen encoder or end-to-end:<br/>all-or-nothing gradient flow"]
        S7["Black-box world models:<br/>no force field interpretability"]
    end

    HYP -->|"exponential vs polynomial<br/>volume growth"| S1
    SPL -->|"orthogonal gradient paths<br/>vs entangled backprop"| S2
    COV -->|"conformal temperature +<br/>Christoffel correction"| S3
    PHYS -->|"symplectic + energy-conserving<br/>vs dissipative RNN"| S4
    SPS -->|"Laplace-Beltrami PDE<br/>vs Bellman backup"| S5
    GSR -->|"selective firewall<br/>vs blanket freeze"| S6
    DIAG -->|"cons/sol/harm ratios<br/>vs opaque loss curves"| S7

    classDef feature fill:#1a1a2e,stroke:#e94560,stroke-width:1px,color:#ffffff;
    classDef sota fill:#16213e,stroke:#0f3460,stroke-width:1px,color:#ffffff;

    class HYP,SPL,COV,PHYS,SPS,GSR,DIAG feature;
    class S1,S2,S3,S4,S5,S6,S7 sota;
```

### 6.1 Hyperbolic atlas latent vs flat Euclidean spaces

**State of the art.** Dreamer-v3 encodes observations into a flat Euclidean latent via an RSSM with discrete + continuous components. IRIS tokenizes frames into a flat discrete grid. Genie and RT-2 use flat continuous latent spaces. In all cases the latent volume grows polynomially: a $D$-dimensional Euclidean ball of radius $r$ has volume $\propto r^D$.

**Fragile.** The Poincare ball model $\mathbb{B}^D$ with conformal factor $\lambda(z) = 2/(1 - \|z\|^2)$ has volume that grows *exponentially* with geodesic radius: $\text{vol}(B_r) \sim e^{(D-1)r}$. This means the representational capacity of the latent interior is exponentially larger than a Euclidean ball of the same coordinate radius. Concretely:

- **Hierarchy without depth.** Tree-structured environments (decision trees, compositional plans, nested goals) embed with bounded distortion into hyperbolic space but require $\Omega(n)$ dimensions in Euclidean space (Sarkar 2011). Fragile encodes these hierarchies natively in the ball geometry.
- **Boundary sharpening.** As states approach the ball boundary ($\|z\| \to 1$), the conformal factor $\lambda \to \infty$ and the metric blows up. This creates a natural "high-resolution rim" where fine distinctions are amplified, analogous to how the Poincare disk crowds detail near its boundary. States near the origin represent coarse, abstract features; states near the boundary represent precise, concrete features. Standard flat latents have uniform resolution everywhere.
- **Chart atlas prevents winner-take-all.** The `CovariantChartRouter` initializes chart centers via Fibonacci-sphere + Lloyd relaxation across the ball, guaranteeing quasi-uniform coverage. Each chart owns a Voronoi-like catchment in hyperbolic metric. In Dreamer's flat latent, there is no such partitioning: the entire space is one undifferentiated coordinate system, making it vulnerable to mode collapse where a few latent directions dominate.

**Example.** Consider a manipulation task with 8 semantically distinct object configurations. Dreamer must learn to separate these 8 modes in a flat space, relying entirely on the reconstruction loss to carve out regions. Fragile assigns each mode to a chart via the router; within each chart, the VQ codebook further discretizes sub-states. The hyperbolic distances between chart centers grow exponentially with separation, so even nearby chart centers are well-separated in geodesic distance. This provides a geometric prior for multi-modal environments that flat latents lack.

### 6.2 Split latent with texture firewall vs monolithic representations

**State of the art.** Standard VAEs and VQ-VAEs encode everything into a single latent vector $z$. World models like Dreamer's RSSM maintain a monolithic hidden state $(h_t, z_t)$ where structure, nuisance, and appearance are entangled. Any gradient from the dynamics loss can reshape the appearance representation and vice versa. Beta-VAE and similar approaches add scalar KL weights but do not enforce orthogonal gradient paths.

**Fragile.** The encoder decomposes the latent into three subspaces with distinct geometric roles:

| Component | Space | Role | Dynamics coupling |
|---|---|---|---|
| $z_{\text{geo}}$ | Poincare ball | Macro state (chart + VQ code) | Full |
| $z_n$ | Tangent space | Gauge-invariant nuisance | Partial (structure_filter only) |
| $z_{\text{tex}}$ | Tangent space | Appearance residual | **None** (texture firewall) |

The texture firewall is not a soft penalty: $z_{\text{tex}}$ is structurally excluded from the world model forward pass. No gradient path exists from the dynamics loss $\mathcal{L}_{\text{dyn}}$ to the texture subspace. This is enforced by code structure, not by a loss weight that could be overwhelmed.

**Why this matters.** In a robotic pick-and-place task, the texture of an object's surface (wood grain, color variations) is irrelevant to dynamics but dominates pixel-level reconstruction. In a monolithic latent, the dynamics loss would fight the reconstruction loss over these dimensions: dynamics wants them constant (irrelevant to motion), reconstruction wants them to vary (relevant to appearance). Fragile resolves this conflict by architecture: $z_{\text{tex}}$ captures appearance for reconstruction, $z_{\text{geo}}$ captures structure for dynamics, and the gradient firewall (Section 5) prevents cross-contamination.

**Contrast with beta-VAE.** Beta-VAE uses $\beta \cdot D_{\mathrm{KL}}$ to pressure the latent toward a standard normal, encouraging disentanglement as a side effect. But beta-VAE cannot guarantee which dimensions capture which factors; the mapping is implicit and can break under distribution shift. Fragile's decomposition is explicit and architecturally enforced: the VQ codebook captures discrete topology, the structure filter captures continuous gauge residuals, and the subtraction $z_{\text{tex}} = \delta_{\text{blended}} - z_n$ captures everything else. Each component has its own gradient pathway.

### 6.3 Covariant routing and attention vs standard dot-product attention

**State of the art.** Standard multi-head attention computes $\text{softmax}(QK^\top / \sqrt{d_k})V$ with Euclidean dot products. The temperature $\sqrt{d_k}$ is a global constant. Keys and queries live in the same flat space regardless of the state's position on the manifold. There is no notion of parallel transport: comparing a query at state $z_a$ with a key at state $z_b$ treats both as if they share a common coordinate frame.

**Fragile.** The `CovariantChartRouter` and `CovariantAttention` modules introduce three geometric corrections:

1. **Parallel transport via conformal scaling.** Before comparing query $q(z)$ with key $k(c_k)$, the key is transported from the chart center $c_k$ to the query position $z$ via $k_{\text{transported}} = k / \lambda(z)$. This is $O(n)$ per key (element-wise division), unlike full $O(n^3)$ Lie-group transport. The conformal factor $\lambda(z) = 2/(1-\|z\|^2)$ ensures the comparison respects the local geometry: near the boundary, keys are downscaled (distances are large), making attention sharper.

2. **Position-dependent temperature.** The softmax temperature is $\tau(z) = \sqrt{d_k} \cdot (1 - \|z\|^2) / 2$, which decreases near the ball boundary. High-curvature regions automatically produce sharp (confident) routing; low-curvature regions produce soft (exploratory) routing. Standard attention has no mechanism for this: its temperature is the same everywhere.

3. **Christoffel correction in queries.** The query projection includes a quadratic term $\gamma(z \otimes z)$ that encodes the Christoffel symbols $\Gamma^k_{ij}$. This forces the attention scores to account for geodesic curvature, not just Euclidean proximity. In standard attention, two states equidistant in Euclidean coordinates receive equal attention weights; in covariant attention, the state in a higher-curvature region receives sharper weighting because the geodesic distance is larger.

**Example.** In a navigation task, two waypoints may be equidistant in pixel coordinates but one is near a decision boundary (high curvature: should the agent turn left or right?) while the other is on a straight path (low curvature). Standard attention treats both identically. Covariant attention automatically sharpens routing near the decision boundary, producing a more decisive chart assignment exactly where disambiguation matters.

### 6.4 Boris-BAOAB symplectic integrator vs RNN/transformer world models

**State of the art.** Dreamer-v3 uses a Recurrent State Space Model (RSSM) with GRU dynamics. IRIS uses a GPT-style transformer to autoregressively predict next tokens. TD-MPC2 uses a learned MLP transition model. None of these models have a notion of phase space $(z, p)$, conservation laws, or symplectic structure. They are dissipative by construction: a GRU can arbitrarily amplify or attenuate information at each step.

**Fragile.** The `GeometricWorldModel` maintains an explicit phase space $(z, p)$ where $z$ is position on the Poincare ball and $p$ is cotangent momentum. The dynamics are integrated via Boris-BAOAB, a splitting integrator with three distinct operator types:

1. **B-step (Kick).** Updates momentum from conservative force $-\nabla\Phi$ and control force $u_\pi$. The force is computed by `CovariantPotentialNet`, which decomposes it as $F = \alpha \cdot \nabla U + (1-\alpha) \cdot f_{\text{critic}} + \gamma \cdot f_{\text{risk}}$, separating intrinsic geometry ($U$), learned value ($f_{\text{critic}}$), and risk adaptation ($f_{\text{risk}}$).

2. **Boris rotation.** Applies the solenoidal (curl) force via a norm-preserving rotation: $|p_{\text{out}}| = |p_{\text{in}}|$ exactly. This is the magnetic analogue: the curl field steers the agent into orbits without changing its kinetic energy. Standard RNNs have no mechanism for norm-preserving rotations; they must learn this as an emergent property, which is fragile and typically lost under distribution shift.

3. **A-step (Drift).** Updates position via geodesic flow with explicit Christoffel corrections $\Gamma^k_{ij} v^i v^j$, then projects back to the ball via `poincare_exp_map`. This ensures the state stays on the manifold. GRU/transformer outputs are unconstrained vectors that can drift off-manifold.

4. **O-step (Thermostat).** Applies Ornstein-Uhlenbeck friction and noise with conformal-factor scaling: $p \leftarrow c_1 p + c_2 \lambda(z) \xi$. The noise amplitude is modulated by the local geometry, providing more exploration in flat regions and less in curved regions.

**Symplectic verification.** The energy conservation loss $\mathcal{L}_{\text{energy}} = \text{Var}(H_t)$ across the horizon measures how well the integrator preserves the Hamiltonian $H = \Phi_{\text{eff}} + \frac{1}{2} p^\top G^{-1} p$. A perfect symplectic integrator keeps $H$ constant; drift indicates numerical or architectural errors. Standard world models have no analogous diagnostic because they do not track energy.

**Why it matters for long-horizon prediction.** Symplectic integrators have bounded energy error over exponentially long times (backward error analysis). Non-symplectic methods (Euler, RNN) accumulate energy error linearly or worse, causing trajectory predictions to diverge. For a 50-step rollout, the BAOAB integrator's energy error remains $O(\Delta t^2)$ regardless of horizon length, while a GRU's hidden state can drift unboundedly.

### 6.5 Screened Poisson PDE critic vs pointwise TD-learning

**State of the art.** Standard critics (TD-learning, SAC, PPO) learn $V(s)$ by minimizing the Bellman residual $\|V(s) - (r + \gamma V(s'))\|^2$. This is a pointwise regression: the critic sees individual $(s, r, s')$ transitions and has no constraint on the spatial structure of $V$. Nothing prevents $V$ from being arbitrarily non-smooth, developing spurious local optima, or failing to generalize between nearby states.

**Fragile.** The critic $V(z)$ is constrained to satisfy the screened Poisson equation on the Poincare ball:

$$(-\Delta_G + \kappa^2) V(z) = \rho_r(z)$$

where $\Delta_G$ is the Laplace-Beltrami operator, $\kappa$ is the screening mass, and $\rho_r$ is the reward source density. This is enforced as a PDE residual loss using batched Hutchinson trace estimation (no second-order autograd).

**Three concrete advantages over TD-learning:**

1. **Spatial smoothness is built in.** The Laplacian $\Delta_G V$ penalizes high-frequency oscillations in $V$ with respect to the manifold metric. Two states close in geodesic distance must have similar values. TD-learning has no such constraint: $V(s)$ and $V(s')$ can differ arbitrarily even for adjacent states, leading to noisy value landscapes.

2. **Screening mass controls spatial decay.** The $\kappa^2 V$ term creates exponential decay: the value field falls off as $V \sim e^{-\kappa r}$ away from reward sources. This is the physics of a Yukawa potential: rewards have a finite "range of influence" controlled by $\kappa$. TD-learning with discount $\gamma$ also creates decay, but it is temporal (per-step), not spatial (per-geodesic-distance). Fragile's spatial screening is geometry-aware.

3. **Hyperbolic Laplacian couples value to curvature.** The Poincare-ball Laplacian includes curvature corrections: $\Delta_G f = \lambda^{-2}[\Delta_E f + (D-2)\lambda(z \cdot \nabla f)]$. Near the ball boundary (high curvature), the Laplacian is suppressed, allowing the value function to vary more sharply. Near the origin (low curvature), the Laplacian enforces smoothness. This automatically adapts the critic's resolution to the latent geometry.

**Example.** In a goal-reaching task, the reward source is at a goal state $z_g$. TD-learning must propagate value backward step-by-step through Bellman backups, requiring many iterations to fill in $V$ across the state space. The screened Poisson equation directly solves for $V$ as a Green's function centered at $z_g$, producing a globally coherent value landscape in a single optimization step. The screening mass $\kappa$ determines how far the "value field" extends from the goal.

### 6.6 Three-pass gradient surgery vs standard encoder-dynamics coupling

**State of the art.** Joint training of encoder and world model faces a fundamental tension: the dynamics loss wants to reshape the encoder's representation to be "easy to predict," which can destroy the perceptual quality of the latent space (catastrophic forgetting). Common solutions include:

- **Frozen encoder** (Dreamer-v2, early RT-2 variants): encoder is fixed after pretraining. Simple but prevents the encoder from adapting to downstream dynamics needs.
- **End-to-end finetuning**: all parameters receive all gradients. Effective but risks representation collapse, especially when dynamics losses dominate reconstruction.
- **EWC / Fisher regularization**: adds a penalty $\sum_i F_i(\theta_i - \theta_i^*)^2$ to protect important parameters. Requires computing the Fisher information matrix, is approximate, and treats all parameters uniformly within each group.

**Fragile.** The three-pass gradient surgery provides *surgical* control over which parameters receive which gradients, without approximations:

| Parameter group | $\mathcal{L}_{\text{base}}$ | $\mathcal{L}_{z_n}$ | $\mathcal{L}_{\text{dyn}}$ | Rationale |
|---|:---:|:---:|:---:|---|
| Feature extractor, val_proj | yes | yes | **no** | Perceptual features must not be reshaped by dynamics |
| CovariantChartRouter, chart_centers | yes | yes | **no** | Chart topology is a structural invariant |
| Codebook (VQ centers) | yes | yes | yes | Interface between perception and dynamics |
| Structure filter ($z_n$) | **no** | yes | yes | Must not receive reconstruction pressure |
| World model | **no** | **no** | yes | Only dynamics-relevant gradients |

This is implemented via three backward passes with explicit gradient save/restore, not via loss weighting or regularization. The firewall is exact: protected parameters receive *zero* gradient from blocked losses, not a scaled-down version.

**Why this is stronger than EWC.** EWC adds a soft penalty proportional to the Fisher information. If the dynamics loss gradient is large enough, it can overwhelm the EWC penalty and shift protected parameters. Fragile's firewall is not a penalty: it is an exact gradient replacement. After Pass 3, the protected parameters' gradients are overwritten with the saved values from Pass 2, regardless of the magnitude of the dynamics gradient. This provides a hard guarantee that the feature extractor, router, and chart centers are updated only by encoder-side losses.

**Why this is better than freezing.** A frozen encoder cannot adapt its codebook to improve dynamics prediction. Fragile's surgery allows the codebook (VQ centers within charts) and structure filter to receive dynamics gradients, enabling the perception-dynamics interface to co-adapt. The codebook is the "meeting point" between the atlas and the world model: the world model can nudge codebook centers to cluster dynamically coherent states, while the encoder-side losses prevent the codebook from drifting away from perceptually meaningful quantization.

### 6.7 Built-in physics diagnostics vs opaque loss curves

**State of the art.** Standard world models expose training loss and validation loss. When a model fails, the practitioner sees a number go up and has no mechanistic insight into *why*. Dreamer logs reconstruction loss and KL divergence; IRIS logs token prediction accuracy. Neither provides information about the *structure* of the learned dynamics.

**Fragile.** The architecture exposes three families of physics-grounded diagnostics that are unavailable in standard models:

**A. Hodge decomposition ratios.** The `HodgeDecomposer` decomposes the total force field at each timestep into conservative, solenoidal, and harmonic components, reporting their norm ratios:

- `hodge_conservative_ratio` $= \|f_{\text{cons}}\| / \|f_{\text{total}}\|$: fraction of force explained by the scalar potential gradient $-\nabla\Phi$.
- `hodge_solenoidal_ratio` $= \|f_{\text{sol}}\| / \|f_{\text{total}}\|$: fraction explained by the curl field (Boris rotation impulse).
- `hodge_harmonic_ratio` $= \|f_{\text{harm}}\| / \|f_{\text{total}}\|$: unexplained residual.

A well-trained model should have $\text{harmonic\_ratio} \approx 0$: all forces are explained by conservative + solenoidal components. A rising harmonic ratio signals that the model is learning unstructured "leftover" forces that do not decompose into physically meaningful components. This is an early warning of model degradation that no standard world model can provide.

**B. Energy conservation variance.** The Hamiltonian $H = \Phi_{\text{eff}} + \frac{1}{2} p^\top G^{-1} p$ is tracked across rollout steps. $\text{Var}(H_t)$ across the horizon measures symplecticity. A symplectic integrator preserves $H$ exactly (up to numerical precision); a non-symplectic model causes $H$ to drift. Monitoring $\text{Var}(H)$ reveals whether the integrator is breaking conservation laws, which directly predicts long-horizon rollout quality.

**C. Jump sparsity and chart transition rates.** The `CovariantJumpRate` outputs Poisson rates $\lambda(z, K)$ at each timestep, and the L1 sparsity loss encourages rare jumps. Monitoring the mean jump rate reveals whether the model is over-switching between charts (unstable macro-state assignment) or under-switching (ignoring genuine mode transitions). Standard models with discrete state variables (IRIS tokens, Dreamer discrete codes) provide no equivalent: they switch at every step with no notion of "jump rarity."

**Diagnostic interpretation table:**

| Diagnostic | Healthy range | Failure mode |
|---|---|---|
| `hodge_harmonic_ratio` | $< 0.05$ | Model learning unstructured forces |
| `Var(H_t)` over horizon | $< 10^{-3}$ | Integrator losing symplecticity |
| Mean jump rate | $< 0.1$ per step | Over-switching between charts |
| `conservative_ratio` | $> 0.7$ | Most forces should be gradient-driven |

These diagnostics transform debugging from "loss went up, try different hyperparameters" to "the harmonic ratio spiked at epoch 50, so the curl network is underfitting the solenoidal component — increase `beta_curl` or add capacity to `CovariantValueCurl`."

### Summary comparison

| Dimension | Fragile | Dreamer-v3 | IRIS | RT-2 / Genie |
|---|---|---|---|---|
| **Latent geometry** | Poincare ball ($e^{Dr}$ volume) | Flat Euclidean ($r^D$ volume) | Discrete tokens (finite) | Flat continuous |
| **Latent structure** | $z_{\text{geo}} + z_n + z_{\text{tex}}$ with firewall | $(h, z)$ monolithic | Token sequence | Single vector |
| **Attention** | Covariant (parallel transport, $\tau(z)$, Christoffel) | Standard dot-product | Standard dot-product | Standard dot-product |
| **Dynamics model** | Boris-BAOAB symplectic integrator | GRU (RSSM) | Autoregressive transformer | None / flat MLP |
| **Value function** | Screened Poisson PDE on $\mathbb{B}^D$ | TD($\lambda$) | TD($\lambda$) | Reward regression |
| **Joint training** | Three-pass gradient surgery (exact firewall) | Stop-gradient heuristics | Frozen encoder | End-to-end |
| **Diagnostics** | Hodge ratios + energy + jump rates | Loss curves only | Token accuracy only | Loss curves only |

---

This document preserves the field-theory framing while specifying the implemented hyperbolic modules, loss operators, and Phase-3 training mechanics.
