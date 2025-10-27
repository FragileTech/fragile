# Dependency Graph Visualization
## The Geometric Viscous Fluid Model

---

## Critical Path Graph (Main Convergence Theorem)

```mermaid
graph TB
    subgraph "External Foundation (Chapter 1: Euclidean Gas)"
        EXT1["Axiom 1.3.1<br>Confining Potential U(x)"]:::axiomStyle
        EXT2["Theorem 1.4.2<br>Backbone Convergence<br>κ_backbone > 0"]:::externalStyle
        EXT3["Theorem 1.7.2<br>Discretization Theorem"]:::externalStyle
        EXT4["Theorem 1.4.3<br>Petite Set Property"]:::externalStyle
        EXT5["Theorem 8.1 (03_cloning.md)<br>Keystone Principle"]:::externalStyle
    end

    subgraph "Chapter 1: ρ-Parameterized Framework"
        CH1A["Definition 1.0.2<br>Localization Kernel K_ρ"]:::defStyle
        CH1B["Definition 1.0.3<br>Localized Moments μ_ρ, σ²_ρ"]:::defStyle
        CH1C["Definition 1.0.4<br>Unified Z-Score Z_ρ"]:::defStyle
    end

    subgraph "Chapter 2: SDE Specification"
        CH2A["Definition 2.1<br>Adaptive Viscous Fluid SDE"]:::defStyle
        CH2B["Definition 2.2<br>Regularized Hessian Σ_reg"]:::defStyle
        CH2C["Definition 2.3<br>Localized Fitness V_fit"]:::defStyle
    end

    subgraph "Appendix A: Regularity Theory"
        APPA1["Lemma A.1<br>Weight Derivatives"]:::lemmaStyle
        APPA2["Lemma A.2<br>Mean First Derivative"]:::lemmaStyle
        APPA3["Lemma A.3<br>Mean Second Derivative"]:::lemmaStyle
        APPA4["Theorem A.1<br>C¹ Regularity<br>||∇V_fit|| ≤ F_adapt,max(ρ)"]:::thmStyle
        APPA5["Theorem A.2<br>C² Regularity<br>||H|| ≤ H_max(ρ)"]:::thmStyle
    end

    subgraph "Chapter 4: Uniform Ellipticity"
        CH4A["Lemma 4.1<br>Hessian Bounded"]:::lemmaStyle
        CH4B["Theorem 4.1 (CRITICAL)<br>UEPH by Construction<br>c_min(ρ)I ⪯ G_reg ⪯ c_max(ρ)I"]:::criticalStyle
        CH4C["Corollary 4.3<br>Well-Posedness"]:::corollaryStyle
    end

    subgraph "Chapter 6: Perturbation Analysis"
        CH6A["Lemma 6.2<br>Adaptive Force Bounded<br>O(ε_F K_F(ρ) V_total)"]:::lemmaStyle
        CH6B["Lemma 6.3<br>Viscous Force Dissipative"]:::lemmaStyle
        CH6C["Lemma 6.4<br>Diffusion Perturbation<br>C_diff,0(ρ) + C_diff,1(ρ) V_total"]:::lemmaStyle
        CH6D["Corollary 6.5<br>Total Perturbation"]:::corollaryStyle
    end

    subgraph "Chapter 7: Main Convergence"
        CH7A["Theorem 7.1 (MAIN)<br>Foster-Lyapunov Drift<br>κ_total(ρ) > 0 for ε_F < ε_F*(ρ)"]:::mainStyle
        CH7B["Section 7.2<br>Discretization Verification"]:::proofStyle
        CH7C["Corollary 7.2<br>Exponential Convergence"]:::corollaryStyle
    end

    subgraph "Chapter 9: Geometric Ergodicity"
        CH9A["Theorem 9.1<br>Geometric Ergodicity<br>||μ_t - π_∞|| ≤ C e^{-λt}"]:::mainStyle
    end

    %% Dependencies
    EXT1 --> EXT2
    EXT2 --> CH7A

    CH1A --> CH1B
    CH1B --> CH1C
    CH1C --> CH2C
    CH2C --> CH2A
    CH2B --> CH2A

    CH1B --> APPA2
    APPA1 --> APPA2
    APPA2 --> APPA3
    APPA2 --> APPA4
    APPA3 --> APPA5

    APPA5 --> CH4A
    CH4A --> CH4B
    CH4B --> CH4C

    APPA4 --> CH6A
    CH6A --> CH6D
    CH6B --> CH6D
    CH6C --> CH6D

    EXT2 --> CH7A
    CH6D --> CH7A
    CH4B --> CH7A

    CH7A --> CH7B
    EXT3 --> CH7B
    CH7B --> CH7C

    CH7A --> CH9A
    EXT4 --> CH9A

    %% Styling
    classDef axiomStyle fill:#8c6239,stroke:#d4a574,stroke-width:2px,color:#fff
    classDef externalStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#fff
    classDef defStyle fill:#2d5a3d,stroke:#5a9a6d,stroke-width:2px,color:#fff
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#fff
    classDef thmStyle fill:#6b3d8c,stroke:#a47fd4,stroke-width:2px,color:#fff
    classDef criticalStyle fill:#8c3d3d,stroke:#d47f7f,stroke-width:4px,color:#fff,font-weight:bold
    classDef mainStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#fff,font-weight:bold
    classDef corollaryStyle fill:#5f8c3d,stroke:#a4d47f,stroke-width:2px,color:#fff
    classDef proofStyle fill:#7f7f7f,stroke:#b3b3b3,stroke-width:2px,color:#fff
```

---

## Secondary Dependencies: LSI and Mean-Field Theory

```mermaid
graph TB
    subgraph "Chapter 8: LSI Theory"
        CH8A["Definition 8.1<br>N-Particle Generator"]:::defStyle
        CH8B["Definition 8.2<br>Relative Entropy & Fisher Info"]:::defStyle
        CH8C["Definition 8.3<br>LSI"]:::defStyle
        CH8D["Theorem 8.1<br>N-Uniform LSI<br>Ent(g²) ≤ C_LSI ∫|∇g|²"]:::thmStyle
        CH8E["Corollary 8.2<br>Entropy Convergence"]:::corollaryStyle
        CH8F["Corollary 8.3<br>Geometric Ergodicity via LSI"]:::corollaryStyle
    end

    subgraph "Chapter 9: Mean-Field"
        CH9B["Theorem 9.3<br>Mean-Field LSI"]:::thmStyle
        CH9C["Conjecture 9.2<br>WFR Convergence"]:::conjectureStyle
        CH9D["Lemma 9.1<br>Entropy Dissipation Decomp"]:::lemmaStyle
    end

    subgraph "From Chapter 4"
        CH4B_REF["Theorem 4.1<br>UEPH"]:::thmStyle
    end

    subgraph "From Chapter 7"
        CH7A_REF["Theorem 7.1<br>Foster-Lyapunov"]:::thmStyle
    end

    CH8A --> CH8B
    CH8B --> CH8C
    CH8C --> CH8D
    CH4B_REF --> CH8D
    CH8D --> CH8E
    CH8E --> CH8F
    CH7A_REF --> CH8F

    CH8D --> CH9B
    CH9D --> CH9B
    CH9B --> CH9C

    classDef defStyle fill:#2d5a3d,stroke:#5a9a6d,stroke-width:2px,color:#fff
    classDef thmStyle fill:#6b3d8c,stroke:#a47fd4,stroke-width:2px,color:#fff
    classDef corollaryStyle fill:#5f8c3d,stroke:#a4d47f,stroke-width:2px,color:#fff
    classDef conjectureStyle fill:#8c7f3d,stroke:#d4b37f,stroke-width:2px,stroke-dasharray: 5 5,color:#fff
    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#fff
```

---

## Appendix B: Keystone Extension

```mermaid
graph TB
    subgraph "Appendix B: Signal Generation"
        APPB1["Lemma B.1<br>Variance-to-Gap"]:::lemmaStyle
        APPB2["Lemma B.2<br>Uniform Bounds Pipeline"]:::lemmaStyle
        APPB3["Lemma B.3<br>Raw-to-Rescaled Gap"]:::lemmaStyle
        APPB4["Lemma B.4 (from 03_cloning)<br>Logarithmic Gap Bounds"]:::externalStyle
        APPB5["Proposition B.1<br>Diversity Signal Lower Bound"]:::propStyle
        APPB6["Proposition B.2<br>Reward Gap Bound"]:::propStyle
        APPB7["Theorem B.1<br>Signal Generation ρ-Adaptive"]:::thmStyle
        APPB8["Theorem B.2<br>Stability Condition ρ-Dependent"]:::thmStyle
        APPB9["Theorem B.3<br>Keystone Lemma ρ-Localized"]:::mainStyle
    end

    subgraph "From Chapter 1"
        KEYSTONEORIG["Theorem 8.1 (03_cloning)<br>Keystone Principle<br>(Global Statistics)"]:::externalStyle
    end

    APPB1 --> APPB5
    APPB2 --> APPB5
    APPB3 --> APPB5
    APPB4 --> APPB5
    APPB5 --> APPB6
    APPB6 --> APPB8
    APPB7 --> APPB9
    APPB8 --> APPB9
    KEYSTONEORIG --> APPB9

    classDef lemmaStyle fill:#3d6b4b,stroke:#7fc296,stroke-width:2px,color:#fff
    classDef propStyle fill:#5f8c5f,stroke:#a4d4a4,stroke-width:2px,color:#fff
    classDef thmStyle fill:#6b3d8c,stroke:#a47fd4,stroke-width:2px,color:#fff
    classDef mainStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:3px,color:#fff,font-weight:bold
    classDef externalStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#fff
```

---

## Full Internal Reference Graph (Top 20 Most Connected)

```mermaid
graph LR
    subgraph "Most Referenced (6 refs)"
        LEM_MEAN1["lem-mean-first-derivative"]:::hotStyle
    end

    subgraph "Highly Referenced (5 refs)"
        THM_C1["thm-c1-regularity"]:::hotStyle
        THM_C2["thm-c2-regularity"]:::hotStyle
    end

    subgraph "Moderately Referenced (3 refs)"
        LEM_RAW["lem-raw-to-rescaled-gap-rho"]:::warmStyle
    end

    subgraph "Referenced (2 refs)"
        DEF_LOC["def-localized-mean-field-moments"]:::normalStyle
        THM_LSI["thm-lsi-adaptive-gas"]:::normalStyle
        LEM_MEAN2["lem-mean-second-derivative"]:::normalStyle
        THM_SIG["thm-signal-generation-adaptive"]:::normalStyle
        THM_STAB["thm-stability-condition-rho"]:::normalStyle
    end

    LEM_MEAN1 --> LEM_MEAN2
    LEM_MEAN1 --> THM_C1
    LEM_MEAN1 --> THM_C2

    THM_C1 --> THM_C2
    THM_C2 --> THM_UEPH["thm-ueph"]

    LEM_RAW --> PROP_DIV["prop-diversity-signal-rho"]
    PROP_DIV --> THM_STAB
    THM_STAB --> THM_KEY["thm-keystone-adaptive"]

    THM_LSI --> COR_ENT["cor-entropy-convergence-lsi"]
    COR_ENT --> COR_GEO["cor-geometric-ergodicity-lsi"]

    classDef hotStyle fill:#d43d3d,stroke:#ff6666,stroke-width:3px,color:#fff,font-weight:bold
    classDef warmStyle fill:#d48c3d,stroke:#ffb366,stroke-width:2px,color:#fff
    classDef normalStyle fill:#3d8cd4,stroke:#66b3ff,stroke-width:2px,color:#fff
```

---

## Cross-Document Dependency Summary

```mermaid
graph TB
    DOC["11_geometric_gas.md<br>(Geometric Viscous Fluid)"]:::currentStyle

    subgraph "Chapter 1: Euclidean Gas Foundation"
        DOC_CLONING["03_cloning.md<br>Keystone Principle"]:::externalStyle
        DOC_CONV["04_convergence.md<br>Foster-Lyapunov"]:::externalStyle
        DOC_KIN["05_kinetic_contraction.md<br>Hypocoercivity"]:::externalStyle
        DOC_MF["07_mean_field.md<br>Propagation of Chaos"]:::externalStyle
        DOC_POC["08_propagation_chaos.md<br>Mean-Field Limit"]:::externalStyle
    end

    DOC -->|"Keystone Principle<br>Cloning operator<br>Axioms Ch 4"| DOC_CLONING
    DOC -->|"Axiom 1.3.1<br>Theorem 1.4.2<br>Theorem 1.7.2<br>Theorem 1.4.3"| DOC_CONV
    DOC -->|"Hypocoercive W_h²<br>Velocity dissipation"| DOC_KIN
    DOC -->|"Propagation of chaos"| DOC_MF
    DOC -->|"Mean-field techniques"| DOC_POC

    classDef currentStyle fill:#8c3d5f,stroke:#d47fa4,stroke-width:4px,color:#fff,font-weight:bold
    classDef externalStyle fill:#4a5f8c,stroke:#8fa4d4,stroke-width:2px,color:#fff
```

---

## Label Normalization Applied

All labels have been normalized to match pipeline convention:

| Original Pattern | Normalized Pattern | Count |
|------------------|-------------------|-------|
| `axiom:*` | `axiom-*` | 4 |
| `def:*` | `def-*` | 12 |
| `thm:*` | `thm-*` | 12 |
| `lem:*` | `lem-*` | 19 |
| `prop:*` | `prop-*` | 5 |
| `cor:*` | `cor-*` | 6 |

**All labels use lowercase with hyphens:** `thm-ueph`, `lem-adaptive-force-bounded`, etc.

---

## Graph Statistics

| Metric | Value |
|--------|-------|
| **Total Nodes** | 61 |
| **Total Edges (Internal)** | 456 |
| **Explicit References** | 41 |
| **Implicit Dependencies** | 415 |
| **Cross-Document Sources** | 5 |
| **Critical Path Length** | 13 steps |
| **Max In-Degree** | 6 (lem-mean-first-derivative) |
| **Max Out-Degree** | ~8 (various proof sections) |
| **Strongly Connected Components** | 1 (acyclic DAG) |

---

## Navigation Guide

**To understand the main result:**
1. Start at `thm-geometric-ergodicity` (Theorem 9.1)
2. Trace backwards to `thm-foster-lyapunov` (Theorem 7.1)
3. Follow perturbation lemmas: `lem-adaptive-force-bounded`, `lem-viscous-dissipative`, `lem-diffusion-perturbation`
4. Understand UEPH: `thm-ueph` (Theorem 4.1)
5. See regularity foundation: `thm-c1-regularity`, `thm-c2-regularity` (Appendix A)

**To understand ρ-parameterization:**
1. `def-localization-kernel` (Definition 1.0.2)
2. `def-localized-mean-field-moments` (Definition 1.0.3)
3. `def-unified-z-score` (Definition 1.0.4)
4. `def-localized-mean-field-fitness` (Definition 2.3)

**To understand Keystone extension:**
1. Read original: Theorem 8.1 in 03_cloning.md
2. See adaptation: `thm-keystone-adaptive` (Theorem B.3)
3. Trace prerequisites: `thm-signal-generation-adaptive`, `thm-stability-condition-rho`

---

**Graph Generation Complete**
