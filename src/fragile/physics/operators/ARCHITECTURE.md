# Strong-Force Companion-Channel Kernels: Architecture

## Overview

The `physics/kernels/` package provides a clean, deduplicated implementation of the
five strong-force companion-channel operator families and their temporal correlators.
Each operator module receives a single shared data structure (`PreparedChannelData`)
instead of accessing `RunHistory` directly, eliminating ~50 lines of duplicated
frame-extraction, color-state computation, and companion-index resolution per channel.

```mermaid
graph LR
    A["RunHistory"] -->|"prepare_channel_data()"| B["PreparedChannelData"]
    B --> C["compute_meson_operators()"]
    B --> D["compute_vector_operators()"]
    B --> E["compute_baryon_operators()"]
    B --> F["compute_glueball_operators()"]
    B --> G["compute_tensor_operators()"]
    C --> H["compute_correlators_batched()"]
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I["PipelineResult"]
```

---

## Module Map

```
physics/kernels/
  __init__.py            Public API re-exports
  config.py              Configuration dataclasses (base + per-channel)
  preparation.py         RunHistory -> PreparedChannelData extraction
  meson_operators.py     J=0 scalar / pseudoscalar operators
  vector_operators.py    J=1 vector / axial-vector operators
  baryon_operators.py    Nucleon (determinant-based) operators
  glueball_operators.py  Color-plaquette glueball operators
  tensor_operators.py    Spin-2 traceless tensor operators
  correlators.py         Batched FFT temporal correlators
  pipeline.py            Top-level orchestrator
```

---

## Data Flow

### Full Pipeline

The orchestrator (`pipeline.py`) drives the entire computation in three stages:

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Preparation"]
        RH["RunHistory"] -->|"frame selection\ncolor states\ncompanion arrays"| PCD["PreparedChannelData"]
    end
    subgraph Stage2["Stage 2: Operators"]
        PCD --> M["Meson\nscalar, pseudoscalar"]
        PCD --> V["Vector\nvector, axial_vector"]
        PCD --> B["Baryon\nnucleon"]
        PCD --> GB["Glueball\nglueball, momentum modes"]
        PCD --> T["Tensor\ntensor, momentum modes"]
    end
    subgraph Stage3["Stage 3: Correlators"]
        M --> CORR["Batched FFT\nCorrelators"]
        V --> CORR
        B --> CORR
        GB --> CORR
        T --> CORR
    end
    CORR --> RES["PipelineResult\n.operators\n.correlators\n.prepared_data"]
```

### Preparation Detail

`prepare_channel_data()` extracts everything operator modules need from `RunHistory`
in a single pass. Downstream modules never import or reference `RunHistory`.

```mermaid
flowchart TD
    H["RunHistory"] --> FR["_resolve_frame_indices()\nwarmup_fraction, end_fraction\nmc_time_index"]
    FR --> SL["Slice frames [start_idx : end_idx]"]
    H --> EL["estimate_ell0()"]
    EL --> CS["compute_color_states_batch()\nh_eff, mass, ell0"]
    SL --> CS
    CS --> COLOR["color [T, N, 3] complex"]
    CS --> CVAL["color_valid [T, N] bool"]
    SL --> ALIVE["alive [T, N] bool"]
    SL --> CD["companions_distance [T, N] long"]
    SL --> CC["companions_clone [T, N] long"]
    SL -->|"if need_scores"| SC["scores [T, N] float"]
    SL -->|"if need_positions"| POS["positions [T, N, 3] float"]
    SL -->|"if need_momentum_axis"| PA["positions_axis [T, N] float"]
    COLOR --> PCD["PreparedChannelData"]
    CVAL --> PCD
    ALIVE --> PCD
    CD --> PCD
    CC --> PCD
    SC --> PCD
    POS --> PCD
    PA --> PCD
```

---

## Configuration Hierarchy

All config dataclasses live in `config.py`. Channel-specific configs inherit from
`ChannelConfigBase`, which holds the shared physics and frame-selection parameters.

```mermaid
classDiagram
    class ChannelConfigBase {
        +float warmup_fraction = 0.1
        +float end_fraction = 1.0
        +int|None mc_time_index
        +float h_eff = 1.0
        +float mass = 1.0
        +float|None ell0
        +tuple|None color_dims
        +float eps = 1e-12
        +str pair_selection = "both"
    }
    class MesonOperatorConfig {
        +str operator_mode = "standard"
    }
    class VectorOperatorConfig {
        +tuple|None position_dims
        +bool use_unit_displacement = False
        +str operator_mode = "standard"
        +str projection_mode = "full"
    }
    class BaryonOperatorConfig {
        +str operator_mode = "det_abs"
        +float flux_exp_alpha = 1.0
    }
    class GlueballOperatorConfig {
        +str|None operator_mode
        +bool use_action_form = False
        +bool use_momentum_projection = False
        +int momentum_axis = 0
        +int momentum_mode_max = 3
    }
    class TensorOperatorConfig {
        +tuple|None position_dims
        +int momentum_axis = 0
        +int momentum_mode_max = 4
        +float|None projection_length
    }
    class CorrelatorConfig {
        +int max_lag = 80
        +bool use_connected = True
    }
    class PipelineConfig {
        +ChannelConfigBase base
        +MesonOperatorConfig meson
        +VectorOperatorConfig vector
        +BaryonOperatorConfig baryon
        +GlueballOperatorConfig glueball
        +TensorOperatorConfig tensor
        +CorrelatorConfig correlator
        +list|None channels
    }
    ChannelConfigBase <|-- MesonOperatorConfig
    ChannelConfigBase <|-- VectorOperatorConfig
    ChannelConfigBase <|-- BaryonOperatorConfig
    ChannelConfigBase <|-- GlueballOperatorConfig
    ChannelConfigBase <|-- TensorOperatorConfig
    PipelineConfig --> ChannelConfigBase : "base"
    PipelineConfig --> MesonOperatorConfig : "meson"
    PipelineConfig --> VectorOperatorConfig : "vector"
    PipelineConfig --> BaryonOperatorConfig : "baryon"
    PipelineConfig --> GlueballOperatorConfig : "glueball"
    PipelineConfig --> TensorOperatorConfig : "tensor"
    PipelineConfig --> CorrelatorConfig : "correlator"
```

---

## Operator Channels

### Uniform Interface

Every operator module exports a single public function with the same signature:

```python
def compute_X_operators(
    data: PreparedChannelData,
    config: XOperatorConfig,
) -> dict[str, Tensor]:
    ...
```

The returned dictionary maps channel names to time-series tensors. Scalar channels
produce `[T]` tensors; vector/tensor channels produce `[T, 3]` or `[T, 5]` tensors.

### Channel Summary

| Module | Function | Output Keys | Shape | Physics |
|--------|----------|-------------|-------|---------|
| `meson_operators` | `compute_meson_operators` | `"scalar"`, `"pseudoscalar"` | `[T]` | `Re(z_ij)`, `Im(z_ij)` |
| `vector_operators` | `compute_vector_operators` | `"vector"`, `"axial_vector"` | `[T, 3]` | `Re(z_ij) * dx`, `Im(z_ij) * dx` |
| `baryon_operators` | `compute_baryon_operators` | `"nucleon"` | `[T]` | `|det(c_i, c_j, c_k)|` |
| `glueball_operators` | `compute_glueball_operators` | `"glueball"`, `"glueball_momentum_cos_n"`, `"glueball_momentum_sin_n"` | `[T]` | `Re(Pi_i)` or `1 - Re(Pi_i)` |
| `tensor_operators` | `compute_tensor_operators` | `"tensor"`, `"tensor_momentum_cos_n"`, `"tensor_momentum_sin_n"` | `[T, 5]` | `Re(z_ij) * Q^{ab}(dx)` |

### Companion Topology

All operators are built from companion relationships between walkers. Two shared
index-construction functions underpin everything:

```mermaid
flowchart TD
    CD["companions_distance [T, N]"] --> BCP["build_companion_pair_indices()\n(meson_operators.py)"]
    CC["companions_clone [T, N]"] --> BCP
    CD --> BCT["build_companion_triplets()\n(baryon_operators.py)"]
    CC --> BCT

    BCP -->|"pair_indices [T, N, P]\nstructural_valid [T, N, P]"| PAIR["Pair-based operators\nmeson, vector, tensor"]

    BCT -->|"anchor_i [T, N]\ncompanion_j [T, N]\ncompanion_k [T, N]\nstructural_valid [T, N]"| TRIP["Triplet-based operators\nbaryon, glueball"]

    BCP -.->|"internally calls"| BCT
```

**Pairs** (used by meson, vector, tensor): Each walker `i` has up to 2 companion
partners -- the distance companion `j = companions_distance[i]` and the clone
companion `k = companions_clone[i]`. The `pair_selection` parameter controls which
partners to use (`"distance"`, `"clone"`, or `"both"`).

**Triplets** (used by baryon, glueball): Each walker `i` forms a triplet
`(i, j, k)` where `j` is the distance companion and `k` is the clone companion.
The triplet must have all three indices distinct and in-range.

### Operator Modes

Each channel supports multiple physics modes via its `operator_mode` field:

```mermaid
flowchart TD
    subgraph Meson["Meson Modes"]
        MS["standard"] --> MSR["Re(z_ij), Im(z_ij)"]
        MSD["score_directed"] --> MSDR["Orient phase uphill"]
        MSW["score_weighted"] --> MSWR["Weight by |delta_score|"]
        MA["abs2_vacsub"] --> MAR["|z_ij|^2 vacuum-subtracted"]
    end

    subgraph Vector["Vector Modes"]
        VS["standard"] --> VSR["Re/Im(z) * dx"]
        VSD["score_directed"] --> VSDR["Orient phase, full dx"]
        VSG["score_gradient"] --> VSGR["Replace dx with score gradient"]
    end

    subgraph Baryon["Baryon Modes"]
        BD["det_abs"] --> BDR["|det(c_i, c_j, c_k)|"]
        BFA["flux_action"] --> BFAR["|det| * (1 - cos(phase))"]
        BFS["flux_sin2"] --> BFSR["|det| * sin^2(phase)"]
        BFE["flux_exp"] --> BFER["|det| * exp(alpha * action)"]
        BSS["score_signed"] --> BSSR["Re(det) score-ordered"]
        BSA["score_abs"] --> BSAR["|det| score-ordered"]
    end

    subgraph Glueball["Glueball Modes"]
        GR["re_plaquette"] --> GRR["Re(Pi_i)"]
        GA["action_re_plaquette"] --> GAR["1 - Re(Pi_i)"]
        GP["phase_action"] --> GPR["1 - cos(angle(Pi_i))"]
        GS["phase_sin2"] --> GSR["sin^2(angle(Pi_i))"]
    end
```

---

## Correlator Engine

`compute_correlators_batched()` processes all operator time-series in a single
pass through `_fft_correlator_batched` (imported from `fractalai.qft`).

```mermaid
flowchart LR
    subgraph Input["Operator Series"]
        S1["scalar [T]"]
        S2["pseudoscalar [T]"]
        S3["nucleon [T]"]
        S4["glueball [T]"]
        V1["vector [T, 3]"]
        V2["axial_vector [T, 3]"]
        T1["tensor [T, 5]"]
    end

    subgraph Batch["Batching"]
        S1 --> SB["Scalar batch [B_s, T]"]
        S2 --> SB
        S3 --> SB
        S4 --> SB
        V1 -->|"split 3 components"| MB["Multi-comp batch [B_m, T]"]
        V2 -->|"split 3 components"| MB
        T1 -->|"split 5 components"| MB
    end

    subgraph FFT["FFT Correlators"]
        SB -->|"single FFT call"| SC["Scalar correlators [B_s, L]"]
        MB -->|"single FFT call"| MC["Component correlators [B_m, L]"]
    end

    subgraph Contract["Contraction"]
        SC --> OUT["Per-channel correlators [L]"]
        MC -->|"sum over components\nC(tau) = sum_mu C_mu(tau)"| OUT
    end
```

- **Scalar channels** (`[T]`): stacked into one batch, one FFT call, results indexed back.
- **Multi-component channels** (`[T, C]`): each component is a separate series in the
  FFT batch. After computing per-component correlators, they are summed
  (dot-product contraction): `C(tau) = sum_mu <O_mu(t) O_mu(t+tau)>`.
- **Connected correlators**: when `use_connected=True`, the mean is subtracted before
  the FFT, yielding the connected two-point function.

---

## Internal Dependencies

```mermaid
flowchart TD
    INIT["__init__.py"] --> CONFIG["config.py"]
    INIT --> PREP["preparation.py"]
    INIT --> MESON["meson_operators.py"]
    INIT --> VECTOR["vector_operators.py"]
    INIT --> BARYON["baryon_operators.py"]
    INIT --> GLUEBALL["glueball_operators.py"]
    INIT --> TENSOR["tensor_operators.py"]
    INIT --> CORR["correlators.py"]
    INIT --> PIPE["pipeline.py"]

    PREP --> CONFIG
    MESON --> CONFIG
    MESON --> PREP
    MESON -.->|"lazy import"| BARYON
    VECTOR --> CONFIG
    VECTOR --> MESON
    VECTOR --> PREP
    BARYON --> CONFIG
    BARYON --> PREP
    GLUEBALL --> CONFIG
    GLUEBALL --> BARYON
    GLUEBALL --> PREP
    TENSOR --> CONFIG
    TENSOR --> MESON
    TENSOR --> PREP

    PIPE --> CONFIG
    PIPE --> PREP
    PIPE --> CORR
    PIPE -.->|"lazy import"| MESON
    PIPE -.->|"lazy import"| VECTOR
    PIPE -.->|"lazy import"| BARYON
    PIPE -.->|"lazy import"| GLUEBALL
    PIPE -.->|"lazy import"| TENSOR

    subgraph External["External Imports"]
        AGG["fractalai.qft.aggregation\ncompute_color_states_batch\nestimate_ell0"]
        FFT["fractalai.qft.correlator_channels\n_fft_correlator_batched"]
        RAD["fractalai.qft.radial_channels\n_apply_pbc_diff_torch\n_slice_bounds"]
        HIST["fractalai.core.history\nRunHistory"]
    end

    PREP --> AGG
    PREP --> HIST
    CORR --> FFT
    TENSOR --> RAD
    PIPE --> HIST
```

Dashed arrows indicate lazy imports (only resolved when the code path is actually
executed), used to avoid circular dependencies and to keep imports fast when only
a subset of channels is needed.

---

## Shared Utilities

Several helper functions are defined once and reused across modules:

| Function | Defined in | Used by |
|----------|-----------|---------|
| `_safe_gather_2d` | `preparation.py` | `meson_operators`, `baryon_operators`, `glueball_operators`, `tensor_operators` |
| `_safe_gather_3d` | `preparation.py` | `meson_operators`, `baryon_operators`, `glueball_operators`, `tensor_operators` |
| `_safe_gather_pairs_2d` | `meson_operators.py` | `vector_operators`, `tensor_operators` |
| `_safe_gather_pairs_3d` | `meson_operators.py` | `vector_operators`, `tensor_operators` |
| `build_companion_pair_indices` | `meson_operators.py` | `vector_operators`, `tensor_operators`, `pipeline` |
| `build_companion_triplets` | `baryon_operators.py` | `meson_operators` (lazy), `glueball_operators`, `pipeline` |
| `_resolve_frame_indices` | `preparation.py` | `prepare_channel_data` |
| `_resolve_3d_dims` | `preparation.py` | `prepare_channel_data` |
| `_det3` | `baryon_operators.py` | `baryon_operators` internal |

---

## Momentum Projection

Glueball and tensor channels support Fourier momentum projection along a spatial axis.
For each mode `n = 0, 1, ..., n_max`:

```
k_n = 2 * pi * n / L

O_cos_n(t) = <O_i(t) * cos(k_n * x_i)>_i
O_sin_n(t) = <O_i(t) * sin(k_n * x_i)>_i
```

where `L` is the projection length (box size along the momentum axis or the
observed span), and the average is over valid walkers weighted by their validity masks.

These projected series are returned as additional dictionary entries
(`"glueball_momentum_cos_0"`, `"tensor_momentum_sin_2"`, etc.) and flow
into the correlator engine like any other operator series.

---

## Relation to Legacy Code

This package is a clean re-implementation of logic previously scattered across:

| Legacy module | Kernels equivalent |
|--------------|-------------------|
| `physics/operators/meson_phase_channels.py` | `meson_operators.py` |
| `physics/operators/vector_meson_channels.py` | `vector_operators.py` |
| `physics/operators/baryon_triplet_channels.py` | `baryon_operators.py` + `preparation.py` utilities |
| `physics/operators/glueball_color_channels.py` | `glueball_operators.py` |
| `physics/operators/tensor_momentum_channels.py` | `tensor_operators.py` |
| `physics/operators/aggregation.py` | Referenced, not copied (imported from `fractalai.qft`) |
| `fractalai/qft/correlator_channels.py` | `correlators.py` (wraps `_fft_correlator_batched`) |

The legacy modules remain intact. No existing code was deleted or modified.

**Key differences from legacy:**

1. **No duplicated frame extraction.** Each legacy module independently sliced
   `RunHistory` arrays and computed color states. Now done once in `preparation.py`.
2. **No per-module correlator loops.** Each legacy module had its own temporal-lag
   loop computing correlators. Now all series go through a single batched FFT in
   `correlators.py`.
3. **Uniform interface.** All operator modules accept `PreparedChannelData` and
   return `dict[str, Tensor]`, making them interchangeable and composable.
4. **Single entry point.** `compute_strong_force_pipeline()` runs the full analysis
   with one function call and one config object.
