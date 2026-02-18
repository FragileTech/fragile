# Companion Strong Force Channels: Code Path Report

## Overview

The "Companion Strong Force" tab in `new_dashboard.py` computes particle correlators
using **companion-based operators** (walker pairs/triplets from the Fractal Gas topology)
rather than the older direct-edge operators. These companion operators override the
defaults for the strong force channels.

The entire pipeline from button click to displayed correlators involves:

1. **UI callback** triggers `_compute_companion_strong_force_bundle()`
2. That function dispatches to **5 channel families**, each with its own module
3. Each module computes an **operator time series** from `RunHistory` data
4. The operator series is turned into a **correlator** via FFT
5. The correlator is fitted to extract a **mass** via AIC window averaging
6. Optionally, **multiscale smearing** re-runs the correlators at multiple scales

---

## Files Involved (copied verbatim into this folder)

| File | Role |
|------|------|
| `baryon_triplet_channels.py` | Nucleon (baryon) channel: 3x3 determinant from companion triplets |
| `meson_phase_channels.py` | Scalar + pseudoscalar channels: Re/Im of color inner product from companion pairs |
| `vector_meson_channels.py` | Vector + axial-vector channels: color phase x displacement from companion pairs |
| `glueball_color_channels.py` | Glueball channel: plaquette trace from companion triplets |
| `tensor_momentum_channels.py` | Tensor (J=2) channel: traceless symmetric tensor with momentum projection |
| `correlator_channels.py` | FFT correlator engine + AIC mass extraction (shared by all channels) |
| `multiscale_strong_force.py` | Multi-scale smearing wrapper that re-runs companion channels at different kernel widths |
| `gevp_channels.py` | GEVP (Generalized Eigenvalue Problem) combination of multiple companion channels |

The orchestrating code lives in **`new_dashboard.py`** (not copied, too large) in these key functions:
- `_compute_companion_strong_force_bundle()` (line ~3657)
- `on_run_companion_strong_force_channels()` (line ~13117)
- Helper wrappers: `_compute_anisotropic_baryon_triplet_result()`, `_compute_anisotropic_meson_phase_results()`, `_compute_anisotropic_vector_meson_results()`, `_compute_anisotropic_glueball_color_result()`, `_compute_tensor_momentum_for_anisotropic_edge()`

---

## Complete Code Path

### Step 0: Loading Run Data

`RunHistory` is already loaded when the dashboard opens. The companion channels use
these key tensors from `history`:

- `history.positions` [T, N, D] -- walker positions across time
- `history.alive_mask` [T, N] -- which walkers are alive at each step
- `history.cloning_scores` [T, N] -- fitness scores used for h_eff auto-calibration
- `history.companions_distance` [T, N] -- nearest-neighbor companion index per walker
- `history.companions_clone` [T, N] -- clone-parent companion index per walker
- `history.delta_t`, `history.record_every` -- time step metadata
- `history.d` -- spatial dimensionality

### Step 1: Button Click -> `on_run_companion_strong_force_channels()`

**Location**: `new_dashboard.py:13117`

1. Reads selected channel variants from the UI MultiSelect widgets
2. Falls back to `DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION` if none selected
3. Calls `_compute_companion_strong_force_bundle(history, settings, requested_channels)`
4. Tags each result with `source="original_companion"`
5. Optionally runs multiscale + GEVP on top

### Step 2: `_compute_companion_strong_force_bundle()`

**Location**: `new_dashboard.py:3657`

This is the main dispatch function. It:

1. **Resolves h_eff** via `_resolve_h_eff()` (line 2935):
   - If `h_eff_mode == "auto_sigma_s"`: computes h_eff = sigma_S / 2 where sigma_S is
     the standard deviation of alive-walker cloning scores in the selected frame range
   - Otherwise uses the manual value

2. **Partitions requested channels** into 5 families:
   - Nucleon family: `{nucleon, nucleon_flux_action, nucleon_flux_sin2, nucleon_flux_exp, nucleon_score_signed, nucleon_score_abs}`
   - Meson family: `{scalar, scalar_raw, scalar_abs2_vacsub, pseudoscalar, scalar_score_directed, scalar_score_weighted, pseudoscalar_score_weighted}`
   - Vector family: `{vector, axial_vector, vector_score_directed, vector_score_directed_longitudinal, ...}`
   - Glueball family: `{glueball, glueball_phase_action, glueball_phase_sin2}`
   - Tensor family: `{tensor}`

3. **Dispatches** to the appropriate compute function for each non-empty family

4. **Returns** `AnisotropicEdgeChannelOutput` with all results merged

### Step 3: Per-Family Computation (inside dashboard wrappers)

Each family wrapper in the dashboard (e.g., `_compute_anisotropic_baryon_triplet_result`)
does three things:

1. **Builds a Config** dataclass from the UI settings
2. **Calls the channel module's main function** (e.g., `compute_companion_baryon_correlator`)
3. **Wraps the result** via `_build_result_from_precomputed_correlator()`

#### 3a. Nucleon -- `_compute_anisotropic_baryon_triplet_result()` (line 2968)

- Builds `BaryonTripletCorrelatorConfig` with operator_mode variants
- Calls `compute_companion_baryon_correlator(history, config)` from `baryon_triplet_channels.py`
- **What the operator computes**:
  - For each alive walker i at frame t, finds companions j = companions_distance[t,i], k = companions_clone[t,i]
  - Extracts 3D color vectors c_i, c_j, c_k from positions (using `color_dims`)
  - Computes det[c_i | c_j | c_k] -- the 3x3 determinant
  - Operator value O(t) = mean over all valid triplets of |det| (or variants: flux_action, flux_sin2, etc.)
- Returns correlator C(lag) = <O(t) O(t+lag)> via FFT

#### 3b. Meson (scalar/pseudoscalar) -- `_compute_anisotropic_meson_phase_results()` (line 3030)

- Builds `MesonPhaseCorrelatorConfig` with operator_mode variants
- Calls `compute_companion_meson_phase_correlator(history, config)` from `meson_phase_channels.py`
- **What the operator computes**:
  - For each alive walker i, finds companion j (distance and/or clone)
  - Computes color inner product z_ij = c_i^dagger * c_j (complex-valued via color dims)
  - Scalar operator: Re(z_ij), Pseudoscalar operator: Im(z_ij)
  - Multiple modes: "standard", "score_directed" (weighted by fitness gradient), "score_weighted", "abs2_vacsub"
- Returns dual correlators for scalar and pseudoscalar simultaneously

#### 3c. Vector/Axial -- `_compute_anisotropic_vector_meson_results()` (line 3294)

- Builds `VectorMesonCorrelatorConfig` with 5 variant specs:
  - (standard, full), (score_directed, full), (score_gradient, full),
  - (score_directed, longitudinal), (score_directed, transverse)
- Calls `compute_companion_vector_meson_correlator(history, config)` from `vector_meson_channels.py`
- **What the operator computes**:
  - For each pair (i, j), computes dx_ij = x_j - x_i (displacement 3-vector)
  - Color phase z_ij same as meson channel
  - Vector operator: V(t) = Re(z_ij) * dx_ij [3-vector per frame]
  - Axial operator: A(t) = Im(z_ij) * dx_ij [3-vector per frame]
  - Correlator is dot-product: C(lag) = <V(t) . V(t+lag)>
- Optional longitudinal/transverse projections decompose along score gradient

#### 3d. Glueball -- `_compute_anisotropic_glueball_color_result()` (line 3402)

- Builds `GlueballColorCorrelatorConfig` with operator_mode variants
- Calls `compute_companion_glueball_color_correlator(history, config)` from `glueball_color_channels.py`
- **What the operator computes**:
  - For each triplet (i, j, k), computes the plaquette:
    Pi_i = (c_i^dag c_j)(c_j^dag c_k)(c_k^dag c_i)
  - This is the trace of a closed color loop -- the gauge-invariant Wilson loop analog
  - Glueball operator: Re(Pi) or 1-Re(Pi) (action form)
  - Optional momentum projection: O_n(t) = sum_i O_i(t) * exp(-i k_n x_i) via FFT
- Variants: "phase_action", "phase_sin2"

#### 3e. Tensor -- `_compute_tensor_momentum_for_anisotropic_edge()` (line 3927)

- Builds `TensorMomentumCorrelatorConfig`
- Calls `compute_companion_tensor_momentum_correlator(history, config)` from `tensor_momentum_channels.py`
- **What the operator computes**:
  - For each pair (i, j), builds the traceless symmetric tensor Q^{ab}(dx_ij):
    5 independent components: q_xy, q_xz, q_yz, q_{xx-yy}, q_{2zz-xx-yy}
  - Weights by Re(z_ij) (color singlet factor)
  - Momentum projection: O_{n,alpha}(t) = sum_i O_{i,alpha}(t) * exp(-i k_n x_i)
  - Contracted correlator: C_n(lag) = sum_alpha <O_{n,alpha}(t) O_{n,alpha}(t+lag)>
- The p=0 mode result is used as the "tensor" channel in the companion strong force bundle

### Step 4: Correlator Computation (shared)

All channels eventually produce an operator time series O(t) [shape: T or T,D].
The correlator is computed in `correlator_channels.py`:

1. **`_fft_correlator_batched(series, max_lag, use_connected)`** (line ~454):
   - Zero-pads the series to 2*T
   - FFT -> multiply by conjugate -> inverse FFT
   - Normalizes by overlap count at each lag
   - If `use_connected=True`: subtracts <O>^2 (disconnected part)
   - Returns C(lag) for lag = 0, 1, ..., max_lag

2. **`_build_result_from_precomputed_correlator()`** in the dashboard (line 2799):
   - Takes the precomputed correlator C(lag)
   - Computes effective mass: m_eff(t) = log(C(t)/C(t+1)) / dt
   - Extracts mass via AIC window averaging or linear fit
   - Packages into `ChannelCorrelatorResult`

### Step 5: Mass Extraction (AIC)

The `CorrelatorConfig.fit_mode` (default "aic") controls how mass is extracted:

- **AIC mode** (`extract_mass_aic` in `correlator_channels.py`):
  - Uses `ConvolutionalAICExtractor` which fits log(C(t)) = -m*t + const
    across many window positions and widths simultaneously via 1D convolutions
  - Computes AIC weight for each (start, width) combination
  - Returns AIC-weighted average mass with uncertainty

- **Linear mode**: Simple linear regression on log(C(t)) in a fixed window

### Step 6: Multiscale (Optional)

If `use_multiscale_kernels=True`, `compute_multiscale_strong_force_channels()` from
`multiscale_strong_force.py` runs the companion channels at N different smearing scales:

1. Generates kernel scales from the neighbor-distance distribution quantiles
2. For each scale, applies Gaussian/exponential/tophat kernel to smooth color states
3. Re-computes all requested companion correlators at that scale
4. Selects the best scale per channel based on AIC quality / R^2 / error criteria

The multiscale results are tagged with `source="multiscale_best"` and merged into the
results dictionary alongside the originals.

### Step 7: GEVP (Optional)

If `use_companion_nucleon_gevp=True`, the `gevp_channels.py` module combines multiple
operator variants for the same physical channel into a GEVP matrix and extracts the
ground-state eigenvalue, improving mass extraction by using operator diversity.

---

## Data Flow Diagram

```
RunHistory
    |
    v
_resolve_h_eff() --> h_eff (auto or manual)
    |
    v
_compute_companion_strong_force_bundle()
    |
    +---> _compute_anisotropic_baryon_triplet_result()
    |         |
    |         +--> compute_companion_baryon_correlator()     [baryon_triplet_channels.py]
    |         |        uses: companions_distance, companions_clone, positions (color_dims)
    |         |        computes: det[c_i, c_j, c_k] per triplet --> O_baryon(t)
    |         |        returns: BaryonTripletCorrelatorOutput with .correlator
    |         |
    |         +--> _build_result_from_precomputed_correlator()
    |                  computes: effective_mass, AIC mass fit
    |                  returns: ChannelCorrelatorResult("nucleon")
    |
    +---> _compute_anisotropic_meson_phase_results()
    |         |
    |         +--> compute_companion_meson_phase_correlator() [meson_phase_channels.py]
    |         |        uses: companions, positions (color_dims)
    |         |        computes: z_ij = c_i^dag c_j --> Re(z) = scalar, Im(z) = pseudoscalar
    |         |        returns: MesonPhaseCorrelatorOutput with .scalar, .pseudoscalar
    |         |
    |         +--> _build_result_from_precomputed_correlator() x N variants
    |                  returns: ChannelCorrelatorResult("scalar"), ("pseudoscalar"), etc.
    |
    +---> _compute_anisotropic_vector_meson_results()
    |         |
    |         +--> compute_companion_vector_meson_correlator() [vector_meson_channels.py]
    |         |        uses: companions, positions (color_dims + position_dims)
    |         |        computes: Re/Im(z_ij) * (x_j - x_i) --> vector/axial 3-vectors
    |         |        correlator: <V(t).V(t+lag)>
    |         |        returns: VectorMesonCorrelatorOutput
    |         |
    |         +--> _build_result_from_precomputed_correlator() x N variants
    |
    +---> _compute_anisotropic_glueball_color_result()
    |         |
    |         +--> compute_companion_glueball_color_correlator() [glueball_color_channels.py]
    |         |        uses: companions (triplets), positions (color_dims)
    |         |        computes: plaquette = (c_i^dag c_j)(c_j^dag c_k)(c_k^dag c_i)
    |         |        returns: GlueballColorCorrelatorOutput
    |         |
    |         +--> _build_result_from_precomputed_correlator()
    |
    +---> _compute_tensor_momentum_for_anisotropic_edge()
              |
              +--> compute_companion_tensor_momentum_correlator() [tensor_momentum_channels.py]
              |        uses: companions (pairs), positions (color_dims + position_dims)
              |        computes: traceless tensor Q^{ab}(dx) weighted by Re(z_ij)
              |        momentum projection via FFT
              |        returns: TensorMomentumCorrelatorOutput
              |
              +--> _build_result_from_precomputed_correlator() per momentum mode
    |
    v
results: dict[str, ChannelCorrelatorResult]
    |
    +---> (optional) compute_multiscale_strong_force_channels()  [multiscale_strong_force.py]
    +---> (optional) compute_companion_channel_gevp()            [gevp_channels.py]
    |
    v
Final results merged into state["companion_strong_force_results"]
```

---

## Default Channel Selection

`DEFAULT_COMPANION_CHANNEL_VARIANT_SELECTION` (dashboard line 203) defines which
channels are computed by default when no UI selection is made:

```python
{
    "pseudoscalar": ("pseudoscalar", "pseudoscalar_score_weighted"),
    "scalar": ("scalar", "scalar_raw", "scalar_abs2_vacsub"),
    "vector": ("vector", "vector_score_directed", "vector_score_gradient",
               "vector_score_directed_longitudinal", "vector_score_directed_transverse"),
    "axial_vector": ("axial_vector",),
    "nucleon": ("nucleon", "nucleon_flux_action", "nucleon_flux_sin2",
                "nucleon_flux_exp", "nucleon_score_signed", "nucleon_score_abs"),
    "glueball": ("glueball", "glueball_phase_action", "glueball_phase_sin2"),
    "tensor": ("tensor",),
}
```

This is **26 channel variants** computed by default across all 5 families.

---

## Key Shared Infrastructure

### Color State Computation

All companion channels extract "color vectors" from walker positions. The color
dimensions are selected via `color_dims = (d1, d2, d3)` -- three of the D spatial
dimensions are interpreted as an SU(3)-like color space. The inner product
`c_i^dag c_j` gives a complex number whose real/imaginary parts encode
scalar/pseudoscalar quantum numbers.

### Companion Topology

Companions are the nearest-neighbor graph of the Fractal Gas. Each walker i has:
- `companions_distance[t, i]`: index of geometrically nearest alive walker
- `companions_clone[t, i]`: index of the walker i was cloned from (or cloned to)

The `pair_selection` parameter controls which companions are used:
"distance", "clone", or "both" (default).

### Frame Selection

All channels use `warmup_fraction` and `end_fraction` to select the MC time range,
and `mc_time_index` for resampling/thinning. The helper `_resolve_baryon_frame_indices()`
converts these to concrete frame indices into the RunHistory tensors.
