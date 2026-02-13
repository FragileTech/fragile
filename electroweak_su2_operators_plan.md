# Electroweak SU(2) Operator Revamp (Directional + Walker-Type Split)

## Goal
Implement SU(2)-aware electroweak operators that:
- respect fitness-score directionality,
- expose three walker types (`cloner`, `resister`, `persister`),
- keep the full SU(2)xU(1) measurement path batched and vectorized in PyTorch,
- reuse directed companion-operator patterns from the strong-force implementation.

## Theory Alignment
Reference: `docs/source/3_fractal_gas/2_fractal_set/04_standard_model.md`

### U(1)
Use the existing fitness phase transport:
- `theta_ij^(U1) = -(Phi_j - Phi_i)/h_eff`
- weighted local transport over recorded neighbors/companions.

### SU(2)
Use cloning score phase and doublet structure:
- `theta_ij^(SU2) = S_i(j)/h_eff`
- `S_i(j) = (V_j - V_i)/(V_i + epsilon_clone)`
- locality amplitude from algorithmic distance with scale `epsilon_c`.

### Directional SU(2)
For score-directed SU(2), orient pair contribution by cloning-score sign:
- base pair contribution: `exp(i * theta_ij)`
- directed pair contribution:
  - `exp(i * theta_ij)` if `S_i(j) >= 0`
  - `conj(exp(i * theta_ij))` if `S_i(j) < 0`

This mirrors the score-directed conjugation logic already used in companion strong-force operators.

## Walker Types
Frame-global labels per walker (`[N]` boolean masks):

1. `cloner`
- alive and Bernoulli cloning event is true (`will_clone[i] == True`).

2. `resister`
- alive, not cloner, and lower fitness than at least one other alive walker in the same frame.

3. `persister`
- alive and not in the previous two groups.

These masks are disjoint and exhaustive over alive walkers.

## Operator Families
### Existing core channels
- `u1_phase`, `u1_dressed`, `u1_phase_q2`, `u1_dressed_q2`
- `su2_phase`, `su2_component`, `su2_doublet`, `su2_doublet_diff`
- `ew_mixed`

### Added directional SU(2) channels
- `su2_phase_directed`
- `su2_component_directed`
- `su2_doublet_directed`
- `su2_doublet_diff_directed`

### Added walker-type SU(2) channels
- `su2_phase_{cloner,resister,persister}`
- `su2_component_{cloner,resister,persister}`
- `su2_doublet_{cloner,resister,persister}`
- `su2_doublet_diff_{cloner,resister,persister}`

## Config Surface
### Electroweak channel config
- `su2_operator_mode`: `standard | score_directed`
- `enable_walker_type_split`: `bool`
- `walker_type_scope`: currently `frame_global`

### Multiscale electroweak config
- same three fields as above for consistent behavior in per-scale and best-scale outputs.

## Data Flow
1. Build packed neighbors from recorded edge weights or companion topology.
2. Compute vectorized U(1)/SU(2) per-walker operators.
3. Select active SU(2) branch (`standard` or `score_directed`) for base channels.
4. Optionally apply walker-type masks to emit split channels.
5. Aggregate over walkers and feed correlator/mass extraction.
6. For multiscale mode, repeat per scale and run existing best-scale filtering.

## Tensor Shapes
- walker states: `[N, d]`
- packed neighbors: indices/weights/valid `[N, K]`
- frame operator outputs: `[N]` complex
- time series: `[T]` real (masked mean over alive walkers)
- multiscale series: `[S, T]`

## Reuse from Companion Strong-Force
Reused implementation patterns:
- score-directed orientation by conjugation,
- strict finite/in-range/alive masking,
- batched reductions with minimal Python loops,
- consistent best-scale quality filtering.

## Validation
Tests cover:
- vectorized operator equivalence (reference vs vectorized),
- directional channel emission,
- walker-type split masks and channel masking behavior,
- multiscale directional/split shape and key checks,
- backward compatibility for standard mode with split disabled.
