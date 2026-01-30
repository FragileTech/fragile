# QFT Calibration Tuning Log (Voronoi Volume Reward)

## Sweep 0 — Baseline feasibility (2026-01-30)

**Run**: `outputs/qft_tuning/sweep0_baseline.json`

**Runtime notes**
- Full Voronoi volume reward + anisotropic diffusion was too slow when using full Hessian.
- Switched to `diffusion_mode=grad_proxy`, `use_diagonal_diffusion=True`, and `fitness_grad_mode=sum` with detached stats/companions.
- Set Voronoi reward cache: `voronoi_reward_update_every=10` to keep reward definition but reuse across steps.

**Key settings (non-default overrides)**
- Reward: `voronoi_volume`, `voronoi_reward_update_every=10`
- Forces: `use_potential_force=False`, `use_fitness_force=False`
- Diffusion: `use_anisotropic_diffusion=True`, `diffusion_mode=grad_proxy`, `diagonal_diffusion=True`
- Fitness grads: `fitness_grad_mode=sum`, `fitness_detach_stats=True`, `fitness_detach_companions=True`
- Neighbors: `neighbor_method=recorded`, `neighbor_graph_method=voronoi`, `neighbor_graph_record=True`
- Viscous: `viscous_neighbor_weighting=uniform`, `viscous_neighbor_penalty=0.0`

**Strong-force summary**
- Masses: `pseudoscalar=0.0`, `scalar=0.0`, `vector=0.0`, `nucleon=0.1161`, `glueball=0.02595`
- Ratios unavailable (π mass collapsed to 0.0), so `rho_pi` and `nucleon_pi` were undefined.

**Electroweak summary**
- Error % (best-fit scaled masses):
  - `u1_phase` +8,134,136%
  - `u1_dressed` +2,385.9%
  - `su2_phase` −96.5%
  - `su2_doublet` −75.5%
  - `ew_mixed` +217.7%

**Learning**
- With the current dynamics, pseudoscalar/vector channels are not developing a stable exponential decay (AIC fit picks ~0 mass).
- Electroweak channels are extremely mismatched; likely need to tune companion epsilons (`epsilon_d`, `epsilon_c`), `lambda_alg`, and `h_eff` before any strong/weak co-calibration is possible.

## Sweep 1 — h_eff scaling (2026-01-30)

Runs: `sweep1_h0p5`, `sweep1_h2`

**Key findings**
- `h_eff=0.5` kept pseudoscalar/vector at ~0, ratios undefined.
- `h_eff=2.0` yielded non-zero masses and ratios:
  - `rho/pi ≈ 0.625` (target 5.5)
  - `nucleon/pi ≈ 0.384` (target 6.7)
- Electroweak error remained enormous (mean > 10^6 %).

## Sweep 2 — Companion epsilons (2026-01-30)

Runs: `sweep2_epsd1`, `sweep2_epsd5`, `sweep2_epsc0p8`, `sweep2_epsc3`

**Key findings**
- Strong ratios unchanged across epsilon sweeps (since strong channels use recorded neighbors, not companions).
- Electroweak errors improved with lower `epsilon_d` (best at `epsilon_d=1.0`), but still > 10^5 %.

## Sweep 3 — Viscous coupling (2026-01-30)

Runs: `sweep3_nu0p5`, `sweep3_nu2`, `sweep3_l0p5`

**Key findings**
- `nu=2.0` gave the best strong ratios so far:
  - `rho/pi ≈ 3.54` (error −1.96)
  - `nucleon/pi ≈ 1.19` (error −5.51)
- Lower `nu` collapsed vector mass to 0.
- Changing viscous length scale alone did not help strong ratios.

## Sweep 4 — Higher nu + cloning epsilon (2026-01-30)

Runs: `sweep4_nu3`, `sweep4_nu4`, `sweep4_eclone0p05`

**Key findings**
- Increasing `nu` above 2 destabilized pseudoscalar/vector fits (masses collapsed to 0).
- Larger `epsilon_clone` distorted nucleon/vector fits and did not improve ratios.

## Sweep 5 — lambda_alg (2026-01-30)

Runs: `sweep5_lambda0p5`, `sweep5_lambda2`

**Key findings**
- `lambda_alg=0.5` collapsed all strong masses.
- `lambda_alg=2.0` raised `rho/pi` modestly (~1.73) but nucleon mass dropped to 0.

## Sweep 6 — delta_t (2026-01-30)

Runs: `sweep6_dt0p2`, `sweep6_dt0p05`

**Key findings**
- `delta_t=0.2` dramatically improved electroweak error (mean ≈ 49.8%) but collapsed strong masses (vector=0).
- `delta_t=0.05` produced a larger nucleon mass but vector stayed at 0 (ratios undefined).

## Sweep 7 — delta_t + higher h_eff (2026-01-30)

Run: `sweep7_dt0p2_h4`

**Key findings**
- Strong masses all collapsed to 0; electroweak comparison rows empty.

## Sweep 8 — abs/abs2 transforms (2026-01-30)

Run: `sweep8_abs`

**Key findings**
- `vector_abs2` and `nucleon_abs2` were much smaller, worsening ratios (`rho/pi ≈ 0.138`, `nucleon/pi ≈ 0.166`).

## Sweep 9 — Electroweak h_eff decoupling (2026-01-30)

Runs: `sweep9_ew0p5`, `sweep9_ew0p2`

**Key findings**
- `ew_h_eff=0.2` reduced EW mean error to ≈ 27,000% (still far from 10%).
- Strong ratios unchanged (since strong uses separate `h_eff=2.0`).

## Sweep 10 — delta_t=0.15 (2026-01-30)

Run: `sweep10_dt0p15`

**Key findings**
- Pseudoscalar collapsed to 0 again; strong ratios undefined; EW error still very large.

## Best So Far (2026-01-30)

**Strong force (closest to targets)**
- Run: `sweep3_nu2`
- Params: `h_eff=2.0`, `nu=2.0`, `companion_epsilon=1.0`
- Ratios: `rho/pi ≈ 3.54`, `nucleon/pi ≈ 1.19`
- Still below targets (5.5 and 6.7).

**Electroweak (closest to targets)**
- Run: `sweep6_dt0p2`
- Params: `delta_t=0.2` (with EW analysis), `h_eff=2.0`
- EW mean error ≈ 49.8%
- Strong masses collapsed (vector=0), so this is not a viable joint solution.

## Open Questions / Next Levers

- Strong ratios may require larger vector/nucleon mass separation without collapsing pseudoscalar. Likely candidates:
  - increase `nu` slightly above 2 but stabilize with `viscous_length_scale` or `gamma/beta`
  - adjust `p_max`, `sigma_x`, and `alpha_restitution` together to raise nucleon mass without killing vector
- Electroweak errors respond strongly to `delta_t` and `epsilon_d`; but EW tuning currently conflicts with strong.
- Consider running a joint sweep that decouples EW analysis (`ew_h_eff`, `epsilon_d/c`) while holding the strong-dynamics parameters fixed at `nu=2`.
