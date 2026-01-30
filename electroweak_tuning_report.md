# Electroweak tuning report (QFT baseline)

Date: 2026-01-30
Script: `src/fragile/fractalai/experiments/qft_baseline_channels.py`
Runs: 200 walkers, 300 steps, zero reward, viscosity-only, CPU, float32

## Summary
- Coupling targets (mZ) are reachable via epsilon_d and epsilon_c; default epsilon values are already within ~3.5% for g1 and g2.
- Electroweak mass ratios are not close to observed values (orders of magnitude off). Parameter tweaks in analysis (epsilon_d, epsilon_c, h_eff) change ratios only modestly.
- Extending to cloning + fitness parameters did not move ratios into the 10% target band. The best su2_doublet/u1_dressed ratio reached ~5.83 (target ~863).
- Added theory-consistent electroweak probe channels (charge-2 harmonics and SU(2) component/difference); baseline ratios remain O(1).
- 500-step sweeps on fitness_rho, fitness_epsilon_dist, lambda_alg, and epsilon_F did not resolve the U1/SU2 scale gap. The best su2_doublet/u1_dressed at 500 steps was ~4.67 (fitness_rho=0.1), still far from 863.

## Reference targets (observed)
- u1_phase/u1_dressed (electron/muon): 0.004836
- su2_doublet/u1_dressed (Z/muon): 863.045
- ew_mixed/u1_dressed (tau/muon): 16.817

## Probe ratio targets (heuristic, theory-consistent)
These are **working targets** for the new probe channels, based on phase-harmonic scaling
and doublet coherence. They are not observed values and should be treated as diagnostics.

- u1_phase_q2/u1_phase: 4.0 (q=2 harmonic → q^2 scaling heuristic)
- u1_dressed_q2/u1_dressed: 4.0 (same harmonic heuristic)
- su2_component/su2_doublet: 1.0 (component vs. summed doublet coherence)
- su2_doublet_diff/su2_doublet: 1.0 (orthogonal doublet coherence)

## Runs and results

### Baseline (ew_tune_base)
Params: ew_epsilon_d=default (2.8), ew_epsilon_c=default (1.68419), ew_h_eff=1.0
Couplings (mZ error): g1=0.3571 (-0.09%), g2=0.6298 (-3.36%)
Ratios:
- u1_phase/u1_dressed = 1.4908 (error +30725%)
- su2_doublet/u1_dressed = 0.9378 (error -99.89%)
- ew_mixed/u1_dressed = 1.0842 (error -93.55%)

### Baseline with new electroweak probes (ew_tune_new_channels)
New channels: u1_phase_q2, u1_dressed_q2, su2_component, su2_doublet_diff
Ratios (base u1_dressed):
- u1_phase_q2/u1_dressed = 3.3363
- u1_dressed_q2/u1_dressed = 0.8875
- su2_component/u1_dressed = 0.9734
- su2_doublet_diff/u1_dressed = n/a (no mass plateau)

## 500-step tuning (new probes enabled)

### Baseline (ew_tune_new_channels_500_base)
Ratios:
- u1_phase/u1_dressed = 1.2288 (error +25308%)
- su2_doublet/u1_dressed = 1.1898 (error -99.86%)
- ew_mixed/u1_dressed = 1.2366 (error -92.65%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6395 (target 4.0, error -84.01%)
- u1_dressed_q2/u1_dressed = 1.0291 (target 4.0, error -74.27%)
- su2_component/su2_doublet = 1.0738 (target 1.0, error +7.38%)
- su2_doublet_diff/su2_doublet = 9.7642 (target 1.0, error +876.42%)

### fitness_rho=0.1 (ew_tune_fitrho0p1_500)
Ratios:
- su2_doublet/u1_dressed = 4.6742 (error -99.46%)
- ew_mixed/u1_dressed = 3.6178 (error -78.49%)
Probe ratios:
- u1_dressed_q2/u1_dressed = 3.9605 (target 4.0, error -0.99%)
- su2_component/su2_doublet = 0.8778 (target 1.0, error -12.22%)
- su2_doublet_diff/su2_doublet = 2.1842 (target 1.0, error +118.42%)

### fitness_rho=0.5 (ew_tune_fitrho0p5_500)
Ratios:
- u1_phase/u1_dressed = 15.6712 (error +323929%)
- su2_doublet/u1_dressed = 4.5563 (error -99.47%)
- ew_mixed/u1_dressed = 1.6845 (error -89.98%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6684 (target 4.0, error -83.29%)
- u1_dressed_q2/u1_dressed = 2.8392 (target 4.0, error -29.02%)
- su2_component/su2_doublet = 0.8739 (target 1.0, error -12.61%)
- su2_doublet_diff/su2_doublet = 0.9827 (target 1.0, error -1.73%)

### fitness_epsilon_dist=1e-6 (ew_tune_fit_epsdist1e-6_500)
Ratios:
- u1_phase/u1_dressed = 1.2289 (error +25309%)
- su2_doublet/u1_dressed = 1.1896 (error -99.86%)
- ew_mixed/u1_dressed = 1.2363 (error -92.65%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6394 (target 4.0, error -84.02%)
- u1_dressed_q2/u1_dressed = 1.0290 (target 4.0, error -74.27%)
- su2_component/su2_doublet = 1.0737 (target 1.0, error +7.37%)
- su2_doublet_diff/su2_doublet = 9.7639 (target 1.0, error +876.39%)

### fitness_epsilon_dist=1e-4 (ew_tune_fit_epsdist1e-4_500)
Ratios:
- u1_phase/u1_dressed = 1.2290 (error +25311%)
- su2_doublet/u1_dressed = 1.1896 (error -99.86%)
- ew_mixed/u1_dressed = 1.2364 (error -92.65%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6393 (target 4.0, error -84.02%)
- u1_dressed_q2/u1_dressed = 1.0290 (target 4.0, error -74.28%)
- su2_component/su2_doublet = 1.0738 (target 1.0, error +7.38%)
- su2_doublet_diff/su2_doublet = 9.7645 (target 1.0, error +876.45%)

### lambda_alg=0.5 (ew_tune_lambda0p5_500)
Ratios:
- u1_phase/u1_dressed = 3.6102 (error +74547%)
- su2_doublet/u1_dressed = 1.4856 (error -99.83%)
- ew_mixed/u1_dressed = 1.2879 (error -92.34%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.8306 (target 4.0, error -79.23%)
- u1_dressed_q2/u1_dressed = 3.0816 (target 4.0, error -22.96%)
- su2_component/su2_doublet = 0.9608 (target 1.0, error -3.92%)
- su2_doublet_diff/su2_doublet = 0.0 (target 1.0, error -100.0)

### lambda_alg=2.0 (ew_tune_lambda2_500)
Ratios:
- u1_phase/u1_dressed = 1.6803 (error +34642%)
- su2_doublet/u1_dressed = 0.7226 (error -99.92%)
- ew_mixed/u1_dressed = 1.2439 (error -92.60%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6358 (target 4.0, error -84.11%)
- u1_dressed_q2/u1_dressed = 1.3342 (target 4.0, error -66.65%)
- su2_component/su2_doublet = 1.0541 (target 1.0, error +5.41%)
- su2_doublet_diff/su2_doublet = 0.0 (target 1.0, error -100.0)

### epsilon_F=10 (ew_tune_epsilonF10_500)
Ratios:
- u1_phase/u1_dressed = 1.2288 (error +25308%)
- su2_doublet/u1_dressed = 1.1898 (error -99.86%)
- ew_mixed/u1_dressed = 1.2366 (error -92.65%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6395 (target 4.0, error -84.01%)
- u1_dressed_q2/u1_dressed = 1.0291 (target 4.0, error -74.27%)
- su2_component/su2_doublet = 1.0738 (target 1.0, error +7.38%)
- su2_doublet_diff/su2_doublet = 9.7642 (target 1.0, error +876.42%)

### epsilon_F=100 (ew_tune_epsilonF100_500)
Ratios:
- u1_phase/u1_dressed = 1.2288 (error +25308%)
- su2_doublet/u1_dressed = 1.1898 (error -99.86%)
- ew_mixed/u1_dressed = 1.2366 (error -92.65%)
Probe ratios:
- u1_phase_q2/u1_phase = 0.6395 (target 4.0, error -84.01%)
- u1_dressed_q2/u1_dressed = 1.0291 (target 4.0, error -74.27%)
- su2_component/su2_doublet = 1.0738 (target 1.0, error +7.38%)
- su2_doublet_diff/su2_doublet = 9.7642 (target 1.0, error +876.42%)

### Lower epsilon_d (ew_tune_ed1)
Params: ew_epsilon_d=1.0, ew_epsilon_c=default, ew_h_eff=1.0
Couplings (mZ error): g1=1.0000 (+179.75%), g2=0.6298 (-3.36%)
Ratios:
- u1_phase/u1_dressed = 1.1038 (error +22724%)
- su2_doublet/u1_dressed = 0.6944 (error -99.92%)
- ew_mixed/u1_dressed = 1.0500 (error -93.76%)

### Higher epsilon_d (ew_tune_ed6)
Params: ew_epsilon_d=6.0, ew_epsilon_c=default, ew_h_eff=1.0
Couplings (mZ error): g1=0.1667 (-53.38%), g2=0.6298 (-3.36%)
Ratios:
- u1_phase/u1_dressed = 0.9064 (error +18642%)
- su2_doublet/u1_dressed = 0.5702 (error -99.93%)
- ew_mixed/u1_dressed = 0.6139 (error -96.35%)

### Lower epsilon_c (ew_tune_ec0p8)
Params: ew_epsilon_d=default, ew_epsilon_c=0.8, ew_h_eff=1.0
Couplings (mZ error): g1=0.3571 (-0.09%), g2=1.3258 (+103.44%)
Ratios:
- u1_phase/u1_dressed = 1.4908 (error +30725%)
- su2_doublet/u1_dressed = 1.6703 (error -99.81%)
- ew_mixed/u1_dressed = 1.2694 (error -92.45%)

### Higher epsilon_c (ew_tune_ec3)
Params: ew_epsilon_d=default, ew_epsilon_c=3.0, ew_h_eff=1.0
Couplings (mZ error): g1=0.3571 (-0.09%), g2=0.3536 (-45.75%)
Ratios:
- u1_phase/u1_dressed = 1.4908 (error +30725%)
- su2_doublet/u1_dressed = 0.7090 (error -99.92%)
- ew_mixed/u1_dressed = 0.9800 (error -94.17%)

### Lower h_eff (ew_tune_heff0p5)
Params: ew_epsilon_d=default, ew_epsilon_c=default, ew_h_eff=0.5
Couplings (mZ error): g1=0.2525 (-29.35%), g2=0.4453 (-31.67%)
Ratios:
- u1_phase/u1_dressed = 3.7594 (error +77632%)
- su2_phase/u1_dressed = 5.6203 (error -99.26%)
- su2_doublet/u1_dressed = 0.8632 (error -99.90%)
- ew_mixed/u1_dressed = 6.8731 (error -59.13%)

### Higher h_eff (ew_tune_heff2)
Params: ew_epsilon_d=default, ew_epsilon_c=default, ew_h_eff=2.0
Couplings (mZ error): g1=0.5051 (+41.29%), g2=0.8906 (+36.67%)
Ratios:
- u1_phase/u1_dressed = 1.2580 (error +25911%)
- su2_phase/u1_dressed = 0.3494 (error -99.95%)
- su2_doublet/u1_dressed = 1.0971 (error -99.87%)
- ew_mixed/u1_dressed = 1.1334 (error -93.26%)

### Low fitness + aggressive cloning (ew_tune_fit_low_clone_high)
Params: fitness_alpha=0.2, fitness_beta=0.2, fitness_eta=0.02, fitness_A=0.2, epsilon_clone=0.005, sigma_x=0.2, alpha_restitution=0.2
Ratios:
- u1_phase/u1_dressed = 3.1787 (error +65624%)
- su2_phase/u1_dressed = 1.4899 (error -99.80%)
- su2_doublet/u1_dressed = 1.3499 (error -99.84%)
- ew_mixed/u1_dressed = 1.4169 (error -91.57%)

### Very low fitness + aggressive cloning (ew_tune_fit_min)
Params: fitness_alpha=0.05, fitness_beta=0.05, fitness_eta=0.01, fitness_A=0.05, epsilon_clone=0.005, sigma_x=0.2, alpha_restitution=0.2
Ratios:
- u1_phase/u1_dressed = 9.4809 (error +195933%)
- su2_phase/u1_dressed = 6.3499 (error -99.17%)
- su2_doublet/u1_dressed = 1.0940 (error -99.87%)
- ew_mixed/u1_dressed = 1.2525 (error -92.55%)

### Low fitness + small companion_epsilon_clone (ew_tune_fit_low_epsclone_small)
Params: fitness_alpha=0.2, fitness_beta=0.2, fitness_eta=0.02, fitness_A=0.2, epsilon_clone=0.005, companion_epsilon_clone=0.5
Ratios:
- u1_phase/u1_dressed = 3.0000 (error +61931%)
- su2_phase/u1_dressed = 2.1628 (error -99.72%)
- su2_doublet/u1_dressed = 5.8323 (error -99.32%)
- ew_mixed/u1_dressed = 1.1079 (error -93.41%)

### Low fitness + large companion_epsilon_clone (ew_tune_fit_low_epsclone_big)
Params: fitness_alpha=0.2, fitness_beta=0.2, fitness_eta=0.02, fitness_A=0.2, epsilon_clone=0.005, companion_epsilon_clone=4.0
Ratios:
- u1_phase/u1_dressed = 3.0000 (error +61931%)
- su2_phase/u1_dressed = 2.1628 (error -99.72%)
- su2_doublet/u1_dressed = 0.9975 (error -99.88%)
- ew_mixed/u1_dressed = 1.0037 (error -94.03%)

### High fitness (ew_tune_fit_high)
Params: fitness_alpha=2.0, fitness_beta=2.0, fitness_eta=0.2, fitness_A=5.0
Ratios:
- su2_phase/u1_dressed = 4.4319 (error -99.42%)
- su2_doublet/u1_dressed = 0.1756 (error -99.98%)
- ew_mixed/u1_dressed = 0.8035 (error -95.22%)

### Cloning noise only (ew_tune_clone_noise)
Params: epsilon_clone=0.001, sigma_x=0.2, alpha_restitution=0.2
Ratios:
- u1_phase/u1_dressed = 1.1601 (error +23888%)
- su2_phase/u1_dressed = 1.5920 (error -99.79%)
- su2_doublet/u1_dressed = 1.0783 (error -99.88%)
- ew_mixed/u1_dressed = 1.3501 (error -91.97%)

## What I learned
- The coupling estimates are sensitive to epsilon_d and epsilon_c as expected, and can be tuned near mZ values. Default epsilon values already do this.
- Electroweak mass ratios remain O(1) across all analysis-level tuning attempts. They are orders of magnitude away from observed ratios, indicating the current electroweak channel construction yields similar mass scales.
- Changing h_eff alters the absolute electroweak masses and can yield a nonzero su2_phase mass, but it does not create the large separation required by observed ratios.
- Aggressive cloning + low fitness changes absolute scales but does not separate U1/SU2 enough to approach the target ratios.

## Next steps to reach 10% target
- Tune simulation parameters more aggressively (fitness_rho, fitness_epsilon_dist, lambda_alg if allowed). If you relax the viscosity-only constraint, test enabling fitness force (keep potential force off) to see if it breaks the U1/SU2 degeneracy.
- Consider electroweak-specific scaling factors in channel definitions (separate normalization for U1 vs SU2 phases) if consistent with the theory section.
- Investigate alternative correlator transforms for U1/SU2 channels (e.g., abs or abs2) and test whether those change relative masses.
- If ratio targets remain unreachable, re-check the mapping between proxy channels and physical masses in the doc, and confirm the required anchor choices.
