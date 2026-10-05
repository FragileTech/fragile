# Chapter 4 full native validation results

48 proposal cases and both N=200, d=3 dense references completed 24,832 native updates. All 543 numerical comparisons passed. Separate saved-data audits added zero engine steps: the frame audit checked 1,028,639 arithmetic predicates; the reference audit checked 2,354 individual force/primitive predicates. Neither audit adds unverified source-expression credits.

Every proposal case has 256 fresh native draws from its declared entering empirical state. Both dense reference trajectories have 128 continuous native steps. Transport errors use probability measures normalized by swarm size and optimal transport plans; they do not use permanent walker labels.

Raw dataset: `outputs/convergence/chapters04-06-experiments/full-20261004/chapter04`. Its archived Chapter 4 source SHA256 is `855c1e92b2f2d0a177f44e8d43fc8299c6174a7e9fe02e2abdb0b89624408ad0`. Source text, inventory, native implementation, complete configs, seed addresses, raw states, sampled fitness, donor plans, noise, stage/component data and checkpoints are retained.

Derived evidence: [frame audit](../outputs/convergence/chapter04-full-frame-audit.json), [reference audit](../outputs/convergence/chapter04-full-reference-audit.json), and [corrected presentation](../outputs/convergence/chapter04-full-corrected-presentation.json). The corrected presentation checks reference predicates individually and states continuous-trajectory provenance. Raw reports remain unchanged. Both audits retain the exact source/archive digests and their own analysis implementation SHA256.

## Quantities and bounds

| Quantity | Numerical prediction or construction | Observation and applicability |
|---|---|---|
| Full phase-space q transport | Full optimal cost = barycenter q cost + centered optimal q cost, q(dx,dv)=|dx|²+|dv|²+0.5 dx·dv. | 12,288 exact checks; largest residual 4.45×10⁻¹⁵. |
| Centered positional transport | Optimal centered positional cost ≤ Var_x(left output)+Var_x(right output). | 12,288 exact checks; largest excess 4.45×10⁻¹⁶. |
| Transport normalization | Every optimal plan has row and column mass 1/N. | 36,864 plan checks; exact retained marginals. |
| C_reset | D_x²+2(1−1/N)d σ_clone². For the ordinary full-live family, D_x=2 and σ_clone=0.1; hence C_reset≤4+0.02d≤4.04 uniformly in N. Selection uses σ_clone=0.2 and C_reset≤4.16. | Both mean output positional proxy and mean direct centered positional cost passed. Partial-input cases use the Chapter 3 eligible-donor certificate and receive no Chapter 4 source-expression credit. |
| Conditional proxy moment | Sum the two exact output positional-variance expectations using the actual retained sampled fitness, donor law, acceptance and jitter covariance. | Every case passed the declared six-standard-error expectation comparison; this is an expectation, not a per-sample inequality. |
| Mandatory revival | Both proposal outputs contain N live particles. Dead incoming storage positions do not enter the physical incoming measure. | 24,576 output-liveness checks; all passed, including half-dead and singleton inputs. |
| Permutation invariance | Same intrinsic canonical representation and primitive source give zero empirical output transport. | Six permutation cases passed; this probe concerns intrinsic representations and records its common seed. |
| Actual donor probabilities | Reconstruct radius-two squashed phase-space Gaussian weights of width two and their excluded-self row normalizer. | 430,504 independently recomputed integrated acceptance rows agree with native moments within 7.78×10⁻¹⁶. |
| a_* | The squashed feature diameter squared is at most 32, so a_*=exp(−32/(2×2²))=exp(−4)=0.01831563889; K_i(j)≥a_*/(N−1). | 430,504 recorded-row floor checks passed. The compensated pressure bound below retains no growing N factor. |
| R_star | R_star=v_max−v_min from configured positive bounded logistic maps; complete realized fitness is checked against the same interval. | Recorded range: 4.4–6.38753487. |
| Delta_V | Half the actual positive arithmetic complement-minus-H fitness gap; no log-gap substitution. | Recorded range: 0.00341277081–0.924680528. |
| s_star_squared | s_*²=f_H f_L Delta_V²; compared with the actual complete sampled fitness variance. | Recorded range: 2.18381336e-06–0.16031889. |
| f_H | Fraction of the geometric set chosen before fitness; H and its complement are nonempty. | Recorded range: 0.25. |
| f_L | Complementary geometric fraction. | Recorded range: 0.75. |
| B_acc | max(R_star,p_max(v_max+epsilon_clone)); clipping is retained. | Recorded range: 4.410001–6.38753487. |
| p_u | a_* s_*²/(2 R_star B_acc); actual minimum target row pressure is compared with this lower bound. | Recorded range: 1.03066192e-09–3.59840636e-05. |
| c_H | Fraction of chosen-swarm centered positional energy captured by H. | Recorded range: 0.75. |
| a_x | Fixed structural-to-positional comparison coefficient; its geometric admission is checked, not fitted after the output. | Recorded range: 0.5. |
| b_x | Nonnegative velocity/cross-term remainder; zero for this matched-velocity input family. | Recorded range: 0. |
| M_j | Recorded other-swarm H positional energy divided by N. | Recorded range: 0–0.5625. |
| B_T | Recorded paired positional error outside the actual target T=H∩U, divided by N. | Recorded range: -0–0.28125. |
| c_err | c_H a_x/2; target error is checked against c_err V_struct−g_err. | Recorded range: 0.1875. |
| g_err | c_H b_x/2+M_j+B_T; no missed-target contribution is discarded. | Recorded range: 0–0.5625. |
| chi | p_u c_err; event-specific pressure coefficient, not a fitted state-uniform rate. | Recorded range: 1.93249111e-10–6.74701192e-06. |
| g_max | max(p_u g_err,chi×0.01); threshold contribution is retained. | Recorded range: 4.68086924e-11–1.85564414e-05. |
| Q | Actual integrated sum of both swarms’ cloning probabilities weighted by entering centered positional error, divided by N. | Recorded range: 0–1.05677464. |
| Q_lower | chi V_struct−g_max; checked against Q individually on every admitted measurement. | Recorded range: -1.85564414e-05–4.86135201e-06. |
| Target overlap and fit/unfit fractions | Actual target fraction ≥ f_H f_L gap/R_V; both fit and unfit fractions ≥ s_*²/(2R_star²). | Each inequality passed on all 7,703 admitted complete sampled fitness vectors. |
| Conditional target applicability | All-alive entering inputs, both geometric sets nonempty, positive actual arithmetic fitness gap and the explicit positional/structural margin. | 7,703 admitted targets; 6,158 had strictly positive Q lower bounds; 4,585 outcomes were excluded and retained. Event-dependent margins are not averaged into uniform constants. |
| Native dense viscous force | Gaussian kernel with nu=0.3, rho=1; divide by eligible count or by actual off-diagonal row mass. | Every B1/B2 stage checked separately in both 128-step references; both full dense influence arrays retained. |
| Native harmonic force | F(x)=−x at each actual force-input stage. | Every reference B1/B2 residual passed individually. |
| Dense influence count | N(N−1)=39,800 directed edges at each B1/B2 stage, 79,600 per update. | All 512 reference force stages passed exact count checks. |
| Count-normalized total viscous force | Sum of pairwise viscous forces =0. | 256 B1/B2 stages passed. Row normalization has its separate force formula and does not assume this conservation identity. |
| N, d, L_D | Reference population, dimension and absorbing box radius. | Actual retained reference: N=200, d=3, L_D=2; both box corners and dimensions are checked during primitive reconstruction. |
| V, alpha_col | Radial velocity cap and collision restitution. | Actual retained values V=2, alpha_col=0.5; consumed in V_c=4. |
| nu, rho | Dense viscosity coefficient and Gaussian bandwidth. | Actual retained values nu=0.3, rho=1; consumed in each reconstructed force and derivative majorant. |
| L_F, B_F | Global harmonic-force derivative and affine remainder constants. | F(x)=−x has L_F=1 and B_F=0 analytically; its actual stage force is measured separately. |
| b_O, sigma_pos | OU noise scale and final position-diffusion amplitude. | Actual retained values b_O=1 and sigma_pos=0.1; Gaussian innovation and unit isotropic factors are checked. |
| J_0 | Survival-event initial jitter radius. | J_0=sigma_J=0.1; unit-ball Gaussian density bound is used. |
| h | Exact BAOAB timestep. | 0.04; independently rebuilt from the retained native configuration. |
| t | h/2. | 0.02; independently rebuilt from the retained native configuration. |
| gamma | OU friction. | 1; independently rebuilt from the retained native configuration. |
| c | exp(−gamma h). | 0.960789439152; independently rebuilt from the retained native configuration. |
| q_squared | −expm1(−2 gamma h)/(2 gamma). | 0.0384418268067; independently rebuilt from the retained native configuration. |
| s | sigma_pos sqrt(h). | 0.02; independently rebuilt from the retained native configuration. |
| sigma_J | Native Gaussian clone jitter amplitude. | 0.1; independently rebuilt from the retained native configuration. |
| V_c | (1+2|alpha_col|)V. | 4; independently rebuilt from the retained native configuration. |
| R_D | sqrt(d)L_D. | 3.46410161514; independently rebuilt from the retained native configuration. |
| B_1 | (1+2t nu)V_c+t[L_F(R_D+J_0)+B_F]. | 4.1192820323; independently rebuilt from the retained native configuration. |
| A | L_D+J_0+t(1+c)B_1. | 2.26154089412; independently rebuilt from the retained native configuration. |
| sigma_h | sqrt(t²q²+s²). | 0.0203807931819; independently rebuilt from the retained native configuration. |
| log_survival_floor_lower | Rigorous Gaussian ball/coordinate-interval log lower bound, avoiding tiny CDF subtraction. | -262.26416933; independently rebuilt from the retained native configuration. |
| r_star_from_survival_lower | sqrt(2d[log(12dN)−2 log(survival lower)]). | 56.572617491; independently rebuilt from the retained native configuration. |
| J | sigma_J r_star. | 5.6572617491; independently rebuilt from the retained native configuration. |
| C_x | Count:4 nu V_c exp(−1/2)/rho; row:16 nu V_c(R_D+J)/rho². | count 2.91134716662; row 175.130176593; independently rebuilt from the retained native configuration. |
| kappa_F | 1−t²L_F−chi_norm t nu>0; chi_norm=1 count or2 row. | count 0.9936; row 0.9876; independently rebuilt from the retained native configuration. |
| beta_F | t²(L_F+C_x); count<0.002, row<0.071. | count 0.00156453886665; row 0.0704520706374; independently rebuilt from the retained native configuration. |

## Population scaling and landscape behavior

All canonical entering centered positional errors are 0.75. The table reports direct optimal output positional transport after the proposal, with 256 trials per cell. It evaluates a single cloning proposal; it does not infer a full-time convergence exponent or a transition-law Wasserstein distance.

| N | d | Quadratic | Sphere | Rastrigin | Constant |
|---|---|---|---|---|---|
| 4 | 1 | 0.302021 ± 0.021988 | 0.301054 ± 0.022114 | 0.270472 ± 0.021690 | 0.800578 ± 0.009512 |
| 4 | 2 | 0.328117 ± 0.022785 | 0.321212 ± 0.022628 | 0.297953 ± 0.022350 | 0.826626 ± 0.010079 |
| 16 | 1 | 0.395398 ± 0.011646 | 0.362943 ± 0.011290 | 0.355426 ± 0.011929 | 0.833458 ± 0.006879 |
| 16 | 2 | 0.369866 ± 0.012269 | 0.363755 ± 0.011691 | 0.357602 ± 0.012344 | 0.849182 ± 0.006357 |
| 64 | 1 | 0.378589 ± 0.006517 | 0.376659 ± 0.005742 | 0.365158 ± 0.005577 | 0.845945 ± 0.003826 |
| 64 | 2 | 0.380569 ± 0.005502 | 0.371383 ± 0.006556 | 0.373006 ± 0.006058 | 0.846187 ± 0.003681 |

Quadratic, Sphere and Rastrigin proposals reduce this entering error in all six population/dimension cells. Constant-landscape proposals can increase it while satisfying the reset bound. At N=64 the Constant case admits no positive-gap target certificates; an unconditional proposal-contraction claim would therefore exceed the tested hypotheses. This behavior is retained in the data and is compatible with the signed-pressure/reset statements actually checked.

## Remaining source scope

The complete source inventory contains 308 required expressions. Six have whole-expression numerical bindings in the native report; 302 remain uncredited, with zero unbound evidence rows. Numerical parameter assembly and finite-stage identities in this table are not promoted to proofs of global hypotheses. In particular, these fixtures do not certify state-uniform target gaps, a complete signed-update contraction rate, the global density/minorization/eigenfunction constants, or QSD convergence from paired errors. Those require their own explicit certificates and distributional experiments.

Independent reanalysis commands:

```sh
python3 proof-validation/audit_chapter04_frames.py DATASET --output DERIVED_FRAME_AUDIT.json
python3 proof-validation/audit_chapter04_references.py DATASET --output DERIVED_REFERENCE_AUDIT.json --presentation-output DERIVED_PRESENTATION.json
```
