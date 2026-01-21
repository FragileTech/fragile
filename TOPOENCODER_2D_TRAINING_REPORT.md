# TopoEncoder 2D Training: Learning Rate and Regularization Weights (Theory-Derived)

This report distills the theory in `docs/source/1_agent` into a principled guide for
computing the learning rate and the regularization weights used by
`src/experiments/topoencoder_2d.py`. The goal is to replace ad-hoc constants with
constraint-driven, measurable quantities.

## 1) Learning rate (eta) as a controlled variable

Theory links step size to stability, information grounding, and geometry:

- **Controlled update law (Governor)**
  The theory frames training as a constrained dynamical system with a controlled
  step size:

  ```text
  theta_{t+1} = theta_t - eta_t * (G^{-1} grad L_task + sum_k lambda_k grad C_k)
  ```

  `G` is a parameter-space metric (natural gradient preconditioner), `C_k` are
  constraint violations (Sieve diagnostics), and `eta_t` is a control input
  rather than a fixed hyperparameter.

- **Stable learning window / coupling window**
  The coupling window requires information inflow to keep up with mixing:

  ```text
  lambda_in = E[I(X;K)]
  lambda_mix = E[(H(K_{t+1}) - H(K_t))_+]
  requirement: lambda_in >= lambda_mix
  ```

  If over-mixing is observed (H(K) near log|K| and I(X;K) low), reduce `eta_t` or
  increase grounding penalties (window loss) until the inequality holds.

- **Scaling hierarchy (BarrierTypeII)**
  Stability requires representation drift to be slower than dynamics drift, and
  policy/model updates not to outrun curvature (alpha/beta diagnostics). When
  the update scale exceeds the critic/geometry scale, reduce `eta_t` or freeze
  updates until `beta <= alpha` holds again.

- **Mass = metric (geometry-driven step size)**
  High curvature implies smaller steps. A practical rule is to scale `eta_t`
  inversely with curvature or gradient norm:

  ```text
  eta_t = min(eta_max, v_max / (||G^{-1} grad L|| + eps))
  ```

  This satisfies the update-speed limit (QSLCheck) and implements the
  "slower near high curvature" rule from the metric law.

- **Implementation rule of thumb**
  Compute `eta_t` per epoch (or per batch) using the measured gradient norm
  and a target step size `v_max` in latent/parameter space. If you want a simple
  schedule, use cosine annealing but still cap by the QSL bound above.

## 2) How to compute lambdas: three principled mechanisms

The theory treats each weight as a *multiplier* enforcing a measurable contract.
Use one of these three mechanisms depending on the loss type.

### A) Primal-dual updates (hard constraints)
Use for non-negotiable contracts (grounding, coupling, hard safety):

```text
lambda_i <- clip(lambda_i + eta_lambda * (C_i - eps_i), 0, lambda_max)
```

`C_i` is the measured violation (positive means violated). If `lambda_i` hits
`lambda_max`, the theory says the constraint is unsatisfied and the model or
architecture must change.

### B) PI/PID setpoint controllers (targets, not zero)
Use when you need a metric to stay near a target (entropy, code usage, KL per
update):

```text
error_t = target - metric_t
lambda_{t+1} = clip(lambda_t + Kp*error_t + Ki*sum(error) + Kd*delta(error))
```

Use PI if derivative noise is high.

### C) Learned precisions (multi-loss scaling)
Use when losses are likelihood-like and have unknown noise scales (recon vs
prediction vs auxiliary):

```text
L_total = sum_i (exp(-s_i) * L_i + s_i)
```

This learns effective weights as inverse variances.

## 3) Loss-by-loss lambda computation for topoencoder_2d

Below, each weight is tied to its theoretical contract and a measurable metric.

### Core reconstruction and VQ
- `vq_commitment_cost` (beta)
  - **Theory hook:** VQ loss balance (Appendix F, Disentangled VAE). Typical
    value is 0.25, but principled tuning is to keep codebook and commitment
    gradients comparable.
  - **Compute:** balance gradient norms:
    `beta = ||grad L_codebook|| / (||grad L_commit|| + eps)`.

### Routing and chart formation
- `entropy_weight` (routing sharpness)
  - **Theory hook:** CompactCheck + coupling window. You want routing entropy
    low enough for sharp charts but not so low that chart usage collapses.
  - **Metric:** mean routing entropy H(K|X).
  - **Compute:** PI controller toward a target `H_target` (e.g., mid-range of
    [0, log K]) or use primal-dual if you impose a hard upper bound.

- `diversity_weight` (chart usage balance)
  - **Theory hook:** Load-balance / BarrierScat.
  - **Metric:** H(K) or KL(p_k || uniform).
  - **Compute:** PI controller with target H(K) close to log K - eps.

- `consistency_weight` (encoder/decoder routing alignment)
  - **Theory hook:** Sync contract between components.
  - **Metric:** KL(enc_w || dec_w) or L2(enc_w - dec_w).
  - **Compute:** primal-dual on constraint KL <= eps_cons.

- `separation_weight` and `separation_margin`
  - **Theory hook:** WFR coupling constant and chart overlap radius.
  - **Metric:** pairwise chart center distances.
  - **Compute:** set `separation_margin` to the WFR overlap length:

    ```text
    margin = injectivity_radius(G)
    default: sqrt(tr(G^{-1}) / n)
    ```

    then use primal-dual on violation `max(0, margin - dist)`.

- `window_weight` and `window_eps_ground`
  - **Theory hook:** coupling window (I(X;K) >= eps and H(K) not saturated).
  - **Metric:** compute_window_loss already yields I(X;K).
  - **Compute:** set `window_eps_ground` to a fraction of log|K| or the estimated
    boundary capacity C_partial. Use primal-dual to enforce I(X;K) >= eps.

### Latent geometry and disentanglement
- `variance_weight`
  - **Theory hook:** prevent collapse (Stiffness/Gap checks).
  - **Metric:** latent energy E||z||^2 vs target.
  - **Compute:** primal-dual on constraint E||z||^2 >= target_std^2 * dim.

- `disentangle_weight`
  - **Theory hook:** gauge coherence (macro-nuisance independence).
  - **Metric:** cross-covariance between router weights and ||z_geo||.
  - **Compute:** primal-dual on Cov <= eps_dis.

- `orthogonality_weight`
  - **Theory hook:** isotropy (basis-invariant spread of singular values).
  - **Metric:** var(log svals) from SVD.
  - **Compute:** primal-dual on var(log svals) <= eps_orth.

### Codebook health
- `code_entropy_weight`
  - **Theory hook:** ComplexCheck / code usage.
  - **Metric:** global code entropy.
  - **Compute:** PI controller to keep H(code) >= log(num_codes) - eps.

- `per_chart_code_entropy_weight`
  - **Theory hook:** per-chart code diversity.
  - **Metric:** per-chart entropy.
  - **Compute:** PI controller per chart or aggregate target; keep each chart's
    entropy near log(num_codes).

### Priors and invariances
- `kl_prior_weight`
  - **Theory hook:** radial energy prior (Appendix F).
  - **Metric:** deviation of mean energy from target.
  - **Compute:** primal-dual on |E||z||^2 - target| <= eps.

- `orbit_weight`
  - **Theory hook:** SymmetryCheck (orbit invariance).
  - **Metric:** symmetric KL(enc_w || enc_w_aug).
  - **Compute:** PI controller toward KL target (e.g., small but nonzero).

- `vicreg_inv_weight`
  - **Theory hook:** Gram invariance (basis-free alignment).
  - **Metric:** MSE(Gram(z), Gram(z_aug)).
  - **Compute:** PI controller to keep Gram MSE below eps.

### Jump operator (chart gluing)
- `jump_weight`, `jump_warmup`, `jump_ramp_end`
  - **Theory hook:** WFR transport vs reaction. Jump cost should match the
    reaction term weight, which scales like 1 / lambda_wfr^2.
  - **Compute:**

    ```text
    jump_weight ~ c_jump / lambda_wfr^2
    lambda_wfr = injectivity_radius(G) or sqrt(tr(G^{-1})/n)
    ```

    Start jump only after the coupling window is satisfied (I(X;K) stable and
    H(K) below saturation), then ramp to the computed weight.

### Supervised topology (when enabled)
- `sup_weight`
  - **Theory hook:** supervised topology is a semantic potential term.
  - **Metric:** route loss vs recon/vq scale.
  - **Compute:** learned precision or gradient ratio so supervised loss does not
    dominate reconstruction early.

- `sup_purity_weight`, `sup_balance_weight`, `sup_metric_weight`
  - **Theory hook:** H(Y|K) (purity), KL(p_k || uniform) (balance), geodesic
    separation (metric).
  - **Compute:** primal-dual or PI to keep each metric near its target. Typical
    starting values (from Appendix F): purity 0.1, balance 0.01, metric 0.01.

- `sup_metric_margin`
  - **Theory hook:** same geometric separation scale as chart separation.
  - **Compute:** set to `separation_margin` or to `lambda_wfr`.

- `sup_temperature`
  - **Theory hook:** class temperature (semantic diffusion).
  - **Compute:** choose to hit a target class entropy; use PI to keep mean
    entropy of P(Y|K) in a desired range.

## 4) Practical training protocol (theory-aligned)

1. **Warmup reconstruction + VQ.** Use learned precisions or gradient matching
   to balance recon and VQ losses before adding higher-tier regularizers.
2. **Activate coupling window early.** Enforce I(X;K) >= eps and H(K) below
   saturation to prevent ungrounded inference.
3. **Introduce invariances and jump later.** Turn on orbit/vicreg and jump
   losses after routing stabilizes (window loss near zero).
4. **Make lambdas adaptive.** Use primal-dual for constraints, PI for setpoints,
   and learned precisions for loss-scale balancing.
5. **Tie step size to diagnostics.** Reduce `eta_t` (or freeze) when scaling
   mismatch or over-mixing is detected.

This yields a principled, measurable path to choosing `lr` and all regularizer
weights for `src/experiments/topoencoder_2d.py`.

## 5) What was implemented (code changes)

The training loop now follows the report directly:

- **Adaptive LR controller:** Replaces cosine scheduling when enabled. It grows
  LR while stable and shrinks it on loss spikes, coupling-window violations, or
  oversized updates. It also caps LR using a max update ratio
  `||dtheta|| / ||theta||`, tracks loss EMA for stability, and adds
  `lr_grounding_warmup_epochs` to avoid premature LR collapse before grounding.
- **Adaptive lambdas:** Uses PI control for setpoints (routing entropy and
  chart/code usage) and primal-dual updates for constraints (window,
  consistency, variance, separation, disentanglement, orthogonality, KL prior,
  orbit, VICReg, jump). These now use signed target errors based on warmup
  baselines (see `adaptive_target_ratio`, `adaptive_target_ema_decay`,
  `adaptive_target_min`) so lambdas can decrease as well as increase.
  Supervised purity/balance/metric weights are updated similarly and synced
  back into the supervised loss module.
- **Learned precisions (optional):** Adds a ParameterDict of log-variances to
  balance recon/VQ/supervised loss scales when `use_learned_precisions=True`.
- **Diagnostics logging:** Adds H(K|X), code entropies, gradient norm, update
  ratio, and LR traces to `info_metrics` and console logs. Adaptive state is
  persisted in checkpoints.

## 6) Example command (MNIST, 3k epochs, no AE/VQ, CPU)

This trains for 3000 epochs on MNIST, disables AE and standard VQ baselines,
saves checkpoints every 50 epochs into a new folder, uses adaptive LR with a
grounding warmup window, and runs on CPU:

```bash
python src/experiments/topoencoder_2d.py \
  --dataset mnist \
  --epochs 3000 \
  --lr 0.01 \
  --lr_max 0.01 \
  --disable_ae \
  --disable_vq \
  --save_every 50 \
  --output_dir outputs/topoencoder_mnist_cpu_3k \
  --device cpu \
  --adaptive_lr true \
  --lr_grounding_warmup_epochs 10 \
  --use_scheduler false
```

```bash
uv run python src/experiments/topoencoder_2d.py \
  --dataset cifar10 \
  --epochs 3000 \
  --save_every 50 \
  --log_every 25 \
  --device cpu \
  --adaptive_lr true \
  --use_scheduler false \
  --lr 0.005 \
  --lr_max 0.005 \
  --kl_prior_weight 0.001 \
  --lr_grounding_warmup_epochs 200 \
  --lr_unstable_patience 10 \
  --output_dir outputs/topoencoder_cifar10_cpu_3k \
  --mlflow true \
  --mlflow_tracking_uri http://127.0.0.1:5000 \
  --mlflow_experiment topoencoder \
  --mlflow_run_name cifar10_adaptive_lr
```

```bash
uv run python src/experiments/topoencoder_2d.py \
  --dataset mnist \
  --epochs 3000 \
  --save_every 50 \
  --log_every 25 \
  --device cpu \
  --adaptive_lr true \
  --use_scheduler false \
  --lr 0.0001 \
  --lr_max 0.0001 \
  --kl_prior_weight 0.001 \
  --lr_grounding_warmup_epochs 200 \
  --lr_unstable_patience 10 \
  --output_dir outputs/topoencoder_mnist_cpu_3k \
  --mlflow true \
  --mlflow_tracking_uri http://127.0.0.1:5000 \
  --mlflow_experiment topoencoder \
  --mlflow_run_name mnist_adaptive_lr
```

```bash
PYTHONUNBUFFERED=1 uv run src/experiments/topoencoder_2d.py \
  --dataset mnist \
  --epochs 1250 \
  --save_every 50 \
  --log_every 25 \
  --device cpu \
  --adaptive_lr true \
  --use_scheduler false \
  --lr 0.001 \
  --lr_max 0.05 \
  --lr_grounding_warmup_epochs 50 \
  --lr_increase_factor 1.05 \
  --lr_decrease_factor 0.8 \
  --lr_unstable_patience 5 \
  --lr_stable_patience 5 \
  --lr_loss_increase_tol 0.1 \
  --output_dir outputs/topoencoder_mnist_cpu_adapt_lr7 \
  --disable_ae \
  --disable_vq \
  --hidden_dim 64 \
  --codes_per_chart 8
```

Resume the MNIST adaptive LR run to 2k epochs (replace `--resume` with the latest checkpoint):

```bash
uv run src/experiments/topoencoder_2d.py \
  --dataset mnist \
  --epochs 2000 \
  --save_every 50 \
  --log_every 25 \
  --device cpu \
  --adaptive_lr true \
  --use_scheduler false \
  --lr 0.001 \
  --lr_max 0.05 \
  --lr_grounding_warmup_epochs 50 \
  --lr_increase_factor 1.05 \
  --lr_decrease_factor 0.8 \
  --lr_unstable_patience 5 \
  --lr_stable_patience 5 \
  --lr_loss_increase_tol 0.1 \
  --output_dir outputs/topoencoder_mnist_cpu_adapt_lr7 \
  --disable_ae \
  --disable_vq \
  --hidden_dim 64 \
  --codes_per_chart 8 \
  --resume outputs/topoencoder_mnist_cpu_adapt_lr7/topo_epoch_01200.pt
```
