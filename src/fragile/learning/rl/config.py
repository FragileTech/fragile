"""Configuration for Geometric Dreamer (Phase 4 model-based RL)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class DreamerConfig:
    """Hyperparameters for Geometric Dreamer training."""

    # --- Environment ---
    domain: str = "walker"
    task: str = "walk"
    action_repeat: int = 2
    max_episode_steps: int = 1000

    # --- Architecture ---
    obs_dim: int = 24  # walker-walk: orientations(14) + height(1) + velocity(9)
    action_dim: int = 6  # walker
    latent_dim: int = 16
    num_charts: int = 8
    num_action_charts: int = 0
    num_action_macros: int = 0
    d_model: int = 128
    hidden_dim: int = 256
    codes_per_chart: int = 32
    hard_routing: bool = True
    hard_routing_tau: float = 1.0
    hard_routing_warmup_epochs: int = 5
    hard_routing_tau_end: float | None = 0.3
    hard_routing_tau_anneal_epochs: int = 200
    commitment_beta: float = 0.25
    codebook_loss_weight: float = 1.0

    # --- World model (match GeometricWorldModel defaults) ---
    wm_dt: float = 0.01
    wm_gamma_friction: float = 1.0
    wm_T_c: float = 0.1
    wm_alpha_potential: float = 0.5
    wm_beta_curl: float = 0.1
    wm_gamma_risk: float = 0.01
    wm_risk_metric_alpha: float = 0.0
    wm_use_boris: bool = True
    wm_use_jump: bool = False  # disabled — encoder trains charts via recon
    wm_n_refine_steps: int = 3
    wm_jump_beta: float = 1.0
    wm_min_length: float = 0.03
    wm_prediction_horizon: int = 4

    # --- RL training ---
    total_epochs: int = 1000
    seed_episodes: int = 5
    batch_size: int = 32
    seq_len: int = 100
    buffer_capacity: int = 250_000
    updates_per_epoch: int = 0  # 0 = auto: buffer_steps // (batch_size * seq_len)
    imagination_horizon: int = 15

    # --- Optimization ---
    lr_actor: float = 1e-3
    lr_wm: float = 1e-3
    lr_encoder: float = 1e-3
    lr_min: float = 1e-5
    gamma: float = 0.99
    lambda_gae: float = 0.95
    T_c_entropy: float = 0.1
    grad_clip: float = 100.0
    w_dynamics: float = 1.0
    w_reward: float = 1.0
    w_critic: float = 0.1
    w_momentum_reg: float = 0.01
    w_energy_conservation: float = 0.01
    w_hodge: float = 0.01
    w_screened_poisson: float = 1.0
    w_action_recon: float = 1.0
    w_control_cycle: float = 0.1
    w_control_supervise: float = 0.5
    w_value_intent_align: float = 0.25
    w_macro_supervise: float = 0.25
    w_motor_nuisance_supervise: float = 0.1
    w_motor_compliance_supervise: float = 0.1
    w_actor_return: float = 1.0
    w_reward_nonconservative: float = 0.01
    screened_poisson_kappa: float = 1.0
    actor_return_horizon: int = 8
    actor_return_batch_size: int = 8
    actor_return_update_every: int = 10
    actor_return_warmup_epochs: int = 10
    lr_chart_centers_scale: float = 0.1
    lr_codebook_scale: float = 0.5
    lr_dyn_transition: float = 3e-3
    w_jump: float = 0.0
    w_jump_warmup: int = 20
    w_jump_ramp_end: int = 40
    w_dyn_transition: float = 0.5
    w_zeno: float = 0.1
    zeno_mode: str = "jsd"
    dyn_transition_hidden_dim: int = 128

    # --- Encoder loss weights (Phase 1 from VLA, applied to topoencoder) ---
    w_feature_recon: float = 1.0
    w_vq: float = 1.0
    w_entropy: float = 0.3
    w_diversity: float = 1.0
    w_chart_ot: float = 1.0
    w_uniformity: float = 0.05
    w_radial_calibration: float = 0.1
    w_confidence_calibration: float = 0.05
    w_hard_routing_nll: float = 0.5
    w_router_margin: float = 2.0
    router_margin_target: float = 0.1
    chart_usage_h_low: float | None = None
    chart_usage_h_high: float | None = None
    chart_ot_epsilon: float = 0.05
    chart_ot_iters: int = 20
    chart_ot_i_target: float = 0.35
    chart_ot_multiplier_lr: float = 1.0
    conf_target_top1: float = 0.55
    conf_multiplier_lr: float = 1.5
    chart_multiplier_lr: float = 1.0
    code_usage_gate_h: float = 0.0
    code_usage_h_low: float | None = None
    code_usage_h_high: float | None = None
    code_usage_temperature: float = 1.0
    code_usage_ramp_epochs: int = 1
    code_multiplier_lr: float = 0.5
    phase1_adaptive_multipliers: bool = True
    phase1_multiplier_max: float = 8.0
    phase1_multiplier_decay: float = 0.05
    w_v_tangent_barrier: float = 0.01
    w_codebook_spread: float = 0.05
    w_codebook_center: float = 0.02
    w_chart_center_mean: float = 0.02
    w_chart_center_radius: float = 0.05
    chart_center_radius_max: float = 0.9
    w_chart_center_sep: float = 0.02
    chart_center_sep_margin: float = 0.2
    w_chart_collapse: float = 0.0
    w_code_collapse: float = 0.5
    w_perp: float = 0.01
    w_window: float = 0.0
    w_window_eps_ground: float = 0.1
    w_consistency: float = 1.0
    radial_quality_alpha: float = 0.0
    radial_vq_alpha: float = 0.0
    radial_quality_rank_mix: float = 0.5
    radial_recon_quality_weight: float = 1.0
    radial_quality_mix: float = 0.5
    radial_quality_base_weight: float = 1.0
    radial_calibration_rho_max: float = 0.98
    radial_calibration_band_width: float = 0.1
    encoder_loss_scale: float = 1.0  # scale for encoder loss vs WM loss

    # --- Gas collection ---
    use_gas: bool = True
    gas_walkers: int = 5000
    gas_steps: int = 150
    gas_reward_coef: float = 2.0
    gas_dist_coef: float = 1.0
    gas_n_elite: int = 0
    gas_collect_every: int = 5
    gas_n_env_workers: int = 4
    gas_use_death_condition: bool = True

    # --- Schedule ---
    collect_every: int = 5
    collect_n_env_workers: int = 1
    log_every: int = 1
    eval_every: int = 50
    eval_episodes: int = 5
    checkpoint_every: int = 100
    freeze_encoder: bool = False
    diagnostics_every_updates: int = 1
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    allow_tf32: bool = True
    matmul_precision: str = "high"
    normalize_observations: bool = True
    obs_norm_min_std: float = 1e-3
    use_motor_texture: bool = True
    sigma_motor: float = 0.1

    # --- Infrastructure ---
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    )
    seed: int = 42
    checkpoint_dir: str = "outputs/dreamer"
    load_checkpoint: str = ""

    # --- MLflow ---
    mlflow: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "geometric-dreamer"

    def __post_init__(self) -> None:
        """Canonicalize theory-facing atlas sizes."""
        if self.num_action_charts <= 0:
            self.num_action_charts = self.num_charts
        if self.num_action_macros <= 0:
            self.num_action_macros = self.num_action_charts
