"""VLA experiment configuration dataclass."""

from dataclasses import dataclass, field

import torch


@dataclass
class VLAConfig:
    """Configuration for the TopoEncoder x SmolVLA experiment."""

    # --- Feature extraction ---
    smolvla_model_id: str = "lerobot/smolvla_base"
    dataset_name: str = "lerobot/svla_so100_pickplace"
    feature_dim: int = 720  # lm_expert hidden_size (SmolVLM2-500M VLM=960, expert=720)
    feature_cache_dir: str = "outputs/vla/features"
    pooling: str = "mean"  # "mean" or "modality_aware"
    max_episodes: int = 0  # 0 = all

    # --- Encoder architecture (maps to TopoEncoderPrimitives args) ---
    input_dim: int = 720  # must match feature_dim
    hidden_dim: int = 256
    latent_dim: int = 16
    num_charts: int = 8
    codes_per_chart: int = 32
    covariant_attn: bool = True
    covariant_attn_tensorization: str = "full"
    covariant_attn_rank: int = 8
    covariant_attn_tau_min: float = 1e-2
    covariant_attn_denom_min: float = 1e-3
    covariant_attn_use_transport: bool = True
    covariant_attn_transport_eps: float = 1e-3
    conv_backbone: bool = False

    # --- World model ---
    action_dim: int = 6  # SO100: 6 joints (shoulder_pan/lift, elbow, wrist_flex/roll, gripper)
    wm_hidden_dim: int = 256
    wm_dt: float = 0.01
    wm_gamma_friction: float = 1.0
    wm_T_c: float = 0.1
    wm_prediction_horizon: int = 4
    wm_alpha_potential: float = 0.5     # generation vs control balance in Phi_eff
    wm_beta_curl: float = 0.1           # Value Curl coupling for Lorentz/Boris
    wm_gamma_risk: float = 0.01         # risk-stress penalty weight
    wm_use_boris: bool = True           # enable Boris rotation
    wm_use_jump: bool = True            # enable Poisson jump process
    wm_jump_rate_hidden: int = 64       # hidden dim for jump rate predictor

    # --- Phase 1 loss weights (encoder) ---
    w_feature_recon: float = 1.0
    w_vq: float = 1.0
    w_entropy: float = 0.1
    w_diversity: float = 0.1
    w_uniformity: float = 0.1
    w_radial_calibration: float = 0.1
    w_codebook_spread: float = 0.05
    w_codebook_spread_margin: float = 1.0
    w_chart_collapse: float = 1.0
    w_code_collapse: float = 0.5
    w_code_collapse_temperature: float = 1.0
    w_window: float = 0.5
    w_window_eps_ground: float = 0.1
    w_consistency: float = 0.1
    w_jump: float = 0.1
    w_jump_warmup: int = 20
    w_jump_ramp_end: int = 40

    # --- Phase 2 loss weights (dynamics) ---
    w_geodesic: float = 1.0
    w_chart_transition: float = 0.5
    w_momentum_reg: float = 0.01
    w_energy_conservation: float = 0.01
    w_jump_dynamics: float = 0.1

    # --- Phase 3 joint scaling ---
    phase3_encoder_scale: float = 0.1
    phase3_dynamics_scale: float = 1.0

    # --- Training ---
    phase1_epochs: int = 100
    phase2_epochs: int = 50
    phase3_epochs: int = 30
    lr_encoder: float = 1e-3
    lr_wm: float = 1e-3
    lr_joint_encoder: float = 1e-4
    lr_joint_wm: float = 1e-3
    batch_size: int = 256
    sequence_length: int = 8
    grad_clip: float = 1.0

    # --- Logging / output ---
    output_dir: str = "outputs/vla"
    save_every: int = 25
    log_every: int = 10
    mlflow: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "vla"
    mlflow_run_name: str = ""

    # --- Device ---
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    )
