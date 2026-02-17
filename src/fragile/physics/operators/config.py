"""Configuration dataclasses for strong-force companion channel operators.

Provides a base config with fields shared across all five companion channels
(meson, vector, baryon, glueball, tensor) plus channel-specific subclasses
and a correlator config.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChannelConfigBase:
    """Shared configuration for companion-channel operator computation."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None
    color_dims: tuple[int, int, int] | None = None
    eps: float = 1e-12
    pair_selection: str = "both"


@dataclass
class MesonOperatorConfig(ChannelConfigBase):
    """Configuration for scalar / pseudoscalar meson operators."""

    operator_mode: str = "standard"


@dataclass
class VectorOperatorConfig(ChannelConfigBase):
    """Configuration for vector / axial-vector meson operators."""

    position_dims: tuple[int, int, int] | None = None
    use_unit_displacement: bool = False
    operator_mode: str = "standard"
    projection_mode: str = "full"


@dataclass
class BaryonOperatorConfig(ChannelConfigBase):
    """Configuration for nucleon (baryon) operators."""

    operator_mode: str = "det_abs"
    flux_exp_alpha: float = 1.0


@dataclass
class GlueballOperatorConfig(ChannelConfigBase):
    """Configuration for glueball plaquette operators."""

    operator_mode: str | None = None
    use_action_form: bool = False
    use_momentum_projection: bool = False
    momentum_axis: int = 0
    momentum_mode_max: int = 3


@dataclass
class TensorOperatorConfig(ChannelConfigBase):
    """Configuration for spin-2 traceless tensor operators."""

    position_dims: tuple[int, int, int] | None = None
    momentum_axis: int = 0
    momentum_mode_max: int = 4
    projection_length: float | None = None


@dataclass
class ElectroweakOperatorConfig(ChannelConfigBase):
    """Configuration for electroweak (U(1)/SU(2)) operators."""

    epsilon_d: float | None = None
    epsilon_clone: float | None = None
    lambda_alg: float = 0.0
    su2_operator_mode: str = "standard"
    enable_walker_type_split: bool = False
    enable_directed_variants: bool = True
    enable_parity_velocity: bool = True


@dataclass
class MultiscaleConfig:
    """Configuration for multiscale operator computation.

    When ``n_scales == 1`` (default), no multiscale processing is applied
    and the pipeline produces identical output to the single-scale path.
    """

    n_scales: int = 1
    mode: str = "companion"  # "companion" | "kernel" | "both"
    kernel_type: str = "gaussian"  # gaussian | exponential | tophat | shell
    distance_method: str = "auto"  # floyd-warshall | tropical | auto
    distance_batch_size: int = 4
    scale_calibration_frames: int = 8
    scale_q_low: float = 0.05
    scale_q_high: float = 0.95
    max_scale_samples: int = 500_000
    min_scale: float = 1e-6
    scales: list[float] | None = None  # User-specified (overrides auto)
    edge_weight_mode: str = "riemannian_kernel_volume"


@dataclass
class CorrelatorConfig:
    """Configuration for batched correlator computation."""

    max_lag: int = 80
    use_connected: bool = True
