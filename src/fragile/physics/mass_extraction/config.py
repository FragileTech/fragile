"""Configuration dataclasses for the mass extraction pipeline.

Provides configuration for covariance estimation, channel fitting,
prior construction, fitting engine, and top-level pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CovarianceConfig:
    """Configuration for covariance matrix estimation.

    Controls how the covariance between correlator time slices is estimated
    from the raw data.
    """

    method: str = "uncorrelated"  # block_jackknife | bootstrap | uncorrelated
    n_bootstrap: int = 200
    block_size: int = 10
    seed: int = 42


@dataclass
class ChannelFitConfig:
    """Per-channel fitting parameters.

    Controls which time slices to include and how many exponentials to use.
    """

    tmin: int = 2
    tmax: int | None = None  # None = use full range
    tp: int | None = None  # periodic boundary time extent (None = no folding)
    nexp: int = 2
    nexp_osc: int = 0  # oscillating states (staggered fermions)
    use_log_dE: bool = True
    use_log_amplitudes: bool = False


@dataclass
class PriorConfig:
    """Prior specification for Bayesian fitting.

    String specs use gvar notation, e.g. ``"0.5(5)"`` means 0.5 +/- 0.5.
    """

    dE_ground: str = "0.5(5)"
    dE_excited: str = "0.5(5)"
    amplitude: str = "0.5(5)"
    use_fastfit_seeding: bool = True


@dataclass
class FitConfig:
    """Global fitting engine parameters."""

    svdcut: float = 1e-4
    tol: float = 1e-8
    maxit: int = 2000
    use_chained: bool = False
    nterm: int | None = None  # marginalization term count


@dataclass
class ChannelGroupConfig:
    """Configuration for a group of correlators sharing the same energy spectrum.

    All correlator keys in a group are fit simultaneously with shared ``dE``
    parameters and independent amplitudes.
    """

    name: str = "channel"
    correlator_keys: list[str] = field(default_factory=list)
    channel_type: str = "meson"  # meson | baryon | glueball | tensor
    fit: ChannelFitConfig = field(default_factory=ChannelFitConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    operator_modes: list[str] = field(default_factory=list)
    include_multiscale: bool = False


@dataclass
class MassExtractionConfig:
    """Top-level configuration for the mass extraction pipeline.

    ``MassExtractionConfig()`` works out of the box with sensible defaults.
    """

    channel_groups: list[ChannelGroupConfig] = field(default_factory=list)
    covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    compute_effective_mass: bool = True
    effective_mass_method: str = "log_ratio"  # log_ratio | cosh
    effective_mass_dt: float = 1.0
