"""Mass extraction pipeline: Bayesian multi-exponential fits for particle masses.

Public API
----------
Pipeline:
    :func:`extract_masses`, :func:`extract_masses_multi_run`

Configuration:
    :class:`MassExtractionConfig`, :class:`ChannelGroupConfig`,
    :class:`CovarianceConfig`, :class:`ChannelFitConfig`,
    :class:`PriorConfig`, :class:`FitConfig`

Results:
    :class:`MassExtractionResult`, :class:`ChannelMassResult`,
    :class:`FitDiagnostics`, :class:`EffectiveMassResult`

Building blocks:
    :func:`correlators_to_gvar`, :func:`multi_run_correlators_to_gvar`,
    :func:`build_all_models`, :func:`build_combined_prior`,
    :func:`run_fit`, :func:`run_stability_scan`,
    :func:`compute_effective_mass`, :func:`compute_effective_mass_for_all`
"""

from .config import (
    ChannelFitConfig,
    ChannelGroupConfig,
    CovarianceConfig,
    FitConfig,
    MassExtractionConfig,
    PriorConfig,
)
from .data_preparation import (
    correlator_tensor_to_numpy,
    correlators_to_gvar,
    multi_run_correlators_to_gvar,
    operator_series_to_correlator_samples,
)
from .effective_mass import compute_effective_mass, compute_effective_mass_for_all
from .fitting import run_fit, run_stability_scan
from .models import (
    build_all_models,
    build_channel_group_models,
    build_corr2_model,
    get_parameter_keys,
)
from .pipeline import extract_masses, extract_masses_multi_run
from .priors import build_combined_prior, build_prior_for_group
from .results import (
    ChannelMassResult,
    EffectiveMassResult,
    extract_channel_results,
    extract_diagnostics,
    FitDiagnostics,
    MassExtractionResult,
)


__all__ = [
    # Config
    "ChannelFitConfig",
    "ChannelGroupConfig",
    # Results
    "ChannelMassResult",
    "CovarianceConfig",
    "EffectiveMassResult",
    "FitConfig",
    "FitDiagnostics",
    "MassExtractionConfig",
    "MassExtractionResult",
    "PriorConfig",
    # Models
    "build_all_models",
    "build_channel_group_models",
    # Priors
    "build_combined_prior",
    "build_corr2_model",
    "build_prior_for_group",
    # Effective mass
    "compute_effective_mass",
    "compute_effective_mass_for_all",
    # Data preparation
    "correlator_tensor_to_numpy",
    "correlators_to_gvar",
    "extract_channel_results",
    "extract_diagnostics",
    # Pipeline
    "extract_masses",
    "extract_masses_multi_run",
    "get_parameter_keys",
    "multi_run_correlators_to_gvar",
    "operator_series_to_correlator_samples",
    # Fitting
    "run_fit",
    "run_stability_scan",
]
