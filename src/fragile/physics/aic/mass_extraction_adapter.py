"""Adapter from ``new_channels`` output dataclasses to the AIC mass-extraction pipeline.

Mirrors the Bayesian adapter at ``new_channels/mass_extraction_adapter.py`` but uses
the AIC sliding-window convolutional fit (``extract_mass_aic``) instead.

By default, pre-computed correlators from the ``new_channels`` outputs are used
directly (the pair-based propagation work is preserved).  Set ``from_series=True``
to recompute correlators from the operator series via FFT autocorrelation.

Example::

    from fragile.physics.aic.mass_extraction_adapter import (
        extract_masses_aic_from_channels,
    )

    results = extract_masses_aic_from_channels(meson_out, baryon_out, dt=1.0)
    for name, r in results.items():
        print(f"{name}: mass={r.mass_fit['mass']:.4f}")
"""

from __future__ import annotations

from typing import Any, Callable

from torch import Tensor

from fragile.physics.new_channels.baryon_triplet_channels import (
    BaryonTripletCorrelatorOutput,
)
from fragile.physics.new_channels.correlator_channels import (
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.physics.new_channels.fitness_bilinear_channels import (
    FitnessBilinearOutput,
)
from fragile.physics.new_channels.fitness_pseudoscalar_channels import (
    FitnessPseudoscalarOutput,
)
from fragile.physics.new_channels.glueball_color_channels import (
    GlueballColorCorrelatorOutput,
)
from fragile.physics.new_channels.meson_phase_channels import (
    MesonPhaseCorrelatorOutput,
)
from fragile.physics.new_channels.multiscale_strong_force import (
    MultiscaleStrongForceOutput,
)
from fragile.physics.new_channels.tensor_momentum_channels import (
    TensorMomentumCorrelatorOutput,
)
from fragile.physics.new_channels.vector_meson_channels import (
    VectorMesonCorrelatorOutput,
)


# Type alias: channel_name → (correlator [max_lag+1], series [T])
_CorrelatorPairs = dict[str, tuple[Tensor, Tensor]]

# Union of all supported output types
ChannelOutput = (
    MesonPhaseCorrelatorOutput
    | BaryonTripletCorrelatorOutput
    | VectorMesonCorrelatorOutput
    | GlueballColorCorrelatorOutput
    | TensorMomentumCorrelatorOutput
    | MultiscaleStrongForceOutput
    | FitnessPseudoscalarOutput
    | FitnessBilinearOutput
)


# ---------------------------------------------------------------------------
# Core fitting function
# ---------------------------------------------------------------------------


def _fit_channel(
    channel_name: str,
    correlator: Tensor,
    series: Tensor,
    dt: float,
    config: CorrelatorConfig,
) -> ChannelCorrelatorResult:
    """Build a :class:`ChannelCorrelatorResult` from a pre-computed correlator.

    Reuses the logic from ``multiscale_strong_force._build_result_from_precomputed_correlator``.

    Args:
        channel_name: Human-readable channel identifier.
        correlator: Pre-computed correlator ``C(τ)`` with shape ``[max_lag+1]``.
        series: Operator time series with shape ``[T]``.
        dt: Temporal spacing (lattice units).
        config: AIC / linear fitting configuration.

    Returns:
        Fully populated :class:`ChannelCorrelatorResult`.
    """
    corr_t = correlator.float()
    effective_mass = compute_effective_mass_torch(corr_t, dt)

    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(corr_t.abs(), dt, config)
        window_data: dict[str, Any] = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(corr_t, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(corr_t, dt, config)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=corr_t,
        correlator_err=None,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series.float(),
        n_samples=int(series.numel()),
        dt=dt,
        **window_data,
    )


# ---------------------------------------------------------------------------
# Per-type extractors — each returns _CorrelatorPairs
# ---------------------------------------------------------------------------


def extract_meson_phase_aic(
    output: MesonPhaseCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract pseudoscalar and scalar correlator pairs from a meson-phase output."""
    suffix = "connected" if use_connected else "raw"
    return {
        f"{prefix}pseudoscalar": (
            getattr(output, f"pseudoscalar_{suffix}"),
            output.operator_pseudoscalar_series,
        ),
        f"{prefix}scalar": (
            getattr(output, f"scalar_{suffix}"),
            output.operator_scalar_series,
        ),
    }


def extract_baryon_aic(
    output: BaryonTripletCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract nucleon correlator pair from a baryon-triplet output."""
    suffix = "connected" if use_connected else "raw"
    return {
        f"{prefix}nucleon": (
            getattr(output, f"correlator_{suffix}"),
            output.operator_baryon_series,
        ),
    }


def extract_vector_meson_aic(
    output: VectorMesonCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract vector and axial-vector correlator pairs.

    The ``operator_vector_series`` has shape ``[T, 3]``; we reduce to ``[T]``
    via ``.norm(dim=-1)`` so the series is a scalar time series suitable for
    AIC fitting.
    """
    suffix = "connected" if use_connected else "raw"
    vector_series = output.operator_vector_series
    if vector_series.dim() > 1:
        vector_series = vector_series.norm(dim=-1)
    axial_series = output.operator_axial_vector_series
    if axial_series.dim() > 1:
        axial_series = axial_series.norm(dim=-1)
    return {
        f"{prefix}vector": (
            getattr(output, f"vector_{suffix}"),
            vector_series,
        ),
        f"{prefix}axial_vector": (
            getattr(output, f"axial_vector_{suffix}"),
            axial_series,
        ),
    }


def extract_glueball_aic(
    output: GlueballColorCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract glueball correlator pair from a glueball-color output."""
    suffix = "connected" if use_connected else "raw"
    return {
        f"{prefix}glueball": (
            getattr(output, f"correlator_{suffix}"),
            output.operator_glueball_series,
        ),
    }


def extract_tensor_momentum_aic(
    output: TensorMomentumCorrelatorOutput,
    *,
    use_connected: bool = True,
    momentum_mode: int | None = None,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract tensor correlator pairs at one or all momentum modes.

    Args:
        output: Tensor momentum channel output.
        use_connected: Use connected correlators.
        momentum_mode: If an ``int``, extract a single mode.  If ``None``,
            extract *all* modes as separate channels (``tensor_mode_0``, etc.).
        prefix: Key prefix.

    Returns:
        Dictionary of ``(correlator, series)`` pairs.
    """
    suffix = "connected" if use_connected else "raw"
    # momentum_contracted_correlator_* has shape [n_modes, max_lag+1]
    contracted = getattr(output, f"momentum_contracted_correlator_{suffix}")
    # cos/sin shapes: [n_modes, 5, T]
    cos_all = output.momentum_operator_cos_series
    sin_all = output.momentum_operator_sin_series

    def _series_for_mode(m: int) -> Tensor:
        cos = cos_all[m]  # [5, T]
        sin = sin_all[m]  # [5, T]
        return (cos**2 + sin**2).sqrt().sum(dim=0)  # [T]

    if momentum_mode is not None:
        return {
            f"{prefix}tensor": (
                contracted[momentum_mode],
                _series_for_mode(momentum_mode),
            ),
        }

    n_modes = contracted.shape[0]
    pairs: _CorrelatorPairs = {}
    for m in range(n_modes):
        pairs[f"{prefix}tensor_mode_{m}"] = (
            contracted[m],
            _series_for_mode(m),
        )
    return pairs


def extract_fitness_pseudoscalar_aic(
    output: FitnessPseudoscalarOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract fitness pseudoscalar, scalar-variance, and axial correlator pairs."""
    suffix = "connected" if use_connected else "raw"
    return {
        f"{prefix}fitness_pseudoscalar": (
            getattr(output, f"cpp_{suffix}"),
            output.operator_pseudoscalar_series,
        ),
        f"{prefix}fitness_scalar_variance": (
            getattr(output, f"css_{suffix}"),
            output.operator_scalar_variance_series,
        ),
        f"{prefix}fitness_axial": (
            getattr(output, f"cjp_{suffix}"),
            output.operator_axial_series,
        ),
    }


def extract_fitness_bilinear_aic(
    output: FitnessBilinearOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract fitness bilinear pseudoscalar, scalar-variance, and axial pairs."""
    suffix = "connected" if use_connected else "raw"
    return {
        f"{prefix}fitness_pseudoscalar": (
            getattr(output, f"fitness_pseudoscalar_{suffix}"),
            output.operator_fitness_pseudoscalar_series,
        ),
        f"{prefix}fitness_scalar_variance": (
            getattr(output, f"fitness_scalar_variance_{suffix}"),
            output.operator_fitness_scalar_variance_series,
        ),
        f"{prefix}fitness_axial": (
            getattr(output, f"fitness_axial_{suffix}"),
            output.operator_fitness_axial_series,
        ),
    }


def extract_multiscale_aic(
    output: MultiscaleStrongForceOutput,
    *,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Extract correlator pairs from the *best* result per channel.

    Uses ``best_results`` (one :class:`ChannelCorrelatorResult` per channel)
    rather than the full per-scale grid.
    """
    pairs: _CorrelatorPairs = {}
    for channel_name, result in output.best_results.items():
        pairs[f"{prefix}{channel_name}"] = (result.correlator, result.series)
    return pairs


# ---------------------------------------------------------------------------
# Dispatcher mapping
# ---------------------------------------------------------------------------

_AIC_EXTRACTORS: dict[type, Callable] = {
    MesonPhaseCorrelatorOutput: extract_meson_phase_aic,
    BaryonTripletCorrelatorOutput: extract_baryon_aic,
    VectorMesonCorrelatorOutput: extract_vector_meson_aic,
    GlueballColorCorrelatorOutput: extract_glueball_aic,
    TensorMomentumCorrelatorOutput: extract_tensor_momentum_aic,
    MultiscaleStrongForceOutput: extract_multiscale_aic,
    FitnessPseudoscalarOutput: extract_fitness_pseudoscalar_aic,
    FitnessBilinearOutput: extract_fitness_bilinear_aic,
}


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


def collect_correlator_pairs(
    *outputs: ChannelOutput,
    use_connected: bool = True,
    prefix: str = "",
) -> _CorrelatorPairs:
    """Combine any number of ``new_channels`` outputs into correlator pairs.

    Auto-dispatches to the correct per-type extractor.  Duplicate keys raise
    ``ValueError``.

    Args:
        *outputs: One or more ``new_channels`` output dataclass instances.
        use_connected: If ``True`` (default), use connected correlators.
        prefix: Optional key prefix for all channels.

    Returns:
        Merged ``_CorrelatorPairs`` dict.
    """
    merged: _CorrelatorPairs = {}

    for out in outputs:
        extractor = _AIC_EXTRACTORS.get(type(out))
        if extractor is None:
            raise TypeError(
                f"Unsupported output type {type(out).__name__}. "
                f"Supported: {', '.join(t.__name__ for t in _AIC_EXTRACTORS)}"
            )
        # MultiscaleStrongForceOutput does not accept use_connected
        if isinstance(out, MultiscaleStrongForceOutput):
            pairs = extractor(out, prefix=prefix)
        else:
            pairs = extractor(out, use_connected=use_connected, prefix=prefix)

        for key in pairs:
            if key in merged:
                raise ValueError(
                    f"Duplicate correlator key '{key}'. Use `prefix` to disambiguate."
                )
        merged.update(pairs)

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def extract_masses_aic_from_channels(
    *outputs: ChannelOutput,
    dt: float,
    config: CorrelatorConfig | None = None,
    use_connected: bool = True,
    prefix: str = "",
    from_series: bool = False,
) -> dict[str, ChannelCorrelatorResult]:
    """One-liner: convert ``new_channels`` outputs and run AIC mass extraction.

    Example::

        from fragile.physics.aic.mass_extraction_adapter import (
            extract_masses_aic_from_channels,
        )

        results = extract_masses_aic_from_channels(meson_out, baryon_out, dt=1.0)
        for name, r in results.items():
            print(f"{name}: mass={r.mass_fit['mass']:.4f}")

    Args:
        *outputs: One or more ``new_channels`` output dataclass instances.
        dt: Temporal spacing (required — not stored in the output dataclasses).
        config: Optional :class:`CorrelatorConfig`; ``None`` uses defaults.
        use_connected: If ``True`` (default), use connected correlators.
        prefix: Optional key prefix for all channels.
        from_series: If ``True``, recompute correlators from operator series via
            FFT autocorrelation instead of using the pre-computed correlators.

    Returns:
        Dictionary mapping channel names to :class:`ChannelCorrelatorResult`.
    """
    if config is None:
        config = CorrelatorConfig()

    pairs = collect_correlator_pairs(*outputs, use_connected=use_connected, prefix=prefix)

    results: dict[str, ChannelCorrelatorResult] = {}
    for name, (correlator, series) in pairs.items():
        if from_series:
            results[name] = compute_channel_correlator(
                series=series,
                dt=dt,
                config=config,
                channel_name=name,
            )
        else:
            results[name] = _fit_channel(
                channel_name=name,
                correlator=correlator,
                series=series,
                dt=dt,
                config=config,
            )

    return results
