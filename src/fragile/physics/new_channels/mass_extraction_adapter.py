"""Adapter from ``new_channels`` output dataclasses to the mass-extraction pipeline.

Provides per-type extractor functions, a combiner that auto-dispatches on type,
and a one-liner convenience entry point::

    from fragile.physics.new_channels.mass_extraction_adapter import (
        extract_masses_from_channels,
    )

    result = extract_masses_from_channels(meson_output, baryon_output)
"""

from __future__ import annotations

from torch import Tensor

from fragile.physics.mass_extraction import (
    extract_masses,
    MassExtractionConfig,
    MassExtractionResult,
)
from fragile.physics.new_channels.baryon_triplet_channels import (
    BaryonTripletCorrelatorOutput,
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
from fragile.physics.operators.pipeline import PipelineResult


# Type alias for the (correlators, operators) pair returned by extractors
_ExtractResult = tuple[dict[str, Tensor], dict[str, Tensor]]

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
# Per-type extractors
# ---------------------------------------------------------------------------


def extract_meson_phase(
    output: MesonPhaseCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract pseudoscalar and scalar correlators from a meson-phase output."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}pseudoscalar": getattr(output, f"pseudoscalar_{corr_suffix}"),
        f"{prefix}scalar": getattr(output, f"scalar_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}pseudoscalar": output.operator_pseudoscalar_series,
        f"{prefix}scalar": output.operator_scalar_series,
    }
    return correlators, operators


def extract_baryon(
    output: BaryonTripletCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract nucleon correlator from a baryon-triplet output."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}nucleon": getattr(output, f"correlator_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}nucleon": output.operator_baryon_series,
    }
    return correlators, operators


def extract_vector_meson(
    output: VectorMesonCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract vector and axial-vector correlators from a vector-meson output."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}vector": getattr(output, f"vector_{corr_suffix}"),
        f"{prefix}axial_vector": getattr(output, f"axial_vector_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}vector": output.operator_vector_series,
        f"{prefix}axial_vector": output.operator_axial_vector_series,
    }
    return correlators, operators


def extract_glueball(
    output: GlueballColorCorrelatorOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract glueball correlator from a glueball-color output."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}glueball": getattr(output, f"correlator_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}glueball": output.operator_glueball_series,
    }
    return correlators, operators


def extract_tensor_momentum(
    output: TensorMomentumCorrelatorOutput,
    *,
    use_connected: bool = True,
    momentum_mode: int = 0,
    prefix: str = "",
) -> _ExtractResult:
    """Extract contracted tensor correlator at a given momentum mode.

    The 5-component tensor structure is already contracted in
    ``momentum_contracted_correlator*``.  We select a single momentum mode
    (default 0, the lowest non-zero mode).

    For the operator series we combine cos and sin quadratures via
    ``sqrt(cos^2 + sin^2)`` summed over components to produce a scalar
    time series ``[T]``.
    """
    corr_suffix = "connected" if use_connected else "raw"
    # momentum_contracted_correlator_* has shape [n_modes, max_lag+1]
    contracted = getattr(output, f"momentum_contracted_correlator_{corr_suffix}")
    correlators: dict[str, Tensor] = {
        f"{prefix}tensor": contracted[momentum_mode],  # [max_lag+1]
    }
    # Operator: combine cos/sin across components → scalar [T]
    # cos/sin shapes: [n_modes, 5, T]
    cos = output.momentum_operator_cos_series[momentum_mode]  # [5, T]
    sin = output.momentum_operator_sin_series[momentum_mode]  # [5, T]
    amplitude = (cos**2 + sin**2).sqrt().sum(dim=0)  # [T]
    operators: dict[str, Tensor] = {
        f"{prefix}tensor": amplitude,
    }
    return correlators, operators


def extract_multiscale(
    output: MultiscaleStrongForceOutput,
    *,
    prefix: str = "",
) -> _ExtractResult:
    """Extract correlators and operators from a multiscale output.

    For each channel present in ``per_scale_results``, the per-scale
    ``ChannelCorrelatorResult.correlator`` tensors are stacked to produce
    a ``[S, max_lag+1]`` correlator matrix (one row per scale).

    Operator series come from ``series_by_channel`` with shape ``[S, T]``.
    """
    correlators: dict[str, Tensor] = {}
    operators: dict[str, Tensor] = {}
    import torch

    for channel_name, scale_results in output.per_scale_results.items():
        key = f"{prefix}{channel_name}"
        # Stack correlators across scales → [S, max_lag+1]
        corr_stack = torch.stack([r.correlator for r in scale_results])
        correlators[key] = corr_stack
        # Operator series [S, T]
        if channel_name in output.series_by_channel:
            operators[key] = output.series_by_channel[channel_name]

    return correlators, operators


def extract_fitness_pseudoscalar(
    output: FitnessPseudoscalarOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract fitness pseudoscalar, scalar variance, and axial correlators."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}fitness_pseudoscalar": getattr(output, f"cpp_{corr_suffix}"),
        f"{prefix}fitness_scalar_variance": getattr(output, f"css_{corr_suffix}"),
        f"{prefix}fitness_axial": getattr(output, f"cjp_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}fitness_pseudoscalar": output.operator_pseudoscalar_series,
        f"{prefix}fitness_scalar_variance": output.operator_scalar_variance_series,
        f"{prefix}fitness_axial": output.operator_axial_series,
    }
    return correlators, operators


def extract_fitness_bilinear(
    output: FitnessBilinearOutput,
    *,
    use_connected: bool = True,
    prefix: str = "",
) -> _ExtractResult:
    """Extract fitness bilinear pseudoscalar, scalar variance, and axial correlators."""
    corr_suffix = "connected" if use_connected else "raw"
    correlators: dict[str, Tensor] = {
        f"{prefix}fitness_pseudoscalar": getattr(output, f"fitness_pseudoscalar_{corr_suffix}"),
        f"{prefix}fitness_scalar_variance": getattr(
            output, f"fitness_scalar_variance_{corr_suffix}"
        ),
        f"{prefix}fitness_axial": getattr(output, f"fitness_axial_{corr_suffix}"),
    }
    operators: dict[str, Tensor] = {
        f"{prefix}fitness_pseudoscalar": output.operator_fitness_pseudoscalar_series,
        f"{prefix}fitness_scalar_variance": output.operator_fitness_scalar_variance_series,
        f"{prefix}fitness_axial": output.operator_fitness_axial_series,
    }
    return correlators, operators


# ---------------------------------------------------------------------------
# Dispatcher mapping
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[type, callable] = {
    MesonPhaseCorrelatorOutput: extract_meson_phase,
    BaryonTripletCorrelatorOutput: extract_baryon,
    VectorMesonCorrelatorOutput: extract_vector_meson,
    GlueballColorCorrelatorOutput: extract_glueball,
    TensorMomentumCorrelatorOutput: extract_tensor_momentum,
    MultiscaleStrongForceOutput: extract_multiscale,
    FitnessPseudoscalarOutput: extract_fitness_pseudoscalar,
    FitnessBilinearOutput: extract_fitness_bilinear,
}


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------


def collect_correlators(
    *outputs: ChannelOutput,
    use_connected: bool = True,
    prefix: str = "",
) -> PipelineResult:
    """Combine any number of ``new_channels`` outputs into a :class:`PipelineResult`.

    Auto-dispatches to the correct per-type extractor.  All correlator and
    operator dicts are merged; duplicate keys raise ``ValueError``.

    Args:
        *outputs: One or more ``new_channels`` output dataclass instances.
        use_connected: If ``True`` (default), use connected correlators.
        prefix: Optional key prefix for all channels.

    Returns:
        A :class:`PipelineResult` ready for :func:`extract_masses`.
    """
    all_correlators: dict[str, Tensor] = {}
    all_operators: dict[str, Tensor] = {}

    for out in outputs:
        extractor = _EXTRACTORS.get(type(out))
        if extractor is None:
            raise TypeError(
                f"Unsupported output type {type(out).__name__}. "
                f"Supported: {', '.join(t.__name__ for t in _EXTRACTORS)}"
            )
        # MultiscaleStrongForceOutput does not accept use_connected
        if isinstance(out, MultiscaleStrongForceOutput):
            corrs, ops = extractor(out, prefix=prefix)
        else:
            corrs, ops = extractor(out, use_connected=use_connected, prefix=prefix)

        # Check for key collisions
        for key in corrs:
            if key in all_correlators:
                raise ValueError(
                    f"Duplicate correlator key '{key}'. " "Use `prefix` to disambiguate."
                )
        all_correlators.update(corrs)
        all_operators.update(ops)

    return PipelineResult(correlators=all_correlators, operators=all_operators)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def extract_masses_from_channels(
    *outputs: ChannelOutput,
    config: MassExtractionConfig | None = None,
    use_connected: bool = True,
    prefix: str = "",
) -> MassExtractionResult:
    """One-liner: convert ``new_channels`` outputs and run mass extraction.

    Example::

        from fragile.physics.new_channels.mass_extraction_adapter import (
            extract_masses_from_channels,
        )

        result = extract_masses_from_channels(meson_out, baryon_out)
        print(result.channels["pseudoscalar"].ground_state_mass)

    Args:
        *outputs: One or more ``new_channels`` output dataclass instances.
        config: Optional :class:`MassExtractionConfig`; ``None`` uses defaults.
        use_connected: If ``True`` (default), use connected correlators.
        prefix: Optional key prefix for all channels.

    Returns:
        :class:`MassExtractionResult` with extracted masses and diagnostics.
    """
    pipeline_result = collect_correlators(*outputs, use_connected=use_connected, prefix=prefix)
    return extract_masses(pipeline_result, config=config)
