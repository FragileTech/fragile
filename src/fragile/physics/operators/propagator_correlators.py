"""Source-pair propagation correlators.

Computes propagator-style correlators where the companion topology is frozen
at the source time and the observable is re-evaluated at the sink time:

    C(tau) = < O(t, pairs_t) * O(t+tau, pairs_t) >

This tracks how a specific color-singlet state (defined by the pair at source
time) evolves forward in Monte Carlo time — the physically correct propagator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from fragile.physics.qft_utils.companions import (
    build_companion_pair_indices,
)

from .baryon_operators import (
    _baryon_flux_weight_from_plaquette,
    _compute_determinants_for_indices,
    _compute_score_ordered_determinants_for_indices,
    _compute_triplet_plaquette_for_indices,
    _resolve_baryon_operator_mode,
)
from .config import (
    BaryonOperatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    VectorOperatorConfig,
)
from .glueball_operators import (
    _compute_color_plaquette_for_triplets,
    _glueball_observable_from_plaquette,
    _resolve_glueball_operator_mode,
)
from .meson_operators import (
    _compute_inner_products_for_pairs,
    _orient_inner_products_by_scores,
    _resolve_meson_operator_mode,
    _weight_inner_products_by_score_gap,
)
from .preparation import PreparedChannelData
from .vector_operators import (
    _compute_pair_observables,
    _resolve_vector_operator_mode,
    _resolve_vector_projection_mode,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PropagatorChannelResult:
    """Propagator correlator output for one channel."""

    raw: Tensor  # [max_lag+1] raw correlator
    connected: Tensor  # [max_lag+1] connected correlator
    counts: Tensor  # [max_lag+1] valid pair counts per lag
    mean_source: float  # mean source observable (for connected subtraction)


@dataclass
class PropagatorResult:
    """Output of compute_propagator_correlators()."""

    channels: dict[str, PropagatorChannelResult]
    prepared_data: PreparedChannelData | None = None


# ---------------------------------------------------------------------------
# Generic lag loop
# ---------------------------------------------------------------------------


def _propagator_lag_loop(
    source_obs: Tensor,
    source_valid: Tensor,
    compute_sink_fn: Callable[[int, int], tuple[Tensor, Tensor]],
    max_lag: int,
    use_connected: bool,
    is_vector: bool = False,
) -> PropagatorChannelResult:
    """Generic lag loop for propagator correlators.

    Args:
        source_obs: [T, N, ...] source observable, pre-computed.
        source_valid: [T, N] source validity mask.
        compute_sink_fn: (lag, source_len) -> (sink_obs, sink_valid).
            sink_obs has same shape as source_obs[:source_len].
            sink_valid has shape [source_len, N].
        max_lag: Maximum temporal lag.
        use_connected: Whether to compute connected correlator.
        is_vector: If True, contract over last dim (dot product for vectors).
    """
    T = source_obs.shape[0]
    n_lags = max(0, int(max_lag)) + 1
    device = source_obs.device

    raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.float32, device=device)

    # Compute mean source for connected subtraction
    src_weights = source_valid.to(torch.float32)
    total_valid = src_weights.sum()
    if is_vector:
        # source_obs: [T, N, 3] — average over valid walkers
        src_weighted = source_obs * src_weights.unsqueeze(-1)
        mean_source_vec = src_weighted.sum(dim=(0, 1)) / total_valid.clamp(min=1.0)
        mean_source_scalar = mean_source_vec.dot(mean_source_vec).item()
    else:
        # source_obs: [T, N] or [T, N, P]
        if source_obs.ndim == 3:
            src_weighted = source_obs * src_weights.unsqueeze(-1)
        else:
            src_weighted = source_obs * src_weights
        mean_val = (src_weighted.sum() / total_valid.clamp(min=1.0)).item()
        mean_source_scalar = mean_val ** 2

    for lag in range(n_lags):
        source_len = T - lag
        if source_len <= 0:
            break

        sink_obs, sink_valid = compute_sink_fn(lag, source_len)
        src_slice = source_obs[:source_len]
        src_valid_slice = source_valid[:source_len]

        # Combined validity
        joint_valid = src_valid_slice & sink_valid

        if is_vector:
            # source/sink: [source_len, N, 3]
            jv = joint_valid.to(torch.float32).unsqueeze(-1)
            product = (src_slice * sink_obs * jv).sum(dim=-1)  # dot product
            corr_val = product.sum()
            count_val = joint_valid.to(torch.float32).sum()
        else:
            if src_slice.ndim == 3:
                jv = joint_valid.to(torch.float32).unsqueeze(-1)
            else:
                jv = joint_valid.to(torch.float32)
            product = src_slice * sink_obs * jv
            corr_val = product.sum()
            count_val = joint_valid.to(torch.float32).sum()

        if count_val > 0:
            raw[lag] = corr_val / count_val
        counts[lag] = count_val

    connected = raw.clone()
    if use_connected:
        connected = connected - mean_source_scalar

    return PropagatorChannelResult(
        raw=raw,
        connected=connected,
        counts=counts,
        mean_source=mean_source_scalar,
    )


# ---------------------------------------------------------------------------
# Meson propagator
# ---------------------------------------------------------------------------


def compute_meson_propagator(
    data: PreparedChannelData,
    config: MesonOperatorConfig,
    max_lag: int = 80,
    use_connected: bool = True,
) -> dict[str, PropagatorChannelResult]:
    """Compute meson propagator correlators (scalar and pseudoscalar).

    Freezes companion topology at source time and re-evaluates the meson
    observable at sink time with that frozen topology.
    """
    device = data.device
    T = int(data.color.shape[0])
    n_lags = max(0, int(max_lag)) + 1

    if T == 0:
        empty = PropagatorChannelResult(
            raw=torch.zeros(n_lags, device=device),
            connected=torch.zeros(n_lags, device=device),
            counts=torch.zeros(n_lags, device=device),
            mean_source=0.0,
        )
        return {"meson_scalar": empty, "meson_pseudoscalar": _clone_result(empty)}

    resolved_mode = _resolve_meson_operator_mode(config.operator_mode)
    pair_selection = str(config.pair_selection).strip().lower()

    # Build pair indices for ALL frames (source topology)
    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=pair_selection,
    )

    # Compute source inner products for all T frames
    inner_all, valid_all = _compute_inner_products_for_pairs(
        color=data.color,
        color_valid=data.color_valid,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=data.eps,
    )

    # Apply score modes to source
    if resolved_mode == "score_directed" and data.scores is not None:
        inner_all, valid_all = _orient_inner_products_by_scores(
            inner=inner_all, valid=valid_all,
            scores=data.scores, pair_indices=pair_indices,
        )
    elif resolved_mode == "score_weighted" and data.scores is not None:
        inner_all, valid_all = _weight_inner_products_by_score_gap(
            inner=inner_all, valid=valid_all,
            scores=data.scores, pair_indices=pair_indices,
        )

    # Extract source observables
    if resolved_mode == "abs2_vacsub":
        scalar_obs_all = inner_all.abs().square().float()
    else:
        scalar_obs_all = inner_all.real.float()
    pseudo_obs_all = inner_all.imag.float()

    # Reduce valid from [T, N, P] to [T, N] by any-valid across pairs
    valid_any = valid_all.any(dim=-1)

    # Per-pair source obs averaged over P dim
    scalar_source = _mean_over_pairs(scalar_obs_all, valid_all)  # [T, N]
    pseudo_source = _mean_over_pairs(pseudo_obs_all, valid_all)  # [T, N]

    def make_sink_fn(obs_type: str):
        def compute_sink(lag: int, source_len: int) -> tuple[Tensor, Tensor]:
            # Re-compute inner products at sink time with source companion topology
            src_pairs = pair_indices[:source_len]
            src_structural = structural_valid[:source_len]
            sink_color = data.color[lag:lag + source_len]
            sink_color_valid = data.color_valid[lag:lag + source_len]

            sink_inner, sink_valid = _compute_inner_products_for_pairs(
                color=sink_color,
                color_valid=sink_color_valid,
                pair_indices=src_pairs,
                structural_valid=src_structural,
                eps=data.eps,
            )

            if resolved_mode == "score_directed" and data.scores is not None:
                sink_inner, sink_valid = _orient_inner_products_by_scores(
                    inner=sink_inner, valid=sink_valid,
                    scores=data.scores[lag:lag + source_len],
                    pair_indices=src_pairs,
                )
            elif resolved_mode == "score_weighted" and data.scores is not None:
                sink_inner, sink_valid = _weight_inner_products_by_score_gap(
                    inner=sink_inner, valid=sink_valid,
                    scores=data.scores[lag:lag + source_len],
                    pair_indices=src_pairs,
                )

            if obs_type == "scalar":
                if resolved_mode == "abs2_vacsub":
                    sink_obs = sink_inner.abs().square().float()
                else:
                    sink_obs = sink_inner.real.float()
            else:
                sink_obs = sink_inner.imag.float()

            sink_obs_avg = _mean_over_pairs(sink_obs, sink_valid)
            sink_valid_any = sink_valid.any(dim=-1)
            return sink_obs_avg, sink_valid_any

        return compute_sink

    scalar_result = _propagator_lag_loop(
        source_obs=scalar_source,
        source_valid=valid_any,
        compute_sink_fn=make_sink_fn("scalar"),
        max_lag=max_lag,
        use_connected=use_connected,
    )
    pseudo_result = _propagator_lag_loop(
        source_obs=pseudo_source,
        source_valid=valid_any,
        compute_sink_fn=make_sink_fn("pseudo"),
        max_lag=max_lag,
        use_connected=use_connected,
    )

    return {"meson_scalar": scalar_result, "meson_pseudoscalar": pseudo_result}


# ---------------------------------------------------------------------------
# Vector propagator
# ---------------------------------------------------------------------------


def compute_vector_propagator(
    data: PreparedChannelData,
    config: VectorOperatorConfig,
    max_lag: int = 80,
    use_connected: bool = True,
) -> dict[str, PropagatorChannelResult]:
    """Compute vector propagator correlators (vector and axial).

    Freezes companion topology at source time and re-evaluates
    vector observables at sink time with frozen topology.
    """
    device = data.device
    T = int(data.color.shape[0])
    n_lags = max(0, int(max_lag)) + 1

    if T == 0:
        empty = PropagatorChannelResult(
            raw=torch.zeros(n_lags, device=device),
            connected=torch.zeros(n_lags, device=device),
            counts=torch.zeros(n_lags, device=device),
            mean_source=0.0,
        )
        return {"vector_full": empty, "axial_full": _clone_result(empty)}

    if data.positions is None:
        raise ValueError("positions must be provided for vector propagator.")

    resolved_op_mode = _resolve_vector_operator_mode(config.operator_mode)
    resolved_proj_mode = _resolve_vector_projection_mode(config.projection_mode)
    pair_selection = str(config.pair_selection).strip().lower()

    pair_indices, structural_valid = build_companion_pair_indices(
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        pair_selection=pair_selection,
    )

    # Compute source pair observables for all T frames
    inner_all, disp_all, valid_all = _compute_pair_observables(
        color=data.color,
        color_valid=data.color_valid,
        positions=data.positions,
        pair_indices=pair_indices,
        structural_valid=structural_valid,
        eps=data.eps,
        use_unit_displacement=config.use_unit_displacement,
        operator_mode=resolved_op_mode,
        projection_mode=resolved_proj_mode,
        scores=data.scores,
    )

    # Build vector/axial observables: [T, N, P, 3]
    vector_obs_all = inner_all.real.float().unsqueeze(-1) * disp_all
    axial_obs_all = inner_all.imag.float().unsqueeze(-1) * disp_all

    # Average over pairs: [T, N, 3]
    vector_source = _mean_over_pairs_vec(vector_obs_all, valid_all)
    axial_source = _mean_over_pairs_vec(axial_obs_all, valid_all)
    valid_any = valid_all.any(dim=-1)

    def make_sink_fn(obs_type: str):
        def compute_sink(lag: int, source_len: int) -> tuple[Tensor, Tensor]:
            src_pairs = pair_indices[:source_len]
            src_structural = structural_valid[:source_len]
            sink_color = data.color[lag:lag + source_len]
            sink_color_valid = data.color_valid[lag:lag + source_len]
            sink_positions = data.positions[lag:lag + source_len]

            sink_scores = None
            if data.scores is not None:
                sink_scores = data.scores[lag:lag + source_len]

            sink_inner, sink_disp, sink_valid = _compute_pair_observables(
                color=sink_color,
                color_valid=sink_color_valid,
                positions=sink_positions,
                pair_indices=src_pairs,
                structural_valid=src_structural,
                eps=data.eps,
                use_unit_displacement=config.use_unit_displacement,
                operator_mode=resolved_op_mode,
                projection_mode=resolved_proj_mode,
                scores=sink_scores,
            )

            if obs_type == "vector":
                sink_obs = sink_inner.real.float().unsqueeze(-1) * sink_disp
            else:
                sink_obs = sink_inner.imag.float().unsqueeze(-1) * sink_disp

            sink_obs_avg = _mean_over_pairs_vec(sink_obs, sink_valid)
            sink_valid_any = sink_valid.any(dim=-1)
            return sink_obs_avg, sink_valid_any

        return compute_sink

    vector_result = _propagator_lag_loop(
        source_obs=vector_source,
        source_valid=valid_any,
        compute_sink_fn=make_sink_fn("vector"),
        max_lag=max_lag,
        use_connected=use_connected,
        is_vector=True,
    )
    axial_result = _propagator_lag_loop(
        source_obs=axial_source,
        source_valid=valid_any,
        compute_sink_fn=make_sink_fn("axial"),
        max_lag=max_lag,
        use_connected=use_connected,
        is_vector=True,
    )

    return {"vector_full": vector_result, "axial_full": axial_result}


# ---------------------------------------------------------------------------
# Baryon propagator
# ---------------------------------------------------------------------------


def compute_baryon_propagator(
    data: PreparedChannelData,
    config: BaryonOperatorConfig,
    max_lag: int = 80,
    use_connected: bool = True,
) -> dict[str, PropagatorChannelResult]:
    """Compute baryon propagator correlator (nucleon channel).

    Freezes companion triplet topology at source time and re-evaluates
    the baryon observable at sink time with frozen topology.
    """
    device = data.device
    T = int(data.color.shape[0])
    n_lags = max(0, int(max_lag)) + 1

    if T == 0:
        empty = PropagatorChannelResult(
            raw=torch.zeros(n_lags, device=device),
            connected=torch.zeros(n_lags, device=device),
            counts=torch.zeros(n_lags, device=device),
            mean_source=0.0,
        )
        return {"baryon_nucleon": empty}

    resolved_mode = _resolve_baryon_operator_mode(config.operator_mode)
    eps = data.eps

    # Compute source observable for all T frames
    source_obs, source_valid = _compute_baryon_obs(
        data.color, data.color_valid, data.scores,
        data.companions_distance, data.companions_clone,
        resolved_mode, eps, config.flux_exp_alpha,
    )

    def compute_sink(lag: int, source_len: int) -> tuple[Tensor, Tensor]:
        sink_color = data.color[lag:lag + source_len]
        sink_color_valid = data.color_valid[lag:lag + source_len]
        src_comp_d = data.companions_distance[:source_len]
        src_comp_c = data.companions_clone[:source_len]

        sink_scores = None
        if data.scores is not None:
            sink_scores = data.scores[lag:lag + source_len]

        sink_obs, sink_valid = _compute_baryon_obs(
            sink_color, sink_color_valid, sink_scores,
            src_comp_d, src_comp_c,
            resolved_mode, eps, config.flux_exp_alpha,
        )
        return sink_obs, sink_valid

    result = _propagator_lag_loop(
        source_obs=source_obs,
        source_valid=source_valid,
        compute_sink_fn=compute_sink,
        max_lag=max_lag,
        use_connected=use_connected,
    )

    return {"baryon_nucleon": result}


def _compute_baryon_obs(
    color: Tensor,
    color_valid: Tensor,
    scores: Tensor | None,
    companion_j: Tensor,
    companion_k: Tensor,
    resolved_mode: str,
    eps: float,
    flux_exp_alpha: float,
) -> tuple[Tensor, Tensor]:
    """Compute per-walker baryon observable and validity."""
    if resolved_mode in {"score_signed", "score_abs"}:
        if scores is None:
            raise ValueError("scores required for score-based baryon modes.")
        det, valid = _compute_score_ordered_determinants_for_indices(
            color=color, color_valid=color_valid, scores=scores,
            companion_j=companion_j, companion_k=companion_k, eps=eps,
        )
        obs = det.real.float() if resolved_mode == "score_signed" else det.abs().float()
    else:
        det, valid = _compute_determinants_for_indices(
            vectors=color, valid_vectors=color_valid,
            companion_j=companion_j, companion_k=companion_k, eps=eps,
        )
        obs = det.abs().float()

    if resolved_mode in {"flux_action", "flux_sin2", "flux_exp"}:
        pi, pi_valid = _compute_triplet_plaquette_for_indices(
            color=color, color_valid=color_valid,
            companion_j=companion_j, companion_k=companion_k, eps=eps,
        )
        flux_weight = _baryon_flux_weight_from_plaquette(
            pi=pi, operator_mode=resolved_mode, flux_exp_alpha=flux_exp_alpha,
        )
        valid = valid & pi_valid
        obs = obs * flux_weight
        obs = torch.where(valid, obs, torch.zeros_like(obs))

    return obs, valid


# ---------------------------------------------------------------------------
# Glueball propagator
# ---------------------------------------------------------------------------


def compute_glueball_propagator(
    data: PreparedChannelData,
    config: GlueballOperatorConfig,
    max_lag: int = 80,
    use_connected: bool = True,
) -> dict[str, PropagatorChannelResult]:
    """Compute glueball propagator correlator.

    Freezes companion triplet topology at source time and re-evaluates
    the glueball plaquette observable at sink time.
    """
    device = data.device
    T = int(data.color.shape[0])
    n_lags = max(0, int(max_lag)) + 1

    if T == 0:
        empty = PropagatorChannelResult(
            raw=torch.zeros(n_lags, device=device),
            connected=torch.zeros(n_lags, device=device),
            counts=torch.zeros(n_lags, device=device),
            mean_source=0.0,
        )
        return {"glueball_plaquette": empty}

    resolved_mode = _resolve_glueball_operator_mode(
        operator_mode=config.operator_mode,
        use_action_form=config.use_action_form,
    )

    # Source: compute plaquettes for all T frames
    source_pi, source_valid = _compute_color_plaquette_for_triplets(
        color=data.color,
        color_valid=data.color_valid,
        companions_distance=data.companions_distance,
        companions_clone=data.companions_clone,
        eps=data.eps,
    )
    source_obs = _glueball_observable_from_plaquette(source_pi, operator_mode=resolved_mode)

    def compute_sink(lag: int, source_len: int) -> tuple[Tensor, Tensor]:
        sink_color = data.color[lag:lag + source_len]
        sink_color_valid = data.color_valid[lag:lag + source_len]
        src_comp_d = data.companions_distance[:source_len]
        src_comp_c = data.companions_clone[:source_len]

        sink_pi, sink_valid = _compute_color_plaquette_for_triplets(
            color=sink_color,
            color_valid=sink_color_valid,
            companions_distance=src_comp_d,
            companions_clone=src_comp_c,
            eps=data.eps,
        )
        sink_obs = _glueball_observable_from_plaquette(sink_pi, operator_mode=resolved_mode)
        return sink_obs, sink_valid

    result = _propagator_lag_loop(
        source_obs=source_obs,
        source_valid=source_valid,
        compute_sink_fn=compute_sink,
        max_lag=max_lag,
        use_connected=use_connected,
    )

    return {"glueball_plaquette": result}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def compute_propagator_pipeline(
    data: PreparedChannelData,
    *,
    meson_config: MesonOperatorConfig | None = None,
    vector_config: VectorOperatorConfig | None = None,
    baryon_config: BaryonOperatorConfig | None = None,
    glueball_config: GlueballOperatorConfig | None = None,
    channels: list[str] | None = None,
    max_lag: int = 80,
    use_connected: bool = True,
) -> PropagatorResult:
    """Run propagator correlator pipeline for requested channels.

    Args:
        data: Pre-extracted channel tensors from prepare_channel_data().
        meson_config: Meson operator config (defaults used if None).
        vector_config: Vector operator config (defaults used if None).
        baryon_config: Baryon operator config (defaults used if None).
        glueball_config: Glueball operator config (defaults used if None).
        channels: List of channels to compute. None = all available.
        max_lag: Maximum temporal lag.
        use_connected: Whether to compute connected correlators.

    Returns:
        PropagatorResult with per-channel correlators.
    """
    if channels is None:
        channels = ["meson", "vector", "baryon", "glueball"]

    all_channels: dict[str, PropagatorChannelResult] = {}

    if "meson" in channels:
        cfg = meson_config or MesonOperatorConfig()
        result = compute_meson_propagator(data, cfg, max_lag, use_connected)
        all_channels.update(result)

    if "vector" in channels and data.positions is not None:
        cfg = vector_config or VectorOperatorConfig()
        result = compute_vector_propagator(data, cfg, max_lag, use_connected)
        all_channels.update(result)

    if "baryon" in channels:
        cfg = baryon_config or BaryonOperatorConfig()
        result = compute_baryon_propagator(data, cfg, max_lag, use_connected)
        all_channels.update(result)

    if "glueball" in channels:
        cfg = glueball_config or GlueballOperatorConfig()
        result = compute_glueball_propagator(data, cfg, max_lag, use_connected)
        all_channels.update(result)

    return PropagatorResult(channels=all_channels, prepared_data=data)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _mean_over_pairs(obs: Tensor, valid: Tensor) -> Tensor:
    """Average observable over the pair dimension P. [T,N,P] -> [T,N]."""
    if obs.ndim == 2:
        return obs
    weights = valid.to(obs.dtype)
    counts = weights.sum(dim=-1).clamp(min=1.0)
    return (obs * weights).sum(dim=-1) / counts


def _mean_over_pairs_vec(obs: Tensor, valid: Tensor) -> Tensor:
    """Average vector observable over pairs. [T,N,P,3] -> [T,N,3]."""
    if obs.ndim == 3:
        return obs
    weights = valid.to(obs.dtype).unsqueeze(-1)
    counts = valid.to(obs.dtype).sum(dim=-1).clamp(min=1.0).unsqueeze(-1)
    return (obs * weights).sum(dim=-2) / counts


def _clone_result(r: PropagatorChannelResult) -> PropagatorChannelResult:
    """Clone a PropagatorChannelResult."""
    return PropagatorChannelResult(
        raw=r.raw.clone(),
        connected=r.connected.clone(),
        counts=r.counts.clone(),
        mean_source=r.mean_source,
    )
