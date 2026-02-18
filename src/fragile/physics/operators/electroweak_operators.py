"""Electroweak (U(1) / SU(2)) operator construction from companion pairs.

Computes per-frame electroweak operator time series from fitness-based
phase factors and position-based amplitude weights.

Channel families:
- U(1) hypercharge: phase and dressed operators using distance companions.
- SU(2) isospin: phase, component, doublet operators using clone companions.
- EW mixed: product of U(1) and SU(2) amplitudes and phases.
- Symmetry-breaking scalars: fitness_phase, clone_indicator.
- Parity velocity: velocity norms split by walker type.

Complex channels output as [T, 2] (Re, Im); scalar channels as [T].
The correlator engine auto-contracts: C(tau) = C_Re(tau) + C_Im(tau).
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.qft_utils.helpers import safe_gather_2d, safe_gather_3d

from .config import ElectroweakOperatorConfig
from .preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Channel name constants
# ---------------------------------------------------------------------------

SU2_BASE_NAMES = ("su2_phase", "su2_component", "su2_doublet", "su2_doublet_diff")
WALKER_TYPE_LABELS = ("cloner", "resister", "persister")


# ---------------------------------------------------------------------------
# Per-frame averaging helpers
# ---------------------------------------------------------------------------


def _average_complex(z: Tensor, valid: Tensor) -> Tensor:
    """Average complex [T, N] over walkers using *valid* mask -> [T, 2]."""
    re_im = torch.stack([z.real.float(), z.imag.float()], dim=-1)  # [T, N, 2]
    mask = valid.unsqueeze(-1).float()  # [T, N, 1]
    counts = valid.sum(dim=1).clamp(min=1).unsqueeze(-1).float()  # [T, 1]
    return (re_im * mask).sum(dim=1) / counts  # [T, 2]


def _average_scalar(x: Tensor, valid: Tensor) -> Tensor:
    """Average real [T, N] over walkers using *valid* mask -> [T]."""
    mask = valid.float()  # [T, N]
    counts = valid.sum(dim=1).clamp(min=1).float()  # [T]
    return (x.float() * mask).sum(dim=1) / counts  # [T]


# ---------------------------------------------------------------------------
# Walker-type classification (batched)
# ---------------------------------------------------------------------------


def _classify_walker_types_batched(
    fitness: Tensor,
    alive: Tensor,
    will_clone: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Classify [T, N] walkers into (cloner, resister, persister) masks."""
    cloner = alive & will_clone

    neg_inf = torch.full_like(fitness, float("-inf"))
    alive_fitness = torch.where(alive, fitness, neg_inf)
    max_fitness = alive_fitness.max(dim=1, keepdim=True).values  # [T, 1]
    has_fitter_peer = alive & torch.isfinite(max_fitness) & (fitness < max_fitness)
    resister = alive & (~will_clone) & has_fitter_peer

    persister = alive & (~cloner) & (~resister)
    return cloner, resister, persister


# ---------------------------------------------------------------------------
# U(1) operators
# ---------------------------------------------------------------------------


def _compute_u1_operators(
    fitness: Tensor,
    companions_distance: Tensor,
    positions_full: Tensor | None,
    velocities: Tensor | None,
    alive: Tensor,
    h_eff: float,
    epsilon_d: float,
    lambda_alg: float,
) -> dict[str, Tensor]:
    """Compute U(1) hypercharge operators -> dict of [T, 2] tensors."""
    cdtype = torch.complex64

    # Gather companion fitness via distance companion
    fitness_j, in_range = safe_gather_2d(fitness, companions_distance)
    valid = alive & in_range

    # Phase: theta = -(S_j - S_i) / h_eff
    theta = -(fitness_j - fitness) / max(h_eff, 1e-12)
    phase = torch.exp(1j * theta.to(cdtype))
    phase_q2 = torch.exp(1j * (2.0 * theta).to(cdtype))

    # Amplitude from positions
    if positions_full is not None:
        pos_j, pos_in_range = safe_gather_3d(positions_full, companions_distance)
        valid = valid & pos_in_range
        diff_x = pos_j - positions_full
        dist_sq = (diff_x**2).sum(dim=-1)
        if velocities is not None and lambda_alg > 0:
            vel_j, _ = safe_gather_3d(velocities, companions_distance)
            diff_v = vel_j - velocities
            dist_sq = dist_sq + lambda_alg * (diff_v**2).sum(dim=-1)
        eps_d = max(epsilon_d, 1e-12)
        amp = torch.exp(-dist_sq / (4.0 * eps_d * eps_d))
    else:
        amp = torch.ones_like(fitness)

    amp_c = amp.to(cdtype)
    valid_c = valid.to(cdtype)

    return {
        "u1_phase": _average_complex(phase * valid_c, valid),
        "u1_dressed": _average_complex(amp_c * phase * valid_c, valid),
        "u1_phase_q2": _average_complex(phase_q2 * valid_c, valid),
        "u1_dressed_q2": _average_complex(amp_c * phase_q2 * valid_c, valid),
    }


# ---------------------------------------------------------------------------
# SU(2) operators
# ---------------------------------------------------------------------------


def _compute_su2_operators(
    fitness: Tensor,
    companions_clone: Tensor,
    positions_full: Tensor | None,
    velocities: Tensor | None,
    alive: Tensor,
    h_eff: float,
    epsilon_clone: float,
    lambda_alg: float,
    su2_mode: str,
    enable_directed: bool,
    enable_walker_type_split: bool,
    will_clone: Tensor | None,
) -> dict[str, Tensor]:
    """Compute SU(2) isospin operators -> dict of [T, 2] tensors."""
    cdtype = torch.complex64

    # Gather companion fitness via clone companion
    fitness_k, in_range = safe_gather_2d(fitness, companions_clone)
    valid = alive & in_range

    # Phase: theta = ((S_k - S_i) / (|S_i| + eps_clone)) / h_eff
    denom = fitness.abs() + max(epsilon_clone, 1e-12)
    theta = ((fitness_k - fitness) / denom) / max(h_eff, 1e-12)
    phase = torch.exp(1j * theta.to(cdtype))

    # Directed phase: conjugate when theta < 0
    phase_directed = torch.where(theta >= 0, phase, torch.conj(phase))

    # Amplitude from positions
    if positions_full is not None:
        pos_k, pos_in_range = safe_gather_3d(positions_full, companions_clone)
        valid = valid & pos_in_range
        diff_x = pos_k - positions_full
        dist_sq = (diff_x**2).sum(dim=-1)
        if velocities is not None and lambda_alg > 0:
            vel_k, _ = safe_gather_3d(velocities, companions_clone)
            diff_v = vel_k - velocities
            dist_sq = dist_sq + lambda_alg * (diff_v**2).sum(dim=-1)
        eps_c = max(epsilon_clone, 1e-12)
        amp = torch.exp(-dist_sq / (4.0 * eps_c * eps_c))
    else:
        amp = torch.ones_like(fitness)

    # Two-hop: gather su2 phase and amplitude at clone companion's clone companion
    comp_phase, comp_in_range = safe_gather_2d(phase.real, companions_clone)
    comp_phase_im, _ = safe_gather_2d(phase.imag, companions_clone)
    comp_phase_complex = torch.complex(comp_phase, comp_phase_im)
    comp_phase_directed_re, _ = safe_gather_2d(phase_directed.real, companions_clone)
    comp_phase_directed_im, _ = safe_gather_2d(phase_directed.imag, companions_clone)
    comp_phase_directed = torch.complex(comp_phase_directed_re, comp_phase_directed_im)
    comp_amp, _ = safe_gather_2d(amp, companions_clone)
    valid = valid & comp_in_range

    amp_c = amp.to(cdtype)
    comp_amp_c = comp_amp.to(cdtype)
    valid_c = valid.to(cdtype)

    # Standard operators
    su2_phase_std = phase * valid_c
    su2_component_std = (amp_c * phase) * valid_c
    su2_doublet_std = (amp_c * phase + comp_amp_c * comp_phase_complex) * valid_c
    su2_doublet_diff_std = (amp_c * phase - comp_amp_c * comp_phase_complex) * valid_c

    # Directed operators
    su2_phase_dir = phase_directed * valid_c
    su2_component_dir = (amp_c * phase_directed) * valid_c
    su2_doublet_dir = (amp_c * phase_directed + comp_amp_c * comp_phase_directed) * valid_c
    su2_doublet_diff_dir = (amp_c * phase_directed - comp_amp_c * comp_phase_directed) * valid_c

    # Select primary operators based on mode
    if su2_mode == "score_directed":
        ops_primary = {
            "su2_phase": su2_phase_dir,
            "su2_component": su2_component_dir,
            "su2_doublet": su2_doublet_dir,
            "su2_doublet_diff": su2_doublet_diff_dir,
        }
    else:
        ops_primary = {
            "su2_phase": su2_phase_std,
            "su2_component": su2_component_std,
            "su2_doublet": su2_doublet_std,
            "su2_doublet_diff": su2_doublet_diff_std,
        }

    # Average primary channels
    results: dict[str, Tensor] = {}
    for name, z in ops_primary.items():
        results[name] = _average_complex(z, valid)

    # Directed variants (always from the directed operators)
    if enable_directed:
        dir_ops = {
            "su2_phase_directed": su2_phase_dir,
            "su2_component_directed": su2_component_dir,
            "su2_doublet_directed": su2_doublet_dir,
            "su2_doublet_diff_directed": su2_doublet_diff_dir,
        }
        for name, z in dir_ops.items():
            results[name] = _average_complex(z, valid)

    # Walker-type split channels
    if enable_walker_type_split and will_clone is not None:
        cloner, resister, persister = _classify_walker_types_batched(fitness, alive, will_clone)
        for label, type_mask in [
            ("cloner", cloner),
            ("resister", resister),
            ("persister", persister),
        ]:
            mask_c = type_mask.to(cdtype)
            for base_name, z in ops_primary.items():
                results[f"{base_name}_{label}"] = _average_complex(z * mask_c, valid & type_mask)
    else:
        # Emit zeroed walker-type channels
        T = fitness.shape[0]
        zero2 = torch.zeros(T, 2, dtype=torch.float32, device=fitness.device)
        for label in WALKER_TYPE_LABELS:
            for base_name in SU2_BASE_NAMES:
                results[f"{base_name}_{label}"] = zero2.clone()

    return results


# ---------------------------------------------------------------------------
# EW mixed operator
# ---------------------------------------------------------------------------


def _compute_ew_mixed(
    fitness: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    positions_full: Tensor | None,
    velocities: Tensor | None,
    alive: Tensor,
    h_eff: float,
    epsilon_d: float,
    epsilon_clone: float,
    lambda_alg: float,
    su2_mode: str,
) -> dict[str, Tensor]:
    """Compute electroweak mixed product operator -> [T, 2]."""
    cdtype = torch.complex64

    # U(1) terms from distance companion
    fitness_j, in_range_j = safe_gather_2d(fitness, companions_distance)
    theta_u1 = -(fitness_j - fitness) / max(h_eff, 1e-12)
    u1_phase = torch.exp(1j * theta_u1.to(cdtype))

    # SU(2) terms from clone companion
    fitness_k, in_range_k = safe_gather_2d(fitness, companions_clone)
    denom = fitness.abs() + max(epsilon_clone, 1e-12)
    theta_su2 = ((fitness_k - fitness) / denom) / max(h_eff, 1e-12)
    su2_phase = torch.exp(1j * theta_su2.to(cdtype))
    if su2_mode == "score_directed":
        su2_phase = torch.where(theta_su2 >= 0, su2_phase, torch.conj(su2_phase))

    valid = alive & in_range_j & in_range_k

    # Amplitudes from positions
    if positions_full is not None:
        pos_j, pj_ir = safe_gather_3d(positions_full, companions_distance)
        pos_k, pk_ir = safe_gather_3d(positions_full, companions_clone)
        valid = valid & pj_ir & pk_ir

        diff_j = pos_j - positions_full
        dist_sq_j = (diff_j**2).sum(dim=-1)
        diff_k = pos_k - positions_full
        dist_sq_k = (diff_k**2).sum(dim=-1)

        if velocities is not None and lambda_alg > 0:
            vel_j, _ = safe_gather_3d(velocities, companions_distance)
            vel_k, _ = safe_gather_3d(velocities, companions_clone)
            dv_j = vel_j - velocities
            dv_k = vel_k - velocities
            dist_sq_j = dist_sq_j + lambda_alg * (dv_j**2).sum(dim=-1)
            dist_sq_k = dist_sq_k + lambda_alg * (dv_k**2).sum(dim=-1)

        eps_d = max(epsilon_d, 1e-12)
        eps_c = max(epsilon_clone, 1e-12)
        u1_amp = torch.exp(-dist_sq_j / (4.0 * eps_d * eps_d))
        su2_amp = torch.exp(-dist_sq_k / (4.0 * eps_c * eps_c))
    else:
        u1_amp = torch.ones_like(fitness)
        su2_amp = torch.ones_like(fitness)

    mixed = u1_amp.to(cdtype) * su2_amp.to(cdtype) * u1_phase * su2_phase * valid.to(cdtype)

    return {"ew_mixed": _average_complex(mixed, valid)}


# ---------------------------------------------------------------------------
# Symmetry-breaking scalar operators
# ---------------------------------------------------------------------------


def _compute_symmetry_breaking(
    fitness: Tensor,
    alive: Tensor,
    will_clone: Tensor | None,
) -> dict[str, Tensor]:
    """Compute fitness_phase and clone_indicator -> [T] scalars."""
    # fitness_phase = -fitness * alive, averaged
    fp = -fitness * alive.float()
    results: dict[str, Tensor] = {"fitness_phase": _average_scalar(fp, alive)}

    # clone_indicator = will_clone * alive, averaged
    if will_clone is not None:
        ci = will_clone.float() * alive.float()
        results["clone_indicator"] = _average_scalar(ci, alive)
    else:
        results["clone_indicator"] = torch.zeros(
            fitness.shape[0], dtype=torch.float32, device=fitness.device
        )

    return results


# ---------------------------------------------------------------------------
# Parity velocity operators
# ---------------------------------------------------------------------------


def _compute_parity_velocity(
    velocities: Tensor | None,
    alive: Tensor,
    fitness: Tensor,
    will_clone: Tensor | None,
) -> dict[str, Tensor]:
    """Compute velocity-norm by walker type -> [T] scalars."""
    T = fitness.shape[0]
    device = fitness.device
    zero = torch.zeros(T, dtype=torch.float32, device=device)

    if velocities is None or will_clone is None:
        return {
            "velocity_norm_cloner": zero.clone(),
            "velocity_norm_resister": zero.clone(),
            "velocity_norm_persister": zero.clone(),
        }

    v_norm = torch.linalg.vector_norm(velocities, dim=-1)  # [T, N]
    cloner, resister, persister = _classify_walker_types_batched(fitness, alive, will_clone)

    results: dict[str, Tensor] = {}
    for label, mask in [("cloner", cloner), ("resister", resister), ("persister", persister)]:
        if mask.any():
            results[f"velocity_norm_{label}"] = _average_scalar(v_norm * mask.float(), mask)
        else:
            results[f"velocity_norm_{label}"] = zero.clone()
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_electroweak_operators(
    data: PreparedChannelData,
    config: ElectroweakOperatorConfig,
) -> dict[str, Tensor]:
    """Compute all electroweak operator time series.

    Args:
        data: Pre-extracted channel tensors from :func:`prepare_channel_data`.
        config: Electroweak operator configuration.

    Returns:
        Dictionary mapping channel names to operator time series.
        Complex channels have shape ``[T, 2]`` (Re, Im); scalar
        channels have shape ``[T]``.
    """
    device = data.device
    T = int(data.companions_distance.shape[0]) if data.companions_distance.numel() > 0 else 0

    if T == 0:
        empty2 = torch.zeros(0, 2, dtype=torch.float32, device=device)
        empty1 = torch.zeros(0, dtype=torch.float32, device=device)
        results: dict[str, Tensor] = {}
        # Complex channels
        for name in (
            "u1_phase",
            "u1_dressed",
            "u1_phase_q2",
            "u1_dressed_q2",
            "su2_phase",
            "su2_component",
            "su2_doublet",
            "su2_doublet_diff",
            "su2_phase_directed",
            "su2_component_directed",
            "su2_doublet_directed",
            "su2_doublet_diff_directed",
            "ew_mixed",
        ):
            results[name] = empty2.clone()
        for label in WALKER_TYPE_LABELS:
            for base in SU2_BASE_NAMES:
                results[f"{base}_{label}"] = empty2.clone()
        # Scalar channels
        for name in (
            "fitness_phase",
            "clone_indicator",
            "velocity_norm_cloner",
            "velocity_norm_resister",
            "velocity_norm_persister",
        ):
            results[name] = empty1.clone()
        return results

    # Validate required fields
    if data.fitness is None:
        msg = "fitness is required for electroweak operators."
        raise ValueError(msg)
    if data.alive is None:
        msg = "alive is required for electroweak operators."
        raise ValueError(msg)

    fitness = data.fitness
    alive = data.alive
    h_eff = float(max(config.h_eff, 1e-8))
    epsilon_d = float(config.epsilon_d) if config.epsilon_d is not None else 1.0
    epsilon_clone = float(config.epsilon_clone) if config.epsilon_clone is not None else 1e-8
    lambda_alg = float(config.lambda_alg)
    su2_mode = str(config.su2_operator_mode).strip().lower()
    if su2_mode not in {"standard", "score_directed"}:
        msg = "su2_operator_mode must be 'standard' or 'score_directed'."
        raise ValueError(msg)

    all_ops: dict[str, Tensor] = {}

    # U(1) operators
    all_ops.update(
        _compute_u1_operators(
            fitness=fitness,
            companions_distance=data.companions_distance,
            positions_full=data.positions_full,
            velocities=data.velocities,
            alive=alive,
            h_eff=h_eff,
            epsilon_d=epsilon_d,
            lambda_alg=lambda_alg,
        )
    )

    # SU(2) operators
    all_ops.update(
        _compute_su2_operators(
            fitness=fitness,
            companions_clone=data.companions_clone,
            positions_full=data.positions_full,
            velocities=data.velocities,
            alive=alive,
            h_eff=h_eff,
            epsilon_clone=epsilon_clone,
            lambda_alg=lambda_alg,
            su2_mode=su2_mode,
            enable_directed=bool(config.enable_directed_variants),
            enable_walker_type_split=bool(config.enable_walker_type_split),
            will_clone=data.will_clone,
        )
    )

    # EW mixed
    all_ops.update(
        _compute_ew_mixed(
            fitness=fitness,
            companions_distance=data.companions_distance,
            companions_clone=data.companions_clone,
            positions_full=data.positions_full,
            velocities=data.velocities,
            alive=alive,
            h_eff=h_eff,
            epsilon_d=epsilon_d,
            epsilon_clone=epsilon_clone,
            lambda_alg=lambda_alg,
            su2_mode=su2_mode,
        )
    )

    # Symmetry-breaking scalars
    all_ops.update(
        _compute_symmetry_breaking(
            fitness=fitness,
            alive=alive,
            will_clone=data.will_clone,
        )
    )

    # Parity velocity
    if config.enable_parity_velocity:
        all_ops.update(
            _compute_parity_velocity(
                velocities=data.velocities,
                alive=alive,
                fitness=fitness,
                will_clone=data.will_clone,
            )
        )
    else:
        zero = torch.zeros(T, dtype=torch.float32, device=device)
        for label in WALKER_TYPE_LABELS:
            all_ops[f"velocity_norm_{label}"] = zero.clone()

    return all_ops
