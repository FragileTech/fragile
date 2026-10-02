"""Numerical checks for the explicit viscous and finite Fock theorems.

These functions evaluate finite matrices and declared parameters. Floating-point
eigenvalues and a user-supplied zero tolerance are diagnostics, not rigorous
spectral enclosures or population-uniform physical mass-gap certificates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class FermionicSpectralDiagnostics:
    """Ground and parity-sector data for the finite directed-edge Hamiltonian."""

    one_particle_energies: Tensor
    sea_energy: float
    zero_modes: int
    ground_dimension: int
    excitation_gap: float | None
    even_ground_energy: float | None
    even_ground_dimension: int
    even_excitation_gap: float | None
    odd_ground_energy: float | None
    odd_ground_dimension: int
    odd_excitation_gap: float | None
    zero_tolerance: float


@dataclass(frozen=True)
class KineticMomentBudget:
    """Explicit mean Lp budgets at the stages of the count-normalized gas."""

    clone_position: float
    clone_velocity: float
    first_kick_velocity: float
    first_drift_position: float
    ou_velocity: float
    second_drift_position: float
    second_kick_velocity: float
    output_position: float
    output_velocity: float


def viscous_kinetic_moment_budget(
    *,
    dimension: int,
    moment_order: float,
    timestep: float,
    viscosity: float,
    force_lipschitz: float,
    force_at_origin: float,
    donor_radius: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
) -> KineticMomentBudget:
    """Evaluate ``lem-cg-mf-moments`` from every stage's declared parameters.

    Amplitudes are standard deviations of the draws at their respective stages,
    not diffusion coefficients. Positions are lengths and velocities length/time;
    the returned budgets keep those units. ``force_lipschitz`` has inverse-time
    squared units. Bandwidth and reward parameters do not enter this bound, as
    the Gaussian kernel is bounded by one and cloning has already selected donors
    inside the declared donor radius. They remain inputs to other theorems.
    """
    values = {
        "timestep": timestep,
        "viscosity": viscosity,
        "force_lipschitz": force_lipschitz,
        "force_at_origin": force_at_origin,
        "donor_radius": donor_radius,
        "clone_jitter": clone_jitter,
        "velocity_cap": velocity_cap,
        "ou_coefficient": ou_coefficient,
        "ou_amplitude": ou_amplitude,
        "position_amplitude": position_amplitude,
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        msg = "moment-budget parameters must be finite and nonnegative"
        raise ValueError(msg)
    if dimension < 1 or not isinstance(dimension, int):
        msg = "dimension must be a positive integer"
        raise ValueError(msg)
    if not math.isfinite(moment_order) or moment_order < 1:
        msg = "moment_order must be finite and at least one"
        raise ValueError(msg)
    if timestep == 0 or velocity_cap == 0 or ou_coefficient > 1:
        msg = "timestep and velocity_cap must be positive and ou_coefficient at most one"
        raise ValueError(msg)
    if not math.isfinite(restitution):
        msg = "restitution must be finite"
        raise ValueError(msg)
    gaussian_norm = math.exp(
        0.5 * math.log(2)
        + (math.lgamma((dimension + moment_order) / 2) - math.lgamma(dimension / 2)) / moment_order
    )
    half_step = timestep / 2
    x0 = donor_radius + clone_jitter * gaussian_norm
    v0 = (1 + 2 * abs(restitution)) * velocity_cap
    v1 = (1 + timestep * viscosity) * v0 + half_step * (force_lipschitz * x0 + force_at_origin)
    x1 = x0 + half_step * v1
    v2 = ou_coefficient * v1 + ou_amplitude * gaussian_norm
    x2 = x1 + half_step * v2
    v3 = (1 + timestep * viscosity) * v2 + half_step * (force_lipschitz * x2 + force_at_origin)
    return KineticMomentBudget(
        clone_position=x0,
        clone_velocity=v0,
        first_kick_velocity=v1,
        first_drift_position=x1,
        ou_velocity=v2,
        second_drift_position=x2,
        second_kick_velocity=v3,
        output_position=x2 + position_amplitude * gaussian_norm,
        output_velocity=min(velocity_cap, v3),
    )


def fermionic_spectral_diagnostics(
    directed_coefficients: Tensor, *, zero_tolerance: float = 1e-12
) -> FermionicSpectralDiagnostics:
    """Evaluate the exact finite-spectrum formulas using numerical eigenvalues.

    Args:
        directed_coefficients: Real directed coefficients ``K`` of shape [m, m].
            Their units are inverse algorithmic time, before energy calibration.
        zero_tolerance: Absolute energy threshold [1/time] for treating a
            computed eigenvalue or equal-energy difference as zero.

    Returns:
        Diagnostics for ``dGamma(i(K-K.T))``. Gaps are measured above the entire
        ground eigenspace. ``None`` means that the sector has no excited level
        (or, for a ground energy, that the parity sector is empty).
    """
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        msg = "zero_tolerance must be finite and nonnegative"
        raise ValueError(msg)
    if directed_coefficients.ndim != 2 or (
        directed_coefficients.shape[0] != directed_coefficients.shape[1]
    ):
        msg = "directed_coefficients must have shape [m, m]"
        raise ValueError(msg)
    if directed_coefficients.is_complex():
        msg = "directed_coefficients must be real"
        raise ValueError(msg)
    if not torch.isfinite(directed_coefficients).all():
        msg = "directed_coefficients must be finite"
        raise ValueError(msg)

    coefficients = directed_coefficients.to(dtype=torch.float64)
    hamiltonian = 1j * (coefficients - coefficients.T)
    energies = torch.linalg.eigvalsh(hamiltonian)
    energies = torch.where(energies.abs() <= zero_tolerance, 0.0, energies)
    negative = energies < 0
    zero_modes = int((energies == 0).sum().item())
    sea_energy = float(energies[negative].sum().item())
    costs = energies[energies != 0].abs().sort().values
    gap = float(costs[0].item()) if costs.numel() else None
    sea_parity = int(negative.sum().item()) % 2

    def parity_data(parity: int) -> tuple[float | None, int, float | None]:
        if zero_modes:
            return sea_energy, 2 ** (zero_modes - 1), gap
        if parity == sea_parity:
            parity_gap = float(costs[:2].sum().item()) if costs.numel() >= 2 else None
            return sea_energy, 1, parity_gap
        if not costs.numel():
            return None, 0, None
        minimum = float(costs[0].item())
        minimal_cost = (costs - minimum).abs() <= zero_tolerance
        multiplicity = int(minimal_cost.sum().item())
        candidates = costs[~minimal_cost]
        next_cost = float(candidates[0].item()) if candidates.numel() else math.inf
        if costs.numel() >= 3:
            next_cost = min(next_cost, float(costs[:3].sum().item()))
        parity_gap = next_cost - minimum if math.isfinite(next_cost) else None
        return sea_energy + minimum, multiplicity, parity_gap

    even_energy, even_dimension, even_gap = parity_data(0)
    odd_energy, odd_dimension, odd_gap = parity_data(1)
    return FermionicSpectralDiagnostics(
        one_particle_energies=energies,
        sea_energy=sea_energy,
        zero_modes=zero_modes,
        ground_dimension=2**zero_modes,
        excitation_gap=gap,
        even_ground_energy=even_energy,
        even_ground_dimension=even_dimension,
        even_excitation_gap=even_gap,
        odd_ground_energy=odd_energy,
        odd_ground_dimension=odd_dimension,
        odd_excitation_gap=odd_gap,
        zero_tolerance=zero_tolerance,
    )


def gaussian_viscous_force(
    positions: Tensor,
    velocities: Tensor,
    *,
    viscosity: float,
    bandwidth: float,
    row_normalized: bool = False,
) -> Tensor:
    """Evaluate the declared complete Gaussian viscous force on [N, d] arrays.

    Count normalization divides by N; row normalization excludes self from the
    denominator. This diagnostic does not modify the production integrator or
    substitute a complete Gaussian graph for a configured sparse geometry graph.
    Its stable row softmax evaluates the mathematical ratio even when raw weights
    would underflow; it does not reproduce the native finite-precision branch
    that skips a row whose accumulated raw mass is zero.
    Positions and bandwidth have length units, velocities length/time, and
    viscosity inverse time. The result is acceleration.
    """
    if not math.isfinite(viscosity) or viscosity < 0:
        msg = "viscosity must be finite and nonnegative"
        raise ValueError(msg)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        msg = "bandwidth must be finite and positive"
        raise ValueError(msg)
    if positions.ndim != 2 or positions.shape != velocities.shape:
        msg = "positions and velocities must have the same [N, d] shape"
        raise ValueError(msg)
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        msg = "the population and spatial dimension must be positive"
        raise ValueError(msg)
    if positions.device != velocities.device or positions.dtype != velocities.dtype:
        msg = "positions and velocities must have the same device and dtype"
        raise ValueError(msg)
    if not positions.is_floating_point() or not velocities.is_floating_point():
        msg = "positions and velocities must be floating-point arrays"
        raise ValueError(msg)
    if not torch.isfinite(positions).all() or not torch.isfinite(velocities).all():
        msg = "positions and velocities must be finite"
        raise ValueError(msg)
    n_walkers = positions.shape[0]
    if n_walkers == 1 or viscosity == 0:
        return torch.zeros_like(velocities)
    differences = positions[:, None, :] - positions[None, :, :]
    log_weights = -0.5 * (differences / bandwidth).square().sum(dim=-1)
    log_weights.fill_diagonal_(-torch.inf)
    if row_normalized:
        if not torch.isfinite(log_weights).any(dim=1).all():
            msg = "row-normalized distances exceed the floating-point range"
            raise ValueError(msg)
        # Softmax preserves the mathematical ratio when every unscaled Gaussian
        # weight underflows. It introduces no floor in the declared denominator.
        weights = log_weights.softmax(dim=1)
    else:
        weights = log_weights.exp() / n_walkers
    velocity_differences = velocities[None, :, :] - velocities[:, None, :]
    return viscosity * (weights[:, :, None] * velocity_differences).sum(dim=1)


def viscous_qsd_coercivity_margin(
    *, timestep: float, viscosity: float, curvature: float, row_normalized: bool = False
) -> float:
    """Evaluate the sufficient quadratic-force finite-N QSD inequality.

    ``curvature`` is lambda in F(x)=-lambda*x [1/time squared]. The margin is
    ``1-lambda*h**2/4-h*nu/2`` for count normalization and
    ``1-lambda*h**2/4-h*nu`` for row normalization. Positivity only verifies this
    particular inequality; the theorem also requires its declared positive
    noises, canonical cloning/cap, donor domain and recording conventions.
    """
    if not math.isfinite(timestep) or timestep <= 0:
        msg = "timestep must be finite and positive"
        raise ValueError(msg)
    if not math.isfinite(viscosity) or viscosity < 0:
        msg = "viscosity must be finite and nonnegative"
        raise ValueError(msg)
    if not math.isfinite(curvature) or curvature < 0:
        msg = "curvature must be finite and nonnegative"
        raise ValueError(msg)
    viscous_factor = 1.0 if row_normalized else 0.5
    return 1.0 - curvature * timestep**2 / 4.0 - viscous_factor * timestep * viscosity
