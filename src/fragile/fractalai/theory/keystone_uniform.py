"""Population-normalized ledgers for the canonical terminal-box gas.

These diagnostics evaluate the signed identities in Chapter 18a. They do not
change a production step or infer full-law contraction from cloning pressure.
The implementation tag, force, companion laws and innovations must still be
checked against the complete execution record. Floating results are not
rigorous interval enclosures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp, ndtri_exp
import torch
from torch import Tensor

from fragile.fractalai.theory.coupled_gas_diagnostics import gaussian_viscous_force


@dataclass(frozen=True)
class FitnessParameters:
    """Native global regularization and logistic-product fitness parameters."""

    reward_floor: float
    diversity_floor: float
    reward_amplitude: float
    diversity_amplitude: float
    reward_offset: float
    diversity_offset: float
    reward_power: float
    diversity_power: float
    clone_scale: float
    clone_regularizer: float

    def __post_init__(self) -> None:
        positive = (
            self.reward_floor,
            self.diversity_floor,
            self.reward_amplitude,
            self.diversity_amplitude,
            self.reward_offset,
            self.diversity_offset,
            self.clone_scale,
            self.clone_regularizer,
        )
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            msg = "regularizers, amplitudes, offsets and clone scale must be positive"
            raise ValueError(msg)
        if any(not math.isfinite(x) or x < 0 for x in (self.reward_power, self.diversity_power)):
            msg = "fitness powers must be finite and nonnegative"
            raise ValueError(msg)

    @property
    def fitness_min(self) -> float:
        return self.reward_offset**self.reward_power * self.diversity_offset**self.diversity_power

    @property
    def fitness_max(self) -> float:
        return (self.reward_amplitude + self.reward_offset) ** self.reward_power * (
            self.diversity_amplitude + self.diversity_offset
        ) ** self.diversity_power

    @property
    def unit_power_mean_bound(self) -> float | None:
        """Uniform actual alive-mean bound, including the shared normalizers.

        This uses the certified scalar tanh remainder .07. For the native
        amplitude 2, offset .1, unit-power record it equals 1.614. Other powers
        require their own moment estimate.
        """
        if self.reward_power != 1 or self.diversity_power != 1:
            return None

        def second(amplitude: float, offset: float) -> float:
            middle = amplitude / 2 + offset
            return middle**2 + 0.07 * amplitude * middle + amplitude**2 / 16

        return math.sqrt(
            second(self.reward_amplitude, self.reward_offset)
            * second(self.diversity_amplitude, self.diversity_offset)
        )

    @property
    def gate_slopes(self) -> tuple[float, float]:
        """Recipient and donor Lipschitz constants, including the clipped branch."""
        denominator = self.clone_scale * (self.fitness_min + self.clone_regularizer)
        return (1 + self.clone_scale) / denominator, 1 / denominator

    @property
    def fitness_slopes(self) -> tuple[float, float]:
        """L2 slopes with respect to actual reward and diversity measurements."""

        def factor(amplitude: float, offset: float, power: float) -> float:
            if power == 0:
                return 0.0
            return (
                amplitude
                * power
                / 4
                * max(offset ** (power - 1), (amplitude + offset) ** (power - 1))
            )

        reward = factor(self.reward_amplitude, self.reward_offset, self.reward_power)
        reward *= (self.diversity_amplitude + self.diversity_offset) ** self.diversity_power
        diversity = factor(self.diversity_amplitude, self.diversity_offset, self.diversity_power)
        diversity *= (self.reward_amplitude + self.reward_offset) ** self.reward_power
        return reward / self.reward_floor, diversity / self.diversity_floor


def _arrays(*arrays: Tensor) -> None:
    if any(not array.is_floating_point() or not torch.isfinite(array).all() for array in arrays):
        msg = "arrays must be finite real floating-point tensors"
        raise ValueError(msg)
    if any(array.device != arrays[0].device or array.dtype != arrays[0].dtype for array in arrays):
        msg = "arrays must share a device and dtype"
        raise ValueError(msg)


def global_standardize(values: Tensor, alive: Tensor, *, floor: float) -> Tensor:
    """Apply the actual alive-only population variance (not sample variance)."""
    _arrays(values)
    if values.ndim != 1 or alive.shape != values.shape or alive.dtype != torch.bool:
        msg = "values and bool alive must have shape [N]"
        raise ValueError(msg)
    if not alive.any() or not math.isfinite(floor) or floor <= 0:
        msg = "a nonextinct population and a positive finite floor are required"
        raise ValueError(msg)
    selected = values[alive]
    centered = selected - selected.mean()
    standardized = torch.zeros_like(values)
    standardized[alive] = centered / torch.sqrt(centered.square().mean() + floor**2)
    return standardized


def complete_fitness(
    reward: Tensor, measured_diversity: Tensor, alive: Tensor, *, parameters: FitnessParameters
) -> Tensor:
    """Compute fitness from the entire actual sampled diversity vector.

    ``reward`` must already have the configured objective orientation, and
    ``measured_diversity`` must use the actual measured companion and distance
    map. Neither is replaced by its mean. Dead entries are zeroed afterwards.
    """
    _arrays(reward, measured_diversity)
    if reward.shape != measured_diversity.shape:
        msg = "reward and measured diversity must have the same shape"
        raise ValueError(msg)
    zr = global_standardize(reward, alive, floor=parameters.reward_floor)
    zs = global_standardize(measured_diversity, alive, floor=parameters.diversity_floor)
    r = parameters.reward_amplitude * torch.sigmoid(zr) + parameters.reward_offset
    s = parameters.diversity_amplitude * torch.sigmoid(zs) + parameters.diversity_offset
    return torch.where(alive, r**parameters.reward_power * s**parameters.diversity_power, 0)


def native_source_pressure_lower_bound(
    fitness: Tensor,
    kernel_weights: Tensor,
    alive: Tensor,
    *,
    clone_regularizer: float,
) -> Tensor:
    """Signed single-position-ancestry excess, with mandatory revival.

    ``kernel_weights`` contains the original symmetric Gaussian clone
    weights in [0,1]. Alive queries exclude self, and dead queries retain
    all alive donors. Fitness is held at its complete sampled value.
    This estimate uses the native unit clone scale.
    The result bounds expected ancestry count minus its entering count
    for each alive source. It does not treat velocities as donor copied.
    """
    _arrays(fitness, kernel_weights)
    n = fitness.numel()
    if (
        fitness.ndim != 1
        or kernel_weights.shape != (n, n)
        or alive.shape != (n,)
        or alive.dtype != torch.bool
        or alive.device != fitness.device
    ):
        msg = "fitness, clone weights and bool alive must have compatible shapes and devices"
        raise ValueError(msg)
    if not alive.any() or (fitness < 0).any():
        msg = "a nonextinct population and nonnegative fitness are required"
        raise ValueError(msg)
    if (
        (kernel_weights < 0).any()
        or (kernel_weights > 1).any()
        or not torch.equal(kernel_weights, kernel_weights.T)
    ):
        msg = "original clone weights must be symmetric and lie in [0,1]"
        raise ValueError(msg)
    if not math.isfinite(clone_regularizer) or clone_regularizer <= 0:
        msg = "clone gate regularizer must be positive and finite"
        raise ValueError(msg)
    eligible = alive.expand(n, n).clone()
    eligible[alive] &= ~torch.eye(n, dtype=torch.bool, device=alive.device)[alive]
    if int(alive.sum()) == 1:
        eligible[alive] = alive
    weights = kernel_weights * eligible
    if int(alive.sum()) == 1:
        # The actual singleton alive row persists independently of its
        # raw self weight. Dead-query revival still uses the original law.
        weights[alive] = alive.to(fitness.dtype)
    denominator = weights.sum(dim=1)
    if (denominator <= 0).any():
        msg = "every original query must have positive eligible donor mass"
        raise ValueError(msg)
    companion = weights / denominator[:, None]
    revival = companion[~alive].sum(dim=0)
    lower = torch.zeros_like(fitness)
    m = int(alive.sum())
    if m >= 2:
        degree = denominator[alive] / (m - 1)
        excluded_mean = (fitness[alive].sum() - fitness[alive]) / (m - 1)
        signed = (fitness[alive] * degree - excluded_mean / degree) / (
            fitness[alive] + clone_regularizer
        )
        lower[alive] = torch.maximum(signed, torch.full_like(signed, -1))
    return torch.where(alive, lower + revival, 0)


def clone_token_law(
    fitness: Tensor, companion: Tensor, alive: Tensor, *, parameters: FitnessParameters
) -> Tensor:
    """Return the complete conditional source law [N, 2, N].

    Token (0,i) means persistence without jitter, token (1,j) accepted cloning
    from frozen donor j with jitter. Dead rows revive mandatorily. A singleton
    alive row persists. All companion probabilities, including self exclusion,
    are supplied by the actual configured law and validated, never inferred
    from fitness or replaced by uniform probabilities. Probability rows are
    normalized after their floating-point stochasticity check; this diagnostic
    evaluates the mathematical law rather than a native arithmetic branch.
    """
    _arrays(fitness, companion)
    n = fitness.numel()
    if fitness.ndim != 1 or companion.shape != (n, n) or alive.shape != (n,):
        msg = "fitness/alive must be [N] and companion [N,N]"
        raise ValueError(msg)
    if alive.dtype != torch.bool or not alive.any():
        msg = "alive must be a nonextinct bool mask"
        raise ValueError(msg)
    if (companion < 0).any() or not torch.allclose(companion.sum(1), torch.ones_like(fitness)):
        msg = "companion must be row stochastic"
        raise ValueError(msg)
    if (companion[:, ~alive] != 0).any():
        msg = "a companion law must be supported on alive donors"
        raise ValueError(msg)
    indices = torch.arange(n, device=fitness.device)
    if int(alive.sum()) > 1:
        if (companion.diagonal()[alive] != 0).any():
            msg = "live self donors must be excluded outside the singleton case"
            raise ValueError(msg)
    else:
        only = int(torch.nonzero(alive, as_tuple=False)[0, 0])
        if companion[only, only] != 1:
            msg = "the singleton companion must be itself"
            raise ValueError(msg)
    if (fitness[alive] < parameters.fitness_min - 1e-12).any() or (
        fitness[alive] > parameters.fitness_max + 1e-12
    ).any():
        msg = "alive fitness lies outside the configured logistic-product range"
        raise ValueError(msg)
    companion = companion / companion.sum(1, keepdim=True)
    denominator = parameters.clone_scale * (fitness[:, None] + parameters.clone_regularizer)
    gate = ((fitness[None, :] - fitness[:, None]).clamp_min(0) / denominator).clamp_max(1)
    accepted = torch.where(alive[:, None], companion * gate, companion)
    token = torch.zeros((n, 2, n), dtype=fitness.dtype, device=fitness.device)
    token[:, 1] = accepted
    token[indices, 0, indices] = torch.where(alive, (1 - accepted.sum(1)).clamp_min(0), 0)
    return token


def maximal_token_coupling(first: Tensor, second: Tensor) -> Tensor:
    """Couple every source and acceptance token, keeping residual donor mass."""
    _arrays(first, second)
    if (
        first.shape != second.shape
        or first.ndim != 3
        or first.shape[1] != 2
        or first.shape[2] != first.shape[0]
    ):
        msg = "source laws must have equal shape [N,2,N]"
        raise ValueError(msg)
    n = first.shape[0]
    p, q = first.reshape(n, -1), second.reshape(n, -1)
    if (
        (p < 0).any()
        or (q < 0).any()
        or not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]))
        or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]))
    ):
        msg = "source laws must be probability distributions"
        raise ValueError(msg)
    common = torch.minimum(p, q)
    coupling = torch.diag_embed(common)
    rp, rq = (p - common).clamp_min(0), (q - common).clamp_min(0)
    mass = rq.sum(1)
    safe_mass = torch.where(mass > 0, mass, 1)
    residual = rp[:, :, None] * (rq / safe_mass[:, None])[:, None, :]
    coupling += torch.where(
        mass[:, None, None] > 0,
        residual,
        0,
    )
    return coupling.reshape(n, 2, n, 2, n)


@dataclass(frozen=True)
class CloningVarianceBalance:
    """Signed conditional variance identity before component collisions."""

    input_variance: Tensor
    recipient_loss: Tensor
    donor_gain: Tensor
    jitter_gain: Tensor
    mean_shift_loss: Tensor
    barycenter_variance_loss: Tensor
    output_variance: Tensor


def cloning_variance_balance(
    positions: Tensor, tokens: Tensor, *, clone_jitter: float
) -> CloningVarianceBalance:
    """Compute exact conditional E[W] including revival and the N^-2 term.

    Conditioning is on the complete actual fitness measurement and companion
    laws, before independent source draws and jitters. Raw dead positions are
    retained in the input variance; the token law replaces them by alive donors.
    Component collisions do not copy donor velocities and do not move positions.
    """
    _arrays(positions, tokens)
    n, d = positions.shape if positions.ndim == 2 else (0, 0)
    if n == 0 or tokens.shape != (n, 2, n):
        msg = "positions must be [N,d] and tokens [N,2,N]"
        raise ValueError(msg)
    if not math.isfinite(clone_jitter) or clone_jitter < 0:
        msg = "clone_jitter must be finite and nonnegative"
        raise ValueError(msg)
    if (tokens < 0).any() or not torch.allclose(
        tokens.sum((1, 2)), torch.ones_like(tokens[:, 0, 0])
    ):
        msg = "tokens must be probability distributions"
        raise ValueError(msg)
    diagonal = torch.diag_embed(tokens[:, 0].diagonal())
    if (tokens[:, 0] != diagonal).any():
        msg = "a persistence token must retain its own row"
        raise ValueError(msg)
    centered = positions - positions.mean(0)
    radii = centered.square().sum(1)
    accepted = tokens[:, 1]
    probability = accepted.sum(1)
    source = tokens.sum(1)
    means = source @ positions
    source_deviation = positions[None, :, :] - means[:, None, :]
    variances = (source * source_deviation.square().sum(2)).sum(1)
    variances += d * clone_jitter**2 * probability
    initial = radii.mean()
    loss = (probability * radii).mean()
    donor = (accepted @ radii).mean()
    jitter = d * clone_jitter**2 * probability.mean()
    shift = (means.mean(0) - positions.mean(0)).square().sum()
    center_variance = variances.sum() / n**2
    # Avoid canceling unbounded raw dead-coordinate squares. The signed terms
    # remain available, but evaluate the resulting expectation in its centered
    # prepared representation.
    output = (means - means.mean(0)).square().sum(1).mean() + (1 - 1 / n) * variances.mean()
    return CloningVarianceBalance(
        initial,
        loss,
        donor,
        jitter,
        shift,
        center_variance,
        output,
    )


def _log_interval(mean: float, width: float, sd: float) -> float:
    upper = float(log_ndtr((width - mean) / sd))
    lower = float(log_ndtr((-width - mean) / sd))
    if not math.isfinite(upper) or lower >= upper:
        # A positive density integral remains a valid lower bound even when two
        # floating log-CDFs coincide. Do not silently turn a positive event off.
        return math.log(2 * width / (sd * math.sqrt(2 * math.pi))) - (abs(mean) + width) ** 2 / (
            2 * sd**2
        )
    return upper + math.log(-math.expm1(lower - upper))


def gaussian_row_column_bound(dimension: int, *, series_terms: int = 0) -> float:
    """KUK.7/KURC.20: finite shell sum plus its decreasing integral tail.

    The default reproduces the original coarse envelope. Increasing the
    finite sum sharpens the same geometric proof, without a degree floor.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        msg = "dimension must be a positive integer"
        raise ValueError(msg)
    if not isinstance(series_terms, int) or isinstance(series_terms, bool) or series_terms < 0:
        msg = "series_terms must be a nonnegative integer"
        raise ValueError(msg)
    annular_log = math.log(5 / 4)
    shell = math.fsum(
        2 * math.exp(-(35 / 128) * (5 / 4) ** (2 * m)) + math.exp(-(3 / 8) * (5 / 4) ** m)
        for m in range(series_terms)
    )
    radial = (5 / 4) ** series_terms
    dense = 2 * math.exp(-(35 / 128) * radial**2) * (1 + 1 / ((35 / 64) * annular_log * radial**2))
    single = math.exp(-(3 / 8) * radial) * (1 + 1 / ((3 / 8) * annular_log * radial))
    return 2 * math.exp(2) + 9**dimension * (1 + shell + dense + single)


@dataclass(frozen=True)
class RastriginBoxSurvivalBudget:
    """Floating evaluation of KURC.10, using a finite lower integral sum."""

    collision_velocity_bound: float
    velocity_energy_bound: float
    row_column_bound: int
    landing_slope_lower: float
    log_accepted_floor: float
    log_alive_floor: float
    persisting_fraction: float
    integration_radius: float
    integration_bins: int

    def log_count_transform_bound(self, n_walkers: int, theta: float) -> float:
        """KURC.11; a Laplace bound, not stochastic binomial domination."""
        if not isinstance(n_walkers, int) or isinstance(n_walkers, bool) or n_walkers < 1:
            msg = "n_walkers must be a positive integer"
            raise ValueError(msg)
        if not math.isfinite(theta) or theta < 0:
            msg = "theta must be finite and nonnegative"
            raise ValueError(msg)
        return n_walkers * math.log1p(math.expm1(-theta) * math.exp(self.log_alive_floor))


def rastrigin_box_survival_budget(
    *,
    dimension: int,
    timestep: float,
    viscosity: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool,
    integration_radius: float = 8,
    integration_bins: int = 400,
    column_series_terms: int = 40,
) -> RastriginBoxSurvivalBudget:
    """Evaluate all gate fractions for the unchanged native Rastrigin force.

    The finite integral sum bounds the original unbounded jitter integral;
    it does not clip the algorithm. Results require interval certification
    before use as rigorous rounded numerical constants.
    """
    _validate_moment_parameters(
        dimension,
        2,
        timestep,
        viscosity,
        terminal_half_width,
        clone_jitter,
        restitution,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    if abs(restitution) > 1:
        msg = "the Haar energy certificate requires absolute restitution <= 1"
        raise ValueError(msg)
    if not math.isfinite(integration_radius) or integration_radius <= 0:
        msg = "integration_radius must be finite and positive"
        raise ValueError(msg)
    if (
        not isinstance(integration_bins, int)
        or isinstance(integration_bins, bool)
        or integration_bins < 1
    ):
        msg = "integration_bins must be a positive integer"
        raise ValueError(msg)
    t, width = timestep / 2, terminal_half_width
    b = t * (1 + ou_coefficient)
    eta, tau = b * t, math.hypot(t * ou_amplitude, position_amplitude)
    collision = (1 + 2 * abs(restitution)) * velocity_cap
    slope = 1 - eta * (2 + 40 * math.pi**2)

    def landing(x: float) -> float:
        return (1 - 2 * eta) * x - 20 * math.pi * eta * math.sin(2 * math.pi * x)

    if tau <= 0 or slope <= 0 or b * collision >= width or landing(width) > width:
        msg = "positive noise, monotone landing into D, and positive eroded width required"
        raise ValueError(msg)
    inner = width - b * collision
    edges = np.linspace(-integration_radius, integration_radius, integration_bins + 1)
    log_values = np.array([
        _log_interval(landing(width + clone_jitter * float(z)), inner, tau) for z in edges
    ])
    log_masses = np.array([
        _log_interval(float(-(lo + hi) / 2), float((hi - lo) / 2), 1) for lo, hi in pairwise(edges)
    ])
    log_accepted = dimension * float(
        logsumexp(log_masses + np.minimum(log_values[:-1], log_values[1:]))
    )
    column = math.ceil(gaussian_row_column_bound(dimension, series_terms=column_series_terms))
    energy = velocity_cap
    if row_normalized:
        energy = min(
            collision, velocity_cap * (1 - t * viscosity + t * viscosity * math.sqrt(column))
        )
    else:
        column = 1
    scale = b / (tau * math.sqrt(dimension))
    log_box_factor = dimension * math.log(-math.expm1(-2 * width**2 / tau**2))

    def log_profile(r: float) -> float:
        return log_box_factor + dimension * float(log_ndtr(-scale * r))

    def log_perspective_derivative(r: float) -> float:
        z = scale * r
        mills = math.exp(-(z**2) / 2 - math.log(2 * math.pi) / 2 - float(log_ndtr(-z)))
        return log_profile(r) + math.log1p(dimension * z * mills / 2)

    fraction = 1.0
    log_alive = log_profile(energy)
    if log_perspective_derivative(energy) > log_accepted:
        upper = 2 * energy
        while log_perspective_derivative(upper) > log_accepted:
            upper *= 2
        root = brentq(lambda r: log_perspective_derivative(r) - log_accepted, energy, upper)
        fraction = (energy / root) ** 2
        log_alive = float(
            logsumexp([
                math.log(fraction) + log_profile(root),
                math.log1p(-fraction) + log_accepted,
            ])
        )
    return RastriginBoxSurvivalBudget(
        collision,
        energy,
        column,
        slope,
        log_accepted,
        log_alive,
        fraction,
        integration_radius,
        integration_bins,
    )


@dataclass(frozen=True)
class UniformKineticBudget:
    """Normalized Lp budgets for the actual uncapped stages, for either mode."""

    prepared_position: float
    first_kick_velocity: float
    ou_velocity: float
    second_drift_position: float
    second_kick_velocity: float
    output_position: float
    output_velocity: float
    row_column_bound: float


def _validate_moment_parameters(
    dimension: int,
    moment_order: float,
    timestep: float,
    viscosity: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
) -> None:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        msg = "dimension must be a positive integer"
        raise ValueError(msg)
    if not math.isfinite(moment_order) or moment_order < 1:
        msg = "moment_order must be finite and at least one"
        raise ValueError(msg)
    positive = (timestep, terminal_half_width, velocity_cap)
    nonnegative = (viscosity, clone_jitter, ou_coefficient, ou_amplitude, position_amplitude)
    if (
        any(not math.isfinite(x) or x <= 0 for x in positive)
        or any(not math.isfinite(x) or x < 0 for x in nonnegative)
        or not math.isfinite(restitution)
        or ou_coefficient > 1
    ):
        msg = "invalid step, region, cap, restitution or innovation parameters"
        raise ValueError(msg)
    if timestep * viscosity / 2 > 1:
        msg = "the explicit moment budget requires h nu / 2 <= 1"
        raise ValueError(msg)


def _gaussian_norm(dimension: int, order: float) -> float:
    return math.exp(
        0.5 * math.log(2)
        + (math.lgamma((dimension + order) / 2) - math.lgamma(dimension / 2)) / order
    )


def quadratic_uniform_kinetic_budget(
    *,
    dimension: int,
    moment_order: float,
    timestep: float,
    viscosity: float,
    quadratic_coefficient: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool,
) -> UniformKineticBudget:
    """Compute every stage's N-independent budget for the configured F=-k*x.

    The second row matrix uses its actual unbounded Gaussian input, through the
    dimension-only column estimate. No positive degree floor is imposed.
    """
    _validate_moment_parameters(
        dimension,
        moment_order,
        timestep,
        viscosity,
        terminal_half_width,
        clone_jitter,
        restitution,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    if not math.isfinite(quadratic_coefficient) or quadratic_coefficient < 0:
        msg = "quadratic_coefficient must be finite and nonnegative"
        raise ValueError(msg)
    p = moment_order
    gaussian_norm = _gaussian_norm(dimension, p)
    t = timestep / 2
    column = gaussian_row_column_bound(dimension) if row_normalized else 1.0
    hp = (1 + t * viscosity * (column - 1)) ** (1 / p)
    x = math.sqrt(dimension) * terminal_half_width + clone_jitter * gaussian_norm
    u = (1 + 2 * abs(restitution)) * velocity_cap + t * quadratic_coefficient * x
    w = ou_coefficient * u + ou_amplitude * gaussian_norm
    y = x + t * (1 + ou_coefficient) * u + t * ou_amplitude * gaussian_norm
    z = hp * w + t * quadratic_coefficient * y
    return UniformKineticBudget(
        x, u, w, y, z, y + position_amplitude * gaussian_norm, velocity_cap, column
    )


def styblinski_tang_uniform_kinetic_budget(
    *,
    dimension: int,
    moment_order: float,
    timestep: float,
    viscosity: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool,
) -> UniformKineticBudget:
    """Compute uniform moments for the unchanged cubic benchmark force.

    Native Styblinski-Tang acceleration is -2*x**3+16*x-2.5, so its
    global force derivative is unbounded. This budget uses moments up to
    order 9*p instead of supplying a false global Lipschitz constant.
    """
    _validate_moment_parameters(
        dimension,
        moment_order,
        timestep,
        viscosity,
        terminal_half_width,
        clone_jitter,
        restitution,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    t, p = timestep / 2, moment_order
    b = t * (1 + ou_coefficient)
    vc = (1 + 2 * abs(restitution)) * velocity_cap
    column = gaussian_row_column_bound(dimension) if row_normalized else 1.0
    hp = (1 + t * viscosity * (column - 1)) ** (1 / p)

    def x(order: float) -> float:
        return math.sqrt(dimension) * terminal_half_width + clone_jitter * _gaussian_norm(
            dimension, order
        )

    def u(order: float) -> float:
        return vc + t * (2 * x(3 * order) ** 3 + 16 * x(order) + 2.5 * math.sqrt(dimension))

    def y(order: float) -> float:
        return x(order) + b * u(order) + t * ou_amplitude * _gaussian_norm(dimension, order)

    w = ou_coefficient * u(p) + ou_amplitude * _gaussian_norm(dimension, p)
    z = hp * w + t * (2 * y(3 * p) ** 3 + 16 * y(p) + 2.5 * math.sqrt(dimension))
    return UniformKineticBudget(
        x(p),
        u(p),
        w,
        y(p),
        z,
        y(p) + position_amplitude * _gaussian_norm(dimension, p),
        velocity_cap,
        column,
    )


def rastrigin_uniform_kinetic_budget(
    *,
    dimension: int,
    moment_order: float,
    timestep: float,
    viscosity: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool,
) -> UniformKineticBudget:
    """Native Rastrigin budgets using its linear plus bounded periodic force.

    F=-2*x-20*pi*sin(2*pi*x); no global convexity or attraction between
    different wells enters these N-independent complete-stage estimates.
    """
    _validate_moment_parameters(
        dimension,
        moment_order,
        timestep,
        viscosity,
        terminal_half_width,
        clone_jitter,
        restitution,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    t, p = timestep / 2, moment_order
    gaussian = _gaussian_norm(dimension, p)
    column = gaussian_row_column_bound(dimension) if row_normalized else 1.0
    hp = min(
        (1 + t * viscosity * (column - 1)) ** (1 / p),
        1 - t * viscosity + t * viscosity * column ** (1 / p),
    )
    periodic = 20 * math.pi * math.sqrt(dimension)
    x = math.sqrt(dimension) * terminal_half_width + clone_jitter * gaussian
    u = (1 + 2 * abs(restitution)) * velocity_cap + t * (2 * x + periodic)
    w = ou_coefficient * u + ou_amplitude * gaussian
    y = x + t * (1 + ou_coefficient) * u + t * ou_amplitude * gaussian
    z = hp * w + t * (2 * y + periodic)
    return UniformKineticBudget(
        x, u, w, y, z, y + position_amplitude * gaussian, velocity_cap, column
    )


@dataclass(frozen=True)
class UniformQuadraticEnvelope:
    """N-independent actual position envelope and binomial survival constant."""

    dimension: int
    position_coefficient: float
    collision_velocity_bound: float
    kinetic_position_variance: float
    maximum_gaussian_variance: float
    bounded_position_mean: float
    log_survival_parameter: float
    log_no_clone_survival: float
    log_clone_survival: float

    def log_exponential_moment(self, delta: float) -> float:
        if (
            not math.isfinite(delta)
            or delta <= 0
            or 4 * delta * self.maximum_gaussian_variance >= 1
        ):
            msg = "delta must satisfy 0 < 4 delta tau^2 < 1"
            raise ValueError(msg)
        return 2 * delta * self.bounded_position_mean**2 - self.dimension / 2 * math.log1p(
            -4 * delta * self.maximum_gaussian_variance
        )

    def log_inverse_alive_bound(self, *, order: float = 1) -> float:
        """Uniform conditional bound for E[(N/A)^order | A>0]."""
        if not math.isfinite(order) or order <= 0:
            msg = "order must be positive and finite"
            raise ValueError(msg)
        chernoff = (1 - math.log(2)) / 2
        terms = np.array([order * math.log(2), order * math.log(order / (math.e * chernoff))])
        return float(np.logaddexp(*terms)) - order * self.log_survival_parameter

    def log_qsd_eigenvalue_lower_bound(self, n_walkers: int) -> float:
        """Bound log(1 - (1-a0)^N), using Na0/(1+Na0) below exponent range."""
        if not isinstance(n_walkers, int) or isinstance(n_walkers, bool) or n_walkers < 1:
            msg = "n_walkers must be a positive integer"
            raise ValueError(msg)
        log_a = self.log_survival_parameter
        if log_a == 0:
            return 0.0
        if log_a < -700:
            log_na = math.log(n_walkers) + log_a
            # Bernoulli: (1-a)^N <= 1/(1+Na). Unlike Na this is a lower bound.
            return log_na - float(np.logaddexp(0, log_na))
        log_extinction = n_walkers * math.log1p(-math.exp(log_a))
        return math.log(-math.expm1(log_extinction))


@dataclass(frozen=True)
class UniformEnergyAliveEnvelope:
    """Actual averaged-energy exponential transform, not binomial domination."""

    velocity_energy_bound: float
    log_alive_exponent: float
    log_accepted_floor: float
    log_fraction_test: float

    def log_count_transform_bound(self, n_walkers: int, theta: float) -> float:
        """Bound log E[exp(-theta M)] for the complete original update."""
        if not isinstance(n_walkers, int) or isinstance(n_walkers, bool) or n_walkers < 1:
            msg = "n_walkers must be a positive integer"
            raise ValueError(msg)
        if math.isnan(theta) or theta < 0:
            msg = "theta must be nonnegative"
            raise ValueError(msg)
        return -n_walkers * -math.expm1(-theta) * math.exp(self.log_alive_exponent)

    def log_inverse_alive_bound(self, *, order: float = 1) -> float:
        """Bound the current survivor/QSD E[(N/M)^order], uniformly in N."""
        if not math.isfinite(order) or order <= 0:
            msg = "order must be positive and finite"
            raise ValueError(msg)
        chernoff = (1 - math.log(2)) / 2
        return (
            float(np.logaddexp(order * math.log(2), order * math.log(order / (math.e * chernoff))))
            - order * self.log_alive_exponent
        )


def quadratic_energy_alive_envelope(
    *,
    dimension: int,
    timestep: float,
    viscosity: float,
    quadratic_coefficient: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    row_normalized: bool,
) -> UniformEnergyAliveEnvelope:
    """Optimize all actual clone fractions using pathwise collision energy.

    The accepted-row jitter integral and the no-copy energy bound are taken
    in the algorithm's order. The row version uses the dimension-only Gaussian
    column envelope and Minkowski for the first small viscous kick. Failure of
    this sufficient parameter test is not evidence of failure of the algorithm.
    """
    if abs(restitution) > 1:
        msg = "averaged collision energy needs abs(restitution) <= 1"
        raise ValueError(msg)
    envelope = quadratic_uniform_envelope(
        dimension=dimension,
        timestep=timestep,
        viscosity=viscosity,
        quadratic_coefficient=quadratic_coefficient,
        terminal_half_width=terminal_half_width,
        clone_jitter=clone_jitter,
        restitution=restitution,
        velocity_cap=velocity_cap,
        ou_coefficient=ou_coefficient,
        ou_amplitude=ou_amplitude,
        position_amplitude=position_amplitude,
    )
    kick = timestep * viscosity / 2
    # ceil(C_d) is the rational envelope used in the printed reference proof.
    column = math.ceil(gaussian_row_column_bound(dimension)) if row_normalized else 1
    energy = min(
        envelope.collision_velocity_bound,
        velocity_cap * (1 - kick + kick * math.sqrt(column)),
    )
    noise = math.sqrt(envelope.kinetic_position_variance)
    log_base = dimension * _log_interval(
        abs(envelope.position_coefficient) * terminal_half_width, terminal_half_width, noise
    )
    if log_base > math.log(0.5):
        msg = "the convex fraction optimization requires the base landing floor <= 1/2"
        raise ValueError(msg)
    z = float(ndtri_exp(log_base))
    k = timestep / 2 * (1 + ou_coefficient) / noise
    argument = z - k * energy
    log_alive = float(log_ndtr(argument))
    # D_E=f(E)-E f'(E)/2, with f'(E)=-k phi(z-k E).
    log_test = float(
        np.logaddexp(
            log_alive, math.log(energy * k / 2) - argument**2 / 2 - math.log(2 * math.pi) / 2
        )
    )
    if clone_jitter > 0 and envelope.log_clone_survival < log_test:
        msg = "the accepted landing floor does not pass the explicit clone-fraction test"
        raise ValueError(msg)
    return UniformEnergyAliveEnvelope(energy, log_alive, envelope.log_clone_survival, log_test)


@dataclass(frozen=True)
class UniformCubicPositionEnvelope:
    """Stretched-exponential position envelope for the native cubic force."""

    exponent: float
    delta: float
    log_moment_upper_bound: float

    def log_tail_upper_bound(self, radius: float) -> float:
        """Return the truncated exponential-Markov bound for one actual row."""
        if not math.isfinite(radius) or radius < 0:
            msg = "radius must be finite and nonnegative"
            raise ValueError(msg)
        return min(0.0, self.log_moment_upper_bound - self.delta * radius**self.exponent)


def styblinski_tang_uniform_position_envelope(
    *,
    dimension: int,
    timestep: float,
    viscosity: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    delta: float | None = None,
) -> UniformCubicPositionEnvelope:
    """Compute (KUP.8) for both actual dense normalizations without clipping.

    The exponent is 2/3 because the executed first position map contains the
    actual cubic force of a Gaussian jitter. The quadratic Gaussian envelope
    is not substituted for that tail. ``delta`` is an analysis choice, not an
    algorithm parameter; omission uses the explicit choice proved in KUP.8.
    """
    _validate_moment_parameters(
        dimension,
        1,
        timestep,
        viscosity,
        terminal_half_width,
        clone_jitter,
        restitution,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    t, r = timestep / 2, 2 / 3
    b, eta = t * (1 + ou_coefficient), t**2 * (1 + ou_coefficient)
    vc = (1 + 2 * abs(restitution)) * velocity_cap
    linear = 1 + 16 * eta
    bounded = b * vc + 2.5 * eta * math.sqrt(dimension)
    cx = linear**r + (2 * eta) ** r
    coefficients = (
        4 * cx * clone_jitter**2,
        2 * (t * ou_amplitude) ** r,
        2 * position_amplitude**r,
    )
    if delta is None:
        total = 2 * sum(coefficients)
        delta = 1 / total if total else 1.0
    if (
        not math.isfinite(delta)
        or delta <= 0
        or any(delta * coefficient >= 1 for coefficient in coefficients)
    ):
        msg = "delta must be positive and make every Gaussian integral finite"
        raise ValueError(msg)
    c0 = (
        linear**r
        + bounded**r
        + (t * ou_amplitude) ** r
        + position_amplitude**r
        + 2 * cx * dimension * terminal_half_width**2
    )
    log_bound = delta * c0 - dimension / 2 * sum(
        math.log1p(-delta * coefficient) for coefficient in coefficients
    )
    return UniformCubicPositionEnvelope(r, delta, log_bound)


def quadratic_uniform_envelope(
    *,
    dimension: int,
    timestep: float,
    viscosity: float,
    quadratic_coefficient: float,
    terminal_half_width: float,
    clone_jitter: float,
    restitution: float,
    velocity_cap: float,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
) -> UniformQuadraticEnvelope:
    """Evaluate both actual dense normalizations without cutting off innovations.

    The actual force is F(x)=-quadratic_coefficient*x. The first viscosity kick
    is a convex average for timestep*viscosity/2 <= 1. B2 and the cap change
    velocity, and terminal classification uses the unchanged output position.
    ``ou_amplitude`` and ``position_amplitude`` are standard deviations at their
    respective stages. No force regularity or alive fraction is supplied.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        msg = "dimension must be a positive integer"
        raise ValueError(msg)
    values = (
        timestep,
        viscosity,
        quadratic_coefficient,
        terminal_half_width,
        clone_jitter,
        velocity_cap,
        ou_coefficient,
        ou_amplitude,
        position_amplitude,
    )
    if any(not math.isfinite(x) or x < 0 for x in values) or not math.isfinite(restitution):
        msg = "parameters must be finite and amplitudes nonnegative"
        raise ValueError(msg)
    if (
        min(timestep, terminal_half_width, velocity_cap, ou_amplitude, position_amplitude) <= 0
        or ou_coefficient > 1
    ):
        msg = "step, box, cap and kinetic amplitudes must be positive; c <= 1"
        raise ValueError(msg)
    t = timestep / 2
    if t * viscosity > 1:
        msg = "the explicit convex-kick regime requires h nu / 2 <= 1"
        raise ValueError(msg)
    b = t * (1 + ou_coefficient)
    a = abs(1 - b * t * quadratic_coefficient)
    vc = (1 + 2 * abs(restitution)) * velocity_cap
    noise = (t * ou_amplitude) ** 2 + position_amplitude**2
    tau = a**2 * clone_jitter**2 + noise
    shift = b * vc
    log_base = dimension * _log_interval(
        a * terminal_half_width, terminal_half_width, math.sqrt(noise)
    )
    # Neyman-Pearson's Gaussian shift bound for a fixed bounded alignment term.
    log_base = min(log_base, math.log(math.nextafter(1.0, 0.0)))
    base_quantile = float(ndtri_exp(log_base))
    log_no_clone = float(log_ndtr(base_quantile - shift / math.sqrt(noise)))
    eroded = terminal_half_width - shift
    log_clone = (
        dimension * _log_interval(a * terminal_half_width, eroded, math.sqrt(tau))
        if eroded > 0
        else -math.inf
    )
    # Accepted and persisting tokens may both occur, so retain the worse branch.
    log_survival = min(log_no_clone, log_clone) if clone_jitter > 0 else log_no_clone
    log_survival = min(log_survival, math.log(math.nextafter(1.0, 0.0)))
    if not math.isfinite(log_survival):
        msg = "the explicit landing bound needs b Vc < terminal_half_width and finite Gaussian log-CDF resolution"
        raise ValueError(msg)
    return UniformQuadraticEnvelope(
        dimension,
        1 - b * t * quadratic_coefficient,
        vc,
        noise,
        tau,
        a * math.sqrt(dimension) * terminal_half_width + shift,
        log_survival,
        log_no_clone,
        log_clone,
    )


@dataclass(frozen=True)
class CoupledBAOABBalance:
    """Full sampled kinetic increments of a coupled normalized quadratic form."""

    input_form: Tensor
    first_force_difference: Tensor
    second_force_difference: Tensor
    raw_position_difference: Tensor
    raw_velocity_difference: Tensor
    cap_correction: Tensor
    signed_kinetic_increment: Tensor
    signed_cap_increment: Tensor
    output_form: Tensor
    first_output_positions: Tensor
    second_output_positions: Tensor
    first_output_velocities: Tensor
    second_output_velocities: Tensor


def coupled_baoab_balance(
    first_positions: Tensor,
    first_velocities: Tensor,
    second_positions: Tensor,
    second_velocities: Tensor,
    ou_innovation: Tensor,
    position_innovation: Tensor,
    *,
    force: Callable[[Tensor], Tensor],
    timestep: float,
    viscosity: float,
    bandwidth: float,
    row_normalized: bool,
    ou_coefficient: float,
    ou_amplitude: float,
    position_amplitude: float,
    velocity_cap: float,
    position_weight: float,
    cross_weight: float,
    velocity_weight: float,
) -> CoupledBAOABBalance:
    """Evaluate both actual kicks with shared, unrestricted Gaussian draws.

    Inputs are the *actual* prepared arrays after frozen copying, accepted-row
    jitter and component collision. The force callback must be the configured
    acceleration map, evaluated at both B stages. Returned expectations require
    averaging this ledger under the actual source/Haar/innovation law; this
    function does not turn a sampled negative drift into a uniform theorem.
    """
    arrays = (
        first_positions,
        first_velocities,
        second_positions,
        second_velocities,
        ou_innovation,
        position_innovation,
    )
    _arrays(*arrays)
    if (
        first_positions.ndim != 2
        or not first_positions.numel()
        or any(a.shape != first_positions.shape for a in arrays)
    ):
        msg = "all six arrays must have identical nonempty shape [N,d]"
        raise ValueError(msg)
    positive = (timestep, bandwidth, velocity_cap, position_weight, velocity_weight)
    nonnegative = (viscosity, ou_coefficient, ou_amplitude, position_amplitude)
    if (
        any(not math.isfinite(v) or v <= 0 for v in positive)
        or any(not math.isfinite(v) or v < 0 for v in nonnegative)
        or ou_coefficient > 1
    ):
        msg = "invalid positive parameters, amplitudes or OU coefficient"
        raise ValueError(msg)
    if not math.isfinite(cross_weight) or position_weight * velocity_weight <= cross_weight**2:
        msg = "the quadratic form must be positive definite"
        raise ValueError(msg)
    t, c = timestep / 2, ou_coefficient
    b, eta = t * (1 + c), t**2 * (1 + c)

    def total(x: Tensor, v: Tensor) -> Tensor:
        external = force(x)
        _arrays(x, external)
        if external.shape != x.shape:
            msg = "the force callback must return [N,d]"
            raise ValueError(msg)
        return external + gaussian_viscous_force(
            x, v, viscosity=viscosity, bandwidth=bandwidth, row_normalized=row_normalized
        )

    def stages(x: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        f = total(x, v)
        v1 = v + t * f
        x1 = x + t * v1
        z = c * v1 + ou_amplitude * ou_innovation
        x2 = x1 + t * z
        g = total(x2, z)
        raw_v = z + t * g
        capped = (
            velocity_cap
            * raw_v
            / (velocity_cap + torch.linalg.vector_norm(raw_v, dim=1, keepdim=True))
        )
        return f, g, x2 + position_amplitude * position_innovation, raw_v, capped

    f1, g1, x1, _raw1, capped1 = stages(first_positions, first_velocities)
    f2, g2, x2, _raw2, capped2 = stages(second_positions, second_velocities)
    r = first_positions - second_positions
    z = first_velocities - second_velocities
    f, g = f1 - f2, g1 - g2
    R, Z = r + b * z + eta * f, c * z + c * t * f + t * g
    cap = capped1 - capped2 - Z

    def dot(a: Tensor, b: Tensor) -> Tensor:
        return (a * b).sum(1).mean()

    def form(x: Tensor, v: Tensor) -> Tensor:
        return (
            position_weight * dot(x, x)
            + 2 * cross_weight * dot(x, v)
            + velocity_weight * dot(v, v)
        )

    # Expand before bounding: preserve every signed external/viscous cross term.
    increment = position_weight * (
        2 * b * dot(r, z)
        + 2 * eta * dot(r, f)
        + b**2 * dot(z, z)
        + 2 * b * eta * dot(z, f)
        + eta**2 * dot(f, f)
    )
    increment += (
        2
        * cross_weight
        * (
            (c - 1) * dot(r, z)
            + c * t * dot(r, f)
            + t * dot(r, g)
            + b * c * dot(z, z)
            + (b * c * t + eta * c) * dot(z, f)
            + b * t * dot(z, g)
            + eta * c * t * dot(f, f)
            + eta * t * dot(f, g)
        )
    )
    increment += velocity_weight * (
        (c**2 - 1) * dot(z, z)
        + 2 * c**2 * t * dot(z, f)
        + 2 * c * t * dot(z, g)
        + c**2 * t**2 * dot(f, f)
        + 2 * c * t**2 * dot(f, g)
        + t**2 * dot(g, g)
    )
    cap_increment = 2 * cross_weight * dot(R, cap) + velocity_weight * (
        2 * dot(Z, cap) + dot(cap, cap)
    )
    return CoupledBAOABBalance(
        form(r, z),
        f,
        g,
        R,
        Z,
        cap,
        increment,
        cap_increment,
        form(x1 - x2, capped1 - capped2),
        x1,
        x2,
        capped1,
        capped2,
    )
