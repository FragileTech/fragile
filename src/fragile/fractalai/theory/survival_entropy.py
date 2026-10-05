"""Diagnostics for the exact physical-survivor entropy chain rule.

The mortality budget applies to the original full marked kernel. Finite
probability tables verify its algebra; they are not a discretization proof
or a substitute for the actual gas's extinction certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr


@dataclass(frozen=True)
class UniformSurvivalEntropyBudget:
    """An actual extinction exponent, with no eigenfunction-ratio input."""

    extinction_exponent: float
    minimum_population: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.extinction_exponent) or self.extinction_exponent <= 0:
            msg = "extinction exponent must be finite and positive"
            raise ValueError(msg)
        self._population(self.minimum_population)

    @classmethod
    def from_alive_floor(
        cls, alive_floor: float, *, minimum_population: int = 1
    ) -> UniformSurvivalEntropyBudget:
        """Use the proved Laplace coefficient: death <= (1-a)**N."""
        if not math.isfinite(alive_floor) or not 0 < alive_floor < 1:
            msg = "alive floor must lie strictly between zero and one"
            raise ValueError(msg)
        return cls(-math.log1p(-alive_floor), minimum_population)

    @staticmethod
    def _population(value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = "population must be a positive integer"
            raise ValueError(msg)

    def log_prefactor(self, n_walkers: int, *, steps: int = 1) -> float:
        """KUE.4: log[(1-exp(-N*a))**(-steps)] without cancellation."""
        self._population(n_walkers)
        if n_walkers < self.minimum_population:
            msg = "population is below the declared minimum"
            raise ValueError(msg)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            msg = "steps must be a nonnegative integer"
            raise ValueError(msg)
        z = n_walkers * self.extinction_exponent
        log_survival = math.log(-math.expm1(-z)) if z <= math.log(2) else math.log1p(-math.exp(-z))
        return -steps * log_survival

    @property
    def uniform_prefactor(self) -> float:
        """The one-step bound valid for all N >= minimum_population."""
        return math.exp(self.log_prefactor(self.minimum_population))


@dataclass(frozen=True)
class SurvivorEntropyBalance:
    """All nonnegative terms of KUE.1 on a finite probability table."""

    input_entropy: float
    output_entropy: float
    survival_probability: float
    qsd_survival: float
    bernoulli_entropy: float
    backward_information: float
    rejected_input_entropy: float

    @property
    def reconstructed_input_entropy(self) -> float:
        return (
            self.bernoulli_entropy
            + self.survival_probability * (self.output_entropy + self.backward_information)
            + (1 - self.survival_probability) * self.rejected_input_entropy
        )


def finite_survivor_entropy_balance(
    killed_kernel: NDArray[np.float64],
    input_law: NDArray[np.float64],
    qsd: NDArray[np.float64],
) -> SurvivorEntropyBalance:
    """Evaluate the complete rejected/surviving chain on a finite state table.

    The qsd argument must satisfy nu*Q = alpha*nu. The function retains input
    rejection and backward conditional information; it never uses a right
    eigenfunction or assumes invariance of separate component kernels.
    """
    kernel = np.asarray(killed_kernel, dtype=float)
    p, nu = np.asarray(input_law, dtype=float), np.asarray(qsd, dtype=float)
    if (
        kernel.ndim != 2
        or kernel.shape[0] != kernel.shape[1]
        or p.shape != (kernel.shape[0],)
        or nu.shape != p.shape
        or not p.size
    ):
        msg = "a nonempty square kernel and matching input/QSD vectors are required"
        raise ValueError(msg)
    if (
        any(not np.isfinite(x).all() for x in (kernel, p, nu))
        or (kernel < 0).any()
        or (p < 0).any()
        or (nu <= 0).any()
        or not (
            math.isclose(float(p.sum()), 1, rel_tol=1e-10, abs_tol=1e-12)
            and math.isclose(float(nu.sum()), 1, rel_tol=1e-10, abs_tol=1e-12)
        )
    ):
        msg = "finite nonnegative probabilities and a strictly positive QSD required"
        raise ValueError(msg)
    p, nu = p / p.sum(), nu / nu.sum()
    k = kernel.sum(1)
    if (k > 1).any():
        msg = "kernel must be sub-Markov"
        raise ValueError(msg)
    death = 1 - k
    a, alpha = float(p @ k), float(nu @ k)
    if a <= 0 or alpha <= 0 or not np.allclose(nu @ kernel, alpha * nu, rtol=1e-10, atol=1e-12):
        msg = "positive survival and the actual QSD equation are required"
        raise ValueError(msg)
    joint_p, joint_nu = p[:, None] * kernel / a, nu[:, None] * kernel / alpha
    output = joint_p.sum(0)
    # Chain rule on the full accepted input/output law.
    joint_entropy = float(rel_entr(joint_p, joint_nu).sum())
    output_entropy = float(rel_entr(output, nu).sum())
    rejected = (
        float(rel_entr(p * death / (1 - a), nu * death / (1 - alpha)).sum()) if a < 1 else 0.0
    )
    bernoulli = float(rel_entr([a, 1 - a], [alpha, 1 - alpha]).sum())
    return SurvivorEntropyBalance(
        float(rel_entr(p, nu).sum()),
        output_entropy,
        a,
        alpha,
        bernoulli,
        joint_entropy - output_entropy,
        rejected,
    )
