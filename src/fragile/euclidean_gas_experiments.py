"""
Euclidean Gas Convergence Experiments

This module provides utilities for running and analyzing convergence experiments
with the Euclidean Gas algorithm. It includes:

1. Helper functions for creating test potentials
2. ConvergenceMetrics class for tracking Lyapunov decay
3. ConvergenceExperiment class for running convergence simulations

All computational logic is here to keep notebooks clean and focused on visualization.

Usage:
    from fragile.euclidean_gas_experiments import (
        create_multimodal_potential,
        ConvergenceExperiment,
        ConvergenceMetrics,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from torch import Tensor

from fragile.benchmarks import MixtureOfGaussians
from fragile.euclidean_gas import EuclideanGas, PotentialParams, SwarmState
from fragile.lyapunov import (
    compute_internal_variance_position,
    compute_internal_variance_velocity,
)


class MixtureBasedPotential(PotentialParams):
    """Potential derived from Mixture of Gaussians.

    This wrapper allows using MixtureOfGaussians from benchmarks.py
    as a PotentialParams instance for EuclideanGas.

    Args:
        mixture: MixtureOfGaussians instance
    """

    mixture: object  # MixtureOfGaussians instance

    model_config = {"arbitrary_types_allowed": True}

    def evaluate(self, x: Tensor) -> Tensor:
        """Evaluate potential U(x) = -log(mixture density).

        Args:
            x: Positions [N, d]

        Returns:
            Potential values [N]
        """
        return self.mixture(x)


def create_multimodal_potential(
    dims: int = 2,
    n_gaussians: int = 3,
    centers: Tensor | np.ndarray | None = None,
    stds: Tensor | np.ndarray | None = None,
    weights: Tensor | np.ndarray | None = None,
    bounds_range: tuple[float, float] = (-10.0, 10.0),
    seed: int | None = None,
) -> tuple[MixtureBasedPotential, MixtureOfGaussians]:
    """Create a multimodal potential from Mixture of Gaussians.

    Args:
        dims: Spatial dimension
        n_gaussians: Number of Gaussian components
        centers: Optional centers [n_gaussians, dims]
        stds: Optional standard deviations [n_gaussians, dims]
        weights: Optional mixture weights [n_gaussians]
        bounds_range: Tuple (low, high) for bounds
        seed: Random seed for reproducibility

    Returns:
        (potential, mixture): Tuple of potential wrapper and mixture instance
    """
    mixture = MixtureOfGaussians(
        dims=dims,
        n_gaussians=n_gaussians,
        centers=centers,
        stds=stds,
        weights=weights,
        bounds_range=bounds_range,
        seed=seed,
    )

    potential = MixtureBasedPotential(mixture=mixture)

    return potential, mixture


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics during Euclidean Gas simulation.

    Tracks Lyapunov function components and alive walker counts over time.

    Attributes:
        time: List of time steps when measurements were taken
        V_total: Total Lyapunov function V_total = V_var_x + V_var_v
        V_var_x: Position variance component (N-normalized)
        V_var_v: Velocity variance component (N-normalized)
        n_alive: Number of alive walkers at each time
    """

    time: list[int] = field(default_factory=list)
    V_total: list[float] = field(default_factory=list)
    V_var_x: list[float] = field(default_factory=list)
    V_var_v: list[float] = field(default_factory=list)
    n_alive: list[int] = field(default_factory=list)

    def add_measurement(
        self,
        t: int,
        state: SwarmState,
        alive_mask: Tensor | None = None,
    ) -> None:
        """Add a measurement at time t.

        Args:
            t: Time step
            state: Current swarm state
            alive_mask: Boolean mask [N] for alive walkers
        """
        # Compute Lyapunov components
        V_var_x = compute_internal_variance_position(state, alive_mask)
        V_var_v = compute_internal_variance_velocity(state, alive_mask)
        V_total = V_var_x + V_var_v

        # Count alive walkers
        if alive_mask is not None:
            n_alive = alive_mask.sum().item()
        else:
            n_alive = state.N

        # Store
        self.time.append(t)
        self.V_total.append(V_total.item())
        self.V_var_x.append(V_var_x.item())
        self.V_var_v.append(V_var_v.item())
        self.n_alive.append(n_alive)

    def fit_exponential_decay(
        self,
        metric_name: str = "V_total",
        fit_start_time: int = 0,
    ) -> tuple[float, float] | None:
        """Fit exponential decay to a metric: y(t) = C exp(-κ t).

        Args:
            metric_name: Name of metric to fit ('V_total', 'V_var_x', 'V_var_v')
            fit_start_time: Start fitting from this time (ignore transient)

        Returns:
            (kappa, C): Decay rate and amplitude, or None if fit fails
        """
        if metric_name not in {"V_total", "V_var_x", "V_var_v"}:
            msg = f"Unknown metric: {metric_name}"
            raise ValueError(msg)

        # Get data
        time_arr = np.array(self.time)
        metric_arr = np.array(getattr(self, metric_name))

        # Filter by start time and positive values
        mask = (time_arr >= fit_start_time) & (metric_arr > 0)
        time_fit = time_arr[mask]
        metric_fit = metric_arr[mask]

        if len(time_fit) < 2:
            return None

        # Linear regression on log(y) vs t: log(y) = log(C) - κ t
        log_y = np.log(metric_fit)

        # Least squares fit
        A = np.vstack([time_fit, np.ones_like(time_fit)]).T
        result = np.linalg.lstsq(A, log_y, rcond=None)
        slope, intercept = result[0]

        kappa = -slope  # Decay rate (negative slope)
        C = np.exp(intercept)  # Amplitude

        return kappa, C


class ConvergenceExperiment:
    """Run convergence experiments with Euclidean Gas.

    This class handles:
    - Running the simulation for n_steps
    - Measuring Lyapunov functions at regular intervals
    - Saving position snapshots at key times
    - Tracking alive/dead walkers

    Args:
        gas: EuclideanGas instance
        save_snapshots_at: List of time steps to save full position snapshots
    """

    def __init__(
        self,
        gas: EuclideanGas,
        save_snapshots_at: list[int] | None = None,
    ):
        self.gas = gas
        self.save_snapshots_at = save_snapshots_at or []

    def run(
        self,
        n_steps: int,
        x_init: Tensor | None = None,
        v_init: Tensor | None = None,
        measure_every: int = 1,
        verbose: bool = False,
    ) -> tuple[ConvergenceMetrics, dict[int, Tensor]]:
        """Run convergence experiment.

        Args:
            n_steps: Number of simulation steps
            x_init: Initial positions [N, d] (optional)
            v_init: Initial velocities [N, d] (optional)
            measure_every: Measure metrics every N steps
            verbose: Print progress information

        Returns:
            (metrics, snapshots): Tuple of ConvergenceMetrics and dict of snapshots
                snapshots is {time: positions} for requested snapshot times
        """
        if verbose:
            print(f"Running convergence experiment for {n_steps} steps...")
            print(f"  N walkers: {self.gas.params.N}")
            print(f"  Dimensions: {self.gas.params.d}")
            print(f"  Measuring every {measure_every} steps")

        # Initialize state
        state = self.gas.initialize_state(x_init, v_init)

        # Initialize metrics
        metrics = ConvergenceMetrics()
        snapshots = {}

        # Get bounds for alive check
        bounds = self.gas.bounds

        # Measure initial state
        if bounds is not None:
            alive_mask = bounds.contains(state.x)
        else:
            alive_mask = None

        metrics.add_measurement(0, state, alive_mask)

        # Save initial snapshot if requested
        if 0 in self.save_snapshots_at:
            snapshots[0] = state.x.clone()

        # Main simulation loop
        for step in range(n_steps):
            # Step the gas
            _, state = self.gas.step(state)

            # Check alive status
            if bounds is not None:
                alive_mask = bounds.contains(state.x)
                n_alive = alive_mask.sum().item()

                # Check for extinction
                if n_alive == 0:
                    if verbose:
                        print(f"  All walkers died at step {step + 1}. Stopping.")
                    break
            else:
                alive_mask = None

            # Measure metrics
            if (step + 1) % measure_every == 0:
                metrics.add_measurement(step + 1, state, alive_mask)

            # Save snapshot if requested
            if (step + 1) in self.save_snapshots_at:
                snapshots[step + 1] = state.x.clone()

        if verbose:
            print("\n✓ Experiment complete!")
            print(f"  Total measurements: {len(metrics.time)}")
            print(f"  Snapshots saved: {len(snapshots)}")
            print(f"  Final V_total: {metrics.V_total[-1]:.6f}")
            print(f"  Final n_alive: {metrics.n_alive[-1]}")

        return metrics, snapshots
