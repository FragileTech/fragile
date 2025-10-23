"""
Convergence Analysis Tools for Geometric Gas.

This module provides computational tools for analyzing exponential convergence
to QSD, separated from visualization logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde
import torch

from fragile.core.benchmarks import MixtureOfGaussians


if TYPE_CHECKING:
    from fragile.core.euclidean_gas import SwarmState
    from fragile.geometric_gas import GeometricGas


def create_multimodal_potential(
    dims: int = 2,
    n_gaussians: int = 3,
    centers: torch.Tensor | None = None,
    stds: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    bounds_range: tuple[float, float] = (-8.0, 8.0),
    seed: int | None = None,
) -> MixtureOfGaussians:
    """Create a multimodal potential for convergence testing.

    Args:
        dims: Dimensionality of the space
        n_gaussians: Number of Gaussian modes
        centers: Optional centers [n_gaussians, dims]
        stds: Optional standard deviations [n_gaussians, dims]
        weights: Optional mixture weights [n_gaussians]
        bounds_range: Bounds for the potential
        seed: Random seed for reproducibility

    Returns:
        MixtureOfGaussians instance (callable, with bounds attribute)
    """
    if centers is None:
        # Default: well-separated modes
        if dims == 2 and n_gaussians == 3:
            centers = torch.tensor([
                [0.0, 0.0],  # Mode 1: Origin (highest weight)
                [4.0, 3.0],  # Mode 2: Upper right
                [-3.0, 2.5],  # Mode 3: Upper left
            ])
        else:
            # Random centers
            torch.manual_seed(seed or 42)
            low, high = bounds_range
            centers = torch.rand(n_gaussians, dims) * (high - low) / 2

    if stds is None:
        # Default: varying spread
        if dims == 2 and n_gaussians == 3:
            stds = torch.tensor([
                [0.8, 0.8],  # Mode 1: Tight peak
                [1.0, 1.0],  # Mode 2: Medium spread
                [1.2, 1.2],  # Mode 3: Wider peak
            ])
        else:
            stds = torch.ones(n_gaussians, dims) * 0.8

    if weights is None:
        # Default: decreasing weights
        if n_gaussians == 3:
            weights = torch.tensor([0.5, 0.3, 0.2])
        else:
            weights = torch.ones(n_gaussians) / n_gaussians

    # Create mixture (returns OptimBenchmark instance that is callable)
    return MixtureOfGaussians(
        dims=dims,
        n_gaussians=n_gaussians,
        centers=centers,
        stds=stds,
        weights=weights,
        bounds_range=bounds_range,
        seed=seed,
    )


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics at each time step."""

    time: list[int] = field(default_factory=list)
    kl_divergence: list[float] = field(default_factory=list)
    wasserstein_distance: list[float] = field(default_factory=list)
    lyapunov_value: list[float] = field(default_factory=list)
    mean_position: list[np.ndarray] = field(default_factory=list)
    variance_position: list[float] = field(default_factory=list)

    def add_snapshot(
        self,
        time: int,
        kl_divergence: float,
        wasserstein_distance: float,
        lyapunov_value: float,
        mean_position: np.ndarray,
        variance_position: float,
    ):
        """Add metrics for a time step."""
        self.time.append(time)
        self.kl_divergence.append(kl_divergence)
        self.wasserstein_distance.append(wasserstein_distance)
        self.lyapunov_value.append(lyapunov_value)
        self.mean_position.append(mean_position)
        self.variance_position.append(variance_position)

    def to_dict(self) -> dict:
        """Convert to dictionary for easy access."""
        return {
            "time": np.array(self.time),
            "kl_divergence": np.array(self.kl_divergence),
            "wasserstein_distance": np.array(self.wasserstein_distance),
            "lyapunov_value": np.array(self.lyapunov_value),
            "mean_position": np.array(self.mean_position),
            "variance_position": np.array(self.variance_position),
        }

    def get_valid_kl_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get valid (finite, positive) KL divergence data.

        Returns:
            Tuple of (time_array, kl_array) with invalid values filtered out
        """
        time_arr = np.array(self.time)
        kl_arr = np.array(self.kl_divergence)

        valid_mask = np.isfinite(kl_arr) & (kl_arr > 0)
        return time_arr[valid_mask], kl_arr[valid_mask]

    def fit_exponential_decay(
        self, metric: str = "kl_divergence", fit_start_time: int = 100
    ) -> tuple[float, float] | None:
        """Fit exponential decay to a metric.

        Args:
            metric: Which metric to fit ("kl_divergence", "wasserstein_distance", "lyapunov_value")
            fit_start_time: Start fitting after this time (skip transient)

        Returns:
            Tuple of (kappa, C) where metric ≈ C * exp(-kappa * t), or None if not enough data
        """
        if metric == "kl_divergence":
            time_arr, values = self.get_valid_kl_data()
        else:
            time_arr = np.array(self.time)
            values = np.array(getattr(self, metric))
            valid_mask = np.isfinite(values) & (values > 0)
            time_arr = time_arr[valid_mask]
            values = values[valid_mask]

        # Filter to fitting region
        fit_mask = time_arr >= fit_start_time
        time_fit = time_arr[fit_mask]
        values_fit = values[fit_mask]

        if len(time_fit) < 10:
            return None

        # Linear fit on log scale: log(metric) = log(C) - kappa * t
        log_values = np.log(values_fit)
        coeffs = np.polyfit(time_fit, log_values, 1)

        kappa = -coeffs[0]  # Convergence rate
        C = np.exp(coeffs[1])  # Initial constant

        return kappa, C


class ConvergenceAnalyzer:
    """Analyzer for computing convergence metrics."""

    def __init__(
        self,
        target_mixture: MixtureOfGaussians,
        target_centers: torch.Tensor,
        target_weights: torch.Tensor,
    ):
        """Initialize analyzer.

        Args:
            target_mixture: Target distribution
            target_centers: Centers of target Gaussians [n_gaussians, dims]
            target_weights: Weights of target Gaussians [n_gaussians]
        """
        self.target_mixture = target_mixture
        self.target_centers = target_centers
        self.target_weights = target_weights

    def compute_kl_divergence_kde(self, samples: np.ndarray, n_grid: int = 1000) -> float:
        """Approximate KL divergence using KDE for empirical distribution.

        KL(empirical || target) = ∫ p_emp(x) log(p_emp(x) / p_target(x)) dx

        Args:
            samples: Sample positions [N, dims]
            n_grid: Number of grid points for integration

        Returns:
            KL divergence (non-negative)
        """
        if len(samples) < 10:
            return float("inf")

        # Create KDE from samples
        try:
            kde = gaussian_kde(samples.T, bw_method="scott")
        except Exception:
            return float("inf")

        # Sample grid points from empirical distribution
        grid_samples = kde.resample(n_grid).T
        grid_samples_torch = torch.tensor(grid_samples, dtype=torch.float32)

        # Evaluate densities
        p_emp = kde(grid_samples.T)

        # Target density (unnormalized)
        U_vals = self.target_mixture(grid_samples_torch).numpy()
        p_target = np.exp(-U_vals)
        Z = p_target.sum()
        p_target /= Z

        # KL divergence (with numerical stability)
        mask = (p_emp > 1e-10) & (p_target > 1e-10)
        if not mask.any():
            return float("inf")

        kl = np.sum(p_emp[mask] * np.log(p_emp[mask] / p_target[mask])) / n_grid

        return max(0.0, kl)  # Ensure non-negative

    def compute_wasserstein_distance(self, samples: torch.Tensor) -> float:
        """Approximate Wasserstein-2 distance using mean.

        Args:
            samples: Sample positions [N, dims]

        Returns:
            L2 distance between empirical and target means
        """
        # Empirical mean
        emp_mean = samples.mean(dim=0)

        # Target mean
        target_mean = (self.target_centers * self.target_weights.unsqueeze(1)).sum(dim=0)

        # L2 distance
        return torch.norm(emp_mean - target_mean).item()

    def compute_lyapunov_function(self, state: SwarmState) -> float:
        """Compute framework-correct Lyapunov function V_total.

        Uses the framework definition from docs/source/1_euclidean_gas/03_cloning.md:

            V_total(S) = V_Var,x(S) + V_Var,v(S)

        where each term is N-normalized internal variance.

        Args:
            state: Swarm state

        Returns:
            Total Lyapunov function value (N-normalized)
        """
        from fragile.lyapunov import compute_total_lyapunov

        return compute_total_lyapunov(state).item()

    def analyze_state(self, state: SwarmState, time: int) -> dict:
        """Compute all metrics for a given state.

        Args:
            state: Swarm state
            time: Current time step

        Returns:
            Dictionary with all metrics
        """
        samples_np = state.x.detach().numpy()

        return {
            "time": time,
            "kl_divergence": self.compute_kl_divergence_kde(samples_np),
            "wasserstein_distance": self.compute_wasserstein_distance(state.x.detach()),
            "lyapunov_value": self.compute_lyapunov_function(state),
            "mean_position": samples_np.mean(axis=0),
            "variance_position": samples_np.var(),
        }


class ConvergenceExperiment:
    """Runner for convergence experiments."""

    def __init__(
        self,
        gas: GeometricGas,
        analyzer: ConvergenceAnalyzer,
        save_snapshots_at: list[int] | None = None,
    ):
        """Initialize experiment.

        Args:
            gas: Geometric Gas instance
            analyzer: Convergence analyzer
            save_snapshots_at: Time steps to save full state snapshots
        """
        self.gas = gas
        self.analyzer = analyzer
        self.save_snapshots_at = save_snapshots_at or []

        self.metrics = ConvergenceMetrics()
        self.snapshots: dict[int, torch.Tensor] = {}

    def run(
        self,
        n_steps: int,
        x_init: torch.Tensor | None = None,
        v_init: torch.Tensor | None = None,
        measure_every: int = 10,
        verbose: bool = True,
    ) -> tuple[ConvergenceMetrics, dict[int, torch.Tensor]]:
        """Run convergence experiment.

        Args:
            n_steps: Number of steps to run
            x_init: Initial positions (optional)
            v_init: Initial velocities (optional)
            measure_every: Compute metrics every N steps
            verbose: Print progress

        Returns:
            Tuple of (metrics, snapshots)
        """
        # Initialize state
        state = self.gas.initialize_state(x_init, v_init)

        if verbose:
            print(f"Running convergence experiment for {n_steps} steps...")
            print(f"  N walkers: {state.N}")
            print(f"  Dimensions: {state.d}")
            print(f"  Measuring every {measure_every} steps")
            print()

        # Initial metrics
        initial_metrics = self.analyzer.analyze_state(state, 0)
        self.metrics.add_snapshot(**initial_metrics)

        # Save initial snapshot
        if 0 in self.save_snapshots_at:
            self.snapshots[0] = state.x.clone()

        # Main loop
        for step in range(n_steps):
            # Perform step
            _, state = self.gas.step(state)

            # Compute metrics
            if (step + 1) % measure_every == 0:
                metrics_dict = self.analyzer.analyze_state(state, step + 1)
                self.metrics.add_snapshot(**metrics_dict)

                if verbose and (step + 1) % (measure_every * 10) == 0:
                    kl = metrics_dict["kl_divergence"]
                    w2 = metrics_dict["wasserstein_distance"]
                    print(f"  Step {step + 1:5d}: KL={kl:.4f}, W2={w2:.4f}")

            # Save snapshot
            if (step + 1) in self.save_snapshots_at:
                self.snapshots[step + 1] = state.x.clone()

        if verbose:
            print("\n✓ Experiment complete!")
            print(f"  Total steps: {n_steps}")
            print(f"  Measurements: {len(self.metrics.time)}")
            print(f"  Snapshots: {len(self.snapshots)}")

        return self.metrics, self.snapshots

    def get_convergence_summary(self) -> dict:
        """Get summary of convergence analysis.

        Returns:
            Dictionary with convergence statistics
        """
        # Fit exponential decay
        kl_fit = self.metrics.fit_exponential_decay("kl_divergence", fit_start_time=100)
        w2_fit = self.metrics.fit_exponential_decay("wasserstein_distance", fit_start_time=100)
        lyap_fit = self.metrics.fit_exponential_decay("lyapunov_value", fit_start_time=100)

        summary = {
            "n_steps": len(self.metrics.time),
            "final_time": self.metrics.time[-1] if self.metrics.time else 0,
        }

        if kl_fit is not None:
            kappa_kl, C_kl = kl_fit
            summary["kl_convergence_rate"] = kappa_kl
            summary["kl_constant"] = C_kl
            summary["kl_half_life"] = np.log(2) / kappa_kl if kappa_kl > 0 else float("inf")

        if w2_fit is not None:
            kappa_w2, C_w2 = w2_fit
            summary["w2_convergence_rate"] = kappa_w2
            summary["w2_constant"] = C_w2

        if lyap_fit is not None:
            kappa_lyap, C_lyap = lyap_fit
            summary["lyapunov_decay_rate"] = kappa_lyap
            summary["lyapunov_constant"] = C_lyap

        # Final metrics
        if self.metrics.time:
            summary["final_kl"] = self.metrics.kl_divergence[-1]
            summary["final_w2"] = self.metrics.wasserstein_distance[-1]
            summary["final_lyapunov"] = self.metrics.lyapunov_value[-1]
            summary["final_mean_position"] = self.metrics.mean_position[-1]

        return summary
