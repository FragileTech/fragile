"""Tests for parameter optimization functionality.

Tests the parameter optimization system including:
- Landscape estimation from RunHistory
- Trajectory data extraction
- Multi-strategy optimization
- GasParams <-> GasConfig conversion
"""

import pytest
import torch
import numpy as np
import holoviews as hv

# Initialize HoloViews for GasConfig tests
hv.extension('bokeh')

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import Sphere
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas
from fragile.core.fitness import FitnessOperator
from fragile.core.kinetic_operator import KineticOperator
from fragile.gas_parameters import (
    GasParams,
    LandscapeParams,
    estimate_landscape_from_history,
    extract_trajectory_data_from_history,
    optimize_parameters_multi_strategy,
    gas_params_from_config,
    gas_params_to_config_dict,
    apply_gas_params_to_config,
)
from fragile.experiments.gas_config_dashboard import GasConfig


@pytest.fixture
def test_simulation():
    """Create a small test simulation for testing."""
    bounds = TorchBounds(
        low=torch.full((2,), -5.0, dtype=torch.float32),
        high=torch.full((2,), 5.0, dtype=torch.float32),
    )
    benchmark = Sphere(dims=2)

    gas = EuclideanGas(
        N=30,
        d=2,
        companion_selection=CompanionSelection(method="softmax", epsilon=0.1, lambda_alg=1.0),
        potential=benchmark,
        kinetic_op=KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.1,
            potential=benchmark,
            device=torch.device("cpu"),
            dtype=torch.float32,
            bounds=bounds,
        ),
        cloning=CloneOperator(sigma_x=0.1, alpha_restitution=0.5, p_max=0.3, epsilon_clone=0.01),
        fitness_op=FitnessOperator(alpha=1.0, beta=0.1, eta=0.1, lambda_alg=1.0),
        bounds=bounds,
        device=torch.device("cpu"),
        dtype="float32",
        enable_cloning=True,
        enable_kinetic=True,
    )

    history = gas.run(n_steps=20)
    return history, benchmark


def test_estimate_landscape_from_history_no_hessian(test_simulation):
    """Test landscape estimation without Hessian data (variance-based fallback)."""
    history, _ = test_simulation

    # Estimate landscape (no Hessian data in Sphere benchmark)
    landscape = estimate_landscape_from_history(history, use_bounds_analysis=True)

    assert isinstance(landscape, LandscapeParams)
    assert landscape.lambda_min > 0
    assert landscape.lambda_max > landscape.lambda_min
    assert landscape.d == 2
    assert landscape.f_typical > 0


def test_extract_trajectory_data(test_simulation):
    """Test trajectory data extraction from RunHistory."""
    history, _ = test_simulation

    trajectory_data = extract_trajectory_data_from_history(history, stage="final")

    assert "V_Var_x" in trajectory_data
    assert "V_Var_v" in trajectory_data
    assert "V_W" in trajectory_data
    assert "W_b" in trajectory_data

    # Check shapes
    assert len(trajectory_data["V_Var_x"]) == history.n_recorded
    assert len(trajectory_data["V_Var_v"]) == history.n_recorded

    # Convert to numpy if needed and check all values are finite
    V_Var_x = trajectory_data["V_Var_x"]
    V_Var_v = trajectory_data["V_Var_v"]

    # Handle torch tensors
    if hasattr(V_Var_x, 'cpu'):
        V_Var_x = V_Var_x.cpu().numpy()
    if hasattr(V_Var_v, 'cpu'):
        V_Var_v = V_Var_v.cpu().numpy()

    assert np.all(np.isfinite(V_Var_x))
    assert np.all(np.isfinite(V_Var_v))


def test_optimize_parameters_balanced_strategy(test_simulation):
    """Test balanced strategy optimization."""
    history, _ = test_simulation

    # Create current params
    landscape = estimate_landscape_from_history(history)
    current_params = GasParams(
        tau=0.1,
        gamma=1.0,
        sigma_v=1.4,
        lambda_clone=1.0,
        N=30,
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_rest=0.5,
        d_safe=3.0,
        kappa_wall=10.0,
    )

    # Optimize with balanced strategy
    optimal_params, diagnostics = optimize_parameters_multi_strategy(
        strategy="balanced",
        landscape=landscape,
        current_params=current_params,
        trajectory_data=None,  # Not needed for balanced
        V_target=0.1,
    )

    assert isinstance(optimal_params, GasParams)
    assert diagnostics["strategy"] == "balanced"
    assert "improvement_ratio" in diagnostics
    assert "kappa_after" in diagnostics
    assert diagnostics["kappa_after"] > 0


def test_optimize_parameters_empirical_strategy(test_simulation):
    """Test empirical strategy optimization (requires trajectory data)."""
    history, _ = test_simulation

    landscape = estimate_landscape_from_history(history)
    current_params = GasParams(
        tau=0.1,
        gamma=1.0,
        sigma_v=1.4,
        lambda_clone=1.0,
        N=30,
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_rest=0.5,
        d_safe=3.0,
        kappa_wall=10.0,
    )
    trajectory_data = extract_trajectory_data_from_history(history)

    # Optimize with empirical strategy
    optimal_params, diagnostics = optimize_parameters_multi_strategy(
        strategy="empirical",
        landscape=landscape,
        current_params=current_params,
        trajectory_data=trajectory_data,
        V_target=0.1,
    )

    assert isinstance(optimal_params, GasParams)
    assert diagnostics["strategy"] == "empirical"


def test_gas_params_from_config():
    """Test GasParams extraction from GasConfig."""
    config = GasConfig(dims=2)
    config.gamma = 1.5
    config.beta = 2.0
    config.delta_t = 0.05
    config.N = 100
    config.sigma_x = 0.2
    config.lambda_alg = 0.8
    config.alpha_restitution = 0.7

    params = gas_params_from_config(config)

    assert isinstance(params, GasParams)
    assert params.gamma == 1.5
    assert params.tau == 0.05
    assert params.N == 100
    assert params.sigma_x == 0.2
    assert params.lambda_alg == 0.8
    assert params.alpha_rest == 0.7

    # Check sigma_v derived correctly
    expected_sigma_v = np.sqrt(2.0 / (1.5 * 2.0))
    assert abs(params.sigma_v - expected_sigma_v) < 1e-6


def test_gas_params_to_config_dict():
    """Test GasParams to GasConfig dict conversion."""
    params = GasParams(
        tau=0.08,
        gamma=1.2,
        sigma_v=1.0,
        lambda_clone=2.0,
        N=150,
        sigma_x=0.15,
        lambda_alg=0.9,
        alpha_rest=0.6,
        d_safe=3.0,
        kappa_wall=10.0,
    )

    config_dict = gas_params_to_config_dict(params, preserve_adaptive=True)

    assert config_dict["delta_t"] == 0.08
    assert config_dict["gamma"] == 1.2
    assert config_dict["N"] == 150
    assert config_dict["sigma_x"] == 0.15
    assert config_dict["lambda_alg"] == 0.9
    assert config_dict["alpha_restitution"] == 0.6

    # Check beta derived correctly
    expected_beta = 2.0 / (1.2 * 1.0**2)
    assert abs(config_dict["beta"] - expected_beta) < 1e-6


def test_apply_gas_params_to_config():
    """Test applying GasParams to GasConfig."""
    config = GasConfig(dims=2)

    # Record initial values
    initial_gamma = config.gamma

    # Create optimized params
    params = GasParams(
        tau=0.08,
        gamma=2.0,  # Different from initial
        sigma_v=1.0,
        lambda_clone=2.0,
        N=200,
        sigma_x=0.25,
        lambda_alg=1.5,
        alpha_rest=0.8,
        d_safe=3.0,
        kappa_wall=10.0,
    )

    # Apply params
    apply_gas_params_to_config(params, config, preserve_adaptive=True)

    # Check values were updated
    assert config.gamma == 2.0
    assert config.gamma != initial_gamma
    assert config.delta_t == 0.08
    assert config.N == 200
    assert config.sigma_x == 0.25
    assert config.lambda_alg == 1.5
    assert config.alpha_restitution == 0.8


def test_optimize_parameters_conservative_strategy(test_simulation):
    """Test conservative strategy produces safer parameters."""
    history, _ = test_simulation

    landscape = estimate_landscape_from_history(history)
    current_params = GasParams(
        tau=0.1,
        gamma=1.0,
        sigma_v=1.4,
        lambda_clone=1.0,
        N=30,
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_rest=0.5,
        d_safe=3.0,
        kappa_wall=10.0,
    )

    # Get balanced params for comparison
    balanced_params, _ = optimize_parameters_multi_strategy(
        strategy="balanced",
        landscape=landscape,
        current_params=current_params,
        trajectory_data=None,
        V_target=0.1,
    )

    # Get conservative params
    conservative_params, _ = optimize_parameters_multi_strategy(
        strategy="conservative",
        landscape=landscape,
        current_params=current_params,
        trajectory_data=None,
        V_target=0.1,
    )

    # Conservative should have smaller timestep (more stable)
    assert conservative_params.tau <= balanced_params.tau

    # Conservative should satisfy stricter stability: γ·τ < 0.4
    assert conservative_params.gamma * conservative_params.tau < 0.4


def test_optimize_parameters_aggressive_strategy(test_simulation):
    """Test aggressive strategy pushes convergence limits."""
    history, _ = test_simulation

    landscape = estimate_landscape_from_history(history)
    current_params = GasParams(
        tau=0.1,
        gamma=1.0,
        sigma_v=1.4,
        lambda_clone=1.0,
        N=30,
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_rest=0.5,
        d_safe=3.0,
        kappa_wall=10.0,
    )

    aggressive_params, diagnostics = optimize_parameters_multi_strategy(
        strategy="aggressive",
        landscape=landscape,
        current_params=current_params,
        trajectory_data=None,
        V_target=0.1,
    )

    # Aggressive should have higher convergence rate
    assert diagnostics["kappa_after"] > 0

    # But still satisfy stability: γ·τ < 0.5
    assert aggressive_params.gamma * aggressive_params.tau < 0.5
