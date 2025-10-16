"""Shared pytest fixtures for Euclidean Gas tests."""

import pytest
import torch

from fragile.bounds import Bounds
from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)


# Trigger Pydantic model rebuild to resolve forward references
EuclideanGasParams.model_rebuild()


@pytest.fixture
def device():
    """Default device for tests."""
    return "cpu"


@pytest.fixture
def dtype():
    """Default dtype for tests."""
    return "float64"


@pytest.fixture
def torch_dtype():
    """Torch dtype object."""
    return torch.float64


@pytest.fixture
def simple_potential():
    """Simple quadratic potential U(x) = 0.5 * ||x||^2."""
    return SimpleQuadraticPotential()


@pytest.fixture
def langevin_params():
    """Standard Langevin parameters."""
    return LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01, integrator="baoab")


@pytest.fixture
def cloning_params():
    """Standard cloning parameters."""
    return CloningParams(
        sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5, use_inelastic_collision=True
    )


@pytest.fixture
def euclidean_gas_params(simple_potential, langevin_params, cloning_params, device, dtype):
    """Complete Euclidean Gas parameters."""
    return EuclideanGasParams(
        N=10,
        d=2,
        potential=simple_potential,
        langevin=langevin_params,
        cloning=cloning_params,
        device=device,
        dtype=dtype,
    )


@pytest.fixture
def small_swarm_params(simple_potential, langevin_params, cloning_params, device, dtype):
    """Small swarm for quick tests."""
    return EuclideanGasParams(
        N=5,
        d=2,
        potential=simple_potential,
        langevin=langevin_params,
        cloning=cloning_params,
        device=device,
        dtype=dtype,
    )


@pytest.fixture
def large_swarm_params(simple_potential, langevin_params, cloning_params, device, dtype):
    """Large swarm for convergence tests."""
    return EuclideanGasParams(
        N=100,
        d=3,
        potential=simple_potential,
        langevin=langevin_params,
        cloning=cloning_params,
        device=device,
        dtype=dtype,
    )


@pytest.fixture(params=["cpu", "cuda"])
def test_device(request):
    """Parametrized device fixture for cross-device testing.

    Tests will run on both CPU and CUDA (if available).
    CUDA tests are automatically skipped if CUDA is not available.
    """
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return device


@pytest.fixture(params=["float32", "float64"])
def test_dtype(request):
    """Parametrized dtype fixture for precision testing."""
    return request.param


@pytest.fixture
def adaptive_params(euclidean_gas_params):
    """Standard adaptive gas parameters."""
    from fragile.adaptive_gas import AdaptiveGasParams, AdaptiveParams

    return AdaptiveGasParams(
        euclidean=euclidean_gas_params,
        adaptive=AdaptiveParams(
            epsilon_F=0.1,
            nu=0.05,
            epsilon_Sigma=2.0,
            A=1.0,
            sigma_prime_min_patch=0.1,
            patch_radius=1.0,
            l_viscous=0.5,
            use_adaptive_diffusion=True,
        ),
        measurement_fn="potential",
    )
