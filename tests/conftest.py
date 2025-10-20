"""Shared pytest fixtures for Euclidean Gas tests."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    SimpleQuadraticPotential,
)
from fragile.core.fitness import FitnessOperator, FitnessParams
from fragile.core.kinetic_operator import KineticOperator, LangevinParams


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
def companion_selection():
    """Standard companion selection strategy."""
    return CompanionSelection(method="uniform")


@pytest.fixture
def cloning_params():
    """Standard cloning parameters."""
    return CloningParams(
        sigma_x=0.1,
        lambda_alg=1.0,
        alpha_restitution=0.5,
    )


@pytest.fixture
def fitness_params():
    """Standard fitness parameters."""
    return FitnessParams(
        alpha=1.0,
        beta=1.0,
        eta=0.1,
        lambda_alg=0.0,
        sigma_min=1e-8,
        A=2.0,
    )


@pytest.fixture
def fitness_op(fitness_params, companion_selection):
    """Standard fitness operator."""
    return FitnessOperator(params=fitness_params, companion_selection=companion_selection)


@pytest.fixture
def kinetic_op(langevin_params, simple_potential, torch_dtype):
    """Standard kinetic operator."""
    return KineticOperator(
        gamma=langevin_params.gamma,
        beta=langevin_params.beta,
        delta_t=langevin_params.delta_t,
        integrator=langevin_params.integrator,
        potential=simple_potential,
        device=torch.device("cpu"),
        dtype=torch_dtype,
    )


@pytest.fixture
def euclidean_gas(
    simple_potential,
    kinetic_op,
    cloning_params,
    fitness_op,
    companion_selection,
    device,
    dtype,
):
    """Complete Euclidean Gas instance."""
    return EuclideanGas(
        N=10,
        d=2,
        companion_selection=companion_selection,
        potential=simple_potential,
        kinetic_op=kinetic_op,
        cloning=cloning_params,
        fitness_op=fitness_op,
        device=torch.device(device),
        dtype=dtype,
    )


@pytest.fixture
def small_swarm_gas(
    simple_potential,
    kinetic_op,
    cloning_params,
    fitness_op,
    companion_selection,
    device,
    dtype,
):
    """Small swarm for quick tests."""
    return EuclideanGas(
        N=5,
        d=2,
        companion_selection=companion_selection,
        potential=simple_potential,
        kinetic_op=kinetic_op,
        cloning=cloning_params,
        fitness_op=fitness_op,
        device=torch.device(device),
        dtype=dtype,
    )


@pytest.fixture
def large_swarm_gas(
    simple_potential,
    kinetic_op,
    cloning_params,
    fitness_op,
    companion_selection,
    device,
    dtype,
):
    """Large swarm for convergence tests."""
    return EuclideanGas(
        N=100,
        d=3,
        companion_selection=companion_selection,
        potential=simple_potential,
        kinetic_op=kinetic_op,
        cloning=cloning_params,
        fitness_op=fitness_op,
        device=torch.device(device),
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
def adaptive_params(euclidean_gas):
    """Standard adaptive gas parameters."""
    from fragile.adaptive_gas import AdaptiveGasParams, AdaptiveParams

    return AdaptiveGasParams(
        euclidean=euclidean_gas,
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
