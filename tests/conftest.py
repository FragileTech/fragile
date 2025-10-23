"""Shared pytest fixtures for Euclidean Gas tests."""

import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import OptimBenchmark
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas
from fragile.core.fitness import FitnessOperator
from fragile.core.kinetic_operator import KineticOperator


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

    def quadratic(x):
        return 0.5 * torch.sum(x**2, dim=-1)

    bounds = TorchBounds(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]))

    return OptimBenchmark(dims=2, function=quadratic, bounds=bounds)


@pytest.fixture
def companion_selection():
    """Standard companion selection strategy."""
    return CompanionSelection(method="uniform")


@pytest.fixture
def clone_op():
    """Standard clone operator."""
    return CloneOperator(
        sigma_x=0.1,
        alpha_restitution=0.5,
        p_max=1.0,
        epsilon_clone=0.01,
    )


@pytest.fixture
def fitness_op():
    """Standard fitness operator."""
    return FitnessOperator(
        alpha=1.0,
        beta=1.0,
        eta=0.1,
        lambda_alg=0.0,
        sigma_min=1e-8,
        A=2.0,
    )


@pytest.fixture
def kinetic_op(simple_potential, torch_dtype):
    """Standard kinetic operator."""
    return KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        integrator="baoab",
        potential=simple_potential,
        device=torch.device("cpu"),
        dtype=torch_dtype,
    )


@pytest.fixture
def euclidean_gas(
    simple_potential,
    kinetic_op,
    clone_op,
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
        cloning=clone_op,
        fitness_op=fitness_op,
        device=torch.device(device),
        dtype=dtype,
    )


@pytest.fixture
def small_swarm_gas(
    simple_potential,
    kinetic_op,
    clone_op,
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
        cloning=clone_op,
        fitness_op=fitness_op,
        device=torch.device(device),
        dtype=dtype,
    )


@pytest.fixture
def large_swarm_gas(
    simple_potential,
    kinetic_op,
    clone_op,
    fitness_op,
    companion_selection,
    device,
    dtype,
):
    """Large swarm for convergence tests."""
    # Provide explicit 3D bounds since simple_potential has 2D bounds
    bounds_3d = TorchBounds(
        low=torch.tensor([-5.0, -5.0, -5.0]), high=torch.tensor([5.0, 5.0, 5.0])
    )

    return EuclideanGas(
        N=100,
        d=3,
        companion_selection=companion_selection,
        potential=simple_potential,
        kinetic_op=kinetic_op,
        cloning=clone_op,
        fitness_op=fitness_op,
        bounds=bounds_3d,  # Override auto-extracted bounds
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
