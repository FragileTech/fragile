"""
Example configurations for Euclidean Gas experiments.

This module provides factory functions for creating common experimental setups.
"""

from __future__ import annotations

import torch

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
)


def default_euclidean_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create Euclidean Gas with default parameters.

    Default values chosen to demonstrate convergence:
    - Langevin: γ=1.0, β=1.0, Δt=0.1
    - Cloning: σ_x=0.1, λ_alg=1.0, e=0.5
    - Potential: Simple quadratic U(x) = 0.5 ||x||²

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device (cpu/cuda)
        dtype: PyTorch dtype (float32/float64)

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def high_temperature_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create high-temperature Euclidean Gas (β=0.1).

    Higher temperature means more thermal noise, slower convergence.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=0.1, delta_t=0.1),  # High temperature
        cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def low_friction_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create low-friction Euclidean Gas (γ=0.1).

    Lower friction means velocities persist longer, more inertia.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=0.1, beta=1.0, delta_t=0.1),  # Low friction
        cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.5),
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def elastic_collision_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create Euclidean Gas with elastic collisions (e=1.0).

    Elastic collisions preserve kinetic energy during cloning.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=1.0),  # Elastic
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def perfectly_inelastic_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create Euclidean Gas with perfectly inelastic collisions (e=0.0).

    Perfectly inelastic collisions: particles stick together after collision.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(
            sigma_x=0.1, lambda_alg=1.0, alpha_restitution=0.0
        ),  # Perfectly inelastic
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def strong_velocity_coupling_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create Euclidean Gas with strong velocity coupling (λ_alg=10.0).

    Higher λ_alg means algorithmic distance is dominated by velocity differences.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(
            sigma_x=0.1, lambda_alg=10.0, alpha_restitution=0.5
        ),  # Strong velocity coupling
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def weak_velocity_coupling_gas(
    N: int = 100,
    d: int = 2,
    device: str = "cpu",
    dtype: str = "float32",
) -> EuclideanGas:
    """
    Create Euclidean Gas with weak velocity coupling (λ_alg=0.1).

    Lower λ_alg means algorithmic distance is dominated by position differences.

    Args:
        N: Number of walkers
        d: Spatial dimension
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Configured EuclideanGas instance
    """
    params = EuclideanGasParams(
        N=N,
        d=d,
        potential=SimpleQuadraticPotential(),
        langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
        cloning=CloningParams(
            sigma_x=0.1, lambda_alg=0.1, alpha_restitution=0.5
        ),  # Weak velocity coupling
        device=device,
        dtype=dtype,
    )
    return EuclideanGas(params)


def run_simple_experiment(
    gas: EuclideanGas,
    n_steps: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Run a simple experiment with the given Euclidean Gas.

    Args:
        gas: Configured EuclideanGas instance
        n_steps: Number of steps to run
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary with trajectory data
    """
    if seed is not None:
        torch.manual_seed(seed)

    return gas.run(n_steps)


def compare_restitution_coefficients(
    N: int = 100,
    d: int = 2,
    n_steps: int = 1000,
    device: str = "cpu",
    dtype: str = "float32",
) -> dict:
    """
    Compare Euclidean Gas behavior for different restitution coefficients.

    Args:
        N: Number of walkers
        d: Spatial dimension
        n_steps: Number of steps
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Dictionary mapping e values to trajectory data
    """
    results = {}

    for e in [0.0, 0.25, 0.5, 0.75, 1.0]:
        params = EuclideanGasParams(
            N=N,
            d=d,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=1.0, alpha_restitution=e),
            device=device,
            dtype=dtype,
        )
        gas = EuclideanGas(params)
        results[f"alpha_restitution={e}"] = gas.run(n_steps)

    return results


def compare_velocity_coupling(
    N: int = 100,
    d: int = 2,
    n_steps: int = 1000,
    device: str = "cpu",
    dtype: str = "float32",
) -> dict:
    """
    Compare Euclidean Gas behavior for different velocity coupling strengths.

    Args:
        N: Number of walkers
        d: Spatial dimension
        n_steps: Number of steps
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Dictionary mapping λ_alg values to trajectory data
    """
    results = {}

    for lambda_alg in [0.1, 0.5, 1.0, 5.0, 10.0]:
        params = EuclideanGasParams(
            N=N,
            d=d,
            potential=SimpleQuadraticPotential(),
            langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.1),
            cloning=CloningParams(sigma_x=0.1, lambda_alg=lambda_alg, alpha_restitution=0.5),
            device=device,
            dtype=dtype,
        )
        gas = EuclideanGas(params)
        results[f"lambda_alg={lambda_alg}"] = gas.run(n_steps)

    return results
