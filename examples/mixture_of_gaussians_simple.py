"""
Simple Example: Using MixtureOfGaussians benchmark

This script demonstrates basic usage of the MixtureOfGaussians benchmark
without requiring visualization.
"""

import torch
import numpy as np

from fragile.benchmarks import MixtureOfGaussians


def example_random_mixture():
    """Example 1: Random Mixture of Gaussians."""
    print("=" * 60)
    print("Example 1: Random Mixture of Gaussians")
    print("=" * 60)

    # Create a random mixture
    mixture = MixtureOfGaussians(
        dims=2,
        n_gaussians=4,
        bounds_range=(-8.0, 8.0),
        seed=42
    )

    print(f"Created mixture with {mixture.n_gaussians} components in {mixture.dims}D")
    print(f"Best state: {mixture.best_state}")
    print(f"Optimal value: {mixture.benchmark:.4f}")

    # Evaluate at some points
    test_points = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [-2.0, 2.0],
        mixture.best_state.numpy()  # Best point
    ])
    values = mixture(test_points)

    print("\nEvaluation at test points:")
    for i, (point, value) in enumerate(zip(test_points, values)):
        print(f"  Point {i+1}: {point.numpy()} -> Value: {value:.4f}")

    print()


def example_custom_mixture():
    """Example 2: Custom Mixture Configuration."""
    print("=" * 60)
    print("Example 2: Custom Mixture Configuration")
    print("=" * 60)

    # Define a specific mixture with known structure
    centers = torch.tensor([
        [0.0, 0.0],    # High-weighted peak at origin
        [4.0, 4.0],    # Medium-weighted peak
        [-3.0, 3.0],   # Low-weighted peak
    ])

    stds = torch.tensor([
        [0.8, 0.8],    # Tight peak
        [1.5, 1.5],    # Wider peak
        [1.0, 2.0],    # Anisotropic peak
    ])

    weights = torch.tensor([0.6, 0.3, 0.1])  # Decreasing weights

    mixture = MixtureOfGaussians(
        dims=2,
        n_gaussians=3,
        centers=centers,
        stds=stds,
        weights=weights,
        bounds_range=(-6.0, 6.0)
    )

    print("Created custom mixture:")
    print(f"  Centers shape: {centers.shape}")
    print(f"  Weights: {weights.tolist()}")
    print(f"  Best state: {mixture.best_state}")
    print(f"  Optimal value: {mixture.benchmark:.4f}")

    # Get component info
    info = mixture.get_component_info()
    print("\nComponent Information:")
    for i in range(info['n_gaussians']):
        center = info['centers'][i]
        std = info['stds'][i]
        weight = info['weights'][i]
        print(f"  Component {i+1}:")
        print(f"    Center: {center.numpy()}")
        print(f"    Std:    {std.numpy()}")
        print(f"    Weight: {weight:.3f}")

    print()


def example_optimization():
    """Example 3: Simple Gradient Descent on Mixture."""
    print("=" * 60)
    print("Example 3: Simple Gradient Descent on Mixture")
    print("=" * 60)

    # Create mixture
    centers = torch.tensor([[0.0, 0.0], [3.0, 3.0]])
    stds = torch.ones(2, 2) * 0.8
    weights = torch.tensor([0.7, 0.3])

    mixture = MixtureOfGaussians(
        dims=2,
        n_gaussians=2,
        centers=centers,
        stds=stds,
        weights=weights,
        bounds_range=(-5.0, 5.0)
    )

    # Run simple gradient descent
    x = torch.tensor([[-4.0, 4.0]], requires_grad=True)

    learning_rate = 0.1
    n_steps = 50

    print(f"Starting position: {x.detach().numpy()}")
    print(f"Target (best state): {mixture.best_state.numpy()}")

    # Optimization loop
    for step in range(n_steps):
        # Compute negative log-likelihood
        value = mixture(x)

        # Gradient descent step
        value.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad

        x.grad.zero_()

        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: position = {x.detach().numpy()}, value = {value.item():.4f}")

    final_pos = x.detach().numpy()
    best_pos = mixture.best_state.numpy()
    distance = np.linalg.norm(final_pos - best_pos)

    print(f"\nFinal position: {final_pos}")
    print(f"Distance to best: {distance:.4f}")
    print(f"Final value: {mixture(x).item():.4f}")
    print(f"Optimal value: {mixture.benchmark.item():.4f}")
    print()


def example_high_dimensional():
    """Example 4: High-dimensional mixture."""
    print("=" * 60)
    print("Example 4: High-Dimensional Mixture")
    print("=" * 60)

    dims = 10
    n_gaussians = 5

    mixture = MixtureOfGaussians(
        dims=dims,
        n_gaussians=n_gaussians,
        seed=123
    )

    print(f"Created {dims}D mixture with {n_gaussians} components")

    # Evaluate at random points
    n_samples = 100
    samples = mixture.sample(n_samples)
    values = mixture(samples)

    print(f"\nEvaluated {n_samples} random points:")
    print(f"  Mean value: {values.mean():.4f}")
    print(f"  Std value: {values.std():.4f}")
    print(f"  Min value: {values.min():.4f}")
    print(f"  Max value: {values.max():.4f}")
    print(f"  Optimal value: {mixture.benchmark:.4f}")

    # Show dimensionality of best state
    print(f"\nBest state shape: {mixture.best_state.shape}")
    print(f"Best state (first 5 dims): {mixture.best_state[:5].numpy()}")
    print()


def example_gradient_flow():
    """Example 5: Demonstrate gradient availability."""
    print("=" * 60)
    print("Example 5: Gradient Computation")
    print("=" * 60)

    mixture = MixtureOfGaussians(dims=2, n_gaussians=3, seed=42)

    # Create test point with gradient tracking
    x = torch.randn(1, 2, requires_grad=True)

    # Evaluate function
    value = mixture(x)

    # Compute gradient
    value.backward()

    print(f"Input position: {x.detach().numpy()}")
    print(f"Function value: {value.item():.4f}")
    print(f"Gradient: {x.grad.numpy()}")
    print(f"Gradient norm: {torch.norm(x.grad).item():.4f}")
    print()


def example_reproducibility():
    """Example 6: Demonstrate reproducibility with seeds."""
    print("=" * 60)
    print("Example 6: Reproducibility with Seeds")
    print("=" * 60)

    # Create two mixtures with same seed
    mixture1 = MixtureOfGaussians(dims=3, n_gaussians=4, seed=999)
    mixture2 = MixtureOfGaussians(dims=3, n_gaussians=4, seed=999)

    # They should be identical
    centers_match = torch.allclose(mixture1.centers, mixture2.centers)
    stds_match = torch.allclose(mixture1.stds, mixture2.stds)
    weights_match = torch.allclose(mixture1.weights, mixture2.weights)

    print(f"Same seed produces identical mixtures:")
    print(f"  Centers match: {centers_match}")
    print(f"  Stds match: {stds_match}")
    print(f"  Weights match: {weights_match}")

    # Different seed should produce different mixture
    mixture3 = MixtureOfGaussians(dims=3, n_gaussians=4, seed=111)
    centers_differ = not torch.allclose(mixture1.centers, mixture3.centers)

    print(f"  Different seeds produce different centers: {centers_differ}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_random_mixture()
    example_custom_mixture()
    example_optimization()
    example_high_dimensional()
    example_gradient_flow()
    example_reproducibility()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
