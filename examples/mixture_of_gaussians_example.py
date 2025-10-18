"""
Example: Using MixtureOfGaussians benchmark with Geometric Gas

This script demonstrates how to use the MixtureOfGaussians benchmark
to test optimization algorithms like the Geometric Gas.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from fragile.benchmarks import MixtureOfGaussians


def visualize_mixture_2d(mixture: MixtureOfGaussians, n_samples: int = 1000):
    """Visualize a 2D Mixture of Gaussians."""
    if mixture.dims != 2:
        print(f"Visualization only works for 2D mixtures (got dims={mixture.dims})")
        return None

    # Get bounds
    low, high = mixture.bounds_range

    # Create grid for contour plot
    x = np.linspace(low, high, 100)
    y = np.linspace(low, high, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on grid
    grid_points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    Z = mixture(grid_points).numpy().reshape(X.shape)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Contour plot with component centers
    contour = ax1.contourf(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax1, label="Negative Log-Likelihood")

    # Plot Gaussian centers
    centers = mixture.centers.numpy()
    weights = mixture.weights.numpy()

    # Size markers by weight
    sizes = weights * 500
    ax1.scatter(
        centers[:, 0],
        centers[:, 1],
        s=sizes,
        c="red",
        marker="*",
        edgecolors="white",
        linewidths=2,
        label="Gaussian Centers",
        zorder=5,
    )

    # Highlight best center
    best_idx = torch.argmax(mixture.weights)
    ax1.scatter(
        centers[best_idx, 0],
        centers[best_idx, 1],
        s=800,
        c="lime",
        marker="*",
        edgecolors="white",
        linewidths=3,
        label="Best Center",
        zorder=6,
    )

    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("Mixture of Gaussians Landscape")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Component information
    ax2.axis("off")
    info = mixture.get_component_info()

    info_text = f"""
    Mixture of Gaussians Information
    ═══════════════════════════════════

    Dimensions: {info["dims"]}
    Number of Components: {info["n_gaussians"]}
    Bounds: [{low:.1f}, {high:.1f}]

    Component Details:
    ─────────────────────────────────
    """

    for i in range(info["n_gaussians"]):
        center = info["centers"][i].numpy()
        std = info["stds"][i].numpy()
        weight = info["weights"][i].item()

        marker = "★" if i == best_idx else "○"
        info_text += f"""
    {marker} Component {i + 1}:
      Center: [{center[0]:.2f}, {center[1]:.2f}]
      Std:    [{std[0]:.2f}, {std[1]:.2f}]
      Weight: {weight:.3f}
    """

    info_text += f"""
    ─────────────────────────────────
    Best State: {mixture.best_state.numpy()}
    Optimal Value: {mixture.benchmark.item():.4f}
    """

    ax2.text(
        0.1,
        0.5,
        info_text,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="center",
        transform=ax2.transAxes,
    )

    plt.tight_layout()
    return fig


def example_random_mixture():
    """Example 1: Random Mixture of Gaussians."""
    print("=" * 60)
    print("Example 1: Random Mixture of Gaussians")
    print("=" * 60)

    # Create a random mixture
    mixture = MixtureOfGaussians(dims=2, n_gaussians=4, bounds_range=(-8.0, 8.0), seed=42)

    print(f"Created mixture with {mixture.n_gaussians} components in {mixture.dims}D")
    print(f"Best state: {mixture.best_state}")
    print(f"Optimal value: {mixture.benchmark:.4f}")

    # Visualize
    visualize_mixture_2d(mixture, n_samples=1000)
    plt.savefig("/tmp/random_mixture.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to /tmp/random_mixture.png")
    plt.close()
    print()


def example_custom_mixture():
    """Example 2: Custom Mixture Configuration."""
    print("=" * 60)
    print("Example 2: Custom Mixture Configuration")
    print("=" * 60)

    # Define a specific mixture with known structure
    centers = torch.tensor([
        [0.0, 0.0],  # High-weighted peak at origin
        [4.0, 4.0],  # Medium-weighted peak
        [-3.0, 3.0],  # Low-weighted peak
    ])

    stds = torch.tensor([
        [0.8, 0.8],  # Tight peak
        [1.5, 1.5],  # Wider peak
        [1.0, 2.0],  # Anisotropic peak
    ])

    weights = torch.tensor([0.6, 0.3, 0.1])  # Decreasing weights

    mixture = MixtureOfGaussians(
        dims=2,
        n_gaussians=3,
        centers=centers,
        stds=stds,
        weights=weights,
        bounds_range=(-6.0, 6.0),
    )

    print("Created custom mixture:")
    print(f"  - Centers: {centers.tolist()}")
    print(f"  - Weights: {weights.tolist()}")
    print(f"  - Best state: {mixture.best_state}")
    print(f"  - Optimal value: {mixture.benchmark:.4f}")

    # Visualize
    visualize_mixture_2d(mixture)
    plt.savefig("/tmp/custom_mixture.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to /tmp/custom_mixture.png")
    plt.close()
    print()


def example_optimization_trajectory():
    """Example 3: Visualize optimization trajectory on mixture."""
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
        bounds_range=(-5.0, 5.0),
    )

    # Run simple gradient descent
    x_init = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    trajectory = [x_init.detach().clone()]

    learning_rate = 0.1
    n_steps = 50

    x = x_init.clone().requires_grad_(True)
    for step in range(n_steps):
        # Compute negative log-likelihood
        value = mixture(x)

        # Gradient descent step
        value.backward()
        with torch.no_grad():
            x -= learning_rate * x.grad
            trajectory.append(x.clone())

        x.grad.zero_()

    trajectory = torch.cat(trajectory, dim=0).numpy()

    print(f"Starting position: {trajectory[0]}")
    print(f"Final position: {trajectory[-1]}")
    print(f"Best state: {mixture.best_state.numpy()}")
    print(f"Distance to best: {np.linalg.norm(trajectory[-1] - mixture.best_state.numpy()):.4f}")

    # Visualize trajectory
    fig = visualize_mixture_2d(mixture)
    ax = fig.axes[0]

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], "r-", linewidth=2, alpha=0.7, label="Trajectory")
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=200,
        c="blue",
        marker="o",
        edgecolors="white",
        linewidths=2,
        label="Start",
        zorder=7,
    )
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        s=200,
        c="red",
        marker="X",
        edgecolors="white",
        linewidths=2,
        label="End",
        zorder=7,
    )

    ax.legend()

    plt.savefig("/tmp/optimization_trajectory.png", dpi=150, bbox_inches="tight")
    print("Saved trajectory visualization to /tmp/optimization_trajectory.png")
    plt.close()
    print()


def example_high_dimensional_mixture():
    """Example 4: High-dimensional mixture."""
    print("=" * 60)
    print("Example 4: High-Dimensional Mixture")
    print("=" * 60)

    dims = 10
    n_gaussians = 5

    mixture = MixtureOfGaussians(dims=dims, n_gaussians=n_gaussians, seed=123)

    print(f"Created {dims}D mixture with {n_gaussians} components")

    # Evaluate at random points
    n_samples = 100
    samples = mixture.sample(n_samples)
    values = mixture(samples)

    print(f"Evaluated {n_samples} random points:")
    print(f"  Mean value: {values.mean():.4f}")
    print(f"  Std value: {values.std():.4f}")
    print(f"  Min value: {values.min():.4f}")
    print(f"  Max value: {values.max():.4f}")
    print(f"  Optimal value: {mixture.benchmark:.4f}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_random_mixture()
    example_custom_mixture()
    example_optimization_trajectory()
    example_high_dimensional_mixture()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
