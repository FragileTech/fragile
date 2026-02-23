"""
Dataset generation utilities for dimensionality reduction experiments.

This module provides synthetic datasets for testing atlas-based encoders:
- Manifold Mixture (2D): Swiss Roll + Circles + Moons
- Nightmare Dataset (3D): Swiss Roll + Sphere + Moons
- MNIST: High-dimensional digit images

Also includes visualization utilities for boundary detection.
"""

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.neighbors import NearestNeighbors
import torch


# ==========================================
# 2D DATASETS
# ==========================================


def get_manifold_mixture_data(
    n_per_manifold: int = 1000,
    seed: int | None = 42,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Generate the Manifold Mixture dataset (2D).

    Creates three distinct geometric shapes:
    - Swiss Roll (projected to 2D, normalized)
    - Circles (shifted right)
    - Moons (shifted up)

    Args:
        n_per_manifold: Number of samples per manifold
        seed: Random seed for reproducibility

    Returns:
        X: [N, 2] tensor of 2D points
        labels: [N] numpy array of manifold labels (0=roll, 1=circles, 2=moons)
        colors: [N] numpy array of [0, 1] values for rainbow colormap
    """
    # Manifold 1: Swiss Roll (project to 2D)
    d1, t_roll = make_swiss_roll(n_per_manifold, noise=0.1, random_state=seed)
    d1 = d1[:, [0, 2]] / 10.0  # Project and normalize
    labels_1 = np.zeros(n_per_manifold)
    # Normalize t_roll to [0, 0.33] for first third of colormap
    colors_1 = (t_roll - t_roll.min()) / (t_roll.max() - t_roll.min() + 1e-6) * 0.33

    # Manifold 2: Circles (topological loop)
    d2, _ = make_circles(n_per_manifold, factor=0.5, noise=0.05, random_state=seed)
    d2 = d2 * 1.5 + np.array([3.0, 0.0])  # Shift right
    labels_2 = np.ones(n_per_manifold)
    # [0.33, 0.66] range for circles
    angles = np.arctan2(d2[:, 1] - 0.0, d2[:, 0] - 3.0)
    colors_2 = 0.33 + (angles + np.pi) / (2 * np.pi) * 0.33

    # Manifold 3: Moons (discontinuous clusters)
    d3, moon_labels = make_moons(n_per_manifold, noise=0.05, random_state=seed)
    d3 = d3 * 1.5 + np.array([0.0, 3.0])  # Shift up
    labels_3 = np.full(n_per_manifold, 2)
    # [0.66, 1.0] range - map moon identity
    colors_3 = 0.66 + moon_labels * 0.17 + 0.08

    # Combine
    X = np.vstack([d1, d2, d3]).astype(np.float32)
    labels = np.concatenate([labels_1, labels_2, labels_3])
    colors = np.concatenate([colors_1, colors_2, colors_3])

    return torch.from_numpy(X), labels, colors


# ==========================================
# 3D DATASETS
# ==========================================


def get_nightmare_data(
    n_samples: int = 3000,
    seed: int | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Generate the 'nightmare dataset': Swiss Roll + Sphere + Moons (3D).

    This is a challenging dataset for atlas-based methods because it combines
    manifolds with different intrinsic dimensions and topologies.

    Args:
        n_samples: Total number of samples (divided by 3 per manifold)
        seed: Random seed for reproducibility

    Returns:
        X: (n_samples, 3) tensor of 3D coordinates
        labels: (n_samples,) array with dataset identity (0=roll, 1=sphere, 2=moons)
        colors: (n_samples,) array with continuous values for rainbow coloring
    """
    if seed is not None:
        np.random.seed(seed)

    n_per = n_samples // 3

    # Swiss Roll - use t parameter for rainbow coloring along the roll
    X1, t_roll = make_swiss_roll(n_per, noise=0.05, random_state=seed)
    X1 = (X1 - X1.mean(0)) / (X1.std(0) + 1e-6)
    X1[:, 0] -= 4.0
    labels1 = np.zeros(n_per)
    # Normalize t_roll to [0, 0.33] for first third of colormap
    colors1 = (t_roll - t_roll.min()) / (t_roll.max() - t_roll.min() + 1e-6) * 0.33

    # Sphere - use latitude (theta) for coloring
    phi = np.random.uniform(0, 2 * np.pi, n_per)
    costheta = np.random.uniform(-1, 1, n_per)
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z_coord = np.cos(theta)
    X2 = np.stack([x, y, z_coord], axis=1)
    labels2 = np.ones(n_per)
    # Normalize theta to [0.33, 0.66] for second third of colormap
    colors2 = 0.33 + (theta / np.pi) * 0.33

    # Moons - use moon identity (0 or 1) for distinct colors
    X3_2d, moon_labels = make_moons(n_per, noise=0.05, random_state=seed)
    X3 = np.zeros((n_per, 3))
    X3[:, 0] = X3_2d[:, 0]
    X3[:, 1] = X3_2d[:, 1]
    X3 = (X3 - X3.mean(0)) / (X3.std(0) + 1e-6)
    X3[:, 0] += 4.0
    labels3 = np.full(n_per, 2)
    # Map moon 0 → 0.72, moon 1 → 0.88 (distinct colors in last third)
    colors3 = 0.66 + moon_labels * 0.17 + 0.08

    X = np.vstack([X1, X2, X3])
    labels = np.concatenate([labels1, labels2, labels3])
    colors = np.concatenate([colors1, colors2, colors3])

    return torch.FloatTensor(X), labels, colors


# ==========================================
# HIGH-DIMENSIONAL DATASETS
# ==========================================


def get_mnist_data(
    n_samples: int = 10000,
    root: str = "/tmp/mnist",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load MNIST dataset for atlas embedding.

    Args:
        n_samples: Number of samples to load
        root: Directory to download/cache MNIST

    Returns:
        X: Tensor of shape (n_samples, 784) with pixel values in [0, 1]
        labels: Array of digit labels (0-9)
        colors: Array of [0, 1] values for rainbow colormap (digit/9)
    """
    from torchvision import datasets

    # Download/load MNIST
    mnist = datasets.MNIST(root=root, train=True, download=True)

    # Flatten to (N, 784), normalize to [0, 1]
    X = mnist.data.float().view(-1, 784) / 255.0
    labels = mnist.targets.numpy()

    # Subsample if needed
    if n_samples < len(X):
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        labels = labels[idx]

    # Colors: map digit class (0-9) to [0, 1] range for rainbow colormap
    colors = labels / 9.0  # 0→0.0, 9→1.0

    return X, labels, colors


def get_fashion_mnist_data(
    n_samples: int = 10000,
    root: str = "/tmp/fashion_mnist",
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST dataset for atlas embedding.

    Args:
        n_samples: Number of samples to load
        root: Directory to download/cache Fashion-MNIST

    Returns:
        X: Tensor of shape (n_samples, 784) with pixel values in [0, 1]
        labels: Array of class labels (0-9)
        colors: Array of [0, 1] values for rainbow colormap (class/9)
    """
    from torchvision import datasets

    fmnist = datasets.FashionMNIST(root=root, train=True, download=True)

    X = fmnist.data.float().view(-1, 784) / 255.0
    labels = fmnist.targets.numpy()

    if n_samples < len(X):
        idx = np.random.choice(len(X), n_samples, replace=False)
        X = X[idx]
        labels = labels[idx]

    colors = labels / 9.0

    return X, labels, colors


# ==========================================
# VISUALIZATION UTILITIES
# ==========================================


def find_boundary_pairs(
    z: np.ndarray,
    hard_assign: np.ndarray,
    X: np.ndarray,
    k: int = 5,
    max_latent_dist: float = 2.0,
) -> list[tuple[int, int]]:
    """Find pairs of points close in input space but in different charts.

    These boundary pairs are useful for visualizing chart transitions
    in the portal view.

    Args:
        z: Latent coordinates [N, D]
        hard_assign: Hard chart assignments [N]
        X: Input coordinates [N, D_in]
        k: Number of neighbors to consider
        max_latent_dist: Maximum latent distance to include pair

    Returns:
        List of (i, j) index pairs straddling chart boundaries
    """
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    pairs = []
    for i in range(len(X)):
        k_i = hard_assign[i]
        for j in indices[i, 1:]:
            k_j = hard_assign[j]
            if k_i != k_j:
                latent_dist = np.linalg.norm(z[i] - z[j])
                if latent_dist < max_latent_dist:
                    pairs.append((i, j))
    return pairs


def compute_chart_colors(
    assignment: np.ndarray,
    num_charts: int,
) -> np.ndarray:
    """Compute soft-blended RGB colors from chart assignment probabilities.

    Args:
        assignment: Soft assignment probabilities [N, K]
        num_charts: Number of charts

    Returns:
        RGB colors [N, 3] in [0, 1]
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    chart_colors = cmap(np.linspace(0, 1, num_charts))[:, :3]

    blended = np.zeros((len(assignment), 3))
    for i in range(num_charts):
        blended += assignment[:, i : i + 1] * chart_colors[i : i + 1]
    return np.clip(blended, 0, 1)
