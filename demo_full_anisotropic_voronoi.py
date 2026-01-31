#!/usr/bin/env python3
"""
Demonstration of full anisotropic Voronoi proxy diffusion.

This script shows:
1. How to enable full anisotropic tensors
2. Comparison between diagonal and full modes
3. Visualization of the tensor differences
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.fragile.fractalai.qft.voronoi_observables import (
    compute_voronoi_diffusion_tensor,
    compute_voronoi_tessellation,
)


def plot_diffusion_ellipses(positions, sigma_tensors, title="Diffusion Tensors", ax=None):
    """
    Plot diffusion tensors as ellipses overlaid on positions.

    For a 2D tensor [[a, b], [b, c]], the ellipse shows the diffusion anisotropy.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot points
    ax.scatter(positions[:, 0], positions[:, 1], c='black', s=50, zorder=3, alpha=0.5)

    # Plot diffusion ellipses
    for i, (pos, sigma) in enumerate(zip(positions, sigma_tensors)):
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # Ellipse parameters
        # Width and height are proportional to sqrt(eigenvalues)
        width = 2 * np.sqrt(eigenvalues[0])
        height = 2 * np.sqrt(eigenvalues[1])

        # Angle of rotation (from eigenvectors)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

        # Create ellipse
        ellipse = Ellipse(
            xy=pos,
            width=width,
            height=height,
            angle=angle,
            facecolor='blue',
            alpha=0.2,
            edgecolor='blue',
            linewidth=1.5,
        )
        ax.add_patch(ellipse)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)

    return ax


def demo_comparison():
    """Compare diagonal vs full anisotropic modes."""
    print("=" * 70)
    print("DEMONSTRATION: Diagonal vs Full Anisotropic Voronoi Proxy")
    print("=" * 70)

    # Create positions with anisotropic structure
    # Positions along a rotated ellipse to create elongated Voronoi cells
    np.random.seed(42)
    N = 40

    # Ellipse parameters
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    a, b = 4.0, 1.5  # Semi-major and semi-minor axes
    angle = np.pi / 6  # 30 degrees rotation

    x_ellipse = a * np.cos(theta)
    y_ellipse = b * np.sin(theta)

    # Rotate
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = cos_a * x_ellipse - sin_a * y_ellipse
    y = sin_a * x_ellipse + cos_a * y_ellipse

    # Add noise
    x += np.random.randn(N) * 0.2
    y += np.random.randn(N) * 0.2

    positions_np = np.column_stack([x, y])
    positions = torch.from_numpy(positions_np).float()
    alive = torch.ones(N, dtype=torch.bool)

    print(f"\n✓ Created {N} positions in rotated elliptical pattern")
    print(f"  Rotation angle: {np.degrees(angle):.1f}°")
    print(f"  Aspect ratio: {a/b:.2f}")

    # Compute Voronoi tessellation
    print("\n✓ Computing Voronoi tessellation...")
    voronoi_data = compute_voronoi_tessellation(
        positions=positions,
        alive=alive,
        bounds=None,
        pbc=False,
    )

    # Compute diagonal mode
    print("\n--- DIAGONAL MODE ---")
    sigma_diag = compute_voronoi_diffusion_tensor(
        voronoi_data=voronoi_data,
        positions=positions_np,
        epsilon_sigma=0.1,
        c2=1.0,
        diagonal_only=True,
    )

    print(f"Shape: {sigma_diag.shape}")
    print(f"Mean σ_x: {sigma_diag[:, 0].mean():.4f} ± {sigma_diag[:, 0].std():.4f}")
    print(f"Mean σ_y: {sigma_diag[:, 1].mean():.4f} ± {sigma_diag[:, 1].std():.4f}")
    print(f"Anisotropy ratio (σ_x/σ_y): {(sigma_diag[:, 0]/sigma_diag[:, 1]).mean():.4f}")

    # Convert to full tensor for plotting
    sigma_diag_full = np.zeros((N, 2, 2))
    for i in range(N):
        sigma_diag_full[i] = np.diag(sigma_diag[i])

    # Compute full anisotropic mode
    print("\n--- FULL ANISOTROPIC MODE ---")
    sigma_full = compute_voronoi_diffusion_tensor(
        voronoi_data=voronoi_data,
        positions=positions_np,
        epsilon_sigma=0.1,
        c2=1.0,
        diagonal_only=False,
    )

    print(f"Shape: {sigma_full.shape}")

    # Analyze off-diagonal elements
    off_diag = sigma_full[:, 0, 1]
    num_significant = np.sum(np.abs(off_diag) > 0.01)
    print(f"Cells with significant off-diagonal: {num_significant}/{N} ({100*num_significant/N:.1f}%)")
    print(f"Max |off-diagonal|: {np.abs(off_diag).max():.4f}")
    print(f"Mean |off-diagonal|: {np.abs(off_diag).mean():.4f}")

    # Check eigenvalues
    eigenvalues_all = np.array([np.linalg.eigvalsh(sigma_full[i]) for i in range(N)])
    print(f"Eigenvalue range: [{eigenvalues_all.min():.4f}, {eigenvalues_all.max():.4f}]")

    # Compute anisotropy ratio from eigenvalues
    anisotropy_ratios = eigenvalues_all[:, 1] / eigenvalues_all[:, 0]
    print(f"Anisotropy ratio (λ_max/λ_min): {anisotropy_ratios.mean():.4f} ± {anisotropy_ratios.std():.4f}")

    # Compare with diagonal
    print("\n--- COMPARISON ---")
    diag_from_full = np.array([sigma_full[i, j, j] for i in range(N) for j in range(2)]).reshape(N, 2)
    rel_diff = np.abs(sigma_diag - diag_from_full) / (sigma_diag + 1e-8)
    print(f"Diagonal element difference: {rel_diff.mean():.4f} (mean relative)")

    # Example tensors
    print("\n--- EXAMPLE TENSORS ---")
    idx = num_significant > 0 and np.argmax(np.abs(off_diag)) or 0
    print(f"\nCell {idx} (position: [{positions_np[idx, 0]:.2f}, {positions_np[idx, 1]:.2f}]):")
    print(f"  Diagonal mode:")
    print(f"    [{sigma_diag[idx, 0]:.4f},  0.0000]")
    print(f"    [0.0000,  {sigma_diag[idx, 1]:.4f}]")
    print(f"\n  Full mode:")
    print(f"    [{sigma_full[idx, 0, 0]:7.4f}, {sigma_full[idx, 0, 1]:7.4f}]")
    print(f"    [{sigma_full[idx, 1, 0]:7.4f}, {sigma_full[idx, 1, 1]:7.4f}]")

    # Visualization
    print("\n✓ Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plot_diffusion_ellipses(
        positions_np,
        sigma_diag_full,
        title="Diagonal Mode (Axis-Aligned Only)",
        ax=axes[0],
    )

    plot_diffusion_ellipses(
        positions_np,
        sigma_full,
        title="Full Anisotropic Mode (Captures Rotation)",
        ax=axes[1],
    )

    plt.tight_layout()
    plt.savefig("demo_full_anisotropic_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: demo_full_anisotropic_comparison.png")

    plt.show()

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("1. Diagonal mode shows ellipses aligned with coordinate axes")
    print("2. Full mode captures rotated/tilted cell geometry")
    print("3. Off-diagonal elements encode correlation between x and y diffusion")
    print("4. Both modes preserve positive-definiteness")
    print("5. Full mode gives more accurate representation of cell anisotropy")
    print("=" * 70)


def demo_usage():
    """Show simple usage example."""
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)

    code_example = '''
# Import
from fragile.fractalai.qft.voronoi_observables import compute_voronoi_diffusion_tensor
from fragile.fractalai.core.kinetic_operator import KineticOperator

# Method 1: Direct function call
sigma_full = compute_voronoi_diffusion_tensor(
    voronoi_data=voronoi_data,
    positions=positions,  # numpy array [N, d]
    epsilon_sigma=0.1,
    c2=1.0,
    diagonal_only=False,  # ← Enable full anisotropic tensors
)
# Returns: [N, d, d] symmetric positive-definite tensors

# Method 2: Through KineticOperator
kinetic_op = KineticOperator(
    gamma=1.0,
    beta=1.0,
    delta_t=0.01,
    use_anisotropic_diffusion=True,
    diffusion_mode="voronoi_proxy",
    diagonal_diffusion=False,  # ← Enable full anisotropic tensors
    epsilon_Sigma=0.1,
    potential=your_potential_function,
)

# The kinetic operator will automatically use full tensors
# when computing the Langevin dynamics
'''

    print(code_example)


if __name__ == "__main__":
    demo_comparison()
    demo_usage()

    print("\n✓ Demo complete!")
