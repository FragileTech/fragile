"""
Run Euclidean Gas with Z-function reward to test QSD localization at zeta zeros.

This experiment tests the hypothesis that:
1. Z-function reward causes QSD to localize at zeta zero locations
2. Yang-Mills Hamiltonian eigenvalues reflect this localization
3. E_n ~ α|t_n| (eigenvalues match zero imaginary parts)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from z_reward import get_first_zeta_zeros, ZFunctionReward

from fragile.euclidean_gas import EuclideanGas, EuclideanGasParams, SwarmState
from fragile.gas_parameters import CloningParams, LangevinParams, PotentialParams


matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CustomPotentialWithZReward(PotentialParams):
    """
    Custom potential that combines confinement with Z-function reward.

    V(x) = (1/2ℓ²)||x||² - r_Z(x)

    where r_Z(x) = 1/(Z(||x||)² + ε²) is the Z-function reward.
    """

    def __init__(self, z_reward: ZFunctionReward, confinement_scale: float = 100.0):
        """
        Args:
            z_reward: ZFunctionReward instance
            confinement_scale: Confinement length scale ℓ_conf
        """
        super().__init__()
        self.z_reward = z_reward
        self.ell_conf = confinement_scale

    def evaluate(self, x: Tensor) -> Tensor:
        """
        Evaluate V(x) = confinement + Z-reward potential.

        Args:
            x: Positions [N, d]

        Returns:
            Potential values [N]
        """
        # Confinement term
        r_squared = torch.sum(x**2, dim=-1)
        v_conf = r_squared / (2 * self.ell_conf**2)

        # Z-reward term (negative because reward → attractive potential)
        v_z = -self.z_reward.reward(x)

        return v_conf + v_z


def run_z_reward_simulation(
    n_walkers: int = 1000,
    d: int = 1,  # Start with 1D for visualization
    n_steps: int = 5000,
    epsilon: float = 0.1,
    ell_conf: float = 50.0,
    output_dir: str = "experiments/z_function_reward/results",
):
    """
    Run Euclidean Gas simulation with Z-function reward.

    Args:
        n_walkers: Number of walkers
        d: Dimension (1 for radial, 3 for full 3D)
        n_steps: Number of simulation steps
        epsilon: Z-function regularization
        ell_conf: Confinement length scale
        output_dir: Directory for saving results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Z-FUNCTION REWARD SIMULATION")
    print(f"{'=' * 60}")
    print("Parameters:")
    print(f"  N = {n_walkers} walkers")
    print(f"  d = {d} dimensions")
    print(f"  ε = {epsilon} (Z-function regularization)")
    print(f"  ℓ_conf = {ell_conf} (confinement scale)")
    print(f"  Steps = {n_steps}")
    print(f"{'=' * 60}\n")

    # Create Z-function reward
    z_reward = ZFunctionReward(epsilon=epsilon, t_max=ell_conf * 1.5)

    # Create potential with Z-reward
    potential = CustomPotentialWithZReward(z_reward, confinement_scale=ell_conf)

    # Create Euclidean Gas parameters
    params = EuclideanGasParams(
        N=n_walkers,
        d=d,
        potential=potential,
        langevin=LangevinParams(
            gamma=1.0,
            beta=1.0,
            delta_t=0.1,
        ),
        cloning=CloningParams(
            sigma_x=0.5,
            lambda_alg=1.0,
            epsilon_c=2.0,
            alpha_reward=1.0,  # Exploitation weight
            beta_diversity=0.5,  # Diversity weight
            eta_virtual=0.1,  # Virtual reward learning rate
            alive_threshold=0.5,  # Survival threshold
            companion_method="hybrid",
        ),
        dtype="float32",
    )

    # Initialize gas
    print("Initializing Euclidean Gas...")
    gas = EuclideanGas(params)

    # Custom reward function that uses Z-function
    def z_reward_fn(x: Tensor, v: Tensor, state: SwarmState) -> Tensor:
        """Custom reward based on Z-function."""
        return z_reward.reward(x)

    # Store history
    radii_history = []
    reward_history = []

    print("\nRunning simulation...")
    for step in range(n_steps):
        # Step the gas
        state = gas.step()

        # Compute radial coordinates
        radii = torch.norm(state.x, dim=-1).cpu().numpy()
        radii_history.append(radii.copy())

        # Compute rewards using Z-function
        rewards = z_reward_fn(state.x, state.v, state).cpu().numpy()
        reward_history.append(rewards.copy())

        if (step + 1) % 500 == 0:
            print(f"  Step {step + 1}/{n_steps}")
            print(f"    Mean radius: {radii.mean():.3f}")
            print(f"    Std radius:  {radii.std():.3f}")
            print(f"    Mean reward: {rewards.mean():.3f}")

    print("\n✓ Simulation complete!")

    # Convert to arrays
    radii_history = np.array(radii_history)  # [n_steps, N]
    reward_history = np.array(reward_history)

    # Save final state
    final_radii = radii_history[-1]
    np.save(f"{output_dir}/final_radii.npy", final_radii)
    np.save(f"{output_dir}/radii_history.npy", radii_history)
    np.save(f"{output_dir}/reward_history.npy", reward_history)

    print(f"\nSaved results to {output_dir}/")

    return radii_history, reward_history, z_reward


def analyze_qsd_localization(
    radii_history: np.ndarray,
    z_reward: ZFunctionReward,
    output_dir: str = "experiments/z_function_reward/results",
):
    """
    Analyze whether QSD localizes at zeta zeros.

    Args:
        radii_history: Radial coordinates over time [n_steps, N]
        z_reward: ZFunctionReward instance
        output_dir: Directory for saving plots
    """
    print(f"\n{'=' * 60}")
    print("QSD LOCALIZATION ANALYSIS")
    print(f"{'=' * 60}\n")

    # Use last 1000 steps as QSD samples
    qsd_radii = radii_history[-1000:].flatten()

    # Get first 20 zeta zeros
    zeros = get_first_zeta_zeros(20)
    zeros_in_range = zeros[zeros < z_reward.t_max]

    print(f"Analyzing {len(qsd_radii)} QSD samples...")
    print("First 10 zeta zeros:")
    for i, t in enumerate(zeros[:10]):
        print(f"  t_{i + 1} = {t:.6f}")

    # Compute histogram
    bins = np.linspace(0, z_reward.t_max, 200)
    hist, bin_edges = np.histogram(qsd_radii, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot QSD distribution vs zeta zeros
    _fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: QSD histogram with zero locations
    axes[0].bar(
        bin_centers,
        hist,
        width=bin_centers[1] - bin_centers[0],
        alpha=0.6,
        color="blue",
        label="QSD density",
    )

    # Mark zeta zeros
    hist.max()
    for t in zeros_in_range:
        axes[0].axvline(t, color="red", linestyle="--", alpha=0.7, linewidth=2)

    axes[0].set_xlabel("Radial coordinate ||x||")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title("QSD Distribution vs Zeta Zero Locations")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zoom on first few zeros
    t_zoom_max = zeros[min(5, len(zeros_in_range) - 1)]
    mask = bin_centers < t_zoom_max * 1.2
    axes[1].bar(
        bin_centers[mask],
        hist[mask],
        width=bin_centers[1] - bin_centers[0],
        alpha=0.6,
        color="blue",
        label="QSD density",
    )

    for t in zeros[:5]:
        if t < t_zoom_max * 1.2:
            axes[1].axvline(
                t,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label=f"t_{zeros.tolist().index(t) + 1}={t:.2f}",
            )

    axes[1].set_xlabel("Radial coordinate ||x||")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("QSD Distribution (Zoomed on First Zeros)")
    axes[1].grid(True, alpha=0.3)

    # Create custom legend for first plot zeros
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", alpha=0.6, linewidth=10, label="QSD"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Zeta zeros"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/qsd_localization.png", dpi=150)
    print(f"\nSaved QSD analysis to {output_dir}/qsd_localization.png")
    plt.close()

    # Quantitative analysis: Check peak locations
    print("\nQuantitative Analysis:")
    print("Finding peaks in QSD density...")

    from scipy.signal import find_peaks

    # Find peaks in histogram
    peaks, _properties = find_peaks(hist, height=hist.max() * 0.1, distance=5)
    peak_locations = bin_centers[peaks]

    print(f"\nFound {len(peak_locations)} peaks:")
    for i, peak in enumerate(peak_locations[:10]):
        # Find nearest zero
        nearest_zero_idx = np.argmin(np.abs(zeros - peak))
        nearest_zero = zeros[nearest_zero_idx]
        distance = abs(peak - nearest_zero)
        print(
            f"  Peak {i + 1}: r = {peak:.4f}, nearest zero t_{nearest_zero_idx + 1} = {nearest_zero:.4f}, distance = {distance:.4f}"
        )

    return peak_locations, zeros_in_range


if __name__ == "__main__":
    # Run simulation
    radii_history, reward_history, z_reward = run_z_reward_simulation(
        n_walkers=1000,
        d=1,  # 1D radial for visualization
        n_steps=5000,
        epsilon=0.5,  # Wider peaks for easier localization
        ell_conf=50.0,
    )

    # Analyze QSD localization
    peak_locations, zeros = analyze_qsd_localization(radii_history, z_reward)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"QSD has {len(peak_locations)} peaks")
    print(f"First {len(zeros)} zeta zeros in range")
    print("\nNext step: Compute Yang-Mills eigenvalues and compare with zeros")
    print("=" * 60)
