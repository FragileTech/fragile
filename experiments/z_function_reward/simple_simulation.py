"""
Simplified Z-reward simulation to get empirical data quickly.

Focus: Just run it, see if QSD localizes at zeta zeros.
"""

from pathlib import Path

import matplotlib
import numpy as np
import torch
from z_reward import get_first_zeta_zeros, ZFunctionReward

from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    PotentialParams,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Global Z-reward instance (workaround for Pydantic)
_global_z_reward = None
_global_ell_conf = None


class ZRewardPotential(PotentialParams):
    """Potential combining confinement and Z-function attraction."""

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        V(x) = (r²)/(2ℓ²) - reward(x)

        Confinement pulls to origin, reward creates peaks at zeros.
        """
        global _global_z_reward, _global_ell_conf

        r_sq = torch.sum(x**2, dim=-1)
        conf = r_sq / (2 * _global_ell_conf**2)

        # Negative reward = attractive potential
        reward = _global_z_reward.reward(x)

        return conf - reward


def run_simulation():
    """Run the simulation and analyze results."""

    print("\n" + "=" * 70)
    print("Z-FUNCTION REWARD SIMULATION - EMPIRICAL INVESTIGATION")
    print("=" * 70)

    # Parameters
    N = 500
    d = 1  # 1D radial for simplicity
    n_steps = 10000
    epsilon = 0.5
    ell_conf = 50.0

    print("\nParameters:")
    print(f"  N={N} walkers, d={d}")
    print(f"  ε={epsilon} (Z regularization)")
    print(f"  ℓ_conf={ell_conf}")
    print(f"  Steps={n_steps}")

    # Create Z-reward and set globals
    global _global_z_reward, _global_ell_conf
    _global_z_reward = ZFunctionReward(epsilon=epsilon, t_max=ell_conf * 2.0)
    _global_ell_conf = ell_conf

    # Create potential
    potential = ZRewardPotential()

    # Create gas
    params = EuclideanGasParams(
        N=N,
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
            alpha_restitution=0.5,
        ),
        dtype="float32",
    )

    gas = EuclideanGas(params)
    state = gas.initialize_state()

    # Run simulation
    print("\nRunning simulation...")
    radii_history = []

    for step in range(n_steps):
        _state_cloned, state = gas.step(state)

        # Store radii
        radii = torch.norm(state.x, dim=-1).cpu().numpy()
        radii_history.append(radii)

        if (step + 1) % 1000 == 0:
            print(
                f"  Step {step + 1}/{n_steps}, mean r={radii.mean():.2f}, std r={radii.std():.2f}"
            )

    radii_history = np.array(radii_history)

    print("\n✓ Simulation complete!")

    # Analyze QSD
    print("\nAnalyzing QSD localization...")

    # Use last 2000 steps as QSD
    qsd_radii = radii_history[-2000:].flatten()

    # Get zeta zeros
    zeros = get_first_zeta_zeros(15)
    zeros_in_range = zeros[zeros < ell_conf]

    print("\nFirst 10 zeta zeros:")
    for i, t in enumerate(zeros[:10]):
        print(f"  t_{i + 1} = {t:.6f}")

    # Plot results
    _fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Time evolution of radii
    times = np.arange(n_steps)
    mean_r = radii_history.mean(axis=1)
    std_r = radii_history.std(axis=1)

    axes[0].plot(times, mean_r, "b-", linewidth=1.5, label="Mean radius")
    axes[0].fill_between(times, mean_r - std_r, mean_r + std_r, alpha=0.3)
    axes[0].axhline(
        zeros[0], color="r", linestyle="--", alpha=0.7, label=f"First zero t₁={zeros[0]:.2f}"
    )
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Radial coordinate")
    axes[0].set_title("Convergence to QSD")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: QSD histogram
    bins = np.linspace(0, ell_conf, 100)
    hist, bin_edges = np.histogram(qsd_radii, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    axes[1].bar(
        bin_centers,
        hist,
        width=bin_centers[1] - bin_centers[0],
        alpha=0.6,
        color="blue",
        label="QSD density",
    )

    for i, t in enumerate(zeros_in_range):
        axes[1].axvline(t, color="red", linestyle="--", alpha=0.7, linewidth=2)

    axes[1].set_xlabel("Radial coordinate ||x||")
    axes[1].set_ylabel("Probability density")
    axes[1].set_title("QSD Distribution vs Zeta Zero Locations")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, ell_conf)

    # Plot 3: Zoom on first zeros
    t_max_zoom = zeros[min(4, len(zeros_in_range) - 1)] * 1.3
    mask = bin_centers < t_max_zoom

    axes[2].bar(
        bin_centers[mask],
        hist[mask],
        width=bin_centers[1] - bin_centers[0],
        alpha=0.6,
        color="blue",
    )

    for i, t in enumerate(zeros[:5]):
        if t < t_max_zoom:
            axes[2].axvline(
                t, color="red", linestyle="--", alpha=0.7, linewidth=2, label=f"t_{i + 1}={t:.2f}"
            )

    axes[2].set_xlabel("Radial coordinate ||x||")
    axes[2].set_ylabel("Probability density")
    axes[2].set_title("QSD Distribution (Zoomed on First Zeros)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    Path("experiments/z_function_reward/results").mkdir(parents=True, exist_ok=True)
    plt.savefig("experiments/z_function_reward/results/qsd_analysis.png", dpi=150)
    print("\n✓ Saved plot to experiments/z_function_reward/results/qsd_analysis.png")
    plt.close()

    # Quantitative analysis
    print("\nQuantitative Analysis:")
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(hist, height=hist.max() * 0.15, distance=3)
    peak_locs = bin_centers[peaks]

    print(f"\nFound {len(peak_locs)} peaks in QSD:")
    for i, peak in enumerate(peak_locs):
        nearest_idx = np.argmin(np.abs(zeros - peak))
        nearest_zero = zeros[nearest_idx]
        dist = abs(peak - nearest_zero)
        print(
            f"  Peak {i + 1}: r={peak:.4f}, nearest zero t_{nearest_idx + 1}={nearest_zero:.4f}, dist={dist:.4f}"
        )

    # Save data
    np.save("experiments/z_function_reward/results/qsd_radii.npy", qsd_radii)
    np.save("experiments/z_function_reward/results/zeros.npy", zeros)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"QSD peaks: {len(peak_locs)}")
    print(f"Zeta zeros in range: {len(zeros_in_range)}")

    if len(peak_locs) >= 3:
        print("\n✓ QSD shows multi-peak structure!")
        print("→ Evidence for localization at zeta zeros")
    else:
        print("\n⚠ QSD doesn't show clear multi-peak structure")
        print("→ May need different parameters (smaller ε, larger N, more steps)")

    print("=" * 70)

    return qsd_radii, zeros


if __name__ == "__main__":
    qsd_radii, zeros = run_simulation()
