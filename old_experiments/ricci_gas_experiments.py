"""
Ricci Fragile Gas Experiments

Run experiments to test the phase transition hypothesis and compare variants.

Usage:
    python experiments/ricci_gas_experiments.py --experiment phase_transition
    python experiments/ricci_gas_experiments.py --experiment ablation
    python experiments/ricci_gas_experiments.py --experiment toy_problems
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fragile.ricci_gas import (
    create_ricci_gas_variants,
    double_well_3d,
    rastrigin_3d,
    RicciGas,
    RicciGasParams,
    SwarmState,
)


def initialize_swarm(
    N: int = 100,
    d: int = 3,
    x_range: tuple[float, float] = (-3.0, 3.0),
    device: str = "cpu",
) -> SwarmState:
    """Initialize swarm with random positions and velocities."""
    x_min, x_max = x_range

    x = torch.rand(N, d, device=device) * (x_max - x_min) + x_min
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    return SwarmState(x=x, v=v, s=s)


def measure_swarm_statistics(state: SwarmState) -> dict:
    """Compute swarm-level statistics for phase transition detection."""
    x = state.x[state.s.bool()]

    if len(x) == 0:
        return {
            "variance": 0.0,
            "entropy": 0.0,
            "alive_fraction": 0.0,
            "max_curvature": 0.0,
        }

    # Spatial variance
    variance = x.var(dim=0).sum().item()

    # Histogram-based entropy (in 3D)
    hist, _ = np.histogramdd(x.cpu().numpy(), bins=10)
    hist = hist.flatten()
    hist = hist[hist > 0] / hist.sum()  # Normalize
    entropy = -(hist * np.log(hist)).sum()

    return {
        "variance": variance,
        "entropy": float(entropy),
        "alive_fraction": state.s.mean().item(),
        "max_curvature": state.R.max().item() if state.R is not None else 0.0,
    }


def experiment_phase_transition(
    output_dir: Path,
    alpha_values: list[float] = [0.01, 0.1, 0.5, 1.0, 2.0],
    N: int = 200,
    T: int = 1000,
):
    """Experiment 1: Detect phase transition by varying epsilon_R.

    Measure variance, entropy, and max curvature as function of α.
    """
    print("=== Experiment: Phase Transition Detection ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {alpha: [] for alpha in alpha_values}

    for alpha in alpha_values:
        print(f"\nRunning α = {alpha:.3f}...")

        params = RicciGasParams(
            epsilon_R=alpha,
            kde_bandwidth=0.3,
            force_mode="pull",
            reward_mode="inverse",
        )
        gas = RicciGas(params)
        state = initialize_swarm(N=N, d=3)

        for t in range(T):
            # Compute geometry
            _R, _H = gas.compute_curvature(state, cache=True)

            # Measure statistics
            stats = measure_swarm_statistics(state)
            stats["alpha"] = alpha
            stats["t"] = t
            results[alpha].append(stats)

            # Simple Langevin step (no cloning for simplicity)
            force = gas.compute_force(state)
            state.v = state.v * 0.9 + force * 0.1 + torch.randn_like(state.v) * 0.05
            state.x += state.v * 0.1

            # Status refresh
            state = gas.apply_singularity_regulation(state)

            if t % 100 == 0:
                print(
                    f"  t={t}: var={stats['variance']:.3f}, "
                    f"entropy={stats['entropy']:.3f}, "
                    f"R_max={stats['max_curvature']:.3f}"
                )

    # Plot results
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for alpha in alpha_values:
        data = results[alpha]
        t_vals = [d["t"] for d in data]
        variance = [d["variance"] for d in data]
        entropy = [d["entropy"] for d in data]
        R_max = [d["max_curvature"] for d in data]
        alive = [d["alive_fraction"] for d in data]

        axes[0, 0].plot(t_vals, variance, label=f"α={alpha:.2f}")
        axes[0, 1].plot(t_vals, entropy, label=f"α={alpha:.2f}")
        axes[1, 0].plot(t_vals, R_max, label=f"α={alpha:.2f}")
        axes[1, 1].plot(t_vals, alive, label=f"α={alpha:.2f}")

    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Spatial Variance")
    axes[0, 0].legend()
    axes[0, 0].set_title("Variance vs Time (Phase Transition)")

    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Entropy")
    axes[0, 1].legend()
    axes[0, 1].set_title("Entropy vs Time")

    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Max Curvature")
    axes[1, 0].legend()
    axes[1, 0].set_title("Max Ricci Curvature")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Alive Fraction")
    axes[1, 1].legend()
    axes[1, 1].set_title("Fraction of Alive Walkers")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_transition.png", dpi=150)
    print(f"\nSaved plot to {output_dir / 'phase_transition.png'}")

    # Summary plot: Final variance vs alpha
    final_variance = [results[alpha][-1]["variance"] for alpha in alpha_values]
    final_entropy = [results[alpha][-1]["entropy"] for alpha in alpha_values]

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(alpha_values, final_variance, "o-")
    axes[0].set_xlabel("Feedback Strength α")
    axes[0].set_ylabel("Final Variance")
    axes[0].set_title("Phase Diagram: Variance")
    axes[0].set_xscale("log")

    axes[1].plot(alpha_values, final_entropy, "o-")
    axes[1].set_xlabel("Feedback Strength α")
    axes[1].set_ylabel("Final Entropy")
    axes[1].set_title("Phase Diagram: Entropy")
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagram.png", dpi=150)
    print(f"Saved phase diagram to {output_dir / 'phase_diagram.png'}")


def experiment_ablation(output_dir: Path, N: int = 150, T: int = 500):
    """Experiment 2: Compare the four variants (A, B, C, D)."""
    print("\n=== Experiment: Ablation Study ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = create_ricci_gas_variants()
    results = {name: [] for name in variants}

    for name, params in variants.items():
        print(f"\nRunning variant: {name}")
        print(f"  Force: {params.force_mode}, Reward: {params.reward_mode}")

        gas = RicciGas(params)
        state = initialize_swarm(N=N, d=3)

        for t in range(T):
            _R, _H = gas.compute_curvature(state, cache=True)
            stats = measure_swarm_statistics(state)
            stats["variant"] = name
            stats["t"] = t
            results[name].append(stats)

            # Simple dynamics
            force = gas.compute_force(state)
            state.v = state.v * 0.9 + force * 0.1 + torch.randn_like(state.v) * 0.05
            state.x += state.v * 0.1
            state = gas.apply_singularity_regulation(state)

            if t % 100 == 0:
                print(f"  t={t}: var={stats['variance']:.3f}")

    # Plot comparison
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for name in variants:
        data = results[name]
        t_vals = [d["t"] for d in data]
        variance = [d["variance"] for d in data]
        entropy = [d["entropy"] for d in data]
        R_max = [d["max_curvature"] for d in data]
        alive = [d["alive_fraction"] for d in data]

        axes[0, 0].plot(t_vals, variance, label=name)
        axes[0, 1].plot(t_vals, entropy, label=name)
        axes[1, 0].plot(t_vals, R_max, label=name)
        axes[1, 1].plot(t_vals, alive, label=name)

    for ax in axes.flat:
        ax.legend()
        ax.set_xlabel("Time")

    axes[0, 0].set_ylabel("Variance")
    axes[0, 0].set_title("Ablation: Variance")

    axes[0, 1].set_ylabel("Entropy")
    axes[0, 1].set_title("Ablation: Entropy")

    axes[1, 0].set_ylabel("Max Curvature")
    axes[1, 0].set_title("Ablation: Max Ricci")
    axes[1, 0].set_yscale("log")

    axes[1, 1].set_ylabel("Alive Fraction")
    axes[1, 1].set_title("Ablation: Survival")

    plt.tight_layout()
    plt.savefig(output_dir / "ablation.png", dpi=150)
    print(f"\nSaved ablation results to {output_dir / 'ablation.png'}")


def experiment_toy_problems(output_dir: Path):
    """Experiment 3: Test on double-well and Rastrigin."""
    print("\n=== Experiment: Toy Problems ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = {
        "double_well": double_well_3d,
        "rastrigin": rastrigin_3d,
    }

    for name, potential_fn in problems.items():
        print(f"\nTesting on {name}...")

        params = RicciGasParams(epsilon_R=0.5, kde_bandwidth=0.3)
        gas = RicciGas(params)
        state = initialize_swarm(N=200, d=3, x_range=(-2.0, 2.0))

        # Track best potential value found
        best_V = float("inf")
        trajectory = []

        for t in range(500):
            _R, _H = gas.compute_curvature(state, cache=True)

            # Evaluate potential
            V = potential_fn(state.x[state.s.bool()])
            V_min = V.min().item()
            best_V = min(V_min, best_V)

            trajectory.append({
                "t": t,
                "V_min": V_min,
                "V_mean": V.mean().item(),
                "variance": state.x[state.s.bool()].var(dim=0).sum().item(),
            })

            # Dynamics
            force = gas.compute_force(state)
            state.v = state.v * 0.9 + force * 0.1 + torch.randn_like(state.v) * 0.05
            state.x += state.v * 0.1

            if t % 100 == 0:
                print(f"  t={t}: V_min={V_min:.4f}, V_mean={V.mean().item():.4f}")

        print(f"  Final best V: {best_V:.6f}")

        # Plot trajectory
        t_vals = [d["t"] for d in trajectory]
        V_min = [d["V_min"] for d in trajectory]
        V_mean = [d["V_mean"] for d in trajectory]

        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, V_min, label="Min Potential")
        plt.plot(t_vals, V_mean, label="Mean Potential", alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("Potential Value")
        plt.title(f"Ricci Gas on {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{name}_optimization.png", dpi=150)
        print(f"  Saved to {output_dir / f'{name}_optimization.png'}")


def visualize_curvature_heatmap(output_dir: Path):
    """Create curvature heatmap visualization."""
    print("\n=== Visualization: Curvature Heatmap ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    params = RicciGasParams(epsilon_R=1.0, kde_bandwidth=0.4)
    gas = RicciGas(params)

    # Create clustered state
    state = initialize_swarm(N=100, d=3, x_range=(-2.0, 2.0))

    # Run for a bit to let structure form
    for _ in range(200):
        _R, _H = gas.compute_curvature(state, cache=True)
        force = gas.compute_force(state)
        state.v = state.v * 0.9 + force * 0.2 + torch.randn_like(state.v) * 0.05
        state.x += state.v * 0.1

    # Compute final curvature
    _R, _H = gas.compute_curvature(state, cache=True)

    # Generate heatmap
    viz_data = gas.visualize_curvature(state, grid_resolution=60, zlevel=0.0)

    # Plot
    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Curvature heatmap
    im = axes[0].contourf(
        viz_data["x_grid"],
        viz_data["y_grid"],
        viz_data["R_grid"],
        levels=20,
        cmap="RdBu_r",
    )
    axes[0].scatter(
        viz_data["walkers"][viz_data["alive"], 0],
        viz_data["walkers"][viz_data["alive"], 1],
        c="black",
        s=10,
        alpha=0.6,
        label="Alive walkers",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Ricci Curvature Heatmap (z=0)")
    axes[0].legend()
    plt.colorbar(im, ax=axes[0], label="Ricci R")

    # Walker distribution with curvature
    scatter = axes[1].scatter(
        viz_data["walkers"][viz_data["alive"], 0],
        viz_data["walkers"][viz_data["alive"], 1],
        c=state.R[state.s.bool()].cpu().numpy(),
        s=30,
        cmap="RdBu_r",
        alpha=0.7,
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Walker Positions Colored by Ricci")
    plt.colorbar(scatter, ax=axes[1], label="Ricci R")

    plt.tight_layout()
    plt.savefig(output_dir / "curvature_heatmap.png", dpi=150)
    print(f"Saved heatmap to {output_dir / 'curvature_heatmap.png'}")


def main():
    parser = argparse.ArgumentParser(description="Ricci Gas Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["phase_transition", "ablation", "toy_problems", "heatmap", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/ricci_gas_results"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.experiment in {"phase_transition", "all"}:
        experiment_phase_transition(args.output_dir / "phase_transition")

    if args.experiment in {"ablation", "all"}:
        experiment_ablation(args.output_dir / "ablation")

    if args.experiment in {"toy_problems", "all"}:
        experiment_toy_problems(args.output_dir / "toy_problems")

    if args.experiment in {"heatmap", "all"}:
        visualize_curvature_heatmap(args.output_dir / "visualization")

    print("\n=== All experiments complete ===")


if __name__ == "__main__":
    main()
