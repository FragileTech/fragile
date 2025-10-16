"""
Simple demonstration of cloning in the Ricci Gas.

This script shows:
1. Walkers start in random positions
2. Cloning causes walkers to aggregate based on distance
3. Number of walkers remains constant
4. Statistics about cloning behavior
"""

import torch

from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState


def main():
    # Setup
    device = torch.device("cpu")
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        epsilon_clone=0.5,  # Moderate distance for companion selection
        sigma_clone=0.1,  # Small jitter
        force_mode="pull",
        reward_mode="inverse",
    )

    gas = RicciGas(params, device=device)

    # Initialize swarm
    N, d = 100, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device) * 2.0
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    print("=" * 60)
    print("Ricci Gas Cloning Demonstration")
    print("=" * 60)
    print("\nInitial state:")
    print(f"  Walkers: {N}")
    print(f"  Dimension: {d}")
    print(f"  Epsilon clone: {params.epsilon_clone}")
    print(f"  Sigma clone: {params.sigma_clone}")
    print(f"  Position spread: {state.x.std(dim=0).mean():.3f}")

    # Show effect of single cloning step
    print("\n" + "-" * 60)
    print("Single cloning step:")
    print("-" * 60)

    state_before = SwarmState(x=state.x.clone(), v=state.v.clone(), s=state.s.clone())
    state_after = gas.apply_cloning(state)

    # Compute displacement statistics
    displacement = (state_after.x - state_before.x).norm(dim=-1)

    print("\nDisplacement statistics:")
    print(f"  Mean: {displacement.mean():.3f}")
    print(f"  Std:  {displacement.std():.3f}")
    print(f"  Min:  {displacement.min():.3f}")
    print(f"  Max:  {displacement.max():.3f}")

    # Analyze companion selection
    print(f"\nPosition spread after cloning: {state_after.x.std(dim=0).mean():.3f}")
    print(f"Walkers still present: {state_after.x.shape[0]} (should be {N})")

    # Run full step with cloning + dynamics
    print("\n" + "-" * 60)
    print("Running 50 steps with cloning + dynamics:")
    print("-" * 60)

    state = SwarmState(x=x.clone(), v=v.clone(), s=s.clone())

    for t in range(50):
        state = gas.step(state, dt=0.1, do_clone=True)

        if t % 10 == 0:
            R_mean = state.R.mean().item() if state.R is not None else 0.0
            spread = state.x.std(dim=0).mean().item()
            print(f"  Step {t:2d}: spread={spread:.3f}, R_mean={R_mean:.4f}")

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    print("\nKey observations:")
    print("  • Cloning creates local jumps (displacement < 1.0)")
    print("  • Swarm size remains constant (N = 100)")
    print("  • Walkers aggregate due to distance-based companion selection")
    print("  • Combined with force, creates push-pull dynamics")
    print("\nNext steps:")
    print("  • Try different epsilon_clone values (large → uniform, small → local)")
    print("  • Visualize walker trajectories")
    print("  • Compare with/without cloning")


if __name__ == "__main__":
    main()
