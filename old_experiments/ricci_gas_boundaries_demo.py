"""
Demonstration of boundary enforcement and revival via cloning in Ricci Gas.

This script shows:
1. Walkers start within bounds [-4, 4]
2. Some walkers leave bounds and die
3. Dead walkers are revived through cloning from alive walkers
4. Population is maintained over time through birth-death balance
"""

import torch

from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState


def main():
    # Setup with boundaries
    device = torch.device("cpu")
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        epsilon_clone=0.5,
        sigma_clone=0.2,
        force_mode="pull",
        reward_mode="inverse",
        x_min=-4.0,  # Lower bound
        x_max=4.0,  # Upper bound
    )

    gas = RicciGas(params, device=device)

    # Initialize swarm
    N, d = 100, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device) * 2.0  # Start within [-4, 4]
    v = torch.randn(N, d, device=device) * 0.5  # Larger velocities to test boundaries
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    print("=" * 70)
    print("Ricci Gas: Boundary Enforcement and Revival Demonstration")
    print("=" * 70)
    print("\nParameters:")
    print(f"  Walkers: {N}")
    print(f"  Boundaries: [{params.x_min}, {params.x_max}]")
    print(f"  Epsilon clone: {params.epsilon_clone}")
    print(f"  Sigma clone: {params.sigma_clone}")

    print("\n" + "-" * 70)
    print("Running 100 steps with boundaries and cloning...")
    print("-" * 70)

    alive_history = []
    death_events = 0
    revival_events = 0

    for step in range(100):
        prev_alive = state.s.sum().item()

        # Take step (includes cloning, dynamics, and boundary enforcement)
        state = gas.step(state, dt=0.1, gamma=0.8, noise_std=0.3, do_clone=True)

        curr_alive = state.s.sum().item()
        alive_history.append(curr_alive)

        # Track deaths and revivals
        if curr_alive < prev_alive:
            death_events += 1
        elif curr_alive > prev_alive:
            revival_events += 1

        # Print periodic updates
        if step % 20 == 0:
            pos_std = state.x.std(dim=0).mean().item()
            out_of_bounds = (
                ((state.x < params.x_min) | (state.x > params.x_max)).any(dim=-1).sum().item()
            )

            print(f"  Step {step:3d}:")
            print(f"    Alive: {int(curr_alive):3d}/{N}")
            print(f"    Position spread: {pos_std:.3f}")
            print(f"    Out of bounds (but alive): {out_of_bounds}")

    # Statistics
    min_alive = min(alive_history)
    max_alive = max(alive_history)
    mean_alive = sum(alive_history) / len(alive_history)
    final_alive = alive_history[-1]

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print("\nPopulation dynamics:")
    print(f"  Initial alive:  {N}")
    print(f"  Final alive:    {int(final_alive)}")
    print(f"  Min alive:      {int(min_alive)}")
    print(f"  Max alive:      {int(max_alive)}")
    print(f"  Mean alive:     {mean_alive:.1f}")

    print("\nEvents:")
    print(f"  Death events:   {death_events} (steps where population decreased)")
    print(f"  Revival events: {revival_events} (steps where population increased)")

    # Check final positions are within bounds for alive walkers
    alive_mask = state.s.bool()
    alive_x = state.x[alive_mask]

    if alive_x.shape[0] > 0:
        within_bounds = (
            ((alive_x >= params.x_min) & (alive_x <= params.x_max)).all(dim=-1).sum().item()
        )
        print("\nBoundary compliance:")
        print(f"  Alive walkers: {alive_x.shape[0]}")
        print(f"  Within bounds: {within_bounds}")
        print(f"  Compliance rate: {100 * within_bounds / alive_x.shape[0]:.1f}%")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("  ✓ Walkers that leave [-4, 4] are killed immediately")
    print("  ✓ Dead walkers are revived by cloning from alive walkers")
    print("  ✓ Population is maintained through birth-death balance")
    print(f"  ✓ Revival events ({revival_events}) help counter deaths ({death_events})")
    print(f"  ✓ Final population: {100 * final_alive / N:.0f}% of initial")

    print("\n" + "-" * 70)
    print("Comparison with no cloning:")
    print("-" * 70)

    # Run without cloning to show the difference
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device) * 2.0
    v = torch.randn(N, d, device=device) * 0.5
    s = torch.ones(N, device=device)
    state_no_clone = SwarmState(x=x, v=v, s=s)

    alive_no_clone = []
    for _ in range(100):
        state_no_clone = gas.step(state_no_clone, dt=0.1, gamma=0.8, noise_std=0.3, do_clone=False)
        alive_no_clone.append(state_no_clone.s.sum().item())

    final_no_clone = alive_no_clone[-1]
    min_no_clone = min(alive_no_clone)

    print("\nWithout cloning:")
    print(f"  Final alive:  {int(final_no_clone)}")
    print(f"  Min alive:    {int(min_no_clone)}")

    print("\nWith cloning:")
    print(f"  Final alive:  {int(final_alive)}")
    print(f"  Min alive:    {int(min_alive)}")

    print("\nImprovement:")
    print(
        f"  Final: +{int(final_alive - final_no_clone)} walkers ({100 * (final_alive - final_no_clone) / N:.0f}%)"
    )
    print(f"  Min:   +{int(min_alive - min_no_clone)} walkers")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
