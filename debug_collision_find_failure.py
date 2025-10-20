"""Debug to find the actual test failure case."""

import torch
from fragile.core.cloning import inelastic_collision_velocity

# Replicate test setup WITHOUT seed
N, d = 20, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run test many times to find failure case where EXTRA walkers change
num_runs = 1000
for run in range(num_runs):
    velocities = torch.randn(N, d, device=device)
    companions = torch.randint(0, N, (N,), device=device)
    will_clone = torch.zeros(N, dtype=torch.bool, device=device)
    will_clone[5] = True  # Only walker 5 clones

    companion_idx = companions[5].item()

    # Skip self-collision case (already analyzed)
    if companion_idx == 5:
        continue

    # Call function
    v_new = inelastic_collision_velocity(velocities, companions, will_clone)

    # Check which walkers changed
    changed_walkers = []
    for i in range(N):
        if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
            changed_walkers.append(i)

    expected_changed = {5, companion_idx}
    actual_changed = set(changed_walkers)

    # Look for case where we have EXTRA changed walkers (not missing)
    if actual_changed != expected_changed and len(actual_changed) > len(expected_changed):
        print("=" * 60)
        print(f"FAILURE ON RUN {run}: EXTRA WALKERS CHANGED")
        print("=" * 60)
        print(f"Walker 5 clones to companion {companion_idx}")
        print(f"Expected changed walkers: {expected_changed}")
        print(f"Actual changed walkers: {actual_changed}")
        print(f"Extra changed: {actual_changed - expected_changed}")

        # Full companion map for walkers that clone
        print("\nWalkers cloning to various companions:")
        for i in range(N):
            if will_clone[i]:
                print(f"  Walker {i} → Companion {companions[i].item()}")

        # For each extra walker, analyze why it changed
        for extra_walker in actual_changed - expected_changed:
            print(f"\nWalker {extra_walker} changed unexpectedly:")
            print(f"  Velocity before: {velocities[extra_walker]}")
            print(f"  Velocity after:  {v_new[extra_walker]}")
            print(f"  This walker clones: {will_clone[extra_walker].item()}")
            print(f"  Its companion: {companions[extra_walker].item()}")

            # Check relationships
            if companions[extra_walker] == 5:
                print(f"  → Walker {extra_walker} has walker 5 as companion (doesn't clone)")
            if companions[extra_walker] == companion_idx:
                print(f"  → Walker {extra_walker} has companion {companion_idx} as companion (doesn't clone)")

        # Reconstruct what collision groups were formed
        print("\nCollision group reconstruction:")
        unique_companions = torch.unique(companions[will_clone])
        for c_idx in unique_companions:
            cloners_mask = (companions == c_idx) & will_clone
            cloner_indices = torch.where(cloners_mask)[0].tolist()
            cloner_indices_no_companion = [idx for idx in cloner_indices if idx != c_idx]
            group_indices = [c_idx] + cloner_indices_no_companion
            print(f"  Companion {c_idx}: collision group = {group_indices}")

        break
else:
    print("No failure found with EXTRA walkers in 1000 runs.")
    print("Let's check for MISSING walkers case...")

# If no extra walkers case found, look for missing walkers
for run in range(1000):
    velocities = torch.randn(N, d, device=device)
    companions = torch.randint(0, N, (N,), device=device)
    will_clone = torch.zeros(N, dtype=torch.bool, device=device)
    will_clone[5] = True

    companion_idx = companions[5].item()

    # Skip self-collision
    if companion_idx == 5:
        continue

    v_new = inelastic_collision_velocity(velocities, companions, will_clone)

    changed_walkers = []
    for i in range(N):
        if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
            changed_walkers.append(i)

    expected_changed = {5, companion_idx}
    actual_changed = set(changed_walkers)

    if actual_changed != expected_changed and len(actual_changed) < len(expected_changed):
        print("=" * 60)
        print(f"FAILURE ON RUN {run}: MISSING WALKERS")
        print("=" * 60)
        print(f"Walker 5 clones to companion {companion_idx}")
        print(f"Expected changed walkers: {expected_changed}")
        print(f"Actual changed walkers: {actual_changed}")
        print(f"Missing: {expected_changed - actual_changed}")
        break
