"""Debug inelastic collision velocity without seed to reproduce test failure."""

import torch

from fragile.core.cloning import inelastic_collision_velocity


# Replicate test setup WITHOUT seed (like the test does)
N, d = 20, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run test multiple times to find failure case
num_runs = 100
for run in range(num_runs):
    velocities = torch.randn(N, d, device=device)
    companions = torch.randint(0, N, (N,), device=device)
    will_clone = torch.zeros(N, dtype=torch.bool, device=device)
    will_clone[5] = True  # Only walker 5 clones

    companion_idx = companions[5].item()

    # Call function
    v_new = inelastic_collision_velocity(velocities, companions, will_clone)

    # Check which walkers changed
    changed_walkers = []
    for i in range(N):
        if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
            changed_walkers.append(i)

    expected_changed = {5, companion_idx}
    actual_changed = set(changed_walkers)

    if actual_changed != expected_changed:
        print("=" * 60)
        print(f"FAILURE ON RUN {run}")
        print("=" * 60)
        print(f"Walker 5 clones to companion {companion_idx}")
        print(f"Expected changed walkers: {expected_changed}")
        print(f"Actual changed walkers: {actual_changed}")
        print(f"Extra changed: {actual_changed - expected_changed}")

        # Analyze the collision groups
        print("\nCollision group analysis:")
        print(f"  Walker 5 → Companion {companion_idx}")

        # Check if companion also clones
        if will_clone[companion_idx]:
            print(f"  Companion {companion_idx} → Companion {companions[companion_idx].item()}")
        else:
            print(f"  Companion {companion_idx} does NOT clone")

        # Check if any unexpected walker changed
        for unexpected in actual_changed - expected_changed:
            print(f"\n  Unexpected change in walker {unexpected}:")
            print(f"    Velocity before: {velocities[unexpected]}")
            print(f"    Velocity after:  {v_new[unexpected]}")
            print(f"    This walker clones: {will_clone[unexpected].item()}")
            print(f"    Its companion: {companions[unexpected].item()}")

            # Check if this walker is the companion of walker 5
            if unexpected == companion_idx:
                print("    → This IS the companion of walker 5 (EXPECTED)")
            else:
                print("    → This is NOT the companion of walker 5 (UNEXPECTED)")

            # Check if walker 5 is somehow in this walker's collision group
            if companions[unexpected] == 5:
                print(f"    → Walker {unexpected} clones to walker 5!")
            if companions[unexpected] == companion_idx:
                print(f"    → Walker {unexpected} clones to companion {companion_idx}!")

        # Check if the issue is self-collision or companion is cloning to walker 5
        if companion_idx == 5:
            print("\n  ⚠ SELF-COLLISION: Walker 5 clones to itself!")

        if not will_clone[companion_idx] and companions[companion_idx] == 5:
            print(
                f"\n  ⚠ Companion {companion_idx} has walker 5 as its companion (but doesn't clone)"
            )

        # Detailed collision group reconstruction
        print("\n  Reconstructing collision group for companion", companion_idx)
        cloners_mask = (companions == companion_idx) & will_clone
        cloner_indices = torch.where(cloners_mask)[0].tolist()
        print(f"    Walkers cloning to {companion_idx}: {cloner_indices}")

        if cloner_indices:
            cloner_indices_no_companion = [idx for idx in cloner_indices if idx != companion_idx]
            group_indices = [companion_idx, *cloner_indices_no_companion]
            print(f"    Full collision group: {group_indices}")
            print(f"    Size: {len(group_indices)}")

            if len(group_indices) != 2:
                print(f"    ⚠ MULTI-BODY COLLISION: {len(group_indices)} walkers in group")

        break
else:
    print("All runs passed!")
