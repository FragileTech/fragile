"""Debug inelastic collision velocity to understand test failure."""

import torch

from fragile.core.cloning import inelastic_collision_velocity


# Replicate test setup
N, d = 20, 3
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

velocities = torch.randn(N, d, device=device)
companions = torch.randint(0, N, (N,), device=device)
will_clone = torch.zeros(N, dtype=torch.bool, device=device)
will_clone[5] = True  # Only walker 5 clones

print("=" * 60)
print("SETUP")
print("=" * 60)
print(f"Walker 5 clones to companion {companions[5].item()}")
print(f"Walker 5 velocity BEFORE: {velocities[5]}")
print(f"Companion {companions[5].item()} velocity BEFORE: {velocities[companions[5]]}")

# Call function
v_new = inelastic_collision_velocity(velocities, companions, will_clone)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Walker 5 velocity AFTER: {v_new[5]}")
print(f"Companion {companions[5].item()} velocity AFTER: {v_new[companions[5]]}")

# Check which walkers changed
companion_idx = companions[5].item()
changed_walkers = []
for i in range(N):
    if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
        changed_walkers.append(i)

print(f"\nWalkers whose velocities changed: {changed_walkers}")
print(f"Expected: [5, {companion_idx}]")

# Check if the problem is that companion_idx also happens to be cloning
print("\n" + "=" * 60)
print("COLLISION GROUP ANALYSIS")
print("=" * 60)
print(f"Companion {companion_idx} is cloning: {will_clone[companion_idx].item()}")

# Check if companion_idx itself clones to someone else
if will_clone[companion_idx]:
    print(f"  → Companion {companion_idx} clones to {companions[companion_idx].item()}")

    # If companion also clones to walker 5, we have a mutual collision
    if companions[companion_idx].item() == 5:
        print("  → MUTUAL COLLISION: Walker 5 ↔ Companion form a collision group")
    else:
        print(
            f"  → Companion {companion_idx} clones to different walker {companions[companion_idx].item()}"
        )
        print("  → This creates a multi-body collision involving multiple walkers!")

# Check for transitive collisions
print("\nFull companion graph:")
for i in range(N):
    if will_clone[i]:
        print(f"  Walker {i} → Companion {companions[i].item()}")

# The key insight: if companion also clones, it forms a larger collision group
if will_clone[companion_idx]:
    # Build the full collision group for walker 5's companion
    group_5_companion = companions[5].item()
    cloners_to_companion = [
        i for i in range(N) if (companions[i] == group_5_companion and will_clone[i])
    ]

    print(f"\nCollision group for companion {group_5_companion}:")
    print(f"  Companion: {group_5_companion}")
    print(f"  Cloners: {cloners_to_companion}")

    # Now check the companion's OWN collision (if it's cloning)
    if will_clone[group_5_companion]:
        target_of_companion = companions[group_5_companion].item()
        cloners_to_target = [
            i for i in range(N) if (companions[i] == target_of_companion and will_clone[i])
        ]

        print(f"\nCompanion {group_5_companion} is also cloning to {target_of_companion}:")
        print(f"  Companion: {target_of_companion}")
        print(f"  Cloners (including {group_5_companion}): {cloners_to_target}")

print("\n" + "=" * 60)
print("TEST ASSERTION")
print("=" * 60)
print(f"Test expects only walkers [5, {companion_idx}] to change")
print(f"Actual walkers that changed: {changed_walkers}")

if set(changed_walkers) == {5, companion_idx}:
    print("✓ TEST WOULD PASS")
else:
    print("✗ TEST FAILS")
    extra_changed = set(changed_walkers) - {5, companion_idx}
    print(f"Extra walkers that changed: {extra_changed}")
