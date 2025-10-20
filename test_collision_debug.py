"""Run the exact test with debug output."""

import pytest
import torch
from fragile.core.cloning import inelastic_collision_velocity


def test_non_cloners_unchanged():
    """Test walkers that don't clone keep their velocities - WITH DEBUG."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, d = 20, 3
    velocities = torch.randn(N, d, device=device)
    companions = torch.randint(0, N, (N,), device=device)
    will_clone = torch.zeros(N, dtype=torch.bool, device=device)
    will_clone[5] = True  # Only one walker clones

    companion_idx = companions[5].item()

    print(f"\n{'='*60}")
    print(f"SETUP")
    print(f"{'='*60}")
    print(f"Walker 5 clones to companion {companion_idx}")
    print(f"will_clone: {will_clone.nonzero().flatten().tolist()}")

    v_new = inelastic_collision_velocity(velocities, companions, will_clone)

    # Check which walkers changed
    changed_walkers = []
    for i in range(N):
        if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
            changed_walkers.append(i)

    print(f"\nWalkers that changed: {changed_walkers}")
    print(f"Expected: [5, {companion_idx}]")

    # Non-cloners should be unchanged (except companion of cloner)
    failed_walker = None
    for i in range(N):
        if i not in {5, companion_idx}:
            if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
                failed_walker = i
                print(f"\n{'='*60}")
                print(f"FAILURE: Walker {i} changed unexpectedly")
                print(f"{'='*60}")
                print(f"  Velocity before: {velocities[i]}")
                print(f"  Velocity after:  {v_new[i]}")
                print(f"  Diff: {v_new[i] - velocities[i]}")
                print(f"  Max diff: {(v_new[i] - velocities[i]).abs().max().item():.6e}")
                print(f"\n  Walker {i} properties:")
                print(f"    Clones: {will_clone[i].item()}")
                print(f"    Companion: {companions[i].item()}")

                # Check relationships
                if companions[i].item() == 5:
                    print(f"    → This walker's companion is walker 5")
                if companions[i].item() == companion_idx:
                    print(f"    → This walker's companion is {companion_idx} (same as walker 5's companion)")

                # Check if walker i is in a collision group
                walkers_cloning_to_i = torch.where((companions == i) & will_clone)[0]
                if walkers_cloning_to_i.numel() > 0:
                    print(f"    → Walkers cloning TO walker {i}: {walkers_cloning_to_i.tolist()}")

                # Check collision groups
                print(f"\n  Collision group analysis:")
                unique_companions = torch.unique(companions[will_clone])
                for c_idx in unique_companions:
                    cloners_mask = (companions == c_idx) & will_clone
                    cloner_indices = torch.where(cloners_mask)[0].tolist()
                    print(f"    Companion {c_idx}: cloners = {cloner_indices}")

                    cloner_indices_no_companion = [idx for idx in cloner_indices if idx != c_idx]
                    group_indices = [c_idx] + cloner_indices_no_companion
                    print(f"      → Full group: {group_indices}")

                    if i in group_indices:
                        print(f"      → ✓ Walker {i} IS in this collision group")

                assert False, f"Walker {i} should not have changed"

    if failed_walker is None:
        print(f"\n✓ Test passed!")


if __name__ == "__main__":
    test_non_cloners_unchanged()
