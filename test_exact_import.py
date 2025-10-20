"""Import test directly and add detailed output."""

import sys


sys.path.insert(0, "tests")

import torch

from fragile.core.cloning import inelastic_collision_velocity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N, d = 20, 3
velocities = torch.randn(N, d, device=device)
companions = torch.randint(0, N, (N,), device=device)
will_clone = torch.zeros(N, dtype=torch.bool, device=device)
will_clone[5] = True  # Only one walker clones

companion_idx = companions[5]

print(f"Companion of walker 5: {companion_idx}")
print(f"companion_idx type: {type(companion_idx)}")
print(
    f"companion_idx value: {companion_idx.item() if hasattr(companion_idx, 'item') else companion_idx}"
)

v_new = inelastic_collision_velocity(velocities, companions, will_clone)

# Non-cloners should be unchanged (except companion of cloner)
for i in range(N):
    # This is the EXACT condition from the test
    if i not in {5, companion_idx}:
        if not torch.allclose(v_new[i], velocities[i], atol=1e-6):
            print(f"\n✗ Walker {i} failed the test")
            print(f"  Velocity before: {velocities[i]}")
            print(f"  Velocity after:  {v_new[i]}")
            print(f"  Diff: {(v_new[i] - velocities[i]).abs()}")
            print(f"  Max diff: {(v_new[i] - velocities[i]).abs().max():.6e}")

            print("\n  Checking condition: i not in {5, companion_idx}")
            print(f"    i = {i}")
            print(f"    companion_idx = {companion_idx}")
            print(f"    companion_idx.item() = {companion_idx.item()}")
            print(f"    i == 5: {i == 5}")
            print(f"    i == companion_idx: {i == companion_idx}")
            print(f"    i == companion_idx.item(): {i == companion_idx.item()}")
            print(f"    5 in {{5, companion_idx}}: {5 in {5, companion_idx}}")
            print(
                f"    companion_idx in {{5, companion_idx}}: {companion_idx in {5, companion_idx}}"
            )
            print(f"    i in {{5, companion_idx}}: {i in {5, companion_idx}}")

            break
else:
    print("\n✓ All walkers passed")
